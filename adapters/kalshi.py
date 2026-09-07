"""Read-only Kalshi historical-data adapter.

Implemented against the public Kalshi Trade API v2 market-data endpoints, all
of which are documented as requiring **no authentication**:

* ``GET /markets`` and ``GET /markets/{ticker}`` - market definitions,
  settlement results and resolution rules.
* ``GET /markets/trades`` and ``GET /historical/trades`` - the public tape.
* ``GET /series/{series_ticker}/markets/{ticker}/candlesticks`` and
  ``GET /historical/markets/{ticker}/candlesticks`` - historical bid/ask series.
* ``GET /historical/cutoff`` - the boundary between the live and historical
  data tiers (Kalshi keeps roughly three months in the live tier).

Only GET is ever issued. This module cannot place, amend or cancel an order,
holds no credentials, and never signs a request.

Conventions this adapter maps onto the project models:

* Kalshi quotes prices as fixed-point **dollar strings** for a contract that
  settles at $1.00, so a price is already a probability in ``[0, 1]``.
* Order prices are limited to $0.01-$0.99, so a ``yes_bid`` of ``0.00`` or a
  ``yes_ask`` of ``1.00`` means *no resting order on that side*, and is mapped
  to a missing side rather than to a real price. Set
  ``treat_extremes_as_missing=False`` to keep the literal values.
* Candlesticks carry no book depth. Sizes are therefore ``0`` unless
  ``depth_from_volume`` is enabled, which substitutes the candle's traded
  volume as a **proxy** and stamps that fact into every quote's ``source``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from adapters.base import AdapterError, RawStore, ReadOnlyClient, Transport, urllib_get
from core.dataset import MarketDataset, sorted_dataset
from core.models import (
    Market,
    MarketStatus,
    Outcome,
    Quote,
    Side,
    Trade,
)

VENUE = "KALSHI"
SOURCE = "KALSHI_API_V2"
CANDLE_SOURCE = "KALSHI_CANDLES"
CANDLE_SOURCE_PROXY_DEPTH = "KALSHI_CANDLES+VOLUME_DEPTH_PROXY"
HISTORICAL_LABEL = "KALSHI HISTORICAL (read-only)"
DEFAULT_BASE_URL = "https://external-api.kalshi.com/trade-api/v2"

VALID_PERIOD_INTERVALS = (1, 60, 1440)
MAX_PAGE_LIMIT = 1000

# Kalshi order prices live in $0.01-$0.99; the bounds mean "no resting order".
NO_BID_PRICE = 0.0
NO_ASK_PRICE = 1.0

_RESOLVED_STATUSES = {"settled", "finalized", "determined", "resolved"}
_CLOSED_STATUSES = {"closed", "inactive", "disputed", "amended", "paused"}


# --------------------------------------------------------------------------
# Field parsing
# --------------------------------------------------------------------------


def parse_dollars(value: Any) -> float | None:
    """Parse a Kalshi fixed-point dollar string into a float probability.

    Args:
        value: A string such as ``"0.5600"``, a number, or ``None``/``""``.

    Returns:
        The value as a float, or ``None`` when absent.

    Raises:
        AdapterError: If the value is present but not numeric.
    """
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as error:
        raise AdapterError(f"unparseable dollar value: {value!r}") from error


def parse_count(value: Any, default: float = 0.0) -> float:
    """Parse a Kalshi fixed-point count string (contracts) into a float."""
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError) as error:
        raise AdapterError(f"unparseable count value: {value!r}") from error


def parse_timestamp(value: Any) -> datetime | None:
    """Parse an ISO 8601 timestamp or Unix seconds into an aware UTC datetime.

    Timestamps are preserved exactly as the venue reported them; only the
    representation changes (to a timezone-aware ``datetime`` in UTC).
    """
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=UTC)
    text = str(value).strip()
    if text.isdigit():
        return datetime.fromtimestamp(int(text), tz=UTC)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as error:
        raise AdapterError(f"unparseable timestamp: {value!r}") from error
    return parsed.astimezone(UTC) if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _first(payload: dict[str, Any], *names: str, default: Any = None) -> Any:
    """Return the first present key from ``names`` (schemas vary by endpoint)."""
    for name in names:
        if name in payload and payload[name] not in (None, ""):
            return payload[name]
    return default


def series_ticker_for(ticker: str) -> str:
    """Derive the series ticker from a market ticker.

    Kalshi market tickers are ``SERIES-EVENT-STRIKE`` (for example
    ``KXHIGHNY-25AUG31-B70``), so the series is the leading segment.
    """
    return ticker.split("-", 1)[0] if ticker else ""


def map_outcome(result: Any) -> Outcome | None:
    """Map the Kalshi ``result`` field onto an :class:`core.models.Outcome`.

    ``"yes"``/``"no"`` map to the binary outcomes. An empty result means the
    market has not settled. ``"scalar"`` markets are not binary and are
    reported as unresolved (the caller filters them out).
    """
    text = str(result or "").strip().lower()
    if text == "yes":
        return Outcome.YES
    if text == "no":
        return Outcome.NO
    return None


def map_status(status: Any, outcome: Outcome | None) -> MarketStatus:
    """Map a Kalshi market status onto a :class:`core.models.MarketStatus`."""
    text = str(status or "").strip().lower()
    if text in _RESOLVED_STATUSES and outcome is not None:
        return MarketStatus.RESOLVED
    if text in _RESOLVED_STATUSES or text in _CLOSED_STATUSES:
        return MarketStatus.CLOSED
    return MarketStatus.OPEN


def _book_side(price: float | None, missing_at: float, enabled: bool) -> float | None:
    """Collapse a bound price ($0.00 bid / $1.00 ask) to a missing side."""
    if price is None:
        return None
    if enabled and abs(price - missing_at) < 1e-9:
        return None
    return price


# --------------------------------------------------------------------------
# Normalization
# --------------------------------------------------------------------------


def normalize_market(payload: dict[str, Any], fee_rate: float = 0.07) -> Market:
    """Convert one Kalshi market payload into a :class:`core.models.Market`.

    Settlement rules (``rules_primary``/``rules_secondary``) and the settlement
    timestamp are carried across verbatim so the resolution basis travels with
    the data.

    Args:
        payload: One entry from the ``markets`` array.
        fee_rate: Fee coefficient recorded on the market.

    Returns:
        The normalized market.

    Raises:
        AdapterError: If the payload has no ticker or no close time.
    """
    ticker = str(_first(payload, "ticker", default="")).strip()
    if not ticker:
        raise AdapterError("market payload has no ticker")

    close_time = parse_timestamp(_first(payload, "close_time"))
    if close_time is None:
        raise AdapterError(f"market {ticker} has no close_time")
    created_at = (
        parse_timestamp(_first(payload, "open_time", "created_time")) or close_time
    )
    outcome = map_outcome(payload.get("result"))
    settlement_ts = parse_timestamp(
        _first(payload, "settlement_ts", "expected_expiration_time", "latest_expiration_time")
    )

    title = str(
        _first(payload, "title", "yes_sub_title", "subtitle", default=ticker)
    ).strip()
    tick = 0.01
    ranges = payload.get("price_ranges") or []
    if ranges and isinstance(ranges, list) and isinstance(ranges[0], dict):
        step = parse_dollars(ranges[0].get("step"))
        if step and step > 0:
            tick = step

    return Market(
        market_id=ticker,
        question=title or ticker,
        category=str(_first(payload, "event_ticker", default=series_ticker_for(ticker))),
        created_at=created_at,
        close_time=close_time,
        resolution_time=settlement_ts if outcome is not None else None,
        outcome=outcome,
        status=map_status(payload.get("status"), outcome),
        tick_size=tick,
        fee_rate=fee_rate,
        source=SOURCE,
        is_demo=False,
        venue=VENUE,
        rules_primary=str(payload.get("rules_primary") or ""),
        rules_secondary=str(payload.get("rules_secondary") or ""),
    )


def normalize_trade(payload: dict[str, Any], index: int = 0) -> Trade:
    """Convert one Kalshi trade payload into a :class:`core.models.Trade`.

    Prices are expressed in YES terms. A taker whose outcome side is ``no`` is
    economically selling YES, so the aggressor is recorded as ``SELL``.
    """
    ticker = str(_first(payload, "ticker", default="")).strip()
    ts = parse_timestamp(_first(payload, "created_time"))
    if ts is None:
        raise AdapterError(f"trade for {ticker or 'unknown market'} has no created_time")

    price = parse_dollars(_first(payload, "yes_price_dollars", "yes_price"))
    if price is None:
        no_price = parse_dollars(_first(payload, "no_price_dollars", "no_price"))
        if no_price is None:
            raise AdapterError(f"trade {payload.get('trade_id')} has no price")
        price = 1.0 - no_price

    taker = str(
        _first(payload, "taker_outcome_side", "taker_side", default="yes")
    ).strip().lower()
    size = parse_count(_first(payload, "count_fp", "count"), default=0.0)
    if size <= 0:
        raise AdapterError(f"trade {payload.get('trade_id')} has non-positive size")

    return Trade(
        trade_id=str(_first(payload, "trade_id", default=f"{ticker}-{index}")),
        market_id=ticker,
        ts=ts,
        price=min(max(price, 0.0), 1.0),
        size=size,
        aggressor=Side.BUY if taker == "yes" else Side.SELL,
        source=SOURCE,
    )


def normalize_quote_from_market(
    payload: dict[str, Any],
    fetched_at: datetime | None = None,
    treat_extremes_as_missing: bool = True,
) -> Quote | None:
    """Build the current top-of-book quote from a market payload.

    ``GET /markets/{ticker}`` carries the live bid/ask and their displayed
    sizes, which is what paper trading polls. Unlike candlesticks this *does*
    carry real depth, so no proxy is needed.

    Args:
        payload: One market payload.
        fetched_at: Fallback timestamp when the venue reports no update time.
        treat_extremes_as_missing: Map a $0.00 bid / $1.00 ask to a missing side.

    Returns:
        The quote, or ``None`` when the payload carries no ticker or no prices.
    """
    ticker = str(_first(payload, "ticker", default="")).strip()
    if not ticker:
        return None

    bid = _book_side(
        parse_dollars(_first(payload, "yes_bid_dollars", "yes_bid")),
        NO_BID_PRICE,
        treat_extremes_as_missing,
    )
    ask = _book_side(
        parse_dollars(_first(payload, "yes_ask_dollars", "yes_ask")),
        NO_ASK_PRICE,
        treat_extremes_as_missing,
    )
    if bid is None and ask is None:
        return None
    if bid is not None and ask is not None and bid > ask:
        return None  # Crossed book: report no quote rather than invent one.

    ts = parse_timestamp(_first(payload, "updated_time", "created_time")) or (
        fetched_at or datetime.now(UTC)
    )
    return Quote(
        market_id=ticker,
        ts=ts,
        bid=bid,
        ask=ask,
        bid_size=(
            parse_count(_first(payload, "yes_bid_size_fp", "yes_bid_size"))
            if bid is not None
            else 0.0
        ),
        ask_size=(
            parse_count(_first(payload, "yes_ask_size_fp", "yes_ask_size"))
            if ask is not None
            else 0.0
        ),
        source=SOURCE,
    )


def normalize_candlesticks(
    ticker: str,
    payload: dict[str, Any],
    depth_from_volume: bool = False,
    treat_extremes_as_missing: bool = True,
) -> list[Quote]:
    """Convert a candlestick response into a chronological list of quotes.

    Each candle contributes one quote stamped at ``end_period_ts`` using the
    closing bid and ask of that period. Candles whose bid would exceed the ask
    are dropped rather than coerced, so no fabricated book is ever produced.

    Args:
        ticker: Market the candles belong to.
        payload: The decoded candlestick response.
        depth_from_volume: Use candle volume as a stand-in for displayed size.
        treat_extremes_as_missing: Map a $0.00 bid / $1.00 ask to a missing side.

    Returns:
        Quotes sorted by timestamp.
    """
    source = CANDLE_SOURCE_PROXY_DEPTH if depth_from_volume else CANDLE_SOURCE
    market_id = str(payload.get("ticker") or ticker)
    quotes: list[Quote] = []

    for candle in payload.get("candlesticks") or []:
        ts = parse_timestamp(candle.get("end_period_ts"))
        if ts is None:
            continue
        bid_block = candle.get("yes_bid") or {}
        ask_block = candle.get("yes_ask") or {}
        bid = _book_side(
            parse_dollars(_first(bid_block, "close_dollars", "close")),
            NO_BID_PRICE,
            treat_extremes_as_missing,
        )
        ask = _book_side(
            parse_dollars(_first(ask_block, "close_dollars", "close")),
            NO_ASK_PRICE,
            treat_extremes_as_missing,
        )
        if bid is not None and ask is not None and bid > ask:
            continue  # Crossed candle: drop rather than invent a book.

        volume = parse_count(_first(candle, "volume_fp", "volume"), default=0.0)
        size = volume if depth_from_volume else 0.0
        quotes.append(
            Quote(
                market_id=market_id,
                ts=ts,
                bid=bid,
                ask=ask,
                bid_size=size if bid is not None else 0.0,
                ask_size=size if ask is not None else 0.0,
                source=source,
            )
        )

    quotes.sort(key=lambda q: q.ts)
    return quotes


# --------------------------------------------------------------------------
# Client
# --------------------------------------------------------------------------


class KalshiClient(ReadOnlyClient):
    """Thin read-only HTTP client for Kalshi public market-data endpoints."""

    def markets(
        self,
        status: str | None = None,
        tickers: str | None = None,
        event_ticker: str | None = None,
        series_ticker: str | None = None,
        min_close_ts: int | None = None,
        max_close_ts: int | None = None,
        limit: int = 200,
        max_pages: int = 10,
    ) -> list[dict[str, Any]]:
        """List markets, following the pagination cursor.

        Args:
            status: One of ``unopened``, ``open``, ``closed``, ``settled``, ...
            tickers: Comma-separated market tickers.
            event_ticker: Restrict to a single event.
            series_ticker: Restrict to a series (requires ``mve_filter=exclude``).
            min_close_ts: Unix seconds; only with ``closed``/empty status.
            max_close_ts: Unix seconds; only with ``closed``/empty status.
            limit: Page size, capped at 1000 by the venue.
            max_pages: Stop after this many pages.

        Returns:
            The raw market payloads, unmodified.
        """
        params: dict[str, Any] = {
            "limit": min(int(limit), MAX_PAGE_LIMIT),
            "status": status,
            "tickers": tickers,
            "event_ticker": event_ticker,
            "min_close_ts": min_close_ts,
            "max_close_ts": max_close_ts,
        }
        if series_ticker:
            params["series_ticker"] = series_ticker
            params["mve_filter"] = "exclude"
        return self._paginate("/markets", params, "markets", max_pages, "markets")

    def market(self, ticker: str) -> dict[str, Any]:
        """Fetch a single market definition."""
        payload = self.get(f"/markets/{ticker}", endpoint="market")
        market = payload.get("market") if isinstance(payload, dict) else None
        if not market:
            raise AdapterError(f"no market payload returned for {ticker}")
        return market

    def trades(
        self,
        ticker: str,
        min_ts: int | None = None,
        max_ts: int | None = None,
        limit: int = 1000,
        max_pages: int = 20,
        historical: bool = False,
    ) -> list[dict[str, Any]]:
        """Fetch public tape prints for one market.

        Args:
            ticker: Market ticker.
            min_ts: Unix seconds lower bound.
            max_ts: Unix seconds upper bound.
            limit: Page size (max 1000).
            max_pages: Page budget.
            historical: Use ``/historical/trades`` for archived data.
        """
        path = "/historical/trades" if historical else "/markets/trades"
        params = {
            "ticker": ticker,
            "min_ts": min_ts,
            "max_ts": max_ts,
            "limit": min(int(limit), MAX_PAGE_LIMIT),
        }
        return self._paginate(path, params, "trades", max_pages, "trades")

    def candlesticks(
        self,
        ticker: str,
        start_ts: int,
        end_ts: int,
        period_interval: int = 60,
        series_ticker: str | None = None,
        historical: bool = True,
    ) -> dict[str, Any]:
        """Fetch the historical bid/ask candlestick series for one market.

        Args:
            ticker: Market ticker.
            start_ts: Unix seconds; candles ending at or after this time.
            end_ts: Unix seconds; candles ending at or before this time.
            period_interval: Candle length in minutes: 1, 60 or 1440.
            series_ticker: Required for the live endpoint; derived if omitted.
            historical: Use ``/historical/markets/{ticker}/candlesticks``, which
                covers markets archived out of the live tier and needs no series.

        Returns:
            The raw candlestick payload.

        Raises:
            ValueError: If ``period_interval`` is not 1, 60 or 1440.
        """
        if period_interval not in VALID_PERIOD_INTERVALS:
            raise ValueError(
                f"period_interval must be one of {VALID_PERIOD_INTERVALS}, got {period_interval}"
            )
        params = {
            "start_ts": int(start_ts),
            "end_ts": int(end_ts),
            "period_interval": int(period_interval),
        }
        if historical:
            path = f"/historical/markets/{ticker}/candlesticks"
        else:
            series = series_ticker or series_ticker_for(ticker)
            path = f"/series/{series}/markets/{ticker}/candlesticks"
        payload = self.get(path, params, endpoint="candlesticks")
        if not isinstance(payload, dict):
            raise AdapterError(f"unexpected candlestick payload for {ticker}")
        return payload

    def cutoff(self) -> dict[str, Any]:
        """Fetch the live/historical tier cutoff timestamps."""
        payload = self.get("/historical/cutoff", endpoint="cutoff")
        return payload if isinstance(payload, dict) else {}

    def _paginate(
        self,
        path: str,
        params: dict[str, Any],
        key: str,
        max_pages: int,
        endpoint: str,
    ) -> list[dict[str, Any]]:
        """Follow ``cursor`` until it is empty or the page budget runs out."""
        items: list[dict[str, Any]] = []
        cursor: str | None = None
        for _ in range(max(1, max_pages)):
            page_params = dict(params)
            if cursor:
                page_params["cursor"] = cursor
            payload = self.get(path, page_params, endpoint=endpoint)
            if not isinstance(payload, dict):
                raise AdapterError(f"unexpected payload from {path}")
            batch = payload.get(key) or []
            items.extend(batch)
            cursor = payload.get("cursor") or ""
            if not cursor or not batch:
                break
        return items


# --------------------------------------------------------------------------
# Adapter
# --------------------------------------------------------------------------


@dataclass
class KalshiFetchSpec:
    """What to fetch: which markets, over which window, at which resolution."""

    tickers: list[str]
    start_ts: int
    end_ts: int
    period_interval: int = 60
    include_trades: bool = True
    historical: bool = True
    max_trade_pages: int = 20


class KalshiAdapter:
    """Read-only adapter that turns Kalshi history into a `MarketDataset`.

    The adapter performs no authentication, issues only GET requests, and
    writes every raw response to ``data/raw/kalshi`` before parsing it.
    """

    venue = VENUE

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        transport: Transport | None = None,
        raw_store: RawStore | None = None,
    ) -> None:
        cfg = dict(config or {})
        self.config = cfg
        self.depth_from_volume = bool(cfg.get("depth_from_volume", True))
        self.treat_extremes_as_missing = bool(cfg.get("treat_extremes_as_missing", True))
        self.fee_rate = float(cfg.get("fee_rate", 0.07))
        self.store_raw = bool(cfg.get("store_raw_responses", True))
        self.raw_store = raw_store or RawStore(
            venue="kalshi",
            root=cfg.get("raw_dir", "data/raw"),
            enabled=self.store_raw,
        )
        self.client = KalshiClient(
            base_url=str(cfg.get("base_url", DEFAULT_BASE_URL)),
            transport=transport or urllib_get,
            timeout=float(cfg.get("timeout_seconds", 20.0)),
            min_request_interval=float(cfg.get("min_request_interval_seconds", 0.2)),
            max_retries=int(cfg.get("max_retries", 3)),
            raw_store=self.raw_store,
        )

    def fetch_dataset(self, spec: KalshiFetchSpec) -> MarketDataset:
        """Fetch and normalize the markets in ``spec`` into one dataset.

        Args:
            spec: Which tickers and time window to pull.

        Returns:
            A :class:`core.dataset.MarketDataset` labelled as Kalshi historical
            data, with any per-market problems recorded in ``notes`` rather than
            raised - one bad market must not lose the rest of the pull.

        Raises:
            AdapterError: If no requested market could be fetched at all.
        """
        markets: list[Market] = []
        quotes: list[Quote] = []
        trades: list[Trade] = []
        notes: list[str] = []

        if self.depth_from_volume:
            notes.append(
                "Quote sizes are a PROXY: candlestick history carries no book depth, "
                "so each candle's traded volume stands in for displayed size."
            )

        for ticker in spec.tickers:
            try:
                payload = self.client.market(ticker)
                market = normalize_market(payload, fee_rate=self.fee_rate)
            except AdapterError as error:
                notes.append(f"{ticker}: skipped ({error})")
                continue

            if str(payload.get("market_type", "binary")).lower() not in ("binary", ""):
                notes.append(f"{ticker}: skipped (not a binary market)")
                continue

            candles = self.client.candlesticks(
                ticker,
                spec.start_ts,
                spec.end_ts,
                spec.period_interval,
                historical=spec.historical,
            )
            market_quotes = normalize_candlesticks(
                ticker,
                candles,
                depth_from_volume=self.depth_from_volume,
                treat_extremes_as_missing=self.treat_extremes_as_missing,
            )
            if not market_quotes:
                notes.append(f"{ticker}: no candlesticks in the requested window")

            market_trades: list[Trade] = []
            if spec.include_trades:
                for index, raw_trade in enumerate(
                    self.client.trades(
                        ticker,
                        min_ts=spec.start_ts,
                        max_ts=spec.end_ts,
                        max_pages=spec.max_trade_pages,
                        historical=spec.historical,
                    )
                ):
                    try:
                        market_trades.append(normalize_trade(raw_trade, index))
                    except AdapterError as error:
                        notes.append(f"{ticker}: dropped a trade ({error})")

            markets.append(market)
            quotes.extend(market_quotes)
            trades.extend(market_trades)

        if not markets:
            raise AdapterError(
                "no markets could be fetched; " + ("; ".join(notes) if notes else "empty request")
            )

        if not any(m.outcome is not None for m in markets):
            notes.append(
                "No market in this pull has settled, so there is nothing to settle "
                "positions against and forecast scoring will be empty."
            )

        return sorted_dataset(
            MarketDataset(
                markets=markets,
                quotes=quotes,
                trades=trades,
                seed=None,
                label=HISTORICAL_LABEL,
                source=SOURCE,
                venue=VENUE,
                fetched_at=datetime.now(UTC),
                notes=notes,
            )
        )
