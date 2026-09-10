"""Read-only Manifold adapter: real resolved markets that walk-forward can split.

Why this venue
--------------
Kalshi's settled population is dominated by markets that live one to six hours
and overlap heavily, so every walk-forward training fold purges to empty
(measured 2026-09-08: 2,000 markets, 127 resolved, **zero** usable folds).
Manifold's resolved binary markets have a median lifetime of roughly 200 days
and resolve in a long sequence, which is the shape the protocol needs.

Manifold is play money. An edge found here is **not tradable**, and that is
accepted: the blocker being cleared is scientific, not financial. The protocol
has never once produced a usable fold on real data, and this is the cheapest
corpus that can prove the machinery works end to end.

The order-book problem, stated honestly
---------------------------------------
Manifold is an automated market maker, not an order book. There is no bid and
no ask - only a single probability that moves when someone bets. The engine
needs two sides, so this adapter **derives** them:

* the mid is the AMM probability at that instant;
* the half-spread is estimated from *observed price impact* - the median
  ``|probAfter - probBefore|`` across nearby bets, which is what a real trade
  actually moved the price by;
* displayed size is a proxy derived from market liquidity.

Every quote produced this way is stamped ``MANIFOLD_AMM+IMPACT_SPREAD_PROXY``
and the dataset carries a note saying so. It is an approximation of trading
cost, not a real book, and results built on it inherit that approximation. This
mirrors how ``adapters/kalshi.py`` handles ``depth_from_volume``.

**Read-only by construction.** The transport issues HTTP GET and takes no
request body. There is no POST/PUT/DELETE path here, no signing, and no
credential header is ever set.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from adapters.base import AdapterError, RawStore, ReadOnlyClient, Transport, urllib_get
from core.dataset import MarketDataset
from core.models import Market, MarketStatus, Outcome, Quote, Side, Trade

VENUE = "manifold"
DEFAULT_BASE_URL = "https://api.manifold.markets/v0"
SOURCE = "MANIFOLD_AMM"
PROXY_SOURCE = "MANIFOLD_AMM+IMPACT_SPREAD_PROXY"
LABEL = "MANIFOLD HISTORICAL (read-only)"

MAX_PAGE_LIMIT = 1000

# Manifold resolves markets to more than YES/NO. Only the two terminal binary
# outcomes can produce a label; MKT (a partial resolution) and CANCEL (voided)
# cannot, and must never be guessed at from the final price.
RESOLVED_OUTCOMES = {"YES": Outcome.YES, "NO": Outcome.NO}
EXCLUDED_RESOLUTIONS = frozenset({"MKT", "CANCEL"})

MIN_SPREAD = 0.005
"""Floor on the derived half-spread, in probability units.

A market with no observed impact would otherwise look free to trade, which
would flatter every result built on it.
"""

MAX_SPREAD = 0.15


def parse_timestamp(value: Any) -> datetime | None:
    """Manifold sends epoch milliseconds. Convert to aware UTC."""
    if value is None or value == "":
        return None
    try:
        return datetime.fromtimestamp(float(value) / 1000.0, UTC)
    except (TypeError, ValueError, OSError, OverflowError):
        return None


def parse_probability(value: Any) -> float | None:
    """A probability already in ``[0, 1]``, or ``None`` when unusable."""
    if value is None:
        return None
    try:
        probability = float(value)
    except (TypeError, ValueError):
        return None
    if not 0.0 <= probability <= 1.0:
        return None
    return probability


def map_outcome(resolution: Any) -> Outcome | None:
    """Map a Manifold resolution to a binary outcome, or ``None``.

    ``MKT`` and ``CANCEL`` deliberately return ``None``: a partially-resolved or
    voided market has no binary answer, and inferring one from the price is the
    exact mistake ``research/labels.py`` exists to prevent.
    """
    if resolution is None:
        return None
    return RESOLVED_OUTCOMES.get(str(resolution).upper())


def market_lifetime_days(payload: dict[str, Any]) -> float | None:
    """How long a listed market was open, in days, or ``None`` if unknown."""
    opened = parse_timestamp(payload.get("createdTime"))
    closed = parse_timestamp(payload.get("closeTime"))
    if opened is None or closed is None:
        return None
    return (closed - opened).total_seconds() / 86400.0


def is_usable_binary(
    payload: dict[str, Any], max_lifetime_days: float | None = None
) -> bool:
    """Whether a listed market can contribute a labelled example.

    ``max_lifetime_days`` bounds how long a market may have been open. That is
    not a quality filter - it controls **concurrency**, which is what the
    walk-forward purge actually depends on. Measured on 286 Manifold markets:
    unbounded lifetimes gave 115 concurrent markets and **zero** usable folds
    even where a fold had 121 training markets, because every training market
    resolved after the test window opened. Capping lifetime at 120 days gave 21
    concurrent and 3 usable folds from the same data.
    """
    if str(payload.get("outcomeType", "")).upper() != "BINARY":
        return False
    if not payload.get("isResolved"):
        return False
    if str(payload.get("resolution", "")).upper() in EXCLUDED_RESOLUTIONS:
        return False
    if map_outcome(payload.get("resolution")) is None or not payload.get("id"):
        return False
    if max_lifetime_days is not None:
        lifetime = market_lifetime_days(payload)
        if lifetime is None or lifetime > max_lifetime_days:
            return False
    return True


def normalize_market(payload: dict[str, Any], fee_rate: float = 0.0) -> Market:
    """Turn a Manifold market payload into the project's typed model.

    Raises:
        AdapterError: If the market has no close time, without which a decision
            grid cannot be built.
    """
    created = parse_timestamp(payload.get("createdTime"))
    close = parse_timestamp(payload.get("closeTime"))
    resolved_at = parse_timestamp(payload.get("resolutionTime"))
    if close is None:
        raise AdapterError(f"market {payload.get('id')} has no close time")

    outcome = map_outcome(payload.get("resolution"))
    # A resolved market with no resolution timestamp is unusable: without one we
    # cannot prove a forecast predates its outcome.
    if outcome is not None and resolved_at is None:
        outcome = None

    return Market(
        market_id=str(payload.get("id")),
        question=str(payload.get("question", "")),
        category=str(payload.get("groupSlugs", [""])[0]) if payload.get("groupSlugs") else "",
        created_at=created or close,
        close_time=close,
        resolution_time=resolved_at,
        outcome=outcome,
        status=MarketStatus.RESOLVED if outcome is not None else MarketStatus.OPEN,
        tick_size=0.01,
        fee_rate=fee_rate,
        source=SOURCE,
        is_demo=False,
    )


def estimate_half_spread(bets: list[dict[str, Any]]) -> float:
    """Estimate a half-spread from observed AMM price impact.

    Each bet reports the probability before and after it landed, so the median
    absolute move *is* the cost a real trade paid. That is a far more defensible
    stand-in for a spread than a constant, because it is measured from the
    venue's own behaviour rather than assumed.
    """
    impacts = [
        abs(after - before)
        for bet in bets
        if (before := parse_probability(bet.get("probBefore"))) is not None
        and (after := parse_probability(bet.get("probAfter"))) is not None
    ]
    moves = [i for i in impacts if i > 0.0]
    if not moves:
        return MIN_SPREAD
    return max(MIN_SPREAD, min(MAX_SPREAD, statistics.median(moves)))


def normalize_bets(
    market_id: str,
    bets: list[dict[str, Any]],
    liquidity: float = 100.0,
) -> tuple[list[Quote], list[Trade]]:
    """Turn a market's bet history into a quote series and a trade tape.

    The AMM has no book, so each bet yields both:

    * a **quote** - the post-bet probability as a mid, widened by the derived
      half-spread into a synthetic bid and ask;
    * a **trade** - the bet itself, at the price it actually executed.

    Redemptions are skipped: they are share settlements, not price-forming
    trades, and counting them would inflate the tape with non-events.
    """
    usable = [
        bet
        for bet in bets
        if not bet.get("isRedemption")
        and parse_timestamp(bet.get("createdTime")) is not None
        and parse_probability(bet.get("probAfter")) is not None
    ]
    if not usable:
        return [], []

    usable.sort(key=lambda b: float(b["createdTime"]))
    half_spread = estimate_half_spread(usable)
    # Displayed size is a proxy: an AMM has depth but no posted size. Liquidity
    # scaled to contracts gives fills something finite to work against.
    size = max(1.0, round(liquidity))

    quotes: list[Quote] = []
    trades: list[Trade] = []
    for index, bet in enumerate(usable):
        ts = parse_timestamp(bet["createdTime"])
        mid = parse_probability(bet.get("probAfter"))
        if ts is None or mid is None:
            continue

        bid = max(0.01, min(0.99, mid - half_spread))
        ask = max(0.01, min(0.99, mid + half_spread))
        if ask <= bid:
            continue
        quotes.append(
            Quote(
                market_id=market_id,
                ts=ts,
                bid=bid,
                ask=ask,
                bid_size=size,
                ask_size=size,
                source=PROXY_SOURCE,
            )
        )

        executed = parse_probability(bet.get("probBefore")) or mid
        outcome = str(bet.get("outcome", "YES")).upper()
        trades.append(
            Trade(
                trade_id=f"{market_id}-B{index:06d}",
                market_id=market_id,
                ts=ts,
                price=max(0.0, min(1.0, executed)),
                size=max(1.0, abs(float(bet.get("amount", 1.0) or 1.0))),
                aggressor=Side.BUY if outcome == "YES" else Side.SELL,
                source=SOURCE,
            )
        )
    return quotes, trades


@dataclass
class ManifoldFetchSpec:
    """Which markets to fetch, and how much of each."""

    market_ids: list[str] = field(default_factory=list)
    max_bet_pages: int = 4
    """Each page is up to 1,000 bets. Four is plenty for most markets."""


class ManifoldClient(ReadOnlyClient):
    """GET-only client for the public Manifold API. No credentials, ever."""

    def search_resolved_binary(
        self, limit: int = 100, sort: str = "most-popular", offset: int = 0
    ) -> list[dict[str, Any]]:
        """List resolved binary markets.

        Args:
            limit: Page size, capped by the venue at 1,000.
            sort: Venue sort order.
            offset: Rows to skip, for paging deeper.
        """
        payload = self.get(
            "/search-markets",
            {
                "filter": "resolved",
                "contractType": "BINARY",
                "limit": min(int(limit), MAX_PAGE_LIMIT),
                "sort": sort,
                "offset": int(offset),
            },
            endpoint="search-markets",
        )
        return payload if isinstance(payload, list) else []

    def market(self, market_id: str) -> dict[str, Any]:
        """Fetch one market's full definition."""
        payload = self.get(f"/market/{market_id}", endpoint="market")
        if not isinstance(payload, dict):
            raise AdapterError(f"unexpected market payload for {market_id}")
        return payload

    def bets(self, market_id: str, max_pages: int = 4) -> list[dict[str, Any]]:
        """Every bet on a market, following the ``before`` cursor."""
        collected: list[dict[str, Any]] = []
        before: str | None = None
        for _ in range(max(1, max_pages)):
            params: dict[str, Any] = {
                "contractId": market_id,
                "limit": MAX_PAGE_LIMIT,
            }
            if before:
                params["before"] = before
            page = self.get("/bets", params, endpoint="bets")
            if not isinstance(page, list) or not page:
                break
            collected.extend(page)
            if len(page) < MAX_PAGE_LIMIT:
                break
            before = str(page[-1].get("id", "")) or None
            if not before:
                break
        return collected


class ManifoldAdapter:
    """Read-only adapter turning Manifold history into a ``MarketDataset``."""

    venue = VENUE

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        transport: Transport | None = None,
        raw_store: RawStore | None = None,
    ) -> None:
        cfg = dict(config or {})
        self.config = cfg
        self.fee_rate = float(cfg.get("fee_rate", 0.0))
        self.store_raw = bool(cfg.get("store_raw_responses", True))
        self.raw_store = raw_store or RawStore(
            venue=VENUE,
            root=cfg.get("raw_dir", "data/raw"),
            enabled=self.store_raw,
        )
        self.client = ManifoldClient(
            base_url=str(cfg.get("base_url", DEFAULT_BASE_URL)),
            transport=transport or urllib_get,
            timeout=float(cfg.get("timeout_seconds", 25.0)),
            min_request_interval=float(cfg.get("min_request_interval_seconds", 0.2)),
            max_retries=int(cfg.get("max_retries", 3)),
            raw_store=self.raw_store,
        )

    def fetch_dataset(self, spec: ManifoldFetchSpec) -> MarketDataset:
        """Fetch and normalise the requested markets.

        Per-market failures are recorded in ``notes`` rather than raised, so one
        bad market cannot lose the rest of the pull.

        Raises:
            AdapterError: If no requested market could be fetched at all.
        """
        markets: list[Market] = []
        quotes: list[Quote] = []
        trades: list[Trade] = []
        notes: list[str] = [
            "Quotes are DERIVED: Manifold is an AMM with no order book. The mid is "
            "the AMM probability; the spread is estimated from observed price impact "
            "and displayed size is a liquidity proxy. Stamped "
            f"{PROXY_SOURCE}.",
            "Manifold is PLAY MONEY. An edge measured here is not tradable; this "
            "corpus exists to prove the pipeline works on real resolved markets.",
        ]

        for market_id in spec.market_ids:
            try:
                payload = self.client.market(market_id)
                market = normalize_market(payload, fee_rate=self.fee_rate)
            except AdapterError as error:
                notes.append(f"{market_id}: skipped ({error})")
                continue

            if not is_usable_binary(payload):
                notes.append(f"{market_id}: skipped (not a resolved binary market)")
                continue

            try:
                raw_bets = self.client.bets(market_id, spec.max_bet_pages)
            except AdapterError as error:
                notes.append(f"{market_id}: no bets ({error})")
                continue

            liquidity = float(payload.get("totalLiquidity") or 100.0)
            market_quotes, market_trades = normalize_bets(market_id, raw_bets, liquidity)
            if not market_quotes:
                notes.append(f"{market_id}: no usable price history")
                continue

            markets.append(market)
            quotes.extend(market_quotes)
            trades.extend(market_trades)

        if not markets:
            raise AdapterError("no Manifold market could be fetched")

        quotes.sort(key=lambda q: (q.ts, q.market_id))
        trades.sort(key=lambda t: (t.ts, t.market_id))
        return MarketDataset(
            markets=markets,
            quotes=quotes,
            trades=trades,
            seed=None,
            label=LABEL,
            source=SOURCE,
            venue=VENUE,
            fetched_at=datetime.now(UTC),
            notes=notes,
        )

    def discover(
        self,
        limit: int = 100,
        offset: int = 0,
        max_lifetime_days: float | None = None,
    ) -> list[dict[str, Any]]:
        """Resolved binary markets that can produce a labelled example.

        ``max_lifetime_days`` filters here rather than after fetching, so
        markets that would only raise concurrency are never pulled at all.
        """
        try:
            listed = self.client.search_resolved_binary(limit=limit, offset=offset)
        except (AdapterError, OSError):
            return []
        return [m for m in listed if is_usable_binary(m, max_lifetime_days)]
