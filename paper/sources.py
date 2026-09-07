"""Live quote sources for paper trading.

A source is the *only* outbound capability a paper session has. The Kalshi
implementation wraps its transport in :class:`paper.guard.ReadOnlyGuard`, so
even a mistaken URL cannot reach a trading endpoint.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from adapters.base import AdapterError, RawStore, urllib_get
from adapters.kalshi import (
    DEFAULT_BASE_URL,
    VENUE,
    KalshiClient,
    normalize_market,
    normalize_quote_from_market,
    normalize_trade,
)
from core.models import Market, Quote, Trade
from paper.guard import ReadOnlyGuard
from paper.session import LiveSnapshot


class KalshiQuoteSource:
    """Polls Kalshi's public market-data endpoints for the current book.

    Every request goes through a :class:`paper.guard.ReadOnlyGuard`, and the
    underlying client only issues HTTP GET. There is no method on this class,
    or anything it holds, that could submit an order.
    """

    venue = VENUE

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        transport: Any = None,
        raw_store: RawStore | None = None,
        trades_per_poll: int = 20,
    ) -> None:
        cfg = dict(config or {})
        self.config = cfg
        self.trades_per_poll = int(trades_per_poll)
        self.treat_extremes_as_missing = bool(cfg.get("treat_extremes_as_missing", True))
        self.fee_rate = float(cfg.get("fee_rate", 0.07))
        self.include_trades = bool(cfg.get("include_trades", True))

        self.guard = ReadOnlyGuard(transport or urllib_get)
        self.client = KalshiClient(
            base_url=str(cfg.get("base_url", DEFAULT_BASE_URL)),
            transport=self.guard,
            timeout=float(cfg.get("timeout_seconds", 20.0)),
            min_request_interval=float(cfg.get("min_request_interval_seconds", 0.2)),
            max_retries=int(cfg.get("max_retries", 3)),
            raw_store=raw_store
            or RawStore(
                venue="kalshi",
                root=cfg.get("raw_dir", "data/raw"),
                enabled=bool(cfg.get("store_raw_responses", True)),
            ),
        )

    def fetch(self, tickers: list[str]) -> dict[str, LiveSnapshot]:
        """Read the current book and recent tape for each ticker.

        A ticker that cannot be read is skipped rather than failing the poll, so
        one bad market never stops the session.

        Args:
            tickers: Market tickers to poll.

        Returns:
            A snapshot per ticker that responded.
        """
        snapshots: dict[str, LiveSnapshot] = {}
        for ticker in tickers:
            fetched_at = datetime.now(UTC)
            try:
                payload = self.client.market(ticker)
                market: Market = normalize_market(payload, fee_rate=self.fee_rate)
                quote: Quote | None = normalize_quote_from_market(
                    payload,
                    fetched_at=fetched_at,
                    treat_extremes_as_missing=self.treat_extremes_as_missing,
                )
            except AdapterError:
                continue

            trades: list[Trade] = []
            if self.include_trades:
                try:
                    for index, raw in enumerate(
                        self.client.trades(ticker, limit=self.trades_per_poll, max_pages=1)
                    ):
                        trades.append(normalize_trade(raw, index))
                except AdapterError:
                    trades = []
            trades.sort(key=lambda t: t.ts)

            snapshots[ticker] = LiveSnapshot(
                market=market,
                quote=quote,
                trades=trades,
                fetched_at=fetched_at,
            )
        return snapshots

    def audit(self) -> dict[str, Any]:
        """What the read-only guard has seen, for display."""
        return self.guard.audit()


class ReplayQuoteSource:
    """Replays a stored dataset as if it were live, for offline paper runs.

    Useful for exercising the paper loop with no network at all: it hands out
    the snapshot in force at the session's current clock time.
    """

    venue = "REPLAY"

    def __init__(self, markets: list[Market], quotes: list[Quote], trades: list[Trade]) -> None:
        self.markets = {m.market_id: m for m in markets}
        self.quotes = sorted(quotes, key=lambda q: q.ts)
        self.trades = sorted(trades, key=lambda t: t.ts)
        self._now: datetime | None = None

    def advance_to(self, moment: datetime) -> None:
        """Set the instant the next :meth:`fetch` should report."""
        self._now = moment

    def fetch(self, tickers: list[str]) -> dict[str, LiveSnapshot]:
        """Return the snapshot in force at the current replay instant."""
        now = self._now or (self.quotes[-1].ts if self.quotes else datetime.now(UTC))
        snapshots: dict[str, LiveSnapshot] = {}
        for ticker in tickers:
            market = self.markets.get(ticker)
            if market is None:
                continue
            visible = [q for q in self.quotes if q.market_id == ticker and q.ts <= now]
            recent = [t for t in self.trades if t.market_id == ticker and t.ts <= now]
            snapshots[ticker] = LiveSnapshot(
                market=market,
                quote=visible[-1] if visible else None,
                trades=recent[-20:],
                fetched_at=now,
            )
        return snapshots
