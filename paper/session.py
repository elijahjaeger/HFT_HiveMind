"""The paper-trading loop: live data in, hypothetical orders out.

The session reuses the backtester's components unchanged - the same forecast
agents, the same :class:`agents.risk.RiskAgent`, the same
:func:`backtest.fills.simulate_fill`, the same
:class:`backtest.engine.Portfolio`. The only thing that differs from a backtest
is where the quotes come from and where the results are written.

Execution is deliberately identical to the backtester's: an approved order is
queued and filled on a later poll against the book observed then, never against
the book that triggered the decision.

This class holds no venue write handle. Its only outbound capability is the
quote source it is given, and the Kalshi implementation of that is wrapped in
:class:`paper.guard.ReadOnlyGuard`.
"""

from __future__ import annotations

import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol, runtime_checkable

from agents.base import Consensus, ForecastAgent, build_default_agents, combine_forecasts
from agents.liquidity import LiquidityAgent
from agents.risk import RiskAgent
from agents.strategy import build_proposal
from backtest.engine import PendingOrder, Portfolio, public_market_view
from backtest.fills import simulate_fill
from core.models import (
    Market,
    MarketState,
    Quote,
    RiskDecision,
    SimulatedFill,
    Trade,
    TradeProposal,
)
from paper.store import PaperStore

MODE = "PAPER"


@dataclass(frozen=True)
class LiveSnapshot:
    """One market as observed at a single poll."""

    market: Market
    quote: Quote | None
    trades: list[Trade] = field(default_factory=list)
    fetched_at: datetime | None = None


@runtime_checkable
class QuoteSource(Protocol):
    """Supplies live market data. Read-only by contract and by construction."""

    venue: str

    def fetch(self, tickers: list[str]) -> dict[str, LiveSnapshot]:
        """Return the current snapshot for each ticker that could be read."""
        ...


@dataclass
class PaperConfig:
    """Settings for one paper-trading session."""

    tickers: list[str] = field(default_factory=list)
    poll_interval_seconds: float = 60.0
    execution_delay_seconds: float = 60.0
    initial_capital: float = 10_000.0
    recent_trades_window: int = 20
    decision_interval_seconds: float = 0.0
    """Minimum gap between decisions in one market; 0 means every poll."""


@dataclass
class PollResult:
    """What one poll of the live feed produced."""

    ts: datetime
    markets_seen: int = 0
    quotes: dict[str, Quote] = field(default_factory=dict)
    forecasts: dict[str, Consensus] = field(default_factory=dict)
    proposals: list[TradeProposal] = field(default_factory=list)
    decisions: list[RiskDecision] = field(default_factory=list)
    fills: list[SimulatedFill] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def executed(self) -> list[SimulatedFill]:
        """Fills that moved inventory."""
        return [f for f in self.fills if f.filled_size > 0.0]


class PaperSession:
    """Runs the agent stack against live data and records hypothetical orders.

    Nothing this class does can reach a venue except through ``quote_source``,
    which is read-only. There is no order-submission path to disable, because
    none was ever written.
    """

    mode = MODE

    def __init__(
        self,
        config: PaperConfig,
        settings: dict[str, Any],
        quote_source: QuoteSource,
        store: PaperStore | None = None,
        session_id: str | None = None,
        clock: Any = None,
        forecast_agents: list[ForecastAgent] | None = None,
        risk_agent: RiskAgent | None = None,
    ) -> None:
        if settings.get("app", {}).get("live_trading_enabled", False):
            raise RuntimeError(
                "refusing to start: live_trading_enabled must be False. Paper "
                "mode never submits orders."
            )

        self.config = config
        self.settings = settings
        self.quote_source = quote_source
        self.store = store if store is not None else PaperStore()
        self.session_id = session_id or f"paper-{uuid.uuid4().hex[:12]}"
        self.clock = clock or (lambda: datetime.now(UTC))

        agents_cfg = settings.get("agents", {})
        self.forecast_agents = list(
            forecast_agents
            if forecast_agents is not None
            else build_default_agents(settings)
        )
        self.liquidity_agent = LiquidityAgent(agents_cfg.get("liquidity", {}))
        self.risk_agent = risk_agent or RiskAgent(settings.get("risk", {}))
        self.ensemble_cfg = agents_cfg.get("ensemble", {})
        self.strategy_cfg = settings.get("strategy", {})
        self.fills_cfg = settings.get("fills", {})

        self.portfolio = Portfolio(config.initial_capital)
        self.execution_delay = timedelta(seconds=float(config.execution_delay_seconds))
        self.books: dict[str, Quote] = {}
        self.markets: dict[str, Market] = {}
        self.marks: dict[str, float] = {}
        self.tape: dict[str, deque[Trade]] = {}
        self.pending: list[PendingOrder] = []
        self.last_decision: dict[str, datetime] = {}
        self.polls = 0
        self.counter = 0
        self.started_at = self.clock()

        self.store.record_event(
            self.session_id,
            self.started_at,
            "SESSION_START",
            f"Paper session started on {getattr(quote_source, 'venue', 'unknown')}. "
            "Orders are hypothetical and are never submitted.",
        )

    # -- the loop ----------------------------------------------------------

    def poll(self) -> PollResult:
        """Read the live book once, then decide, execute and persist.

        Returns:
            A :class:`PollResult` describing what this poll produced.
        """
        now = self.clock()
        result = PollResult(ts=now)
        self.polls += 1

        try:
            snapshots = self.quote_source.fetch(list(self.config.tickers))
        except Exception as error:  # noqa: BLE001 - surfaced, never fatal
            message = f"{type(error).__name__}: {error}"
            result.errors.append(message)
            self.store.record_event(self.session_id, now, "FETCH_ERROR", message)
            return result

        # Orders in flight execute against the book observed BEFORE this poll,
        # exactly as the backtester does, so a decision never fills on the
        # quote that triggered it.
        self._execute_pending(now, result)

        for ticker, snapshot in snapshots.items():
            self.markets[ticker] = snapshot.market
            if snapshot.quote is not None:
                self.books[ticker] = snapshot.quote
                result.quotes[ticker] = snapshot.quote
                mark = _mark_price(snapshot.quote)
                if mark is not None:
                    self.marks[ticker] = mark
            window = self.tape.setdefault(
                ticker, deque(maxlen=self.config.recent_trades_window)
            )
            known = {t.trade_id for t in window}
            for trade in snapshot.trades:
                if trade.trade_id not in known:
                    window.append(trade)
        result.markets_seen = len(snapshots)

        for ticker in snapshots:
            self._decide(ticker, now, result)

        snapshot = self.portfolio.snapshot(now, self.marks)
        self.store.record_equity(self.session_id, snapshot)
        return result

    def run(self, polls: int = 1) -> list[PollResult]:
        """Poll ``polls`` times in a row. Pacing is the caller's job."""
        return [self.poll() for _ in range(max(0, polls))]

    def stop(self) -> None:
        """Record the end of the session."""
        self.store.record_event(
            self.session_id,
            self.clock(),
            "SESSION_STOP",
            f"Paper session stopped after {self.polls} poll(s). "
            f"{len(self.pending)} hypothetical order(s) were still in flight.",
        )

    # -- internals ---------------------------------------------------------

    def _decide(self, ticker: str, now: datetime, result: PollResult) -> None:
        """Run the agent chain for one market and queue any approved order."""
        market = self.markets.get(ticker)
        quote = self.books.get(ticker)
        if market is None or quote is None:
            return
        if now >= market.close_time:
            return
        previous = self.last_decision.get(ticker)
        if (
            previous is not None
            and (now - previous).total_seconds() < self.config.decision_interval_seconds
        ):
            return
        self.last_decision[ticker] = now

        state = self._build_state(market, quote, now)
        forecasts = [agent.analyze(state) for agent in self.forecast_agents]
        consensus = combine_forecasts(forecasts, self.ensemble_cfg, ticker)
        result.forecasts[ticker] = consensus

        report = self.liquidity_agent.analyze(state)
        if report.warnings:
            self.store.record_event(
                self.session_id, now, "LIQUIDITY", "; ".join(report.warnings), ticker
            )

        self.counter += 1
        proposal = build_proposal(
            state,
            consensus,
            proposal_id=f"{self.session_id}-P{self.counter:06d}",
            strategy_config=self.strategy_cfg,
            fills_config=self.fills_cfg,
        )
        if proposal is None:
            return

        state.proposal = proposal
        decision = self.risk_agent.evaluate(proposal, state)
        result.proposals.append(proposal)
        result.decisions.append(decision)
        self.store.record_order(
            self.session_id,
            proposal,
            decision,
            venue=getattr(self.quote_source, "venue", ""),
        )

        if decision.is_blocked:
            self.store.record_event(
                self.session_id,
                now,
                "RISK_REJECTED",
                f"{proposal.proposal_id} rejected: "
                + ", ".join(c.value for c in decision.reason_codes),
                ticker,
            )
            return

        self.pending.append(
            PendingOrder(
                proposal=proposal,
                size=decision.approved_size,
                decided_at=now,
                execute_at=now + self.execution_delay,
                sequence=self.counter,
            )
        )
        self.store.record_event(
            self.session_id,
            now,
            "ORDER_QUEUED",
            f"Hypothetical {proposal.side.value} {decision.approved_size:.0f} "
            f"queued for simulated execution. Nothing was submitted to any venue.",
            ticker,
        )

    def _execute_pending(self, now: datetime, result: PollResult) -> None:
        """Simulate any queued order whose delay has elapsed."""
        if not self.pending:
            return
        due = [order for order in self.pending if order.execute_at <= now]
        if not due:
            return
        self.pending = [order for order in self.pending if order.execute_at > now]

        for order in sorted(due, key=lambda o: (o.execute_at, o.sequence)):
            proposal = order.proposal
            ticker = proposal.market_id
            market = self.markets.get(ticker)
            filled_at = order.execute_at

            if market is not None and filled_at >= market.close_time:
                self.store.record_event(
                    self.session_id,
                    filled_at,
                    "ORDER_CANCELLED",
                    f"{proposal.proposal_id} cancelled: the market closed while it "
                    "was in flight.",
                    ticker,
                )
                continue

            fill = simulate_fill(
                quote=self.books.get(ticker),
                side=proposal.side,
                size=order.size,
                ts=filled_at,
                config=self.fills_cfg,
                proposal_id=proposal.proposal_id,
                fill_id=f"{self.session_id}-F{order.sequence:06d}",
                limit_price=proposal.limit_price,
                decided_at=order.decided_at,
            )
            self.portfolio.apply_fill(fill)
            result.fills.append(fill)
            self.store.record_fill(self.session_id, fill)

    def _build_state(self, market: Market, quote: Quote, now: datetime) -> MarketState:
        """Assemble the agent-visible state from what has actually been seen."""
        position = self.portfolio.positions.get(market.market_id)
        return MarketState(
            as_of=now,
            market=public_market_view(market, now),
            quote=quote,
            recent_trades=list(self.tape.get(market.market_id, [])),
            position=position.model_copy() if position else None,
            equity=self.portfolio.equity(self.marks),
            peak_equity=self.portfolio.peak_equity,
            cash=self.portfolio.cash,
            event_exposure=self.portfolio.event_exposure(market.market_id, self.marks),
            total_exposure=self.portfolio.total_exposure(self.marks),
        )

    # -- reporting ---------------------------------------------------------

    def status(self) -> dict[str, Any]:
        """Current session state, for display."""
        return {
            "session_id": self.session_id,
            "mode": self.mode,
            "venue": getattr(self.quote_source, "venue", "unknown"),
            "polls": self.polls,
            "markets_tracked": len(self.markets),
            "orders_in_flight": len(self.pending),
            "cash": self.portfolio.cash,
            "equity": self.portfolio.equity(self.marks),
            "open_positions": sum(
                1 for p in self.portfolio.positions.values() if p.quantity != 0.0
            ),
            "is_hypothetical": True,
        }


def _mark_price(quote: Quote) -> float | None:
    """Best available mark from a quote: mid, else the single visible side."""
    if quote.mid is not None:
        return quote.mid
    return quote.bid if quote.bid is not None else quote.ask
