"""Event-driven backtest simulator.

Design rules, in order of importance:

1. **Chronology.** Every record is processed in timestamp order and the engine
   raises if a stream is out of order.
2. **No look-ahead.** Agents receive a :class:`core.models.MarketState` built
   only from events already processed, and the market handed to them is a
   masked view with ``outcome`` and ``resolution_time`` stripped. The realised
   outcome is used exclusively for settlement and post-hoc scoring.
3. **Everything is logged.** Proposals, risk decisions, fills, rejections and
   settlements all land in a single audit log.

The simulator never contacts a venue; it replays local DEMO DATA.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import pandas as pd

from agents.base import Consensus, ForecastAgent, build_default_agents, combine_forecasts
from agents.liquidity import LiquidityAgent
from agents.risk import RiskAgent
from agents.strategy import build_proposal, explain_no_proposal
from backtest.fills import simulate_fill
from backtest.skips import SkippedDecision, SkipReason
from core.models import (
    AuditEvent,
    EventKind,
    Forecast,
    LiquidityReport,
    Market,
    MarketState,
    MarketStatus,
    Outcome,
    PortfolioSnapshot,
    Position,
    Quote,
    RiskAction,
    RiskDecision,
    Settlement,
    Side,
    SimulatedFill,
    Trade,
    TradeProposal,
)

QUOTE_PRIORITY, TRADE_PRIORITY, RESOLUTION_PRIORITY, CLOCK_PRIORITY = 0, 1, 2, 3

_SKIP_CODES: dict[str, SkipReason] = {
    "INVALID_DATA": SkipReason.INVALID_DATA,
    "MISSING_FORECAST": SkipReason.MISSING_FORECAST,
    "INSUFFICIENT_EDGE": SkipReason.INSUFFICIENT_EDGE,
    "NEGATIVE_EXPECTED_VALUE": SkipReason.NEGATIVE_EXPECTED_VALUE,
}


@dataclass(frozen=True)
class PendingOrder:
    """An approved order in flight between the decision and its execution."""

    proposal: TradeProposal
    size: float
    decided_at: datetime
    execute_at: datetime
    sequence: int


@dataclass(frozen=True)
class Event:
    """One item in the chronological replay stream."""

    ts: datetime
    priority: int
    seq: int
    kind: str
    payload: Any


def build_event_stream(
    quotes: Sequence[Quote],
    trades: Sequence[Trade],
    markets: Sequence[Market],
    decision_interval_seconds: float = 0.0,
) -> list[Event]:
    """Merge quotes, trades, resolutions and decision ticks into one stream.

    Ties are broken deterministically: quotes are applied first, then trades,
    then resolutions, and decision ticks last - so a decision always sees every
    observation timestamped at that instant, and never a market that has just
    resolved.

    Decision ticks are generated on a fixed clock rather than on data arrival.
    That matters: it is what lets the simulator make a decision while a feed is
    frozen, so stale quotes are actually exercised instead of silently skipped.

    Args:
        quotes: Book snapshots.
        trades: Public tape prints.
        markets: Markets, whose resolution times generate settlement events.
        decision_interval_seconds: Spacing of decision ticks; ``0`` disables them.

    Returns:
        Events sorted by ``(ts, priority, seq)``.
    """
    events: list[Event] = []
    seq = 0
    for quote in quotes:
        events.append(Event(quote.ts, QUOTE_PRIORITY, seq, "QUOTE", quote))
        seq += 1
    for trade in trades:
        events.append(Event(trade.ts, TRADE_PRIORITY, seq, "TRADE", trade))
        seq += 1
    for market in markets:
        if market.outcome is None:
            continue
        ts = market.resolution_time or market.close_time
        events.append(Event(ts, RESOLUTION_PRIORITY, seq, "RESOLUTION", market))
        seq += 1

    if decision_interval_seconds and events:
        start = min(e.ts for e in events)
        end = max(e.ts for e in events)
        tick = start
        step = timedelta(seconds=float(decision_interval_seconds))
        while tick <= end:
            events.append(Event(tick, CLOCK_PRIORITY, seq, "CLOCK", None))
            seq += 1
            tick += step

    events.sort(key=lambda e: (e.ts, e.priority, e.seq))
    return events


def public_market_view(market: Market, as_of: datetime) -> Market:
    """Return the market as an agent may see it at ``as_of``.

    The realised ``outcome`` and ``resolution_time`` are removed and the status
    is recomputed from ``as_of``, so no agent can inspect the future.
    """
    status = MarketStatus.OPEN if as_of < market.close_time else MarketStatus.CLOSED
    return market.model_copy(
        update={"outcome": None, "resolution_time": None, "status": status}
    )


class Portfolio:
    """Simulated account: cash, signed positions and mark-to-market equity."""

    def __init__(self, initial_capital: float = 10_000.0) -> None:
        self.initial_capital = float(initial_capital)
        self.cash = float(initial_capital)
        self.positions: dict[str, Position] = {}
        self.peak_equity = float(initial_capital)
        self.total_fees = 0.0
        self.realized_pnl = 0.0

    def position(self, market_id: str) -> Position:
        """Return (creating if needed) the position for ``market_id``."""
        return self.positions.setdefault(market_id, Position(market_id=market_id))

    def quantity(self, market_id: str) -> float:
        """Signed contract inventory for ``market_id``."""
        pos = self.positions.get(market_id)
        return pos.quantity if pos else 0.0

    def apply_fill(self, fill: SimulatedFill) -> None:
        """Apply a simulated fill to cash and inventory (average-cost basis)."""
        if fill.filled_size <= 0.0 or fill.avg_price is None:
            self.cash -= fill.fee
            self.total_fees += fill.fee
            return

        pos = self.position(fill.market_id)
        price = fill.avg_price
        delta = fill.side.sign * fill.filled_size
        old_qty = pos.quantity

        if old_qty == 0.0 or (old_qty > 0) == (delta > 0):
            total = abs(old_qty) + abs(delta)
            pos.avg_price = (abs(old_qty) * pos.avg_price + abs(delta) * price) / total
            pos.quantity = old_qty + delta
        else:
            closed = min(abs(delta), abs(old_qty))
            direction = 1.0 if old_qty > 0 else -1.0
            realized = closed * (price - pos.avg_price) * direction
            pos.realized_pnl += realized
            self.realized_pnl += realized
            remaining = abs(delta) - closed
            pos.quantity = old_qty + delta
            if remaining > 0.0:
                pos.avg_price = price
            elif pos.quantity == 0.0:
                pos.avg_price = 0.0

        pos.fees_paid += fill.fee
        self.total_fees += fill.fee
        self.cash += fill.cash_flow

    def settle(self, market_id: str, outcome: Outcome, ts: datetime) -> Settlement | None:
        """Cash-settle any open position in ``market_id`` at the binary payout."""
        pos = self.positions.get(market_id)
        if pos is None or pos.quantity == 0.0:
            return None
        payout = outcome.payout
        cash_flow = pos.quantity * payout
        realized = pos.quantity * (payout - pos.avg_price)
        self.cash += cash_flow
        self.realized_pnl += realized
        pos.realized_pnl += realized
        settlement = Settlement(
            market_id=market_id,
            ts=ts,
            outcome=outcome,
            position=pos.quantity,
            payout_per_contract=payout,
            cash_flow=cash_flow,
        )
        pos.quantity = 0.0
        pos.avg_price = 0.0
        return settlement

    def position_value(self, marks: dict[str, float]) -> float:
        """Mark-to-market value of all open positions."""
        return sum(
            pos.quantity * marks.get(mid, pos.avg_price)
            for mid, pos in self.positions.items()
            if pos.quantity != 0.0
        )

    def equity(self, marks: dict[str, float]) -> float:
        """Cash plus mark-to-market position value."""
        return self.cash + self.position_value(marks)

    def event_exposure(self, market_id: str, marks: dict[str, float]) -> float:
        """Worst-case loss on one market at current marks."""
        pos = self.positions.get(market_id)
        if pos is None or pos.quantity == 0.0:
            return 0.0
        return pos.exposure(marks.get(market_id, pos.avg_price))

    def total_exposure(self, marks: dict[str, float]) -> float:
        """Sum of worst-case losses across every open position."""
        return sum(self.event_exposure(mid, marks) for mid in self.positions)

    def snapshot(self, ts: datetime, marks: dict[str, float]) -> PortfolioSnapshot:
        """Record equity, drawdown and exposure at ``ts``."""
        equity = self.equity(marks)
        self.peak_equity = max(self.peak_equity, equity)
        drawdown = (
            (self.peak_equity - equity) / self.peak_equity if self.peak_equity > 0 else 0.0
        )
        return PortfolioSnapshot(
            ts=ts,
            cash=self.cash,
            position_value=self.position_value(marks),
            equity=equity,
            peak_equity=self.peak_equity,
            drawdown=max(0.0, drawdown),
            total_exposure=self.total_exposure(marks),
            open_positions=sum(1 for p in self.positions.values() if p.quantity != 0.0),
        )


@dataclass
class BacktestResult:
    """Everything the simulator produced, ready for metrics and the dashboard."""

    markets: list[Market] = field(default_factory=list)
    forecasts: list[Forecast] = field(default_factory=list)
    liquidity_reports: list[LiquidityReport] = field(default_factory=list)
    consensus: list[dict[str, Any]] = field(default_factory=list)
    proposals: list[TradeProposal] = field(default_factory=list)
    decisions: list[RiskDecision] = field(default_factory=list)
    fills: list[SimulatedFill] = field(default_factory=list)
    settlements: list[Settlement] = field(default_factory=list)
    snapshots: list[PortfolioSnapshot] = field(default_factory=list)
    audit_log: list[AuditEvent] = field(default_factory=list)
    skipped: list[SkippedDecision] = field(default_factory=list)
    """Every decision point that did not become a trade, with its reason."""
    initial_capital: float = 0.0
    final_equity: float = 0.0
    n_events: int = 0
    metrics: dict[str, Any] = field(default_factory=dict)
    calibration: list[dict[str, Any]] = field(default_factory=list)

    @property
    def executed_fills(self) -> list[SimulatedFill]:
        """Fills that actually traded (non-zero size)."""
        return [f for f in self.fills if f.filled_size > 0.0]

    def equity_frame(self) -> pd.DataFrame:
        """Equity curve with drawdown, one row per processed event."""
        if not self.snapshots:
            return pd.DataFrame(
                columns=["ts", "cash", "position_value", "equity", "peak_equity", "drawdown"]
            )
        frame = pd.DataFrame([s.model_dump() for s in self.snapshots])
        frame["ts"] = pd.to_datetime(frame["ts"], utc=True)
        return frame

    def audit_frame(self) -> pd.DataFrame:
        """Audit log as a flat table."""
        if not self.audit_log:
            return pd.DataFrame(columns=["ts", "kind", "market_id", "message"])
        rows = [
            {
                "ts": e.ts,
                "kind": e.kind.value,
                "market_id": e.market_id,
                "message": e.message,
                **{f"p_{k}": v for k, v in e.payload.items()},
            }
            for e in self.audit_log
        ]
        frame = pd.DataFrame(rows)
        frame["ts"] = pd.to_datetime(frame["ts"], utc=True)
        return frame

    def fills_frame(self) -> pd.DataFrame:
        """Every execution attempt, filled or not."""
        if not self.fills:
            return pd.DataFrame(
                columns=["ts", "market_id", "side", "filled_size", "avg_price", "fee"]
            )
        return pd.DataFrame([f.model_dump() for f in self.fills])

    def decisions_frame(self) -> pd.DataFrame:
        """Risk decisions with their reason codes joined into a string."""
        if not self.decisions:
            return pd.DataFrame(columns=["ts", "market_id", "action", "reasons"])
        rows = []
        for decision in self.decisions:
            rows.append(
                {
                    "ts": decision.ts,
                    "market_id": decision.market_id,
                    "proposal_id": decision.proposal_id,
                    "action": decision.action.value,
                    "original_size": decision.original_size,
                    "approved_size": decision.approved_size,
                    "reasons": ", ".join(c.value for c in decision.reason_codes),
                }
            )
        return pd.DataFrame(rows)

    def forecasts_frame(self) -> pd.DataFrame:
        """Per-agent forecasts recorded at every decision instant."""
        if not self.forecasts:
            return pd.DataFrame(columns=["ts", "market_id", "agent_name", "probability"])
        rows = [
            {
                "ts": f.ts,
                "market_id": f.market_id,
                "agent_name": f.agent_name,
                "probability": f.probability,
                "confidence": f.confidence,
                "is_placeholder": f.is_placeholder,
            }
            for f in self.forecasts
        ]
        return pd.DataFrame(rows)

    def consensus_frame(self) -> pd.DataFrame:
        """Blended probability and agent disagreement over time."""
        if not self.consensus:
            return pd.DataFrame(columns=["ts", "market_id", "probability", "disagreement"])
        return pd.DataFrame(self.consensus)


class BacktestEngine:
    """Replays a dataset through the agent stack and records what happened."""

    def __init__(
        self,
        settings: dict[str, Any],
        forecast_agents: Iterable[ForecastAgent] | None = None,
        liquidity_agent: LiquidityAgent | None = None,
        risk_agent: RiskAgent | None = None,
    ) -> None:
        self.settings = settings
        agents_cfg = settings.get("agents", {})
        self.forecast_agents: list[ForecastAgent] = list(
            forecast_agents if forecast_agents is not None else build_default_agents(settings)
        )
        self.liquidity_agent = liquidity_agent or LiquidityAgent(
            agents_cfg.get("liquidity", {})
        )
        self.risk_agent = risk_agent or RiskAgent(settings.get("risk", {}))
        self.ensemble_cfg = agents_cfg.get("ensemble", {})
        self.strategy_cfg = settings.get("strategy", {})
        self.fills_cfg = settings.get("fills", {})
        backtest_cfg = settings.get("backtest", {})
        self.initial_capital = float(backtest_cfg.get("initial_capital", 10_000.0))
        self.decision_interval = float(backtest_cfg.get("decision_interval_seconds", 600))
        self.trade_window = int(backtest_cfg.get("recent_trades_window", 20))
        # Orders do not execute at the instant they are decided. This is the
        # gap between the decision and the order reaching the book.
        self.execution_delay = timedelta(
            seconds=float(backtest_cfg.get("execution_delay_seconds", 60.0))
        )
        self.record_every_event = bool(backtest_cfg.get("record_equity_every_event", True))

    def run(
        self,
        markets: Sequence[Market],
        quotes: Sequence[Quote],
        trades: Sequence[Trade],
    ) -> BacktestResult:
        """Replay the dataset chronologically and return the full result set.

        Args:
            markets: Markets, including their realised outcomes (used only for
                settlement and post-hoc scoring).
            quotes: Book snapshots.
            trades: Public tape prints.

        Returns:
            A populated :class:`BacktestResult`.

        Raises:
            ValueError: If the merged event stream is not chronological.
        """
        result = BacktestResult(markets=list(markets), initial_capital=self.initial_capital)
        portfolio = Portfolio(self.initial_capital)
        by_id = {m.market_id: m for m in markets}
        books: dict[str, Quote] = {}
        tape: dict[str, deque[Trade]] = {
            m.market_id: deque(maxlen=self.trade_window) for m in markets
        }
        marks: dict[str, float] = {}
        last_decision: dict[str, datetime] = {}
        resolved: set[str] = set()
        counter = 0
        previous_ts: datetime | None = None
        pending: list[PendingOrder] = []

        events = build_event_stream(quotes, trades, markets, self.decision_interval)
        result.n_events = len(events)

        for event in events:
            if previous_ts is not None and event.ts < previous_ts:
                raise ValueError(
                    f"event stream is not chronological: {event.ts} after {previous_ts}"
                )
            previous_ts = event.ts

            # Orders in flight execute FIRST, against the book in force
            # immediately before this event is applied. Draining here rather
            # than after the quote update matters: an order that arrives at the
            # same instant a new quote is published must not get the new price,
            # and it can never see a book published after it arrived.
            self._execute_pending(
                pending, event.ts, books, by_id, resolved, portfolio, result
            )

            if event.kind == "QUOTE":
                quote: Quote = event.payload
                books[quote.market_id] = quote
                mark = _mark_price(quote)
                if mark is not None:
                    marks[quote.market_id] = mark
            elif event.kind == "CLOCK":
                for market_id, market in by_id.items():
                    if market_id in resolved:
                        continue
                    quote = books.get(market_id)
                    if quote is None:
                        continue  # No book observed yet; nothing to analyse.
                    counter = self._maybe_decide(
                        market=market,
                        as_of=event.ts,
                        quote=quote,
                        tape=tape,
                        portfolio=portfolio,
                        marks=marks,
                        last_decision=last_decision,
                        result=result,
                        counter=counter,
                        pending=pending,
                    )
            elif event.kind == "TRADE":
                trade: Trade = event.payload
                tape.setdefault(trade.market_id, deque(maxlen=self.trade_window)).append(trade)
            elif event.kind == "RESOLUTION":
                market = event.payload
                self._settle(market, event.ts, portfolio, marks, resolved, result)

            if self.record_every_event or event.kind == "RESOLUTION":
                result.snapshots.append(portfolio.snapshot(event.ts, marks))

        result.final_equity = portfolio.equity(marks)
        return result

    # -- internals ----------------------------------------------------------

    def _maybe_decide(
        self,
        market: Market,
        as_of: datetime,
        quote: Quote,
        tape: dict[str, deque[Trade]],
        portfolio: Portfolio,
        marks: dict[str, float],
        last_decision: dict[str, datetime],
        result: BacktestResult,
        counter: int,
        pending: list[PendingOrder],
    ) -> int:
        """Run one decision cycle for a market if the cadence allows it."""
        if as_of >= market.close_time:
            return counter
        previous = last_decision.get(market.market_id)
        if previous is not None and (as_of - previous).total_seconds() < self.decision_interval:
            return counter
        last_decision[market.market_id] = as_of

        state = self._build_state(market, as_of, quote, tape, portfolio, marks)

        forecasts = [agent.analyze(state) for agent in self.forecast_agents]
        result.forecasts.extend(forecasts)
        for forecast in forecasts:
            result.audit_log.append(
                AuditEvent(
                    ts=as_of,
                    kind=EventKind.FORECAST,
                    market_id=market.market_id,
                    message=f"{forecast.agent_name} p={_fmt(forecast.probability)}",
                    payload={
                        "agent": forecast.agent_name,
                        "probability": forecast.probability,
                        "placeholder": forecast.is_placeholder,
                    },
                )
            )

        report = self.liquidity_agent.analyze(state)
        result.liquidity_reports.append(report)
        if report.warnings:
            result.audit_log.append(
                AuditEvent(
                    ts=as_of,
                    kind=EventKind.LIQUIDITY,
                    market_id=market.market_id,
                    message="; ".join(report.warnings),
                    payload={"spread": report.spread, "depth": report.depth},
                )
            )

        consensus: Consensus = combine_forecasts(
            forecasts, self.ensemble_cfg, market.market_id
        )
        result.consensus.append(
            {
                "ts": as_of,
                "market_id": market.market_id,
                "probability": consensus.probability,
                "disagreement": consensus.disagreement,
                "stdev": consensus.stdev,
                "n_contributors": consensus.n_contributors,
            }
        )

        counter += 1
        proposal = build_proposal(
            state,
            consensus,
            proposal_id=f"P{counter:06d}",
            strategy_config=self.strategy_cfg,
            fills_config=self.fills_cfg,
        )
        if proposal is None:
            detail = explain_no_proposal(
                state, consensus, self.strategy_cfg, self.fills_cfg
            )
            code, _, message = detail.partition(": ")
            result.skipped.append(
                SkippedDecision(
                    ts=as_of,
                    market_id=market.market_id,
                    reason=_SKIP_CODES.get(code, SkipReason.INSUFFICIENT_EDGE),
                    detail=message or detail,
                    forecast_probability=consensus.probability,
                    market_probability=quote.mid,
                )
            )
            return counter

        result.proposals.append(proposal)
        result.audit_log.append(
            AuditEvent(
                ts=as_of,
                kind=EventKind.PROPOSAL,
                market_id=market.market_id,
                message=(
                    f"{proposal.side.value} {proposal.size:.0f} @ {proposal.reference_price:.3f} "
                    f"(fair {proposal.fair_probability:.3f}, "
                    f"EV {proposal.expected_value_total:+.2f})"
                ),
                payload={
                    "proposal_id": proposal.proposal_id,
                    "side": proposal.side.value,
                    "size": proposal.size,
                    "edge": proposal.edge,
                    "ev_total": proposal.expected_value_total,
                },
            )
        )

        state.proposal = proposal
        decision = self.risk_agent.evaluate(proposal, state)
        result.decisions.append(decision)
        codes = ", ".join(c.value for c in decision.reason_codes)
        result.audit_log.append(
            AuditEvent(
                ts=as_of,
                kind=EventKind.RISK_DECISION,
                market_id=market.market_id,
                message=(
                    f"{decision.action.value} {decision.approved_size:.0f}/"
                    f"{decision.original_size:.0f} [{codes}]"
                ),
                payload={
                    "proposal_id": proposal.proposal_id,
                    "action": decision.action.value,
                    "approved_size": decision.approved_size,
                    "reasons": ",".join(c.value for c in decision.reason_codes),
                },
            )
        )

        if decision.is_blocked:
            result.skipped.append(
                SkippedDecision(
                    ts=as_of,
                    market_id=market.market_id,
                    reason=SkipReason.RISK_CAP,
                    detail=", ".join(c.value for c in decision.reason_codes),
                    forecast_probability=consensus.probability,
                    market_probability=quote.mid,
                )
            )
            result.audit_log.append(
                AuditEvent(
                    ts=as_of,
                    kind=EventKind.REJECTION,
                    market_id=market.market_id,
                    message=f"Proposal {proposal.proposal_id} rejected by risk.",
                    payload={
                        "proposal_id": proposal.proposal_id,
                        "reasons": ",".join(c.value for c in decision.reason_codes),
                    },
                )
            )
            return counter

        pending.append(
            PendingOrder(
                proposal=proposal,
                size=decision.approved_size,
                decided_at=as_of,
                execute_at=as_of + self.execution_delay,
                sequence=counter,
            )
        )
        return counter

    def _execute_pending(
        self,
        pending: list[PendingOrder],
        now: datetime,
        books: dict[str, Quote],
        by_id: dict[str, Market],
        resolved: set[str],
        portfolio: Portfolio,
        result: BacktestResult,
    ) -> None:
        """Execute every queued order whose delay has elapsed.

        The book used is the latest one observed *strictly before* ``now``, so a
        delayed order can only ever be filled with information that already
        existed when it arrived.

        The configured delay is a minimum, not an exact latency: an order can
        only be executed when the simulator next wakes, so the realised latency
        is quantised up to the next event in the stream. Orders that would land
        at or after their market closed are cancelled rather than filled.
        """
        if not pending:
            return
        due = [order for order in pending if order.execute_at <= now]
        if not due:
            return
        pending[:] = [order for order in pending if order.execute_at > now]

        for order in sorted(due, key=lambda o: (o.execute_at, o.sequence)):
            proposal = order.proposal
            market = by_id.get(proposal.market_id)
            market_id = proposal.market_id
            # The order arrived at execute_at and is filled against the book in
            # force then, so that is the fill's timestamp. Nothing observable
            # happened between execute_at and now, which is why the simulator
            # only gets to it now.
            filled_at = order.execute_at

            if market is None or market_id in resolved or filled_at >= market.close_time:
                result.audit_log.append(
                    AuditEvent(
                        ts=filled_at,
                        kind=EventKind.REJECTION,
                        market_id=market_id,
                        message=(
                            f"Order {proposal.proposal_id} cancelled: the market closed "
                            "while it was in flight."
                        ),
                        payload={
                            "proposal_id": proposal.proposal_id,
                            "reason": "MARKET_CLOSED_IN_FLIGHT",
                        },
                    )
                )
                continue

            fill = simulate_fill(
                quote=books.get(market_id),
                side=proposal.side,
                size=order.size,
                ts=filled_at,
                config=self.fills_cfg,
                proposal_id=proposal.proposal_id,
                fill_id=f"F{order.sequence:06d}",
                limit_price=proposal.limit_price,
                decided_at=order.decided_at,
            )
            result.fills.append(fill)
            portfolio.apply_fill(fill)

            if fill.filled_size > 0.0:
                result.audit_log.append(
                    AuditEvent(
                        ts=filled_at,
                        kind=EventKind.FILL,
                        market_id=market_id,
                        message=(
                            f"{fill.side.value} {fill.filled_size:.0f} @ "
                            f"{fill.avg_price:.4f} (slippage {fill.slippage:.4f}, "
                            f"fee {fill.fee:.2f}, latency "
                            f"{fill.latency_seconds:.0f}s)"
                            + (" PARTIAL" if fill.is_partial else "")
                        ),
                        payload={
                            "proposal_id": proposal.proposal_id,
                            "filled_size": fill.filled_size,
                            "avg_price": fill.avg_price,
                            "fee": fill.fee,
                            "partial": fill.is_partial,
                            "latency_seconds": fill.latency_seconds,
                        },
                    )
                )
            else:
                result.skipped.append(
                    SkippedDecision(
                        ts=filled_at,
                        market_id=market_id,
                        reason=SkipReason.EXECUTION_CONSTRAINT,
                        detail=fill.unfilled_reason or "no fill",
                    )
                )
                result.audit_log.append(
                    AuditEvent(
                        ts=filled_at,
                        kind=EventKind.REJECTION,
                        market_id=market_id,
                        message=f"No fill: {fill.unfilled_reason}",
                        payload={
                            "proposal_id": proposal.proposal_id,
                            "reason": fill.unfilled_reason,
                        },
                    )
                )

    def _build_state(
        self,
        market: Market,
        as_of: datetime,
        quote: Quote,
        tape: dict[str, deque[Trade]],
        portfolio: Portfolio,
        marks: dict[str, float],
    ) -> MarketState:
        """Assemble the agent-visible state from already-processed data only."""
        position = portfolio.positions.get(market.market_id)
        return MarketState(
            as_of=as_of,
            market=public_market_view(market, as_of),
            quote=quote,
            recent_trades=list(tape.get(market.market_id, [])),
            position=position.model_copy() if position else None,
            equity=portfolio.equity(marks),
            peak_equity=portfolio.peak_equity,
            cash=portfolio.cash,
            event_exposure=portfolio.event_exposure(market.market_id, marks),
            total_exposure=portfolio.total_exposure(marks),
        )

    def _settle(
        self,
        market: Market,
        ts: datetime,
        portfolio: Portfolio,
        marks: dict[str, float],
        resolved: set[str],
        result: BacktestResult,
    ) -> None:
        """Settle a resolving market and log the cash flow."""
        if market.outcome is None:
            return
        resolved.add(market.market_id)
        marks[market.market_id] = market.outcome.payout
        settlement = portfolio.settle(market.market_id, market.outcome, ts)
        if settlement is None:
            result.audit_log.append(
                AuditEvent(
                    ts=ts,
                    kind=EventKind.SETTLEMENT,
                    market_id=market.market_id,
                    message=f"Resolved {market.outcome.value}; no open position.",
                    payload={"outcome": market.outcome.value, "position": 0.0},
                )
            )
            return
        result.settlements.append(settlement)
        result.audit_log.append(
            AuditEvent(
                ts=ts,
                kind=EventKind.SETTLEMENT,
                market_id=market.market_id,
                message=(
                    f"Resolved {market.outcome.value}; settled {settlement.position:+.0f} "
                    f"contracts for {settlement.cash_flow:+.2f}."
                ),
                payload={
                    "outcome": market.outcome.value,
                    "position": settlement.position,
                    "cash_flow": settlement.cash_flow,
                },
            )
        )


def _mark_price(quote: Quote) -> float | None:
    """Best available mark from a quote: mid, else the single visible side."""
    if quote.mid is not None:
        return quote.mid
    if quote.bid is not None:
        return quote.bid
    return quote.ask


def _fmt(value: float | None) -> str:
    """Format an optional probability for log messages."""
    return "n/a" if value is None or math.isnan(value) else f"{value:.3f}"


def run_backtest(
    markets: Sequence[Market],
    quotes: Sequence[Quote],
    trades: Sequence[Trade],
    settings: dict[str, Any],
    forecast_agents: Iterable[ForecastAgent] | None = None,
    liquidity_agent: LiquidityAgent | None = None,
    risk_agent: RiskAgent | None = None,
    with_metrics: bool = True,
) -> BacktestResult:
    """Convenience wrapper: build an engine, run it and attach metrics.

    Args:
        markets: Markets to replay.
        quotes: Book snapshots.
        trades: Public tape prints.
        settings: Full settings dictionary.
        forecast_agents: Override the default forecast agents.
        liquidity_agent: Override the default liquidity agent.
        risk_agent: Override the default risk agent.
        with_metrics: Compute and attach performance metrics.

    Returns:
        A populated :class:`BacktestResult`.
    """
    engine = BacktestEngine(settings, forecast_agents, liquidity_agent, risk_agent)
    result = engine.run(markets, quotes, trades)
    if with_metrics:
        from backtest.metrics import compute_metrics

        summary = compute_metrics(result)
        result.metrics = summary.model_dump()
        result.calibration = [bucket.model_dump() for bucket in summary.calibration]
    return result


__all__ = [
    "BacktestEngine",
    "BacktestResult",
    "Event",
    "Portfolio",
    "build_event_stream",
    "public_market_view",
    "run_backtest",
    "RiskAction",
    "Side",
]
