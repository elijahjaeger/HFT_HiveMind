"""Presentation logic for the desktop app, with no Qt dependency.

Everything the UI shows is computed here as plain data, so the interesting
behaviour can be tested without starting a GUI. ``app/desktop.py`` only turns
these results into widgets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from agents.base import Consensus, build_default_agents, combine_forecasts
from agents.liquidity import LiquidityAgent
from agents.risk import RiskAgent
from agents.strategy import build_proposal
from backtest.engine import public_market_view
from calibration.evaluate import CalibrationReport, evaluate_calibrators
from calibration.splits import chronological_split, observations_from_result
from core.dataset import MarketDataset
from core.models import (
    Forecast,
    LiquidityReport,
    Market,
    MarketState,
    Quote,
    RiskDecision,
    TradeProposal,
)

LIVE_TRADING_BANNER = (
    "LIVE TRADING DISABLED - simulation only. This application has no venue "
    "connection for orders, no credentials, and no order-placement code."
)


def provenance_banner(dataset: MarketDataset) -> str:
    """One line describing where the loaded data came from."""
    if dataset.is_demo:
        return (
            f"DEMO DATA - fabricated by a seeded generator (seed {dataset.seed}). "
            "Not real market data."
        )
    fetched = dataset.fetched_at.strftime("%Y-%m-%d %H:%M UTC") if dataset.fetched_at else "-"
    return (
        f"{dataset.label} - read-only historical data from {dataset.venue}, "
        f"fetched {fetched}. Replayed offline; no orders are ever placed."
    )


@dataclass
class DecisionPreview:
    """Everything the agents produced at one instant for one market."""

    state: MarketState
    forecasts: list[Forecast] = field(default_factory=list)
    consensus: Consensus | None = None
    liquidity: LiquidityReport | None = None
    proposal: TradeProposal | None = None
    decision: RiskDecision | None = None

    @property
    def no_proposal_reason(self) -> str:
        """Why no trade was proposed at this instant."""
        if self.consensus is None or not self.consensus.has_view:
            return "No agent produced a usable probability at this timestamp."
        return (
            "The blended probability is not far enough from the price that would "
            "have to be crossed, or the expected value after fees and slippage is "
            "not positive."
        )


def state_at(
    dataset: MarketDataset,
    market: Market,
    as_of: datetime,
    capital: float = 10_000.0,
    window: int = 20,
) -> MarketState:
    """Build a market state from data at or before ``as_of`` - never after.

    The market runs through the same masking helper the backtester uses, so the
    preview cannot see a resolution the agents would not have known.

    Args:
        dataset: Dataset to read from.
        market: Market to build state for.
        as_of: Decision instant; nothing later is included.
        capital: Simulated capital for the (flat) preview portfolio.
        window: How many recent tape prints to include.

    Returns:
        A :class:`core.models.MarketState` for a flat portfolio.
    """
    quotes = [q for q in dataset.quotes_for(market.market_id) if q.ts <= as_of]
    trades = [t for t in dataset.trades_for(market.market_id) if t.ts <= as_of]
    return MarketState(
        as_of=as_of,
        market=public_market_view(market, as_of),
        quote=quotes[-1] if quotes else None,
        recent_trades=trades[-window:],
        equity=capital,
        peak_equity=capital,
        cash=capital,
    )


def decision_preview(
    dataset: MarketDataset,
    market: Market,
    as_of: datetime,
    settings: dict[str, Any],
) -> DecisionPreview:
    """Run the full agent chain once, for one market at one instant.

    This is the same sequence the backtester runs: forecast agents, blend,
    liquidity report, proposal, then the risk veto.
    """
    capital = float(settings.get("backtest", {}).get("initial_capital", 10_000.0))
    state = state_at(dataset, market, as_of, capital)
    preview = DecisionPreview(state=state)

    preview.forecasts = [agent.analyze(state) for agent in build_default_agents(settings)]
    preview.consensus = combine_forecasts(
        preview.forecasts, settings.get("agents", {}).get("ensemble", {}), market.market_id
    )
    preview.liquidity = LiquidityAgent(
        settings.get("agents", {}).get("liquidity", {})
    ).analyze(state)

    preview.proposal = build_proposal(
        state,
        preview.consensus,
        "PREVIEW",
        settings.get("strategy", {}),
        settings.get("fills", {}),
    )
    if preview.proposal is not None:
        state.proposal = preview.proposal
        preview.decision = RiskAgent(settings.get("risk", {})).evaluate(
            preview.proposal, state
        )
    return preview


def market_rows(dataset: MarketDataset, as_of: datetime) -> list[dict[str, Any]]:
    """One summary row per market, as of ``as_of``."""
    rows: list[dict[str, Any]] = []
    for market in dataset.markets:
        quotes = [q for q in dataset.quotes_for(market.market_id) if q.ts <= as_of]
        quote: Quote | None = quotes[-1] if quotes else None
        rows.append(
            {
                "market_id": market.market_id,
                "question": market.question,
                "category": market.category,
                "bid": quote.bid if quote else None,
                "ask": quote.ask if quote else None,
                "mid": quote.mid if quote else None,
                "spread": quote.spread if quote else None,
                "bid_size": quote.bid_size if quote else None,
                "ask_size": quote.ask_size if quote else None,
                "quote_age_s": round(quote.age_seconds(as_of)) if quote else None,
                "quotes_seen": len(quotes),
                "status": "OPEN" if as_of < market.close_time else "CLOSED",
            }
        )
    return rows


def forecast_rows(preview: DecisionPreview) -> list[dict[str, Any]]:
    """Per-agent forecast rows for the decision panel."""
    return forecast_rows_from(preview.forecasts)


def forecast_rows_from(forecasts: list[Forecast]) -> list[dict[str, Any]]:
    """Per-agent forecast rows, including status and declared input times.

    Agents that declare no inputs leave the ``inputs`` column blank, so this
    renders unchanged for the agents that predate provenance reporting.
    """
    return [
        {
            "agent": f.agent_name,
            "probability": f.probability,
            "confidence": f.confidence,
            "status": getattr(f.status, "value", str(f.status)),
            "inputs": ", ".join(f"{i.name}@{i.ts:%H:%M:%S}" for i in f.inputs),
            "placeholder": "YES - not a forecast" if f.is_placeholder else "",
            "rationale": f.rationale,
        }
        for f in forecasts
    ]


def forecast_provenance_rows(forecasts: list[Forecast]) -> list[dict[str, Any]]:
    """Declared inputs and stated assumptions, one row each.

    Only agents that actually declare provenance appear here. Input rows carry
    the age of the data at the moment the forecast was made, which is what
    makes a look-ahead check readable at a glance.
    """
    rows: list[dict[str, Any]] = []
    for forecast in forecasts:
        for item in forecast.inputs:
            age = (forecast.ts - item.ts).total_seconds()
            rows.append(
                {
                    "agent": forecast.agent_name,
                    "kind": "input",
                    "detail": (
                        f"{item.name} @ {item.ts:%Y-%m-%d %H:%M:%S} UTC "
                        f"(age {age:.0f}s, source {item.source or 'n/a'})"
                    ),
                }
            )
        for assumption in forecast.assumptions:
            rows.append(
                {"agent": forecast.agent_name, "kind": "assumption", "detail": assumption}
            )
    return rows


def risk_reason_rows(decision: RiskDecision | None) -> list[dict[str, Any]]:
    """Machine-readable risk reasons as table rows."""
    if decision is None:
        return []
    return [
        {
            "code": reason.code.value,
            "reason": reason.message,
            "observed": reason.observed,
            "limit": reason.limit,
        }
        for reason in decision.reasons
    ]


def price_series(
    dataset: MarketDataset, market_id: str, as_of: datetime
) -> dict[str, list[tuple[datetime, float]]]:
    """Bid/ask/mid series and tape prints for charting, truncated at ``as_of``.

    Points with a missing side are omitted from that series rather than
    interpolated, so gaps in the chart are real gaps in the data.
    """
    quotes = [q for q in dataset.quotes_for(market_id) if q.ts <= as_of]
    trades = [t for t in dataset.trades_for(market_id) if t.ts <= as_of]
    return {
        "bid": [(q.ts, q.bid) for q in quotes if q.bid is not None],
        "ask": [(q.ts, q.ask) for q in quotes if q.ask is not None],
        "mid": [(q.ts, q.mid) for q in quotes if q.mid is not None],
        "trades": [(t.ts, t.price) for t in trades],
    }


def equity_series(result: Any) -> dict[str, list[tuple[datetime, float]]]:
    """Equity, running peak and drawdown series from a backtest result."""
    snapshots = getattr(result, "snapshots", [])
    return {
        "equity": [(s.ts, s.equity) for s in snapshots],
        "peak": [(s.ts, s.peak_equity) for s in snapshots],
        "drawdown": [(s.ts, -100.0 * s.drawdown) for s in snapshots],
    }


def calibration_rows(result: Any) -> list[dict[str, Any]]:
    """Calibration buckets as table rows."""
    return [
        {
            "bucket": bucket["bucket"],
            "count": bucket["count"],
            "mean_predicted": bucket["mean_predicted"],
            "observed_frequency": bucket["observed_frequency"],
        }
        for bucket in getattr(result, "calibration", [])
    ]


def audit_rows(result: Any, kinds: set[str] | None = None) -> list[dict[str, Any]]:
    """Audit-log rows, optionally filtered to a set of event kinds."""
    rows = []
    for event in getattr(result, "audit_log", []):
        if kinds and event.kind.value not in kinds:
            continue
        rows.append(
            {
                "ts": event.ts,
                "kind": event.kind.value,
                "market_id": event.market_id,
                "message": event.message,
            }
        )
    return rows


def headline_metrics(result: Any) -> list[tuple[str, str]]:
    """The five headline numbers shown above the equity curve."""
    metrics = getattr(result, "metrics", {}) or {}
    win_rate = metrics.get("win_rate")
    return [
        ("Total P&L", f"${metrics.get('total_pnl', 0.0):+,.2f}"),
        ("Return", f"{metrics.get('total_return', 0.0):+.2%}"),
        ("Max drawdown", f"{metrics.get('max_drawdown', 0.0):.2%}"),
        ("Trades", str(metrics.get("n_trades", 0))),
        ("Win rate", "n/a" if win_rate is None else f"{win_rate:.1%}"),
    ]


# --------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------


def calibration_sources(result: Any) -> list[str]:
    """Forecast sources available to calibrate: the blend, then each agent."""
    agents = sorted({f.agent_name for f in getattr(result, "forecasts", [])})
    return ["consensus", *agents]


def run_calibration(
    result: Any,
    source: str = "consensus",
    train_fraction: float = 0.5,
    validation_fraction: float = 0.25,
    n_buckets: int = 10,
    selection_metric: str = "brier",
) -> tuple[CalibrationReport | None, str]:
    """Build a chronological split from a backtest and compare calibrators.

    Args:
        result: A completed backtest result.
        source: ``"consensus"`` or an agent name.
        train_fraction: Share of observations used for fitting.
        validation_fraction: Share used to select between methods.
        n_buckets: Buckets for reliability diagrams and ECE.
        selection_metric: Metric used on the validation fold to pick a winner.

    Returns:
        ``(report, message)``. ``report`` is ``None`` when the split cannot
        support a calibration at all, and ``message`` always explains why.
    """
    observations = observations_from_result(result, source)
    if not observations:
        return None, (
            f"No scored forecasts for {source!r}: calibration needs markets that "
            "have resolved."
        )
    split = chronological_split(
        observations,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
    )
    if not split.train.observations:
        return None, "The training fold is empty; nothing can be fitted."
    if not split.is_usable:
        empty = [f.name for f in split.folds if len(f) == 0]
        return None, (
            f"Fold(s) {', '.join(empty)} are empty after the leakage guard dropped "
            f"{split.total_dropped} observation(s). The markets overlap too much in "
            "time to split chronologically."
        )
    report = evaluate_calibrators(
        split,
        n_buckets=n_buckets,
        selection_metric=selection_metric,
        source=source,
    )
    selected = report.selected()
    message = (
        f"Fitted on {split.train.effective_n} market(s); "
        f"selected {selected.name if selected else 'nothing'} on the validation fold."
    )
    return report, message


def split_rows(report: CalibrationReport) -> list[dict[str, Any]]:
    """Fold summary rows for the split table."""
    return report.split.summary_rows()


# --------------------------------------------------------------------------
# Paper trading
# --------------------------------------------------------------------------


def paper_status_rows(session: Any) -> list[dict[str, Any]]:
    """Session status as label/value rows."""
    if session is None:
        return []
    status = session.status()
    return [
        {"field": key.replace("_", " "), "value": value} for key, value in status.items()
    ]


def paper_poll_summary(result: Any) -> str:
    """One line describing what a poll produced."""
    if result is None:
        return "No polls yet."
    if result.errors:
        return "Poll failed: " + "; ".join(result.errors)
    return (
        f"{result.ts:%H:%M:%S UTC} - read {result.markets_seen} market(s), "
        f"{len(result.proposals)} hypothetical order(s) proposed, "
        f"{len(result.executed)} simulated fill(s). Nothing was submitted."
    )


def paper_frame_rows(frame: Any, limit: int = 200) -> list[dict[str, Any]]:
    """Turn a stored paper-trading frame into display rows, newest first."""
    if frame is None or getattr(frame, "empty", True):
        return []
    return frame.tail(limit).iloc[::-1].to_dict("records")


# --------------------------------------------------------------------------
# Research pipeline
# --------------------------------------------------------------------------

RESEARCH_BANNER = (
    "RESEARCH PIPELINE - models are trained, calibrated and simulated locally. "
    "Nothing here is deployed, and approving a candidate grants a label only: "
    "it starts nothing and can place no order."
)


def readiness_rows(prepared: Any) -> list[dict[str, Any]]:
    """Data-readiness checks, blockers first."""
    return [] if prepared is None else prepared.readiness.as_rows()


def corpus_rows(prepared: Any) -> list[dict[str, Any]]:
    """Where the research corpus came from and what it can support."""
    return [] if prepared is None else prepared.corpus.as_rows()


def dataset_summary_rows(prepared: Any) -> list[dict[str, Any]]:
    """Dataset counts and provenance as field/value rows."""
    if prepared is None:
        return []
    summary = prepared.dataset.summary()
    rows = [
        {"field": "Rows built", "value": summary["rows_total"]},
        {"field": "Rows with a settlement label", "value": summary["rows_labelled"]},
        {"field": "Resolved markets (effective n)", "value": summary["effective_n_markets"]},
        {"field": "Base rate (YES share)",
         "value": None if summary["base_rate"] is None else round(summary["base_rate"], 4)},
        {"field": "Feature version", "value": summary["feature_version"]},
        {"field": "Label policy", "value": summary["label_policy_version"]},
        {"field": "Data version", "value": summary["data_version"]},
    ]
    for status, count in summary["label_status_counts"].items():
        if count:
            rows.append({"field": f"Markets: {status}", "value": count})
    for reason, count in summary["rows_skipped"].items():
        rows.append({"field": f"Rows skipped: {reason}", "value": count})
    return rows


def comparison_rows(outcome: Any) -> list[dict[str, Any]]:
    """Candidate against every frozen baseline, on identical rows."""
    if outcome is None:
        return []
    return [outcome.candidate.as_row()] + [b.as_row() for b in outcome.comparison.baselines]


def gate_rows(outcome: Any) -> list[dict[str, Any]]:
    """The promotion gate, check by check."""
    return [] if outcome is None else outcome.gate.as_rows()


def backtest_metric_rows(result: Any) -> list[dict[str, Any]]:
    """Backtest metrics beside the flat no-trade baseline."""
    if result is None or not result.metrics:
        return []
    metrics = result.metrics
    baseline = result.baseline_metrics
    def fmt(value: Any, spec: str = ",.2f") -> str:
        return "n/a" if value is None else format(value, spec)
    return [
        {"metric": "Forecasts available", "model": str(result.n_forecasts), "flat baseline": "-"},
        {"metric": "Simulated trades", "model": str(metrics.get("n_trades", 0)),
         "flat baseline": str(baseline.get("n_trades", 0))},
        {"metric": "Net P&L", "model": fmt(metrics.get("total_pnl"), "+,.2f"),
         "flat baseline": fmt(baseline.get("total_pnl"), "+,.2f")},
        {"metric": "Return", "model": fmt(metrics.get("total_return"), "+.2%"),
         "flat baseline": fmt(baseline.get("total_return"), "+.2%")},
        {"metric": "Max drawdown", "model": fmt(metrics.get("max_drawdown"), ".2%"),
         "flat baseline": fmt(baseline.get("max_drawdown"), ".2%")},
        {"metric": "Fees paid", "model": fmt(metrics.get("total_fees")), "flat baseline": "0.00"},
        {"metric": "Slippage cost", "model": fmt(metrics.get("total_slippage_cost")),
         "flat baseline": "0.00"},
        {"metric": "Turnover (x capital)", "model": fmt(metrics.get("turnover"), ".2f"),
         "flat baseline": "0.00"},
        {"metric": "Win rate", "model": fmt(metrics.get("win_rate"), ".2%"), "flat baseline": "-"},
        {"metric": "Decisions declined", "model": str(len(result.skips)), "flat baseline": "-"},
    ]


def drift_rows(report: Any) -> list[dict[str, Any]]:
    """Drift measures, most severe first."""
    if report is None:
        return []
    order = {"MATERIAL": 0, "MINOR": 1, "UNKNOWN": 2, "STABLE": 3}
    rows = [m.as_row() for m in report.metrics]
    rows.sort(key=lambda r: order.get(r["level"], 9))
    return rows

