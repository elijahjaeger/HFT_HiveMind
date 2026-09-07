"""Performance and forecast-quality metrics.

Every function here is a pure calculation over already-produced backtest
output. Metrics describe what happened on synthetic DEMO DATA in one
simulation; they are not evidence that anything works, and they are not a
projection of future results.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, Field

from core.models import Market, Outcome, SimulatedFill

EPSILON = 1e-12


class CalibrationBucket(BaseModel):
    """One probability bucket of the calibration table."""

    bucket: str
    lower: float
    upper: float
    count: int
    mean_predicted: float | None = None
    observed_frequency: float | None = None

    @property
    def gap(self) -> float | None:
        """Signed calibration error (predicted minus observed)."""
        if self.mean_predicted is None or self.observed_frequency is None:
            return None
        return self.mean_predicted - self.observed_frequency


class MetricsSummary(BaseModel):
    """Headline results of one backtest run."""

    initial_capital: float = 0.0
    final_equity: float = 0.0
    total_pnl: float = 0.0
    total_return: float = 0.0
    max_drawdown: float = 0.0
    n_trades: int = 0
    win_rate: float | None = None
    turnover: float = 0.0
    gross_notional: float = 0.0
    total_fees: float = 0.0
    total_slippage_cost: float = 0.0
    n_proposals: int = 0
    n_approved: int = 0
    n_reduced: int = 0
    n_rejected: int = 0
    n_partial_fills: int = 0
    n_unfilled: int = 0
    n_settlements: int = 0
    n_forecasts_scored: int = 0
    brier_score: float | None = None
    log_loss: float | None = None
    calibration: list[CalibrationBucket] = Field(default_factory=list)


def drawdown_series(equity: Sequence[float]) -> list[float]:
    """Drawdown at each point as a positive fraction of the running peak."""
    out: list[float] = []
    peak = -math.inf
    for value in equity:
        peak = max(peak, value)
        out.append(0.0 if peak <= 0 else max(0.0, (peak - value) / peak))
    return out


def max_drawdown(equity: Sequence[float]) -> float:
    """Largest peak-to-trough decline, as a positive fraction of the peak.

    Args:
        equity: Equity values in chronological order.

    Returns:
        ``0.0`` for an empty or monotonically rising curve.
    """
    series = drawdown_series(equity)
    return max(series) if series else 0.0


def brier_score(probabilities: Sequence[float], outcomes: Sequence[float]) -> float | None:
    """Mean squared error of probabilistic forecasts (lower is better)."""
    pairs = _pairs(probabilities, outcomes)
    if not pairs:
        return None
    return sum((p - y) ** 2 for p, y in pairs) / len(pairs)


def log_loss(
    probabilities: Sequence[float], outcomes: Sequence[float], eps: float = 1e-9
) -> float | None:
    """Mean negative log-likelihood of the forecasts (lower is better)."""
    pairs = _pairs(probabilities, outcomes)
    if not pairs:
        return None
    total = 0.0
    for p, y in pairs:
        clipped = min(max(p, eps), 1.0 - eps)
        total -= y * math.log(clipped) + (1.0 - y) * math.log(1.0 - clipped)
    return total / len(pairs)


def calibration_table(
    probabilities: Sequence[float],
    outcomes: Sequence[float],
    n_buckets: int = 10,
) -> list[CalibrationBucket]:
    """Bucket forecasts by predicted probability and compare to realised frequency.

    Args:
        probabilities: Forecast probabilities in ``[0, 1]``.
        outcomes: Matching realised outcomes encoded as ``1.0``/``0.0``.
        n_buckets: Number of equal-width buckets spanning ``[0, 1]``.

    Returns:
        One :class:`CalibrationBucket` per bucket, always ``n_buckets`` long.
    """
    pairs = _pairs(probabilities, outcomes)
    width = 1.0 / n_buckets
    buckets: list[CalibrationBucket] = []
    for index in range(n_buckets):
        lower = index * width
        upper = lower + width
        if index == n_buckets - 1:
            members = [(p, y) for p, y in pairs if lower <= p <= upper + EPSILON]
        else:
            members = [(p, y) for p, y in pairs if lower <= p < upper]
        if members:
            mean_p = sum(p for p, _ in members) / len(members)
            observed = sum(y for _, y in members) / len(members)
        else:
            mean_p = None
            observed = None
        buckets.append(
            CalibrationBucket(
                bucket=f"[{lower:.1f}, {upper:.1f}{']' if index == n_buckets - 1 else ')'}",
                lower=lower,
                upper=upper,
                count=len(members),
                mean_predicted=mean_p,
                observed_frequency=observed,
            )
        )
    return buckets


def expected_calibration_error(
    probabilities: Sequence[float],
    outcomes: Sequence[float],
    n_buckets: int = 10,
) -> float | None:
    """Weighted mean gap between predicted probability and realised frequency.

    Each bucket contributes ``|mean predicted - observed frequency|`` weighted by
    the share of observations it holds. ``0`` is perfect calibration. Note that
    ECE is sensitive to the bucket count and says nothing about whether the
    forecasts are *informative* - a constant forecast at the base rate scores
    perfectly.

    Args:
        probabilities: Forecast probabilities in ``[0, 1]``.
        outcomes: Matching realised outcomes as ``1.0``/``0.0``.
        n_buckets: Number of equal-width buckets spanning ``[0, 1]``.

    Returns:
        The expected calibration error, or ``None`` without observations.
    """
    pairs = _pairs(probabilities, outcomes)
    if not pairs:
        return None
    total = 0.0
    for bucket in calibration_table(probabilities, outcomes, n_buckets):
        if bucket.count == 0 or bucket.gap is None:
            continue
        total += (bucket.count / len(pairs)) * abs(bucket.gap)
    return total


def fill_pnl(fill: SimulatedFill, outcome: Outcome) -> float:
    """Realised P&L of one fill once its market settles, net of the fee.

    A binary contract pays ``1`` on YES and ``0`` on NO, so a BUY earns
    ``payout - avg_price`` per contract and a SELL earns the negative of that.
    """
    if fill.filled_size <= 0.0 or fill.avg_price is None:
        return -fill.fee
    payout = outcome.payout
    return fill.side.sign * fill.filled_size * (payout - fill.avg_price) - fill.fee


def compute_metrics(result: Any, n_buckets: int = 10) -> MetricsSummary:
    """Compute the full metric set for a :class:`backtest.engine.BacktestResult`.

    Forecast scoring uses the blended probability recorded at each decision
    instant, scored against the market's realised outcome. Note that a market
    contributes one observation per decision, so scores are dominated by markets
    that were evaluated most often - see the README limitations.

    Args:
        result: A completed backtest result.
        n_buckets: Buckets for the calibration table.

    Returns:
        A :class:`MetricsSummary`.
    """
    outcomes: dict[str, Outcome] = {
        m.market_id: m.outcome for m in result.markets if m.outcome is not None
    }

    equity = [snapshot.equity for snapshot in result.snapshots]
    final_equity = result.final_equity or (equity[-1] if equity else result.initial_capital)
    initial = result.initial_capital or (equity[0] if equity else 0.0)
    total_pnl = final_equity - initial

    executed = result.executed_fills
    gross_notional = sum(f.notional for f in executed)
    total_fees = sum(f.fee for f in result.fills)
    slippage_cost = sum(f.slippage * f.filled_size for f in executed)

    wins = 0
    scored_trades = 0
    for fill in executed:
        outcome = outcomes.get(fill.market_id)
        if outcome is None:
            continue
        scored_trades += 1
        if fill_pnl(fill, outcome) > 0.0:
            wins += 1

    probabilities: list[float] = []
    realised: list[float] = []
    for record in result.consensus:
        probability = record.get("probability")
        outcome = outcomes.get(record.get("market_id", ""))
        if probability is None or outcome is None:
            continue
        probabilities.append(float(probability))
        realised.append(1.0 if outcome is Outcome.YES else 0.0)

    actions = [d.action.value for d in result.decisions]
    return MetricsSummary(
        initial_capital=initial,
        final_equity=final_equity,
        total_pnl=total_pnl,
        total_return=total_pnl / initial if initial else 0.0,
        max_drawdown=max_drawdown(equity),
        n_trades=len(executed),
        win_rate=(wins / scored_trades) if scored_trades else None,
        turnover=(gross_notional / initial) if initial else 0.0,
        gross_notional=gross_notional,
        total_fees=total_fees,
        total_slippage_cost=slippage_cost,
        n_proposals=len(result.proposals),
        n_approved=actions.count("APPROVE"),
        n_reduced=actions.count("REDUCE"),
        n_rejected=actions.count("REJECT"),
        n_partial_fills=sum(1 for f in executed if f.is_partial),
        n_unfilled=sum(1 for f in result.fills if f.filled_size <= 0.0),
        n_settlements=len(result.settlements),
        n_forecasts_scored=len(probabilities),
        brier_score=brier_score(probabilities, realised),
        log_loss=log_loss(probabilities, realised),
        calibration=calibration_table(probabilities, realised, n_buckets),
    )


def metrics_rows(summary: MetricsSummary) -> list[dict[str, Any]]:
    """Flatten a summary into ``metric``/``value`` rows for display."""
    def fmt(value: float | None, spec: str) -> str:
        return "n/a" if value is None else format(value, spec)

    return [
        {"metric": "Initial capital", "value": fmt(summary.initial_capital, ",.2f")},
        {"metric": "Final equity", "value": fmt(summary.final_equity, ",.2f")},
        {"metric": "Total P&L", "value": fmt(summary.total_pnl, "+,.2f")},
        {"metric": "Return", "value": fmt(summary.total_return, "+.2%")},
        {"metric": "Max drawdown", "value": fmt(summary.max_drawdown, ".2%")},
        {"metric": "Trades", "value": str(summary.n_trades)},
        {"metric": "Win rate", "value": fmt(summary.win_rate, ".2%")},
        {"metric": "Turnover (x capital)", "value": fmt(summary.turnover, ".2f")},
        {"metric": "Gross notional", "value": fmt(summary.gross_notional, ",.2f")},
        {"metric": "Fees paid", "value": fmt(summary.total_fees, ",.2f")},
        {"metric": "Slippage cost", "value": fmt(summary.total_slippage_cost, ",.2f")},
        {"metric": "Proposals", "value": str(summary.n_proposals)},
        {"metric": "Risk approved", "value": str(summary.n_approved)},
        {"metric": "Risk reduced", "value": str(summary.n_reduced)},
        {"metric": "Risk rejected", "value": str(summary.n_rejected)},
        {"metric": "Partial fills", "value": str(summary.n_partial_fills)},
        {"metric": "Unfilled attempts", "value": str(summary.n_unfilled)},
        {"metric": "Settlements", "value": str(summary.n_settlements)},
        {"metric": "Forecasts scored", "value": str(summary.n_forecasts_scored)},
        {"metric": "Brier score", "value": fmt(summary.brier_score, ".4f")},
        {"metric": "Log loss", "value": fmt(summary.log_loss, ".4f")},
    ]


def _pairs(
    probabilities: Sequence[float], outcomes: Sequence[float]
) -> list[tuple[float, float]]:
    """Zip and validate probability/outcome pairs, dropping missing values."""
    if len(probabilities) != len(outcomes):
        raise ValueError("probabilities and outcomes must have the same length")
    pairs: list[tuple[float, float]] = []
    for p, y in zip(probabilities, outcomes, strict=True):
        if p is None or y is None:
            continue
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"probability out of range: {p}")
        pairs.append((float(p), float(y)))
    return pairs


def market_outcome_map(markets: Sequence[Market]) -> dict[str, Outcome]:
    """Map market id to realised outcome for resolved markets."""
    return {m.market_id: m.outcome for m in markets if m.outcome is not None}
