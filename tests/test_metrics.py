"""Drawdown, P&L attribution and forecast-quality scoring."""

from __future__ import annotations

import math

import pytest

from backtest.metrics import (
    brier_score,
    calibration_table,
    compute_metrics,
    drawdown_series,
    fill_pnl,
    log_loss,
    max_drawdown,
    metrics_rows,
)
from core.models import Outcome, Side, SimulatedFill
from tests.conftest import T0


def test_max_drawdown_of_a_rising_curve_is_zero():
    assert max_drawdown([100.0, 110.0, 120.0]) == 0.0


def test_max_drawdown_measures_peak_to_trough():
    # Peak 120, trough 90 => 25%.
    assert max_drawdown([100.0, 120.0, 90.0, 110.0]) == pytest.approx(0.25)


def test_max_drawdown_uses_the_worst_decline_not_the_last():
    assert max_drawdown([100.0, 50.0, 100.0, 80.0]) == pytest.approx(0.5)


def test_max_drawdown_of_an_empty_curve_is_zero():
    assert max_drawdown([]) == 0.0


def test_drawdown_series_tracks_the_running_peak():
    assert drawdown_series([100.0, 90.0, 100.0]) == pytest.approx([0.0, 0.1, 0.0])


def test_brier_score_of_a_perfect_forecast_is_zero():
    assert brier_score([1.0, 0.0], [1.0, 0.0]) == pytest.approx(0.0)


def test_brier_score_of_a_coin_flip_is_a_quarter():
    assert brier_score([0.5, 0.5], [1.0, 0.0]) == pytest.approx(0.25)


def test_brier_score_is_none_without_observations():
    assert brier_score([], []) is None


def test_log_loss_punishes_confident_mistakes():
    confident_and_right = log_loss([0.99], [1.0])
    confident_and_wrong = log_loss([0.99], [0.0])
    assert confident_and_right < 0.02 < confident_and_wrong


def test_log_loss_is_finite_at_the_boundaries():
    assert math.isfinite(log_loss([0.0], [1.0]))


def test_mismatched_lengths_are_rejected():
    with pytest.raises(ValueError):
        brier_score([0.5], [1.0, 0.0])


def test_out_of_range_probabilities_are_rejected():
    with pytest.raises(ValueError):
        brier_score([1.5], [1.0])


def test_calibration_buckets_group_forecasts_by_probability():
    table = calibration_table([0.05, 0.15, 0.95], [0.0, 1.0, 1.0], n_buckets=10)
    assert len(table) == 10
    assert table[0].count == 1
    assert table[1].count == 1
    assert table[9].count == 1
    assert table[9].mean_predicted == pytest.approx(0.95)
    assert table[9].observed_frequency == pytest.approx(1.0)


def test_empty_calibration_buckets_report_no_values():
    table = calibration_table([0.05], [1.0], n_buckets=10)
    assert table[5].count == 0
    assert table[5].mean_predicted is None
    assert table[5].gap is None


def test_calibration_gap_is_predicted_minus_observed():
    table = calibration_table([0.75, 0.75], [1.0, 0.0], n_buckets=10)
    bucket = table[7]
    assert bucket.observed_frequency == pytest.approx(0.5)
    assert bucket.gap == pytest.approx(0.25)


def _fill(side: Side, size: float, price: float, fee: float = 0.0) -> SimulatedFill:
    return SimulatedFill(
        fill_id="F1",
        proposal_id="P1",
        market_id="M1",
        ts=T0,
        side=side,
        requested_size=size,
        filled_size=size,
        reference_price=price,
        avg_price=price,
        fee=fee,
    )


def test_a_long_wins_when_the_market_resolves_yes():
    assert fill_pnl(_fill(Side.BUY, 100.0, 0.40), Outcome.YES) == pytest.approx(60.0)
    assert fill_pnl(_fill(Side.BUY, 100.0, 0.40), Outcome.NO) == pytest.approx(-40.0)


def test_a_short_wins_when_the_market_resolves_no():
    assert fill_pnl(_fill(Side.SELL, 100.0, 0.40), Outcome.NO) == pytest.approx(40.0)
    assert fill_pnl(_fill(Side.SELL, 100.0, 0.40), Outcome.YES) == pytest.approx(-60.0)


def test_fees_reduce_realised_pnl():
    assert fill_pnl(_fill(Side.BUY, 100.0, 0.40, fee=5.0), Outcome.YES) == pytest.approx(55.0)


class _Result:
    """Minimal stand-in for a BacktestResult."""

    def __init__(self, markets, snapshots, fills, consensus):
        self.markets = markets
        self.snapshots = snapshots
        self.fills = fills
        self.consensus = consensus
        self.proposals = []
        self.decisions = []
        self.settlements = []
        self.initial_capital = 1_000.0
        self.final_equity = snapshots[-1].equity if snapshots else 1_000.0

    @property
    def executed_fills(self):
        return [f for f in self.fills if f.filled_size > 0]


def test_compute_metrics_summarises_a_run():
    from core.models import PortfolioSnapshot
    from tests.conftest import make_market

    market = make_market(outcome=Outcome.YES)
    snapshots = [
        PortfolioSnapshot(
            ts=T0, cash=1_000.0, position_value=0.0, equity=equity,
            peak_equity=max(1_000.0, equity), drawdown=0.0, total_exposure=0.0,
            open_positions=0,
        )
        for equity in (1_000.0, 1_200.0, 900.0, 1_100.0)
    ]
    result = _Result(
        markets=[market],
        snapshots=snapshots,
        fills=[_fill(Side.BUY, 100.0, 0.40, fee=1.0)],
        consensus=[{"ts": T0, "market_id": market.market_id, "probability": 0.6}],
    )
    summary = compute_metrics(result)

    assert summary.total_pnl == pytest.approx(100.0)
    assert summary.total_return == pytest.approx(0.10)
    assert summary.max_drawdown == pytest.approx(0.25)
    assert summary.n_trades == 1
    assert summary.win_rate == pytest.approx(1.0)
    assert summary.turnover == pytest.approx(40.0 / 1_000.0)
    assert summary.brier_score == pytest.approx(0.16)
    assert summary.n_forecasts_scored == 1
    assert len(summary.calibration) == 10


def test_metrics_rows_are_display_ready():
    from backtest.metrics import MetricsSummary

    rows = metrics_rows(MetricsSummary(initial_capital=1_000.0))
    assert {"metric", "value"} == set(rows[0])
    assert any(row["metric"] == "Win rate" and row["value"] == "n/a" for row in rows)
