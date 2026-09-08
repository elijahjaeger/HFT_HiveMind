"""Repeated resampled backtests: the distribution, and the honesty around it.

The properties that matter: the band is ordered, a stopped campaign keeps its
work, and the verdict never claims more than resampling can support.
"""

from __future__ import annotations

import random

import pytest

from backtest.campaign import (
    CampaignConfig,
    CampaignResult,
    ReturnBand,
    RunOutcome,
    build_band,
    normalise_curve,
    run_campaign,
    sample_markets,
)
from core.config import get_settings
from core.synthetic import generate_demo_dataset


@pytest.fixture(scope="module")
def settings() -> dict:
    return get_settings()


@pytest.fixture(scope="module")
def dataset(settings):
    return generate_demo_dataset(settings["data"])


# --------------------------------------------------------------------------
# Curve normalisation
# --------------------------------------------------------------------------


def test_curves_are_resampled_onto_a_fixed_grid():
    """Runs differ in length, so curves must share an x-axis to be compared."""
    assert len(normalise_curve([100.0, 110.0, 120.0], 100.0, grid_points=10)) == 10
    assert len(normalise_curve([100.0] * 500, 100.0, grid_points=10)) == 10


def test_equity_becomes_a_return_fraction():
    curve = normalise_curve([100.0, 150.0], 100.0, grid_points=2)
    assert curve[0] == pytest.approx(0.0)
    assert curve[-1] == pytest.approx(0.5)


def test_a_flat_run_is_a_flat_line_at_zero():
    assert all(v == pytest.approx(0.0) for v in normalise_curve([100.0] * 5, 100.0))


def test_an_empty_or_degenerate_curve_does_not_explode():
    assert normalise_curve([], 100.0, grid_points=5) == (0.0,) * 5
    assert normalise_curve([120.0], 100.0, grid_points=3) == pytest.approx((0.2,) * 3)
    assert normalise_curve([100.0, 110.0], 0.0, grid_points=3) == (0.0,) * 3


def test_a_grid_of_one_point_is_refused():
    with pytest.raises(ValueError, match="at least 2"):
        normalise_curve([100.0, 110.0], 100.0, grid_points=1)


# --------------------------------------------------------------------------
# The band
# --------------------------------------------------------------------------


def test_the_band_percentiles_are_ordered():
    rng = random.Random(7)
    curves = [
        tuple(rng.gauss(0.0, 0.05) * step for step in range(10)) for _ in range(50)
    ]
    band = build_band(curves)
    assert band is not None
    for low, mid, high in zip(band.p05, band.median, band.p95, strict=True):
        assert low <= mid <= high


def test_identical_runs_collapse_the_band_to_a_line():
    """A band with no width is repetition, not confidence."""
    band = build_band([(0.0, 0.1, 0.2)] * 20)
    assert band.p05 == band.median == band.p95


def test_curves_on_different_grids_are_refused():
    with pytest.raises(ValueError, match="same grid"):
        build_band([(0.0, 0.1), (0.0, 0.1, 0.2)])


def test_no_curves_yields_no_band():
    assert build_band([]) is None
    assert build_band([()]) is None


def test_a_band_spanning_zero_is_flagged():
    """The single most important read on the chart."""
    straddling = ReturnBand(
        grid=(0.0, 1.0), p05=(0.0, -0.05), p25=(0.0, -0.01),
        median=(0.0, 0.01), p75=(0.0, 0.03), p95=(0.0, 0.08),
    )
    assert straddling.straddles_zero is True

    clear = ReturnBand(
        grid=(0.0, 1.0), p05=(0.0, 0.02), p25=(0.0, 0.03),
        median=(0.0, 0.05), p75=(0.0, 0.07), p95=(0.0, 0.09),
    )
    assert clear.straddles_zero is False


# --------------------------------------------------------------------------
# Sampling
# --------------------------------------------------------------------------


def test_sampling_draws_a_subset_without_replacement(dataset):
    ids = sample_markets(dataset, 0.5, random.Random(1))
    assert len(ids) == len(set(ids))
    assert set(ids) <= {m.market_id for m in dataset.markets}


def test_sampling_is_reproducible_for_a_given_seed(dataset):
    assert sample_markets(dataset, 0.5, random.Random(3)) == sample_markets(
        dataset, 0.5, random.Random(3)
    )


def test_sampling_never_returns_fewer_than_the_minimum(dataset):
    ids = sample_markets(dataset, 0.01, random.Random(1), minimum=3)
    assert len(ids) >= min(3, len(dataset.markets))


def test_sampling_cannot_exceed_the_corpus(dataset):
    ids = sample_markets(dataset, 5.0, random.Random(1))
    assert len(ids) == len(dataset.markets)


# --------------------------------------------------------------------------
# Running a campaign
# --------------------------------------------------------------------------


def test_a_campaign_respects_its_time_budget(dataset, settings):
    result = run_campaign(dataset, settings, CampaignConfig(duration_seconds=3.0))
    assert result.outcomes, "no run completed inside the budget"
    assert result.elapsed < 12.0, "the budget must bound the run, allowing one overrun"


def test_a_campaign_produces_many_runs_not_one(dataset, settings):
    """The whole point: one backtest is one sample and says almost nothing."""
    result = run_campaign(dataset, settings, CampaignConfig(duration_seconds=3.0))
    assert len(result.outcomes) > 1
    assert len(result.returns) == len(result.outcomes)


def test_a_cancelled_campaign_keeps_every_completed_run(dataset, settings):
    seen: list = []

    def stop_after_two() -> bool:
        return len(seen) >= 2

    result = run_campaign(
        dataset, settings, CampaignConfig(duration_seconds=30.0),
        on_progress=seen.append, should_stop=stop_after_two,
    )
    assert result.cancelled is True
    assert result.outcomes, "completed runs must survive a stop"
    assert result.elapsed < 30.0


def test_progress_carries_a_band_once_runs_exist(dataset, settings):
    seen: list = []
    run_campaign(
        dataset, settings, CampaignConfig(duration_seconds=3.0), on_progress=seen.append
    )
    assert seen
    assert seen[-1].runs > 0
    assert seen[-1].band is not None


def test_a_synthetic_corpus_is_flagged_on_the_result(dataset, settings):
    """Fabricated data describes the generator, never a market."""
    result = run_campaign(dataset, settings, CampaignConfig(duration_seconds=2.0))
    assert result.is_synthetic is True
    assert any("FABRICATED" in note for note in result.notes)


def test_a_corpus_too_small_to_resample_says_so(settings):
    tiny = generate_demo_dataset({"n_markets": 3, "n_steps": 20})
    result = run_campaign(
        tiny, settings, CampaignConfig(duration_seconds=1.0, min_markets=99)
    )
    assert result.outcomes == []
    assert any("Too few markets" in note for note in result.notes)


# --------------------------------------------------------------------------
# The verdict - it must never claim more than resampling supports
# --------------------------------------------------------------------------


def _result(returns: list[float], band: ReturnBand | None) -> CampaignResult:
    return CampaignResult(
        outcomes=[
            RunOutcome(index=i, n_markets=5, total_return=r, max_drawdown=0.0, n_trades=1)
            for i, r in enumerate(returns)
        ],
        band=band,
    )


def test_too_few_runs_refuses_to_conclude_anything():
    verdict = _result([0.01, 0.02], None).verdict()
    assert "too few" in verdict.lower()


def test_a_band_over_zero_is_reported_as_not_distinguished():
    band = ReturnBand(
        grid=(0.0, 1.0), p05=(0.0, -0.02), p25=(0.0, 0.0),
        median=(0.0, 0.01), p75=(0.0, 0.02), p95=(0.0, 0.04),
    )
    verdict = _result([0.01] * 10, band).verdict()
    assert "includes zero" in verdict
    assert "NOT been distinguished" in verdict


def test_a_clear_band_still_refuses_to_claim_out_of_sample_validity():
    """Resampling measures robustness to market selection, nothing more."""
    band = ReturnBand(
        grid=(0.0, 1.0), p05=(0.0, 0.02), p25=(0.0, 0.03),
        median=(0.0, 0.05), p75=(0.0, 0.07), p95=(0.0, 0.09),
    )
    verdict = _result([0.05] * 20, band).verdict()
    assert "robustness to market selection" in verdict
    assert "NOT out-of-sample" in verdict


def test_the_win_rate_counts_runs_above_zero():
    result = _result([0.1, -0.1, 0.1, -0.1], None)
    assert result.win_rate == pytest.approx(0.5)


def test_an_empty_campaign_reports_zero_rather_than_dividing_by_zero():
    empty = CampaignResult()
    assert empty.win_rate == 0.0
    assert empty.median_return == 0.0
