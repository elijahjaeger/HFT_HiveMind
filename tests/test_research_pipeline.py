"""Pipeline orchestration, scoring, drift, and reproducibility end to end."""

from __future__ import annotations

import pytest

from core.config import load_settings
from research.corpus import build_research_corpus, describe, max_concurrent_markets
from research.drift import (
    DriftLevel,
    population_stability_index,
    quality_drift,
)
from research.evaluate import expected_calibration_error, reliability_points, score
from research.pipeline import check_drift, prepare_data, run_model_backtest, train_and_evaluate
from research.registry import ResearchRegistry

# Keeps the shipped corpus shape (markets must outlive the 1-hour horizon) but
# uses fewer markets so the suite stays quick.
SMALL_CORPUS = {"n_markets": 60, "n_steps": 4000, "lifetime_fraction": 0.03,
                "stagger_fraction": 0.95, "quote_interval_seconds": 300, "seed": 5}


@pytest.fixture(scope="module")
def settings():
    return load_settings(use_env=False)


@pytest.fixture(scope="module")
def corpus():
    return build_research_corpus(SMALL_CORPUS)


@pytest.fixture(scope="module")
def prepared(corpus, settings):
    return prepare_data(corpus, settings)


@pytest.fixture(scope="module")
def outcome(prepared, settings):
    return train_and_evaluate(prepared, settings, "logistic_regression")


# -- the corpus ------------------------------------------------------------


def test_the_research_corpus_is_labelled_as_fabricated(corpus):
    assert corpus.is_demo is True
    assert "DEMO" in corpus.label.upper()
    info = describe(corpus)
    assert info.is_synthetic is True
    assert "fabricated" in info.caveat.lower()
    assert "tells you nothing about real markets" in info.caveat


def test_the_corpus_is_reproducible_from_its_seed():
    a = build_research_corpus(SMALL_CORPUS)
    b = build_research_corpus(SMALL_CORPUS)
    assert [m.market_id for m in a.markets] == [m.market_id for m in b.markets]
    assert [q.mid for q in a.quotes] == [q.mid for q in b.quotes]


def test_a_different_seed_gives_a_different_corpus():
    other = build_research_corpus({**SMALL_CORPUS, "seed": 99})
    base = build_research_corpus(SMALL_CORPUS)
    assert [q.mid for q in other.quotes] != [q.mid for q in base.quotes]


def test_the_corpus_reports_how_many_markets_overlap(corpus):
    """Concurrency is what makes purging necessary, so it is measured."""
    assert max_concurrent_markets(corpus) >= 1


# -- stage 1: data preparation ---------------------------------------------


def test_preparation_produces_labelled_examples(prepared):
    assert prepared.dataset.labelled
    assert prepared.dataset.effective_n > 0
    assert prepared.dataset.data_version.startswith("data-")


def test_preparation_reports_readiness(prepared):
    assert prepared.readiness is not None
    assert isinstance(prepared.readiness.ok, bool)
    assert prepared.readiness.effective_n == prepared.dataset.effective_n


def test_preparation_records_a_run_in_the_registry(tmp_path, corpus, settings):
    registry = ResearchRegistry(tmp_path / "reg.duckdb")
    prepare_data(corpus, settings, registry)
    assert not registry.runs().empty


# -- stage 2: training and scoring -----------------------------------------


def test_training_scores_the_candidate_against_the_frozen_baseline(outcome):
    assert outcome.candidate is not None
    assert outcome.baseline is not None
    assert outcome.baseline.model_name == "market_price_baseline"
    assert outcome.comparison is not None


def test_the_candidate_is_scored_out_of_sample_only(outcome):
    assert outcome.report.usable_folds
    assert outcome.candidate.effective_n <= outcome.report.effective_n


def test_scores_carry_the_sample_size_they_rest_on(outcome):
    assert outcome.candidate.effective_n > 0
    row = outcome.candidate.as_row()
    assert "markets (effective n)" in row


def test_brier_and_log_loss_are_reported(outcome):
    assert outcome.candidate.brier is not None
    assert outcome.candidate.log_loss is not None
    assert 0.0 <= outcome.candidate.brier <= 1.0


def test_a_perfect_forecast_scores_zero_brier():
    card = score("perfect", [1.0, 0.0, 1.0], [1.0, 0.0, 1.0], effective_n=3)
    assert card.brier == pytest.approx(0.0)


def test_a_coin_flip_forecast_scores_a_quarter():
    card = score("coinflip", [0.5, 0.5], [1.0, 0.0], effective_n=2)
    assert card.brier == pytest.approx(0.25)


def test_expected_calibration_error_is_zero_for_honest_forecasts():
    probabilities = [0.0] * 50 + [1.0] * 50
    outcomes = [0.0] * 50 + [1.0] * 50
    assert expected_calibration_error(probabilities, outcomes) == pytest.approx(0.0)


def test_expected_calibration_error_catches_overconfidence():
    """Always says 90%, is right half the time."""
    probabilities = [0.9] * 100
    outcomes = [1.0] * 50 + [0.0] * 50
    assert expected_calibration_error(probabilities, outcomes) == pytest.approx(0.4, abs=0.01)


def test_reliability_points_pair_predicted_with_observed(outcome):
    points = reliability_points(
        [f.probability for fold in outcome.report.usable_folds for f in fold.forecasts],
        [o for fold in outcome.report.usable_folds for o in fold.outcomes],
    )
    assert points, "expected at least one populated bin"
    for point in points:
        assert 0.0 <= point["mean_predicted"] <= 1.0
        assert 0.0 <= point["observed_frequency"] <= 1.0
        assert point["count"] > 0


# -- stage 3: backtest -----------------------------------------------------


@pytest.fixture(scope="module")
def backtest(prepared, outcome, corpus, settings):
    return run_model_backtest(prepared, outcome, corpus, settings)


def test_the_backtest_reports_how_many_forecasts_it_considered(backtest):
    assert backtest.n_forecasts > 0


def test_every_skipped_trade_records_a_reason(backtest):
    from backtest.skips import skip_summary

    for row in skip_summary(backtest.skips):
        assert row["reason"]
        assert row["count"] > 0
    for skipped in backtest.skips:
        assert skipped.reason is not None
        assert skipped.detail


def test_the_backtest_reports_costs_separately_from_pnl(backtest):
    metrics = backtest.metrics
    for key in ("total_pnl", "total_fees", "total_slippage_cost", "max_drawdown",
                "turnover", "n_trades"):
        assert key in metrics, f"missing {key}"


def test_a_baseline_backtest_is_available_for_comparison(backtest):
    """The candidate is only interesting relative to the frozen baseline."""
    assert backtest.baseline_metrics is not None


def test_the_backtest_only_ever_sees_out_of_sample_forecasts(backtest, outcome):
    """It is fed the walk-forward forecasts, which are test-fold only."""
    oos = sum(len(fold.forecasts) for fold in outcome.report.usable_folds)
    assert backtest.n_forecasts <= oos


# -- stage 4: drift --------------------------------------------------------


def test_psi_is_zero_for_an_unchanged_distribution():
    values = [float(i % 10) for i in range(500)]
    assert population_stability_index(values, list(values)) == pytest.approx(0.0, abs=1e-9)


def test_psi_grows_when_the_distribution_shifts():
    reference = [float(i % 10) for i in range(500)]
    shifted = [v + 5.0 for v in reference]
    assert population_stability_index(reference, shifted) > 0.25


def test_psi_is_none_without_enough_data():
    assert population_stability_index([], []) is None


def test_quality_drift_notices_more_missing_data():
    """Missingness counts per row, against a fixed feature count."""
    stable = quality_drift([0, 1, 0, 1], [0, 1, 1, 0], n_features=10)
    worse = quality_drift([0, 0, 0, 0], [9, 9, 8, 9], n_features=10)
    assert all(m.level is DriftLevel.STABLE for m in stable)
    assert any(m.level in (DriftLevel.MINOR, DriftLevel.MATERIAL) for m in worse)


def test_the_drift_report_never_retrains_or_promotes_by_itself(prepared, outcome, settings):
    """Drift produces findings for a human, never an action."""
    report = check_drift(prepared, outcome, settings)
    assert report is not None
    # `material` lists the metrics that breached, so a human can read why.
    assert all(m.level is DriftLevel.MATERIAL for m in report.material)
    for forbidden in ("retrain", "promote", "deploy", "apply"):
        assert not hasattr(report, forbidden), f"drift report exposes {forbidden}()"


def test_drift_metrics_carry_their_own_level_and_explanation(prepared, outcome, settings):
    report = check_drift(prepared, outcome, settings)
    assert report.metrics
    for metric in report.metrics:
        assert metric.level in set(DriftLevel)
        assert metric.name
        assert metric.note, f"{metric.name} has no explanation"
    assert report.worst_level in set(DriftLevel)


# -- reproducibility -------------------------------------------------------


def test_the_same_config_and_data_reproduce_the_same_scores(prepared, settings):
    first = train_and_evaluate(prepared, settings, "logistic_regression")
    second = train_and_evaluate(prepared, settings, "logistic_regression")
    assert first.candidate.brier == pytest.approx(second.candidate.brier)
    assert first.candidate.log_loss == pytest.approx(second.candidate.log_loss)
    assert first.candidate.n_forecasts == second.candidate.n_forecasts


def test_run_ids_are_unique_even_for_identical_configs(tmp_path, corpus, settings):
    registry = ResearchRegistry(tmp_path / "reg.duckdb")
    prepared = prepare_data(corpus, settings, registry)
    a = train_and_evaluate(prepared, settings, "logistic_regression", registry)
    b = train_and_evaluate(prepared, settings, "logistic_regression", registry)
    assert a.run_id != b.run_id, "each run must be its own immutable record"


def test_the_baseline_is_frozen_across_runs(prepared, settings):
    """The bar must not move between candidates."""
    first = train_and_evaluate(prepared, settings, "logistic_regression")
    second = train_and_evaluate(prepared, settings, "persistence_baseline")
    assert first.baseline.model_name == second.baseline.model_name
    assert first.baseline.brier == pytest.approx(second.baseline.brier)
