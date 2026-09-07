"""Fitting, scoring and method selection across folds."""

from __future__ import annotations

import random
from datetime import UTC, datetime, timedelta

import pytest

from backtest.metrics import expected_calibration_error
from calibration.evaluate import evaluate_calibrators
from calibration.methods import IdentityCalibrator, IsotonicCalibrator, LogisticCalibrator
from calibration.splits import Observation, chronological_split

T0 = datetime(2026, 1, 5, 9, tzinfo=UTC)


def miscalibrated_observations(
    n_markets: int = 400, per_market: int = 4, factor: float = 2.0, seed: int = 0
) -> list[Observation]:
    """A rolling sequence of markets whose forecasts are too confident.

    Each market opens after the previous one resolves, so a chronological split
    keeps every fold clean without dropping anything.
    """
    rng = random.Random(seed)
    observations: list[Observation] = []
    for index in range(n_markets):
        opened = T0 + timedelta(hours=index * 2)
        resolution = opened + timedelta(hours=1)
        raw = rng.uniform(0.05, 0.95)
        true_probability = 0.5 + (raw - 0.5) / factor
        outcome = 1.0 if rng.random() < true_probability else 0.0
        for step in range(per_market):
            observations.append(
                Observation(
                    ts=opened + timedelta(minutes=step * 10),
                    market_id=f"M{index:03d}",
                    probability=min(max(raw + rng.gauss(0, 0.01), 0.01), 0.99),
                    outcome=outcome,
                    resolution_ts=resolution,
                )
            )
    return observations


@pytest.fixture(scope="module")
def report():
    split = chronological_split(miscalibrated_observations())
    return evaluate_calibrators(split)


# -- ECE -------------------------------------------------------------------


def test_perfect_calibration_scores_zero_ece():
    assert expected_calibration_error([0.0, 1.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_ece_measures_the_weighted_gap():
    # Ten forecasts of 0.9 where only half resolved YES: gap of 0.4.
    assert expected_calibration_error([0.9] * 10, [1.0] * 5 + [0.0] * 5) == pytest.approx(
        0.4
    )


def test_ece_is_none_without_observations():
    assert expected_calibration_error([], []) is None


# -- fitting and scoring ---------------------------------------------------


def test_every_method_is_scored_on_every_fold(report):
    assert {m.name for m in report.methods} == {"uncalibrated", "isotonic", "logistic"}
    for method in report.methods:
        assert set(method.scores) == {"train", "validation", "test"}
        assert method.error is None


def test_train_scores_are_labelled_in_sample(report):
    for method in report.methods:
        assert method.scores["train"].in_sample is True
        assert method.scores["test"].in_sample is False
    assert any("in-sample" in row["fold"] for row in report.comparison_rows())


def test_calibration_improves_a_known_overconfident_forecaster(report):
    """The whole point: on held-out data, calibration should beat the baseline."""
    baseline = report.method("uncalibrated").test
    isotonic = report.method("isotonic").test
    logistic = report.method("logistic").test
    assert isotonic.brier < baseline.brier
    assert logistic.brier < baseline.brier
    assert logistic.ece < baseline.ece


def test_the_baseline_is_scored_on_the_raw_probabilities(report):
    scores = report.method("uncalibrated").test
    assert scores.mean_prediction is not None
    assert scores.n > 0


def test_the_report_exposes_a_delta_against_the_baseline(report):
    rows = {row["method"]: row for row in report.test_rows()}
    assert rows["uncalibrated"]["brier_vs_baseline"] == pytest.approx(0.0)
    assert rows["isotonic"]["brier_vs_baseline"] < 0


def test_a_calibrator_is_never_fitted_on_validation_or_test():
    """Fitting must depend only on the training fold."""
    observations = miscalibrated_observations()
    split = chronological_split(observations)

    fitted_on: list[int] = []

    class SpyCalibrator(IdentityCalibrator):
        name = "spy"

        def fit(self, probabilities, outcomes):
            fitted_on.append(len(probabilities))
            return super().fit(probabilities, outcomes)

    evaluate_calibrators(split, [SpyCalibrator()])
    assert fitted_on == [len(split.train)]


def test_selection_uses_validation_and_never_the_test_fold(report):
    selected = report.selected()
    assert selected is not None
    best_on_validation = min(
        (m for m in report.methods if m.validation and m.validation.brier is not None),
        key=lambda m: m.validation.brier,
    )
    assert selected.name == best_on_validation.name


def test_the_selection_metric_can_be_changed():
    split = chronological_split(miscalibrated_observations())
    by_log_loss = evaluate_calibrators(split, selection_metric="log_loss")
    assert by_log_loss.selection_metric == "log_loss"
    assert by_log_loss.selected() is not None


def test_an_unknown_selection_metric_is_rejected():
    split = chronological_split(miscalibrated_observations())
    with pytest.raises(ValueError, match="unknown selection metric"):
        evaluate_calibrators(split, selection_metric="accuracy")


def test_an_empty_training_fold_is_refused():
    split = chronological_split([])
    with pytest.raises(ValueError, match="training fold is empty"):
        evaluate_calibrators(split)


def test_a_failing_calibrator_is_reported_not_raised():
    class Exploding(IdentityCalibrator):
        name = "exploding"

        def fit(self, probabilities, outcomes):
            raise ValueError("nope")

    split = chronological_split(miscalibrated_observations())
    report = evaluate_calibrators(split, [Exploding()])
    assert report.method("exploding").error is not None
    assert any("error" in row for row in report.comparison_rows())


# -- display data ----------------------------------------------------------


def test_reliability_series_only_contains_populated_buckets(report):
    series = report.reliability_series("test")
    assert set(series) == {"uncalibrated", "isotonic", "logistic"}
    for points in series.values():
        assert points
        assert all(0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 for x, y in points)


def test_a_calibrated_reliability_curve_sits_closer_to_the_diagonal(report):
    def mean_gap(name: str) -> float:
        points = report.reliability_series("test")[name]
        return sum(abs(x - y) for x, y in points) / len(points)

    assert mean_gap("isotonic") < mean_gap("uncalibrated")


def test_calibration_rows_cover_every_bucket(report):
    rows = report.calibration_rows("isotonic", "test")
    assert len(rows) == report.n_buckets
    assert {"bucket", "count", "mean_predicted", "observed_frequency", "gap"} == set(
        rows[0]
    )


def test_calibration_rows_for_an_unknown_method_are_empty(report):
    assert report.calibration_rows("nope") == []


def test_a_clean_split_reports_no_warnings(report):
    assert report.is_trustworthy
    assert report.warnings == []


def test_small_samples_do_not_reliably_improve_on_the_baseline():
    """Calibration needs data. With ~45 training markets, neither method helps.

    This is the behaviour that makes the demo dataset unusable for calibration,
    so it is pinned rather than left as a surprise.
    """
    split = chronological_split(miscalibrated_observations(n_markets=90))
    report = evaluate_calibrators(split)
    baseline = report.method("uncalibrated").test.brier
    assert report.method("isotonic").test.brier >= baseline
    assert split.train.effective_n < 50


def test_a_thin_split_reports_warnings():
    split = chronological_split(miscalibrated_observations(n_markets=9))
    report = evaluate_calibrators(split)
    assert not report.is_trustworthy
    assert any("far too few" in w for w in report.warnings)


# -- integration with the backtester --------------------------------------


def test_the_pipeline_runs_end_to_end_on_demo_data(demo_result):
    from calibration.splits import observations_from_result

    split = chronological_split(observations_from_result(demo_result))
    if not split.train.observations:
        pytest.skip("demo split produced no training fold")
    report = evaluate_calibrators(split)
    assert report.methods
    assert report.comparison_rows()
    # The demo dataset is far too small to calibrate on, and must say so.
    assert not report.is_trustworthy


def test_isotonic_and_logistic_can_be_supplied_explicitly():
    split = chronological_split(miscalibrated_observations())
    report = evaluate_calibrators(split, [IsotonicCalibrator(), LogisticCalibrator()])
    assert [m.name for m in report.methods] == ["isotonic", "logistic"]
    assert report.baseline is None
