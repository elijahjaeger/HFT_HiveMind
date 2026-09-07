"""Isotonic and logistic calibrators, checked against known-good answers."""

from __future__ import annotations

import math
import random

import pytest

from calibration.methods import (
    IdentityCalibrator,
    IsotonicCalibrator,
    LogisticCalibrator,
    default_calibrators,
    pool_adjacent_violators,
)


def calibrated_sample(n: int = 20_000, seed: int = 0) -> tuple[list[float], list[float]]:
    """Forecasts that are already perfectly calibrated."""
    rng = random.Random(seed)
    probabilities = [rng.uniform(0.05, 0.95) for _ in range(n)]
    outcomes = [1.0 if rng.random() < p else 0.0 for p in probabilities]
    return probabilities, outcomes


def overconfident_sample(
    n: int = 20_000, seed: int = 1, factor: float = 2.0
) -> tuple[list[float], list[float]]:
    """Forecasts pushed ``factor`` times too far from 0.5."""
    rng = random.Random(seed)
    probabilities = [rng.uniform(0.05, 0.95) for _ in range(n)]
    outcomes = [
        1.0 if rng.random() < 0.5 + (p - 0.5) / factor else 0.0 for p in probabilities
    ]
    return probabilities, outcomes


# -- pool adjacent violators ----------------------------------------------


def test_pava_pools_a_violating_pair():
    assert pool_adjacent_violators([1.0, 0.0, 1.0]) == pytest.approx([0.5, 0.5, 1.0])


def test_pava_leaves_a_monotone_sequence_alone():
    assert pool_adjacent_violators([0.0, 0.5, 1.0]) == pytest.approx([0.0, 0.5, 1.0])


def test_pava_output_is_always_non_decreasing():
    rng = random.Random(3)
    values = [rng.random() for _ in range(200)]
    fitted = pool_adjacent_violators(values)
    assert all(a <= b + 1e-12 for a, b in zip(fitted, fitted[1:], strict=False))


def test_pava_preserves_the_total():
    values = [0.9, 0.1, 0.4, 0.8]
    assert sum(pool_adjacent_violators(values)) == pytest.approx(sum(values))


def test_pava_handles_an_empty_input():
    assert pool_adjacent_violators([]) == []


def test_pava_rejects_mismatched_weights():
    with pytest.raises(ValueError):
        pool_adjacent_violators([1.0, 2.0], [1.0])


# -- identity --------------------------------------------------------------


def test_the_identity_calibrator_leaves_probabilities_alone():
    calibrator = IdentityCalibrator().fit([0.2, 0.8], [0.0, 1.0])
    assert calibrator.transform([0.25, 0.75]) == pytest.approx([0.25, 0.75])


def test_the_identity_calibrator_still_clips_the_extremes():
    calibrator = IdentityCalibrator().fit([0.5], [1.0])
    low, high = calibrator.transform([0.0, 1.0])
    assert 0.0 < low < 1e-3
    assert 1.0 - 1e-3 < high < 1.0


# -- isotonic --------------------------------------------------------------


def test_isotonic_recovers_the_empirical_frequency_of_each_level():
    probabilities = [0.2] * 100 + [0.8] * 100
    outcomes = [1.0] * 10 + [0.0] * 90 + [1.0] * 90 + [0.0] * 10
    calibrator = IsotonicCalibrator().fit(probabilities, outcomes)
    assert calibrator.transform([0.2])[0] == pytest.approx(0.10, abs=1e-9)
    assert calibrator.transform([0.8])[0] == pytest.approx(0.90, abs=1e-9)


def test_isotonic_is_monotone_non_decreasing():
    probabilities, outcomes = overconfident_sample(2_000, seed=5)
    calibrator = IsotonicCalibrator().fit(probabilities, outcomes)
    grid = [i / 100 for i in range(101)]
    mapped = calibrator.transform(grid)
    assert all(a <= b + 1e-12 for a, b in zip(mapped, mapped[1:], strict=False))


def test_isotonic_never_extrapolates_beyond_the_training_range():
    calibrator = IsotonicCalibrator().fit([0.4, 0.5, 0.6], [0.0, 1.0, 1.0])
    below, above = calibrator.transform([0.0, 1.0])
    inside = calibrator.transform([0.4, 0.6])
    assert below == pytest.approx(inside[0])
    assert above == pytest.approx(inside[1])


def test_isotonic_output_stays_inside_the_unit_interval():
    probabilities, outcomes = calibrated_sample(1_000, seed=7)
    mapped = IsotonicCalibrator().fit(probabilities, outcomes).transform(
        [0.0, 0.5, 1.0]
    )
    assert all(0.0 < value < 1.0 for value in mapped)


def test_isotonic_corrects_a_known_overconfident_forecaster():
    probabilities, outcomes = overconfident_sample(40_000, seed=11, factor=2.0)
    calibrator = IsotonicCalibrator().fit(probabilities, outcomes)
    # A raw 0.90 should be pulled back toward 0.5 + (0.90 - 0.5) / 2 = 0.70.
    assert calibrator.transform([0.90])[0] == pytest.approx(0.70, abs=0.05)


def test_isotonic_refuses_to_fit_nothing():
    with pytest.raises(ValueError):
        IsotonicCalibrator().fit([], [])


def test_isotonic_refuses_to_transform_before_fitting():
    with pytest.raises(RuntimeError, match="before fit"):
        IsotonicCalibrator().transform([0.5])


# -- logistic (Platt) ------------------------------------------------------


def test_logistic_is_near_identity_on_already_calibrated_data():
    probabilities, outcomes = calibrated_sample(20_000, seed=0)
    calibrator = LogisticCalibrator().fit(probabilities, outcomes)
    assert calibrator.slope == pytest.approx(1.0, abs=0.1)
    assert calibrator.intercept == pytest.approx(0.0, abs=0.1)
    assert calibrator.transform([0.7])[0] == pytest.approx(0.7, abs=0.03)


def test_logistic_shrinks_an_overconfident_forecaster():
    probabilities, outcomes = overconfident_sample(40_000, seed=1, factor=2.0)
    calibrator = LogisticCalibrator().fit(probabilities, outcomes)
    assert calibrator.slope < 1.0
    assert calibrator.transform([0.90])[0] == pytest.approx(0.70, abs=0.05)
    assert calibrator.transform([0.10])[0] == pytest.approx(0.30, abs=0.05)


def test_logistic_is_monotone_in_the_input():
    probabilities, outcomes = overconfident_sample(2_000, seed=2)
    mapped = LogisticCalibrator().fit(probabilities, outcomes).transform(
        [i / 20 for i in range(21)]
    )
    assert all(a <= b + 1e-12 for a, b in zip(mapped, mapped[1:], strict=False))


def test_logistic_converges_on_well_behaved_data():
    probabilities, outcomes = calibrated_sample(5_000, seed=4)
    calibrator = LogisticCalibrator().fit(probabilities, outcomes)
    assert calibrator.converged
    assert calibrator.iterations < 50


def test_target_smoothing_keeps_a_separable_fold_finite():
    """Perfectly separable training data would otherwise blow the fit up."""
    probabilities = [0.1] * 50 + [0.9] * 50
    outcomes = [0.0] * 50 + [1.0] * 50
    calibrator = LogisticCalibrator(smoothing=True).fit(probabilities, outcomes)
    assert math.isfinite(calibrator.slope)
    assert math.isfinite(calibrator.intercept)
    assert all(0.0 < p < 1.0 for p in calibrator.transform([0.1, 0.5, 0.9]))


def test_logistic_output_stays_inside_the_unit_interval():
    probabilities, outcomes = overconfident_sample(500, seed=6)
    calibrator = LogisticCalibrator().fit(probabilities, outcomes)
    assert all(0.0 < p < 1.0 for p in calibrator.transform([0.0, 0.5, 1.0]))


def test_logistic_reports_its_fitted_parameters():
    probabilities, outcomes = calibrated_sample(1_000, seed=8)
    parameters = LogisticCalibrator().fit(probabilities, outcomes).parameters
    assert set(parameters) == {"slope", "intercept"}


def test_logistic_refuses_to_transform_before_fitting():
    with pytest.raises(RuntimeError, match="before fit"):
        LogisticCalibrator().transform([0.5])


# -- shared contract -------------------------------------------------------


@pytest.mark.parametrize("calibrator", default_calibrators(), ids=lambda c: c.name)
def test_every_calibrator_validates_its_inputs(calibrator):
    with pytest.raises(ValueError):
        calibrator.fit([0.5, 0.5], [1.0])
    with pytest.raises(ValueError):
        calibrator.fit([1.5], [1.0])
    with pytest.raises(ValueError):
        calibrator.fit([0.5], [0.5])


@pytest.mark.parametrize("calibrator", default_calibrators(), ids=lambda c: c.name)
def test_every_calibrator_is_deterministic(calibrator):
    probabilities, outcomes = overconfident_sample(1_000, seed=9)
    first = calibrator.fit(probabilities, outcomes).transform(probabilities[:50])
    second = calibrator.fit(probabilities, outcomes).transform(probabilities[:50])
    assert first == pytest.approx(second)


def test_the_default_set_leads_with_the_uncalibrated_baseline():
    assert [c.name for c in default_calibrators()] == [
        "uncalibrated",
        "isotonic",
        "logistic",
    ]
