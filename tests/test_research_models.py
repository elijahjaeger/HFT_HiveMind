"""Forecasters: the contract, determinism, and the models that are refused."""

from __future__ import annotations

import math

import pytest

from research.features import FEATURE_NAMES, StandardScaler
from research.models import (
    TREE_MODEL_BAR,
    FittedPipeline,
    Forecaster,
    LogisticForecaster,
    MarketPriceBaseline,
    PersistenceBaseline,
    ProbabilityForecast,
    build_forecaster,
)
from tests.conftest import T0


def separable_rows(n: int = 80):
    """A linearly separable problem the logistic model should solve."""
    rows, labels = [], []
    for i in range(n):
        positive = i % 2 == 0
        base = 0.8 if positive else 0.2
        rows.append([base] + [0.0] * (len(FEATURE_NAMES) - 1))
        labels.append(1.0 if positive else 0.0)
    return rows, labels


# -- the forecast contract -------------------------------------------------


def test_the_forecast_carries_every_required_field():
    forecast = ProbabilityForecast(
        market_ticker="M1",
        decision_timestamp=T0,
        raw_probability_yes=0.6,
        calibrated_probability_yes=0.55,
        confidence=0.4,
        feature_version="features-v1",
        model_version="logistic_regression/1.0",
        data_version="data-abc",
        assumptions={"scaler_fitted_on": "training fold only"},
    )
    assert forecast.probability == pytest.approx(0.55)
    row = forecast.as_row()
    assert {"market", "decision_ts", "raw", "calibrated", "confidence",
            "model", "features", "data"} <= set(row)


def test_the_actionable_probability_falls_back_to_raw_when_uncalibrated():
    forecast = ProbabilityForecast(
        market_ticker="M1", decision_timestamp=T0, raw_probability_yes=0.6,
        calibrated_probability_yes=None, confidence=0.4,
        feature_version="f", model_version="m", data_version="d",
    )
    assert forecast.probability == pytest.approx(0.6)
    assert forecast.as_row()["calibrated"] is None


# -- baselines -------------------------------------------------------------


def test_the_market_price_baseline_returns_the_market_probability():
    model = MarketPriceBaseline()
    index = FEATURE_NAMES.index("market_prob")
    row = [0.0] * len(FEATURE_NAMES)
    row[index] = 0.37
    model.fit([row], [1.0])
    assert model.predict_proba([row])[0] == pytest.approx(0.37)


def test_the_market_price_baseline_ignores_its_training_labels():
    """It is a frozen reference, not something that learns."""
    model = MarketPriceBaseline()
    index = FEATURE_NAMES.index("market_prob")
    row = [0.0] * len(FEATURE_NAMES)
    row[index] = 0.37
    model.fit([row] * 20, [1.0] * 20)  # all YES
    assert model.predict_proba([row])[0] == pytest.approx(0.37)


def test_the_persistence_baseline_carries_the_last_probability_forward():
    model = PersistenceBaseline()
    index = FEATURE_NAMES.index("market_prob")
    row = [0.0] * len(FEATURE_NAMES)
    row[index] = 0.62
    model.fit([row], [1.0])
    assert model.predict_proba([row])[0] == pytest.approx(0.62, abs=1e-6)


def test_baseline_probabilities_stay_inside_the_unit_interval():
    model = MarketPriceBaseline()
    index = FEATURE_NAMES.index("market_prob")
    for value in (-5.0, 0.0, 0.5, 1.0, 5.0):
        row = [0.0] * len(FEATURE_NAMES)
        row[index] = value
        model.fit([row], [1.0])
        assert 0.0 <= model.predict_proba([row])[0] <= 1.0


# -- logistic regression ---------------------------------------------------


def test_logistic_regression_learns_a_separable_problem():
    rows, labels = separable_rows()
    model = LogisticForecaster().fit(rows, labels)
    predictions = model.predict_proba(rows)
    positives = [p for p, y in zip(predictions, labels, strict=True) if y == 1.0]
    negatives = [p for p, y in zip(predictions, labels, strict=True) if y == 0.0]
    assert min(positives) > max(negatives)


def test_logistic_probabilities_are_valid():
    rows, labels = separable_rows()
    model = LogisticForecaster().fit(rows, labels)
    assert all(0.0 < p < 1.0 for p in model.predict_proba(rows))


def test_logistic_regression_is_deterministic():
    rows, labels = separable_rows()
    first = LogisticForecaster(l2=1.0).fit(rows, labels).predict_proba(rows)
    second = LogisticForecaster(l2=1.0).fit(rows, labels).predict_proba(rows)
    assert first == second


def test_stronger_regularisation_shrinks_the_coefficients():
    rows, labels = separable_rows()
    weak = LogisticForecaster(l2=0.01).fit(rows, labels)
    strong = LogisticForecaster(l2=100.0).fit(rows, labels)
    assert sum(abs(w) for w in strong.weights) < sum(abs(w) for w in weak.weights)


def test_fitted_coefficients_are_inspectable_by_feature_name():
    """Interpretability is the point of choosing this model."""
    rows, labels = separable_rows()
    model = LogisticForecaster().fit(rows, labels)
    rows_out = model.coefficient_rows()
    assert rows_out
    assert {r["feature"] for r in rows_out} <= set(FEATURE_NAMES)
    assert "market_prob" in {r["feature"] for r in rows_out}


def test_a_single_class_training_set_does_not_crash():
    """Degenerate but real: an all-NO training fold must fail safely."""
    rows = [[0.5] + [0.0] * (len(FEATURE_NAMES) - 1) for _ in range(10)]
    model = LogisticForecaster().fit(rows, [0.0] * 10)
    predictions = model.predict_proba(rows)
    assert all(0.0 <= p <= 1.0 for p in predictions)
    assert all(math.isfinite(p) for p in predictions)


def test_predicting_before_fitting_is_refused():
    with pytest.raises(RuntimeError):
        LogisticForecaster().predict_proba([[0.0] * len(FEATURE_NAMES)])


# -- the model registry of allowed types -----------------------------------


def test_the_supported_models_can_all_be_built():
    for name in ("market_price_baseline", "persistence_baseline", "logistic_regression"):
        model = build_forecaster(name)
        assert isinstance(model, Forecaster)
        assert model.name and model.version


@pytest.mark.parametrize("name", ["mlp", "lstm", "neural_network"])
def test_out_of_scope_model_families_are_refused(name):
    with pytest.raises(ValueError, match="out of scope"):
        build_forecaster(name)


@pytest.mark.parametrize("name", ["rl", "mappo", "ppo"])
def test_a_control_policy_is_redirected_rather_than_banned(name):
    """RL is part of the project since 2026-09-07, but not as a forecaster.

    A policy emits actions, so it cannot be scored by Brier or log loss. The
    refusal now has to say where it belongs instead of only saying no.
    """
    with pytest.raises(ValueError) as excinfo:
        build_forecaster(name)
    message = str(excinfo.value)
    assert "not a probability forecaster" in message
    assert "src/rl_policy.py" in message


def test_the_tree_model_is_refused_with_the_bar_it_would_have_to_clear():
    with pytest.raises(ValueError) as excinfo:
        build_forecaster("decision_tree")
    assert "not enabled" in str(excinfo.value)
    assert "200 resolved markets" in TREE_MODEL_BAR


def test_an_unknown_model_is_refused():
    with pytest.raises(ValueError, match="unknown forecaster"):
        build_forecaster("wishful_thinking")


# -- the frozen pipeline ---------------------------------------------------


def test_a_fitted_pipeline_produces_raw_and_calibrated_probabilities():
    from calibration.methods import LogisticCalibrator

    rows, labels = separable_rows()
    scaler = StandardScaler().fit(rows, list(FEATURE_NAMES))
    model = LogisticForecaster().fit(scaler.transform(rows), labels)
    calibrator = LogisticCalibrator().fit(model.predict_proba(scaler.transform(rows)), labels)
    pipeline = FittedPipeline(
        model=model, scaler=scaler, calibrator=calibrator,
        feature_names=FEATURE_NAMES, data_version="data-abc",
        calibration_method="logistic",
    )
    raw, calibrated = pipeline.probabilities(rows)
    assert len(raw) == len(rows)
    assert calibrated is not None and len(calibrated) == len(rows)
    assert all(0.0 <= p <= 1.0 for p in raw + calibrated)


def test_the_pipeline_applies_its_scaler_not_the_raw_rows():
    """Skipping the scaler would silently change every probability."""
    rows, labels = separable_rows()
    scaler = StandardScaler().fit(rows, list(FEATURE_NAMES))
    model = LogisticForecaster().fit(scaler.transform(rows), labels)
    pipeline = FittedPipeline(
        model=model, scaler=scaler, calibrator=None,
        feature_names=FEATURE_NAMES, data_version="d", calibration_method="none",
    )
    through_pipeline, _ = pipeline.probabilities(rows)
    unscaled = model.predict_proba(rows)
    assert through_pipeline == model.predict_proba(scaler.transform(rows))
    assert through_pipeline != unscaled


def test_the_pipeline_records_its_own_assumptions():
    rows, labels = separable_rows()
    scaler = StandardScaler().fit(rows, list(FEATURE_NAMES))
    pipeline = FittedPipeline(
        model=LogisticForecaster().fit(scaler.transform(rows), labels),
        scaler=scaler, calibrator=None, feature_names=FEATURE_NAMES,
        data_version="d", calibration_method="none",
        assumptions={"scaler_fitted_on": "training fold only"},
    )
    assert pipeline.assumptions["scaler_fitted_on"] == "training fold only"
    assert pipeline.model_version


# -- no deep learning anywhere in the package ------------------------------


def test_the_research_package_imports_no_deep_learning_library():
    import ast
    import pathlib

    root = pathlib.Path(__file__).resolve().parent.parent / "research"
    banned = {"torch", "tensorflow", "keras", "jax", "sklearn", "scikit_learn",
              "stable_baselines3", "gymnasium"}
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom):
                names = {(node.module or "").split(".")[0]}
            else:
                continue
            assert not (names & banned), f"{path.name} imports {names & banned}"
