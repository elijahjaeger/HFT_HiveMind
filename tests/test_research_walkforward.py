"""Walk-forward protocol: grouping, purge, embargo, and where each fit happens.

These are the leakage tests. If any of them regress, every metric the pipeline
produces downstream becomes meaningless, so they assert the invariants directly
rather than checking summary numbers.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from core.dataset import MarketDataset
from core.models import MarketStatus, Outcome
from research.datasets import build_feature_dataset
from research.walkforward import (
    MIN_ISOTONIC_MARKETS,
    plan_folds,
    run_walk_forward,
)
from tests.conftest import T0, make_market, make_quote, make_trade


def staggered_dataset(n_markets: int = 40, lifetime_hours: int = 8,
                      stagger_hours: int = 4, yes_every: int = 2) -> MarketDataset:
    """A rolling sequence of markets that resolve at well-separated times."""
    markets, quotes, trades = [], [], []
    for i in range(n_markets):
        opened = T0 + timedelta(hours=stagger_hours * i)
        close = opened + timedelta(hours=lifetime_hours)
        outcome = Outcome.YES if i % yes_every == 0 else Outcome.NO
        market_id = f"M{i:03d}"
        market = make_market(
            market_id=market_id, outcome=outcome, close_time=close,
            status=MarketStatus.RESOLVED,
        ).model_copy(update={"created_at": opened,
                             "resolution_time": close + timedelta(hours=1)})
        markets.append(market)
        # A mildly informative book: YES markets drift up, NO markets drift down.
        drift = 0.004 if outcome is Outcome.YES else -0.004
        for step in range(32):
            ts = opened + timedelta(minutes=15 * step)
            mid = min(0.9, max(0.1, 0.5 + drift * step))
            quotes.append(make_quote(bid=round(mid - 0.01, 4), ask=round(mid + 0.01, 4),
                                     bid_size=200.0, ask_size=150.0, ts=ts,
                                     market_id=market_id))
            trades.append(make_trade(price=round(mid, 4), size=8.0, ts=ts,
                                     trade_id=f"{market_id}-T{step}",
                                     market_id=market_id))
    quotes.sort(key=lambda q: q.ts)
    trades.sort(key=lambda t: t.ts)
    return MarketDataset(markets=markets, quotes=quotes, trades=trades,
                         label="TEST", source="TEST")


@pytest.fixture(scope="module")
def dataset():
    return build_feature_dataset(staggered_dataset(40))


@pytest.fixture(scope="module")
def plans(dataset):
    return plan_folds(dataset, {"n_folds": 3, "embargo_seconds": 3600.0})


# -- chronology ------------------------------------------------------------


def test_folds_are_laid_out_in_time_order(plans, dataset):
    assert plans
    resolution = {m: lbl.resolution_ts for m, lbl in dataset.labels.items() if lbl.is_usable}
    for plan in plans:
        train_last = max(resolution[m] for m in plan.train_markets)
        test_first = min(resolution[m] for m in plan.test_markets)
        assert train_last < test_first, f"fold {plan.index} trains on the future"


def test_the_test_block_walks_forward_across_folds(plans):
    starts = [p.test_start_ts for p in plans]
    assert starts == sorted(starts)
    assert len(set(starts)) == len(starts), "each fold must test a later block"


def test_nothing_is_shuffled(dataset):
    """Rows enter each fold in decision order."""
    stamps = [e.decision_ts for e in dataset.labelled]
    assert stamps == sorted(stamps)


# -- grouping: a market is never split -------------------------------------


def test_a_market_never_appears_in_two_parts_of_a_fold(plans):
    for plan in plans:
        train = set(plan.train_markets)
        val = set(plan.validation_markets)
        test = set(plan.test_markets)
        assert not train & val, f"fold {plan.index}: market in train and validation"
        assert not train & test, f"fold {plan.index}: market in train and test"
        assert not val & test, f"fold {plan.index}: market in validation and test"


def test_a_market_resolves_once_so_it_belongs_to_one_fold_part(plans):
    """The single most direct leak: the same outcome on both sides of the split."""
    for plan in plans:
        everything = list(plan.train_markets) + list(plan.validation_markets) + \
            list(plan.test_markets)
        assert len(everything) == len(set(everything))


# -- purge and embargo -----------------------------------------------------


def test_every_training_market_resolves_before_the_embargoed_test_start(plans, dataset):
    """The headline invariant from the design doc."""
    resolution = {m: lbl.resolution_ts for m, lbl in dataset.labels.items() if lbl.is_usable}
    for plan in plans:
        cutoff = plan.test_start_ts - timedelta(seconds=plan.embargo_seconds)
        for market in list(plan.train_markets) + list(plan.validation_markets):
            assert resolution[market] <= cutoff, (
                f"fold {plan.index}: {market} resolved at {resolution[market]}, "
                f"after the embargoed cutoff {cutoff}"
            )


def test_a_longer_embargo_purges_more_markets(dataset):
    short = plan_folds(dataset, {"n_folds": 3, "embargo_seconds": 0.0})
    long = plan_folds(dataset, {"n_folds": 3, "embargo_seconds": 86_400.0 * 3})
    assert short and long
    assert sum(len(p.purged_markets) for p in long) > sum(len(p.purged_markets) for p in short)


def test_purged_markets_are_recorded_not_silently_dropped(dataset):
    plans = plan_folds(dataset, {"n_folds": 3, "embargo_seconds": 86_400.0 * 2})
    assert any(p.purged_markets for p in plans)
    for plan in plans:
        assert not set(plan.purged_markets) & set(plan.train_markets)


def test_too_few_markets_yields_no_folds():
    tiny = build_feature_dataset(staggered_dataset(2))
    assert plan_folds(tiny, {"n_folds": 3}) == []


# -- where each fit happens ------------------------------------------------


@pytest.fixture(scope="module")
def report(dataset):
    return run_walk_forward(dataset, "logistic_regression",
                            {"n_folds": 3, "calibration_method": "logistic"})


def test_the_walk_forward_produces_out_of_sample_forecasts(report):
    assert report.usable_folds
    assert report.effective_n > 0


def test_every_forecast_belongs_to_a_test_market_of_its_own_fold(report):
    for fold in report.usable_folds:
        test_markets = set(fold.plan.test_markets)
        assert {f.market_ticker for f in fold.forecasts} <= test_markets


def test_the_scaler_is_fitted_on_training_rows_only(report):
    for fold in report.usable_folds:
        assert fold.pipeline is not None
        assert fold.pipeline.assumptions["scaler_fitted_on"] == "training fold only"


def test_calibration_is_fitted_on_validation_only(report):
    for fold in report.usable_folds:
        assert fold.pipeline is not None
        assert fold.pipeline.assumptions["calibration_fitted_on"] == "validation fold only"


def test_calibrating_on_test_data_would_change_the_result(dataset):
    """Confidence check that the calibrator is doing something at all.

    If fitting on validation vs on everything gave identical output, the
    'fitted on validation only' assertion above would be vacuous.
    """
    from calibration.methods import LogisticCalibrator

    report = run_walk_forward(dataset, "logistic_regression",
                              {"n_folds": 3, "calibration_method": "logistic"})
    fold = report.usable_folds[0]
    honest = fold.calibrated_probabilities

    cheating = LogisticCalibrator().fit(fold.raw_probabilities, fold.outcomes)
    leaked = cheating.transform(fold.raw_probabilities)
    assert honest != leaked, "calibrator output is insensitive to what it was fitted on"


def test_each_fold_freezes_its_own_pipeline(report):
    """Folds must not share a fitted object or its parameters.

    ``model_version`` deliberately identifies the model *type*, so it is the
    same across folds; identity has to be checked on the fitted objects and
    their learned parameters instead.
    """
    pipelines = [f.pipeline for f in report.usable_folds]
    assert len({id(p) for p in pipelines}) == len(pipelines)
    assert len({id(p.model) for p in pipelines}) == len(pipelines)
    assert len({id(p.scaler) for p in pipelines}) == len(pipelines)

    # Different training windows must produce different fitted parameters.
    scaler_means = [tuple(p.scaler.means) for p in pipelines]
    assert len(set(scaler_means)) == len(scaler_means), "folds saw the same training data"
    assert all(p.assumptions["fold"] == f.plan.index
               for p, f in zip(pipelines, report.usable_folds, strict=True))


# -- forecast contract -----------------------------------------------------


def test_every_forecast_carries_full_provenance(report):
    for fold in report.usable_folds:
        for forecast in fold.forecasts:
            assert forecast.market_ticker
            assert forecast.decision_timestamp is not None
            assert 0.0 <= forecast.raw_probability_yes <= 1.0
            assert forecast.calibrated_probability_yes is None or \
                0.0 <= forecast.calibrated_probability_yes <= 1.0
            assert 0.0 <= forecast.confidence <= 1.0
            assert forecast.feature_version and forecast.model_version
            assert forecast.data_version
            assert forecast.assumptions


def test_a_forecast_is_never_stamped_after_its_market_resolved(report, dataset):
    resolution = {m: lbl.resolution_ts for m, lbl in dataset.labels.items() if lbl.is_usable}
    for fold in report.usable_folds:
        for forecast in fold.forecasts:
            assert forecast.decision_timestamp < resolution[forecast.market_ticker]


# -- calibration safety ----------------------------------------------------


def test_isotonic_is_skipped_and_warned_about_on_a_small_validation_fold(dataset):
    report = run_walk_forward(dataset, "logistic_regression",
                              {"n_folds": 3, "calibration_method": "isotonic"})
    small = [f for f in report.usable_folds
             if len(f.plan.validation_markets) < MIN_ISOTONIC_MARKETS]
    assert small, "expected at least one fold below the isotonic minimum"
    for fold in small:
        assert fold.calibration_method != "isotonic"
        assert any("Isotonic calibration skipped" in w for w in fold.warnings)


def test_an_unusable_dataset_reports_a_warning_not_a_crash():
    tiny = build_feature_dataset(staggered_dataset(2))
    report = run_walk_forward(tiny, "logistic_regression", {"n_folds": 3})
    assert report.usable_folds == []
    assert report.warnings


# -- reproducibility -------------------------------------------------------


def test_the_same_dataset_and_config_reproduce_identical_forecasts(dataset):
    config = {"n_folds": 3, "calibration_method": "logistic"}
    a = run_walk_forward(dataset, "logistic_regression", config)
    b = run_walk_forward(dataset, "logistic_regression", config)
    assert [f.raw_probability_yes for fold in a.usable_folds for f in fold.forecasts] == \
           [f.raw_probability_yes for fold in b.usable_folds for f in fold.forecasts]


def test_changing_the_config_changes_the_folds(dataset):
    three = plan_folds(dataset, {"n_folds": 3})
    five = plan_folds(dataset, {"n_folds": 5})
    assert [p.as_row() for p in three] != [p.as_row() for p in five]


def test_fold_plans_render_display_rows(plans):
    for plan in plans:
        row = plan.as_row()
        assert {"fold", "train_markets", "val_markets", "test_markets", "purged"} <= set(row)
