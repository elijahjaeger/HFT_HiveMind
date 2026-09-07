"""Chronological walk-forward evaluation with purge and embargo.

The protocol, per fold:

1. Train the model on an earlier time range.
2. Fit preprocessing (scaling) on that training range **only**.
3. Fit calibration on a strictly later validation range **only**.
4. Freeze the whole pipeline.
5. Score a strictly later, untouched test range.
6. Backtest only that fold's out-of-sample forecasts.

Why grouping and purging are not optional
-----------------------------------------

Prediction markets overlap in time, so a plain chronological cut leaks in two
distinct ways:

* **Split markets.** One market resolves once. If its rows appear in both
  training and test, the model has already been shown the answer it is being
  asked for. Markets are therefore assigned whole to a single fold.
* **Overlapping horizons.** A training market that resolves *after* the test
  fold begins was still unresolved when the test forecasts were made - so its
  outcome could not have been known to a real forecaster at that time. Those
  markets are **purged** from training.

An **embargo** then skips a configurable gap after the last training
resolution, because information leaks between correlated markets even without
a direct overlap.

The invariant the tests assert: *every training market resolves at least
``embargo_seconds`` before the earliest test decision timestamp.*
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from calibration.methods import (
    IdentityCalibrator,
    IsotonicCalibrator,
    LogisticCalibrator,
)
from research.datasets import FeatureDataset, TrainingExample
from research.features import StandardScaler
from research.models import FittedPipeline, ProbabilityForecast, build_forecaster

DEFAULT_EMBARGO_SECONDS = 3600.0
MIN_ISOTONIC_MARKETS = 30
"""Below this many independent outcomes, isotonic calibration is skipped.

Isotonic regression is non-parametric and will happily fit a step function
through a handful of points, producing a calibration map that looks perfect on
the data it was fitted to and is meaningless off it.
"""


@dataclass(frozen=True)
class FoldPlan:
    """Which markets belong to each part of one fold, and why."""

    index: int
    train_markets: tuple[str, ...]
    validation_markets: tuple[str, ...]
    test_markets: tuple[str, ...]
    purged_markets: tuple[str, ...] = ()
    embargo_seconds: float = DEFAULT_EMBARGO_SECONDS
    train_knowledge_ts: datetime | None = None
    test_start_ts: datetime | None = None

    @property
    def is_usable(self) -> bool:
        """True when every part of the fold has at least one market."""
        return bool(self.train_markets and self.validation_markets and self.test_markets)

    def as_row(self) -> dict[str, Any]:
        """Display row for the dashboard."""
        return {
            "fold": self.index,
            "train_markets": len(self.train_markets),
            "val_markets": len(self.validation_markets),
            "test_markets": len(self.test_markets),
            "purged": len(self.purged_markets),
            "train_known_by": self.train_knowledge_ts,
            "test_starts": self.test_start_ts,
        }


@dataclass
class FoldResult:
    """Everything one fold produced."""

    plan: FoldPlan
    pipeline: FittedPipeline | None = None
    forecasts: list[ProbabilityForecast] = field(default_factory=list)
    test_examples: list[TrainingExample] = field(default_factory=list)
    raw_probabilities: list[float] = field(default_factory=list)
    calibrated_probabilities: list[float] = field(default_factory=list)
    outcomes: list[float] = field(default_factory=list)
    market_probabilities: list[float] = field(default_factory=list)
    calibration_method: str = "none"
    warnings: list[str] = field(default_factory=list)
    skipped_reason: str = ""

    @property
    def ok(self) -> bool:
        """True when the fold produced scored forecasts."""
        return bool(self.outcomes) and not self.skipped_reason

    @property
    def effective_n(self) -> int:
        """Independent outcomes in the test fold."""
        return len(self.plan.test_markets)


@dataclass
class WalkForwardReport:
    """Every fold, for one model."""

    model_name: str
    folds: list[FoldResult] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def usable_folds(self) -> list[FoldResult]:
        """Folds that produced out-of-sample forecasts."""
        return [f for f in self.folds if f.ok]

    def pooled(self) -> tuple[list[float], list[float], list[float]]:
        """Pooled ``(calibrated, outcomes, market_prob)`` across usable folds."""
        probs: list[float] = []
        outcomes: list[float] = []
        market: list[float] = []
        for fold in self.usable_folds:
            probs.extend(fold.calibrated_probabilities or fold.raw_probabilities)
            outcomes.extend(fold.outcomes)
            market.extend(fold.market_probabilities)
        return probs, outcomes, market

    @property
    def effective_n(self) -> int:
        """Total independent outcomes evaluated out-of-sample."""
        return sum(f.effective_n for f in self.usable_folds)


def _market_resolution(dataset: FeatureDataset) -> dict[str, datetime]:
    """When each labelled market's outcome became knowable."""
    return {
        market_id: label.resolution_ts
        for market_id, label in dataset.labels.items()
        if label.is_usable and label.resolution_ts is not None
    }


def _first_decision(examples: Sequence[TrainingExample], markets: set[str]) -> datetime | None:
    """Earliest decision timestamp among the given markets."""
    stamps = [e.decision_ts for e in examples if e.market_id in markets]
    return min(stamps) if stamps else None


def plan_folds(
    dataset: FeatureDataset, config: dict[str, Any] | None = None
) -> list[FoldPlan]:
    """Lay out chronological folds, applying grouping, purge and embargo.

    Markets are ordered by the instant their outcome became knowable, then
    sliced into expanding-window folds. Purging and the embargo are applied
    after slicing, so the guarantee holds regardless of how the slices land.

    Args:
        dataset: The built feature dataset.
        config: The ``research.walkforward`` settings section.

    Returns:
        One :class:`FoldPlan` per fold, in chronological order.
    """
    cfg = dict(config or {})
    n_folds = max(1, int(cfg.get("n_folds", 3)))
    val_frac = float(cfg.get("validation_fraction", 0.2))
    embargo = float(cfg.get("embargo_seconds", DEFAULT_EMBARGO_SECONDS))

    resolution = _market_resolution(dataset)
    ordered = sorted(resolution, key=lambda m: (resolution[m], m))
    total = len(ordered)
    if total < 3:
        return []

    # Fold size is expressed in *markets*, because the market is the unit of
    # independent information. Saying "30 markets per test block" is both
    # clearer and more honest than a row fraction, which would let a fold look
    # large while resting on a handful of outcomes.
    requested = cfg.get("test_markets_per_fold")
    if requested:
        test_size = max(1, int(requested))
    else:
        test_size = max(1, int(round(total * float(cfg.get("test_fraction", 0.2)))))
    # Never consume so much of the history that no training data remains.
    test_size = min(test_size, max(1, total // (n_folds + 1)))
    plans: list[FoldPlan] = []

    for fold_index in range(n_folds):
        # Expanding window: the test block walks forward, training grows.
        test_end = total - (n_folds - fold_index - 1) * test_size
        test_start = test_end - test_size
        if test_start <= 1:
            continue

        history = ordered[:test_start]
        val_size = max(1, int(round(len(history) * val_frac)))
        train_markets = history[: len(history) - val_size]
        validation_markets = history[len(history) - val_size :]
        test_markets = ordered[test_start:test_end]
        if not train_markets or not validation_markets or not test_markets:
            continue

        test_first_decision = _first_decision(dataset.labelled, set(test_markets))
        if test_first_decision is None:
            continue
        cutoff = test_first_decision - timedelta(seconds=embargo)

        # Purge: a training market whose outcome was not yet knowable by the
        # cutoff cannot legitimately inform a forecast made at that time.
        kept, purged = [], []
        for market in train_markets:
            (kept if resolution[market] <= cutoff else purged).append(market)
        kept_val, purged_val = [], []
        for market in validation_markets:
            (kept_val if resolution[market] <= cutoff else purged_val).append(market)

        plans.append(
            FoldPlan(
                index=fold_index,
                train_markets=tuple(kept),
                validation_markets=tuple(kept_val),
                test_markets=tuple(test_markets),
                purged_markets=tuple(purged + purged_val),
                embargo_seconds=embargo,
                train_knowledge_ts=(
                    max(resolution[m] for m in kept + kept_val) if kept + kept_val else None
                ),
                test_start_ts=test_first_decision,
            )
        )
    return plans


def _rows_for(
    examples: Sequence[TrainingExample], markets: Sequence[str], names: Sequence[str]
) -> tuple[list[list[float]], list[float], list[TrainingExample]]:
    """Feature rows, labels and examples for a set of markets."""
    wanted = set(markets)
    chosen = [e for e in examples if e.market_id in wanted and e.is_labelled]
    chosen.sort(key=lambda e: (e.decision_ts, e.market_id))
    rows = [e.features.vector(names) for e in chosen]
    labels = [float(e.label) for e in chosen if e.label is not None]
    return rows, labels, chosen


def _choose_calibrator(
    method: str, effective_n: int
) -> tuple[Any, str, list[str]]:
    """Pick a calibrator, downgrading safely when the sample is too small."""
    warnings: list[str] = []
    if method == "isotonic" and effective_n < MIN_ISOTONIC_MARKETS:
        warnings.append(
            f"Isotonic calibration skipped: only {effective_n} independent outcomes in "
            f"the validation fold, below the minimum of {MIN_ISOTONIC_MARKETS}. "
            "Fell back to logistic calibration, which has two parameters instead of "
            "one per point."
        )
        method = "logistic"
    if method == "isotonic":
        return IsotonicCalibrator(), "isotonic", warnings
    if method == "logistic":
        return LogisticCalibrator(), "logistic", warnings
    return IdentityCalibrator(), "uncalibrated", warnings


def run_walk_forward(
    dataset: FeatureDataset,
    model_name: str,
    config: dict[str, Any] | None = None,
    model_config: dict[str, Any] | None = None,
    forecaster_factory: Callable[[], Any] | None = None,
) -> WalkForwardReport:
    """Run the full walk-forward protocol for one model.

    Args:
        dataset: The built feature dataset.
        model_name: Which forecaster to run.
        config: The ``research.walkforward`` settings section.
        model_config: Model-specific settings.
        forecaster_factory: Supply a custom unfitted forecaster per fold.

    Returns:
        A :class:`WalkForwardReport` with one entry per fold.
    """
    cfg = dict(config or {})
    method = str(cfg.get("calibration_method", "isotonic"))
    names = list(dataset.feature_names)
    report = WalkForwardReport(model_name=model_name, config=cfg)

    plans = plan_folds(dataset, cfg)
    if not plans:
        report.warnings.append(
            "No usable folds: too few resolved markets to split chronologically."
        )
        return report

    for plan in plans:
        result = FoldResult(plan=plan)
        if not plan.is_usable:
            result.skipped_reason = (
                "Fold has an empty train, validation or test set after purging."
            )
            report.folds.append(result)
            continue

        train_rows, train_labels, _ = _rows_for(dataset.labelled, plan.train_markets, names)
        val_rows, val_labels, _ = _rows_for(
            dataset.labelled, plan.validation_markets, names
        )
        test_rows, test_labels, test_examples = _rows_for(
            dataset.labelled, plan.test_markets, names
        )
        if not train_rows or not val_rows or not test_rows:
            result.skipped_reason = "Fold has no rows in train, validation or test."
            report.folds.append(result)
            continue

        # Preprocessing is fitted on training rows only.
        scaler = StandardScaler().fit(train_rows, names)
        model = (
            forecaster_factory() if forecaster_factory
            else build_forecaster(model_name, model_config)
        )
        model.fit(scaler.transform(train_rows), train_labels)

        # Calibration is fitted on validation rows only, never on train or test.
        calibrator, chosen_method, warnings = _choose_calibrator(
            method, len(plan.validation_markets)
        )
        result.warnings.extend(warnings)
        val_raw = model.predict_proba(scaler.transform(val_rows))
        try:
            calibrator.fit(val_raw, val_labels)
        except (ValueError, RuntimeError) as error:
            result.warnings.append(
                f"Calibration failed ({error}); reporting uncalibrated probabilities."
            )
            calibrator = IdentityCalibrator().fit(val_raw, val_labels)
            chosen_method = "uncalibrated"

        pipeline = FittedPipeline(
            model=model,
            scaler=scaler,
            calibrator=calibrator,
            feature_names=tuple(names),
            data_version=dataset.data_version,
            calibration_method=chosen_method,
            assumptions={
                "fold": plan.index,
                "train_markets": len(plan.train_markets),
                "validation_markets": len(plan.validation_markets),
                "calibration_fitted_on": "validation fold only",
                "scaler_fitted_on": "training fold only",
                "embargo_seconds": plan.embargo_seconds,
                "purged_markets": len(plan.purged_markets),
            },
        )

        raw, calibrated = pipeline.probabilities(test_rows)
        result.pipeline = pipeline
        result.calibration_method = chosen_method
        result.raw_probabilities = raw
        result.calibrated_probabilities = calibrated or list(raw)
        result.outcomes = test_labels
        result.test_examples = test_examples
        result.market_probabilities = [e.market_prob for e in test_examples]
        result.forecasts = [
            ProbabilityForecast(
                market_ticker=example.market_id,
                decision_timestamp=example.decision_ts,
                raw_probability_yes=raw_p,
                calibrated_probability_yes=cal_p,
                confidence=_confidence(example),
                feature_version=dataset.feature_version,
                model_version=pipeline.model_version,
                data_version=dataset.data_version,
                assumptions=dict(pipeline.assumptions),
            )
            for example, raw_p, cal_p in zip(
                test_examples, raw, result.calibrated_probabilities, strict=True
            )
        ]
        report.folds.append(result)

    if not report.usable_folds:
        report.warnings.append("No fold produced out-of-sample forecasts.")
    return report


def _confidence(example: TrainingExample) -> float:
    """Confidence in one forecast, from data completeness and book quality.

    Deliberately not derived from the model's own probability: a model being
    confident is not evidence that it should be trusted.
    """
    completeness = 1.0 if example.features.is_complete else max(
        0.0, 1.0 - 0.15 * len(example.features.missing)
    )
    spread = example.features.values.get("spread", 1.0)
    liquidity = max(0.0, 1.0 - min(1.0, spread / 0.10))
    return round(max(0.0, min(1.0, 0.5 * completeness + 0.5 * liquidity)), 4)
