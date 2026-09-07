"""Fit calibrators on the training fold and score them on held-out data.

The protocol is the conventional one, and the fold roles are kept strictly
separate:

* **train** - the only fold a calibrator is fitted on.
* **validation** - used to *choose* between methods.
* **test** - scored once, reported, and never used to choose anything.

Train scores are in-sample by construction and are reported only so the gap
between train and test is visible; isotonic regression in particular will look
excellent on train and much worse on test when the training fold is thin.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from backtest.metrics import (
    CalibrationBucket,
    brier_score,
    calibration_table,
    expected_calibration_error,
    log_loss,
)
from calibration.methods import Calibrator, default_calibrators
from calibration.splits import ChronologicalSplit, Fold

LOWER_IS_BETTER = {"brier", "log_loss", "ece"}


@dataclass(frozen=True)
class FoldScores:
    """Scores for one method on one fold."""

    fold: str
    n: int
    brier: float | None = None
    log_loss: float | None = None
    ece: float | None = None
    mean_prediction: float | None = None
    base_rate: float | None = None
    in_sample: bool = False
    buckets: list[CalibrationBucket] = field(default_factory=list)

    def metric(self, name: str) -> float | None:
        """Look a metric up by name."""
        return {"brier": self.brier, "log_loss": self.log_loss, "ece": self.ece}[name]


@dataclass
class MethodResult:
    """One calibration method, fitted and scored across every fold."""

    name: str
    calibrator: Calibrator
    scores: dict[str, FoldScores] = field(default_factory=dict)
    parameters: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    @property
    def test(self) -> FoldScores | None:
        """Scores on the held-out test fold."""
        return self.scores.get("test")

    @property
    def validation(self) -> FoldScores | None:
        """Scores on the validation fold."""
        return self.scores.get("validation")


@dataclass
class CalibrationReport:
    """The full comparison across methods and folds."""

    split: ChronologicalSplit
    methods: list[MethodResult] = field(default_factory=list)
    source: str = "consensus"
    n_buckets: int = 10
    selection_metric: str = "brier"

    @property
    def warnings(self) -> list[str]:
        """Statistical caveats that apply to every number in this report."""
        return list(self.split.warnings)

    @property
    def is_trustworthy(self) -> bool:
        """True when no fold is too thin or degenerate to learn from."""
        return not self.split.warnings

    def method(self, name: str) -> MethodResult | None:
        """Look up one method's result by name."""
        return next((m for m in self.methods if m.name == name), None)

    @property
    def baseline(self) -> MethodResult | None:
        """The uncalibrated baseline every other method is compared against."""
        return self.method("uncalibrated")

    def selected(self) -> MethodResult | None:
        """The method with the best validation score on the selection metric.

        Selection uses the validation fold only. The test fold is never
        consulted to pick a winner.
        """
        candidates = [
            m
            for m in self.methods
            if m.error is None
            and m.validation is not None
            and m.validation.metric(self.selection_metric) is not None
        ]
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda m: m.validation.metric(self.selection_metric),  # type: ignore[union-attr]
        )

    def comparison_rows(self) -> list[dict[str, Any]]:
        """One row per method and fold, for the comparison table."""
        rows: list[dict[str, Any]] = []
        for method in self.methods:
            if method.error is not None:
                rows.append({"method": method.name, "fold": "-", "error": method.error})
                continue
            for fold_name in ("train", "validation", "test"):
                scores = method.scores.get(fold_name)
                if scores is None:
                    continue
                rows.append(
                    {
                        "method": method.name,
                        "fold": fold_name + (" (in-sample)" if scores.in_sample else ""),
                        "n": scores.n,
                        "brier": scores.brier,
                        "log_loss": scores.log_loss,
                        "ece": scores.ece,
                        "mean_prediction": scores.mean_prediction,
                        "base_rate": scores.base_rate,
                    }
                )
        return rows

    def test_rows(self) -> list[dict[str, Any]]:
        """One row per method on the held-out test fold, with the delta."""
        baseline = self.baseline.test if self.baseline else None
        rows: list[dict[str, Any]] = []
        for method in self.methods:
            scores = method.test
            if scores is None:
                continue
            delta = None
            if baseline is not None and baseline.brier is not None and scores.brier is not None:
                delta = scores.brier - baseline.brier
            rows.append(
                {
                    "method": method.name,
                    "n": scores.n,
                    "brier": scores.brier,
                    "brier_vs_baseline": delta,
                    "log_loss": scores.log_loss,
                    "ece": scores.ece,
                }
            )
        return rows

    def reliability_series(
        self, fold: str = "test"
    ) -> dict[str, list[tuple[float, float]]]:
        """``(mean predicted, observed frequency)`` points per method.

        Only populated buckets are returned, so the diagram never implies data
        where there is none.
        """
        series: dict[str, list[tuple[float, float]]] = {}
        for method in self.methods:
            scores = method.scores.get(fold)
            if scores is None:
                continue
            points = [
                (bucket.mean_predicted, bucket.observed_frequency)
                for bucket in scores.buckets
                if bucket.count > 0
                and bucket.mean_predicted is not None
                and bucket.observed_frequency is not None
            ]
            if points:
                series[method.name] = points
        return series

    def calibration_rows(self, method: str, fold: str = "test") -> list[dict[str, Any]]:
        """The calibration table for one method and fold."""
        result = self.method(method)
        scores = result.scores.get(fold) if result else None
        if scores is None:
            return []
        return [
            {
                "bucket": bucket.bucket,
                "count": bucket.count,
                "mean_predicted": bucket.mean_predicted,
                "observed_frequency": bucket.observed_frequency,
                "gap": bucket.gap,
            }
            for bucket in scores.buckets
        ]


def _score_fold(
    fold: Fold,
    calibrator: Calibrator,
    n_buckets: int,
    in_sample: bool,
) -> FoldScores:
    """Apply a fitted calibrator to one fold and score the result."""
    if not fold.observations:
        return FoldScores(fold=fold.name, n=0, in_sample=in_sample)
    calibrated = calibrator.transform(fold.probabilities)
    outcomes = fold.outcomes
    return FoldScores(
        fold=fold.name,
        n=len(calibrated),
        brier=brier_score(calibrated, outcomes),
        log_loss=log_loss(calibrated, outcomes),
        ece=expected_calibration_error(calibrated, outcomes, n_buckets),
        mean_prediction=sum(calibrated) / len(calibrated),
        base_rate=fold.base_rate,
        in_sample=in_sample,
        buckets=calibration_table(calibrated, outcomes, n_buckets),
    )


def evaluate_calibrators(
    split: ChronologicalSplit,
    calibrators: Sequence[Calibrator] | None = None,
    n_buckets: int = 10,
    selection_metric: str = "brier",
    source: str = "consensus",
) -> CalibrationReport:
    """Fit every calibrator on the training fold and score it on all three.

    Args:
        split: A chronological split. Only ``split.train`` is used for fitting.
        calibrators: Methods to compare; defaults to identity, isotonic, logistic.
        n_buckets: Buckets for the calibration table and ECE.
        selection_metric: ``"brier"``, ``"log_loss"`` or ``"ece"`` - used on the
            validation fold to pick a winner.
        source: Which forecast source these observations came from.

    Returns:
        A :class:`CalibrationReport`.

    Raises:
        ValueError: If the training fold is empty or the metric is unknown.
    """
    if selection_metric not in LOWER_IS_BETTER:
        raise ValueError(f"unknown selection metric: {selection_metric}")
    if not split.train.observations:
        raise ValueError(
            "the training fold is empty; there is nothing to fit a calibrator on"
        )

    methods: list[MethodResult] = []
    train_probabilities = split.train.probabilities
    train_outcomes = split.train.outcomes

    for calibrator in calibrators if calibrators is not None else default_calibrators():
        result = MethodResult(name=calibrator.name, calibrator=calibrator)
        try:
            calibrator.fit(train_probabilities, train_outcomes)
        except (ValueError, RuntimeError, ZeroDivisionError) as error:
            result.error = f"{type(error).__name__}: {error}"
            methods.append(result)
            continue

        parameters = getattr(calibrator, "parameters", None)
        if isinstance(parameters, dict):
            result.parameters = dict(parameters)

        for fold in split.folds:
            result.scores[fold.name] = _score_fold(
                fold, calibrator, n_buckets, in_sample=(fold.name == "train")
            )
        methods.append(result)

    return CalibrationReport(
        split=split,
        methods=methods,
        source=source,
        n_buckets=n_buckets,
        selection_metric=selection_metric,
    )
