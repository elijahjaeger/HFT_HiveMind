"""Drift monitoring: has the world moved away from what the model was fitted on?

Deliberately simple and transparent measures, chosen so a reader can check them
by hand: missingness rates, means, standard deviations, quantiles, and the
Population Stability Index. Nothing here is a hypothesis test, and nothing here
is a trigger.

**Drift never retrains or promotes anything.** It raises a flag for a human.
That restraint is the point: a drift measure that fires automatically becomes a
mechanism for fitting the model to recent noise.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from backtest.metrics import brier_score, log_loss

PSI_BINS = 10
PSI_MINOR, PSI_MAJOR = 0.10, 0.25
"""Conventional Population Stability Index thresholds: <0.10 stable, >0.25 material."""


class DriftLevel(StrEnum):
    """How far a measure has moved from its reference."""

    STABLE = "STABLE"
    MINOR = "MINOR"
    MATERIAL = "MATERIAL"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class DriftMetric:
    """One measured comparison between a reference window and a recent one."""

    name: str
    kind: str
    reference: float | None
    current: float | None
    change: float | None
    level: DriftLevel
    note: str = ""

    def as_row(self) -> dict[str, Any]:
        """Display row for the dashboard."""
        return {
            "measure": self.name,
            "kind": self.kind,
            "reference": _round(self.reference),
            "current": _round(self.current),
            "change": _round(self.change),
            "level": self.level.value,
            "note": self.note,
        }


@dataclass
class DriftReport:
    """Every drift measure for one comparison, plus what it does and does not mean."""

    metrics: list[DriftMetric] = field(default_factory=list)
    reference_n: int = 0
    current_n: int = 0
    reference_markets: int = 0
    current_markets: int = 0

    @property
    def material(self) -> list[DriftMetric]:
        """Measures that moved materially."""
        return [m for m in self.metrics if m.level is DriftLevel.MATERIAL]

    @property
    def worst_level(self) -> DriftLevel:
        """The most severe level observed."""
        for level in (DriftLevel.MATERIAL, DriftLevel.MINOR, DriftLevel.STABLE):
            if any(m.level is level for m in self.metrics):
                return level
        return DriftLevel.UNKNOWN

    def summary(self) -> str:
        """A plain-language statement of what was found, and what happens next."""
        if not self.metrics:
            return "No drift measures could be computed."
        if self.current_markets < 20:
            return (
                f"Only {self.current_markets} recent market(s): too few to distinguish "
                "drift from noise. Treat these numbers as indicative."
            )
        if self.material:
            names = ", ".join(m.name for m in self.material[:4])
            return (
                f"{len(self.material)} measure(s) moved materially ({names}). "
                "Nothing has been retrained or promoted; this is a flag for review."
            )
        return "No material drift detected. No action taken, and none is automatic."


def population_stability_index(
    reference: Sequence[float], current: Sequence[float], bins: int = PSI_BINS
) -> float | None:
    """Population Stability Index between two samples of one variable.

    Bin edges come from the **reference** quantiles, which is what makes the
    measure asymmetric and correct: the question is how far current data has
    moved from what the model was fitted on, not the reverse.

    Returns ``None`` when either sample is too small to bin.
    """
    reference = [v for v in reference if v is not None and not math.isnan(v)]
    current = [v for v in current if v is not None and not math.isnan(v)]
    if len(reference) < bins or not current:
        return None

    ordered = sorted(reference)
    edges = [
        ordered[min(len(ordered) - 1, int(round(q * len(ordered) / bins)))]
        for q in range(1, bins)
    ]
    edges = sorted(set(edges))
    if not edges:
        return None

    def distribute(values: Sequence[float]) -> list[float]:
        counts = [0] * (len(edges) + 1)
        for value in values:
            index = 0
            while index < len(edges) and value > edges[index]:
                index += 1
            counts[index] += 1
        total = len(values) or 1
        # Floor each share so an empty bin cannot send the index to infinity.
        return [max(c / total, 1e-4) for c in counts]

    reference_shares = distribute(reference)
    current_shares = distribute(current)
    return sum(
        (c - r) * math.log(c / r)
        for r, c in zip(reference_shares, current_shares, strict=True)
    )


def _level_from_psi(value: float | None) -> DriftLevel:
    """Classify a PSI value against the conventional thresholds."""
    if value is None:
        return DriftLevel.UNKNOWN
    if value >= PSI_MAJOR:
        return DriftLevel.MATERIAL
    if value >= PSI_MINOR:
        return DriftLevel.MINOR
    return DriftLevel.STABLE


def _level_from_relative(change: float | None, minor: float, major: float) -> DriftLevel:
    """Classify a relative change against two thresholds."""
    if change is None:
        return DriftLevel.UNKNOWN
    magnitude = abs(change)
    if magnitude >= major:
        return DriftLevel.MATERIAL
    if magnitude >= minor:
        return DriftLevel.MINOR
    return DriftLevel.STABLE


def feature_drift(
    reference_rows: Sequence[Sequence[float]],
    current_rows: Sequence[Sequence[float]],
    feature_names: Sequence[str],
) -> list[DriftMetric]:
    """Per-feature distribution drift: PSI, mean shift and dispersion shift."""
    if not reference_rows or not current_rows:
        return []
    metrics: list[DriftMetric] = []
    reference_cols = list(zip(*reference_rows, strict=True))
    current_cols = list(zip(*current_rows, strict=True))

    for name, ref, cur in zip(feature_names, reference_cols, current_cols, strict=False):
        psi = population_stability_index(ref, cur)
        metrics.append(
            DriftMetric(
                name=f"{name} (PSI)", kind="distribution",
                reference=None, current=psi, change=psi, level=_level_from_psi(psi),
                note="Bins are reference quantiles." if psi is not None
                else "Too few reference rows to bin.",
            )
        )
        ref_mean, cur_mean = statistics.fmean(ref), statistics.fmean(cur)
        ref_sd = statistics.pstdev(ref) if len(ref) > 1 else 0.0
        scale = ref_sd or 1.0
        shift = (cur_mean - ref_mean) / scale
        metrics.append(
            DriftMetric(
                name=f"{name} (mean shift)", kind="location",
                reference=ref_mean, current=cur_mean, change=shift,
                level=_level_from_relative(shift, 0.25, 0.75),
                note="Change is in reference standard deviations.",
            )
        )
        cur_sd = statistics.pstdev(cur) if len(cur) > 1 else 0.0
        ratio = (cur_sd / ref_sd) if ref_sd else None
        metrics.append(
            DriftMetric(
                name=f"{name} (dispersion)", kind="scale",
                reference=ref_sd, current=cur_sd,
                change=None if ratio is None else ratio - 1.0,
                level=_level_from_relative(
                    None if ratio is None else ratio - 1.0, 0.30, 0.75
                ),
                note="Ratio of standard deviations, minus one.",
            )
        )
    return metrics


def quality_drift(
    reference_missing: Sequence[int],
    current_missing: Sequence[int],
    n_features: int,
) -> list[DriftMetric]:
    """Data-quality drift: how much more often features are missing now."""
    if not reference_missing or not current_missing:
        return []
    ref_rate = statistics.fmean(reference_missing) / max(1, n_features)
    cur_rate = statistics.fmean(current_missing) / max(1, n_features)
    return [
        DriftMetric(
            name="missingness rate", kind="quality",
            reference=ref_rate, current=cur_rate, change=cur_rate - ref_rate,
            level=_level_from_relative(cur_rate - ref_rate, 0.05, 0.15),
            note="Share of features that had to fall back to a default.",
        )
    ]


def prediction_drift(
    reference_probabilities: Sequence[float], current_probabilities: Sequence[float]
) -> list[DriftMetric]:
    """Drift in the shape of the model's own output."""
    if not reference_probabilities or not current_probabilities:
        return []
    psi = population_stability_index(reference_probabilities, current_probabilities)
    ref_mean = statistics.fmean(reference_probabilities)
    cur_mean = statistics.fmean(current_probabilities)
    return [
        DriftMetric(
            name="forecast distribution (PSI)", kind="prediction",
            reference=None, current=psi, change=psi, level=_level_from_psi(psi),
            note="How much the shape of the model's output has moved.",
        ),
        DriftMetric(
            name="mean forecast", kind="prediction",
            reference=ref_mean, current=cur_mean, change=cur_mean - ref_mean,
            level=_level_from_relative(cur_mean - ref_mean, 0.05, 0.12),
            note="A drifting mean forecast with a stable base rate is a calibration problem.",
        ),
    ]


def calibration_drift(
    reference_probabilities: Sequence[float],
    reference_outcomes: Sequence[float],
    current_probabilities: Sequence[float],
    current_outcomes: Sequence[float],
) -> list[DriftMetric]:
    """Degradation in scored accuracy, once outcomes are known.

    This is the measure that matters most and arrives last: it cannot be
    computed until the recent markets have actually settled.
    """
    if not reference_outcomes or not current_outcomes:
        return []
    metrics: list[DriftMetric] = []
    for name, fn, minor, major in (
        ("Brier score", brier_score, 0.02, 0.05),
        ("log loss", log_loss, 0.05, 0.12),
    ):
        ref = fn(reference_probabilities, reference_outcomes)
        cur = fn(current_probabilities, current_outcomes)
        change = None if ref is None or cur is None else cur - ref
        metrics.append(
            DriftMetric(
                name=name, kind="calibration",
                reference=ref, current=cur, change=change,
                level=_level_from_relative(change, minor, major),
                note="Higher is worse. Requires settled outcomes, so it lags.",
            )
        )
    return metrics


def build_drift_report(
    reference_rows: Sequence[Sequence[float]],
    current_rows: Sequence[Sequence[float]],
    feature_names: Sequence[str],
    reference_missing: Sequence[int] = (),
    current_missing: Sequence[int] = (),
    reference_probabilities: Sequence[float] = (),
    current_probabilities: Sequence[float] = (),
    reference_outcomes: Sequence[float] = (),
    current_outcomes: Sequence[float] = (),
    reference_markets: int = 0,
    current_markets: int = 0,
) -> DriftReport:
    """Assemble every drift measure into one report.

    Args:
        reference_rows: Feature rows from the training reference window.
        current_rows: Feature rows from the recent window.
        feature_names: Column names, in order.
        reference_missing: Per-row missing-feature counts, reference window.
        current_missing: Per-row missing-feature counts, recent window.
        reference_probabilities: Model output over the reference window.
        current_probabilities: Model output over the recent window.
        reference_outcomes: Settled outcomes, reference window.
        current_outcomes: Settled outcomes, recent window.
        reference_markets: Independent outcomes in the reference window.
        current_markets: Independent outcomes in the recent window.

    Returns:
        A :class:`DriftReport`. It reports; it never acts.
    """
    metrics: list[DriftMetric] = []
    metrics += feature_drift(reference_rows, current_rows, feature_names)
    metrics += quality_drift(reference_missing, current_missing, len(feature_names))
    metrics += prediction_drift(reference_probabilities, current_probabilities)
    metrics += calibration_drift(
        reference_probabilities, reference_outcomes,
        current_probabilities, current_outcomes,
    )
    return DriftReport(
        metrics=metrics,
        reference_n=len(reference_rows),
        current_n=len(current_rows),
        reference_markets=reference_markets,
        current_markets=current_markets,
    )


def _round(value: float | None, digits: int = 4) -> float | None:
    """Round, tolerating ``None`` and NaN."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return round(value, digits)
