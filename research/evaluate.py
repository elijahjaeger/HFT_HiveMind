"""Scoring candidates against frozen baselines.

Everything here is computed on out-of-sample fold test data only. Two habits
are enforced rather than left to the reader:

* **Every metric is reported next to the baseline's.** A Brier score of 0.21 is
  meaningless alone; against a market baseline of 0.20 it is a loss.
* **Every metric carries its effective sample size.** ``effective_n`` counts
  resolved markets, not forecast rows, because rows from one market share one
  outcome. A difference that looks decisive across 4000 rows may rest on 30
  coin flips.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from backtest.metrics import brier_score, calibration_table, log_loss
from research.walkforward import WalkForwardReport

MIN_MARKETS_FOR_RELIABLE = 50


@dataclass(frozen=True)
class ScoreCard:
    """Out-of-sample scores for one model."""

    model_name: str
    n_forecasts: int
    effective_n: int
    brier: float | None
    log_loss: float | None
    ece: float | None
    mean_forecast: float | None
    base_rate: float | None
    calibration_method: str = "none"
    warnings: tuple[str, ...] = ()

    @property
    def is_reliable(self) -> bool:
        """Whether the sample supports drawing a conclusion at all."""
        return self.effective_n >= MIN_MARKETS_FOR_RELIABLE

    def as_row(self) -> dict[str, Any]:
        """Display row for the comparison table."""
        return {
            "model": self.model_name,
            "brier": None if self.brier is None else round(self.brier, 4),
            "log_loss": None if self.log_loss is None else round(self.log_loss, 4),
            "ece": None if self.ece is None else round(self.ece, 4),
            "mean_forecast": (
                None if self.mean_forecast is None else round(self.mean_forecast, 4)
            ),
            "base_rate": None if self.base_rate is None else round(self.base_rate, 4),
            "forecasts": self.n_forecasts,
            "markets (effective n)": self.effective_n,
            "calibration": self.calibration_method,
            "reliable?": "yes" if self.is_reliable
            else f"NO - under {MIN_MARKETS_FOR_RELIABLE} markets",
        }


@dataclass
class Comparison:
    """A candidate measured against the frozen baselines."""

    candidate: ScoreCard
    baselines: list[ScoreCard] = field(default_factory=list)

    @property
    def market_baseline(self) -> ScoreCard | None:
        """The market-price baseline, which is the bar that matters."""
        for card in self.baselines:
            if "market" in card.model_name:
                return card
        return self.baselines[0] if self.baselines else None

    def brier_delta(self) -> float | None:
        """Candidate Brier minus baseline Brier. Negative is better."""
        baseline = self.market_baseline
        if baseline is None or baseline.brier is None or self.candidate.brier is None:
            return None
        return self.candidate.brier - baseline.brier

    def beats_baseline(self) -> bool | None:
        """Whether the candidate scores better, or ``None`` if unmeasurable."""
        delta = self.brier_delta()
        return None if delta is None else delta < 0.0

    def verdict(self) -> str:
        """A plain-language verdict a non-specialist can act on."""
        delta = self.brier_delta()
        if delta is None:
            return "Not comparable: one of the scores could not be computed."
        if not self.candidate.is_reliable:
            return (
                f"Not enough data to judge. {self.candidate.effective_n} resolved "
                f"market(s) is below the {MIN_MARKETS_FOR_RELIABLE} needed for a "
                "conclusion, whichever way the numbers point."
            )
        if delta < -0.005:
            return (
                f"Beats the market baseline on Brier score by {abs(delta):.4f}. "
                "This is a simulation result on out-of-sample folds, not evidence "
                "of executable profit."
            )
        if delta > 0.005:
            return f"Worse than the market baseline by {delta:.4f} Brier. Do not promote."
        return (
            "Indistinguishable from the market baseline. The market price already "
            "contains what this model knows."
        )


def expected_calibration_error(
    probabilities: Sequence[float], outcomes: Sequence[float], n_bins: int = 10
) -> float | None:
    """Average gap between predicted probability and observed frequency.

    Weighted by bin population, so sparsely-populated bins cannot dominate.
    """
    table = calibration_table(probabilities, outcomes, n_bins)
    total = sum(bucket.count for bucket in table)
    if not total:
        return None
    error = 0.0
    for bucket in table:
        if bucket.count and bucket.mean_predicted is not None:
            gap = abs(bucket.mean_predicted - (bucket.observed_frequency or 0.0))
            error += (bucket.count / total) * gap
    return error


def score(
    model_name: str,
    probabilities: Sequence[float],
    outcomes: Sequence[float],
    effective_n: int,
    calibration_method: str = "none",
    warnings: Sequence[str] = (),
) -> ScoreCard:
    """Score one model's out-of-sample forecasts."""
    extra = list(warnings)
    if effective_n < MIN_MARKETS_FOR_RELIABLE:
        extra.append(
            f"Only {effective_n} independent outcomes: metrics are indicative at best."
        )
    return ScoreCard(
        model_name=model_name,
        n_forecasts=len(probabilities),
        effective_n=effective_n,
        brier=brier_score(probabilities, outcomes),
        log_loss=log_loss(probabilities, outcomes),
        ece=expected_calibration_error(probabilities, outcomes),
        mean_forecast=(sum(probabilities) / len(probabilities)) if probabilities else None,
        base_rate=(sum(outcomes) / len(outcomes)) if outcomes else None,
        calibration_method=calibration_method,
        warnings=tuple(extra),
    )


def score_report(report: WalkForwardReport) -> ScoreCard:
    """Score a whole walk-forward run by pooling its out-of-sample folds."""
    probabilities, outcomes, _ = report.pooled()
    methods = {f.calibration_method for f in report.usable_folds} or {"none"}
    warnings = list(report.warnings)
    for fold in report.usable_folds:
        warnings.extend(fold.warnings)
    return score(
        report.model_name,
        probabilities,
        outcomes,
        report.effective_n,
        "/".join(sorted(methods)),
        warnings,
    )


def score_market_baseline(report: WalkForwardReport) -> ScoreCard:
    """Score the market price on exactly the same rows the candidate saw.

    Scoring the baseline on the identical rows is what makes the comparison
    fair: a baseline evaluated on a different sample is not a baseline.
    """
    _, outcomes, market = report.pooled()
    return score("market_price_baseline", market, outcomes, report.effective_n, "none")


def reliability_points(
    probabilities: Sequence[float], outcomes: Sequence[float], n_bins: int = 10
) -> list[dict[str, Any]]:
    """Points for a reliability diagram, one per populated bin."""
    return [
        {
            "bin": bucket.bucket,
            "mean_predicted": bucket.mean_predicted,
            "observed_frequency": bucket.observed_frequency,
            "count": bucket.count,
        }
        for bucket in calibration_table(probabilities, outcomes, n_bins)
        if bucket.count
    ]


def calibration_rows(
    probabilities: Sequence[float], outcomes: Sequence[float], n_bins: int = 10
) -> list[dict[str, Any]]:
    """The calibration table, including empty bins so gaps are visible."""
    rows: list[dict[str, Any]] = []
    for bucket in calibration_table(probabilities, outcomes, n_bins):
        gap = bucket.gap
        rows.append(
            {
                "bin": bucket.bucket,
                "forecasts": bucket.count,
                "mean_predicted": (
                    None if bucket.mean_predicted is None else round(bucket.mean_predicted, 4)
                ),
                "observed_yes_rate": (
                    None if bucket.observed_frequency is None
                    else round(bucket.observed_frequency, 4)
                ),
                "gap": None if gap is None else round(gap, 4),
            }
        )
    return rows


def fold_rows(report: WalkForwardReport) -> list[dict[str, Any]]:
    """Per-fold scores, so fold-to-fold variation is visible.

    A candidate that wins on average but loses on a third of folds has not
    demonstrated anything durable, and pooling hides exactly that.
    """
    rows: list[dict[str, Any]] = []
    for fold in report.folds:
        if not fold.ok:
            rows.append(
                {
                    "fold": fold.plan.index,
                    "status": f"skipped - {fold.skipped_reason}",
                    "forecasts": 0,
                    "markets": 0,
                }
            )
            continue
        probabilities = fold.calibrated_probabilities or fold.raw_probabilities
        rows.append(
            {
                "fold": fold.plan.index,
                "status": "ok",
                "train_markets": len(fold.plan.train_markets),
                "val_markets": len(fold.plan.validation_markets),
                "test_markets": len(fold.plan.test_markets),
                "purged": len(fold.plan.purged_markets),
                "forecasts": len(fold.outcomes),
                "calibration": fold.calibration_method,
                "brier": _round(brier_score(probabilities, fold.outcomes)),
                "brier_market": _round(
                    brier_score(fold.market_probabilities, fold.outcomes)
                ),
                "log_loss": _round(log_loss(probabilities, fold.outcomes)),
            }
        )
    return rows


def calibration_comparison(report: WalkForwardReport) -> list[dict[str, Any]]:
    """Uncalibrated vs calibrated vs market, on identical out-of-sample rows."""
    raw: list[float] = []
    calibrated: list[float] = []
    outcomes: list[float] = []
    market: list[float] = []
    for fold in report.usable_folds:
        raw.extend(fold.raw_probabilities)
        calibrated.extend(fold.calibrated_probabilities or fold.raw_probabilities)
        outcomes.extend(fold.outcomes)
        market.extend(fold.market_probabilities)

    methods = {f.calibration_method for f in report.usable_folds} or {"none"}
    label = "/".join(sorted(methods))
    return [
        card.as_row()
        for card in (
            score(f"{report.model_name} (uncalibrated)", raw, outcomes, report.effective_n),
            score(
                f"{report.model_name} ({label})", calibrated, outcomes,
                report.effective_n, label,
            ),
            score("market price (baseline)", market, outcomes, report.effective_n),
        )
    ]


def _round(value: float | None, digits: int = 4) -> float | None:
    """Round, tolerating ``None``."""
    return None if value is None or math.isnan(value) else round(value, digits)
