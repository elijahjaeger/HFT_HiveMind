"""Chronological train/validation/test splits that cannot leak the future.

A calibrator maps raw forecast probabilities onto corrected ones. Fitting it
needs *outcomes*, and outcomes only exist once a market resolves - which makes
it very easy to build a calibration study that quietly cheats. Three distinct
leaks are possible, and this module blocks all three:

1. **Random splitting.** Shuffling observations mixes late data into the
   training set. The split here is always chronological.

2. **Same market on both sides.** A market is forecast many times, so its
   observations would otherwise land in several folds. The calibrator would
   then learn that market's realised outcome from the training rows and be
   scored on predicting the very same outcome. Splits are therefore grouped:
   **every observation for one market goes to exactly one fold.**

3. **Using an outcome before it was known.** Even with grouped folds, a
   calibrator fitted on markets that resolve in June cannot be applied to a
   forecast made in March. Each fold records when its outcomes became knowable
   (``knowledge_time``), and later folds drop any observation made before that
   instant. This is the rule most calibration code omits.

The dropped observations are counted and reported rather than silently
discarded, because a large drop count means the split is too thin to trust.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

CONSENSUS = "consensus"

MIN_MARKETS_PER_FOLD = 20
"""Below this many resolved markets in a fold, calibration results are noise."""


@dataclass(frozen=True)
class Observation:
    """One forecast paired with the outcome it was later scored against."""

    ts: datetime
    """When the forecast was made."""

    market_id: str
    probability: float
    outcome: float
    """Realised result encoded as ``1.0`` (YES) or ``0.0`` (NO)."""

    resolution_ts: datetime
    """When the outcome became knowable. Never earlier than ``ts``."""

    source: str = CONSENSUS
    """Which agent (or the blend) produced the forecast."""

    def __post_init__(self) -> None:
        """Reject impossible observations, including outcomes known too early."""
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError(f"probability out of range: {self.probability}")
        if self.outcome not in (0.0, 1.0):
            raise ValueError(f"outcome must be 0.0 or 1.0, got {self.outcome}")
        if self.resolution_ts < self.ts:
            raise ValueError(
                f"market {self.market_id} resolves at {self.resolution_ts}, "
                f"before the forecast at {self.ts}"
            )


@dataclass(frozen=True)
class Fold:
    """One chronological fold of observations."""

    name: str
    observations: list[Observation] = field(default_factory=list)
    market_ids: list[str] = field(default_factory=list)
    dropped: int = 0
    """Observations removed because they predate an earlier fold's outcomes."""

    def __len__(self) -> int:
        """Number of observations in the fold."""
        return len(self.observations)

    @property
    def probabilities(self) -> list[float]:
        """Raw forecast probabilities, in chronological order."""
        return [o.probability for o in self.observations]

    @property
    def outcomes(self) -> list[float]:
        """Realised outcomes, in chronological order."""
        return [o.outcome for o in self.observations]

    @property
    def start(self) -> datetime | None:
        """Timestamp of the first forecast in the fold."""
        return self.observations[0].ts if self.observations else None

    @property
    def end(self) -> datetime | None:
        """Timestamp of the last forecast in the fold."""
        return self.observations[-1].ts if self.observations else None

    @property
    def knowledge_time(self) -> datetime | None:
        """When every outcome in this fold had become knowable.

        A calibrator fitted on this fold may only be applied to forecasts made
        at or after this instant.
        """
        if not self.observations:
            return None
        return max(o.resolution_ts for o in self.observations)

    @property
    def base_rate(self) -> float | None:
        """Share of observations that resolved YES."""
        if not self.observations:
            return None
        return sum(o.outcome for o in self.observations) / len(self.observations)

    @property
    def effective_n(self) -> int:
        """Number of *independent* observations, which is the market count.

        This is the number that governs how much can be learned. A market is
        forecast many times, but it resolves once, so those forecasts share a
        single Bernoulli outcome and are very far from independent. A fold with
        4000 observations across 8 markets carries roughly as much calibration
        information as 8 coin flips - not 4000.
        """
        return len(self.market_ids)


@dataclass(frozen=True)
class ChronologicalSplit:
    """Train/validation/test folds, ordered in time and free of leakage."""

    train: Fold
    validation: Fold
    test: Fold
    grouped_by_market: bool = True
    enforce_knowledge_time: bool = True

    @property
    def folds(self) -> list[Fold]:
        """The three folds in chronological order."""
        return [self.train, self.validation, self.test]

    @property
    def total_dropped(self) -> int:
        """Observations dropped across all folds to prevent leakage."""
        return sum(fold.dropped for fold in self.folds)

    @property
    def is_usable(self) -> bool:
        """True when every fold retained at least one observation."""
        return all(len(fold) > 0 for fold in self.folds)

    @property
    def warnings(self) -> list[str]:
        """Reasons to distrust any calibration fitted on this split.

        These are about *statistical* validity, not correctness of the split.
        The split can be perfectly clean and still far too thin to learn from.
        """
        messages: list[str] = []
        for fold in self.folds:
            if len(fold) == 0:
                messages.append(f"The {fold.name} fold is empty.")
                continue
            # These are independent problems: a fold can be both too thin and
            # degenerate, and each one invalidates the results differently.
            if fold.effective_n < MIN_MARKETS_PER_FOLD:
                messages.append(
                    f"The {fold.name} fold covers only {fold.effective_n} resolved "
                    f"market(s) ({len(fold)} observations). Calibration is learned "
                    "per market, not per observation, so this is far too few to "
                    "fit or judge a calibrator on."
                )
            if fold.base_rate in (0.0, 1.0):
                messages.append(
                    f"Every market in the {fold.name} fold resolved the same way "
                    f"(base rate {fold.base_rate:.0%}); scores from it are meaningless."
                )
        if self.total_dropped:
            share = self.total_dropped / max(
                1, self.total_dropped + sum(len(f) for f in self.folds)
            )
            messages.append(
                f"{self.total_dropped} observation(s) ({share:.0%}) were dropped "
                "because they were made before an earlier fold's outcomes were "
                "known. A large share here means the folds overlap in time."
            )
        return messages

    def summary_rows(self) -> list[dict[str, Any]]:
        """One row per fold, for display."""
        rows = []
        for fold in self.folds:
            rows.append(
                {
                    "fold": fold.name,
                    "observations": len(fold),
                    "markets": len(fold.market_ids),
                    "first_forecast": fold.start,
                    "last_forecast": fold.end,
                    "outcomes_known_at": fold.knowledge_time,
                    "effective_n_markets": fold.effective_n,
                    "base_rate": fold.base_rate,
                    "dropped_for_leakage": fold.dropped,
                }
            )
        return rows


def observations_from_result(
    result: Any,
    source: str = CONSENSUS,
) -> list[Observation]:
    """Extract calibration observations from a completed backtest result.

    Args:
        result: A :class:`backtest.engine.BacktestResult`.
        source: ``"consensus"`` for the blended probability actually traded on,
            or an agent name to score that agent's own forecasts.

    Returns:
        Observations sorted by forecast time, covering only markets that both
        resolved and carry a resolution timestamp.
    """
    resolved: dict[str, tuple[float, datetime]] = {}
    for market in getattr(result, "markets", []):
        if market.outcome is None:
            continue
        resolution_ts = market.resolution_time or market.close_time
        resolved[market.market_id] = (
            1.0 if market.outcome.value == "YES" else 0.0,
            resolution_ts,
        )

    records: Iterable[tuple[datetime, str, float | None]]
    if source == CONSENSUS:
        records = (
            (r["ts"], r["market_id"], r.get("probability"))
            for r in getattr(result, "consensus", [])
        )
    else:
        records = (
            (f.ts, f.market_id, f.probability)
            for f in getattr(result, "forecasts", [])
            if f.agent_name == source
        )

    observations: list[Observation] = []
    for ts, market_id, probability in records:
        if probability is None or market_id not in resolved:
            continue
        outcome, resolution_ts = resolved[market_id]
        if ts > resolution_ts:
            # A forecast made after resolution is not a forecast; skip it.
            continue
        observations.append(
            Observation(
                ts=ts,
                market_id=market_id,
                probability=float(probability),
                outcome=outcome,
                resolution_ts=resolution_ts,
                source=source,
            )
        )
    observations.sort(key=lambda o: (o.ts, o.market_id))
    return observations


def _market_order(observations: Sequence[Observation]) -> list[str]:
    """Markets ordered by when they resolved, then by first forecast, then id."""
    first_seen: dict[str, tuple[datetime, datetime, str]] = {}
    for observation in observations:
        key = (observation.resolution_ts, observation.ts, observation.market_id)
        current = first_seen.get(observation.market_id)
        if current is None or key < current:
            first_seen[observation.market_id] = key
    return [mid for mid, _ in sorted(first_seen.items(), key=lambda item: item[1])]


def chronological_split(
    observations: Sequence[Observation],
    train_fraction: float = 0.5,
    validation_fraction: float = 0.25,
    group_by_market: bool = True,
    enforce_knowledge_time: bool = True,
) -> ChronologicalSplit:
    """Split observations into chronological train/validation/test folds.

    Args:
        observations: Observations to split; order does not matter.
        train_fraction: Target share of observations for training.
        validation_fraction: Target share for validation. The remainder is test.
        group_by_market: Keep every observation for a market in one fold. Turning
            this off permits same-market leakage and is only for experiments.
        enforce_knowledge_time: Drop observations in later folds that predate an
            earlier fold's outcomes becoming knowable.

    Returns:
        A :class:`ChronologicalSplit`.

    Raises:
        ValueError: If the fractions are not a valid three-way division.
    """
    if train_fraction <= 0 or validation_fraction < 0:
        raise ValueError("train_fraction must be > 0 and validation_fraction >= 0")
    if train_fraction + validation_fraction >= 1.0:
        raise ValueError(
            "train_fraction + validation_fraction must leave room for a test fold"
        )

    ordered = sorted(observations, key=lambda o: (o.ts, o.market_id))
    if not ordered:
        return ChronologicalSplit(
            train=Fold("train"),
            validation=Fold("validation"),
            test=Fold("test"),
            grouped_by_market=group_by_market,
            enforce_knowledge_time=enforce_knowledge_time,
        )

    total = len(ordered)
    train_target = train_fraction * total
    validation_target = (train_fraction + validation_fraction) * total

    buckets: dict[str, list[Observation]] = {"train": [], "validation": [], "test": []}
    fold_markets: dict[str, list[str]] = {"train": [], "validation": [], "test": []}

    if group_by_market:
        by_market: dict[str, list[Observation]] = {}
        for observation in ordered:
            by_market.setdefault(observation.market_id, []).append(observation)
        assigned = 0
        for market_id in _market_order(ordered):
            group = by_market[market_id]
            if assigned < train_target:
                name = "train"
            elif assigned < validation_target:
                name = "validation"
            else:
                name = "test"
            buckets[name].extend(group)
            fold_markets[name].append(market_id)
            assigned += len(group)
    else:
        for index, observation in enumerate(ordered):
            if index < train_target:
                name = "train"
            elif index < validation_target:
                name = "validation"
            else:
                name = "test"
            buckets[name].append(observation)
        for name, group in buckets.items():
            fold_markets[name] = sorted({o.market_id for o in group})

    for group in buckets.values():
        group.sort(key=lambda o: (o.ts, o.market_id))

    train = Fold("train", buckets["train"], fold_markets["train"])
    validation = _apply_knowledge_time(
        Fold("validation", buckets["validation"], fold_markets["validation"]),
        [train],
        enforce_knowledge_time,
    )
    test = _apply_knowledge_time(
        Fold("test", buckets["test"], fold_markets["test"]),
        [train, validation],
        enforce_knowledge_time,
    )
    return ChronologicalSplit(
        train=train,
        validation=validation,
        test=test,
        grouped_by_market=group_by_market,
        enforce_knowledge_time=enforce_knowledge_time,
    )


def _apply_knowledge_time(
    fold: Fold, earlier: Sequence[Fold], enforce: bool
) -> Fold:
    """Drop observations made before the earlier folds' outcomes were knowable."""
    if not enforce or not fold.observations:
        return fold
    times = [f.knowledge_time for f in earlier if f.knowledge_time is not None]
    if not times:
        return fold
    cutoff = max(times)
    kept = [o for o in fold.observations if o.ts >= cutoff]
    return Fold(
        name=fold.name,
        observations=kept,
        market_ids=sorted({o.market_id for o in kept}),
        dropped=len(fold.observations) - len(kept),
    )
