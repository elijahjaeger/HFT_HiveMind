"""Repeated backtests over resampled markets, for a distribution rather than a number.

Why this exists
---------------
A single backtest is a single sample. It produces one equity curve, one total
return, one drawdown - and none of those numbers tell you whether the result is
an edge or an accident of which markets happened to be in the dataset. Run the
same strategy over a different subset of markets and the number moves, often a
lot.

A campaign runs the same strategy many times over different random subsets of
the same corpus and keeps every outcome. What you read afterwards is a
*distribution*: the median return, and the range the middle of the outcomes fell
in. "The 5th-95th percentile band straddles zero" is a far more useful sentence
than "the backtest returned +1.2%".

What this is NOT
----------------
**This is not out-of-sample validation and it cannot replace the promotion
gate.** Every run here draws from the same corpus, and every market in it was
already used to build the strategy being tested. Resampling measures
*robustness to market selection* - how much the answer depends on which markets
you happened to look at. It says nothing about whether the strategy generalises
to markets it has never seen. That question belongs to
``research/walkforward.py`` and the gate in ``research/registry.py``, and no
amount of resampling substitutes for them.

Read a campaign as "how fragile is this result?", never as "does this work?".

**Simulation-only.** Every run goes through ``backtest.engine.run_backtest``,
which terminates in an in-memory portfolio. No order path, no venue writes.
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from backtest.engine import run_backtest
from core.dataset import MarketDataset
from core.jobs import ProgressThrottle

CAMPAIGN_VERSION = "campaign-v1"

DEFAULT_GRID_POINTS = 40
"""Points each equity curve is resampled onto before percentiles are taken.

Runs differ in length, so curves must share an x-axis before they can be
compared pointwise. The axis is *progress through the run* in [0, 1], not wall
time, because the runs cover different market subsets and different spans.
"""


@dataclass(frozen=True)
class CampaignConfig:
    """How long to run, and how to sample."""

    duration_seconds: float = 60.0
    """Wall-clock budget. The campaign stops after this, mid-sequence if needed."""

    sample_fraction: float = 0.6
    """Share of the corpus's markets drawn for each run.

    Below about half the runs get too few markets to trade at all; above about
    0.8 every run sees nearly the same markets and the band collapses to a line,
    which looks like confidence but is really just repetition.
    """

    min_markets: int = 3
    seed: int = 20260101
    max_runs: int = 100_000
    """A safety cap. The duration is the real limit."""

    grid_points: int = DEFAULT_GRID_POINTS


@dataclass(frozen=True)
class RunOutcome:
    """What one resampled backtest produced."""

    index: int
    n_markets: int
    total_return: float
    max_drawdown: float
    n_trades: int
    brier: float | None = None
    curve: tuple[float, ...] = ()
    """Equity as a return fraction, resampled onto the shared progress grid."""


@dataclass(frozen=True)
class ReturnBand:
    """Pointwise percentiles across every run's equity curve.

    This is the "confidence" a campaign can honestly report: the range the
    middle of the outcomes fell in. It is a description of the runs that
    happened, not a probability statement about the future.
    """

    grid: tuple[float, ...]
    p05: tuple[float, ...]
    p25: tuple[float, ...]
    median: tuple[float, ...]
    p75: tuple[float, ...]
    p95: tuple[float, ...]

    @property
    def final_median(self) -> float:
        """Median return at the end of the run."""
        return self.median[-1] if self.median else 0.0

    @property
    def final_low(self) -> float:
        """5th-percentile return at the end of the run."""
        return self.p05[-1] if self.p05 else 0.0

    @property
    def final_high(self) -> float:
        """95th-percentile return at the end of the run."""
        return self.p95[-1] if self.p95 else 0.0

    @property
    def straddles_zero(self) -> bool:
        """True when the 5-95 band contains zero at the end of the run.

        The single most important thing on the chart. When this is true the
        campaign has not distinguished the strategy from doing nothing.
        """
        return self.final_low <= 0.0 <= self.final_high


@dataclass
class CampaignProgress:
    """A snapshot of a campaign in flight, safe to hand to another thread."""

    runs: int = 0
    elapsed: float = 0.0
    duration: float = 0.0
    returns: tuple[float, ...] = ()
    band: ReturnBand | None = None
    latest: RunOutcome | None = None

    @property
    def fraction(self) -> float:
        """Completion against the time budget, in ``[0, 1]``."""
        if self.duration <= 0:
            return 0.0
        return max(0.0, min(1.0, self.elapsed / self.duration))

    @property
    def percent(self) -> int:
        """Completion against the time budget, as a whole percentage."""
        return int(round(100 * self.fraction))

    @property
    def median_return(self) -> float:
        """Median total return across every run so far."""
        return float(np.median(self.returns)) if self.returns else 0.0

    @property
    def win_rate(self) -> float:
        """Share of runs that ended above zero."""
        if not self.returns:
            return 0.0
        return sum(1 for r in self.returns if r > 0.0) / len(self.returns)


@dataclass
class CampaignResult:
    """Every run, and what the spread of them says."""

    outcomes: list[RunOutcome] = field(default_factory=list)
    band: ReturnBand | None = None
    elapsed: float = 0.0
    cancelled: bool = False
    corpus_markets: int = 0
    is_synthetic: bool = True
    notes: list[str] = field(default_factory=list)

    @property
    def returns(self) -> list[float]:
        """Total return from each completed run."""
        return [o.total_return for o in self.outcomes]

    @property
    def median_return(self) -> float:
        """Median total return across every completed run."""
        return float(np.median(self.returns)) if self.outcomes else 0.0

    @property
    def win_rate(self) -> float:
        """Share of completed runs that ended above zero."""
        if not self.outcomes:
            return 0.0
        return sum(1 for o in self.outcomes if o.total_return > 0.0) / len(self.outcomes)

    def verdict(self) -> str:
        """One plain sentence a reader can act on."""
        if len(self.outcomes) < 5:
            return (
                f"Only {len(self.outcomes)} run(s) completed - far too few to say "
                "anything. Give it longer."
            )
        band = self.band
        median = self.median_return
        if band is None:
            return f"{len(self.outcomes)} runs, median return {median:+.2%}."
        if band.straddles_zero:
            return (
                f"{len(self.outcomes)} runs, median {median:+.2%}, but the 5-95% band "
                f"({band.final_low:+.2%} to {band.final_high:+.2%}) includes zero. "
                "This has NOT been distinguished from doing nothing."
            )
        direction = "profitable" if band.final_low > 0 else "loss-making"
        return (
            f"{len(self.outcomes)} runs, median {median:+.2%}, 5-95% band "
            f"{band.final_low:+.2%} to {band.final_high:+.2%} - consistently "
            f"{direction} across market subsets. This measures robustness to market "
            "selection only, NOT out-of-sample validity."
        )


# --------------------------------------------------------------------------
# Curves and percentiles
# --------------------------------------------------------------------------


def normalise_curve(
    equity: list[float], initial_capital: float, grid_points: int = DEFAULT_GRID_POINTS
) -> tuple[float, ...]:
    """Resample an equity series onto a fixed progress grid, as return fractions.

    Runs cover different market subsets over different spans, so their raw
    curves cannot be compared pointwise. Interpolating each onto the same
    ``[0, 1]`` progress axis makes percentiles across runs meaningful.
    """
    if grid_points < 2:
        raise ValueError("grid_points must be at least 2")
    if not equity or initial_capital <= 0:
        return tuple(0.0 for _ in range(grid_points))

    returns = np.asarray(equity, dtype=float) / initial_capital - 1.0
    if len(returns) == 1:
        return tuple(float(returns[0]) for _ in range(grid_points))

    source = np.linspace(0.0, 1.0, len(returns))
    target = np.linspace(0.0, 1.0, grid_points)
    return tuple(float(v) for v in np.interp(target, source, returns))


def build_band(curves: list[tuple[float, ...]]) -> ReturnBand | None:
    """Pointwise percentiles across equally-gridded curves."""
    usable = [c for c in curves if c]
    if not usable:
        return None
    width = len(usable[0])
    if any(len(c) != width for c in usable):
        raise ValueError("every curve must share the same grid")

    matrix = np.asarray(usable, dtype=float)
    percentiles = np.percentile(matrix, [5, 25, 50, 75, 95], axis=0)
    return ReturnBand(
        grid=tuple(float(v) for v in np.linspace(0.0, 1.0, width)),
        p05=tuple(float(v) for v in percentiles[0]),
        p25=tuple(float(v) for v in percentiles[1]),
        median=tuple(float(v) for v in percentiles[2]),
        p75=tuple(float(v) for v in percentiles[3]),
        p95=tuple(float(v) for v in percentiles[4]),
    )


def sample_markets(
    dataset: MarketDataset, fraction: float, rng: random.Random, minimum: int = 3
) -> list[str]:
    """Draw a random subset of market ids, without replacement."""
    ids = [m.market_id for m in dataset.markets]
    if not ids:
        return []
    wanted = max(minimum, int(round(len(ids) * fraction)))
    return rng.sample(ids, min(wanted, len(ids)))


def run_one(
    dataset: MarketDataset,
    settings: dict[str, Any],
    market_ids: list[str],
    index: int,
    grid_points: int,
) -> RunOutcome:
    """One backtest restricted to ``market_ids``."""
    chosen = set(market_ids)
    markets = [m for m in dataset.markets if m.market_id in chosen]
    quotes = [q for q in dataset.quotes if q.market_id in chosen]
    trades = [t for t in dataset.trades if t.market_id in chosen]

    result = run_backtest(markets, quotes, trades, settings)
    metrics = result.metrics or {}
    return RunOutcome(
        index=index,
        n_markets=len(markets),
        total_return=float(metrics.get("total_return", 0.0)),
        max_drawdown=float(metrics.get("max_drawdown", 0.0)),
        n_trades=int(metrics.get("n_trades", 0)),
        brier=metrics.get("brier_score"),
        curve=normalise_curve(
            [s.equity for s in result.snapshots], result.initial_capital, grid_points
        ),
    )


def run_campaign(
    dataset: MarketDataset,
    settings: dict[str, Any],
    config: CampaignConfig | None = None,
    on_progress: Callable[[CampaignProgress], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> CampaignResult:
    """Run backtests over resampled markets until the time budget is spent.

    Args:
        dataset: The corpus to resample from.
        settings: Full settings dictionary, unchanged between runs - the point
            is to vary the *markets*, not the strategy.
        config: Duration and sampling parameters.
        on_progress: Called with a throttled :class:`CampaignProgress`.
        should_stop: Checked between runs; a stopped campaign keeps every run
            it already completed.

    Returns:
        A :class:`CampaignResult`. Read its ``verdict()`` before its median.
    """
    cfg = config or CampaignConfig()
    rng = random.Random(cfg.seed)
    throttle = ProgressThrottle()
    started = time.monotonic()

    result = CampaignResult(
        corpus_markets=len(dataset.markets),
        is_synthetic=dataset.is_demo,
    )
    if dataset.is_demo:
        result.notes.append(
            "Corpus is FABRICATED demo data. The spread below describes the "
            "generator, not any market."
        )
    if len(dataset.markets) < cfg.min_markets:
        result.notes.append("Too few markets in the corpus to resample meaningfully.")
        return result

    curves: list[tuple[float, ...]] = []
    while True:
        elapsed = time.monotonic() - started
        if elapsed >= cfg.duration_seconds or len(result.outcomes) >= cfg.max_runs:
            break
        if should_stop is not None and should_stop():
            result.cancelled = True
            break

        ids = sample_markets(dataset, cfg.sample_fraction, rng, cfg.min_markets)
        outcome = run_one(dataset, settings, ids, len(result.outcomes), cfg.grid_points)
        result.outcomes.append(outcome)
        curves.append(outcome.curve)
        result.band = build_band(curves)

        if on_progress is not None and throttle.should_emit():
            on_progress(
                CampaignProgress(
                    runs=len(result.outcomes),
                    elapsed=time.monotonic() - started,
                    duration=cfg.duration_seconds,
                    returns=tuple(result.returns),
                    band=result.band,
                    latest=outcome,
                )
            )

    result.elapsed = time.monotonic() - started
    if on_progress is not None:
        on_progress(
            CampaignProgress(
                runs=len(result.outcomes),
                elapsed=result.elapsed,
                duration=cfg.duration_seconds,
                returns=tuple(result.returns),
                band=result.band,
                latest=result.outcomes[-1] if result.outcomes else None,
            )
        )
    return result
