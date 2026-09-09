"""The self-improving loop: acquire real markets, retrain, score, report.

What "self-learning" means here, and what it does not
-----------------------------------------------------
The system improves by **accumulating evidence**, not by tuning itself toward a
number it likes. Each pass:

1. acquires more real resolved markets from a read-only venue;
2. rebuilds causal features and settlement labels over the grown corpus;
3. retrains every candidate under the walk-forward protocol;
4. scores each one **out-of-sample against a frozen baseline**;
5. records the trial and reports the promotion gate verdict.

Earnings are feedback - but only earnings measured on markets the candidate was
never fitted on. A loop that searched parameters and kept whatever maximised
in-sample profit would find something every time, and that something would be
noise. This module therefore never varies model hyper-parameters in response to
a result, and never approves anything.

The multiple-comparisons problem, made visible
----------------------------------------------
Automation creates a hazard that manual work does not: run enough candidates
against the same test data and one of them looks good by luck. Twenty
independent candidates at a 5% threshold produce a false winner about
two-thirds of the time.

:class:`TrialLedger` counts every evaluation against the corpus and reports the
bar a result would have to clear *given how many things were tried*. It does not
silently adjust anything - it makes the cost of searching visible, so a
promising twentieth trial is read as the twentieth trial rather than as a
discovery.

**Simulation-only.** Acquisition is read-only GET; scoring runs the existing
event-driven simulator into an in-memory portfolio; promotion still requires a
named human and a written rationale. There is no code path from this loop to an
order.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from adapters.base import AdapterError
from core.dataset import MarketDataset
from core.store import read_dataset, write_dataset
from research.corpus import describe
from research.pipeline import prepare_data, run_model_backtest, train_and_evaluate
from research.registry import ResearchRegistry
from research.run_loop import StopReason, merge_corpora

AUTOPILOT_VERSION = "autopilot-v1"

DEFAULT_CANDIDATES: tuple[str, ...] = ("logistic_regression",)
"""Models evaluated each pass.

Deliberately short. Every extra candidate is another draw against the same test
data, and the ledger charges for it.
"""


@dataclass(frozen=True)
class AutopilotConfig:
    """How long to run, what to acquire, and what would count as done."""

    duration_seconds: float = 900.0
    venue: str = "manifold"
    markets_per_batch: int = 60
    max_batches: int = 40

    target_effective_n: int = 50
    """The gate's own bar for independent out-of-sample outcomes."""

    target_folds: int = 3
    candidates: tuple[str, ...] = DEFAULT_CANDIDATES

    cache_path: str = "data/processed/manifold_corpus.duckdb"
    max_bet_pages: int = 1

    # Long-lived markets need a coarse decision grid. A market that lives 200
    # days would otherwise produce ~10,000 rows on a 30-minute stride, none of
    # which carry independent information.
    stride_seconds: float = 86_400.0
    horizon_seconds: float = 86_400.0
    short_window_seconds: float = 172_800.0
    long_window_seconds: float = 604_800.0


@dataclass(frozen=True)
class Trial:
    """One candidate, evaluated once, against one state of the corpus."""

    index: int
    model_name: str
    corpus_markets: int
    corpus_effective_n: int
    effective_n: int
    fold_count: int
    brier: float | None = None
    baseline_brier: float | None = None
    beats_baseline: bool | None = None
    oos_return: float | None = None
    """Out-of-sample simulated return - the "earnings" feedback, measured only
    on fold test windows the candidate was never fitted on."""

    n_trades: int = 0
    gate_passed: bool = False
    gate_failures: tuple[str, ...] = ()
    run_id: str = ""

    def as_row(self) -> dict[str, Any]:
        """One display row per trial."""
        return {
            "trial": self.index,
            "model": self.model_name,
            "corpus": self.corpus_markets,
            "resolved": self.corpus_effective_n,
            "oos_markets": self.effective_n,
            "folds": self.fold_count,
            "brier": None if self.brier is None else round(self.brier, 4),
            "baseline": None if self.baseline_brier is None else round(self.baseline_brier, 4),
            "oos_return": None if self.oos_return is None else round(self.oos_return, 4),
            "gate": "PASSED" if self.gate_passed else "failed",
        }


@dataclass
class TrialLedger:
    """Counts what has been tried, so a lucky result is read as one.

    This is the guard automation needs and manual work does not. Nothing here
    changes a threshold; it reports what the threshold would have to be.
    """

    trials: int = 0
    beat_baseline: int = 0

    def record(self, trial: Trial) -> None:
        """Add one evaluation to the count."""
        self.trials += 1
        if trial.beats_baseline:
            self.beat_baseline += 1

    @property
    def family_wise_bar(self) -> float:
        """The per-trial significance a 5% family-wise bar implies.

        Bonferroni: with ``k`` draws against the same data, each needs to clear
        ``0.05 / k`` for the *set* to hold at 5%.
        """
        return 0.05 / max(1, self.trials)

    @property
    def expected_false_winners(self) -> float:
        """How many candidates would look good by luck alone at a 5% bar."""
        return 0.05 * self.trials

    def warning(self) -> str:
        """A sentence to print next to any promising result."""
        if self.trials <= 1:
            return ""
        return (
            f"{self.trials} candidate evaluations have been run against this corpus. "
            f"At a naive 5% bar, roughly {self.expected_false_winners:.1f} would look "
            f"good by chance alone; holding the whole set to 5% requires each to clear "
            f"{self.family_wise_bar:.4f}. Read a promising result as the "
            f"{self.trials}{_ordinal(self.trials)} thing tried, not as a discovery."
        )


def _ordinal(n: int) -> str:
    """Ordinal suffix for a count."""
    if 11 <= (n % 100) <= 13:
        return "th"
    return {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")


@dataclass
class AutopilotReport:
    """Everything the loop did, and why it stopped."""

    trials: list[Trial] = field(default_factory=list)
    ledger: TrialLedger = field(default_factory=TrialLedger)
    corpus_markets: int = 0
    corpus_effective_n: int = 0
    batches: int = 0
    elapsed: float = 0.0
    stopped_because: StopReason = StopReason.MAX_ITERATIONS
    errors: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def gate_passed(self) -> bool:
        """Whether any trial cleared the promotion gate."""
        return any(t.gate_passed for t in self.trials)

    @property
    def best(self) -> Trial | None:
        """The trial with the lowest out-of-sample Brier, if any scored."""
        scored = [t for t in self.trials if t.brier is not None]
        return min(scored, key=lambda t: t.brier) if scored else None

    def as_rows(self) -> list[dict[str, Any]]:
        """Every trial as a display row."""
        return [t.as_row() for t in self.trials]

    def summary(self) -> str:
        """The honest read, including what searching cost."""
        if not self.trials:
            return (
                f"No candidate was evaluated ({self.stopped_because.value}). "
                f"Corpus holds {self.corpus_markets} market(s), "
                f"{self.corpus_effective_n} resolved."
            )
        best = self.best
        lines = [
            f"Stopped: {self.stopped_because.value} after {self.batches} batch(es) "
            f"and {len(self.trials)} trial(s) in {self.elapsed:.0f}s.",
            f"Corpus: {self.corpus_markets} market(s), {self.corpus_effective_n} resolved.",
        ]
        if best is not None:
            # The baseline can be unscored when no fold produced rows, so it is
            # formatted defensively rather than assumed.
            baseline = (
                "n/a" if best.baseline_brier is None else f"{best.baseline_brier:.4f}"
            )
            lines.append(
                f"Best out-of-sample Brier {best.brier:.4f} (baseline {baseline}) "
                f"across {best.fold_count} fold(s) and {best.effective_n} resolved "
                "market(s)."
            )
        if self.gate_passed:
            lines.append(
                "PROMOTION GATE PASSED - this is a candidate for human review, NOT an "
                "approval. Approving still requires a named approver and a written "
                "rationale."
            )
        else:
            failures = self.trials[-1].gate_failures
            lines.append("Gate did NOT pass. Failing checks on the last trial:")
            lines.extend(f"  - {f}" for f in failures)
        warning = self.ledger.warning()
        if warning:
            lines.append("")
            lines.append(warning)
        return "\n".join(lines)


# --------------------------------------------------------------------------
# Acquisition
# --------------------------------------------------------------------------


def build_adapter(venue: str, settings: dict[str, Any]) -> Any:
    """Construct the read-only adapter for a venue."""
    venues = settings.get("venues", {})
    if venue == "manifold":
        from adapters.manifold import ManifoldAdapter

        return ManifoldAdapter(venues.get("manifold", {}))
    if venue == "kalshi":
        from adapters.kalshi import KalshiAdapter

        return KalshiAdapter(venues.get("kalshi", {}))
    raise ValueError(f"unknown venue: {venue!r}")


def acquire_batch(
    adapter: Any,
    venue: str,
    seen: set[str],
    config: AutopilotConfig,
    offset: int,
    errors: list[str] | None = None,
) -> MarketDataset | None:
    """Fetch one batch of resolved markets not already in the corpus."""
    if venue != "manifold":
        raise ValueError("only the manifold venue is wired into autopilot today")

    from adapters.manifold import ManifoldFetchSpec

    try:
        listed = adapter.discover(limit=100, offset=offset)
    except (AdapterError, OSError) as error:
        if errors is not None:
            errors.append(f"discover: {type(error).__name__}: {error}")
        return None

    fresh = [m["id"] for m in listed if m.get("id") and m["id"] not in seen]
    if not fresh:
        return None

    try:
        return adapter.fetch_dataset(
            ManifoldFetchSpec(
                market_ids=fresh[: config.markets_per_batch],
                max_bet_pages=config.max_bet_pages,
            )
        )
    except (AdapterError, OSError) as error:
        if errors is not None:
            errors.append(f"fetch: {type(error).__name__}: {error}")
        return None


def research_settings(settings: dict[str, Any], config: AutopilotConfig) -> dict[str, Any]:
    """Settings with a decision grid sized for long-lived markets.

    Not a tuning knob: a market that lives 200 days sampled every 30 minutes
    yields ~10,000 rows carrying one resolved outcome between them. The grid is
    matched to market lifetime, never to a result.
    """
    merged = dict(settings)
    merged["research"] = {
        **settings.get("research", {}),
        "dataset": {
            **settings.get("research", {}).get("dataset", {}),
            "stride_seconds": config.stride_seconds,
            "horizon_seconds": config.horizon_seconds,
            "short_window_seconds": config.short_window_seconds,
            "long_window_seconds": config.long_window_seconds,
        },
    }
    return merged


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------


def evaluate_candidate(
    corpus: MarketDataset,
    settings: dict[str, Any],
    model_name: str,
    index: int,
    registry: ResearchRegistry | None = None,
) -> Trial:
    """Train one candidate and score it out-of-sample. Never approves."""
    prepared = prepare_data(corpus, settings, registry)
    outcome = train_and_evaluate(prepared, settings, model_name, registry)
    gate = outcome.gate

    oos_return: float | None = None
    n_trades = 0
    if outcome.report.usable_folds:
        backtest = run_model_backtest(prepared, outcome, corpus, settings, registry)
        metrics = backtest.metrics or {}
        oos_return = metrics.get("total_return")
        n_trades = int(metrics.get("n_trades", 0) or 0)

    return Trial(
        index=index,
        model_name=model_name,
        corpus_markets=len(corpus.markets),
        corpus_effective_n=prepared.dataset.effective_n,
        effective_n=outcome.candidate.effective_n,
        fold_count=len(outcome.report.usable_folds),
        brier=outcome.candidate.brier,
        baseline_brier=outcome.baseline.brier if outcome.baseline else None,
        beats_baseline=outcome.comparison.beats_baseline(),
        oos_return=oos_return,
        n_trades=n_trades,
        gate_passed=gate.passed,
        gate_failures=tuple(
            f"{name}: {detail}" for name, ok, detail in gate.checks if not ok
        ),
        run_id=outcome.run_id,
    )


# --------------------------------------------------------------------------
# The loop
# --------------------------------------------------------------------------


def run_autopilot(
    settings: dict[str, Any],
    config: AutopilotConfig | None = None,
    adapter: Any = None,
    registry: ResearchRegistry | None = None,
    on_trial: Callable[[Trial], None] | None = None,
    on_status: Callable[[str], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> AutopilotReport:
    """Grow a real corpus and re-evaluate until the gate passes or time runs out.

    Args:
        settings: Full settings dictionary.
        config: Duration, venue and targets.
        adapter: Read-only venue adapter; built from settings when omitted.
        registry: Where every run is recorded.
        on_trial: Called as each candidate finishes.
        on_status: Called with short progress lines.
        should_stop: Cancellation predicate, checked between phases.

    Returns:
        An :class:`AutopilotReport`. A passing gate yields a recommendation for
        a human; nothing is ever approved here.
    """
    cfg = config or AutopilotConfig()
    started = time.monotonic()
    report = AutopilotReport()
    tuned = research_settings(settings, cfg)
    venue_adapter = adapter or build_adapter(cfg.venue, settings)

    def say(message: str) -> None:
        if on_status is not None:
            on_status(message)

    corpus: MarketDataset | None = None
    resolved = read_dataset(cfg.cache_path) if _exists(cfg.cache_path) else None
    if resolved is not None:
        corpus = resolved
        say(f"Loaded {len(corpus.markets)} cached market(s).")
    seen: set[str] = {m.market_id for m in corpus.markets} if corpus else set()

    offset = 0
    for batch in range(1, cfg.max_batches + 1):
        if should_stop is not None and should_stop():
            report.stopped_because = StopReason.CANCELLED
            break
        if time.monotonic() - started >= cfg.duration_seconds:
            report.stopped_because = StopReason.MAX_ITERATIONS
            break

        say(f"Batch {batch}: acquiring resolved markets from {cfg.venue}...")
        fetched = acquire_batch(
            venue_adapter, cfg.venue, seen, cfg, offset, report.errors
        )
        offset += 100
        if fetched is None:
            if corpus is None:
                report.stopped_because = StopReason.NO_DATA
                break
            if offset > 100 * cfg.max_batches:
                report.stopped_because = StopReason.NO_NEW_MARKETS
                break
            continue

        corpus = merge_corpora(corpus, fetched)
        seen.update(m.market_id for m in corpus.markets)
        write_dataset(corpus, cfg.cache_path)
        report.batches = batch
        report.corpus_markets = len(corpus.markets)
        say(f"Corpus now {len(corpus.markets)} market(s). Training...")

        if describe(corpus).is_synthetic:
            report.notes.append(
                "Corpus reports as synthetic - the gate cannot pass. Check provenance."
            )

        for model_name in cfg.candidates:
            if should_stop is not None and should_stop():
                report.stopped_because = StopReason.CANCELLED
                break
            # The budget is checked here as well as at the top of the batch.
            # One evaluation is not interruptible once started - feature
            # building and walk-forward run to completion - so a large corpus
            # can still overrun. Measured: a 179-market corpus took hours on a
            # 14-minute budget. The overrun is recorded rather than hidden.
            spent = time.monotonic() - started
            if spent >= cfg.duration_seconds:
                report.stopped_because = StopReason.MAX_ITERATIONS
                break
            trial = evaluate_candidate(
                corpus, tuned, model_name, len(report.trials) + 1, registry
            )
            report.trials.append(trial)
            report.ledger.record(trial)
            report.corpus_effective_n = trial.corpus_effective_n
            if on_trial is not None:
                on_trial(trial)

            if trial.gate_passed:
                report.stopped_because = StopReason.GATE_PASSED
                break

        if report.stopped_because in (StopReason.GATE_PASSED, StopReason.CANCELLED):
            break

    report.elapsed = time.monotonic() - started
    if corpus is not None:
        report.corpus_markets = len(corpus.markets)
    if report.elapsed > cfg.duration_seconds * 1.5:
        report.notes.append(
            f"Ran {report.elapsed:.0f}s against a {cfg.duration_seconds:.0f}s budget. "
            "A candidate evaluation cannot be interrupted once started, and the cost "
            "of rebuilding features grows with the corpus. Use a smaller batch, or "
            "expect the last evaluation to overrun."
        )
    return report


def _exists(path: str) -> bool:
    """Whether a cached corpus is already on disk."""
    from core.config import resolve_path

    return resolve_path(path).exists()


def target_progress(report: AutopilotReport, config: AutopilotConfig) -> list[dict[str, Any]]:
    """How far the run is from the conditions the gate requires.

    This is the "outputs fulfilled" checklist: concrete, measurable, and read
    from the same numbers the gate reads.
    """
    best = report.best
    return [
        {
            "target": "Real market data",
            "have": "yes" if report.corpus_markets else "no corpus",
            "need": "a non-synthetic corpus",
            "met": report.corpus_markets > 0,
        },
        {
            "target": "Usable walk-forward folds",
            "have": best.fold_count if best else 0,
            "need": config.target_folds,
            "met": bool(best and best.fold_count >= config.target_folds),
        },
        {
            "target": "Independent out-of-sample markets",
            "have": best.effective_n if best else 0,
            "need": config.target_effective_n,
            "met": bool(best and best.effective_n >= config.target_effective_n),
        },
        {
            "target": "Beats frozen baseline",
            "have": "yes" if best and best.beats_baseline else "no",
            "need": "yes",
            "met": bool(best and best.beats_baseline),
        },
    ]


def datetime_stamp() -> str:
    """UTC stamp for run labelling."""
    return datetime.now(UTC).isoformat(timespec="seconds")


def bonferroni_bar(trials: int, family_alpha: float = 0.05) -> float:
    """The per-trial bar implied by a family-wise error rate."""
    return family_alpha / max(1, trials)


def probability_of_a_false_winner(trials: int, alpha: float = 0.05) -> float:
    """Chance at least one of ``trials`` independent draws looks good by luck."""
    return 1.0 - math.pow(1.0 - alpha, max(0, trials))


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Run the loop from the command line.

    Usage::

        python -m research.autopilot --minutes 15 --venue manifold
    """
    import argparse

    from core.config import get_settings

    parser = argparse.ArgumentParser(
        description="Grow a real corpus and re-evaluate until the gate passes."
    )
    parser.add_argument("--minutes", type=float, default=15.0)
    parser.add_argument("--venue", default="manifold", choices=("manifold", "kalshi"))
    parser.add_argument("--batch", type=int, default=AutopilotConfig.markets_per_batch)
    parser.add_argument("--max-batches", type=int, default=AutopilotConfig.max_batches)
    parser.add_argument("--cache", default=AutopilotConfig.cache_path)
    args = parser.parse_args(argv)

    settings = get_settings()
    config = AutopilotConfig(
        duration_seconds=args.minutes * 60.0,
        venue=args.venue,
        markets_per_batch=args.batch,
        max_batches=args.max_batches,
        cache_path=args.cache,
    )

    def status(message: str) -> None:
        print(f"  {message}", flush=True)

    def trial(record: Trial) -> None:
        print(
            f"[trial {record.index}] corpus {record.corpus_markets} "
            f"({record.corpus_effective_n} resolved) | OOS markets "
            f"{record.effective_n} across {record.fold_count} fold(s) | brier "
            f"{'n/a' if record.brier is None else format(record.brier, '.4f')} "
            f"vs baseline "
            f"{'n/a' if record.baseline_brier is None else format(record.baseline_brier, '.4f')}"
            f" | gate {'PASSED' if record.gate_passed else 'failed'}",
            flush=True,
        )
        for failure in record.gate_failures:
            print(f"      - {failure}", flush=True)

    registry = ResearchRegistry(
        settings.get("research", {}).get("registry_path", "data/research/registry.duckdb")
    )
    report = run_autopilot(
        settings, config, registry=registry, on_trial=trial, on_status=status
    )

    print()
    print("TARGETS")
    for row in target_progress(report, config):
        mark = "OK " if row["met"] else "-- "
        print(f"  {mark} {row['target']:<38} have={row['have']!s:<10} need={row['need']}")
    print()
    print(report.summary())
    return 0 if report.gate_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
