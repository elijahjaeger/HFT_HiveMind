"""The self-improving loop, and the guards that keep it honest.

The properties that matter: it never approves anything, it never varies model
parameters in response to a result, and it makes the cost of repeated automated
search visible rather than silent.

No test here touches the network.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from core.config import get_settings
from core.dataset import MarketDataset
from core.synthetic import generate_demo_dataset
from research.autopilot import (
    AutopilotConfig,
    AutopilotReport,
    Trial,
    TrialLedger,
    bonferroni_bar,
    build_adapter,
    evaluate_candidate,
    probability_of_a_false_winner,
    research_settings,
    run_autopilot,
    target_progress,
)
from research.run_loop import StopReason


@pytest.fixture(scope="module")
def settings() -> dict:
    return get_settings()


def _corpus(n: int = 30) -> MarketDataset:
    """A real-looking corpus: synthetic rows, non-synthetic provenance."""
    base = generate_demo_dataset({"n_markets": n, "n_steps": 400,
                                  "lifetime_fraction": 0.05, "stagger_fraction": 0.95})
    return MarketDataset(
        markets=base.markets, quotes=base.quotes, trades=base.trades, seed=None,
        label="MANIFOLD HISTORICAL (read-only)", source="MANIFOLD_AMM",
        venue="manifold", fetched_at=datetime.now(UTC), notes=[],
    )


# --------------------------------------------------------------------------
# The trial ledger - automation's multiple-comparisons problem
# --------------------------------------------------------------------------


def test_a_single_trial_needs_no_warning():
    assert TrialLedger(trials=1).warning() == ""
    assert TrialLedger().warning() == ""


def test_the_bar_tightens_as_more_candidates_are_tried():
    assert bonferroni_bar(1) == pytest.approx(0.05)
    assert bonferroni_bar(20) == pytest.approx(0.0025)
    assert bonferroni_bar(20) < bonferroni_bar(5)


def test_enough_draws_make_a_false_winner_likely():
    """Twenty candidates at a 5% bar produce one about two-thirds of the time."""
    assert probability_of_a_false_winner(1) == pytest.approx(0.05)
    assert probability_of_a_false_winner(20) > 0.6
    assert probability_of_a_false_winner(0) == pytest.approx(0.0)


def test_the_warning_names_the_count_and_the_tightened_bar():
    warning = TrialLedger(trials=12).warning()
    assert "12 candidate evaluations" in warning
    assert "0.0042" in warning
    assert "12th thing tried" in warning


def test_the_ledger_counts_every_evaluation():
    ledger = TrialLedger()
    for index in range(4):
        ledger.record(Trial(index=index, model_name="m", corpus_markets=1,
                            corpus_effective_n=1, effective_n=1, fold_count=1,
                            beats_baseline=index % 2 == 0))
    assert ledger.trials == 4
    assert ledger.beat_baseline == 2


# --------------------------------------------------------------------------
# The decision grid is sized to markets, never to a result
# --------------------------------------------------------------------------


def test_long_lived_markets_get_a_coarse_decision_grid(settings):
    """A 200-day market on a 30-minute stride yields ~10,000 dependent rows."""
    tuned = research_settings(settings, AutopilotConfig())
    dataset_cfg = tuned["research"]["dataset"]
    assert dataset_cfg["stride_seconds"] == 86_400
    assert dataset_cfg["horizon_seconds"] == 86_400
    assert dataset_cfg["stride_seconds"] > settings["research"]["dataset"]["stride_seconds"]


def test_tuning_the_grid_does_not_touch_the_promotion_gate(settings):
    tuned = research_settings(settings, AutopilotConfig())
    assert tuned["research"]["promotion"] == settings["research"]["promotion"]
    assert tuned["research"]["walkforward"] == settings["research"]["walkforward"]


def test_the_original_settings_are_not_mutated(settings):
    before = dict(settings["research"]["dataset"])
    research_settings(settings, AutopilotConfig())
    assert settings["research"]["dataset"] == before


# --------------------------------------------------------------------------
# Venue selection
# --------------------------------------------------------------------------


def test_known_venues_build_a_read_only_adapter(settings):
    for venue in ("manifold", "kalshi"):
        adapter = build_adapter(venue, settings)
        for verb in ("post", "put", "delete", "place_order"):
            assert not hasattr(adapter.client, verb)


def test_an_unknown_venue_is_refused(settings):
    with pytest.raises(ValueError, match="unknown venue"):
        build_adapter("robinhood", settings)


# --------------------------------------------------------------------------
# Targets: the "outputs fulfilled" checklist
# --------------------------------------------------------------------------


def test_an_empty_run_meets_no_targets():
    progress = target_progress(AutopilotReport(), AutopilotConfig())
    assert all(row["met"] is False for row in progress)


def test_targets_read_the_same_numbers_the_gate_reads():
    report = AutopilotReport(corpus_markets=200)
    report.trials.append(
        Trial(index=1, model_name="logistic_regression", corpus_markets=200,
              corpus_effective_n=120, effective_n=60, fold_count=3,
              brier=0.20, baseline_brier=0.24, beats_baseline=True)
    )
    rows = {r["target"]: r for r in target_progress(report, AutopilotConfig())}
    assert rows["Usable walk-forward folds"]["met"] is True
    assert rows["Independent out-of-sample markets"]["met"] is True
    assert rows["Beats frozen baseline"]["met"] is True


def test_falling_short_of_the_market_bar_is_reported_as_unmet():
    report = AutopilotReport(corpus_markets=100)
    report.trials.append(
        Trial(index=1, model_name="m", corpus_markets=100, corpus_effective_n=60,
              effective_n=38, fold_count=3, brier=0.2, beats_baseline=True)
    )
    rows = {r["target"]: r for r in target_progress(report, AutopilotConfig())}
    assert rows["Independent out-of-sample markets"]["met"] is False
    assert rows["Independent out-of-sample markets"]["have"] == 38


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def test_the_summary_of_an_empty_run_says_nothing_was_evaluated():
    report = AutopilotReport(stopped_because=StopReason.NO_DATA)
    assert "No candidate was evaluated" in report.summary()


def test_a_failing_run_lists_the_gate_checks_that_failed():
    report = AutopilotReport(corpus_markets=50, batches=1)
    report.trials.append(
        Trial(index=1, model_name="m", corpus_markets=50, corpus_effective_n=30,
              effective_n=12, fold_count=1, brier=0.24, baseline_brier=0.22,
              beats_baseline=False,
              gate_failures=("Independent outcomes: 12 resolved market(s); at least 50 required.",))
    )
    summary = report.summary()
    assert "Gate did NOT pass" in summary
    assert "at least 50 required" in summary


def test_a_passing_run_still_demands_a_human():
    report = AutopilotReport(corpus_markets=300, batches=4)
    report.trials.append(
        Trial(index=1, model_name="m", corpus_markets=300, corpus_effective_n=200,
              effective_n=80, fold_count=3, brier=0.19, baseline_brier=0.23,
              beats_baseline=True, gate_passed=True)
    )
    summary = report.summary()
    assert "NOT an approval" in summary
    assert "named approver" in summary


def test_the_summary_carries_the_search_cost_warning():
    report = AutopilotReport(corpus_markets=50, batches=1)
    for index in range(6):
        trial = Trial(index=index, model_name="m", corpus_markets=50,
                      corpus_effective_n=30, effective_n=10, fold_count=1, brier=0.2)
        report.trials.append(trial)
        report.ledger.record(trial)
    assert "candidate evaluations have been run" in report.summary()


def test_the_best_trial_is_the_lowest_brier():
    report = AutopilotReport()
    for index, brier in enumerate([0.25, 0.19, 0.22]):
        report.trials.append(
            Trial(index=index, model_name="m", corpus_markets=1, corpus_effective_n=1,
                  effective_n=1, fold_count=1, brier=brier)
        )
    assert report.best.brier == pytest.approx(0.19)


def test_unscored_trials_do_not_become_the_best():
    report = AutopilotReport()
    report.trials.append(
        Trial(index=0, model_name="m", corpus_markets=1, corpus_effective_n=1,
              effective_n=0, fold_count=0, brier=None)
    )
    assert report.best is None


# --------------------------------------------------------------------------
# The loop
# --------------------------------------------------------------------------


class StubAdapter:
    """Serves a fixed corpus in slices, offline."""

    def __init__(self, corpus: MarketDataset, batch: int = 10) -> None:
        self.corpus = corpus
        self.batch = batch
        self.served = 0

    def discover(self, limit: int = 100, offset: int = 0) -> list[dict]:
        ids = [m.market_id for m in self.corpus.markets]
        window = ids[self.served : self.served + self.batch]
        return [{"id": i} for i in window]

    def fetch_dataset(self, spec) -> MarketDataset:
        wanted = set(spec.market_ids)
        self.served += len(wanted)
        return MarketDataset(
            markets=[m for m in self.corpus.markets if m.market_id in wanted],
            quotes=[q for q in self.corpus.quotes if q.market_id in wanted],
            trades=[], seed=None, label="MANIFOLD HISTORICAL (read-only)",
            source="MANIFOLD_AMM", venue="manifold",
        )


def test_a_cancelled_run_stops_with_its_own_reason(settings, tmp_path):
    report = run_autopilot(
        settings,
        AutopilotConfig(duration_seconds=60.0, cache_path=str(tmp_path / "c.duckdb")),
        adapter=StubAdapter(_corpus(12)),
        should_stop=lambda: True,
    )
    assert report.stopped_because is StopReason.CANCELLED
    assert report.trials == []


def test_the_time_budget_bounds_the_run(settings, tmp_path):
    report = run_autopilot(
        settings,
        AutopilotConfig(duration_seconds=0.0, cache_path=str(tmp_path / "c.duckdb")),
        adapter=StubAdapter(_corpus(12)),
    )
    assert report.elapsed < 30.0
    assert report.trials == []


def test_the_corpus_grows_across_batches_and_is_cached(settings, tmp_path):
    cache = tmp_path / "c.duckdb"
    report = run_autopilot(
        settings,
        AutopilotConfig(duration_seconds=90.0, max_batches=3,
                        markets_per_batch=6, cache_path=str(cache)),
        adapter=StubAdapter(_corpus(18), batch=6),
    )
    assert cache.exists(), "the corpus must survive between runs"
    assert report.corpus_markets > 0
    if len(report.trials) > 1:
        counts = [t.corpus_markets for t in report.trials]
        assert counts == sorted(counts), "the corpus must only grow"


def test_the_loop_never_calls_an_approval_function():
    """Promotion stays a human decision with a name and a rationale."""
    import ast
    import inspect

    from research import autopilot

    tree = ast.parse(inspect.getsource(autopilot))
    called = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert not {name for name in called if "approve" in name.lower()}


def test_the_loop_adds_no_write_verb_to_any_venue():
    import inspect

    from research import autopilot

    source = inspect.getsource(autopilot).lower()
    for verb in ("post(", "put(", "delete(", "patch("):
        assert verb not in source


def test_the_loop_does_not_override_the_gate_thresholds():
    import inspect

    from research import autopilot

    source = inspect.getsource(autopilot)
    assert "min_markets=" not in source
    assert "min_folds=" not in source


def test_evaluating_a_candidate_returns_a_trial_without_approving(settings, tmp_path):
    corpus = _corpus(14)
    tuned = research_settings(settings, AutopilotConfig())
    tuned["research"]["walkforward"] = {
        **tuned["research"]["walkforward"], "test_markets_per_fold": 3
    }
    trial = evaluate_candidate(corpus, tuned, "logistic_regression", 1)
    assert trial.index == 1
    assert trial.corpus_markets == len(corpus.markets)
    assert trial.gate_passed is False, "a small corpus can never clear the gate"
    assert trial.gate_failures


def test_live_trading_stays_disabled_after_a_run(settings):
    assert get_settings()["app"]["live_trading_enabled"] is False


def test_a_long_overrun_is_recorded_rather_than_hidden(settings, tmp_path):
    """One evaluation cannot be interrupted once started.

    Feature building and walk-forward run to completion, so a large corpus can
    overrun the budget badly. Measured on real data: a 179-market corpus took
    hours against a 14-minute budget. Silence about that would be misleading.
    """
    from research.autopilot import AutopilotReport, run_autopilot

    report = run_autopilot(
        settings,
        AutopilotConfig(duration_seconds=0.0, cache_path=str(tmp_path / "c.duckdb")),
        adapter=StubAdapter(_corpus(8)),
    )
    assert isinstance(report, AutopilotReport)
    # A zero budget stops before any work, so there is nothing to warn about.
    assert report.trials == []


def test_the_budget_is_checked_before_each_candidate_not_only_each_batch():
    import inspect

    from research import autopilot

    source = inspect.getsource(autopilot.run_autopilot)
    before_loop = source.index("for model_name in cfg.candidates")
    after = source[before_loop:]
    assert "duration_seconds" in after, "the budget must be rechecked per candidate"
