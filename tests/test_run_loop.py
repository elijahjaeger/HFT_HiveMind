"""Tests for research/run_loop.py and the core/store.py provenance fix.

No test here touches the network: every request is served by the same fake
transport the adapter tests use.

The properties that matter most, and that a refactor would quietly break:
the loop never loosens the gate, the corpus only grows, and a fetched corpus
survives caching still looking fetched.
"""

from __future__ import annotations

import copy
from datetime import UTC, datetime, timedelta

import pytest

from adapters.base import AdapterError
from core.config import get_settings
from core.dataset import MarketDataset
from core.store import read_dataset, write_dataset
from core.synthetic import generate_demo_dataset
from research.corpus import build_research_corpus, describe
from research.registry import evaluate_promotion_gate
from research.run_loop import (
    DiscoveredMarket,
    LoopConfig,
    LoopReport,
    StopReason,
    cache_corpus,
    discover_settled_markets,
    discover_settled_tickers,
    fetch_corpus,
    fetch_corpus_chunked,
    gate_failures,
    historical_cutoff,
    is_settled_binary,
    load_cached_corpus,
    market_lifetime_hours,
    merge_corpora,
    recommendation_for,
    run_loop,
    run_stages,
    summarise_iteration,
    window_for,
)
from tests.fixtures import kalshi_payloads as fx
from tests.test_kalshi_adapter import FakeTransport, make_adapter

CUTOFF = datetime(2026, 7, 9, tzinfo=UTC)


# ==========================================================================
# core/store.py - the two defects that made real corpora uncacheable
# ==========================================================================


def _real_corpus() -> MarketDataset:
    base = generate_demo_dataset({"n_markets": 4, "n_steps": 40})
    return MarketDataset(
        markets=base.markets,
        quotes=base.quotes,
        trades=base.trades,
        seed=None,
        label="KALSHI HISTORICAL",
        source="KALSHI_CANDLES",
        venue="kalshi",
        fetched_at=datetime(2026, 7, 1, tzinfo=UTC),
        notes=["depth proxy"],
    )


def test_a_fetched_corpus_has_no_seed_and_still_round_trips(tmp_path):
    """Regression: int(NULL) used to raise and make real data uncacheable."""
    path = tmp_path / "corpus.duckdb"
    write_dataset(_real_corpus(), path)
    assert read_dataset(path).seed is None


def test_caching_a_fetched_corpus_does_not_make_it_look_fabricated(tmp_path):
    """Regression: provenance loss failed the gate's "Real market data" check."""
    path = tmp_path / "corpus.duckdb"
    write_dataset(_real_corpus(), path)
    back = read_dataset(path)

    assert back.label == "KALSHI HISTORICAL"
    assert back.source == "KALSHI_CANDLES"
    assert back.venue == "kalshi"
    assert back.notes == ["depth proxy"]
    assert back.is_demo is False
    assert describe(back).is_synthetic is False


def test_a_generated_corpus_still_round_trips_as_synthetic(tmp_path):
    path = tmp_path / "demo.duckdb"
    demo = generate_demo_dataset({"n_markets": 4, "n_steps": 40})
    write_dataset(demo, path)
    back = read_dataset(path)

    assert back.seed == demo.seed
    assert back.is_demo is True
    assert describe(back).is_synthetic is True


# ==========================================================================
# Windows and cutoff
# ==========================================================================


def test_the_historical_cutoff_is_read_from_the_venue(tmp_path):
    adapter, _ = make_adapter(tmp_path, FakeTransport([("cutoff", fx.CUTOFF)]))
    assert historical_cutoff(adapter) == datetime(2026, 6, 1, tzinfo=UTC)


def test_an_unreadable_cutoff_falls_back_instead_of_raising(tmp_path):
    def boom(url: str):
        raise OSError("no network")

    adapter, _ = make_adapter(tmp_path, FakeTransport([("", boom)]))
    assert historical_cutoff(adapter) < datetime.now(UTC)


def test_each_iteration_reaches_further_back_from_a_pinned_end():
    config = LoopConfig(window_days_initial=30, window_days_increment=30)
    first = window_for(CUTOFF, 1, config)
    second = window_for(CUTOFF, 2, config)

    assert first[1] == second[1] == CUTOFF, "the end must stay at the cutoff"
    assert second[0] < first[0], "the start must reach further back"
    assert first[0] == CUTOFF - timedelta(days=30)
    assert second[0] == CUTOFF - timedelta(days=60)


# ==========================================================================
# Discovery
# ==========================================================================


def test_only_settled_binary_markets_with_a_yes_no_result_are_eligible():
    assert is_settled_binary(fx.SETTLED_MARKET) is True


@pytest.mark.parametrize(
    ("overrides", "why"),
    [
        ({"result": ""}, "unsettled markets have no label"),
        ({"result": "scalar"}, "scalar markets are not binary"),
        ({"market_type": "scalar"}, "non-binary market type"),
        ({"ticker": ""}, "cannot be fetched without a ticker"),
    ],
)
def test_markets_that_cannot_produce_a_label_are_excluded(overrides, why):
    payload = {**copy.deepcopy(fx.SETTLED_MARKET), **overrides}
    assert is_settled_binary(payload) is False, why


def test_discovery_returns_settled_tickers_it_has_not_seen(tmp_path):
    adapter, _ = make_adapter(tmp_path)
    found = discover_settled_tickers(adapter, CUTOFF - timedelta(days=30), CUTOFF, 10)
    assert "KXHIGHNY-25AUG31-B70" in found


def test_discovery_skips_tickers_already_in_the_corpus(tmp_path):
    adapter, _ = make_adapter(tmp_path)
    seen = {"KXHIGHNY-25AUG31-B70"}
    assert discover_settled_tickers(
        adapter, CUTOFF - timedelta(days=30), CUTOFF, 10, seen
    ) == []


def test_discovery_respects_its_limit(tmp_path):
    adapter, _ = make_adapter(tmp_path)
    found = discover_settled_tickers(adapter, CUTOFF - timedelta(days=30), CUTOFF, 1)
    assert len(found) <= 1


def test_a_failed_discovery_returns_nothing_rather_than_raising(tmp_path):
    def boom(url: str):
        raise OSError("no network")

    adapter, _ = make_adapter(tmp_path, FakeTransport([("", boom)]))
    assert discover_settled_tickers(adapter, CUTOFF - timedelta(days=30), CUTOFF, 5) == []


def test_a_failed_discovery_is_distinguishable_from_an_exhausted_archive(tmp_path):
    """Both return no tickers; only one is a finding about the data."""
    def boom(url: str):
        raise OSError("connection reset")

    adapter, _ = make_adapter(tmp_path, FakeTransport([("", boom)]))
    errors: list[str] = []
    found = discover_settled_tickers(
        adapter, CUTOFF - timedelta(days=30), CUTOFF, 5, None, None, errors
    )
    assert found == []
    assert errors and "connection reset" in errors[0]


def test_an_exhausted_archive_records_no_error(tmp_path):
    adapter, _ = make_adapter(tmp_path)
    errors: list[str] = []
    discover_settled_tickers(
        adapter, CUTOFF - timedelta(days=30), CUTOFF, 5,
        {"KXHIGHNY-25AUG31-B70"}, None, errors,
    )
    assert errors == [], "no error means the history really is exhausted"


def test_a_transient_discovery_failure_is_retried_not_reported_as_no_data(
    monkeypatch, small_corpus, small_settings, tmp_path
):
    """A venue blip must not be reported as a finding about the archive."""
    import research.run_loop as module

    state = {"calls": 0}

    def flaky(adapter, start, end, limit, seen=None, config=None, errors=None):
        state["calls"] += 1
        if state["calls"] == 1:
            if errors is not None:
                errors.append("OSError: connection reset")
            return []
        return [
            DiscoveredMarket(m.market_id, CUTOFF - timedelta(days=2), CUTOFF)
            for m in small_corpus.markets[:10]
        ]

    monkeypatch.setattr(module, "discover_settled_markets", flaky)
    monkeypatch.setattr(
        module, "fetch_corpus_chunked",
        lambda a, t, s2, e, c=None, errors=None, markets=None: MarketDataset(
            markets=[m for m in small_corpus.markets if m.market_id in set(t)],
            quotes=[q for q in small_corpus.quotes if q.market_id in set(t)],
            trades=[], seed=None, label="KALSHI HISTORICAL", source="KALSHI_CANDLES",
        ),
    )
    config = LoopConfig(cache_path=str(tmp_path / "c.duckdb"), max_iterations=3)
    result = run_loop(small_settings, config, adapter=StubAdapter([], small_corpus))

    assert result.stopped_because is not StopReason.NO_DATA
    assert state["calls"] >= 2, "the loop must retry after a transient failure"


def test_repeated_discovery_failures_stop_with_their_own_reason(
    monkeypatch, small_corpus, small_settings, tmp_path
):
    import research.run_loop as module

    def always_fails(adapter, start, end, limit, seen=None, config=None, errors=None):
        if errors is not None:
            errors.append("AdapterError: HTTP 503")
        return []

    monkeypatch.setattr(module, "discover_settled_markets", always_fails)
    config = LoopConfig(
        cache_path=str(tmp_path / "c.duckdb"), max_iterations=10, max_discovery_failures=2
    )
    result = run_loop(small_settings, config, adapter=StubAdapter([], small_corpus))

    assert result.stopped_because is StopReason.DISCOVERY_FAILED
    assert any("503" in e for e in result.errors)


# ==========================================================================
# Fetching
# ==========================================================================


def test_fetching_no_tickers_is_not_an_error(tmp_path):
    adapter, _ = make_adapter(tmp_path)
    assert fetch_corpus(adapter, [], CUTOFF - timedelta(days=1), CUTOFF) is None


def test_a_fetched_corpus_carries_real_provenance(tmp_path):
    adapter, _ = make_adapter(tmp_path)
    corpus = fetch_corpus(
        adapter, ["KXHIGHNY-25AUG31-B70"], CUTOFF - timedelta(days=30), CUTOFF
    )
    assert corpus is not None
    assert corpus.is_demo is False
    assert describe(corpus).is_synthetic is False


def test_a_batch_that_cannot_be_fetched_returns_none_rather_than_raising(tmp_path):
    class Failing:
        def fetch_dataset(self, spec):
            raise AdapterError("venue unavailable")

    assert fetch_corpus(Failing(), ["X"], CUTOFF - timedelta(days=1), CUTOFF) is None


def test_trades_are_off_by_default_to_stay_inside_the_throttle():
    assert LoopConfig().include_trades is False


# ==========================================================================
# Merging - the corpus must only ever grow
# ==========================================================================


def _slice(dataset: MarketDataset, ids: set[str]) -> MarketDataset:
    return MarketDataset(
        markets=[m for m in dataset.markets if m.market_id in ids],
        quotes=[q for q in dataset.quotes if q.market_id in ids],
        trades=[t for t in dataset.trades if t.market_id in ids],
        seed=None,
        label="KALSHI HISTORICAL",
        source="KALSHI_CANDLES",
        venue="kalshi",
    )


def test_merging_unions_two_corpora():
    base = generate_demo_dataset({"n_markets": 4, "n_steps": 40})
    first = _slice(base, {"DEMO-01", "DEMO-02"})
    second = _slice(base, {"DEMO-03", "DEMO-04"})

    merged = merge_corpora(first, second)
    assert len(merged.markets) == 4
    assert len(merged.quotes) == len(first.quotes) + len(second.quotes)


def test_merging_the_same_market_twice_does_not_duplicate_it():
    base = generate_demo_dataset({"n_markets": 4, "n_steps": 40})
    first = _slice(base, {"DEMO-01", "DEMO-02"})

    merged = merge_corpora(first, _slice(base, {"DEMO-02", "DEMO-03"}))
    assert len(merged.markets) == 3
    assert len(merged.market_ids) == len(set(merged.market_ids))
    assert len(merged.quotes) == len({(q.market_id, q.ts) for q in merged.quotes})


def test_merging_with_nothing_returns_the_other_side():
    base = _slice(generate_demo_dataset({"n_markets": 4, "n_steps": 40}), {"DEMO-01"})
    assert merge_corpora(None, base) is base
    assert merge_corpora(base, None) is base
    assert merge_corpora(None, None) is None


def test_a_merged_corpus_keeps_real_provenance():
    base = generate_demo_dataset({"n_markets": 4, "n_steps": 40})
    merged = merge_corpora(_slice(base, {"DEMO-01"}), _slice(base, {"DEMO-02"}))
    assert describe(merged).is_synthetic is False


# ==========================================================================
# Caching
# ==========================================================================


def test_the_corpus_cache_round_trips_through_store(tmp_path):
    path = str(tmp_path / "corpus.duckdb")
    corpus = _real_corpus()
    cache_corpus(corpus, path)

    back = load_cached_corpus(path)
    assert back is not None
    assert len(back.markets) == len(corpus.markets)
    assert describe(back).is_synthetic is False


def test_a_missing_cache_is_not_an_error(tmp_path):
    assert load_cached_corpus(str(tmp_path / "absent.duckdb")) is None


def test_a_corrupt_cache_is_not_an_error(tmp_path):
    path = tmp_path / "corrupt.duckdb"
    path.write_bytes(b"not a duckdb file")
    assert load_cached_corpus(str(path)) is None


# ==========================================================================
# The gate - reported, never loosened
# ==========================================================================


def test_every_failing_gate_check_is_reported_in_the_registrys_own_words():
    gate = evaluate_promotion_gate(
        effective_n=38, beats_baseline=False, is_synthetic=True, fold_count=2
    )
    failures = gate_failures(gate)

    assert len(failures) == 4
    assert any("38 resolved market(s); at least 50 required." in f for f in failures)
    assert any("2 usable fold(s); at least 3 required." in f for f in failures)
    assert any("does not beat the market-price baseline" in f.lower() for f in failures)


def test_a_passing_gate_reports_no_failures():
    gate = evaluate_promotion_gate(
        effective_n=80, beats_baseline=True, is_synthetic=False, fold_count=3
    )
    assert gate.passed is True
    assert gate_failures(gate) == []


def _called_names(module) -> set[str]:
    """Every function/method name run_loop actually calls, via AST.

    A string search would trip over the word in a docstring, which is exactly
    where this module explains why it does not do these things.
    """
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(module))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                names.add(func.id)
            elif isinstance(func, ast.Attribute):
                names.add(func.attr)
    return names


def test_the_loop_never_lowers_the_gate_thresholds():
    """The registry defaults are the gate. run_loop may not pass its own."""
    import inspect

    from research import run_loop as module

    source = inspect.getsource(module)
    assert "min_markets=" not in source, "run_loop must not override the market bar"
    assert "min_folds=" not in source, "run_loop must not override the fold bar"


def test_the_loop_never_calls_an_approval_function():
    from research import run_loop as module

    approving = {n for n in _called_names(module) if "approve" in n.lower()}
    assert not approving, f"run_loop must never approve a run, called: {approving}"


def test_the_detector_would_notice_an_approval_call():
    """Confidence check: a vacuous detector would make the test above worthless."""
    import ast

    guilty = ast.parse(
        "def f(registry):\n"
        "    registry.approve('someone', 'a rationale')\n"
    )
    names = {
        node.func.attr
        for node in ast.walk(guilty)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert any("approve" in name for name in names), "the detector is vacuous"


def test_a_passing_gate_recommends_review_not_approval():
    from research.run_loop import IterationReport

    report = IterationReport(
        iteration=2,
        window_start=CUTOFF - timedelta(days=60),
        window_end=CUTOFF,
        effective_n=80,
        fold_count=3,
        brier=0.20,
        baseline_brier=0.22,
        gate_passed=True,
    )
    text = recommendation_for(report)
    assert "NOT an approval" in text
    assert "named approver" in text
    assert "Nothing has been promoted" in text


# ==========================================================================
# Stages and iteration reporting
# ==========================================================================


@pytest.fixture(scope="module")
def small_settings() -> dict:
    """Real settings with fold sizes small enough for a fast test.

    This shrinks the *test corpus*, never the promotion gate: min_markets and
    min_folds are untouched.
    """
    settings = get_settings()
    settings["research"]["walkforward"] = {
        **settings["research"]["walkforward"],
        "n_folds": 3,
        "test_markets_per_fold": 4,
    }
    settings["research"]["dataset"] = {
        **settings["research"]["dataset"],
        "stride_seconds": 3600,
    }
    return settings


@pytest.fixture(scope="module")
def small_corpus() -> MarketDataset:
    return build_research_corpus({"n_markets": 30, "n_steps": 600})


def test_running_the_stages_produces_a_scored_candidate(small_corpus, small_settings):
    results = run_stages(small_corpus, small_settings)
    assert results.prepared.dataset.effective_n > 0
    assert results.outcome.candidate is not None


def test_a_corpus_with_no_usable_folds_skips_the_later_stages(small_settings):
    tiny = build_research_corpus({"n_markets": 6, "n_steps": 120})
    results = run_stages(tiny, small_settings)
    if not results.outcome.report.usable_folds:
        assert results.backtest is None
        assert results.drift_level is None


def test_an_iteration_report_carries_the_evidence_and_the_verdict(
    small_corpus, small_settings
):
    results = run_stages(small_corpus, small_settings)
    window = (CUTOFF - timedelta(days=30), CUTOFF)
    report = summarise_iteration(1, window, small_corpus, 30, results)

    assert report.iteration == 1
    assert report.corpus_markets == len(small_corpus.markets)
    assert report.new_tickers == 30
    assert report.effective_n == results.outcome.candidate.effective_n
    assert report.fold_count == len(results.outcome.report.usable_folds)
    # The synthetic research corpus can never clear the gate, by design.
    assert report.gate_passed is False
    assert any("fabricated" in f.lower() for f in report.gate_failures)


def test_the_iteration_row_is_displayable(small_corpus, small_settings):
    results = run_stages(small_corpus, small_settings)
    window = (CUTOFF - timedelta(days=30), CUTOFF)
    row = summarise_iteration(1, window, small_corpus, 30, results).as_row()
    assert set(row) >= {"iteration", "effective_n", "folds", "brier", "gate"}
    assert row["gate"] == "failed"


# ==========================================================================
# The loop itself
# ==========================================================================


class StubAdapter:
    """Serves a scripted sequence of discoveries and corpora, offline."""

    def __init__(self, batches: list[list[str]], corpus: MarketDataset) -> None:
        self.batches = batches
        self.corpus = corpus
        self.calls = 0

    class _Client:
        def cutoff(self):
            return {"market_settled_ts": "2026-07-09T00:00:00Z"}

        def markets(self, **kwargs):
            return []

    client = _Client()

    def next_batch(self) -> list[str]:
        batch = self.batches[self.calls] if self.calls < len(self.batches) else []
        self.calls += 1
        return batch


def _patch_loop(monkeypatch, stub: StubAdapter, ids_per_batch: list[list[str]]):
    """Wire discovery/fetch to the stub so the loop runs with no network."""
    import research.run_loop as module

    state = {"i": 0}

    def discover(adapter, start, end, limit, seen=None, config=None, errors=None):
        batch = ids_per_batch[state["i"]] if state["i"] < len(ids_per_batch) else []
        state["i"] += 1
        return [
            DiscoveredMarket(t, CUTOFF - timedelta(days=2), CUTOFF - timedelta(days=1))
            for t in batch
        ]

    def fetch(adapter, tickers, start, end, config=None, errors=None, markets=None):
        ids = set(tickers)
        return MarketDataset(
            markets=[m for m in stub.corpus.markets if m.market_id in ids],
            quotes=[q for q in stub.corpus.quotes if q.market_id in ids],
            trades=[t for t in stub.corpus.trades if t.market_id in ids],
            seed=None,
            label="KALSHI HISTORICAL",
            source="KALSHI_CANDLES",
            venue="kalshi",
        )

    monkeypatch.setattr(module, "discover_settled_markets", discover)
    monkeypatch.setattr(module, "fetch_corpus_chunked", fetch)


def test_the_loop_stops_when_discovery_finds_nothing_new(
    monkeypatch, small_corpus, small_settings, tmp_path
):
    stub = StubAdapter([], small_corpus)
    _patch_loop(monkeypatch, stub, [[]])
    config = LoopConfig(cache_path=str(tmp_path / "c.duckdb"), max_iterations=3)

    result = run_loop(small_settings, config, adapter=stub)
    assert result.stopped_because in {StopReason.NO_NEW_MARKETS, StopReason.NO_DATA}
    assert result.gate_passed is False


def test_the_loop_grows_the_corpus_across_iterations(
    monkeypatch, small_corpus, small_settings, tmp_path
):
    ids = [m.market_id for m in small_corpus.markets]
    stub = StubAdapter([], small_corpus)
    _patch_loop(monkeypatch, stub, [ids[:10], ids[10:20], ids[20:]])
    config = LoopConfig(cache_path=str(tmp_path / "c.duckdb"), max_iterations=3)

    result = run_loop(small_settings, config, adapter=stub)
    counts = [i.corpus_markets for i in result.iterations]
    assert counts == sorted(counts), f"corpus must only grow, got {counts}"
    assert counts[-1] > counts[0]


def test_the_loop_reports_why_the_gate_failed_rather_than_retrying(
    monkeypatch, small_corpus, small_settings, tmp_path
):
    ids = [m.market_id for m in small_corpus.markets]
    stub = StubAdapter([], small_corpus)
    _patch_loop(monkeypatch, stub, [ids])
    config = LoopConfig(cache_path=str(tmp_path / "c.duckdb"), max_iterations=1)

    result = run_loop(small_settings, config, adapter=stub)
    assert result.gate_passed is False
    assert "did NOT pass" in result.recommendation
    assert "No parameters were varied" in result.recommendation
    assert result.iterations[-1].gate_failures


def test_the_loop_gives_up_after_the_configured_stale_iterations(
    monkeypatch, small_corpus, small_settings, tmp_path
):
    ids = [m.market_id for m in small_corpus.markets]
    stub = StubAdapter([], small_corpus)
    # The same markets every time: the corpus cannot grow, so effective_n stalls.
    _patch_loop(monkeypatch, stub, [ids[:10]] * 6)
    config = LoopConfig(
        cache_path=str(tmp_path / "c.duckdb"), max_iterations=10, max_stale_iterations=2
    )

    result = run_loop(small_settings, config, adapter=stub)
    assert result.stopped_because in {StopReason.NO_GROWTH, StopReason.NO_NEW_MARKETS}
    assert len(result.iterations) <= 10, "the loop must not run forever"


def test_the_default_stale_limit_is_five():
    assert LoopConfig().max_stale_iterations == 5


def test_an_empty_report_says_there_is_nothing_to_review():
    from research.run_loop import _no_pass_summary

    assert "Nothing to review" in _no_pass_summary(LoopReport())


# ==========================================================================
# Safety posture
# ==========================================================================


def test_the_loop_adds_no_write_verb_to_the_venue():
    import inspect

    from research import run_loop as module

    source = inspect.getsource(module).lower()
    for verb in ("post(", "put(", "delete(", "patch("):
        assert verb not in source, f"run_loop must never issue {verb}"


def test_live_trading_stays_force_disabled_after_a_loop_run():
    assert get_settings()["app"]["live_trading_enabled"] is False


# ==========================================================================
# Minimum lifetime - markets that cannot yield a row must not enter the corpus
# ==========================================================================


def test_market_lifetime_is_measured_from_open_to_close():
    payload = {"open_time": "2026-07-01T00:00:00Z", "close_time": "2026-07-02T12:00:00Z"}
    assert market_lifetime_hours(payload) == pytest.approx(36.0)


def test_a_market_without_timestamps_has_no_measurable_lifetime():
    assert market_lifetime_hours({"open_time": "2026-07-01T00:00:00Z"}) is None
    assert market_lifetime_hours({}) is None


def test_a_market_too_short_to_produce_a_decision_timestamp_is_excluded():
    """A one-hour market yields zero rows at horizon=1h.

    However many of them we pull, effective_n stays at zero.
    """
    short = {**copy.deepcopy(fx.SETTLED_MARKET),
             "open_time": "2026-07-08T20:00:00Z", "close_time": "2026-07-08T21:00:00Z"}
    assert is_settled_binary(short) is True, "still a valid settled binary market"
    assert is_settled_binary(short, min_lifetime_hours=6.0) is False


def test_a_long_enough_market_survives_the_lifetime_filter():
    long_lived = {**copy.deepcopy(fx.SETTLED_MARKET),
                  "open_time": "2026-07-06T00:00:00Z", "close_time": "2026-07-08T00:00:00Z"}
    assert is_settled_binary(long_lived, min_lifetime_hours=6.0) is True


def test_the_lifetime_filter_is_off_unless_asked_for():
    short = {**copy.deepcopy(fx.SETTLED_MARKET),
             "open_time": "2026-07-08T20:00:00Z", "close_time": "2026-07-08T21:00:00Z"}
    assert is_settled_binary(short, min_lifetime_hours=0.0) is True


def test_the_default_lifetime_floor_exceeds_the_feature_horizon():
    """The floor must clear the feature horizon.

    Otherwise the corpus fills with markets that produce nothing.
    """
    horizon_hours = get_settings()["research"]["dataset"]["horizon_seconds"] / 3600
    assert LoopConfig().min_lifetime_hours > horizon_hours


# ==========================================================================
# Historical endpoint fallback
# ==========================================================================


def test_the_fetch_falls_back_to_the_series_path_when_historical_404s():
    """The historical candlestick endpoint 404s for every market observed.

    A 404 there aborts the whole batch rather than skipping one ticker.
    """

    class Flaky:
        def __init__(self):
            self.attempts = []

        def fetch_dataset(self, spec):
            self.attempts.append(spec.historical)
            if spec.historical:
                raise AdapterError("HTTP 404 ... not found")
            return MarketDataset(markets=[], quotes=[], trades=[], seed=None,
                                 label="KALSHI HISTORICAL", source="KALSHI_CANDLES")

    adapter = Flaky()
    result = fetch_corpus(adapter, ["X"], CUTOFF - timedelta(days=1), CUTOFF)
    assert adapter.attempts == [True, False], "must try historical, then fall back"
    assert result is not None


def test_the_fallback_can_be_turned_off():
    class Flaky:
        def __init__(self):
            self.attempts = []

        def fetch_dataset(self, spec):
            self.attempts.append(spec.historical)
            return MarketDataset(markets=[], quotes=[], trades=[], seed=None)

    adapter = Flaky()
    fetch_corpus(adapter, ["X"], CUTOFF - timedelta(days=1), CUTOFF,
                 LoopConfig(historical_first=False))
    assert adapter.attempts == [False]


def test_both_paths_failing_returns_none_rather_than_raising():
    class Broken:
        def fetch_dataset(self, spec):
            raise AdapterError("gone")

    assert fetch_corpus(Broken(), ["X"], CUTOFF - timedelta(days=1), CUTOFF) is None


# ==========================================================================
# Entry point
# ==========================================================================


def test_the_module_is_runnable_as_a_script():
    from research.run_loop import main

    assert callable(main)


def test_a_corpus_with_no_tape_can_still_be_cached(tmp_path):
    """Regression: run_loop fetches without trades to stay inside the throttle.

    An empty list gave pd.DataFrame([]) - zero columns - and DuckDB refused to
    create a table from it, so the loop died before its first iteration report.
    """
    base = generate_demo_dataset({"n_markets": 4, "n_steps": 40})
    no_tape = MarketDataset(
        markets=base.markets,
        quotes=base.quotes,
        trades=[],
        seed=None,
        label="KALSHI HISTORICAL",
        source="KALSHI_CANDLES",
        venue="kalshi",
    )
    path = tmp_path / "no_tape.duckdb"
    write_dataset(no_tape, path)
    back = read_dataset(path)

    assert back.trades == []
    assert len(back.markets) == len(no_tape.markets)
    assert len(back.quotes) == len(no_tape.quotes)
    assert describe(back).is_synthetic is False


def test_a_completely_empty_corpus_round_trips(tmp_path):
    path = tmp_path / "empty.duckdb"
    write_dataset(MarketDataset(markets=[], quotes=[], trades=[], seed=None), path)
    back = read_dataset(path)
    assert (back.markets, back.quotes, back.trades) == ([], [], [])


# ==========================================================================
# Corpus vs out-of-sample effective_n - the distinction that matters
# ==========================================================================


def test_the_report_separates_corpus_and_out_of_sample_market_counts(
    small_corpus, small_settings
):
    """A corpus can hold resolved markets while the OOS count is zero.

    Conflating the two reads as "no data" when the real problem is empty folds.
    """
    results = run_stages(small_corpus, small_settings)
    window = (CUTOFF - timedelta(days=30), CUTOFF)
    report = summarise_iteration(1, window, small_corpus, 30, results)

    assert report.corpus_effective_n == results.prepared.dataset.effective_n
    assert report.rows_labelled == len(results.prepared.dataset.labelled)
    assert report.effective_n == results.outcome.candidate.effective_n


def test_empty_folds_are_diagnosed_as_folds_not_as_missing_data():
    from research.run_loop import IterationReport, _no_pass_summary

    report = IterationReport(
        iteration=1,
        window_start=CUTOFF - timedelta(days=30),
        window_end=CUTOFF,
        corpus_markets=750,
        corpus_effective_n=94,
        rows_labelled=5553,
        effective_n=0,
        fold_count=0,
        gate_failures=["Walk-forward folds: 0 usable fold(s); at least 3 required."],
    )
    text = _no_pass_summary(LoopReport(iterations=[report]))

    assert "94 resolved" in text
    assert "The data is there; the folds are empty" in text
    assert "research.walkforward" in text


def test_no_usable_rows_is_diagnosed_as_a_data_problem():
    from research.run_loop import IterationReport, _no_pass_summary

    report = IterationReport(
        iteration=1,
        window_start=CUTOFF - timedelta(days=30),
        window_end=CUTOFF,
        corpus_markets=750,
        corpus_effective_n=0,
        effective_n=0,
        fold_count=0,
    )
    text = _no_pass_summary(LoopReport(iterations=[report]))
    assert "period_interval" in text


def test_growth_is_measured_on_the_corpus_not_the_out_of_sample_count(
    monkeypatch, small_corpus, small_settings, tmp_path
):
    """Out-of-sample effective_n is 0 whenever folds are empty.

    Using it to detect growth would call a genuinely growing corpus stale.
    """
    ids = [m.market_id for m in small_corpus.markets]
    stub = StubAdapter([], small_corpus)
    _patch_loop(monkeypatch, stub, [ids[:10], ids[10:20], ids[20:]])
    config = LoopConfig(
        cache_path=str(tmp_path / "c.duckdb"), max_iterations=3, max_stale_iterations=1
    )

    result = run_loop(small_settings, config, adapter=stub)
    corpus_counts = [i.corpus_effective_n for i in result.iterations]
    assert corpus_counts == sorted(corpus_counts)
    assert len(result.iterations) >= 2, "a growing corpus must not stop after one pass"


# ==========================================================================
# Chunked fetching - one bad ticker must not lose the whole iteration
# ==========================================================================


def test_a_failing_chunk_does_not_lose_the_markets_that_fetched_cleanly():
    """fetch_dataset does not guard the candlestick call.

    One 404 there aborts the batch, so fetching 300 at once means one bad
    market loses all 300 - which then reads as "no data available".
    """
    base = generate_demo_dataset({"n_markets": 4, "n_steps": 40})
    good = {"DEMO-01", "DEMO-02"}

    class PartlyBroken:
        def fetch_dataset(self, spec):
            if any(t not in good for t in spec.tickers):
                raise AdapterError("HTTP 404 candlesticks")
            ids = set(spec.tickers)
            return MarketDataset(
                markets=[m for m in base.markets if m.market_id in ids],
                quotes=[q for q in base.quotes if q.market_id in ids],
                trades=[], seed=None, label="KALSHI HISTORICAL",
                source="KALSHI_CANDLES",
            )

    errors: list[str] = []
    corpus = fetch_corpus_chunked(
        PartlyBroken(), ["DEMO-01", "DEMO-02", "DEMO-03", "DEMO-04"],
        CUTOFF - timedelta(days=1), CUTOFF,
        LoopConfig(fetch_chunk_size=1, historical_first=False), errors,
    )
    assert corpus is not None, "the good chunks must survive"
    assert {m.market_id for m in corpus.markets} == good
    assert len(errors) == 2, "each failed chunk must be recorded, not swallowed"


def test_every_chunk_failing_returns_nothing_and_says_so():
    class Broken:
        def fetch_dataset(self, spec):
            raise AdapterError("gone")

    errors: list[str] = []
    result = fetch_corpus_chunked(
        Broken(), ["A", "B"], CUTOFF - timedelta(days=1), CUTOFF,
        LoopConfig(fetch_chunk_size=1), errors,
    )
    assert result is None
    assert len(errors) == 2


def test_chunking_covers_every_ticker_exactly_once():
    seen: list[str] = []
    base = generate_demo_dataset({"n_markets": 4, "n_steps": 40})

    class Recording:
        def fetch_dataset(self, spec):
            seen.extend(spec.tickers)
            ids = set(spec.tickers)
            return MarketDataset(
                markets=[m for m in base.markets if m.market_id in ids],
                quotes=[], trades=[], seed=None,
                label="KALSHI HISTORICAL", source="KALSHI_CANDLES",
            )

    tickers = [f"DEMO-0{i}" for i in range(1, 5)]
    fetch_corpus_chunked(
        Recording(), tickers, CUTOFF - timedelta(days=1), CUTOFF,
        LoopConfig(fetch_chunk_size=3, historical_first=False),
    )
    assert seen == tickers


# ==========================================================================
# Per-chunk fetch windows - the venue refuses too many candles at once
# ==========================================================================


def test_discovery_keeps_each_market_own_window(tmp_path):
    adapter, _ = make_adapter(tmp_path)
    found = discover_settled_markets(adapter, CUTOFF - timedelta(days=30), CUTOFF, 10)
    assert found, "expected the settled fixture market"
    for market in found:
        assert market.closed > market.opened


def test_discovered_markets_come_back_in_close_order(tmp_path):
    adapter, _ = make_adapter(tmp_path)
    found = discover_settled_markets(adapter, CUTOFF - timedelta(days=30), CUTOFF, 10)
    assert [m.closed for m in found] == sorted(m.closed for m in found)


def test_a_chunk_is_fetched_over_its_own_span_not_the_whole_window():
    """A 30-day window at 1-minute candles asks for ~43,000 candles.

    The venue answers HTTP 400 rather than truncating, so the request must be
    bounded to what the chunk actually spans.
    """
    seen: list[tuple[int, int]] = []

    class Recording:
        def fetch_dataset(self, spec):
            seen.append((spec.start_ts, spec.end_ts))
            return MarketDataset(markets=[], quotes=[], trades=[], seed=None,
                                 label="KALSHI HISTORICAL", source="KALSHI_CANDLES")

    opened = CUTOFF - timedelta(days=2)
    closed = CUTOFF - timedelta(days=1)
    markets = [DiscoveredMarket("A", opened, closed), DiscoveredMarket("B", opened, closed)]

    fetch_corpus_chunked(
        Recording(), ["A", "B"], CUTOFF - timedelta(days=120), CUTOFF,
        LoopConfig(fetch_chunk_size=25, historical_first=False), None, markets,
    )
    assert len(seen) == 1
    span_hours = (seen[0][1] - seen[0][0]) / 3600
    assert span_hours < 24 * 5, f"window should track the chunk, got {span_hours:.0f}h"


def test_a_chunk_with_no_known_span_falls_back_to_the_iteration_window():
    seen: list[tuple[int, int]] = []

    class Recording:
        def fetch_dataset(self, spec):
            seen.append((spec.start_ts, spec.end_ts))
            return MarketDataset(markets=[], quotes=[], trades=[], seed=None)

    lo, hi = CUTOFF - timedelta(days=10), CUTOFF
    fetch_corpus_chunked(Recording(), ["A"], lo, hi,
                         LoopConfig(historical_first=False), None, None)
    assert seen and seen[0][1] == int(hi.timestamp())


def test_a_run_with_no_iterations_still_says_what_went_wrong():
    from research.run_loop import _no_pass_summary

    report = LoopReport(
        stopped_because=StopReason.NO_DATA,
        errors=["chunk 1 (25 ticker(s)) failed to fetch"],
    )
    text = _no_pass_summary(report)
    assert "What went wrong" in text
    assert "failed to fetch" in text
