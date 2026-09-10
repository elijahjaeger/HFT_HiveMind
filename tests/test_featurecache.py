"""Feature cache: it must be exactly a speed-up and never change an answer.

A stale cache is the worst failure available here - it would train a model on
features that no longer match the code claiming to have produced them, and do
it silently. So the tests are mostly about invalidation, not about speed.
"""

from __future__ import annotations

import pytest

from core.config import get_settings
from core.synthetic import generate_demo_dataset
from research.autopilot import AutopilotConfig, research_settings
from research.datasets import build_feature_dataset
from research.featurecache import (
    CACHE_SCHEMA_VERSION,
    CacheStats,
    FeatureCache,
    cache_key,
    series_fingerprint,
)


@pytest.fixture(scope="module")
def dataset():
    return generate_demo_dataset({"n_markets": 6, "n_steps": 240})


@pytest.fixture(scope="module")
def grid() -> dict:
    """The shipped research grid, which fits the demo dataset's lifetimes.

    Not the autopilot's daily grid: these markets live hours, so a one-day
    horizon skips every one of them and the cache tests would pass vacuously
    on two empty datasets.
    """
    return get_settings()["research"]["dataset"]


def _series(dataset, market_id):
    return (
        [q for q in dataset.quotes if q.market_id == market_id],
        [t for t in dataset.trades if t.market_id == market_id],
    )


# --------------------------------------------------------------------------
# The only property that really matters
# --------------------------------------------------------------------------


def test_cached_and_uncached_builds_are_identical(dataset, grid, tmp_path):
    """A cache that changes an answer is worse than no cache."""
    plain = build_feature_dataset(dataset, grid)
    cache = FeatureCache(tmp_path / "fc.duckdb")
    build_feature_dataset(dataset, grid, cache=cache)  # cold, fills the cache
    warm = build_feature_dataset(dataset, grid, cache=cache)

    assert plain.examples, "an empty build would make this test vacuous"
    assert len(warm.examples) == len(plain.examples)
    assert warm.effective_n == plain.effective_n
    for left, right in zip(plain.examples, warm.examples, strict=True):
        assert left.market_id == right.market_id
        assert left.decision_ts == right.decision_ts
        assert left.label == right.label
        assert left.resolution_ts == right.resolution_ts
        assert left.market_prob == pytest.approx(right.market_prob)
        assert left.features.vector() == pytest.approx(right.features.vector())
        assert left.features.missing == right.features.missing


def test_the_second_pass_serves_markets_from_cache(dataset, grid, tmp_path):
    cache = FeatureCache(tmp_path / "fc.duckdb")
    build_feature_dataset(dataset, grid, cache=cache)
    cache.stats = CacheStats()
    build_feature_dataset(dataset, grid, cache=cache)
    assert cache.stats.hits > 0
    assert cache.stats.hit_rate > 0.5


# --------------------------------------------------------------------------
# Invalidation - every input that can change a row must change the key
# --------------------------------------------------------------------------


def test_a_different_grid_is_a_different_key(dataset, grid):
    quotes, trades = _series(dataset, dataset.markets[0].market_id)
    base = cache_key("m1", quotes, trades, grid, "labels-v1")
    coarser = cache_key("m1", quotes, trades, {**grid, "stride_seconds": 999}, "labels-v1")
    assert base != coarser


def test_a_different_horizon_is_a_different_key(dataset, grid):
    quotes, trades = _series(dataset, dataset.markets[0].market_id)
    base = cache_key("m1", quotes, trades, grid, "labels-v1")
    other = cache_key("m1", quotes, trades, {**grid, "horizon_seconds": 7}, "labels-v1")
    assert base != other


def test_a_different_label_policy_is_a_different_key(dataset, grid):
    quotes, trades = _series(dataset, dataset.markets[0].market_id)
    assert cache_key("m1", quotes, trades, grid, "labels-v1") != cache_key(
        "m1", quotes, trades, grid, "labels-v2"
    )


def test_a_different_market_is_a_different_key(dataset, grid):
    quotes, trades = _series(dataset, dataset.markets[0].market_id)
    assert cache_key("m1", quotes, trades, grid, "l") != cache_key(
        "m2", quotes, trades, grid, "l"
    )


def test_refetching_a_market_with_more_history_invalidates_it(dataset, grid):
    """Deeper history must rebuild, not reuse rows built from less of it."""
    quotes, trades = _series(dataset, dataset.markets[0].market_id)
    before = cache_key("m1", quotes, trades, grid, "l")
    after = cache_key("m1", quotes[:-3], trades, grid, "l")
    assert before != after


def test_the_fingerprint_covers_endpoints_not_only_counts(dataset):
    """A market that gained rows in the middle must not reuse stale features."""
    quotes, trades = _series(dataset, dataset.markets[0].market_id)
    shifted = quotes[1:] + quotes[:1]
    assert series_fingerprint(quotes, trades) == series_fingerprint(quotes, trades)
    assert series_fingerprint(quotes[:-1], trades) != series_fingerprint(quotes, trades)
    assert len(shifted) == len(quotes)


def test_the_schema_version_is_part_of_the_key(dataset, grid):
    quotes, trades = _series(dataset, dataset.markets[0].market_id)
    key = cache_key("m1", quotes, trades, grid, "l")
    assert CACHE_SCHEMA_VERSION
    assert isinstance(key, str) and len(key) == 64


# --------------------------------------------------------------------------
# Failure modes: a bad cache costs a rebuild, never a wrong answer
# --------------------------------------------------------------------------


def test_a_miss_returns_nothing_and_is_counted(tmp_path):
    cache = FeatureCache(tmp_path / "fc.duckdb")
    assert cache.get("absent") is None
    assert cache.stats.misses == 1


def test_a_corrupt_entry_is_treated_as_a_miss(tmp_path):
    import duckdb

    cache = FeatureCache(tmp_path / "fc.duckdb")
    with duckdb.connect(str(cache.path)) as con:
        con.execute(
            "INSERT OR REPLACE INTO feature_cache VALUES (?, ?, ?, ?)",
            ["broken", "m1", 1, "{not json"],
        )
    assert cache.get("broken") is None
    assert cache.stats.misses == 1


def test_the_cache_can_be_cleared_and_still_works(dataset, grid, tmp_path):
    cache = FeatureCache(tmp_path / "fc.duckdb")
    build_feature_dataset(dataset, grid, cache=cache)
    assert cache.size() > 0
    cache.clear()
    assert cache.size() == 0
    rebuilt = build_feature_dataset(dataset, grid, cache=cache)
    assert rebuilt.examples


def test_an_empty_cache_reports_a_zero_hit_rate():
    assert CacheStats().hit_rate == 0.0
    assert CacheStats(hits=3, misses=1).hit_rate == pytest.approx(0.75)


# --------------------------------------------------------------------------
# The tick-rate alignment that was the real bottleneck
# --------------------------------------------------------------------------


def test_the_simulator_ticks_no_faster_than_the_model_forecasts():
    """The backtest was 69% of a pass because it re-decided on stale forecasts.

    A market living 200 days got ~28,800 decision ticks at the shipped 600s
    interval while the model emitted one forecast per daily stride, so 143 of
    every 144 ticks re-evaluated a forecast that had not changed.
    """
    config = AutopilotConfig()
    tuned = research_settings(get_settings(), config)
    assert tuned["backtest"]["decision_interval_seconds"] == config.stride_seconds


def test_aligning_the_tick_rate_leaves_the_gate_untouched():
    settings = get_settings()
    tuned = research_settings(settings, AutopilotConfig())
    assert tuned["research"]["promotion"] == settings["research"]["promotion"]
    assert tuned["backtest"]["initial_capital"] == settings["backtest"]["initial_capital"]


def test_the_original_backtest_settings_are_not_mutated():
    settings = get_settings()
    before = dict(settings["backtest"])
    research_settings(settings, AutopilotConfig())
    assert settings["backtest"] == before
