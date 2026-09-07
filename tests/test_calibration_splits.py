"""Chronological splits and the three leakage rules they enforce."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from calibration.splits import (
    ChronologicalSplit,
    Observation,
    chronological_split,
    observations_from_result,
)

T0 = datetime(2026, 1, 5, 9, tzinfo=UTC)


def observation(
    market: str,
    minutes: int,
    probability: float = 0.5,
    outcome: float = 1.0,
    resolves_after: int = 600,
) -> Observation:
    """Build one observation ``minutes`` after T0."""
    ts = T0 + timedelta(minutes=minutes)
    return Observation(
        ts=ts,
        market_id=market,
        probability=probability,
        outcome=outcome,
        resolution_ts=ts + timedelta(minutes=resolves_after),
    )


def rolling_observations(
    n_markets: int = 8, per_market: int = 10, lifetime: int = 60
) -> list[Observation]:
    """Markets that open and resolve in sequence, like the demo data."""
    out: list[Observation] = []
    for index in range(n_markets):
        open_minutes = index * lifetime
        resolution = T0 + timedelta(minutes=open_minutes + lifetime)
        for step in range(per_market):
            ts = T0 + timedelta(
                minutes=open_minutes + step * (lifetime / per_market)
            )
            out.append(
                Observation(
                    ts=ts,
                    market_id=f"M{index:02d}",
                    probability=0.3 + 0.4 * (step / per_market),
                    outcome=float(index % 2),
                    resolution_ts=resolution,
                )
            )
    return out


# -- the Observation contract ---------------------------------------------


def test_an_outcome_cannot_predate_the_forecast_that_predicted_it():
    with pytest.raises(ValueError, match="before the forecast"):
        Observation(
            ts=T0,
            market_id="M1",
            probability=0.5,
            outcome=1.0,
            resolution_ts=T0 - timedelta(minutes=1),
        )


@pytest.mark.parametrize("probability", [-0.01, 1.01])
def test_probabilities_must_be_in_the_unit_interval(probability):
    with pytest.raises(ValueError, match="probability out of range"):
        observation("M1", 0, probability=probability)


def test_outcomes_must_be_binary():
    with pytest.raises(ValueError, match="must be 0.0 or 1.0"):
        observation("M1", 0, outcome=0.5)


# -- leak 1: the split is chronological -----------------------------------


def test_folds_are_ordered_in_time():
    split = chronological_split(rolling_observations())
    assert split.train.end is not None
    assert split.validation.start is not None
    assert split.train.end <= split.validation.start
    assert split.validation.end <= split.test.start


def test_input_order_does_not_change_the_split():
    observations = rolling_observations()
    forwards = chronological_split(observations)
    backwards = chronological_split(list(reversed(observations)))
    assert forwards.train.market_ids == backwards.train.market_ids
    assert forwards.test.market_ids == backwards.test.market_ids


# -- leak 2: no market spans two folds ------------------------------------


def test_every_market_lands_in_exactly_one_fold():
    split = chronological_split(rolling_observations())
    train = set(split.train.market_ids)
    validation = set(split.validation.market_ids)
    test = set(split.test.market_ids)
    assert train & validation == set()
    assert train & test == set()
    assert validation & test == set()


def test_grouping_can_be_disabled_for_experiments():
    """Ungrouped splitting is available but permits same-market leakage."""
    split = chronological_split(rolling_observations(), group_by_market=False)
    assert split.grouped_by_market is False


def test_the_effective_sample_size_is_the_market_count_not_the_row_count():
    split = chronological_split(rolling_observations(n_markets=8, per_market=50))
    assert len(split.train) > 100
    assert split.train.effective_n == len(split.train.market_ids)
    assert split.train.effective_n < 10


# -- leak 3: outcomes cannot be used before they were known ----------------


def test_a_fold_reports_when_its_outcomes_became_knowable():
    split = chronological_split(rolling_observations())
    assert split.train.knowledge_time == max(
        o.resolution_ts for o in split.train.observations
    )


def test_later_folds_drop_forecasts_made_before_the_training_outcomes_were_known():
    split = chronological_split(rolling_observations())
    cutoff = split.train.knowledge_time
    assert cutoff is not None
    assert all(o.ts >= cutoff for o in split.validation.observations)
    assert all(o.ts >= cutoff for o in split.test.observations)


def test_the_test_fold_also_respects_the_validation_knowledge_time():
    split = chronological_split(rolling_observations())
    cutoff = split.validation.knowledge_time
    if cutoff is not None and split.test.observations:
        assert all(o.ts >= cutoff for o in split.test.observations)


def test_dropped_observations_are_counted_not_silently_discarded():
    """Concurrent markets force drops, and the split must say so."""
    concurrent = [
        observation(f"M{i}", minutes=step, outcome=float(i % 2), resolves_after=1000)
        for i in range(6)
        for step in range(10)
    ]
    split = chronological_split(concurrent)
    assert split.total_dropped > 0
    assert sum(f.dropped for f in split.folds) == split.total_dropped
    assert any("dropped" in w for w in split.warnings)


def test_the_knowledge_time_guard_can_be_disabled_explicitly():
    concurrent = [
        observation(f"M{i}", minutes=step, outcome=float(i % 2), resolves_after=1000)
        for i in range(6)
        for step in range(10)
    ]
    guarded = chronological_split(concurrent)
    unguarded = chronological_split(concurrent, enforce_knowledge_time=False)
    assert unguarded.total_dropped == 0
    assert len(unguarded.test) >= len(guarded.test)


# -- fractions and edge cases ---------------------------------------------


def test_the_fractions_shape_the_folds():
    observations = rolling_observations(n_markets=10, per_market=10)
    split = chronological_split(observations, train_fraction=0.8, validation_fraction=0.1)
    assert len(split.train.market_ids) > len(split.test.market_ids)


@pytest.mark.parametrize(
    "train,validation", [(0.0, 0.25), (0.9, 0.2), (1.0, 0.0), (-0.1, 0.5)]
)
def test_invalid_fractions_are_rejected(train, validation):
    with pytest.raises(ValueError):
        chronological_split(rolling_observations(), train, validation)


def test_an_empty_input_yields_empty_folds():
    split = chronological_split([])
    assert isinstance(split, ChronologicalSplit)
    assert not split.is_usable
    assert split.total_dropped == 0


def test_a_thin_split_is_flagged_as_untrustworthy():
    split = chronological_split(rolling_observations(n_markets=6))
    assert not split.is_usable or split.warnings
    assert any("far too few" in w for w in split.warnings)


def test_a_degenerate_fold_is_flagged():
    """Every market resolving the same way makes the fold's scores meaningless."""
    same = [
        Observation(
            ts=T0 + timedelta(minutes=i * 60 + step),
            market_id=f"M{i:02d}",
            probability=0.5,
            outcome=1.0,
            resolution_ts=T0 + timedelta(minutes=i * 60 + 60),
        )
        for i in range(9)
        for step in range(5)
    ]
    assert any("resolved the same way" in w for w in chronological_split(same).warnings)


# -- extraction from a backtest result ------------------------------------


def test_observations_are_extracted_from_a_backtest(demo_result):
    observations = observations_from_result(demo_result)
    assert observations
    assert all(0.0 <= o.probability <= 1.0 for o in observations)
    assert all(o.outcome in (0.0, 1.0) for o in observations)
    assert [o.ts for o in observations] == sorted(o.ts for o in observations)


def test_extracted_forecasts_always_predate_their_own_resolution(demo_result):
    for observation_ in observations_from_result(demo_result):
        assert observation_.ts <= observation_.resolution_ts


def test_a_single_agent_can_be_scored_on_its_own_forecasts(demo_result):
    consensus = observations_from_result(demo_result, source="consensus")
    market_agent = observations_from_result(demo_result, source="MarketAgent")
    assert market_agent
    assert {o.source for o in market_agent} == {"MarketAgent"}
    assert len(market_agent) >= len(consensus) - len(consensus)  # both non-empty


def test_unresolved_markets_contribute_no_observations(demo_result):
    resolved = {m.market_id for m in demo_result.markets if m.outcome is not None}
    assert {o.market_id for o in observations_from_result(demo_result)} <= resolved
