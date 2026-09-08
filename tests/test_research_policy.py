"""The threshold policy: what it costs to trade, and every reason it declines.

The policy's job is to say *no* well. These tests concentrate on the decline
paths, because a policy that trades when it should not is how a research
pipeline turns into a loss.
"""

from __future__ import annotations

import pytest

from backtest.skips import SkipReason, skip_summary
from research.policy import (
    PolicyConfig,
    freshness_window,
    strategy_config_for,
)
from tests.conftest import T0

FILLS = {
    "base_slippage": 0.002,
    "impact_coefficient": 0.0,
    "fee_rate": 0.02,
    "fee_model": "kalshi",
    "tick_size": 0.01,
}


def policy(**overrides) -> PolicyConfig:
    return PolicyConfig(**{"safety_margin": 0.02, "min_confidence": 0.0,
                           "max_quote_age_seconds": 900.0, **overrides})


# -- the cost stack --------------------------------------------------------


# -- declining safely ------------------------------------------------------


def test_the_freshness_window_scales_with_the_decision_grid():
    assert freshness_window(1800.0, multiplier=1.5) == pytest.approx(2700.0)
    assert freshness_window(1.0) >= 60.0, "never shorter than a sensible floor"


def test_the_strategy_config_carries_the_policy_thresholds():
    config = strategy_config_for(policy(safety_margin=0.05), FILLS)
    assert config["min_edge"] >= 0.05
    assert config["require_positive_ev"] is True


# -- skip reasons ----------------------------------------------------------


def test_every_skip_reason_is_distinct_and_named():
    values = [r.value for r in SkipReason]
    assert len(values) == len(set(values))
    assert {"INSUFFICIENT_EDGE", "STALE_DATA", "MISSING_FORECAST",
            "INVALID_DATA", "RISK_CAP"} <= set(values)


def test_the_skip_summary_counts_by_reason():
    from backtest.skips import SkippedDecision

    skips = [
        SkippedDecision(ts=T0, market_id="A", reason=SkipReason.INSUFFICIENT_EDGE,
                        detail="edge too small"),
        SkippedDecision(ts=T0, market_id="B", reason=SkipReason.INSUFFICIENT_EDGE,
                        detail="edge too small"),
        SkippedDecision(ts=T0, market_id="C", reason=SkipReason.STALE_DATA,
                        detail="quote too old"),
    ]
    rows = {r["reason"]: r["count"] for r in skip_summary(skips)}
    assert rows["INSUFFICIENT_EDGE"] == 2
    assert rows["STALE_DATA"] == 1


def test_an_empty_skip_list_summarises_to_nothing():
    assert skip_summary([]) == []


# -- integration with the simulator ---------------------------------------


def forecast_for(market_id: str, ts, probability: float):
    """A ProbabilityForecast the policy agent can consume."""
    from research.models import ProbabilityForecast

    return ProbabilityForecast(
        market_ticker=market_id,
        decision_timestamp=ts,
        raw_probability_yes=probability,
        calibrated_probability_yes=probability,
        confidence=0.8,
        feature_version="features-v1",
        model_version="logistic_regression/1.0",
        data_version="data-test",
    )




def test_there_is_only_one_implementation_of_the_threshold_rule():
    """Regression guard against reintroducing the parallel policy engine.

    research/policy.py used to carry a second implementation of "edge must beat
    the cost stack" - assess_edge/evaluate_policy - reachable only from its own
    unit tests. It measured edge against the midpoint while the shipped path
    (agents.strategy.build_proposal, configured by strategy_config_for)
    measures it against the crossing price, so the two disagreed about which
    trades existed. Removed 2026-09-08; see DECISIONS.md.
    """
    from research import policy as module

    for name in ("assess_edge", "evaluate_policy", "EdgeAssessment",
                 "PolicyRun", "PolicyDecision"):
        assert not hasattr(module, name), (
            f"{name} is back. The threshold rule has one implementation: "
            "strategy_config_for feeding agents.strategy.build_proposal."
        )


def test_the_pipeline_uses_the_shared_freshness_window():
    """The constant was defined here and inlined in the pipeline separately."""
    import inspect

    from research import pipeline

    source = inspect.getsource(pipeline.run_model_backtest)
    assert "freshness_window(" in source
    assert "stride * 1.5" not in source, "the inlined duplicate is back"
