"""The threshold policy: what it costs to trade, and every reason it declines.

The policy's job is to say *no* well. These tests concentrate on the decline
paths, because a policy that trades when it should not is how a research
pipeline turns into a loss.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from backtest.skips import SkipReason, skip_summary
from core.models import Side
from research.policy import (
    PolicyConfig,
    assess_edge,
    freshness_window,
    strategy_config_for,
)
from tests.conftest import T0, make_quote, make_state

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


def test_the_required_edge_is_the_sum_of_every_modelled_cost():
    state = make_state(quote=make_quote(bid=0.40, ask=0.44, ts=T0), as_of=T0)
    assessment = assess_edge(state, 0.80, policy(), FILLS)
    assert assessment.required_edge == pytest.approx(
        assessment.half_spread + assessment.fee
        + assessment.slippage + assessment.safety_margin
    )
    assert assessment.net_edge == pytest.approx(
        assessment.raw_edge - assessment.required_edge
    )


def test_a_confident_forecast_above_the_ask_wants_to_buy():
    state = make_state(quote=make_quote(bid=0.40, ask=0.44, ts=T0), as_of=T0)
    assessment = assess_edge(state, 0.85, policy(), FILLS)
    assert assessment.side is Side.BUY
    assert assessment.is_tradable is True


def test_a_confident_forecast_below_the_bid_wants_to_sell():
    state = make_state(quote=make_quote(bid=0.60, ask=0.64, ts=T0), as_of=T0)
    assessment = assess_edge(state, 0.10, policy(), FILLS)
    assert assessment.side is Side.SELL
    assert assessment.is_tradable is True


def test_agreeing_with_the_market_is_not_tradable():
    quote = make_quote(bid=0.40, ask=0.44, ts=T0)
    state = make_state(quote=quote, as_of=T0)
    assessment = assess_edge(state, quote.mid, policy(), FILLS)
    assert assessment.is_tradable is False
    assert assessment.net_edge < 0


def test_a_thin_edge_is_eaten_by_the_cost_stack():
    """The half-spread alone is 0.02 here, before fees, slippage or margin."""
    state = make_state(quote=make_quote(bid=0.40, ask=0.44, ts=T0), as_of=T0)
    assessment = assess_edge(state, 0.45, policy(), FILLS)
    assert assessment.raw_edge > 0
    assert assessment.is_tradable is False


def test_a_larger_safety_margin_makes_fewer_trades_tradable():
    state = make_state(quote=make_quote(bid=0.40, ask=0.44, ts=T0), as_of=T0)
    lenient = assess_edge(state, 0.55, policy(safety_margin=0.0), FILLS)
    strict = assess_edge(state, 0.55, policy(safety_margin=0.20), FILLS)
    assert strict.required_edge > lenient.required_edge
    assert strict.net_edge < lenient.net_edge


def test_the_cost_stack_renders_for_the_dashboard():
    state = make_state(quote=make_quote(bid=0.40, ask=0.44, ts=T0), as_of=T0)
    rows = assess_edge(state, 0.85, policy(), FILLS).as_rows()
    components = {r["component"] for r in rows}
    assert {"Market probability", "Model probability", "Raw edge"} <= components
    assert any("fee" in c for c in components)
    assert any("spread" in c for c in components)


# -- declining safely ------------------------------------------------------


def test_no_quote_yields_no_assessment_at_all():
    """There is nothing to assess without a book, so ``None`` is returned."""
    assert assess_edge(make_state(quote=None, as_of=T0), 0.9, policy(), FILLS) is None


def test_a_one_sided_book_yields_no_assessment():
    """Half a book cannot price the cost of crossing it."""
    state = make_state(quote=make_quote(bid=0.40, ask=None, ts=T0), as_of=T0)
    assert assess_edge(state, 0.99, policy(), FILLS) is None


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


def test_stale_data_produces_a_no_trade_not_a_trade():
    """A quote past the freshness window must never be crossed."""
    from research.policy import ModelForecastAgent, evaluate_policy

    stale_state = make_state(
        quote=make_quote(bid=0.40, ask=0.44, ts=T0),
        as_of=T0 + timedelta(seconds=7200),
    )
    agent = ModelForecastAgent([forecast_for("M1", stale_state.as_of, 0.95)])
    run = evaluate_policy(agent, [stale_state], policy(max_quote_age_seconds=900.0), FILLS)
    assert run.decisions
    assert all(d.skip is not None for d in run.decisions)
    assert any(d.skip.reason is SkipReason.STALE_DATA for d in run.decisions)


def test_a_missing_forecast_produces_a_no_trade():
    from research.policy import ModelForecastAgent, evaluate_policy

    state = make_state(quote=make_quote(bid=0.40, ask=0.44, ts=T0), as_of=T0)
    agent = ModelForecastAgent([])  # nothing forecast for this market
    run = evaluate_policy(agent, [state], policy(), FILLS)
    assert all(d.skip is not None for d in run.decisions)
    assert any(d.skip.reason is SkipReason.MISSING_FORECAST for d in run.decisions)


def test_an_insufficient_edge_is_recorded_with_its_reason():
    from research.policy import ModelForecastAgent, evaluate_policy

    quote = make_quote(bid=0.40, ask=0.44, ts=T0)
    state = make_state(quote=quote, as_of=T0)
    agent = ModelForecastAgent([forecast_for("M1", T0, quote.mid)])
    run = evaluate_policy(agent, [state], policy(), FILLS)
    reasons = {d.skip.reason for d in run.decisions if d.skip}
    assert SkipReason.INSUFFICIENT_EDGE in reasons


def test_a_tradable_forecast_is_not_skipped():
    from research.policy import ModelForecastAgent, evaluate_policy

    state = make_state(quote=make_quote(bid=0.40, ask=0.44, ts=T0), as_of=T0)
    agent = ModelForecastAgent([forecast_for("M1", T0, 0.95)])
    run = evaluate_policy(agent, [state], policy(), FILLS)
    assert any(d.skip is None and d.assessment and d.assessment.is_tradable
               for d in run.decisions)


def test_every_decision_is_accounted_for_either_way():
    """No decision may vanish: it is a trade candidate or a recorded skip."""
    from research.policy import ModelForecastAgent, evaluate_policy

    states = [
        make_state(quote=make_quote(bid=0.40, ask=0.44, ts=T0), as_of=T0),
        make_state(quote=None, as_of=T0),
        make_state(quote=make_quote(bid=0.40, ask=0.44, ts=T0),
                   as_of=T0 + timedelta(seconds=9000)),
    ]
    agent = ModelForecastAgent([forecast_for("M1", s.as_of, 0.95) for s in states])
    run = evaluate_policy(agent, states, policy(), FILLS)
    assert len(run.decisions) == len(states)
    for decision in run.decisions:
        assert (decision.skip is not None) != (
            decision.assessment is not None and decision.assessment.is_tradable
        ) or decision.skip is not None
