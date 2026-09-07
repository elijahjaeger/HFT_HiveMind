"""Probability and midpoint calculations, and forecast blending."""

from __future__ import annotations

import pytest

from agents.base import combine_forecasts
from agents.baseline import BaseRateAgent
from agents.market import MarketAgent, implied_probability, midpoint
from agents.tape import TapeAgent, volume_weighted_price
from core.models import Forecast
from tests.conftest import T0, make_quote, make_state, make_trade


def test_midpoint_is_the_average_of_bid_and_ask():
    assert midpoint(0.40, 0.44) == pytest.approx(0.42)


@pytest.mark.parametrize("bid,ask", [(None, 0.5), (0.5, None), (None, None)])
def test_midpoint_requires_both_sides(bid, ask):
    assert midpoint(bid, ask) is None


def test_implied_probability_uses_the_midpoint_when_two_sided():
    assert implied_probability(make_quote(bid=0.30, ask=0.34)) == pytest.approx(0.32)


def test_implied_probability_adjusts_a_one_sided_book():
    bid_only = implied_probability(make_quote(bid=0.30, ask=None), 0.01)
    ask_only = implied_probability(make_quote(bid=None, ask=0.34), 0.01)
    assert bid_only == pytest.approx(0.31)
    assert ask_only == pytest.approx(0.33)


def test_implied_probability_is_none_without_a_book():
    assert implied_probability(None) is None
    assert implied_probability(make_quote(bid=None, ask=None)) is None


def test_market_agent_reports_the_implied_probability():
    agent = MarketAgent({"confidence": 0.6, "one_sided_adjustment": 0.01})
    forecast = agent.analyze(make_state(quote=make_quote(bid=0.20, ask=0.24)))
    assert forecast.probability == pytest.approx(0.22)
    assert forecast.is_placeholder is False
    assert forecast.confidence == pytest.approx(0.6)


def test_market_agent_halves_confidence_on_a_one_sided_book():
    agent = MarketAgent({"confidence": 0.6})
    forecast = agent.analyze(make_state(quote=make_quote(bid=0.20, ask=None)))
    assert forecast.confidence == pytest.approx(0.3)


def test_market_agent_returns_no_probability_without_a_quote():
    forecast = MarketAgent().analyze(make_state(quote=None))
    assert forecast.probability is None
    assert forecast.confidence == 0.0


def test_base_rate_agent_is_marked_as_a_placeholder():
    agent = BaseRateAgent({"placeholder_probability": 0.5})
    forecast = agent.analyze(make_state())
    assert forecast.is_placeholder is True
    assert forecast.probability == pytest.approx(0.5)
    assert "PLACEHOLDER" in forecast.rationale


def test_vwap_is_size_weighted():
    trades = [
        make_trade(price=0.40, size=10.0, trade_id="a"),
        make_trade(price=0.60, size=30.0, trade_id="b"),
    ]
    assert volume_weighted_price(trades) == pytest.approx(0.55)


def test_vwap_is_none_without_prints():
    assert volume_weighted_price([]) is None


def test_tape_agent_needs_a_minimum_number_of_prints():
    agent = TapeAgent({"min_trades": 5})
    forecast = agent.analyze(make_state(trades=[make_trade()]))
    assert forecast.probability is None


def test_tape_agent_blends_vwap_with_the_midpoint():
    trades = [
        make_trade(price=0.60, size=10.0, trade_id=f"t{i}") for i in range(5)
    ]
    agent = TapeAgent({"min_trades": 5, "tape_weight": 0.5})
    state = make_state(quote=make_quote(bid=0.38, ask=0.42), trades=trades)
    # 0.5 * VWAP(0.60) + 0.5 * mid(0.40)
    assert agent.analyze(state).probability == pytest.approx(0.50)


def _forecast(name: str, probability: float | None, placeholder: bool = False) -> Forecast:
    return Forecast(
        agent_name=name,
        market_id="M1",
        ts=T0,
        probability=probability,
        is_placeholder=placeholder,
    )


def test_combine_forecasts_excludes_placeholders_by_default():
    consensus = combine_forecasts(
        [_forecast("MarketAgent", 0.40), _forecast("BaseRateAgent", 0.50, True)],
        {"include_placeholder": False},
    )
    assert consensus.probability == pytest.approx(0.40)
    assert "BaseRateAgent" in consensus.excluded


def test_combine_forecasts_applies_weights_and_measures_disagreement():
    consensus = combine_forecasts(
        [_forecast("MarketAgent", 0.40), _forecast("TapeAgent", 0.60)],
        {"weights": {"MarketAgent": 0.25, "TapeAgent": 0.75}},
    )
    assert consensus.probability == pytest.approx(0.55)
    assert consensus.disagreement == pytest.approx(0.20)
    assert consensus.n_contributors == 2


def test_combine_forecasts_has_no_view_when_nothing_qualifies():
    consensus = combine_forecasts([_forecast("MarketAgent", None)], {})
    assert consensus.has_view is False
    assert consensus.probability is None
