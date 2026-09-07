"""Risk vetoes, sizing limits and stale-data rejection."""

from __future__ import annotations

from datetime import timedelta

import pytest

from agents.risk import RiskAgent
from core.models import MarketStatus, RiskAction, RiskReasonCode, Side
from tests.conftest import T0, make_market, make_proposal, make_quote, make_state

LIMITS = {
    "kill_switch": False,
    "max_position_size": 300.0,
    "max_event_exposure": 400.0,
    "max_total_exposure": 1200.0,
    "max_spread": 0.06,
    "max_slippage": 0.02,
    "max_data_age_seconds": 900.0,
    "max_drawdown": 0.25,
    "min_order_size": 10.0,
}


def agent(**overrides) -> RiskAgent:
    """Build a risk agent from the shared limits plus overrides."""
    limits = dict(LIMITS)
    limits.update(overrides)
    return RiskAgent(limits)


def test_a_clean_proposal_is_approved_in_full():
    decision = agent().evaluate(make_proposal(size=100.0), make_state())
    assert decision.action is RiskAction.APPROVE
    assert decision.approved_size == pytest.approx(100.0)
    assert decision.reason_codes == [RiskReasonCode.OK]


def test_kill_switch_rejects_everything():
    decision = agent(kill_switch=True).evaluate(make_proposal(), make_state())
    assert decision.action is RiskAction.REJECT
    assert decision.approved_size == 0.0
    assert RiskReasonCode.KILL_SWITCH in decision.reason_codes


def test_stale_data_is_rejected():
    quote = make_quote(ts=T0)
    state = make_state(quote=quote, as_of=T0 + timedelta(seconds=1800))
    decision = agent(max_data_age_seconds=900).evaluate(make_proposal(), state)
    assert decision.action is RiskAction.REJECT
    assert RiskReasonCode.STALE_DATA in decision.reason_codes
    stale = next(r for r in decision.reasons if r.code is RiskReasonCode.STALE_DATA)
    assert stale.observed == pytest.approx(1800.0)
    assert stale.limit == pytest.approx(900.0)


def test_fresh_data_just_inside_the_limit_is_allowed():
    state = make_state(quote=make_quote(ts=T0), as_of=T0 + timedelta(seconds=900))
    decision = agent(max_data_age_seconds=900).evaluate(make_proposal(), state)
    assert decision.action is RiskAction.APPROVE


def test_wide_spread_is_rejected():
    state = make_state(quote=make_quote(bid=0.30, ask=0.45))
    decision = agent(max_spread=0.06).evaluate(make_proposal(reference_price=0.45), state)
    assert decision.action is RiskAction.REJECT
    assert RiskReasonCode.SPREAD_TOO_WIDE in decision.reason_codes


def test_excessive_estimated_slippage_is_rejected():
    decision = agent(max_slippage=0.01).evaluate(
        make_proposal(expected_slippage=0.05), make_state()
    )
    assert decision.action is RiskAction.REJECT
    assert RiskReasonCode.SLIPPAGE_TOO_HIGH in decision.reason_codes


def test_drawdown_breach_halts_trading():
    state = make_state(equity=7_000.0, peak_equity=10_000.0)
    decision = agent(max_drawdown=0.25).evaluate(make_proposal(), state)
    assert decision.action is RiskAction.REJECT
    assert RiskReasonCode.MAX_DRAWDOWN_BREACHED in decision.reason_codes


def test_drawdown_inside_the_limit_is_allowed():
    state = make_state(equity=9_000.0, peak_equity=10_000.0)
    assert agent(max_drawdown=0.25).evaluate(make_proposal(), state).action is RiskAction.APPROVE


def test_missing_side_of_the_book_is_rejected():
    state = make_state(quote=make_quote(bid=0.40, ask=None))
    decision = agent().evaluate(make_proposal(side=Side.BUY), state)
    assert decision.action is RiskAction.REJECT
    assert RiskReasonCode.NO_QUOTE in decision.reason_codes


def test_closed_market_is_rejected():
    market = make_market(status=MarketStatus.CLOSED)
    decision = agent().evaluate(make_proposal(), make_state(market=market))
    assert decision.action is RiskAction.REJECT
    assert RiskReasonCode.MARKET_NOT_OPEN in decision.reason_codes


def test_position_limit_reduces_the_size():
    # Already long 250 of a 300 limit: only 50 more contracts may be bought.
    state = make_state(position_qty=250.0, avg_price=0.40)
    decision = agent(max_position_size=300.0).evaluate(make_proposal(size=200.0), state)
    assert decision.action is RiskAction.REDUCE
    assert decision.approved_size == pytest.approx(50.0)
    assert RiskReasonCode.POSITION_LIMIT in decision.reason_codes


def test_position_limit_rejects_when_no_room_is_left():
    state = make_state(position_qty=300.0, avg_price=0.40)
    decision = agent(max_position_size=300.0).evaluate(make_proposal(size=100.0), state)
    assert decision.action is RiskAction.REJECT
    assert decision.approved_size == 0.0
    assert RiskReasonCode.SIZE_BELOW_MINIMUM in decision.reason_codes


def test_selling_is_limited_by_the_short_side_of_the_position_limit():
    state = make_state(position_qty=-250.0, avg_price=0.40)
    proposal = make_proposal(side=Side.SELL, size=200.0, reference_price=0.40)
    decision = agent(max_position_size=300.0).evaluate(proposal, state)
    assert decision.action is RiskAction.REDUCE
    assert decision.approved_size == pytest.approx(50.0)


def test_event_exposure_limit_caps_the_size():
    # Worst-case loss on a long is size * price: 400 / 0.40 = 1000 contracts.
    proposal = make_proposal(size=2_000.0, reference_price=0.40)
    decision = agent(max_position_size=10_000.0, max_event_exposure=400.0).evaluate(
        proposal, make_state()
    )
    assert decision.action is RiskAction.REDUCE
    assert decision.approved_size == pytest.approx(1_000.0)
    assert RiskReasonCode.EVENT_EXPOSURE_LIMIT in decision.reason_codes


def test_event_exposure_limit_for_a_short_uses_one_minus_price():
    # Worst-case loss on a short is size * (1 - price): 400 / 0.60 contracts.
    proposal = make_proposal(side=Side.SELL, size=2_000.0, reference_price=0.40)
    decision = agent(max_position_size=10_000.0, max_event_exposure=400.0).evaluate(
        proposal, make_state()
    )
    assert decision.action is RiskAction.REDUCE
    assert decision.approved_size == pytest.approx(400.0 / 0.60)


def test_total_exposure_limit_accounts_for_other_markets():
    # 900 of the 1000 book-wide budget is used elsewhere, leaving 100 => 250
    # contracts at a price of 0.40.
    state = make_state(total_exposure=900.0, event_exposure=0.0)
    proposal = make_proposal(size=1_000.0, reference_price=0.40)
    decision = agent(
        max_position_size=10_000.0, max_event_exposure=10_000.0, max_total_exposure=1_000.0
    ).evaluate(proposal, state)
    assert decision.action is RiskAction.REDUCE
    assert decision.approved_size == pytest.approx(250.0)
    assert RiskReasonCode.TOTAL_EXPOSURE_LIMIT in decision.reason_codes


def test_insufficient_cash_caps_the_size():
    # 50 dollars of cash at 0.40 plus a 0.005 fee per contract.
    state = make_state(cash=50.0)
    proposal = make_proposal(size=1_000.0, reference_price=0.40, fee=0.005)
    decision = agent(max_position_size=10_000.0, max_event_exposure=10_000.0).evaluate(
        proposal, state
    )
    assert decision.action is RiskAction.REDUCE
    assert decision.approved_size == pytest.approx(50.0 / 0.405)
    assert RiskReasonCode.INSUFFICIENT_CASH in decision.reason_codes


def test_the_tightest_limit_binds():
    state = make_state(position_qty=200.0, avg_price=0.40, cash=10_000.0)
    proposal = make_proposal(size=500.0, reference_price=0.40)
    decision = agent(max_position_size=300.0, max_event_exposure=400.0).evaluate(
        proposal, state
    )
    # Position allows 100 more; event exposure would allow 800. Position binds.
    assert decision.approved_size == pytest.approx(100.0)


def test_risk_agent_analyze_requires_a_proposal_on_the_state():
    with pytest.raises(ValueError):
        agent().analyze(make_state())


def test_risk_agent_analyze_delegates_to_evaluate():
    state = make_state()
    state.proposal = make_proposal()
    assert agent().analyze(state).action is RiskAction.APPROVE


def test_risk_agent_has_no_order_placement_surface():
    """The risk agent may only describe a decision - never execute one."""
    forbidden = {"submit", "send", "place_order", "execute", "route", "cancel"}
    assert forbidden.isdisjoint(dir(RiskAgent))


def test_rounding_dust_does_not_produce_an_empty_reduce():
    """A cap a hair below the request is an approval, not a reasonless REDUCE."""
    state = make_state(cash=1_000_000.0)
    proposal = make_proposal(size=100.000000_4, reference_price=0.40)
    decision = agent(max_position_size=100.0).evaluate(proposal, state)
    assert decision.action is RiskAction.APPROVE
    assert decision.reason_codes == [RiskReasonCode.OK]
    assert decision.approved_size <= proposal.size


def test_every_reduce_carries_at_least_one_reason():
    state = make_state(position_qty=250.0, avg_price=0.40)
    decision = agent(max_position_size=300.0).evaluate(make_proposal(size=200.0), state)
    assert decision.action is RiskAction.REDUCE
    assert decision.reasons, "a REDUCE must explain itself"
