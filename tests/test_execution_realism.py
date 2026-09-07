"""Execution realism: bid/ask crossing, partial fills, staleness, costs, delay.

These tests exist because of an audit of the engine that found three defects:
the order executed against the very book the agents had just analysed (zero
latency), the limit price was computed but never enforced, and the fill
simulator had no staleness guard of its own. Each is pinned here.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from backtest.engine import BacktestEngine, run_backtest
from backtest.fills import compute_fee, estimate_slippage, simulate_fill
from core.config import load_settings
from core.models import (
    EventKind,
    Market,
    MarketStatus,
    Outcome,
    Quote,
    Side,
    Trade,
)
from tests.conftest import T0, make_market, make_quote

NO_COST = {
    "base_slippage": 0.0,
    "impact_coefficient": 0.0,
    "fee_rate": 0.0,
    "max_quote_age_seconds": None,
}
COSTED = {
    "base_slippage": 0.002,
    "impact_coefficient": 0.010,
    "allow_partial_fills": True,
    "max_fraction_of_displayed": 1.0,
    "min_fill_size": 1.0,
    "fee_rate": 0.02,
    "fee_model": "kalshi",
    "tick_size": 0.01,
    "max_quote_age_seconds": 900.0,
    "enforce_limit_price": True,
}


# -- bid/ask execution -----------------------------------------------------


def test_a_buy_lifts_the_ask_never_the_mid_or_the_bid():
    quote = make_quote(bid=0.40, ask=0.46, ask_size=1000.0)
    fill = simulate_fill(quote, Side.BUY, 10.0, T0, NO_COST, "P1")
    assert fill.reference_price == pytest.approx(0.46)
    assert fill.avg_price == pytest.approx(0.46)
    assert fill.avg_price > quote.mid


def test_a_sell_hits_the_bid_never_the_mid_or_the_ask():
    quote = make_quote(bid=0.40, ask=0.46, bid_size=1000.0)
    fill = simulate_fill(quote, Side.SELL, 10.0, T0, NO_COST, "P1")
    assert fill.reference_price == pytest.approx(0.40)
    assert fill.avg_price < quote.mid


def test_execution_is_never_better_than_the_touch():
    """Slippage is signed against the taker; a fill can never improve on it."""
    quote = make_quote(bid=0.40, ask=0.46, bid_size=500.0, ask_size=500.0)
    buy = simulate_fill(quote, Side.BUY, 100.0, T0, COSTED, "P1")
    sell = simulate_fill(quote, Side.SELL, 100.0, T0, COSTED, "P2")
    assert buy.avg_price >= quote.ask
    assert sell.avg_price <= quote.bid


def test_crossing_the_spread_costs_the_spread():
    """Buying then immediately selling the same size must lose money."""
    quote = make_quote(bid=0.40, ask=0.46, bid_size=1000.0, ask_size=1000.0)
    buy = simulate_fill(quote, Side.BUY, 100.0, T0, NO_COST, "P1")
    sell = simulate_fill(quote, Side.SELL, 100.0, T0, NO_COST, "P2")
    assert sell.avg_price < buy.avg_price
    assert buy.cash_flow + sell.cash_flow < 0


def test_a_missing_side_cannot_be_traded():
    assert (
        simulate_fill(make_quote(bid=0.40, ask=None), Side.BUY, 10.0, T0, COSTED).unfilled_reason
        == "NO_PRICE_ON_SIDE"
    )
    assert (
        simulate_fill(make_quote(bid=None, ask=0.46), Side.SELL, 10.0, T0, COSTED).unfilled_reason
        == "NO_PRICE_ON_SIDE"
    )


# -- partial fills ---------------------------------------------------------


def test_an_order_cannot_take_more_than_the_displayed_size():
    fill = simulate_fill(
        make_quote(bid=0.40, ask=0.46, ask_size=60.0), Side.BUY, 500.0, T0, COSTED, "P1"
    )
    assert fill.filled_size == pytest.approx(60.0)
    assert fill.requested_size == pytest.approx(500.0)
    assert fill.is_partial is True


def test_a_fully_filled_order_is_not_marked_partial():
    fill = simulate_fill(
        make_quote(bid=0.40, ask=0.46, ask_size=600.0), Side.BUY, 500.0, T0, COSTED, "P1"
    )
    assert fill.is_partial is False


def test_the_displayed_fraction_caps_how_much_is_takeable():
    """Taking 100% of the touch is optimistic; the cap makes that tunable."""
    config = dict(COSTED, max_fraction_of_displayed=0.25)
    fill = simulate_fill(
        make_quote(bid=0.40, ask=0.46, ask_size=400.0), Side.BUY, 400.0, T0, config, "P1"
    )
    assert fill.filled_size == pytest.approx(100.0)


def test_partial_fills_can_be_forbidden_entirely():
    config = dict(COSTED, allow_partial_fills=False)
    fill = simulate_fill(
        make_quote(bid=0.40, ask=0.46, ask_size=60.0), Side.BUY, 500.0, T0, config, "P1"
    )
    assert fill.filled_size == 0.0
    assert fill.unfilled_reason == "PARTIAL_FILLS_DISABLED"


def test_a_partial_fill_only_pays_fees_on_what_traded():
    fill = simulate_fill(
        make_quote(bid=0.40, ask=0.46, ask_size=60.0), Side.BUY, 500.0, T0, COSTED, "P1"
    )
    assert fill.fee == pytest.approx(
        compute_fee(fill.avg_price, 60.0, COSTED["fee_rate"], COSTED["fee_model"])
    )


# -- stale quotes ----------------------------------------------------------


def test_execution_refuses_a_book_older_than_the_limit():
    """Independent of the risk agent: a frozen feed is not a tradable price."""
    quote = make_quote(bid=0.40, ask=0.46, ask_size=500.0, ts=T0)
    fill = simulate_fill(
        quote, Side.BUY, 10.0, T0 + timedelta(seconds=1800), COSTED, "P1"
    )
    assert fill.filled_size == 0.0
    assert fill.unfilled_reason == "STALE_QUOTE"


def test_a_book_inside_the_age_limit_still_trades():
    quote = make_quote(bid=0.40, ask=0.46, ask_size=500.0, ts=T0)
    fill = simulate_fill(
        quote, Side.BUY, 10.0, T0 + timedelta(seconds=900), COSTED, "P1"
    )
    assert fill.filled_size > 0.0


def test_the_staleness_guard_can_be_disabled_for_experiments():
    config = dict(COSTED, max_quote_age_seconds=None)
    fill = simulate_fill(
        make_quote(bid=0.40, ask=0.46, ask_size=500.0, ts=T0),
        Side.BUY,
        10.0,
        T0 + timedelta(days=7),
        config,
        "P1",
    )
    assert fill.filled_size > 0.0


# -- fees ------------------------------------------------------------------


def test_fees_are_charged_on_the_executed_price_not_the_quoted_one():
    quote = make_quote(bid=0.40, ask=0.46, ask_size=200.0)
    fill = simulate_fill(quote, Side.BUY, 200.0, T0, COSTED, "P1")
    assert fill.avg_price > 0.46
    assert fill.fee == pytest.approx(compute_fee(fill.avg_price, 200.0, 0.02, "kalshi"))


def test_fees_always_reduce_the_cash_received_or_increase_the_cash_paid():
    quote = make_quote(bid=0.40, ask=0.46, bid_size=500.0, ask_size=500.0)
    buy = simulate_fill(quote, Side.BUY, 100.0, T0, COSTED, "P1")
    sell = simulate_fill(quote, Side.SELL, 100.0, T0, COSTED, "P2")
    assert buy.cash_flow == pytest.approx(-buy.notional - buy.fee)
    assert sell.cash_flow == pytest.approx(sell.notional - sell.fee)
    assert buy.fee > 0 and sell.fee > 0


def test_the_two_fee_models_differ_and_both_are_non_negative():
    quote = make_quote(bid=0.40, ask=0.46, ask_size=500.0)
    kalshi = simulate_fill(quote, Side.BUY, 100.0, T0, dict(COSTED, fee_model="kalshi"), "P1")
    notional = simulate_fill(
        quote, Side.BUY, 100.0, T0, dict(COSTED, fee_model="notional"), "P2"
    )
    assert kalshi.fee >= 0 and notional.fee >= 0
    assert kalshi.fee != notional.fee


# -- slippage --------------------------------------------------------------


def test_slippage_grows_with_the_share_of_depth_consumed():
    small = simulate_fill(
        make_quote(bid=0.40, ask=0.46, ask_size=10_000.0), Side.BUY, 100.0, T0, COSTED, "P1"
    )
    large = simulate_fill(
        make_quote(bid=0.40, ask=0.46, ask_size=100.0), Side.BUY, 100.0, T0, COSTED, "P2"
    )
    assert small.slippage < large.slippage


def test_slippage_matches_the_documented_model():
    quote = make_quote(bid=0.40, ask=0.46, ask_size=400.0)
    fill = simulate_fill(quote, Side.BUY, 100.0, T0, COSTED, "P1")
    expected = estimate_slippage(100.0, 400.0, 0.002, 0.010)
    assert fill.slippage == pytest.approx(expected)
    assert fill.avg_price == pytest.approx(0.46 + expected)


def test_slippage_is_never_favourable():
    quote = make_quote(bid=0.40, ask=0.46, bid_size=50.0, ask_size=50.0)
    for side in (Side.BUY, Side.SELL):
        fill = simulate_fill(quote, side, 5_000.0, T0, COSTED, "P1")
        assert fill.slippage >= 0.0


def test_execution_price_is_clamped_inside_the_tradable_range():
    quote = make_quote(bid=0.98, ask=0.99, ask_size=10.0)
    fill = simulate_fill(quote, Side.BUY, 10.0, T0, dict(COSTED, enforce_limit_price=False), "P1")
    assert fill.avg_price <= 1.0 - COSTED["tick_size"]


# -- limit price -----------------------------------------------------------


def test_a_marketable_limit_order_never_fills_worse_than_its_limit():
    """The audit's second finding: the limit was computed but never enforced."""
    quote = make_quote(bid=0.40, ask=0.50, ask_size=500.0)
    fill = simulate_fill(quote, Side.BUY, 100.0, T0, COSTED, "P1", limit_price=0.45)
    assert fill.filled_size == 0.0
    assert fill.unfilled_reason == "LIMIT_NOT_MARKETABLE"


def test_a_sell_limit_is_enforced_on_the_other_side():
    quote = make_quote(bid=0.30, ask=0.50, bid_size=500.0)
    fill = simulate_fill(quote, Side.SELL, 100.0, T0, COSTED, "P1", limit_price=0.40)
    assert fill.unfilled_reason == "LIMIT_NOT_MARKETABLE"


def test_a_limit_that_covers_the_slippage_still_fills():
    quote = make_quote(bid=0.40, ask=0.46, ask_size=500.0)
    fill = simulate_fill(quote, Side.BUY, 100.0, T0, COSTED, "P1", limit_price=0.50)
    assert fill.filled_size == pytest.approx(100.0)
    assert fill.avg_price <= 0.50


# -- delayed execution -----------------------------------------------------


def rolling_market(close_hours: int = 6) -> Market:
    """A market that stays open long enough for delayed orders to land."""
    return make_market("M1", outcome=Outcome.YES, close_time=T0 + timedelta(hours=close_hours))


def settings_with(delay: float, **backtest: object) -> dict:
    """Project settings with an execution delay and easy trading conditions."""
    settings = load_settings(use_env=False)
    settings["backtest"].update(
        {
            "initial_capital": 10_000.0,
            "decision_interval_seconds": 600,
            "execution_delay_seconds": delay,
            **backtest,
        }
    )
    settings["strategy"].update({"min_edge": 0.0, "require_positive_ev": False})
    settings["risk"].update({"max_data_age_seconds": 86_400, "max_spread": 1.0})
    settings["fills"].update({"max_quote_age_seconds": 86_400})
    return settings


def moving_book(market_id: str = "M1") -> list[Quote]:
    """A book that climbs at every snapshot, so any delay changes the price."""
    return [
        Quote(
            market_id=market_id,
            ts=T0 + timedelta(minutes=i * 10),
            bid=round(0.20 + 0.08 * i, 4),
            ask=round(0.22 + 0.08 * i, 4),
            bid_size=1_000.0,
            ask_size=1_000.0,
        )
        for i in range(6)
    ]


def tape(market_id: str = "M1") -> list[Trade]:
    """Enough prints for the tape agent to form a view."""
    return [
        Trade(
            trade_id=f"T{i}",
            market_id=market_id,
            ts=T0 + timedelta(minutes=i),
            price=0.80,
            size=25.0,
            aggressor=Side.BUY,
        )
        for i in range(10)
    ]


def test_every_fill_happens_strictly_after_the_decision_that_caused_it():
    """The audit's first finding: execution used to be simultaneous."""
    settings = settings_with(60.0)
    result = run_backtest([rolling_market()], moving_book(), tape(), settings)
    assert result.executed_fills, "expected at least one fill"
    for fill in result.fills:
        assert fill.decided_at is not None
        assert fill.ts > fill.decided_at
        assert fill.latency_seconds == pytest.approx(60.0)


def test_a_longer_delay_changes_the_book_the_order_executes_against():
    """With a delay past the next update, the order fills at the new price."""
    quick = run_backtest([rolling_market()], moving_book(), tape(), settings_with(60.0))
    slow = run_backtest([rolling_market()], moving_book(), tape(), settings_with(900.0))
    quick_first = quick.executed_fills[0]
    slow_first = slow.executed_fills[0]
    assert slow_first.latency_seconds > quick_first.latency_seconds
    assert slow_first.avg_price != quick_first.avg_price


def test_a_delayed_order_never_sees_a_quote_published_after_it_arrived():
    """No look-ahead: the executing book must predate the execution instant."""
    settings = settings_with(60.0)
    quotes = moving_book()
    result = run_backtest([rolling_market()], quotes, tape(), settings)
    for fill in result.executed_fills:
        visible = [q for q in quotes if q.ts < fill.ts]
        assert visible, "a fill happened before any quote existed"
        assert fill.reference_price == pytest.approx(visible[-1].ask)


def test_an_order_in_flight_when_the_market_closes_is_cancelled():
    market = make_market("M1", outcome=Outcome.YES, close_time=T0 + timedelta(minutes=25))
    quotes = [
        Quote(
            market_id="M1",
            ts=T0 + timedelta(minutes=i * 10),
            bid=0.20,
            ask=0.22,
            bid_size=1_000.0,
            ask_size=1_000.0,
        )
        for i in range(3)
    ]
    settings = settings_with(3_600.0)
    result = run_backtest([market], quotes, tape(), settings)
    assert not result.executed_fills
    cancelled = [
        e
        for e in result.audit_log
        if e.kind is EventKind.REJECTION
        and e.payload.get("reason") == "MARKET_CLOSED_IN_FLIGHT"
    ]
    assert cancelled


def test_zero_delay_is_still_supported_for_comparison():
    settings = settings_with(0.0)
    result = run_backtest([rolling_market()], moving_book(), tape(), settings)
    for fill in result.executed_fills:
        assert fill.latency_seconds == pytest.approx(0.0)


def test_delayed_execution_does_not_reorder_the_event_stream():
    settings = settings_with(60.0)
    result = run_backtest([rolling_market()], moving_book(), tape(), settings)
    timestamps = [s.ts for s in result.snapshots]
    assert timestamps == sorted(timestamps)


# -- settlement ------------------------------------------------------------


def test_a_long_position_settles_at_one_dollar_on_yes():
    settings = settings_with(60.0)
    market = make_market("M1", outcome=Outcome.YES, close_time=T0 + timedelta(hours=6))
    result = run_backtest([market], moving_book(), tape(), settings)
    settlements = [s for s in result.settlements if s.market_id == "M1"]
    assert settlements
    settlement = settlements[0]
    assert settlement.payout_per_contract == 1.0
    assert settlement.cash_flow == pytest.approx(settlement.position * 1.0)


def test_a_position_settles_at_zero_on_no():
    settings = settings_with(60.0)
    market = make_market("M1", outcome=Outcome.NO, close_time=T0 + timedelta(hours=6))
    result = run_backtest([market], moving_book(), tape(), settings)
    for settlement in result.settlements:
        assert settlement.payout_per_contract == 0.0
        assert settlement.cash_flow == pytest.approx(0.0)


def test_settlement_happens_at_the_resolution_timestamp():
    settings = settings_with(60.0)
    market = make_market("M1", outcome=Outcome.YES, close_time=T0 + timedelta(hours=6))
    result = run_backtest([market], moving_book(), tape(), settings)
    for settlement in result.settlements:
        assert settlement.ts == market.resolution_time


def test_no_position_survives_settlement():
    settings = settings_with(60.0)
    market = make_market("M1", outcome=Outcome.YES, close_time=T0 + timedelta(hours=6))
    result = run_backtest([market], moving_book(), tape(), settings)
    assert result.final_equity == pytest.approx(result.snapshots[-1].equity)
    assert result.snapshots[-1].open_positions == 0


def test_nothing_trades_after_a_market_resolves():
    settings = settings_with(60.0)
    market = make_market("M1", outcome=Outcome.YES, close_time=T0 + timedelta(hours=6))
    result = run_backtest([market], moving_book(), tape(), settings)
    assert market.resolution_time is not None
    for fill in result.fills:
        assert fill.ts < market.resolution_time


def test_the_engine_refuses_a_market_marked_resolved_without_an_outcome():
    with pytest.raises(ValueError):
        Market(
            market_id="M1",
            question="[DEMO DATA] broken",
            created_at=T0,
            close_time=T0 + timedelta(hours=1),
            status=MarketStatus.RESOLVED,
            outcome=None,
        )


# -- end-to-end audit invariants ------------------------------------------


def test_the_default_configuration_has_a_non_zero_execution_delay():
    """Guards against the zero-latency default silently coming back."""
    settings = load_settings(use_env=False)
    assert float(settings["backtest"]["execution_delay_seconds"]) > 0.0


def test_the_engine_reads_the_delay_from_configuration():
    engine = BacktestEngine(settings_with(120.0))
    assert engine.execution_delay == timedelta(seconds=120.0)


def test_every_executed_fill_traces_to_an_earlier_approved_decision(demo_result):
    approved = {
        d.proposal_id: (d.ts, d.approved_size)
        for d in demo_result.decisions
        if not d.is_blocked
    }
    for fill in demo_result.executed_fills:
        decided_ts, approved_size = approved[fill.proposal_id]
        assert fill.ts > decided_ts
        assert fill.filled_size <= approved_size + 1e-9
