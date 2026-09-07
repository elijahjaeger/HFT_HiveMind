"""Fees, slippage, partial fills and execution against bid/ask."""

from __future__ import annotations

import pytest

from backtest.fills import compute_fee, estimate_slippage, fee_per_contract, simulate_fill
from core.models import Side
from tests.conftest import T0, make_quote


def test_notional_fee_is_a_share_of_traded_value():
    assert compute_fee(0.50, 100.0, fee_rate=0.02, fee_model="notional") == pytest.approx(1.0)


def test_kalshi_fee_peaks_at_a_coin_flip():
    at_half = compute_fee(0.50, 100.0, fee_rate=0.02, fee_model="kalshi")
    near_certainty = compute_fee(0.95, 100.0, fee_rate=0.02, fee_model="kalshi")
    assert at_half == pytest.approx(0.02 * 100 * 0.25)
    assert near_certainty < at_half


def test_fees_are_never_negative_and_zero_for_empty_orders():
    assert compute_fee(0.5, 0.0) == 0.0
    assert compute_fee(0.5, 100.0, fee_rate=0.0) == 0.0


def test_unknown_fee_model_is_rejected():
    with pytest.raises(ValueError):
        compute_fee(0.5, 10.0, fee_model="mystery")


def test_fee_per_contract_matches_a_single_contract_fill():
    assert fee_per_contract(0.40) == pytest.approx(compute_fee(0.40, 1.0))


def test_slippage_grows_with_participation():
    small = estimate_slippage(10.0, 1000.0, 0.002, 0.010)
    large = estimate_slippage(1000.0, 1000.0, 0.002, 0.010)
    assert small == pytest.approx(0.002 + 0.010 * 0.01)
    assert large == pytest.approx(0.012)
    assert small < large


def test_slippage_is_capped_at_full_participation():
    assert estimate_slippage(5000.0, 1000.0, 0.002, 0.010) == pytest.approx(0.012)


def test_slippage_of_an_empty_order_is_zero():
    assert estimate_slippage(0.0, 100.0) == 0.0


CONFIG = {
    "base_slippage": 0.002,
    "impact_coefficient": 0.010,
    "allow_partial_fills": True,
    "max_fraction_of_displayed": 1.0,
    "min_fill_size": 1.0,
    "fee_rate": 0.02,
    "fee_model": "kalshi",
    "tick_size": 0.01,
}


def test_a_buy_crosses_the_spread_and_pays_slippage():
    quote = make_quote(bid=0.40, ask=0.42, ask_size=1000.0)
    fill = simulate_fill(quote, Side.BUY, 100.0, T0, CONFIG, "P1")
    expected_slippage = estimate_slippage(100.0, 1000.0, 0.002, 0.010)
    assert fill.reference_price == pytest.approx(0.42)
    assert fill.avg_price == pytest.approx(0.42 + expected_slippage)
    assert fill.avg_price > quote.ask
    assert fill.filled_size == pytest.approx(100.0)


def test_a_sell_hits_the_bid_and_pays_slippage():
    quote = make_quote(bid=0.40, ask=0.42, bid_size=1000.0)
    fill = simulate_fill(quote, Side.SELL, 100.0, T0, CONFIG, "P1")
    assert fill.avg_price < quote.bid
    assert fill.slippage == pytest.approx(0.40 - fill.avg_price)


def test_the_fee_is_charged_on_the_executed_price_and_size():
    quote = make_quote(bid=0.40, ask=0.42, ask_size=1000.0)
    fill = simulate_fill(quote, Side.BUY, 100.0, T0, CONFIG, "P1")
    assert fill.fee == pytest.approx(compute_fee(fill.avg_price, 100.0, 0.02, "kalshi"))
    assert fill.fee > 0.0


def test_size_beyond_the_displayed_depth_fills_partially():
    quote = make_quote(bid=0.40, ask=0.42, ask_size=60.0)
    fill = simulate_fill(quote, Side.BUY, 200.0, T0, CONFIG, "P1")
    assert fill.filled_size == pytest.approx(60.0)
    assert fill.is_partial is True
    assert fill.requested_size == pytest.approx(200.0)


def test_partial_fills_can_be_disabled():
    quote = make_quote(bid=0.40, ask=0.42, ask_size=60.0)
    config = dict(CONFIG, allow_partial_fills=False)
    fill = simulate_fill(quote, Side.BUY, 200.0, T0, config, "P1")
    assert fill.filled_size == 0.0
    assert fill.unfilled_reason == "PARTIAL_FILLS_DISABLED"


def test_no_fill_without_a_price_on_the_required_side():
    quote = make_quote(bid=0.40, ask=None)
    fill = simulate_fill(quote, Side.BUY, 50.0, T0, CONFIG, "P1")
    assert fill.filled_size == 0.0
    assert fill.unfilled_reason == "NO_PRICE_ON_SIDE"
    assert fill.notional == 0.0


def test_no_fill_without_displayed_size():
    quote = make_quote(bid=0.40, ask=0.42, ask_size=0.0)
    fill = simulate_fill(quote, Side.BUY, 50.0, T0, CONFIG, "P1")
    assert fill.unfilled_reason == "NO_DISPLAYED_SIZE"


def test_no_fill_without_a_quote():
    fill = simulate_fill(None, Side.BUY, 50.0, T0, CONFIG, "P1")
    assert fill.filled_size == 0.0
    assert fill.unfilled_reason == "NO_QUOTE"


def test_execution_price_stays_inside_the_tick_bounds():
    quote = make_quote(bid=0.98, ask=0.99, ask_size=10.0)
    fill = simulate_fill(quote, Side.BUY, 10.0, T0, CONFIG, "P1")
    assert fill.avg_price <= 1.0 - CONFIG["tick_size"]


def test_cash_flow_is_negative_for_a_buy_and_positive_for_a_sell():
    quote = make_quote(bid=0.40, ask=0.42, bid_size=1000.0, ask_size=1000.0)
    buy = simulate_fill(quote, Side.BUY, 100.0, T0, CONFIG, "P1")
    sell = simulate_fill(quote, Side.SELL, 100.0, T0, CONFIG, "P2")
    assert buy.cash_flow == pytest.approx(-(buy.notional) - buy.fee)
    assert sell.cash_flow == pytest.approx(sell.notional - sell.fee)
    assert buy.cash_flow < 0 < sell.cash_flow
