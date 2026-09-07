"""Proposal construction, expected value after costs, and configuration."""

from __future__ import annotations

from pathlib import Path

import pytest

from agents.base import Consensus
from agents.strategy import build_proposal, expected_value_per_contract
from core.config import PROJECT_ROOT, load_settings, resolve_path, section
from core.models import Side
from tests.conftest import make_quote, make_state

FILLS = {
    "base_slippage": 0.0,
    "impact_coefficient": 0.0,
    "fee_rate": 0.0,
    "fee_model": "kalshi",
    "tick_size": 0.01,
}
STRATEGY = {
    "min_edge": 0.03,
    "base_size": 100.0,
    "edge_size_multiplier": 0.0,
    "max_order_size": 400.0,
    "min_order_size": 10.0,
    "require_positive_ev": True,
}


def consensus(probability: float | None) -> Consensus:
    return Consensus(market_id="M1", probability=probability)


def test_expected_value_of_a_buy_is_fair_minus_price_and_fees():
    assert expected_value_per_contract(Side.BUY, 0.60, 0.50, 0.01) == pytest.approx(0.09)


def test_expected_value_of_a_sell_is_price_minus_fair_and_fees():
    assert expected_value_per_contract(Side.SELL, 0.40, 0.50, 0.01) == pytest.approx(0.09)


def test_a_cheap_ask_produces_a_buy_proposal():
    state = make_state(quote=make_quote(bid=0.38, ask=0.40))
    proposal = build_proposal(state, consensus(0.50), "P1", STRATEGY, FILLS)
    assert proposal is not None
    assert proposal.side is Side.BUY
    assert proposal.reference_price == pytest.approx(0.40)
    assert proposal.edge == pytest.approx(0.10)
    assert proposal.expected_value_per_contract == pytest.approx(0.10)
    assert proposal.expected_value_total == pytest.approx(0.10 * proposal.size)


def test_a_rich_bid_produces_a_sell_proposal():
    state = make_state(quote=make_quote(bid=0.60, ask=0.62))
    proposal = build_proposal(state, consensus(0.50), "P1", STRATEGY, FILLS)
    assert proposal is not None
    assert proposal.side is Side.SELL
    assert proposal.reference_price == pytest.approx(0.60)


def test_no_proposal_when_the_edge_is_inside_the_spread():
    state = make_state(quote=make_quote(bid=0.48, ask=0.52))
    assert build_proposal(state, consensus(0.50), "P1", STRATEGY, FILLS) is None


def test_no_proposal_without_a_consensus_view():
    assert build_proposal(make_state(), consensus(None), "P1", STRATEGY, FILLS) is None


def test_no_proposal_without_a_book():
    assert build_proposal(make_state(quote=None), consensus(0.9), "P1", STRATEGY, FILLS) is None


def test_a_market_mid_can_never_beat_the_price_it_would_cross():
    """Sanity check on the arithmetic: crossing the spread costs the spread."""
    quote = make_quote(bid=0.40, ask=0.42)
    state = make_state(quote=quote)
    assert build_proposal(state, consensus(quote.mid), "P1", STRATEGY, FILLS) is None


def test_costs_can_veto_a_marginal_proposal():
    state = make_state(quote=make_quote(bid=0.38, ask=0.40))
    expensive = dict(FILLS, base_slippage=0.05, fee_rate=0.10, fee_model="notional")
    assert build_proposal(state, consensus(0.44), "P1", STRATEGY, expensive) is None


def test_negative_expected_value_can_be_allowed_explicitly():
    state = make_state(quote=make_quote(bid=0.38, ask=0.40))
    expensive = dict(FILLS, base_slippage=0.05, fee_rate=0.10, fee_model="notional")
    allowed = dict(STRATEGY, require_positive_ev=False)
    proposal = build_proposal(state, consensus(0.44), "P1", allowed, expensive)
    assert proposal is not None
    assert proposal.expected_value_per_contract < 0


def test_size_scales_with_the_edge_and_respects_the_cap():
    state = make_state(quote=make_quote(bid=0.38, ask=0.40))
    scaling = dict(STRATEGY, edge_size_multiplier=1_000.0, max_order_size=150.0)
    proposal = build_proposal(state, consensus(0.50), "P1", scaling, FILLS)
    assert proposal is not None
    assert proposal.size == pytest.approx(150.0)


def test_settings_load_from_the_project_yaml():
    settings = load_settings(use_env=False)
    for name in ("app", "data", "agents", "strategy", "fills", "risk", "backtest"):
        assert section(settings, name), f"missing settings section: {name}"


def test_live_trading_is_always_disabled(tmp_path: Path):
    rogue = tmp_path / "settings.yaml"
    rogue.write_text("app:\n  live_trading_enabled: true\n", encoding="utf-8")
    assert load_settings(rogue, use_env=False)["app"]["live_trading_enabled"] is False


def test_paths_resolve_against_the_project_root():
    assert resolve_path("data/processed") == PROJECT_ROOT / "data" / "processed"
    assert resolve_path(PROJECT_ROOT).is_absolute()
