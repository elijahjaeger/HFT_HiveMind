"""Spread, depth and stale-data calculations."""

from __future__ import annotations

from datetime import timedelta

import pytest

from agents.liquidity import (
    LiquidityAgent,
    compute_depth,
    compute_spread,
    data_age_seconds,
    is_stale,
)
from tests.conftest import T0, make_quote, make_state


def test_spread_is_the_bid_ask_width():
    assert compute_spread(make_quote(bid=0.40, ask=0.45)) == pytest.approx(0.05)


def test_spread_is_none_on_a_one_sided_book():
    assert compute_spread(make_quote(bid=0.40, ask=None)) is None
    assert compute_spread(None) is None


def test_depth_is_the_smaller_displayed_side():
    quote = make_quote(bid_size=120.0, ask_size=80.0)
    assert compute_depth(quote) == pytest.approx(80.0)


def test_depth_of_a_one_sided_book_is_the_visible_side():
    assert compute_depth(make_quote(bid=None, ask=0.42, ask_size=70.0)) == pytest.approx(70.0)
    assert compute_depth(None) == 0.0


def test_data_age_counts_seconds_since_the_quote():
    quote = make_quote(ts=T0)
    assert data_age_seconds(quote, T0 + timedelta(seconds=90)) == pytest.approx(90.0)


def test_data_age_never_goes_negative():
    quote = make_quote(ts=T0 + timedelta(seconds=30))
    assert data_age_seconds(quote, T0) == 0.0


def test_is_stale_compares_age_to_the_limit():
    quote = make_quote(ts=T0)
    assert is_stale(quote, T0 + timedelta(seconds=100), 900) is False
    assert is_stale(quote, T0 + timedelta(seconds=1000), 900) is True
    assert is_stale(None, T0, 900) is True


def test_liquidity_agent_reports_a_healthy_book_without_warnings():
    agent = LiquidityAgent({"max_data_age_seconds": 900, "warn_spread": 0.04, "warn_depth": 50})
    report = agent.analyze(make_state(quote=make_quote(bid=0.40, ask=0.42)))
    assert report.spread == pytest.approx(0.02)
    assert report.depth == pytest.approx(500.0)
    assert report.is_stale is False
    assert report.warnings == []


def test_liquidity_agent_warns_on_wide_spread_thin_depth_and_stale_data():
    agent = LiquidityAgent({"max_data_age_seconds": 60, "warn_spread": 0.03, "warn_depth": 100})
    quote = make_quote(bid=0.30, ask=0.40, bid_size=10.0, ask_size=10.0, ts=T0)
    report = agent.analyze(make_state(quote=quote, as_of=T0 + timedelta(seconds=600)))
    codes = " ".join(report.warnings)
    assert "WIDE_SPREAD" in codes
    assert "THIN_DEPTH" in codes
    assert "STALE_DATA" in codes
    assert report.is_stale is True


def test_liquidity_agent_flags_a_missing_book():
    report = LiquidityAgent().analyze(make_state(quote=None))
    assert report.is_two_sided is False
    assert any(w.startswith("NO_QUOTE") for w in report.warnings)
