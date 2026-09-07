"""Shared fixtures and builders for the test suite."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from core.config import load_settings
from core.models import (
    Market,
    MarketState,
    MarketStatus,
    Outcome,
    Position,
    Quote,
    Side,
    Trade,
    TradeProposal,
)

T0 = datetime(2026, 1, 5, 9, 0, tzinfo=UTC)


@pytest.fixture(scope="session")
def settings() -> dict[str, Any]:
    """The project settings, with environment overrides disabled."""
    return load_settings(use_env=False)


@pytest.fixture(scope="session")
def demo_dataset():
    """A small rolling-sequence demo dataset shared across test modules."""
    from core.synthetic import generate_demo_dataset

    return generate_demo_dataset({"seed": 11, "n_markets": 8, "n_steps": 160})


@pytest.fixture(scope="session")
def demo_result(demo_dataset, settings):
    """One backtest over the shared demo dataset."""
    from backtest.engine import run_backtest

    return run_backtest(
        demo_dataset.markets, demo_dataset.quotes, demo_dataset.trades, settings
    )


def make_market(
    market_id: str = "M1",
    outcome: Outcome | None = None,
    close_time: datetime | None = None,
    status: MarketStatus = MarketStatus.OPEN,
) -> Market:
    """Build a market for tests."""
    return Market(
        market_id=market_id,
        question=f"[DEMO DATA] test market {market_id}",
        created_at=T0,
        close_time=close_time or (T0 + timedelta(hours=10)),
        resolution_time=(close_time or (T0 + timedelta(hours=10))) + timedelta(hours=1),
        outcome=outcome,
        status=status,
    )


def make_quote(
    bid: float | None = 0.40,
    ask: float | None = 0.42,
    bid_size: float = 500.0,
    ask_size: float = 500.0,
    ts: datetime | None = None,
    market_id: str = "M1",
) -> Quote:
    """Build a top-of-book quote for tests."""
    return Quote(
        market_id=market_id,
        ts=ts or T0,
        bid=bid,
        ask=ask,
        bid_size=0.0 if bid is None else bid_size,
        ask_size=0.0 if ask is None else ask_size,
    )


def make_trade(
    price: float = 0.41,
    size: float = 10.0,
    ts: datetime | None = None,
    market_id: str = "M1",
    trade_id: str = "T1",
    aggressor: Side = Side.BUY,
) -> Trade:
    """Build a tape print for tests."""
    return Trade(
        trade_id=trade_id,
        market_id=market_id,
        ts=ts or T0,
        price=price,
        size=size,
        aggressor=aggressor,
    )


MISSING = object()
"""Sentinel so ``make_state(quote=None)`` really means "no book"."""


def make_state(
    quote: Quote | None | Any = MISSING,
    as_of: datetime | None = None,
    market: Market | None = None,
    position_qty: float = 0.0,
    avg_price: float = 0.0,
    cash: float = 10_000.0,
    equity: float = 10_000.0,
    peak_equity: float = 10_000.0,
    event_exposure: float = 0.0,
    total_exposure: float = 0.0,
    trades: list[Trade] | None = None,
) -> MarketState:
    """Build a :class:`core.models.MarketState` for tests."""
    market = market or make_market()
    position = (
        Position(market_id=market.market_id, quantity=position_qty, avg_price=avg_price)
        if position_qty
        else None
    )
    return MarketState(
        as_of=as_of or T0,
        market=market,
        quote=make_quote() if quote is MISSING else quote,
        recent_trades=trades or [],
        position=position,
        cash=cash,
        equity=equity,
        peak_equity=peak_equity,
        event_exposure=event_exposure,
        total_exposure=total_exposure,
    )


def make_proposal(
    side: Side = Side.BUY,
    size: float = 100.0,
    reference_price: float = 0.42,
    fair_probability: float = 0.50,
    expected_slippage: float = 0.002,
    fee: float = 0.005,
    market_id: str = "M1",
    ts: datetime | None = None,
) -> TradeProposal:
    """Build a trade proposal for tests."""
    return TradeProposal(
        proposal_id="P1",
        ts=ts or T0,
        market_id=market_id,
        agent_name="TestStrategy",
        side=side,
        size=size,
        limit_price=reference_price,
        reference_price=reference_price,
        fair_probability=fair_probability,
        edge=abs(fair_probability - reference_price),
        expected_fee_per_contract=fee,
        expected_slippage=expected_slippage,
        expected_value_per_contract=0.05,
    )
