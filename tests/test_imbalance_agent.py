"""BookImbalanceAgent: the signal, its provenance, and how it fails."""

from __future__ import annotations

from datetime import timedelta

import pytest

from agents.base import build_default_agents, combine_forecasts
from agents.imbalance import ASSUMPTIONS, BookImbalanceAgent, imbalance, microprice
from agents.market import MarketAgent
from agents.tape import TapeAgent
from core.config import load_settings
from core.models import ForecastStatus
from tests.conftest import T0, make_quote, make_state, make_trade

CONFIG = {
    "confidence": 0.5,
    "reference_depth": 200.0,
    "max_data_age_seconds": 900.0,
    "hard_cutoff_seconds": 3600.0,
}


def agent(**overrides) -> BookImbalanceAgent:
    return BookImbalanceAgent({**CONFIG, **overrides})


# -- the signal ------------------------------------------------------------


def test_a_balanced_book_gives_the_midpoint():
    quote = make_quote(bid=0.40, ask=0.44, bid_size=100.0, ask_size=100.0)
    assert microprice(quote) == pytest.approx(0.42)
    assert imbalance(quote) == pytest.approx(0.0)


def test_a_heavy_bid_pushes_the_estimate_toward_the_ask():
    quote = make_quote(bid=0.40, ask=0.44, bid_size=300.0, ask_size=100.0)
    assert microprice(quote) == pytest.approx(0.43)
    assert imbalance(quote) == pytest.approx(0.5)


def test_a_heavy_ask_pushes_the_estimate_toward_the_bid():
    quote = make_quote(bid=0.40, ask=0.44, bid_size=100.0, ask_size=300.0)
    assert microprice(quote) == pytest.approx(0.41)
    assert imbalance(quote) < 0


def test_the_microprice_never_leaves_the_spread():
    for bid_size, ask_size in [(1.0, 999.0), (999.0, 1.0), (50.0, 50.0)]:
        quote = make_quote(bid=0.30, ask=0.35, bid_size=bid_size, ask_size=ask_size)
        assert 0.30 <= microprice(quote) <= 0.35


@pytest.mark.parametrize(
    "quote",
    [None, make_quote(bid=0.40, ask=None), make_quote(bid=None, ask=0.44),
     make_quote(bid=0.40, ask=0.44, bid_size=0.0, ask_size=0.0)],
)
def test_the_signal_is_undefined_without_a_two_sided_sized_book(quote):
    assert microprice(quote) is None
    assert imbalance(quote) is None


# -- the forecast ----------------------------------------------------------


def test_a_healthy_book_yields_a_full_confidence_forecast():
    # Both sides at or above the 200-contract reference depth, so the depth
    # factor is capped at 1.0 and confidence is the configured base.
    quote = make_quote(bid=0.40, ask=0.44, bid_size=600.0, ask_size=200.0, ts=T0)
    forecast = agent().analyze(make_state(quote=quote, as_of=T0))
    assert forecast.status is ForecastStatus.OK
    assert forecast.probability == pytest.approx(0.43)
    assert forecast.confidence == pytest.approx(0.5)


def test_confidence_scales_with_the_thinner_side_of_the_book():
    thin = make_quote(bid=0.40, ask=0.44, bid_size=50.0, ask_size=1000.0, ts=T0)
    deep = make_quote(bid=0.40, ask=0.44, bid_size=500.0, ask_size=1000.0, ts=T0)
    thin_c = agent().analyze(make_state(quote=thin, as_of=T0)).confidence
    deep_c = agent().analyze(make_state(quote=deep, as_of=T0)).confidence
    assert thin_c == pytest.approx(0.5 * 50.0 / 200.0)
    assert deep_c == pytest.approx(0.5)  # capped at reference depth
    assert thin_c < deep_c


def test_the_forecast_is_not_a_placeholder():
    quote = make_quote(bid=0.40, ask=0.44, bid_size=200.0, ask_size=200.0, ts=T0)
    assert agent().analyze(make_state(quote=quote, as_of=T0)).is_placeholder is False


# -- declared inputs -------------------------------------------------------


def test_the_forecast_declares_the_quote_timestamp_as_its_only_input():
    quote = make_quote(bid=0.40, ask=0.44, bid_size=200.0, ask_size=200.0, ts=T0)
    forecast = agent().analyze(make_state(quote=quote, as_of=T0 + timedelta(seconds=30)))
    assert [i.name for i in forecast.inputs] == ["quote"]
    assert forecast.inputs[0].ts == T0
    assert forecast.inputs[0].source == quote.source
    assert forecast.latest_input_ts == T0
    assert forecast.uses_future_input is False


def test_declared_inputs_are_never_after_the_forecast_itself():
    for offset in (0, 60, 600, 899):
        quote = make_quote(bid=0.40, ask=0.44, bid_size=200.0, ask_size=200.0, ts=T0)
        forecast = agent().analyze(
            make_state(quote=quote, as_of=T0 + timedelta(seconds=offset))
        )
        assert all(i.ts <= forecast.ts for i in forecast.inputs)


def test_a_missing_quote_declares_no_inputs():
    forecast = agent().analyze(make_state(quote=None, as_of=T0))
    assert forecast.inputs == []
    assert forecast.latest_input_ts is None


# -- exposed assumptions ---------------------------------------------------


def test_every_forecast_exposes_its_assumptions():
    quote = make_quote(bid=0.40, ask=0.44, bid_size=200.0, ask_size=200.0, ts=T0)
    ok = agent().analyze(make_state(quote=quote, as_of=T0))
    failed = agent().analyze(make_state(quote=None, as_of=T0))
    assert ok.assumptions == list(ASSUMPTIONS)
    assert failed.assumptions == list(ASSUMPTIONS), "assumptions travel even with no forecast"
    assert len(ASSUMPTIONS) >= 4
    assert any("size" in a.lower() for a in ASSUMPTIONS)


# -- fails safely ----------------------------------------------------------


def test_no_quote_fails_safely():
    forecast = agent().analyze(make_state(quote=None, as_of=T0))
    assert forecast.probability is None
    assert forecast.confidence == 0.0
    assert forecast.status is ForecastStatus.NO_DATA
    assert "No quote" in forecast.rationale


def test_a_one_sided_book_fails_safely():
    forecast = agent().analyze(make_state(quote=make_quote(bid=0.40, ask=None, ts=T0), as_of=T0))
    assert forecast.probability is None
    assert forecast.status is ForecastStatus.NO_DATA
    assert "One-sided" in forecast.rationale


def test_an_empty_two_sided_book_fails_safely():
    quote = make_quote(bid=0.40, ask=0.44, bid_size=0.0, ask_size=0.0, ts=T0)
    forecast = agent().analyze(make_state(quote=quote, as_of=T0))
    assert forecast.probability is None
    assert forecast.status is ForecastStatus.NO_DATA


def test_a_stale_quote_inside_the_cutoff_degrades_rather_than_fails():
    quote = make_quote(bid=0.40, ask=0.44, bid_size=200.0, ask_size=200.0, ts=T0)
    as_of = T0 + timedelta(seconds=900 + 1350)  # halfway between 900s and 3600s
    forecast = agent().analyze(make_state(quote=quote, as_of=as_of))
    assert forecast.status is ForecastStatus.DEGRADED
    assert forecast.probability == pytest.approx(0.42)
    assert forecast.confidence == pytest.approx(0.25)  # freshness 0.5 * base 0.5
    assert "DEGRADED" in forecast.rationale


def test_a_quote_past_the_hard_cutoff_fails_safely():
    quote = make_quote(bid=0.40, ask=0.44, bid_size=200.0, ask_size=200.0, ts=T0)
    forecast = agent().analyze(make_state(quote=quote, as_of=T0 + timedelta(hours=2)))
    assert forecast.probability is None
    assert forecast.status is ForecastStatus.NO_DATA
    assert "cut-off" in forecast.rationale


def test_nothing_raises_on_any_missing_data_shape():
    shapes = [
        None,
        make_quote(bid=None, ask=None),
        make_quote(bid=0.5, ask=None),
        make_quote(bid=None, ask=0.5),
        make_quote(bid=0.5, ask=0.5, bid_size=0.0, ask_size=0.0),
    ]
    for quote in shapes:
        forecast = agent().analyze(make_state(quote=quote, as_of=T0))
        assert forecast.probability is None
        assert forecast.status is not ForecastStatus.OK


# -- no future information -------------------------------------------------


def test_a_quote_from_the_future_is_rejected_outright():
    future = make_quote(bid=0.40, ask=0.44, bid_size=200.0, ask_size=200.0,
                        ts=T0 + timedelta(seconds=1))
    forecast = agent().analyze(make_state(quote=future, as_of=T0))
    assert forecast.probability is None
    assert forecast.status is ForecastStatus.REJECTED_FUTURE_INPUT
    assert forecast.uses_future_input is True
    assert "future" in forecast.rationale.lower()


def test_the_agent_reads_nothing_but_the_quote():
    """Trades on the tape must not influence the forecast at all."""
    quote = make_quote(bid=0.40, ask=0.44, bid_size=200.0, ask_size=200.0, ts=T0)
    without = agent().analyze(make_state(quote=quote, as_of=T0))
    with_tape = agent().analyze(
        make_state(quote=quote, as_of=T0, trades=[make_trade(price=0.99, size=1e6)])
    )
    assert with_tape.probability == without.probability
    assert with_tape.confidence == without.confidence


# -- integration without touching existing agents -------------------------


def test_existing_agents_are_unaffected_by_the_new_one():
    quote = make_quote(bid=0.40, ask=0.44, bid_size=300.0, ask_size=100.0, ts=T0)
    trades = [make_trade(price=0.60, size=10.0, trade_id=f"t{i}") for i in range(6)]
    state = make_state(quote=quote, as_of=T0, trades=trades)
    market_alone = MarketAgent().analyze(state)
    tape_alone = TapeAgent({"min_trades": 5}).analyze(state)
    # Running the new agent on the same state changes nothing about the others.
    agent().analyze(state)
    assert MarketAgent().analyze(state).probability == market_alone.probability
    assert TapeAgent({"min_trades": 5}).analyze(state).probability == tape_alone.probability
    assert market_alone.inputs == [] and market_alone.assumptions == []


def test_the_agent_is_part_of_the_default_stack():
    agents = build_default_agents(load_settings(use_env=False))
    names = [a.name for a in agents]
    assert "BookImbalanceAgent" in names
    # Existing agents are still there, in their original order, ahead of it.
    assert names.index("MarketAgent") < names.index("BookImbalanceAgent")
    assert names.index("TapeAgent") < names.index("BookImbalanceAgent")


def test_the_ensemble_blends_it_with_its_configured_weight():
    settings = load_settings(use_env=False)
    weights = settings["agents"]["ensemble"]["weights"]
    assert weights.get("BookImbalanceAgent", 0) > 0

    quote = make_quote(bid=0.40, ask=0.44, bid_size=300.0, ask_size=100.0, ts=T0)
    state = make_state(quote=quote, as_of=T0)
    forecasts = [MarketAgent().analyze(state), agent().analyze(state)]
    consensus = combine_forecasts(
        forecasts, {"weights": {"MarketAgent": 1.0, "BookImbalanceAgent": 1.0}}
    )
    assert consensus.probability == pytest.approx((0.42 + 0.43) / 2)
    assert consensus.n_contributors == 2


def test_a_failed_forecast_is_excluded_from_the_ensemble_not_zeroed():
    quote = make_quote(bid=0.40, ask=None, ts=T0)
    state = make_state(quote=quote, as_of=T0)
    forecasts = [MarketAgent().analyze(state), agent().analyze(state)]
    consensus = combine_forecasts(forecasts, {})
    assert "BookImbalanceAgent" in consensus.excluded
    assert consensus.probability == pytest.approx(MarketAgent().analyze(state).probability)


def test_the_backtester_never_hands_it_future_data():
    from backtest.engine import BacktestEngine
    from core.synthetic import generate_demo_dataset

    dataset = generate_demo_dataset({"seed": 3, "n_markets": 3, "n_steps": 60})
    settings = load_settings(use_env=False)
    engine = BacktestEngine(settings)
    result = engine.run(dataset.markets, dataset.quotes, dataset.trades)
    mine = [f for f in result.forecasts if f.agent_name == "BookImbalanceAgent"]
    assert mine, "expected the agent to have run"
    assert all(not f.uses_future_input for f in mine)
    assert all(i.ts <= f.ts for f in mine for i in f.inputs)
    assert not any(f.status is ForecastStatus.REJECTED_FUTURE_INPUT for f in mine)


# -- dashboard output ------------------------------------------------------


def test_presenter_rows_carry_status_and_provenance():
    from app import presenter

    quote = make_quote(bid=0.40, ask=0.44, bid_size=300.0, ask_size=100.0, ts=T0)
    state = make_state(quote=quote, as_of=T0 + timedelta(seconds=45))
    forecasts = [MarketAgent().analyze(state), agent().analyze(state)]

    rows = {r["agent"]: r for r in presenter.forecast_rows_from(forecasts)}
    assert rows["BookImbalanceAgent"]["status"] == "OK"
    assert rows["BookImbalanceAgent"]["inputs"] == "quote@" + T0.strftime("%H:%M:%S")
    assert rows["MarketAgent"]["inputs"] == ""  # undeclared, and unchanged

    provenance = presenter.forecast_provenance_rows(forecasts)
    mine = [r for r in provenance if r["agent"] == "BookImbalanceAgent"]
    kinds = {r["kind"] for r in mine}
    assert kinds == {"input", "assumption"}
    inputs = [r for r in mine if r["kind"] == "input"]
    assert inputs[0]["detail"].startswith("quote")
    assert "age 45s" in inputs[0]["detail"]
