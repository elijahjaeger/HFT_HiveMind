"""Chronological replay, look-ahead protection, settlement and accounting."""

from __future__ import annotations

from datetime import timedelta

import pytest

from agents.base import ForecastAgent
from backtest.engine import (
    BacktestEngine,
    Event,
    Portfolio,
    build_event_stream,
    public_market_view,
    run_backtest,
)
from backtest.fills import simulate_fill
from core.models import (
    EventKind,
    Forecast,
    MarketState,
    MarketStatus,
    Outcome,
    Side,
)
from core.synthetic import generate_demo_dataset
from tests.conftest import T0, make_market, make_quote, make_trade

NO_SLIPPAGE = {"base_slippage": 0.0, "impact_coefficient": 0.0}


class SpyAgent(ForecastAgent):
    """Records every state it is shown so tests can inspect for leakage."""

    name = "SpyAgent"

    def __init__(self) -> None:
        super().__init__({})
        self.seen: list[MarketState] = []

    def analyze(self, state: MarketState) -> Forecast:
        self.seen.append(state)
        return Forecast(
            agent_name=self.name,
            market_id=state.market.market_id,
            ts=state.as_of,
            probability=None,
        )


@pytest.fixture(scope="module")
def dataset():
    """A small deterministic demo dataset."""
    return generate_demo_dataset({"seed": 7, "n_markets": 3, "n_steps": 80})


@pytest.fixture(scope="module")
def demo_result(dataset, module_settings):
    """One full backtest over the small dataset."""
    return run_backtest(dataset.markets, dataset.quotes, dataset.trades, module_settings)


@pytest.fixture(scope="module")
def module_settings():
    from core.config import load_settings

    return load_settings(use_env=False)


# -- event stream ----------------------------------------------------------


def test_event_stream_is_sorted_even_when_inputs_are_shuffled(dataset):
    events = build_event_stream(
        list(reversed(dataset.quotes)),
        list(reversed(dataset.trades)),
        dataset.markets,
        decision_interval_seconds=600,
    )
    timestamps = [e.ts for e in events]
    assert timestamps == sorted(timestamps)


def test_quotes_are_applied_before_trades_at_the_same_instant():
    quote = make_quote(ts=T0)
    trade = make_trade(ts=T0)
    market = make_market(outcome=Outcome.YES)
    kinds = [e.kind for e in build_event_stream([quote], [trade], [market]) if e.ts == T0]
    assert kinds == ["QUOTE", "TRADE"]


def test_decision_ticks_come_after_resolution_at_the_same_instant():
    market = make_market(outcome=Outcome.YES, close_time=T0 + timedelta(hours=1))
    events = build_event_stream(
        [make_quote(ts=T0)], [], [market], decision_interval_seconds=3600
    )
    at_resolution = [e.kind for e in events if e.ts == market.resolution_time]
    assert at_resolution.index("RESOLUTION") < at_resolution.index("CLOCK")


def test_engine_rejects_an_out_of_order_stream(monkeypatch, module_settings, dataset):
    engine = BacktestEngine(module_settings)
    late, early = make_quote(ts=T0 + timedelta(hours=1)), make_quote(ts=T0)
    monkeypatch.setattr(
        "backtest.engine.build_event_stream",
        lambda *a, **k: [
            Event(late.ts, 0, 0, "QUOTE", late),
            Event(early.ts, 0, 1, "QUOTE", early),
        ],
    )
    with pytest.raises(ValueError, match="not chronological"):
        engine.run(dataset.markets, dataset.quotes, dataset.trades)


def test_snapshots_are_recorded_in_chronological_order(demo_result):
    timestamps = [s.ts for s in demo_result.snapshots]
    assert timestamps == sorted(timestamps)


# -- look-ahead protection -------------------------------------------------


def test_the_market_view_handed_to_agents_hides_the_outcome():
    market = make_market(outcome=Outcome.YES, status=MarketStatus.RESOLVED)
    view = public_market_view(market, T0)
    assert view.outcome is None
    assert view.resolution_time is None
    assert view.status is MarketStatus.OPEN


def test_the_market_view_closes_once_the_close_time_has_passed():
    market = make_market(outcome=Outcome.NO, close_time=T0 + timedelta(hours=1))
    assert public_market_view(market, T0 + timedelta(hours=2)).status is MarketStatus.CLOSED


def test_agents_never_see_future_data(dataset, module_settings):
    spy = SpyAgent()
    engine = BacktestEngine(module_settings, forecast_agents=[spy])
    engine.run(dataset.markets, dataset.quotes, dataset.trades)

    assert spy.seen, "the spy agent was never called"
    for state in spy.seen:
        assert state.quote is not None
        assert state.quote.ts <= state.as_of
        assert all(t.ts <= state.as_of for t in state.recent_trades)
        assert state.market.outcome is None
        assert state.market.resolution_time is None
        assert state.as_of < state.market.close_time


def test_results_do_not_depend_on_the_realised_outcomes(dataset, module_settings):
    """Flipping every outcome must not change a single trading decision."""
    flipped = [
        m.model_copy(
            update={"outcome": Outcome.NO if m.outcome is Outcome.YES else Outcome.YES}
        )
        for m in dataset.markets
    ]
    base = run_backtest(dataset.markets, dataset.quotes, dataset.trades, module_settings)
    other = run_backtest(flipped, dataset.quotes, dataset.trades, module_settings)

    def signature(result):
        return [
            (f.ts, f.market_id, f.side, round(f.filled_size, 6), f.avg_price)
            for f in result.fills
        ]

    assert signature(base) == signature(other)
    assert [p.proposal_id for p in base.proposals] == [p.proposal_id for p in other.proposals]


# -- portfolio accounting --------------------------------------------------


def test_buying_moves_cash_and_inventory():
    portfolio = Portfolio(1_000.0)
    quote = make_quote(bid=0.40, ask=0.40, ask_size=1_000.0)
    fill = simulate_fill(quote, Side.BUY, 100.0, T0, NO_SLIPPAGE, "P1")
    portfolio.apply_fill(fill)
    assert portfolio.quantity("M1") == pytest.approx(100.0)
    assert portfolio.cash == pytest.approx(1_000.0 - 40.0 - fill.fee)
    assert portfolio.position("M1").avg_price == pytest.approx(0.40)


def test_a_binary_contract_settles_at_one_on_yes():
    portfolio = Portfolio(1_000.0)
    quote = make_quote(bid=0.40, ask=0.40, ask_size=1_000.0)
    fill = simulate_fill(quote, Side.BUY, 100.0, T0, NO_SLIPPAGE, "P1")
    portfolio.apply_fill(fill)
    settlement = portfolio.settle("M1", Outcome.YES, T0)
    assert settlement is not None
    assert settlement.payout_per_contract == 1.0
    assert settlement.cash_flow == pytest.approx(100.0)
    assert portfolio.cash == pytest.approx(1_000.0 - 40.0 - fill.fee + 100.0)
    assert portfolio.quantity("M1") == 0.0


def test_a_binary_contract_settles_at_zero_on_no():
    portfolio = Portfolio(1_000.0)
    quote = make_quote(bid=0.40, ask=0.40, ask_size=1_000.0)
    fill = simulate_fill(quote, Side.BUY, 100.0, T0, NO_SLIPPAGE, "P1")
    portfolio.apply_fill(fill)
    portfolio.settle("M1", Outcome.NO, T0)
    assert portfolio.cash == pytest.approx(1_000.0 - 40.0 - fill.fee)
    assert portfolio.equity({}) == pytest.approx(1_000.0 - 40.0 - fill.fee)


def test_a_short_position_pays_out_on_yes():
    portfolio = Portfolio(1_000.0)
    quote = make_quote(bid=0.60, ask=0.60, bid_size=1_000.0)
    fill = simulate_fill(quote, Side.SELL, 100.0, T0, NO_SLIPPAGE, "P1")
    portfolio.apply_fill(fill)
    assert portfolio.quantity("M1") == pytest.approx(-100.0)
    portfolio.settle("M1", Outcome.YES, T0)
    assert portfolio.cash == pytest.approx(1_000.0 + 60.0 - fill.fee - 100.0)


def test_settling_a_flat_market_returns_nothing():
    assert Portfolio(100.0).settle("M1", Outcome.YES, T0) is None


def test_exposure_is_worst_case_loss_for_longs_and_shorts():
    portfolio = Portfolio(1_000.0)
    long_fill = simulate_fill(
        make_quote(bid=0.40, ask=0.40, ask_size=1_000.0), Side.BUY, 100.0, T0,
        {"base_slippage": 0.0, "impact_coefficient": 0.0}, "P1"
    )
    portfolio.apply_fill(long_fill)
    assert portfolio.event_exposure("M1", {"M1": 0.40}) == pytest.approx(40.0)

    short = Portfolio(1_000.0)
    short_fill = simulate_fill(
        make_quote(bid=0.40, ask=0.40, bid_size=1_000.0), Side.SELL, 100.0, T0,
        {"base_slippage": 0.0, "impact_coefficient": 0.0}, "P2"
    )
    short.apply_fill(short_fill)
    assert short.event_exposure("M1", {"M1": 0.40}) == pytest.approx(60.0)


def test_closing_a_position_realises_the_profit():
    portfolio = Portfolio(1_000.0)
    no_cost = {"base_slippage": 0.0, "impact_coefficient": 0.0, "fee_rate": 0.0}
    portfolio.apply_fill(
        simulate_fill(
            make_quote(bid=0.40, ask=0.40, ask_size=999.0), Side.BUY, 100.0, T0, no_cost, "P1"
        )
    )
    portfolio.apply_fill(
        simulate_fill(
            make_quote(bid=0.50, ask=0.50, bid_size=999.0), Side.SELL, 100.0, T0, no_cost, "P2"
        )
    )
    assert portfolio.quantity("M1") == 0.0
    assert portfolio.realized_pnl == pytest.approx(10.0)
    assert portfolio.cash == pytest.approx(1_010.0)


# -- integration -----------------------------------------------------------


def test_every_fill_happens_before_its_market_resolves(demo_result, dataset):
    resolution = {m.market_id: m.resolution_time for m in dataset.markets}
    for fill in demo_result.fills:
        assert fill.ts < resolution[fill.market_id]


def test_no_position_survives_resolution(demo_result):
    assert demo_result.metrics["n_settlements"] == len(demo_result.settlements)
    assert demo_result.final_equity == pytest.approx(
        demo_result.snapshots[-1].equity, abs=1e-9
    )


def test_the_audit_log_records_the_full_decision_chain(demo_result):
    kinds = {event.kind for event in demo_result.audit_log}
    assert EventKind.FORECAST in kinds
    assert EventKind.PROPOSAL in kinds
    assert EventKind.RISK_DECISION in kinds
    assert EventKind.SETTLEMENT in kinds
    assert len(demo_result.decisions) == len(demo_result.proposals)


def test_every_executed_fill_traces_back_to_an_approved_decision(demo_result):
    approved = {
        d.proposal_id: d.approved_size for d in demo_result.decisions if not d.is_blocked
    }
    for fill in demo_result.executed_fills:
        assert fill.proposal_id in approved
        assert fill.filled_size <= approved[fill.proposal_id] + 1e-9


def test_the_backtest_is_reproducible(dataset, module_settings):
    first = run_backtest(dataset.markets, dataset.quotes, dataset.trades, module_settings)
    second = run_backtest(dataset.markets, dataset.quotes, dataset.trades, module_settings)
    assert first.metrics == second.metrics


# --------------------------------------------------------------------------
# Live progress and cancellation
#
# The invariant that matters most: with neither hook passed, the run is
# byte-for-bit what it was before these existed.
# --------------------------------------------------------------------------


def test_the_untouched_path_is_unaffected_by_the_new_hooks(dataset, module_settings):
    """Both hooks default to None and every new branch is guarded on that."""
    plain = run_backtest(dataset.markets, dataset.quotes, dataset.trades, module_settings)
    watched = run_backtest(
        dataset.markets, dataset.quotes, dataset.trades, module_settings,
        on_progress=lambda p: None,
    )
    assert plain.metrics == watched.metrics
    assert plain.cancelled is False
    assert len(plain.fills) == len(watched.fills)


def test_progress_reaches_one_hundred_percent_on_the_final_event(
    dataset, module_settings
):
    seen: list = []
    result = run_backtest(
        dataset.markets, dataset.quotes, dataset.trades, module_settings,
        on_progress=seen.append,
    )
    assert seen, "at least one progress update is required"
    assert seen[-1].index == result.n_events - 1
    assert seen[-1].percent == 100


def test_progress_is_throttled_rather_than_emitted_per_event(dataset, module_settings):
    """A UI cannot repaint thousands of times a second and should not be asked to."""
    seen: list = []
    result = run_backtest(
        dataset.markets, dataset.quotes, dataset.trades, module_settings,
        on_progress=seen.append,
    )
    assert result.n_events > 100, "fixture too small to prove throttling"
    assert len(seen) < result.n_events / 10, (
        f"{len(seen)} updates for {result.n_events} events is not throttled"
    )


def test_a_progress_payload_matches_the_state_actually_recorded(
    dataset, module_settings
):
    seen: list = []
    result = run_backtest(
        dataset.markets, dataset.quotes, dataset.trades, module_settings,
        on_progress=seen.append,
    )
    final = seen[-1]
    assert final.last_snapshot is not None
    assert final.last_snapshot.ts == result.snapshots[-1].ts
    assert final.last_snapshot.equity == result.snapshots[-1].equity
    assert final.open_positions == len(final.positions_view)


def test_positions_view_is_copied_out_not_shared(dataset, module_settings):
    """Nothing mutable may cross a thread boundary."""
    seen: list = []
    run_backtest(
        dataset.markets, dataset.quotes, dataset.trades, module_settings,
        on_progress=seen.append,
    )
    for update in seen:
        assert isinstance(update.positions_view, tuple)
        for row in update.positions_view:
            assert isinstance(row, tuple) and len(row) == 4
            assert row[1] in {"LONG", "SHORT"}


def test_should_stop_halts_the_run_and_marks_it_partial(dataset, module_settings):
    calls = {"n": 0}

    def stop_after_fifty() -> bool:
        calls["n"] += 1
        return calls["n"] > 50

    full = run_backtest(dataset.markets, dataset.quotes, dataset.trades, module_settings)
    stopped = run_backtest(
        dataset.markets, dataset.quotes, dataset.trades, module_settings,
        should_stop=stop_after_fifty,
    )
    assert stopped.cancelled is True
    assert full.cancelled is False
    assert len(stopped.snapshots) < len(full.snapshots)
    assert stopped.n_events == full.n_events, "n_events is the stream size, not progress"


def test_a_cancelled_result_is_still_a_well_formed_result(dataset, module_settings):
    """Partial, not broken: the same shape, built from fewer events."""
    stopped = run_backtest(
        dataset.markets, dataset.quotes, dataset.trades, module_settings,
        should_stop=lambda: True,
    )
    assert stopped.cancelled is True
    assert stopped.markets, "markets are recorded before the loop runs"
    assert stopped.final_equity > 0.0, "equity is still marked after an early break"
    assert isinstance(stopped.metrics, dict)


def test_a_cancellation_token_can_be_passed_directly_as_should_stop(
    dataset, module_settings
):
    from core.jobs import CancellationToken

    token = CancellationToken()
    token.cancel()
    result = run_backtest(
        dataset.markets, dataset.quotes, dataset.trades, module_settings,
        should_stop=token,
    )
    assert result.cancelled is True
