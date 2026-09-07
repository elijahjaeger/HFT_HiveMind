"""Presentation logic for the desktop app (no Qt required)."""

from __future__ import annotations

from datetime import timedelta

import pytest

from app import presenter
from core.config import load_settings
from core.dataset import MarketDataset
from core.models import MarketStatus, Outcome, Quote
from core.synthetic import generate_demo_dataset


@pytest.fixture(scope="module")
def dataset():
    return generate_demo_dataset({"seed": 3, "n_markets": 3, "n_steps": 80})


@pytest.fixture(scope="module")
def app_settings():
    return load_settings(use_env=False)


def test_state_at_hides_the_outcome_from_the_preview(dataset):
    market = dataset.markets[0]
    state = presenter.state_at(dataset, market, dataset.quotes[-1].ts, 5_000.0)
    assert state.market.outcome is None
    assert state.market.resolution_time is None
    assert state.cash == pytest.approx(5_000.0)


def test_state_at_never_includes_future_records(dataset):
    market = dataset.markets[0]
    quotes = dataset.quotes_for(market.market_id)
    as_of = quotes[len(quotes) // 2].ts
    state = presenter.state_at(dataset, market, as_of)
    assert state.quote is not None
    assert state.quote.ts <= as_of
    assert all(trade.ts <= as_of for trade in state.recent_trades)


def test_state_at_reports_no_book_before_the_first_quote(dataset):
    market = dataset.markets[0]
    before = dataset.quotes[0].ts - timedelta(days=1)
    assert presenter.state_at(dataset, market, before).quote is None


def test_market_rows_summarise_every_market(dataset):
    rows = presenter.market_rows(dataset, dataset.quotes[-1].ts)
    assert len(rows) == len(dataset.markets)
    assert {"market_id", "bid", "ask", "spread", "quote_age_s", "status"} <= set(rows[0])


def test_market_rows_report_closed_after_the_close_time(dataset):
    after = max(m.close_time for m in dataset.markets) + timedelta(hours=1)
    assert all(row["status"] == "CLOSED" for row in presenter.market_rows(dataset, after))


def test_decision_preview_runs_the_whole_agent_chain(dataset, app_settings):
    market = dataset.markets[0]
    preview = presenter.decision_preview(
        dataset, market, dataset.quotes[-1].ts, app_settings
    )
    assert preview.forecasts, "expected forecasts from every agent"
    assert preview.consensus is not None
    assert preview.liquidity is not None
    assert preview.state.market.outcome is None
    # A proposal is optional, but if there is one it must carry a risk decision.
    assert (preview.proposal is None) == (preview.decision is None)


def test_the_placeholder_agent_is_flagged_in_the_forecast_rows(dataset, app_settings):
    preview = presenter.decision_preview(
        dataset, dataset.markets[0], dataset.quotes[-1].ts, app_settings
    )
    rows = presenter.forecast_rows(preview)
    placeholders = [r for r in rows if r["placeholder"]]
    assert placeholders, "the BaseRateAgent must be marked as a placeholder"
    assert placeholders[0]["agent"] == "BaseRateAgent"


def test_the_kill_switch_rejects_the_previewed_proposal(dataset, app_settings):
    settings = dict(app_settings)
    settings["risk"] = {**app_settings["risk"], "kill_switch": True}
    settings["strategy"] = {**app_settings["strategy"], "min_edge": 0.0,
                            "require_positive_ev": False}
    market = dataset.markets[0]
    for quote in reversed(dataset.quotes_for(market.market_id)):
        preview = presenter.decision_preview(dataset, market, quote.ts, settings)
        if preview.decision is not None:
            assert preview.decision.action.value == "REJECT"
            codes = [r["code"] for r in presenter.risk_reason_rows(preview.decision)]
            assert "KILL_SWITCH" in codes
            return
    pytest.skip("no proposal was produced to veto")


def test_price_series_omits_missing_sides_rather_than_interpolating():
    from tests.conftest import T0, make_market

    market = make_market("M1", outcome=Outcome.YES, status=MarketStatus.RESOLVED)
    quotes = [
        Quote(market_id="M1", ts=T0, bid=0.40, ask=0.42, bid_size=10, ask_size=10),
        Quote(market_id="M1", ts=T0 + timedelta(minutes=5), bid=None, ask=0.43, ask_size=5),
    ]
    dataset = MarketDataset(markets=[market], quotes=quotes, trades=[], seed=1)
    series = presenter.price_series(dataset, "M1", T0 + timedelta(hours=1))
    assert len(series["ask"]) == 2
    assert len(series["bid"]) == 1
    assert len(series["mid"]) == 1


def test_price_series_is_truncated_at_the_as_of_instant(dataset):
    market_id = dataset.market_ids[0]
    quotes = dataset.quotes_for(market_id)
    as_of = quotes[10].ts
    series = presenter.price_series(dataset, market_id, as_of)
    assert all(ts <= as_of for ts, _ in series["mid"])
    assert all(ts <= as_of for ts, _ in series["trades"])


def test_provenance_banner_names_demo_data(dataset):
    text = presenter.provenance_banner(dataset)
    assert "DEMO DATA" in text
    assert str(dataset.seed) in text


def test_provenance_banner_names_the_venue_for_fetched_data():
    from datetime import UTC, datetime

    dataset = MarketDataset(
        markets=[], quotes=[], trades=[], seed=None,
        label="KALSHI HISTORICAL (read-only)", source="KALSHI_API_V2",
        venue="KALSHI", fetched_at=datetime(2026, 8, 31, 12, tzinfo=UTC),
    )
    text = presenter.provenance_banner(dataset)
    assert "KALSHI" in text
    assert "read-only" in text
    assert "no orders are ever placed" in text


def test_live_trading_banner_is_unambiguous():
    assert "LIVE TRADING DISABLED" in presenter.LIVE_TRADING_BANNER
    assert "no order-placement code" in presenter.LIVE_TRADING_BANNER


def test_backtest_derived_rows_come_out_display_ready(dataset, app_settings):
    from backtest.engine import run_backtest

    result = run_backtest(dataset.markets, dataset.quotes, dataset.trades, app_settings)
    assert len(presenter.headline_metrics(result)) == 5
    assert len(presenter.calibration_rows(result)) == 10
    assert presenter.equity_series(result)["equity"]

    everything = presenter.audit_rows(result)
    only_fills = presenter.audit_rows(result, {"FILL"})
    assert len(only_fills) <= len(everything)
    assert all(row["kind"] == "FILL" for row in only_fills)
