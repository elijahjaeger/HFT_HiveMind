"""The read-only Kalshi adapter, driven entirely by mocked responses.

No test here touches the network: every request is served by a fake transport
that also records exactly what was asked for.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from adapters.base import AdapterError, RateLimitedError, RawStore, ReadOnlyClient
from adapters.kalshi import (
    CANDLE_SOURCE,
    CANDLE_SOURCE_PROXY_DEPTH,
    SOURCE,
    VENUE,
    KalshiAdapter,
    KalshiClient,
    KalshiFetchSpec,
    map_outcome,
    map_status,
    normalize_candlesticks,
    normalize_market,
    normalize_trade,
    parse_count,
    parse_dollars,
    parse_timestamp,
    series_ticker_for,
)
from core.models import MarketStatus, Outcome, Side
from tests.fixtures import kalshi_payloads as fx


class FakeTransport:
    """Serves canned payloads by URL substring and records every request."""

    def __init__(self, routes: list[tuple[str, object]], status: int = 200) -> None:
        self.routes = routes
        self.status = status
        self.calls: list[tuple[str, dict[str, str]]] = []

    def __call__(self, url: str, headers: dict[str, str], timeout: float):
        self.calls.append((url, headers))
        for needle, payload in self.routes:
            if needle in url:
                if callable(payload):
                    return payload(url)
                return self.status, fx.encode(payload)
        raise AssertionError(f"no fixture route matched {url}")

    @property
    def urls(self) -> list[str]:
        return [url for url, _ in self.calls]


def kalshi_routes() -> list[tuple[str, object]]:
    """Default routing table covering every endpoint the adapter uses."""

    def trades(url: str):
        page = fx.TRADES_PAGE_2 if "cursor=PAGE2" in url else fx.TRADES_PAGE_1
        return 200, fx.encode(page)

    # Ordered most specific first: a candlestick URL also contains the
    # single-market path, so it has to be matched before it.
    return [
        ("candlesticks", fx.CANDLESTICKS),
        ("trades", trades),
        ("/markets/KXHIGHNY-25AUG31-B70", {"market": fx.SETTLED_MARKET}),
        ("/markets/KXSCALAR-25AUG31", {"market": fx.SCALAR_MARKET}),
        ("/markets", fx.MARKETS_PAGE),
    ]


def make_adapter(tmp_path, transport=None, **config):
    """Build an adapter wired to a fake transport and a temporary raw store."""
    transport = transport or FakeTransport(kalshi_routes())
    cfg = {"min_request_interval_seconds": 0.0, "raw_dir": str(tmp_path), **config}
    store = RawStore("kalshi", root=tmp_path, enabled=cfg.get("store_raw_responses", True))
    return KalshiAdapter(cfg, transport=transport, raw_store=store), transport


# -- field parsing ---------------------------------------------------------


def test_fixed_point_dollar_strings_parse_to_probabilities():
    assert parse_dollars("0.5600") == pytest.approx(0.56)
    assert parse_dollars("0.010000") == pytest.approx(0.01)
    assert parse_dollars(0.42) == pytest.approx(0.42)


@pytest.mark.parametrize("value", [None, ""])
def test_absent_dollar_values_become_none(value):
    assert parse_dollars(value) is None


def test_unparseable_dollar_value_is_rejected():
    with pytest.raises(AdapterError):
        parse_dollars("not-a-price")


def test_fixed_point_counts_parse_to_floats():
    assert parse_count("10.00") == pytest.approx(10.0)
    assert parse_count("0.01") == pytest.approx(0.01)
    assert parse_count(None, default=0.0) == 0.0


def test_iso_timestamps_are_preserved_exactly_in_utc():
    parsed = parse_timestamp("2026-08-31T23:00:00Z")
    assert parsed == datetime(2026, 8, 31, 23, 0, tzinfo=UTC)
    assert parsed.tzinfo is not None


def test_offset_timestamps_are_converted_not_truncated():
    assert parse_timestamp("2026-08-31T19:00:00-04:00") == datetime(
        2026, 8, 31, 23, 0, tzinfo=UTC
    )


def test_unix_timestamps_are_accepted():
    assert parse_timestamp(1788181200) == datetime(2026, 8, 31, 13, 0, tzinfo=UTC)


def test_series_ticker_is_the_leading_segment():
    assert series_ticker_for("KXHIGHNY-25AUG31-B70") == "KXHIGHNY"


# -- resolution rules ------------------------------------------------------


@pytest.mark.parametrize(
    "result,expected",
    [("yes", Outcome.YES), ("no", Outcome.NO), ("", None), (None, None), ("scalar", None)],
)
def test_result_maps_to_a_binary_outcome(result, expected):
    assert map_outcome(result) is expected


def test_status_is_resolved_only_with_a_settled_result():
    assert map_status("settled", Outcome.YES) is MarketStatus.RESOLVED
    assert map_status("settled", None) is MarketStatus.CLOSED
    assert map_status("closed", None) is MarketStatus.CLOSED
    assert map_status("active", None) is MarketStatus.OPEN


def test_market_normalization_preserves_resolution_rules_and_timestamps():
    market = normalize_market(fx.SETTLED_MARKET)
    assert market.market_id == "KXHIGHNY-25AUG31-B70"
    assert market.outcome is Outcome.YES
    assert market.status is MarketStatus.RESOLVED
    assert market.close_time == datetime(2026, 8, 31, 23, 0, tzinfo=UTC)
    assert market.created_at == datetime(2026, 8, 30, 13, 0, tzinfo=UTC)
    assert market.resolution_time == datetime(2026, 9, 1, 0, 15, tzinfo=UTC)
    assert market.rules_primary == fx.SETTLED_MARKET["rules_primary"]
    assert market.rules_secondary == fx.SETTLED_MARKET["rules_secondary"]
    assert market.venue == VENUE
    assert market.is_demo is False
    assert market.source == SOURCE


def test_an_unsettled_market_has_no_outcome_or_resolution_time():
    market = normalize_market(fx.UNSETTLED_MARKET)
    assert market.outcome is None
    assert market.resolution_time is None
    assert market.status is MarketStatus.OPEN


def test_tick_size_comes_from_the_venue_price_ranges():
    assert normalize_market(fx.SETTLED_MARKET).tick_size == pytest.approx(0.01)


def test_a_market_without_a_close_time_is_rejected():
    payload = dict(fx.SETTLED_MARKET)
    payload.pop("close_time")
    with pytest.raises(AdapterError):
        normalize_market(payload)


# -- trades ----------------------------------------------------------------


def test_a_yes_taker_is_recorded_as_a_buy():
    trade = normalize_trade(fx.TRADES_PAGE_1["trades"][0])
    assert trade.aggressor is Side.BUY
    assert trade.price == pytest.approx(0.42)
    assert trade.size == pytest.approx(25.0)
    assert trade.ts == datetime(2026, 8, 31, 13, 5, tzinfo=UTC)
    assert trade.trade_id == "3f0a1b2c-0001"


def test_a_no_taker_is_economically_selling_yes():
    trade = normalize_trade(fx.TRADES_PAGE_1["trades"][1])
    assert trade.aggressor is Side.SELL
    assert trade.price == pytest.approx(0.41)


def test_the_deprecated_taker_side_field_is_still_honoured():
    payload = dict(fx.TRADES_PAGE_1["trades"][1])
    payload.pop("taker_outcome_side")
    payload["taker_side"] = "no"
    assert normalize_trade(payload).aggressor is Side.SELL


def test_a_yes_price_is_derived_from_the_no_price_when_absent():
    payload = dict(fx.TRADES_PAGE_1["trades"][0])
    payload.pop("yes_price_dollars")
    assert normalize_trade(payload).price == pytest.approx(0.42)


def test_a_trade_without_a_timestamp_is_rejected():
    payload = dict(fx.TRADES_PAGE_1["trades"][0])
    payload.pop("created_time")
    with pytest.raises(AdapterError):
        normalize_trade(payload)


# -- candlesticks ----------------------------------------------------------


def test_candlesticks_become_a_chronological_quote_series():
    quotes = normalize_candlesticks("KXHIGHNY-25AUG31-B70", fx.CANDLESTICKS)
    assert [q.ts for q in quotes] == sorted(q.ts for q in quotes)
    assert quotes[0].ts == datetime(2026, 8, 31, 13, 0, tzinfo=UTC)
    assert quotes[0].bid == pytest.approx(0.39)
    assert quotes[0].ask == pytest.approx(0.42)
    assert quotes[0].spread == pytest.approx(0.03)


def test_a_zero_bid_means_no_resting_order_not_a_price_of_zero():
    quotes = normalize_candlesticks("KXHIGHNY-25AUG31-B70", fx.CANDLESTICKS)
    one_sided = quotes[2]
    assert one_sided.bid is None
    assert one_sided.ask == pytest.approx(0.48)
    assert one_sided.is_two_sided is False


def test_extreme_prices_can_be_kept_literally():
    quotes = normalize_candlesticks(
        "KXHIGHNY-25AUG31-B70", fx.CANDLESTICKS, treat_extremes_as_missing=False
    )
    assert quotes[2].bid == pytest.approx(0.0)


def test_a_crossed_candle_is_dropped_rather_than_coerced():
    quotes = normalize_candlesticks("KXHIGHNY-25AUG31-B70", fx.CANDLESTICKS)
    assert len(quotes) == 3  # the fourth candle is crossed
    assert all(q.bid is None or q.ask is None or q.bid <= q.ask for q in quotes)


def test_candlesticks_carry_no_depth_by_default():
    quotes = normalize_candlesticks("KXHIGHNY-25AUG31-B70", fx.CANDLESTICKS)
    assert all(q.bid_size == 0.0 and q.ask_size == 0.0 for q in quotes)
    assert all(q.source == CANDLE_SOURCE for q in quotes)


def test_the_volume_depth_proxy_is_labelled_on_every_quote():
    quotes = normalize_candlesticks(
        "KXHIGHNY-25AUG31-B70", fx.CANDLESTICKS, depth_from_volume=True
    )
    assert quotes[0].bid_size == pytest.approx(1200.0)
    assert quotes[0].ask_size == pytest.approx(1200.0)
    assert all(q.source == CANDLE_SOURCE_PROXY_DEPTH for q in quotes)


def test_the_historical_endpoint_field_names_are_also_understood():
    quotes = normalize_candlesticks(
        "KXHIGHNY-25AUG31-B70", fx.HISTORICAL_CANDLESTICKS, depth_from_volume=True
    )
    assert len(quotes) == 1
    assert quotes[0].bid_size == pytest.approx(1200.0)


# -- client behaviour ------------------------------------------------------


def test_pagination_follows_the_cursor_until_it_is_empty(tmp_path):
    transport = FakeTransport(kalshi_routes())
    client = KalshiClient(
        base_url="https://example.test/v2", transport=transport, min_request_interval=0.0
    )
    trades = client.trades("KXHIGHNY-25AUG31-B70")
    assert len(trades) == 3
    assert any("cursor=PAGE2" in url for url in transport.urls)


def test_query_parameters_are_encoded_and_empty_ones_dropped():
    client = ReadOnlyClient(base_url="https://example.test/v2")
    url = client.build_url("/markets/trades", {"ticker": "ABC", "min_ts": 100, "cursor": None})
    assert url == "https://example.test/v2/markets/trades?ticker=ABC&min_ts=100"


def test_an_invalid_candle_interval_is_refused(tmp_path):
    adapter, _ = make_adapter(tmp_path)
    with pytest.raises(ValueError, match="period_interval"):
        adapter.client.candlesticks("T", 0, 1, period_interval=5)


def test_the_historical_candlestick_path_needs_no_series_ticker(tmp_path):
    adapter, transport = make_adapter(tmp_path)
    adapter.client.candlesticks("KXHIGHNY-25AUG31-B70", 0, 1, 60, historical=True)
    assert "/historical/markets/KXHIGHNY-25AUG31-B70/candlesticks" in transport.urls[-1]


def test_the_live_candlestick_path_derives_the_series_ticker(tmp_path):
    adapter, transport = make_adapter(tmp_path)
    adapter.client.candlesticks("KXHIGHNY-25AUG31-B70", 0, 1, 60, historical=False)
    assert "/series/KXHIGHNY/markets/KXHIGHNY-25AUG31-B70/candlesticks" in transport.urls[-1]


def test_rate_limiting_is_retried_then_surfaced():
    sleeps: list[float] = []
    transport = FakeTransport([("", lambda url: (429, b'{"error":"too many requests"}'))])
    client = ReadOnlyClient(
        base_url="https://example.test/v2",
        transport=transport,
        min_request_interval=0.0,
        max_retries=2,
        sleep=sleeps.append,
    )
    with pytest.raises(RateLimitedError):
        client.get("/markets")
    assert len(transport.calls) == 3  # initial attempt plus two retries
    assert sleeps == [1.0, 2.0]  # exponential backoff


def test_a_server_error_is_reported_with_context():
    transport = FakeTransport([("", lambda url: (500, b"boom"))])
    client = ReadOnlyClient(
        base_url="https://example.test/v2", transport=transport, min_request_interval=0.0
    )
    with pytest.raises(AdapterError, match="HTTP 500"):
        client.get("/markets")


def test_a_non_json_body_is_reported_clearly():
    transport = FakeTransport([("", lambda url: (200, b"<html>nope</html>"))])
    client = ReadOnlyClient(
        base_url="https://example.test/v2", transport=transport, min_request_interval=0.0
    )
    with pytest.raises(AdapterError, match="could not decode JSON"):
        client.get("/markets")


# -- raw storage -----------------------------------------------------------


def test_raw_response_bodies_are_stored_byte_for_byte(tmp_path):
    adapter, _ = make_adapter(tmp_path)
    adapter.client.market("KXHIGHNY-25AUG31-B70")

    files = sorted((tmp_path / "kalshi").rglob("*.json"))
    payloads = [p for p in files if not p.name.endswith(".meta.json")]
    assert payloads, "expected a stored raw response"
    stored = payloads[0].read_bytes()
    assert stored == fx.encode({"market": fx.SETTLED_MARKET})
    assert json.loads(stored)["market"]["rules_primary"]


def test_raw_provenance_is_recorded_in_a_sidecar(tmp_path):
    adapter, _ = make_adapter(tmp_path)
    adapter.client.market("KXHIGHNY-25AUG31-B70")
    meta_files = sorted((tmp_path / "kalshi").rglob("*.meta.json"))
    meta = json.loads(meta_files[0].read_text(encoding="utf-8"))
    assert meta["venue"] == "kalshi"
    assert meta["status"] == 200
    assert "KXHIGHNY-25AUG31-B70" in meta["url"]
    assert meta["bytes"] > 0


def test_raw_storage_can_be_disabled(tmp_path):
    adapter, _ = make_adapter(tmp_path, store_raw_responses=False)
    adapter.client.market("KXHIGHNY-25AUG31-B70")
    assert not list((tmp_path / "kalshi").rglob("*.json"))


def test_a_stored_response_can_be_read_back_unchanged(tmp_path):
    adapter, _ = make_adapter(tmp_path)
    adapter.client.market("KXHIGHNY-25AUG31-B70")
    saved = adapter.client.responses[-1]
    assert saved.path is not None
    reloaded = adapter.raw_store.load(saved.path)
    assert reloaded.body == saved.body
    assert reloaded.json() == saved.json()


# -- end to end ------------------------------------------------------------


def test_fetch_dataset_normalizes_markets_quotes_and_trades(tmp_path):
    adapter, _ = make_adapter(tmp_path)
    dataset = adapter.fetch_dataset(
        KalshiFetchSpec(
            tickers=["KXHIGHNY-25AUG31-B70"],
            start_ts=1788181200,
            end_ts=1788192000,
            period_interval=60,
        )
    )
    assert dataset.venue == VENUE
    assert dataset.is_demo is False
    assert "KALSHI" in dataset.label
    assert dataset.market_ids == ["KXHIGHNY-25AUG31-B70"]
    assert len(dataset.quotes) == 3
    assert len(dataset.trades) == 3
    assert dataset.markets[0].outcome is Outcome.YES
    assert [q.ts for q in dataset.quotes] == sorted(q.ts for q in dataset.quotes)
    assert [t.ts for t in dataset.trades] == sorted(t.ts for t in dataset.trades)


def test_the_depth_proxy_is_disclosed_in_the_dataset_notes(tmp_path):
    adapter, _ = make_adapter(tmp_path, depth_from_volume=True)
    dataset = adapter.fetch_dataset(
        KalshiFetchSpec(tickers=["KXHIGHNY-25AUG31-B70"], start_ts=0, end_ts=1)
    )
    assert any("PROXY" in note for note in dataset.notes)


def test_a_non_binary_market_is_skipped_with_a_note(tmp_path):
    adapter, _ = make_adapter(tmp_path)
    with pytest.raises(AdapterError, match="no markets could be fetched"):
        adapter.fetch_dataset(KalshiFetchSpec(tickers=["KXSCALAR-25AUG31"], start_ts=0, end_ts=1))


def test_a_kalshi_dataset_runs_through_the_existing_backtester(tmp_path):
    from backtest.engine import run_backtest
    from core.config import load_settings

    adapter, _ = make_adapter(tmp_path, depth_from_volume=True)
    dataset = adapter.fetch_dataset(
        KalshiFetchSpec(tickers=["KXHIGHNY-25AUG31-B70"], start_ts=0, end_ts=1)
    )
    settings = load_settings(use_env=False)
    settings["backtest"]["decision_interval_seconds"] = 3600
    result = run_backtest(dataset.markets, dataset.quotes, dataset.trades, settings)

    assert result.n_events > 0
    assert result.metrics["initial_capital"] > 0
    # The market settled YES, so any open position must have been settled out.
    assert all(s.outcome is Outcome.YES for s in result.settlements)


# -- read-only guarantees --------------------------------------------------


def test_the_client_exposes_no_order_placement_surface():
    forbidden = {"post", "put", "delete", "patch", "place_order", "create_order", "cancel"}
    assert forbidden.isdisjoint({name.lower() for name in dir(KalshiClient)})


def test_the_transport_only_ever_issues_get(tmp_path):
    adapter, transport = make_adapter(tmp_path)
    adapter.fetch_dataset(
        KalshiFetchSpec(tickers=["KXHIGHNY-25AUG31-B70"], start_ts=0, end_ts=1)
    )
    assert transport.calls, "expected requests to have been made"
    # The transport signature has no body parameter at all, and every recorded
    # request carries only headers.
    for _, headers in transport.calls:
        assert set(headers) <= {"User-Agent", "Accept"}


def test_no_credential_headers_are_ever_sent(tmp_path):
    adapter, transport = make_adapter(tmp_path)
    adapter.client.market("KXHIGHNY-25AUG31-B70")
    sent = {key.lower() for _, headers in transport.calls for key in headers}
    assert not sent & {
        "authorization", "kalshi-access-key", "kalshi-access-signature",
        "kalshi-access-timestamp", "cookie",
    }
