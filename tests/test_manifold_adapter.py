"""Manifold adapter: normalisation, exclusions, and the derived-book caveat.

No test here touches the network: every request is served by a fake transport.

The properties that matter most: a market with no binary answer never produces
a label, and the synthetic bid/ask is always stamped as derived so nothing
downstream can mistake it for a real order book.
"""

from __future__ import annotations

import pytest

from adapters.base import AdapterError, RawStore
from adapters.manifold import (
    LABEL,
    MIN_SPREAD,
    PROXY_SOURCE,
    SOURCE,
    ManifoldAdapter,
    ManifoldFetchSpec,
    estimate_half_spread,
    is_usable_binary,
    map_outcome,
    normalize_bets,
    normalize_market,
    parse_probability,
    parse_timestamp,
)
from core.models import MarketStatus, Outcome, Side
from research.corpus import describe
from tests.fixtures import manifold_payloads as fx


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


def manifold_routes() -> list[tuple[str, object]]:
    """Ordered most specific first."""
    return [
        ("/bets", fx.BETS),
        ("/search-markets", fx.SEARCH_PAGE),
        ("/market/resolved-yes-1", fx.RESOLVED_YES_MARKET),
        ("/market/resolved-mkt-1", fx.AMBIGUOUS_MARKET),
        ("/market/multi-1", fx.MULTI_MARKET),
        ("/market/", fx.RESOLVED_MARKET),
    ]


def make_adapter(tmp_path, transport=None, **config):
    transport = transport or FakeTransport(manifold_routes())
    cfg = {"min_request_interval_seconds": 0.0, "raw_dir": str(tmp_path), **config}
    store = RawStore("manifold", root=tmp_path, enabled=True)
    return ManifoldAdapter(cfg, transport=transport, raw_store=store), transport


# -- field parsing ---------------------------------------------------------


def test_epoch_milliseconds_parse_to_utc():
    moment = parse_timestamp(1723334400000)
    assert moment is not None
    assert moment.year == 2024
    assert moment.tzinfo is not None


@pytest.mark.parametrize("value", [None, "", "not-a-number", object()])
def test_unusable_timestamps_become_none(value):
    assert parse_timestamp(value) is None


def test_probabilities_outside_the_unit_interval_are_rejected():
    assert parse_probability(0.42) == pytest.approx(0.42)
    assert parse_probability(1.5) is None
    assert parse_probability(-0.1) is None
    assert parse_probability("nope") is None


# -- outcomes: the exclusions that protect the labels ----------------------


@pytest.mark.parametrize(
    ("resolution", "expected"),
    [("YES", Outcome.YES), ("NO", Outcome.NO), ("yes", Outcome.YES)],
)
def test_binary_resolutions_map_to_outcomes(resolution, expected):
    assert map_outcome(resolution) is expected


@pytest.mark.parametrize("resolution", ["MKT", "CANCEL", "", None, "PROB"])
def test_a_market_without_a_binary_answer_has_no_outcome(resolution):
    """Inferring one from the final price is exactly the mistake labels.py bans."""
    assert map_outcome(resolution) is None


def test_only_resolved_binary_markets_are_usable():
    assert is_usable_binary(fx.RESOLVED_MARKET) is True
    assert is_usable_binary(fx.RESOLVED_YES_MARKET) is True


@pytest.mark.parametrize(
    ("payload", "why"),
    [
        (fx.AMBIGUOUS_MARKET, "MKT is a partial resolution"),
        (fx.CANCELLED_MARKET, "CANCEL voided the market"),
        (fx.UNRESOLVED_MARKET, "not resolved yet"),
        (fx.MULTI_MARKET, "not a binary market"),
    ],
)
def test_markets_that_cannot_produce_a_label_are_excluded(payload, why):
    assert is_usable_binary(payload) is False, why


# -- market normalisation --------------------------------------------------


def test_a_resolved_market_carries_its_outcome_and_settlement_time():
    market = normalize_market(fx.RESOLVED_MARKET)
    assert market.outcome is Outcome.NO
    assert market.resolution_time is not None
    assert market.status is MarketStatus.RESOLVED
    assert market.is_demo is False
    assert market.source == SOURCE


def test_a_resolved_market_without_a_settlement_time_loses_its_outcome():
    """Without one we cannot prove a forecast predates the outcome."""
    market = normalize_market(fx.NO_SETTLEMENT_TIME)
    assert market.outcome is None
    assert market.status is MarketStatus.OPEN


def test_a_market_without_a_close_time_is_rejected():
    with pytest.raises(AdapterError, match="close time"):
        normalize_market(fx.NO_CLOSE_TIME)


# -- the derived book ------------------------------------------------------


def test_the_half_spread_is_estimated_from_observed_price_impact():
    """The median move a real bet caused is a defensible stand-in for a spread."""
    spread = estimate_half_spread(fx.BETS)
    assert spread >= MIN_SPREAD
    assert spread < 0.15


def test_a_market_with_no_observed_impact_still_costs_something_to_trade():
    """Otherwise it would look free, which would flatter every result."""
    flat = [{"probBefore": 0.5, "probAfter": 0.5} for _ in range(5)]
    assert estimate_half_spread(flat) == pytest.approx(MIN_SPREAD)
    assert estimate_half_spread([]) == pytest.approx(MIN_SPREAD)


def test_bets_become_a_two_sided_quote_series_and_a_tape():
    quotes, trades = normalize_bets("m1", fx.BETS, liquidity=200.0)
    assert quotes and trades
    for quote in quotes:
        assert quote.bid is not None and quote.ask is not None
        assert quote.ask > quote.bid
        assert 0.0 < quote.bid < 1.0


def test_every_derived_quote_is_stamped_as_a_proxy():
    """Nothing downstream may mistake this for a real order book."""
    quotes, _ = normalize_bets("m1", fx.BETS)
    assert quotes
    assert all(q.source == PROXY_SOURCE for q in quotes)


def test_redemptions_are_not_counted_as_trades():
    """They are share settlements, not price-forming events."""
    _, trades = normalize_bets("m1", fx.BETS)
    real_bets = [b for b in fx.BETS if not b["isRedemption"]]
    assert len(trades) == len(real_bets)


def test_the_tape_records_the_side_the_bettor_took():
    _, trades = normalize_bets("m1", fx.BETS)
    assert trades[0].aggressor is Side.BUY
    assert trades[1].aggressor is Side.SELL


def test_quotes_are_chronological():
    quotes, _ = normalize_bets("m1", list(reversed(fx.BETS)))
    assert [q.ts for q in quotes] == sorted(q.ts for q in quotes)


def test_bets_without_a_usable_probability_are_dropped():
    quotes, _ = normalize_bets("m1", [{"createdTime": 1, "probAfter": None}])
    assert quotes == []


# -- the adapter -----------------------------------------------------------


def test_a_fetched_dataset_is_real_and_says_where_it_came_from(tmp_path):
    adapter, _ = make_adapter(tmp_path)
    dataset = adapter.fetch_dataset(ManifoldFetchSpec(market_ids=["resolved-no-1"]))

    assert dataset.label == LABEL
    assert dataset.venue == "manifold"
    assert dataset.is_demo is False
    assert describe(dataset).is_synthetic is False
    assert dataset.markets and dataset.quotes


def test_the_dataset_declares_that_its_book_is_derived(tmp_path):
    adapter, _ = make_adapter(tmp_path)
    dataset = adapter.fetch_dataset(ManifoldFetchSpec(market_ids=["resolved-no-1"]))
    notes = " ".join(dataset.notes)
    assert "DERIVED" in notes
    assert "no order book" in notes


def test_the_dataset_says_the_venue_is_play_money(tmp_path):
    """An edge found here is not tradable, and the corpus must say so."""
    adapter, _ = make_adapter(tmp_path)
    dataset = adapter.fetch_dataset(ManifoldFetchSpec(market_ids=["resolved-no-1"]))
    assert any("PLAY MONEY" in note for note in dataset.notes)


def test_a_market_that_cannot_be_labelled_is_skipped_with_a_note(tmp_path):
    adapter, _ = make_adapter(tmp_path)
    dataset = adapter.fetch_dataset(
        ManifoldFetchSpec(market_ids=["resolved-no-1", "resolved-mkt-1"])
    )
    assert [m.market_id for m in dataset.markets] == ["resolved-no-1"]
    assert any("resolved-mkt-1" in note for note in dataset.notes)


def test_a_pull_where_nothing_resolves_is_an_error_not_an_empty_dataset(tmp_path):
    adapter, _ = make_adapter(tmp_path)
    with pytest.raises(AdapterError, match="no Manifold market"):
        adapter.fetch_dataset(ManifoldFetchSpec(market_ids=["multi-1"]))


def test_discovery_returns_only_markets_that_can_be_labelled(tmp_path):
    adapter, _ = make_adapter(tmp_path)
    found = adapter.discover(limit=50)
    ids = {m["id"] for m in found}
    assert ids == {"resolved-no-1", "resolved-yes-1"}


def test_a_failed_discovery_returns_nothing_rather_than_raising(tmp_path):
    def boom(url: str):
        raise OSError("no network")

    adapter, _ = make_adapter(tmp_path, FakeTransport([("", boom)]))
    assert adapter.discover() == []


# -- read-only by construction ---------------------------------------------


def test_every_request_is_a_get_with_no_credentials(tmp_path):
    adapter, transport = make_adapter(tmp_path)
    adapter.fetch_dataset(ManifoldFetchSpec(market_ids=["resolved-no-1"]))

    assert transport.calls, "no request was made"
    for url, headers in transport.calls:
        assert url.startswith("https://")
        lowered = {k.lower() for k in headers}
        assert "authorization" not in lowered
        assert "cookie" not in lowered
        assert not any("key" in k for k in lowered)


def test_the_client_has_no_write_method():
    from adapters.manifold import ManifoldClient

    for verb in ("post", "put", "delete", "patch", "place_order", "bet"):
        assert not hasattr(ManifoldClient, verb), f"{verb} must not exist"
