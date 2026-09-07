"""Settlement labels: only the venue decides an outcome, and never the price."""

from __future__ import annotations

from datetime import timedelta

from core.models import Market, MarketStatus, Outcome
from research.labels import (
    LABEL_POLICY_VERSION,
    LabelStatus,
    label_for_market,
    label_markets,
    label_summary,
    usable_labels,
)
from tests.conftest import T0, make_market


def resolved(outcome: Outcome = Outcome.YES, **kwargs) -> Market:
    """A normally-settled market."""
    defaults = {
        "outcome": outcome,
        "status": MarketStatus.RESOLVED,
        "close_time": T0 + timedelta(hours=8),
    }
    defaults.update(kwargs)
    market = make_market(**{k: v for k, v in defaults.items() if k != "resolution_time"})
    return market.model_copy(
        update={"resolution_time": kwargs.get("resolution_time", T0 + timedelta(hours=9))}
    )


# -- the happy path --------------------------------------------------------


def test_a_settled_yes_market_is_labelled_one():
    label = label_for_market(resolved(Outcome.YES), venue_status="settled")
    assert label.status is LabelStatus.LABELLED
    assert label.is_usable is True
    assert label.value == 1.0
    assert label.outcome is Outcome.YES
    assert label.resolution_ts == T0 + timedelta(hours=9)
    assert label.policy_version == LABEL_POLICY_VERSION


def test_a_settled_no_market_is_labelled_zero():
    label = label_for_market(resolved(Outcome.NO), venue_status="settled")
    assert label.value == 0.0
    assert label.is_usable is True


def test_finalized_and_determined_statuses_are_accepted():
    for status in ("finalized", "determined", "resolved"):
        assert label_for_market(resolved(), venue_status=status).is_usable, status


def test_the_label_carries_the_instant_the_outcome_became_knowable():
    """Without a knowledge timestamp we cannot prove a forecast came first."""
    label = label_for_market(resolved(), venue_status="settled")
    assert label.resolution_ts is not None
    assert label.resolution_ts > T0


# -- exclusions ------------------------------------------------------------


def test_a_contested_market_is_excluded_even_with_an_outcome():
    """A disputed result can still flip; training on it corrupts every metric."""
    for status in ("disputed", "amended"):
        label = label_for_market(resolved(Outcome.YES), venue_status=status)
        assert label.status is LabelStatus.CONTESTED
        assert label.is_usable is False
        assert label.value is None
        assert "can still change" in label.reason


def test_a_market_with_no_outcome_is_unresolved():
    market = make_market(outcome=None)
    label = label_for_market(market)
    assert label.status is LabelStatus.UNRESOLVED
    assert label.is_usable is False
    assert "price is never used" in label.reason


def test_an_outcome_on_a_non_terminal_status_is_not_trusted():
    market = make_market(outcome=Outcome.YES, status=MarketStatus.CLOSED)
    label = label_for_market(market)
    assert label.status is LabelStatus.UNRESOLVED
    assert "not terminal" in label.reason


def test_an_unrecognised_venue_status_is_treated_as_unresolved():
    """Unknown statuses fail safe: unusable, never wrongly labelled."""
    label = label_for_market(resolved(), venue_status="some_new_status")
    assert label.status is LabelStatus.UNRESOLVED
    assert label.is_usable is False


def test_a_resolved_market_without_a_settlement_time_is_excluded():
    market = resolved().model_copy(update={"resolution_time": None})
    label = label_for_market(market, venue_status="settled")
    assert label.status is LabelStatus.NO_SETTLEMENT_TIME
    assert label.is_usable is False
    assert "before the outcome was known" in label.reason


# -- price must never imply an outcome -------------------------------------


def test_an_extreme_price_does_not_create_a_label():
    """A market at $0.99 has no label until the venue says it settled."""
    from core.models import Quote

    market = make_market(outcome=None)
    Quote(market_id=market.market_id, ts=T0, bid=0.98, ask=0.99,
          bid_size=500.0, ask_size=500.0)
    label = label_for_market(market)
    assert label.value is None
    assert label.status is LabelStatus.UNRESOLVED


def test_labelling_reads_no_price_fields_at_all():
    """The label function's only inputs are the market and its venue status.

    Scanned on the parsed tree, not the raw text: the source legitimately
    *mentions* price in a comment explaining that price is never used, so a
    substring search would flag its own disclaimer.
    """
    import ast
    import inspect
    import textwrap

    tree = ast.parse(textwrap.dedent(inspect.getsource(label_for_market)))
    read = {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    } | {
        node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
    }
    forbidden = {"bid", "ask", "mid", "price", "last_price", "quote", "spread"}
    assert not (read & forbidden), f"label logic reads price fields: {read & forbidden}"
    # And confirm the scan sees the fields it *is* allowed to read.
    assert {"outcome", "status", "resolution_time"} <= read


# -- collection helpers ----------------------------------------------------


def test_labelling_many_markets_keys_by_id():
    markets = [
        resolved(Outcome.YES, market_id="A"),
        resolved(Outcome.NO, market_id="B"),
        make_market(market_id="C", outcome=None),
    ]
    labels = label_markets(markets, {"A": "settled", "B": "settled"})
    assert set(labels) == {"A", "B", "C"}
    assert labels["A"].value == 1.0
    assert labels["C"].is_usable is False


def test_usable_labels_filters_out_everything_unresolved():
    markets = [
        resolved(Outcome.YES, market_id="A"),
        make_market(market_id="B", outcome=None),
        resolved(Outcome.NO, market_id="C"),
    ]
    labels = label_markets(markets, {"A": "settled", "C": "disputed"})
    usable = usable_labels(labels)
    assert set(usable) == {"A"}


def test_the_summary_counts_every_status():
    markets = [
        resolved(Outcome.YES, market_id="A"),
        resolved(Outcome.NO, market_id="B"),
        make_market(market_id="C", outcome=None),
    ]
    counts = label_summary(label_markets(markets, {"A": "settled", "B": "disputed"}))
    assert counts["LABELLED"] == 1
    assert counts["CONTESTED"] == 1
    assert counts["UNRESOLVED"] == 1
    assert set(counts) == {s.value for s in LabelStatus}
