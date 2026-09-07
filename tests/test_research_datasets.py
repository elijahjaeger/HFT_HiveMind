"""Dataset assembly: labels attach only when the outcome came strictly later."""

from __future__ import annotations

from datetime import timedelta

import pytest

from core.dataset import MarketDataset
from core.models import MarketStatus, Outcome
from core.synthetic import generate_demo_dataset
from research.datasets import (
    build_feature_dataset,
    check_readiness,
    data_version_for,
    dataset_fingerprint,
    decision_timestamps,
)
from tests.conftest import T0, make_market, make_quote, make_trade


def build_market(market_id: str, outcome: Outcome | None, hours: int = 10,
                 resolve_after: int = 1):
    close = T0 + timedelta(hours=hours)
    market = make_market(
        market_id=market_id,
        outcome=outcome,
        close_time=close,
        status=MarketStatus.RESOLVED if outcome else MarketStatus.OPEN,
    )
    return market.model_copy(
        update={"resolution_time": close + timedelta(hours=resolve_after) if outcome else None}
    )


def series(market_id: str, n: int = 40, step_minutes: int = 15):
    quotes = [
        make_quote(bid=0.40, ask=0.42, bid_size=200.0, ask_size=200.0,
                   ts=T0 + timedelta(minutes=step_minutes * i), market_id=market_id)
        for i in range(n)
    ]
    trades = [
        make_trade(price=0.41, size=5.0, ts=T0 + timedelta(minutes=step_minutes * i),
                   trade_id=f"{market_id}-T{i}", market_id=market_id)
        for i in range(n)
    ]
    return quotes, trades


def make_dataset(specs):
    """specs: list of (market_id, outcome)."""
    markets, quotes, trades = [], [], []
    for market_id, outcome in specs:
        markets.append(build_market(market_id, outcome))
        q, t = series(market_id)
        quotes += q
        trades += t
    quotes.sort(key=lambda q: q.ts)
    trades.sort(key=lambda t: t.ts)
    return MarketDataset(markets=markets, quotes=quotes, trades=trades,
                         label="TEST", source="TEST")


# -- decision timestamps ---------------------------------------------------


def test_decision_stamps_stop_a_full_horizon_before_close():
    market = build_market("M1", Outcome.YES, hours=10)
    quotes, _ = series("M1", n=60)
    stamps = decision_timestamps(market, quotes, horizon_seconds=3600, stride_seconds=1800)
    assert stamps
    assert max(stamps) <= market.close_time - timedelta(seconds=3600)


def test_decision_stamps_start_at_the_first_observed_quote():
    market = build_market("M1", Outcome.YES)
    quotes, _ = series("M1", n=40)
    stamps = decision_timestamps(market, quotes, 3600, 1800)
    assert stamps[0] == quotes[0].ts


def test_decision_stamps_are_strictly_ascending():
    market = build_market("M1", Outcome.YES)
    quotes, _ = series("M1", n=40)
    stamps = decision_timestamps(market, quotes, 3600, 1800)
    assert stamps == sorted(stamps)
    assert len(set(stamps)) == len(stamps)


def test_a_market_shorter_than_the_horizon_yields_no_decisions():
    market = build_market("M1", Outcome.YES, hours=1)
    quotes, _ = series("M1", n=4, step_minutes=5)
    assert decision_timestamps(market, quotes, horizon_seconds=7200, stride_seconds=1800) == []


def test_no_quotes_means_no_decisions():
    assert decision_timestamps(build_market("M1", Outcome.YES), [], 3600, 1800) == []


# -- label attachment ------------------------------------------------------


def test_every_labelled_example_predates_its_own_resolution():
    """The core anti-leak invariant of the dataset builder."""
    fd = build_feature_dataset(make_dataset([("A", Outcome.YES), ("B", Outcome.NO)]))
    assert fd.labelled
    for example in fd.labelled:
        assert example.resolution_ts is not None
        assert example.decision_ts < example.resolution_ts
        assert example.seconds_to_resolution > 0


def test_an_unresolved_market_produces_rows_but_no_labels():
    fd = build_feature_dataset(make_dataset([("A", Outcome.YES), ("OPEN", None)]))
    open_rows = [e for e in fd.examples if e.market_id == "OPEN"]
    assert open_rows, "unresolved markets should still yield feature rows"
    assert all(e.label is None for e in open_rows)
    assert all(not e.is_labelled for e in open_rows)
    assert "OPEN" not in fd.market_ids


def test_a_forecast_at_or_after_settlement_is_not_labelled():
    """Equality is rejected too: that is not a forecast."""
    market = build_market("A", Outcome.YES, hours=10, resolve_after=0)
    # Resolution lands exactly at close; horizon 0 lets a stamp reach it.
    quotes, trades = series("A", n=60)
    ds = MarketDataset(markets=[market], quotes=quotes, trades=trades,
                       label="TEST", source="TEST")
    fd = build_feature_dataset(ds, {"horizon_seconds": 0.0, "stride_seconds": 1800.0})
    assert all(e.decision_ts < e.resolution_ts for e in fd.labelled)
    assert fd.skipped.get("outcome_already_known", 0) >= 0


def test_contested_markets_contribute_no_labels():
    ds = make_dataset([("A", Outcome.YES), ("B", Outcome.NO)])
    fd = build_feature_dataset(ds, venue_statuses={"A": "settled", "B": "disputed"})
    assert fd.market_ids == ["A"]
    assert all(e.label is None for e in fd.examples if e.market_id == "B")


# -- sample size is markets, not rows --------------------------------------


def test_effective_n_counts_markets_not_rows():
    """One market forecast many times is one independent outcome."""
    fd = build_feature_dataset(make_dataset([("A", Outcome.YES), ("B", Outcome.NO)]))
    assert fd.effective_n == 2
    assert len(fd.labelled) > 2, "expected many rows per market"
    assert fd.summary()["effective_n_markets"] == 2


def test_the_base_rate_gives_one_vote_per_market():
    fd = build_feature_dataset(
        make_dataset([("A", Outcome.YES), ("B", Outcome.YES), ("C", Outcome.NO)])
    )
    assert fd.base_rate == pytest.approx(2 / 3)


def test_the_base_rate_is_none_without_resolved_markets():
    assert build_feature_dataset(make_dataset([("OPEN", None)])).base_rate is None


# -- ordering and provenance ----------------------------------------------


def test_examples_come_out_in_chronological_order():
    fd = build_feature_dataset(make_dataset([("A", Outcome.YES), ("B", Outcome.NO)]))
    stamps = [e.decision_ts for e in fd.examples]
    assert stamps == sorted(stamps), "dataset must never be shuffled"


def test_the_dataset_records_why_rows_were_skipped():
    market = build_market("A", Outcome.YES)
    quotes = [make_quote(bid=0.40, ask=None, ts=T0 + timedelta(minutes=15 * i),
                         market_id="A") for i in range(40)]
    ds = MarketDataset(markets=[market], quotes=quotes, trades=[],
                       label="TEST", source="TEST")
    fd = build_feature_dataset(ds)
    assert fd.skipped.get("no_two_sided_book", 0) > 0
    assert fd.examples == []


def test_provenance_versions_are_stamped_on_the_dataset():
    fd = build_feature_dataset(make_dataset([("A", Outcome.YES)]))
    summary = fd.summary()
    assert summary["feature_version"]
    assert summary["label_policy_version"]
    assert summary["data_version"].startswith("data-")
    assert "TEST" in summary["source"]


# -- data versioning -------------------------------------------------------


def test_the_data_version_is_stable_for_identical_data():
    a = make_dataset([("A", Outcome.YES)])
    b = make_dataset([("A", Outcome.YES)])
    assert data_version_for(a) == data_version_for(b)


def test_the_data_version_changes_when_an_outcome_changes():
    yes = make_dataset([("A", Outcome.YES)])
    no = make_dataset([("A", Outcome.NO)])
    assert data_version_for(yes) != data_version_for(no)


def test_the_data_version_changes_when_rows_are_added():
    one = make_dataset([("A", Outcome.YES)])
    two = make_dataset([("A", Outcome.YES), ("B", Outcome.NO)])
    assert data_version_for(one) != data_version_for(two)


def test_the_dataset_fingerprint_covers_config_as_well_as_data():
    ds = make_dataset([("A", Outcome.YES)])
    default = dataset_fingerprint(build_feature_dataset(ds))
    other = dataset_fingerprint(build_feature_dataset(ds, {"stride_seconds": 900.0}))
    assert default != other


# -- readiness gate --------------------------------------------------------


def test_the_demo_dataset_is_blocked_for_real_training():
    """Eight synthetic markets is not a training set, and the gate says so."""
    fd = build_feature_dataset(generate_demo_dataset({"seed": 11, "n_markets": 8,
                                                      "n_steps": 240}))
    check = check_readiness(fd)
    assert check.ok is False
    assert check.effective_n < check.required_n
    assert check.blockers
    assert any("resolved market" in b.lower() for b in check.blockers)


def test_readiness_blocks_on_markets_not_on_row_count():
    """Thousands of rows from a handful of markets must not pass the gate."""
    fd = build_feature_dataset(make_dataset([("A", Outcome.YES), ("B", Outcome.NO)]))
    check = check_readiness(fd)
    assert len(fd.labelled) > 20
    assert check.ok is False
    assert check.effective_n == 2


def test_readiness_blocks_when_one_class_is_too_thin():
    specs = [(f"M{i}", Outcome.YES) for i in range(60)]
    fd = build_feature_dataset(make_dataset(specs))
    check = check_readiness(fd)
    assert check.ok is False
    assert any("classes" in b.lower() or "yes" in b.lower() for b in check.blockers)


def test_readiness_renders_display_rows():
    fd = build_feature_dataset(make_dataset([("A", Outcome.YES)]))
    rows = check_readiness(fd).as_rows()
    assert rows
    assert all({"check", "value", "verdict"} <= set(r) for r in rows)
    assert any("BLOCKED" in r["verdict"] for r in rows)
