"""Causal feature construction: nothing after the decision timestamp is read."""

from __future__ import annotations

import math
from datetime import timedelta

import pytest

from research.features import (
    FEATURE_NAMES,
    FEATURE_SPECS,
    FEATURE_VERSION,
    StandardScaler,
    build_features,
    feature_dictionary_rows,
)
from tests.conftest import T0, make_market, make_quote, make_trade


def quote_series(n: int = 40, step_minutes: int = 5, start_mid: float = 0.40,
                 drift: float = 0.002, market_id: str = "M1"):
    """A rising, evenly spaced two-sided book."""
    out = []
    for i in range(n):
        mid = start_mid + drift * i
        out.append(
            make_quote(
                bid=round(mid - 0.01, 4), ask=round(mid + 0.01, 4),
                bid_size=200.0, ask_size=100.0,
                ts=T0 + timedelta(minutes=step_minutes * i), market_id=market_id,
            )
        )
    return out


def trade_series(n: int = 20, step_minutes: int = 10, price: float = 0.45,
                 market_id: str = "M1"):
    return [
        make_trade(price=price, size=10.0, ts=T0 + timedelta(minutes=step_minutes * i),
                   trade_id=f"T{i}", market_id=market_id)
        for i in range(n)
    ]


MARKET = make_market(close_time=T0 + timedelta(hours=10))


# -- causality -------------------------------------------------------------


def test_features_ignore_everything_after_the_decision_timestamp():
    quotes = quote_series(40)
    trades = trade_series(20)
    t = T0 + timedelta(hours=1)

    full = build_features(MARKET, quotes, trades, t)
    truncated = build_features(
        MARKET,
        [q for q in quotes if q.ts <= t],
        [x for x in trades if x.ts <= t],
        t,
    )
    assert full is not None and truncated is not None
    assert full.values == truncated.values


def test_appending_future_data_cannot_change_a_past_feature_vector():
    """The strongest form: bolt on wildly different future data, expect no change."""
    quotes = quote_series(20)
    t = quotes[-1].ts
    before = build_features(MARKET, quotes, trade_series(10), t)

    shocked = quotes + [
        make_quote(bid=0.90, ask=0.92, bid_size=9e5, ask_size=9e5,
                   ts=t + timedelta(minutes=m))
        for m in (5, 10, 15)
    ]
    after = build_features(MARKET, shocked, trade_series(10), t)
    assert before is not None and after is not None
    assert before.values == after.values


def test_the_declared_source_range_never_crosses_the_decision_timestamp():
    quotes = quote_series(30)
    t = T0 + timedelta(hours=1)
    vector = build_features(MARKET, quotes, trade_series(10), t)
    assert vector is not None
    assert vector.source_end_ts is not None
    assert vector.source_end_ts <= t
    assert vector.source_start_ts <= vector.source_end_ts


def test_the_feature_version_is_stamped_on_every_vector():
    vector = build_features(MARKET, quote_series(10), [], T0 + timedelta(minutes=30))
    assert vector is not None
    assert vector.feature_version == FEATURE_VERSION


# -- individual features ---------------------------------------------------


def test_market_prob_is_the_current_mid():
    quotes = [make_quote(bid=0.30, ask=0.34, ts=T0)]
    vector = build_features(MARKET, quotes, [], T0)
    assert vector is not None
    assert vector.values["market_prob"] == pytest.approx(0.32)
    assert vector.values["spread"] == pytest.approx(0.04)


def test_depth_imbalance_is_signed_and_normalised():
    quotes = [make_quote(bid=0.30, ask=0.34, bid_size=300.0, ask_size=100.0, ts=T0)]
    vector = build_features(MARKET, quotes, [], T0)
    assert vector is not None
    assert vector.values["depth_imbalance"] == pytest.approx(0.5)
    assert vector.values["log_depth"] == pytest.approx(math.log1p(100.0))


def test_returns_measure_change_over_their_window():
    quotes = quote_series(n=40, step_minutes=5, start_mid=0.40, drift=0.002)
    t = quotes[-1].ts  # 195 minutes in
    vector = build_features(MARKET, quotes, [], t)
    assert vector is not None
    # 30 minutes back is 6 steps of 0.002.
    assert vector.values["return_30m"] == pytest.approx(0.012, abs=1e-6)
    # 120 minutes back is 24 steps.
    assert vector.values["return_120m"] == pytest.approx(0.048, abs=1e-6)


def test_time_to_close_shrinks_as_the_decision_moves_later():
    early = build_features(MARKET, quote_series(30), [], T0 + timedelta(minutes=30))
    late = build_features(MARKET, quote_series(30), [], T0 + timedelta(hours=2))
    assert early is not None and late is not None
    assert early.values["time_to_close_frac"] > late.values["time_to_close_frac"]
    assert early.values["log_seconds_to_close"] > late.values["log_seconds_to_close"]


def test_time_to_close_uses_only_the_published_close_time():
    """close_time is published at listing, so using it is not look-ahead."""
    t = T0 + timedelta(hours=1)
    vector = build_features(MARKET, quote_series(20), [], t)
    assert vector is not None
    expected = (MARKET.close_time - t).total_seconds()
    assert vector.values["log_seconds_to_close"] == pytest.approx(math.log1p(expected))


def test_quote_age_measures_staleness_at_the_decision_instant():
    quotes = [make_quote(bid=0.30, ask=0.34, ts=T0)]
    vector = build_features(MARKET, quotes, [], T0 + timedelta(seconds=450))
    assert vector is not None
    assert vector.values["quote_age"] == pytest.approx(450.0)


def test_volume_is_a_real_zero_when_the_tape_is_empty():
    """No prints is informative, not missing - it is not flagged."""
    vector = build_features(MARKET, quote_series(20), [], T0 + timedelta(hours=1))
    assert vector is not None
    assert vector.values["volume_120m"] == 0.0
    assert "volume_120m" not in vector.missing


# -- missing data ----------------------------------------------------------


def test_no_quote_at_all_drops_the_row():
    assert build_features(MARKET, [], [], T0) is None


def test_a_quote_only_in_the_future_drops_the_row():
    future = [make_quote(bid=0.3, ask=0.34, ts=T0 + timedelta(hours=1))]
    assert build_features(MARKET, future, [], T0) is None


def test_a_one_sided_book_drops_the_row_rather_than_imputing_a_price():
    """Imputing here would fabricate the target itself."""
    assert build_features(MARKET, [make_quote(bid=0.30, ask=None, ts=T0)], [], T0) is None
    assert build_features(MARKET, [make_quote(bid=None, ask=0.34, ts=T0)], [], T0) is None


def test_absent_depth_defaults_to_zero_and_is_flagged():
    quotes = [make_quote(bid=0.30, ask=0.34, bid_size=0.0, ask_size=0.0, ts=T0)]
    vector = build_features(MARKET, quotes, [], T0)
    assert vector is not None
    assert vector.values["depth_imbalance"] == 0.0
    assert "depth_imbalance" in vector.missing
    assert "log_depth" in vector.missing


def test_a_short_history_flags_the_windowed_features():
    quotes = [make_quote(bid=0.30, ask=0.34, ts=T0)]
    vector = build_features(MARKET, quotes, [], T0)
    assert vector is not None
    for name in ("return_30m", "return_120m", "volatility_120m", "tape_vs_mid"):
        assert name in vector.missing
        assert vector.values[name] == 0.0


def test_missing_count_matches_the_flagged_features():
    quotes = [make_quote(bid=0.30, ask=0.34, bid_size=0.0, ask_size=0.0, ts=T0)]
    vector = build_features(MARKET, quotes, [], T0)
    assert vector is not None
    assert vector.values["missing_count"] == float(len(vector.missing))
    assert vector.is_complete is False


def test_a_complete_row_reports_itself_complete():
    vector = build_features(MARKET, quote_series(40), trade_series(20),
                            T0 + timedelta(hours=3))
    assert vector is not None
    assert vector.missing == ()
    assert vector.is_complete is True


# -- the feature contract --------------------------------------------------


def test_every_declared_feature_is_actually_produced():
    vector = build_features(MARKET, quote_series(40), trade_series(20),
                            T0 + timedelta(hours=3))
    assert vector is not None
    assert set(vector.values) == set(FEATURE_NAMES)


def test_the_vector_order_is_fixed():
    vector = build_features(MARKET, quote_series(40), trade_series(20),
                            T0 + timedelta(hours=3))
    assert vector is not None
    assert vector.vector() == [vector.values[n] for n in FEATURE_NAMES]
    assert len(vector.vector()) == len(FEATURE_SPECS)


def test_every_feature_declares_sources_and_a_missing_policy():
    for spec in FEATURE_SPECS:
        assert spec.name and spec.description
        assert spec.source_fields, f"{spec.name} declares no source fields"
        assert spec.missing_policy, f"{spec.name} declares no missing-data policy"


def test_the_feature_dictionary_renders_every_feature():
    rows = feature_dictionary_rows()
    assert len(rows) == len(FEATURE_SPECS)
    assert {r["feature"] for r in rows} == set(FEATURE_NAMES)
    assert all({"feature", "description", "sources", "window", "missing", "scaled"} <= set(r)
               for r in rows)


# -- scaling ---------------------------------------------------------------


def test_the_scaler_standardises_its_training_rows():
    scaler = StandardScaler().fit([[0.0, 10.0], [2.0, 20.0], [4.0, 30.0]], ["a", "b"])
    assert scaler.means == pytest.approx([2.0, 20.0])
    out = scaler.transform([[2.0, 20.0]])
    assert out[0] == pytest.approx([0.0, 0.0])


def test_a_constant_column_does_not_divide_by_zero():
    scaler = StandardScaler().fit([[5.0], [5.0], [5.0]], ["c"])
    assert scaler.stdevs == [1.0]
    assert scaler.transform([[5.0]])[0] == pytest.approx([0.0])


def test_the_scaler_refuses_to_transform_before_it_is_fitted():
    with pytest.raises(RuntimeError, match="training fold"):
        StandardScaler().transform([[1.0]])


def test_the_scaler_refuses_to_fit_on_nothing():
    with pytest.raises(ValueError):
        StandardScaler().fit([], [])


def test_scaler_parameters_are_plain_serialisable_numbers():
    scaler = StandardScaler().fit([[1.0, 2.0], [3.0, 4.0]], ["a", "b"])
    payload = scaler.as_dict()
    assert set(payload) == {"names", "means", "stdevs"}
    assert all(isinstance(v, float) for v in payload["means"])


def test_test_rows_are_scaled_with_training_parameters_only():
    """The give-away of leakage would be test rows re-centring themselves."""
    scaler = StandardScaler().fit([[0.0], [2.0]], ["a"])
    scaled = scaler.transform([[100.0], [200.0]])
    assert scaler.means == pytest.approx([1.0])  # unchanged by the transform
    assert scaled[0][0] == pytest.approx((100.0 - 1.0) / scaler.stdevs[0])
