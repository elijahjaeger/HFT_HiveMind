"""Causal feature construction.

Every feature here is a function of data stamped at or before the decision
timestamp ``t``. That is enforced in one place - :func:`_visible` - rather than
trusted to each feature, so a new feature cannot accidentally reach forward.

Each feature declares what it is built from, what happens when the inputs are
missing, and whether it is scaled. Scaling is *not* applied here: it is fitted
on the training fold only, inside the walk-forward loop, because fitting a
scaler on all the data would leak the test distribution into training.
"""

from __future__ import annotations

import math
import statistics
from bisect import bisect_right
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from core.models import Market, Quote, Trade

FEATURE_VERSION = "features-v1"


@dataclass(frozen=True)
class FeatureSpec:
    """What one feature is, where it comes from, and how it fails."""

    name: str
    description: str
    source_fields: tuple[str, ...]
    window_seconds: float | None
    missing_policy: str
    scaled: bool = True

    def as_row(self) -> dict[str, Any]:
        """Display row for the dashboard's feature dictionary."""
        return {
            "feature": self.name,
            "description": self.description,
            "sources": ", ".join(self.source_fields),
            "window": "point-in-time" if not self.window_seconds
            else f"{self.window_seconds / 60:.0f} min",
            "missing": self.missing_policy,
            "scaled": "yes" if self.scaled else "no",
        }


SHORT_WINDOW = 1800.0
LONG_WINDOW = 7200.0

FEATURE_SPECS: tuple[FeatureSpec, ...] = (
    FeatureSpec(
        "market_prob", "Mid-price of the book, the market's own implied probability.",
        ("quote.bid", "quote.ask"), None,
        "No two-sided book: row dropped. This is the one feature nothing works without.",
    ),
    FeatureSpec(
        "spread", "Ask minus bid, in probability units.",
        ("quote.bid", "quote.ask"), None, "No two-sided book: row dropped.",
    ),
    FeatureSpec(
        "depth_imbalance", "(bid_size - ask_size) / (bid_size + ask_size).",
        ("quote.bid_size", "quote.ask_size"), None, "No displayed size: 0.0, flagged missing.",
    ),
    FeatureSpec(
        "log_depth", "log1p of the smaller displayed side, a liquidity proxy.",
        ("quote.bid_size", "quote.ask_size"), None, "No displayed size: 0.0, flagged missing.",
    ),
    FeatureSpec(
        "return_30m", "Change in mid-price over the last 30 minutes.",
        ("quote.bid", "quote.ask"), SHORT_WINDOW, "No quote 30 min ago: 0.0, flagged missing.",
    ),
    FeatureSpec(
        "return_120m", "Change in mid-price over the last 2 hours.",
        ("quote.bid", "quote.ask"), LONG_WINDOW, "No quote 2 h ago: 0.0, flagged missing.",
    ),
    FeatureSpec(
        "volatility_120m", "Standard deviation of mid-price over the last 2 hours.",
        ("quote.bid", "quote.ask"), LONG_WINDOW, "Fewer than 3 quotes: 0.0, flagged missing.",
    ),
    FeatureSpec(
        "volume_120m", "Total traded size on the tape over the last 2 hours.",
        ("trade.size",), LONG_WINDOW, "No prints: 0.0 (a real, informative zero).",
    ),
    FeatureSpec(
        "tape_vs_mid", "Volume-weighted trade price minus the mid-price.",
        ("trade.price", "trade.size", "quote.bid", "quote.ask"), LONG_WINDOW,
        "No prints: 0.0, flagged missing.",
    ),
    FeatureSpec(
        "time_to_close_frac", "Time left until close, as a share of the market's life.",
        ("market.close_time", "market.created_at"), None,
        "Always available: close_time is published at listing, not derived from the future.",
    ),
    FeatureSpec(
        "log_seconds_to_close", "log1p of seconds remaining until close.",
        ("market.close_time",), None, "Always available.",
    ),
    FeatureSpec(
        "quote_age", "Seconds since the last quote update, a staleness measure.",
        ("quote.ts",), None, "No quote: row dropped.",
    ),
    FeatureSpec(
        "missing_count", "How many of the above were unavailable and defaulted.",
        ("(internal)",), None, "Never missing; this is the data-quality flag.", scaled=False,
    ),
)

FEATURE_NAMES: tuple[str, ...] = tuple(spec.name for spec in FEATURE_SPECS)


@dataclass(frozen=True)
class FeatureVector:
    """Features for one ``(market, decision timestamp)`` pair."""

    market_id: str
    decision_ts: datetime
    values: dict[str, float]
    missing: tuple[str, ...] = ()
    source_start_ts: datetime | None = None
    source_end_ts: datetime | None = None
    feature_version: str = FEATURE_VERSION

    def vector(self, names: Sequence[str] = FEATURE_NAMES) -> list[float]:
        """Values in a fixed order, for the model."""
        return [self.values.get(name, 0.0) for name in names]

    @property
    def is_complete(self) -> bool:
        """True when no feature had to fall back to a default."""
        return not self.missing


def _visible(items: Sequence[Any], decision_ts: datetime) -> list[Any]:
    """Every record stamped at or before ``decision_ts``.

    The single choke point for causality. Inputs are assumed sorted by ``ts``,
    which the dataset builder guarantees, so this is a binary search.
    """
    timestamps = [item.ts for item in items]
    return list(items[: bisect_right(timestamps, decision_ts)])


def _mid_at_or_before(quotes: Sequence[Quote], moment: datetime) -> float | None:
    """Mid-price of the most recent two-sided quote at or before ``moment``."""
    for quote in reversed(_visible(quotes, moment)):
        if quote.mid is not None:
            return quote.mid
    return None


def build_features(
    market: Market,
    quotes: Sequence[Quote],
    trades: Sequence[Trade],
    decision_ts: datetime,
    config: dict[str, Any] | None = None,
) -> FeatureVector | None:
    """Build one causal feature vector, or ``None`` if the row is unusable.

    Args:
        market: The market being forecast.
        quotes: That market's quotes, sorted by timestamp.
        trades: That market's tape prints, sorted by timestamp.
        decision_ts: The instant the forecast is made. Nothing stamped after
            this is read.
        config: Optional overrides for the rolling-window lengths.

    Returns:
        A :class:`FeatureVector`, or ``None`` when there is no usable book at
        ``decision_ts`` (no quote at all, or no two-sided quote).
    """
    cfg = dict(config or {})
    short_window = float(cfg.get("short_window_seconds", SHORT_WINDOW))
    long_window = float(cfg.get("long_window_seconds", LONG_WINDOW))

    seen_quotes = _visible(quotes, decision_ts)
    if not seen_quotes:
        return None

    current = seen_quotes[-1]
    mid = current.mid
    if mid is None or current.spread is None:
        # A one-sided book gives no usable probability. Dropping the row is
        # honest; imputing a price here would fabricate the target itself.
        return None

    seen_trades = _visible(trades, decision_ts)
    missing: list[str] = []
    values: dict[str, float] = {}

    values["market_prob"] = mid
    values["spread"] = current.spread

    total_size = current.bid_size + current.ask_size
    if total_size > 0:
        values["depth_imbalance"] = (current.bid_size - current.ask_size) / total_size
        values["log_depth"] = math.log1p(min(current.bid_size, current.ask_size))
    else:
        values["depth_imbalance"] = 0.0
        values["log_depth"] = 0.0
        missing.append("depth_imbalance")
        missing.append("log_depth")

    for name, window in (("return_30m", short_window), ("return_120m", long_window)):
        past = _mid_at_or_before(seen_quotes, decision_ts - timedelta(seconds=window))
        if past is None:
            values[name] = 0.0
            missing.append(name)
        else:
            values[name] = mid - past

    window_start = decision_ts - timedelta(seconds=long_window)
    window_mids = [
        q.mid for q in seen_quotes if q.ts >= window_start and q.mid is not None
    ]
    if len(window_mids) >= 3:
        values["volatility_120m"] = statistics.pstdev(window_mids)
    else:
        values["volatility_120m"] = 0.0
        missing.append("volatility_120m")

    window_trades = [t for t in seen_trades if t.ts >= window_start]
    volume = sum(t.size for t in window_trades)
    values["volume_120m"] = math.log1p(volume)
    if window_trades and volume > 0:
        vwap = sum(t.price * t.size for t in window_trades) / volume
        values["tape_vs_mid"] = vwap - mid
    else:
        values["tape_vs_mid"] = 0.0
        missing.append("tape_vs_mid")

    seconds_left = max(0.0, (market.close_time - decision_ts).total_seconds())
    lifetime = max(1.0, (market.close_time - market.created_at).total_seconds())
    values["time_to_close_frac"] = min(1.0, seconds_left / lifetime)
    values["log_seconds_to_close"] = math.log1p(seconds_left)
    values["quote_age"] = max(0.0, (decision_ts - current.ts).total_seconds())
    values["missing_count"] = float(len(missing))

    return FeatureVector(
        market_id=market.market_id,
        decision_ts=decision_ts,
        values=values,
        missing=tuple(missing),
        source_start_ts=seen_quotes[0].ts,
        source_end_ts=current.ts,
        feature_version=FEATURE_VERSION,
    )


def feature_dictionary_rows() -> list[dict[str, Any]]:
    """The feature dictionary, for display in the dashboard."""
    return [spec.as_row() for spec in FEATURE_SPECS]


@dataclass
class StandardScaler:
    """Zero-mean, unit-variance scaling. Fitted on training data only.

    Kept here rather than taken from a library so that the fitted parameters
    are plain numbers that can be stored, inspected and diffed in the registry.
    """

    means: list[float] = field(default_factory=list)
    stdevs: list[float] = field(default_factory=list)
    names: list[str] = field(default_factory=list)
    is_fitted: bool = False

    def fit(self, rows: Sequence[Sequence[float]], names: Sequence[str]) -> StandardScaler:
        """Fit on training rows only.

        Raises:
            ValueError: If ``rows`` is empty.
        """
        if not rows:
            raise ValueError("cannot fit a scaler on zero rows")
        columns = list(zip(*rows, strict=True))
        self.means = [sum(col) / len(col) for col in columns]
        self.stdevs = [
            (statistics.pstdev(col) if len(col) > 1 else 0.0) or 1.0 for col in columns
        ]
        self.names = list(names)
        self.is_fitted = True
        return self

    def transform(self, rows: Sequence[Sequence[float]]) -> list[list[float]]:
        """Apply the fitted scaling.

        Raises:
            RuntimeError: If called before :meth:`fit`.
        """
        if not self.is_fitted:
            raise RuntimeError("scaler must be fitted on the training fold first")
        return [
            [(value - m) / s for value, m, s in zip(row, self.means, self.stdevs, strict=True)]
            for row in rows
        ]

    def as_dict(self) -> dict[str, Any]:
        """Serialisable parameters, for the registry."""
        return {"names": self.names, "means": self.means, "stdevs": self.stdevs}
