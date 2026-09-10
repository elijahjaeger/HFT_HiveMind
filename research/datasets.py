"""Assembling training examples from stored, versioned market data.

The rule this module exists to enforce: **features and labels come from
disjoint time regions**. Features are built from data at or before the decision
timestamp; the label becomes knowable strictly later, at settlement. Unresolved
markets can still produce feature rows - useful for drift monitoring and for
inference - but they carry no label and are excluded from anything that trains
or scores.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from core.dataset import MarketDataset
from core.models import Market, Quote, Trade
from research.features import (
    FEATURE_NAMES,
    FEATURE_VERSION,
    FeatureVector,
    build_features,
)
from research.labels import (
    LABEL_POLICY_VERSION,
    LabelStatus,
    SettlementLabel,
    label_markets,
    label_summary,
)

DEFAULT_HORIZON_SECONDS = 3600.0
DEFAULT_STRIDE_SECONDS = 1800.0


@dataclass(frozen=True)
class TrainingExample:
    """One ``(market, decision timestamp)`` pair with its eventual outcome."""

    market_id: str
    decision_ts: datetime
    features: FeatureVector
    label: float | None
    resolution_ts: datetime | None
    market_prob: float

    @property
    def is_labelled(self) -> bool:
        """True when this example may be trained or scored on."""
        return self.label is not None and self.resolution_ts is not None

    @property
    def seconds_to_resolution(self) -> float | None:
        """How far ahead the outcome was, at the moment of the forecast."""
        if self.resolution_ts is None:
            return None
        return (self.resolution_ts - self.decision_ts).total_seconds()


@dataclass
class FeatureDataset:
    """Every example built from one source dataset, plus its provenance."""

    examples: list[TrainingExample] = field(default_factory=list)
    labels: dict[str, SettlementLabel] = field(default_factory=dict)
    feature_names: tuple[str, ...] = FEATURE_NAMES
    feature_version: str = FEATURE_VERSION
    label_policy_version: str = LABEL_POLICY_VERSION
    data_version: str = ""
    source_label: str = ""
    config: dict[str, Any] = field(default_factory=dict)
    skipped: dict[str, int] = field(default_factory=dict)

    @property
    def labelled(self) -> list[TrainingExample]:
        """Only the examples that carry a settlement label."""
        return [e for e in self.examples if e.is_labelled]

    @property
    def market_ids(self) -> list[str]:
        """Markets with at least one labelled example, in first-seen order."""
        seen: dict[str, None] = {}
        for example in self.labelled:
            seen.setdefault(example.market_id, None)
        return list(seen)

    @property
    def effective_n(self) -> int:
        """Independent outcomes: the number of resolved markets.

        This, not ``len(labelled)``, is the sample size that governs what can
        be learned. One market forecast 400 times is one coin flip.
        """
        return len(self.market_ids)

    @property
    def base_rate(self) -> float | None:
        """Share of resolved markets that settled YES, one vote per market."""
        outcomes = self.market_outcomes()
        if not outcomes:
            return None
        return sum(outcomes.values()) / len(outcomes)

    def market_outcomes(self) -> dict[str, float]:
        """The single outcome of each resolved market."""
        return {
            market_id: label.value
            for market_id, label in self.labels.items()
            if label.is_usable and label.value is not None
        }

    def summary(self) -> dict[str, Any]:
        """Headline counts for display and readiness checks."""
        return {
            "rows_total": len(self.examples),
            "rows_labelled": len(self.labelled),
            "effective_n_markets": self.effective_n,
            "base_rate": self.base_rate,
            "feature_version": self.feature_version,
            "label_policy_version": self.label_policy_version,
            "data_version": self.data_version,
            "source": self.source_label,
            "label_status_counts": label_summary(self.labels),
            "rows_skipped": dict(self.skipped),
        }


def data_version_for(dataset: MarketDataset) -> str:
    """A stable content hash identifying exactly this input data.

    Derived from market identifiers, their settlement fields and the quote and
    trade counts, so any change to the underlying data produces a new version
    string and every artifact built from it is traceable.
    """
    digest = hashlib.sha256()
    digest.update(dataset.source.encode())
    for market in sorted(dataset.markets, key=lambda m: m.market_id):
        digest.update(market.market_id.encode())
        digest.update(str(market.outcome).encode())
        digest.update(str(market.resolution_time).encode())
        digest.update(str(market.close_time).encode())
    digest.update(str(len(dataset.quotes)).encode())
    digest.update(str(len(dataset.trades)).encode())
    return f"data-{digest.hexdigest()[:16]}"


def decision_timestamps(
    market: Market,
    quotes: Sequence[Quote],
    horizon_seconds: float = DEFAULT_HORIZON_SECONDS,
    stride_seconds: float = DEFAULT_STRIDE_SECONDS,
) -> list[datetime]:
    """The grid of instants at which a forecast is made for one market.

    Starts at the market's first observed quote and stops ``horizon_seconds``
    before close. The horizon matters: a forecast made a minute before
    settlement is nearly free, and including such rows would flatter every
    metric without representing a decision anyone could act on.

    Args:
        market: The market.
        quotes: Its quotes, sorted by timestamp.
        horizon_seconds: Minimum lead time between forecast and close.
        stride_seconds: Spacing of the grid.

    Returns:
        Decision timestamps in ascending order; empty if the market is shorter
        than the horizon.
    """
    if not quotes or stride_seconds <= 0:
        return []
    start = quotes[0].ts
    last = market.close_time - timedelta(seconds=horizon_seconds)
    if last <= start:
        return []
    stamps: list[datetime] = []
    current = start
    step = timedelta(seconds=stride_seconds)
    while current <= last:
        stamps.append(current)
        current += step
    return stamps


def build_feature_dataset(
    dataset: MarketDataset,
    config: dict[str, Any] | None = None,
    venue_statuses: dict[str, str] | None = None,
    cache: Any = None,
) -> FeatureDataset:
    """Build every causal feature row from a stored dataset.

    Args:
        dataset: Source markets, quotes and trades.
        config: The ``research.dataset`` settings section.
        venue_statuses: Raw venue status strings by market id, when available,
            so contested settlements can be excluded.
        cache: Optional :class:`research.featurecache.FeatureCache`. Features
            for one market are a pure function of that market's own series and
            the grid config, so they cache per market. Without it, a growing
            corpus recomputes everything on every pass - measured at hours for
            179 markets, which makes the autopilot unusable exactly when it
            starts working.

    Returns:
        A :class:`FeatureDataset` carrying its own provenance.
    """
    cfg = dict(config or {})
    horizon = float(cfg.get("horizon_seconds", DEFAULT_HORIZON_SECONDS))
    stride = float(cfg.get("stride_seconds", DEFAULT_STRIDE_SECONDS))

    labels = label_markets(list(dataset.markets), venue_statuses)
    quotes_by_market: dict[str, list[Quote]] = defaultdict(list)
    trades_by_market: dict[str, list[Trade]] = defaultdict(list)
    for quote in dataset.quotes:
        quotes_by_market[quote.market_id].append(quote)
    for trade in dataset.trades:
        trades_by_market[trade.market_id].append(trade)
    for series in quotes_by_market.values():
        series.sort(key=lambda q: q.ts)
    for series in trades_by_market.values():
        series.sort(key=lambda t: t.ts)

    examples: list[TrainingExample] = []
    skipped: dict[str, int] = defaultdict(int)

    for market in dataset.markets:
        quotes = quotes_by_market.get(market.market_id, [])
        trades = trades_by_market.get(market.market_id, [])
        label = labels[market.market_id]

        key: str | None = None
        if cache is not None:
            from research.featurecache import cache_key

            key = cache_key(
                market.market_id, quotes, trades, cfg, label.policy_version
            )
            cached = cache.get(key)
            if cached is not None:
                examples.extend(cached)
                continue

        stamps = decision_timestamps(market, quotes, horizon, stride)
        if not stamps:
            skipped["market_shorter_than_horizon"] += 1
            continue

        built: list[TrainingExample] = []
        for decision_ts in stamps:
            vector = build_features(market, quotes, trades, decision_ts, cfg)
            if vector is None:
                skipped["no_two_sided_book"] += 1
                continue

            # The label is attached only if it was knowable strictly after the
            # forecast. Equality is rejected too: a forecast made at the instant
            # of settlement is not a forecast.
            usable = label.is_usable and label.resolution_ts is not None
            if usable and label.resolution_ts <= decision_ts:
                skipped["outcome_already_known"] += 1
                usable = False

            built.append(
                TrainingExample(
                    market_id=market.market_id,
                    decision_ts=decision_ts,
                    features=vector,
                    label=label.value if usable else None,
                    resolution_ts=label.resolution_ts if usable else None,
                    market_prob=vector.values["market_prob"],
                )
            )

        examples.extend(built)
        if cache is not None and key is not None:
            cache.store(key, market.market_id, built)

    examples.sort(key=lambda e: (e.decision_ts, e.market_id))
    return FeatureDataset(
        examples=examples,
        labels=labels,
        data_version=data_version_for(dataset),
        source_label=f"{dataset.label} ({dataset.source})",
        config={"horizon_seconds": horizon, "stride_seconds": stride},
        skipped=dict(skipped),
    )


@dataclass(frozen=True)
class ReadinessCheck:
    """Whether there is enough resolved data to train anything meaningful."""

    ok: bool
    effective_n: int
    required_n: int
    blockers: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def as_rows(self) -> list[dict[str, Any]]:
        """Display rows for the dashboard."""
        rows = [
            {"check": "Resolved markets (independent outcomes)",
             "value": str(self.effective_n),
             "verdict": "OK" if self.effective_n >= self.required_n
             else f"BLOCKED - need {self.required_n}"},
        ]
        rows += [{"check": "Blocker", "value": b, "verdict": "BLOCKED"} for b in self.blockers]
        rows += [{"check": "Warning", "value": w, "verdict": "WARN"} for w in self.warnings]
        return rows


def check_readiness(
    dataset: FeatureDataset, config: dict[str, Any] | None = None
) -> ReadinessCheck:
    """Decide whether this data can support training, and say why not if it cannot.

    The bar is expressed in **resolved markets**, not rows, because rows from
    one market share a single outcome. This check gates training and promotion
    in the dashboard.
    """
    cfg = dict(config or {})
    required = int(cfg.get("min_resolved_markets", 50))
    min_per_class = int(cfg.get("min_markets_per_class", 10))

    blockers: list[str] = []
    warnings: list[str] = []
    effective_n = dataset.effective_n

    if effective_n < required:
        blockers.append(
            f"Only {effective_n} resolved market(s); at least {required} are needed "
            "before a fitted model can be taken seriously."
        )

    outcomes = list(dataset.market_outcomes().values())
    yes_markets = int(sum(outcomes))
    no_markets = len(outcomes) - yes_markets
    if outcomes and min(yes_markets, no_markets) < min_per_class:
        blockers.append(
            f"Outcome classes are too thin: {yes_markets} YES / {no_markets} NO markets, "
            f"minimum {min_per_class} of each."
        )

    if not dataset.labelled:
        blockers.append("No labelled examples at all: nothing to train on.")

    contested = sum(
        1 for label in dataset.labels.values() if label.status is LabelStatus.CONTESTED
    )
    if contested:
        warnings.append(f"{contested} market(s) excluded as disputed or amended.")

    no_time = sum(
        1 for label in dataset.labels.values()
        if label.status is LabelStatus.NO_SETTLEMENT_TIME
    )
    if no_time:
        warnings.append(
            f"{no_time} market(s) had an outcome but no settlement timestamp and were excluded."
        )

    if dataset.skipped.get("no_two_sided_book"):
        warnings.append(
            f"{dataset.skipped['no_two_sided_book']} row(s) dropped for having no two-sided book."
        )

    return ReadinessCheck(
        ok=not blockers,
        effective_n=effective_n,
        required_n=required,
        blockers=tuple(blockers),
        warnings=tuple(warnings),
    )


def dataset_fingerprint(dataset: FeatureDataset) -> str:
    """Stable hash over the built dataset, for reproducibility records."""
    payload = json.dumps(
        {
            "data_version": dataset.data_version,
            "feature_version": dataset.feature_version,
            "label_policy_version": dataset.label_policy_version,
            "config": dataset.config,
            "rows": len(dataset.examples),
            "markets": sorted(dataset.market_ids),
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]
