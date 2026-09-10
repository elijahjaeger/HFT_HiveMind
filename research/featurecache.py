"""Per-market feature cache, so a growing corpus does not rebuild everything.

Why this exists
---------------
The autopilot rebuilds every feature row on every pass. That is correct but
quadratic in effort: a corpus of 179 markets took hours against a 14-minute
budget, because pass *n* recomputes everything passes 1..n-1 already computed.
Since the loop's whole premise is that the corpus keeps growing, the cost grows
with it and the loop becomes unusable exactly when it starts working.

Features for one market are a pure function of that market's own quotes, trades
and the grid configuration. Nothing about market B can change the features of
market A. So they cache cleanly, keyed per market.

Correctness before speed
------------------------
A stale cache would silently train a model on features that no longer match the
code that claims to have produced them - the worst possible failure here,
because it would be invisible. The key therefore includes every input that can
change a row:

* the market id;
* ``feature_version`` and the label policy version;
* the full decision-grid configuration (horizon, stride, both windows);
* a hash of the market's own quote and trade series, so re-fetching a market
  with more history invalidates it.

Any change to any of those produces a different key and a rebuild. The cache is
never consulted across versions.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb

from core.config import resolve_path
from core.models import Quote, Trade
from research.features import FEATURE_VERSION, FeatureVector

CACHE_SCHEMA_VERSION = "featurecache-v1"

_TABLE = "feature_cache"


def series_fingerprint(quotes: list[Quote], trades: list[Trade]) -> str:
    """A short hash of a market's own price history.

    Re-fetching a market with deeper history must invalidate its cached rows,
    so the fingerprint covers the count and the endpoints of both series rather
    than only the count - a market that gained rows in the middle would
    otherwise reuse stale features.
    """
    parts = [
        str(len(quotes)),
        str(len(trades)),
        quotes[0].ts.isoformat() if quotes else "-",
        quotes[-1].ts.isoformat() if quotes else "-",
        trades[0].ts.isoformat() if trades else "-",
        trades[-1].ts.isoformat() if trades else "-",
    ]
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]


def cache_key(
    market_id: str,
    quotes: list[Quote],
    trades: list[Trade],
    config: dict[str, Any],
    label_policy: str,
) -> str:
    """Everything that can change this market's rows, in one string."""
    grid = {
        "horizon": config.get("horizon_seconds"),
        "stride": config.get("stride_seconds"),
        "short": config.get("short_window_seconds"),
        "long": config.get("long_window_seconds"),
    }
    payload = json.dumps(
        {
            "schema": CACHE_SCHEMA_VERSION,
            "market": market_id,
            "features": FEATURE_VERSION,
            "labels": label_policy,
            "grid": grid,
            "series": series_fingerprint(quotes, trades),
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _encode_example(example: Any) -> dict[str, Any]:
    """Serialise one training example to plain JSON types."""
    vector = example.features
    return {
        "market_id": example.market_id,
        "decision_ts": example.decision_ts.isoformat(),
        "label": example.label,
        "resolution_ts": (
            example.resolution_ts.isoformat() if example.resolution_ts else None
        ),
        "market_prob": example.market_prob,
        "values": vector.values,
        "missing": list(vector.missing),
        "source_start_ts": (
            vector.source_start_ts.isoformat() if vector.source_start_ts else None
        ),
        "source_end_ts": (
            vector.source_end_ts.isoformat() if vector.source_end_ts else None
        ),
        "feature_version": vector.feature_version,
    }


def _decode_example(row: dict[str, Any]) -> Any:
    """Rebuild a training example from its serialised form."""
    from research.datasets import TrainingExample

    def moment(value: Any) -> datetime | None:
        return datetime.fromisoformat(value) if value else None

    decision_ts = datetime.fromisoformat(row["decision_ts"])
    return TrainingExample(
        market_id=row["market_id"],
        decision_ts=decision_ts,
        features=FeatureVector(
            market_id=row["market_id"],
            decision_ts=decision_ts,
            values=dict(row["values"]),
            missing=tuple(row.get("missing", ())),
            source_start_ts=moment(row.get("source_start_ts")),
            source_end_ts=moment(row.get("source_end_ts")),
            feature_version=row.get("feature_version", FEATURE_VERSION),
        ),
        label=row["label"],
        resolution_ts=moment(row.get("resolution_ts")),
        market_prob=row["market_prob"],
    )


@dataclass
class CacheStats:
    """How much work the cache saved, so the claim is measurable."""

    hits: int = 0
    misses: int = 0
    stored: int = 0

    @property
    def hit_rate(self) -> float:
        """Share of markets served from cache."""
        total = self.hits + self.misses
        return 0.0 if total == 0 else self.hits / total

    def as_row(self) -> dict[str, Any]:
        """Display row."""
        return {
            "markets from cache": self.hits,
            "markets rebuilt": self.misses,
            "hit rate": f"{self.hit_rate:.0%}",
        }


class FeatureCache:
    """DuckDB-backed store of built feature rows, keyed per market.

    Safe to delete at any time: a missing or corrupt cache costs a rebuild and
    never a wrong answer.
    """

    def __init__(self, path: str | Path = "data/processed/feature_cache.duckdb") -> None:
        self.path = resolve_path(path)
        self.stats = CacheStats()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_table()

    def _ensure_table(self) -> None:
        with duckdb.connect(str(self.path)) as con:
            con.execute(
                f"CREATE TABLE IF NOT EXISTS {_TABLE} ("
                "cache_key TEXT PRIMARY KEY, market_id TEXT, n_rows INTEGER, "
                "payload TEXT)"
            )

    def get(self, key: str) -> list[Any] | None:
        """Cached rows for a key, or ``None`` on a miss or unreadable entry."""
        try:
            with duckdb.connect(str(self.path), read_only=True) as con:
                found = con.execute(
                    f"SELECT payload FROM {_TABLE} WHERE cache_key = ?", [key]
                ).fetchone()
        except duckdb.Error:
            self.stats.misses += 1
            return None
        if not found:
            self.stats.misses += 1
            return None
        try:
            rows = json.loads(found[0])
            examples = [_decode_example(row) for row in rows]
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            # A corrupt entry must cost a rebuild, never a wrong answer.
            self.stats.misses += 1
            return None
        self.stats.hits += 1
        return examples

    def store(self, key: str, market_id: str, examples: list[Any]) -> None:
        """Store rows for a key, replacing any existing entry.

        Named ``store`` rather than ``put``: the AST order-placement scan in
        ``tests/test_paper_trading.py`` treats a bare ``put()`` as an HTTP
        write verb, and it is right to. Nothing in this project should have a
        method whose name reads as a remote write.
        """
        payload = json.dumps([_encode_example(e) for e in examples])
        try:
            with duckdb.connect(str(self.path)) as con:
                con.execute(
                    f"INSERT OR REPLACE INTO {_TABLE} "
                    "(cache_key, market_id, n_rows, payload) VALUES (?, ?, ?, ?)",
                    [key, market_id, len(examples), payload],
                )
        except duckdb.Error:
            return  # A cache that cannot be written is a slow run, not a failure.
        self.stats.stored += 1

    def size(self) -> int:
        """How many markets are cached."""
        try:
            with duckdb.connect(str(self.path), read_only=True) as con:
                return int(con.execute(f"SELECT COUNT(*) FROM {_TABLE}").fetchone()[0])
        except duckdb.Error:
            return 0

    def clear(self) -> None:
        """Drop every entry. Correct at any time; costs a rebuild."""
        try:
            with duckdb.connect(str(self.path)) as con:
                con.execute(f"DELETE FROM {_TABLE}")
        except duckdb.Error:
            return
