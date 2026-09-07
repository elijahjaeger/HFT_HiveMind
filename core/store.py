"""DuckDB-backed local storage for the synthetic DEMO DATA.

The store is a convenience layer: it turns the in-memory dataset into pandas
frames, persists them in a local DuckDB file under ``data/processed`` and reads
them back. There is no network component and no credential handling.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
from pydantic import ValidationError

from core.config import resolve_path
from core.models import Market, Quote, Trade
from core.synthetic import DemoDataset, generate_demo_dataset

_MODEL_FIELDS = {
    "markets": Market,
    "quotes": Quote,
    "trades": Trade,
}


def _records(models: list[Any]) -> list[dict[str, Any]]:
    """Dump pydantic models to plain dicts with enum values as strings."""
    return [m.model_dump(mode="json") for m in models]


def _clean_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Rows as dicts with pandas nulls (``NaN``/``NaT``) turned back into ``None``.

    A one-sided book stores ``NULL`` for the missing side; pandas reads that
    back as ``NaN``, which the models reject. This restores ``None``.
    """
    rows: list[dict[str, Any]] = []
    for record in frame.to_dict("records"):
        clean: dict[str, Any] = {}
        for key, value in record.items():
            try:
                clean[key] = None if value is None or pd.isna(value) else value
            except (TypeError, ValueError):  # non-scalar values are kept as-is
                clean[key] = value
        rows.append(clean)
    return rows


def dataset_to_frames(dataset: DemoDataset) -> dict[str, pd.DataFrame]:
    """Convert a :class:`DemoDataset` into ``markets``/``quotes``/``trades`` frames."""
    frames = {
        "markets": pd.DataFrame(_records(dataset.markets)),
        "quotes": pd.DataFrame(_records(dataset.quotes)),
        "trades": pd.DataFrame(_records(dataset.trades)),
    }
    for name, frame in frames.items():
        for column in ("ts", "created_at", "close_time", "resolution_time"):
            if column in frame.columns:
                frame[column] = pd.to_datetime(frame[column], utc=True, format="ISO8601")
        frames[name] = frame
    return frames


def write_dataset(dataset: DemoDataset, db_path: str | Path) -> Path:
    """Persist a demo dataset into a local DuckDB file, replacing any contents.

    Args:
        dataset: The dataset to store.
        db_path: Destination ``.duckdb`` file (relative paths resolve to the
            project root).

    Returns:
        The resolved path of the DuckDB file.
    """
    path = resolve_path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = dataset_to_frames(dataset)
    with duckdb.connect(str(path)) as con:
        for name, frame in frames.items():
            con.register(f"_tmp_{name}", frame)
            con.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM _tmp_{name}")
            con.unregister(f"_tmp_{name}")
        con.execute(
            "CREATE OR REPLACE TABLE meta AS SELECT ? AS seed, ? AS label",
            [dataset.seed, dataset.label],
        )
    return path


def read_dataset(db_path: str | Path) -> DemoDataset:
    """Load a demo dataset previously written by :func:`write_dataset`."""
    path = resolve_path(db_path)
    if not path.exists():
        raise FileNotFoundError(f"no DuckDB file at {path}")
    with duckdb.connect(str(path), read_only=True) as con:
        frames = {
            name: con.execute(f"SELECT * FROM {name}").fetch_df()
            for name in _MODEL_FIELDS
        }
        meta = con.execute("SELECT * FROM meta").fetch_df()

    markets = [Market.model_validate(r) for r in _clean_records(frames["markets"])]
    quotes = [Quote.model_validate(r) for r in _clean_records(frames["quotes"])]
    trades = [Trade.model_validate(r) for r in _clean_records(frames["trades"])]
    quotes.sort(key=lambda q: (q.ts, q.market_id))
    trades.sort(key=lambda t: (t.ts, t.market_id))
    seed = int(meta["seed"].iloc[0]) if not meta.empty else 0
    return DemoDataset(markets=markets, quotes=quotes, trades=trades, seed=seed)


def load_or_create_dataset(settings: dict[str, Any], refresh: bool = False) -> DemoDataset:
    """Return the demo dataset, generating and caching it in DuckDB if needed.

    Args:
        settings: Full settings dictionary (uses the ``data`` section).
        refresh: When True, regenerate and overwrite the cached DuckDB file.
    """
    data_cfg = settings.get("data", {})
    db_path = resolve_path(data_cfg.get("duckdb_path", "data/processed/demo.duckdb"))
    if not refresh and db_path.exists():
        try:
            return read_dataset(db_path)
        except (duckdb.Error, FileNotFoundError, KeyError, ValidationError):
            pass  # Corrupt or stale cache: fall through and regenerate.
    dataset = generate_demo_dataset(data_cfg)
    write_dataset(dataset, db_path)
    return dataset


def query(db_path: str | Path, sql: str) -> pd.DataFrame:
    """Run a read-only SQL query against the local DuckDB demo store."""
    path = resolve_path(db_path)
    with duckdb.connect(str(path), read_only=True) as con:
        return con.execute(sql).fetch_df()
