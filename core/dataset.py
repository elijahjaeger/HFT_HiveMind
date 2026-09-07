"""The dataset container shared by every data source.

Both the synthetic generator and the read-only venue adapters produce a
:class:`MarketDataset`. The backtester only ever sees this type, so nothing
downstream needs to know where the records came from - only the ``label`` and
``source`` fields record the provenance, and they are surfaced in the UI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from core.models import Market, Quote, Trade

DEMO_LABEL = "DEMO DATA"
"""Label for fabricated data. Must be shown wherever the data is shown."""


@dataclass(frozen=True)
class MarketDataset:
    """Markets, quotes and trades from one source, sorted chronologically."""

    markets: list[Market]
    quotes: list[Quote]
    trades: list[Trade]
    seed: int | None = None
    label: str = DEMO_LABEL
    source: str = "SYNTHETIC_DEMO"
    venue: str | None = None
    fetched_at: datetime | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def market_ids(self) -> list[str]:
        """Identifiers of every market in the dataset."""
        return [m.market_id for m in self.markets]

    @property
    def is_demo(self) -> bool:
        """True when this dataset is fabricated rather than fetched."""
        return self.label == DEMO_LABEL

    def quotes_for(self, market_id: str) -> list[Quote]:
        """Chronological quotes for a single market."""
        return [q for q in self.quotes if q.market_id == market_id]

    def trades_for(self, market_id: str) -> list[Trade]:
        """Chronological tape prints for a single market."""
        return [t for t in self.trades if t.market_id == market_id]

    def market(self, market_id: str) -> Market:
        """Look up one market by identifier."""
        for candidate in self.markets:
            if candidate.market_id == market_id:
                return candidate
        raise KeyError(market_id)

    def summary(self) -> dict[str, Any]:
        """Small provenance summary for display."""
        return {
            "label": self.label,
            "source": self.source,
            "venue": self.venue or "-",
            "markets": len(self.markets),
            "quotes": len(self.quotes),
            "trades": len(self.trades),
            "seed": self.seed,
            "notes": list(self.notes),
        }


def sorted_dataset(dataset: MarketDataset) -> MarketDataset:
    """Return a copy with quotes and trades in strict chronological order."""
    quotes = sorted(dataset.quotes, key=lambda q: (q.ts, q.market_id))
    trades = sorted(dataset.trades, key=lambda t: (t.ts, t.market_id))
    return MarketDataset(
        markets=list(dataset.markets),
        quotes=quotes,
        trades=trades,
        seed=dataset.seed,
        label=dataset.label,
        source=dataset.source,
        venue=dataset.venue,
        fetched_at=dataset.fetched_at,
        notes=list(dataset.notes),
    )
