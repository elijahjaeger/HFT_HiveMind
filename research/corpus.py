"""Building a research corpus, and being honest about where it comes from.

Two sources are supported:

* **Synthetic DEMO DATA** - a seeded, reproducible corpus shaped so the
  walk-forward protocol has something to work with. It is fabricated. It can
  demonstrate that the machinery is correct; it can say nothing whatsoever
  about whether a strategy works.
* **Kalshi historical data** - real settled markets pulled through the existing
  read-only adapter and cached locally. This is the only source whose results
  mean anything, and it is the source the pipeline is designed for.

Why the synthetic corpus is shaped the way it is
------------------------------------------------

The default demo dataset runs its markets concurrently: around fifty are open
at once. That is fine for the trading dashboard, but it makes a rigorous
walk-forward split nearly empty, because almost every training market is still
unresolved when the test window opens and is therefore purged. That is the
purge doing its job, not a bug.

A research corpus therefore uses short-lived markets spread over a long span,
so markets resolve in a rolling sequence and there is genuine chronological
separation to train on. :data:`RESEARCH_CORPUS` encodes that shape.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.dataset import MarketDataset
from core.synthetic import generate_demo_dataset

RESEARCH_CORPUS: dict[str, Any] = {
    "seed": 20260101,
    "n_markets": 200,
    "n_steps": 4000,
    "lifetime_fraction": 0.03,
    "stagger_fraction": 0.95,
    "quote_interval_seconds": 300,
    "base_spread": 0.015,
    "spread_jitter": 0.015,
}
"""Shape of the synthetic research corpus: many short markets over a long span."""


@dataclass(frozen=True)
class CorpusInfo:
    """Where a research corpus came from and what it can support."""

    label: str
    source: str
    is_synthetic: bool
    n_markets: int
    n_quotes: int
    n_trades: int
    max_concurrent_markets: int
    caveat: str

    def as_rows(self) -> list[dict[str, Any]]:
        """Display rows for the dashboard."""
        return [
            {"field": "Source", "value": self.label},
            {"field": "Real or fabricated",
             "value": "FABRICATED - demo only" if self.is_synthetic else "Real venue data"},
            {"field": "Markets", "value": str(self.n_markets)},
            {"field": "Quotes", "value": str(self.n_quotes)},
            {"field": "Trades", "value": str(self.n_trades)},
            {"field": "Peak concurrent markets", "value": str(self.max_concurrent_markets)},
            {"field": "What results mean", "value": self.caveat},
        ]


SYNTHETIC_CAVEAT = (
    "Results on this corpus test the PIPELINE, not a strategy. The data is "
    "fabricated by a seeded generator, so any edge measured on it is a property "
    "of the generator and tells you nothing about real markets."
)

REAL_CAVEAT = (
    "Real settled Kalshi markets. Results are still an out-of-sample simulation, "
    "not evidence of executable profit."
)


def max_concurrent_markets(dataset: MarketDataset) -> int:
    """Largest number of markets open simultaneously.

    High overlap is what makes purging expensive, so this number explains fold
    sizes before anyone has to go read the walk-forward code.
    """
    events: list[tuple[Any, int]] = []
    for market in dataset.markets:
        events.append((market.created_at, 1))
        events.append((market.close_time, -1))
    events.sort()
    current = peak = 0
    for _, delta in events:
        current += delta
        peak = max(peak, current)
    return peak


def describe(dataset: MarketDataset) -> CorpusInfo:
    """Summarise a corpus and state plainly what its results can support."""
    synthetic = dataset.is_demo or "SYNTHETIC" in dataset.source.upper()
    return CorpusInfo(
        label=f"{dataset.label} ({dataset.source})",
        source=dataset.source,
        is_synthetic=synthetic,
        n_markets=len(dataset.markets),
        n_quotes=len(dataset.quotes),
        n_trades=len(dataset.trades),
        max_concurrent_markets=max_concurrent_markets(dataset),
        caveat=SYNTHETIC_CAVEAT if synthetic else REAL_CAVEAT,
    )


def build_research_corpus(overrides: dict[str, Any] | None = None) -> MarketDataset:
    """Generate the reproducible synthetic research corpus.

    Args:
        overrides: Settings overriding :data:`RESEARCH_CORPUS`.

    Returns:
        A seeded :class:`core.dataset.MarketDataset` labelled DEMO DATA.
    """
    config = dict(RESEARCH_CORPUS)
    config.update(overrides or {})
    return generate_demo_dataset(config)
