"""LiquidityAgent - measures spread, depth and staleness.

The report it produces is advisory: it describes how tradable the book looks.
Only the :class:`agents.risk.RiskAgent` can block a trade.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from agents.base import Agent
from core.models import LiquidityReport, MarketState, Quote


def compute_spread(quote: Quote | None) -> float | None:
    """Absolute bid/ask width in probability units, or ``None`` if one-sided."""
    if quote is None or quote.bid is None or quote.ask is None:
        return None
    return quote.ask - quote.bid


def compute_depth(quote: Quote | None) -> float:
    """Tradable depth: the smaller of the two displayed sizes."""
    if quote is None:
        return 0.0
    if quote.bid is None or quote.ask is None:
        # A one-sided book only supports one direction; report what is there.
        return quote.ask_size if quote.bid is None else quote.bid_size
    return min(quote.bid_size, quote.ask_size)


def data_age_seconds(quote: Quote | None, as_of: datetime) -> float | None:
    """Age of the quote in seconds at ``as_of``, or ``None`` when absent."""
    if quote is None:
        return None
    return quote.age_seconds(as_of)


def is_stale(quote: Quote | None, as_of: datetime, max_age_seconds: float) -> bool:
    """True when the quote is missing or older than ``max_age_seconds``."""
    age = data_age_seconds(quote, as_of)
    return age is None or age > max_age_seconds


class LiquidityAgent(Agent):
    """Reports spread, depth, data age and human-readable liquidity warnings."""

    name = "LiquidityAgent"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.max_data_age_seconds = float(self.config.get("max_data_age_seconds", 900))
        self.warn_spread = float(self.config.get("warn_spread", 0.04))
        self.warn_depth = float(self.config.get("warn_depth", 50))

    def analyze(self, state: MarketState) -> LiquidityReport:
        """Assess how tradable the current book is.

        Args:
            state: Market state at the decision instant.

        Returns:
            A :class:`core.models.LiquidityReport` with warnings attached.
        """
        quote = state.quote
        spread = compute_spread(quote)
        depth = compute_depth(quote)
        age = data_age_seconds(quote, state.as_of)
        stale = is_stale(quote, state.as_of, self.max_data_age_seconds)

        warnings: list[str] = []
        if quote is None:
            warnings.append("NO_QUOTE: no book has been observed for this market.")
        else:
            if not quote.is_two_sided:
                warnings.append("ONE_SIDED_BOOK: only one side of the book is quoted.")
            if stale:
                warnings.append(
                    f"STALE_DATA: quote is {age:.0f}s old "
                    f"(limit {self.max_data_age_seconds:.0f}s)."
                )
            if spread is not None and spread > self.warn_spread:
                warnings.append(
                    f"WIDE_SPREAD: {spread:.3f} exceeds warning level {self.warn_spread:.3f}."
                )
            if depth < self.warn_depth:
                warnings.append(
                    f"THIN_DEPTH: {depth:.0f} contracts below warning level {self.warn_depth:.0f}."
                )

        return LiquidityReport(
            agent_name=self.name,
            market_id=state.market.market_id,
            ts=state.as_of,
            spread=spread,
            mid=quote.mid if quote else None,
            depth=depth,
            bid_size=quote.bid_size if quote else 0.0,
            ask_size=quote.ask_size if quote else 0.0,
            data_age_seconds=age,
            is_stale=stale,
            is_two_sided=bool(quote and quote.is_two_sided),
            warnings=warnings,
            details={
                "max_data_age_seconds": self.max_data_age_seconds,
                "warn_spread": self.warn_spread,
                "warn_depth": self.warn_depth,
            },
        )
