"""MarketAgent - reads the implied probability out of the order book.

For a binary contract that pays 1 on YES, the price *is* the market-implied
probability. This agent therefore reports the bid/ask midpoint, degrading
gracefully when the book is one-sided or missing.
"""

from __future__ import annotations

from typing import Any

from agents.base import ForecastAgent
from core.models import Forecast, MarketState, Quote


def midpoint(bid: float | None, ask: float | None) -> float | None:
    """Return the bid/ask midpoint, or ``None`` if either side is missing."""
    if bid is None or ask is None:
        return None
    return (bid + ask) / 2.0


def implied_probability(quote: Quote | None, one_sided_adjustment: float = 0.01) -> float | None:
    """Convert a quote into a market-implied probability in ``[0, 1]``.

    A two-sided book yields the midpoint. A one-sided book is nudged by
    ``one_sided_adjustment`` away from the visible side (a lone bid is a lower
    bound on fair value, a lone ask an upper bound). An empty book yields
    ``None``.

    Args:
        quote: The most recent quote, or ``None``.
        one_sided_adjustment: Nudge applied when only one side is present.

    Returns:
        The implied probability, or ``None`` when no price is available.
    """
    if quote is None:
        return None
    mid = midpoint(quote.bid, quote.ask)
    if mid is not None:
        return max(0.0, min(1.0, mid))
    if quote.bid is not None:
        return max(0.0, min(1.0, quote.bid + one_sided_adjustment))
    if quote.ask is not None:
        return max(0.0, min(1.0, quote.ask - one_sided_adjustment))
    return None


class MarketAgent(ForecastAgent):
    """Derives the implied probability from the current bid/ask."""

    name = "MarketAgent"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.base_confidence = float(self.config.get("confidence", 0.6))
        self.one_sided_adjustment = float(self.config.get("one_sided_adjustment", 0.01))

    def analyze(self, state: MarketState) -> Forecast:
        """Report the book-implied probability for the YES outcome.

        Args:
            state: Market state at the decision instant.

        Returns:
            A :class:`core.models.Forecast`; ``probability`` is ``None`` when
            the book carries no usable price.
        """
        quote = state.quote
        probability = implied_probability(quote, self.one_sided_adjustment)
        two_sided = bool(quote and quote.is_two_sided)
        spread = quote.spread if quote else None

        if probability is None:
            confidence = 0.0
            rationale = "No quote available; the book carries no price."
        elif two_sided:
            confidence = self.base_confidence
            rationale = f"Midpoint of bid/ask ({quote.bid:.3f} / {quote.ask:.3f})."
        else:
            confidence = self.base_confidence * 0.5
            side = "bid" if quote and quote.bid is not None else "ask"
            rationale = f"One-sided book: {side} only, adjusted by {self.one_sided_adjustment:.3f}."

        return Forecast(
            agent_name=self.name,
            market_id=state.market.market_id,
            ts=state.as_of,
            probability=probability,
            confidence=max(0.0, min(1.0, confidence)),
            is_placeholder=False,
            rationale=rationale,
            details={
                "bid": quote.bid if quote else None,
                "ask": quote.ask if quote else None,
                "mid": midpoint(quote.bid, quote.ask) if quote else None,
                "spread": spread,
                "two_sided": two_sided,
                "quote_age_seconds": quote.age_seconds(state.as_of) if quote else None,
            },
        )
