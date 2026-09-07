"""BookImbalanceAgent - a size-weighted microprice with declared provenance.

The signal
----------
The midpoint treats both sides of the book as equally informative. The
*microprice* weights each side by the size resting on the other::

    microprice = (bid * ask_size + ask * bid_size) / (bid_size + ask_size)

A heavy bid pushes the estimate toward the ask (buyers are queuing), a heavy
ask pushes it toward the bid. The estimate always lies inside the spread, so on
its own it can never justify crossing that spread; its role is to tilt the
ensemble, not to trade alone.

What this agent guarantees
--------------------------
* **Declared inputs.** Every forecast lists the data it used and the timestamp
  each item carried. There is exactly one input - the quote - so the audit is
  trivial: ``quote.ts <= as_of`` or the agent refuses to forecast.
* **Exposed assumptions.** The modelling assumptions are listed on every
  forecast, not buried in a docstring.
* **Fails safely.** No quote, a one-sided book, an empty book or a quote past
  the hard staleness cut-off all produce ``probability=None`` with a
  machine-readable ``status`` and a plain-language reason. A quote that is
  stale but inside the cut-off still forecasts, with confidence scaled down
  and ``status=DEGRADED``. Nothing here raises on missing data.
* **No look-ahead.** A quote stamped after ``as_of`` is rejected outright,
  independently of the engine's own guarantees.
"""

from __future__ import annotations

from typing import Any

from agents.base import ForecastAgent
from core.models import Forecast, ForecastInput, ForecastStatus, MarketState, Quote

ASSUMPTIONS: tuple[str, ...] = (
    "Displayed size reflects genuine intent to trade; hidden or iceberg liquidity is ignored.",
    "Only the top level of the book is observed; deeper levels are treated as empty.",
    "Fair value lies inside the current spread, weighted linearly by opposite-side size.",
    "A binary contract paying $1.00 on YES, so a price is directly a probability.",
    "Book imbalance is informative on the horizon of the next few quotes only.",
)


def microprice(quote: Quote | None) -> float | None:
    """Size-weighted price inside the spread, or ``None`` if not computable.

    Requires both sides of the book and positive total displayed size.
    """
    if quote is None or quote.bid is None or quote.ask is None:
        return None
    total = quote.bid_size + quote.ask_size
    if total <= 0.0:
        return None
    value = (quote.bid * quote.ask_size + quote.ask * quote.bid_size) / total
    return max(0.0, min(1.0, value))


def imbalance(quote: Quote | None) -> float | None:
    """Signed size imbalance in ``[-1, 1]``: positive when the bid is heavier."""
    if quote is None or quote.bid is None or quote.ask is None:
        return None
    total = quote.bid_size + quote.ask_size
    if total <= 0.0:
        return None
    return (quote.bid_size - quote.ask_size) / total


class BookImbalanceAgent(ForecastAgent):
    """Forecasts the YES probability from the size-weighted microprice."""

    name = "BookImbalanceAgent"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.base_confidence = float(self.config.get("confidence", 0.5))
        self.reference_depth = float(self.config.get("reference_depth", 200.0))
        self.max_data_age_seconds = float(self.config.get("max_data_age_seconds", 900.0))
        self.hard_cutoff_seconds = float(self.config.get("hard_cutoff_seconds", 3600.0))

    def analyze(self, state: MarketState) -> Forecast:
        """Return a microprice forecast, or a safe non-forecast with a reason.

        Args:
            state: Market state at the decision instant. Only ``state.quote``
                is read.

        Returns:
            A :class:`core.models.Forecast` whose ``inputs`` declare the quote
            timestamp, whose ``assumptions`` list the modelling assumptions,
            and whose ``status`` says whether a probability was produced.
        """
        quote = state.quote
        market_id = state.market.market_id
        inputs = (
            [ForecastInput(name="quote", ts=quote.ts, source=quote.source)]
            if quote is not None
            else []
        )

        if quote is None:
            return self._no_forecast(
                state, inputs, ForecastStatus.NO_DATA, "No quote has been observed."
            )

        if quote.ts > state.as_of:
            return self._no_forecast(
                state,
                inputs,
                ForecastStatus.REJECTED_FUTURE_INPUT,
                f"Quote is stamped {quote.ts.isoformat()} which is after the decision "
                f"instant {state.as_of.isoformat()}; refusing to use future data.",
            )

        if not quote.is_two_sided:
            side = "bid" if quote.bid is not None else "ask"
            return self._no_forecast(
                state,
                inputs,
                ForecastStatus.NO_DATA,
                f"One-sided book ({side} only): imbalance cannot be computed.",
            )

        price = microprice(quote)
        skew = imbalance(quote)
        if price is None or skew is None:
            return self._no_forecast(
                state,
                inputs,
                ForecastStatus.NO_DATA,
                "Both sides quoted but no displayed size on either.",
            )

        age = quote.age_seconds(state.as_of)
        if age > self.hard_cutoff_seconds:
            return self._no_forecast(
                state,
                inputs,
                ForecastStatus.NO_DATA,
                f"Quote is {age:.0f}s old, past the hard cut-off of "
                f"{self.hard_cutoff_seconds:.0f}s.",
            )

        depth = min(quote.bid_size, quote.ask_size)
        depth_factor = min(1.0, depth / self.reference_depth) if self.reference_depth > 0 else 1.0
        if age <= self.max_data_age_seconds:
            freshness = 1.0
            status = ForecastStatus.OK
        else:
            span = max(1.0, self.hard_cutoff_seconds - self.max_data_age_seconds)
            freshness = max(0.0, 1.0 - (age - self.max_data_age_seconds) / span)
            status = ForecastStatus.DEGRADED
        confidence = max(0.0, min(1.0, self.base_confidence * depth_factor * freshness))

        rationale = (
            f"Microprice {price:.4f} vs mid {quote.mid:.4f}; imbalance {skew:+.2f} "
            f"(bid {quote.bid_size:.0f} / ask {quote.ask_size:.0f}); quote age {age:.0f}s."
        )
        if status is ForecastStatus.DEGRADED:
            rationale += f" DEGRADED: older than {self.max_data_age_seconds:.0f}s."

        return Forecast(
            agent_name=self.name,
            market_id=market_id,
            ts=state.as_of,
            probability=price,
            confidence=confidence,
            is_placeholder=False,
            rationale=rationale,
            status=status,
            inputs=inputs,
            assumptions=list(ASSUMPTIONS),
            details={
                "microprice": price,
                "mid": quote.mid,
                "imbalance": skew,
                "bid_size": quote.bid_size,
                "ask_size": quote.ask_size,
                "depth_factor": depth_factor,
                "freshness": freshness,
                "quote_age_seconds": age,
            },
        )

    def _no_forecast(
        self,
        state: MarketState,
        inputs: list[ForecastInput],
        status: ForecastStatus,
        reason: str,
    ) -> Forecast:
        """A structured non-forecast: no probability, zero confidence, a reason."""
        return Forecast(
            agent_name=self.name,
            market_id=state.market.market_id,
            ts=state.as_of,
            probability=None,
            confidence=0.0,
            is_placeholder=False,
            rationale=reason,
            status=status,
            inputs=inputs,
            assumptions=list(ASSUMPTIONS),
            details={"reason": reason},
        )
