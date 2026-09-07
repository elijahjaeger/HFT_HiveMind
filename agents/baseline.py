"""BaseRateAgent - an explicit placeholder, not a forecaster.

This agent exists to hold the seat that a real base-rate model (reference
classes, historical frequencies, or a calibrated ensemble) would occupy later.
It returns a constant from the configuration and marks the result
``is_placeholder=True`` so downstream code and the dashboard can label it
clearly. It is excluded from the blended probability by default.
"""

from __future__ import annotations

from typing import Any

from agents.base import ForecastAgent
from core.models import Forecast, MarketState

PLACEHOLDER_NOTE = (
    "PLACEHOLDER - constant from config, not a forecast. "
    "No historical base rates are modelled yet."
)


class BaseRateAgent(ForecastAgent):
    """Returns a clearly marked placeholder probability."""

    name = "BaseRateAgent"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.placeholder_probability = float(
            self.config.get("placeholder_probability", 0.5)
        )
        self.confidence = float(self.config.get("confidence", 0.05))

    def analyze(self, state: MarketState) -> Forecast:
        """Return the configured placeholder probability for this market.

        Args:
            state: Market state at the decision instant (ignored except for
                identifiers - the value is constant by design).

        Returns:
            A :class:`core.models.Forecast` with ``is_placeholder=True``.
        """
        return Forecast(
            agent_name=self.name,
            market_id=state.market.market_id,
            ts=state.as_of,
            probability=self.placeholder_probability,
            confidence=self.confidence,
            is_placeholder=True,
            rationale=PLACEHOLDER_NOTE,
            details={
                "category": state.market.category,
                "source": "config:agents.base_rate.placeholder_probability",
            },
        )
