"""The threshold trading policy, expressed in the backtester's own terms.

The policy is deliberately blunt: trade only when the calibrated forecast
disagrees with the executable price by more than every modelled cost plus a
safety margin. Otherwise stay flat.

**There is one implementation of that rule, and it is not here.**
:func:`strategy_config_for` translates the policy into the configuration
``agents.strategy.build_proposal`` already reads, so the simulator charges
exactly the costs the policy assumed. A second, parallel implementation used to
live in this module; it was reachable only from its own unit tests, and it
measured edge against the midpoint while the shipped path measures it against
the crossing price - so the two disagreed about which trades existed. Removed
2026-09-08; see DECISIONS.md.

The costs are stacked conservatively, because each one is a real deduction that
a naive backtest quietly omits:

* **half the spread** - you cross it to get in;
* **fees** - charged on the executed price;
* **slippage** - modelled impact against displayed depth;
* **a safety margin** - explicit compensation for everything not modelled:
  queue position, adverse selection, and the fact that the forecast itself is
  an estimate rather than a known quantity.

Nothing here places an order. It produces a decision the existing simulator
consumes, and the simulator writes only to an in-memory portfolio.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from agents.base import ForecastAgent
from backtest.fills import fee_per_contract
from core.models import Forecast, ForecastInput, ForecastStatus, MarketState
from research.models import ProbabilityForecast

POLICY_VERSION = "threshold-v1"


@dataclass
class PolicyConfig:
    """Thresholds for the trading policy."""

    safety_margin: float = 0.02
    """Extra edge demanded beyond modelled costs, in probability units."""

    min_confidence: float = 0.0
    """Forecasts below this confidence are not traded."""

    max_quote_age_seconds: float = 900.0
    reference_size: float = 100.0
    """Size used when estimating slippage for the threshold calculation."""


class ModelForecastAgent(ForecastAgent):
    """Serves stored out-of-sample model forecasts to the backtester.

    The engine asks agents for a view at ``state.as_of``. This agent replies
    with the forecast whose decision timestamp is the latest **at or before**
    that instant, and never one from the future. If there is no such forecast
    it reports ``NO_DATA`` rather than guessing, which is what produces an
    honest ``MISSING_FORECAST`` skip downstream.
    """

    name = "ModelForecastAgent"

    def __init__(
        self,
        forecasts: Sequence[ProbabilityForecast],
        max_age_seconds: float = 3600.0,
        config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(config)
        self.max_age = float(max_age_seconds)
        self.by_market: dict[str, list[ProbabilityForecast]] = {}
        for forecast in forecasts:
            self.by_market.setdefault(forecast.market_ticker, []).append(forecast)
        for series in self.by_market.values():
            series.sort(key=lambda f: f.decision_timestamp)
        self.model_version = forecasts[0].model_version if forecasts else "none"

    def forecast_at(
        self, market_id: str, as_of: datetime
    ) -> ProbabilityForecast | None:
        """The most recent forecast at or before ``as_of``, if fresh enough."""
        series = self.by_market.get(market_id)
        if not series:
            return None
        chosen: ProbabilityForecast | None = None
        for forecast in series:
            if forecast.decision_timestamp > as_of:
                break
            chosen = forecast
        if chosen is None:
            return None
        age = (as_of - chosen.decision_timestamp).total_seconds()
        return chosen if age <= self.max_age else None

    def analyze(self, state: MarketState) -> Forecast:
        """Return the stored model forecast for this instant, or a non-forecast."""
        market_id = state.market.market_id
        stored = self.forecast_at(market_id, state.as_of)

        if stored is None:
            return Forecast(
                agent_name=self.name,
                market_id=market_id,
                ts=state.as_of,
                probability=None,
                confidence=0.0,
                status=ForecastStatus.NO_DATA,
                rationale=(
                    "No model forecast at or before this instant within the freshness "
                    f"window of {self.max_age:.0f}s."
                ),
                details={"model_version": self.model_version},
            )

        return Forecast(
            agent_name=self.name,
            market_id=market_id,
            ts=state.as_of,
            probability=stored.probability,
            confidence=stored.confidence,
            is_placeholder=False,
            status=ForecastStatus.OK,
            rationale=(
                f"Model {stored.model_version} forecast {stored.probability:.4f} "
                f"(raw {stored.raw_probability_yes:.4f}) made at "
                f"{stored.decision_timestamp:%Y-%m-%d %H:%M:%S} UTC."
            ),
            inputs=[
                ForecastInput(
                    name="model_forecast",
                    ts=stored.decision_timestamp,
                    source=stored.model_version,
                )
            ],
            assumptions=[
                f"Feature set {stored.feature_version} built only from data at or "
                "before the decision timestamp.",
                f"Data version {stored.data_version}.",
                "Calibration fitted on a validation fold that predates this forecast.",
                "The forecast is an estimate, not a known probability.",
            ],
            details={
                "raw": stored.raw_probability_yes,
                "calibrated": stored.calibrated_probability_yes,
                "model_version": stored.model_version,
                "decision_timestamp": stored.decision_timestamp,
            },
        )


def strategy_config_for(policy: PolicyConfig, fills_config: dict[str, Any]) -> dict[str, Any]:
    """Translate the policy into the existing strategy's configuration.

    The backtester's strategy already implements "edge must exceed a threshold
    and expected value after costs must be positive". Rather than duplicate
    that logic, the policy is expressed in its terms, so the simulator charges
    exactly the costs the policy assumed.
    """
    typical_fee = fee_per_contract(
        0.5,
        float(fills_config.get("fee_rate", 0.02)),
        str(fills_config.get("fee_model", "kalshi")),
    )
    base_slippage = float(fills_config.get("base_slippage", 0.002))
    return {
        "min_edge": policy.safety_margin + typical_fee + base_slippage,
        "require_positive_ev": True,
    }


def freshness_window(stride_seconds: float, multiplier: float = 1.5) -> float:
    """How long a forecast stays usable, given the decision grid spacing."""
    return max(60.0, stride_seconds * multiplier)
