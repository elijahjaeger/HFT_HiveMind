"""The threshold trading policy, and the bridge into the backtester.

The policy is deliberately blunt: trade only when the calibrated forecast
disagrees with the executable price by more than every modelled cost plus a
safety margin. Otherwise stay flat, and record why.

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
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from agents.base import ForecastAgent
from backtest.fills import estimate_slippage, fee_per_contract
from core.models import Forecast, ForecastInput, ForecastStatus, MarketState, Side
from research.models import ProbabilityForecast
from research.skips import SkippedDecision, SkipReason

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


@dataclass
class EdgeAssessment:
    """The full cost stack for one decision, kept for display."""

    market_probability: float
    forecast_probability: float
    side: Side | None
    reference_price: float | None
    raw_edge: float
    half_spread: float
    fee: float
    slippage: float
    safety_margin: float

    @property
    def required_edge(self) -> float:
        """Total edge needed before a trade is worth making."""
        return self.half_spread + self.fee + self.slippage + self.safety_margin

    @property
    def net_edge(self) -> float:
        """Edge remaining after every modelled cost."""
        return self.raw_edge - self.required_edge

    @property
    def is_tradable(self) -> bool:
        """True when the forecast clears the whole cost stack."""
        return self.side is not None and self.net_edge > 0.0

    def as_rows(self) -> list[dict[str, Any]]:
        """Cost-stack breakdown, for the dashboard."""
        return [
            {"component": "Market probability", "value": round(self.market_probability, 4)},
            {"component": "Model probability", "value": round(self.forecast_probability, 4)},
            {"component": "Raw edge", "value": round(self.raw_edge, 4)},
            {"component": "- half spread", "value": -round(self.half_spread, 4)},
            {"component": "- fee", "value": -round(self.fee, 4)},
            {"component": "- slippage", "value": -round(self.slippage, 4)},
            {"component": "- safety margin", "value": -round(self.safety_margin, 4)},
            {"component": "= net edge", "value": round(self.net_edge, 4)},
        ]


def assess_edge(
    state: MarketState,
    forecast_probability: float,
    policy: PolicyConfig,
    fills_config: dict[str, Any] | None = None,
) -> EdgeAssessment | None:
    """Work out whether a forecast beats the cost of acting on it.

    Args:
        state: Market state at the decision instant.
        forecast_probability: The calibrated model probability.
        policy: Threshold configuration.
        fills_config: The ``fills`` settings, so the cost model matches the
            one the simulator will actually charge.

    Returns:
        An :class:`EdgeAssessment`, or ``None`` when there is no usable book.
    """
    quote = state.quote
    if quote is None or not quote.is_two_sided or quote.mid is None:
        return None

    fills_cfg = dict(fills_config or {})
    mid = quote.mid
    spread = quote.spread or 0.0

    if forecast_probability > mid:
        side, reference = Side.BUY, quote.ask
    elif forecast_probability < mid:
        side, reference = Side.SELL, quote.bid
    else:
        side, reference = None, None

    raw_edge = abs(forecast_probability - mid)
    slippage = (
        estimate_slippage(
            policy.reference_size,
            quote.size_for(side) if side else 0.0,
            float(fills_cfg.get("base_slippage", 0.002)),
            float(fills_cfg.get("impact_coefficient", 0.010)),
        )
        if side
        else 0.0
    )
    fee = fee_per_contract(
        reference if reference is not None else mid,
        float(fills_cfg.get("fee_rate", 0.02)),
        str(fills_cfg.get("fee_model", "kalshi")),
    )

    return EdgeAssessment(
        market_probability=mid,
        forecast_probability=forecast_probability,
        side=side,
        reference_price=reference,
        raw_edge=raw_edge,
        half_spread=spread / 2.0,
        fee=fee,
        slippage=slippage,
        safety_margin=policy.safety_margin,
    )


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


@dataclass
class PolicyDecision:
    """The outcome of applying the policy at one decision point."""

    ts: datetime
    market_id: str
    assessment: EdgeAssessment | None
    skip: SkippedDecision | None = None

    @property
    def would_trade(self) -> bool:
        """True when the policy would have proposed a trade."""
        return self.skip is None and self.assessment is not None


@dataclass
class PolicyRun:
    """Every decision point the policy saw, traded or not."""

    decisions: list[PolicyDecision] = field(default_factory=list)
    skips: list[SkippedDecision] = field(default_factory=list)
    policy_version: str = POLICY_VERSION

    @property
    def eligible(self) -> int:
        """Decision points with a usable forecast and a usable book."""
        return sum(
            1 for d in self.decisions
            if d.skip is None or d.skip.reason
            in {SkipReason.INSUFFICIENT_EDGE, SkipReason.NEGATIVE_EXPECTED_VALUE}
        )

    @property
    def tradable(self) -> int:
        """Decision points that cleared the whole cost stack."""
        return sum(1 for d in self.decisions if d.would_trade)


def evaluate_policy(
    agent: ModelForecastAgent,
    states: Sequence[MarketState],
    policy: PolicyConfig,
    fills_config: dict[str, Any] | None = None,
) -> PolicyRun:
    """Apply the threshold policy to a sequence of decision points.

    Every point yields either a tradable assessment or a recorded skip, so the
    counts always reconcile.

    Args:
        agent: Serves the stored model forecasts.
        states: Decision points, in chronological order.
        policy: Threshold configuration.
        fills_config: Cost model settings.

    Returns:
        A :class:`PolicyRun`.
    """
    run = PolicyRun()

    for state in states:
        market_id = state.market.market_id
        forecast = agent.analyze(state)

        def record(
            reason: SkipReason,
            detail: str,
            *,
            _ts: datetime = state.as_of,
            _market: str = market_id,
            **extra: Any,
        ) -> None:
            """Record a skip. Loop variables are bound now, not captured."""
            skip = SkippedDecision(
                ts=_ts, market_id=_market, reason=reason, detail=detail, **extra
            )
            run.skips.append(skip)
            run.decisions.append(
                PolicyDecision(ts=_ts, market_id=_market, assessment=None, skip=skip)
            )

        if state.as_of >= state.market.close_time:
            record(SkipReason.MARKET_CLOSED, "Past the market close time.")
            continue
        if forecast.probability is None:
            record(SkipReason.MISSING_FORECAST, forecast.rationale)
            continue
        if forecast.confidence < policy.min_confidence:
            record(
                SkipReason.LOW_CONFIDENCE,
                f"Confidence {forecast.confidence:.3f} below the floor of "
                f"{policy.min_confidence:.3f}.",
                forecast_probability=forecast.probability,
            )
            continue

        quote = state.quote
        if quote is None or not quote.is_two_sided:
            record(SkipReason.INVALID_DATA, "No two-sided book at this instant.")
            continue
        age = quote.age_seconds(state.as_of)
        if age > policy.max_quote_age_seconds:
            record(
                SkipReason.STALE_DATA,
                f"Book is {age:.0f}s old, over the {policy.max_quote_age_seconds:.0f}s limit.",
            )
            continue

        assessment = assess_edge(state, forecast.probability, policy, fills_config)
        if assessment is None:
            record(SkipReason.INVALID_DATA, "Edge could not be assessed.")
            continue
        if not assessment.is_tradable:
            record(
                SkipReason.INSUFFICIENT_EDGE,
                f"Edge {assessment.raw_edge:.4f} below the {assessment.required_edge:.4f} "
                "needed to cover spread, fees, slippage and the safety margin.",
                edge=assessment.raw_edge,
                required_edge=assessment.required_edge,
                forecast_probability=forecast.probability,
                market_probability=assessment.market_probability,
            )
            continue

        run.decisions.append(
            PolicyDecision(ts=state.as_of, market_id=market_id, assessment=assessment)
        )

    return run


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


def horizon_delta(seconds: float) -> timedelta:
    """Convenience wrapper for configuring horizons in seconds."""
    return timedelta(seconds=seconds)
