"""Agent interface and forecast aggregation helpers.

The interface is deliberately narrow::

    result = agent.analyze(market_state)

``market_state`` is a :class:`core.models.MarketState` containing only
information available at ``market_state.as_of``. Agents must not mutate it and
must not perform I/O.
"""

from __future__ import annotations

import statistics
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field

from core.models import AgentResult, Forecast, MarketState


class Agent(ABC):
    """Base class for every agent in the sandbox.

    Subclasses implement :meth:`analyze` and return a structured
    :class:`core.models.AgentResult`. Agents never place, route or submit
    orders; the most any agent can do is describe or veto a proposal.
    """

    name: str = "Agent"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config: dict[str, Any] = dict(config or {})

    @abstractmethod
    def analyze(self, state: MarketState) -> AgentResult:
        """Analyse a market state and return a structured result."""

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        """Readable representation naming the concrete agent."""
        return f"{type(self).__name__}(name={self.name!r})"


class ForecastAgent(Agent):
    """An agent whose :meth:`analyze` returns a :class:`core.models.Forecast`."""

    @abstractmethod
    def analyze(self, state: MarketState) -> Forecast:
        """Produce a probability estimate for the YES outcome."""


class Consensus(BaseModel):
    """Blended view across forecast agents, plus a disagreement measure."""

    market_id: str
    probability: float | None = None
    disagreement: float = 0.0
    stdev: float = 0.0
    n_contributors: int = 0
    contributors: dict[str, float] = Field(default_factory=dict)
    excluded: dict[str, str] = Field(default_factory=dict)

    @property
    def has_view(self) -> bool:
        """True when at least one agent contributed a usable probability."""
        return self.probability is not None


def combine_forecasts(
    forecasts: list[Forecast],
    config: dict[str, Any] | None = None,
    market_id: str = "",
) -> Consensus:
    """Blend agent forecasts into a single probability plus a disagreement score.

    Placeholder forecasts (``is_placeholder=True``) are excluded unless
    ``config["include_placeholder"]`` is True, so a hard-coded constant is never
    silently presented as a prediction. Agents may be weighted through
    ``config["weights"]``; a weight of ``0`` removes an agent from the blend.

    Args:
        forecasts: Forecasts produced at one decision instant.
        config: The ``agents.ensemble`` settings section.
        market_id: Market identifier used when ``forecasts`` is empty.

    Returns:
        A :class:`Consensus`. ``probability`` is ``None`` when no agent
        contributed a usable view.
    """
    cfg = dict(config or {})
    include_placeholder = bool(cfg.get("include_placeholder", False))
    weights: dict[str, float] = dict(cfg.get("weights", {}) or {})

    market = market_id or (forecasts[0].market_id if forecasts else "")
    contributors: dict[str, float] = {}
    excluded: dict[str, str] = {}
    weighted_sum = 0.0
    weight_total = 0.0

    for forecast in forecasts:
        if forecast.probability is None:
            excluded[forecast.agent_name] = "no probability"
            continue
        if forecast.is_placeholder and not include_placeholder:
            excluded[forecast.agent_name] = "placeholder excluded"
            continue
        weight = float(weights.get(forecast.agent_name, 1.0))
        if weight <= 0.0:
            excluded[forecast.agent_name] = "zero weight"
            continue
        contributors[forecast.agent_name] = forecast.probability
        weighted_sum += weight * forecast.probability
        weight_total += weight

    if weight_total <= 0.0:
        return Consensus(market_id=market, excluded=excluded)

    values = list(contributors.values())
    disagreement = max(values) - min(values) if len(values) > 1 else 0.0
    stdev = statistics.pstdev(values) if len(values) > 1 else 0.0
    return Consensus(
        market_id=market,
        probability=weighted_sum / weight_total,
        disagreement=disagreement,
        stdev=stdev,
        n_contributors=len(values),
        contributors=contributors,
        excluded=excluded,
    )


def build_default_agents(settings: dict[str, Any]) -> list[ForecastAgent]:
    """Instantiate the default forecast agents from a settings dictionary."""
    from agents.baseline import BaseRateAgent
    from agents.imbalance import BookImbalanceAgent
    from agents.market import MarketAgent
    from agents.tape import TapeAgent

    agent_cfg = settings.get("agents", {})
    return [
        MarketAgent(agent_cfg.get("market", {})),
        TapeAgent(agent_cfg.get("tape", {})),
        BookImbalanceAgent(agent_cfg.get("imbalance", {})),
        BaseRateAgent(agent_cfg.get("base_rate", {})),
    ]
