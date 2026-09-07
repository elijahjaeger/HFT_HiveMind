"""Deterministic, rule-based agents.

No LLMs are used anywhere in this package. Each agent is a pure function of the
:class:`core.models.MarketState` it is handed.
"""

from agents.base import Agent, build_default_agents
from agents.baseline import BaseRateAgent
from agents.imbalance import BookImbalanceAgent
from agents.liquidity import LiquidityAgent
from agents.market import MarketAgent
from agents.risk import RiskAgent
from agents.tape import TapeAgent

__all__ = [
    "Agent",
    "BaseRateAgent",
    "BookImbalanceAgent",
    "LiquidityAgent",
    "MarketAgent",
    "RiskAgent",
    "TapeAgent",
    "build_default_agents",
]
