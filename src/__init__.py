"""Self-learning trading architecture: LangGraph control flow over CrewAI research and an RL policy.

The three nodes this package adds sit *on top of* the existing sandbox rather
than replacing it. They reuse ``core.models``, ``agents.risk``,
``backtest.fills`` and ``backtest.engine`` unchanged, so every forward-only and
cost-model guarantee the base project makes still holds inside the graph.

**Simulation-only is unchanged.** Adding a learned policy does not add a way to
act on it: there is still no order-placement code path here, no venue write
method and no credential handling. The RL node's action space terminates in
``backtest.fills.simulate_fill`` and an in-memory portfolio, exactly like the
backtester's does.
"""

from src.contracts import AlphaVector, PolicyMetrics, SupervisorState

__all__ = ["AlphaVector", "PolicyMetrics", "SupervisorState"]
