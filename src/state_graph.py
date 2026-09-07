"""LangGraph supervisor state machine: the control flow for the whole desk.

Shape
-----
A classic supervisor topology. Every worker node returns to the supervisor,
which owns *all* routing decisions, so the order of operations lives in exactly
one function instead of being smeared across the edges::

              +-----------------------------------------------+
              v                                               |
    START -> supervisor -> ingest ---------------------------->+
                  |     -> research_desk (CrewAI) ------------>+
                  |     -> rl_policy (FinRL/SB3) ------------->+
                  |     -> risk (existing RiskAgent) --------->+
                  |     -> execute (existing simulate_fill) -->+
                  +-> END

The five workers map onto the pipeline the base project already had - build a
view, size it, veto it, fill it - with the CrewAI desk supplying the view and
the RL policy supplying the sizing intent.

What the graph does *not* change
--------------------------------
``risk`` and ``execute`` call ``agents.risk.RiskAgent`` and
``backtest.fills.simulate_fill`` unchanged, so the fee model, the spread
crossing, the depth cap, the staleness guard and every risk limit apply exactly
as they do in the backtester. The graph adds routing, not permissions: there is
still no order-placement path anywhere in it.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from langgraph.graph import END, START, StateGraph

from agents.base import Consensus
from agents.risk import RiskAgent
from agents.strategy import build_proposal
from backtest.engine import public_market_view
from backtest.fills import simulate_fill
from core.dataset import MarketDataset
from core.models import MarketState, RiskAction
from src.contracts import AlphaVector, PolicyMetrics, SupervisorState
from src.research_desk import ResearchDesk
from src.rl_policy import RLPolicyPipeline

GRAPH_VERSION = "supervisor-v1"

# The fixed order the supervisor walks. Kept as data so the routing rule is one
# lookup rather than a chain of if-statements.
_PIPELINE: tuple[str, ...] = ("ingest", "research_desk", "rl_policy", "risk", "execute")


def build_states(
    dataset: MarketDataset, as_of: datetime, capital: float = 10_000.0, window: int = 20
) -> list[MarketState]:
    """One :class:`MarketState` per market, built only from data at or before ``as_of``.

    Runs each market through :func:`backtest.engine.public_market_view`, so the
    outcome and resolution time are stripped exactly as they are for the
    backtester's agents.
    """
    states: list[MarketState] = []
    for market in dataset.markets:
        quotes = [q for q in dataset.quotes_for(market.market_id) if q.ts <= as_of]
        trades = [t for t in dataset.trades_for(market.market_id) if t.ts <= as_of]
        states.append(
            MarketState(
                as_of=as_of,
                market=public_market_view(market, as_of),
                quote=quotes[-1] if quotes else None,
                recent_trades=trades[-window:],
                equity=capital,
                peak_equity=capital,
                cash=capital,
            )
        )
    return states


class TradingGraph:
    """Builds and runs the supervisor graph over a dataset."""

    def __init__(
        self,
        dataset: MarketDataset,
        settings: dict[str, Any] | None = None,
        use_llm: bool = False,
        train_timesteps: int = 1_000,
    ) -> None:
        self.dataset = dataset
        self.settings = dict(settings or {})
        self.desk = ResearchDesk(self.settings, use_llm=use_llm)
        self.rl = RLPolicyPipeline(self.settings, total_timesteps=train_timesteps)
        self.risk_agent = RiskAgent(self.settings.get("risk", {}))
        self._states: list[MarketState] = []
        self.graph = self._build()

    # -- nodes ------------------------------------------------------------

    def ingest(self, state: SupervisorState) -> dict[str, Any]:
        """Assemble the causal market states for this decision instant."""
        as_of = state.as_of or _default_as_of(self.dataset)
        self._states = build_states(
            self.dataset,
            as_of,
            capital=float(self.settings.get("backtest", {}).get("initial_capital", 10_000.0)),
        )
        return {
            "as_of": as_of,
            "market_ids": [s.market.market_id for s in self._states],
            "audit": [*state.audit, f"ingest: {len(self._states)} market state(s) at {as_of}"],
        }

    def research_desk(self, state: SupervisorState) -> dict[str, Any]:
        """Run the CrewAI personas and emit alpha vectors."""
        alpha = self.desk.run(self._states)
        with_view = sum(1 for v in alpha if v.has_view)
        return {
            "alpha": alpha,
            "audit": [
                *state.audit,
                f"research_desk: {len(alpha)} alpha vector(s), {with_view} with a usable view",
            ],
        }

    def rl_policy(self, state: SupervisorState) -> dict[str, Any]:
        """Train / refresh the RL policy on the alpha vectors and report metrics.

        A single instant does not contain enough episodes to learn from, so the
        node reports honestly and skips training rather than producing a policy
        fitted to one timestep per market.
        """
        usable = [v for v in state.alpha if v.has_view]
        if not usable:
            return {
                "policy": PolicyMetrics(warnings=["No usable alpha; policy not updated."]),
                "audit": [*state.audit, "rl_policy: skipped, no usable alpha"],
            }
        try:
            metrics = self.rl.train(self.dataset, usable)
            note = (
                f"rl_policy: trained {metrics.timesteps} steps over "
                f"{metrics.effective_n} resolved market(s)"
            )
        except (ValueError, RuntimeError) as error:
            metrics = PolicyMetrics(warnings=[f"training skipped: {error}"])
            note = f"rl_policy: skipped ({type(error).__name__}: {error})"
        return {"policy": metrics, "audit": [*state.audit, note]}

    def risk(self, state: SupervisorState) -> dict[str, Any]:
        """Turn alpha into proposals and run the existing risk veto over them."""
        proposals: list[Any] = []
        decisions: list[Any] = []
        by_market = {s.market.market_id: s for s in self._states}

        for vector in state.alpha:
            if not vector.has_view:
                continue
            market_state = by_market.get(vector.market_id)
            if market_state is None:
                continue
            consensus = Consensus(
                market_id=vector.market_id,
                probability=vector.consensus_probability,
                disagreement=vector.disagreement,
                n_contributors=1,
            )
            proposal = build_proposal(
                market_state,
                consensus,
                f"graph-{vector.market_id}-{state.cycles}",
                self.settings.get("strategy", {}),
                self.settings.get("fills", {}),
            )
            if proposal is None:
                continue
            market_state.proposal = proposal
            proposals.append(proposal)
            decisions.append(self.risk_agent.evaluate(proposal, market_state))

        approved = sum(1 for d in decisions if d.action is not RiskAction.REJECT)
        return {
            "proposals": proposals,
            "decisions": decisions,
            "audit": [
                *state.audit,
                f"risk: {len(proposals)} proposal(s), {approved} survived the veto",
            ],
        }

    def execute(self, state: SupervisorState) -> dict[str, Any]:
        """Simulate fills for surviving proposals. Nothing is ever submitted."""
        fills: list[Any] = []
        by_market = {s.market.market_id: s for s in self._states}

        for proposal, decision in zip(state.proposals, state.decisions, strict=False):
            if decision.action is RiskAction.REJECT:
                continue
            size = getattr(decision, "approved_size", None) or proposal.size
            market_state = by_market.get(proposal.market_id)
            if market_state is None:
                continue
            fills.append(
                simulate_fill(
                    quote=market_state.quote,
                    side=proposal.side,
                    size=float(size),
                    ts=proposal.ts,
                    config=self.settings.get("fills", {}),
                    proposal_id=proposal.proposal_id,
                    fill_id=f"fill-{proposal.proposal_id}",
                    limit_price=proposal.limit_price,
                    decided_at=proposal.ts,
                )
            )
        filled = sum(1 for f in fills if f.filled_size > 0)
        return {
            "fills": fills,
            "cycles": state.cycles + 1,
            "audit": [
                *state.audit,
                f"execute: {len(fills)} simulated fill(s), {filled} non-zero (HYPOTHETICAL)",
            ],
        }

    # -- supervisor -------------------------------------------------------

    def supervisor(self, state: SupervisorState) -> dict[str, Any]:
        """Decide the next hop. The only place routing is decided."""
        if state.route == "done" or state.cycles >= state.max_cycles:
            return {"route": "done"}
        try:
            nxt = (
                _PIPELINE[_PIPELINE.index(state.route) + 1]
                if state.route in _PIPELINE
                else _PIPELINE[0]
            )
        except IndexError:
            nxt = "done"
        return {"route": nxt}

    def _route(self, state: SupervisorState) -> str:
        """Conditional-edge function: map the routing decision to a node name."""
        return END if state.route == "done" else state.route

    def _build(self) -> Any:
        """Wire the graph: supervisor fans out, every worker returns to it."""
        graph = StateGraph(SupervisorState)
        graph.add_node("supervisor", self.supervisor)
        for name in _PIPELINE:
            graph.add_node(name, getattr(self, name))
        graph.add_edge(START, "supervisor")
        graph.add_conditional_edges(
            "supervisor", self._route, {**{n: n for n in _PIPELINE}, END: END}
        )
        for name in _PIPELINE:
            graph.add_edge(name, "supervisor")
        return graph.compile()

    def run(self, as_of: datetime | None = None, max_cycles: int = 1) -> SupervisorState:
        """Execute the graph once and return the final state."""
        # "start" is deliberately not a pipeline stage: the supervisor advances
        # the route *before* dispatching, so seeding it with _PIPELINE[0] would
        # route straight past ingest and leave every later node with no state.
        initial = SupervisorState(as_of=as_of, route="start", max_cycles=max_cycles)
        result = self.graph.invoke(initial)
        return SupervisorState(**result) if isinstance(result, dict) else result


def _default_as_of(dataset: MarketDataset) -> datetime:
    """A decision instant with data on both sides: the median quote timestamp."""
    stamps = sorted(q.ts for q in dataset.quotes)
    if not stamps:
        raise ValueError("dataset has no quotes to pick a decision instant from")
    return stamps[len(stamps) // 2]


def compile_graph(
    dataset: MarketDataset | None = None,
    settings: dict[str, Any] | None = None,
    **kwargs: Any,
) -> Any:
    """Build and compile the supervisor graph.

    Exposed as a module-level helper so ``verify_architecture.py`` (and any
    caller that only wants to check the topology) can compile the state machine
    without constructing a dataset first.
    """
    if dataset is None:
        from core.config import get_settings
        from core.synthetic import generate_demo_dataset

        settings = settings or get_settings()
        dataset = generate_demo_dataset(settings.get("data", {}))
    return TradingGraph(dataset, settings, **kwargs).graph


__all__ = [
    "GRAPH_VERSION",
    "AlphaVector",
    "SupervisorState",
    "TradingGraph",
    "build_states",
    "compile_graph",
]
