"""Typed payloads passed between the graph's nodes.

Keeping these in their own module means :mod:`src.research_desk` (which
produces alpha vectors) and :mod:`src.rl_policy` (which consumes them) never
import each other, so the graph wiring stays the only thing that knows about
both.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

# Feature order is fixed and versioned: the RL policy's observation space is
# built from it, so a silent reordering would invalidate a trained policy
# without changing any shape.
ALPHA_FEATURE_NAMES: tuple[str, ...] = (
    "macro_probability",
    "macro_confidence",
    "micro_probability",
    "micro_confidence",
    "consensus_probability",
    "disagreement",
    "spread",
    "depth_imbalance",
    "quote_age_seconds",
)

ALPHA_VECTOR_VERSION = "alpha-v1"


@dataclass(frozen=True)
class AlphaVector:
    """One research-desk view of one market at one instant.

    Every field is derived from data stamped at or before ``as_of``. The vector
    carries its own ``as_of`` so a downstream consumer can re-check causality
    rather than trusting the producer.
    """

    market_id: str
    as_of: datetime
    macro_probability: float | None = None
    macro_confidence: float = 0.0
    micro_probability: float | None = None
    micro_confidence: float = 0.0
    consensus_probability: float | None = None
    disagreement: float = 0.0
    spread: float | None = None
    depth_imbalance: float = 0.0
    quote_age_seconds: float = 0.0
    source: str = "deterministic"
    notes: list[str] = field(default_factory=list)

    @property
    def has_view(self) -> bool:
        """True when at least one persona produced a usable probability."""
        return self.consensus_probability is not None

    def vector(self) -> list[float]:
        """Dense observation in :data:`ALPHA_FEATURE_NAMES` order.

        ``None`` probabilities become ``0.5`` (maximum uncertainty) paired with
        a zero confidence, so a missing view is representable without a NaN
        reaching the policy network.
        """
        values: list[float] = []
        for name in ALPHA_FEATURE_NAMES:
            raw = getattr(self, name, None)
            if raw is None:
                raw = 0.5 if name.endswith("probability") else 0.0
            values.append(float(raw))
        return values

    def as_row(self) -> dict[str, Any]:
        """Display row for the dashboard and audit log."""
        return {
            "market": self.market_id,
            "as_of": self.as_of,
            "macro": self.macro_probability,
            "micro": self.micro_probability,
            "consensus": self.consensus_probability,
            "disagreement": round(self.disagreement, 4),
            "source": self.source,
        }


@dataclass
class PolicyMetrics:
    """What one RL training pass learned, and how much to trust it.

    ``effective_n`` is carried through from the base project's central
    constraint: the statistical unit is the resolved market, not the timestep.
    A policy trained over 400,000 steps across 200 markets has seen 200
    independent outcomes, and this field is what stops the step count from
    flattering it.
    """

    model_version: str = "none"
    timesteps: int = 0
    n_markets: int = 0
    effective_n: int = 0
    mean_episode_reward: float | None = None
    final_equity: float | None = None
    total_return: float | None = None
    n_trades: int = 0
    explained_variance: float | None = None
    entropy_loss: float | None = None
    is_synthetic: bool = True
    warnings: list[str] = field(default_factory=list)

    def as_rows(self) -> list[dict[str, Any]]:
        """Metric rows, effective_n first because it governs the rest."""
        return [
            {"metric": "Resolved markets (effective n)", "value": self.effective_n},
            {"metric": "Training timesteps", "value": self.timesteps},
            {"metric": "Mean episode reward", "value": self.mean_episode_reward},
            {"metric": "Simulated final equity", "value": self.final_equity},
            {"metric": "Simulated return", "value": self.total_return},
            {"metric": "Simulated trades", "value": self.n_trades},
            {"metric": "Explained variance", "value": self.explained_variance},
            {"metric": "Model version", "value": self.model_version},
        ]


@dataclass
class SupervisorState:
    """The LangGraph state object threaded through every node.

    LangGraph merges partial dicts returned by nodes into this structure, so
    each node declares only the keys it owns.
    """

    as_of: datetime | None = None
    market_ids: list[str] = field(default_factory=list)
    alpha: list[AlphaVector] = field(default_factory=list)
    policy: PolicyMetrics | None = None
    proposals: list[Any] = field(default_factory=list)
    decisions: list[Any] = field(default_factory=list)
    fills: list[Any] = field(default_factory=list)
    audit: list[str] = field(default_factory=list)
    route: str = "research_desk"
    cycles: int = 0
    max_cycles: int = 1
    errors: list[str] = field(default_factory=list)
