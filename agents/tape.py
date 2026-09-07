"""TapeAgent - a reference price built from recent prints.

Why this agent exists: an agent stack whose only real view is the bid/ask
midpoint can never produce a trade, because the midpoint is by construction
inside the spread it would have to cross. To exercise the proposal, risk and
execution paths end to end, the demo needs at least one view that can disagree
with the current quote.

The signal is a size-weighted average of recent tape prints, blended toward the
current midpoint. When the quote has moved away from where volume recently
traded, this agent leans back toward that traded level. That is a toy
mean-reversion assumption. It is not validated, it is not predictive, and it is
not trading advice - it is plumbing that makes the simulation do something.
"""

from __future__ import annotations

from typing import Any

from agents.base import ForecastAgent
from agents.market import implied_probability
from core.models import Forecast, MarketState, Trade

NOTE = (
    "TOY SIGNAL - size-weighted recent trade price blended with the midpoint. "
    "Mean-reversion assumption, not a validated forecast."
)


def volume_weighted_price(trades: list[Trade], max_trades: int = 20) -> float | None:
    """Size-weighted average price of the most recent ``max_trades`` prints.

    Args:
        trades: Recent trades, oldest first.
        max_trades: How many of the most recent prints to include.

    Returns:
        The VWAP, or ``None`` when there are no prints with positive size.
    """
    recent = trades[-max_trades:] if max_trades > 0 else list(trades)
    volume = sum(t.size for t in recent)
    if volume <= 0.0:
        return None
    return sum(t.price * t.size for t in recent) / volume


class TapeAgent(ForecastAgent):
    """Blends a recent-trade VWAP with the midpoint into a fair-value estimate."""

    name = "TapeAgent"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.max_trades = int(self.config.get("max_trades", 20))
        self.tape_weight = float(self.config.get("tape_weight", 0.7))
        self.min_trades = int(self.config.get("min_trades", 5))
        self.confidence = float(self.config.get("confidence", 0.4))

    def analyze(self, state: MarketState) -> Forecast:
        """Estimate fair value from recent prints and the current midpoint.

        Args:
            state: Market state at the decision instant. Only trades already on
                the tape at ``state.as_of`` are visible.

        Returns:
            A :class:`core.models.Forecast`; ``probability`` is ``None`` when
            there is not enough tape to form a view.
        """
        trades = state.recent_trades
        vwap = volume_weighted_price(trades, self.max_trades)
        mid = implied_probability(state.quote)

        if vwap is None or len(trades) < self.min_trades:
            return Forecast(
                agent_name=self.name,
                market_id=state.market.market_id,
                ts=state.as_of,
                probability=None,
                confidence=0.0,
                rationale=f"Only {len(trades)} recent prints; below the minimum "
                f"of {self.min_trades}.",
                details={"n_trades": len(trades), "vwap": vwap, "mid": mid},
            )

        if mid is None:
            probability = vwap
        else:
            probability = self.tape_weight * vwap + (1.0 - self.tape_weight) * mid

        probability = max(0.0, min(1.0, probability))
        return Forecast(
            agent_name=self.name,
            market_id=state.market.market_id,
            ts=state.as_of,
            probability=probability,
            confidence=self.confidence,
            is_placeholder=False,
            rationale=(
                f"{NOTE} VWAP {vwap:.3f} over {min(len(trades), self.max_trades)} prints"
                + (f", midpoint {mid:.3f}." if mid is not None else ".")
            ),
            details={
                "vwap": vwap,
                "mid": mid,
                "n_trades": len(trades),
                "tape_weight": self.tape_weight,
            },
        )
