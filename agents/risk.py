"""RiskAgent - the final, deterministic veto.

The risk agent is the last stop before the fill simulator. It answers exactly
one question: *may this proposal be executed, and at what size?* It returns
APPROVE, REDUCE or REJECT together with machine-readable reason codes.

It never submits, routes or places an order. It has no execution capability at
all - it only returns a decision object that the backtester obeys.
"""

from __future__ import annotations

from typing import Any

from agents.base import Agent
from core.models import (
    MarketState,
    MarketStatus,
    RiskAction,
    RiskDecision,
    RiskReason,
    RiskReasonCode,
    Side,
    TradeProposal,
)

SIZE_TOLERANCE = 1e-6
"""Sizes within this of the request count as a full approval, not a reduction."""

DEFAULT_RISK_LIMITS: dict[str, Any] = {
    "kill_switch": False,
    "max_position_size": 300.0,
    "max_event_exposure": 400.0,
    "max_total_exposure": 1200.0,
    "max_spread": 0.06,
    "max_slippage": 0.02,
    "max_data_age_seconds": 900.0,
    "max_drawdown": 0.25,
    "min_order_size": 1.0,
}


class RiskAgent(Agent):
    """Applies deterministic, configurable limits to a single trade proposal."""

    name = "RiskAgent"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        merged = dict(DEFAULT_RISK_LIMITS)
        merged.update(config or {})
        super().__init__(merged)

        self.kill_switch = bool(merged["kill_switch"])
        self.max_position_size = float(merged["max_position_size"])
        self.max_event_exposure = float(merged["max_event_exposure"])
        self.max_total_exposure = float(merged["max_total_exposure"])
        self.max_spread = float(merged["max_spread"])
        self.max_slippage = float(merged["max_slippage"])
        self.max_data_age_seconds = float(merged["max_data_age_seconds"])
        self.max_drawdown = float(merged["max_drawdown"])
        self.min_order_size = float(merged["min_order_size"])

    # -- public API ---------------------------------------------------------

    def analyze(self, state: MarketState) -> RiskDecision:
        """Evaluate ``state.proposal``.

        Args:
            state: Market state carrying the proposal under review.

        Returns:
            The :class:`core.models.RiskDecision` for that proposal.

        Raises:
            ValueError: If ``state.proposal`` is not set.
        """
        if state.proposal is None:
            raise ValueError("RiskAgent.analyze requires state.proposal to be set")
        return self.evaluate(state.proposal, state)

    def evaluate(self, proposal: TradeProposal, state: MarketState) -> RiskDecision:
        """Apply every risk rule to ``proposal`` and return a decision.

        Rules run in a fixed order. Hard blocks (kill switch, market closed,
        missing or stale data, wide spread, excessive slippage, drawdown
        breach) produce REJECT. Size limits produce REDUCE when a smaller size
        is still viable, and REJECT when it is not.

        Args:
            proposal: The proposed trade.
            state: Market state at the decision instant.

        Returns:
            A :class:`core.models.RiskDecision`.
        """
        reasons: list[RiskReason] = []
        blocked = self._hard_blocks(proposal, state, reasons)

        details: dict[str, Any] = {
            "position_before": state.position_quantity,
            "event_exposure_before": state.event_exposure,
            "total_exposure_before": state.total_exposure,
            "drawdown": state.drawdown,
            "limits": {
                "max_position_size": self.max_position_size,
                "max_event_exposure": self.max_event_exposure,
                "max_total_exposure": self.max_total_exposure,
                "max_spread": self.max_spread,
                "max_slippage": self.max_slippage,
                "max_data_age_seconds": self.max_data_age_seconds,
                "max_drawdown": self.max_drawdown,
                "kill_switch": self.kill_switch,
            },
        }

        if blocked:
            return RiskDecision(
                proposal_id=proposal.proposal_id,
                market_id=proposal.market_id,
                ts=state.as_of,
                action=RiskAction.REJECT,
                original_size=proposal.size,
                approved_size=0.0,
                reasons=reasons,
                details=details,
            )

        allowed, sizing_reasons = self._max_allowed_size(proposal, state)
        details["max_allowed_size"] = allowed

        if allowed < self.min_order_size or allowed <= 0.0:
            reasons.extend(sizing_reasons)
            reasons.append(
                RiskReason(
                    code=RiskReasonCode.SIZE_BELOW_MINIMUM,
                    message=(
                        f"Largest size permitted by the limits is {allowed:.2f}, "
                        f"below the minimum order size {self.min_order_size:.2f}."
                    ),
                    observed=allowed,
                    limit=self.min_order_size,
                )
            )
            return RiskDecision(
                proposal_id=proposal.proposal_id,
                market_id=proposal.market_id,
                ts=state.as_of,
                action=RiskAction.REJECT,
                original_size=proposal.size,
                approved_size=0.0,
                reasons=reasons,
                details=details,
            )

        # Only report a reduction when it is material. Rounding dust must not
        # turn a full-size approval into a REDUCE carrying no reason.
        if allowed < proposal.size - SIZE_TOLERANCE:
            reasons.extend(sizing_reasons)
            return RiskDecision(
                proposal_id=proposal.proposal_id,
                market_id=proposal.market_id,
                ts=state.as_of,
                action=RiskAction.REDUCE,
                original_size=proposal.size,
                approved_size=allowed,
                reasons=reasons,
                details=details,
            )

        reasons.append(
            RiskReason(code=RiskReasonCode.OK, message="All risk limits satisfied.")
        )
        return RiskDecision(
            proposal_id=proposal.proposal_id,
            market_id=proposal.market_id,
            ts=state.as_of,
            action=RiskAction.APPROVE,
            original_size=proposal.size,
            approved_size=min(allowed, proposal.size),
            reasons=reasons,
            details=details,
        )

    # -- rules --------------------------------------------------------------

    def _hard_blocks(
        self,
        proposal: TradeProposal,
        state: MarketState,
        reasons: list[RiskReason],
    ) -> bool:
        """Append every triggered hard-block reason; return True if any fired."""
        blocked = False

        if self.kill_switch:
            reasons.append(
                RiskReason(
                    code=RiskReasonCode.KILL_SWITCH,
                    message="Kill switch is engaged; all trading is halted.",
                )
            )
            blocked = True

        if proposal.size <= 0.0:
            reasons.append(
                RiskReason(
                    code=RiskReasonCode.INVALID_PROPOSAL,
                    message="Proposal size must be positive.",
                    observed=proposal.size,
                )
            )
            blocked = True

        if state.market.status is not MarketStatus.OPEN:
            reasons.append(
                RiskReason(
                    code=RiskReasonCode.MARKET_NOT_OPEN,
                    message=f"Market status is {state.market.status.value}.",
                )
            )
            blocked = True

        quote = state.quote
        if quote is None or quote.price_for(proposal.side) is None:
            reasons.append(
                RiskReason(
                    code=RiskReasonCode.NO_QUOTE,
                    message=(
                        "No quote available on the required side of the book "
                        f"({proposal.side.value})."
                    ),
                )
            )
            return True  # Nothing else can be measured without a price.

        age = quote.age_seconds(state.as_of)
        if age > self.max_data_age_seconds:
            reasons.append(
                RiskReason(
                    code=RiskReasonCode.STALE_DATA,
                    message=f"Quote is {age:.0f}s old.",
                    observed=age,
                    limit=self.max_data_age_seconds,
                )
            )
            blocked = True

        spread = quote.spread
        if spread is None:
            reasons.append(
                RiskReason(
                    code=RiskReasonCode.NO_QUOTE,
                    message="One-sided book: spread cannot be measured.",
                )
            )
            blocked = True
        elif spread > self.max_spread:
            reasons.append(
                RiskReason(
                    code=RiskReasonCode.SPREAD_TOO_WIDE,
                    message=f"Spread {spread:.4f} exceeds the limit.",
                    observed=spread,
                    limit=self.max_spread,
                )
            )
            blocked = True

        if proposal.expected_slippage > self.max_slippage:
            reasons.append(
                RiskReason(
                    code=RiskReasonCode.SLIPPAGE_TOO_HIGH,
                    message=(
                        f"Estimated slippage {proposal.expected_slippage:.4f} "
                        "exceeds the limit."
                    ),
                    observed=proposal.expected_slippage,
                    limit=self.max_slippage,
                )
            )
            blocked = True

        drawdown = state.drawdown
        if drawdown > self.max_drawdown:
            reasons.append(
                RiskReason(
                    code=RiskReasonCode.MAX_DRAWDOWN_BREACHED,
                    message=f"Portfolio drawdown {drawdown:.2%} exceeds the limit.",
                    observed=drawdown,
                    limit=self.max_drawdown,
                )
            )
            blocked = True

        return blocked

    def _max_allowed_size(
        self, proposal: TradeProposal, state: MarketState
    ) -> tuple[float, list[RiskReason]]:
        """Return the largest permissible size and the constraints that bind it.

        Exposure is defined as worst-case loss: ``quantity * price`` for a long
        position (it expires worthless on NO) and ``|quantity| * (1 - price)``
        for a short position (it pays out on YES).
        """
        price = float(proposal.reference_price)
        quantity = state.position_quantity
        requested = proposal.size
        caps: list[tuple[RiskReasonCode, float, float, str]] = []

        # 1. Per-market position limit.
        position_cap = self.max_position_size - proposal.side.sign * quantity
        caps.append(
            (
                RiskReasonCode.POSITION_LIMIT,
                position_cap,
                self.max_position_size,
                f"position would exceed max_position_size ({self.max_position_size:.0f})",
            )
        )

        # 2. Per-event exposure limit.
        denom = price if proposal.side is Side.BUY else (1.0 - price)
        if denom <= 1e-9:
            event_cap = float("inf")
        elif proposal.side is Side.BUY:
            event_cap = self.max_event_exposure / denom - quantity
        else:
            event_cap = self.max_event_exposure / denom + quantity
        caps.append(
            (
                RiskReasonCode.EVENT_EXPOSURE_LIMIT,
                event_cap,
                self.max_event_exposure,
                f"event exposure would exceed max_event_exposure ({self.max_event_exposure:.2f})",
            )
        )

        # 3. Portfolio-wide exposure limit (budget left after other markets).
        other_exposure = max(0.0, state.total_exposure - state.event_exposure)
        budget = max(0.0, self.max_total_exposure - other_exposure)
        if denom <= 1e-9:
            total_cap = float("inf")
        elif proposal.side is Side.BUY:
            total_cap = budget / denom - quantity
        else:
            total_cap = budget / denom + quantity
        caps.append(
            (
                RiskReasonCode.TOTAL_EXPOSURE_LIMIT,
                total_cap,
                self.max_total_exposure,
                f"book exposure would exceed max_total_exposure ({self.max_total_exposure:.2f})",
            )
        )

        # 4. Cash. Buying always costs premium; selling beyond a long position
        #    requires collateral for the worst-case payout.
        cash = max(0.0, state.cash)
        per_contract_cost = proposal.expected_fee_per_contract + (
            price if proposal.side is Side.BUY else 0.0
        )
        if proposal.side is Side.BUY:
            cash_cap = cash / per_contract_cost if per_contract_cost > 1e-9 else float("inf")
        else:
            collateral = 1.0 - price
            owned = max(0.0, quantity)
            cash_cap = owned + (cash / collateral if collateral > 1e-9 else float("inf"))
        caps.append(
            (
                RiskReasonCode.INSUFFICIENT_CASH,
                cash_cap,
                cash,
                "insufficient simulated cash",
            )
        )

        allowed = requested
        binding: list[RiskReason] = []
        for code, cap, limit, message in caps:
            cap = max(0.0, cap)
            if cap < requested - 1e-9:
                binding.append(
                    RiskReason(
                        code=code,
                        message=(
                            f"Size reduced from {requested:.2f} to at most {cap:.2f}: {message}."
                        ),
                        observed=requested,
                        limit=limit,
                    )
                )
            allowed = min(allowed, cap)

        return max(0.0, round(allowed, 6)), binding
