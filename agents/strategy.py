"""Deterministic proposal builder.

Turns a blended probability into a :class:`core.models.TradeProposal` when the
edge against the crossing price survives modelled fees and slippage.

This is toy plumbing whose purpose is to exercise the risk and execution paths.
It is not a strategy anyone should trade, it makes no claim of profitability,
and it is not financial advice.
"""

from __future__ import annotations

from typing import Any

from agents.base import Consensus
from backtest.fills import estimate_slippage, fee_per_contract
from core.models import MarketState, Side, TradeProposal

DEFAULT_STRATEGY_CONFIG: dict[str, Any] = {
    "min_edge": 0.03,
    "base_size": 100.0,
    "edge_size_multiplier": 1500.0,
    "max_order_size": 400.0,
    "min_order_size": 10.0,
    "require_positive_ev": True,
}

STRATEGY_NAME = "EdgeStrategy"


def expected_value_per_contract(
    side: Side, fair_probability: float, effective_price: float, fee: float
) -> float:
    """Expected value of one contract after costs, in dollars.

    A long YES contract pays ``1`` with probability ``fair_probability`` and
    costs ``effective_price``; a short YES contract is the mirror image. ``fee``
    is the per-contract fee.
    """
    if side is Side.BUY:
        return fair_probability - effective_price - fee
    return effective_price - fair_probability - fee


def _sizing(edge: float, cfg: dict[str, Any]) -> float:
    """Scale order size with the size of the edge, then clamp to configured bounds."""
    min_edge = float(cfg["min_edge"])
    raw = float(cfg["base_size"]) + float(cfg["edge_size_multiplier"]) * max(
        0.0, edge - min_edge
    )
    return float(min(float(cfg["max_order_size"]), max(float(cfg["min_order_size"]), raw)))


def explain_no_proposal(
    state: MarketState,
    consensus: Consensus,
    strategy_config: dict[str, Any] | None = None,
    fills_config: dict[str, Any] | None = None,
) -> str:
    """Say why :func:`build_proposal` would decline, in one line.

    Deliberately mirrors the branches in :func:`build_proposal` and lives
    beside it, so the two cannot drift apart unnoticed. Returns an empty string
    when a proposal *would* be produced.
    """
    cfg = dict(DEFAULT_STRATEGY_CONFIG)
    cfg.update(strategy_config or {})
    fills_cfg = dict(fills_config or {})

    quote = state.quote
    fair = consensus.probability
    if quote is None:
        return "INVALID_DATA: no quote at this instant."
    if fair is None:
        return "MISSING_FORECAST: no agent produced a usable probability."

    min_edge = float(cfg["min_edge"])
    if quote.ask is not None and fair - quote.ask >= min_edge:
        side, reference, edge = Side.BUY, quote.ask, fair - quote.ask
    elif quote.bid is not None and quote.bid - fair >= min_edge:
        side, reference, edge = Side.SELL, quote.bid, quote.bid - fair
    else:
        best_edge = max(
            (fair - quote.ask) if quote.ask is not None else -1.0,
            (quote.bid - fair) if quote.bid is not None else -1.0,
        )
        return (
            f"INSUFFICIENT_EDGE: best edge {best_edge:+.4f} against the crossing "
            f"price is below the {min_edge:.4f} required."
        )

    size = _sizing(edge, cfg)
    slippage = estimate_slippage(
        size,
        quote.size_for(side),
        float(fills_cfg.get("base_slippage", 0.002)),
        float(fills_cfg.get("impact_coefficient", 0.010)),
    )
    tick = float(fills_cfg.get("tick_size", 0.01))
    effective_price = min(max(reference + side.sign * slippage, tick), 1.0 - tick)
    fee = fee_per_contract(
        effective_price,
        float(fills_cfg.get("fee_rate", 0.02)),
        str(fills_cfg.get("fee_model", "kalshi")),
    )
    ev = expected_value_per_contract(side, fair, effective_price, fee)
    if bool(cfg["require_positive_ev"]) and ev <= 0.0:
        return (
            f"NEGATIVE_EXPECTED_VALUE: edge {edge:.4f} survives the threshold but "
            f"expected value after fees and slippage is {ev:+.4f} per contract."
        )
    return ""


def build_proposal(
    state: MarketState,
    consensus: Consensus,
    proposal_id: str,
    strategy_config: dict[str, Any] | None = None,
    fills_config: dict[str, Any] | None = None,
) -> TradeProposal | None:
    """Build a trade proposal from a blended probability, or return ``None``.

    A proposal is produced only when all of the following hold: a usable
    consensus probability exists, the book quotes the side that would be
    crossed, the edge against that crossing price is at least ``min_edge``, and
    (when ``require_positive_ev`` is set) the expected value after modelled fees
    and slippage is positive.

    Args:
        state: Market state at the decision instant.
        consensus: Blended agent view of the YES probability.
        proposal_id: Identifier to stamp on the proposal.
        strategy_config: The ``strategy`` settings section.
        fills_config: The ``fills`` settings section (cost model).

    Returns:
        A :class:`core.models.TradeProposal`, or ``None`` when no trade is warranted.
    """
    cfg = dict(DEFAULT_STRATEGY_CONFIG)
    cfg.update(strategy_config or {})
    fills_cfg = dict(fills_config or {})

    quote = state.quote
    fair = consensus.probability
    if quote is None or fair is None:
        return None

    min_edge = float(cfg["min_edge"])
    if quote.ask is not None and fair - quote.ask >= min_edge:
        side = Side.BUY
        reference = quote.ask
        edge = fair - quote.ask
    elif quote.bid is not None and quote.bid - fair >= min_edge:
        side = Side.SELL
        reference = quote.bid
        edge = quote.bid - fair
    else:
        return None

    size = _sizing(edge, cfg)
    slippage = estimate_slippage(
        size,
        quote.size_for(side),
        float(fills_cfg.get("base_slippage", 0.002)),
        float(fills_cfg.get("impact_coefficient", 0.010)),
    )
    tick = float(fills_cfg.get("tick_size", 0.01))
    effective_price = min(max(reference + side.sign * slippage, tick), 1.0 - tick)
    fee = fee_per_contract(
        effective_price,
        float(fills_cfg.get("fee_rate", 0.02)),
        str(fills_cfg.get("fee_model", "kalshi")),
    )
    ev = expected_value_per_contract(side, fair, effective_price, fee)

    if bool(cfg["require_positive_ev"]) and ev <= 0.0:
        return None

    return TradeProposal(
        proposal_id=proposal_id,
        ts=state.as_of,
        market_id=state.market.market_id,
        agent_name=STRATEGY_NAME,
        side=side,
        size=size,
        limit_price=effective_price,
        reference_price=reference,
        fair_probability=fair,
        edge=edge,
        expected_fee_per_contract=fee,
        expected_slippage=slippage,
        expected_value_per_contract=ev,
        rationale=(
            f"Blended probability {fair:.3f} vs {side.value} reference {reference:.3f} "
            f"(edge {edge:.3f}); EV after costs {ev:+.4f}/contract."
        ),
    )
