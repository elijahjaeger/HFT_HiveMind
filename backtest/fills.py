"""Simulated execution: fees, slippage and partial fills.

This module is the single source of truth for what a trade costs. The strategy
uses it to estimate costs before proposing, and the engine uses it to execute,
so the expected value shown on a proposal is computed with the same model that
later fills it.

Orders cross the spread: a BUY lifts the ask, a SELL hits the bid. Size is
capped by the displayed depth, which is what produces partial fills.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from core.models import Quote, Side, SimulatedFill

DEFAULT_FILL_CONFIG: dict[str, Any] = {
    "max_quote_age_seconds": 900.0,
    "enforce_limit_price": True,
    "base_slippage": 0.002,
    "impact_coefficient": 0.010,
    "allow_partial_fills": True,
    "max_fraction_of_displayed": 1.0,
    "min_fill_size": 1.0,
    "fee_rate": 0.02,
    "fee_model": "kalshi",
    "tick_size": 0.01,
}


def compute_fee(
    price: float,
    size: float,
    fee_rate: float = 0.02,
    fee_model: str = "kalshi",
) -> float:
    """Return the total fee for a fill, in dollars.

    Two models are supported:

    ``"kalshi"``
        ``fee_rate * size * price * (1 - price)`` - the fee peaks near 0.50 and
        vanishes near certainty, mirroring how several venues charge.
    ``"notional"``
        ``fee_rate * size * price`` - a flat percentage of traded notional.

    Args:
        price: Execution price in probability units.
        size: Number of contracts.
        fee_rate: Fee coefficient from configuration.
        fee_model: ``"kalshi"`` or ``"notional"``.

    Returns:
        The fee, never negative.
    """
    if size <= 0.0 or fee_rate <= 0.0:
        return 0.0
    price = min(max(price, 0.0), 1.0)
    if fee_model == "notional":
        fee = fee_rate * size * price
    elif fee_model == "kalshi":
        fee = fee_rate * size * price * (1.0 - price)
    else:
        raise ValueError(f"unknown fee_model: {fee_model!r}")
    return max(0.0, fee)


def fee_per_contract(
    price: float, fee_rate: float = 0.02, fee_model: str = "kalshi"
) -> float:
    """Fee for a single contract at ``price`` (used for expected-value maths)."""
    return compute_fee(price, 1.0, fee_rate, fee_model)


def estimate_slippage(
    size: float,
    available_size: float,
    base_slippage: float = 0.002,
    impact_coefficient: float = 0.010,
) -> float:
    """Estimate adverse price movement for an order, in probability units.

    The model is deliberately simple and deterministic: a constant crossing
    cost plus a linear market-impact term that grows with the fraction of
    displayed depth consumed. An order with no visible depth pays the full
    impact coefficient.

    Args:
        size: Contracts to execute.
        available_size: Displayed size on the side being taken.
        base_slippage: Constant component.
        impact_coefficient: Impact at 100% of displayed depth.

    Returns:
        A non-negative slippage estimate.
    """
    if size <= 0.0:
        return 0.0
    if available_size <= 0.0:
        return max(0.0, base_slippage + impact_coefficient)
    participation = min(1.0, size / available_size)
    return max(0.0, base_slippage + impact_coefficient * participation)


def _config(overrides: dict[str, Any] | None) -> dict[str, Any]:
    merged = dict(DEFAULT_FILL_CONFIG)
    merged.update(overrides or {})
    return merged


def simulate_fill(
    quote: Quote | None,
    side: Side,
    size: float,
    ts: datetime,
    config: dict[str, Any] | None = None,
    proposal_id: str = "",
    fill_id: str = "",
    limit_price: float | None = None,
    decided_at: datetime | None = None,
) -> SimulatedFill:
    """Execute ``size`` contracts against ``quote`` and return the fill.

    A fill with ``filled_size == 0`` and an ``unfilled_reason`` is returned
    rather than an exception whenever execution is impossible, so the audit log
    keeps a record of the attempt.

    Two guards make this an order rather than a wish:

    * **Staleness.** Execution refuses to trade against a book older than
      ``max_quote_age_seconds``, independently of what the risk agent allowed at
      decision time. A book that stopped updating is not a tradable price.
    * **Limit price.** A marketable limit order never fills worse than its
      limit. This matters once execution is delayed: if the book moves against
      the order while it is in flight, it simply does not fill.

    Args:
        quote: Book at the time of execution (``None`` means no market).
        side: BUY lifts the ask, SELL hits the bid.
        size: Requested contracts.
        ts: Execution timestamp.
        config: The ``fills`` settings section.
        proposal_id: Proposal this fill belongs to.
        fill_id: Identifier for the fill record.
        limit_price: Worst acceptable execution price. ``None`` executes at
            market, which is only realistic with no delay.
        decided_at: When the order was decided, recorded for the audit trail.

    Returns:
        A :class:`core.models.SimulatedFill`.
    """
    cfg = _config(config)
    market_id = quote.market_id if quote else ""

    def unfilled(reason: str, reference: float | None = None) -> SimulatedFill:
        return SimulatedFill(
            fill_id=fill_id or f"{proposal_id}-nofill",
            proposal_id=proposal_id,
            market_id=market_id,
            ts=ts,
            decided_at=decided_at,
            side=side,
            requested_size=max(0.0, size),
            filled_size=0.0,
            reference_price=reference,
            avg_price=None,
            unfilled_reason=reason,
        )

    if size <= 0.0:
        return unfilled("NON_POSITIVE_SIZE")
    if quote is None:
        return unfilled("NO_QUOTE")

    max_age = cfg.get("max_quote_age_seconds")
    if max_age is not None and quote.age_seconds(ts) > float(max_age):
        return unfilled("STALE_QUOTE")

    reference = quote.price_for(side)
    if reference is None:
        return unfilled("NO_PRICE_ON_SIDE")

    available = quote.size_for(side) * float(cfg["max_fraction_of_displayed"])
    if available <= 0.0:
        return unfilled("NO_DISPLAYED_SIZE", reference)

    fillable = min(size, available)
    if fillable < size and not bool(cfg["allow_partial_fills"]):
        return unfilled("PARTIAL_FILLS_DISABLED", reference)
    if fillable < float(cfg["min_fill_size"]):
        return unfilled("BELOW_MIN_FILL_SIZE", reference)

    slippage = estimate_slippage(
        fillable,
        available,
        float(cfg["base_slippage"]),
        float(cfg["impact_coefficient"]),
    )
    tick = float(cfg["tick_size"])
    raw_price = reference + side.sign * slippage
    avg_price = min(max(raw_price, tick), 1.0 - tick)

    if limit_price is not None and bool(cfg.get("enforce_limit_price", True)):
        too_expensive = side is Side.BUY and avg_price > limit_price + 1e-12
        too_cheap = side is Side.SELL and avg_price < limit_price - 1e-12
        if too_expensive or too_cheap:
            return unfilled("LIMIT_NOT_MARKETABLE", reference)

    fee = compute_fee(
        avg_price, fillable, float(cfg["fee_rate"]), str(cfg["fee_model"])
    )

    return SimulatedFill(
        fill_id=fill_id or f"{proposal_id}-fill",
        proposal_id=proposal_id,
        market_id=market_id,
        ts=ts,
        decided_at=decided_at,
        side=side,
        requested_size=size,
        filled_size=fillable,
        reference_price=reference,
        avg_price=avg_price,
        slippage=abs(avg_price - reference),
        fee=fee,
        unfilled_reason=None,
    )
