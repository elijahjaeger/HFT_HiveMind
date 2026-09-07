"""Settlement labels: the only place an outcome is decided.

A training label says what actually happened. It comes from the venue's
settlement fields and **never** from price. A market trading at $0.98 did not
"probably resolve YES" - it resolved however the venue says it resolved, and
until the venue says so it has no label at all.

The status table is documented in ``research/README_DESIGN.md`` and enforced
here. Anything not explicitly listed as terminal is treated as unresolved,
which is the safe direction: an unresolved market is merely unusable for
training, whereas a wrongly-labelled one silently corrupts every metric
downstream.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from core.models import Market, MarketStatus, Outcome

LABEL_POLICY_VERSION = "labels-v1"

TERMINAL_STATUSES: frozenset[str] = frozenset({"settled", "finalized", "resolved"})
"""Venue statuses whose result is final and will not change."""

DETERMINED_STATUSES: frozenset[str] = frozenset({"determined"})
"""Outcome decided but not yet final. Usable only with a settlement timestamp."""

CONTESTED_STATUSES: frozenset[str] = frozenset({"disputed", "amended"})
"""Excluded outright: the recorded outcome can still change."""


class LabelStatus(StrEnum):
    """Why a market did or did not yield a training label."""

    LABELLED = "LABELLED"
    UNRESOLVED = "UNRESOLVED"
    CONTESTED = "CONTESTED"
    NOT_BINARY = "NOT_BINARY"
    NO_SETTLEMENT_TIME = "NO_SETTLEMENT_TIME"
    MISSING_OUTCOME = "MISSING_OUTCOME"


@dataclass(frozen=True)
class SettlementLabel:
    """The outcome of one market, with the instant it became knowable."""

    market_id: str
    status: LabelStatus
    outcome: Outcome | None = None
    resolution_ts: datetime | None = None
    reason: str = ""
    policy_version: str = LABEL_POLICY_VERSION

    @property
    def is_usable(self) -> bool:
        """True only when this market may contribute a training label."""
        return (
            self.status is LabelStatus.LABELLED
            and self.outcome is not None
            and self.resolution_ts is not None
        )

    @property
    def value(self) -> float | None:
        """The label as ``1.0`` (YES) / ``0.0`` (NO), or ``None`` if unusable."""
        if not self.is_usable or self.outcome is None:
            return None
        return 1.0 if self.outcome is Outcome.YES else 0.0


def label_for_market(market: Market, venue_status: str | None = None) -> SettlementLabel:
    """Derive the settlement label for one market.

    Args:
        market: The normalized market.
        venue_status: The raw venue status string, when the caller has it.
            Normalization collapses several venue statuses onto
            :class:`core.models.MarketStatus`, which loses the distinction
            between "settled" and "disputed". Passing the raw string preserves
            it; without it, a contested market cannot be told apart from a
            settled one and the normalized status is used instead.

    Returns:
        A :class:`SettlementLabel`. Only ``is_usable`` labels may train a model.
    """
    raw = (venue_status or "").strip().lower()

    if raw in CONTESTED_STATUSES:
        return SettlementLabel(
            market_id=market.market_id,
            status=LabelStatus.CONTESTED,
            reason=(
                f"Venue status {raw!r}: the recorded outcome can still change, "
                "so it must not become a training label."
            ),
        )

    if market.outcome is None:
        return SettlementLabel(
            market_id=market.market_id,
            status=LabelStatus.UNRESOLVED,
            reason=(
                "No settlement outcome recorded. The market may still be open; "
                "price is never used to infer an outcome."
            ),
        )

    if market.status is not MarketStatus.RESOLVED:
        return SettlementLabel(
            market_id=market.market_id,
            status=LabelStatus.UNRESOLVED,
            reason=(
                f"An outcome is present but the market status is "
                f"{market.status.value}, which is not terminal."
            ),
        )

    if raw and raw not in TERMINAL_STATUSES and raw not in DETERMINED_STATUSES:
        return SettlementLabel(
            market_id=market.market_id,
            status=LabelStatus.UNRESOLVED,
            reason=f"Venue status {raw!r} is not a terminal settlement status.",
        )

    if market.resolution_time is None:
        return SettlementLabel(
            market_id=market.market_id,
            status=LabelStatus.NO_SETTLEMENT_TIME,
            reason=(
                "Outcome is recorded but carries no settlement timestamp, so we "
                "cannot prove a forecast was made before the outcome was known."
            ),
        )

    return SettlementLabel(
        market_id=market.market_id,
        status=LabelStatus.LABELLED,
        outcome=market.outcome,
        resolution_ts=market.resolution_time,
        reason=f"Settled {market.outcome.value} at {market.resolution_time}.",
    )


def label_markets(
    markets: list[Market], venue_statuses: dict[str, str] | None = None
) -> dict[str, SettlementLabel]:
    """Label every market, keyed by market id."""
    statuses = venue_statuses or {}
    return {
        m.market_id: label_for_market(m, statuses.get(m.market_id)) for m in markets
    }


def label_summary(labels: dict[str, SettlementLabel]) -> dict[str, int]:
    """Count markets by label status, for display and data-readiness checks."""
    counts: dict[str, int] = {status.value: 0 for status in LabelStatus}
    for label in labels.values():
        counts[label.status.value] += 1
    return counts


def usable_labels(labels: dict[str, SettlementLabel]) -> dict[str, SettlementLabel]:
    """Only the labels that may be used for training or scoring."""
    return {k: v for k, v in labels.items() if v.is_usable}
