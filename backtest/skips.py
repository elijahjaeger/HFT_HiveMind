"""Why a decision did not become a trade.

The backtester used to return silently when the strategy declined to propose,
so "nothing happened" was indistinguishable from "the data was unusable". For
research that difference is the whole point: a policy that trades twice out of
four thousand opportunities is not the same as one that found four thousand
opportunities and rejected them all on cost.

Every decision point now produces exactly one outcome - a proposal, or a
:class:`SkipReason` explaining what stopped it.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any


class SkipReason(StrEnum):
    """The mutually exclusive reasons a decision did not produce a trade."""

    MISSING_FORECAST = "MISSING_FORECAST"
    """No model forecast was available at this decision timestamp."""

    INVALID_DATA = "INVALID_DATA"
    """No usable book: missing, one-sided, or crossed."""

    STALE_DATA = "STALE_DATA"
    """The book was older than the configured tolerance."""

    INSUFFICIENT_EDGE = "INSUFFICIENT_EDGE"
    """The forecast disagreed with the market, but by less than costs plus margin."""

    NEGATIVE_EXPECTED_VALUE = "NEGATIVE_EXPECTED_VALUE"
    """Edge cleared the threshold but expected value after costs was not positive."""

    RISK_CAP = "RISK_CAP"
    """The risk agent rejected or fully reduced the order."""

    EXECUTION_CONSTRAINT = "EXECUTION_CONSTRAINT"
    """Approved, but nothing could fill: no depth, or the limit was not marketable."""

    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    """The forecast's own confidence was below the configured floor."""

    MARKET_CLOSED = "MARKET_CLOSED"
    """The market was past its close time."""


@dataclass(frozen=True)
class SkippedDecision:
    """One decision point that did not become a trade, and why."""

    ts: datetime
    market_id: str
    reason: SkipReason
    detail: str = ""
    edge: float | None = None
    required_edge: float | None = None
    forecast_probability: float | None = None
    market_probability: float | None = None

    def as_row(self) -> dict[str, Any]:
        """Display row for the dashboard."""
        return {
            "ts": self.ts,
            "market": self.market_id,
            "reason": self.reason.value,
            "detail": self.detail,
            "edge": None if self.edge is None else round(self.edge, 4),
            "required": None if self.required_edge is None else round(self.required_edge, 4),
            "forecast": (
                None if self.forecast_probability is None
                else round(self.forecast_probability, 4)
            ),
            "market_prob": (
                None if self.market_probability is None
                else round(self.market_probability, 4)
            ),
        }


def skip_summary(skips: list[SkippedDecision]) -> list[dict[str, Any]]:
    """Count skips by reason, most frequent first.

    This is the table that tells a reader whether a policy is genuinely
    selective or simply starved of data.
    """
    counts: dict[str, int] = {}
    for skip in skips:
        counts[skip.reason.value] = counts.get(skip.reason.value, 0) + 1
    total = sum(counts.values()) or 1
    rows = [
        {"reason": reason, "count": count, "share": f"{count / total:.1%}"}
        for reason, count in counts.items()
    ]
    rows.sort(key=lambda r: r["count"], reverse=True)
    return rows
