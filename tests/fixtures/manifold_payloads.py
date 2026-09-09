"""Canned Manifold API payloads, shaped from real responses.

Field names and types match what the venue actually returns (epoch
milliseconds, probabilities already in [0, 1]) so the normalisers are exercised
against the real contract rather than a convenient one.
"""

from __future__ import annotations

import json
from typing import Any

# 2024-08-11 -> 2025-01-02, resolved NO.
RESOLVED_MARKET: dict[str, Any] = {
    "id": "resolved-no-1",
    "question": "Will the demo claim replicate before 2025?",
    "outcomeType": "BINARY",
    "mechanism": "cpmm-1",
    "createdTime": 1723334400000,
    "closeTime": 1735689600000,
    "isResolved": True,
    "resolution": "NO",
    "resolutionTime": 1735780870155,
    "resolutionProbability": 0.02,
    "totalLiquidity": 420.0,
    "volume": 9100.0,
    "groupSlugs": ["science"],
    "url": "https://manifold.markets/demo/resolved-no-1",
}

RESOLVED_YES_MARKET: dict[str, Any] = {
    **RESOLVED_MARKET,
    "id": "resolved-yes-1",
    "question": "Will the demo milestone be reached?",
    "resolution": "YES",
    "resolutionProbability": 0.97,
}

# Resolved to MKT: a partial resolution with no binary answer.
AMBIGUOUS_MARKET: dict[str, Any] = {
    **RESOLVED_MARKET,
    "id": "resolved-mkt-1",
    "resolution": "MKT",
}

CANCELLED_MARKET: dict[str, Any] = {
    **RESOLVED_MARKET,
    "id": "resolved-cancel-1",
    "resolution": "CANCEL",
}

UNRESOLVED_MARKET: dict[str, Any] = {
    **RESOLVED_MARKET,
    "id": "open-1",
    "isResolved": False,
    "resolution": None,
    "resolutionTime": None,
}

MULTI_MARKET: dict[str, Any] = {
    **RESOLVED_MARKET,
    "id": "multi-1",
    "outcomeType": "MULTIPLE_CHOICE",
    "mechanism": "cpmm-multi-1",
}

# A resolved market whose settlement time never arrived: unusable, because a
# forecast cannot be shown to predate an outcome with no timestamp.
NO_SETTLEMENT_TIME: dict[str, Any] = {
    **RESOLVED_MARKET,
    "id": "resolved-no-time",
    "resolutionTime": None,
}

NO_CLOSE_TIME: dict[str, Any] = {
    **RESOLVED_MARKET,
    "id": "no-close",
    "closeTime": None,
}


def _bet(index: int, before: float, after: float, amount: float = 25.0,
         outcome: str = "YES", redemption: bool = False) -> dict[str, Any]:
    return {
        "id": f"bet-{index:04d}",
        "contractId": "resolved-no-1",
        "createdTime": 1723334400000 + index * 3_600_000,
        "probBefore": before,
        "probAfter": after,
        "amount": amount,
        "shares": amount * 2,
        "outcome": outcome,
        "isRedemption": redemption,
    }


# A drifting price series with realistic small impacts.
BETS: list[dict[str, Any]] = [
    _bet(0, 0.50, 0.54),
    _bet(1, 0.54, 0.51, outcome="NO"),
    _bet(2, 0.51, 0.47, outcome="NO"),
    _bet(3, 0.47, 0.49),
    _bet(4, 0.49, 0.44, outcome="NO"),
    _bet(5, 0.44, 0.40, outcome="NO"),
    # A redemption: a share settlement, not a price-forming trade.
    _bet(6, 0.40, 0.40, redemption=True),
    _bet(7, 0.40, 0.36, outcome="NO"),
]

SEARCH_PAGE: list[dict[str, Any]] = [
    RESOLVED_MARKET,
    RESOLVED_YES_MARKET,
    AMBIGUOUS_MARKET,
    CANCELLED_MARKET,
    UNRESOLVED_MARKET,
    MULTI_MARKET,
]


def encode(payload: Any) -> bytes:
    """Serialize a fixture the way the venue would send it."""
    return json.dumps(payload).encode("utf-8")
