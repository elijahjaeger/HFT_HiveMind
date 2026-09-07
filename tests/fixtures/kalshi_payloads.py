"""Canned Kalshi API payloads used by the adapter tests.

These are hand-built to match the field names and fixed-point string formats in
the published Kalshi Trade API v2 schemas. They are FIXTURES, not captured
production data, and no test in this suite touches the network.
"""

from __future__ import annotations

import json
from typing import Any

SETTLED_MARKET: dict[str, Any] = {
    "ticker": "KXHIGHNY-25AUG31-B70",
    "event_ticker": "KXHIGHNY-25AUG31",
    "market_type": "binary",
    "title": "Will the high temperature in NYC on Aug 31 be 70-71F?",
    "yes_sub_title": "70-71F",
    "no_sub_title": "Not 70-71F",
    "created_time": "2026-08-24T12:00:00Z",
    "open_time": "2026-08-30T13:00:00Z",
    "close_time": "2026-08-31T23:00:00Z",
    "expected_expiration_time": "2026-08-31T23:30:00Z",
    "latest_expiration_time": "2026-09-01T01:00:00Z",
    "status": "settled",
    "yes_bid_dollars": "0.4200",
    "yes_ask_dollars": "0.4500",
    "no_bid_dollars": "0.5500",
    "no_ask_dollars": "0.5800",
    "yes_bid_size_fp": "250.00",
    "yes_ask_size_fp": "310.00",
    "last_price_dollars": "0.4400",
    "volume_fp": "18400.00",
    "volume_24h_fp": "5200.00",
    "open_interest_fp": "9100.00",
    "result": "yes",
    "settlement_value_dollars": "1.0000",
    "settlement_ts": "2026-09-01T00:15:00Z",
    "rules_primary": (
        "If the high temperature in New York City on Aug 31, 2026 is between "
        "70 and 71 degrees Fahrenheit inclusive, then the market resolves to Yes."
    ),
    "rules_secondary": (
        "The high temperature is determined by the NWS daily climate report for "
        "Central Park. If the report is unavailable the market may be voided."
    ),
    "price_ranges": [{"start": "0.0100", "end": "0.9900", "step": "0.0100"}],
    "can_close_early": True,
    "is_provisional": False,
}

UNSETTLED_MARKET: dict[str, Any] = {
    **SETTLED_MARKET,
    "ticker": "KXHIGHNY-25AUG31-B72",
    "title": "Will the high temperature in NYC on Aug 31 be 72-73F?",
    "status": "active",
    "result": "",
    "settlement_value_dollars": None,
    "settlement_ts": None,
}

SCALAR_MARKET: dict[str, Any] = {
    **SETTLED_MARKET,
    "ticker": "KXSCALAR-25AUG31",
    "market_type": "scalar",
    "result": "scalar",
}

CANDLESTICKS: dict[str, Any] = {
    "ticker": "KXHIGHNY-25AUG31-B70",
    "candlesticks": [
        {
            "end_period_ts": 1788181200,  # 2026-08-31T13:00:00Z
            "yes_bid": {
                "open_dollars": "0.3800", "low_dollars": "0.3700",
                "high_dollars": "0.4000", "close_dollars": "0.3900",
            },
            "yes_ask": {
                "open_dollars": "0.4100", "low_dollars": "0.4000",
                "high_dollars": "0.4300", "close_dollars": "0.4200",
            },
            "price": {
                "open_dollars": "0.4000", "low_dollars": "0.3800",
                "high_dollars": "0.4200", "close_dollars": "0.4100",
                "mean_dollars": "0.4050", "previous_dollars": "0.3900",
            },
            "volume_fp": "1200.00",
            "open_interest_fp": "8400.00",
        },
        {
            "end_period_ts": 1788184800,  # 2026-08-31T14:00:00Z
            "yes_bid": {
                "open_dollars": "0.3900", "low_dollars": "0.3900",
                "high_dollars": "0.4400", "close_dollars": "0.4300",
            },
            "yes_ask": {
                "open_dollars": "0.4200", "low_dollars": "0.4200",
                "high_dollars": "0.4700", "close_dollars": "0.4600",
            },
            "price": {
                "open_dollars": "0.4100", "low_dollars": "0.4000",
                "high_dollars": "0.4500", "close_dollars": "0.4400",
                "mean_dollars": "0.4300", "previous_dollars": "0.4100",
            },
            "volume_fp": "900.00",
            "open_interest_fp": "8600.00",
        },
        {
            # One-sided: no resting bid ($0.00) and no resting ask ($1.00).
            "end_period_ts": 1788188400,  # 2026-08-31T15:00:00Z
            "yes_bid": {
                "open_dollars": "0.0000", "low_dollars": "0.0000",
                "high_dollars": "0.0000", "close_dollars": "0.0000",
            },
            "yes_ask": {
                "open_dollars": "0.4600", "low_dollars": "0.4600",
                "high_dollars": "0.5000", "close_dollars": "0.4800",
            },
            "price": {
                "open_dollars": None, "low_dollars": None, "high_dollars": None,
                "close_dollars": None, "mean_dollars": None, "previous_dollars": "0.4400",
            },
            "volume_fp": "0.00",
            "open_interest_fp": "8600.00",
        },
        {
            # Crossed candle (bid above ask): must be dropped, never coerced.
            "end_period_ts": 1788192000,  # 2026-08-31T16:00:00Z
            "yes_bid": {"close_dollars": "0.6000"},
            "yes_ask": {"close_dollars": "0.5000"},
            "price": {"close_dollars": "0.5500"},
            "volume_fp": "50.00",
            "open_interest_fp": "8600.00",
        },
    ],
}

# The historical endpoint uses `volume`/`open_interest` rather than the
# `_fp`-suffixed names used by the live endpoint.
HISTORICAL_CANDLESTICKS: dict[str, Any] = {
    "ticker": "KXHIGHNY-25AUG31-B70",
    "candlesticks": [
        {
            "end_period_ts": 1788181200,
            "yes_bid": {"close_dollars": "0.3900"},
            "yes_ask": {"close_dollars": "0.4200"},
            "price": {"close_dollars": "0.4100"},
            "volume": "1200.00",
            "open_interest": "8400.00",
        }
    ],
}

TRADES_PAGE_1: dict[str, Any] = {
    "trades": [
        {
            "trade_id": "3f0a1b2c-0001",
            "ticker": "KXHIGHNY-25AUG31-B70",
            "count_fp": "25.00",
            "yes_price_dollars": "0.4200",
            "no_price_dollars": "0.5800",
            "taker_outcome_side": "yes",
            "taker_book_side": "ask",
            "created_time": "2026-08-31T13:05:00Z",
            "is_block_trade": False,
        },
        {
            "trade_id": "3f0a1b2c-0002",
            "ticker": "KXHIGHNY-25AUG31-B70",
            "count_fp": "10.50",
            "yes_price_dollars": "0.4100",
            "no_price_dollars": "0.5900",
            "taker_outcome_side": "no",
            "taker_book_side": "bid",
            "created_time": "2026-08-31T13:20:00Z",
            "is_block_trade": False,
        },
    ],
    "cursor": "PAGE2",
}

TRADES_PAGE_2: dict[str, Any] = {
    "trades": [
        {
            "trade_id": "3f0a1b2c-0003",
            "ticker": "KXHIGHNY-25AUG31-B70",
            "count_fp": "5.00",
            "yes_price_dollars": "0.4400",
            "no_price_dollars": "0.5600",
            "taker_outcome_side": "yes",
            "taker_book_side": "ask",
            "created_time": "2026-08-31T14:10:00Z",
            "is_block_trade": False,
        }
    ],
    "cursor": "",
}

MARKETS_PAGE: dict[str, Any] = {"markets": [SETTLED_MARKET, UNSETTLED_MARKET], "cursor": ""}

CUTOFF: dict[str, Any] = {
    "market_settled_ts": "2026-06-01T00:00:00Z",
    "trades_created_ts": "2026-06-01T00:00:00Z",
}


def encode(payload: Any) -> bytes:
    """Serialize a fixture the way the venue would send it."""
    return json.dumps(payload).encode("utf-8")
