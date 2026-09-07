"""Deterministic synthetic DEMO DATA generator.

Everything produced here is fabricated by a seeded pseudo-random number
generator so the app runs with no network access and no API keys. It is **not**
real market data, it is not derived from any venue, and it must never be
presented as such: every record carries ``source="SYNTHETIC_DEMO"`` and every
market question is prefixed with ``[DEMO DATA]``.
"""

from __future__ import annotations

import random
from datetime import UTC, datetime, timedelta
from typing import Any

from core.dataset import DEMO_LABEL, MarketDataset
from core.models import (
    DEMO_SOURCE,
    Market,
    MarketStatus,
    Outcome,
    Quote,
    Side,
    Trade,
)

# Question templates are intentionally generic and obviously fictional.
_TEMPLATES: list[tuple[str, str]] = [
    ("Will the fictional Blue team win the DEMO league final?", "sports"),
    ("Will DEMO City record above-average rainfall this quarter?", "weather"),
    ("Will the imaginary Widget Corp ship product X before the deadline?", "business"),
    ("Will the sample index close above its fictional threshold?", "finance"),
    ("Will the demo satellite launch happen in the simulated window?", "science"),
    ("Will the made-up policy proposal pass the fictional council?", "civics"),
    ("Will the notional ferry service resume on the demo route?", "transport"),
    ("Will the invented startup announce a funding round this window?", "business"),
    ("Will the simulated harvest exceed its fictional baseline?", "agriculture"),
    ("Will the pretend film top the imaginary box-office chart?", "media"),
]


DemoDataset = MarketDataset
"""Backwards-compatible alias: the demo generator returns a MarketDataset."""


def _parse_start(value: str | datetime) -> datetime:
    """Parse the configured start timestamp into an aware UTC datetime."""
    if isinstance(value, datetime):
        dt = value
    else:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return dt.astimezone(UTC) if dt.tzinfo else dt.replace(tzinfo=UTC)


def _clamp(value: float, low: float, high: float) -> float:
    """Clamp ``value`` into ``[low, high]``."""
    return max(low, min(high, value))


def _round_tick(value: float, tick: float) -> float:
    """Round a price to the nearest tick and keep it strictly inside (0, 1)."""
    ticks = round(value / tick)
    return _clamp(round(ticks * tick, 10), tick, 1.0 - tick)


def generate_demo_dataset(
    config: dict[str, Any] | None = None, **overrides: Any
) -> MarketDataset:
    """Generate a reproducible synthetic dataset of markets, quotes and trades.

    The same ``seed`` always yields byte-identical output. Each dataset
    deliberately contains imperfections a real feed would have: variable
    spreads, one-sided (missing) quotes and stale periods where a market stops
    updating.

    Args:
        config: The ``data`` section of ``config/settings.yaml``.
        **overrides: Individual settings overrides (e.g. ``seed=1``).

    Returns:
        A :class:`core.dataset.MarketDataset` sorted chronologically.
    """
    cfg: dict[str, Any] = dict(config or {})
    cfg.update(overrides)

    seed = int(cfg.get("seed", 20240617))
    n_markets = int(cfg.get("n_markets", 4))
    start = _parse_start(cfg.get("start", "2026-01-05T09:00:00Z"))
    step = timedelta(seconds=int(cfg.get("quote_interval_seconds", 300)))
    n_steps = int(cfg.get("n_steps", 240))
    trades_per_step = float(cfg.get("trades_per_step", 1.2))
    base_spread = float(cfg.get("base_spread", 0.02))
    spread_jitter = float(cfg.get("spread_jitter", 0.015))
    min_size = float(cfg.get("min_size", 25))
    max_size = float(cfg.get("max_size", 800))
    missing_quote_rate = float(cfg.get("missing_quote_rate", 0.04))
    stale_gap_rate = float(cfg.get("stale_gap_rate", 0.03))
    stale_gap_steps = int(cfg.get("stale_gap_steps", 12))
    tick = float(cfg.get("tick_size", 0.01))
    fee_rate = float(cfg.get("fee_rate", 0.02))
    # Markets run in a rolling sequence rather than all at once: market i opens
    # later than market i-1 and closes after a fixed lifetime. Without this,
    # every market would resolve at the same instant and a chronological
    # calibration split would have no train/test separation to work with.
    lifetime_fraction = float(cfg.get("lifetime_fraction", 0.40))
    stagger_fraction = float(cfg.get("stagger_fraction", 0.60))

    if n_markets < 3:
        raise ValueError("the demo dataset requires at least 3 markets")

    markets: list[Market] = []
    quotes: list[Quote] = []
    trades: list[Trade] = []

    lifetime_steps = max(10, round(n_steps * lifetime_fraction))
    settle_delay = timedelta(hours=1)

    for index in range(n_markets):
        rng = random.Random(seed + index * 1_000)
        question, category = _TEMPLATES[index % len(_TEMPLATES)]
        market_id = f"DEMO-{index + 1:02d}"

        # Anchors are spread deterministically across the probability range so a
        # demo dataset contains both likely and unlikely markets (and therefore
        # a mix of YES/NO outcomes to score calibration against).
        span = max(1, n_markets - 1)
        anchor = _clamp(0.25 + 0.5 * index / span + rng.gauss(0.0, 0.04), 0.1, 0.9)
        true_prob = _clamp(anchor + rng.gauss(0.0, 0.05), 0.05, 0.95)
        drift_strength = rng.uniform(0.01, 0.05)
        volatility = rng.uniform(0.006, 0.020)

        # Rolling window: this market is live from open_step to close_step.
        open_step = round(index * n_steps * stagger_fraction / max(1, n_markets))
        close_step = min(n_steps, open_step + lifetime_steps)
        open_time = start + step * open_step
        close_time = start + step * close_step
        resolution_time = close_time + settle_delay

        # Two imperfections are forced (not sampled), positioned relative to this
        # market's own window so every dataset exhibits them at any size.
        forced_one_sided_step = open_step + max(5, lifetime_steps // 4)
        forced_stale_start = open_step + max(8, lifetime_steps // 2)

        stale_until = -1
        for stepno in range(open_step, close_step):
            ts = start + step * stepno
            true_prob = _clamp(
                true_prob
                + drift_strength * (anchor - true_prob)
                + rng.gauss(0.0, volatility),
                0.02,
                0.98,
            )

            # During a freeze the book stops updating (our quote goes stale) but
            # the tape keeps printing at the true price - which is exactly the
            # situation the stale-data risk rule exists to catch.
            frozen = stepno < stale_until
            if not frozen and (
                stepno == forced_stale_start
                or (stepno > open_step + 5 and rng.random() < stale_gap_rate)
            ):
                stale_until = stepno + stale_gap_steps
                frozen = True

            observed = _clamp(true_prob + rng.gauss(0.0, 0.012), 0.02, 0.98)
            spread = max(tick, base_spread + abs(rng.gauss(0.0, spread_jitter)))
            bid = _round_tick(observed - spread / 2.0, tick)
            ask = _round_tick(observed + spread / 2.0, tick)
            if ask <= bid:
                ask = _round_tick(bid + tick, tick)
            bid_size = float(round(rng.uniform(min_size, max_size)))
            ask_size = float(round(rng.uniform(min_size, max_size)))

            if not frozen:
                drop_bid: bool | None = None
                if stepno == forced_one_sided_step:
                    drop_bid = True
                elif rng.random() < missing_quote_rate:
                    drop_bid = rng.random() < 0.5

                quotes.append(
                    Quote(
                        market_id=market_id,
                        ts=ts,
                        bid=None if drop_bid is True else bid,
                        ask=None if drop_bid is False else ask,
                        bid_size=0.0 if drop_bid is True else bid_size,
                        ask_size=0.0 if drop_bid is False else ask_size,
                        source=DEMO_SOURCE,
                    )
                )

            n_prints = int(trades_per_step) + (
                1 if rng.random() < (trades_per_step % 1.0) else 0
            )
            for print_no in range(n_prints):
                aggressor = Side.BUY if rng.random() < 0.5 else Side.SELL
                price = ask if aggressor is Side.BUY else bid
                trades.append(
                    Trade(
                        trade_id=f"{market_id}-T{stepno:04d}-{print_no}",
                        market_id=market_id,
                        ts=ts + timedelta(seconds=1 + print_no),
                        price=_clamp(price, 0.0, 1.0),
                        size=float(round(rng.uniform(1.0, min_size))),
                        aggressor=aggressor,
                        source=DEMO_SOURCE,
                    )
                )

        # The outcome is drawn from a dedicated generator so it depends only on
        # the seed and the final probability, not on how many draws the quote
        # loop happened to consume.
        outcome_rng = random.Random(seed + index * 1_000 + 777)
        outcome = Outcome.YES if outcome_rng.random() < true_prob else Outcome.NO
        markets.append(
            Market(
                market_id=market_id,
                question=f"[{DEMO_LABEL}] {question}",
                category=category,
                created_at=open_time,
                close_time=close_time,
                resolution_time=resolution_time,
                outcome=outcome,
                status=MarketStatus.RESOLVED,
                tick_size=tick,
                fee_rate=fee_rate,
                source=DEMO_SOURCE,
                is_demo=True,
            )
        )

    quotes.sort(key=lambda q: (q.ts, q.market_id))
    trades.sort(key=lambda t: (t.ts, t.market_id))
    return MarketDataset(
        markets=markets,
        quotes=quotes,
        trades=trades,
        seed=seed,
        label=DEMO_LABEL,
        source=DEMO_SOURCE,
    )
