"""Named strategy presets, in plain language, for people learning the domain.

The settings dock exposes thirty-five numeric controls. Every one of them is
real and matters, but almost none of them mean anything until you already know
what they do - and a wrong value produces a plausible-looking result rather
than an error, which is the worst kind of confusing.

A preset is a small, named set of those numbers with an explanation of what it
changes and, more importantly, **what you should expect to see**. The point is
not to hide the controls. It is to give a beginner a defensible starting point
and a reason to reach for a specific knob later.

No Qt import here, so the text and the numbers can be tested without a GUI.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Preset:
    """One named configuration, with the reasoning attached."""

    key: str
    name: str
    summary: str
    """One line. Shown next to the selector."""

    expect: str
    """What the user should expect to SEE. The most useful sentence here -
    "you will see very few trades" turns a confusing empty table into a
    confirmation that the setting is working."""

    changes: tuple[str, ...] = ()
    """Plain-language description of each meaningful difference."""

    overrides: dict[str, dict[str, Any]] = field(default_factory=dict)
    """Settings sections to merge over the loaded configuration."""


CAUTIOUS = Preset(
    key="cautious",
    name="Cautious",
    summary="Trades only on large disagreements with the market.",
    expect=(
        "Very few trades - often none at all on a short run. That is the preset "
        "working, not a bug. A strategy that rarely trades is how a demanding "
        "edge threshold looks from the outside."
    ),
    changes=(
        "Needs a 5% edge over the price it would have to pay, versus 2% normally.",
        "Trades in smaller size, so any single market matters less.",
        "Refuses wider spreads and staler prices.",
    ),
    overrides={
        "strategy": {"min_edge": 0.05, "base_size": 50, "max_order_size": 150},
        "risk": {
            "max_position_size": 150,
            "max_event_exposure": 200.0,
            "max_total_exposure": 600.0,
            "max_spread": 0.04,
            "max_data_age_seconds": 600,
        },
    },
)

BALANCED = Preset(
    key="balanced",
    name="Balanced",
    summary="The project's default settings.",
    expect=(
        "A handful of trades on the demo data. Enough to see the machinery work "
        "end to end without the result being dominated by one position."
    ),
    changes=(
        "Needs a 2% edge after fees and estimated slippage.",
        "Standard position sizes and risk limits.",
    ),
    overrides={},
)

ADVENTUROUS = Preset(
    key="adventurous",
    name="Adventurous",
    summary="Trades more often, on thinner edges.",
    expect=(
        "Many more trades, and a noticeably wider spread of outcomes when you run "
        "a campaign. More trades is not better - watch whether the confidence "
        "band gets wider rather than the median getting higher."
    ),
    changes=(
        "Accepts a 1% edge, so it acts on weaker signals.",
        "Larger positions, so single markets move the result more.",
        "Tolerates wider spreads.",
    ),
    overrides={
        "strategy": {"min_edge": 0.01, "base_size": 150, "max_order_size": 600},
        "risk": {
            "max_position_size": 500,
            "max_event_exposure": 800.0,
            "max_total_exposure": 2400.0,
            "max_spread": 0.08,
        },
    },
)

NO_TRADING = Preset(
    key="flat",
    name="Do nothing",
    summary="The kill switch: every proposal is rejected.",
    expect=(
        "Zero trades and a flat line at 0%. Useful as a sanity check - it is the "
        "baseline every strategy has to beat, and it costs nothing to lose to."
    ),
    changes=("The risk agent rejects every proposal before it can be filled.",),
    overrides={"risk": {"kill_switch": True}},
)

PRESETS: tuple[Preset, ...] = (CAUTIOUS, BALANCED, ADVENTUROUS, NO_TRADING)
DEFAULT_PRESET = "balanced"

BY_KEY: dict[str, Preset] = {p.key: p for p in PRESETS}


def get(key: str) -> Preset:
    """Look up a preset, falling back to Balanced for an unknown key."""
    return BY_KEY.get(key, BALANCED)


def apply(settings: dict[str, Any], key: str) -> dict[str, Any]:
    """Return a copy of ``settings`` with the preset's overrides merged in.

    Merges one level into each named section, so a preset only has to state the
    values it actually changes and everything else keeps whatever the loaded
    configuration said.
    """
    merged = copy.deepcopy(settings)
    for section, values in get(key).overrides.items():
        current = merged.get(section)
        merged[section] = {**current, **values} if isinstance(current, dict) else dict(values)
    return merged


# --------------------------------------------------------------------------
# Explanations for the things a beginner will misread
# --------------------------------------------------------------------------

CAMPAIGN_EXPLAINER = (
    "A single backtest is a single sample. Run the same strategy over a slightly "
    "different set of markets and the number moves - often enough to flip its "
    "sign. So this runs it many times over random subsets of the data and shows "
    "the whole spread of outcomes.\n\n"
    "The dark line is the median run. The shaded band holds the middle 90% of "
    "them.\n\n"
    "The question to ask is not \"is the line up?\" It is \"does the band stay "
    "clear of zero?\" If the band includes zero, the strategy has not been "
    "distinguished from doing nothing, however good the median looks."
)

CAMPAIGN_CAVEAT = (
    "This measures how much the result depends on which markets you happened to "
    "look at. It does NOT show that the strategy works on markets it has never "
    "seen - that is what the Research tab's walk-forward and promotion gate are "
    "for, and no amount of resampling substitutes for them."
)

TERMS: tuple[tuple[str, str], ...] = (
    (
        "Edge",
        "How far the agents' estimated probability sits from the price you would "
        "actually have to pay. Only edge that survives fees and slippage is worth "
        "acting on.",
    ),
    (
        "Return",
        "Profit or loss as a percentage of starting capital. +2% means the "
        "portfolio ended 2% above where it began.",
    ),
    (
        "Drawdown",
        "The worst peak-to-trough fall during a run. A strategy that ends up 5% "
        "but fell 30% on the way is not the same as one that never fell.",
    ),
    (
        "Confidence band",
        "The range the middle 90% of runs fell in. Wide means the outcome depends "
        "heavily on luck; narrow means it is consistent. Narrow and above zero is "
        "what you are looking for.",
    ),
    (
        "Brier score",
        "How good the probability estimates are, lower is better. 0.25 is what you "
        "get by always guessing 50%. Beating it means the forecasts carry real "
        "information.",
    ),
    (
        "Effective n",
        "The number of markets that actually resolved - not the number of rows. A "
        "market forecast 400 times still only resolves once, so it is one piece of "
        "evidence, not 400.",
    ),
)
