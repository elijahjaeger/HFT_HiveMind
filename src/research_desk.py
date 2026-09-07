"""CrewAI research desk: two named personas producing an :class:`AlphaVector`.

The desk has two seats, and they are deliberately given *different information*
rather than different prompts over the same information:

``MacroAnalyst``
    Works from the market-level view - the implied probability in the book and
    the drift of recent prints. Answers "what does this market think, and which
    way has it been moving?"

``MicrostructureQuant``
    Works from the shape of the book - the size-weighted microprice, the depth
    imbalance, the spread and how stale the quote is. Answers "is the top of
    book telling us anything, and is it even tradable right now?"

Two execution paths, one contract
---------------------------------
Both paths return the same :class:`AlphaVector`, so nothing downstream knows or
cares which ran:

* **Deterministic (default).** The personas are backed by the existing
  ``MarketAgent``/``TapeAgent`` and ``BookImbalanceAgent``/``LiquidityAgent``.
  No network, no key, reproducible. This is what runs offline and what the
  tests exercise.
* **LLM-backed.** When an LLM key is present in the environment *and* the
  caller passes ``use_llm=True``, the CrewAI crew runs and its structured
  output is merged over the deterministic vector - never replacing it, so a
  hallucinated or unparseable answer degrades to the deterministic view instead
  of poisoning the policy.

This project ships **no credentials and accepts none**. ``use_llm`` defaults to
``False`` and the desk is fully functional without it.
"""

from __future__ import annotations

import dataclasses
import os
import re
from typing import Any

from crewai import Agent as CrewAgent
from crewai import Crew, Process, Task

from agents.base import ForecastAgent, combine_forecasts
from agents.imbalance import BookImbalanceAgent
from agents.liquidity import LiquidityAgent
from agents.market import MarketAgent
from agents.tape import TapeAgent
from core.models import Forecast, ForecastInput, ForecastStatus, MarketState
from src.contracts import AlphaVector

RESEARCH_DESK_VERSION = "research-desk-v1"

# Environment variables that indicate an LLM is actually reachable. Absence of
# every one of them is what forces the deterministic path.
_LLM_KEY_VARS = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GROQ_API_KEY")

MACRO_ANALYST = {
    "role": "MacroAnalyst",
    "goal": (
        "Judge where this prediction market should settle from its own implied "
        "probability and the direction of recent trade prints, and say how much "
        "to trust that judgement."
    ),
    "backstory": (
        "You price event risk for a research desk. You are sceptical of short "
        "bursts of activity and you never treat a price near certainty as "
        "evidence of the outcome - a market at $0.98 has not settled YES. You "
        "state a probability and a confidence, and you say plainly when the "
        "data does not support either."
    ),
}

MICROSTRUCTURE_QUANT = {
    "role": "MicrostructureQuant",
    "goal": (
        "Read the top of the order book - microprice, depth imbalance, spread "
        "and quote staleness - and report what it implies for fair value and "
        "whether the market is tradable at all right now."
    ),
    "backstory": (
        "You study order books. You know the microprice always sits inside the "
        "spread, so it can tilt a view but can never on its own justify "
        "crossing that spread. You treat a one-sided book, a zero-size book or "
        "a stale quote as an absence of information, not as a signal."
    ),
}


def llm_is_configured() -> bool:
    """True when some LLM credential is present in the environment."""
    return any(os.environ.get(var) for var in _LLM_KEY_VARS)


def build_personas(verbose: bool = False) -> dict[str, CrewAgent]:
    """Instantiate the two CrewAI personas.

    Constructed with ``allow_delegation=False``: the desk is two independent
    opinions that the supervisor blends, not a conversation in which one seat
    can talk the other into agreeing.
    """
    return {
        spec["role"]: CrewAgent(
            role=spec["role"],
            goal=spec["goal"],
            backstory=spec["backstory"],
            allow_delegation=False,
            verbose=verbose,
        )
        for spec in (MACRO_ANALYST, MICROSTRUCTURE_QUANT)
    }


def build_crew(verbose: bool = False) -> Crew:
    """Build the two-persona crew that produces a research view."""
    personas = build_personas(verbose=verbose)
    macro, micro = personas["MacroAnalyst"], personas["MicrostructureQuant"]

    macro_task = Task(
        description=(
            "Market {market_id} at {as_of}. Implied probability {mid}, recent "
            "print trend {tape}. State P(YES) in [0,1] and a confidence in "
            "[0,1]. Reply exactly as: probability=<x> confidence=<y>"
        ),
        expected_output="probability=<float> confidence=<float>",
        agent=macro,
    )
    micro_task = Task(
        description=(
            "Market {market_id} at {as_of}. Microprice {microprice}, depth "
            "imbalance {imbalance}, spread {spread}, quote age {quote_age}s. "
            "State P(YES) in [0,1] and a confidence in [0,1]. If the book is "
            "one-sided, zero-size or stale, return confidence=0. "
            "Reply exactly as: probability=<x> confidence=<y>"
        ),
        expected_output="probability=<float> confidence=<float>",
        agent=micro,
    )
    return Crew(
        agents=[macro, micro],
        tasks=[macro_task, micro_task],
        process=Process.sequential,
        verbose=verbose,
    )


def _parse_persona_output(text: str) -> tuple[float | None, float]:
    """Pull ``probability=`` and ``confidence=`` out of a reply from a persona.

    Returns ``(None, 0.0)`` on anything unparseable, which is what makes an LLM
    failure degrade to the deterministic view rather than corrupt it.
    """
    prob = re.search(r"probability\s*=\s*([0-9]*\.?[0-9]+)", text, re.I)
    conf = re.search(r"confidence\s*=\s*([0-9]*\.?[0-9]+)", text, re.I)
    if prob is None:
        return None, 0.0
    try:
        value = float(prob.group(1))
        certainty = float(conf.group(1)) if conf else 0.0
    except (TypeError, ValueError):
        return None, 0.0
    if not 0.0 <= value <= 1.0:
        return None, 0.0
    return value, max(0.0, min(1.0, certainty))


class ResearchDesk:
    """Runs the two personas over a :class:`core.models.MarketState`."""

    def __init__(
        self,
        settings: dict[str, Any] | None = None,
        use_llm: bool = False,
        verbose: bool = False,
    ) -> None:
        cfg = dict(settings or {}).get("agents", {})
        self.settings = dict(settings or {})
        self.use_llm = bool(use_llm)
        self.verbose = verbose
        # Deterministic backings for each seat.
        self._macro_price = MarketAgent(cfg.get("market", {}))
        self._macro_tape = TapeAgent(cfg.get("tape", {}))
        self._micro_book = BookImbalanceAgent(cfg.get("imbalance", {}))
        self._micro_liquidity = LiquidityAgent(cfg.get("liquidity", {}))
        self._ensemble_cfg = cfg.get("ensemble", {})
        self._crew: Crew | None = None

    @property
    def crew(self) -> Crew:
        """The CrewAI crew, built lazily so offline use never constructs it."""
        if self._crew is None:
            self._crew = build_crew(verbose=self.verbose)
        return self._crew

    def analyze(self, state: MarketState) -> AlphaVector:
        """Produce one alpha vector for one market at one instant."""
        macro_forecasts = [
            self._macro_price.analyze(state),
            self._macro_tape.analyze(state),
        ]
        micro_forecast = self._micro_book.analyze(state)
        liquidity = self._micro_liquidity.analyze(state)

        macro = combine_forecasts(
            macro_forecasts, self._ensemble_cfg, state.market.market_id
        )
        blended = combine_forecasts(
            [*macro_forecasts, micro_forecast], self._ensemble_cfg, state.market.market_id
        )

        quote = state.quote
        depth_imbalance = 0.0
        if quote is not None and quote.bid_size + quote.ask_size > 0.0:
            depth_imbalance = (quote.bid_size - quote.ask_size) / (
                quote.bid_size + quote.ask_size
            )

        notes: list[str] = list(getattr(liquidity, "warnings", []) or [])
        macro_confidence = max(
            (f.confidence for f in macro_forecasts if f.probability is not None),
            default=0.0,
        )

        vector = AlphaVector(
            market_id=state.market.market_id,
            as_of=state.as_of,
            macro_probability=macro.probability,
            macro_confidence=macro_confidence,
            micro_probability=micro_forecast.probability,
            micro_confidence=micro_forecast.confidence,
            consensus_probability=blended.probability,
            disagreement=blended.disagreement,
            spread=quote.spread if quote else None,
            depth_imbalance=depth_imbalance,
            quote_age_seconds=quote.age_seconds(state.as_of) if quote else 0.0,
            source="deterministic",
            notes=notes,
        )

        if self.use_llm and llm_is_configured():
            vector = self._merge_llm_view(vector)
        return vector

    def _merge_llm_view(self, vector: AlphaVector) -> AlphaVector:
        """Overlay the answer from the crew on the deterministic vector.

        Any failure - network, parsing, an out-of-range probability - leaves the
        deterministic vector untouched and records why in ``notes``.
        """
        try:
            output = self.crew.kickoff(
                inputs={
                    "market_id": vector.market_id,
                    "as_of": vector.as_of.isoformat(),
                    "mid": vector.macro_probability,
                    "tape": vector.macro_probability,
                    "microprice": vector.micro_probability,
                    "imbalance": round(vector.depth_imbalance, 4),
                    "spread": vector.spread,
                    "quote_age": round(vector.quote_age_seconds),
                }
            )
        except Exception as error:  # noqa: BLE001 - degrade, never raise
            return dataclasses.replace(
                vector, notes=[*vector.notes, f"LLM path failed: {type(error).__name__}"]
            )

        outputs = [str(t.raw) for t in getattr(output, "tasks_output", [])] or [str(output)]
        macro_p, macro_c = _parse_persona_output(outputs[0])
        micro_p, micro_c = (
            _parse_persona_output(outputs[-1]) if len(outputs) > 1 else (None, 0.0)
        )

        merged = dataclasses.replace(
            vector,
            macro_probability=macro_p if macro_p is not None else vector.macro_probability,
            macro_confidence=macro_c if macro_p is not None else vector.macro_confidence,
            micro_probability=micro_p if micro_p is not None else vector.micro_probability,
            micro_confidence=micro_c if micro_p is not None else vector.micro_confidence,
            source="crewai-llm",
        )
        usable = [
            (p, c)
            for p, c in (
                (merged.macro_probability, merged.macro_confidence),
                (merged.micro_probability, merged.micro_confidence),
            )
            if p is not None and c > 0.0
        ]
        if usable:
            total = sum(c for _, c in usable)
            probabilities = [p for p, _ in usable]
            merged = dataclasses.replace(
                merged,
                consensus_probability=(
                    sum(p * c for p, c in usable) / total if total > 0 else None
                ),
                disagreement=(
                    max(probabilities) - min(probabilities)
                    if len(probabilities) > 1
                    else 0.0
                ),
            )
        return merged

    def run(self, states: list[MarketState]) -> list[AlphaVector]:
        """Analyse many decision points, preserving order."""
        return [self.analyze(state) for state in states]


class ResearchDeskAgent(ForecastAgent):
    """Exposes the desk through the existing ``analyze()`` interface.

    This is the seat README Roadmap item 4 reserved for "LLM-backed research
    agents behind the same ``analyze()`` interface", so the desk can be dropped
    into ``run_backtest`` and graded by the evaluation harness that already
    exists rather than judged on its own say-so.
    """

    name = "ResearchDeskAgent"

    def __init__(
        self, settings: dict[str, Any] | None = None, use_llm: bool = False
    ) -> None:
        super().__init__((settings or {}).get("agents", {}))
        self.desk = ResearchDesk(settings, use_llm=use_llm)

    def analyze(self, state: MarketState) -> Forecast:
        """Return the blended view from the desk as a standard forecast."""
        vector = self.desk.analyze(state)
        inputs: list[ForecastInput] = []
        if state.quote is not None:
            inputs.append(
                ForecastInput(name="quote", ts=state.quote.ts, source=state.quote.source)
            )
        has_view = vector.has_view
        return Forecast(
            agent_name=self.name,
            market_id=state.market.market_id,
            ts=state.as_of,
            probability=vector.consensus_probability,
            confidence=(
                max(vector.macro_confidence, vector.micro_confidence) if has_view else 0.0
            ),
            status=ForecastStatus.OK if has_view else ForecastStatus.NO_DATA,
            rationale=(
                f"MacroAnalyst {vector.macro_probability} / MicrostructureQuant "
                f"{vector.micro_probability} -> {vector.consensus_probability} "
                f"(disagreement {vector.disagreement:.3f}, source {vector.source})."
            ),
            inputs=inputs,
            assumptions=[
                "MacroAnalyst reads implied price and print drift only.",
                "MicrostructureQuant reads top-of-book shape only; deeper levels unobserved.",
                "Personas are blended by confidence, not by a fitted weight.",
                f"Alpha contract {RESEARCH_DESK_VERSION}.",
            ],
            details={"alpha": vector.as_row()},
        )
