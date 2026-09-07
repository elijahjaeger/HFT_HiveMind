"""Tests for the LangGraph / CrewAI / FinRL architecture in ``src/``.

These cover the three properties that matter and are easy to lose in a
refactor: the graph routes through every stage in order, the research desk
never reads data stamped after the decision instant, and the RL node reports
its evidence in resolved markets rather than in timesteps.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from core.config import get_settings
from core.models import ForecastStatus
from core.synthetic import generate_demo_dataset
from src.contracts import ALPHA_FEATURE_NAMES, AlphaVector, PolicyMetrics
from src.research_desk import (
    MACRO_ANALYST,
    MICROSTRUCTURE_QUANT,
    ResearchDesk,
    ResearchDeskAgent,
    _parse_persona_output,
    build_personas,
    llm_is_configured,
)
from src.rl_policy import OBSERVATION_NAMES, AlphaTradingEnv, RLPolicyPipeline
from src.state_graph import _PIPELINE, TradingGraph, build_states, compile_graph


@pytest.fixture(scope="module")
def settings() -> dict:
    return get_settings()


@pytest.fixture(scope="module")
def dataset(settings):
    return generate_demo_dataset(settings["data"])


@pytest.fixture(scope="module")
def as_of(dataset) -> datetime:
    return sorted(q.ts for q in dataset.quotes)[len(dataset.quotes) // 2]


# --------------------------------------------------------------------------
# Contracts
# --------------------------------------------------------------------------


def test_the_alpha_vector_is_dense_and_matches_its_declared_feature_order():
    vector = AlphaVector(market_id="X", as_of=datetime.now(UTC))
    assert len(vector.vector()) == len(ALPHA_FEATURE_NAMES)


def test_a_missing_probability_becomes_maximum_uncertainty_not_a_nan():
    vector = AlphaVector(market_id="X", as_of=datetime.now(UTC))
    values = dict(zip(ALPHA_FEATURE_NAMES, vector.vector(), strict=True))
    assert values["macro_probability"] == 0.5
    assert values["macro_confidence"] == 0.0
    assert not vector.has_view


def test_policy_metrics_lead_with_effective_n_not_the_step_count():
    rows = PolicyMetrics(timesteps=999_999, effective_n=4).as_rows()
    assert rows[0]["metric"].startswith("Resolved markets")
    assert rows[0]["value"] == 4


# --------------------------------------------------------------------------
# CrewAI research desk
# --------------------------------------------------------------------------


def test_the_desk_has_exactly_the_two_named_personas():
    personas = build_personas()
    assert set(personas) == {"MacroAnalyst", "MicrostructureQuant"}
    assert MACRO_ANALYST["role"] == "MacroAnalyst"
    assert MICROSTRUCTURE_QUANT["role"] == "MicrostructureQuant"


def test_personas_cannot_delegate_to_each_other():
    for persona in build_personas().values():
        assert persona.allow_delegation is False


def test_the_desk_runs_offline_with_no_credentials(monkeypatch):
    for var in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GROQ_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    assert llm_is_configured() is False


def test_the_desk_produces_alpha_without_touching_the_network(settings, dataset, as_of):
    states = build_states(dataset, as_of)
    alpha = ResearchDesk(settings).run(states)
    assert len(alpha) == len(states)
    assert any(v.has_view for v in alpha), "no persona produced a usable view"
    assert all(v.source == "deterministic" for v in alpha)


def test_alpha_never_carries_data_stamped_after_the_decision_instant(
    settings, dataset, as_of
):
    states = build_states(dataset, as_of)
    for state, vector in zip(states, ResearchDesk(settings).run(states), strict=True):
        assert vector.as_of == as_of
        if state.quote is not None:
            assert state.quote.ts <= as_of
        assert vector.quote_age_seconds >= 0.0


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("probability=0.7 confidence=0.4", (0.7, 0.4)),
        ("Probability = 0.25  Confidence = 0.9", (0.25, 0.9)),
        ("no numbers here", (None, 0.0)),
        ("probability=1.5 confidence=0.5", (None, 0.0)),
    ],
)
def test_persona_replies_that_cannot_be_parsed_degrade_rather_than_corrupt(text, expected):
    assert _parse_persona_output(text) == expected


def test_the_desk_plugs_into_the_existing_forecast_agent_interface(
    settings, dataset, as_of
):
    agent = ResearchDeskAgent(settings)
    forecasts = [agent.analyze(s) for s in build_states(dataset, as_of)]
    assert all(f.agent_name == "ResearchDeskAgent" for f in forecasts)
    assert not any(f.uses_future_input for f in forecasts), "look-ahead in the desk"
    assert all(
        f.status in {ForecastStatus.OK, ForecastStatus.NO_DATA} for f in forecasts
    )


def test_a_forecast_without_a_view_reports_no_data_and_zero_confidence(
    settings, dataset
):
    # An instant before any market opens: nothing is observable yet.
    earliest = min(q.ts for q in dataset.quotes) - timedelta(days=1)
    agent = ResearchDeskAgent(settings)
    forecast = agent.analyze(build_states(dataset, earliest)[0])
    assert forecast.probability is None
    assert forecast.confidence == 0.0
    assert forecast.status is ForecastStatus.NO_DATA


# --------------------------------------------------------------------------
# FinRL / SB3 policy
# --------------------------------------------------------------------------


def test_the_observation_space_is_alpha_plus_portfolio_state():
    assert len(OBSERVATION_NAMES) == len(ALPHA_FEATURE_NAMES) + 3


def test_the_environment_refuses_a_corpus_with_nothing_to_settle(settings, dataset):
    with pytest.raises(ValueError, match="settled outcome"):
        AlphaTradingEnv(dataset, alpha_by_market={}, settings=settings)


def _alpha_over_time(settings, dataset, n: int = 6) -> list[AlphaVector]:
    """Alpha vectors on a grid, which is what the env walks along."""
    stamps = sorted({q.ts for q in dataset.quotes})
    desk = ResearchDesk(settings)
    picked = stamps[:: max(1, len(stamps) // n)][:n]
    return [v for ts in picked for v in desk.run(build_states(dataset, ts)) if v.has_view]


def test_training_reports_evidence_in_resolved_markets(settings, dataset):
    alpha = _alpha_over_time(settings, dataset)
    pipeline = RLPolicyPipeline(settings, total_timesteps=256)
    metrics = pipeline.train(dataset, alpha)
    assert metrics.timesteps == 256
    assert metrics.effective_n > 0
    assert metrics.effective_n <= len(dataset.markets)
    assert metrics.is_synthetic is True


def test_the_gate_refuses_a_policy_trained_on_fabricated_data(settings, dataset):
    pipeline = RLPolicyPipeline(settings, total_timesteps=256)
    pipeline.train(dataset, _alpha_over_time(settings, dataset))
    passed, failures = pipeline.evaluate_gate()
    assert passed is False
    assert any("Fabricated" in f for f in failures)


def test_the_gate_counts_markets_not_timesteps(settings, dataset):
    pipeline = RLPolicyPipeline(settings, total_timesteps=256)
    pipeline.train(dataset, _alpha_over_time(settings, dataset))
    pipeline.update_policy_metrics(is_synthetic=False, effective_n=3, total_return=99.0)
    passed, failures = pipeline.evaluate_gate(min_markets=50)
    assert passed is False
    assert any("resolved markets" in f for f in failures)


# --------------------------------------------------------------------------
# LangGraph supervisor
# --------------------------------------------------------------------------


def test_the_graph_compiles_with_every_stage_present():
    nodes = set(compile_graph().nodes)
    assert {"supervisor", *_PIPELINE} <= nodes


def test_the_graph_walks_every_stage_in_order(settings, dataset, as_of):
    final = TradingGraph(dataset, settings, train_timesteps=128).run(as_of=as_of)
    stages = [line.split(":")[0] for line in final.audit]
    assert stages == list(_PIPELINE), f"unexpected route: {stages}"


def test_the_graph_produces_alpha_and_a_policy_reading(settings, dataset, as_of):
    final = TradingGraph(dataset, settings, train_timesteps=128).run(as_of=as_of)
    assert len(final.alpha) == len(dataset.markets)
    assert final.policy is not None
    assert final.as_of == as_of


def test_ingest_never_hands_a_node_data_from_the_future(dataset, as_of):
    for state in build_states(dataset, as_of):
        assert state.as_of == as_of
        if state.quote is not None:
            assert state.quote.ts <= as_of
        assert all(t.ts <= as_of for t in state.recent_trades)
        # The masked market must not leak how it resolved.
        assert state.market.outcome is None or as_of >= state.market.close_time


def test_the_graph_stops_after_its_configured_cycle_count(settings, dataset, as_of):
    final = TradingGraph(dataset, settings, train_timesteps=128).run(
        as_of=as_of, max_cycles=1
    )
    assert final.cycles == 1
    assert final.route == "done"


# --------------------------------------------------------------------------
# Safety posture
# --------------------------------------------------------------------------


def test_the_stubbed_broker_clients_refuse_to_be_used():
    import alpaca_trade_api
    from compat import RefusedExternalClient

    with pytest.raises(RefusedExternalClient):
        alpaca_trade_api.REST("key", "secret")


def test_no_graph_node_exposes_an_order_placement_method(settings, dataset):
    graph = TradingGraph(dataset, settings, train_timesteps=128)
    forbidden = {"place_order", "submit_order", "create_order", "cancel_order", "post"}
    for name in (*_PIPELINE, "supervisor"):
        node = getattr(graph, name)
        assert not (forbidden & set(dir(node)))
    assert not (forbidden & set(dir(graph)))
