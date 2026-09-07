"""Model registry: immutable versions, and a promotion no code can grant itself.

The safety-critical claim in this file is that nothing automatic can move a
model into ``APPROVED_FOR_PAPER_EVALUATION``. It is attacked statically (no
call site sets it) and behaviourally (the registry refuses without an explicit
human decision record).
"""

from __future__ import annotations

import ast
import json
import pathlib

import pytest

from research.registry import (
    PromotionState,
    ResearchRegistry,
    evaluate_promotion_gate,
)


@pytest.fixture
def registry(tmp_path):
    return ResearchRegistry(tmp_path / "registry.duckdb")


def record(registry, **overrides) -> str:
    """Record a plausible evaluated run and return its id."""
    payload = {
        "kind": "walk_forward",
        "model_name": "logistic_regression",
        "model_version": "logistic_regression/1.0",
        "data_version": "data-abc123",
        "feature_version": "features-v1",
        "label_policy": "labels-v1",
        "corpus_label": "TEST CORPUS",
        "is_synthetic": True,
        "effective_n": 80,
        "n_forecasts": 1200,
        "config": {"n_folds": 3, "seed": 7},
        "metrics": {"brier": 0.21, "log_loss": 0.60},
    }
    payload.update(overrides)
    return registry.record_run(**payload)


# -- versioning and immutability -------------------------------------------


def test_a_recorded_run_can_be_read_back(registry):
    run_id = record(registry)
    stored = registry.run(run_id)
    assert stored is not None
    assert stored["model_name"] == "logistic_regression"
    assert stored["data_version"] == "data-abc123"
    assert json.loads(stored["metrics_json"])["brier"] == pytest.approx(0.21)


def test_every_run_gets_a_distinct_id(registry):
    ids = {record(registry) for _ in range(5)}
    assert len(ids) == 5


def test_recording_again_creates_a_new_version_rather_than_overwriting(registry):
    """A retrained model must never replace the one before it."""
    first = record(registry, metrics={"brier": 0.30})
    second = record(registry, metrics={"brier": 0.20})
    assert first != second
    assert json.loads(registry.run(first)["metrics_json"])["brier"] == pytest.approx(0.30)
    assert json.loads(registry.run(second)["metrics_json"])["brier"] == pytest.approx(0.20)
    assert len(registry.runs()) == 2


def test_an_earlier_run_survives_a_later_one_being_promoted(registry):
    baseline = record(registry)
    candidate = record(registry)
    gate = evaluate_promotion_gate(
        effective_n=200, beats_baseline=True, is_synthetic=False, fold_count=3
    )
    registry.request_promotion(candidate, gate, "tester", "reviewed the folds")
    assert registry.run(baseline) is not None
    assert registry.state_of(baseline) is PromotionState.CANDIDATE


def test_runs_persist_across_reopening_the_registry(tmp_path):
    path = tmp_path / "registry.duckdb"
    run_id = record(ResearchRegistry(path))
    assert ResearchRegistry(path).run(run_id) is not None


def test_an_unknown_run_id_returns_nothing(registry):
    assert registry.run("run-does-not-exist") is None


# -- the promotion gate ----------------------------------------------------


def test_a_fresh_run_starts_as_a_candidate(registry):
    assert registry.state_of(record(registry)) is PromotionState.CANDIDATE


def test_the_gate_blocks_on_too_few_independent_outcomes():
    gate = evaluate_promotion_gate(
        effective_n=8, beats_baseline=True, is_synthetic=False, fold_count=3
    )
    assert gate.passed is False
    assert any(not ok for _, ok, _ in gate.checks)


def test_the_gate_blocks_a_model_that_does_not_beat_the_baseline():
    gate = evaluate_promotion_gate(
        effective_n=500, beats_baseline=False, is_synthetic=False, fold_count=3
    )
    assert gate.passed is False


def test_the_gate_blocks_on_synthetic_data():
    """Nothing measured on fabricated data may be approved, however good."""
    gate = evaluate_promotion_gate(
        effective_n=5000, beats_baseline=True, is_synthetic=True, fold_count=10
    )
    assert gate.passed is False
    assert any("fabricated" in reason.lower() for _, _, reason in gate.checks)
    assert any(label == "Real market data" for label, ok, _ in gate.checks if not ok)


def test_the_gate_blocks_on_too_few_folds():
    gate = evaluate_promotion_gate(
        effective_n=500, beats_baseline=True, is_synthetic=False, fold_count=1
    )
    assert gate.passed is False


def test_the_gate_passes_only_when_every_check_passes():
    gate = evaluate_promotion_gate(
        effective_n=500, beats_baseline=True, is_synthetic=False, fold_count=3
    )
    assert gate.passed is True
    assert all(ok for _, ok, _ in gate.checks)


def test_an_unknown_baseline_comparison_is_treated_as_failure():
    """``None`` means "we could not tell", which must not pass."""
    gate = evaluate_promotion_gate(
        effective_n=500, beats_baseline=None, is_synthetic=False, fold_count=3
    )
    assert gate.passed is False


# -- promotion requires an explicit human decision -------------------------


def test_promotion_records_who_decided_and_why(registry):
    run_id = record(registry)
    gate = evaluate_promotion_gate(
        effective_n=500, beats_baseline=True, is_synthetic=False, fold_count=3
    )
    registry.request_promotion(run_id, gate, "eli", "reviewed folds and calibration")
    assert registry.state_of(run_id) is PromotionState.APPROVED_FOR_PAPER
    promotion = registry.promotions().iloc[-1]
    assert promotion["approved_by"] == "eli"
    assert "reviewed" in promotion["rationale"]


def test_promotion_is_refused_when_the_gate_failed(registry):
    run_id = record(registry)
    failed = evaluate_promotion_gate(
        effective_n=5, beats_baseline=False, is_synthetic=True, fold_count=1
    )
    with pytest.raises(ValueError):
        registry.request_promotion(run_id, failed, "eli", "I really want this")
    assert registry.state_of(run_id) is PromotionState.CANDIDATE


def test_promotion_is_refused_without_a_rationale(registry):
    run_id = record(registry)
    gate = evaluate_promotion_gate(
        effective_n=500, beats_baseline=True, is_synthetic=False, fold_count=3
    )
    with pytest.raises(ValueError):
        registry.request_promotion(run_id, gate, "eli", "   ")


def test_promotion_is_refused_without_an_approver(registry):
    run_id = record(registry)
    gate = evaluate_promotion_gate(
        effective_n=500, beats_baseline=True, is_synthetic=False, fold_count=3
    )
    with pytest.raises(ValueError):
        registry.request_promotion(run_id, gate, "", "a sufficiently long reason")


def test_a_rejection_is_recorded_with_its_reason(registry):
    run_id = record(registry)
    registry.record_rejection(run_id, "eli", "calibration is worse than the market")
    assert registry.state_of(run_id) is PromotionState.REJECTED


def test_approved_runs_are_listed_separately(registry):
    approved = record(registry)
    record(registry)
    gate = evaluate_promotion_gate(
        effective_n=500, beats_baseline=True, is_synthetic=False, fold_count=3
    )
    registry.request_promotion(approved, gate, "eli", "reviewed out-of-sample folds")
    approved_frame = registry.approved_runs()
    assert list(approved_frame["run_id"]) == [approved]


# -- nothing can promote itself --------------------------------------------


def test_no_module_outside_the_registry_sets_the_approved_state():
    """Static scan: only the registry may name the approved state at all."""
    root = pathlib.Path(__file__).resolve().parent.parent
    offenders: list[str] = []
    for path in list((root / "research").rglob("*.py")) + list((root / "app").rglob("*.py")):
        if path.name == "registry.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            # Attribute access such as PromotionState.APPROVED_FOR_PAPER_EVALUATION
            if isinstance(node, ast.Attribute) and "APPROVED" in node.attr:
                offenders.append(f"{path.name}:{node.lineno} sets {node.attr}")
            if isinstance(node, ast.Constant) and isinstance(node.value, str) \
                    and "approved_for_paper" in node.value.lower():
                offenders.append(f"{path.name}:{node.lineno} uses the literal state string")
    assert not offenders, "auto-promotion risk: " + "; ".join(offenders)


def test_the_static_scan_would_catch_an_offender():
    """Confidence check: a vacuous scanner would make the test above worthless."""
    tree = ast.parse("state = PromotionState.APPROVED_FOR_PAPER_EVALUATION")
    hits = [n.attr for n in ast.walk(tree)
            if isinstance(n, ast.Attribute) and "APPROVED" in n.attr]
    assert hits == ["APPROVED_FOR_PAPER_EVALUATION"]


def test_running_the_whole_pipeline_promotes_nothing(tmp_path):
    """End to end: train and evaluate, then confirm nothing self-approved."""
    from core.config import load_settings
    from research.corpus import build_research_corpus
    from research.pipeline import prepare_data, train_and_evaluate

    settings = load_settings(use_env=False)
    reg = ResearchRegistry(tmp_path / "reg.duckdb")
    corpus = build_research_corpus({"n_markets": 40, "n_steps": 120})
    prepared = prepare_data(corpus, settings, reg)
    train_and_evaluate(prepared, settings, "logistic_regression", reg)

    assert not reg.runs().empty, "expected the pipeline to record runs"
    assert reg.approved_runs().empty
    assert all(reg.state_of(rid) is PromotionState.CANDIDATE
               for rid in reg.runs()["run_id"])
    assert reg.promotions().empty


def test_promotion_does_not_enable_any_trading(registry):
    """Approval is a label. It starts nothing and unlocks no venue path."""
    from core.config import load_settings

    run_id = record(registry, is_synthetic=False)
    gate = evaluate_promotion_gate(
        effective_n=500, beats_baseline=True, is_synthetic=False, fold_count=3
    )
    registry.request_promotion(run_id, gate, "eli", "reviewed out-of-sample folds")
    assert load_settings(use_env=False)["app"]["live_trading_enabled"] is False
    assert not hasattr(registry, "deploy")
    assert not hasattr(registry, "activate")
