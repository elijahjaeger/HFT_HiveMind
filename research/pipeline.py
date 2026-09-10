"""The orchestrator behind the dashboard buttons.

Four stages, each independently runnable and each recorded in the registry:

1. **Prepare data** - build causal features and settlement labels from a corpus.
2. **Train and evaluate** - run the walk-forward protocol and score against
   frozen baselines.
3. **Backtest** - replay each fold's out-of-sample forecasts through the
   existing event-driven simulator, with the threshold policy in front and the
   risk agent behind it.
4. **Check drift** - compare recent data against the training reference window.

Promotion is not a stage. It is a separate, explicit human decision.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from backtest.engine import run_backtest
from core.dataset import MarketDataset
from research.corpus import CorpusInfo, describe
from research.datasets import (
    FeatureDataset,
    ReadinessCheck,
    build_feature_dataset,
    check_readiness,
    dataset_fingerprint,
)
from research.drift import DriftReport, build_drift_report
from research.evaluate import (
    Comparison,
    ScoreCard,
    calibration_comparison,
    calibration_rows,
    fold_rows,
    reliability_points,
    score_market_baseline,
    score_report,
)
from research.models import DEFAULT_BASELINES
from research.policy import (
    ModelForecastAgent,
    PolicyConfig,
    freshness_window,
    strategy_config_for,
)
from research.registry import PromotionGate, ResearchRegistry, evaluate_promotion_gate
from research.skips import SkippedDecision, skip_summary
from research.walkforward import WalkForwardReport, run_walk_forward

PIPELINE_VERSION = "pipeline-v1"


@dataclass
class PreparedData:
    """Stage 1 output."""

    dataset: FeatureDataset
    corpus: CorpusInfo
    readiness: ReadinessCheck
    fingerprint: str
    run_id: str = ""


@dataclass
class TrainingOutcome:
    """Stage 2 output for one candidate."""

    report: WalkForwardReport
    candidate: ScoreCard
    baseline: ScoreCard
    comparison: Comparison
    baseline_reports: dict[str, WalkForwardReport] = field(default_factory=dict)
    run_id: str = ""

    @property
    def gate(self) -> PromotionGate:
        """The promotion gate as the evidence currently stands."""
        return evaluate_promotion_gate(
            effective_n=self.candidate.effective_n,
            beats_baseline=self.comparison.beats_baseline(),
            is_synthetic=self.is_synthetic,
            fold_count=len(self.report.usable_folds),
        )

    is_synthetic: bool = True


@dataclass
class BacktestOutcome:
    """Stage 3 output."""

    metrics: dict[str, Any] = field(default_factory=dict)
    baseline_metrics: dict[str, Any] = field(default_factory=dict)
    skips: list[SkippedDecision] = field(default_factory=list)
    audit_rows: list[dict[str, Any]] = field(default_factory=list)
    equity: list[dict[str, Any]] = field(default_factory=list)
    n_forecasts: int = 0
    run_id: str = ""
    warnings: list[str] = field(default_factory=list)


def prepare_data(
    corpus: MarketDataset,
    settings: dict[str, Any],
    registry: ResearchRegistry | None = None,
    cache: Any = None,
) -> PreparedData:
    """Stage 1: build causal features and settlement labels.

    Args:
        corpus: The source market data.
        settings: Full settings dictionary.
        registry: Where to record the run, if recording is wanted.
        cache: Optional feature cache, so a growing corpus does not rebuild
            rows it has already built.

    Returns:
        A :class:`PreparedData` carrying the dataset and its readiness verdict.
    """
    research_cfg = settings.get("research", {})
    dataset = build_feature_dataset(
        corpus, research_cfg.get("dataset", {}), cache=cache
    )
    info = describe(corpus)
    readiness = check_readiness(dataset, research_cfg.get("readiness", {}))
    fingerprint = dataset_fingerprint(dataset)

    run_id = ""
    if registry is not None:
        run_id = registry.record_run(
            kind="prepare_data",
            model_name="-",
            model_version="-",
            data_version=dataset.data_version,
            feature_version=dataset.feature_version,
            label_policy=dataset.label_policy_version,
            corpus_label=info.label,
            is_synthetic=info.is_synthetic,
            effective_n=dataset.effective_n,
            n_forecasts=len(dataset.labelled),
            config=dataset.config,
            metrics=dataset.summary(),
            notes=f"fingerprint={fingerprint}",
        )
    return PreparedData(
        dataset=dataset,
        corpus=info,
        readiness=readiness,
        fingerprint=fingerprint,
        run_id=run_id,
    )


def train_and_evaluate(
    prepared: PreparedData,
    settings: dict[str, Any],
    model_name: str = "logistic_regression",
    registry: ResearchRegistry | None = None,
) -> TrainingOutcome:
    """Stage 2: walk-forward training, calibration and scoring vs baselines.

    Args:
        prepared: Stage 1 output.
        settings: Full settings dictionary.
        model_name: Candidate model to train.
        registry: Where to record the run and artifacts.

    Returns:
        A :class:`TrainingOutcome`.
    """
    research_cfg = settings.get("research", {})
    wf_cfg = research_cfg.get("walkforward", {})
    model_cfg = research_cfg.get("models", {}).get(model_name, {})

    report = run_walk_forward(prepared.dataset, model_name, wf_cfg, model_cfg)
    candidate = score_report(report)
    baseline = score_market_baseline(report)

    baseline_reports: dict[str, WalkForwardReport] = {}
    for name in DEFAULT_BASELINES:
        if name == "market_price_baseline":
            continue
        baseline_reports[name] = run_walk_forward(prepared.dataset, name, wf_cfg)

    comparison = Comparison(
        candidate=candidate,
        baselines=[baseline] + [score_report(r) for r in baseline_reports.values()],
    )
    outcome = TrainingOutcome(
        report=report,
        candidate=candidate,
        baseline=baseline,
        comparison=comparison,
        baseline_reports=baseline_reports,
        is_synthetic=prepared.corpus.is_synthetic,
    )

    if registry is not None:
        outcome.run_id = registry.record_run(
            kind="train_evaluate",
            model_name=model_name,
            model_version=(
                report.usable_folds[0].pipeline.model_version
                if report.usable_folds and report.usable_folds[0].pipeline
                else model_name
            ),
            data_version=prepared.dataset.data_version,
            feature_version=prepared.dataset.feature_version,
            label_policy=prepared.dataset.label_policy_version,
            corpus_label=prepared.corpus.label,
            is_synthetic=prepared.corpus.is_synthetic,
            effective_n=candidate.effective_n,
            n_forecasts=candidate.n_forecasts,
            config={"walkforward": wf_cfg, "model": model_cfg},
            metrics={
                "candidate": candidate.as_row(),
                "baseline": baseline.as_row(),
                "verdict": comparison.verdict(),
            },
            notes=f"fingerprint={prepared.fingerprint}",
        )
        registry.save_artifact(
            outcome.run_id, "folds", fold_rows(report), "per-fold out-of-sample scores"
        )
        probabilities, outcomes, _ = report.pooled()
        registry.save_artifact(
            outcome.run_id, "calibration",
            {
                "table": calibration_rows(probabilities, outcomes),
                "reliability": reliability_points(probabilities, outcomes),
                "comparison": calibration_comparison(report),
            },
            "calibration table and reliability diagram",
        )
        for fold in report.usable_folds:
            if fold.pipeline is not None and hasattr(fold.pipeline.model, "parameters"):
                registry.save_artifact(
                    outcome.run_id, f"model_fold{fold.plan.index}",
                    {
                        "parameters": fold.pipeline.model.parameters(),
                        "scaler": (
                            fold.pipeline.scaler.as_dict() if fold.pipeline.scaler else None
                        ),
                        "assumptions": fold.pipeline.assumptions,
                    },
                    f"fitted parameters for fold {fold.plan.index}",
                )
    return outcome


def run_model_backtest(
    prepared: PreparedData,
    outcome: TrainingOutcome,
    corpus: MarketDataset,
    settings: dict[str, Any],
    registry: ResearchRegistry | None = None,
) -> BacktestOutcome:
    """Stage 3: backtest the out-of-sample forecasts through the simulator.

    Only forecasts from fold test windows are used, so nothing the model was
    fitted on can be traded. The existing event-driven engine does the
    execution, which means the same fees, spread crossing, slippage, latency,
    partial fills and risk limits apply as anywhere else in the project.

    Args:
        prepared: Stage 1 output.
        outcome: Stage 2 output, whose forecasts are replayed.
        corpus: The market data to replay.
        settings: Full settings dictionary.
        registry: Where to record the run.

    Returns:
        A :class:`BacktestOutcome`.
    """
    research_cfg = settings.get("research", {})
    policy_cfg = research_cfg.get("policy", {})
    policy = PolicyConfig(
        safety_margin=float(policy_cfg.get("safety_margin", 0.02)),
        min_confidence=float(policy_cfg.get("min_confidence", 0.0)),
        max_quote_age_seconds=float(policy_cfg.get("max_quote_age_seconds", 900.0)),
        reference_size=float(policy_cfg.get("reference_size", 100.0)),
    )

    forecasts = [f for fold in outcome.report.usable_folds for f in fold.forecasts]
    result = BacktestOutcome(n_forecasts=len(forecasts))
    if not forecasts:
        result.warnings.append(
            "No out-of-sample forecasts to trade: the walk-forward produced none."
        )
        return result

    # Restrict the replay to markets that appear in a fold test window, so the
    # simulator can never trade a market the model was trained on.
    test_markets = {
        market_id
        for fold in outcome.report.usable_folds
        for market_id in fold.plan.test_markets
    }
    markets = [m for m in corpus.markets if m.market_id in test_markets]
    quotes = [q for q in corpus.quotes if q.market_id in test_markets]
    trades = [t for t in corpus.trades if t.market_id in test_markets]

    stride = float(
        research_cfg.get("dataset", {}).get("stride_seconds", 1800.0)
    )
    agent = ModelForecastAgent(forecasts, max_age_seconds=freshness_window(stride))

    run_settings = dict(settings)
    run_settings["strategy"] = {
        **settings.get("strategy", {}),
        **strategy_config_for(policy, settings.get("fills", {})),
    }
    run_settings["agents"] = {
        **settings.get("agents", {}),
        "ensemble": {"include_placeholder": False, "weights": {agent.name: 1.0}},
    }

    model_run = run_backtest(
        markets, quotes, trades, run_settings, forecast_agents=[agent]
    )
    result.metrics = dict(model_run.metrics)
    result.skips = list(getattr(model_run, "skipped", []))
    result.equity = [
        {"ts": s.ts, "equity": s.equity, "drawdown": s.drawdown}
        for s in model_run.snapshots
    ]
    result.audit_rows = [
        {"ts": e.ts, "kind": e.kind.value, "market": e.market_id, "message": e.message}
        for e in model_run.audit_log[-500:]
    ]

    # The flat baseline: the same markets, with nothing traded. Its P&L is
    # exactly zero by construction, which is the number a candidate has to beat
    # after costs before any of this is worth doing.
    result.baseline_metrics = {
        "total_pnl": 0.0,
        "total_return": 0.0,
        "max_drawdown": 0.0,
        "n_trades": 0,
        "note": "Flat baseline: no trades, no costs, no risk.",
    }

    if registry is not None:
        result.run_id = registry.record_run(
            kind="backtest",
            model_name=outcome.report.model_name,
            model_version=outcome.candidate.model_name,
            data_version=prepared.dataset.data_version,
            feature_version=prepared.dataset.feature_version,
            label_policy=prepared.dataset.label_policy_version,
            corpus_label=prepared.corpus.label,
            is_synthetic=prepared.corpus.is_synthetic,
            effective_n=outcome.candidate.effective_n,
            n_forecasts=len(forecasts),
            config={"policy": policy_cfg, "strategy": run_settings["strategy"]},
            metrics=result.metrics,
            notes="out-of-sample fold forecasts only",
        )
        registry.save_artifact(
            result.run_id, "skips", skip_summary(result.skips), "skipped-decision reasons"
        )
        registry.save_artifact(
            result.run_id, "metrics", result.metrics, "backtest metrics"
        )
    return result


def check_drift(
    prepared: PreparedData,
    outcome: TrainingOutcome,
    settings: dict[str, Any],
    registry: ResearchRegistry | None = None,
) -> DriftReport:
    """Stage 4: compare the most recent window against the training reference.

    The reference window is the first fold's training markets; the current
    window is the last fold's test markets. Nothing is retrained or promoted as
    a result - the report is a flag for a human.
    """
    report = outcome.report
    if not report.usable_folds:
        return DriftReport()

    first, last = report.usable_folds[0], report.usable_folds[-1]
    names = list(prepared.dataset.feature_names)
    reference_markets = set(first.plan.train_markets)
    current_markets = set(last.plan.test_markets)

    reference_rows, reference_missing = [], []
    current_rows, current_missing = [], []
    for example in prepared.dataset.labelled:
        if example.market_id in reference_markets:
            reference_rows.append(example.features.vector(names))
            reference_missing.append(len(example.features.missing))
        elif example.market_id in current_markets:
            current_rows.append(example.features.vector(names))
            current_missing.append(len(example.features.missing))

    reference_probabilities: list[float] = []
    reference_outcomes: list[float] = []
    if first.pipeline is not None and reference_rows:
        raw, calibrated = first.pipeline.probabilities(reference_rows)
        reference_probabilities = calibrated or raw
        reference_outcomes = [
            float(e.label)
            for e in prepared.dataset.labelled
            if e.market_id in reference_markets and e.label is not None
        ]

    drift = build_drift_report(
        reference_rows=reference_rows,
        current_rows=current_rows,
        feature_names=names,
        reference_missing=reference_missing,
        current_missing=current_missing,
        reference_probabilities=reference_probabilities,
        current_probabilities=last.calibrated_probabilities or last.raw_probabilities,
        reference_outcomes=reference_outcomes,
        current_outcomes=last.outcomes,
        reference_markets=len(reference_markets),
        current_markets=len(current_markets),
    )

    if registry is not None:
        registry.record_run(
            kind="drift",
            model_name=report.model_name,
            model_version=report.model_name,
            data_version=prepared.dataset.data_version,
            feature_version=prepared.dataset.feature_version,
            label_policy=prepared.dataset.label_policy_version,
            corpus_label=prepared.corpus.label,
            is_synthetic=prepared.corpus.is_synthetic,
            effective_n=drift.current_markets,
            n_forecasts=drift.current_n,
            metrics={"summary": drift.summary(), "worst_level": drift.worst_level.value},
            notes="drift never triggers retraining or promotion",
        )
    return drift
