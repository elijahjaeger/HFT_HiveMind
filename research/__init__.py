"""Controlled research pipeline: features, labels, models, walk-forward.

This package adds **no** trading capability. It has no order-placement code, no
broker or account calls, and no write path to any venue. It reads market data
only through the existing read-only adapter, and no model is ever deployed,
replaced or promoted automatically - promotion is a manual, recorded human
decision that starts nothing.

See ``research/README_DESIGN.md`` for the label policy, the purge/embargo rule
and the sample-size bar that gates every claim this package can make.
"""

from research.datasets import (
    FeatureDataset,
    ReadinessCheck,
    TrainingExample,
    build_feature_dataset,
    check_readiness,
    data_version_for,
    dataset_fingerprint,
    decision_timestamps,
)
from research.features import (
    FEATURE_NAMES,
    FEATURE_SPECS,
    FEATURE_VERSION,
    FeatureSpec,
    FeatureVector,
    StandardScaler,
    build_features,
    feature_dictionary_rows,
)
from research.labels import (
    LABEL_POLICY_VERSION,
    LabelStatus,
    SettlementLabel,
    label_for_market,
    label_markets,
    label_summary,
    usable_labels,
)
from research.models import (
    Forecaster,
    LogisticForecaster,
    MarketPriceBaseline,
    PersistenceBaseline,
    ProbabilityForecast,
    build_forecaster,
)
from research.walkforward import (
    FoldPlan,
    FoldResult,
    WalkForwardReport,
    plan_folds,
    run_walk_forward,
)

__all__ = [
    "FEATURE_NAMES",
    "FEATURE_SPECS",
    "FEATURE_VERSION",
    "LABEL_POLICY_VERSION",
    "FeatureDataset",
    "FeatureSpec",
    "FeatureVector",
    "Forecaster",
    "FoldPlan",
    "FoldResult",
    "LabelStatus",
    "LogisticForecaster",
    "MarketPriceBaseline",
    "PersistenceBaseline",
    "ProbabilityForecast",
    "ReadinessCheck",
    "SettlementLabel",
    "StandardScaler",
    "TrainingExample",
    "WalkForwardReport",
    "build_feature_dataset",
    "build_features",
    "build_forecaster",
    "check_readiness",
    "data_version_for",
    "dataset_fingerprint",
    "decision_timestamps",
    "feature_dictionary_rows",
    "label_for_market",
    "label_markets",
    "label_summary",
    "plan_folds",
    "run_walk_forward",
    "usable_labels",
]
