"""Pluggable probability forecasters. Simple and interpretable only.

Three models ship here:

* :class:`MarketPriceBaseline` - the market's own implied probability. This is
  the **frozen baseline** every candidate is measured against, and it is a hard
  bar: prediction-market prices are already an aggregate of everyone's
  information and their own money.
* :class:`PersistenceBaseline` - the last observed probability, carried
  forward. A sanity check that a model is doing something a lag cannot.
* :class:`LogisticForecaster` - L2-regularised logistic regression fitted by
  Newton/IRLS.

**No tree model is included**, though the task permits one. The condition it
attaches - "only if enough data exists and its addition is well justified" - is
not met at the data volumes available here. A shallow tree needs enough
*independent outcomes* to place its splits, and the statistical unit is the
resolved market, not the feature row. :data:`TREE_MODEL_BAR` records the bar a
tree would have to clear before it would be justified.

No neural networks and no automated hyperparameter search *in this module*: the
fitted parameters stay plain numbers so the registry can store, diff and display
them.

**Reinforcement learning is no longer excluded from the project** - it moved,
rather than being admitted here. A policy emits actions, not calibrated
probabilities, so it cannot be scored by Brier or log loss and does not belong
behind :func:`build_forecaster`. It lives in :mod:`src.rl_policy`, downstream of
the alpha vectors this module's forecasters help produce. The sample-size
argument that used to justify the ban survives as a warning rather than a
prohibition: :data:`TREE_MODEL_BAR` still states the bar, and the RL node
reports ``effective_n`` in resolved markets for the same reason. See
``DECISIONS.md`` (2026-09-07).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol, runtime_checkable

from research.features import FEATURE_NAMES, FEATURE_VERSION, StandardScaler

MODEL_SCOPE_VERSION = "models-v1"

TREE_MODEL_BAR = (
    "A shallow tree would need at least 200 resolved markets with both outcome "
    "classes well represented, and would have to beat regularised logistic "
    "regression out-of-sample on the validation folds by more than the spread "
    "of fold-to-fold variation. Below that, a tree places splits on noise from "
    "a handful of independent outcomes and its apparent skill is an artifact."
)


@dataclass(frozen=True)
class ProbabilityForecast:
    """One model's forecast, carrying everything needed to audit it later."""

    market_ticker: str
    decision_timestamp: datetime
    raw_probability_yes: float
    calibrated_probability_yes: float | None
    confidence: float
    feature_version: str
    model_version: str
    data_version: str
    assumptions: dict[str, Any] = field(default_factory=dict)

    @property
    def probability(self) -> float:
        """The probability to act on: calibrated when available, else raw."""
        return (
            self.calibrated_probability_yes
            if self.calibrated_probability_yes is not None
            else self.raw_probability_yes
        )

    def as_row(self) -> dict[str, Any]:
        """Display row for the dashboard."""
        return {
            "market": self.market_ticker,
            "decision_ts": self.decision_timestamp,
            "raw": round(self.raw_probability_yes, 4),
            "calibrated": (
                round(self.calibrated_probability_yes, 4)
                if self.calibrated_probability_yes is not None
                else None
            ),
            "confidence": round(self.confidence, 3),
            "model": self.model_version,
            "features": self.feature_version,
            "data": self.data_version,
        }


@runtime_checkable
class Forecaster(Protocol):
    """The pluggable model interface.

    Implementations must be deterministic: the same training data and
    configuration must produce the same fitted parameters, so a stored run can
    be reproduced exactly.
    """

    name: str
    version: str

    def fit(self, rows: Sequence[Sequence[float]], labels: Sequence[float]) -> Forecaster:
        """Fit on training rows only and return self."""
        ...

    def predict_proba(self, rows: Sequence[Sequence[float]]) -> list[float]:
        """Return P(YES) for each row."""
        ...


def _clip(value: float, low: float = 1e-6, high: float = 1.0 - 1e-6) -> float:
    """Keep a probability strictly inside ``(0, 1)``."""
    return max(low, min(high, value))


def _sigmoid(value: float) -> float:
    """Numerically stable logistic function."""
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-min(value, 60.0)))
    exp_value = math.exp(max(value, -60.0))
    return exp_value / (1.0 + exp_value)


class MarketPriceBaseline:
    """The frozen baseline: the market's own implied probability.

    Reads the ``market_prob`` feature straight through. It is not fitted, so it
    cannot overfit, and beating it out-of-sample after costs is the entire
    research question.
    """

    name = "market_price_baseline"
    version = f"{name}/1.0"
    is_baseline = True

    def __init__(self, feature_names: Sequence[str] = FEATURE_NAMES) -> None:
        self.feature_names = list(feature_names)
        self.index = self.feature_names.index("market_prob")
        self.is_fitted = True

    def fit(
        self, rows: Sequence[Sequence[float]], labels: Sequence[float]
    ) -> MarketPriceBaseline:
        """No-op: this baseline has nothing to learn."""
        return self

    def predict_proba(self, rows: Sequence[Sequence[float]]) -> list[float]:
        """Return the market-implied probability for each row."""
        return [_clip(row[self.index]) for row in rows]

    def parameters(self) -> dict[str, Any]:
        """Nothing is fitted; recorded for completeness."""
        return {"reads_feature": "market_prob", "fitted": False}


class PersistenceBaseline:
    """Carries the last observed probability forward, with a shrink to 0.5.

    Distinct from the market baseline only when ``shrink`` is non-zero, which
    is what makes it a check on whether a model is adding anything beyond a lag.
    """

    name = "persistence_baseline"
    version = f"{name}/1.0"
    is_baseline = True

    def __init__(
        self, feature_names: Sequence[str] = FEATURE_NAMES, shrink: float = 0.0
    ) -> None:
        self.feature_names = list(feature_names)
        self.index = self.feature_names.index("market_prob")
        self.return_index = self.feature_names.index("return_30m")
        self.shrink = float(shrink)
        self.is_fitted = True

    def fit(
        self, rows: Sequence[Sequence[float]], labels: Sequence[float]
    ) -> PersistenceBaseline:
        """No-op: this baseline has nothing to learn."""
        return self

    def predict_proba(self, rows: Sequence[Sequence[float]]) -> list[float]:
        """Undo the most recent move, then shrink toward 0.5."""
        out: list[float] = []
        for row in rows:
            lagged = row[self.index] - row[self.return_index]
            out.append(_clip((1.0 - self.shrink) * lagged + self.shrink * 0.5))
        return out

    def parameters(self) -> dict[str, Any]:
        """Fitted parameters (only the configured shrink)."""
        return {"shrink": self.shrink, "fitted": False}


class LogisticForecaster:
    """L2-regularised logistic regression, fitted by Newton/IRLS.

    The regularisation is not optional. With few independent outcomes an
    unpenalised fit separates the classes and drives its weights to infinity,
    producing confident nonsense. The penalty is a fixed configuration value,
    deliberately **not** searched over: an automated hyperparameter search on
    this much data would be fitting the validation set.
    """

    name = "logistic_regression"
    version = f"{name}/1.0"
    is_baseline = False

    def __init__(
        self,
        l2: float = 1.0,
        max_iterations: int = 50,
        tolerance: float = 1e-8,
        feature_names: Sequence[str] = FEATURE_NAMES,
    ) -> None:
        self.l2 = float(l2)
        self.max_iterations = int(max_iterations)
        self.tolerance = float(tolerance)
        self.feature_names = list(feature_names)
        self.weights: list[float] = []
        self.intercept = 0.0
        self.iterations = 0
        self.converged = False
        self.is_fitted = False

    def fit(
        self, rows: Sequence[Sequence[float]], labels: Sequence[float]
    ) -> LogisticForecaster:
        """Fit weights on training rows only.

        Args:
            rows: Scaled feature rows.
            labels: Outcomes as ``1.0``/``0.0``.

        Returns:
            Self.

        Raises:
            ValueError: If the inputs are empty or mismatched.
        """
        if not rows:
            raise ValueError("logistic regression needs at least one training row")
        if len(rows) != len(labels):
            raise ValueError("rows and labels must be the same length")

        n_features = len(rows[0])
        beta = [0.0] * (n_features + 1)  # index 0 is the intercept
        design = [[1.0, *row] for row in rows]

        for iteration in range(self.max_iterations):
            gradient = [0.0] * (n_features + 1)
            hessian = [[0.0] * (n_features + 1) for _ in range(n_features + 1)]

            for x, y in zip(design, labels, strict=True):
                p = _sigmoid(sum(b * v for b, v in zip(beta, x, strict=True)))
                w = max(p * (1.0 - p), 1e-9)
                residual = y - p
                for i in range(n_features + 1):
                    gradient[i] += residual * x[i]
                    for j in range(i, n_features + 1):
                        hessian[i][j] += w * x[i] * x[j]

            # L2 penalty on the weights, never on the intercept: penalising it
            # would bias the model away from the observed base rate.
            for i in range(1, n_features + 1):
                gradient[i] -= self.l2 * beta[i]
                hessian[i][i] += self.l2
            for i in range(n_features + 1):
                for j in range(i):
                    hessian[i][j] = hessian[j][i]

            step = _solve(hessian, gradient)
            if step is None:
                break
            beta = [b + s for b, s in zip(beta, step, strict=True)]
            self.iterations = iteration + 1
            if max(abs(s) for s in step) < self.tolerance:
                self.converged = True
                break

        self.intercept = beta[0]
        self.weights = beta[1:]
        self.is_fitted = True
        return self

    def predict_proba(self, rows: Sequence[Sequence[float]]) -> list[float]:
        """Return P(YES) for each row.

        Raises:
            RuntimeError: If called before :meth:`fit`.
        """
        if not self.is_fitted:
            raise RuntimeError("model must be fitted before predicting")
        return [
            _clip(
                _sigmoid(
                    self.intercept
                    + sum(w * v for w, v in zip(self.weights, row, strict=True))
                )
            )
            for row in rows
        ]

    def parameters(self) -> dict[str, Any]:
        """Fitted coefficients, keyed by feature name."""
        return {
            "intercept": self.intercept,
            "weights": dict(zip(self.feature_names, self.weights, strict=False)),
            "l2": self.l2,
            "iterations": self.iterations,
            "converged": self.converged,
        }

    def coefficient_rows(self) -> list[dict[str, Any]]:
        """Coefficients sorted by absolute size, for display."""
        rows = [
            {"feature": name, "weight": weight, "abs": abs(weight)}
            for name, weight in zip(self.feature_names, self.weights, strict=False)
        ]
        rows.sort(key=lambda r: r["abs"], reverse=True)
        for row in rows:
            row.pop("abs")
        return rows


def _solve(matrix: list[list[float]], vector: list[float]) -> list[float] | None:
    """Solve ``matrix @ x = vector`` by Gaussian elimination with pivoting.

    Returns ``None`` when the system is singular, which tells the caller to
    stop iterating rather than take a nonsense step.
    """
    n = len(vector)
    augmented = [[*row, v] for row, v in zip(matrix, vector, strict=True)]

    for column in range(n):
        pivot = max(range(column, n), key=lambda r: abs(augmented[r][column]))
        if abs(augmented[pivot][column]) < 1e-12:
            return None
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        pivot_value = augmented[column][column]
        for row in range(column + 1, n):
            factor = augmented[row][column] / pivot_value
            if factor == 0.0:
                continue
            for col in range(column, n + 1):
                augmented[row][col] -= factor * augmented[column][col]

    solution = [0.0] * n
    for row in range(n - 1, -1, -1):
        total = augmented[row][n] - sum(
            augmented[row][col] * solution[col] for col in range(row + 1, n)
        )
        solution[row] = total / augmented[row][row]
    return solution


@dataclass
class FittedPipeline:
    """A frozen scaler + model + calibrator, ready to forecast.

    Freezing matters: once a fold's pipeline is built it must not see another
    row of training data, or the out-of-sample evaluation stops being out of
    sample.
    """

    model: Any
    scaler: StandardScaler | None
    calibrator: Any | None
    feature_names: tuple[str, ...] = FEATURE_NAMES
    feature_version: str = FEATURE_VERSION
    data_version: str = ""
    calibration_method: str = "none"
    assumptions: dict[str, Any] = field(default_factory=dict)

    @property
    def model_version(self) -> str:
        """Version string of the wrapped model."""
        return getattr(self.model, "version", getattr(self.model, "name", "unknown"))

    def raw_probabilities(self, rows: Sequence[Sequence[float]]) -> list[float]:
        """Model output before calibration."""
        scaled = self.scaler.transform(rows) if self.scaler is not None else list(rows)
        return self.model.predict_proba(scaled)

    def probabilities(
        self, rows: Sequence[Sequence[float]]
    ) -> tuple[list[float], list[float] | None]:
        """Return ``(raw, calibrated)``; calibrated is ``None`` if uncalibrated."""
        raw = self.raw_probabilities(rows)
        if self.calibrator is None:
            return raw, None
        return raw, self.calibrator.transform(raw)

    def forecast(
        self,
        market_ticker: str,
        decision_timestamp: datetime,
        row: Sequence[float],
        confidence: float = 0.0,
    ) -> ProbabilityForecast:
        """Produce a single fully-attributed :class:`ProbabilityForecast`."""
        raw, calibrated = self.probabilities([row])
        return ProbabilityForecast(
            market_ticker=market_ticker,
            decision_timestamp=decision_timestamp,
            raw_probability_yes=raw[0],
            calibrated_probability_yes=calibrated[0] if calibrated else None,
            confidence=confidence,
            feature_version=self.feature_version,
            model_version=self.model_version,
            data_version=self.data_version,
            assumptions=dict(self.assumptions),
        )


def build_forecaster(name: str, config: dict[str, Any] | None = None) -> Any:
    """Instantiate a forecaster by name.

    Args:
        name: One of ``market_price_baseline``, ``persistence_baseline``,
            ``logistic_regression``.
        config: Model-specific settings.

    Returns:
        An unfitted forecaster.

    Raises:
        ValueError: If the name is unknown, or names a model class this task
            explicitly excludes.
    """
    cfg = dict(config or {})
    if name in {"tree", "decision_tree", "random_forest", "gradient_boosting"}:
        raise ValueError(f"{name!r} is not enabled. {TREE_MODEL_BAR}")
    if name in {"rl", "mappo", "ppo"}:
        raise ValueError(
            f"{name!r} is a control policy, not a probability forecaster: it emits "
            "actions, which this pipeline cannot calibrate or score with Brier / log "
            "loss. Reinforcement learning now lives in src/rl_policy.py, which "
            "consumes the alpha vectors this pipeline produces. See DECISIONS.md "
            "(2026-09-07)."
        )
    if name in {"mlp", "lstm", "neural_network"}:
        raise ValueError(
            f"{name!r} is out of scope for the forecaster slot: this pipeline is "
            "restricted to models whose fitted parameters the registry can store, "
            "diff and display as plain numbers."
        )
    if name == "market_price_baseline":
        return MarketPriceBaseline()
    if name == "persistence_baseline":
        return PersistenceBaseline(shrink=float(cfg.get("shrink", 0.0)))
    if name == "logistic_regression":
        return LogisticForecaster(
            l2=float(cfg.get("l2", 1.0)),
            max_iterations=int(cfg.get("max_iterations", 50)),
        )
    raise ValueError(f"unknown forecaster: {name!r}")


DEFAULT_CANDIDATES: tuple[str, ...] = ("logistic_regression",)
DEFAULT_BASELINES: tuple[str, ...] = ("market_price_baseline", "persistence_baseline")
