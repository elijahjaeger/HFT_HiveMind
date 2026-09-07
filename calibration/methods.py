"""Calibration methods: identity, isotonic regression and logistic (Platt).

Both real methods are implemented directly rather than pulled from scikit-learn.
They are short, exact and deterministic, and the project would otherwise gain a
large dependency tree for two textbook fits:

* **Isotonic regression** via the Pool Adjacent Violators Algorithm, which finds
  the exact least-squares monotonic fit. Non-parametric and flexible, but it can
  overfit a small training fold and only ever produces values seen in training.
* **Logistic (Platt) scaling** fits ``sigmoid(a * logit(p) + b)`` by Newton's
  method. Two parameters only, so it is stable on little data, but it can only
  apply a smooth monotone stretch - it cannot fix a non-monotone miscalibration.

Every calibrator is fitted on the training fold alone. ``transform`` is a pure
function of the fitted parameters.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

EPS = 1e-6
"""Probabilities are clipped into ``[EPS, 1 - EPS]`` so logits stay finite."""


def _clip(value: float, low: float = EPS, high: float = 1.0 - EPS) -> float:
    """Clamp a probability away from the open ends of the unit interval."""
    return max(low, min(high, value))


def _logit(probability: float) -> float:
    """Log-odds of a probability, with clipping for numerical safety."""
    clipped = _clip(probability)
    return math.log(clipped / (1.0 - clipped))


def _sigmoid(value: float) -> float:
    """Numerically stable logistic function."""
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def _validate(probabilities: Sequence[float], outcomes: Sequence[float]) -> None:
    if len(probabilities) != len(outcomes):
        raise ValueError("probabilities and outcomes must have the same length")
    for probability in probabilities:
        if not 0.0 <= probability <= 1.0:
            raise ValueError(f"probability out of range: {probability}")
    for outcome in outcomes:
        if outcome not in (0.0, 1.0):
            raise ValueError(f"outcome must be 0.0 or 1.0, got {outcome}")


@runtime_checkable
class Calibrator(Protocol):
    """Maps raw forecast probabilities onto calibrated ones."""

    name: str

    def fit(self, probabilities: Sequence[float], outcomes: Sequence[float]) -> Calibrator:
        """Fit on the training fold and return self."""
        ...

    def transform(self, probabilities: Sequence[float]) -> list[float]:
        """Apply the fitted mapping."""
        ...


class IdentityCalibrator:
    """The uncalibrated baseline: returns the raw probabilities unchanged."""

    name = "uncalibrated"

    def __init__(self) -> None:
        self.is_fitted = False

    def fit(
        self, probabilities: Sequence[float], outcomes: Sequence[float]
    ) -> IdentityCalibrator:
        """Validate the inputs; there is nothing to learn.

        Args:
            probabilities: Training probabilities.
            outcomes: Training outcomes as ``1.0``/``0.0``.

        Returns:
            Self.
        """
        _validate(probabilities, outcomes)
        self.is_fitted = True
        return self

    def transform(self, probabilities: Sequence[float]) -> list[float]:
        """Return the probabilities unchanged (clipped for numerical safety)."""
        return [_clip(p) for p in probabilities]


def pool_adjacent_violators(
    values: Sequence[float], weights: Sequence[float] | None = None
) -> list[float]:
    """Least-squares fit of a non-decreasing sequence (the PAVA algorithm).

    Args:
        values: Target values in the order of their (sorted) inputs.
        weights: Optional per-value weights; defaults to 1 each.

    Returns:
        A non-decreasing sequence of the same length, minimising the weighted
        squared error against ``values``.
    """
    if not values:
        return []
    weights = list(weights) if weights is not None else [1.0] * len(values)
    if len(weights) != len(values):
        raise ValueError("values and weights must have the same length")

    # Each block holds (weighted mean, total weight, count).
    blocks: list[list[float]] = []
    for value, weight in zip(values, weights, strict=True):
        blocks.append([float(value), float(weight), 1.0])
        while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0] + 1e-15:
            mean_b, weight_b, count_b = blocks.pop()
            mean_a, weight_a, count_a = blocks.pop()
            total_weight = weight_a + weight_b
            merged = (
                (mean_a * weight_a + mean_b * weight_b) / total_weight
                if total_weight > 0
                else (mean_a + mean_b) / 2.0
            )
            blocks.append([merged, total_weight, count_a + count_b])

    fitted: list[float] = []
    for mean, _weight, count in blocks:
        fitted.extend([mean] * int(count))
    return fitted


class IsotonicCalibrator:
    """Non-parametric monotone calibration fitted by pool-adjacent-violators.

    The fitted mapping is piecewise linear between the pooled knots and is
    clamped outside the training range, so it never extrapolates.
    """

    name = "isotonic"

    def __init__(self, clip_to: tuple[float, float] = (EPS, 1.0 - EPS)) -> None:
        self.clip_low, self.clip_high = clip_to
        self.knots_x: list[float] = []
        self.knots_y: list[float] = []
        self.is_fitted = False

    def fit(
        self, probabilities: Sequence[float], outcomes: Sequence[float]
    ) -> IsotonicCalibrator:
        """Fit a monotone step function mapping probabilities to frequencies.

        Args:
            probabilities: Training probabilities.
            outcomes: Training outcomes as ``1.0``/``0.0``.

        Returns:
            Self.

        Raises:
            ValueError: If the inputs are empty or mismatched.
        """
        _validate(probabilities, outcomes)
        if not probabilities:
            raise ValueError("isotonic calibration needs at least one observation")

        order = sorted(range(len(probabilities)), key=lambda i: probabilities[i])
        xs = [float(probabilities[i]) for i in order]
        ys = [float(outcomes[i]) for i in order]
        fitted = pool_adjacent_violators(ys)

        # Collapse duplicate x values, keeping the last fitted value for each.
        knots_x: list[float] = []
        knots_y: list[float] = []
        for x, y in zip(xs, fitted, strict=True):
            if knots_x and abs(x - knots_x[-1]) < 1e-12:
                knots_y[-1] = y
            else:
                knots_x.append(x)
                knots_y.append(y)

        self.knots_x = knots_x
        self.knots_y = knots_y
        self.is_fitted = True
        return self

    def transform(self, probabilities: Sequence[float]) -> list[float]:
        """Apply the fitted monotone mapping with linear interpolation.

        Raises:
            RuntimeError: If called before :meth:`fit`.
        """
        if not self.is_fitted:
            raise RuntimeError("IsotonicCalibrator.transform called before fit")
        return [self._interpolate(p) for p in probabilities]

    def _interpolate(self, probability: float) -> float:
        xs, ys = self.knots_x, self.knots_y
        if not xs:
            return _clip(probability)
        if probability <= xs[0]:
            return self._bound(ys[0])
        if probability >= xs[-1]:
            return self._bound(ys[-1])

        low, high = 0, len(xs) - 1
        while high - low > 1:
            middle = (low + high) // 2
            if xs[middle] <= probability:
                low = middle
            else:
                high = middle
        span = xs[high] - xs[low]
        if span <= 0:
            return self._bound(ys[high])
        weight = (probability - xs[low]) / span
        return self._bound(ys[low] + weight * (ys[high] - ys[low]))

    def _bound(self, value: float) -> float:
        return max(self.clip_low, min(self.clip_high, value))


class LogisticCalibrator:
    """Platt scaling: ``sigmoid(a * logit(p) + b)`` fitted by Newton's method.

    Platt's target smoothing is applied by default, which shrinks the targets
    away from 0 and 1 in proportion to the fold size. Without it a perfectly
    separable training fold drives the coefficients to infinity.
    """

    name = "logistic"

    def __init__(
        self,
        smoothing: bool = True,
        max_iterations: int = 100,
        tolerance: float = 1e-9,
        ridge: float = 1e-10,
    ) -> None:
        self.smoothing = smoothing
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.ridge = ridge
        self.slope = 1.0
        self.intercept = 0.0
        self.iterations = 0
        self.converged = False
        self.is_fitted = False

    def fit(
        self, probabilities: Sequence[float], outcomes: Sequence[float]
    ) -> LogisticCalibrator:
        """Fit the two coefficients on the training fold.

        Args:
            probabilities: Training probabilities.
            outcomes: Training outcomes as ``1.0``/``0.0``.

        Returns:
            Self.

        Raises:
            ValueError: If the inputs are empty or mismatched.
        """
        _validate(probabilities, outcomes)
        n = len(probabilities)
        if n == 0:
            raise ValueError("logistic calibration needs at least one observation")

        features = [_logit(p) for p in probabilities]
        positives = sum(outcomes)
        negatives = n - positives
        if self.smoothing:
            high = (positives + 1.0) / (positives + 2.0)
            low = 1.0 / (negatives + 2.0)
            targets = [high if y == 1.0 else low for y in outcomes]
        else:
            targets = [float(y) for y in outcomes]

        slope, intercept = 1.0, 0.0
        for iteration in range(1, self.max_iterations + 1):
            gradient_a = gradient_b = 0.0
            hessian_aa = hessian_ab = hessian_bb = 0.0
            for feature, target in zip(features, targets, strict=True):
                predicted = _sigmoid(slope * feature + intercept)
                residual = predicted - target
                weight = max(predicted * (1.0 - predicted), 1e-12)
                gradient_a += residual * feature
                gradient_b += residual
                hessian_aa += weight * feature * feature
                hessian_ab += weight * feature
                hessian_bb += weight

            hessian_aa += self.ridge
            hessian_bb += self.ridge
            determinant = hessian_aa * hessian_bb - hessian_ab * hessian_ab
            if abs(determinant) < 1e-18:
                break
            step_a = (hessian_bb * gradient_a - hessian_ab * gradient_b) / determinant
            step_b = (hessian_aa * gradient_b - hessian_ab * gradient_a) / determinant
            slope -= step_a
            intercept -= step_b
            self.iterations = iteration
            if max(abs(step_a), abs(step_b)) < self.tolerance:
                self.converged = True
                break

        self.slope = slope
        self.intercept = intercept
        self.is_fitted = True
        return self

    def transform(self, probabilities: Sequence[float]) -> list[float]:
        """Apply the fitted sigmoid.

        Raises:
            RuntimeError: If called before :meth:`fit`.
        """
        if not self.is_fitted:
            raise RuntimeError("LogisticCalibrator.transform called before fit")
        return [
            _clip(_sigmoid(self.slope * _logit(p) + self.intercept))
            for p in probabilities
        ]

    @property
    def parameters(self) -> dict[str, float]:
        """The fitted coefficients, for display."""
        return {"slope": self.slope, "intercept": self.intercept}


def default_calibrators() -> list[Calibrator]:
    """The three methods compared throughout the app, baseline first."""
    return [IdentityCalibrator(), IsotonicCalibrator(), LogisticCalibrator()]
