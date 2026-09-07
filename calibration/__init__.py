"""Probability calibration: chronological splits, calibrators and evaluation.

Nothing in this package may look forward in time. See :mod:`calibration.splits`
for the three distinct leakage rules the split enforces.
"""

from calibration.evaluate import (
    CalibrationReport,
    FoldScores,
    MethodResult,
    evaluate_calibrators,
)
from calibration.methods import (
    Calibrator,
    IdentityCalibrator,
    IsotonicCalibrator,
    LogisticCalibrator,
    default_calibrators,
)
from calibration.splits import (
    ChronologicalSplit,
    Observation,
    chronological_split,
    observations_from_result,
)

__all__ = [
    "CalibrationReport",
    "Calibrator",
    "ChronologicalSplit",
    "FoldScores",
    "IdentityCalibrator",
    "IsotonicCalibrator",
    "LogisticCalibrator",
    "MethodResult",
    "Observation",
    "chronological_split",
    "default_calibrators",
    "evaluate_calibrators",
    "observations_from_result",
]
