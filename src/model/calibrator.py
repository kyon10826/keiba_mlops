"""Probability calibration using holdout data.

Fixes the data leakage issue in the original notebook where calibration
was performed on training data.
"""

from __future__ import annotations

import os
import pickle

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression


class HoldoutCalibrator:
    """Isotonic regression calibrator trained on a holdout split.

    Usage:
        1. Split validation data into two halves.
        2. Fit calibrator on the first half's raw predictions vs true labels.
        3. Evaluate on the second half.
    """

    def __init__(self, method: str = "isotonic"):
        self.method = method
        self.calibrator = IsotonicRegression(out_of_bounds="clip")

    def fit(self, raw_probs: np.ndarray, y_true: np.ndarray) -> None:
        """Fit calibrator on holdout predictions."""
        self.calibrator.fit(raw_probs, y_true)

    def predict(self, raw_probs: np.ndarray) -> np.ndarray:
        """Calibrate raw probabilities."""
        return self.calibrator.predict(raw_probs)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.calibrator, f)

    @classmethod
    def load(cls, path: str) -> "HoldoutCalibrator":
        obj = cls()
        with open(path, "rb") as f:
            obj.calibrator = pickle.load(f)
        return obj


def calibrate_model(
    model: lgb.Booster,
    valid_x: pd.DataFrame,
    valid_y: pd.Series,
    holdout_fraction: float = 0.5,
    seed: int = 17,
) -> tuple[HoldoutCalibrator, pd.DataFrame, pd.Series]:
    """Calibrate model on holdout split of validation data.

    Uses a temporal (front/back) split instead of random permutation
    to prevent data leakage on time-series data.  The first
    ``holdout_fraction`` of the validation set (earlier period) is used
    for calibration, and the remainder (later period) for evaluation.
    The caller is expected to pass validation data that is already
    sorted chronologically.

    Args:
        model: Trained LightGBM model
        valid_x: Validation features (sorted by time)
        valid_y: Validation labels (sorted by time)
        holdout_fraction: Fraction used for calibration (rest for evaluation)
        seed: Random seed (unused, kept for API compatibility)

    Returns:
        (calibrator, eval_x, eval_y) — calibrator and the evaluation split
    """
    n = len(valid_x)
    split = int(n * holdout_fraction)

    cal_x = valid_x.iloc[:split]
    cal_y = valid_y.iloc[:split].values

    eval_x = valid_x.iloc[split:]
    eval_y = valid_y.iloc[split:]

    raw_probs = model.predict(cal_x)

    calibrator = HoldoutCalibrator()
    calibrator.fit(raw_probs, cal_y)

    return calibrator, eval_x, eval_y
