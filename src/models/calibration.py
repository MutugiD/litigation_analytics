"""Probability calibration for litigation outcome models.

Lawyers need to trust that "70% predicted win probability" actually
means ~70% of such cases are won. Raw model outputs are often
poorly calibrated. Platt scaling (logistic calibration) on the
validation set corrects this.

Usage:
    from src.models.calibration import CalibratedModel
    cal = CalibratedModel(model, method="platt")
    cal.fit(val_probs, val_labels)
    calibrated_probs = cal.predict_proba(test_probs)
"""

import logging

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

logger = logging.getLogger(__name__)


class CalibratedModel:
    """Post-hoc probability calibration wrapper.

    Supports:
    - Platt scaling (logistic regression on model outputs)
    - Isotonic regression (non-parametric)
    """

    def __init__(self, method: str = "platt"):
        if method not in ("platt", "isotonic"):
            raise ValueError(f"Unknown calibration method: {method}")
        self.method = method
        self._calibrator = None

    def fit(self, raw_probs: np.ndarray, true_labels: np.ndarray):
        """Fit the calibration model on validation data.

        Args:
            raw_probs: Uncalibrated probabilities from the model (shape: [n,])
            true_labels: True binary labels (shape: [n,])
        """
        if self.method == "platt":
            self._calibrator = LogisticRegression(C=1e10, solver="lbfgs", max_iter=10000)
            self._calibrator.fit(raw_probs.reshape(-1, 1), true_labels)
        else:
            self._calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            self._calibrator.fit(raw_probs, true_labels)

        # Report calibration improvement
        raw_brier = brier_score_loss(true_labels, raw_probs)
        cal_probs = self.predict_proba(raw_probs)
        cal_brier = brier_score_loss(true_labels, cal_probs)

        logger.info(
            "Calibration (%s): Brier score %.4f -> %.4f (%.1f%% improvement)",
            self.method,
            raw_brier,
            cal_brier,
            100 * (raw_brier - cal_brier) / raw_brier if raw_brier > 0 else 0,
        )

    def predict_proba(self, raw_probs: np.ndarray) -> np.ndarray:
        """Calibrate raw probabilities.

        Args:
            raw_probs: Uncalibrated probabilities (shape: [n,])

        Returns:
            Calibrated probabilities (shape: [n,])
        """
        if self._calibrator is None:
            raise RuntimeError("Calibrator not fitted. Call .fit() first.")

        if self.method == "platt":
            return self._calibrator.predict_proba(raw_probs.reshape(-1, 1))[:, 1]
        else:
            return self._calibrator.predict(raw_probs)


def reliability_diagram_data(
    true_labels: np.ndarray,
    predicted_probs: np.ndarray,
    n_bins: int = 10,
) -> dict:
    """Compute data for a reliability (calibration) diagram.

    Returns:
        Dict with 'bin_centers', 'true_frequencies', 'bin_counts'
        for plotting.
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = []
    true_frequencies = []
    bin_counts = []

    for i in range(n_bins):
        mask = (predicted_probs >= bin_edges[i]) & (predicted_probs < bin_edges[i + 1])
        count = mask.sum()
        bin_counts.append(int(count))
        bin_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)

        if count > 0:
            true_frequencies.append(true_labels[mask].mean())
        else:
            true_frequencies.append(np.nan)

    return {
        "bin_centers": bin_centers,
        "true_frequencies": true_frequencies,
        "bin_counts": bin_counts,
        "n_bins": n_bins,
    }
