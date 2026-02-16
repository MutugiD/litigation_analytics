"""Tests for model evaluation and calibration utilities."""

import numpy as np
import pytest

from src.models.calibration import CalibratedModel, reliability_diagram_data
from src.models.evaluation import check_gates, compute_fairness, compute_metrics


class TestComputeMetrics:
    """Tests for metric computation."""

    def test_perfect_predictions(self):
        y_true = np.array([1, 1, 0, 0, 1])
        y_probs = np.array([0.9, 0.8, 0.1, 0.2, 0.95])
        metrics = compute_metrics(y_true, y_probs)
        assert metrics["accuracy"] == 1.0
        assert metrics["auc_roc"] == 1.0
        assert metrics["n_samples"] == 5

    def test_random_predictions(self):
        rng = np.random.RandomState(42)
        y_true = rng.randint(0, 2, size=100)
        y_probs = rng.uniform(0, 1, size=100)
        metrics = compute_metrics(y_true, y_probs)
        assert 0.3 <= metrics["accuracy"] <= 0.7  # Around chance
        assert metrics["n_samples"] == 100

    def test_all_same_prediction(self):
        y_true = np.array([1, 0, 1, 0])
        y_probs = np.array([0.6, 0.6, 0.6, 0.6])  # All predict positive
        metrics = compute_metrics(y_true, y_probs)
        assert metrics["accuracy"] == 0.5


class TestComputeFairness:
    """Tests for fairness metric computation."""

    def test_equal_groups(self):
        y_true = np.array([1, 0, 1, 0, 1, 0])
        y_probs = np.array([0.9, 0.1, 0.8, 0.2, 0.7, 0.3])
        groups = np.array(["A", "A", "A", "B", "B", "B"])
        result = compute_fairness(y_true, y_probs, groups)
        assert result["fairness_pass"]  # Both groups have 100% accuracy

    def test_small_group_excluded(self):
        """Groups with <5 samples are excluded from fairness check."""
        y_true = np.ones(10)
        y_probs = np.ones(10) * 0.9
        groups = np.array(["A"] * 8 + ["B"] * 2)  # B has only 2
        result = compute_fairness(y_true, y_probs, groups)
        assert "B" not in result["group_metrics"]


class TestCheckGates:
    """Tests for evaluation gate checking."""

    def test_all_gates_pass(self):
        metrics = {
            "metrics": {
                "accuracy": 0.75,
                "auc_roc": 0.75,
                "brier_score": 0.18,
            }
        }
        passed, report = check_gates(metrics)
        assert passed
        assert "PASS" in report

    def test_accuracy_gate_fails(self):
        metrics = {
            "metrics": {
                "accuracy": 0.55,  # Below 0.60 threshold
                "auc_roc": 0.70,
                "brier_score": 0.20,
            }
        }
        passed, report = check_gates(metrics)
        assert not passed
        assert "FAIL" in report


class TestCalibratedModel:
    """Tests for probability calibration."""

    def test_platt_scaling(self):
        """Platt scaling produces valid probabilities."""
        cal = CalibratedModel(method="platt")
        raw_probs = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        labels = np.array([0, 0, 1, 1, 1])
        cal.fit(raw_probs, labels)
        calibrated = cal.predict_proba(raw_probs)
        assert all(0 <= p <= 1 for p in calibrated)
        assert len(calibrated) == 5

    def test_isotonic_regression(self):
        """Isotonic calibration produces monotonic probabilities."""
        cal = CalibratedModel(method="isotonic")
        rng = np.random.RandomState(42)
        raw_probs = rng.uniform(0, 1, 50)
        labels = (raw_probs > 0.5).astype(int)
        cal.fit(raw_probs, labels)
        calibrated = cal.predict_proba(raw_probs)
        assert all(0 <= p <= 1 for p in calibrated)

    def test_invalid_method(self):
        with pytest.raises(ValueError):
            CalibratedModel(method="invalid")

    def test_predict_before_fit(self):
        cal = CalibratedModel()
        with pytest.raises(RuntimeError):
            cal.predict_proba(np.array([0.5]))


class TestReliabilityDiagram:
    """Tests for reliability diagram data computation."""

    def test_basic(self):
        y_true = np.array([0, 0, 1, 1])
        y_probs = np.array([0.1, 0.2, 0.8, 0.9])
        data = reliability_diagram_data(y_true, y_probs, n_bins=5)
        assert len(data["bin_centers"]) == 5
        assert len(data["bin_counts"]) == 5
        assert data["n_bins"] == 5
