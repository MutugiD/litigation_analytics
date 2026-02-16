"""Model evaluation: metrics, fairness checks, and SHAP explanations.

Evaluation gates (from settings, literature-aligned):
- Test accuracy >= 70% (Katz 70.2%, Aletras 79%, JES 2024 72%)
- AUC-ROC >= 0.70 (PILOT 2024: 0.83 on ECHR)
- Brier score <= 0.22 (standard for well-calibrated models)
- Fairness: accuracy delta across groups < 5 percentage points

Usage:
    from src.models.evaluation import evaluate_model, check_gates
    metrics = evaluate_model(y_true, y_probs, metadata_df)
    passed, report = check_gates(metrics)
"""

import logging
from typing import Any

import numpy as np
import pandas as pd
import shap
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    brier_score_loss,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from configs.settings import settings

logger = logging.getLogger(__name__)


def compute_metrics(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute all evaluation metrics for binary classification.

    Args:
        y_true: True binary labels
        y_probs: Predicted probabilities
        threshold: Classification threshold

    Returns:
        Dict of metric_name -> value
    """
    y_pred = (y_probs >= threshold).astype(int)

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "auc_roc": roc_auc_score(y_true, y_probs),
        "brier_score": brier_score_loss(y_true, y_probs),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "n_samples": len(y_true),
        "positive_rate": y_true.mean(),
        "predicted_positive_rate": y_pred.mean(),
    }


def compute_fairness(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    group_labels: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Compute fairness metrics across groups.

    Checks that accuracy difference between any two groups
    is within the allowed delta (default 5 percentage points).

    Args:
        y_true: True binary labels
        y_probs: Predicted probabilities
        group_labels: Group membership for each sample

    Returns:
        Dict with per-group metrics, max delta, and pass/fail.
    """
    y_pred = (y_probs >= threshold).astype(int)
    max_delta = settings.model.max_fairness_delta_pp

    group_metrics = {}
    for group in np.unique(group_labels):
        mask = group_labels == group
        if mask.sum() < 5:  # Too few samples to be meaningful
            continue
        group_metrics[str(group)] = {
            "accuracy": accuracy_score(y_true[mask], y_pred[mask]),
            "n_samples": int(mask.sum()),
        }

    # Compute max accuracy delta between any two groups
    accuracies = [m["accuracy"] for m in group_metrics.values()]
    accuracy_delta_pp = (max(accuracies) - min(accuracies)) * 100 if len(accuracies) >= 2 else 0.0

    return {
        "group_metrics": group_metrics,
        "accuracy_delta_pp": accuracy_delta_pp,
        "max_allowed_delta_pp": max_delta,
        "fairness_pass": accuracy_delta_pp <= max_delta,
    }


def compute_shap_explanations(
    model,
    X: np.ndarray | pd.DataFrame,
    feature_names: list[str] | None = None,
    max_samples: int = 200,
) -> dict:
    """Compute SHAP values for model explanations.

    Args:
        model: Trained model (XGBoost booster or sklearn pipeline)
        X: Feature matrix
        feature_names: Feature names for SHAP output
        max_samples: Max samples to explain (SHAP is expensive)

    Returns:
        Dict with SHAP values and global importance ranking.
    """
    if len(X) > max_samples:
        indices = np.random.RandomState(42).choice(len(X), max_samples, replace=False)
        X_sample = X[indices] if isinstance(X, np.ndarray) else X.iloc[indices]
    else:
        X_sample = X

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # Global feature importance (mean absolute SHAP)
    if feature_names is None and isinstance(X, pd.DataFrame):
        feature_names = list(X.columns)
    elif feature_names is None:
        feature_names = [f"f{i}" for i in range(shap_values.shape[1])]

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance_ranking = sorted(
        zip(feature_names, mean_abs_shap),
        key=lambda x: x[1],
        reverse=True,
    )

    return {
        "shap_values": shap_values,
        "feature_names": feature_names,
        "global_importance": importance_ranking,
        "top_5_features": [name for name, _ in importance_ranking[:5]],
    }


def per_prediction_explanation(
    shap_values: np.ndarray,
    feature_names: list[str],
    index: int,
    top_k: int = 5,
) -> list[dict]:
    """Get top-k SHAP drivers for a single prediction.

    Returns list of dicts with feature name, SHAP value, and direction.
    """
    values = shap_values[index]
    top_indices = np.argsort(np.abs(values))[::-1][:top_k]

    return [
        {
            "feature": feature_names[i],
            "shap_value": float(values[i]),
            "direction": "positive" if values[i] > 0 else "negative",
        }
        for i in top_indices
    ]


def evaluate_model(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    metadata_df: pd.DataFrame | None = None,
    group_column: str | None = None,
) -> dict:
    """Full model evaluation: metrics + fairness + report.

    Args:
        y_true: True binary labels
        y_probs: Predicted probabilities
        metadata_df: Optional DataFrame with group columns for fairness
        group_column: Column name in metadata_df for fairness analysis

    Returns:
        Comprehensive evaluation dict.
    """
    metrics = compute_metrics(y_true, y_probs)

    result = {"metrics": metrics}

    if metadata_df is not None and group_column and group_column in metadata_df.columns:
        fairness = compute_fairness(
            y_true, y_probs, metadata_df[group_column].values
        )
        result["fairness"] = fairness

    # Confusion matrix
    y_pred = (y_probs >= 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    result["confusion_matrix"] = cm.tolist()
    result["classification_report"] = classification_report(
        y_true, y_pred, output_dict=True
    )

    return result


def check_gates(metrics: dict) -> tuple[bool, str]:
    """Check whether the model passes all evaluation gates.

    Gates (from settings, literature-aligned):
    - Accuracy >= 70%
    - AUC-ROC >= 0.70
    - Brier score <= 0.22
    - Fairness delta < 5pp

    Returns:
        Tuple of (all_passed: bool, report: str)
    """
    cfg = settings.model
    m = metrics.get("metrics", metrics)
    f = metrics.get("fairness", {})

    checks = [
        ("accuracy", m.get("accuracy", 0) >= cfg.min_test_accuracy,
         f"accuracy {m.get('accuracy', 0):.3f} >= {cfg.min_test_accuracy}"),
        ("auc_roc", m.get("auc_roc", 0) >= cfg.min_test_auc_roc,
         f"AUC-ROC {m.get('auc_roc', 0):.3f} >= {cfg.min_test_auc_roc}"),
        ("brier", m.get("brier_score", 1) <= cfg.max_brier_score,
         f"Brier {m.get('brier_score', 1):.3f} <= {cfg.max_brier_score}"),
    ]

    if f:
        checks.append((
            "fairness",
            f.get("fairness_pass", False),
            f"fairness delta {f.get('accuracy_delta_pp', 0):.1f}pp <= {cfg.max_fairness_delta_pp}pp",
        ))

    all_passed = all(passed for _, passed, _ in checks)
    lines = []
    for name, passed, desc in checks:
        status = "PASS" if passed else "FAIL"
        lines.append(f"  [{status}] {desc}")

    report = "Evaluation Gates:\n" + "\n".join(lines)
    if all_passed:
        report += "\n\n  Result: ALL GATES PASSED - proceed to Phase 4"
    else:
        report += "\n\n  Result: GATES FAILED - iterate on features/model"

    logger.info(report)
    return all_passed, report
