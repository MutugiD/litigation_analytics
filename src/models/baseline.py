"""Baseline models for litigation outcome prediction.

Establishes performance floors before building complex models:
1. MajorityClassifier - always predicts the most common outcome
2. LogisticRegressionBaseline - L2-regularized logistic regression on tabular features

If logistic regression beats 55% on the test set, that's already
a publishable result for Kenyan legal prediction.

Usage:
    from src.models.baseline import train_baselines
    results = train_baselines(train_df, val_df, test_df, target_col="outcome_binary")
"""

import logging

import mlflow
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from configs.settings import settings

logger = logging.getLogger(__name__)

# Numeric features (from tabular_features.py + lawyer_features.py)
NUMERIC_FEATURES = [
    "filing_year",
    "num_judges",
    "num_citations",
    "text_length",
    "claim_amount_kes",
    "case_age_days",
    "month_sin",
    "month_cos",
    "judge_win_rate",
    "log_claim_amount",
    # Lawyer features - literature shows counsel is a top predictor
    "plaintiff_lawyer_win_rate",
    "defendant_lawyer_win_rate",
    "lawyer_win_rate_diff",
    "plaintiff_lawyer_experience",
    "defendant_lawyer_experience",
    "num_plaintiff_advocates",
    "num_defendant_advocates",
]

# Binary features
BINARY_FEATURES = [
    "cites_dpa",
    "has_monetary_claim",
    "is_post_2018",
    "has_advocate_data",
]

# Categorical features
CATEGORICAL_FEATURES = [
    "court_code",
]


class MajorityClassifier:
    """Always predicts the majority class. Performance floor."""

    def __init__(self):
        self.majority_class = None
        self.class_probability = None

    def fit(self, X, y):
        counts = pd.Series(y).value_counts()
        self.majority_class = counts.idxmax()
        self.class_probability = counts[self.majority_class] / len(y)
        return self

    def predict(self, X):
        return np.full(len(X), self.majority_class)

    def predict_proba(self, X):
        probs = np.full((len(X), 2), 1.0 - self.class_probability)
        probs[:, int(self.majority_class)] = self.class_probability
        return probs


def build_preprocessing_pipeline() -> ColumnTransformer:
    """Build a sklearn ColumnTransformer for mixed feature types."""
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    binary_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("bin", binary_transformer, BINARY_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",  # Drop any columns not listed
    )


def _available_features(df: pd.DataFrame, feature_list: list[str]) -> list[str]:
    """Return only features that actually exist in the DataFrame."""
    return [f for f in feature_list if f in df.columns]


def train_baselines(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str = "outcome_binary",
) -> dict:
    """Train and evaluate baseline models.

    Args:
        train_df: Training features + target
        val_df: Validation features + target
        test_df: Test features + target (touched ONCE for final evaluation)
        target_col: Name of the binary target column

    Returns:
        Dict with model names -> metrics dict
    """
    mlflow.set_tracking_uri(settings.model.mlflow_tracking_uri)
    mlflow.set_experiment(settings.model.experiment_name)

    # Filter to only available features
    avail_numeric = _available_features(train_df, NUMERIC_FEATURES)
    avail_binary = _available_features(train_df, BINARY_FEATURES)
    avail_cat = _available_features(train_df, CATEGORICAL_FEATURES)
    all_features = avail_numeric + avail_binary + avail_cat

    X_train = train_df[all_features]
    y_train = train_df[target_col]
    X_val = val_df[all_features]
    y_val = val_df[target_col]
    X_test = test_df[all_features]
    y_test = test_df[target_col]

    results = {}

    # --- Model 1: Majority Class ---
    with mlflow.start_run(run_name="majority_class"):
        majority = MajorityClassifier()
        majority.fit(X_train, y_train)

        test_preds = majority.predict(X_test)
        test_probs = majority.predict_proba(X_test)[:, 1]

        metrics = {
            "test_accuracy": accuracy_score(y_test, test_preds),
            "test_auc_roc": 0.5,  # By definition for constant predictor
            "test_brier": brier_score_loss(y_test, test_probs),
        }
        mlflow.log_metrics(metrics)
        mlflow.log_param("model_type", "majority_class")
        results["majority_class"] = metrics
        logger.info("Majority class: %s", metrics)

    # --- Model 2: Logistic Regression ---
    with mlflow.start_run(run_name="logistic_regression"):
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]), avail_numeric),
                ("bin", Pipeline([
                    ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
                ]), avail_binary),
                ("cat", Pipeline([
                    ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                ]), avail_cat),
            ],
            remainder="drop",
        )

        pipeline = Pipeline([
            ("preprocess", preprocessor),
            ("classifier", LogisticRegression(
                C=1.0,
                penalty="l2",
                solver="lbfgs",
                max_iter=1000,
                random_state=42,
            )),
        ])

        pipeline.fit(X_train, y_train)

        # Validation metrics (for model selection)
        val_preds = pipeline.predict(X_val)
        val_probs = pipeline.predict_proba(X_val)[:, 1]
        val_metrics = {
            "val_accuracy": accuracy_score(y_val, val_preds),
            "val_auc_roc": roc_auc_score(y_val, val_probs),
            "val_brier": brier_score_loss(y_val, val_probs),
        }

        # Test metrics (final evaluation)
        test_preds = pipeline.predict(X_test)
        test_probs = pipeline.predict_proba(X_test)[:, 1]
        test_metrics = {
            "test_accuracy": accuracy_score(y_test, test_preds),
            "test_auc_roc": roc_auc_score(y_test, test_probs),
            "test_brier": brier_score_loss(y_test, test_probs),
        }

        metrics = {**val_metrics, **test_metrics}
        mlflow.log_metrics(metrics)
        mlflow.log_param("model_type", "logistic_regression")
        mlflow.log_param("features", all_features)
        mlflow.sklearn.log_model(pipeline, "model")

        results["logistic_regression"] = metrics
        logger.info("Logistic Regression: %s", metrics)

    return results
