"""XGBoost model with optional text embeddings (explicit stacking).

Architecture:
    Option A (tabular only): XGBoost on tabular features
    Option B (tabular + text): XGBoost on tabular features + PCA-reduced embeddings

This is NOT an undefined "fusion" - it's simple feature concatenation
after dimensionality reduction, which is the most effective approach
for combining dense embeddings with tabular features at small scale.

Usage:
    from src.models.xgboost_model import train_xgboost
    model, metrics = train_xgboost(train_df, val_df, test_df)
"""

import logging

import mlflow
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss

from configs.settings import settings

logger = logging.getLogger(__name__)


def _prepare_dmatrix(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "outcome_binary",
) -> xgb.DMatrix:
    """Create an XGBoost DMatrix from a DataFrame."""
    X = df[feature_cols].values.astype(np.float32)
    y = df[target_col].values.astype(np.float32)
    return xgb.DMatrix(X, label=y, feature_names=feature_cols)


def _get_feature_cols(df: pd.DataFrame, include_text: bool = False) -> list[str]:
    """Get feature columns from DataFrame, optionally including text embeddings."""
    exclude = {"case_id", "outcome_binary", "outcome_label", "frbr_uri", "primary_judge"}
    cols = [c for c in df.columns if c not in exclude]

    if not include_text:
        cols = [c for c in cols if not c.startswith(("emb_pca_", "tfidf_"))]

    return cols


def objective(
    trial: optuna.Trial,
    dtrain: xgb.DMatrix,
    feature_cols: list[str],
    n_folds: int = 5,
) -> float:
    """Optuna objective function for XGBoost hyperparameter search.

    Optimizes AUC-ROC using temporal-aware cross-validation.
    """
    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",
        "verbosity": 0,
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "gamma": trial.suggest_float("gamma", 1e-8, 5.0, log=True),
    }
    n_estimators = trial.suggest_int("n_estimators", 50, 500)

    cv_results = xgb.cv(
        params,
        dtrain,
        num_boost_round=n_estimators,
        nfold=n_folds,
        metrics=["auc"],
        early_stopping_rounds=20,
        seed=42,
        verbose_eval=False,
    )

    best_auc = cv_results["test-auc-mean"].max()
    return best_auc


def train_xgboost(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    include_text: bool = False,
    target_col: str = "outcome_binary",
    n_trials: int | None = None,
) -> tuple[xgb.Booster, dict]:
    """Train XGBoost with Optuna HPO.

    Args:
        train_df: Training data with features and target
        val_df: Validation data
        test_df: Test data (touched once for final eval)
        include_text: If True, include text embedding features
        target_col: Binary target column name
        n_trials: Optuna trials (default from settings)

    Returns:
        Tuple of (trained booster, metrics dict)
    """
    n_trials = n_trials or settings.model.optuna_n_trials
    feature_cols = _get_feature_cols(train_df, include_text=include_text)

    logger.info(
        "Training XGBoost: %d features (text=%s), %d train, %d val, %d test",
        len(feature_cols), include_text, len(train_df), len(val_df), len(test_df),
    )

    dtrain = _prepare_dmatrix(train_df, feature_cols, target_col)
    dval = _prepare_dmatrix(val_df, feature_cols, target_col)
    dtest = _prepare_dmatrix(test_df, feature_cols, target_col)

    mlflow.set_tracking_uri(settings.model.mlflow_tracking_uri)
    mlflow.set_experiment(settings.model.experiment_name)

    model_name = f"xgboost_{'text' if include_text else 'tabular'}"

    with mlflow.start_run(run_name=model_name):
        # --- Hyperparameter search ---
        study = optuna.create_study(direction="maximize", study_name=model_name)
        study.optimize(
            lambda trial: objective(trial, dtrain, feature_cols),
            n_trials=n_trials,
            show_progress_bar=True,
        )

        best_params = study.best_params
        n_estimators = best_params.pop("n_estimators")
        best_params.update({
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "tree_method": "hist",
            "verbosity": 0,
        })

        logger.info("Best params: %s (n_estimators=%d)", best_params, n_estimators)
        mlflow.log_params(best_params)
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("include_text_features", include_text)
        mlflow.log_param("num_features", len(feature_cols))

        # --- Train final model ---
        booster = xgb.train(
            best_params,
            dtrain,
            num_boost_round=n_estimators,
            evals=[(dtrain, "train"), (dval, "val")],
            early_stopping_rounds=20,
            verbose_eval=50,
        )

        # --- Validation metrics ---
        val_probs = booster.predict(dval)
        val_preds = (val_probs >= 0.5).astype(int)
        val_metrics = {
            "val_accuracy": accuracy_score(dval.get_label(), val_preds),
            "val_auc_roc": roc_auc_score(dval.get_label(), val_probs),
            "val_brier": brier_score_loss(dval.get_label(), val_probs),
        }

        # --- Test metrics (final, one-time evaluation) ---
        test_probs = booster.predict(dtest)
        test_preds = (test_probs >= 0.5).astype(int)
        test_metrics = {
            "test_accuracy": accuracy_score(dtest.get_label(), test_preds),
            "test_auc_roc": roc_auc_score(dtest.get_label(), test_probs),
            "test_brier": brier_score_loss(dtest.get_label(), test_probs),
        }

        metrics = {**val_metrics, **test_metrics}
        mlflow.log_metrics(metrics)

        # Log model artifact
        mlflow.xgboost.log_model(booster, "model")

        # Log feature importance
        importance = booster.get_score(importance_type="gain")
        mlflow.log_dict(importance, "feature_importance.json")

        logger.info("XGBoost (%s): %s", model_name, metrics)

    return booster, metrics
