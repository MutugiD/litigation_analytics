"""Random Forest model for litigation outcome prediction.

Literature grounding:
- Katz et al. (2017) achieved 70.2% on SCOTUS using Random Forest
- JES 2024 confirmed RF as a competitive baseline for legal prediction
- RF handles mixed feature types (numeric + categorical) well
- Strong interpretability via feature importance (critical for lawyer trust)

In our model progression:
    1. Majority class -> floor
    2. Logistic Regression -> simple baseline
    3. Random Forest -> this model (competitive with XGBoost, better interpretability)
    4. XGBoost -> primary candidate

Usage:
    from src.models.random_forest_model import train_random_forest
    model, metrics = train_random_forest(train_df, val_df, test_df)
"""

import logging

import mlflow
import optuna
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from configs.settings import settings
from src.models.baseline import BINARY_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES

logger = logging.getLogger(__name__)


def _available_features(df: pd.DataFrame, feature_list: list[str]) -> list[str]:
    """Return only features that actually exist in the DataFrame."""
    return [f for f in feature_list if f in df.columns]


def train_random_forest(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str = "outcome_binary",
    n_trials: int | None = None,
) -> tuple[Pipeline, dict]:
    """Train Random Forest with Optuna HPO.

    Uses the same feature set as the logistic regression baseline but with
    a more expressive model class. Katz 2017 showed RF captures judge-level
    patterns that linear models miss.

    Args:
        train_df: Training data with features and target
        val_df: Validation data
        test_df: Test data (touched once for final eval)
        target_col: Binary target column name
        n_trials: Optuna trials (default from settings)

    Returns:
        Tuple of (trained pipeline, metrics dict)
    """
    n_trials = n_trials or settings.model.optuna_n_trials
    mlflow.set_tracking_uri(settings.model.mlflow_tracking_uri)
    mlflow.set_experiment(settings.model.experiment_name)

    # Filter to available features
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

    logger.info(
        "Training Random Forest: %d features, %d train, %d val, %d test",
        len(all_features),
        len(X_train),
        len(X_val),
        len(X_test),
    )

    # Build preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                avail_numeric,
            ),
            (
                "bin",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
                    ]
                ),
                avail_binary,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                avail_cat,
            ),
        ],
        remainder="drop",
    )

    with mlflow.start_run(run_name="random_forest"):
        # --- Optuna HPO ---
        def objective(trial: optuna.Trial) -> float:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 15),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            }

            pipeline = Pipeline(
                [
                    ("preprocess", preprocessor),
                    (
                        "classifier",
                        RandomForestClassifier(
                            **params,
                            random_state=42,
                            n_jobs=-1,
                        ),
                    ),
                ]
            )
            pipeline.fit(X_train, y_train)
            val_probs = pipeline.predict_proba(X_val)[:, 1]
            return roc_auc_score(y_val, val_probs)

        study = optuna.create_study(direction="maximize", study_name="random_forest")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        best_params = study.best_params
        logger.info("Best RF params: %s", best_params)
        mlflow.log_params(best_params)
        mlflow.log_param("model_type", "random_forest")
        mlflow.log_param("features", all_features)

        # --- Train final model with best params ---
        pipeline = Pipeline(
            [
                ("preprocess", preprocessor),
                (
                    "classifier",
                    RandomForestClassifier(
                        **best_params,
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
        pipeline.fit(X_train, y_train)

        # --- Validation metrics ---
        val_preds = pipeline.predict(X_val)
        val_probs = pipeline.predict_proba(X_val)[:, 1]
        val_metrics = {
            "val_accuracy": accuracy_score(y_val, val_preds),
            "val_auc_roc": roc_auc_score(y_val, val_probs),
            "val_brier": brier_score_loss(y_val, val_probs),
        }

        # --- Test metrics (final, one-time evaluation) ---
        test_preds = pipeline.predict(X_test)
        test_probs = pipeline.predict_proba(X_test)[:, 1]
        test_metrics = {
            "test_accuracy": accuracy_score(y_test, test_preds),
            "test_auc_roc": roc_auc_score(y_test, test_probs),
            "test_brier": brier_score_loss(y_test, test_probs),
        }

        metrics = {**val_metrics, **test_metrics}
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(pipeline, "model")

        # Log feature importance from the forest
        rf = pipeline.named_steps["classifier"]
        preprocess = pipeline.named_steps["preprocess"]
        try:
            feature_names = preprocess.get_feature_names_out()
            importance = dict(
                zip(
                    [str(f) for f in feature_names],
                    [float(v) for v in rf.feature_importances_],
                    strict=False,
                )
            )
            mlflow.log_dict(importance, "feature_importance.json")
        except Exception:
            logger.warning("Could not extract feature names for importance logging")

        logger.info("Random Forest: %s", metrics)

    return pipeline, metrics
