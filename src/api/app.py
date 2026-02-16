"""FastAPI prediction service for litigation outcome prediction.

Endpoints:
    POST /predict - Predict case outcome with SHAP explanations
    GET  /health  - Health check with model status
    GET  /model/info - Model metadata and metrics

Loads the latest model from MLflow at startup. All predictions
are logged to a JSON file for post-hoc analysis.

Usage:
    uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000
"""

import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException

from src.api.schemas import (
    FeatureContribution,
    HealthResponse,
    ModelInfoResponse,
    PredictionRequest,
    PredictionResponse,
)
from configs.settings import settings

logger = logging.getLogger(__name__)

# Global state for the loaded model
_model_state = {
    "model": None,
    "model_version": None,
    "model_type": None,
    "feature_names": [],
    "test_metrics": {},
    "calibrator": None,
}

LOG_DIR = Path("logs")
PREDICTION_LOG = LOG_DIR / "predictions.jsonl"


def _load_model():
    """Load the latest model from MLflow."""
    try:
        import mlflow

        mlflow.set_tracking_uri(settings.model.mlflow_tracking_uri)
        client = mlflow.tracking.MlflowClient()

        # Get the latest run from our experiment
        experiment = client.get_experiment_by_name(settings.model.experiment_name)
        if experiment is None:
            logger.warning("No MLflow experiment found. Model not loaded.")
            return

        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["metrics.test_auc_roc DESC"],
            max_results=1,
        )
        if not runs:
            logger.warning("No MLflow runs found. Model not loaded.")
            return

        best_run = runs[0]
        model_uri = f"runs:/{best_run.info.run_id}/model"

        _model_state["model"] = mlflow.xgboost.load_model(model_uri)
        _model_state["model_version"] = best_run.info.run_id[:8]
        _model_state["model_type"] = best_run.data.params.get("model_type", "xgboost")
        _model_state["test_metrics"] = {
            k: v for k, v in best_run.data.metrics.items() if k.startswith("test_")
        }

        logger.info("Loaded model: run=%s, metrics=%s", best_run.info.run_id, _model_state["test_metrics"])

    except Exception as e:
        logger.error("Failed to load model from MLflow: %s", e)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: load model. Shutdown: cleanup."""
    LOG_DIR.mkdir(exist_ok=True)
    _load_model()
    yield


app = FastAPI(
    title="Litigation Analytics Kenya",
    description="AI-assisted case outcome prediction for Kenyan courts",
    version="0.1.0",
    lifespan=lifespan,
)


def _log_prediction(request: dict, response: dict):
    """Append prediction to the JSON Lines log file."""
    entry = {
        "timestamp": datetime.now().isoformat(),
        "request": request,
        "response": response,
    }
    with open(PREDICTION_LOG, "a") as f:
        f.write(json.dumps(entry, default=str) + "\n")


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Predict case outcome and return top-5 SHAP drivers."""
    if _model_state["model"] is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Train a model first and restart the service.",
        )

    # Build feature vector from request
    features = {
        "filing_year": request.filing_year,
        "num_judges": request.num_judges,
        "num_citations": request.num_statutes_cited or 0,
        "cites_dpa": int(request.cites_dpa),
        "has_monetary_claim": int(request.has_monetary_claim),
        "claim_amount_kes": request.claim_amount_kes or 0,
        "log_claim_amount": np.log1p(request.claim_amount_kes or 0),
        "is_post_2018": int(request.filing_year >= 2018),
        "text_length": len(request.case_text) if request.case_text else 0,
    }

    # Convert to DMatrix for XGBoost
    import xgboost as xgb

    feature_names = list(features.keys())
    feature_values = np.array([list(features.values())], dtype=np.float32)
    dmatrix = xgb.DMatrix(feature_values, feature_names=feature_names)

    # Predict
    raw_prob = float(_model_state["model"].predict(dmatrix)[0])

    # Apply calibration if available
    if _model_state["calibrator"] is not None:
        prob = float(_model_state["calibrator"].predict_proba(np.array([raw_prob]))[0])
    else:
        prob = raw_prob

    # Determine prediction and confidence
    prediction = "LIKELY_WIN" if prob >= 0.5 else "LIKELY_LOSS"
    distance_from_boundary = abs(prob - 0.5)
    if distance_from_boundary >= 0.2:
        confidence = "HIGH"
    elif distance_from_boundary >= 0.1:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    # SHAP explanations
    try:
        import shap

        explainer = shap.TreeExplainer(_model_state["model"])
        shap_values = explainer.shap_values(dmatrix)
        top_indices = np.argsort(np.abs(shap_values[0]))[::-1][:5]
        top_features = [
            FeatureContribution(
                feature=feature_names[i],
                shap_value=float(shap_values[0][i]),
                direction="positive" if shap_values[0][i] > 0 else "negative",
            )
            for i in top_indices
        ]
    except Exception as e:
        logger.warning("SHAP explanation failed: %s", e)
        top_features = []

    response = PredictionResponse(
        probability_win=round(prob, 4),
        prediction=prediction,
        confidence=confidence,
        top_features=top_features,
        model_version=_model_state["model_version"] or "unknown",
    )

    # Log the prediction
    _log_prediction(request.model_dump(), response.model_dump())

    return response


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model_loaded=_model_state["model"] is not None,
        model_version=_model_state["model_version"],
    )


@app.get("/model/info", response_model=ModelInfoResponse)
async def model_info():
    """Model metadata and performance metrics."""
    if _model_state["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    return ModelInfoResponse(
        model_type=_model_state["model_type"] or "unknown",
        model_version=_model_state["model_version"] or "unknown",
        training_date=None,
        num_features=len(_model_state["feature_names"]),
        feature_names=_model_state["feature_names"],
        test_metrics=_model_state["test_metrics"],
    )
