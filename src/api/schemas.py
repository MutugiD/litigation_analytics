"""Pydantic request/response schemas for the prediction API."""

from pydantic import BaseModel, Field


class FeatureContribution(BaseModel):
    """A single feature's contribution to the prediction."""

    feature: str
    shap_value: float
    direction: str  # "positive" or "negative"


class PredictionRequest(BaseModel):
    """Request body for the /predict endpoint."""

    case_id: str | None = None
    court_code: str = Field(default="kehc", description="Court code (e.g., kehc)")
    filing_year: int = Field(ge=2000, le=2030, description="Year case was filed")
    judge_id: str | None = Field(None, description="Primary judge identifier")
    case_text: str | None = Field(
        None,
        description="Judgment or filing text. If provided, generates text features.",
        max_length=500_000,
    )
    claim_amount_kes: float | None = Field(None, ge=0, description="Claim amount in KES")
    num_statutes_cited: int | None = Field(None, ge=0)
    cites_dpa: bool = False
    num_judges: int = Field(default=1, ge=1, le=10)
    has_monetary_claim: bool = False

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "court_code": "kehc",
                    "filing_year": 2023,
                    "judge_id": "J. Smith",
                    "claim_amount_kes": 5_000_000,
                    "num_statutes_cited": 3,
                    "cites_dpa": True,
                    "num_judges": 1,
                    "has_monetary_claim": True,
                }
            ]
        }
    }


class PredictionResponse(BaseModel):
    """Response body for the /predict endpoint."""

    probability_win: float = Field(
        description="Calibrated probability of plaintiff/applicant success"
    )
    prediction: str = Field(description="LIKELY_WIN or LIKELY_LOSS")
    confidence: str = Field(description="HIGH (>0.7), MEDIUM (0.5-0.7), or LOW (<0.5)")
    top_features: list[FeatureContribution] = Field(
        description="Top features driving this prediction"
    )
    model_version: str
    disclaimer: str = (
        "This prediction is for research purposes only and does not "
        "constitute legal advice. Always consult a qualified advocate."
    )


class HealthResponse(BaseModel):
    """Response body for the /health endpoint."""

    status: str
    model_loaded: bool
    model_version: str | None


class ModelInfoResponse(BaseModel):
    """Response body for the /model/info endpoint."""

    model_type: str
    model_version: str
    training_date: str | None
    num_features: int
    feature_names: list[str]
    test_metrics: dict
