"""Centralized configuration using Pydantic Settings.

All configuration is loaded from environment variables and .env file.
No hardcoded constants scattered across the codebase.
"""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings

# Project root (two levels up from this file)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TausiAPISettings(BaseSettings):
    """Tausi (Kenya Law) API configuration."""

    token: str = Field(alias="TAUSI_TOKEN")
    base_url: str = "https://caselaw.kenyalaw.org/api/v1"
    initial_rate_limit: float = 1.0  # requests per second (conservative start)
    max_rate_limit: float = 3.0  # maximum requests per second
    backoff_factor: float = 2.0  # exponential backoff multiplier
    max_retries: int = 5
    circuit_breaker_threshold: int = 5  # consecutive failures before pausing
    circuit_breaker_pause_seconds: int = 300  # 5 minute pause
    timeout_seconds: int = 30

    model_config = {"env_file": PROJECT_ROOT / ".env", "extra": "ignore"}


class DataSettings(BaseSettings):
    """Data paths and parameters."""

    raw_dir: Path = PROJECT_ROOT / "data" / "raw"
    interim_dir: Path = PROJECT_ROOT / "data" / "interim"
    processed_dir: Path = PROJECT_ROOT / "data" / "processed"
    reference_dir: Path = PROJECT_ROOT / "data" / "reference"

    # Data collection scope
    filing_year_start: int = 2015
    filing_year_end: int = 2023
    target_court_code: str = "kehc"  # High Court of Kenya
    target_court_station: str = "HCNRB"  # Milimani, Nairobi

    # Exchange rate (configurable, not hardcoded)
    kes_per_usd: float = 129.0  # As of early 2026

    model_config = {"env_file": PROJECT_ROOT / ".env", "extra": "ignore"}


class ModelSettings(BaseSettings):
    """ML model configuration."""

    mlflow_tracking_uri: str = Field(
        default=(PROJECT_ROOT / "mlruns").as_uri(), alias="MLFLOW_TRACKING_URI"
    )
    experiment_name: str = "litigation-analytics-kenya"

    # Data split (temporal)
    train_years: tuple[int, ...] = (2015, 2016, 2017, 2018, 2019, 2020)
    val_years: tuple[int, ...] = (2021,)
    test_years: tuple[int, ...] = (2022, 2023)

    # Feature engineering
    judge_win_rate_min_cases: int = 15  # Minimum cases for judge win rate
    text_embedding_dim: int = 384  # all-MiniLM-L6-v2 output dimension
    text_pca_components: int = 30  # PCA reduction for embeddings
    sentence_transformer_model: str = "all-MiniLM-L6-v2"

    # Optuna HPO
    optuna_n_trials: int = 100
    optuna_cv_folds: int = 5

    # Evaluation gates (literature-aligned: Katz 70.2%, Aletras 79%, PILOT 0.83 AUC)
    min_test_accuracy: float = 0.70
    min_test_auc_roc: float = 0.70
    max_brier_score: float = 0.22
    max_fairness_delta_pp: float = 5.0  # percentage points

    model_config = {"env_file": PROJECT_ROOT / ".env", "extra": "ignore"}


class Settings(BaseSettings):
    """Top-level settings aggregator."""

    tausi: TausiAPISettings = TausiAPISettings()
    data: DataSettings = DataSettings()
    model: ModelSettings = ModelSettings()

    model_config = {"env_file": PROJECT_ROOT / ".env", "extra": "ignore"}


# Singleton instance
settings = Settings()
