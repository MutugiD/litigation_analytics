"""Tabular feature engineering from structured case metadata.

Builds features from the Tausi API, extracted text, and computed lawyer data.

Literature grounding (see docs/EXPERIMENT_PLAN.md for full citations):
- Counsel characteristics are a top predictor (LexEdge 2024, Pre/Dicta)
- Judge assignment patterns explain significant variance (Katz et al. 2017)
- Case-type-specific accuracy varies 70-92% (LexEdge 2024, Aletras 2016)
- Facts/circumstances > pure legal text for prediction (Aletras et al. 2016)

Features:
    Court & temporal: court_code, filing_year, case_age_days, is_post_2018, month
    Judge: judge_win_rate (Bayesian-smoothed)
    Lawyer: plaintiff/defendant win rates, experience proxy, win rate differential
    Case: num_citations, cites_dpa, text_length, claim_amount_kes, case_type
"""

import logging
import re

import numpy as np
import pandas as pd

from configs.settings import settings

logger = logging.getLogger(__name__)

# Multi-pattern monetary amount extraction for Kenyan judgments
AMOUNT_PATTERNS = [
    # "KES 5,000,000" or "Kshs. 5,000,000.00" or "Kshs 5,000,000/="
    re.compile(r"(?:KES|Kshs?\.?)\s*([\d,]+(?:\.\d{1,2})?)(?:/=)?", re.IGNORECASE),
    # "Kenya Shillings 5,000,000" or "Kenya Shillings Five Million"
    re.compile(r"Kenya\s+Shillings?\s*([\d,]+(?:\.\d{1,2})?)", re.IGNORECASE),
    # "5,000,000/=" (Kenyan notation for money)
    re.compile(r"\b([\d,]{4,}(?:\.\d{1,2})?)/="),
    # "a sum of 5,000,000"
    re.compile(r"sum\s+of\s+([\d,]+(?:\.\d{1,2})?)", re.IGNORECASE),
    # "awarded 5,000,000" or "damages of 5,000,000"
    re.compile(r"(?:awarded?|damages?\s+of)\s+([\d,]+(?:\.\d{1,2})?)", re.IGNORECASE),
]

# Word-to-number mapping for Kenyan legal text
WORD_NUMBERS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "hundred": 100,
    "thousand": 1_000,
    "million": 1_000_000,
    "billion": 1_000_000_000,
}


def extract_monetary_amount(text: str) -> float | None:
    """Extract the largest monetary amount from judgment text.

    Uses multiple regex patterns to handle the variety of formats
    found in Kenyan legal documents. Returns the largest amount found,
    which is typically the claim amount rather than costs/fees.

    Returns:
        Amount in KES, or None if no amount found.
    """
    if not text:
        return None

    amounts = []
    for pattern in AMOUNT_PATTERNS:
        for match in pattern.finditer(text):
            try:
                amount_str = match.group(1).replace(",", "")
                amount = float(amount_str)
                if 1_000 <= amount <= 100_000_000_000:  # Reasonable range: 1K to 100B KES
                    amounts.append(amount)
            except (ValueError, IndexError):
                continue

    return max(amounts) if amounts else None


def compute_judge_win_rates(
    df: pd.DataFrame,
    min_cases: int | None = None,
) -> dict[str, float]:
    """Compute historical win rates per judge with Bayesian shrinkage.

    Uses a Beta-Binomial model: the shrinkage target is the global
    win rate. Judges with few cases are pulled toward the global mean.

    Args:
        df: DataFrame with 'primary_judge' and 'outcome_binary' columns.
        min_cases: Minimum cases threshold (from settings if None).

    Returns:
        Dict mapping judge name -> smoothed win rate.
    """
    min_cases = min_cases or settings.model.judge_win_rate_min_cases

    # Global win rate (prior)
    labeled = df.dropna(subset=["outcome_binary"])
    global_rate = labeled["outcome_binary"].mean()
    prior_strength = min_cases  # Strength of the prior (pseudo-observations)

    judge_rates = {}
    for judge, group in labeled.groupby("primary_judge"):
        n = len(group)
        wins = group["outcome_binary"].sum()

        if n < 3:  # Too few to be meaningful at all
            judge_rates[judge] = global_rate
        else:
            # Bayesian shrinkage: (wins + prior_strength * global_rate) / (n + prior_strength)
            smoothed = (wins + prior_strength * global_rate) / (n + prior_strength)
            judge_rates[judge] = smoothed

    logger.info(
        "Computed win rates for %d judges (global rate: %.3f, min_cases: %d)",
        len(judge_rates),
        global_rate,
        min_cases,
    )
    return judge_rates


def build_tabular_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build the full tabular feature matrix from case records.

    Input DataFrame should have columns matching CaseRecord fields.
    Includes lawyer features when advocate data is available.

    Returns:
        DataFrame with engineered features, indexed by case_id.
    """
    from src.features.lawyer_features import build_lawyer_features

    features = pd.DataFrame(index=df["case_id"])

    # --- Direct features ---
    features["filing_year"] = df["filing_year"].values
    features["num_judges"] = df["num_judges"].values
    features["num_citations"] = df["num_citations"].values
    features["cites_dpa"] = df["cites_dpa"].astype(int).values
    features["text_length"] = df["text_length"].fillna(0).values
    features["has_monetary_claim"] = df["has_monetary_claim"].astype(int).values
    features["claim_amount_kes"] = df["claim_amount_kes"].values  # nullable

    # --- Derived features ---

    # Case age in days (judgment_date - approximate filing date)
    if "judgment_date" in df.columns and "filing_year" in df.columns:
        filing_approx = pd.to_datetime(df["filing_year"].astype(str) + "-01-01")
        judgment = pd.to_datetime(df["judgment_date"], errors="coerce")
        features["case_age_days"] = (judgment - filing_approx).dt.days
        features["case_age_days"] = features["case_age_days"].clip(lower=0)

    # Month of decision (cyclical encoding)
    if "judgment_date" in df.columns:
        judgment = pd.to_datetime(df["judgment_date"], errors="coerce")
        month = judgment.dt.month
        features["month_sin"] = np.sin(2 * np.pi * month / 12)
        features["month_cos"] = np.cos(2 * np.pi * month / 12)

    # Post-DCSM indicator (Digital Case Management System, 2018+)
    features["is_post_2018"] = (df["filing_year"] >= 2018).astype(int).values

    # Judge win rate (requires outcome_binary to be populated)
    if "primary_judge" in df.columns and "outcome_binary" in df.columns:
        judge_rates = compute_judge_win_rates(df)
        features["judge_win_rate"] = df["primary_judge"].map(judge_rates).values

    # Log-transformed claim amount (for scale normalization)
    features["log_claim_amount"] = np.log1p(features["claim_amount_kes"].fillna(0))

    # --- Lawyer features (from PDF extraction + API) ---
    # Literature: counsel characteristics are among the top predictors
    # (LexEdge 2024, Pre/Dicta: 50-100 features per case including attorneys)
    if "plaintiff_advocates" in df.columns or "advocate_names" in df.columns:
        try:
            lawyer_df = build_lawyer_features(df)
            for col in lawyer_df.columns:
                features[col] = lawyer_df[col].values
            logger.info(
                "Added %d lawyer features (%.1f%% of cases have advocate data)",
                len(lawyer_df.columns),
                100 * lawyer_df["has_advocate_data"].mean(),
            )
        except Exception as e:
            logger.warning("Lawyer feature extraction failed: %s", e)

    logger.info(
        "Built tabular features: %d cases x %d features",
        len(features),
        len(features.columns),
    )
    return features
