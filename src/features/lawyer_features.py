"""Lawyer/advocate feature extraction from Kenyan judgment text and API data.

Literature finding (LexEdge 2024, Pre/Dicta): Counsel characteristics are
a significant predictor of case outcomes. Pre/Dicta uses 50-100 data points
per case including attorney information across 15M federal cases.

Approach:
1. Use Tausi API advocates[] field where available (~30-40% of cases)
2. Extract lawyer names from PDF judgment text (supplementary)
3. Compute lawyer-level features: historical win rates, case volume, experience
4. These features are IMPORTANT for the target audience (Kenyan lawyers)

Kenyan judgment text typically contains advocate references in patterns like:
  "Mr. Ochieng for the Plaintiff"
  "Ms. Wanjiku, Advocate, for the Defendant"
  "Learned Counsel for the Applicant, Mr. Kamau"
  "Hamilton Harrison & Mathews Advocates for the Plaintiff"
"""

import logging
import re

import numpy as np
import pandas as pd

from configs.settings import settings

logger = logging.getLogger(__name__)

# Regex patterns for extracting advocate names from Kenyan judgment text
# These target the standard phrasing used in Kenyan judgments
ADVOCATE_PATTERNS = [
    # "Mr./Ms./Mrs. Name for the Plaintiff/Defendant/Applicant/Respondent"
    re.compile(
        r"(?:Mr\.?|Ms\.?|Mrs\.?|Dr\.?)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)"
        r"\s+(?:for|representing|instructed by|appeared for)\s+the\s+"
        r"(Plaintiff|Defendant|Applicant|Respondent|Petitioner|Claimant)",
        re.IGNORECASE,
    ),
    # "Name Advocates for the Plaintiff"
    re.compile(
        r"([A-Z][a-zA-Z]+(?:\s+(?:&|and)\s+[A-Z][a-zA-Z]+)*\s+Advocates?)"
        r"\s+for\s+the\s+"
        r"(Plaintiff|Defendant|Applicant|Respondent|Petitioner|Claimant)",
        re.IGNORECASE,
    ),
    # "Learned Counsel for the Plaintiff, Mr. Name"
    re.compile(
        r"[Ll]earned\s+[Cc]ounsel\s+for\s+the\s+"
        r"(Plaintiff|Defendant|Applicant|Respondent)"
        r"[,\s]+(?:Mr\.?|Ms\.?|Mrs\.?)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
    ),
    # "Advocates: Name1 (P), Name2 (D)" - header section format
    re.compile(
        r"(?:Mr\.?|Ms\.?|Mrs\.?)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+"
        r"(?:holding brief\s+)?for\s+the\s+"
        r"(Plaintiff|Defendant|Applicant|Respondent|1st|2nd)",
        re.IGNORECASE,
    ),
]


def extract_advocates_from_text(text: str) -> dict[str, list[str]]:
    """Extract advocate names and their party affiliation from judgment text.

    Searches the first 15% of the text (header/introduction section)
    where advocate appearances are typically recorded.

    Returns:
        Dict with 'plaintiff_advocates', 'defendant_advocates', 'all_advocates'
    """
    if not text or len(text) < 200:
        return {"plaintiff_advocates": [], "defendant_advocates": [], "all_advocates": []}

    # Focus on the header section (first 15% of text)
    header_end = int(len(text) * 0.15)
    header_text = text[:header_end]

    plaintiff_names = set()
    defendant_names = set()

    plaintiff_roles = {"plaintiff", "applicant", "petitioner", "claimant", "appellant"}
    defendant_roles = {"defendant", "respondent"}

    for pattern in ADVOCATE_PATTERNS:
        for match in pattern.finditer(header_text):
            groups = match.groups()
            if len(groups) >= 2:
                name = groups[0].strip()
                role = groups[1].strip().lower()

                # Normalize role to plaintiff/defendant side
                if any(r in role for r in plaintiff_roles):
                    plaintiff_names.add(name)
                elif any(r in role for r in defendant_roles):
                    defendant_names.add(name)

    all_names = list(plaintiff_names | defendant_names)
    return {
        "plaintiff_advocates": sorted(plaintiff_names),
        "defendant_advocates": sorted(defendant_names),
        "all_advocates": sorted(all_names),
    }


def compute_lawyer_win_rates(
    df: pd.DataFrame,
    min_cases: int = 10,
) -> dict[str, dict]:
    """Compute historical win rates per lawyer with Bayesian shrinkage.

    For each lawyer, computes:
    - win_rate: Bayesian-smoothed win rate
    - case_count: Total cases in the dataset
    - experience_proxy: Number of distinct years active (proxy for seniority)

    Args:
        df: DataFrame with advocate columns and outcome_binary.
        min_cases: Prior strength for Bayesian shrinkage.

    Returns:
        Dict mapping lawyer name -> {win_rate, case_count, experience_proxy}
    """
    labeled = df.dropna(subset=["outcome_binary"]).copy()
    global_rate = labeled["outcome_binary"].mean()

    # Build a lawyer-case mapping from both plaintiff and defendant advocates
    lawyer_records: dict[str, list[dict]] = {}

    for _, row in labeled.iterrows():
        outcome = row["outcome_binary"]
        year = row.get("filing_year", 0)

        # Plaintiff advocates: they "win" when outcome_binary == 1
        for name in row.get("plaintiff_advocates", []) or []:
            if name:
                lawyer_records.setdefault(name, []).append(
                    {"won": outcome == 1, "year": year}
                )

        # Defendant advocates: they "win" when outcome_binary == 0
        for name in row.get("defendant_advocates", []) or []:
            if name:
                lawyer_records.setdefault(name, []).append(
                    {"won": outcome == 0, "year": year}
                )

    # Compute features per lawyer
    lawyer_features = {}
    for name, records in lawyer_records.items():
        n = len(records)
        wins = sum(1 for r in records if r["won"])
        years_active = len(set(r["year"] for r in records))

        # Bayesian shrinkage toward global rate
        smoothed_rate = (wins + min_cases * global_rate) / (n + min_cases)

        lawyer_features[name] = {
            "win_rate": smoothed_rate,
            "case_count": n,
            "experience_proxy": years_active,
        }

    logger.info(
        "Computed features for %d lawyers (global rate: %.3f)",
        len(lawyer_features), global_rate,
    )
    return lawyer_features


def build_lawyer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build lawyer-related features for each case.

    Features generated:
    - plaintiff_lawyer_win_rate: Best plaintiff advocate's historical win rate
    - defendant_lawyer_win_rate: Best defendant advocate's historical win rate
    - lawyer_win_rate_diff: plaintiff - defendant (positive = plaintiff has stronger counsel)
    - plaintiff_lawyer_experience: Max years active among plaintiff advocates
    - defendant_lawyer_experience: Max years active among defendant advocates
    - num_plaintiff_advocates: Number of plaintiff-side advocates
    - num_defendant_advocates: Number of defendant-side advocates
    - has_advocate_data: Whether any advocate information is available
    """
    lawyer_stats = compute_lawyer_win_rates(df)
    features = pd.DataFrame(index=df["case_id"])

    p_win_rates = []
    d_win_rates = []
    p_experience = []
    d_experience = []
    n_p_advocates = []
    n_d_advocates = []

    for _, row in df.iterrows():
        p_advs = row.get("plaintiff_advocates", []) or []
        d_advs = row.get("defendant_advocates", []) or []

        # Plaintiff side: best win rate
        p_rates = [lawyer_stats[n]["win_rate"] for n in p_advs if n in lawyer_stats]
        d_rates = [lawyer_stats[n]["win_rate"] for n in d_advs if n in lawyer_stats]
        p_exp = [lawyer_stats[n]["experience_proxy"] for n in p_advs if n in lawyer_stats]
        d_exp = [lawyer_stats[n]["experience_proxy"] for n in d_advs if n in lawyer_stats]

        p_win_rates.append(max(p_rates) if p_rates else np.nan)
        d_win_rates.append(max(d_rates) if d_rates else np.nan)
        p_experience.append(max(p_exp) if p_exp else np.nan)
        d_experience.append(max(d_exp) if d_exp else np.nan)
        n_p_advocates.append(len(p_advs))
        n_d_advocates.append(len(d_advs))

    features["plaintiff_lawyer_win_rate"] = p_win_rates
    features["defendant_lawyer_win_rate"] = d_win_rates
    features["lawyer_win_rate_diff"] = (
        features["plaintiff_lawyer_win_rate"] - features["defendant_lawyer_win_rate"]
    )
    features["plaintiff_lawyer_experience"] = p_experience
    features["defendant_lawyer_experience"] = d_experience
    features["num_plaintiff_advocates"] = n_p_advocates
    features["num_defendant_advocates"] = n_d_advocates
    features["has_advocate_data"] = (
        (features["num_plaintiff_advocates"] > 0) |
        (features["num_defendant_advocates"] > 0)
    ).astype(int)

    logger.info(
        "Built lawyer features: %d cases, %.1f%% have advocate data",
        len(features),
        100 * features["has_advocate_data"].mean(),
    )
    return features
