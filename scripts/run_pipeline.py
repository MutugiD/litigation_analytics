"""End-to-end training pipeline: fetch data, engineer features, train models, report.

Usage:
    python scripts/run_pipeline.py --scrape --n-cases 200   # Scrape 200 cases from kenyalaw.org
    python scripts/run_pipeline.py --scrape --n-cases 750   # Scrape 750 cases
    python scripts/run_pipeline.py --skip-download          # Use cached scraped data
    python scripts/run_pipeline.py --synthetic              # Use synthetic data for testing
"""

import argparse
import asyncio
import json
import logging
import re
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from configs.settings import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("pipeline")


# ---------------------------------------------------------------------------
# Step 1: Data Collection
# ---------------------------------------------------------------------------

async def fetch_cases_from_api(max_pages: int = 50) -> list[dict]:
    """Fetch KEHC decisions from Tausi API for 2015-2023."""
    from src.data.api_client import TausiClient

    all_decisions = []
    async with TausiClient() as client:
        for year in range(settings.data.filing_year_start, settings.data.filing_year_end + 1):
            logger.info("Fetching year %d...", year)
            count = 0
            async for decision in client.iter_decisions(
                court=settings.data.target_court_code,
                year=year,
            ):
                all_decisions.append(decision)
                count += 1
                if count >= max_pages * 20:  # ~20 results per page
                    break
            logger.info("Year %d: fetched %d decisions", year, count)

    logger.info("Total decisions fetched: %d", len(all_decisions))
    return all_decisions


def save_raw_decisions(decisions: list[dict], output_dir: Path) -> Path:
    """Save raw API decisions to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "all_decisions.json"
    with open(output_path, "w") as f:
        json.dump(decisions, f, indent=2, default=str)
    logger.info("Saved %d decisions to %s", len(decisions), output_path)
    return output_path


def load_raw_decisions(raw_dir: Path) -> list[dict]:
    """Load previously saved raw decisions."""
    path = raw_dir / "all_decisions.json"
    if not path.exists():
        raise FileNotFoundError(f"No cached data at {path}. Run without --skip-download.")
    with open(path) as f:
        decisions = json.load(f)
    logger.info("Loaded %d cached decisions from %s", len(decisions), path)
    return decisions


# ---------------------------------------------------------------------------
# Step 2: Validate and Build Case Records
# ---------------------------------------------------------------------------

def build_case_records(raw_decisions: list[dict]) -> pd.DataFrame:
    """Convert raw API responses to validated CaseRecords."""
    from src.data.validators import TausiDecisionRaw, CaseRecord, OutcomeLabel

    records = []
    validation_errors = 0

    for raw in raw_decisions:
        try:
            tausi = TausiDecisionRaw(**raw)

            record = CaseRecord(
                case_id=tausi.case_id,
                frbr_uri=tausi.frbr_uri,
                court_code=tausi.court_code or "kehc",
                filing_year=tausi.filing_year or 2020,
                judgment_date=tausi.judgment_date,
                case_type=tausi.casetype.get("name") if tausi.casetype else None,
                judge_names=tausi.judge_names,
                advocate_names=tausi.advocate_names,
                cited_statutes=tausi.cited_statute_titles,
                has_pdf=bool(tausi.content_url),
                text_length=0,
            )
            records.append(record.model_dump())
        except Exception as e:
            validation_errors += 1
            if validation_errors <= 5:
                logger.warning("Validation error: %s", e)

    logger.info(
        "Built %d case records (%d validation errors)",
        len(records), validation_errors,
    )
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Step 3: Parse Outcomes
# ---------------------------------------------------------------------------

def parse_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    """Apply outcome parser to extract binary labels.

    For cases without PDF text, we attempt to infer from metadata.
    """
    from src.features.outcome_parser import parse_outcome

    outcomes = []
    for _, row in df.iterrows():
        # Try to parse from text if available
        text = ""  # Would come from PDF extraction
        result = parse_outcome(text) if text else {
            "label": "UNDETERMINED",
            "confidence": 0.0,
        }
        outcomes.append(result)

    df["outcome_label"] = [o["label"] for o in outcomes]
    df["outcome_confidence"] = [o["confidence"] for o in outcomes]

    # Map to binary
    from src.data.validators import POSITIVE_OUTCOMES, NEGATIVE_OUTCOMES, OutcomeLabel
    df["outcome_binary"] = df["outcome_label"].apply(
        lambda x: 1 if x in {o.value for o in POSITIVE_OUTCOMES}
        else (0 if x in {o.value for o in NEGATIVE_OUTCOMES} else None)
    )

    labeled = df["outcome_binary"].notna().sum()
    logger.info("Outcome labels: %d labeled / %d total", labeled, len(df))
    return df


# ---------------------------------------------------------------------------
# Step 4: Generate Synthetic Data (for pipeline testing)
# ---------------------------------------------------------------------------

def generate_synthetic_data(n_cases: int = 600, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic data that mimics real Kenyan court case structure.

    This is for pipeline testing ONLY. Real training requires actual Tausi data.
    Synthetic data follows the distributions observed in literature:
    - ~55-60% plaintiff win rate (typical for commercial cases)
    - Judge win rates varying around 0.5-0.7
    - Lawyer features with Bayesian-smoothed rates
    """
    rng = np.random.RandomState(seed)

    judges = [f"Justice_{chr(65+i)}" for i in range(15)]
    p_lawyers = [f"P_Advocate_{i}" for i in range(30)]
    d_lawyers = [f"D_Advocate_{i}" for i in range(30)]

    # Generate years with temporal distribution
    years = rng.choice(range(2015, 2024), size=n_cases, p=[
        0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.13, 0.12, 0.12  # More recent = more cases
    ])

    records = []
    for i in range(n_cases):
        year = years[i]
        judge = rng.choice(judges)
        p_lawyer = rng.choice(p_lawyers)
        d_lawyer = rng.choice(d_lawyers)

        # Outcome influenced by judge, lawyers, and randomness
        judge_effect = hash(judge) % 20 / 100 - 0.1  # -0.1 to +0.1
        lawyer_effect = (hash(p_lawyer) % 15 - hash(d_lawyer) % 15) / 100
        base_prob = 0.57 + judge_effect + lawyer_effect
        outcome = 1 if rng.random() < np.clip(base_prob, 0.3, 0.8) else 0

        has_monetary = rng.random() < 0.7
        claim_amount = rng.lognormal(16, 1.5) if has_monetary else 0  # ~10M KES median
        text_length = int(rng.lognormal(9, 0.8))  # ~8000 chars median
        num_citations = rng.poisson(3)
        judgment_month = rng.randint(1, 13)

        records.append({
            "case_id": f"KEHC_{year}_{i:04d}",
            "frbr_uri": f"/akn/ke/judgment/kehc/{year}/{i}",
            "court_code": "kehc",
            "filing_year": int(year),
            "judgment_date": f"{year}-{judgment_month:02d}-15",
            "primary_judge": judge,
            "judge_names": [judge],
            "num_judges": 1,
            "advocate_names": [p_lawyer, d_lawyer],
            "plaintiff_advocates": [p_lawyer],
            "defendant_advocates": [d_lawyer],
            "num_advocates": 2,
            "has_advocate_data": True,
            "cited_statutes": [f"Act_{j}" for j in range(num_citations)],
            "num_citations": num_citations,
            "cites_dpa": rng.random() < 0.05,
            "has_pdf": True,
            "has_text": True,
            "text_length": text_length,
            "outcome_label": "JUDGMENT_FOR_PLAINTIFF" if outcome == 1 else "JUDGMENT_FOR_DEFENDANT",
            "outcome_confidence": 0.9,
            "outcome_binary": outcome,
            "claim_amount_kes": claim_amount if has_monetary else None,
            "has_monetary_claim": has_monetary,
        })

    df = pd.DataFrame(records)
    logger.info(
        "Generated %d synthetic cases (%.1f%% plaintiff wins, %d-%d years)",
        len(df), 100 * df["outcome_binary"].mean(),
        df["filing_year"].min(), df["filing_year"].max(),
    )
    return df


# ---------------------------------------------------------------------------
# Step 4b: Scrape from kenyalaw.org
# ---------------------------------------------------------------------------

OUTCOME_MAP = {
    # Positive outcomes (plaintiff/applicant wins) -> 1
    "allowed": 1,
    "granted": 1,
    "upheld": 1,
    "succeeded": 1,
    # Negative outcomes (defendant/respondent wins) -> 0
    "dismissed": 0,
    "struck out": 0,
    "declined": 0,
    "rejected": 0,
    "refused": 0,
    "acquitted": 0,
    "convicted": 0,  # criminal - treat as negative from plaintiff perspective
    # Excluded
    "settled by consent": None,
    "withdrawn": None,
}


def map_outcome_to_binary(outcome_str: str) -> int | None:
    """Map scraped outcome string to binary label."""
    if not outcome_str:
        return None
    outcome_lower = outcome_str.strip().lower()
    for key, val in OUTCOME_MAP.items():
        if key in outcome_lower:
            return val
    return None


def scrape_cases(n_cases: int = 200) -> pd.DataFrame:
    """Scrape KEHC cases from new.kenyalaw.org.

    Distributes cases across years 2015-2023, proportional to target.
    """
    from src.data.scraper import KenyaLawScraper

    years = list(range(2015, 2024))
    per_year = max(5, n_cases // len(years))
    # Give more to recent years (more data available)
    allocations = {}
    remaining = n_cases
    for y in years:
        alloc = min(per_year + (5 if y >= 2020 else 0), remaining)
        allocations[y] = alloc
        remaining -= alloc
    # Distribute remaining to latest years
    for y in reversed(years):
        if remaining <= 0:
            break
        add = min(remaining, 20)
        allocations[y] += add
        remaining -= add

    logger.info("Scraping plan: %s (total target: %d)", allocations, n_cases)

    with KenyaLawScraper(delay=0.8) as scraper:
        all_cases = []
        for year, max_cases in allocations.items():
            if max_cases <= 0:
                continue
            logger.info("Scraping year %d (target: %d cases)...", year, max_cases)
            cases = scraper.scrape_kehc_cases(
                years=[year],
                max_per_year=max_cases,
                max_pages=min(10, (max_cases // 50) + 1),
            )
            all_cases.extend(cases)
            logger.info("Year %d: scraped %d cases (running total: %d)", year, len(cases), len(all_cases))

    df = pd.DataFrame(all_cases)
    logger.info("Total scraped: %d cases", len(df))

    # Map outcomes to binary
    df["outcome_binary"] = df["outcome"].apply(map_outcome_to_binary)
    labeled = df["outcome_binary"].notna().sum()
    logger.info("Outcome mapping: %d labeled / %d total (%.1f%%)",
                labeled, len(df), 100 * labeled / len(df) if len(df) > 0 else 0)

    # Log outcome distribution
    if "outcome" in df.columns:
        logger.info("Outcome distribution:\n%s", df["outcome"].value_counts().to_string())

    # Save to disk for reuse
    save_path = settings.data.raw_dir / "scraped_cases.json"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(save_path, orient="records", indent=2)
    logger.info("Saved scraped data to %s", save_path)

    return df


def load_scraped_cases() -> pd.DataFrame:
    """Load previously scraped cases."""
    path = settings.data.raw_dir / "scraped_cases.json"
    if not path.exists():
        raise FileNotFoundError(f"No scraped data at {path}. Run with --scrape first.")
    df = pd.read_json(path)
    logger.info("Loaded %d cached scraped cases", len(df))
    return df


def prepare_scraped_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Transform scraped data into the format expected by feature engineering.

    Scraped data has raw fields (judges list, outcome string, judgment_text).
    This computes derived columns needed by build_tabular_features:
      num_judges, num_citations, cites_dpa, has_monetary_claim, claim_amount_kes,
      has_text, num_advocates, has_advocate_data.
    """
    from src.features.tabular_features import extract_monetary_amount

    # Ensure judges is a list and compute num_judges
    if "judges" in df.columns:
        df["judges"] = df["judges"].apply(lambda x: x if isinstance(x, list) else [])
        df["num_judges"] = df["judges"].apply(len)
    else:
        df["num_judges"] = 1

    # Compute citation count from judgment_text
    def count_citations(text):
        if not text or not isinstance(text, str):
            return 0
        # Count references to Acts, Cap., Sections
        acts = len(re.findall(r'\b(?:Act|Cap\.|Section)\b', text, re.IGNORECASE))
        return min(acts, 50)  # cap at 50

    df["num_citations"] = df["judgment_text"].apply(count_citations)

    # Check for Data Protection Act citation
    def check_dpa(text):
        if not text or not isinstance(text, str):
            return False
        return bool(re.search(r'Data\s+Protection\s+Act', text, re.IGNORECASE))

    df["cites_dpa"] = df["judgment_text"].apply(check_dpa)

    # Extract monetary amounts from judgment text
    df["claim_amount_kes"] = df["judgment_text"].apply(extract_monetary_amount)
    df["has_monetary_claim"] = df["claim_amount_kes"].notna()

    # Text availability
    df["has_text"] = df["text_length"] > 0

    # Advocate counts
    for col in ["advocate_names", "plaintiff_advocates", "defendant_advocates"]:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: x if isinstance(x, list) else [])
        else:
            df[col] = [[] for _ in range(len(df))]

    df["num_advocates"] = df["advocate_names"].apply(len)
    df["has_advocate_data"] = df["num_advocates"] > 0

    # Ensure primary_judge exists
    if "primary_judge" not in df.columns or df["primary_judge"].isna().all():
        if "judges" in df.columns:
            df["primary_judge"] = df["judges"].apply(
                lambda x: x[0] if isinstance(x, list) and x else "Unknown"
            )

    # For cases with text but no outcome, try text-based extraction
    if "judgment_text" in df.columns:
        from src.data.scraper import KenyaLawScraper
        _scraper = KenyaLawScraper()
        empty_outcome_mask = (df["outcome"] == "") & (df["text_length"] > 200)
        fixed = 0
        for idx in df[empty_outcome_mask].index:
            extracted = _scraper._extract_outcome_from_text(df.at[idx, "judgment_text"])
            if extracted:
                df.at[idx, "outcome"] = extracted
                fixed += 1
        if fixed > 0:
            logger.info("Extracted %d additional outcomes from judgment text", fixed)

    # Map outcome to binary
    df["outcome_binary"] = df["outcome"].apply(map_outcome_to_binary)

    labeled = df["outcome_binary"].notna().sum()
    logger.info(
        "Prepared scraped data: %d cases, %d labeled (%.1f%%), %d with advocates",
        len(df), labeled, 100 * labeled / len(df) if len(df) > 0 else 0,
        df["has_advocate_data"].sum(),
    )
    return df


# ---------------------------------------------------------------------------
# Step 5: Feature Engineering
# ---------------------------------------------------------------------------

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build full feature matrix from case records."""
    from src.features.tabular_features import build_tabular_features

    features = build_tabular_features(df)

    # Add back the target and metadata columns
    features["outcome_binary"] = df.set_index("case_id")["outcome_binary"]
    features["primary_judge"] = df.set_index("case_id")["primary_judge"]
    features["filing_year"] = df.set_index("case_id")["filing_year"]

    # Drop rows without binary labels
    before = len(features)
    features = features.dropna(subset=["outcome_binary"])
    features["outcome_binary"] = features["outcome_binary"].astype(int)
    logger.info("Features: %d cases x %d columns (dropped %d unlabeled)",
                len(features), len(features.columns), before - len(features))

    return features


def temporal_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by filing year (temporal, not random). Critical for legal ML (PILOT 2024)."""
    train = df[df["filing_year"].isin(settings.model.train_years)]
    val = df[df["filing_year"].isin(settings.model.val_years)]
    test = df[df["filing_year"].isin(settings.model.test_years)]

    logger.info("Temporal split: train=%d, val=%d, test=%d", len(train), len(val), len(test))
    logger.info("  Train years: %s", sorted(train["filing_year"].unique()))
    logger.info("  Val years: %s", sorted(val["filing_year"].unique()))
    logger.info("  Test years: %s", sorted(test["filing_year"].unique()))
    logger.info("  Train positive rate: %.1f%%", 100 * train["outcome_binary"].mean())
    logger.info("  Test positive rate: %.1f%%", 100 * test["outcome_binary"].mean())

    return train, val, test


# ---------------------------------------------------------------------------
# Step 6: Train Models
# ---------------------------------------------------------------------------

def train_all_models(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    n_trials: int = 30,
) -> dict:
    """Train all models in the literature-aligned progression.

    Progression:
    1. Majority class (floor)
    2. Logistic Regression (Zeleznikow 2023)
    3. Random Forest (Katz 2017, JES 2024)
    4. XGBoost tabular (JES 2024: top performer)
    """
    from src.models.baseline import train_baselines
    from src.models.random_forest_model import train_random_forest
    from src.models.xgboost_model import train_xgboost

    all_results = {}

    # --- Baselines (Majority Class + Logistic Regression) ---
    logger.info("=" * 60)
    logger.info("Training baselines (Majority Class + Logistic Regression)...")
    baseline_results = train_baselines(train_df, val_df, test_df)
    all_results.update(baseline_results)

    # --- Random Forest (Katz 2017: 70.2% on SCOTUS) ---
    logger.info("=" * 60)
    logger.info("Training Random Forest (Katz 2017, JES 2024)...")
    try:
        rf_model, rf_metrics = train_random_forest(
            train_df, val_df, test_df, n_trials=n_trials,
        )
        all_results["random_forest"] = rf_metrics
    except Exception as e:
        logger.error("Random Forest failed: %s", e)

    # --- XGBoost Tabular (JES 2024: 72% best performer) ---
    logger.info("=" * 60)
    logger.info("Training XGBoost tabular (JES 2024: top performer)...")
    try:
        xgb_model, xgb_metrics = train_xgboost(
            train_df, val_df, test_df,
            include_text=False,
            n_trials=n_trials,
        )
        all_results["xgboost_tabular"] = xgb_metrics
    except Exception as e:
        logger.error("XGBoost failed: %s", e)

    return all_results


# ---------------------------------------------------------------------------
# Step 7: Report Results
# ---------------------------------------------------------------------------

def report_results(results: dict, test_df: pd.DataFrame) -> str:
    """Generate a comprehensive results report."""
    lines = []
    lines.append("=" * 70)
    lines.append("LITIGATION ANALYTICS KENYA - MODEL TRAINING RESULTS")
    lines.append("=" * 70)
    lines.append("")

    # Dataset summary
    lines.append("DATASET:")
    lines.append(f"  Test set size: {len(test_df)} cases")
    lines.append(f"  Test positive rate: {100 * test_df['outcome_binary'].mean():.1f}%")
    lines.append("")

    # Model comparison table
    lines.append("MODEL COMPARISON:")
    lines.append(f"  {'Model':<25} {'Test Acc':>10} {'Test AUC':>10} {'Test Brier':>12}")
    lines.append("  " + "-" * 57)

    best_model = None
    best_auc = 0

    for name, metrics in results.items():
        acc = metrics.get("test_accuracy", 0)
        auc = metrics.get("test_auc_roc", 0)
        brier = metrics.get("test_brier", 1)
        lines.append(f"  {name:<25} {acc:>10.4f} {auc:>10.4f} {brier:>12.4f}")
        if auc > best_auc:
            best_auc = auc
            best_model = name

    lines.append("")
    lines.append(f"  Best model (by AUC): {best_model} (AUC={best_auc:.4f})")
    lines.append("")

    # Gate check
    lines.append("EVALUATION GATES (literature-aligned):")
    cfg = settings.model
    best = results.get(best_model, {})
    acc = best.get("test_accuracy", 0)
    auc = best.get("test_auc_roc", 0)
    brier = best.get("test_brier", 1)

    acc_pass = acc >= cfg.min_test_accuracy
    auc_pass = auc >= cfg.min_test_auc_roc
    brier_pass = brier <= cfg.max_brier_score

    lines.append(f"  [{'PASS' if acc_pass else 'FAIL'}] Accuracy {acc:.4f} >= {cfg.min_test_accuracy} (Katz 70.2%, JES 72%)")
    lines.append(f"  [{'PASS' if auc_pass else 'FAIL'}] AUC-ROC {auc:.4f} >= {cfg.min_test_auc_roc} (PILOT 0.83)")
    lines.append(f"  [{'PASS' if brier_pass else 'FAIL'}] Brier {brier:.4f} <= {cfg.max_brier_score}")
    lines.append("")

    all_pass = acc_pass and auc_pass and brier_pass
    if all_pass:
        lines.append("  RESULT: ALL GATES PASSED -> Proceed to Phase 4 (API Deployment)")
    else:
        lines.append("  RESULT: GATES NOT MET -> Iterate on features/model")
        lines.append("  (Note: synthetic data results are for pipeline validation only)")
    lines.append("")

    # Literature comparison
    lines.append("LITERATURE COMPARISON:")
    lines.append(f"  Our best:          {acc:.1%} accuracy, {auc:.4f} AUC")
    lines.append(f"  Katz 2017 (SCOTUS): 70.2% accuracy (RF)")
    lines.append(f"  JES 2024 (SCOTUS):  72.0% accuracy (XGBoost)")
    lines.append(f"  Aletras 2016 (ECHR): 79.0% accuracy (SVM)")
    lines.append(f"  PILOT 2024 (ECHR):   0.83 AUC")
    lines.append(f"  LexEdge (commercial): 80-87% accuracy")
    lines.append("")
    lines.append("=" * 70)

    report = "\n".join(lines)
    return report


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main_async(args):
    """Async main for API data fetching."""
    decisions = await fetch_cases_from_api(max_pages=args.max_pages)
    save_raw_decisions(decisions, settings.data.raw_dir)
    return decisions


def main():
    parser = argparse.ArgumentParser(description="Litigation Analytics Training Pipeline")
    parser.add_argument("--scrape", action="store_true", help="Scrape data from kenyalaw.org")
    parser.add_argument("--skip-download", action="store_true", help="Use cached data (scraped or API)")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data for testing")
    parser.add_argument("--max-pages", type=int, default=50, help="Max pages per year from API")
    parser.add_argument("--n-trials", type=int, default=30, help="Optuna trials per model")
    parser.add_argument("--n-cases", type=int, default=600, help="Number of cases (scrape or synthetic)")
    args = parser.parse_args()

    mode = "scrape" if args.scrape else (
        "synthetic" if args.synthetic else (
            "cached" if args.skip_download else "api"
        )
    )
    logger.info("Starting litigation analytics training pipeline...")
    logger.info("Mode: %s, n_cases: %d", mode, args.n_cases)

    # Step 1: Get data
    if args.synthetic:
        logger.info("Using synthetic data for pipeline testing...")
        df = generate_synthetic_data(n_cases=args.n_cases)
    elif args.scrape:
        logger.info("Scraping %d cases from kenyalaw.org...", args.n_cases)
        df = scrape_cases(n_cases=args.n_cases)
        df = prepare_scraped_dataframe(df)
    elif args.skip_download:
        # Try scraped data first, then API data
        try:
            df = load_scraped_cases()
            df = prepare_scraped_dataframe(df)
        except FileNotFoundError:
            raw = load_raw_decisions(settings.data.raw_dir)
            df = build_case_records(raw)
            df = parse_outcomes(df)
    else:
        raw = asyncio.run(main_async(args))
        df = build_case_records(raw)
        df = parse_outcomes(df)

    # Step 2: Feature engineering
    logger.info("Engineering features...")
    features_df = engineer_features(df)

    # Step 3: Temporal split
    train_df, val_df, test_df = temporal_split(features_df)

    min_train = 20 if args.n_cases <= 300 else 50  # Relaxed for sample runs
    min_split = 5 if args.n_cases <= 300 else 10
    if len(train_df) < min_train or len(val_df) < min_split or len(test_df) < min_split:
        logger.error("Insufficient data for training: train=%d, val=%d, test=%d (min: %d/%d/%d)",
                      len(train_df), len(val_df), len(test_df), min_train, min_split, min_split)
        logger.error("Try --synthetic for pipeline testing, or scrape more data.")
        sys.exit(1)

    # Step 4: Train all models
    logger.info("Training models...")
    results = train_all_models(train_df, val_df, test_df, n_trials=args.n_trials)

    # Step 5: Report
    report = report_results(results, test_df)
    print(report)

    # Save report
    report_path = settings.data.processed_dir / "training_report.txt"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report)
    logger.info("Report saved to %s", report_path)


if __name__ == "__main__":
    main()
