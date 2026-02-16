# Model Card: Litigation Outcome Predictor

## Model Description

Binary classifier predicting the outcome of Commercial Division cases at the
Milimani High Court, Nairobi, Kenya. The model predicts `P(plaintiff/applicant wins)`
as a calibrated probability.

**Model type:** Ensemble of gradient-boosted and tree-based models, with model selection
based on validation performance. Model progression (literature-grounded):
1. Majority class (floor)
2. Logistic Regression (L2-regularized, minimum viable baseline)
3. Random Forest (Katz 2017: 70.2% SCOTUS; JES 2024: competitive legal prediction)
4. XGBoost (primary candidate - JES 2024: 72% best performer on SCOTUS)
5. XGBoost + text embeddings (only if text adds measurable value over tabular-only)

**Architecture:** Tabular features (including lawyer/counsel features) + (optional) 30 PCA
components from `all-MiniLM-L6-v2` embeddings, fed into XGBoost/RF with Platt-calibrated outputs.

## Training Data

- **Source:** Web-scraped from new.kenyalaw.org (National Council for Law Reporting)
- **Scope:** KEHC civil and commercial decisions, 2015-2023 (741 scraped, filtered to civil/commercial)
- **Split:** Temporal: Train 2015-2020, Validation 2021, Test 2022-2023
- **Target:** Binary outcome (1=plaintiff/applicant success, 0=defendant/respondent success)
- **Exclusions:** Criminal and Family cases, consent orders, withdrawn cases
- **Labeling:** ~60% of cases labeled via metadata "Outcome" field + text-based extraction from judgment tail

## Features

Key features (see `docs/DATA_DICTIONARY.md` for full list):

**Counsel features** (Literature: LexEdge 2024, Pre/Dicta — counsel is among the strongest predictors):
- Plaintiff/defendant lawyer win rates (Bayesian-smoothed)
- Lawyer win rate differential (relative strength of opposing counsel)
- Lawyer experience proxy (years active)
- Number of advocates per side
- Whether advocate data is available

**Judge features:**
- Judge historical win rate (Bayesian-smoothed)

**Case features:**
- Filing year and case age
- Number of statute citations
- Whether Data Protection Act is cited
- Claim amount (log-transformed)
- Post-2018 DCSM indicator

**Text features** (optional, only if they improve over tabular):
- Judgment text embeddings (PCA-reduced sentence-transformer)
- TF-IDF on legal keyword vocabulary

## Performance

### Baseline Run (741 unfiltered cases, 50 Optuna trials)

| Model | Test Acc | Test AUC | Test Brier |
|---|---|---|---|
| Majority class | 0.408 | 0.500 | 0.255 |
| Logistic Regression | 0.585 | 0.634 | 0.237 |
| Random Forest | 0.577 | 0.626 | 0.240 |
| **XGBoost** | **0.569** | **0.641** | **0.236** |

**Notes:** Baseline run used all case types (including criminal/family), minimal advocate data (1.5%), no text features. Gates not met — expected as this was pre-filtering.

## Fairness

- Accuracy delta across judge gender groups must be <= 5 percentage points
- Accuracy delta across case years must be reported
- SHAP feature importance must not show problematic proxies

## Limitations

1. **Kenyan courts only** - trained exclusively on KEHC/HCNRB data, not generalizable to other jurisdictions
2. **Historical data bias** - model reflects patterns from 2015-2023; judicial appointments, law changes, and precedent evolution may cause drift
3. **Outcome taxonomy** - outcome labels are rule-extracted from judgment text, not ground truth; ~10-15% may need manual correction
4. **Small dataset** - with ~300-500 labeled training cases, overfitting risk is real; model confidence intervals are wide
5. **Advocate data incomplete** - advocate names extracted from metadata "Attorneys" field (~22%) + text extraction (~2% sided); most names are unsided (not classified as plaintiff/defendant)
6. **Not legal advice** - predictions are for research/decision-support only; AI is an assistive tool, not a replacement for legal judgment (GAP 2025)

## Ethical Considerations

- Model predictions must never replace legal judgment
- Risk of anchoring bias if lawyers over-rely on model predictions
- Potential for gaming if parties learn what features the model uses
- Privacy: anonymized cases must remain anonymized in model outputs

## Maintenance

- Retrain quarterly with new decisions
- Monitor for distribution drift (judge turnover, new legislation)
- Track calibration quality on recent predictions
- Domain expert review of misclassified cases
