# Experiment Plan: AI-Assisted Litigation Prediction for Kenyan Courts

## 1. Ideation & Problem Statement

### Why This Experiment

Senior advocates in Kenya spend 3-5 hours per case researching past judgments. Outcome forecasts are highly variable, with no data-driven support. This project builds an AI prediction tool that:

1. Improves case outcome forecast accuracy over unassisted predictions
2. Reduces research time by >= 30%
3. Provides transparent explanations (judge proclivity, counsel track record, statutory citations)
4. Is specifically designed for **Kenyan commercial litigation lawyers** as the primary user

### Literature-Grounded Accuracy Expectations

Prediction accuracy varies significantly by case type and jurisdiction. The following benchmarks from peer-reviewed studies inform our realistic targets:

| Study | Jurisdiction | Case Type | Accuracy | Method |
|-------|-------------|-----------|----------|--------|
| Aletras et al. 2016 (PeerJ CS) | ECHR | Human rights (Art 3,6,8) | **79%** avg | SVM + N-grams + topics |
| Katz et al. 2017 (PLOS ONE) | US Supreme Court | All case types | **70.2%** case-level | Random Forest |
| PILOT (NAACL 2024) | ECHR | Multi-label violation | **71.5%** F1, **0.83** AUC | BERT + temporal |
| Brazilian appeals (PMC 2022) | Brazilian federal | Appeals (affirm/reverse) | MCC **0.37** (beats experts at 0.13) | ULMFiT |
| MambaEffNet 2024 | Nigerian Supreme Court | Criminal + civil | **92%** | Deep learning |
| JES 2024 (ResearchGate) | US Supreme Court | All case types | **72%** XGBoost, 61% NB, 52% DT | XGBoost, NB, DT, SVM, RF, k-NN |
| LexEdge 2024 (industry) | General | Contract disputes | **85-92%** | Not specified |
| LexEdge 2024 (industry) | General | Commercial litigation | **80-87%** | Not specified |
| LexEdge 2024 (industry) | General | Patent litigation | **78-85%** | Not specified |
| Pre/Dicta (industry) | US Federal | Motions to dismiss | **85-90%** | ML on 15M cases |
| Human experts (Katz 2017) | US Supreme Court | All | **~66%** | Legal expertise |
| Human experts (Brazil 2022) | Brazilian federal | Appeals | MCC **0.13** | 22 judges/clerks |

**Key findings that shape our approach:**

1. **Accuracy varies by case type**: Contract disputes (85-92%) >> Patent cases (78-85%). Our commercial litigation target should expect **75-85%** at maturity, **65-75%** in pilot.
2. **ML consistently beats human experts**: Brazilian study (MCC 0.37 vs 0.13), US SCOTUS (~70% vs ~66%). Even modest ML improvement is meaningful.
3. **Facts matter more than law**: Aletras found "circumstances" (factual description) were the strongest predictor. Pure "law" section was the weakest. This supports using full judgment text features.
4. **Temporal shift degrades accuracy**: PILOT showed accuracy drops from 80% to 68% when using chronological (realistic) splits vs random splits. Our temporal split is essential.
5. **Counsel characteristics are predictive**: LexEdge and Pre/Dicta both identify attorney/counsel as a significant feature. Pre/Dicta uses 50-100 data points per case including attorney information.
6. **Small datasets can work**: Aletras achieved 79% with only 584 cases. Our target of 400-800 cases is within the proven range.
7. **No African legal ML studies exist at scale**: The Nigerian MambaEffNet study is the only African court ML paper. We are building novel ground for Kenyan courts.

### Revised Hypothesis

> "An ML model trained on historical Milimani High Court commercial decisions (2015-2023) achieves binary outcome prediction accuracy of 70% or higher on held-out test data (2022-2023), and when deployed as a decision-support tool, improves Kenyan advocate prediction accuracy by at least 8 percentage points over their unassisted baseline."

**Why 70% and 8pp:**
- 70% offline accuracy aligns with Katz (70.2% on SCOTUS), Aletras (79% on ECHR), and is conservative for commercial cases (industry reports 80-87%)
- 8pp lift over lawyer baseline mirrors Brazilian finding where ML significantly outperformed experts, and aligns with ECHR temporal-split reality (~68%)
- Previous "10pp lift" was changed to 8pp because literature shows the realistic range of human-expert accuracy in common-law systems is 60-70%, and a lift from 65% to 73% (8pp) is both achievable and practically significant

## 2. Workflow: End-to-End Data Pipeline

### Phase 0: Data Discovery (Weeks 1-3)
**Why:** Validate what actually exists before building models. Literature shows data availability is the #1 risk in legal ML (Zeleznikow 2023).

```
Tausi API --> Count cases by year/court --> Audit field completeness
          --> Check PDF availability --> Sample 30 cases for outcome taxonomy
          --> Check advocate data coverage (API + PDF extraction)
```

**Gate:** >= 300 cases with published judgments and extractable outcomes.

### Phase 1: Data Collection & Labeling (Weeks 3-6)
**Why:** Outcome label quality determines everything. Aletras 2016 showed that factual/circumstantial text predicts better than legal text, so full-text extraction is critical.

```
Tausi API --> Download JSON metadata (all KEHC/HCNRB 2015-2023)
          --> Download PDF judgments
          --> Extract text (PyMuPDF -> pdfminer -> OCR fallback)
          --> Extract advocate names from text headers + API advocates[]
          --> Run outcome parser (rule-based, confidence-scored)
          --> Domain expert validates 100+ labels (Cohen's kappa > 0.80)
```

**Data needed for TRAINING:** 2015-2020 cases (~60% of dataset)
**Data needed for VALIDATION:** 2021 cases (~15%)
**Data needed for TESTING:** 2022-2023 cases (~25%)

**Why temporal split (not random):** PILOT (NAACL 2024) demonstrated accuracy drops from 80% to 68% with chronological vs random splits. Legal ML must respect temporal ordering because judges rotate, laws change, and precedent evolves.

### Phase 2: Feature Engineering (Weeks 6-9)
**Why each feature category exists (literature-grounded):**

| Feature Category | Why It Matters | Literature Support |
|-----------------|---------------|-------------------|
| **Judge win rate** | Judicial behavior patterns are among the strongest predictors | Katz 2017: judge-level features in RF; Pre/Dicta: judge modeling |
| **Lawyer win rates** | Counsel quality/experience is a top predictor | LexEdge 2024; Pre/Dicta uses 50-100 data points inc. attorneys |
| **Lawyer win rate diff** | Relative strength of opposing counsel predicts outcome direction | Pre/Dicta: comparative analysis between sides |
| **Case facts (text)** | Circumstances/facts outperform legal reasoning text | Aletras 2016: "circumstances" >> "law" section accuracy |
| **Claim amount** | Case economic magnitude affects outcomes | Pre/Dicta: party/economic characteristics matter |
| **Citation patterns** | Statutory references signal legal framework | Aletras 2016: topic modeling on legal references |
| **Temporal features** | Post-DCSM (2018+) and seasonal patterns | PILOT 2024: temporal handling improves prediction |

### Phase 3: Model Development (Weeks 9-13)
**Why this progression:**

1. **Majority class** -> Floor. If dataset is 60/40, majority class = 60%.
2. **Logistic Regression** -> Simple, interpretable baseline. Zeleznikow 2023 advocates starting simple.
3. **Random Forest** -> Added because Katz 2017 achieved 70.2% on SCOTUS with RF and it handles mixed feature types well. JES 2024 confirms RF is a competitive baseline for legal prediction tasks.
4. **XGBoost (tabular)** -> Primary candidate. Strongest for structured features at this scale. JES 2024 independently confirmed XGBoost as top performer (72%) over NB (61%), DT (52%) on SCOTUS data.
5. **XGBoost + TF-IDF text features** -> Aletras showed text adds ~3-5pp. TF-IDF on judgment text captures legal vocabulary signals. Park et al. (2021) survey confirms supervised learning on text is the dominant approach in legal tech research.

**Why TF-IDF over embeddings:** With ~300-500 labeled cases, dense embeddings risk overfitting. TF-IDF with SVD dimensionality reduction is more stable at this scale and provides interpretable legal keyword features.

**Why NOT deep learning:** Our dataset is 400-800 cases. Deep learning (BERT fine-tuning) needs thousands. Brazilian study used 765K cases. Nigerian study had large corpora. Park et al. (2021) note that ANNs are widely used but primarily in large-dataset contexts. We don't have that scale.

### Phase 4: API Deployment (Weeks 13-16)
FastAPI service with /predict endpoint, SHAP explanations, mandatory legal disclaimer. JSON logging for pilot scale (not Prometheus/Grafana).

### Phase 5: Shadow Mode (Weeks 16-20)
**Why shadow mode before A/B:** Must empirically measure the lawyer baseline. The assumed "55%" in the original framework was unsubstantiated. Brazilian study showed experts had MCC=0.13 (barely above random). Katz showed ~66% for SCOTUS experts. We need the real Kenyan number.

```
5-10 lawyers --> Record predictions (Google Forms) --> No model shown
Model --> Independently predicts same cases --> Compare when outcomes known
```

**This produces:** Empirical lawyer baseline accuracy for Kenyan commercial cases.

### Phase 6: A/B Test (Weeks 20-30+)
**Why stratification matters:** Different case types have different accuracy ceilings (LexEdge: contracts 85-92% vs patent 78-85%). Stratify by case type, judge, and claim bucket.

## 3. Assumptions & Rationale

| Assumption | Rationale | Risk if Wrong |
|-----------|-----------|---------------|
| 400+ commercial cases available (2015-2023) | KEHC processes thousands yearly; Milimani is Kenya's busiest court | Broaden to all KEHC or reduce model scope |
| Advocate names extractable from PDF text | Kenyan judgments consistently list advocates in header section | Fall back to API advocates[] only; ~30-40% coverage |
| Lawyer win rates are predictive | LexEdge 2024, Pre/Dicta use counsel as top feature | Model still works on judge + case features alone |
| Binary outcome extractable from judgment text | Rule-based parser with 12 outcome patterns | Domain expert reviews all low-confidence labels |
| Temporal split adequate for generalization | PILOT 2024 shows temporal > random for legal ML | Include drift monitoring in deployment |
| 70% accuracy achievable at pilot | Aletras (79%), Katz (70.2%) with similar/smaller datasets | Lower target to 65%; focus on time-saving metric |
| Lawyers baseline is 60-70% | Katz (66%), Brazilian experts (MCC 0.13) | Shadow mode will measure the actual number |

## 4. Data Used for the Experiment

### Data Source
- **Source:** Web-scraped from new.kenyalaw.org (National Council for Law Reporting)
- **Note:** Laws.Africa Tausi API token lacks Kenya content permissions; direct scraping used instead
- **Scraper:** `src/data/scraper.py` — KenyaLawScraper class, handles two page layouts (akn-judgmentBody and document_content divs)

### Initial Scrape Results (741 cases, all KEHC 2015-2023)
| Court Division | Count | Included |
|---|---|---|
| Civil | 266 | Yes |
| Criminal | 263 | No (different outcome dynamics) |
| Family | 106 | No (different domain) |
| Commercial and Tax | 45 | Yes |
| Judicial Review | 29 | Yes |
| Constitutional | 20 | Yes |
| Other | 12 | Case-by-case |

**Filtering:** Criminal and Family cases are excluded for the commercial/civil prediction model. This reduces noise from fundamentally different case types (conviction/acquittal vs plaintiff/defendant outcomes).

### Outcome Labeling
- **Metadata "Outcome" field:** ~40% of cases have outcome in structured metadata
- **Text-based extraction:** Additional ~20% extracted from judgment tail using regex patterns
- **Combined labeling rate:** ~60% of scraped cases receive binary labels
- **Outcome distribution (labeled):** ~53% Allowed, ~42% Dismissed, ~5% Other

### Training Data (2015-2020)
- **Size:** ~200-300 labeled civil/commercial cases
- **Content:** Full judgment text + structured metadata
- **Features:** Judge, lawyer, case type, citations, temporal, text (TF-IDF)
- **Labels:** Rule-extracted outcome from metadata + judgment text

### Validation Data (2021)
- **Purpose:** Model selection, hyperparameter tuning, calibration
- **Size:** ~40-60 labeled cases

### Test Data (2022-2023) - HELD OUT
- **Purpose:** Final one-time evaluation. Touched ONCE.
- **Size:** ~80-130 labeled cases
- **Integrity:** No information from test set leaks into training (temporal barrier)

### Prospective Data (2024+) - for Shadow Mode and A/B Test
- **Source:** New cases as they are decided
- **Purpose:** Real-world validation with lawyer predictions
- **Size:** Target 50 cases for shadow mode, 300+ per arm for A/B test

## 5. Evaluation Gates

| Gate | Metric | Threshold | Source |
|------|--------|-----------|--------|
| Offline accuracy | Test accuracy | >= 70% | Katz 2017 (70.2% on SCOTUS) |
| Discrimination | AUC-ROC | >= 0.70 | PILOT 2024 (0.83 on ECHR) |
| Calibration | Brier score | <= 0.22 | Standard for well-calibrated models |
| Fairness | Accuracy delta (judge gender) | <= 5pp | Project constraint |
| Shadow mode | Model vs lawyer accuracy | >= 5pp lift | Conservative; Brazilian showed large gap |

## 6. Metrics

| Metric | Type | Target | Why |
|--------|------|--------|-----|
| Binary accuracy | Primary | >= 70% test, >= 8pp lift over lawyers | Literature-aligned (Katz 70%, Aletras 79%) |
| AUC-ROC | Primary | >= 0.70 | Standard discrimination metric |
| Brier score | Secondary | <= 0.22 | Calibration quality for lawyer trust |
| Research time reduction | Secondary | >= 30% | Business goal |
| User trust (% Accept) | Secondary | > 60% | Adoption metric |
| Precision at 0.5 | Diagnostic | Report | Error analysis |
| Per-case-type accuracy | Diagnostic | Report | LexEdge shows type matters |
| Lawyer feature importance | Diagnostic | Report | Validates counsel-as-predictor hypothesis |

## 7. Statistical Analysis Plan

1. **Two-proportion Z-test** for accuracy lift (primary)
2. **Independent t-test** for time-saving (secondary)
3. **Mixed-effects logistic regression**: `Outcome ~ Treatment + JudgeID + CaseType + ClaimBucket + (1|LawyerID)`
4. **SHAP analysis** to understand which features (especially lawyer features) drive predictions
5. **Per-case-type breakdown** to identify where model helps most vs least

## 8. Roll-out Criteria

| Decision | Condition |
|----------|-----------|
| GREEN (deploy) | Test accuracy >= 70% AND lift >= 8pp AND time saved >= 30% |
| YELLOW (extend pilot) | One metric passes, others marginal |
| RED (iterate) | No uplift or bias detected - iterate on features/model |

## 9. References

1. Aletras N, Tsarapatsanis D, Preotiuc-Pietro D, Lampos V. (2016). "Predicting judicial decisions of the ECHR: a NLP perspective." PeerJ Computer Science 2:e93.
2. Katz DM, Bommarito MJ II, Blackman J. (2017). "A general approach for predicting the behavior of the Supreme Court." PLOS ONE 12(4).
3. Cao L, Wang Z, Xiao C, Sun J. (2024). "PILOT: Legal Case Outcome Prediction with Case Law." NAACL 2024.
4. Lage-Freitas A, et al. (2022). "Using deep learning to predict outcomes of legal appeals better than human experts." PMC/PLOS ONE.
5. Zeleznikow J. (2023). "Benefits and dangers of using ML to support legal predictions." WIREs Data Mining.
6. LexEdge. (2024). "Machine Learning for Case Outcome Prediction."
7. ABA. (2024). "Using AI for Predictive Analytics in Litigation."
8. Pre/Dicta. (2024). Litigation prediction platform documentation.
9. Dina NZ, Ravana SD, Idris N. (2025). "Legal Judgment Prediction using NLP and ML: A Systematic Literature Review." SAGE Open.
10. Predictive Modelling in Legal Decision-Making. (2024). "Leveraging Machine Learning for Forecasting Legal Outcomes." Journal of Electrical Systems. [XGBoost=72%, NB=61%, DT=52% on SCOTUS; confirms XGBoost as top performer for legal prediction.]
11. Park S, et al. (2021). "A Survey of Research on Data Analytics-Based Legal Tech." Sustainability 13(14):8085. [Supervised learning dominant; ANNs widely used; preprocessing legal text is critical.]
12. GAP Interdisciplinarities. (2025). "AI in Legal Research and Case Prediction: Transforming the Future of Law." [AI as assistive tool not replacement; ethical/bias concerns; LexisNexis/Westlaw precedent.]
