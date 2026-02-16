
You are a senior Machine‑Learning Engineer, product analyst, and legal‑tech consultant tasked with delivering a **ready‑to‑run Git repository** that implements the entire litigation‑analytics experiment described below.

---
## CONTEXT & BUSINESS PROBLEM
- **Domain:** Kenyan courts (High Court, Court of Appeal, Supreme Court).
- **Use‑case:** Predict the outcome of a **Data Protection Act** dispute or a **commercial injunction** (win/loss, settlement probability, expected monetary award).
- **Current pain:** Senior advocates spend **3–5 hours** per case researching past judgments; forecasts are highly variable.
- **Goal:** Build a data‑driven tool that:
  1. Improves forecast accuracy by **≥ 20 %** over senior‑advocate‑only predictions.
  2. Cuts research time by **≥ 30 %**.
  3. Provides transparent driver explanations (judge proclivity, statutory citations, claim size).

---
## EXPERIMENT DESIGN (end‑to‑end)
1. **Hypothesis**
   *“Providing AI‑assisted predictions for a Motion for Injunction in the Milimani Commercial Court will increase the binary‑win accuracy by at least 20 % compared with manual senior‑advocate forecasts.”*

2. **Effect size & Power**
   - Baseline accuracy (manual) = 55 %.
   - Target accuracy (with model) = ≥ 66 % (≥ 20 % relative lift).
   - Two‑proportion Z‑test, α = 0.05, power = 0.80 → **≈ 308 cases per arm**.
   - We will collect **≥ 650 cases** (to cover drop‑outs).

3. **Stratified Randomisation** (to balance confounders)
   - **Strata:**
     - Case‑type = C02 (Commercial Contract).
     - Judge (top‑5 most‑active judges in Milimani).
     - Claim‑bucket (Low < 5 M KES, Mid 5‑20 M KES, High > 20 M KES).
   - Within each stratum assign cases 1:1 to **Control** (manual only) or **Treatment** (ML assistance).

4. **Workflow**
   a. **Data extraction** – pull all judgments (2015‑2023) from the **Tausi API** (Kenya Law).
   b. **Baseline generation** – senior counsel reads the case, records prediction + time spent (Google‑Form).
   c. **Model inference** – the ML service returns `{prob_win, prob_settle, expected_award, top_features}`.
   d. **Decision logging** – user either **Accept** the model suggestion or **Override**; both the final decision and the user action are stored.
   e. **Outcome capture** – when the real judgment is published, the actual outcome is fetched automatically from the API and compared to the predicted label.

5. **Metrics**
   - **Primary:** Binary win‑accuracy (Treatment vs. Control).
   - **Secondary:** Precision@0.5, settlement‑probability AUC, RMSE of award prediction, research‑time reduction, user‑trust (% Accept).
   - **Fairness:** Δ win‑rate between male/female judges, Δ across counties (must be ≤ 5 pp).

6. **Statistical analysis**
   - Two‑proportion Z‑test for accuracy lift (p < 0.05).
   - Paired t‑test for time‑saving.
   - Mixed‑effects logistic regression:
     `Outcome ~ Treatment + JudgeID + ClaimBucket + (1|LawyerID)`.
   - SHAP explanations to understand where the model helped vs. hurt.

7. **Roll‑out criteria**
   - **Green:** Accuracy uplift ≥ 20 % **and** time saved ≥ 30 % (both p < 0.05).
   - **Yellow:** One metric passes, the other marginal – run a second pilot with more data.
   - **Red:** No uplift or bias detected – abort, revisit features.

---
## DATA REQUIREMENTS & DICTIONARY
All raw data will live under `data/raw/<year>/`.  The ETL script (`src/etl/download_data.py`) must pull **both** the structured JSON metadata **and** the PDF judgment for every decision filed between **2015 – 2023**.

| Field (canonical) | Type | Source | Description | Transformation |
|-------------------|------|--------|-------------|-----------------|
| `case_id` | string | `short_mnc` / `full_mnc` (JSON) | Unique case identifier (e.g., `KEHC2023_00123`) | Upper‑case, strip spaces |
| `court_code` | string | `court.code` | Court identifier (e.g., `HCNRB`) | Map to human‑readable name via `/courts` endpoint |
| `filing_year` | int | `filing_year` | Year of filing | Direct |
| `judgment_date` | date | `judgment_date` | Date judgment rendered | ISO‑8601 |
| `case_type_code` | string | `casetype.name` | Taxonomy C01‑C09 (see `docs/DATA_DICTIONARY.md`) | Map to description |
| `claim_amount_kes` | float | Regex on PDF text (`KES\s*[\d,]+`) | Monetary claim in Kenyan Shillings | Remove commas, cast to float |
| `claim_amount_usd` | float | Derived | Claim amount in USD (1 USD ≈ 140 KES) | `claim_amount_kes / 140` |
| `outcome_label` | enum | Parsed from `action.name` & textual cues | `WIN_PLAINTIFF`, `WIN_DEFENDANT`, `SETTLEMENT`, `DISMISSAL` | Rules in `src/features/build_features.py` |
| `outcome_binary` | int (0/1) | Derived from `outcome_label` | 1 = plaintiff (or prosecution) win, 0 = loss (partial win = 0.5) | Mapping table |
| `judge_ids` | array[string] | `judges[].short_name` | Judges on the panel (primary first) | Keep as list |
| `judge_win_rate` | float | Computed from historic decisions (≥ 30 cases) | Historical win‑rate for primary judge | Updated nightly |
| `lawyer_ids` | array[string] | `advocates[].short_name` | Counsel for both sides | N/A |
| `lawyer_seniority` | int | `AdvocateRegistry.bar_year` | Years since bar admission (2024 – bar_year) | Derived on load |
| `is_senior_counsel` | bool | `AdvocateRegistry.is_senior` | “Advocate of the Supreme Court” flag | N/A |
| `statutes_cited` | array[string] | `cited_documents[].title` (JSON) – fallback regex on PDF | List of statutes (e.g., `Data Protection Act – Sec 25`) | Normalise to `Act_ID:Section` |
| `num_cites` | int | `len(statutes_cited)` | Number of statutory citations | N/A |
| `top_cited_act` | string | Most frequent act in `statutes_cited` | E.g., `Data Protection Act` | N/A |
| `temporal_month` | int | `judgment_date.month` | Month of decision | N/A |
| `post_dcsm` | bool | `filing_year` ≥ 2018 | Indicator of Digital Case Management System rollout | N/A |
| `text_path` | string | Local FS path to PDF | Used for text extraction (Legal‑BERT) | N/A |
| `full_text` | string | Extracted from PDF (pdfminer / PyMuPDF) | Plain‑text of judgment | Stored in processed table for embedding |

All transformations must be recorded in a **Feast feature store** to guarantee reproducibility.

---
## REPO STRUCTURE & FILE CONTENTS
Your answer **must output** each file as a separate fenced code block.
The first line of every block should be the **relative file path**, e.g.:

```markdown
# docs/README.md
<file content>
```

Do **NOT** include any extra commentary outside the blocks.
At the very end, output a plain‑text **directory tree** that exactly matches the repository layout.

### Files to generate (with required content)

| Path | What to include |
|------|-----------------|
| `docs/README.md` | High‑level project overview, quick‑start instructions, next steps. |
| `docs/PRD.md` | Vision, personas, user stories, success metrics, scope, acceptance criteria (as written above). |
| `docs/DATA_DICTIONARY.md` | Full table shown above (the canonical fields, source, description, transformation). |
| `docs/EXPERIMENT_PLAN.md` | Detailed A/B experiment protocol (hypothesis, power analysis, stratification, workflow, metrics, statistical plan, roll‑out criteria). |
| `docs/MODEL_CARD.md` | Model description, training data summary, performance (overall & slice‑wise), fairness, limitations, maintenance. |
| `docs/RISK_REGISTER.md` | Table of risks (privacy, bias, legal liability, API limits, drift, outage) with likelihood, impact, mitigation, owner. |
| `docs/DEPLOYMENT_PLAYBOOK.md` | Architecture diagram (text), CI/CD steps (GitHub Actions), secrets handling, monitoring (Prometheus + Grafana), rollback, DR plan. |
| `src/etl/download_data.py` | Complete Python script that:  <br>• reads `TAUSI_TOKEN` from env, <br>• pages `/api/v1/decisions`, <br>• filters `filing_year` 2015‑2023, <br>• saves JSON to `data/raw/<year>/<case_id>.json`, <br>• builds PDF URL from FRBR URI, downloads PDF to same folder, <br>• respects rate‑limit (≤ 5 req/s), exponential back‑off, tqdm progress, logging. |
| `src/features/build_features.py` | Placeholder with a top‑level comment explaining: load JSON, extract fields defined in `DATA_DICTIONARY`, compute `judge_win_rate`, `lawyer_win_rate`, generate Legal‑BERT embeddings from PDFs, store feature table (`data/processed/cases.feather`). |
| `src/models/train.py` | Placeholder with comment: split into train/val/test (temporal split), build XGBoost‑BERT fusion model, hyper‑parameter search (Optuna), calibrate (Platt), log to MLflow, save model artifact. |
| `src/api/app.py` | Minimal FastAPI app exposing `/predict` (loads the latest model from MLflow, reads a JSON payload with case features, returns prediction + top‑5 SHAP drivers). |
| `requirements.txt` | List of all Python dependencies needed for ETL, feature engineering, modeling, FastAPI, monitoring. |
| `.gitignore` | Common ignores (`__pycache__/`, `*.pyc`, `data/raw/`, `data/processed/`, `.env`, `.venv/`). |
| `directory_tree.txt` (or just plain text at the end) | Exact folder layout (see below). |

---
## SPECIFIC CONTENT GUIDELINES

1. **Markdown files:** Use GitHub‑flavoured markdown tables, headings (`#`, `##`), and bullet lists.
2. **Python scripts:** Use `#!/usr/bin/env python` shebang, proper imports, `if __name__ == "__main__":` entry point, and thorough inline comments.
3. **Requirements:** Include at least: `pandas`, `numpy`, `requests`, `tqdm`, `feast[postgres]`, `scikit-learn`, `xgboost`, `optuna`, `mlflow`, `fastapi`, `uvicorn[standard]`, `python-dotenv`, `pdfminer.six`, `PyMuPDF`, `transformers`, `torch`, `shap`, `prometheus-client`.
4. **Directory tree:** Use the exact structure shown at the end of this prompt.

---
## FINAL OUTPUT FORMAT

Your response **must** consist of a series of fenced code blocks, one for each file listed above, in any order you prefer.
Each block starts with a comment line that is the **relative file path** (e.g. `# docs/README.md`).
After all file blocks, output a plain‑text directory tree (no markdown fences).

When this answer is copied into a local folder, the repo should be instantly buildable, and the ETL script should be runnable after setting `TAUSI_TOKEN`.

---
**END OF PROMPT**. Please generate the files accordingly.
```

---

**Directory tree (paste this after all code blocks):**
```text
litigation-analytics-kenya/
├─ data/
│  ├─ raw/
│  └─ processed/
├─ docs/
│  ├─ README.md
│  ├─ PRD.md
│  ├─ DATA_DICTIONARY.md
│  ├─ EXPERIMENT_PLAN.md
│  ├─ MODEL_CARD.md
│  ├─ RISK_REGISTER.md
│  └─ DEPLOYMENT_PLAYBOOK.md
├─ src/
│  ├─ etl/
│  │  └─ download_data.py
│  ├─ features/
│  │  └─ build_features.py
│  ├─ models/
│  │  └─ train.py
│  └─ api/
│     └─ app.py
├─ requirements.txt
└─ .gitignore
```

