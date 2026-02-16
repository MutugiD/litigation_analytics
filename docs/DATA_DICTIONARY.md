# Data Dictionary

## Source: Kenya Law (new.kenyalaw.org)

All case data is web-scraped from the Kenya Law website maintained by the National Council for Law Reporting of Kenya. The scraper (`src/data/scraper.py`) handles two page layout variants and extracts metadata from dt/dd HTML elements plus judgment text from the content div.

## Canonical Fields

| Field | Type | Source | Description | Transformation |
|-------|------|--------|-------------|----------------|
| `case_id` | string | FRBR URI | Unique identifier (e.g., `KEHC_2020_123`) | Derived from `/akn/ke/judgment/{court}/{year}/{number}` |
| `frbr_uri` | string | API JSON | Full FRBR URI for the decision | Direct |
| `court_code` | string | `court.code` | Court identifier (e.g., `kehc`) | Direct from API |
| `filing_year` | int | `filing_year` | Year of filing | Direct |
| `judgment_date` | date | `judgment_date` | Date judgment rendered | ISO-8601 |
| `case_type` | string | `casetype.name` | Case type description from API | Direct (not C01-C09) |
| `judge_names` | array[str] | `judges[].short_name` | Judges on the panel | Direct from API |
| `primary_judge` | string | First in `judges[]` | Primary judge | Derived |
| `num_judges` | int | `len(judges)` | Number of judges | Computed |
| `advocate_names` | array[str] | `advocates[].short_name` | Counsel (may be empty) | Direct; **low completeness** |
| `cited_statutes` | array[str] | `cited_documents[].title` | Statutes cited | Direct; fallback regex on PDF |
| `num_citations` | int | `len(cited_statutes)` | Count of statute citations | Computed |
| `cites_dpa` | bool | Text search | Whether case cites Data Protection Act | Text matching |
| `has_pdf` | bool | Download check | Whether PDF judgment was downloaded | Binary |
| `has_text` | bool | Extraction check | Whether text was extracted from PDF | Binary |
| `text_length` | int | `len(full_text)` | Character count of judgment | Computed |
| `outcome_label` | enum | Parsed from judgment text | See Outcome Taxonomy below | Rule-based extraction |
| `outcome_confidence` | float | Parser output | Confidence in outcome extraction (0-1) | Computed |
| `outcome_binary` | int/null | Derived | 1=positive, 0=negative, null=excluded | Mapping table |
| `claim_amount_kes` | float/null | Multi-regex on PDF | Monetary claim in KES | Multi-pattern extraction |
| `has_monetary_claim` | bool | Derived | Whether any monetary amount found | Binary |

## Outcome Taxonomy

Built from analysis of actual Kenyan High Court judgments. See `configs/outcomes.yaml` for patterns.

| Outcome | Binary | Description |
|---------|--------|-------------|
| `APPLICATION_ALLOWED` | 1 | Interlocutory application granted |
| `JUDGMENT_FOR_PLAINTIFF` | 1 | Substantive judgment for plaintiff |
| `APPEAL_ALLOWED` | 1 | Appeal upheld |
| `APPLICATION_DISMISSED` | 0 | Application refused |
| `JUDGMENT_FOR_DEFENDANT` | 0 | Substantive judgment for defendant |
| `SUIT_DISMISSED` | 0 | Case dismissed (want of prosecution, etc.) |
| `STRUCK_OUT` | 0 | Case struck off the register |
| `APPEAL_DISMISSED` | 0 | Appeal rejected |
| `CONSENT_ORDER` | excluded | Settlement by consent |
| `WITHDRAWN` | excluded | Case withdrawn |
| `PARTIAL` | excluded | Partial success (ambiguous binary) |
| `UNDETERMINED` | excluded | Could not be extracted |

## Engineered Features

### Judge Features
| Feature | Type | Description |
|---------|------|-------------|
| `judge_win_rate` | float | Historical win rate (Bayesian-smoothed, min 15 cases) |

### Lawyer/Advocate Features (Literature: LexEdge 2024, Pre/Dicta — counsel is a top predictor)
| Feature | Type | Description |
|---------|------|-------------|
| `plaintiff_lawyer_win_rate` | float | Best plaintiff advocate's Bayesian-smoothed historical win rate |
| `defendant_lawyer_win_rate` | float | Best defendant advocate's Bayesian-smoothed historical win rate |
| `lawyer_win_rate_diff` | float | plaintiff - defendant win rate (positive = plaintiff has stronger counsel) |
| `plaintiff_lawyer_experience` | int | Max years active among plaintiff advocates (proxy for seniority) |
| `defendant_lawyer_experience` | int | Max years active among defendant advocates |
| `num_plaintiff_advocates` | int | Number of plaintiff-side advocates |
| `num_defendant_advocates` | int | Number of defendant-side advocates |
| `has_advocate_data` | int | Whether any advocate information is available (1/0) |

### Temporal & Case Features
| Feature | Type | Description |
|---------|------|-------------|
| `case_age_days` | int | Days between filing and judgment |
| `month_sin`, `month_cos` | float | Cyclical month encoding |
| `is_post_2018` | int | Post-DCSM rollout indicator |
| `log_claim_amount` | float | log(1 + claim_amount_kes) |

### Text Features
| Feature | Type | Description |
|---------|------|-------------|
| `emb_pca_0..29` | float | PCA-reduced sentence-transformer embeddings |
| `tfidf_*` | float | TF-IDF on legal keyword vocabulary |

## Advocate Data Pipeline

Lawyer/advocate data is sourced from **three channels** (literature supports counsel as a top predictor):

1. **Metadata "Attorneys" field** — Available on ~22% of case pages; provides names but not party sides
2. **Judgment text extraction** — Regex patterns targeting Kenyan judgment phrasing:
   - "Mr./Ms. Name for the Plaintiff/Defendant"
   - "Name Advocates for the Plaintiff"
   - "Learned Counsel for the Plaintiff, Mr. Name"
   - Searches first 15% of judgment text (header/introduction section)
3. **Improved text extraction (v2)** — Extended patterns targeting additional Kenyan judgment phrasing:
   - "Name for the [1st/2nd] Plaintiff/Defendant"
   - "Name instructed by / holding brief for"
   - "Name, Advocate" with role inference from context

**Side assignment:** When advocates are listed in the "Attorneys" metadata without side information, the system infers sides from judgment text context where possible. Advocates "for the Plaintiff/Applicant" are assigned as plaintiff-side; "for the Defendant/Respondent" as defendant-side.

Win rates use **Bayesian shrinkage** toward the global mean (Beta-Binomial model), so lawyers with few cases are regularized. Plaintiff advocates "win" when `outcome_binary == 1`; defendant advocates "win" when `outcome_binary == 0`.

## Data Quality Notes

- **Advocate data**: Attorneys metadata available on ~22% of cases. Text extraction adds sided advocates for ~15-25% more. Combined coverage estimated at 30-40%.
- **Outcome labels**: ~40% from metadata "Outcome" field, ~20% additional from text-based extraction. 282/741 cases (38%) have no extractable outcome.
- **Citation data**: Extracted via regex from judgment text (Act/Cap./Section counts); varies by case type.
- **Monetary amounts**: Multi-pattern extraction; injunction/criminal cases often lack monetary claims.
- **Court division filtering**: Criminal (263) and Family (106) cases excluded for commercial prediction model.
- **Text features**: TF-IDF on judgment text with SVD dimensionality reduction (30 components).
