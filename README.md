# Litigation Analytics Kenya

AI-assisted litigation outcome prediction for Kenyan courts, focused on Commercial Division cases at the Milimani High Court (HCNRB).

## Problem

Senior advocates spend 3-5 hours per case researching past judgments. Outcome forecasts are highly variable and lack data-driven support. This project builds a prediction tool that:

1. Improves forecast accuracy by >=10 percentage points over unassisted predictions
2. Reduces case research time by >=30%
3. Provides transparent explanations (judge proclivity, statutory citations, claim size)

## Quick Start

```bash
# Install dependencies
pip install -e ".[dev]"

# Set up environment
cp .env.example .env
# Edit .env with your Tausi API token

# Run data discovery
jupyter notebook notebooks/01_data_discovery.ipynb

# Run the API (after training a model)
uvicorn src.api.app:app --reload
```

## Project Structure

- `configs/` - Settings, court mappings, outcome taxonomy
- `src/data/` - API client, downloader, PDF extraction, validation
- `src/features/` - Outcome parsing, tabular/text feature engineering
- `src/models/` - Baseline, XGBoost, calibration, evaluation
- `src/api/` - FastAPI prediction service
- `src/analysis/` - Power analysis, A/B test statistics
- `notebooks/` - Sequential analysis notebooks (01-05)
- `tests/` - Test suite
- `docs/` - Data dictionary, experiment plan, model card

## Phased Approach

1. **Phase 0** - Data discovery & hypothesis validation
2. **Phase 1** - Data collection & outcome labeling
3. **Phase 2** - Feature engineering
4. **Phase 3** - Model development (Logistic Regression -> XGBoost -> text-enhanced)
5. **Phase 4** - API deployment
6. **Phase 5** - Shadow mode & offline validation
7. **Phase 6** - Pilot A/B test with advocates
