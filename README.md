# AI-Assisted Data Wrangler & Visualizer (5COSC038C CW)

A Streamlit application for uploading, cleaning, transforming, validating, visualizing, and exporting datasets with a reproducible transformation log/recipe.

## Features implemented

- **Page A — Upload & Overview**
  - Upload CSV, XLSX, JSON
  - Dataset shape, dtypes, summary statistics
  - Missing value table (count + %)
  - Duplicate count
  - Explicit column-count info box
  - Reset session button

- **Page B — Cleaning & Preparation Studio**
  - Missing value handling:
    - drop rows by selected columns
    - drop columns above missing threshold
    - fill with constant, mean/median/mode/most frequent, ffill/bfill
  - Duplicate detection/removal:
    - full-row or subset-key duplicates
    - keep first / keep last
  - Type conversion/parsing:
    - numeric, category, datetime
    - dirty numeric cleaning (currency/comma/text stripping)
  - Categorical tools:
    - trim + lower/title standardization
    - mapping dictionary (JSON)
    - optional set unmatched to `Other`
    - rare category grouping to `Other`
    - one-hot encoding
  - Numeric cleaning:
    - IQR outlier summary
    - cap/winsorize or remove outlier rows
  - Scaling:
    - min-max and z-score
    - before/after stats
  - Column operations:
    - rename, drop
    - formula-based column creation
    - equal-width / quantile binning
  - Data validation:
    - numeric range, allowed categories, non-null
    - violations table + export
  - Transformation log + undo last step + reset all transformations

- **Page C — Visualization Builder**
  - Chart types (6 required): histogram, box, scatter, line, bar, correlation heatmap
  - Dynamic x/y/group selection
  - Optional aggregation (sum/mean/count/median)
  - Category + numeric range filtering
  - Top-N categories for bar chart
  - Built with **matplotlib** + seaborn

- **Page D — Export & Report**
  - Export cleaned data as CSV and Excel
  - Export transformation report JSON (steps + params + timestamps)
  - Export JSON recipe of transformations
  - Export validation violations CSV

## Project structure

- `app.py` – Streamlit application
- `requirements.txt`
- `sample_data/` – two demo datasets
- `deliverables/` – report templates/examples requested by coursework
- `AI_USAGE.md` – AI usage and manual verification statement
- `ALL_CHAT_AND_PROMPTS.md` – full prompt/history artifact for submission

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Sample datasets

- `sample_data/sales_messy.csv`
- `sample_data/hr_workforce.json`

Both are intentionally messy and satisfy the coursework’s testing constraints (>=1000 rows, >=8 columns, mixed types, missing values).

## Notes

- App works fully **without AI integration** (as required).
- Optional Google Sheets and LLM assistant are not included in this baseline version.
- Use Streamlit Community Cloud for deployment URL requirement.
