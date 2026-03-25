# AI_USAGE.md

## AI usage disclosure

AI assistance was used to:
- draft project structure,
- generate implementation ideas,
- produce boilerplate Streamlit/pandas code,
- draft README and deliverable checklists.

## What was verified manually

- Verified every required page exists and is navigable.
- Verified CSV/XLSX/JSON upload logic and fallback errors.
- Verified each cleaning operation updates the working dataframe.
- Verified transformation log appends operation name, parameters, and timestamp.
- Verified undo/reset controls behave safely.
- Verified visualization builder supports six required chart types.
- Verified export buttons generate cleaned dataset/report/recipe outputs.
- Verified validation rules generate a violations table and export.

## Limitations reviewed manually

- Formula input uses a restricted `eval`; malformed formulas are handled with user-facing errors.
- Very large datasets may still require optimization beyond baseline caching/session-state strategy.
- Optional LLM and Google Sheets features were intentionally omitted.
