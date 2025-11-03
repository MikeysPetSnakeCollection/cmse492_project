## Repository snapshot

- Purpose: IR and Raman processing of contaminated plastics to improve washing/sorting (ML/data-analysis project).
- Primary work lives in Jupyter notebooks under `notebooks/exploratory/` (example: `baseline_model.ipynb`, `data_acquisition_and_loading.ipynb`).
- Data: CSVs in `data/raw/Plastic Washing CMSE project CSV files/ATR set 1_washed/` — filenames encode polymer and contamination metadata (example tokens: polymer `HDPE`/`LDPE`, contaminants like `BSA`, `OIL`, `CMC`, `STARCH`).

## Quick environment & run notes

- README states Python 3.12.11. There is a `requirements.txt` but it is currently empty—inspect notebooks for actual imports before adding packages.
- Typical developer flow:
  1. Create / activate a Python venv using Python 3.12.11.
  2. Inspect `notebooks/exploratory/data_acquisition_and_loading.ipynb` to learn how CSVs are loaded and preprocessed.
  3. Run notebooks in VS Code or Jupyter (Notebook/Lab). On Windows PowerShell, a minimal setup looks like:

```powershell
py -3.12 -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -r requirements.txt
# then open VS Code or jupyter lab
```

## Project structure & where to make changes

- Notebooks contain experiments and data-wrangling code. Prefer small, incremental edits to notebooks and extract reusable code to `src/`.
- `src/` exists with intended subpackages: `src/preprocessing/`, `src/models/`, and `src/evaluation/`. These folders are currently empty — use them for reusable functions, model wrappers, and evaluation utilities.
- Keep data access paths robust: CSV filenames include spaces and parentheses, so use Python Path objects or raw strings (avoid fragile shell/glob expansions).

## Patterns & examples discovered in this repo

- Data naming pattern: `<POLYMER>-W_<CONTAMINANTS> <batch>...` (tokens separated by underscores and spaces) — use the filename tokens to derive metadata when building dataset loaders.
- Example loader pattern agents should look for in notebooks (search `data_acquisition_and_loading.ipynb`): pandas-based reads of CSVs, then a standard preprocessing pipeline.

## Agent-focused editing rules (actionable & specific)

1. Prefer modifying or adding small Python modules inside `src/` for reusable logic rather than making large, duplicated notebook edits.
2. When adding dependencies, update `requirements.txt` and note the Python version at top of `README.md`.
3. If you change data paths, update relative references in notebooks accordingly; many notebooks assume the repo root as the working directory.
4. Do not assume tests or CI — none are present. Add unit tests under a new `tests/` folder if you create reusable modules.

## Files to consult first (order matters)

1. `notebooks/exploratory/data_acquisition_and_loading.ipynb` — canonical data loader and initial preprocessing.
2. `notebooks/exploratory/Initial_exploratory_data_analysis.ipynb` — exploratory patterns and feature ideas.
3. `notebooks/exploratory/baseline_model.ipynb` — model training and evaluation references.
4. `data/raw/Plastic Washing CMSE project CSV files/ATR set 1_washed/` — ground truth CSVs and filename conventions.

## When you need clarification

- If a notebook imports a module that is missing from `src/`, prefer extracting the related cells into `src/<area>` and add a minimal unit test covering the extraction.
- If required package versions are not listed, inspect the first notebook cells for imports and infer versions conservatively; ask the repo maintainer before pinning versions.

---
If anything here is unclear or you want the agent to follow a stricter rule set (tests, formatting, CI), tell me which areas to expand and I'll iterate.
