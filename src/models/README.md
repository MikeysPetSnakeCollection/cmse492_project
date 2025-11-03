# Baseline model — how to re-run

This short README explains how to re-run the baseline training notebook and where its outputs are saved.

## Files
- Notebook: `src/models/baseline_model.ipynb` — trains a MultiOutput RandomForest baseline on the spectra dataset (uses `src.preprocessing.loader.prepare_ml_dataset_spectra`).
- Trained model: `src/models/baseline_multioutput_rf.joblib`
- Metrics (JSON): `src/models/baseline_metrics.json`
- Metrics (Excel): `src/models/baseline_metrics.xlsx`

## Quick steps (PowerShell)
1. Create & activate a Python virtual environment (recommended Python 3.12 per repository README):

```powershell
py -3.12 -m venv .venv; .\.venv\Scripts\Activate.ps1
```

2. Install dependencies. The repository `requirements.txt` may not contain all needed packages used by the notebook. Install at least the following if they are missing:

```powershell
pip install -r requirements.txt
pip install numpy pandas scikit-learn matplotlib seaborn joblib openpyxl jupyter
```

3. Open the notebook in Jupyter or VS Code and run it interactively:

```powershell
# start Jupyter Lab (optional)
jupyter lab
# then open src/models/baseline_model.ipynb and run the cells
```

4. (Non-interactive) Execute the notebook end-to-end from the command line (useful for CI):

```powershell
# increase timeout if training takes longer than the default
jupyter nbconvert --to notebook --inplace --execute src/models/baseline_model.ipynb --ExecutePreprocessor.timeout=1200
```

## Reproducibility & options
- The notebook sets a RANDOM_STATE variable; change it to reproduce different splits/trials.
- Test/train split is controlled by TEST_SIZE in the notebook.
- The loader function `prepare_ml_dataset_spectra` resamples spectra to a common grid; check or change its parameters in `src/preprocessing/loader.py` if you want different resampling or normalization.

## Where outputs are written
By default the notebook saves outputs under `src/models/`:
- `baseline_multioutput_rf.joblib` — the trained sklearn estimator (joblib).
- `baseline_metrics.json` — evaluation metrics (JSON).
- `baseline_metrics.xlsx` — Excel workbook with a summary sheet and a per-target sheet.

## Troubleshooting
- If the notebook fails to import `src.preprocessing.loader`, ensure the notebook's working directory is the repository root or that `src` is on `sys.path`. The notebook already includes a repo-root discovery snippet which adds the repository root to `sys.path`.
- Missing package errors: install the packages listed above.
- Long training times: reduce `n_estimators` in the RandomForest instantiation or run on a machine with more CPU resources.

If you'd like, I can add a small test that verifies the existence of the saved files after notebook execution, or I can commit these changes for you. Let me know which you'd prefer.
