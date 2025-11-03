from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
import numpy as np


DEFAULT_PLASTICS = ["HDPE", "LDPE", "LLDPE", "PP"]
DEFAULT_CONTAMINANTS = ["BSA", "OIL", "GUAR", "CMC", "STARCH"]


def _parse_filename_tokens(filename: str) -> tuple[str, list[str]]:
    """Extract polymer and contaminant tokens from a filename stem.

    Expected filename examples (from repo):
      - HDPE-W_BSA 1203-67(#4)1.CSV
      - HDPE-W_BSA_OIL_CMC_STARCH 1203-72(#8)1.CSV

    Strategy:
      - take the stem (filename without extension), split on first space and use the left token
      - split that token on the literal '-W' marker: left is polymer, right are contaminants separated by '_'
      - normalize tokens to upper-case names
    Returns: (polymer, [contaminant_tokens...])
    """
    stem = Path(filename).stem
    left_token = stem.split(" ", 1)[0]
    if "-W" in left_token:
        parts = left_token.split("-W", 1)
        polymer = parts[0].upper()
        rest = parts[1].lstrip("-_ ")
        contaminants = [t.upper() for t in rest.split("_") if t]
    else:
        # fallback: try first token as polymer, no contaminants
        polymer = left_token.upper()
        contaminants = []
    return polymer, contaminants


def load_raw_data(raw_root: str | Path, *,
                  plastics: Iterable[str] | None = None,
                  contaminants: Iterable[str] | None = None,
                  save_path: str | Path | None = None) -> pd.DataFrame:
    """Load all CSVs under `raw_root` recursively, attach metadata and one-hot columns.

    - raw_root: path to `data/raw` (or the repository root) where CSVs exist under subfolders.
    - plastics: iterable of plastic types to one-hot (defaults to HDPE, LDPE, LLDPE, PP)
    - contaminants: iterable of contaminant types to one-hot (defaults to BSA, OIL, GUAR, CMC, STARCH)
    - save_path: optional path to save the combined dataframe as a pickle file (recommended: `data/processed/raw_combined.pkl`)

    Returns the combined pandas DataFrame.
    """
    raw_root = Path(raw_root)
    plastics = [p.upper() for p in (plastics or DEFAULT_PLASTICS)]
    contaminants = [c.upper() for c in (contaminants or DEFAULT_CONTAMINANTS)]

    rows: list[pd.DataFrame] = []

    for csv_path in raw_root.rglob("*.CSV"):
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            # try lowercase extension too
            try:
                df = pd.read_csv(csv_path)
            except Exception:
                # Skip files that cannot be read; caller can investigate
                continue

        polymer, file_contaminants = _parse_filename_tokens(csv_path.name)

        # add metadata columns
        df = df.copy()
        df["source_file"] = str(csv_path)
        df["polymer"] = polymer

        # one-hot plastics
        for p in plastics:
            df[f"is_{p}"] = 1 if polymer == p else 0

        # contaminants presence
        for c in contaminants:
            df[f"has_{c}"] = 1 if c in file_contaminants else 0

        rows.append(df)

    if not rows:
        combined = pd.DataFrame()
    else:
        combined = pd.concat(rows, ignore_index=True)

    if save_path:
        outp = Path(save_path)
        outp.parent.mkdir(parents=True, exist_ok=True)
        combined.to_pickle(outp)

    return combined


def save_combined_dataframe(df: pd.DataFrame, out_path: str | Path) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(out)


def files_to_numpy_summary_by_file(raw_root: str | Path, *,
                                  plastics: Iterable[str] | None = None,
                                  contaminants: Iterable[str] | None = None,
                                  agg: str = "mean") -> tuple[np.ndarray, list[str], list[Path]]:
    """Aggregate each CSV file into a single-row feature vector and return as a numpy array.

    Each row corresponds to one CSV file. By default we compute the aggregation (mean)
    of the first two numeric columns in the CSV and append one-hot encodings for plastics
    and contaminants. Returns (array, feature_names, file_paths).

    Assumptions / notes:
      - Each CSV contains at least two numeric columns. We aggregate each column to a single
        scalar (default: mean). If you need a different aggregation, set `agg` to 'median' or
        'sum'.
      - Filenames are parsed using the same logic as `load_raw_data` to detect polymer and
        contaminant tokens.
      - One-hot order: plastics (DEFAULT_PLASTICS) then contaminants (DEFAULT_CONTAMINANTS).

    This function is conservative and will skip files that cannot be parsed or that don't
    contain at least two numeric columns.
    """
    raw_root = Path(raw_root)
    plastics = [p.upper() for p in (plastics or DEFAULT_PLASTICS)]
    contaminants = [c.upper() for c in (contaminants or DEFAULT_CONTAMINANTS)]

    rows: list[np.ndarray] = []
    feature_names: list[str] = []
    file_paths: list[Path] = []

    for csv_path in raw_root.rglob("*.CSV"):
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            # skip unreadable files
            continue

        # select first two columns that are numeric
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if len(numeric_cols) < 2:
            # not enough numeric data
            continue

        c1, c2 = numeric_cols[0], numeric_cols[1]
        if agg == "mean":
            v1 = float(df[c1].mean())
            v2 = float(df[c2].mean())
        elif agg == "median":
            v1 = float(df[c1].median())
            v2 = float(df[c2].median())
        elif agg == "sum":
            v1 = float(df[c1].sum())
            v2 = float(df[c2].sum())
        else:
            raise ValueError(f"Unsupported agg: {agg}")

        polymer, file_contaminants = _parse_filename_tokens(csv_path.name)

        onehots: list[int] = []
        for p in plastics:
            onehots.append(1 if polymer == p else 0)
        for c in contaminants:
            onehots.append(1 if c in file_contaminants else 0)

        row = np.array([v1, v2] + onehots, dtype=float)
        rows.append(row)
        file_paths.append(csv_path)

    if not rows:
        return np.empty((0, 2 + len(plastics) + len(contaminants))), [], []

    arr = np.stack(rows, axis=0)

    # build stable feature names (use generic col1/col2 names so names don't depend on a particular file)
    feature_names = [f"col1_agg_{agg}", f"col2_agg_{agg}"] + [f"is_{p}" for p in plastics] + [f"has_{c}" for c in contaminants]

    return arr, feature_names, file_paths


def prepare_ml_dataset(raw_root: str | Path, *,
                       plastics: Iterable[str] | None = None,
                       contaminants: Iterable[str] | None = None,
                       agg: str = "mean",
                       return_file_paths: bool = False) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Prepare ML-ready X, y arrays from raw CSV files.

    - X: numpy array of shape (n_files, n_features) where features are aggregated numeric summaries (default: mean of first two numeric cols).
    - y: numpy array of shape (n_files, n_targets) representing the one-hot targets (plastics then contaminants).
    - returns (X, y, feature_names, target_names) by default. If return_file_paths=True, returns (X, y, feature_names, target_names, file_paths).

    This is suitable for scikit-learn estimators. For multi-output classification use
    sklearn.multioutput.MultiOutputClassifier with a base estimator such as RandomForestClassifier.
    """
    arr, feature_names, file_paths = files_to_numpy_summary_by_file(raw_root, plastics=plastics, contaminants=contaminants, agg=agg)

    if arr.size == 0:
        # empty outputs
        plastics = [p.upper() for p in (plastics or DEFAULT_PLASTICS)]
        contaminants = [c.upper() for c in (contaminants or DEFAULT_CONTAMINANTS)]
        target_names = [f"is_{p}" for p in plastics] + [f"has_{c}" for c in contaminants]
        X = np.empty((0, 2))
        y = np.empty((0, len(target_names)))
        if return_file_paths:
            return X, y, feature_names, target_names, file_paths
        return X, y, feature_names, target_names

    # first two columns are numeric features, rest are one-hot targets
    X = arr[:, :2]
    y = arr[:, 2:]

    plastics = [p.upper() for p in (plastics or DEFAULT_PLASTICS)]
    contaminants = [c.upper() for c in (contaminants or DEFAULT_CONTAMINANTS)]
    target_names = [f"is_{p}" for p in plastics] + [f"has_{c}" for c in contaminants]

    if return_file_paths:
        return X, y, feature_names, target_names, file_paths
    return X, y, feature_names, target_names


def files_to_spectra_by_file(raw_root: str | Path, *,
                             x_col: str | None = None,
                             y_col: str | None = None,
                             resample_n: int | None = None,
                             plastics: Iterable[str] | None = None,
                             contaminants: Iterable[str] | None = None) -> tuple[np.ndarray, list[str], list[Path]]:
    """Load full spectra from each CSV and return stacked numpy array per file.

    Returns (spectra_array, feature_names, file_paths)
    - spectra_array: shape (n_files, n_points)
    - feature_names: list of column names ['spec_0', 'spec_1', ...]
    - file_paths: original file paths in the same order as rows

    Behavior / assumptions:
    - By default the function will read the first two columns as (x, y) if x_col/y_col are not provided.
    - If files have different x-grids or lengths, spectra will be resampled to a common grid of length `resample_n`.
      If `resample_n` is None the median number of points across readable files is used.
    - Resampling uses numpy.interp over the union min/max range across files.
    - One-hot target columns are NOT included in the returned spectra array; they can be built using
      the same filename parsing logic and returned separately by calling `prepare_ml_dataset_spectra`.
    """
    raw_root = Path(raw_root)
    plastics = [p.upper() for p in (plastics or DEFAULT_PLASTICS)]
    contaminants = [c.upper() for c in (contaminants or DEFAULT_CONTAMINANTS)]

    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    file_paths: list[Path] = []

    for csv_path in raw_root.rglob("*.CSV"):
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue

        # determine columns
        cols = list(df.columns)
        if x_col is None:
            if len(cols) < 2:
                continue
            xc, yc = cols[0], cols[1]
        else:
            xc = x_col
            yc = y_col if y_col is not None else (cols[1] if len(cols) > 1 else None)

        if yc is None or xc not in df.columns or yc not in df.columns:
            continue

        xvals = df[xc].to_numpy(dtype=float)
        yvals = df[yc].to_numpy(dtype=float)

        # sort by x if necessary
        if not np.all(np.diff(xvals) >= 0):
            order = np.argsort(xvals)
            xvals = xvals[order]
            yvals = yvals[order]

        xs.append(xvals)
        ys.append(yvals)
        file_paths.append(csv_path)

    if not xs:
        return np.empty((0, 0)), [], []

    # determine resample length
    lengths = [len(x) for x in xs]
    if resample_n is None:
        resample_n = int(np.median(lengths))

    # determine global x range
    global_min = min(x[0] for x in xs)
    global_max = max(x[-1] for x in xs)
    new_x = np.linspace(global_min, global_max, resample_n)

    spectra: list[np.ndarray] = []
    for xvals, yvals in zip(xs, ys):
        # interpolate to new_x; values outside original range are filled via extrapolation using edge values
        interp = np.interp(new_x, xvals, yvals, left=yvals[0], right=yvals[-1])
        spectra.append(interp)

    arr = np.stack(spectra, axis=0)
    feature_names = [f"spec_{i}" for i in range(arr.shape[1])]
    return arr, feature_names, file_paths


def prepare_ml_dataset_spectra(raw_root: str | Path, *,
                                x_col: str | None = None,
                                y_col: str | None = None,
                                resample_n: int | None = None,
                                plastics: Iterable[str] | None = None,
                                contaminants: Iterable[str] | None = None,
                                return_file_paths: bool = False) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Prepare ML-ready X (spectra) and y (one-hot targets) arrays where X contains full spectra per file.

    Returns (X, y, feature_names, target_names) or with file_paths appended if return_file_paths=True.
    """
    spectra_arr, feature_names, file_paths = files_to_spectra_by_file(raw_root, x_col=x_col, y_col=y_col, resample_n=resample_n, plastics=plastics, contaminants=contaminants)

    # build targets from filenames
    plastics = [p.upper() for p in (plastics or DEFAULT_PLASTICS)]
    contaminants = [c.upper() for c in (contaminants or DEFAULT_CONTAMINANTS)]
    target_names = [f"is_{p}" for p in plastics] + [f"has_{c}" for c in contaminants]

    ys: list[np.ndarray] = []
    for p in file_paths:
        polymer, file_contaminants = _parse_filename_tokens(p.name)
        onehots = [1 if polymer == ptype else 0 for ptype in plastics]
        onehots += [1 if c in file_contaminants else 0 for c in contaminants]
        ys.append(np.array(onehots, dtype=float))

    if not ys:
        X = np.empty((0, spectra_arr.shape[1] if spectra_arr.size else 0))
        y = np.empty((0, len(target_names)))
    else:
        X = spectra_arr
        y = np.stack(ys, axis=0)

    if return_file_paths:
        return X, y, feature_names, target_names, file_paths
    return X, y, feature_names, target_names


if __name__ == "__main__":
    # Minimal runner for local debugging — does not run on import.
    root = Path(__file__).parents[2] / "data" / "raw"
    processed = Path(__file__).parents[2] / "data" / "processed" / "raw_combined.pkl"
    df = load_raw_data(root, save_path=processed)
    print("Loaded rows:", len(df))
    if not df.empty:
        print(df[["source_file", "polymer"] + [c for c in df.columns if c.startswith("is_") or c.startswith("has_")]].head())