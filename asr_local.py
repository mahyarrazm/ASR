#!/usr/bin/env python3
"""
ASR expansion: scope comparison, built to run locally in VS Code.

Earlier runs pooled three very different kinds of test into one model: the
accelerated mortar-bar test (ASTM C1260/C1567, 14 days at 80 C in 1N NaOH),
the concrete prism test (ASTM C1293, a year at 38 C), and a large block of
non-standard tests. Pooled mixture-grouped R2 was about 0.62, but the pooled
model already scored 0.77 on the C1260 rows alone. Averaging one well-defined
protocol together with two others is what produces the low headline number.

This script fits and scores each scope separately under identical
mixture-grouped cross-validation, so the cost of pooling is measured rather
than assumed.

Everything here runs on CPU in a few minutes on an Apple silicon laptop.
TabPFN is off by default because it is roughly fifty times slower on CPU for a
gain of about 0.03 R2; enable it with --tabpfn once a scope is chosen.

Usage:
    python asr_local.py
    python asr_local.py --csv "/path/to/ASR_FinalE-ComCo.csv"
    python asr_local.py --scope c1260 --tabpfn
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.ensemble import (
    ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

DEFAULT_CSV = "~/Smart Lab/Lab/ML/Final Dataset/ASR_FinalE-ComCo.csv"
DATA_FILENAME = "ASR_FinalE-ComCo.csv"
TARGET = "y"
AGE_COLUMN = "x29"
STANDARD_COLUMN = "Standard"
RANDOM_STATE = 256
N_FOLDS = 5

NUMERIC_FEATURES = [
    "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x9", "x10",
    "x11", "x12", "x13", "x14", "x16", "x17", "x18", "x19", "x20",
    "X21", "x22", "x23", "x24", "x25", "x26", "x27", "x29",
]
CATEGORICAL_FEATURES = ["x15", "x28", "Standard"]
BASE_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
COMPOSITION_FEATURES = [c for c in BASE_FEATURES if c != AGE_COLUMN]
MK_COLS = ["x11", "x12", "x14"]
FA_COLS = ["x16", "x17", "x18", "x19", "X21"]
CNS_COLS = ["x22", "x23", "x24", "x26", "x27"]


def resolve_csv(given: str) -> Path:
    """Accept the given path, or find the dataset by name under the home dir."""
    candidate = Path(given).expanduser()
    if candidate.is_file():
        return candidate
    print(f"not found: {candidate}\nsearching for {DATA_FILENAME}...")
    matches = sorted(Path.home().rglob(DATA_FILENAME))
    if not matches:
        sys.exit(f"could not find {DATA_FILENAME} under {Path.home()}. "
                 f"Pass the path explicitly with --csv")
    print(f"using: {matches[0]}")
    return matches[0]


def load(csv_path: Path):
    frame = pd.read_csv(csv_path)
    missing = [c for c in BASE_FEATURES + [TARGET] if c not in frame.columns]
    if missing:
        sys.exit(f"columns missing from {csv_path}: {missing}")
    for column in NUMERIC_FEATURES + [TARGET]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame[frame[TARGET].notna()].reset_index(drop=True)
    X = frame[BASE_FEATURES].copy()
    for column in CATEGORICAL_FEATURES:
        X[column] = X[column].fillna("Unknown").astype(str).str.strip()
        X[column] = X[column].replace({"": "Unknown", "nan": "Unknown"})
    y = frame[TARGET].astype(float)
    keys = pd.util.hash_pandas_object(
        X[COMPOSITION_FEATURES].fillna("missing").astype(str), index=False
    )
    groups = pd.factorize(keys)[0]
    return X.reset_index(drop=True), y.reset_index(drop=True), groups


def scope_mask(X: pd.DataFrame, scope: str) -> np.ndarray:
    standard = X[STANDARD_COLUMN].astype(str)
    if scope == "all":
        return np.ones(len(X), dtype=bool)
    if scope == "standard":
        return (standard != "Unknown").to_numpy()
    if scope == "c1260":
        return standard.str.contains("1260|1567", regex=True).to_numpy()
    if scope == "c1293":
        return standard.str.contains("1293").to_numpy()
    if scope == "nonstandard":
        return (standard == "Unknown").to_numpy()
    sys.exit(f"unknown scope: {scope}")


def grouped_folds(groups, n_folds, seed):
    """Group-disjoint folds balanced on row count."""
    groups = np.asarray(groups)
    unique, inverse = np.unique(groups, return_inverse=True)
    sizes = np.bincount(inverse)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(unique))
    order = order[np.argsort(-sizes[order], kind="stable")]
    fold_of_group = np.empty(len(unique), dtype=int)
    loads = np.zeros(n_folds)
    for index in order:
        target = int(np.argmin(loads))
        fold_of_group[index] = target
        loads[target] += sizes[index]
    row_fold = fold_of_group[inverse]
    folds = []
    for f in range(n_folds):
        test = np.flatnonzero(row_fold == f)
        train = np.flatnonzero(row_fold != f)
        if len(test) and len(train):
            folds.append((train, test))
    return folds


def fit_preprocessing(train_df):
    fill = {}

    def conditional(prop_cols, gate):
        present = train_df[gate].fillna(0) > 0
        for column in prop_cols:
            values = train_df.loc[present, column].dropna()
            fill[column] = float(values.median()) if len(values) else 0.0

    conditional(MK_COLS, "x13")
    conditional(FA_COLS, "x20")
    conditional(CNS_COLS, "x25")
    for column in NUMERIC_FEATURES:
        if column not in fill:
            values = train_df[column].dropna()
            fill[column] = float(values.median()) if len(values) else 0.0
    categories = {c: sorted(train_df[c].astype(str).unique())
                  for c in CATEGORICAL_FEATURES}
    return {"fill": fill, "categories": categories}


def build_matrix(frame, state):
    out = frame[NUMERIC_FEATURES].fillna(state["fill"]).astype(float).copy()
    for column, levels in state["categories"].items():
        mapping = {value: index for index, value in enumerate(levels)}
        out[column] = frame[column].astype(str).map(mapping).fillna(
            len(levels)).astype(float)
    return out.to_numpy()


def model_zoo(seed, use_tabpfn):
    """Fast, well-separated learners. TabPFN only when explicitly requested."""
    zoo = {
        "HistGBM": HistGradientBoostingRegressor(
            max_iter=600, learning_rate=0.05, max_leaf_nodes=31,
            min_samples_leaf=10, l2_regularization=0.5, random_state=seed),
        "RandomForest": RandomForestRegressor(
            n_estimators=500, min_samples_leaf=2, max_features=0.6,
            random_state=seed, n_jobs=-1),
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=500, min_samples_leaf=2, max_features=0.6,
            random_state=seed, n_jobs=-1),
    }
    try:
        from lightgbm import LGBMRegressor

        zoo["LightGBM"] = LGBMRegressor(
            n_estimators=800, learning_rate=0.05, num_leaves=31,
            min_child_samples=10, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.8, reg_lambda=1.0, random_state=seed,
            n_jobs=-1, verbose=-1)
    except Exception as error:  # noqa: BLE001
        print(f"  LightGBM unavailable ({type(error).__name__}); skipping."
              f" On macOS this usually means: brew install libomp")
    try:
        from xgboost import XGBRegressor

        zoo["XGBoost"] = XGBRegressor(
            n_estimators=800, learning_rate=0.05, max_depth=6, subsample=0.8,
            colsample_bytree=0.8, reg_lambda=2.0, random_state=seed,
            n_jobs=-1, verbosity=0, tree_method="hist")
    except Exception as error:  # noqa: BLE001
        print(f"  XGBoost unavailable ({type(error).__name__}); skipping."
              f" On macOS this usually means: brew install libomp")
    try:
        from catboost import CatBoostRegressor

        zoo["CatBoost"] = CatBoostRegressor(
            iterations=800, learning_rate=0.05, depth=6, l2_leaf_reg=3.0,
            random_seed=seed, verbose=0, allow_writing_files=False)
    except Exception as error:  # noqa: BLE001
        print(f"  CatBoost unavailable ({type(error).__name__}); skipping")
    if use_tabpfn:
        try:
            import torch
            from tabpfn import TabPFNRegressor

            device = "mps" if torch.backends.mps.is_available() else "cpu"
            zoo["TabPFN"] = TabPFNRegressor(device=device, random_state=seed)
            print(f"  TabPFN enabled on {device}")
        except Exception as error:  # noqa: BLE001
            print(f"  TabPFN unavailable ({type(error).__name__}); skipping")
    return zoo


def evaluate_scope(X, y, groups, scope, seed, n_folds, use_tabpfn):
    mask = scope_mask(X, scope)
    if mask.sum() < 100:
        print(f"\n{scope}: only {mask.sum()} rows; skipped")
        return []
    Xs = X.loc[mask].reset_index(drop=True)
    ys = y.loc[mask].reset_index(drop=True)
    gs = np.asarray(groups)[mask]
    y_values = ys.to_numpy()
    folds = grouped_folds(gs, n_folds, seed)

    print(f"\n{'=' * 70}\nSCOPE: {scope}   rows={len(Xs)}   "
          f"mixtures={pd.Series(gs).nunique()}   "
          f"mean={y_values.mean():.4f}  sd={y_values.std():.4f}\n{'=' * 70}")

    names = list(model_zoo(seed, use_tabpfn))
    predictions = {name: np.full(len(y_values), np.nan) for name in names}
    for train_index, test_index in folds:
        state = fit_preprocessing(Xs.iloc[train_index])
        matrix_tr = build_matrix(Xs.iloc[train_index], state)
        matrix_te = build_matrix(Xs.iloc[test_index], state)
        for name, model in model_zoo(seed, use_tabpfn).items():
            try:
                model.fit(matrix_tr, y_values[train_index])
                predictions[name][test_index] = model.predict(matrix_te)
            except Exception as error:  # noqa: BLE001
                print(f"  {name} failed: {type(error).__name__}")
                predictions[name][:] = np.nan

    rows = []
    usable = {}
    for name, prediction in predictions.items():
        if not np.isfinite(prediction).all():
            continue
        usable[name] = prediction
        rows.append({
            "scope": scope, "model": name,
            "R2": float(r2_score(y_values, prediction)),
            "RMSE": float(np.sqrt(mean_squared_error(y_values, prediction))),
            "MAE": float(mean_absolute_error(y_values, prediction)),
            "n_rows": len(y_values), "n_mixtures": int(pd.Series(gs).nunique()),
        })

    # Non-negative blend, with weights fitted inside a second grouped split so
    # the blend is not scored on the rows that chose its weights.
    if len(usable) >= 2:
        matrix = np.column_stack(list(usable.values()))
        blend = np.full(len(y_values), np.nan)
        for train_index, test_index in grouped_folds(gs, n_folds, seed + 5150):
            weights, _ = nnls(matrix[train_index], y_values[train_index])
            blend[test_index] = matrix[test_index] @ weights
        if np.isfinite(blend).all():
            rows.append({
                "scope": scope, "model": "BLEND",
                "R2": float(r2_score(y_values, blend)),
                "RMSE": float(np.sqrt(mean_squared_error(y_values, blend))),
                "MAE": float(mean_absolute_error(y_values, blend)),
                "n_rows": len(y_values),
                "n_mixtures": int(pd.Series(gs).nunique()),
            })

    table = pd.DataFrame(rows).sort_values("R2", ascending=False)
    print(table[["model", "R2", "RMSE", "MAE"]].to_string(
        index=False, float_format=lambda v: f"{v:9.4f}"))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=DEFAULT_CSV, help="dataset path")
    parser.add_argument("--scope", default="compare",
                        choices=["compare", "all", "standard", "c1260",
                                 "c1293", "nonstandard"])
    parser.add_argument("--folds", type=int, default=N_FOLDS)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--tabpfn", action="store_true",
                        help="include TabPFN (much slower on CPU)")
    parser.add_argument("--out", default="asr_local_results.csv")
    args = parser.parse_args()

    started = time.time()
    csv_path = resolve_csv(args.csv)
    X, y, groups = load(csv_path)
    print(f"loaded {csv_path}")
    print(f"  {len(X)} rows, {pd.Series(groups).nunique()} mixtures")
    counts = X[STANDARD_COLUMN].value_counts()
    for name, count in counts.items():
        subset = y[X[STANDARD_COLUMN] == name]
        print(f"  {str(name):<24} rows={count:5d}  mean={subset.mean():.4f}  "
              f"sd={subset.std():.4f}")

    scopes = (["all", "standard", "c1260", "c1293", "nonstandard"]
              if args.scope == "compare" else [args.scope])
    results = []
    for scope in scopes:
        results.extend(evaluate_scope(X, y, groups, scope, args.seed,
                                      args.folds, args.tabpfn))

    if not results:
        sys.exit("no scope produced a usable result")
    table = pd.DataFrame(results)
    table.to_csv(args.out, index=False)

    print(f"\n{'=' * 70}\nBEST MODEL PER SCOPE\n{'=' * 70}")
    best = (table.sort_values("R2", ascending=False)
            .drop_duplicates("scope")
            .sort_values("R2", ascending=False))
    print(best[["scope", "model", "R2", "RMSE", "MAE", "n_rows",
                "n_mixtures"]].to_string(
        index=False, float_format=lambda v: f"{v:9.4f}"))
    print(f"\nelapsed {(time.time() - started) / 60:.1f} min; wrote {args.out}")
    print("\nAll scores are mixture-grouped cross-validation: no mixture "
          "appears in both a training fold and the fold that scores it.")


if __name__ == "__main__":
    main()
