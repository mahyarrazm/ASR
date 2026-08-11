#!/usr/bin/env python3
"""
ASR expansion: does restructuring the problem beat tuning it?

Hyperparameter search plateaued near R2 0.62 under mixture-grouped
cross-validation, with every model family landing in a narrow band. That
pattern says the limit is how the problem is posed, not how the model is
configured. This script tests four framings on identical grouped folds:

  A  pooled          one model over all rows, the current baseline
  B  per-standard    a separate model per test standard, since the accelerated
                     mortar-bar test and the concrete prism test differ in
                     temperature, solution, duration and specimen geometry
  C  kinetic         predict each mixture's expansion curve, not each row:
                     fit y = A (1 - exp(-k t)) per training mixture, regress A
                     and k on composition, then evaluate the predicted curve
  D  hybrid          the kinetic curve prediction supplied as an extra input
                     to the pooled row-level model

It first reports a variance decomposition and oracle ceilings, so the result
can be read against what is actually attainable.

Usage (Colab, GPU runtime):
    !python asr_structure.py
Environment:
    ASR_CSV_PATH  dataset location
"""

import os
import time
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

CSV_PATH = os.environ.get("ASR_CSV_PATH", "/content/ASR_FinalE-ComCo.csv")
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
COMPOSITION_NUMERIC = [c for c in NUMERIC_FEATURES if c != AGE_COLUMN]


def load():
    frame = pd.read_csv(CSV_PATH)
    for column in NUMERIC_FEATURES + [TARGET]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame[frame[TARGET].notna()].reset_index(drop=True)
    X = frame[BASE_FEATURES].copy()
    for column in CATEGORICAL_FEATURES:
        X[column] = X[column].fillna("Unknown").astype(str).str.strip()
    y = frame[TARGET].astype(float)
    keys = pd.util.hash_pandas_object(
        X[COMPOSITION_FEATURES].fillna("missing").astype(str), index=False
    )
    groups = pd.factorize(keys)[0]
    print(f"loaded {CSV_PATH}: {len(X)} rows, "
          f"{pd.Series(groups).nunique()} mixtures")
    return X.reset_index(drop=True), y.reset_index(drop=True), groups


def grouped_folds(groups, n_folds, seed):
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
    return [(np.flatnonzero(row_fold != f), np.flatnonzero(row_fold == f))
            for f in range(n_folds)]


def encode(train_X, other_X, numeric=None, categorical=None):
    """Median-impute numerics and ordinally encode categoricals, train-fit."""
    numeric = NUMERIC_FEATURES if numeric is None else numeric
    categorical = CATEGORICAL_FEATURES if categorical is None else categorical
    fill = train_X[numeric].median(numeric_only=True)
    levels = {c: sorted(train_X[c].astype(str).unique()) for c in categorical}

    def apply(frame):
        out = frame[numeric].fillna(fill).astype(float).copy()
        for column in categorical:
            mapping = {value: index for index, value in enumerate(levels[column])}
            out[column] = frame[column].astype(str).map(mapping).fillna(
                len(levels[column])).astype(float)
        return out.to_numpy()

    return apply(train_X), apply(other_X)


def gbm(seed=RANDOM_STATE):
    return HistGradientBoostingRegressor(
        max_iter=600, learning_rate=0.05, max_leaf_nodes=31,
        min_samples_leaf=10, l2_regularization=0.5, random_state=seed,
    )


def score(y_true, y_pred, label):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    result = {
        "approach": label,
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
    }
    print(f"  {label:<34} R2={result['R2']:.4f}  RMSE={result['RMSE']:.4f}  "
          f"MAE={result['MAE']:.4f}")
    return result


# =============================================================================
# Kinetic curve
# =============================================================================
def kinetic(t, ultimate, rate):
    """Saturating ASR expansion: rises quickly, then flattens."""
    return ultimate * (1.0 - np.exp(-rate * np.clip(t, 0, None)))


def fit_curve(ages, values):
    """Fit (ultimate, rate) for one mixture; fall back to sensible defaults."""
    ages = np.asarray(ages, float)
    values = np.asarray(values, float)
    if len(ages) < 3 or np.ptp(ages) <= 0:
        return float(np.max(values)) if len(values) else 0.0, 0.05
    start = [max(float(np.max(values)), 1e-4), 0.05]
    try:
        params, _ = curve_fit(
            kinetic, ages, values, p0=start, maxfev=5000,
            bounds=([0.0, 1e-4], [max(float(np.max(values)) * 5, 1.0), 5.0]),
        )
        return float(params[0]), float(params[1])
    except Exception:
        return start[0], start[1]


def mixture_table(X, y, groups, index):
    """One row per mixture: composition inputs plus fitted curve parameters."""
    frame = X.iloc[index].copy()
    frame["_y"] = np.asarray(y)[index]
    frame["_g"] = np.asarray(groups)[index]
    rows = []
    for group, block in frame.groupby("_g"):
        ultimate, rate = fit_curve(block[AGE_COLUMN], block["_y"])
        record = block.iloc[0][COMPOSITION_FEATURES].to_dict()
        record.update({"_g": group, "ultimate": ultimate, "rate": rate})
        rows.append(record)
    return pd.DataFrame(rows)


def main():
    started = time.time()
    X, y, groups = load()
    y_values = y.to_numpy()

    # ---------------------------------------------------------------- audit
    print("\n" + "=" * 72)
    print("WHERE THE VARIANCE LIVES")
    print("=" * 72)
    total = float(y.var(ddof=1))
    mixture_mean = pd.Series(y_values).groupby(groups).transform("mean")
    between = float(mixture_mean.var(ddof=1))
    within = float((y_values - mixture_mean).var(ddof=1))
    print(f"total variance        : {total:.5f}")
    print(f"  between mixtures    : {between:.5f} ({between / total:6.1%})  "
          f"predictable only from composition")
    print(f"  within mixture      : {within:.5f} ({within / total:6.1%})  "
          f"driven by testing age")
    print(f"\noracle: perfect mixture mean, no age model -> R2 = "
          f"{between / total:.4f}")
    print(f"oracle: perfect age model, no mixture info -> R2 = "
          f"{within / total:.4f}")

    print("\nper standard")
    for name, block in X.groupby(X[STANDARD_COLUMN]):
        subset = y_values[block.index]
        print(f"  {str(name):<26} rows={len(block):5d}  "
              f"mixtures={pd.Series(groups[block.index]).nunique():4d}  "
              f"mean={subset.mean():.4f}  sd={subset.std():.4f}  "
              f"max={subset.max():.4f}")

    folds = grouped_folds(groups, N_FOLDS, RANDOM_STATE)
    results = []

    # ------------------------------------------------------- A: pooled
    print("\n" + "=" * 72)
    print("FRAMINGS, IDENTICAL GROUPED FOLDS")
    print("=" * 72)
    pooled = np.full(len(y_values), np.nan)
    for train_index, test_index in folds:
        matrix_tr, matrix_te = encode(X.iloc[train_index], X.iloc[test_index])
        model = gbm().fit(matrix_tr, y_values[train_index])
        pooled[test_index] = model.predict(matrix_te)
    results.append(score(y_values, pooled, "A  pooled (baseline)"))

    # ------------------------------------------------- B: per standard
    per_standard = np.full(len(y_values), np.nan)
    for train_index, test_index in folds:
        for name in X[STANDARD_COLUMN].unique():
            tr = train_index[X[STANDARD_COLUMN].to_numpy()[train_index] == name]
            te = test_index[X[STANDARD_COLUMN].to_numpy()[test_index] == name]
            if len(te) == 0:
                continue
            if len(tr) < 40:          # too few rows to fit a separate model
                tr = train_index
            matrix_tr, matrix_te = encode(X.iloc[tr], X.iloc[te])
            model = gbm().fit(matrix_tr, y_values[tr])
            per_standard[te] = model.predict(matrix_te)
    results.append(score(y_values, per_standard, "B  per standard"))

    # ----------------------------------------------------- C: kinetic
    kinetic_pred = np.full(len(y_values), np.nan)
    for train_index, test_index in folds:
        table = mixture_table(X, y, groups, train_index)
        features_tr, _ = encode(table[COMPOSITION_FEATURES],
                                table[COMPOSITION_FEATURES],
                                COMPOSITION_NUMERIC)
        ultimate_model = gbm().fit(features_tr, table["ultimate"].to_numpy())
        rate_model = gbm(RANDOM_STATE + 1).fit(features_tr,
                                               table["rate"].to_numpy())
        test_frame = X.iloc[test_index]
        _, features_te = encode(table[COMPOSITION_FEATURES],
                                test_frame[COMPOSITION_FEATURES],
                                COMPOSITION_NUMERIC)
        ultimate_hat = ultimate_model.predict(features_te)
        rate_hat = np.clip(rate_model.predict(features_te), 1e-4, 5.0)
        kinetic_pred[test_index] = kinetic(
            test_frame[AGE_COLUMN].to_numpy(float), ultimate_hat, rate_hat
        )
    results.append(score(y_values, kinetic_pred, "C  kinetic curve"))

    # ------------------------------------------------------ D: hybrid
    hybrid = np.full(len(y_values), np.nan)
    for train_index, test_index in folds:
        table = mixture_table(X, y, groups, train_index)
        features_tr, _ = encode(table[COMPOSITION_FEATURES],
                                table[COMPOSITION_FEATURES],
                                COMPOSITION_NUMERIC)
        ultimate_model = gbm().fit(features_tr, table["ultimate"].to_numpy())
        rate_model = gbm(RANDOM_STATE + 1).fit(features_tr,
                                               table["rate"].to_numpy())

        def curve_feature(index):
            frame = X.iloc[index]
            _, features = encode(table[COMPOSITION_FEATURES],
                                 frame[COMPOSITION_FEATURES],
                                 COMPOSITION_NUMERIC)
            ultimate_hat = ultimate_model.predict(features)
            rate_hat = np.clip(rate_model.predict(features), 1e-4, 5.0)
            return kinetic(frame[AGE_COLUMN].to_numpy(float), ultimate_hat,
                           rate_hat)

        matrix_tr, matrix_te = encode(X.iloc[train_index], X.iloc[test_index])
        matrix_tr = np.column_stack([matrix_tr, curve_feature(train_index)])
        matrix_te = np.column_stack([matrix_te, curve_feature(test_index)])
        model = gbm().fit(matrix_tr, y_values[train_index])
        hybrid[test_index] = model.predict(matrix_te)
    results.append(score(y_values, hybrid, "D  hybrid (curve as input)"))

    # ------------------------------------------------------- reporting
    table = pd.DataFrame(results).sort_values("R2", ascending=False)
    table.to_csv("asr_structure_results.csv", index=False)

    print("\n" + "=" * 72)
    print("RESULT")
    print("=" * 72)
    print(table.to_string(index=False,
                          float_format=lambda v: f"{v:9.4f}"))
    best = table.iloc[0]
    baseline = table.loc[table["approach"].str.startswith("A")].iloc[0]
    gain = best["R2"] - baseline["R2"]
    print(f"\nbest framing: {best['approach']}  "
          f"({gain:+.4f} R2 versus pooled)")

    print("\nper-standard breakdown of the best framing")
    column = {"A": pooled, "B": per_standard, "C": kinetic_pred,
              "D": hybrid}[best["approach"][0]]
    for name, block in X.groupby(X[STANDARD_COLUMN]):
        index = block.index
        print(f"  {str(name):<26} R2={r2_score(y_values[index], column[index]):.4f}  "
              f"RMSE={np.sqrt(mean_squared_error(y_values[index], column[index])):.4f}")
    print(f"\nelapsed {(time.time() - started) / 60:.1f} min; "
          f"wrote asr_structure_results.csv")


if __name__ == "__main__":
    main()
