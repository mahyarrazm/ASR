#!/usr/bin/env python3
"""
ASR expansion: recover accuracy the current pipeline is leaving on the table.

The pipeline's CatBoost runs at depth=10 for 2600 iterations. Those values were
inherited from a configuration tuned under random-row cross-validation, where
the model can interpolate a specimen's own expansion curve and depth is
rewarded for memorising it. Under mixture-disjoint validation that reward
disappears, and the in-sample development fit of R2 0.9955 says the model is
memorising rather than generalising. Nothing has yet re-tuned the model against
the protocol it is actually scored under. That is the first lever here.

The second is the shape of the age response. ASR expansion saturates with time,
so expansion is close to linear in log(age), but an axis-aligned tree has to
spend many splits approximating that curve. Testing age is the single most
important input, so handing the model log(age), sqrt(age) and a few composition
aggregates directly is worth measuring. These are deterministic functions of
measured inputs -- ordinary feature engineering, not leakage -- and the pipeline
excludes them as a purity choice rather than a correctness one.

Both levers are searched under mixture-grouped cross-validation on the
development partition alone. The locked holdout is scored twice at the very
end: once for the current pipeline configuration, once for the winner. Both
evaluations are declared, so the comparison is not a multiple-testing artefact.

The partition reproduces the pipeline exactly -- same scope, same grouping,
same seed -- so the holdout numbers printed here are directly comparable to the
pipeline's reported R2 0.7662.

Usage:
    python asr_boost.py                       # standard scope, 60 trials
    python asr_boost.py --trials 150          # a longer search
    python asr_boost.py --scope c1260
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
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

# Matched to asr_publication_pipeline.py so the partition is identical.
RANDOM_STATE = 256
TEST_SIZE = 0.20
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

# Supplementary-cementitious-material replacement levels, in percent.
MK_CONTENT, FA_CONTENT, CNS_CONTENT = "x13", "x20", "x25"

# The configuration the pipeline currently ships, carried over from a
# random-row-tuned run. Kept here as the baseline the search has to beat.
PIPELINE_CAT_PARAMS = dict(
    iterations=2600, depth=10, learning_rate=0.0825, l2_leaf_reg=15.34,
    bagging_temperature=0.242, random_strength=0.414,
)


# =============================================================================
# Data
# =============================================================================
def resolve_csv(given: str) -> Path:
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
    sys.exit(f"unknown scope: {scope}")


def grouped_holdout(groups, test_size, seed):
    """Reserve whole mixtures until the requested row fraction is reached."""
    groups = np.asarray(groups)
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(np.unique(groups))
    target_rows = int(round(test_size * len(groups)))
    membership = pd.Series(groups)
    held, count = [], 0
    for group in shuffled:
        if count >= target_rows:
            break
        held.append(group)
        count += int((membership == group).sum())
    mask = membership.isin(held).to_numpy()
    return np.flatnonzero(~mask), np.flatnonzero(mask)


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


# =============================================================================
# Fold-pure preprocessing
# =============================================================================
def fit_preprocessing(train_df):
    fill = {}

    def conditional(prop_cols, gate):
        # SCM composition is only defined where that material is present, so
        # its median is conditioned on a non-zero replacement level.
        present = train_df[gate].fillna(0) > 0
        for column in prop_cols:
            values = train_df.loc[present, column].dropna()
            fill[column] = float(values.median()) if len(values) else 0.0

    conditional(MK_COLS, MK_CONTENT)
    conditional(FA_COLS, FA_CONTENT)
    conditional(CNS_COLS, CNS_CONTENT)
    for column in NUMERIC_FEATURES:
        if column not in fill:
            values = train_df[column].dropna()
            fill[column] = float(values.median()) if len(values) else 0.0
    categories = {c: sorted(train_df[c].astype(str).unique())
                  for c in CATEGORICAL_FEATURES}
    return {"fill": fill, "categories": categories}


def derived_block(numeric: pd.DataFrame) -> pd.DataFrame:
    """Deterministic functions of the measured inputs.

    Expansion saturates with time, so it is closer to linear in log(age) than
    in age. An axis-aligned tree can only approximate that curve with a
    staircase of splits, and every split it spends on the shape of the age
    response is one it does not spend on composition.
    """
    age = numeric[AGE_COLUMN].clip(lower=0)
    mk = numeric[MK_CONTENT].fillna(0)
    fa = numeric[FA_CONTENT].fillna(0)
    cns = numeric[CNS_CONTENT].fillna(0)
    total_scm = mk + fa + cns
    log_age = np.log1p(age)
    return pd.DataFrame({
        "log_age": log_age,
        "sqrt_age": np.sqrt(age),
        "total_scm": total_scm,
        "scm_count": (mk > 0).astype(float) + (fa > 0).astype(float)
                     + (cns > 0).astype(float),
        "log_age_x_scm": log_age * total_scm,
        "scm_share_mk": np.where(total_scm > 0, mk / total_scm.replace(0, 1), 0.0),
        "scm_share_fa": np.where(total_scm > 0, fa / total_scm.replace(0, 1), 0.0),
    }, index=numeric.index)


def build_matrix(frame, state, derived: bool):
    numeric = frame[NUMERIC_FEATURES].fillna(state["fill"]).astype(float)
    parts = [numeric]
    if derived:
        parts.append(derived_block(numeric))
    encoded = {}
    for column, levels in state["categories"].items():
        mapping = {value: index for index, value in enumerate(levels)}
        encoded[column] = (frame[column].astype(str).map(mapping)
                           .fillna(len(levels)).astype(float))
    parts.append(pd.DataFrame(encoded, index=frame.index))
    return pd.concat(parts, axis=1).to_numpy()


# =============================================================================
# Target transforms
# =============================================================================
TRANSFORMS = {
    "none": (lambda v: v, lambda v: v),
    "log1p": (lambda v: np.log1p(np.clip(v, 0, None)),
              lambda v: np.expm1(v)),
    "sqrt": (lambda v: np.sqrt(np.clip(v, 0, None)),
             lambda v: np.clip(v, 0, None) ** 2),
}


# =============================================================================
# Search space
# =============================================================================
def available_families():
    families = ["HistGBM", "ExtraTrees", "RandomForest"]
    for label, module, symbol in (("CatBoost", "catboost", "CatBoostRegressor"),
                                  ("LightGBM", "lightgbm", "LGBMRegressor"),
                                  ("XGBoost", "xgboost", "XGBRegressor")):
        try:
            getattr(__import__(module, fromlist=[symbol]), symbol)
            families.append(label)
        except Exception as error:  # noqa: BLE001
            print(f"  {label} unavailable ({type(error).__name__}); skipped."
                  + ("  On macOS: brew install libomp"
                     if label in ("LightGBM", "XGBoost") else ""))
    return families


def sample_config(family, rng):
    if family == "CatBoost":
        return dict(
            iterations=int(rng.integers(400, 2200)),
            depth=int(rng.integers(4, 9)),
            learning_rate=float(rng.uniform(0.02, 0.10)),
            l2_leaf_reg=float(rng.uniform(1.0, 30.0)),
            bagging_temperature=float(rng.uniform(0.0, 1.0)),
            random_strength=float(rng.uniform(0.0, 2.0)),
        )
    if family == "HistGBM":
        return dict(
            max_iter=int(rng.integers(200, 1200)),
            learning_rate=float(rng.uniform(0.02, 0.12)),
            max_leaf_nodes=int(rng.integers(8, 48)),
            min_samples_leaf=int(rng.integers(5, 40)),
            l2_regularization=float(rng.uniform(0.0, 5.0)),
        )
    if family == "LightGBM":
        return dict(
            n_estimators=int(rng.integers(300, 1600)),
            learning_rate=float(rng.uniform(0.02, 0.10)),
            num_leaves=int(rng.integers(8, 48)),
            min_child_samples=int(rng.integers(5, 40)),
            subsample=float(rng.uniform(0.6, 1.0)),
            colsample_bytree=float(rng.uniform(0.5, 1.0)),
            reg_lambda=float(rng.uniform(0.0, 10.0)),
        )
    if family == "XGBoost":
        return dict(
            n_estimators=int(rng.integers(300, 1600)),
            learning_rate=float(rng.uniform(0.02, 0.10)),
            max_depth=int(rng.integers(3, 9)),
            subsample=float(rng.uniform(0.6, 1.0)),
            colsample_bytree=float(rng.uniform(0.5, 1.0)),
            min_child_weight=float(rng.uniform(1.0, 10.0)),
            reg_lambda=float(rng.uniform(0.0, 10.0)),
        )
    return dict(
        n_estimators=int(rng.integers(300, 900)),
        min_samples_leaf=int(rng.integers(1, 9)),
        max_features=float(rng.uniform(0.3, 1.0)),
    )


def build_estimator(family, config, seed):
    if family == "CatBoost":
        from catboost import CatBoostRegressor

        return CatBoostRegressor(**config, random_seed=seed, verbose=0,
                                 allow_writing_files=False, thread_count=-1)
    if family == "HistGBM":
        return HistGradientBoostingRegressor(**config, random_state=seed)
    if family == "LightGBM":
        from lightgbm import LGBMRegressor

        return LGBMRegressor(**config, random_state=seed, n_jobs=-1,
                             verbose=-1, subsample_freq=1)
    if family == "XGBoost":
        from xgboost import XGBRegressor

        return XGBRegressor(**config, random_state=seed, n_jobs=-1,
                            verbosity=0, tree_method="hist")
    if family == "ExtraTrees":
        return ExtraTreesRegressor(**config, random_state=seed, n_jobs=-1)
    return RandomForestRegressor(**config, random_state=seed, n_jobs=-1)


# =============================================================================
# Evaluation
# =============================================================================
def out_of_fold(X, y, folds, family, config, transform, derived, seed):
    """Grouped out-of-fold predictions. Preprocessing refits inside each fold."""
    forward, inverse = TRANSFORMS[transform]
    prediction = np.full(len(y), np.nan)
    for train_index, test_index in folds:
        state = fit_preprocessing(X.iloc[train_index])
        matrix_tr = build_matrix(X.iloc[train_index], state, derived)
        matrix_te = build_matrix(X.iloc[test_index], state, derived)
        model = build_estimator(family, config, seed)
        model.fit(matrix_tr, forward(y[train_index]))
        prediction[test_index] = inverse(model.predict(matrix_te))
    return prediction


def score(y_true, y_pred):
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
    }


def fit_and_score_holdout(X_dev, y_dev, X_hold, y_hold, family, config,
                          transform, derived, seed):
    forward, inverse = TRANSFORMS[transform]
    state = fit_preprocessing(X_dev)
    model = build_estimator(family, config, seed)
    model.fit(build_matrix(X_dev, state, derived), forward(y_dev))
    prediction = inverse(model.predict(build_matrix(X_hold, state, derived)))
    return score(y_hold, prediction)


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=DEFAULT_CSV)
    parser.add_argument("--scope", default="standard",
                        choices=["all", "standard", "c1260", "c1293"])
    parser.add_argument("--trials", type=int, default=60,
                        help="random configurations to evaluate")
    parser.add_argument("--folds", type=int, default=N_FOLDS)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--out", default="asr_boost_results.csv")
    args = parser.parse_args()

    started = time.time()
    X, y, groups = load(resolve_csv(args.csv))

    mask = scope_mask(X, args.scope)
    X = X.loc[mask].reset_index(drop=True)
    y = y.loc[mask].reset_index(drop=True).to_numpy()
    groups = np.asarray(groups)[mask]

    dev_index, hold_index = grouped_holdout(groups, TEST_SIZE, args.seed)
    X_dev, y_dev, groups_dev = X.iloc[dev_index], y[dev_index], groups[dev_index]
    X_hold, y_hold = X.iloc[hold_index], y[hold_index]
    X_dev = X_dev.reset_index(drop=True)
    folds = grouped_folds(groups_dev, args.folds, args.seed)

    print(f"scope {args.scope}: {len(X)} rows, "
          f"{pd.Series(groups).nunique()} mixtures")
    print(f"  development {len(dev_index)} rows, "
          f"{pd.Series(groups_dev).nunique()} mixtures, {len(folds)} folds")
    print(f"  holdout     {len(hold_index)} rows, "
          f"{pd.Series(groups[hold_index]).nunique()} mixtures (locked)\n")

    families = available_families()
    print(f"families: {', '.join(families)}\n")

    rng = np.random.default_rng(args.seed)
    records = []
    print(f"{'#':>4}  {'family':<13}{'transform':<10}{'feat':<7}"
          f"{'R2':>9}{'RMSE':>9}")
    print("-" * 56)
    for trial in range(args.trials):
        family = families[int(rng.integers(len(families)))]
        transform = list(TRANSFORMS)[int(rng.integers(len(TRANSFORMS)))]
        derived = bool(rng.integers(2))
        config = sample_config(family, rng)
        try:
            prediction = out_of_fold(X_dev, y_dev, folds, family, config,
                                     transform, derived, args.seed)
        except Exception as error:  # noqa: BLE001
            print(f"{trial:>4}  {family:<13}failed: {type(error).__name__}")
            continue
        if not np.isfinite(prediction).all():
            continue
        result = score(y_dev, prediction)
        records.append({"family": family, "transform": transform,
                        "derived": derived, "config": config, **result})
        print(f"{trial:>4}  {family:<13}{transform:<10}"
              f"{'base+der' if derived else 'base':<7}"
              f"{result['R2']:>9.4f}{result['RMSE']:>9.4f}")

    if not records:
        sys.exit("no configuration produced a usable result")

    table = pd.DataFrame(records).sort_values("R2", ascending=False)
    table.to_csv(args.out, index=False)

    print(f"\n{'=' * 70}\nTOP 10 BY MIXTURE-GROUPED CV (development only)\n"
          f"{'=' * 70}")
    print(table.head(10)[["family", "transform", "derived", "R2", "RMSE",
                          "MAE"]].to_string(
        index=False, float_format=lambda v: f"{v:9.4f}"))

    print(f"\n{'=' * 70}\nDOES THE DERIVED AGE BLOCK HELP?\n{'=' * 70}")
    print(table.groupby(["family", "derived"])["R2"].max().unstack()
          .rename(columns={False: "base", True: "base+derived"})
          .to_string(float_format=lambda v: f"{v:9.4f}"))

    best = table.iloc[0]
    print(f"\n{'=' * 70}\nLOCKED HOLDOUT -- scored twice, both declared\n"
          f"{'=' * 70}")

    rows = []
    if "CatBoost" in families:
        baseline = fit_and_score_holdout(
            X_dev, y_dev, X_hold, y_hold, "CatBoost", PIPELINE_CAT_PARAMS,
            "none", False, args.seed + 2)
        rows.append({"configuration": "pipeline CatBoost (depth 10)",
                     **baseline})
    rows.append({
        "configuration": f"search winner: {best['family']} "
                         f"/ {best['transform']}"
                         f"{' / +derived' if best['derived'] else ''}",
        **fit_and_score_holdout(X_dev, y_dev, X_hold, y_hold, best["family"],
                                best["config"], best["transform"],
                                bool(best["derived"]), args.seed),
    })
    print(pd.DataFrame(rows).to_string(
        index=False, float_format=lambda v: f"{v:9.4f}"))

    print(f"\n{'=' * 70}\nPASTE INTO asr_publication_pipeline.py\n{'=' * 70}")
    if best["family"] == "CatBoost":
        body = ", ".join(
            f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
            for k, v in best["config"].items())
        print(f"CAT_PARAMS = dict(\n    {body},\n"
              f"    verbose=0, allow_writing_files=False, thread_count=-1,\n)")
    else:
        print(f"winner is {best['family']}, not CatBoost:\n  {best['config']}")
    if best["derived"]:
        print("\nThe derived age block won. Port derived_block() into the "
              "pipeline's build_matrix() to carry the gain across.")

    print(f"\nelapsed {(time.time() - started) / 60:.1f} min; "
          f"wrote {args.out}")
    print("Every search score is mixture-grouped: no mixture appears in both "
          "a training fold and the fold that scores it.")


if __name__ == "__main__":
    main()
