#!/usr/bin/env python3
"""
ASR expansion: fast model search.

Optimises mixture-grouped cross-validated accuracy. No figures, no SHAP, no
conformal intervals, no feature elimination - only the search loop, so the
wall-clock cost goes into finding a better model.

Three stages:
  1. Screen many families and hyperparameter draws on fixed grouped folds.
  2. Spend extra budget refining the families that led the screen.
  3. Blend the best diverse models by non-negative least squares on their
     out-of-fold predictions, scored with a second grouped split so the blend
     weights are not evaluated on the data that fitted them.

The locked holdout is never touched. Keep it that way until the search is
finished, then score the winner once with the full pipeline.

Usage (Colab, GPU runtime):
    !python asr_model_search.py
Environment:
    ASR_CSV_PATH      dataset location
    ASR_SEARCH_BUDGET multiplier on the number of configurations (default 1.0)
    ASR_SKIP_INSTALL  set to 1 to skip dependency checks
"""

import importlib.metadata
import os
import subprocess
import sys
import time
import warnings
from pathlib import Path

REQUIREMENTS = [
    ("numpy", "1.26"), ("pandas", "2.2"), ("scipy", "1.13"),
    ("scikit-learn", "1.5"), ("lightgbm", "4.5"), ("catboost", "1.2"),
    ("xgboost", "2.1"),
]


def _ensure(requirements):
    missing = []
    for name, minimum in requirements:
        try:
            installed = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            missing.append(f"{name}>={minimum}")
            continue
        parts, want = [], []
        for text, store in ((installed, parts), (minimum, want)):
            for chunk in str(text).split(".")[:3]:
                digits = "".join(c for c in chunk if c.isdigit())
                store.append(int(digits) if digits else 0)
        if tuple(parts) < tuple(want):
            missing.append(f"{name}>={minimum}")
    if missing:
        print("installing:", ", ".join(missing))
        subprocess.check_call([sys.executable, "-m", "pip", "install",
                               "--quiet", *missing])


if os.environ.get("ASR_SKIP_INSTALL", "0") != "1":
    _ensure(REQUIREMENTS)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.optimize import nnls  # noqa: E402
from scipy.special import boxcox  # noqa: E402
from sklearn.cross_decomposition import PLSRegression  # noqa: E402
from sklearn.ensemble import (  # noqa: E402
    ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor,
)
from sklearn.exceptions import ConvergenceWarning  # noqa: E402
from sklearn.kernel_ridge import KernelRidge  # noqa: E402
from sklearn.linear_model import (  # noqa: E402
    BayesianRidge, ElasticNet, HuberRegressor, Ridge,
)
from sklearn.metrics import (  # noqa: E402
    mean_absolute_error, mean_squared_error, r2_score,
)
from sklearn.neighbors import KNeighborsRegressor  # noqa: E402
from sklearn.neural_network import MLPRegressor  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import QuantileTransformer, StandardScaler  # noqa: E402
from sklearn.svm import SVR  # noqa: E402

from catboost import CatBoostRegressor  # noqa: E402
from lightgbm import LGBMRegressor  # noqa: E402
from xgboost import XGBRegressor  # noqa: E402

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

try:
    import torch

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    torch = None
    DEVICE = "cpu"

os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")
try:
    from tabpfn import TabPFNRegressor

    HAS_TABPFN = True
except ImportError:
    TabPFNRegressor = None
    HAS_TABPFN = False

# =============================================================================
# Configuration
# =============================================================================
CSV_PATH = os.environ.get("ASR_CSV_PATH", "/content/ASR_FinalE-ComCo.csv")
TARGET = "y"
RANDOM_STATE = 256
TEST_SIZE = 0.20
AGE_COLUMN = "x29"
BUDGET = float(os.environ.get("ASR_SEARCH_BUDGET", "1.0"))

SCREEN_FOLDS = 5           # grouped folds used during the screen
FINAL_REPEATS = 2          # extra repeats when re-scoring the leaders
TOP_FOR_REFINE = 3         # families carried into stage 2
TOP_FOR_BLEND = 8          # models offered to the blender
RESULTS_PATH = Path("asr_model_search_results.csv")

NUMERIC_FEATURES = [
    "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x9", "x10",
    "x11", "x12", "x13", "x14", "x16", "x17", "x18", "x19", "x20",
    "X21", "x22", "x23", "x24", "x25", "x26", "x27", "x29",
]
CATEGORICAL_FEATURES = ["x15", "x28", "Standard"]
BASE_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
MK_COLS = ["x11", "x12", "x14"]
FA_COLS = ["x16", "x17", "x18", "x19", "X21"]
CNS_COLS = ["x22", "x23", "x24", "x26", "x27"]


# =============================================================================
# Data
# =============================================================================
def load_data(path):
    frame = pd.read_csv(path)
    missing = [c for c in BASE_FEATURES + [TARGET] if c not in frame.columns]
    if missing:
        raise ValueError(f"columns missing from {path}: {missing}")
    for column in NUMERIC_FEATURES + [TARGET]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    n_before = len(frame)
    frame = frame[frame[TARGET].notna()].reset_index(drop=True)
    print(f"loaded {path}: dropped {n_before - len(frame)} description row(s)")
    X = frame[BASE_FEATURES].reset_index(drop=True)
    y = frame[TARGET].astype(float).reset_index(drop=True)
    key_columns = [c for c in BASE_FEATURES if c != AGE_COLUMN]
    keys = pd.util.hash_pandas_object(
        X[key_columns].fillna("missing").astype(str), index=False
    )
    groups = pd.factorize(keys)[0]
    print(f"  {len(X)} rows, {pd.Series(groups).nunique()} mixtures")
    return X, y, groups


def noise_ceiling(X, y):
    """
    Estimate irreducible error from replicate rows.

    Rows sharing an identical predictor vector describe the same experiment,
    so any spread in their measured expansion is measurement and
    between-bar variability that no model can explain. The pooled
    within-replicate standard deviation is therefore a floor on achievable
    RMSE, and it converts to a ceiling on achievable R2.
    """
    frame = X.copy()
    frame["_y"] = np.asarray(y, float)
    keys = pd.util.hash_pandas_object(
        X.fillna("missing").astype(str), index=False
    )
    frame["_key"] = keys.to_numpy()
    sizes = frame.groupby("_key")["_y"].transform("size")
    replicates = frame.loc[sizes > 1]
    if replicates.empty:
        print("  replicate rows: none; noise ceiling cannot be estimated")
        return None
    deviations, counts = [], 0
    for _, block in replicates.groupby("_key"):
        values = block["_y"].to_numpy()
        deviations.append(values - values.mean())
        counts += 1
    residual = np.concatenate(deviations)
    degrees = max(len(residual) - counts, 1)
    sigma = float(np.sqrt(np.sum(residual ** 2) / degrees))
    total_variance = float(np.var(np.asarray(y, float), ddof=1))
    ceiling = 1.0 - (sigma ** 2) / total_variance if total_variance > 0 else np.nan
    print(f"  replicate sets: {counts} covering {len(residual)} rows")
    print(f"  irreducible RMSE >= {sigma:.4f} %  ->  achievable R2 <= "
          f"{ceiling:.4f}")
    return {"replicate_sets": counts, "irreducible_rmse": sigma,
            "max_r2": ceiling}


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
    return [(np.flatnonzero(row_fold != f), np.flatnonzero(row_fold == f))
            for f in range(n_folds)]


def grouped_holdout(groups, test_size, seed):
    groups = np.asarray(groups)
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(np.unique(groups))
    membership = pd.Series(groups)
    target_rows, held, count = int(round(test_size * len(groups))), [], 0
    for group in shuffled:
        if count >= target_rows:
            break
        held.append(group)
        count += int((membership == group).sum())
    mask = membership.isin(held).to_numpy()
    return np.flatnonzero(~mask), np.flatnonzero(mask)


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
    categories = {}
    for column in CATEGORICAL_FEATURES:
        values = train_df[column].fillna("Unknown").astype(str).str.strip()
        levels = sorted(v for v in values.unique() if v != "Unknown")
        categories[column] = levels + ["Unknown"]
    return {"fill": fill, "categories": categories}


def build_matrices(raw, state):
    processed = raw.copy()
    for column, value in state["fill"].items():
        processed[column] = processed[column].fillna(value)
    for column, levels in state["categories"].items():
        values = processed[column].fillna("Unknown").astype(str).str.strip()
        processed[column] = values.where(values.isin(levels), "Unknown")
    numeric = processed[NUMERIC_FEATURES].astype(float)
    onehot_parts, ordinal = [numeric], numeric.copy()
    for column in CATEGORICAL_FEATURES:
        categorical = pd.Categorical(processed[column],
                                     categories=state["categories"][column])
        dummies = pd.get_dummies(categorical, prefix=column, prefix_sep="=",
                                 dtype=float)
        dummies.index = processed.index
        onehot_parts.append(dummies)
        ordinal[column] = categorical.codes.astype(float)
    return pd.concat(onehot_parts, axis=1), ordinal


# Target transforms. The published run found the identity transform gave the
# lowest grouped-CV RMSE, so it is the default and the others are searched.
def transform_pair(name, y_train):
    shift = float(np.min(y_train) - 1e-4)
    if name == "identity":
        return (lambda v: np.asarray(v, float)), (lambda v: np.asarray(v, float))
    if name == "log1p":
        return ((lambda v: np.log1p(np.clip(np.asarray(v, float) - shift, 0, None))),
                (lambda v: np.expm1(np.asarray(v, float)) + shift))
    if name == "sqrt":
        return ((lambda v: np.sqrt(np.clip(np.asarray(v, float) - shift, 0, None))),
                (lambda v: np.clip(np.asarray(v, float), 0, None) ** 2 + shift))
    lam = 0.85
    return ((lambda v: boxcox(np.clip(np.asarray(v, float) - shift, 1e-12, None), lam)),
            (lambda v: np.clip(lam * np.asarray(v, float) + 1.0, 1e-9, None)
             ** (1.0 / lam) + shift))


# =============================================================================
# Model zoo
# =============================================================================
# Each family returns (estimator, matrix_kind). "ordinal" gives one column per
# input; "onehot" expands the three categorical inputs.
def sample_config(family, rng):
    """Draw one hyperparameter configuration for a family."""
    if family == "LightGBM":
        return dict(
            n_estimators=int(rng.choice([400, 800, 1500, 2900])),
            learning_rate=float(rng.choice([0.01, 0.02, 0.05, 0.08])),
            num_leaves=int(rng.choice([15, 31, 63, 127])),
            max_depth=int(rng.choice([4, 6, 8, -1])),
            min_child_samples=int(rng.choice([5, 10, 20, 40])),
            subsample=float(rng.choice([0.6, 0.8, 1.0])),
            colsample_bytree=float(rng.choice([0.6, 0.8, 1.0])),
            reg_alpha=float(rng.choice([0.0, 0.1, 1.0])),
            reg_lambda=float(rng.choice([0.0, 1.0, 5.0])),
        )
    if family == "XGBoost":
        return dict(
            n_estimators=int(rng.choice([400, 800, 1800])),
            learning_rate=float(rng.choice([0.01, 0.03, 0.06])),
            max_depth=int(rng.choice([3, 5, 7, 9])),
            min_child_weight=float(rng.choice([1, 2, 5, 10])),
            subsample=float(rng.choice([0.6, 0.8, 1.0])),
            colsample_bytree=float(rng.choice([0.6, 0.8, 1.0])),
            reg_alpha=float(rng.choice([0.0, 0.1, 1.0])),
            reg_lambda=float(rng.choice([0.5, 2.0, 5.0])),
        )
    if family == "CatBoost":
        return dict(
            iterations=int(rng.choice([500, 1200, 2600])),
            learning_rate=float(rng.choice([0.02, 0.05, 0.08])),
            depth=int(rng.choice([4, 6, 8, 10])),
            l2_leaf_reg=float(rng.choice([1.0, 3.0, 8.0, 15.0])),
        )
    if family == "HistGBM":
        return dict(
            max_iter=int(rng.choice([300, 600, 1200])),
            learning_rate=float(rng.choice([0.02, 0.05, 0.1])),
            max_leaf_nodes=int(rng.choice([15, 31, 63])),
            min_samples_leaf=int(rng.choice([5, 10, 20])),
            l2_regularization=float(rng.choice([0.0, 0.5, 2.0])),
        )
    if family in ("RandomForest", "ExtraTrees"):
        return dict(
            n_estimators=int(rng.choice([300, 600])),
            max_depth=int(rng.choice([8, 16, 32, 0])) or None,
            min_samples_leaf=int(rng.choice([1, 2, 4, 8])),
            max_features=float(rng.choice([0.3, 0.6, 1.0])),
        )
    if family == "MLP":
        return dict(
            hidden_layer_sizes=[
                (128, 64), (256, 128), (256, 128, 64), (512, 256),
                (128, 128, 128), (64, 64),
            ][int(rng.integers(0, 6))],
            alpha=float(10 ** rng.uniform(-5, -1)),
            learning_rate_init=float(10 ** rng.uniform(-3.5, -2)),
            activation=str(rng.choice(["relu", "tanh"])),
            batch_size=int(rng.choice([32, 64, 128])),
            scaler=str(rng.choice(["standard", "quantile"])),
        )
    if family == "SVR":
        return dict(
            C=float(10 ** rng.uniform(-1, 3)),
            gamma=float(10 ** rng.uniform(-3, 0)),
            epsilon=float(10 ** rng.uniform(-3, -1)),
            kernel=str(rng.choice(["rbf", "rbf", "poly"])),
        )
    if family == "KernelRidge":
        return dict(
            alpha=float(10 ** rng.uniform(-4, 1)),
            gamma=float(10 ** rng.uniform(-3, 0)),
            kernel=str(rng.choice(["rbf", "laplacian"])),
        )
    if family == "KNN":
        return dict(
            n_neighbors=int(rng.choice([3, 5, 10, 20])),
            weights=str(rng.choice(["distance", "uniform"])),
            p=int(rng.choice([1, 2])),
        )
    if family == "ElasticNet":
        return dict(alpha=float(10 ** rng.uniform(-4, 0)),
                    l1_ratio=float(rng.uniform(0.05, 0.95)))
    if family == "PLS":
        return dict(n_components=int(rng.integers(2, 20)))
    return {}


def build_estimator(family, config, seed):
    """Return (estimator, matrix_kind)."""
    if family == "LightGBM":
        return LGBMRegressor(**config, random_state=seed, n_jobs=-1,
                             verbose=-1), "ordinal"
    if family == "XGBoost":
        return XGBRegressor(**config, random_state=seed, n_jobs=-1,
                            verbosity=0, tree_method="hist",
                            objective="reg:squarederror"), "ordinal"
    if family == "CatBoost":
        return CatBoostRegressor(**config, random_seed=seed, verbose=0,
                                 allow_writing_files=False,
                                 thread_count=-1), "ordinal"
    if family == "HistGBM":
        return HistGradientBoostingRegressor(**config,
                                             random_state=seed), "ordinal"
    if family == "RandomForest":
        return RandomForestRegressor(**config, random_state=seed,
                                     n_jobs=-1), "ordinal"
    if family == "ExtraTrees":
        return ExtraTreesRegressor(**config, random_state=seed,
                                   n_jobs=-1), "ordinal"
    if family == "MLP":
        settings = dict(config)
        scaler = (QuantileTransformer(output_distribution="normal",
                                      random_state=seed)
                  if settings.pop("scaler") == "quantile" else StandardScaler())
        return make_pipeline(scaler, MLPRegressor(
            **settings, solver="adam", max_iter=3000, early_stopping=True,
            validation_fraction=0.15, n_iter_no_change=40, tol=1e-6,
            random_state=seed)), "onehot"
    if family == "SVR":
        return make_pipeline(StandardScaler(), SVR(**config)), "onehot"
    if family == "KernelRidge":
        return make_pipeline(StandardScaler(),
                             KernelRidge(**config)), "onehot"
    if family == "KNN":
        return make_pipeline(StandardScaler(),
                             KNeighborsRegressor(**config, n_jobs=-1)), "onehot"
    if family == "ElasticNet":
        return make_pipeline(StandardScaler(),
                             ElasticNet(**config, max_iter=20000,
                                        random_state=seed)), "onehot"
    if family == "Ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=1.0)), "onehot"
    if family == "BayesianRidge":
        return make_pipeline(StandardScaler(), BayesianRidge()), "onehot"
    if family == "Huber":
        return make_pipeline(StandardScaler(),
                             HuberRegressor(max_iter=2000)), "onehot"
    if family == "PLS":
        return make_pipeline(StandardScaler(),
                             PLSRegression(**config)), "onehot"
    if family == "TabPFN":
        kwargs = dict(device=DEVICE, random_state=seed)
        if torch is not None:
            kwargs["inference_precision"] = torch.float32
        for attempt in ({**kwargs, "n_estimators": config.get("n_estimators", 8)},
                        kwargs, {"device": DEVICE}):
            try:
                return TabPFNRegressor(**attempt), "ordinal"
            except TypeError:
                continue
        raise RuntimeError("could not construct TabPFNRegressor")
    raise KeyError(family)


# Families that need no search (fixed form) versus those worth sampling.
FIXED_FAMILIES = ["Ridge", "BayesianRidge", "Huber"]
SEARCH_FAMILIES = [
    "LightGBM", "XGBoost", "CatBoost", "HistGBM", "RandomForest",
    "ExtraTrees", "MLP", "SVR", "KernelRidge", "KNN", "ElasticNet", "PLS",
]
SCREEN_DRAWS = {
    "LightGBM": 8, "XGBoost": 8, "CatBoost": 6, "HistGBM": 6,
    "RandomForest": 4, "ExtraTrees": 4, "MLP": 10, "SVR": 10,
    "KernelRidge": 6, "KNN": 4, "ElasticNet": 4, "PLS": 4,
}
TRANSFORMS = ["identity", "log1p", "sqrt", "boxcox"]


# =============================================================================
# Evaluation
# =============================================================================
def evaluate(family, config, transform, folds, X, y, seed, collect=True):
    """Grouped cross-validation of one configuration. Returns metrics + OOF."""
    y_values = y.to_numpy()
    oof = np.full(len(y_values), np.nan)
    for train_index, test_index in folds:
        state = fit_preprocessing(X.iloc[train_index])
        onehot_tr, ordinal_tr = build_matrices(X.iloc[train_index], state)
        onehot_te, ordinal_te = build_matrices(X.iloc[test_index], state)
        forward, inverse = transform_pair(transform, y_values[train_index])
        estimator, kind = build_estimator(family, config, seed)
        matrix_tr = ordinal_tr if kind == "ordinal" else onehot_tr
        matrix_te = ordinal_te if kind == "ordinal" else onehot_te
        estimator.fit(matrix_tr.to_numpy(),
                      forward(y_values[train_index]).ravel())
        prediction = np.asarray(
            inverse(np.asarray(estimator.predict(matrix_te.to_numpy())).ravel()),
            float,
        )
        if not np.isfinite(prediction).all():
            return None
        oof[test_index] = prediction
    if not np.isfinite(oof).all():
        return None
    result = {
        "R2": float(r2_score(y_values, oof)),
        "RMSE": float(np.sqrt(mean_squared_error(y_values, oof))),
        "MAE": float(mean_absolute_error(y_values, oof)),
    }
    return (result, oof) if collect else (result, None)


# =============================================================================
# Search
# =============================================================================
def scaled(count):
    return max(1, int(round(count * BUDGET)))


def run_search():
    started = time.time()
    X_all, y_all, groups_all = load_data(CSV_PATH)
    ceiling = noise_ceiling(X_all, y_all)

    # Reserve the same locked holdout the full pipeline uses, then ignore it.
    dev_index, holdout_index = grouped_holdout(groups_all, TEST_SIZE,
                                               RANDOM_STATE)
    X = X_all.iloc[dev_index].reset_index(drop=True)
    y = y_all.iloc[dev_index].reset_index(drop=True)
    groups = groups_all[dev_index]
    print(f"  development: {len(X)} rows / {pd.Series(groups).nunique()} "
          f"mixtures  (holdout of {len(holdout_index)} rows left untouched)\n")

    folds = grouped_folds(groups, SCREEN_FOLDS, RANDOM_STATE)
    rng = np.random.default_rng(RANDOM_STATE)
    records, oof_store = [], {}

    def trial(family, config, transform, tag):
        key = f"{family}|{transform}|{tag}"
        try:
            outcome = evaluate(family, config, transform, folds, X, y,
                               RANDOM_STATE)
        except Exception as error:
            print(f"  {family:<13} {transform:<8} failed: "
                  f"{type(error).__name__}")
            return
        if outcome is None:
            return
        result, oof = outcome
        records.append({"family": family, "transform": transform,
                        "config": config, "key": key, **result})
        oof_store[key] = oof
        print(f"  {family:<13} {transform:<8} R2={result['R2']:.4f}  "
              f"RMSE={result['RMSE']:.4f}  MAE={result['MAE']:.4f}")

    print("=" * 72)
    print("STAGE 1  SCREEN")
    print("=" * 72)
    for family in FIXED_FAMILIES:
        for transform in ["identity", "boxcox"]:
            trial(family, {}, transform, "fixed")
    if HAS_TABPFN:
        for transform in TRANSFORMS:
            trial("TabPFN", {"n_estimators": 8}, transform, "default")
    for family in SEARCH_FAMILIES:
        for draw in range(scaled(SCREEN_DRAWS[family])):
            transform = str(rng.choice(TRANSFORMS))
            trial(family, sample_config(family, rng), transform, f"s{draw}")

    if not records:
        raise RuntimeError("no configuration completed successfully")

    leaderboard = pd.DataFrame(records).sort_values("R2", ascending=False)
    print("\ntop of screen")
    print(leaderboard.head(10)[["family", "transform", "R2", "RMSE", "MAE"]]
          .to_string(index=False, float_format=lambda v: f"{v:8.4f}"))

    print("\n" + "=" * 72)
    print("STAGE 2  REFINE LEADING FAMILIES")
    print("=" * 72)
    best_by_family = leaderboard.groupby("family")["R2"].max().sort_values(
        ascending=False)
    leaders = [f for f in best_by_family.index[:TOP_FOR_REFINE]
               if f in SEARCH_FAMILIES]
    print(f"refining: {', '.join(leaders) if leaders else 'none'}")
    for family in leaders:
        best_transform = leaderboard.loc[
            leaderboard["family"].eq(family)
        ].iloc[0]["transform"]
        for draw in range(scaled(12)):
            transform = (best_transform if draw % 3 else str(rng.choice(TRANSFORMS)))
            trial(family, sample_config(family, rng), transform, f"r{draw}")

    leaderboard = pd.DataFrame(records).sort_values("R2", ascending=False)

    print("\n" + "=" * 72)
    print("STAGE 3  BLEND")
    print("=" * 72)
    # Take the best configuration of each distinct family so the blend gets
    # genuinely different error patterns rather than near-duplicates.
    best_rows = (leaderboard.sort_values("R2", ascending=False)
                 .drop_duplicates("family").head(TOP_FOR_BLEND))
    keys = best_rows["key"].tolist()
    matrix = np.column_stack([oof_store[k] for k in keys])
    y_values = y.to_numpy()

    # Weights are fitted inside a second grouped split and scored on the held
    # part, so the blend is not judged on the rows that chose its weights.
    blend_folds = grouped_folds(groups, SCREEN_FOLDS, RANDOM_STATE + 5150)
    blend_oof = np.full(len(y_values), np.nan)
    for train_index, test_index in blend_folds:
        weights, _ = nnls(matrix[train_index], y_values[train_index])
        blend_oof[test_index] = matrix[test_index] @ weights
    blend_result = {
        "R2": float(r2_score(y_values, blend_oof)),
        "RMSE": float(np.sqrt(mean_squared_error(y_values, blend_oof))),
        "MAE": float(mean_absolute_error(y_values, blend_oof)),
    }
    full_weights, _ = nnls(matrix, y_values)
    print("members and weights:")
    for key, weight in zip(keys, full_weights):
        print(f"  {weight:6.3f}  {key}")
    print(f"blend    R2={blend_result['R2']:.4f}  "
          f"RMSE={blend_result['RMSE']:.4f}  MAE={blend_result['MAE']:.4f}")

    print("\n" + "=" * 72)
    print("STAGE 4  RE-SCORE LEADERS ON FRESH FOLDS")
    print("=" * 72)
    # The screen picked winners on one fold set, so the winning score is
    # optimistic. Re-scoring the leaders on different folds shows how much of
    # the advantage survives.
    finalists = leaderboard.head(5).to_dict("records")
    confirm_rows = []
    for entry in finalists:
        scores = []
        for repeat in range(FINAL_REPEATS):
            confirm_folds = grouped_folds(groups, SCREEN_FOLDS,
                                          RANDOM_STATE + 9000 + 137 * repeat)
            outcome = evaluate(entry["family"], entry["config"],
                               entry["transform"], confirm_folds, X, y,
                               RANDOM_STATE, collect=False)
            if outcome:
                scores.append(outcome[0])
        if not scores:
            continue
        confirm_rows.append({
            "family": entry["family"], "transform": entry["transform"],
            "screen_R2": entry["R2"],
            "confirm_R2": float(np.mean([s["R2"] for s in scores])),
            "confirm_RMSE": float(np.mean([s["RMSE"] for s in scores])),
            "confirm_MAE": float(np.mean([s["MAE"] for s in scores])),
            "config": entry["config"],
        })
    confirmed = pd.DataFrame(confirm_rows).sort_values("confirm_R2",
                                                       ascending=False)
    print(confirmed[["family", "transform", "screen_R2", "confirm_R2",
                     "confirm_RMSE", "confirm_MAE"]].to_string(
        index=False, float_format=lambda v: f"{v:9.4f}"))

    output = leaderboard.copy()
    output["config"] = output["config"].astype(str)
    output.to_csv(RESULTS_PATH, index=False)
    confirmed_path = RESULTS_PATH.with_name("asr_model_search_confirmed.csv")
    confirmed_out = confirmed.copy()
    confirmed_out["config"] = confirmed_out["config"].astype(str)
    confirmed_out.to_csv(confirmed_path, index=False)

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    if len(confirmed):
        best = confirmed.iloc[0]
        best_score, score_label = best["confirm_R2"], "confirmed on fresh folds"
    else:
        best = leaderboard.iloc[0]
        best_score, score_label = best["R2"], "screen only"
    print(f"configurations evaluated : {len(records)}")
    print(f"best single model        : {best['family']} "
          f"({best['transform']})  R2={best_score:.4f}  [{score_label}]")
    # The blend is scored on the screen fold set, so compare it with the
    # screen column rather than the confirmed one.
    print(f"blend of {len(keys)} families      : R2={blend_result['R2']:.4f}  "
          f"[screen fold basis; compare with screen_R2]")
    if ceiling and np.isfinite(ceiling["max_r2"]):
        print(f"replicate-noise ceiling  : R2 <= {ceiling['max_r2']:.4f}  "
              f"(RMSE >= {ceiling['irreducible_rmse']:.4f} %)")
    print(f"elapsed                  : {(time.time() - started) / 60:.1f} min")
    print(f"wrote {RESULTS_PATH} and {confirmed_path}")
    print("\nThe locked holdout was not touched. Score the winner on it once, "
          "with the full pipeline, when the search is finished.")
    return leaderboard, confirmed, blend_result


if __name__ == "__main__":
    run_search()
