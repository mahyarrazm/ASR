# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Machine-learning prediction of ASR mortar expansion
#
# Stacked ensemble (LightGBM + CatBoost + TabPFN → BayesianRidge meta-learner)
# for predicting alkali–silica reaction (ASR) mortar-bar expansion from
# mixture, binder-composition, and exposure descriptors.
#
# **Modeling protocol**
#
# | Component | Setting |
# |---|---|
# | Features | 17 base + 5 engineered = 22 (ablation-selected) |
# | Imputation | conditional median by SCM presence + FA-class mode, refit per fold |
# | Target transform | Box–Cox (λ = 0.85), shift refit per fold |
# | Ensemble | LightGBM + CatBoost + TabPFN, BayesianRidge meta-learner |
# | Stacking | fold-pure: meta-learner trained on inner out-of-fold predictions only |
# | Evaluation | 10-fold stratified CV (y-quantile bins) + 80/20 holdout (seed 256) |
#
# **Analysis suite and outputs**
#
# | Analysis | Figure | Table |
# |---|---|---|
# | Dataset overview | Fig. S1 | Table S0 |
# | 10-fold fold-pure cross-validation | Fig. 10a | Table S1 |
# | Baseline / single-model comparison | Fig. 4 | Table S2 |
# | 80/20 holdout (parity, residual diagnostics) | Figs. 1, 6 | Table S4 |
# | Split-conformal 90 % prediction intervals | Fig. 9 | Table S9 |
# | SHAP feature attribution | Figs. 2, 3 | Table S5 |
# | Feature–target correlation structure | Fig. 7 | Table S6 |
# | Learning curve (data efficiency) | Fig. 5 | Table S7 |
# | y-randomization (chance-performance test) | Fig. 8 | Table S8 |
# | Multi-seed holdout stability | Fig. 10b | Table S3 |
#
# All figures are rendered inline and saved to `figures/` (600-dpi PNG +
# vector PDF); all tables are saved to `results/` (CSV). A ZIP archive of both
# folders is created at the end.
#
# **Usage (Google Colab).** Select a GPU runtime (recommended for TabPFN),
# upload the dataset CSV to `/content/`, add your TabPFN license token as a
# Colab secret named `TABPFN_TOKEN` (key icon in the sidebar, notebook access
# ON; free token from https://ux.priorlabs.ai/account), then *Runtime → Run
# all*. Without a token or without `tabpfn` installed, the pipeline
# automatically falls back to the two-learner (LightGBM + CatBoost) stack.

# %% [markdown]
# ## 1. Setup

# %%
import importlib.util
import os
import subprocess
import sys


def _ensure(packages):
    missing = [p for p in packages
               if importlib.util.find_spec(p.replace("-", "_")) is None]
    if missing:
        subprocess.check_call([sys.executable, "-m", "pip", "install",
                               "--quiet", *missing])


if os.environ.get("ASR_SKIP_INSTALL", "0") != "1":
    _ensure(["lightgbm", "catboost", "shap"])
    try:  # TabPFN is optional: the ensemble degrades gracefully without it
        _ensure(["tabpfn"])
    except Exception as exc:
        print(f"tabpfn could not be installed ({exc}); continuing without it.")

# %%
import gc
import platform
import random
import warnings
import zipfile
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st
import sklearn
from scipy.special import boxcox
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import BayesianRidge, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

print(f"python      : {platform.python_version()}")
for _m in (np, pd, sklearn, mpl):
    print(f"{_m.__name__:<12}: {_m.__version__}")
import lightgbm as _lgb, catboost as _cb  # noqa: E401
print(f"lightgbm    : {_lgb.__version__}")
print(f"catboost    : {_cb.__version__}")

# %% [markdown]
# ## 2. Configuration

# %%
CSV_PATH     = os.environ.get("ASR_CSV_PATH", "/content/ASR_FinalE-ComCo.csv")
TARGET       = "y"
TARGET_LABEL = "expansion (%)"   # axis label used in figures
UNIT         = "%"

RANDOM_STATE = 256               # holdout split + model seeds
TEST_SIZE    = 0.20
CV_FOLDS     = 10
LAMBDA_BC    = 0.85              # Box-Cox exponent for the target
EPS_SHIFT    = 1e-4              # shift so the transformed target is > 0

USE_TABPFN    = True             # False reproduces the 2-learner stack
TABPFN_DEVICE = "auto"           # "cpu", "cuda", or "auto"

CONF_ALPHA = 0.10                # 1 - target coverage (0.10 -> 90 % intervals)
STABILITY_SEEDS = [256, 7, 42, 101, 2024]
N_PERMUTATIONS = 10              # y-randomization repeats
LEARNING_CURVE_SIZES = [0.2, 0.4, 0.6, 0.8, 1.0]

RUN_MODEL_COMPARISON = True
RUN_CONFORMAL        = True
RUN_SHAP             = True
RUN_LEARNING_CURVE   = True
RUN_Y_RANDOMIZATION  = True
RUN_SEED_STABILITY   = True

# Map feature codes to descriptive names for figure labels (codes are kept
# where no entry is given); edit to match the manuscript's variable names.
FEATURE_LABELS = {}

FAST_MODE = os.environ.get("ASR_FAST_MODE", "0") == "1"  # reduced smoke test
if FAST_MODE:
    CV_FOLDS, N_PERMUTATIONS = 3, 2
    STABILITY_SEEDS = STABILITY_SEEDS[:2]
    LEARNING_CURVE_SIZES = [0.5, 1.0]

FIG_DIR = Path("figures")
RES_DIR = Path("results")
FIG_DIR.mkdir(exist_ok=True)
RES_DIR.mkdir(exist_ok=True)

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

# %% [markdown]
# ## 3. Figure style (journal defaults)

# %%
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.5,
    "legend.frameon": False,
    "figure.dpi": 110,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,  # embed TrueType fonts -> editable text in PDFs
    "ps.fonttype": 42,
})

# Okabe-Ito colorblind-safe palette
C_BLUE, C_ORANGE, C_GREEN, C_RED, C_PURPLE, C_GREY = (
    "#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#7F7F7F",
)


def save_fig(fig, name):
    """Save as 600-dpi PNG + vector PDF, then render inline."""
    fig.savefig(FIG_DIR / f"{name}.png")
    fig.savefig(FIG_DIR / f"{name}.pdf")
    print(f"saved {FIG_DIR}/{name}.png/.pdf")
    plt.show()
    plt.close(fig)

# %% [markdown]
# ## 4. Features
#
# 27 base columns are loaded from the CSV. Ten of them were removed from the
# model input by the ablation study (FA composition/class and CNS composition
# descriptors), but remain loaded because `X21`/`x27` feed the engineered
# `SCM_alkali` feature and `x15` is needed for imputation. Five engineered
# features are added, giving 22 active model features.

# %%
BASE_FEATURES_LOADED = [
    "x1", "x2", "x3", "x4", "x5", "x7", "x9", "x10",
    "x11", "x12", "x13", "x14", "x15", "x16", "x17", "x18", "x19", "x20",
    "X21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "x29",
]

DROPPED_FROM_BASELINE = [
    "x15", "x16", "x17", "x19", "X21",   # FA composition / class
    "x22", "x23", "x24", "x26", "x27",   # CNS composition
]

ENGINEERED = ["total_SCM", "age_x_temp", "silica_x_alkali", "log_age", "SCM_alkali"]

_ALL_BASELINE = BASE_FEATURES_LOADED + ENGINEERED
ALL_FEATURES  = [f for f in _ALL_BASELINE if f not in DROPPED_FROM_BASELINE]  # 22
DISPLAY_NAMES = [FEATURE_LABELS.get(f, f) for f in ALL_FEATURES]

# imputation groups: SCM composition columns conditioned on their content column
mk_prop_cols  = ["x11", "x12", "x14"]                # gated by x13 (MK content %)
fa_prop_cols  = ["x16", "x17", "x18", "x19", "X21"]  # gated by x20 (FA content %)
cns_prop_cols = ["x22", "x23", "x24", "x26", "x27"]  # gated by x25 (CNS content %)

# %% [markdown]
# ## 5. Preprocessing and target transform
#
# Both are *fold-pure*: imputation statistics and the Box–Cox shift are
# learned from the training portion of each split only, and then applied to
# the corresponding validation/test portion.

# %%
def fit_preprocessing(train_df):
    """Learn fill values from the training split only."""
    fill = {}

    def cond_median(prop_cols, content_col):
        present = train_df[content_col].fillna(0) > 0
        for c in prop_cols:
            src = train_df.loc[present, c].dropna()
            fill[c] = float(src.median()) if len(src) else 0.0

    cond_median(mk_prop_cols,  "x13")
    cond_median(fa_prop_cols,  "x20")
    cond_median(cns_prop_cols, "x25")

    for c in ["x13", "x20", "x25"]:
        src = train_df[c].dropna()
        fill[c] = float(src.median()) if len(src) else 0.0

    fa_mode = train_df.loc[train_df["x20"].fillna(0) > 0, "x15"].dropna().mode()
    x15_fill = float(fa_mode.iloc[0]) if len(fa_mode) else 1.0

    for c in BASE_FEATURES_LOADED:
        if c == "x15" or c in fill:
            continue
        src = train_df[c].dropna()
        fill[c] = float(src.median()) if len(src) else 0.0

    return {"fill": fill, "x15_fill": x15_fill}


def apply_preprocessing(df, state):
    out = df.copy()
    for c, v in state["fill"].items():
        if c in out.columns:
            out[c] = out[c].fillna(v)
    out["x15"] = out["x15"].fillna(state["x15_fill"])
    return out


def add_engineered(df):
    d = df.copy()
    d["total_SCM"]       = d["x13"] + d["x20"] + d["x25"]
    d["age_x_temp"]      = d["x29"] * d["x7"]
    d["silica_x_alkali"] = d["x1"]  * d["x5"]
    d["log_age"]         = np.log1p(d["x29"].clip(lower=0))
    d["SCM_alkali"]      = d["x14"] + d["X21"] + d["x27"]
    return d


def build_matrix(raw_df, state):
    """raw -> impute -> engineer -> select the 22 active features."""
    return add_engineered(apply_preprocessing(raw_df, state))[ALL_FEATURES]


def fit_bc_shift(y_train):
    """shift = train_min - eps, so (y - shift) >= eps > 0 for Box-Cox."""
    return float(np.min(y_train) - EPS_SHIFT)


def bc_forward(y, shift, lam=LAMBDA_BC):
    return boxcox(np.asarray(y, float) - shift, lam)


def bc_inverse(z, shift, lam=LAMBDA_BC, eps=1e-9):
    base = np.clip(lam * np.asarray(z, float) + 1.0, eps, None) ** (1.0 / lam)
    return base + shift

# %% [markdown]
# ## 6. Models
#
# The stack uses scikit-learn's `StackingRegressor`: each base learner
# produces inner out-of-fold predictions (5-fold, fixed seed), the
# BayesianRidge meta-learner is fitted on those leak-free predictions only,
# and the base learners are then refit on the full training data for
# inference. `n_jobs=1` across the stack avoids nesting joblib parallelism
# with GPU-backed learners.

# %%
LGBM_PARAMS = dict(
    n_estimators=2900, max_depth=8, learning_rate=0.0561,
    subsample=0.624, colsample_bytree=0.783, reg_alpha=0.030, reg_lambda=2.884,
    min_child_samples=7, num_leaves=32, n_jobs=-1, verbose=-1,
)
CAT_PARAMS = dict(
    iterations=2600, depth=10, learning_rate=0.0825,
    l2_leaf_reg=15.34, bagging_temperature=0.242, random_strength=0.414,
    verbose=0, allow_writing_files=False,
)
if FAST_MODE:
    LGBM_PARAMS["n_estimators"] = 300
    CAT_PARAMS["iterations"] = 300


def resolve_device(spec=TABPFN_DEVICE):
    if spec != "auto":
        return spec
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def make_lgbm(random_state=RANDOM_STATE):
    return LGBMRegressor(**LGBM_PARAMS, random_state=random_state + 1)


def make_catboost(random_state=RANDOM_STATE):
    return CatBoostRegressor(**CAT_PARAMS, random_seed=random_state + 2)


def make_tabpfn(random_state=RANDOM_STATE):
    from tabpfn import TabPFNRegressor
    return TabPFNRegressor(device=resolve_device(), random_state=random_state,
                           ignore_pretraining_limits=True)


def make_stack(use_tabpfn, random_state=RANDOM_STATE):
    estimators = [("lgbm", make_lgbm(random_state)),
                  ("catboost", make_catboost(random_state))]
    if use_tabpfn:
        estimators.append(("tabpfn", make_tabpfn(random_state)))
    return StackingRegressor(
        estimators=estimators,
        final_estimator=BayesianRidge(),
        cv=KFold(n_splits=5, shuffle=True, random_state=random_state),
        n_jobs=1,
        passthrough=False,
    )


def cleanup():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

# %% [markdown]
# ### TabPFN availability
#
# TabPFN weights require a one-time free license token. The token is read
# from the environment (`TABPFN_TOKEN`) or from a Colab secret — it is never
# hard-coded in this notebook. A small probe fit verifies that the weights
# can actually be loaded; otherwise the pipeline falls back to the
# two-learner stack.

# %%
def bootstrap_tabpfn_token():
    if os.environ.get("TABPFN_TOKEN"):
        return True
    try:
        from google.colab import userdata  # only available inside Colab
        candidates = ("TABPFN_TOKEN", "TABPFN", "TABPFN_1", "TABPFN_T",
                      "TABPFN_KEY", "TABPFN_API_KEY", "PRIORLABS_TOKEN")
        for name in candidates:
            try:
                tok = userdata.get(name)
            except Exception:
                tok = None
            if tok:
                os.environ["TABPFN_TOKEN"] = tok
                print(f"TabPFN token loaded from Colab secret '{name}'.")
                return True
        print("No TabPFN Colab secret found; name your secret TABPFN_TOKEN "
              "(key icon, notebook access ON).")
    except Exception:
        pass
    return bool(os.environ.get("TABPFN_TOKEN"))


def tabpfn_available():
    if not USE_TABPFN:
        return False
    try:
        import tabpfn  # noqa: F401
    except ImportError:
        print("tabpfn is not installed -> falling back to LGBM+CatBoost.")
        return False
    bootstrap_tabpfn_token()
    try:
        rng = np.random.default_rng(0)
        probe = make_tabpfn()
        probe.fit(rng.random((20, 3)), rng.random(20))
        probe.predict(rng.random((2, 3)))
        del probe
        cleanup()
        return True
    except Exception as exc:
        print("tabpfn is installed but unusable -> falling back to LGBM+CatBoost.")
        print(f"  reason: {type(exc).__name__}: {str(exc).splitlines()[0]}")
        return False


HAS_TABPFN = tabpfn_available()
ENSEMBLE_LABEL = "LGBM+CatBoost+TabPFN" if HAS_TABPFN else "LGBM+CatBoost"
print(f"ensemble: {ENSEMBLE_LABEL}"
      + (f" (TabPFN device: {resolve_device()})" if HAS_TABPFN else ""))

# %% [markdown]
# ## 7. Metrics

# %%
def metrics(y_true, y_pred):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    nz = np.abs(y_true) > 1e-8  # MAPE is undefined at zero expansion
    ape = (np.abs((y_true[nz] - y_pred[nz]) / y_true[nz]) * 100
           if nz.any() else np.array([np.nan]))
    return {
        "R2":     float(r2_score(y_true, y_pred)),
        "RMSE":   float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE":    float(mean_absolute_error(y_true, y_pred)),
        "MAPE":   float(np.mean(ape)),     # inflated near zero expansion
        "MedAPE": float(np.median(ape)),   # robust to small denominators
    }

# %% [markdown]
# ## 8. Data loading and overview (Fig. S1, Table S0)

# %%
def load_data(path=CSV_PATH):
    df = pd.read_csv(path)
    need = BASE_FEATURES_LOADED + [TARGET]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"Columns missing from {path}: {missing}")

    # Some exports carry a second header row of descriptions and/or rows with
    # no target. Coerce to numeric and drop rows whose target is missing;
    # genuine missing feature values are preserved (NaN) for the fold-pure
    # imputer.
    for c in need:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    n0 = len(df)
    df = df[df[TARGET].notna()].reset_index(drop=True)
    if n0 - len(df):
        print(f"dropped {n0 - len(df)} row(s) with non-numeric/missing target "
              f"(e.g. a description header row).")

    X_raw = df[BASE_FEATURES_LOADED].astype(float).reset_index(drop=True)
    y = df[TARGET].astype(float).reset_index(drop=True)
    print(f"Loaded {path}: raw X={X_raw.shape}, y={y.shape}  "
          f"(active model features = {len(ALL_FEATURES)})")
    return X_raw, y


X_raw, y = load_data()

# %%
# Table S0 — descriptive statistics of the loaded variables
overview = X_raw.copy()
overview[TARGET] = y
table_s0 = pd.DataFrame({
    "mean": overview.mean(), "std": overview.std(), "min": overview.min(),
    "median": overview.median(), "max": overview.max(),
    "missing_%": overview.isna().mean() * 100,
}).round(4)
table_s0.to_csv(RES_DIR / "Table_S0_feature_summary.csv")
table_s0

# %%
# Fig. S1 — target distribution and feature missingness
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))
axes[0].hist(y, bins=40, color=C_BLUE, alpha=0.85)
axes[0].set_xlabel(f"Measured {TARGET_LABEL}")
axes[0].set_ylabel("Count")
miss = (X_raw.isna().mean() * 100).sort_values(ascending=False)
miss = miss[miss > 0]
if len(miss):
    axes[1].bar(range(len(miss)), miss.values, color=C_ORANGE, alpha=0.9)
    axes[1].set_xticks(range(len(miss)), miss.index, rotation=90, fontsize=6)
    axes[1].set_ylabel("Missing values (%)")
else:
    axes[1].text(0.5, 0.5, "no missing values", ha="center", va="center",
                 transform=axes[1].transAxes)
    axes[1].set_axis_off()
for ax, letter in zip(axes, "ab"):
    ax.text(-0.14, 1.05, letter, transform=ax.transAxes,
            fontweight="bold", fontsize=10)
fig.tight_layout()
save_fig(fig, "FigS1_DataOverview")

# %% [markdown]
# ## 9. Cross-validation machinery
#
# A single set of stratified splits (y-quantile bins) is generated from the
# training portion of the 80/20 holdout and reused for every model, so all
# comparisons in this notebook share identical folds. Within each fold the
# imputer and the Box–Cox shift are refit on the fold's training portion.

# %%
def make_strata(y_vals, n_bins=CV_FOLDS):
    for q in range(n_bins, 1, -1):
        try:
            s = pd.qcut(y_vals, q=q, labels=False, duplicates="drop")
            if pd.Series(s).value_counts().min() >= 2:
                return np.asarray(s, dtype=int)
        except ValueError:
            continue
    return None


def make_cv_splits(Xtr_df, ytr_ser):
    strata = make_strata(ytr_ser)
    if strata is not None:
        splitter = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True,
                                   random_state=RANDOM_STATE)
        return list(splitter.split(Xtr_df, strata))
    splitter = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    return list(splitter.split(Xtr_df, ytr_ser))


def cross_validate(model_factory, Xtr_df, ytr_ser, splits, verbose=True):
    """Fold-pure CV of any regressor factory under the shared protocol."""
    oof = np.full(len(ytr_ser), np.nan)
    rows = []
    for fold, (tr, va) in enumerate(splits, 1):
        tr_raw, va_raw = Xtr_df.iloc[tr], Xtr_df.iloc[va]
        y_tr, y_va = ytr_ser.iloc[tr].to_numpy(), ytr_ser.iloc[va].to_numpy()

        state = fit_preprocessing(tr_raw)
        X_tr, X_va = build_matrix(tr_raw, state), build_matrix(va_raw, state)
        shift = fit_bc_shift(y_tr)

        model = model_factory()
        model.fit(X_tr.to_numpy(), bc_forward(y_tr, shift))
        pred = bc_inverse(model.predict(X_va.to_numpy()), shift)

        oof[va] = pred
        m = metrics(y_va, pred)
        rows.append(m)
        if verbose:
            print(f"  fold {fold:2d}:  R2={m['R2']:.4f}  RMSE={m['RMSE']:.4f}  "
                  f"MAE={m['MAE']:.4f}")

        del model, state, X_tr, X_va
        cleanup()
    return pd.DataFrame(rows, index=range(1, len(splits) + 1)), oof

# %% [markdown]
# ## 10. 10-fold fold-pure cross-validation of the stack (Table S1)

# %%
idx_tr, idx_te = train_test_split(np.arange(len(y)), test_size=TEST_SIZE,
                                  random_state=RANDOM_STATE)
Xtr_raw, Xte_raw = X_raw.iloc[idx_tr].reset_index(drop=True), \
                   X_raw.iloc[idx_te].reset_index(drop=True)
ytr, yte = y.iloc[idx_tr].reset_index(drop=True), \
           y.iloc[idx_te].reset_index(drop=True)
CV_SPLITS = make_cv_splits(Xtr_raw, ytr)

print(f"=== {CV_FOLDS}-fold fold-pure CV ({ENSEMBLE_LABEL}) ===")
cv_table, oof_pred = cross_validate(lambda: make_stack(HAS_TABPFN),
                                    Xtr_raw, ytr, CV_SPLITS)
print("\n  per-fold mean +/- std:")
for col in cv_table.columns:
    print(f"    {col:<6}: {cv_table[col].mean():.4f} +/- {cv_table[col].std():.4f}")
oof_metrics = metrics(ytr.to_numpy(), oof_pred)
print(f"  pooled OOF R2 = {oof_metrics['R2']:.4f}")
cv_table.round(4).to_csv(RES_DIR / "Table_S1_cv_per_fold_metrics.csv",
                         index_label="fold")

# %% [markdown]
# ## 11. Baseline and single-model comparison (Fig. 4, Table S2)
#
# Every model is evaluated with the same folds, the same fold-pure
# preprocessing, and the same Box–Cox target transform, so differences
# reflect the learner only. The stacked-ensemble row reuses the CV results
# from the previous section.

# %%
if RUN_MODEL_COMPARISON:
    def comparison_factories():
        f = {
            "Ridge (linear)": lambda: make_pipeline(
                StandardScaler(), Ridge(alpha=1.0)),
            "k-NN (k=5)": lambda: make_pipeline(
                StandardScaler(),
                KNeighborsRegressor(n_neighbors=5, weights="distance")),
            "Random Forest": lambda: RandomForestRegressor(
                n_estimators=500, min_samples_leaf=2,
                random_state=RANDOM_STATE, n_jobs=-1),
            "LightGBM": lambda: make_lgbm(),
            "CatBoost": lambda: make_catboost(),
        }
        if HAS_TABPFN:
            f["TabPFN"] = lambda: make_tabpfn()
        return f

    rows = []
    for name, factory in comparison_factories().items():
        table, _ = cross_validate(factory, Xtr_raw, ytr, CV_SPLITS, verbose=False)
        rows.append({"model": name,
                     **{f"{c}_mean": table[c].mean() for c in table.columns},
                     **{f"{c}_std": table[c].std() for c in table.columns}})
        print(f"  {name:<18}: R2={rows[-1]['R2_mean']:.4f}+/-{rows[-1]['R2_std']:.4f}  "
              f"RMSE={rows[-1]['RMSE_mean']:.4f}+/-{rows[-1]['RMSE_std']:.4f}")
    rows.append({"model": "Stacked ensemble",
                 **{f"{c}_mean": cv_table[c].mean() for c in cv_table.columns},
                 **{f"{c}_std": cv_table[c].std() for c in cv_table.columns}})
    print(f"  {'Stacked ensemble':<18}: R2={rows[-1]['R2_mean']:.4f}"
          f"+/-{rows[-1]['R2_std']:.4f}  "
          f"RMSE={rows[-1]['RMSE_mean']:.4f}+/-{rows[-1]['RMSE_std']:.4f}")
    comp = pd.DataFrame(rows).set_index("model")
    comp.round(4).to_csv(RES_DIR / "Table_S2_model_comparison.csv")

    order = comp["R2_mean"].sort_values().index
    colors = [C_ORANGE if m == "Stacked ensemble" else C_BLUE for m in order]
    ypos = np.arange(len(order))
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))
    axes[0].barh(ypos, comp.loc[order, "R2_mean"],
                 xerr=comp.loc[order, "R2_std"], color=colors, height=0.65,
                 error_kw={"lw": 0.8})
    axes[0].set_yticks(ypos, order)
    axes[0].set_xlabel(rf"$R^2$ ({CV_FOLDS}-fold CV)")
    axes[0].set_xlim(max(0.0, comp["R2_mean"].min() - 0.05), 1.0)
    axes[1].barh(ypos, comp.loc[order, "RMSE_mean"],
                 xerr=comp.loc[order, "RMSE_std"], color=colors, height=0.65,
                 error_kw={"lw": 0.8})
    axes[1].set_yticks(ypos, ["" for _ in order])
    axes[1].set_xlabel(f"RMSE ({CV_FOLDS}-fold CV), {UNIT}")
    for ax, letter in zip(axes, "ab"):
        ax.text(-0.05, 1.05, letter, transform=ax.transAxes,
                fontweight="bold", fontsize=10)
    fig.tight_layout()
    save_fig(fig, "Fig4_ModelComparison")

# %% [markdown]
# ## 12. Final fit and 80/20 holdout (Fig. 1, Table S4)

# %%
final_state = fit_preprocessing(Xtr_raw)
X_tr = build_matrix(Xtr_raw, final_state)
X_te = build_matrix(Xte_raw, final_state)
final_shift = fit_bc_shift(ytr.to_numpy())

final_stack = make_stack(HAS_TABPFN)
final_stack.fit(X_tr.to_numpy(), bc_forward(ytr.to_numpy(), final_shift))

pred_tr = bc_inverse(final_stack.predict(X_tr.to_numpy()), final_shift)
pred_te = bc_inverse(final_stack.predict(X_te.to_numpy()), final_shift)
m_tr, m_te = metrics(ytr.to_numpy(), pred_tr), metrics(yte.to_numpy(), pred_te)

print(f"=== final fit / 80-20 holdout (random_state={RANDOM_STATE}) ===")
print("  train:", {k: round(v, 4) for k, v in m_tr.items()})
print("  test :", {k: round(v, 4) for k, v in m_te.items()})
meta_names = [n for n, _ in final_stack.estimators]
meta_weights = {n: round(float(c), 4)
                for n, c in zip(meta_names, final_stack.final_estimator_.coef_)}
print("  meta (BayesianRidge) weights:", meta_weights)

pd.DataFrame([{"split": "train", **m_tr}, {"split": "test", **m_te}]).round(4) \
    .to_csv(RES_DIR / "Table_S4_holdout_metrics.csv", index=False)

# %%
# Fig. 1 — parity plots (train / test)
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6))
fig.subplots_adjust(wspace=0.38)
for ax, yt, yp, m, lbl, col in [
    (axes[0], ytr.to_numpy(), pred_tr, m_tr, "Training set", C_BLUE),
    (axes[1], yte.to_numpy(), pred_te, m_te, "Test set",     C_RED),
]:
    lim = [0, max(yt.max(), yp.max()) * 1.07]
    ax.plot(lim, lim, color="#333333", lw=1.0, ls="--", zorder=3, label="1:1 line")
    ax.scatter(yt, yp, s=14, c=col, alpha=0.45, lw=0, zorder=4,
               label=f"{lbl} (n={len(yt)})")
    txt = (f"$R^2$ = {m['R2']:.4f}\nRMSE = {m['RMSE']:.4f} {UNIT}\n"
           f"MAE = {m['MAE']:.4f} {UNIT}\nMedAPE = {m['MedAPE']:.1f} %")
    ax.text(0.97, 0.05, txt, transform=ax.transAxes, fontsize=8,
            va="bottom", ha="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#CCCCCC", lw=0.5, alpha=0.9))
    ax.set_xlim(lim), ax.set_ylim(lim)
    ax.set_aspect("equal")
    ax.set_xlabel(f"Measured {TARGET_LABEL}")
    ax.set_ylabel(f"Predicted {TARGET_LABEL}")
    ax.set_title(lbl, fontweight="bold", pad=4)
    ax.legend(loc="upper left")
for ax, letter in zip(axes, "ab"):
    ax.text(-0.14, 1.05, letter, transform=ax.transAxes,
            fontweight="bold", fontsize=10)
save_fig(fig, "Fig1_ActualVsPredicted")

# %% [markdown]
# ## 13. Residual diagnostics on the test set (Fig. 6)

# %%
resid = yte.to_numpy() - pred_te
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))
axes[0].axhline(0, color="#333333", lw=0.8, ls="--")
axes[0].scatter(pred_te, resid, s=10, c=C_RED, alpha=0.5, lw=0)
axes[0].set_xlabel(f"Predicted {TARGET_LABEL}")
axes[0].set_ylabel(f"Residual ({UNIT})")

axes[1].hist(resid, bins=30, color=C_RED, alpha=0.7, density=True,
             edgecolor="white", lw=0.4)
xs = np.linspace(resid.min(), resid.max(), 200)
axes[1].plot(xs, st.norm.pdf(xs, resid.mean(), resid.std()),
             color="#333333", lw=1)
axes[1].axvline(0, color="#333333", lw=0.8, ls="--")
axes[1].set_xlabel(f"Residual ({UNIT})")
axes[1].set_ylabel("Density")

st.probplot(resid, dist="norm", plot=axes[2])
axes[2].set_title("")
axes[2].get_lines()[0].set(marker="o", markersize=2.5, color=C_RED, alpha=0.6)
axes[2].get_lines()[1].set(color="#333333", lw=1)
axes[2].set_xlabel("Theoretical quantiles")
axes[2].set_ylabel("Ordered residuals")
for ax, letter in zip(axes, "abc"):
    ax.text(-0.2, 1.06, letter, transform=ax.transAxes,
            fontweight="bold", fontsize=10)
fig.tight_layout()
save_fig(fig, "Fig6_Residuals")

# %% [markdown]
# ## 14. Split-conformal prediction intervals (Fig. 9, Table S9)
#
# The symmetric half-width *q* is the finite-sample-corrected
# (1 − α)-quantile of the absolute out-of-fold residuals from the fold-pure
# CV (a leak-free calibration set that costs no extra training data).
# Intervals `prediction ± q` then carry ≈ 90 % marginal coverage, which is
# verified empirically on the holdout test set.

# %%
if RUN_CONFORMAL:
    def conformal_halfwidth(abs_residuals, alpha=CONF_ALPHA):
        r = np.sort(np.asarray(abs_residuals, float))
        n = len(r)
        k = int(np.ceil((n + 1) * (1 - alpha)))
        return float(r[min(max(k, 1), n) - 1])

    oof_resid_abs = np.abs(ytr.to_numpy() - oof_pred)
    q = conformal_halfwidth(oof_resid_abs)
    covered = np.abs(yte.to_numpy() - pred_te) <= q
    coverage = float(covered.mean())
    print(f"conformal: half-width q={q:.4f} {UNIT}, target "
          f"{int((1 - CONF_ALPHA) * 100)}% -> empirical coverage "
          f"{coverage * 100:.1f}% on {len(yte)} test points")
    pd.DataFrame([{"alpha": CONF_ALPHA, "half_width_q": q,
                   "target_coverage": 1 - CONF_ALPHA,
                   "empirical_coverage": coverage,
                   "n_test": len(yte)}]).round(4) \
        .to_csv(RES_DIR / "Table_S9_conformal_summary.csv", index=False)

    order = np.argsort(pred_te)
    pos = np.arange(len(yte))
    cov_s = covered[order]
    fig, ax = plt.subplots(figsize=(7.0, 2.8))
    ax.fill_between(pos, pred_te[order] - q, pred_te[order] + q,
                    color=C_BLUE, alpha=0.18, lw=0,
                    label=f"{int((1 - CONF_ALPHA) * 100)}% conformal interval "
                          f"(±{q:.3f} {UNIT})")
    ax.plot(pos, pred_te[order], color=C_BLUE, lw=0.9, label="prediction")
    ax.scatter(pos[cov_s], yte.to_numpy()[order][cov_s], s=7, c=C_GREY, lw=0,
               zorder=3, label="measured (covered)")
    ax.scatter(pos[~cov_s], yte.to_numpy()[order][~cov_s], s=10, c=C_RED, lw=0,
               zorder=4, label="measured (outside)")
    ax.set_xlabel("Test samples (sorted by predicted value)")
    ax.set_ylabel(TARGET_LABEL.capitalize())
    ax.text(0.02, 0.96, f"empirical coverage = {coverage * 100:.1f} %",
            transform=ax.transAxes, va="top")
    ax.legend(loc="lower right", ncols=2)
    fig.tight_layout()
    save_fig(fig, "Fig9_ConformalIntervals")

# %% [markdown]
# ## 15. SHAP feature attribution (Figs. 2–3, Table S5)
#
# Exact TreeSHAP values are computed for the gradient-boosting base learners
# and combined with their renormalized meta-learner weights. TabPFN has no
# exact SHAP algorithm and is excluded from attribution (noted in the figure
# caption). Because the base learners operate on the Box–Cox-transformed
# target, SHAP magnitudes are in transformed units; the feature ranking is
# unaffected.

# %%
if RUN_SHAP:
    import shap

    fitted = dict(zip(meta_names, final_stack.estimators_))
    tree_names = [n for n in meta_names if n in ("lgbm", "catboost")]
    w = np.array([abs(meta_weights[n]) for n in tree_names], float)
    w = w / w.sum()
    shap_vals = np.zeros((len(X_te), len(ALL_FEATURES)))
    for name, wi in zip(tree_names, w):
        shap_vals += wi * shap.TreeExplainer(fitted[name]).shap_values(
            X_te.to_numpy())

    X_te_disp = pd.DataFrame(X_te.to_numpy(), columns=DISPLAY_NAMES)
    plt.figure(figsize=(5.2, 4.6))
    shap.summary_plot(shap_vals, X_te_disp, show=False, max_display=15,
                      plot_size=None, alpha=0.6)
    fig = plt.gcf()
    fig.tight_layout()
    save_fig(fig, "Fig2_SHAP_beeswarm")

    mean_abs = pd.Series(np.abs(shap_vals).mean(axis=0), index=DISPLAY_NAMES)
    mean_abs.sort_values(ascending=False).round(5) \
        .to_csv(RES_DIR / "Table_S5_shap_importance.csv",
                header=["mean_abs_shap"])
    top = mean_abs.sort_values().tail(15)
    fig, ax = plt.subplots(figsize=(3.6, 3.8))
    ax.barh(np.arange(len(top)), top.values, color=C_BLUE, height=0.65)
    ax.set_yticks(np.arange(len(top)), top.index)
    ax.set_xlabel("mean |SHAP value| (Box–Cox units)")
    fig.tight_layout()
    save_fig(fig, "Fig3_SHAP_bar")

# %% [markdown]
# ## 16. Feature–target correlation structure (Fig. 7, Table S6)

# %%
corr_df = X_tr.copy()
corr_df.columns = DISPLAY_NAMES
corr_df[TARGET_LABEL] = ytr.to_numpy()
corr = corr_df.corr(method="pearson")
fig, ax = plt.subplots(figsize=(6.4, 5.6))
im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(corr)), corr.columns, rotation=90, fontsize=6)
ax.set_yticks(range(len(corr)), corr.columns, fontsize=6)
cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
cbar.set_label("Pearson correlation")
fig.tight_layout()
save_fig(fig, "Fig7_CorrelationHeatmap")
corr.round(3).to_csv(RES_DIR / "Table_S6_correlation_matrix.csv")

# %% [markdown]
# ## 17. Learning curve (Fig. 5, Table S7)
#
# The stack is retrained on random subsets of the training split (fold-pure
# preprocessing refit per subset) and evaluated on the fixed holdout test
# set, showing how performance scales with the amount of training data.

# %%
if RUN_LEARNING_CURVE:
    lc_rows = []
    rng = np.random.default_rng(RANDOM_STATE)
    for frac in LEARNING_CURVE_SIZES:
        n = int(round(frac * len(ytr)))
        sub = np.sort(rng.choice(len(ytr), size=n, replace=False))
        sub_raw, sub_y = Xtr_raw.iloc[sub], ytr.iloc[sub].to_numpy()

        state = fit_preprocessing(sub_raw)
        shift = fit_bc_shift(sub_y)
        model = make_stack(HAS_TABPFN)
        model.fit(build_matrix(sub_raw, state).to_numpy(),
                  bc_forward(sub_y, shift))
        pred = bc_inverse(model.predict(build_matrix(Xte_raw, state).to_numpy()),
                          shift)
        m = metrics(yte.to_numpy(), pred)
        lc_rows.append({"n_train": n, **m})
        print(f"  n_train={n:5d}:  R2={m['R2']:.4f}  RMSE={m['RMSE']:.4f}")
        del model, state
        cleanup()
    lc = pd.DataFrame(lc_rows)
    lc.round(4).to_csv(RES_DIR / "Table_S7_learning_curve.csv", index=False)

    fig, ax1 = plt.subplots(figsize=(3.6, 2.8))
    ax1.plot(lc["n_train"], lc["R2"], "o-", color=C_BLUE, ms=4, lw=1.2)
    ax1.set_xlabel("Training set size")
    ax1.set_ylabel(r"Test $R^2$", color=C_BLUE)
    ax1.tick_params(axis="y", labelcolor=C_BLUE)
    ax2 = ax1.twinx()
    ax2.plot(lc["n_train"], lc["RMSE"], "s--", color=C_ORANGE, ms=4, lw=1.2)
    ax2.set_ylabel(f"Test RMSE ({UNIT})", color=C_ORANGE)
    ax2.tick_params(axis="y", labelcolor=C_ORANGE)
    ax2.spines["right"].set_visible(True)
    fig.tight_layout()
    save_fig(fig, "Fig5_LearningCurve")

# %% [markdown]
# ## 18. y-randomization test (Fig. 8, Table S8)
#
# The training targets are randomly permuted and the full pipeline retrained.
# If performance on permuted targets collapses to chance (R² ≤ 0), the real
# model's performance cannot be an artifact of leakage or overfitting.

# %%
if RUN_Y_RANDOMIZATION:
    rng = np.random.default_rng(RANDOM_STATE)
    perm_r2 = []
    for i in range(N_PERMUTATIONS):
        y_perm = rng.permutation(ytr.to_numpy())
        shift = fit_bc_shift(y_perm)
        model = make_stack(HAS_TABPFN)
        model.fit(X_tr.to_numpy(), bc_forward(y_perm, shift))
        r2 = metrics(yte.to_numpy(),
                     bc_inverse(model.predict(X_te.to_numpy()), shift))["R2"]
        perm_r2.append(r2)
        print(f"  permutation {i + 1:2d}: R2={r2:.4f}")
        del model
        cleanup()
    perm_r2 = np.array(perm_r2)
    print(f"  permuted R2 = {perm_r2.mean():.4f} +/- {perm_r2.std():.4f}  "
          f"(true model R2 = {m_te['R2']:.4f})")
    pd.DataFrame({"permutation": np.arange(1, len(perm_r2) + 1),
                  "R2": perm_r2}).round(4) \
        .to_csv(RES_DIR / "Table_S8_y_randomization.csv", index=False)

    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    ax.hist(perm_r2, bins=max(5, len(perm_r2) // 2), color=C_GREY, alpha=0.85,
            label=f"permuted targets (n={len(perm_r2)})")
    ax.axvline(m_te["R2"], color=C_RED, lw=1.4,
               label=f"true model ($R^2$={m_te['R2']:.3f})")
    ax.set_xlabel(r"Holdout $R^2$")
    ax.set_ylabel("Count")
    ax.legend(loc="upper center")
    fig.tight_layout()
    save_fig(fig, "Fig8_yRandomization")

# %% [markdown]
# ## 19. Multi-seed holdout stability (Fig. 10, Table S3)
#
# The 80/20 split and all model seeds are varied jointly across five seeds;
# the spread of the resulting test metrics measures the sensitivity of the
# reported performance to the choice of split.

# %%
if RUN_SEED_STABILITY:
    print(f"=== seed-averaged holdout over {len(STABILITY_SEEDS)} seeds "
          f"({ENSEMBLE_LABEL}) ===")
    seed_rows = []
    for s in STABILITY_SEEDS:
        tr, te = train_test_split(np.arange(len(y)), test_size=TEST_SIZE,
                                  random_state=s)
        Xa, Xb = X_raw.iloc[tr], X_raw.iloc[te]
        ya, yb = y.iloc[tr].to_numpy(), y.iloc[te].to_numpy()

        state = fit_preprocessing(Xa)
        shift = fit_bc_shift(ya)
        model = make_stack(HAS_TABPFN, random_state=s)
        model.fit(build_matrix(Xa, state).to_numpy(), bc_forward(ya, shift))
        pred = bc_inverse(model.predict(build_matrix(Xb, state).to_numpy()),
                          shift)
        seed_rows.append({"seed": s, **metrics(yb, pred)})
        del model, state
        cleanup()
    seed_df = pd.DataFrame(seed_rows).set_index("seed")
    print(f"  R2={seed_df['R2'].mean():.4f}+/-{seed_df['R2'].std():.4f}  "
          f"RMSE={seed_df['RMSE'].mean():.4f}+/-{seed_df['RMSE'].std():.4f}  "
          f"MAE={seed_df['MAE'].mean():.4f}+/-{seed_df['MAE'].std():.4f}  "
          f"MedAPE={seed_df['MedAPE'].mean():.2f}+/-{seed_df['MedAPE'].std():.2f}")
    print(f"  seeds: {STABILITY_SEEDS}")
    seed_df.round(4).to_csv(RES_DIR / "Table_S3_seed_stability.csv")

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))
    axes[0].boxplot([cv_table["R2"], cv_table["RMSE"] * 10],
                    tick_labels=[r"$R^2$", r"RMSE $\times$ 10"],
                    widths=0.5, medianprops={"color": C_RED})
    axes[0].set_ylabel(f"{CV_FOLDS}-fold CV metric")
    xpos = np.arange(len(seed_df))
    axes[1].scatter(xpos, seed_df["R2"], s=28, c=C_BLUE, zorder=3)
    axes[1].axhline(seed_df["R2"].mean(), color=C_RED, lw=1, ls="--",
                    label=f"mean = {seed_df['R2'].mean():.4f}")
    axes[1].set_xticks(xpos, [str(s) for s in seed_df.index])
    axes[1].set_xlim(-0.5, len(seed_df) - 0.5)
    axes[1].set_xlabel("Random seed (80/20 split)")
    axes[1].set_ylabel(r"Holdout $R^2$")
    axes[1].margins(y=0.25)
    axes[1].legend(loc="best")
    for ax, letter in zip(axes, "ab"):
        ax.text(-0.12, 1.05, letter, transform=ax.transAxes,
                fontweight="bold", fontsize=10)
    fig.tight_layout()
    save_fig(fig, "Fig10_Stability")

# %% [markdown]
# ## 20. Archive outputs

# %%
archive = Path("ASR_publication_outputs.zip")
with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
    for folder in (FIG_DIR, RES_DIR):
        for f in sorted(folder.iterdir()):
            zf.write(f)
print(f"created {archive} with:")
for folder in (FIG_DIR, RES_DIR):
    for f in sorted(folder.iterdir()):
        print(f"  {f}")

try:
    from google.colab import files
    files.download(str(archive))
except ImportError:
    pass
