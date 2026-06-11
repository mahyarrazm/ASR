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
# # ASR mortar expansion — publication pipeline (v11)
#
# Machine-learning prediction of alkali–silica reaction (ASR) mortar-bar
# expansion. Stacked ensemble (LightGBM + CatBoost + TabPFN, BayesianRidge
# meta-learner) with a complete, reproducible evaluation suite:
#
# | Section | Output |
# |---|---|
# | 10-fold fold-pure cross-validation | Table S1, Fig. 10a |
# | Baseline / single-model comparison | Table S2, Fig. 4 |
# | 80/20 holdout evaluation | Fig. 1 (parity), Fig. 6 (residual diagnostics) |
# | Split-conformal prediction intervals | Fig. 9 |
# | SHAP feature attribution | Fig. 2 (beeswarm), Fig. 3 (bar) |
# | Feature correlation structure | Fig. 7 |
# | Learning curve (data efficiency) | Fig. 5 |
# | y-randomization (chance-performance) test | Fig. 8 |
# | Multi-seed holdout stability | Table S3, Fig. 10b |
#
# All figures are rendered inline and saved to `figures/` as 600-dpi PNG and
# vector PDF; all tables are saved to `results/` as CSV. A ZIP archive of both
# folders is created at the end for download.
#
# **Usage (Google Colab).** Upload the dataset CSV (or mount Drive), set
# `CSV_PATH` in the *Configuration* cell, then *Runtime → Run all*. A GPU
# runtime is recommended for TabPFN; without a GPU (or without `tabpfn`
# installed) the pipeline automatically falls back to the two
# gradient-boosting learners.

# %% [markdown]
# ## 1. Setup

# %%
import importlib.util
import subprocess
import sys


def _ensure(packages):
    missing = [p for p in packages if importlib.util.find_spec(p.split("==")[0].replace("-", "_")) is None]
    if missing:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", *missing])


_ensure(["lightgbm", "catboost", "shap"])
try:  # TabPFN is optional: the ensemble degrades gracefully without it
    _ensure(["tabpfn"])
except Exception as exc:  # noqa: BLE001
    print(f"tabpfn could not be installed ({exc}); continuing without it.")

# %%
import os
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
import shap
import sklearn
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import BayesianRidge, Ridge
from sklearn.model_selection import KFold, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")

try:
    import torch

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    DEVICE = "cpu"

try:
    from tabpfn import TabPFNRegressor

    HAS_TABPFN = True
except ImportError:
    TabPFNRegressor = None
    HAS_TABPFN = False

if HAS_TABPFN:  # verify TabPFN can actually load its model weights
    try:
        _rng = np.random.default_rng(0)
        _probe = TabPFNRegressor(device=DEVICE)
        _probe.fit(_rng.random((20, 3)), _rng.random(20))
        _probe.predict(_rng.random((2, 3)))
    except Exception as exc:  # noqa: BLE001
        HAS_TABPFN = False
        print("tabpfn is installed but unusable; falling back to LGBM+CatBoost.")
        print(f"  reason: {type(exc).__name__}: {str(exc).splitlines()[0]}")
        print("  (If the model is gated, accept its license on Hugging Face and set HF_TOKEN.)")

print(f"python    : {platform.python_version()}")
for _m in (np, pd, sklearn, mpl, shap):
    print(f"{_m.__name__:<10}: {_m.__version__}")
print(f"tabpfn    : {'available, device=' + DEVICE if HAS_TABPFN else 'NOT available (fallback: LGBM+CatBoost)'}")

# %% [markdown]
# ## 2. Configuration

# %%
CSV_PATH = os.environ.get("ASR_CSV_PATH", "/content/ASR_FinalE-ComCo.csv")  # path to the dataset
TARGET_COLUMN = None        # None -> last column of the CSV is the target
EXCLUDED_FEATURES = []      # feature columns to exclude from the model
TARGET_LABEL = "Expansion (%)"  # axis label used in figures
UNIT = "%"                  # unit of the target, used in printed reports

SEED = 256                  # primary random seed (holdout split, models)
N_FOLDS = 10                # outer cross-validation folds
N_INNER_FOLDS = 5           # inner folds for out-of-fold meta-feature stacking
TEST_FRACTION = 0.20        # holdout fraction
CALIB_FRACTION = 0.15       # fraction of the training set used for conformal calibration
CONFORMAL_LEVEL = 0.90      # target coverage of conformal intervals
STABILITY_SEEDS = [256, 7, 42, 101, 2024]
N_PERMUTATIONS = 10         # y-randomization repeats
LEARNING_CURVE_SIZES = [0.2, 0.4, 0.6, 0.8, 1.0]

RUN_CV = True
RUN_MODEL_COMPARISON = True
RUN_CONFORMAL = True
RUN_SHAP = True
RUN_LEARNING_CURVE = True
RUN_Y_RANDOMIZATION = True
RUN_SEED_STABILITY = True

FAST_MODE = os.environ.get("ASR_FAST_MODE", "0") == "1"  # reduced settings for smoke tests
if FAST_MODE:
    N_FOLDS, N_INNER_FOLDS, N_PERMUTATIONS = 3, 3, 3
    STABILITY_SEEDS = STABILITY_SEEDS[:2]
    LEARNING_CURVE_SIZES = [0.4, 1.0]

FIG_DIR = Path("figures")
RES_DIR = Path("results")
FIG_DIR.mkdir(exist_ok=True)
RES_DIR.mkdir(exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)

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
    "pdf.fonttype": 42,  # embed TrueType fonts (editable text in PDFs)
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

# %% [markdown]
# ## 4. Data loading

# %%
def load_dataset(csv_path, target_column=None, excluded=()):
    df = pd.read_csv(csv_path)
    target = target_column or df.columns[-1]

    # Drop rows whose target is non-numeric/missing (e.g. a units/description row)
    y_raw = pd.to_numeric(df[target], errors="coerce")
    bad = y_raw.isna()
    if bad.any():
        print(f"dropped {int(bad.sum())} row(s) with non-numeric/missing target.")
    df = df.loc[~bad].reset_index(drop=True)

    y = pd.to_numeric(df[target], errors="coerce").astype(float)
    X_raw = df.drop(columns=[target]).apply(pd.to_numeric, errors="coerce")
    X_raw = X_raw.dropna(axis=1, how="all")
    X = X_raw.drop(columns=[c for c in excluded if c in X_raw.columns])

    n_imputed = int(X.isna().sum().sum())
    if n_imputed:
        print(f"imputed {n_imputed} missing feature value(s) with column medians.")
        X = X.fillna(X.median(numeric_only=True))

    print(f"Loaded {csv_path}: raw X={X_raw.shape}, y={y.shape}  "
          f"(active model features = {X.shape[1]})")
    return X, y.to_numpy(), X_raw


X_df, y, X_raw_df = load_dataset(CSV_PATH, TARGET_COLUMN, EXCLUDED_FEATURES)
FEATURES = list(X_df.columns)
X = X_df.to_numpy(dtype=float)

summary = pd.DataFrame({
    "mean": X_df.mean(), "std": X_df.std(), "min": X_df.min(),
    "median": X_df.median(), "max": X_df.max(),
}).round(4)
summary.to_csv(RES_DIR / "Table_S0_feature_summary.csv")
summary.head(10)

# %% [markdown]
# ## 5. Metrics

# %%
def metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    resid = y_true - y_pred
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    nz = np.abs(y_true) > 1e-12  # MAPE is undefined at zero targets
    ape = np.abs(resid[nz] / y_true[nz]) * 100.0
    return {
        "R2": 1.0 - ss_res / ss_tot,
        "RMSE": float(np.sqrt(np.mean(resid**2))),
        "MAE": float(np.mean(np.abs(resid))),
        "MAPE": float(np.mean(ape)),
        "MedAPE": float(np.median(ape)),
    }


def fmt(m, keys=("R2", "RMSE", "MAE")):
    return "  ".join(f"{k}={m[k]:.4f}" for k in keys)

# %% [markdown]
# ## 6. Models
#
# `StackedEnsemble` implements *fold-pure* stacking: the BayesianRidge
# meta-learner is trained exclusively on inner out-of-fold predictions of the
# base learners, so no base learner ever predicts a sample it was trained on.
# The base learners are then refit on the full training set for inference.

# %%
def make_lgbm(seed):
    return LGBMRegressor(
        n_estimators=2000, learning_rate=0.02, num_leaves=63,
        min_child_samples=10, subsample=0.85, subsample_freq=1,
        colsample_bytree=0.85, reg_alpha=0.1, reg_lambda=1.0,
        random_state=seed, n_jobs=-1, verbosity=-1,
    )


def make_catboost(seed):
    return CatBoostRegressor(
        iterations=1500, learning_rate=0.04, depth=6, l2_leaf_reg=3.0,
        loss_function="RMSE", random_seed=seed, verbose=0, allow_writing_files=False,
    )


def make_tabpfn(seed):
    try:
        return TabPFNRegressor(device=DEVICE, random_state=seed)
    except TypeError:  # older tabpfn versions use different constructor args
        return TabPFNRegressor(device=DEVICE)


def base_learners(seed, use_tabpfn=True):
    models = {"lgbm": make_lgbm(seed), "catboost": make_catboost(seed)}
    if use_tabpfn and HAS_TABPFN:
        models["tabpfn"] = make_tabpfn(seed)
    return models


class StackedEnsemble:
    """Fold-pure stacking of base regressors with a BayesianRidge meta-learner."""

    def __init__(self, seed=SEED, n_inner_folds=N_INNER_FOLDS, use_tabpfn=True):
        self.seed = seed
        self.n_inner_folds = n_inner_folds
        self.use_tabpfn = use_tabpfn

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y, float)
        names = list(base_learners(self.seed, self.use_tabpfn))
        oof = np.zeros((len(y), len(names)))
        inner = KFold(self.n_inner_folds, shuffle=True, random_state=self.seed)
        for tr, va in inner.split(X):
            fold_models = base_learners(self.seed, self.use_tabpfn)
            for j, name in enumerate(names):
                fold_models[name].fit(X[tr], y[tr])
                oof[va, j] = fold_models[name].predict(X[va])
        self.meta_ = BayesianRidge().fit(oof, y)
        self.models_ = base_learners(self.seed, self.use_tabpfn)
        for model in self.models_.values():
            model.fit(X, y)
        self.names_ = names
        return self

    def base_predictions(self, X):
        return np.column_stack([self.models_[n].predict(np.asarray(X, float)) for n in self.names_])

    def predict(self, X):
        return self.meta_.predict(self.base_predictions(X))

    @property
    def weights(self):
        return {n: round(float(w), 4) for n, w in zip(self.names_, self.meta_.coef_)}


def comparison_models(seed):
    """Baselines and single learners evaluated under the same CV protocol."""
    models = {
        "Ridge (linear)": make_pipeline(StandardScaler(), Ridge(alpha=1.0, random_state=seed)),
        "k-NN": make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=5, weights="distance")),
        "Random Forest": RandomForestRegressor(n_estimators=500, min_samples_leaf=2,
                                               random_state=seed, n_jobs=-1),
        "LightGBM": make_lgbm(seed),
        "CatBoost": make_catboost(seed),
    }
    if HAS_TABPFN:
        models["TabPFN"] = make_tabpfn(seed)
    models["Stacked ensemble"] = StackedEnsemble(seed)
    return models

# %% [markdown]
# ## 7. 10-fold fold-pure cross-validation

# %%
def cross_validate(model_factory, X, y, n_folds=N_FOLDS, seed=SEED, verbose=True):
    folds, oof = [], np.zeros(len(y))
    for k, (tr, te) in enumerate(KFold(n_folds, shuffle=True, random_state=seed).split(X), 1):
        model = model_factory()
        model.fit(X[tr], y[tr])
        pred = model.predict(X[te])
        oof[te] = pred
        m = metrics(y[te], pred)
        folds.append(m)
        if verbose:
            print(f"  fold {k:2d}:  R2={m['R2']:.4f}  RMSE={m['RMSE']:.4f}  MAE={m['MAE']:.4f}")
    return pd.DataFrame(folds, index=range(1, n_folds + 1)), oof


ensemble_label = "LGBM+CatBoost+TabPFN" if HAS_TABPFN else "LGBM+CatBoost"
if RUN_CV:
    print(f"=== {N_FOLDS}-fold fold-pure CV ({ensemble_label}) ===")
    cv_table, oof_pred = cross_validate(lambda: StackedEnsemble(SEED), X, y)
    print("\n  per-fold mean +/- std:")
    for col in cv_table.columns:
        print(f"    {col:<6}: {cv_table[col].mean():.4f} +/- {cv_table[col].std():.4f}")
    oof_metrics = metrics(y, oof_pred)
    print(f"  pooled OOF R2 = {oof_metrics['R2']:.4f}")
    cv_table.round(4).to_csv(RES_DIR / "Table_S1_cv_per_fold_metrics.csv", index_label="fold")

# %% [markdown]
# ## 8. Model comparison (Fig. 4)

# %%
if RUN_MODEL_COMPARISON:
    rows = []
    for name, model in comparison_models(SEED).items():
        table, _ = cross_validate(lambda m=model: m if isinstance(m, StackedEnsemble)
                                  else sklearn.base.clone(m), X, y, verbose=False)
        rows.append({"model": name,
                     **{f"{c}_mean": table[c].mean() for c in table.columns},
                     **{f"{c}_std": table[c].std() for c in table.columns}})
        print(f"  {name:<18}: R2={rows[-1]['R2_mean']:.4f}+/-{rows[-1]['R2_std']:.4f}  "
              f"RMSE={rows[-1]['RMSE_mean']:.4f}+/-{rows[-1]['RMSE_std']:.4f}")
    comp = pd.DataFrame(rows).set_index("model")
    comp.round(4).to_csv(RES_DIR / "Table_S2_model_comparison.csv")

    order = comp["R2_mean"].sort_values().index
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))
    ypos = np.arange(len(order))
    colors = [C_ORANGE if m == "Stacked ensemble" else C_BLUE for m in order]
    axes[0].barh(ypos, comp.loc[order, "R2_mean"], xerr=comp.loc[order, "R2_std"],
                 color=colors, height=0.65, error_kw={"lw": 0.8})
    axes[0].set_yticks(ypos, order)
    axes[0].set_xlabel(r"$R^2$ (10-fold CV)")
    axes[0].set_xlim(max(0.0, comp["R2_mean"].min() - 0.05), 1.0)
    axes[1].barh(ypos, comp.loc[order, "RMSE_mean"], xerr=comp.loc[order, "RMSE_std"],
                 color=colors, height=0.65, error_kw={"lw": 0.8})
    axes[1].set_yticks(ypos, ["" for _ in order])
    axes[1].set_xlabel(f"RMSE (10-fold CV), {UNIT}")
    for ax, letter in zip(axes, "ab"):
        ax.text(-0.05, 1.05, letter, transform=ax.transAxes, fontweight="bold", fontsize=10)
    fig.tight_layout()
    save_fig(fig, "Fig4_ModelComparison")

# %% [markdown]
# ## 9. Final fit and 80/20 holdout (Fig. 1, Fig. 6)

# %%
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=TEST_FRACTION, random_state=SEED)
final_model = StackedEnsemble(SEED).fit(X_tr, y_tr)
pred_tr, pred_te = final_model.predict(X_tr), final_model.predict(X_te)
m_tr, m_te = metrics(y_tr, pred_tr), metrics(y_te, pred_te)

print(f"=== final fit / 80-20 holdout (random_state={SEED}) ===")
print(f"  train: { {k: round(v, 4) for k, v in m_tr.items()} }")
print(f"  test : { {k: round(v, 4) for k, v in m_te.items()} }")
print(f"  meta (BayesianRidge) weights: {final_model.weights}")
pd.DataFrame([{"split": "train", **m_tr}, {"split": "test", **m_te}]).round(4) \
    .to_csv(RES_DIR / "Table_S4_holdout_metrics.csv", index=False)

# %%
# Fig. 1 — parity plot
fig, ax = plt.subplots(figsize=(3.4, 3.4))
lims = [min(y.min(), pred_te.min()), max(y.max(), pred_te.max())]
pad = 0.04 * (lims[1] - lims[0])
lims = [lims[0] - pad, lims[1] + pad]
ax.plot(lims, lims, color=C_GREY, lw=0.8, zorder=1)
ax.scatter(y_tr, pred_tr, s=10, c=C_GREY, alpha=0.35, lw=0,
           label=f"train (n={len(y_tr)})", zorder=2)
ax.scatter(y_te, pred_te, s=14, c=C_BLUE, alpha=0.8, lw=0,
           label=f"test (n={len(y_te)})", zorder=3)
ax.set_xlim(lims), ax.set_ylim(lims)
ax.set_xlabel(f"Measured {TARGET_LABEL}")
ax.set_ylabel(f"Predicted {TARGET_LABEL}")
ax.set_aspect("equal")
ax.text(0.04, 0.96, f"test $R^2$ = {m_te['R2']:.3f}\nRMSE = {m_te['RMSE']:.3f}\n"
        f"MAE = {m_te['MAE']:.3f}", transform=ax.transAxes, va="top")
ax.legend(loc="lower right")
fig.tight_layout()
save_fig(fig, "Fig1_ActualVsPredicted")

# %%
# Fig. 6 — residual diagnostics (test set)
resid = y_te - pred_te
fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.4))
axes[0].axhline(0, color=C_GREY, lw=0.8)
axes[0].scatter(pred_te, resid, s=10, c=C_BLUE, alpha=0.7, lw=0)
axes[0].set_xlabel(f"Predicted {TARGET_LABEL}")
axes[0].set_ylabel(f"Residual {TARGET_LABEL}")

axes[1].hist(resid, bins=30, color=C_BLUE, alpha=0.85, density=True)
xs = np.linspace(resid.min(), resid.max(), 200)
axes[1].plot(xs, st.norm.pdf(xs, resid.mean(), resid.std()), color=C_RED, lw=1)
axes[1].set_xlabel(f"Residual {TARGET_LABEL}")
axes[1].set_ylabel("Density")

st.probplot(resid, dist="norm", plot=axes[2])
axes[2].set_title("")
axes[2].get_lines()[0].set(marker="o", markersize=2.5, color=C_BLUE, alpha=0.7)
axes[2].get_lines()[1].set(color=C_RED, lw=1)
axes[2].set_xlabel("Theoretical quantiles")
axes[2].set_ylabel("Ordered residuals")
for ax, letter in zip(axes, "abc"):
    ax.text(-0.18, 1.06, letter, transform=ax.transAxes, fontweight="bold", fontsize=10)
fig.tight_layout()
save_fig(fig, "Fig6_Residuals")

# %% [markdown]
# ## 10. Split-conformal prediction intervals (Fig. 9)

# %%
if RUN_CONFORMAL:
    X_fit, X_cal, y_fit, y_cal = train_test_split(
        X_tr, y_tr, test_size=CALIB_FRACTION, random_state=SEED)
    conf_model = StackedEnsemble(SEED).fit(X_fit, y_fit)
    cal_scores = np.abs(y_cal - conf_model.predict(X_cal))
    n_cal = len(cal_scores)
    q_level = min(1.0, np.ceil((n_cal + 1) * CONFORMAL_LEVEL) / n_cal)
    q = float(np.quantile(cal_scores, q_level, method="higher"))

    pred_conf = conf_model.predict(X_te)
    covered = np.abs(y_te - pred_conf) <= q
    coverage = covered.mean()
    print(f"conformal: half-width q={q:.4f} {UNIT}, target {CONFORMAL_LEVEL:.0%} "
          f"-> empirical coverage {coverage:.1%} on {len(y_te)} test points")

    order = np.argsort(pred_conf)
    idx = np.arange(len(y_te))
    cov_s = covered[order]
    fig, ax = plt.subplots(figsize=(7.0, 2.6))
    ax.fill_between(idx, pred_conf[order] - q, pred_conf[order] + q,
                    color=C_BLUE, alpha=0.18, lw=0,
                    label=f"{CONFORMAL_LEVEL:.0%} conformal interval")
    ax.plot(idx, pred_conf[order], color=C_BLUE, lw=0.9, label="prediction")
    ax.scatter(idx[cov_s], y_te[order][cov_s], s=7, c=C_GREY, lw=0, zorder=3,
               label="measured (covered)")
    ax.scatter(idx[~cov_s], y_te[order][~cov_s], s=9, c=C_RED, lw=0, zorder=4,
               label="measured (outside interval)")
    ax.set_xlabel("Test samples (sorted by predicted value)")
    ax.set_ylabel(TARGET_LABEL)
    ax.text(0.02, 0.96, f"empirical coverage = {coverage:.1%}",
            transform=ax.transAxes, va="top")
    ax.legend(loc="lower right", ncols=2)
    fig.tight_layout()
    save_fig(fig, "Fig9_ConformalIntervals")

# %% [markdown]
# ## 11. SHAP feature attribution (Fig. 2, Fig. 3)
#
# TreeSHAP values are computed exactly for the gradient-boosting components and
# combined with their (renormalized) meta-learner weights; TabPFN has no exact
# SHAP algorithm and is excluded from attribution, which is noted in captions.

# %%
if RUN_SHAP:
    tree_names = [n for n in final_model.names_ if n in ("lgbm", "catboost")]
    w = np.array([final_model.weights[n] for n in tree_names], float)
    w = np.abs(w) / np.abs(w).sum()
    shap_vals = np.zeros((len(X_te), len(FEATURES)))
    for name, wi in zip(tree_names, w):
        explainer = shap.TreeExplainer(final_model.models_[name])
        shap_vals += wi * explainer.shap_values(X_te)

    X_te_df = pd.DataFrame(X_te, columns=FEATURES)
    plt.figure(figsize=(5.2, 4.6))
    shap.summary_plot(shap_vals, X_te_df, show=False, max_display=15, plot_size=None)
    fig = plt.gcf()
    fig.tight_layout()
    save_fig(fig, "Fig2_SHAP_beeswarm")

    mean_abs = pd.Series(np.abs(shap_vals).mean(axis=0), index=FEATURES) \
        .sort_values(ascending=True)
    mean_abs.sort_values(ascending=False).round(5) \
        .to_csv(RES_DIR / "Table_S5_shap_importance.csv", header=["mean_abs_shap"])
    top = mean_abs.tail(15)
    fig, ax = plt.subplots(figsize=(3.6, 3.8))
    ax.barh(np.arange(len(top)), top.values, color=C_BLUE, height=0.65)
    ax.set_yticks(np.arange(len(top)), top.index)
    ax.set_xlabel(f"mean |SHAP value| ({UNIT})")
    fig.tight_layout()
    save_fig(fig, "Fig3_SHAP_bar")

# %% [markdown]
# ## 12. Feature correlation structure (Fig. 7)

# %%
corr_df = X_df.copy()
corr_df[TARGET_LABEL] = y
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
# ## 13. Learning curve (Fig. 5)

# %%
if RUN_LEARNING_CURVE:
    lc_rows = []
    rng = np.random.default_rng(SEED)
    for frac in LEARNING_CURVE_SIZES:
        n = int(round(frac * len(X_tr)))
        sub = rng.choice(len(X_tr), size=n, replace=False)
        model = StackedEnsemble(SEED).fit(X_tr[sub], y_tr[sub])
        m = metrics(y_te, model.predict(X_te))
        lc_rows.append({"n_train": n, **m})
        print(f"  n_train={n:4d}:  R2={m['R2']:.4f}  RMSE={m['RMSE']:.4f}")
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
# ## 14. y-randomization test (Fig. 8)
#
# The target vector is randomly permuted and the full pipeline retrained; if
# performance on permuted targets collapses to chance ($R^2 \le 0$), the real
# model's performance cannot be explained by leakage or overfitting.

# %%
if RUN_Y_RANDOMIZATION:
    rng = np.random.default_rng(SEED)
    perm_r2 = []
    for i in range(N_PERMUTATIONS):
        y_perm = rng.permutation(y_tr)
        model = StackedEnsemble(SEED).fit(X_tr, y_perm)
        r2 = metrics(y_te, model.predict(X_te))["R2"]
        perm_r2.append(r2)
        print(f"  permutation {i + 1:2d}: R2={r2:.4f}")
    perm_r2 = np.array(perm_r2)
    print(f"  permuted R2 = {perm_r2.mean():.4f} +/- {perm_r2.std():.4f}  "
          f"(true model R2 = {m_te['R2']:.4f})")
    pd.DataFrame({"permutation": np.arange(1, len(perm_r2) + 1), "R2": perm_r2}) \
        .round(4).to_csv(RES_DIR / "Table_S8_y_randomization.csv", index=False)

    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    bins = np.linspace(perm_r2.min() - 0.02, perm_r2.max() + 0.02,
                       max(5, len(perm_r2) // 2))
    ax.hist(perm_r2, bins=bins, color=C_GREY, alpha=0.85,
            label=f"permuted targets (n={len(perm_r2)})")
    ax.axvline(m_te["R2"], color=C_RED, lw=1.4, label="true model")
    ax.set_xlabel(r"Holdout $R^2$")
    ax.set_ylabel("Count")
    ax.legend(loc="upper center")
    fig.tight_layout()
    save_fig(fig, "Fig8_yRandomization")

# %% [markdown]
# ## 15. Multi-seed holdout stability (Fig. 10)

# %%
if RUN_SEED_STABILITY:
    print(f"=== seed-averaged holdout over {len(STABILITY_SEEDS)} seeds ({ensemble_label}) ===")
    seed_rows = []
    for s in STABILITY_SEEDS:
        Xa, Xb, ya, yb = train_test_split(X, y, test_size=TEST_FRACTION, random_state=s)
        model = StackedEnsemble(s).fit(Xa, ya)
        m = metrics(yb, model.predict(Xb))
        seed_rows.append({"seed": s, **m})
    seed_df = pd.DataFrame(seed_rows).set_index("seed")
    print(f"  R2={seed_df['R2'].mean():.4f}+/-{seed_df['R2'].std():.4f}  "
          f"RMSE={seed_df['RMSE'].mean():.4f}+/-{seed_df['RMSE'].std():.4f}  "
          f"MAE={seed_df['MAE'].mean():.4f}+/-{seed_df['MAE'].std():.4f}  "
          f"MedAPE={seed_df['MedAPE'].mean():.2f}+/-{seed_df['MedAPE'].std():.2f}")
    print(f"  seeds: {STABILITY_SEEDS}")
    seed_df.round(4).to_csv(RES_DIR / "Table_S3_seed_stability.csv")

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))
    if RUN_CV:
        axes[0].boxplot([cv_table["R2"], cv_table["RMSE"] * 10], tick_labels=[r"$R^2$", r"RMSE $\times$ 10"],
                        widths=0.5, medianprops={"color": C_RED})
        axes[0].set_ylabel("10-fold CV metric")
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
        ax.text(-0.12, 1.05, letter, transform=ax.transAxes, fontweight="bold", fontsize=10)
    fig.tight_layout()
    save_fig(fig, "Fig10_Stability")

# %% [markdown]
# ## 16. Archive outputs

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

# In Colab, offer the archive for download
try:
    from google.colab import files  # noqa: PLC0415

    files.download(str(archive))
except ImportError:
    pass
