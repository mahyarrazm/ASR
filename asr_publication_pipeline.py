#!/usr/bin/env python3
"""
Alkali-silica reaction (ASR) mortar-bar expansion: predictive modelling,
parsimony analysis, and model interpretation.

Analysis stages
---------------
 0. Data audit: duplicate rows, repeated-measures structure, target and
    predictor distributions.
 1. Model comparison under two resampling schemes: random-row cross-validation
    and mixture-grouped cross-validation. The difference between them
    quantifies optimism caused by repeated measurements of the same mixture.
 2. Model selection on the development partition only, by grouped-CV RMSE.
 3. Grouped-permutation importance for the selected model, aggregated over
    cross-validation folds.
 4. Cumulative least-important-first input removal against prespecified
    practical tolerances.
 5. Locked, mixture-disjoint holdout evaluation with paired bootstrap
    intervals; the random-row holdout is retained as an optimistic reference.
 6. Split-conformal prediction intervals with empirical coverage validation.
 7. Reactivity-classification performance at standard expansion limits.
 8. y-randomization and learning-curve controls.
 9. SHAP attribution with a stability check, calibration, and residual
    diagnostics.

Every figure is rendered inline and written to figures/ as 600-dpi PNG,
vector PDF, and editable SVG. Every table is written to results/ as CSV.
A verified ZIP archive of all artefacts is produced at the end.

Usage (Google Colab)
--------------------
Select a GPU runtime, upload the dataset CSV, set CSV_PATH below, and run all
cells. TabPFN model weights are gated: accept the licence at
https://huggingface.co/Prior-Labs and provide a token via the TABPFN_TOKEN
Colab secret or environment variable. Set ASR_USE_TABPFN=0 to run the
complete analysis without TabPFN.
"""

import importlib.metadata
import os
import subprocess
import sys

# Minimum bounds rather than exact pins: exact pins force downgrades of the
# Colab base image and can require a runtime restart mid-analysis. The exact
# resolved versions of every dependency are recorded in the run manifest and
# in results/environment_lock.txt so a run remains reproducible.
REQUIREMENTS = [
    ("numpy", "1.26"), ("pandas", "2.2"), ("scipy", "1.13"),
    ("scikit-learn", "1.5"), ("matplotlib", "3.9"), ("lightgbm", "4.5"),
    ("catboost", "1.2"), ("xgboost", "2.1"), ("shap", "0.46"),
]
OPTIONAL_REQUIREMENTS = [("tabpfn", "2.0"), ("pytorch-tabnet", "4.1")]


def _version_tuple(text):
    parts = []
    for chunk in str(text).split(".")[:3]:
        digits = "".join(itertools.takewhile(str.isdigit, chunk))
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


def _ensure(requirements, optional=False):
    missing = []
    for name, minimum in requirements:
        try:
            installed = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            missing.append(f"{name}>={minimum}")
            continue
        if _version_tuple(installed) < _version_tuple(minimum):
            missing.append(f"{name}>={minimum}")
    if not missing:
        return
    print("installing:", ", ".join(missing))
    command = [sys.executable, "-m", "pip", "install", "--quiet", *missing]
    if optional:
        try:
            subprocess.check_call(command)
        except subprocess.CalledProcessError as error:
            print(f"optional dependencies unavailable ({error}); continuing")
    else:
        subprocess.check_call(command)


import itertools  # noqa: E402  (used by _version_tuple above)

if os.environ.get("ASR_SKIP_INSTALL", "0") != "1":
    _ensure(REQUIREMENTS)
    _ensure(OPTIONAL_REQUIREMENTS, optional=True)

import gc  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import random  # noqa: E402
import shutil  # noqa: E402
import textwrap  # noqa: E402
import warnings  # noqa: E402
import zipfile  # noqa: E402
from contextlib import redirect_stdout  # noqa: E402
from io import StringIO  # noqa: E402
from pathlib import Path  # noqa: E402

# TabPFN inference runs locally against a cached checkpoint; no model inputs or
# predictions leave the runtime.
os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")
if "TABPFN_TOKEN" not in os.environ:
    try:
        from google.colab import userdata

        _token = userdata.get("TABPFN_TOKEN")
        if _token:
            os.environ["TABPFN_TOKEN"] = _token
    except Exception:
        pass

import matplotlib as mpl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy.stats as st  # noqa: E402
import shap  # noqa: E402
import sklearn  # noqa: E402
from scipy.special import boxcox  # noqa: E402
from scipy.stats import boxcox_normmax  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from sklearn.exceptions import ConvergenceWarning  # noqa: E402
from sklearn.linear_model import BayesianRidge, Ridge  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    mean_absolute_error, mean_squared_error, r2_score, roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split  # noqa: E402
from sklearn.neural_network import MLPRegressor  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

# The gradient-boosting libraries load a compiled backend that needs an
# OpenMP runtime. Where that runtime is absent the import raises OSError from
# dlopen rather than ImportError, so each is guarded and the affected
# candidates are dropped from the comparison instead of ending the run.
BOOSTERS = {}
for _label, _module, _symbol in (("LightGBM", "lightgbm", "LGBMRegressor"),
                                 ("XGBoost", "xgboost", "XGBRegressor"),
                                 ("CatBoost", "catboost", "CatBoostRegressor")):
    try:
        BOOSTERS[_label] = getattr(__import__(_module, fromlist=[_symbol]),
                                   _symbol)
    except Exception as _error:  # noqa: BLE001
        print(f"{_label} unavailable ({type(_error).__name__}); it is dropped "
              f"from the comparison. On macOS install the OpenMP runtime "
              f"(brew install libomp) to enable it.")
LGBMRegressor = BOOSTERS.get("LightGBM")
XGBRegressor = BOOSTERS.get("XGBoost")
CatBoostRegressor = BOOSTERS.get("CatBoost")

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

try:
    from tabpfn import TabPFNRegressor

    HAS_TABPFN_PACKAGE = True
except ImportError:
    TabPFNRegressor = None
    HAS_TABPFN_PACKAGE = False

try:
    from pytorch_tabnet import TabNetRegressor

    HAS_TABNET = True
except ImportError:
    try:
        from pytorch_tabnet.tab_model import TabNetRegressor

        HAS_TABNET = True
    except ImportError:
        TabNetRegressor = None
        HAS_TABNET = False

warnings.filterwarnings("ignore", message=r"X does not have valid feature names.*")
# The neural benchmarks stop on their internal validation score by design, so
# a convergence notice is expected rather than a problem to report per fold.
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=FutureWarning,
                        module=r"sklearn\.compose\._column_transformer")
warnings.filterwarnings("ignore", category=DeprecationWarning,
                        module=r"matplotlib\._(?:mathtext|fontconfig_pattern)")
warnings.filterwarnings("ignore", category=DeprecationWarning,
                        module=r"matplotlib\.backends\.backend_pdf")

# =============================================================================
# Configuration
# =============================================================================
CSV_PATH = os.environ.get("ASR_CSV_PATH", "/content/ASR_FinalE-ComCo.csv")

# Analysis scope. The dataset pools three kinds of experiment: the accelerated
# mortar-bar test (ASTM C1260/C1567), the year-long concrete prism test
# (ASTM C1293), and a block of non-standard tests. They differ in temperature,
# solution, duration and specimen, and pooling them costs accuracy. Restricting
# the analysis to one protocol is legitimate provided the paper states the
# scope it reports; "c1260" is the most widely used screening test.
#   all         every row (pooled)
#   standard    ASTM-labelled rows only
#   c1260       accelerated mortar-bar test only
#   c1293       concrete prism test only
SCOPE = os.environ.get("ASR_SCOPE", "all")
STANDARD_COLUMN = "Standard"
TARGET = "y"
TARGET_LABEL = "Expansion (%)"
RANDOM_STATE = 256
TEST_SIZE = 0.20

# Fold-pure Box-Cox target transform. LAMBDA_BC is the value used in earlier
# versions of this analysis and is retained as the default; stage 1b compares
# it against no transform, log1p, and a development-fitted maximum-likelihood
# lambda so the choice is evidence-based rather than assumed.
LAMBDA_BC = 0.85
EPS_SHIFT = 1e-4

# Resampling. Random-row CV reproduces the conventional protocol used in the
# concrete-durability machine-learning literature. Grouped CV keeps every
# measurement of one mixture inside a single fold and is the primary basis for
# model selection and all reported generalization estimates.
CV_FOLDS = 5
CV_REPEATS = 3
GROUPED_CV_FOLDS = 5
GROUPED_CV_REPEATS = 2

# Mixture grouping. "auto" derives a mixture identifier by hashing every
# predictor except the testing age, so all measurement ages of one mixture form
# one group. "row" disables grouping and reproduces the optimistic protocol.
GROUPING_MODE = "auto"
AGE_COLUMN = "x29"
IDENTIFIER_CANDIDATES = ["x0"]

# Feature elimination.
MAX_RAW_FACTORS_TO_REMOVE = 20
RANKING_PERMUTATION_REPEATS = 10
R2_MAX_ABSOLUTE_LOSS = 0.005
RMSE_MAX_RELATIVE_INCREASE_PCT = 5.0
MAE_MAX_RELATIVE_INCREASE_PCT = 5.0

# Uncertainty and interpretation.
CI_LEVEL = 0.95
HOLDOUT_BOOTSTRAP_REPLICATES = 5000
CONFORMAL_ALPHA = 0.10
CONFORMAL_CALIBRATION_FRACTION = 0.25
SHAP_BACKGROUND_ROWS = 50
SHAP_EXPLAIN_ROWS = 200
SHAP_PERMUTATION_ROUNDS = 3
SHAP_MAX_DISPLAY = 15

# Reactivity limits in expansion percent. ASTM C1260/C1567 classify 0.10 % at
# 14 days as innocuous and 0.20 % as deleterious; ASTM C1293 uses 0.04 % at one
# year. Values outside the observed target range are skipped automatically.
REACTIVITY_LIMITS = [
    (0.04, "ASTM C1293 one-year limit"),
    (0.10, "ASTM C1260 innocuous limit"),
    (0.20, "ASTM C1260 deleterious limit"),
]

N_PERMUTATION_CONTROLS = 10
LEARNING_CURVE_FRACTIONS = [0.2, 0.4, 0.6, 0.8, 1.0]

# TabPFN is the strongest single model but needs Python 3.10+, a licence
# token, and roughly fifty times the runtime on CPU for about +0.03 R2. Set
# ASR_USE_TABPFN=0 to leave it out entirely, which is the sensible default for
# a laptop run.
USE_TABPFN = os.environ.get("ASR_USE_TABPFN", "1") == "1"
REQUIRE_TABPFN = True
TABPFN_MODEL_VERSION = "V3"
TABPFN_N_ESTIMATORS = 8

RUN_TARGET_TRANSFORM_CHECK = True
RUN_REMOVAL_PATH = True
RUN_CONFORMAL = True
RUN_REACTIVITY_CLASSIFICATION = True
RUN_PERMUTATION_CONTROL = True
RUN_LEARNING_CURVE = True
RUN_SHAP = True

FAST_MODE = os.environ.get("ASR_FAST_MODE", "0") == "1"
if FAST_MODE:
    CV_FOLDS, CV_REPEATS = 3, 1
    GROUPED_CV_FOLDS, GROUPED_CV_REPEATS = 3, 1
    MAX_RAW_FACTORS_TO_REMOVE = 3
    RANKING_PERMUTATION_REPEATS = 2
    HOLDOUT_BOOTSTRAP_REPLICATES = 200
    SHAP_BACKGROUND_ROWS, SHAP_EXPLAIN_ROWS = 10, 20
    SHAP_PERMUTATION_ROUNDS = 1
    N_PERMUTATION_CONTROLS = 2
    LEARNING_CURVE_FRACTIONS = [0.5, 1.0]
    REQUIRE_TABPFN = False

ROOT_DIR = Path("ASR_publication_outputs")
FIG_DIR = ROOT_DIR / "figures"
RES_DIR = ROOT_DIR / "results"
if ROOT_DIR.is_symlink():
    raise RuntimeError(f"refusing to replace symlinked output path: {ROOT_DIR}")
if ROOT_DIR.exists():
    shutil.rmtree(ROOT_DIR)
FIG_DIR.mkdir(parents=True, exist_ok=True)
RES_DIR.mkdir(parents=True, exist_ok=True)

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

# =============================================================================
# Figure style
# =============================================================================
FULL_W, HALF_W = 7.08, 3.46
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7.5, "axes.labelsize": 8.0, "axes.titlesize": 8.0,
    "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "xtick.labelsize": 7.0, "ytick.labelsize": 7.0,
    "legend.fontsize": 6.8, "legend.frameon": False,
    "figure.dpi": 110, "savefig.dpi": 600, "savefig.bbox": "tight",
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
})
C_BLUE, C_ORANGE, C_GREEN, C_RED, C_PURPLE, C_GREY = (
    "#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#7F7F7F"
)
C_DARK = "#333333"
FIGURE_INDEX = []


def save_fig(fig, name, caption=""):
    for extension in ("png", "pdf", "svg"):
        fig.savefig(FIG_DIR / f"{name}.{extension}")
    FIGURE_INDEX.append({"figure": name, "draft_caption": caption})
    print(f"saved figures/{name}.png/.pdf/.svg")
    plt.show()
    plt.close(fig)


def panel_label(ax, letter, dx=-0.15, dy=1.03):
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=8,
            fontweight="bold", va="bottom", ha="right")


def save_table(frame, name, index=False):
    frame.to_csv(RES_DIR / f"{name}.csv", index=index)


def section(title):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def cleanup():
    gc.collect()
    if TORCH_AVAILABLE and torch.cuda.is_available():
        torch.cuda.empty_cache()


section("ENVIRONMENT")
print(f"python      : {platform.python_version()}")
for _module in (np, pd, sklearn, mpl, shap):
    print(f"{_module.__name__:<12}: {_module.__version__}")
ENVIRONMENT_VERSIONS = {"python": platform.python_version()}
for _package in ["numpy", "pandas", "scipy", "scikit-learn", "matplotlib",
                 "lightgbm", "catboost", "xgboost", "shap", "torch",
                 "tabpfn", "pytorch-tabnet2"]:
    try:
        ENVIRONMENT_VERSIONS[_package] = importlib.metadata.version(_package)
    except importlib.metadata.PackageNotFoundError:
        ENVIRONMENT_VERSIONS[_package] = "not installed"
(RES_DIR / "environment_lock.txt").write_text(
    "\n".join(f"{key}=={value}" for key, value in ENVIRONMENT_VERSIONS.items()
              if value != "not installed") + "\n"
)

TABPFN_DEVICE = "cuda" if (TORCH_AVAILABLE and torch.cuda.is_available()) else "cpu"
print(f"tabpfn      : {ENVIRONMENT_VERSIONS['tabpfn']} (device={TABPFN_DEVICE})")
print(f"tabnet      : {'available' if HAS_TABNET else 'not installed'}")

# =============================================================================
# Feature policy
# =============================================================================
NUMERIC_FEATURES_LOADED = [
    "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x9", "x10",
    "x11", "x12", "x13", "x14", "x16", "x17", "x18", "x19", "x20",
    "X21", "x22", "x23", "x24", "x25", "x26", "x27", "x29",
]
CATEGORICAL_FEATURES_LOADED = ["x15", "x28", "Standard"]
BASE_FEATURES_LOADED = NUMERIC_FEATURES_LOADED + CATEGORICAL_FEATURES_LOADED

# Deterministic combinations of the measured inputs. They are documented for
# the feature audit only: no function constructs them, so no fitted model
# matrix in this analysis can contain them.
EXCLUDED_ENGINEERED_FACTORS = {
    "total_SCM": "Total SCM content",
    "age_x_temp": "Testing age x curing temperature",
    "silica_x_alkali": "Silica content x alkali content",
    "log_age": "log(1 + testing age)",
    "SCM_alkali": "SCM-weighted alkali contribution",
}
EXCLUDED_MEASURED_INPUTS = {
    "x8": "relative humidity, constant across all analysed rows",
}
FEATURE_LABEL_OVERRIDES = {"x3": "Mean reactive aggregate size"}

MK_PROP_COLS = ["x11", "x12", "x14"]
FA_PROP_COLS = ["x16", "x17", "x18", "x19", "X21"]
CNS_PROP_COLS = ["x22", "x23", "x24", "x26", "x27"]


def load_data(path):
    """Read the dataset, drop description rows, and harvest column labels."""
    frame = pd.read_csv(path)
    missing = [c for c in BASE_FEATURES_LOADED + [TARGET]
               if c not in frame.columns]
    if missing:
        raise ValueError(f"columns missing from {path}: {missing}")

    # Some exports carry a description row beneath the header, e.g. the text
    # "Silica content (Sand)" under x1. Those rows have a non-numeric target
    # and are used to recover human-readable labels before being dropped.
    numeric_target = pd.to_numeric(frame[TARGET], errors="coerce")
    description_rows = frame.loc[numeric_target.isna()]
    labels = {}
    for column in BASE_FEATURES_LOADED:
        for value in description_rows[column]:
            if not isinstance(value, str) or not value.strip():
                continue
            if pd.isna(pd.to_numeric(pd.Series([value.strip()]),
                                     errors="coerce").iloc[0]):
                labels[column] = " ".join(value.split())
                break

    for column in NUMERIC_FEATURES_LOADED + [TARGET]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    n_before = len(frame)
    frame = frame[frame[TARGET].notna()].reset_index(drop=True)
    print(f"dropped {n_before - len(frame)} description/missing-target row(s)")

    identifier = None
    for candidate in IDENTIFIER_CANDIDATES:
        if candidate in frame.columns:
            identifier = frame[candidate].astype(str).reset_index(drop=True)
            break

    X = frame[BASE_FEATURES_LOADED].reset_index(drop=True)
    y = frame[TARGET].astype(float).reset_index(drop=True)
    print(f"loaded {path}: X={X.shape}, y={y.shape}")
    for column in CATEGORICAL_FEATURES_LOADED:
        levels = sorted(X[column].dropna().astype(str).str.strip().unique())
        print(f"  {column} levels ({len(levels)}): {levels[:6]}"
              f"{' ...' if len(levels) > 6 else ''}")
    return X, y, labels, identifier


def derive_mixture_groups(X, identifier):
    """
    Assign every row to a mixture group.

    Mortar-bar testing measures the same specimen repeatedly over time, so the
    rows of one mixture differ only in testing age. Hashing every predictor
    except the age column recovers that structure directly from the data and
    does not depend on an identifier column being present or meaningful.
    """
    if GROUPING_MODE == "row":
        return np.arange(len(X)), "row (no grouping)"
    key_columns = [c for c in BASE_FEATURES_LOADED if c != AGE_COLUMN]
    keys = pd.util.hash_pandas_object(
        X[key_columns].fillna("missing").astype(str), index=False
    )
    groups = pd.factorize(keys)[0]
    description = f"composition hash of {len(key_columns)} predictors excluding {AGE_COLUMN}"
    if identifier is not None:
        n_identifier_groups = identifier.nunique()
        print(f"  identifier column: {n_identifier_groups} unique values "
              f"over {len(identifier)} rows")
    return groups, description


def audit_data(X, y, groups, group_description, identifier):
    """Quantify duplication and repeated-measures structure before modelling."""
    duplicate_predictor_rows = int(X.duplicated().sum())
    duplicate_full_rows = int(
        pd.concat([X, y.rename(TARGET)], axis=1).duplicated().sum()
    )
    sizes = pd.Series(groups).value_counts()
    audit = pd.DataFrame([{
        "n_rows": len(X),
        "n_predictors": X.shape[1],
        "n_mixture_groups": int(sizes.size),
        "grouping_definition": group_description,
        "median_rows_per_group": float(sizes.median()),
        "max_rows_per_group": int(sizes.max()),
        "share_of_rows_in_multi_row_groups": float(
            sizes[sizes > 1].sum() / len(X)
        ),
        "duplicate_predictor_rows": duplicate_predictor_rows,
        "duplicate_predictor_and_target_rows": duplicate_full_rows,
        "identifier_unique_values": (
            int(identifier.nunique()) if identifier is not None else np.nan
        ),
        "target_min": float(y.min()), "target_median": float(y.median()),
        "target_max": float(y.max()),
        "target_share_below_0.01_pct": float((y < 0.01).mean()),
    }])
    save_table(audit, "Table_01_data_audit")
    print(f"  mixture groups: {sizes.size} for {len(X)} rows "
          f"(median {sizes.median():.0f} rows per group, max {sizes.max()})")
    print(f"  rows sharing a mixture with another row: "
          f"{sizes[sizes > 1].sum() / len(X):.1%}")
    print(f"  duplicated predictor rows: {duplicate_predictor_rows}")
    return audit, sizes


# =============================================================================
# Fold-pure preprocessing
# =============================================================================
def fit_preprocessing(train_df, excluded=None):
    """Learn imputation values and categorical levels from a training fold."""
    excluded = set(excluded or ())
    unknown = excluded.difference(BASE_FEATURES_LOADED)
    if unknown:
        raise ValueError(f"unknown excluded features: {sorted(unknown)}")
    fill = {}

    def conditional_median(property_columns, content_column):
        # Supplementary-cementitious-material composition is only defined where
        # that material is present, so its median is conditioned on a non-zero
        # replacement level rather than taken over all rows.
        present = (pd.Series(True, index=train_df.index)
                   if content_column in excluded
                   else train_df[content_column].fillna(0) > 0)
        for column in property_columns:
            if column in excluded:
                continue
            values = train_df.loc[present, column].dropna()
            fill[column] = float(values.median()) if len(values) else 0.0

    conditional_median(MK_PROP_COLS, "x13")
    conditional_median(FA_PROP_COLS, "x20")
    conditional_median(CNS_PROP_COLS, "x25")
    for column in NUMERIC_FEATURES_LOADED:
        if column in excluded or column in fill:
            continue
        values = train_df[column].dropna()
        fill[column] = float(values.median()) if len(values) else 0.0

    categories = {}
    for column in CATEGORICAL_FEATURES_LOADED:
        if column in excluded:
            continue
        values = train_df[column].fillna("Unknown").astype(str).str.strip()
        values = values.mask(values.eq(""), "Unknown")
        levels = sorted(v for v in values.unique() if v != "Unknown")
        categories[column] = levels + ["Unknown"]
    return {"fill": fill, "categories": categories}


def apply_preprocessing(frame, state):
    output = frame.copy()
    for column, value in state["fill"].items():
        if column in output.columns:
            output[column] = output[column].fillna(value)
    for column, levels in state["categories"].items():
        values = output[column].fillna("Unknown").astype(str).str.strip()
        values = values.mask(values.eq(""), "Unknown")
        output[column] = values.where(values.isin(levels), "Unknown")
    return output


def build_matrix(raw_df, state, excluded=None):
    """One-hot design matrix for the conventional learners."""
    excluded = set(excluded or ())
    processed = apply_preprocessing(raw_df, state)
    numeric = [c for c in NUMERIC_FEATURES_LOADED if c not in excluded]
    parts = [processed[numeric].astype(float)]
    for column in CATEGORICAL_FEATURES_LOADED:
        if column in excluded:
            continue
        categorical = pd.Categorical(processed[column],
                                     categories=state["categories"][column])
        encoded = pd.get_dummies(categorical, prefix=column, prefix_sep="=",
                                 dtype=float)
        encoded.index = processed.index
        parts.append(encoded)
    output = pd.concat(parts, axis=1)
    if output.isna().any().any():
        raise ValueError(
            f"preprocessing left missing values: "
            f"{output.columns[output.isna().any()].tolist()}"
        )
    return output


def build_tabpfn_matrix(raw_df, state, excluded=None):
    """Ordinal design matrix; TabPFN consumes categorical codes directly."""
    excluded = set(excluded or ())
    processed = apply_preprocessing(raw_df, state)
    numeric = [c for c in NUMERIC_FEATURES_LOADED if c not in excluded]
    output = processed[numeric].astype(float).copy()
    for column in CATEGORICAL_FEATURES_LOADED:
        if column in excluded:
            continue
        codes = pd.Categorical(processed[column],
                               categories=state["categories"][column]).codes
        if np.any(codes < 0):
            raise ValueError(f"categorical encoding failed for {column}")
        output[column] = codes.astype(float)
    if output.isna().any().any():
        raise ValueError("TabPFN preprocessing left missing values")
    return output


def build_model_matrix(model_name, raw_df, state, excluded=None):
    if model_name == TABPFN_NAME:
        return build_tabpfn_matrix(raw_df, state, excluded)
    return build_matrix(raw_df, state, excluded)


def matrix_columns_for_factor(matrix_columns, factor):
    return [position for position, name in enumerate(matrix_columns)
            if name == factor or name.startswith(f"{factor}=")]


# =============================================================================
# Target transform
# =============================================================================
def fit_transform_state(y_train, method="boxcox_fixed"):
    """Fit the target transform on training data only."""
    shift = float(np.min(y_train) - EPS_SHIFT)
    if method == "identity":
        return {"method": method}
    if method == "log1p":
        return {"method": method, "shift": shift}
    if method == "boxcox_fixed":
        return {"method": method, "shift": shift, "lambda": LAMBDA_BC}
    if method == "boxcox_mle":
        shifted = np.asarray(y_train, float) - shift
        try:
            lam = float(np.clip(boxcox_normmax(shifted, method="mle"), 0.05, 2.0))
        except Exception:
            lam = LAMBDA_BC
        return {"method": method, "shift": shift, "lambda": lam}
    raise KeyError(method)


def transform_forward(y, state):
    values = np.asarray(y, float)
    if state["method"] == "identity":
        return values
    shifted = values - state["shift"]
    if state["method"] == "log1p":
        return np.log1p(np.clip(shifted, 0, None))
    return boxcox(np.clip(shifted, 1e-12, None), state["lambda"])


def transform_inverse(z, state):
    values = np.asarray(z, float)
    if state["method"] == "identity":
        return values
    if state["method"] == "log1p":
        return np.expm1(values) + state["shift"]
    lam = state["lambda"]
    base = np.clip(lam * values + 1.0, 1e-9, None) ** (1.0 / lam)
    return base + state["shift"]


# =============================================================================
# Models
# =============================================================================
LGBM_PARAMS = dict(
    n_estimators=2900, max_depth=8, learning_rate=0.0561, subsample=0.624,
    subsample_freq=1, colsample_bytree=0.783, reg_alpha=0.030,
    reg_lambda=2.884, min_child_samples=7, num_leaves=32, n_jobs=-1,
    verbose=-1, deterministic=True, force_col_wise=True,
)
CAT_PARAMS = dict(
    iterations=2600, depth=10, learning_rate=0.0825, l2_leaf_reg=15.34,
    bagging_temperature=0.242, random_strength=0.414, verbose=0,
    allow_writing_files=False, thread_count=-1,
)
XGB_PARAMS = dict(
    n_estimators=1800, max_depth=7, learning_rate=0.035, subsample=0.80,
    colsample_bytree=0.80, min_child_weight=2.0, reg_alpha=0.03,
    reg_lambda=2.0, objective="reg:squarederror", tree_method="hist",
    n_jobs=-1, verbosity=0,
)
MLP_PARAMS = dict(
    hidden_layer_sizes=(128, 64), activation="relu", solver="adam", alpha=1e-3,
    learning_rate_init=1e-3, max_iter=1500, early_stopping=True,
    validation_fraction=0.15, n_iter_no_change=40, tol=1e-5,
)
TABNET_INIT_PARAMS = dict(
    n_d=16, n_a=16, n_steps=4, gamma=1.5, lambda_sparse=1e-4,
    mask_type="entmax", optimizer_params={"lr": 0.02}, verbose=0,
    device_name="cpu",
)
TABNET_FIT_PARAMS = dict(
    max_epochs=300, patience=30, batch_size=256, virtual_batch_size=64,
    num_workers=0, drop_last=False,
)
TABPFN_NAME = "TabPFN"
STACK_NAME = "Stacked ensemble"
BLEND_NAME = "LightGBM+CatBoost (50/50)"
TABPFN_PARAMS = dict(
    n_estimators=TABPFN_N_ESTIMATORS, device=TABPFN_DEVICE,
    fit_mode="fit_with_cache", n_preprocessing_jobs=1, show_progress_bar=False,
)

if FAST_MODE:
    LGBM_PARAMS["n_estimators"] = 200
    CAT_PARAMS["iterations"] = 200
    XGB_PARAMS["n_estimators"] = 200
    MLP_PARAMS["max_iter"] = 200
    TABNET_FIT_PARAMS.update(max_epochs=20, patience=5)
    TABPFN_PARAMS["n_estimators"] = 2


class TabNetAdapter:
    """Deterministic adapter with an internal, fold-pure early-stopping split."""

    def __init__(self, seed):
        self.seed = int(seed)
        self.scaler = StandardScaler()
        self.model = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).reshape(-1, 1)
        scaled = self.scaler.fit_transform(X).astype(np.float32)
        positions = np.arange(len(scaled))
        fit_positions, validation_positions = train_test_split(
            positions, test_size=0.15, random_state=self.seed
        )
        self.model = TabNetRegressor(**TABNET_INIT_PARAMS, seed=self.seed)
        with redirect_stdout(StringIO()):
            self.model.fit(
                X_train=scaled[fit_positions], y_train=y[fit_positions],
                eval_set=[(scaled[validation_positions],
                           y[validation_positions])],
                eval_name=["internal_validation"], eval_metric=["rmse"],
                **TABNET_FIT_PARAMS,
            )
        return self

    def predict(self, X):
        scaled = self.scaler.transform(
            np.asarray(X, dtype=np.float32)
        ).astype(np.float32)
        return np.asarray(self.model.predict(scaled), float).reshape(-1)


def make_tabpfn(seed, feature_names):
    categorical_indices = [
        position for position, name in enumerate(feature_names)
        if name in CATEGORICAL_FEATURES_LOADED
    ]
    kwargs = dict(TABPFN_PARAMS)
    if TORCH_AVAILABLE:
        kwargs["inference_precision"] = torch.float32
    kwargs["categorical_features_indices"] = categorical_indices
    kwargs["random_state"] = seed
    # Newer TabPFN releases expose an explicit checkpoint selector; older ones
    # take the same keyword arguments on the constructor.
    try:
        from tabpfn.constants import ModelVersion

        return TabPFNRegressor.create_default_for_version(
            getattr(ModelVersion, TABPFN_MODEL_VERSION), **kwargs
        )
    except Exception:
        pass
    for attempt in (kwargs,
                    {k: v for k, v in kwargs.items()
                     if k not in ("n_preprocessing_jobs", "fit_mode",
                                  "inference_precision")},
                    {"device": TABPFN_DEVICE, "random_state": seed}):
        try:
            return TabPFNRegressor(**attempt)
        except TypeError:
            continue
    raise RuntimeError("could not construct a TabPFNRegressor")


class StackedEnsemble:
    """Fold-pure stack: the meta-learner sees only inner out-of-fold values."""

    def __init__(self, seed, components, n_inner_folds=5):
        self.seed = int(seed)
        self.components = list(components)
        self.n_inner_folds = n_inner_folds

    def fit(self, matrices, y):
        y = np.asarray(y, float)
        meta_features = np.zeros((len(y), len(self.components)))
        inner = KFold(self.n_inner_folds, shuffle=True, random_state=self.seed)
        for fit_index, out_index in inner.split(y.reshape(-1, 1)):
            for position, name in enumerate(self.components):
                matrix = matrices[name]
                model = make_model(name, self.seed, matrix.columns)
                model.fit(matrix.to_numpy()[fit_index], y[fit_index])
                meta_features[out_index, position] = model.predict(
                    matrix.to_numpy()[out_index]
                )
                del model
        self.meta_ = BayesianRidge().fit(meta_features, y)
        self.models_ = {}
        for name in self.components:
            matrix = matrices[name]
            model = make_model(name, self.seed, matrix.columns)
            model.fit(matrix.to_numpy(), y)
            self.models_[name] = model
        return self

    def predict(self, matrices):
        stacked = np.column_stack([
            self.models_[name].predict(matrices[name].to_numpy())
            for name in self.components
        ])
        return self.meta_.predict(stacked)

    @property
    def weights(self):
        return {name: round(float(weight), 4)
                for name, weight in zip(self.components, self.meta_.coef_)}


def make_model(name, seed, feature_names=None):
    if name == "Ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    if name == "LightGBM":
        return LGBMRegressor(**LGBM_PARAMS, random_state=seed + 1)
    if name == "CatBoost":
        return CatBoostRegressor(**CAT_PARAMS, random_seed=seed + 2)
    if name == "XGBoost":
        return XGBRegressor(**XGB_PARAMS, random_state=seed + 3)
    if name == "MLP":
        return make_pipeline(StandardScaler(),
                             MLPRegressor(**MLP_PARAMS, random_state=seed + 4))
    if name == "TabNet":
        return TabNetAdapter(seed=seed + 5)
    if name == TABPFN_NAME:
        if feature_names is None:
            raise ValueError("TabPFN requires design-matrix column names")
        return make_tabpfn(seed + 6, feature_names)
    raise KeyError(name)


STACK_COMPONENTS = [n for n in ("LightGBM", "CatBoost") if n in BOOSTERS]
MATRIX_KIND = {TABPFN_NAME: "ordinal"}


def build_all_matrices(raw_df, state, excluded=None):
    return {
        "onehot": build_matrix(raw_df, state, excluded),
        "ordinal": build_tabpfn_matrix(raw_df, state, excluded),
    }


def matrix_for(name, matrices):
    return matrices[MATRIX_KIND.get(name, "onehot")]


def fit_named(name, matrices, y_transformed, seed):
    """Fit one candidate on an already-built set of design matrices."""
    if name == BLEND_NAME:
        fitted = {}
        for component in ("LightGBM", "CatBoost"):
            model = make_model(component, seed)
            model.fit(matrix_for(component, matrices).to_numpy(), y_transformed)
            fitted[component] = model
        return fitted
    if name == STACK_NAME:
        component_matrices = {c: matrix_for(c, matrices)
                              for c in STACK_COMPONENTS}
        return StackedEnsemble(seed, STACK_COMPONENTS).fit(
            component_matrices, y_transformed
        )
    matrix = matrix_for(name, matrices)
    model = make_model(name, seed, matrix.columns)
    model.fit(matrix.to_numpy(), y_transformed)
    return model


def predict_named(name, fitted, matrices, transform_state):
    if name == BLEND_NAME:
        parts = [
            transform_inverse(
                fitted[component].predict(
                    matrix_for(component, matrices).to_numpy()
                ), transform_state
            )
            for component in ("LightGBM", "CatBoost")
        ]
        prediction = np.mean(parts, axis=0)
    elif name == STACK_NAME:
        component_matrices = {c: matrix_for(c, matrices)
                              for c in STACK_COMPONENTS}
        prediction = transform_inverse(fitted.predict(component_matrices),
                                       transform_state)
    else:
        prediction = transform_inverse(
            fitted.predict(matrix_for(name, matrices).to_numpy()),
            transform_state,
        )
    prediction = np.asarray(prediction, float)
    if not np.isfinite(prediction).all():
        raise FloatingPointError(f"non-finite predictions from {name}")
    return prediction


def verify_tabpfn_runtime():
    """Fail early and explicitly if the gated TabPFN checkpoint is unusable."""
    if not HAS_TABPFN_PACKAGE:
        return False, "tabpfn package not installed"
    rng = np.random.default_rng(RANDOM_STATE)
    try:
        probe = make_tabpfn(RANDOM_STATE, pd.Index(["x1", "x28"]))
        probe.fit(np.column_stack([rng.normal(size=24),
                                   rng.integers(0, 2, size=24)]),
                  rng.normal(size=24))
        output = np.asarray(probe.predict(rng.normal(size=(3, 2))), float)
        if not np.isfinite(output).all():
            raise FloatingPointError("non-finite probe prediction")
        del probe
        cleanup()
        return True, f"{TABPFN_MODEL_VERSION} on {TABPFN_DEVICE}"
    except Exception as error:
        return False, f"{type(error).__name__}: {str(error).splitlines()[0]}"


section("TABPFN AVAILABILITY")
if USE_TABPFN:
    HAS_TABPFN, TABPFN_STATUS = verify_tabpfn_runtime()
else:
    HAS_TABPFN, TABPFN_STATUS = False, "disabled by ASR_USE_TABPFN=0"
if HAS_TABPFN:
    print(f"TabPFN verified: {TABPFN_STATUS}")
    STACK_COMPONENTS = ["LightGBM", "CatBoost", TABPFN_NAME]
else:
    message = textwrap.dedent(f"""
        TabPFN is unavailable: {TABPFN_STATUS}
        Accept the model licence at https://huggingface.co/Prior-Labs and
        supply a token via the TABPFN_TOKEN Colab secret or environment
        variable, and select a GPU runtime.
    """).strip()
    if REQUIRE_TABPFN and USE_TABPFN:
        raise RuntimeError(message)
    print(message + "\nContinuing without TabPFN.")

CANDIDATE_MODELS = ["Ridge", "MLP"]
if HAS_TABNET:
    CANDIDATE_MODELS.append("TabNet")
CANDIDATE_MODELS += [n for n in ("LightGBM", "XGBoost", "CatBoost")
                     if n in BOOSTERS]
if {"LightGBM", "CatBoost"} <= set(BOOSTERS):
    CANDIDATE_MODELS.append(BLEND_NAME)
if HAS_TABPFN:
    CANDIDATE_MODELS.append(TABPFN_NAME)
if STACK_COMPONENTS:
    CANDIDATE_MODELS.append(STACK_NAME)


# =============================================================================
# Metrics and resampling
# =============================================================================
def metric_row(y_true, y_pred):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    residual = y_true - y_pred
    nonzero = np.abs(y_true) > 1e-8
    medape = (float(np.median(np.abs(residual[nonzero] / y_true[nonzero]) * 100))
              if nonzero.any() else np.nan)
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MedAPE": medape,
        "residual_mean": float(np.mean(residual)),
        "residual_sd": float(np.std(residual, ddof=1)) if len(residual) > 1 else 0.0,
        "p95_absolute_residual": float(np.quantile(np.abs(residual), 0.95)),
    }


def corrected_t_interval(values, n_test, n_train, level=CI_LEVEL):
    """
    Nadeau-Bengio corrected resampled t interval.

    Fold scores from repeated cross-validation share training data, so the
    naive standard error understates variability. The correction inflates the
    variance by (1/J + n_test/n_train) and is the appropriate interval to
    report for repeated k-fold estimates.
    """
    values = np.asarray(values, float)
    mean = float(np.mean(values))
    n_folds = len(values)
    if n_folds < 2:
        return mean, mean, mean
    variance = float(np.var(values, ddof=1))
    if variance == 0 or n_train <= 0:
        return mean, mean, mean
    factor = (1.0 / n_folds) + (n_test / n_train)
    standard_error = float(np.sqrt(max(factor, 1e-12) * variance))
    critical = st.t.ppf((1.0 + level) / 2.0, df=n_folds - 1)
    return mean, mean - critical * standard_error, mean + critical * standard_error


def bootstrap_metric_interval(y_true, y_pred, replicates, seed):
    """Paired-row percentile intervals, conditional on the fitted model."""
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    rng = np.random.default_rng(seed)
    names = ["R2", "RMSE", "MAE", "MedAPE"]
    draws = {name: np.empty(replicates, float) for name in names}
    for index in range(replicates):
        positions = rng.integers(0, len(y_true), size=len(y_true))
        result = metric_row(y_true[positions], y_pred[positions])
        for name in names:
            draws[name][index] = result[name]
    alpha = 1.0 - CI_LEVEL
    output = {}
    for name in names:
        output[f"{name}_ci_lower"] = float(np.quantile(draws[name], alpha / 2))
        output[f"{name}_ci_upper"] = float(np.quantile(draws[name], 1 - alpha / 2))
    output["bootstrap_replicates"] = replicates
    output["ci_level"] = CI_LEVEL
    return output


def paired_difference_bootstrap(y_true, reference, candidate, replicates, seed):
    """Paired holdout differences; positive error change means degradation."""
    y_true = np.asarray(y_true, float)
    reference = np.asarray(reference, float)
    candidate = np.asarray(candidate, float)

    def differences(positions):
        a = metric_row(y_true[positions], reference[positions])
        b = metric_row(y_true[positions], candidate[positions])
        return {
            "delta_R2_candidate_minus_reference": b["R2"] - a["R2"],
            "relative_RMSE_change_pct": 100 * (b["RMSE"] / a["RMSE"] - 1),
            "relative_MAE_change_pct": 100 * (b["MAE"] / a["MAE"] - 1),
            "relative_MedAPE_change_pct": 100 * (b["MedAPE"] / a["MedAPE"] - 1),
        }

    point = differences(np.arange(len(y_true)))
    rng = np.random.default_rng(seed)
    draws = {name: np.empty(replicates, float) for name in point}
    for index in range(replicates):
        positions = rng.integers(0, len(y_true), size=len(y_true))
        for name, value in differences(positions).items():
            draws[name][index] = value
    alpha = 1.0 - CI_LEVEL
    rows = []
    for name, estimate in point.items():
        values = draws[name]
        rows.append({
            "metric_difference": name,
            "point_estimate": estimate,
            "ci_lower": float(np.quantile(values, alpha / 2)),
            "ci_upper": float(np.quantile(values, 1 - alpha / 2)),
            "probability_candidate_worse": float(
                np.mean(values < 0) if name.startswith("delta_R2")
                else np.mean(values > 0)
            ),
            "bootstrap_replicates": replicates,
        })
    return pd.DataFrame(rows)


def holm_adjust(pvalues):
    pvalues = np.asarray(pvalues, float)
    adjusted = np.full_like(pvalues, np.nan)
    valid = np.flatnonzero(np.isfinite(pvalues))
    order = valid[np.argsort(pvalues[valid])]
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(order) - rank) * pvalues[index])
        adjusted[index] = min(running, 1.0)
    return adjusted


def make_strata(y_values, n_bins):
    for q in range(min(n_bins, max(len(y_values) // 2, 2)), 1, -1):
        try:
            strata = pd.qcut(y_values, q=q, labels=False, duplicates="drop")
            if pd.Series(strata).value_counts().min() >= 2:
                return np.asarray(strata, dtype=int)
        except ValueError:
            continue
    return None


def random_row_splits(y, n_folds, n_repeats, seed):
    splits = []
    strata = make_strata(np.asarray(y, float), n_folds)
    for repeat in range(1, n_repeats + 1):
        fold_seed = seed + 1000 * repeat
        if strata is not None:
            iterator = StratifiedKFold(
                n_splits=n_folds, shuffle=True, random_state=fold_seed
            ).split(np.zeros(len(y)), strata)
        else:
            iterator = KFold(
                n_splits=n_folds, shuffle=True, random_state=fold_seed
            ).split(np.zeros(len(y)))
        for fold, (train_index, test_index) in enumerate(iterator, 1):
            splits.append((repeat, fold, train_index, test_index))
    return splits


def grouped_splits(groups, n_folds, n_repeats, seed):
    """
    Group-disjoint k-fold with size-balanced folds.

    Groups are shuffled per repeat, then assigned largest-first to whichever
    fold currently holds the fewest rows. This keeps folds comparable in size
    even when mixtures contribute very different numbers of measurements, and
    it is deterministic across scikit-learn versions.
    """
    groups = np.asarray(groups)
    unique, inverse = np.unique(groups, return_inverse=True)
    sizes = np.bincount(inverse)
    splits = []
    for repeat in range(1, n_repeats + 1):
        rng = np.random.default_rng(seed + 977 * repeat)
        order = rng.permutation(len(unique))
        order = order[np.argsort(-sizes[order], kind="stable")]
        fold_of_group = np.empty(len(unique), dtype=int)
        loads = np.zeros(n_folds)
        for group_index in order:
            target = int(np.argmin(loads))
            fold_of_group[group_index] = target
            loads[target] += sizes[group_index]
        row_fold = fold_of_group[inverse]
        for fold in range(n_folds):
            test_index = np.flatnonzero(row_fold == fold)
            train_index = np.flatnonzero(row_fold != fold)
            if len(test_index) == 0 or len(train_index) == 0:
                continue
            splits.append((repeat, fold + 1, train_index, test_index))
    return splits


def grouped_holdout(groups, test_size, seed):
    """Reserve whole mixtures until the requested row fraction is reached."""
    groups = np.asarray(groups)
    unique = np.unique(groups)
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique)
    target_rows = int(round(test_size * len(groups)))
    membership = pd.Series(groups)
    held, count = [], 0
    for group in shuffled:
        if count >= target_rows:
            break
        held.append(group)
        count += int((membership == group).sum())
    holdout_mask = membership.isin(held).to_numpy()
    return np.flatnonzero(~holdout_mask), np.flatnonzero(holdout_mask)


# =============================================================================
# Stage 0: load, group, audit
# =============================================================================
def apply_scope(X, y, identifier, scope):
    """Restrict the analysis to one test protocol and report what was kept."""
    labels = X[STANDARD_COLUMN].fillna("Unknown").astype(str).str.strip()
    labels = labels.mask(labels.eq("") | labels.eq("nan"), "Unknown")
    if scope == "all":
        mask = pd.Series(True, index=X.index)
    elif scope == "standard":
        mask = labels.ne("Unknown")
    elif scope == "c1260":
        mask = labels.str.contains("1260|1567", regex=True)
    elif scope == "c1293":
        mask = labels.str.contains("1293")
    else:
        raise ValueError(f"unknown ASR_SCOPE: {scope}")
    if int(mask.sum()) < 100:
        raise ValueError(f"scope '{scope}' keeps only {int(mask.sum())} rows")
    if scope != "all":
        print(f"scope '{scope}': kept {int(mask.sum())} of {len(X)} rows "
              f"({', '.join(sorted(labels[mask].unique()))})")
    return (X.loc[mask].reset_index(drop=True),
            y.loc[mask].reset_index(drop=True),
            None if identifier is None else
            identifier.loc[mask].reset_index(drop=True))


section("STAGE 0  DATA AUDIT")
X_RAW, Y, AUTO_LABELS, IDENTIFIER = load_data(CSV_PATH)
X_RAW, Y, IDENTIFIER = apply_scope(X_RAW, Y, IDENTIFIER, SCOPE)
LABEL_MAP = {**AUTO_LABELS, **FEATURE_LABEL_OVERRIDES}
GROUPS, GROUP_DESCRIPTION = derive_mixture_groups(X_RAW, IDENTIFIER)
DATA_AUDIT, GROUP_SIZES = audit_data(X_RAW, Y, GROUPS, GROUP_DESCRIPTION,
                                     IDENTIFIER)
REPEATED_MEASURE_SHARE = float(DATA_AUDIT.loc[0, "share_of_rows_in_multi_row_groups"])

factor_inventory = pd.DataFrame([{
    "measured_inputs_included": len(BASE_FEATURES_LOADED),
    "numeric_inputs": len(NUMERIC_FEATURES_LOADED),
    "categorical_inputs": len(CATEGORICAL_FEATURES_LOADED),
    "engineered_factors_available": len(EXCLUDED_ENGINEERED_FACTORS),
    "engineered_factors_included": 0,
    "engineered_factors_excluded": "; ".join(
        f"{code}: {label}" for code, label in EXCLUDED_ENGINEERED_FACTORS.items()
    ),
    "measured_inputs_excluded": "; ".join(
        f"{code}: {reason}" for code, reason in EXCLUDED_MEASURED_INPUTS.items()
    ),
    "feature_policy": "measured inputs only; deterministic engineered terms "
                      "excluded before any model fitting",
    "analysis_scope": SCOPE,
    "scope_definition": {
        "all": "every row, pooling all test protocols",
        "standard": "ASTM-labelled rows only",
        "c1260": "accelerated mortar-bar test (ASTM C1260/C1567) only",
        "c1293": "concrete prism test (ASTM C1293) only",
    }[SCOPE],
}])
save_table(factor_inventory, "Table_02_factor_inventory")

descriptive = X_RAW[NUMERIC_FEATURES_LOADED].describe().T[
    ["count", "mean", "std", "min", "50%", "max"]
]
descriptive.insert(0, "label", [LABEL_MAP.get(c, c) for c in descriptive.index])
descriptive["missing_fraction"] = (
    X_RAW[NUMERIC_FEATURES_LOADED].isna().mean().reindex(descriptive.index)
)
save_table(descriptive.round(5), "Table_03_predictor_summary", index=True)

fig = plt.figure(figsize=(FULL_W, 5.4))
grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.25], hspace=0.45, wspace=0.30)

ax = fig.add_subplot(grid[0, 0])
ax.hist(Y, bins=45, color=C_BLUE, alpha=0.85)
for style, (limit, _) in zip([":", "--", "-."], REACTIVITY_LIMITS):
    if Y.min() <= limit <= Y.max():
        ax.axvline(limit, color=C_RED, ls=style, lw=0.8,
                   label=f"{limit:.2f} % limit")
if ax.get_legend_handles_labels()[0]:
    ax.legend(loc="upper right", fontsize=6.0)
ax.set_xlabel(f"Measured {TARGET_LABEL}")
ax.set_ylabel("Number of measurements")
panel_label(ax, "a", dx=-0.16)

ax = fig.add_subplot(grid[0, 1])
if AGE_COLUMN in X_RAW.columns:
    ax.scatter(X_RAW[AGE_COLUMN], Y, s=5, alpha=0.25, color=C_BLUE,
               edgecolors="none")
    ax.set_xlabel(LABEL_MAP.get(AGE_COLUMN, AGE_COLUMN))
    ax.set_ylabel(TARGET_LABEL)
panel_label(ax, "b", dx=-0.16)

ax = fig.add_subplot(grid[1, 0])
counts = GROUP_SIZES.value_counts().sort_index()
ax.bar(counts.index, counts.to_numpy(), color=C_ORANGE, alpha=0.9, width=0.8)
ax.set_xlim(counts.index.min() - 1, counts.index.max() + 1)
ax.set_xlabel("Measurements per mixture")
ax.set_ylabel("Number of mixtures")
ax.set_title(f"{GROUP_SIZES.size} mixtures; "
             f"{REPEATED_MEASURE_SHARE:.0%} of rows are repeated measures",
             fontsize=7)
panel_label(ax, "c", dx=-0.16)

ax = fig.add_subplot(grid[1, 1])
correlation_frame = X_RAW[NUMERIC_FEATURES_LOADED].copy()
correlation_frame[TARGET_LABEL] = Y
correlation = correlation_frame.corr(method="spearman")
image = ax.imshow(correlation.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(correlation)), correlation.columns, rotation=90,
              fontsize=4.4)
ax.set_yticks(range(len(correlation)), correlation.columns, fontsize=4.4)
colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
colorbar.set_label("Spearman correlation", fontsize=6.5)
colorbar.ax.tick_params(labelsize=5.5)
panel_label(ax, "d", dx=-0.16)
save_fig(fig, "Fig_01_DatasetOverview",
         "Dataset structure. a, Distribution of measured expansion with "
         "standard reactivity limits. b, Expansion against testing age. "
         "c, Number of measurements contributed by each mixture. d, Spearman "
         "correlation among measured predictors and the target.")
save_table(correlation.round(3), "Table_04_predictor_correlation", index=True)

# =============================================================================
# Partitioning
# =============================================================================
section("STAGE 0b  PARTITIONING")
DEV_INDEX, HOLDOUT_INDEX = grouped_holdout(GROUPS, TEST_SIZE, RANDOM_STATE)
RANDOM_DEV_INDEX, RANDOM_HOLDOUT_INDEX = train_test_split(
    np.arange(len(Y)), test_size=TEST_SIZE, random_state=RANDOM_STATE
)

X_DEV = X_RAW.iloc[DEV_INDEX].reset_index(drop=True)
Y_DEV = Y.iloc[DEV_INDEX].reset_index(drop=True)
GROUPS_DEV = GROUPS[DEV_INDEX]
X_HOLDOUT = X_RAW.iloc[HOLDOUT_INDEX].reset_index(drop=True)
Y_HOLDOUT = Y.iloc[HOLDOUT_INDEX].reset_index(drop=True)

shared_groups = set(GROUPS[DEV_INDEX]).intersection(GROUPS[HOLDOUT_INDEX])
if shared_groups:
    raise RuntimeError("mixture leakage between development and holdout")

split_audit = pd.DataFrame([{
    "n_total": len(Y),
    "n_development_rows": len(DEV_INDEX),
    "n_holdout_rows": len(HOLDOUT_INDEX),
    "n_development_mixtures": int(pd.Series(GROUPS[DEV_INDEX]).nunique()),
    "n_holdout_mixtures": int(pd.Series(GROUPS[HOLDOUT_INDEX]).nunique()),
    "holdout_is_mixture_disjoint": True,
    "mixtures_shared_between_partitions": 0,
    "split_seed": RANDOM_STATE,
    "holdout_used_for_selection": False,
    "holdout_used_for_importance_or_elimination": False,
    "reference_random_row_holdout_rows": len(RANDOM_HOLDOUT_INDEX),
}])
save_table(split_audit, "Table_05_split_audit")
print(f"  development: {len(DEV_INDEX)} rows / "
      f"{split_audit.loc[0, 'n_development_mixtures']} mixtures")
print(f"  locked holdout: {len(HOLDOUT_INDEX)} rows / "
      f"{split_audit.loc[0, 'n_holdout_mixtures']} mixtures (disjoint)")

RANDOM_SPLITS = random_row_splits(Y_DEV, CV_FOLDS, CV_REPEATS, RANDOM_STATE)
GROUPED_SPLITS = grouped_splits(GROUPS_DEV, GROUPED_CV_FOLDS,
                                GROUPED_CV_REPEATS, RANDOM_STATE)
SCHEMES = {
    "random-row CV": RANDOM_SPLITS,
    "mixture-grouped CV": GROUPED_SPLITS,
}
PRIMARY_SCHEME = "mixture-grouped CV"


# =============================================================================
# Stage 1: model comparison under both resampling schemes
# =============================================================================
def evaluate_candidates(splits, model_names, X_source, y_source,
                        transform_method="boxcox_fixed", excluded=None,
                        label="", collect_predictions=True):
    """Fit every candidate on identical folds and return per-fold metrics."""
    fold_rows, prediction_rows = [], []
    total = len(splits)
    for position, (repeat, fold, train_index, test_index) in enumerate(splits, 1):
        raw_train = X_source.iloc[train_index]
        raw_test = X_source.iloc[test_index]
        y_train = y_source.iloc[train_index].to_numpy()
        y_test = y_source.iloc[test_index].to_numpy()

        state = fit_preprocessing(raw_train, excluded)
        train_matrices = build_all_matrices(raw_train, state, excluded)
        test_matrices = build_all_matrices(raw_test, state, excluded)
        transform_state = fit_transform_state(y_train, transform_method)
        y_train_transformed = transform_forward(y_train, transform_state)
        seed = RANDOM_STATE + 10000 * repeat + 100 * fold

        for name in model_names:
            fitted = fit_named(name, train_matrices, y_train_transformed, seed)
            prediction = predict_named(name, fitted, test_matrices,
                                       transform_state)
            fold_rows.append({
                "model": name, "repeat": repeat, "fold": fold,
                "n_train": len(train_index), "n_test": len(test_index),
                **metric_row(y_test, prediction),
            })
            if collect_predictions:
                for source_row, measured, predicted in zip(
                        test_index, y_test, prediction):
                    prediction_rows.append({
                        "model": name, "repeat": repeat, "fold": fold,
                        "development_row": int(source_row),
                        "measured": float(measured),
                        "predicted": float(predicted),
                        "residual": float(measured - predicted),
                    })
            del fitted, prediction
            cleanup()
        del state, train_matrices, test_matrices
        cleanup()
        if label:
            print(f"  {label}: fold {position}/{total} complete")
    return pd.DataFrame(fold_rows), pd.DataFrame(prediction_rows)


def summarise_folds(fold_frame):
    rows = []
    metrics = ["R2", "RMSE", "MAE", "MedAPE", "residual_mean",
               "p95_absolute_residual"]
    for model, group in fold_frame.groupby("model", sort=False):
        row = {"model": model, "n_cv_estimates": len(group)}
        n_test = float(group["n_test"].mean())
        n_train = float(group["n_train"].mean())
        for metric in metrics:
            mean, lower, upper = corrected_t_interval(
                group[metric].to_numpy(), n_test, n_train
            )
            row[f"{metric}_mean"] = mean
            row[f"{metric}_ci_lower"] = lower
            row[f"{metric}_ci_upper"] = upper
            row[f"{metric}_sd"] = float(group[metric].std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


section("STAGE 1  MODEL COMPARISON")
print(f"candidates: {', '.join(CANDIDATE_MODELS)}")
comparison_folds, comparison_predictions = {}, {}
comparison_summary = {}
for scheme_name, splits in SCHEMES.items():
    print(f"\n{scheme_name}: {len(splits)} folds x {len(CANDIDATE_MODELS)} models")
    folds, predictions = evaluate_candidates(
        splits, CANDIDATE_MODELS, X_DEV, Y_DEV, label=scheme_name
    )
    folds.insert(0, "scheme", scheme_name)
    predictions.insert(0, "scheme", scheme_name)
    comparison_folds[scheme_name] = folds
    comparison_predictions[scheme_name] = predictions
    summary = summarise_folds(folds)
    summary.insert(0, "scheme", scheme_name)
    comparison_summary[scheme_name] = summary

all_comparison_folds = pd.concat(comparison_folds.values(), ignore_index=True)
all_comparison_summary = pd.concat(comparison_summary.values(), ignore_index=True)
save_table(all_comparison_folds, "Table_06_model_comparison_fold_metrics")
save_table(all_comparison_summary.round(6), "Table_07_model_comparison_summary")
save_table(pd.concat(comparison_predictions.values(), ignore_index=True),
           "Table_08_model_comparison_out_of_fold_predictions")

for scheme_name in SCHEMES:
    compact = comparison_summary[scheme_name][
        ["model", "R2_mean", "RMSE_mean", "MAE_mean", "MedAPE_mean"]
    ].sort_values("RMSE_mean")
    print(f"\n{scheme_name}")
    print(compact.to_string(index=False,
                            float_format=lambda value: f"{value:9.4f}"))

# Optimism attributable to repeated measurements of the same mixture.
optimism_rows = []
random_summary = comparison_summary["random-row CV"].set_index("model")
grouped_summary = comparison_summary[PRIMARY_SCHEME].set_index("model")
for model in CANDIDATE_MODELS:
    optimism_rows.append({
        "model": model,
        "R2_random_row_CV": random_summary.loc[model, "R2_mean"],
        "R2_mixture_grouped_CV": grouped_summary.loc[model, "R2_mean"],
        "R2_optimism": (random_summary.loc[model, "R2_mean"]
                        - grouped_summary.loc[model, "R2_mean"]),
        "RMSE_random_row_CV": random_summary.loc[model, "RMSE_mean"],
        "RMSE_mixture_grouped_CV": grouped_summary.loc[model, "RMSE_mean"],
        "RMSE_understatement_pct": 100 * (
            1 - random_summary.loc[model, "RMSE_mean"]
            / grouped_summary.loc[model, "RMSE_mean"]
        ),
    })
optimism = pd.DataFrame(optimism_rows)
save_table(optimism.round(6), "Table_09_repeated_measure_optimism")
print("\nrepeated-measure optimism (random-row CV minus mixture-grouped CV)")
print(optimism[["model", "R2_optimism", "RMSE_understatement_pct"]].to_string(
    index=False, float_format=lambda value: f"{value:9.4f}"))

# Model selection: lowest grouped-CV RMSE, then MAE, then R2.
selection_order = grouped_summary.reset_index().sort_values(
    ["RMSE_mean", "MAE_mean", "R2_mean"], ascending=[True, True, False],
    kind="stable",
)
BEST_MODEL_NAME = str(selection_order.iloc[0]["model"])
print(f"\nselected model: {BEST_MODEL_NAME} "
      f"(lowest {PRIMARY_SCHEME} RMSE on the development partition)")
save_table(pd.DataFrame([{
    "selected_model": BEST_MODEL_NAME,
    "selection_scheme": PRIMARY_SCHEME,
    "selection_metric": "mean cross-validated RMSE",
    "tie_breakers": "mean MAE, then mean R2",
    "selection_data": "development partition only",
    "holdout_used_for_selection": False,
}]), "Table_10_model_selection_audit")

# Paired Wilcoxon tests over identical folds, Holm-adjusted across all pairs.
pair_rows = []
for scheme_name, folds in comparison_folds.items():
    for model_a, model_b in itertools.combinations(CANDIDATE_MODELS, 2):
        left = folds.loc[folds["model"].eq(model_a),
                         ["repeat", "fold", "RMSE"]].sort_values(["repeat", "fold"])
        right = folds.loc[folds["model"].eq(model_b),
                          ["repeat", "fold", "RMSE"]].sort_values(["repeat", "fold"])
        if not np.array_equal(left[["repeat", "fold"]].to_numpy(),
                              right[["repeat", "fold"]].to_numpy()):
            raise RuntimeError(f"unpaired folds for {model_a} vs {model_b}")
        differences = left["RMSE"].to_numpy() - right["RMSE"].to_numpy()
        try:
            p_value = float(st.wilcoxon(left["RMSE"], right["RMSE"],
                                        alternative="two-sided").pvalue)
        except ValueError:
            p_value = 1.0
        pair_rows.append({
            "scheme": scheme_name, "model_a": model_a, "model_b": model_b,
            "n_paired_folds": len(differences),
            "mean_RMSE_a_minus_b": float(np.mean(differences)),
            "p_wilcoxon_raw": p_value,
        })
pair_tests = pd.DataFrame(pair_rows)
pair_tests["p_wilcoxon_holm"] = np.concatenate([
    holm_adjust(group["p_wilcoxon_raw"].to_numpy())
    for _, group in pair_tests.groupby("scheme", sort=False)
])
pair_tests["inference_scope"] = (
    "descriptive: cross-validation training folds overlap and the winner is "
    "chosen on the same development results"
)
save_table(pair_tests, "Table_11_pairwise_model_tests")

order = grouped_summary["RMSE_mean"].sort_values(ascending=False).index.tolist()
fig, axes = plt.subplots(1, 3, figsize=(FULL_W, 3.1))
positions = np.arange(len(order))
height = 0.38
for ax, metric, xlabel, letter in zip(
        axes, ["R2", "RMSE", "MedAPE"],
        [r"Cross-validated $R^2$", "Cross-validated RMSE (%)",
         "Cross-validated MedAPE (%)"], "abc"):
    for offset, (scheme_name, colour) in enumerate(
            [("random-row CV", C_GREY), (PRIMARY_SCHEME, C_BLUE)]):
        frame = comparison_summary[scheme_name].set_index("model").loc[order]
        values = frame[f"{metric}_mean"].to_numpy(float)
        errors = np.vstack([
            np.clip(values - frame[f"{metric}_ci_lower"].to_numpy(float), 0, None),
            np.clip(frame[f"{metric}_ci_upper"].to_numpy(float) - values, 0, None),
        ])
        bar_colours = [
            C_ORANGE if (model == BEST_MODEL_NAME and scheme_name == PRIMARY_SCHEME)
            else colour for model in order
        ]
        ax.barh(positions + (0.5 - offset) * height, values, height=height,
                color=bar_colours, alpha=0.92)
        ax.errorbar(values, positions + (0.5 - offset) * height, xerr=errors,
                    fmt="none", ecolor="#222222", elinewidth=0.6, capsize=1.5)
    ax.set_yticks(positions, order if letter == "a" else ["" for _ in order],
                  fontsize=6.0)
    ax.set_xlabel(xlabel)
    if metric == "R2":
        ax.set_xlim(min(0.0, float(grouped_summary["R2_mean"].min()) - 0.05), 1.0)
    panel_label(ax, letter, dx=-0.05 if letter != "a" else -0.55)
legend_handles = [
    Patch(facecolor=C_GREY, label="random-row CV"),
    Patch(facecolor=C_BLUE, label="mixture-grouped CV"),
    Patch(facecolor=C_ORANGE, label=f"selected: {BEST_MODEL_NAME}"),
]
fig.legend(handles=legend_handles, loc="lower center", ncols=3, fontsize=6.6,
           frameon=False, bbox_to_anchor=(0.5, -0.03))
fig.suptitle(
    "Measured inputs only. Bars are cross-validated means; whiskers are "
    "Nadeau-Bengio corrected 95% intervals.", fontsize=7.4,
)
fig.tight_layout()
save_fig(fig, "Fig_02_ModelComparison",
         "Candidate model performance under random-row and mixture-grouped "
         "cross-validation. Grouped cross-validation keeps every measurement "
         "of a mixture in one fold and is the basis for model selection.")

fig, ax = plt.subplots(figsize=(HALF_W, 2.9))
sorted_optimism = optimism.sort_values("R2_optimism")
ax.barh(np.arange(len(sorted_optimism)), sorted_optimism["R2_optimism"],
        color=C_RED, alpha=0.85, height=0.6)
ax.set_yticks(np.arange(len(sorted_optimism)), sorted_optimism["model"],
              fontsize=6.2)
ax.axvline(0, color=C_DARK, lw=0.7)
ax.set_xlabel(r"$R^2$ optimism (random-row $-$ mixture-grouped)")
fig.tight_layout()
save_fig(fig, "Fig_03_RepeatedMeasureOptimism",
         "Apparent performance gain produced by splitting rows at random "
         "rather than by mixture. The difference is the optimism introduced "
         "when repeated measurements of one specimen appear in both the "
         "training and the evaluation fold.")


# =============================================================================
# Stage 1b: target-transform sensitivity
# =============================================================================
if RUN_TARGET_TRANSFORM_CHECK:
    section("STAGE 1b  TARGET-TRANSFORM SENSITIVITY")
    transform_rows = []
    for method in ["boxcox_fixed", "boxcox_mle", "log1p", "identity"]:
        folds, _ = evaluate_candidates(
            GROUPED_SPLITS, [BEST_MODEL_NAME], X_DEV, Y_DEV,
            transform_method=method, collect_predictions=False,
        )
        summary = summarise_folds(folds).iloc[0]
        transform_rows.append({
            "target_transform": method,
            "R2_mean": summary["R2_mean"], "RMSE_mean": summary["RMSE_mean"],
            "MAE_mean": summary["MAE_mean"], "MedAPE_mean": summary["MedAPE_mean"],
        })
        print(f"  {method:<14}: R2={summary['R2_mean']:.4f}  "
              f"RMSE={summary['RMSE_mean']:.4f}  MAE={summary['MAE_mean']:.4f}")
    transform_check = pd.DataFrame(transform_rows)
    transform_check["selected_for_analysis"] = (
        transform_check["target_transform"] == "boxcox_fixed"
    )
    save_table(transform_check.round(6), "Table_12_target_transform_sensitivity")
    best_transform = transform_check.sort_values("RMSE_mean").iloc[0]
    print(f"  lowest RMSE: {best_transform['target_transform']}; the analysis "
          f"retains the fixed Box-Cox transform for comparability")

TRANSFORM_METHOD = "boxcox_fixed"


# =============================================================================
# Stage 2: grouped-permutation importance for the selected model
# =============================================================================
section("STAGE 2  PERMUTATION IMPORTANCE")
IMPORTANCE_SPLITS = [split for split in GROUPED_SPLITS if split[0] == 1]
importance_records = []
for repeat, fold, train_index, test_index in IMPORTANCE_SPLITS:
    raw_train = X_DEV.iloc[train_index]
    raw_test = X_DEV.iloc[test_index]
    y_train = Y_DEV.iloc[train_index].to_numpy()
    y_test = Y_DEV.iloc[test_index].to_numpy()

    state = fit_preprocessing(raw_train)
    train_matrices = build_all_matrices(raw_train, state)
    test_matrices = build_all_matrices(raw_test, state)
    transform_state = fit_transform_state(y_train, TRANSFORM_METHOD)
    seed = RANDOM_STATE + 10000 * repeat + 100 * fold
    fitted = fit_named(BEST_MODEL_NAME, train_matrices,
                       transform_forward(y_train, transform_state), seed)
    baseline_rmse = metric_row(
        y_test, predict_named(BEST_MODEL_NAME, fitted, test_matrices,
                              transform_state)
    )["RMSE"]

    rng = np.random.default_rng(seed + 731)
    for factor in BASE_FEATURES_LOADED:
        for permutation in range(RANKING_PERMUTATION_REPEATS):
            row_order = rng.permutation(len(raw_test))
            permuted = {}
            for kind, matrix in test_matrices.items():
                positions = matrix_columns_for_factor(matrix.columns, factor)
                values = matrix.to_numpy().copy()
                if positions:
                    # Permute all encoded columns of one factor with a single
                    # row order so the factor stays internally consistent.
                    values[:, positions] = values[row_order][:, positions]
                permuted[kind] = pd.DataFrame(values, columns=matrix.columns)
            permuted_rmse = metric_row(
                y_test, predict_named(BEST_MODEL_NAME, fitted, permuted,
                                      transform_state)
            )["RMSE"]
            importance_records.append({
                "raw_factor": factor, "fold": fold, "permutation": permutation,
                "delta_RMSE": permuted_rmse - baseline_rmse,
            })
    del fitted, state, train_matrices, test_matrices
    cleanup()
    print(f"  fold {fold}/{len(IMPORTANCE_SPLITS)} complete")

importance_detail = pd.DataFrame(importance_records)
save_table(importance_detail, "Table_13_permutation_importance_detail")
importance = (
    importance_detail.groupby("raw_factor", as_index=False)
    .agg(delta_RMSE_mean=("delta_RMSE", "mean"),
         delta_RMSE_sd=("delta_RMSE", "std"),
         n_estimates=("delta_RMSE", "size"))
)
importance["label"] = importance["raw_factor"].map(
    lambda code: LABEL_MAP.get(code, code)
)
importance["selected_model"] = BEST_MODEL_NAME
importance = importance.sort_values(["delta_RMSE_mean", "raw_factor"],
                                    kind="stable").reset_index(drop=True)
importance["removal_step"] = np.arange(1, len(importance) + 1)
importance["importance_rank"] = len(importance) - importance["removal_step"] + 1
importance = importance[[
    "removal_step", "importance_rank", "raw_factor", "label", "selected_model",
    "delta_RMSE_mean", "delta_RMSE_sd", "n_estimates",
]]
save_table(importance.round(6), "Table_14_permutation_importance")
REMOVAL_ORDER = importance["raw_factor"].tolist()
print(f"\nmost important inputs for {BEST_MODEL_NAME}:")
print(importance.tail(8)[["raw_factor", "label", "delta_RMSE_mean"]]
      .iloc[::-1].to_string(index=False,
                            float_format=lambda value: f"{value:8.4f}"))

plot_importance = importance.sort_values("delta_RMSE_mean")
fig, ax = plt.subplots(figsize=(FULL_W, 5.6))
ax.barh(np.arange(len(plot_importance)), plot_importance["delta_RMSE_mean"],
        xerr=plot_importance["delta_RMSE_sd"].fillna(0.0), color=C_BLUE,
        alpha=0.9, error_kw={"elinewidth": 0.6, "capsize": 1.4})
ax.set_yticks(np.arange(len(plot_importance)),
              [f"{row.raw_factor}: {row.label}"
               for row in plot_importance.itertuples()], fontsize=5.8)
ax.axvline(0, color=C_DARK, lw=0.7)
ax.set_xlabel("Increase in held-out RMSE after grouped permutation (%)")
ax.set_title(f"{BEST_MODEL_NAME}: grouped-permutation importance, "
             f"mean and SD over {len(IMPORTANCE_SPLITS)} mixture-disjoint "
             f"folds x {RANKING_PERMUTATION_REPEATS} permutations", fontsize=7.4)
fig.tight_layout()
save_fig(fig, "Fig_04_PermutationImportance",
         "Grouped-permutation importance for the selected model. All encoded "
         "columns of an input are permuted together, so importance is "
         "reported per measured input rather than per encoded column.")


# =============================================================================
# Stage 3: cumulative least-important-first removal
# =============================================================================
LAST_SAFE_REMOVAL, FIRST_OUT_OF_TOLERANCE = 0, 1
SELECTED_REMOVALS = []
removal_summary = pd.DataFrame()
removal_out_of_fold = pd.DataFrame()

if RUN_REMOVAL_PATH:
    section("STAGE 3  CUMULATIVE INPUT REMOVAL")
    n_steps = min(MAX_RAW_FACTORS_TO_REMOVE, len(REMOVAL_ORDER) - 1)
    # An ensemble refits several learners per fold, so the removal path is by
    # far the most expensive stage. State the cost before starting it.
    per_fold = {STACK_NAME: len(STACK_COMPONENTS) * 6, BLEND_NAME: 2}.get(
        BEST_MODEL_NAME, 1
    )
    print(f"  {(n_steps + 1) * len(GROUPED_SPLITS) * per_fold} model fits "
          f"({n_steps + 1} steps x {len(GROUPED_SPLITS)} folds x {per_fold} "
          f"fit(s) per fold). Lower MAX_RAW_FACTORS_TO_REMOVE or "
          f"GROUPED_CV_REPEATS to shorten this stage.")
    removal_fold_rows, removal_prediction_rows = [], []
    for removal_count in range(n_steps + 1):
        excluded = REMOVAL_ORDER[:removal_count]
        removed = excluded[-1] if excluded else "none"
        folds, predictions = evaluate_candidates(
            GROUPED_SPLITS, [BEST_MODEL_NAME], X_DEV, Y_DEV,
            transform_method=TRANSFORM_METHOD, excluded=excluded,
        )
        folds["removal_count"] = removal_count
        folds["removed_code"] = removed
        folds["removed_label"] = LABEL_MAP.get(removed, removed) if excluded else "full model"
        folds["cumulative_removed"] = "; ".join(excluded)
        folds["inputs_retained"] = len(BASE_FEATURES_LOADED) - removal_count
        predictions["removal_count"] = removal_count
        removal_fold_rows.append(folds)
        removal_prediction_rows.append(predictions)
        print(f"  step {removal_count:2d}/{n_steps}: removed {removed:<9} "
              f"RMSE={folds['RMSE'].mean():.4f}  R2={folds['R2'].mean():.4f}")

    removal_folds = pd.concat(removal_fold_rows, ignore_index=True)
    removal_out_of_fold = pd.concat(removal_prediction_rows, ignore_index=True)
    save_table(removal_folds, "Table_15_removal_path_fold_metrics")
    save_table(removal_out_of_fold, "Table_16_removal_path_out_of_fold")

    summary_rows = []
    for removal_count, group in removal_folds.groupby("removal_count"):
        first = group.iloc[0]
        row = {
            "removal_count": removal_count,
            "removed_code": first["removed_code"],
            "removed_label": first["removed_label"],
            "cumulative_removed": first["cumulative_removed"],
            "inputs_retained": first["inputs_retained"],
            "n_cv_estimates": len(group),
        }
        for metric in ["R2", "RMSE", "MAE", "MedAPE"]:
            mean, lower, upper = corrected_t_interval(
                group[metric].to_numpy(), float(group["n_test"].mean()),
                float(group["n_train"].mean()),
            )
            row[f"{metric}_mean"] = mean
            row[f"{metric}_ci_lower"] = lower
            row[f"{metric}_ci_upper"] = upper
        summary_rows.append(row)
    removal_summary = pd.DataFrame(summary_rows).sort_values("removal_count")

    baseline_folds = removal_folds.loc[
        removal_folds["removal_count"].eq(0)
    ].sort_values(["repeat", "fold"])["RMSE"].to_numpy()
    test_rows = []
    for removal_count in range(1, n_steps + 1):
        current = removal_folds.loc[
            removal_folds["removal_count"].eq(removal_count)
        ].sort_values(["repeat", "fold"])["RMSE"].to_numpy()
        try:
            p_value = float(st.wilcoxon(current, baseline_folds,
                                        alternative="greater").pvalue)
        except ValueError:
            p_value = 1.0
        test_rows.append({"removal_count": removal_count,
                          "p_wilcoxon_RMSE_greater_raw": p_value})
    tests = pd.DataFrame(test_rows)
    tests["p_wilcoxon_RMSE_greater_holm"] = holm_adjust(
        tests["p_wilcoxon_RMSE_greater_raw"].to_numpy()
    )
    removal_summary = removal_summary.merge(tests, on="removal_count", how="left")

    baseline = removal_summary.loc[
        removal_summary["removal_count"].eq(0)
    ].iloc[0]
    removal_summary["delta_R2_vs_full"] = (
        removal_summary["R2_mean"] - baseline["R2_mean"]
    )
    removal_summary["relative_RMSE_change_pct"] = 100 * (
        removal_summary["RMSE_mean"] / baseline["RMSE_mean"] - 1
    )
    removal_summary["relative_MAE_change_pct"] = 100 * (
        removal_summary["MAE_mean"] / baseline["MAE_mean"] - 1
    )
    removal_summary["within_practical_tolerance"] = (
        (removal_summary["delta_R2_vs_full"] >= -R2_MAX_ABSOLUTE_LOSS)
        & (removal_summary["relative_RMSE_change_pct"] <= RMSE_MAX_RELATIVE_INCREASE_PCT)
        & (removal_summary["relative_MAE_change_pct"] <= MAE_MAX_RELATIVE_INCREASE_PCT)
    )
    removal_summary.loc[
        removal_summary["removal_count"].eq(0), "within_practical_tolerance"
    ] = True
    removal_summary["statistically_worse_after_holm"] = (
        removal_summary["p_wilcoxon_RMSE_greater_holm"] < 0.05
    )
    save_table(removal_summary.round(6), "Table_17_removal_path_summary")

    # The boundary is sequential: once a step falls outside tolerance, later
    # accidental recoveries do not reopen the path.
    unsafe = removal_summary.loc[
        (removal_summary["removal_count"] > 0)
        & (~removal_summary["within_practical_tolerance"])
    ].sort_values("removal_count")
    FIRST_OUT_OF_TOLERANCE = (int(unsafe.iloc[0]["removal_count"])
                              if len(unsafe) else n_steps + 1)
    LAST_SAFE_REMOVAL = FIRST_OUT_OF_TOLERANCE - 1
    SELECTED_REMOVALS = REMOVAL_ORDER[:LAST_SAFE_REMOVAL]
    safe_row = removal_summary.loc[
        removal_summary["removal_count"].eq(LAST_SAFE_REMOVAL)
    ].iloc[0]
    print(f"\n  last safe removal: {LAST_SAFE_REMOVAL} input(s); "
          f"{len(BASE_FEATURES_LOADED) - LAST_SAFE_REMOVAL} retained")
    if FIRST_OUT_OF_TOLERANCE <= n_steps:
        harmful = removal_summary.loc[
            removal_summary["removal_count"].eq(FIRST_OUT_OF_TOLERANCE)
        ].iloc[0]
        print(f"  first step outside tolerance: {FIRST_OUT_OF_TOLERANCE} "
              f"({harmful['removed_code']}: {harmful['removed_label']}), "
              f"RMSE {harmful['relative_RMSE_change_pct']:+.1f}%")
    save_table(pd.DataFrame([{
        "selected_model": BEST_MODEL_NAME,
        "last_safe_removal_count": LAST_SAFE_REMOVAL,
        "inputs_retained": len(BASE_FEATURES_LOADED) - LAST_SAFE_REMOVAL,
        "removed_inputs": "; ".join(SELECTED_REMOVALS),
        "removed_labels": "; ".join(LABEL_MAP.get(code, code)
                                    for code in SELECTED_REMOVALS),
        "first_out_of_tolerance_count": FIRST_OUT_OF_TOLERANCE,
        "max_absolute_R2_loss": R2_MAX_ABSOLUTE_LOSS,
        "max_relative_RMSE_increase_pct": RMSE_MAX_RELATIVE_INCREASE_PCT,
        "max_relative_MAE_increase_pct": MAE_MAX_RELATIVE_INCREASE_PCT,
    }]), "Table_18_removal_boundary")

    fig, axes = plt.subplots(2, 2, figsize=(FULL_W, 4.6), sharex=True)
    for ax, metric, ylabel, letter in zip(
            axes.ravel(), ["R2", "RMSE", "MAE", "MedAPE"],
            [r"Cross-validated $R^2$", "RMSE (%)", "MAE (%)", "MedAPE (%)"],
            "abcd"):
        ax.plot(removal_summary["removal_count"], removal_summary[f"{metric}_mean"],
                marker="o", ms=2.8, lw=1.2, color=C_BLUE)
        ax.fill_between(removal_summary["removal_count"].to_numpy(),
                        removal_summary[f"{metric}_ci_lower"].to_numpy(),
                        removal_summary[f"{metric}_ci_upper"].to_numpy(),
                        color=C_BLUE, alpha=0.13, linewidth=0)
        ax.axvline(LAST_SAFE_REMOVAL, color=C_DARK, ls="--", lw=0.9)
        if FIRST_OUT_OF_TOLERANCE <= n_steps:
            ax.axvline(FIRST_OUT_OF_TOLERANCE, color=C_RED, ls=":", lw=1.0)
        ax.set_ylabel(ylabel)
        panel_label(ax, letter)
    for ax in axes.ravel():
        ax.set_xticks(np.arange(0, n_steps + 1, 1 if n_steps <= 10 else 2))
    for ax in axes[1]:
        ax.set_xlabel("Measured inputs cumulatively removed")
    fig.suptitle(
        f"{BEST_MODEL_NAME} under mixture-grouped cross-validation. Dashed "
        "line, last step within tolerance; dotted line, first step outside.",
        fontsize=7.4,
    )
    fig.tight_layout()
    save_fig(fig, "Fig_05_CumulativeRemoval",
             "Performance along the fixed least-important-first removal path. "
             "Shading shows corrected 95% intervals across mixture-disjoint "
             "folds.")

FINAL_EXCLUDED = list(SELECTED_REMOVALS)
FINAL_INPUTS = [c for c in BASE_FEATURES_LOADED if c not in FINAL_EXCLUDED]


# =============================================================================
# Stage 4: final fit and locked holdout evaluation
# =============================================================================
section("STAGE 4  LOCKED HOLDOUT")


def fit_final(raw_train, y_train, excluded, seed):
    state = fit_preprocessing(raw_train, excluded)
    matrices = build_all_matrices(raw_train, state, excluded)
    transform_state = fit_transform_state(y_train.to_numpy()
                                          if hasattr(y_train, "to_numpy")
                                          else y_train, TRANSFORM_METHOD)
    y_values = np.asarray(y_train, float)
    fitted = fit_named(BEST_MODEL_NAME, matrices,
                       transform_forward(y_values, transform_state), seed)
    return {"state": state, "matrices": matrices, "transform": transform_state,
            "fitted": fitted, "excluded": list(excluded)}


def predict_final(bundle, raw_frame):
    matrices = build_all_matrices(raw_frame, bundle["state"], bundle["excluded"])
    return predict_named(BEST_MODEL_NAME, bundle["fitted"], matrices,
                         bundle["transform"]), matrices


holdout_rows, holdout_predictions, bundles = [], [], {}
subsets = [("all measured inputs", [])]
if FINAL_EXCLUDED:
    subsets.append((f"reduced inputs (remove {len(FINAL_EXCLUDED)})",
                    FINAL_EXCLUDED))
FINAL_SUBSET_NAME = subsets[-1][0]

for subset_name, excluded in subsets:
    bundle = fit_final(X_DEV, Y_DEV, excluded, RANDOM_STATE + 90000)
    holdout_prediction, holdout_matrices = predict_final(bundle, X_HOLDOUT)
    development_prediction = predict_named(
        BEST_MODEL_NAME, bundle["fitted"], bundle["matrices"], bundle["transform"]
    )
    bundle.update({"holdout_prediction": holdout_prediction,
                   "development_prediction": development_prediction,
                   "holdout_matrices": holdout_matrices})
    bundles[subset_name] = bundle
    point = metric_row(Y_HOLDOUT.to_numpy(), holdout_prediction)
    holdout_rows.append({
        "subset": subset_name, "model": BEST_MODEL_NAME,
        "inputs_removed": len(excluded),
        "removed_codes": "; ".join(excluded),
        "n_holdout_rows": len(Y_HOLDOUT),
        "n_holdout_mixtures": int(pd.Series(GROUPS[HOLDOUT_INDEX]).nunique()),
        **point,
        **bootstrap_metric_interval(Y_HOLDOUT.to_numpy(), holdout_prediction,
                                    HOLDOUT_BOOTSTRAP_REPLICATES,
                                    RANDOM_STATE + 91000 + len(excluded)),
    })
    for position, predicted in enumerate(holdout_prediction):
        holdout_predictions.append({
            "subset": subset_name, "holdout_row": position,
            "measured": float(Y_HOLDOUT.iloc[position]),
            "predicted": float(predicted),
            "residual": float(Y_HOLDOUT.iloc[position] - predicted),
        })

holdout_results = pd.DataFrame(holdout_rows)
save_table(holdout_results.round(6), "Table_19_locked_holdout")
save_table(pd.DataFrame(holdout_predictions), "Table_20_locked_holdout_predictions")
print(holdout_results[["subset", "R2", "RMSE", "MAE", "MedAPE"]].to_string(
    index=False, float_format=lambda value: f"{value:9.4f}"))

FINAL_BUNDLE = bundles[FINAL_SUBSET_NAME]
FINAL_HOLDOUT_PREDICTION = FINAL_BUNDLE["holdout_prediction"]
FINAL_DEVELOPMENT_PREDICTION = FINAL_BUNDLE["development_prediction"]
holdout_metrics = metric_row(Y_HOLDOUT.to_numpy(), FINAL_HOLDOUT_PREDICTION)
development_metrics = metric_row(Y_DEV.to_numpy(), FINAL_DEVELOPMENT_PREDICTION)

paired_holdout_differences = pd.DataFrame()
if FINAL_EXCLUDED:
    paired_holdout_differences = paired_difference_bootstrap(
        Y_HOLDOUT.to_numpy(), bundles["all measured inputs"]["holdout_prediction"],
        FINAL_HOLDOUT_PREDICTION, HOLDOUT_BOOTSTRAP_REPLICATES,
        RANDOM_STATE + 92500,
    )
    paired_holdout_differences.insert(0, "candidate", FINAL_SUBSET_NAME)
    paired_holdout_differences.insert(0, "reference", "all measured inputs")
    save_table(paired_holdout_differences.round(6),
               "Table_21_paired_holdout_difference")

# Optimistic reference: the same model under a random-row holdout, reported so
# the magnitude of the repeated-measure effect is visible at the final stage.
reference_bundle = fit_final(
    X_RAW.iloc[RANDOM_DEV_INDEX].reset_index(drop=True),
    Y.iloc[RANDOM_DEV_INDEX].reset_index(drop=True),
    FINAL_EXCLUDED, RANDOM_STATE + 90500,
)
reference_prediction, _ = predict_final(
    reference_bundle, X_RAW.iloc[RANDOM_HOLDOUT_INDEX].reset_index(drop=True)
)
reference_metrics = metric_row(Y.iloc[RANDOM_HOLDOUT_INDEX].to_numpy(),
                               reference_prediction)
save_table(pd.DataFrame([
    {"holdout_design": "mixture-disjoint (primary)", **holdout_metrics},
    {"holdout_design": "random-row (optimistic reference)", **reference_metrics},
]).round(6), "Table_22_holdout_design_comparison")
print(f"\n  mixture-disjoint holdout : R2={holdout_metrics['R2']:.4f}  "
      f"RMSE={holdout_metrics['RMSE']:.4f}")
print(f"  random-row holdout       : R2={reference_metrics['R2']:.4f}  "
      f"RMSE={reference_metrics['RMSE']:.4f}  (optimistic reference)")
del reference_bundle
cleanup()

# Pooled out-of-fold predictions of the final input set, averaged per row.
final_out_of_fold = pd.DataFrame()
if RUN_REMOVAL_PATH and len(removal_out_of_fold):
    subset = removal_out_of_fold.loc[
        removal_out_of_fold["removal_count"].eq(LAST_SAFE_REMOVAL)
    ]
else:
    subset = comparison_predictions[PRIMARY_SCHEME].loc[
        comparison_predictions[PRIMARY_SCHEME]["model"].eq(BEST_MODEL_NAME)
    ]
final_out_of_fold = (
    subset.groupby("development_row", as_index=False)
    .agg(measured=("measured", "first"), predicted=("predicted", "mean"),
         prediction_sd=("predicted", "std"), n_predictions=("predicted", "size"))
    .sort_values("development_row").reset_index(drop=True)
)
final_out_of_fold["residual"] = (
    final_out_of_fold["measured"] - final_out_of_fold["predicted"]
)
save_table(final_out_of_fold.round(6), "Table_23_out_of_fold_predictions")
out_of_fold_metrics = metric_row(final_out_of_fold["measured"].to_numpy(),
                                 final_out_of_fold["predicted"].to_numpy())

evaluation_summary = pd.DataFrame([
    {"Evaluation": "Mixture-grouped CV (mean of folds)",
     **{k: v for k, v in zip(
         ["R2", "RMSE", "MAE", "MedAPE"],
         [grouped_summary.loc[BEST_MODEL_NAME, f"{m}_mean"]
          for m in ["R2", "RMSE", "MAE", "MedAPE"]])},
     "Role": "primary generalization estimate"},
    {"Evaluation": "Mixture-grouped CV (pooled out-of-fold)",
     **{k: out_of_fold_metrics[k] for k in ["R2", "RMSE", "MAE", "MedAPE"]},
     "Role": "out-of-sample prediction plot"},
    {"Evaluation": "Locked mixture-disjoint holdout",
     **{k: holdout_metrics[k] for k in ["R2", "RMSE", "MAE", "MedAPE"]},
     "Role": "final internal test"},
    {"Evaluation": "Random-row holdout",
     **{k: reference_metrics[k] for k in ["R2", "RMSE", "MAE", "MedAPE"]},
     "Role": "optimistic reference only"},
    {"Evaluation": "Development fit (in-sample)",
     **{k: development_metrics[k] for k in ["R2", "RMSE", "MAE", "MedAPE"]},
     "Role": "overfitting check only"},
])
save_table(evaluation_summary.round(6), "Table_24_evaluation_summary")
print("\nevaluation summary")
print(evaluation_summary.to_string(index=False,
                                   float_format=lambda value: f"{value:9.4f}"))

axis_low = float(min(final_out_of_fold["measured"].min(), Y_HOLDOUT.min(),
                     final_out_of_fold["predicted"].min(),
                     FINAL_HOLDOUT_PREDICTION.min()))
axis_high = float(max(final_out_of_fold["measured"].max(), Y_HOLDOUT.max(),
                      final_out_of_fold["predicted"].max(),
                      FINAL_HOLDOUT_PREDICTION.max()))
pad = 0.03 * (axis_high - axis_low)
LIMITS = (axis_low - pad, axis_high + pad)

fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 3.3), sharex=True, sharey=True)
for ax, measured, predicted, title, colour, metrics_here, letter in [
    (axes[0], final_out_of_fold["measured"].to_numpy(),
     final_out_of_fold["predicted"].to_numpy(),
     f"Mixture-grouped out-of-fold (n={len(final_out_of_fold)})", C_BLUE,
     out_of_fold_metrics, "a"),
    (axes[1], Y_HOLDOUT.to_numpy(), FINAL_HOLDOUT_PREDICTION,
     f"Locked mixture-disjoint holdout (n={len(Y_HOLDOUT)})", C_ORANGE,
     holdout_metrics, "b"),
]:
    ax.plot(LIMITS, LIMITS, "--", color=C_DARK, lw=0.9, label="1:1 line")
    ax.scatter(measured, predicted, s=11, alpha=0.5, color=colour,
               edgecolors="none")
    ax.set_xlim(LIMITS)
    ax.set_ylim(LIMITS)
    ax.set_xlabel(f"Measured {TARGET_LABEL}")
    ax.set_ylabel(f"Predicted {TARGET_LABEL}")
    ax.set_title(title, fontsize=7.2)
    ax.text(0.97, 0.04,
            f"$R^2$ = {metrics_here['R2']:.4f}\n"
            f"RMSE = {metrics_here['RMSE']:.4f} %\n"
            f"MAE = {metrics_here['MAE']:.4f} %\n"
            f"MedAPE = {metrics_here['MedAPE']:.1f} %",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6.2,
            bbox={"facecolor": "white", "edgecolor": "#BBBBBB", "alpha": 0.85})
    ax.legend(loc="upper left")
    panel_label(ax, letter)
fig.suptitle(f"Final model: {BEST_MODEL_NAME} on {len(FINAL_INPUTS)} measured "
             f"inputs", fontsize=7.6)
fig.tight_layout()
save_fig(fig, "Fig_06_ActualVsPredicted",
         "Measured against predicted expansion for out-of-fold development "
         "predictions and the locked holdout. No mixture contributes rows to "
         "both a training fold and its evaluation fold.")

residuals = Y_HOLDOUT.to_numpy() - FINAL_HOLDOUT_PREDICTION
fig, axes = plt.subplots(1, 3, figsize=(FULL_W, 2.5))
axes[0].scatter(FINAL_HOLDOUT_PREDICTION, residuals, s=11, alpha=0.55,
                color=C_ORANGE, edgecolors="none")
axes[0].axhline(0, color=C_DARK, ls="--", lw=0.8)
axes[0].set_xlabel(f"Predicted {TARGET_LABEL}")
axes[0].set_ylabel("Residual (%)")
axes[1].hist(residuals, bins=28, density=True, color=C_ORANGE, alpha=0.7)
grid_values = np.linspace(residuals.min(), residuals.max(), 250)
axes[1].plot(grid_values, st.norm.pdf(grid_values, residuals.mean(),
                                      residuals.std(ddof=1)),
             color=C_DARK, lw=1.0)
axes[1].axvline(0, color=C_RED, ls="--", lw=0.8)
axes[1].set_xlabel("Residual (%)")
axes[1].set_ylabel("Density")
theoretical, ordered = st.probplot(residuals, dist="norm", fit=False)
slope, intercept, _ = st.probplot(residuals, dist="norm", fit=True)[1]
axes[2].scatter(theoretical, ordered, s=10, color=C_ORANGE, alpha=0.75,
                edgecolors="none")
line_x = np.array([theoretical.min(), theoretical.max()])
axes[2].plot(line_x, intercept + slope * line_x, color=C_DARK, lw=0.9)
axes[2].set_xlabel("Theoretical quantiles")
axes[2].set_ylabel("Ordered residuals (%)")
for ax, letter in zip(axes, "abc"):
    panel_label(ax, letter, dx=-0.18)
fig.tight_layout()
save_fig(fig, "Fig_07_ResidualDiagnostics",
         "Locked-holdout residual diagnostics: residuals against predictions, "
         "residual density with a fitted normal curve, and a normal "
         "quantile-quantile plot.")


# =============================================================================
# Stage 5: split-conformal prediction intervals
# =============================================================================
conformal_summary = pd.DataFrame()
if RUN_CONFORMAL:
    section("STAGE 5  CONFORMAL PREDICTION INTERVALS")
    proper_index, calibration_index = grouped_holdout(
        GROUPS_DEV, CONFORMAL_CALIBRATION_FRACTION, RANDOM_STATE + 4242
    )
    conformal_bundle = fit_final(
        X_DEV.iloc[proper_index].reset_index(drop=True),
        Y_DEV.iloc[proper_index].reset_index(drop=True),
        FINAL_EXCLUDED, RANDOM_STATE + 93000,
    )
    calibration_prediction, _ = predict_final(
        conformal_bundle, X_DEV.iloc[calibration_index].reset_index(drop=True)
    )
    scores = np.abs(Y_DEV.iloc[calibration_index].to_numpy()
                    - calibration_prediction)
    conformal_prediction, _ = predict_final(conformal_bundle, X_HOLDOUT)

    rows = []
    for alpha in [0.20, 0.10, 0.05]:
        rank = int(np.ceil((len(scores) + 1) * (1 - alpha)))
        half_width = float(np.sort(scores)[min(max(rank, 1), len(scores)) - 1])
        covered = np.abs(Y_HOLDOUT.to_numpy() - conformal_prediction) <= half_width
        rows.append({
            "nominal_coverage": 1 - alpha,
            "half_width": half_width,
            "empirical_coverage": float(covered.mean()),
            "mean_interval_width": 2 * half_width,
            "n_calibration_rows": len(scores),
            "n_holdout_rows": len(Y_HOLDOUT),
        })
    conformal_summary = pd.DataFrame(rows)
    save_table(conformal_summary.round(6), "Table_25_conformal_coverage")
    print(conformal_summary.to_string(index=False,
                                      float_format=lambda value: f"{value:9.4f}"))

    primary = conformal_summary.loc[
        conformal_summary["nominal_coverage"].sub(1 - CONFORMAL_ALPHA).abs().idxmin()
    ]
    half_width = float(primary["half_width"])
    covered = np.abs(Y_HOLDOUT.to_numpy() - conformal_prediction) <= half_width
    order = np.argsort(conformal_prediction)
    positions = np.arange(len(Y_HOLDOUT))
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.8),
                             gridspec_kw={"width_ratios": [2.1, 1.0]})
    axes[0].fill_between(positions, conformal_prediction[order] - half_width,
                         conformal_prediction[order] + half_width,
                         color=C_BLUE, alpha=0.18, lw=0,
                         label=f"{primary['nominal_coverage']:.0%} interval")
    axes[0].plot(positions, conformal_prediction[order], color=C_BLUE, lw=0.9,
                 label="prediction")
    axes[0].scatter(positions[covered[order]],
                    Y_HOLDOUT.to_numpy()[order][covered[order]], s=7,
                    color=C_GREY, lw=0, zorder=3, label="covered")
    axes[0].scatter(positions[~covered[order]],
                    Y_HOLDOUT.to_numpy()[order][~covered[order]], s=9,
                    color=C_RED, lw=0, zorder=4, label="outside interval")
    axes[0].set_xlabel("Holdout specimens, ordered by prediction")
    axes[0].set_ylabel(TARGET_LABEL)
    axes[0].text(0.02, 0.96,
                 f"empirical coverage {covered.mean():.1%}\n"
                 f"half-width {half_width:.3f} %",
                 transform=axes[0].transAxes, va="top", fontsize=6.5)
    axes[0].legend(loc="lower right", ncols=2, fontsize=6.2)
    panel_label(axes[0], "a", dx=-0.07)

    axes[1].plot([0.7, 1.0], [0.7, 1.0], "--", color=C_DARK, lw=0.8)
    axes[1].plot(conformal_summary["nominal_coverage"],
                 conformal_summary["empirical_coverage"], "o-", color=C_ORANGE,
                 ms=4, lw=1.2)
    axes[1].set_xlabel("Nominal coverage")
    axes[1].set_ylabel("Empirical coverage")
    axes[1].set_xlim(0.72, 1.0)
    axes[1].set_ylim(0.72, 1.0)
    panel_label(axes[1], "b", dx=-0.22)
    fig.tight_layout()
    save_fig(fig, "Fig_08_ConformalIntervals",
             "Split-conformal prediction intervals calibrated on a "
             "mixture-disjoint subset of the development partition. a, "
             "Intervals on the locked holdout. b, Empirical against nominal "
             "coverage.")
    del conformal_bundle
    cleanup()


# =============================================================================
# Stage 6: reactivity classification at standard expansion limits
# =============================================================================
reactivity_results = pd.DataFrame()
if RUN_REACTIVITY_CLASSIFICATION:
    section("STAGE 6  REACTIVITY CLASSIFICATION")
    from sklearn.metrics import confusion_matrix, matthews_corrcoef

    measured = Y_HOLDOUT.to_numpy()
    applicable = [
        (limit, name) for limit, name in REACTIVITY_LIMITS
        if (measured > limit).sum() >= 5 and (measured <= limit).sum() >= 5
    ]
    rows, matrices = [], {}
    for limit, name in applicable:
        truth = measured > limit
        predicted_class = FINAL_HOLDOUT_PREDICTION > limit
        matrix = confusion_matrix(truth, predicted_class, labels=[False, True])
        matrices[limit] = matrix
        true_negative, false_positive, false_negative, true_positive = matrix.ravel()
        rows.append({
            "limit_percent": limit, "criterion": name,
            "n_exceeding": int(truth.sum()), "n_not_exceeding": int((~truth).sum()),
            "accuracy": float((truth == predicted_class).mean()),
            "sensitivity": float(true_positive / max(true_positive + false_negative, 1)),
            "specificity": float(true_negative / max(true_negative + false_positive, 1)),
            "precision": float(true_positive / max(true_positive + false_positive, 1)),
            "matthews_correlation": float(matthews_corrcoef(truth, predicted_class)),
            "roc_auc_using_predicted_expansion": float(
                roc_auc_score(truth, FINAL_HOLDOUT_PREDICTION)
            ),
            "false_negative_count": int(false_negative),
            "false_positive_count": int(false_positive),
        })
    reactivity_results = pd.DataFrame(rows)
    if len(reactivity_results):
        save_table(reactivity_results.round(5), "Table_26_reactivity_classification")
        print(reactivity_results[
            ["limit_percent", "accuracy", "sensitivity", "specificity",
             "matthews_correlation", "roc_auc_using_predicted_expansion"]
        ].to_string(index=False, float_format=lambda value: f"{value:8.4f}"))

        n_limits = len(applicable)
        fig, axes = plt.subplots(1, n_limits + 1,
                                 figsize=(FULL_W, 2.5 + 0.0 * n_limits))
        axes = np.atleast_1d(axes)
        for colour, (limit, name) in zip([C_BLUE, C_ORANGE, C_GREEN], applicable):
            truth = measured > limit
            false_positive_rate, true_positive_rate, _ = roc_curve(
                truth, FINAL_HOLDOUT_PREDICTION
            )
            axes[0].plot(false_positive_rate, true_positive_rate, lw=1.2,
                         color=colour,
                         label=f"{limit:.2f} % (AUC "
                               f"{roc_auc_score(truth, FINAL_HOLDOUT_PREDICTION):.3f})")
        axes[0].plot([0, 1], [0, 1], "--", color=C_DARK, lw=0.7)
        axes[0].set_xlabel("False-positive rate")
        axes[0].set_ylabel("True-positive rate")
        axes[0].legend(loc="lower right", fontsize=6.0, frameon=True,
                       framealpha=0.85, edgecolor="#CCCCCC")
        panel_label(axes[0], "a", dx=-0.22)
        for position, (limit, name) in enumerate(applicable, start=1):
            matrix = matrices[limit]
            ax = axes[position]
            ax.imshow(matrix, cmap="Blues", vmin=0, vmax=matrix.max())
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, f"{matrix[i, j]}", ha="center", va="center",
                            fontsize=7,
                            color="white" if matrix[i, j] > matrix.max() / 2
                            else C_DARK)
            ax.set_xticks([0, 1], ["below", "above"], fontsize=6)
            ax.set_yticks([0, 1], ["below", "above"], fontsize=6)
            ax.set_xlabel("Predicted")
            if position == 1:
                ax.set_ylabel("Measured")
            ax.set_title(f"{limit:.2f} % limit", fontsize=7)
            panel_label(ax, "abcd"[position], dx=-0.22)
        fig.suptitle("Classification of reactivity against standard expansion "
                     "limits on the locked holdout", fontsize=7.4)
        fig.tight_layout()
        save_fig(fig, "Fig_09_ReactivityClassification",
                 "Ability of the continuous expansion prediction to reproduce "
                 "the reactivity classification used in practice. a, Receiver "
                 "operating characteristic curves. b-d, Confusion matrices "
                 "obtained by applying each limit directly to the predicted "
                 "expansion.")
    else:
        print("  no limit has enough holdout specimens on both sides; skipped")


# =============================================================================
# Stage 7: y-randomization control
# =============================================================================
permutation_control = pd.DataFrame()
if RUN_PERMUTATION_CONTROL:
    section("STAGE 7  Y-RANDOMIZATION CONTROL")
    rng = np.random.default_rng(RANDOM_STATE + 5150)
    rows = []
    for replicate in range(1, N_PERMUTATION_CONTROLS + 1):
        shuffled = pd.Series(rng.permutation(Y_DEV.to_numpy()))
        bundle = fit_final(X_DEV, shuffled, FINAL_EXCLUDED,
                           RANDOM_STATE + 94000 + replicate)
        prediction, _ = predict_final(bundle, X_HOLDOUT)
        rows.append({"replicate": replicate,
                     **metric_row(Y_HOLDOUT.to_numpy(), prediction)})
        del bundle
        cleanup()
    permutation_control = pd.DataFrame(rows)
    save_table(permutation_control.round(6), "Table_27_y_randomization")
    print(f"  permuted-target holdout R2 = "
          f"{permutation_control['R2'].mean():.4f} +/- "
          f"{permutation_control['R2'].std(ddof=1):.4f}  "
          f"(true model {holdout_metrics['R2']:.4f})")


# =============================================================================
# Stage 8: learning curve over mixtures
# =============================================================================
learning_curve = pd.DataFrame()
if RUN_LEARNING_CURVE:
    section("STAGE 8  LEARNING CURVE")
    unique_groups = np.unique(GROUPS_DEV)
    rng = np.random.default_rng(RANDOM_STATE + 6060)
    shuffled_groups = rng.permutation(unique_groups)
    rows = []
    for fraction in LEARNING_CURVE_FRACTIONS:
        n_groups = max(int(round(fraction * len(shuffled_groups))), 2)
        chosen = set(shuffled_groups[:n_groups])
        mask = np.array([group in chosen for group in GROUPS_DEV])
        bundle = fit_final(
            X_DEV.loc[mask].reset_index(drop=True),
            Y_DEV.loc[mask].reset_index(drop=True),
            FINAL_EXCLUDED, RANDOM_STATE + 95000,
        )
        prediction, _ = predict_final(bundle, X_HOLDOUT)
        result = metric_row(Y_HOLDOUT.to_numpy(), prediction)
        rows.append({"fraction_of_mixtures": fraction, "n_mixtures": n_groups,
                     "n_rows": int(mask.sum()), **result})
        print(f"  {n_groups:4d} mixtures ({int(mask.sum()):4d} rows): "
              f"R2={result['R2']:.4f}  RMSE={result['RMSE']:.4f}")
        del bundle
        cleanup()
    learning_curve = pd.DataFrame(rows)
    save_table(learning_curve.round(6), "Table_28_learning_curve")

if len(permutation_control) or len(learning_curve):
    n_panels = int(len(permutation_control) > 0) + int(len(learning_curve) > 0)
    fig, axes = plt.subplots(1, n_panels, figsize=(FULL_W * n_panels / 2, 2.7))
    axes = np.atleast_1d(axes)
    index = 0
    if len(permutation_control):
        ax = axes[index]
        ax.hist(permutation_control["R2"], bins=max(5, N_PERMUTATION_CONTROLS // 2),
                color=C_GREY, alpha=0.85,
                label=f"permuted targets (n={len(permutation_control)})")
        ax.axvline(holdout_metrics["R2"], color=C_RED, lw=1.4,
                   label="fitted model")
        ax.set_xlabel(r"Holdout $R^2$")
        ax.set_ylabel("Count")
        ax.legend(loc="upper center", fontsize=6.2)
        panel_label(ax, "ab"[index])
        index += 1
    if len(learning_curve):
        ax = axes[index]
        ax.plot(learning_curve["n_mixtures"], learning_curve["R2"], "o-",
                color=C_BLUE, ms=4, lw=1.2)
        ax.set_xlabel("Mixtures used for training")
        ax.set_ylabel(r"Holdout $R^2$", color=C_BLUE)
        ax.tick_params(axis="y", labelcolor=C_BLUE)
        twin = ax.twinx()
        twin.plot(learning_curve["n_mixtures"], learning_curve["RMSE"], "s--",
                  color=C_ORANGE, ms=4, lw=1.2)
        twin.set_ylabel("Holdout RMSE (%)", color=C_ORANGE)
        twin.tick_params(axis="y", labelcolor=C_ORANGE)
        twin.spines["right"].set_visible(True)
        panel_label(ax, "ab"[index])
    fig.tight_layout()
    save_fig(fig, "Fig_10_ControlsAndDataEfficiency",
             "Model controls. Left, holdout performance after randomly "
             "permuting the training targets, which collapses to chance and "
             "rules out leakage as an explanation of the fitted result. "
             "Right, holdout performance as the number of training mixtures "
             "increases.")


# =============================================================================
# Stage 9: calibration and error stratification
# =============================================================================
section("STAGE 9  CALIBRATION AND ERROR REGIMES")


def calibration_table(measured_values, predicted_values, source, n_bins=10):
    frame = pd.DataFrame({"measured": np.asarray(measured_values, float),
                          "predicted": np.asarray(predicted_values, float)})
    frame["bin"] = pd.qcut(frame["predicted"], q=n_bins, labels=False,
                           duplicates="drop")
    rows = []
    for bin_index, group in frame.groupby("bin", observed=True):
        values = group["measured"].to_numpy()
        mean = float(values.mean())
        if len(values) > 1 and values.std(ddof=1) > 0:
            half = st.t.ppf((1 + CI_LEVEL) / 2, len(values) - 1) * st.sem(values)
        else:
            half = 0.0
        rows.append({
            "source": source, "predicted_decile": int(bin_index) + 1,
            "n": len(group), "mean_predicted": float(group["predicted"].mean()),
            "mean_measured": mean, "mean_measured_ci_lower": mean - half,
            "mean_measured_ci_upper": mean + half,
            "mean_residual": float((group["measured"] - group["predicted"]).mean()),
        })
    return pd.DataFrame(rows)


calibration = pd.concat([
    calibration_table(final_out_of_fold["measured"], final_out_of_fold["predicted"],
                      "Grouped out-of-fold"),
    calibration_table(Y_HOLDOUT.to_numpy(), FINAL_HOLDOUT_PREDICTION,
                      "Locked holdout"),
], ignore_index=True)
save_table(calibration.round(6), "Table_29_calibration")

error_frame = pd.DataFrame({
    "measured": Y_HOLDOUT.to_numpy(), "predicted": FINAL_HOLDOUT_PREDICTION,
    "absolute_error": np.abs(residuals),
})
error_frame["measured_quartile"] = pd.qcut(
    error_frame["measured"], q=4, labels=["Q1", "Q2", "Q3", "Q4"],
    duplicates="drop",
)
save_table(error_frame.round(6), "Table_30_error_by_measured_quartile")

fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 3.0))
bounds = [
    float(min(calibration["mean_predicted"].min(),
              calibration["mean_measured_ci_lower"].min())),
    float(max(calibration["mean_predicted"].max(),
              calibration["mean_measured_ci_upper"].max())),
]
span = 0.04 * (bounds[1] - bounds[0])
bounds = (bounds[0] - span, bounds[1] + span)
for source, colour, marker in [("Grouped out-of-fold", C_BLUE, "o"),
                               ("Locked holdout", C_ORANGE, "s")]:
    rows = calibration[calibration["source"].eq(source)]
    axes[0].errorbar(
        rows["mean_predicted"], rows["mean_measured"],
        yerr=np.vstack([rows["mean_measured"] - rows["mean_measured_ci_lower"],
                        rows["mean_measured_ci_upper"] - rows["mean_measured"]]),
        color=colour, marker=marker, ms=3.5, lw=0.9, capsize=2, label=source,
    )
axes[0].plot(bounds, bounds, "--", color=C_DARK, lw=0.8, label="perfect calibration")
axes[0].set_xlim(bounds)
axes[0].set_ylim(bounds)
axes[0].set_xlabel(f"Mean predicted {TARGET_LABEL}")
axes[0].set_ylabel(f"Mean measured {TARGET_LABEL}")
axes[0].legend(loc="upper left", fontsize=6.2)
panel_label(axes[0], "a", dx=-0.18)

quartiles = ["Q1", "Q2", "Q3", "Q4"]
quartile_errors = [
    error_frame.loc[error_frame["measured_quartile"].eq(q),
                    "absolute_error"].to_numpy() for q in quartiles
]
quartile_errors = [values for values in quartile_errors if len(values)]
axes[1].boxplot(quartile_errors, tick_labels=quartiles[:len(quartile_errors)],
                showfliers=False,
                medianprops={"color": C_ORANGE, "linewidth": 1.2})
jitter_rng = np.random.default_rng(RANDOM_STATE + 96000)
for position, values in enumerate(quartile_errors, 1):
    axes[1].scatter(position + jitter_rng.normal(0, 0.045, len(values)), values,
                    s=7, alpha=0.22, color=C_BLUE, edgecolors="none")
axes[1].set_xlabel("Measured-expansion quartile")
axes[1].set_ylabel("Absolute holdout error (%)")
panel_label(axes[1], "b", dx=-0.18)
fig.tight_layout()
save_fig(fig, "Fig_11_CalibrationAndErrorRegimes",
         "Calibration by predicted-value decile and absolute error across "
         "measured-expansion quartiles on the locked holdout.")


# =============================================================================
# Stage 10: SHAP attribution
# =============================================================================
# SHAP is computed in the ordinal input space, which carries exactly one column
# per measured input. Attribution is therefore reported per input rather than
# per encoded column, and no post-hoc aggregation of one-hot levels is needed.
ORDINAL_TRAIN = FINAL_BUNDLE["matrices"]["ordinal"]
ORDINAL_HOLDOUT = FINAL_BUNDLE["holdout_matrices"]["ordinal"]
ONEHOT_COLUMNS = FINAL_BUNDLE["matrices"]["onehot"].columns


def ordinal_to_matrices(values, state=None):
    """Rebuild both design matrices from ordinal-space values."""
    state = state or FINAL_BUNDLE["state"]
    ordinal = pd.DataFrame(np.asarray(values, float),
                           columns=ORDINAL_TRAIN.columns)
    parts = [ordinal[[c for c in ORDINAL_TRAIN.columns
                      if c not in CATEGORICAL_FEATURES_LOADED]]]
    for column in CATEGORICAL_FEATURES_LOADED:
        if column not in ORDINAL_TRAIN.columns:
            continue
        levels = state["categories"][column]
        codes = np.clip(np.rint(ordinal[column].to_numpy()).astype(int),
                        0, len(levels) - 1)
        encoded = np.zeros((len(ordinal), len(levels)))
        encoded[np.arange(len(ordinal)), codes] = 1.0
        parts.append(pd.DataFrame(
            encoded, columns=[f"{column}={level}" for level in levels]
        ))
    onehot = pd.concat(parts, axis=1).reindex(columns=ONEHOT_COLUMNS,
                                              fill_value=0.0)
    return {"ordinal": ordinal, "onehot": onehot}


def final_prediction_function(values):
    return predict_named(BEST_MODEL_NAME, FINAL_BUNDLE["fitted"],
                         ordinal_to_matrices(values), FINAL_BUNDLE["transform"])


def display_label(code):
    label = " ".join(str(LABEL_MAP.get(code, code)).split())
    if not label.startswith(code):
        label = f"{code}: {label}"
    return label if len(label) <= 44 else label[:43] + "…"


INPUT_LABELS = [display_label(code) for code in ORDINAL_TRAIN.columns]
shap_importance = pd.DataFrame()

if RUN_SHAP:
    section("STAGE 10  SHAP ATTRIBUTION")
    rng = np.random.default_rng(RANDOM_STATE + 94500)
    background_count = min(SHAP_BACKGROUND_ROWS, len(ORDINAL_TRAIN))
    explain_count = min(SHAP_EXPLAIN_ROWS, len(ORDINAL_HOLDOUT))
    background_rows = np.sort(rng.choice(len(ORDINAL_TRAIN),
                                         size=background_count, replace=False))
    explain_rows = np.sort(rng.choice(len(ORDINAL_HOLDOUT),
                                      size=explain_count, replace=False))
    background = ORDINAL_TRAIN.iloc[background_rows].to_numpy()
    explain_matrix = ORDINAL_HOLDOUT.iloc[explain_rows].to_numpy()

    n_inputs = ORDINAL_TRAIN.shape[1]
    max_evals = SHAP_PERMUTATION_ROUNDS * (2 * n_inputs + 1)
    print(f"  explaining {explain_count} holdout rows against "
          f"{background_count} background rows, {max_evals} evaluations per row")
    print(f"  approximately {explain_count * max_evals * background_count:,} "
          f"model rows; reduce SHAP_EXPLAIN_ROWS or SHAP_BACKGROUND_ROWS to "
          f"shorten this stage")
    explainer = shap.Explainer(
        final_prediction_function,
        shap.maskers.Independent(background, max_samples=background_count),
        feature_names=INPUT_LABELS, algorithm="permutation",
    )
    shap_values = explainer(explain_matrix, max_evals=max_evals, batch_size=16)
    values = np.asarray(shap_values.values, float)

    # Additivity audit: base value plus attributions must reproduce the model.
    reconstruction = (np.asarray(shap_values.base_values, float).reshape(-1)
                      + values.sum(axis=1))
    direct = final_prediction_function(explain_matrix)
    save_table(pd.DataFrame({
        "explained_holdout_row": explain_rows,
        "model_prediction": direct,
        "shap_reconstruction": reconstruction,
        "absolute_difference": np.abs(direct - reconstruction),
    }).round(8), "Table_31_shap_additivity_audit")
    print(f"  additivity check: max |model - base - sum(SHAP)| = "
          f"{np.max(np.abs(direct - reconstruction)):.2e}")

    mean_absolute = np.abs(values).mean(axis=0)
    shap_importance = pd.DataFrame({
        "input_code": ORDINAL_TRAIN.columns,
        "input": INPUT_LABELS,
        "input_type": ["categorical" if c in CATEGORICAL_FEATURES_LOADED
                       else "numeric" for c in ORDINAL_TRAIN.columns],
        "mean_absolute_shap_percentage_points": mean_absolute,
    }).sort_values("mean_absolute_shap_percentage_points",
                   ascending=False).reset_index(drop=True)
    shap_importance.insert(0, "rank", np.arange(1, len(shap_importance) + 1))
    save_table(shap_importance.round(6), "Table_32_shap_importance")

    # Stability: independent halves of the explained set should rank inputs
    # consistently if the number of permutation rounds is adequate.
    half = explain_count // 2
    first_half = np.abs(values[:half]).mean(axis=0)
    second_half = np.abs(values[half:]).mean(axis=0)
    stability = float(st.spearmanr(first_half, second_half).statistic)
    agreement = len(
        set(np.argsort(first_half)[-10:]).intersection(np.argsort(second_half)[-10:])
    )
    save_table(pd.DataFrame([{
        "split_half_spearman_correlation": stability,
        "top10_overlap_between_halves": agreement,
        "permutation_rounds": SHAP_PERMUTATION_ROUNDS,
        "evaluations_per_row": max_evals,
    }]).round(4), "Table_33_shap_stability")
    print(f"  split-half rank correlation = {stability:.3f}; "
          f"top-10 overlap = {agreement}/10")

    top = np.argsort(mean_absolute)[-SHAP_MAX_DISPLAY:][::-1]
    fig, ax = plt.subplots(figsize=(FULL_W, 4.6))
    swarm_rng = np.random.default_rng(RANDOM_STATE + 95500)
    has_categorical = False
    for position, index in enumerate(top):
        code = ORDINAL_TRAIN.columns[index]
        jitter = swarm_rng.normal(0, 0.115, size=explain_count)
        if code in CATEGORICAL_FEATURES_LOADED:
            has_categorical = True
            ax.scatter(values[:, index], position + jitter, s=12, alpha=0.68,
                       color="#595959", edgecolors="none")
        else:
            feature = explain_matrix[:, index]
            low, high = np.nanpercentile(feature, [5, 95])
            scaled = (np.full(explain_count, 0.5) if not np.isfinite(high - low)
                      or high <= low
                      else np.clip((feature - low) / (high - low), 0, 1))
            ax.scatter(values[:, index], position + jitter, s=12, alpha=0.72,
                       c=scaled, cmap="coolwarm", vmin=0, vmax=1,
                       edgecolors="none")
    ax.axvline(0, color="#777777", lw=0.8)
    ax.set_yticks(np.arange(len(top)), [INPUT_LABELS[i] for i in top],
                  fontsize=6.4)
    ax.invert_yaxis()
    ax.set_xlabel("SHAP value (expansion percentage points)")
    scale = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0, 1), cmap="coolwarm")
    scale.set_array([])
    colorbar = fig.colorbar(scale, ax=ax, pad=0.018, aspect=35)
    colorbar.set_ticks([0, 1], labels=["Low", "High"])
    colorbar.set_label("Value of a numeric input")
    if has_categorical:
        ax.scatter([], [], s=16, color="#595959", label="unordered categorical input")
        ax.legend(loc="lower right")
    fig.tight_layout(pad=0.8)
    save_fig(fig, "Fig_12_ShapBeeswarm",
             "Permutation SHAP values on the locked holdout, in expansion "
             "percentage points. Numeric inputs are coloured by their value; "
             "unordered categorical inputs use a neutral colour because a "
             "low-to-high scale would imply an ordering they do not have.")

    top_bar = shap_importance.head(SHAP_MAX_DISPLAY).iloc[::-1]
    fig, ax = plt.subplots(figsize=(4.8, 4.0))
    ax.barh(np.arange(len(top_bar)),
            top_bar["mean_absolute_shap_percentage_points"],
            color=[C_ORANGE if kind == "categorical" else C_BLUE
                   for kind in top_bar["input_type"]], height=0.66)
    ax.set_yticks(np.arange(len(top_bar)), top_bar["input"], fontsize=6.3)
    ax.set_xlabel("Mean |SHAP| (expansion percentage points)")
    fig.tight_layout()
    save_fig(fig, "Fig_13_ShapImportance",
             "Global SHAP importance per measured input. Orange marks "
             "categorical inputs.")

    fig, axes = plt.subplots(2, 2, figsize=(FULL_W, 4.8))
    for ax, index, letter in zip(axes.ravel(),
                                 np.argsort(mean_absolute)[-4:][::-1], "abcd"):
        code = ORDINAL_TRAIN.columns[index]
        feature = explain_matrix[:, index]
        if code in CATEGORICAL_FEATURES_LOADED:
            levels = FINAL_BUNDLE["state"]["categories"][code]
            ax.scatter(feature + swarm_rng.normal(0, 0.03, explain_count),
                       values[:, index], s=11, alpha=0.55, color=C_PURPLE,
                       edgecolors="none")
            observed = sorted({int(round(v)) for v in feature if np.isfinite(v)})
            ax.set_xticks(observed,
                          [str(levels[c])[:16] for c in observed], rotation=20)
        else:
            ax.scatter(feature, values[:, index], s=11, alpha=0.6,
                       color=C_BLUE, edgecolors="none")
        ax.axhline(0, color="#777777", lw=0.6)
        ax.set_xlabel(INPUT_LABELS[index])
        ax.set_ylabel("SHAP value (pp)")
        panel_label(ax, letter, dx=-0.18)
    fig.suptitle("Strongest input-response relationships in the fitted model",
                 fontsize=7.6)
    fig.tight_layout()
    save_fig(fig, "Fig_14_ShapDependence",
             "SHAP dependence for the four most influential inputs. The "
             "relationships describe the fitted model and are not causal "
             "material effects.")

    # Age contribution stratified by curing condition.
    if AGE_COLUMN in ORDINAL_TRAIN.columns and "x28" in ORDINAL_TRAIN.columns:
        age_index = list(ORDINAL_TRAIN.columns).index(AGE_COLUMN)
        curing_index = list(ORDINAL_TRAIN.columns).index("x28")
        curing_levels = FINAL_BUNDLE["state"]["categories"]["x28"]
        codes = np.clip(np.rint(explain_matrix[:, curing_index]).astype(int),
                        0, len(curing_levels) - 1)
        labels = np.array([curing_levels[c] for c in codes])
        save_table(pd.DataFrame({
            "explained_holdout_row": explain_rows,
            "testing_age": explain_matrix[:, age_index],
            "curing_condition": labels,
            "testing_age_shap": values[:, age_index],
            "curing_condition_shap": values[:, curing_index],
        }).round(6), "Table_34_age_shap_by_curing")

        observed_levels = [level for level in curing_levels
                           if level in set(labels)]
        palette = [C_BLUE, C_ORANGE, C_GREEN, C_PURPLE, C_RED]
        fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 3.1))
        grouped_values = []
        for position, level in enumerate(observed_levels):
            mask = labels == level
            colour = palette[position % len(palette)]
            axes[0].scatter(explain_matrix[mask, age_index],
                            values[mask, age_index], s=12, alpha=0.45,
                            color=colour, edgecolors="none",
                            label=f"{level} (n={int(mask.sum())})")
            frame = pd.DataFrame({"age": explain_matrix[mask, age_index],
                                  "shap": values[mask, age_index]})
            if len(frame) >= 10 and frame["age"].nunique() >= 3:
                frame["bin"] = pd.qcut(frame["age"],
                                       q=min(6, frame["age"].nunique()),
                                       labels=False, duplicates="drop")
                trend = frame.groupby("bin", observed=True).median()
                axes[0].plot(trend["age"], trend["shap"], color=colour, lw=1.3)
            grouped_values.append(values[mask, curing_index])
        axes[0].axhline(0, color="#777777", lw=0.7)
        axes[0].set_xlabel(display_label(AGE_COLUMN))
        axes[0].set_ylabel("Testing-age SHAP value (pp)")
        axes[0].legend(loc="best", fontsize=5.8)
        panel_label(axes[0], "a", dx=-0.18)

        stable = [(position, group) for position, group
                  in enumerate(grouped_values, 1) if len(group) >= 10]
        if stable:
            axes[1].boxplot([group for _, group in stable],
                            positions=[position for position, _ in stable],
                            showfliers=False,
                            medianprops={"color": C_ORANGE, "linewidth": 1.2})
        for position, group in enumerate(grouped_values, 1):
            axes[1].scatter(position + swarm_rng.normal(0, 0.045, len(group)),
                            group, s=8, alpha=0.3,
                            color=palette[(position - 1) % len(palette)],
                            edgecolors="none")
            if len(group) < 10 and len(group):
                axes[1].annotate(f"n={len(group)}", (position, np.max(group)),
                                 xytext=(0, 5), textcoords="offset points",
                                 ha="center", fontsize=5.8)
        axes[1].axhline(0, color="#777777", lw=0.7)
        axes[1].set_xticks(np.arange(1, len(observed_levels) + 1),
                           observed_levels)
        axes[1].tick_params(axis="x", rotation=20)
        axes[1].set_xlabel("Curing condition")
        axes[1].set_ylabel("Curing SHAP value (pp)")
        panel_label(axes[1], "b", dx=-0.18)
        fig.suptitle("Exploratory interpretation: testing age and curing "
                     "condition", fontsize=7.6)
        fig.tight_layout()
        save_fig(fig, "Fig_15_AgeShapByCuring",
                 "Testing-age SHAP contribution stratified by curing "
                 "condition, and the distribution of curing-condition "
                 "contributions. Groups with fewer than ten explained "
                 "specimens are shown descriptively.")


# =============================================================================
# Reporting artefacts
# =============================================================================
section("REPORTING")

figure_index = pd.DataFrame(FIGURE_INDEX)
figure_index.insert(0, "order", np.arange(1, len(figure_index) + 1))
save_table(figure_index, "FIGURE_INDEX_AND_DRAFT_CAPTIONS")

headline = pd.DataFrame([
    {"Item": "Analysis scope", "Value": SCOPE},
    {"Item": "Measurements", "Value": f"{len(Y)} rows"},
    {"Item": "Distinct mixtures", "Value": f"{GROUP_SIZES.size}"},
    {"Item": "Repeated-measure rows", "Value": f"{REPEATED_MEASURE_SHARE:.0%}"},
    {"Item": "Measured inputs", "Value": f"{len(BASE_FEATURES_LOADED)}"},
    {"Item": "Engineered inputs", "Value": "0 (excluded by design)"},
    {"Item": "Candidates compared", "Value": f"{len(CANDIDATE_MODELS)}"},
    {"Item": "Selected model", "Value": BEST_MODEL_NAME},
    {"Item": "Grouped CV R2",
     "Value": f"{grouped_summary.loc[BEST_MODEL_NAME, 'R2_mean']:.4f}"},
    {"Item": "Grouped CV RMSE (%)",
     "Value": f"{grouped_summary.loc[BEST_MODEL_NAME, 'RMSE_mean']:.4f}"},
    {"Item": "Holdout R2 (mixture-disjoint)", "Value": f"{holdout_metrics['R2']:.4f}"},
    {"Item": "Holdout RMSE (%)", "Value": f"{holdout_metrics['RMSE']:.4f}"},
    {"Item": "Holdout R2 (random-row reference)",
     "Value": f"{reference_metrics['R2']:.4f}"},
    {"Item": "Inputs retained after reduction", "Value": f"{len(FINAL_INPUTS)}"},
])
save_table(headline, "HEADLINE_RESULTS")
print(headline.to_string(index=False))

manifest = {
    "analysis": "ASR mortar-bar expansion: comparison, parsimony, validation, "
                "interpretation",
    "dataset": {
        "path": str(CSV_PATH),
        "analysis_scope": SCOPE,
        "rows": int(len(Y)),
        "measured_inputs": BASE_FEATURES_LOADED,
        "excluded_engineered_factors": list(EXCLUDED_ENGINEERED_FACTORS),
        "excluded_measured_inputs": EXCLUDED_MEASURED_INPUTS,
        "mixture_grouping": GROUP_DESCRIPTION,
        "n_mixture_groups": int(GROUP_SIZES.size),
        "share_of_rows_in_multi_row_groups": REPEATED_MEASURE_SHARE,
    },
    "protocol": {
        "random_state": RANDOM_STATE,
        "holdout_fraction": TEST_SIZE,
        "holdout_design": "mixture-disjoint; a random-row holdout is reported "
                          "only as an optimistic reference",
        "random_row_cv": f"{CV_FOLDS}-fold x {CV_REPEATS} repeats",
        "grouped_cv": f"{GROUPED_CV_FOLDS}-fold x {GROUPED_CV_REPEATS} repeats",
        "primary_scheme": PRIMARY_SCHEME,
        "target_transform": TRANSFORM_METHOD,
        "box_cox_lambda": LAMBDA_BC,
        "interval_method": "Nadeau-Bengio corrected resampled t for "
                           "cross-validation; paired-row percentile bootstrap "
                           "for the fixed holdout",
        "holdout_bootstrap_replicates": HOLDOUT_BOOTSTRAP_REPLICATES,
    },
    "candidates": CANDIDATE_MODELS,
    "selected_model": BEST_MODEL_NAME,
    "selection_rule": "lowest mixture-grouped CV RMSE on development data; "
                      "MAE then R2 tie-breakers; holdout never consulted",
    "results": {
        "grouped_cv": {
            metric: float(grouped_summary.loc[BEST_MODEL_NAME, f"{metric}_mean"])
            for metric in ["R2", "RMSE", "MAE", "MedAPE"]
        },
        "pooled_out_of_fold": {k: float(v) for k, v in out_of_fold_metrics.items()},
        "locked_holdout": {k: float(v) for k, v in holdout_metrics.items()},
        "random_row_holdout_reference": {k: float(v)
                                         for k, v in reference_metrics.items()},
        "development_fit_in_sample": {k: float(v)
                                      for k, v in development_metrics.items()},
        "repeated_measure_optimism": json.loads(optimism.to_json(orient="records")),
        "conformal_coverage": (json.loads(conformal_summary.to_json(orient="records"))
                               if len(conformal_summary) else []),
        "reactivity_classification": (
            json.loads(reactivity_results.to_json(orient="records"))
            if len(reactivity_results) else []
        ),
        "y_randomization_mean_R2": (float(permutation_control["R2"].mean())
                                    if len(permutation_control) else None),
    },
    "feature_reduction": {
        "removal_order": REMOVAL_ORDER,
        "last_safe_removal_count": int(LAST_SAFE_REMOVAL),
        "removed_inputs": FINAL_EXCLUDED,
        "retained_inputs": FINAL_INPUTS,
        "tolerances": {
            "max_absolute_R2_loss": R2_MAX_ABSOLUTE_LOSS,
            "max_relative_RMSE_increase_pct": RMSE_MAX_RELATIVE_INCREASE_PCT,
            "max_relative_MAE_increase_pct": MAE_MAX_RELATIVE_INCREASE_PCT,
        },
    },
    "tabpfn": {
        "used": bool(HAS_TABPFN), "status": TABPFN_STATUS,
        "device": TABPFN_DEVICE, "n_estimators": TABPFN_PARAMS["n_estimators"],
        "inference": "local checkpoint; no data sent to a hosted service",
        "licence": "model weights are non-commercial and require acceptance "
                   "of the Prior Labs licence",
    },
    "limitations": [
        "Mixture grouping is derived from the predictor values; if two "
        "genuinely independent mixtures share every non-age predictor they "
        "are treated as one group, which is conservative.",
        "The locked holdout is internal. It is disjoint in mixture but drawn "
        "from the same compiled sources, so it does not establish external "
        "validity for new aggregates, binders, or laboratories.",
        "Cross-validated intervals use the Nadeau-Bengio correction but "
        "remain approximate because training folds overlap.",
        "Paired model tests are descriptive: the winner is chosen on the same "
        "development results used for the tests.",
        "The cumulative removal path is exploratory model selection and its "
        "boundary depends on the prespecified tolerances.",
        "SHAP values and the age-by-curing pattern describe the fitted "
        "prediction function and must not be read as causal material effects.",
        "Reactivity classification applies each expansion limit directly to "
        "the predicted value and does not recalibrate a decision threshold.",
    ],
    "environment": ENVIRONMENT_VERSIONS,
}
with open(RES_DIR / "run_manifest.json", "w") as handle:
    json.dump(manifest, handle, indent=2)

inventory = []
for artefact in sorted(path for path in ROOT_DIR.rglob("*") if path.is_file()):
    inventory.append({
        "relative_path": artefact.relative_to(ROOT_DIR).as_posix(),
        "size_bytes": artefact.stat().st_size,
        "sha256": hashlib.sha256(artefact.read_bytes()).hexdigest(),
    })
pd.DataFrame(inventory).to_csv(RES_DIR / "artefact_inventory.csv", index=False)

archive = Path("ASR_publication_outputs.zip")
files = sorted(path for path in ROOT_DIR.rglob("*") if path.is_file())
with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
    for path in files:
        bundle.write(path, path.as_posix())
with zipfile.ZipFile(archive, "r") as bundle:
    archived = set(bundle.namelist())
expected = {path.as_posix() for path in files}
if archived != expected:
    raise RuntimeError(f"archive verification failed; missing "
                       f"{sorted(expected - archived)}")
print(f"\nwrote and verified {archive} "
      f"({len(files)} files, {archive.stat().st_size:,} bytes)")
print(f"  figures: {len(list(FIG_DIR.glob('*.png')))}")
print(f"  tables : {len(list(RES_DIR.glob('*.csv')))}")

try:
    from google.colab import files as colab_files

    colab_files.download(str(archive))
except ImportError:
    pass
except Exception as error:
    print(f"automatic download unavailable ({error}); download {archive} "
          f"from the Colab file browser")
