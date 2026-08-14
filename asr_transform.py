#!/usr/bin/env python3
"""
ASR expansion: does transforming the target fix the systematic under-prediction?

The parity plot shows the defect clearly. Below about 0.25 % expansion the
predictions track the 1:1 line; above about 0.4 % nearly every point falls
below it. Expansion is right-skewed, so squared error on the raw percentage
lets a model buy accuracy on the crowded low end by giving up the tail.
Transforming the target is the standard response.

There is a trap in it. Fitting on transformed y and back-transforming the
prediction returns the conditional MEDIAN, not the mean, and for a
right-skewed target the median sits below the mean. A naive Box-Cox round-trip
can therefore deepen the very under-prediction it was meant to fix. Duan's
smearing estimator corrects this non-parametrically: it back-transforms the
prediction once per residual and averages, so the correction carries whatever
skew the residual distribution actually has. Every transform here is scored
both ways so the correction is measured rather than assumed.

Two further details matter for the result to mean anything:

  * Lambda is fitted inside each training fold. Fitting it once on the whole
    dataset would leak the test distribution into the transform.
  * Box-Cox needs strictly positive input, so zero-expansion rows are shifted
    by a small constant fitted on training data. Yeo-Johnson handles zeros
    natively and is the cleaner choice if the two score alike.

Accuracy is always reported on the original percentage scale. Scores on the
transformed scale look better and are not comparable to anything.

The question is not only whether R2 improves. It is whether the bias against
measured expansion flattens, so the run reports mean signed error inside bins
bounded by the ASTM limits, plus the calibration slope, which is 1.0 for an
unbiased model and below 1.0 when high values are shrunk toward the mean.

Usage:
    python asr_transform.py                    # all transforms, CatBoost
    python asr_transform.py --families all     # every family, slower
    python asr_transform.py --quick            # fewer iterations, a smoke test
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
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
TEST_SIZE = 0.20
N_FOLDS = 5

# Duan's smearing averages the back-transform over training residuals. Using
# every residual is quadratic in fold size for no extra accuracy, so a fixed
# quantile grid stands in for the residual distribution.
SMEARING_POINTS = 200

# Bin edges follow the ASTM reactivity limits, so the bias table reads in the
# units the standards make decisions in.
BIAS_BINS = [0.0, 0.04, 0.10, 0.20, 0.40, np.inf]

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
MK_CONTENT, FA_CONTENT, CNS_CONTENT = "x13", "x20", "x25"

# The configuration the pipeline ships. An 80-trial grouped search failed to
# beat it, so it is held fixed here and only the target transform varies.
CAT_PARAMS = dict(
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
    return X.reset_index(drop=True), y.reset_index(drop=True).to_numpy(), \
        pd.factorize(keys)[0]


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


def build_matrix(frame, state):
    out = frame[NUMERIC_FEATURES].fillna(state["fill"]).astype(float).copy()
    for column, levels in state["categories"].items():
        mapping = {value: index for index, value in enumerate(levels)}
        out[column] = (frame[column].astype(str).map(mapping)
                       .fillna(len(levels)).astype(float))
    return out.to_numpy()


# =============================================================================
# Target transforms
#
# Each is fitted on training targets only and exposes forward/inverse. The
# fitted parameter is reported so the paper can quote the lambda it used.
# =============================================================================
class Transform:
    """Fitted on training targets only; inverts safely outside its support.

    Box-Cox and Yeo-Johnson with a negative lambda are bounded above, and
    smearing deliberately evaluates the inverse at prediction-plus-residual,
    which can land past that bound and return NaN or a nonsensical magnitude.
    Predictions are therefore clipped into the range the training target
    actually spanned, widened by half its span so the tail is not truncated at
    the largest value ever observed.
    """

    def __init__(self, name):
        self.name = name
        self.parameter = np.nan

    def fit(self, y_train):
        self._fit(np.asarray(y_train, dtype=float))
        self._record(np.asarray(y_train, dtype=float))
        return self

    def _fit(self, y_train):
        return None

    def _record(self, y_train):
        self._y_hi = float(np.max(y_train))
        span_y = self._y_hi - float(np.min(y_train)) or 1.0
        self._ceiling = self._y_hi + 0.5 * span_y
        z = np.asarray(self.forward(y_train), dtype=float)
        low, high = float(np.min(z)), float(np.max(z))
        span = (high - low) or 1.0
        self._z_lo, self._z_hi = low - 0.5 * span, high + 0.5 * span

    def safe_inverse(self, z):
        clipped = np.clip(np.asarray(z, dtype=float), self._z_lo, self._z_hi)
        out = np.asarray(self.inverse(clipped), dtype=float)
        out = np.where(np.isfinite(out), out, self._y_hi)
        return np.clip(out, 0.0, self._ceiling)

    def forward(self, y):
        raise NotImplementedError

    def inverse(self, z):
        raise NotImplementedError


class Identity(Transform):
    def forward(self, y):
        return np.asarray(y, dtype=float)

    def inverse(self, z):
        return np.asarray(z, dtype=float)


class Log1p(Transform):
    def forward(self, y):
        return np.log1p(np.clip(y, 0, None))

    def inverse(self, z):
        return np.expm1(np.clip(z, -50, 50))


class Sqrt(Transform):
    def forward(self, y):
        return np.sqrt(np.clip(y, 0, None))

    def inverse(self, z):
        return np.clip(z, 0, None) ** 2


class BoxCox(Transform):
    """Box-Cox with a shift, since expansion has exact zeros.

    The shift is a fraction of the smallest positive training value rather than
    a round constant, so it stays small relative to the data rather than
    relative to nothing in particular.
    """

    def _fit(self, y_train):
        positive = y_train[y_train > 0]
        self.shift = float(positive.min() / 2) if len(positive) else 1e-4
        _, self.parameter = stats.boxcox(np.clip(y_train, 0, None) + self.shift)

    def forward(self, y):
        shifted = np.clip(y, 0, None) + self.shift
        if abs(self.parameter) < 1e-8:
            return np.log(shifted)
        return (np.power(shifted, self.parameter) - 1.0) / self.parameter

    def inverse(self, z):
        if abs(self.parameter) < 1e-8:
            return np.exp(np.clip(z, -50, 50)) - self.shift
        base = self.parameter * np.asarray(z, dtype=float) + 1.0
        # Below the transform's support the inverse is undefined; the target is
        # bounded at zero, so clipping there is the faithful choice.
        base = np.clip(base, 1e-12, None)
        return np.power(base, 1.0 / self.parameter) - self.shift


class YeoJohnson(Transform):
    """Handles zeros natively, so no shift is needed."""

    def _fit(self, y_train):
        from sklearn.preprocessing import PowerTransformer

        self._transformer = PowerTransformer(method="yeo-johnson",
                                             standardize=False)
        self._transformer.fit(y_train.reshape(-1, 1))
        self.parameter = float(self._transformer.lambdas_[0])

    def forward(self, y):
        return self._transformer.transform(
            np.asarray(y, dtype=float).reshape(-1, 1)).ravel()

    def inverse(self, z):
        return self._transformer.inverse_transform(
            np.asarray(z, dtype=float).reshape(-1, 1)).ravel()


class QuantileNormal(Transform):
    """Maps the training target onto a normal distribution by rank.

    Included as the upper bound of what a marginal transform can do: it makes
    the target exactly normal. If it does not fix the bias, no transform will.
    """

    def _fit(self, y_train):
        from sklearn.preprocessing import QuantileTransformer

        self._transformer = QuantileTransformer(
            output_distribution="normal",
            n_quantiles=int(min(1000, max(10, len(y_train)))),
            subsample=1_000_000, random_state=0)
        self._transformer.fit(y_train.reshape(-1, 1))

    def forward(self, y):
        return self._transformer.transform(
            np.asarray(y, dtype=float).reshape(-1, 1)).ravel()

    def inverse(self, z):
        return self._transformer.inverse_transform(
            np.asarray(z, dtype=float).reshape(-1, 1)).ravel()


TRANSFORMS = {
    "none": Identity, "log1p": Log1p, "sqrt": Sqrt,
    "boxcox": BoxCox, "yeojohnson": YeoJohnson, "quantile": QuantileNormal,
}


# =============================================================================
# Duan's smearing estimator
# =============================================================================
def smearing_inverse(transform, z_pred, residuals):
    """Back-transform averaged over the residual distribution.

    A plain inverse returns the conditional median. Averaging the inverse over
    the training residuals returns an estimate of the conditional mean, which
    is what a squared-error metric is scored against. The residual distribution
    is summarised by a quantile grid so cost stays linear in the test size.
    """
    if residuals is None or len(residuals) == 0:
        return transform.safe_inverse(z_pred)
    grid = np.quantile(residuals,
                       np.linspace(0.005, 0.995, SMEARING_POINTS))
    stacked = np.asarray(z_pred, dtype=float)[:, None] + grid[None, :]
    back = transform.safe_inverse(stacked.ravel()).reshape(stacked.shape)
    return back.mean(axis=1)


# =============================================================================
# Models
# =============================================================================
def available_families(requested):
    families = []
    for label in ("CatBoost", "HistGBM", "ExtraTrees", "RandomForest"):
        if label == "CatBoost":
            try:
                __import__("catboost")
            except Exception as error:  # noqa: BLE001
                print(f"  CatBoost unavailable ({type(error).__name__}); "
                      f"skipped")
                continue
        families.append(label)
    if requested != "all":
        families = [f for f in families if f.lower() == requested.lower()]
        if not families:
            sys.exit(f"family not available: {requested}")
    return families


def build_estimator(family, seed, quick):
    if family == "CatBoost":
        from catboost import CatBoostRegressor

        params = dict(CAT_PARAMS)
        if quick:
            params["iterations"] = 300
        return CatBoostRegressor(**params, random_seed=seed, verbose=0,
                                 allow_writing_files=False, thread_count=-1)
    if family == "HistGBM":
        return HistGradientBoostingRegressor(
            max_iter=150 if quick else 600, learning_rate=0.05,
            max_leaf_nodes=31, min_samples_leaf=10, l2_regularization=0.5,
            random_state=seed)
    if family == "ExtraTrees":
        return ExtraTreesRegressor(
            n_estimators=200 if quick else 650, min_samples_leaf=1,
            max_features=0.97, random_state=seed, n_jobs=-1)
    return RandomForestRegressor(
        n_estimators=200 if quick else 500, min_samples_leaf=2,
        max_features=0.6, random_state=seed, n_jobs=-1)


# =============================================================================
# Evaluation
# =============================================================================
def run_folds(X, y, folds, family, transform_name, smear, seed, quick):
    """Grouped out-of-fold predictions on the original percentage scale.

    Preprocessing, the transform and its lambda are all fitted inside the
    training fold. Smearing residuals come from an inner grouped split of the
    training fold, because a boosted ensemble fits its own training rows almost
    exactly and in-sample residuals would report a spread near zero.
    """
    prediction = np.full(len(y), np.nan)
    parameters = []
    for train_index, test_index in folds:
        train_df = X.iloc[train_index]
        state = fit_preprocessing(train_df)
        matrix_tr = build_matrix(train_df, state)
        matrix_te = build_matrix(X.iloc[test_index], state)

        transform = TRANSFORMS[transform_name](transform_name)
        transform.fit(y[train_index])
        parameters.append(transform.parameter)
        z_train = transform.forward(y[train_index])

        residuals = None
        if smear:
            inner = max(2, len(train_index) // 4)
            rng = np.random.default_rng(seed + 17)
            order = rng.permutation(len(train_index))
            held, kept = order[:inner], order[inner:]
            inner_model = build_estimator(family, seed, quick)
            inner_model.fit(matrix_tr[kept], z_train[kept])
            residuals = z_train[held] - inner_model.predict(matrix_tr[held])

        model = build_estimator(family, seed, quick)
        model.fit(matrix_tr, z_train)
        z_pred = model.predict(matrix_te)
        prediction[test_index] = (smearing_inverse(transform, z_pred, residuals)
                                  if smear else transform.safe_inverse(z_pred))
    return prediction, float(np.nanmean(parameters))


def score(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    slope, intercept = np.polyfit(y_true, y_pred, 1)
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "bias": float(np.mean(y_pred - y_true)),
        "slope": float(slope),
        "tail_bias": float(np.mean((y_pred - y_true)[y_true >= 0.40]))
                     if (y_true >= 0.40).any() else np.nan,
    }


def bias_table(y_true, predictions: dict):
    """Mean signed error inside bins bounded by the ASTM reactivity limits."""
    y_true = np.asarray(y_true, dtype=float)
    bins = pd.cut(y_true, BIAS_BINS, right=False)
    rows = []
    for interval, index in pd.Series(range(len(y_true))).groupby(
            bins, observed=True):
        position = index.to_numpy()
        row = {"measured expansion": str(interval), "n": len(position)}
        for label, prediction in predictions.items():
            row[label] = float(np.mean(
                np.asarray(prediction)[position] - y_true[position]))
        rows.append(row)
    return pd.DataFrame(rows)


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=DEFAULT_CSV)
    parser.add_argument("--scope", default="standard",
                        choices=["all", "standard", "c1260", "c1293"])
    parser.add_argument("--families", default="CatBoost",
                        help="'all' or one of CatBoost, HistGBM, ExtraTrees, "
                             "RandomForest")
    parser.add_argument("--folds", type=int, default=N_FOLDS)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--quick", action="store_true",
                        help="fewer iterations; for checking it runs")
    parser.add_argument("--out", default="asr_transform_results.csv")
    args = parser.parse_args()

    started = time.time()
    X, y, groups = load(resolve_csv(args.csv))
    mask = scope_mask(X, args.scope)
    X, y, groups = X.loc[mask].reset_index(drop=True), y[mask], groups[mask]

    dev_index, hold_index = grouped_holdout(groups, TEST_SIZE, args.seed)
    X_dev = X.iloc[dev_index].reset_index(drop=True)
    y_dev, groups_dev = y[dev_index], groups[dev_index]
    X_hold, y_hold = X.iloc[hold_index].reset_index(drop=True), y[hold_index]
    folds = grouped_folds(groups_dev, args.folds, args.seed)

    print(f"scope {args.scope}: {len(X)} rows, "
          f"{pd.Series(groups).nunique()} mixtures")
    print(f"  development {len(dev_index)} rows, "
          f"{pd.Series(groups_dev).nunique()} mixtures, {len(folds)} folds")
    print(f"  holdout     {len(hold_index)} rows (locked, scored at the end)")
    print(f"  target: {(y_dev == 0).sum()} exact zeros, skew "
          f"{stats.skew(y_dev):.2f}\n")

    families = available_families(args.families)
    print(f"families: {', '.join(families)}")
    print("smearing: Duan's estimator, corrects the median-vs-mean bias "
          "a back-transform introduces\n")

    records, oof = [], {}
    print(f"{'family':<14}{'transform':<12}{'smear':<7}{'lambda':>9}"
          f"{'R2':>9}{'RMSE':>9}{'bias':>9}{'slope':>8}{'tail':>9}")
    print("-" * 86)
    for family in families:
        for transform_name in TRANSFORMS:
            for smear in (False, True):
                if transform_name == "none" and smear:
                    continue          # identity: smearing is a no-op
                try:
                    prediction, parameter = run_folds(
                        X_dev, y_dev, folds, family, transform_name, smear,
                        args.seed, args.quick)
                except Exception as error:  # noqa: BLE001
                    print(f"{family:<14}{transform_name:<12}"
                          f"failed: {type(error).__name__}: {error}")
                    continue
                if not np.isfinite(prediction).all():
                    print(f"{family:<14}{transform_name:<12}"
                          f"{'yes' if smear else 'no':<7}"
                          f"non-finite predictions; skipped")
                    continue
                result = score(y_dev, prediction)
                label = (f"{family}/{transform_name}"
                         f"{'+smear' if smear else ''}")
                oof[label] = prediction
                records.append({"family": family, "transform": transform_name,
                                "smearing": smear, "lambda": parameter,
                                **result})
                print(f"{family:<14}{transform_name:<12}"
                      f"{'yes' if smear else 'no':<7}"
                      f"{parameter:>9.4f}{result['R2']:>9.4f}"
                      f"{result['RMSE']:>9.4f}{result['bias']:>9.4f}"
                      f"{result['slope']:>8.3f}{result['tail_bias']:>9.4f}")

    if not records:
        sys.exit("no transform produced a usable result")

    table = pd.DataFrame(records).sort_values("R2", ascending=False)
    table.to_csv(args.out, index=False)

    print(f"\n{'=' * 86}\nRANKED BY GROUPED-CV R2 (development only)\n{'=' * 86}")
    print(table.to_string(index=False,
                          float_format=lambda v: f"{v:9.4f}"))

    baseline = table[(table["transform"] == "none")].iloc[0]
    best = table.iloc[0]
    baseline_label = f"{baseline['family']}/none"
    best_label = (f"{best['family']}/{best['transform']}"
                  f"{'+smear' if best['smearing'] else ''}")

    print(f"\n{'=' * 86}\nDID THE SYSTEMATIC ERROR RESOLVE?\n{'=' * 86}")
    print("Mean signed error (predicted - measured, %) inside ASTM limit "
          "bands.\nNegative means under-prediction. The defect is the growing "
          "negative bias\nin the upper bands; a transform fixes it only if "
          "those numbers move toward zero.\n")
    comparison = {baseline_label: oof[baseline_label]}
    if best_label != baseline_label:
        comparison[best_label] = oof[best_label]
    print(bias_table(y_dev, comparison).to_string(
        index=False, float_format=lambda v: f"{v:9.4f}"))
    print(f"\ncalibration slope  {baseline_label}: {baseline['slope']:.3f}"
          + (f"   {best_label}: {best['slope']:.3f}"
             if best_label != baseline_label else ""))
    print("1.000 is unbiased; below 1.000 means high values are shrunk "
          "toward the mean.")

    print(f"\n{'=' * 86}\nLOCKED HOLDOUT -- scored twice, both declared\n"
          f"{'=' * 86}")
    rows = []
    for label, row in ((baseline_label, baseline), (best_label, best)):
        if label == baseline_label and rows:
            continue
        transform = TRANSFORMS[row["transform"]](row["transform"])
        transform.fit(y_dev)
        state = fit_preprocessing(X_dev)
        matrix_dev = build_matrix(X_dev, state)
        z_dev = transform.forward(y_dev)

        residuals = None
        if row["smearing"]:
            rng = np.random.default_rng(args.seed + 17)
            order = rng.permutation(len(y_dev))
            inner = max(2, len(y_dev) // 4)
            held, kept = order[:inner], order[inner:]
            inner_model = build_estimator(row["family"], args.seed, args.quick)
            inner_model.fit(matrix_dev[kept], z_dev[kept])
            residuals = z_dev[held] - inner_model.predict(matrix_dev[held])

        model = build_estimator(row["family"], args.seed, args.quick)
        model.fit(matrix_dev, z_dev)
        z_pred = model.predict(build_matrix(X_hold, state))
        prediction = (smearing_inverse(transform, z_pred, residuals)
                      if row["smearing"] else transform.safe_inverse(z_pred))
        rows.append({"configuration": label, **score(y_hold, prediction)})
        if best_label == baseline_label:
            break
    print(pd.DataFrame(rows).to_string(
        index=False, float_format=lambda v: f"{v:9.4f}"))

    print(f"\n{'=' * 86}\nREADING THIS\n{'=' * 86}")
    print("A transform is worth adopting only if the holdout R2 improves AND "
          "the upper\nbias bands move toward zero. A higher CV R2 alone is not "
          "enough: an 80-trial\nsearch already showed grouped CV on 118 "
          "mixtures cannot rank models reliably,\nso treat CV gaps below about "
          "0.1 R2 as noise.")
    print("\nIf every transform leaves the upper bands strongly negative, the "
          "under-prediction\nis not a target-scale problem. It is the model "
          "declining to commit to extreme\nvalues it cannot predict from "
          "composition -- which is what 71 % between-mixture\nvariance implies, "
          "and no transform reaches that.")
    print(f"\nelapsed {(time.time() - started) / 60:.1f} min; "
          f"wrote {args.out}")


if __name__ == "__main__":
    main()
