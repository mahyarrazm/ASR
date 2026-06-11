# ASR mortar expansion — stacked-ensemble publication pipeline

Reproducible machine-learning pipeline for predicting alkali–silica reaction
(ASR) mortar-bar expansion from mixture, binder-composition, and exposure
descriptors, prepared for journal submission.

The model is a *fold-pure* stacked ensemble of **LightGBM**, **CatBoost**, and
**TabPFN**, combined by a **BayesianRidge** meta-learner trained only on inner
out-of-fold base-model predictions (no information leakage between the base
and meta levels).

## Modeling protocol

| Component | Setting |
|---|---|
| Features | 17 base + 5 engineered = 22 (ablation-selected) |
| Imputation | conditional median by SCM presence + FA-class mode, refit per fold |
| Target transform | Box–Cox (λ = 0.85), shift refit per fold |
| Cross-validation | 10-fold, stratified by target-quantile bins, fold-pure |
| Holdout | 80/20 split, `random_state = 256` |
| Stability | seeds 256, 7, 42, 101, 2024 |

## Contents

| File | Description |
|---|---|
| `ASR_pipeline.ipynb` | Main notebook (Google Colab ready) — runs the full analysis and renders every figure inline |
| `asr_pipeline.py` | Same pipeline as a plain Python script (jupytext-paired with the notebook) |
| `requirements.txt` | Python dependencies |

## Quick start (Google Colab)

1. Open `ASR_pipeline.ipynb` in Colab and select a **GPU runtime**
   (recommended for TabPFN).
2. Upload the dataset CSV to `/content/` (default name
   `ASR_FinalE-ComCo.csv`; adjust `CSV_PATH` in the *Configuration* cell if
   needed).
3. Add your TabPFN license token as a Colab secret named `TABPFN_TOKEN`
   (key icon in the sidebar, notebook access ON; free token from
   https://ux.priorlabs.ai/account). Without a token or without `tabpfn`
   installed, the pipeline automatically falls back to the two-learner
   (LightGBM + CatBoost) stack.
4. *Runtime → Run all*. All figures appear inline; at the end a ZIP archive
   (`ASR_publication_outputs.zip`) with all figures and tables is offered for
   download.

## Data format

A single CSV with one row per specimen: the 27 base feature columns
(`x1` … `x29`, including `X21`) plus the numeric target column `y`
(expansion, %). Rows with a non-numeric target (e.g. a units/description
header row) are detected and dropped automatically; missing feature values
are preserved and imputed fold-pure inside each split.

## Analysis suite and outputs

| Analysis | Figures (`figures/`, 600-dpi PNG + vector PDF) | Tables (`results/`) |
|---|---|---|
| Dataset overview (target distribution, missingness) | Fig. S1 | Table S0 |
| 10-fold fold-pure cross-validation | Fig. 10a | Table S1 |
| Baseline / single-model comparison (Ridge, k-NN, Random Forest, LightGBM, CatBoost, TabPFN, stack) under identical folds and preprocessing | Fig. 4 | Table S2 |
| 80/20 holdout evaluation | Fig. 1 (parity), Fig. 6 (residual diagnostics incl. Q–Q) | Table S4 |
| Split-conformal 90 % prediction intervals, calibrated on leak-free out-of-fold CV residuals | Fig. 9 | Table S9 |
| SHAP feature attribution (TreeSHAP on the GBDT components, meta-weighted) | Fig. 2 (beeswarm), Fig. 3 (bar) | Table S5 |
| Feature–target Pearson correlation structure | Fig. 7 | Table S6 |
| Learning curve (data efficiency) | Fig. 5 | Table S7 |
| y-randomization (chance-performance / leakage check) | Fig. 8 | Table S8 |
| Multi-seed holdout stability (5 seeds) | Fig. 10b | Table S3 |

## Reproducibility

- All stochastic steps are seeded (`RANDOM_STATE = 256`); package versions
  are printed at the start of every run.
- All models in the comparison share one fixed set of stratified CV splits
  and the same fold-pure preprocessing and target transform, so differences
  reflect the learner only.
- The conformal calibration set (out-of-fold CV residuals) is disjoint from
  the holdout test set on which coverage is verified.

## Local execution

```bash
pip install -r requirements.txt
ASR_CSV_PATH=/path/to/data.csv python asr_pipeline.py
```

Set `ASR_FAST_MODE=1` for a quick smoke test (reduced folds/repeats and
truncated boosting), and `ASR_SKIP_INSTALL=1` to skip the dependency
auto-install cell.

## License

Code released for academic use; please cite the accompanying article when
using this pipeline.
