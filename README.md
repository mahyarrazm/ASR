# ASR mortar expansion — stacked ensemble pipeline (v11)

Reproducible machine-learning pipeline for predicting alkali–silica reaction
(ASR) mortar-bar expansion from mixture/exposure descriptors, prepared for
journal submission. Successor to the *v10F + TabPFN* script: same modeling
core, extended with the comparison and robustness analyses expected for a
high-impact submission. The model is a *fold-pure*
stacked ensemble of **LightGBM**, **CatBoost**, and **TabPFN**, combined by a
**BayesianRidge** meta-learner that is trained only on out-of-fold base-model
predictions (no information leakage between base and meta levels).

## Contents

| File | Description |
|---|---|
| `ASR_pipeline.ipynb` | Main notebook (Google Colab ready) — runs the full analysis and renders every figure inline |
| `asr_pipeline.py` | Same pipeline as a plain Python script (jupytext-paired with the notebook) |
| `requirements.txt` | Python dependencies |

## Quick start (Google Colab)

1. Open `ASR_pipeline.ipynb` in Colab and select a **GPU runtime**
   (recommended for TabPFN; without a GPU or without `tabpfn` installed, the
   pipeline automatically falls back to the LightGBM + CatBoost ensemble).
2. Upload the dataset CSV to `/content/` (or mount Google Drive).
3. Adjust the **Configuration** cell if needed:
   - `CSV_PATH` — dataset location (default `/content/ASR_FinalE-ComCo.csv`)
   - `TARGET_COLUMN` — target column name (`None` = last column)
   - `EXCLUDED_FEATURES` — feature columns to exclude from modeling
4. *Runtime → Run all*. All figures appear inline; at the end a ZIP archive
   (`ASR_publication_outputs.zip`) with all figures and tables is offered for
   download.

## Data format

A single CSV with one row per sample: numeric feature columns plus a numeric
target column. Rows with a non-numeric target (e.g. a units/description
header row) are detected and dropped automatically; missing feature values
are median-imputed.

## Analysis suite and outputs

| Analysis | Figures (`figures/`, 600-dpi PNG + vector PDF) | Tables (`results/`) |
|---|---|---|
| 10-fold fold-pure cross-validation | Fig. 10a | Table S1 |
| Baseline / single-model comparison (Ridge, k-NN, Random Forest, LightGBM, CatBoost, TabPFN, stack) | Fig. 4 | Table S2 |
| 80/20 holdout evaluation | Fig. 1 (parity), Fig. 6 (residual diagnostics) | Table S4 |
| Split-conformal 90 % prediction intervals with empirical coverage | Fig. 9 | — |
| SHAP feature attribution (TreeSHAP on GBDT components, meta-weighted) | Fig. 2 (beeswarm), Fig. 3 (bar) | Table S5 |
| Feature–target Pearson correlation structure | Fig. 7 | Table S6 |
| Learning curve (data efficiency) | Fig. 5 | Table S7 |
| y-randomization (chance-performance / leakage check) | Fig. 8 | Table S8 |
| Multi-seed holdout stability (5 seeds) | Fig. 10b | Table S3 |

## Reproducibility

- All stochastic steps are seeded (`SEED = 256`; stability analysis over
  seeds 256, 7, 42, 101, 2024).
- Package versions are printed at the start of every run.
- The cross-validation is *fold-pure*: within each outer fold, the
  meta-learner is fitted on inner out-of-fold predictions only, and the
  conformal calibration set is disjoint from both the model-fitting and test
  sets.

## Local execution

```bash
pip install -r requirements.txt
ASR_CSV_PATH=/path/to/data.csv python asr_pipeline.py
```

Set `ASR_FAST_MODE=1` for a quick smoke test with reduced folds/repeats.

## License

Code released for academic use; please cite the accompanying article when
using this pipeline.
