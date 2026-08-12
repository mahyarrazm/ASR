# Machine-learning prediction of ASR mortar-bar expansion

Reproducible pipeline for predicting alkali–silica reaction (ASR) expansion of
mortar bars from measured mixture and exposure descriptors. The analysis
compares eight model families under a common protocol, selects one on
development data only, reduces the input set against prespecified tolerances,
and validates the result on a locked holdout that shares no mixture with the
training data.

## Contents

Three scripts, each with a distinct job. The publication pipeline is the
deliverable; the other two are the tools used to decide what goes into it.

| File | Purpose |
|---|---|
| `asr_publication_pipeline.py` | **Complete publication analysis.** Fifteen figures, thirty-seven tables, run manifest. Runs top to bottom in Colab or locally. |
| `ASR_pipeline.ipynb` | The same code as a Colab notebook, one cell per stage. |
| `make_notebook.py` | Rebuilds the notebook from the script, which is the single source of truth. |
| `asr_model_search.py` | Model and hyperparameter search over sixteen families, four target transforms, per-mixture weighting and a monotone age constraint. No figures, so the time goes into the search. |
| `asr_structure.py` | Compares four ways of framing the problem — pooled, per test standard, kinetic curve, and hybrid — and reports a variance decomposition with oracle ceilings. Answers whether accuracy is limited by the model or by the formulation. |
| `requirements.txt` | Minimum dependency versions. |
| `requirements-lock.txt` | Exact versions of a verified reference run. |

Run `python make_notebook.py` after editing the pipeline to regenerate the
notebook.

### Which to run

- Reproducing the paper: `asr_publication_pipeline.py`.
- Asking whether a better model exists: `asr_model_search.py`.
- Asking whether a better *formulation* exists: `asr_structure.py`. Run this
  first if the search has plateaued, since a plateau shared across unrelated
  model families points at the formulation rather than the configuration.

Both search scripts reserve the same locked holdout as the pipeline and never
read it.

## Quick start (Google Colab)

1. Open `ASR_pipeline.ipynb` and select a **GPU runtime** (Runtime → Change
   runtime type → GPU). TabPFN is impractically slow on CPU. Alternatively,
   clone this repository in a Colab cell and run the script directly with
   `%run asr_publication_pipeline.py`, which also renders every figure inline.
2. TabPFN model weights are gated. Accept the licence at
   <https://huggingface.co/Prior-Labs>, then add your token as a Colab secret
   named `TABPFN_TOKEN` (key icon in the sidebar, notebook access on). To run
   everything else without TabPFN, set `REQUIRE_TABPFN = False`.
3. Upload the dataset CSV to `/content/`, or mount Google Drive.
4. Set `CSV_PATH` in the **Configuration** cell.
5. Runtime → Run all. Figures appear inline; a verified ZIP of all figures and
   tables is offered for download at the end.

Local use is identical:

```bash
pip install -r requirements.txt
ASR_CSV_PATH=/path/to/data.csv python asr_publication_pipeline.py
```

Set `ASR_FAST_MODE=1` for a reduced-setting smoke test, and
`ASR_SKIP_INSTALL=1` to skip the dependency check.

## Data format

One CSV row per measurement. Required columns are `x1`–`x29` (note the
capitalised `X21`), `Standard`, and the target `y` in expansion percent.
`x15`, `x28`, and `Standard` are treated as categorical; the rest are numeric.
A description row beneath the header is detected automatically: its text is
harvested as figure labels and the row is then dropped. Missing composition
values are imputed inside each fold using medians conditioned on whether the
corresponding supplementary cementitious material is present.

## Analysis protocol

**Measured inputs only.** The five deterministic engineered terms used in
earlier versions (`total_SCM`, `age_x_temp`, `silica_x_alkali`, `log_age`,
`SCM_alkali`) are excluded before any model is fitted, so no result depends on
a transformation of the inputs. `x8` is dropped because relative humidity is
constant across the analysed rows.

**Mixture grouping.** Mortar-bar testing measures the same specimen repeatedly
over time, so rows are not independent. Each row is assigned to a mixture group
by hashing every predictor except testing age. All cross-validation folds and
the locked holdout are mixture-disjoint: no mixture contributes rows to both a
training set and the set used to score it. The conventional random-row protocol
is also run, and the difference between the two is reported as the optimism
that random splitting introduces.

**Everything below the holdout line.** Model selection, importance ranking, and
the input-removal boundary use the development partition only. The holdout is
scored once per reported model.

| Stage | Output |
|---|---|
| Data audit: duplication, repeated-measures structure, distributions | Fig. 1, Tables 1–4 |
| Model comparison under both resampling schemes | Fig. 2, Fig. 3, Tables 6–11 |
| Target-transform sensitivity | Table 12 |
| Grouped-permutation importance over folds | Fig. 4, Tables 13–14 |
| Cumulative least-important-first input removal | Fig. 5, Tables 15–18 |
| Locked mixture-disjoint holdout, paired bootstrap intervals | Fig. 6, Fig. 7, Tables 19–24 |
| Split-conformal prediction intervals and coverage | Fig. 8, Table 25 |
| Reactivity classification at standard expansion limits | Fig. 9, Table 26 |
| y-randomization and learning curve controls | Fig. 10, Tables 27–28 |
| Calibration and error stratification | Fig. 11, Tables 29–30 |
| SHAP attribution with additivity and stability audits | Figs. 12–15, Tables 31–34 |

## Models compared

Ridge as a linear benchmark, a multilayer perceptron, TabNet, LightGBM,
XGBoost, CatBoost, a fixed 50/50 LightGBM–CatBoost blend, TabPFN, and a
fold-pure stacked ensemble with a BayesianRidge meta-learner. All
hyperparameters are fixed in advance; no tuning occurs inside this workflow, so
the comparison is not confounded by unequal search effort.

`asr_model_search.py` covers a wider set for exploration — adding
HistGradientBoosting, ExtraTrees, Random Forest, SVR, kernel ridge, k-nearest
neighbours, elastic net, PLS and Huber regression — and searches
hyperparameters within each. Anything promising found there should be promoted
into the pipeline's fixed specifications rather than tuned inside it, so the
reported comparison stays free of unequal search effort.

## Statistical reporting

- Cross-validated intervals use the Nadeau–Bengio corrected resampled *t*
  interval, which accounts for the training-set overlap that makes a naive
  standard error too narrow.
- Holdout intervals are paired-row percentile bootstrap intervals, conditional
  on the fitted model.
- Model comparisons are paired Wilcoxon tests on identical folds with Holm
  correction across all pairs. They are descriptive: folds overlap and the
  winner is chosen on the same development results.
- SHAP values carry an additivity audit and a split-half stability check.

## Outputs

`ASR_publication_outputs/figures/` holds every figure as 600-dpi PNG, vector
PDF, and editable SVG. `ASR_publication_outputs/results/` holds every table as
CSV, plus `run_manifest.json` (full protocol and results record),
`environment_lock.txt`, `artefact_inventory.csv` with SHA-256 checksums,
`HEADLINE_RESULTS.csv`, and `FIGURE_INDEX_AND_DRAFT_CAPTIONS.csv`.

## Reproducibility

All stochastic steps derive from `RANDOM_STATE`. Preprocessing, the target
transform, and the stacking meta-learner are refitted inside every fold. The
run manifest records the resolved version of every dependency, the selected
model, the exact removal order, and the stated limitations.

## Licence and citation

Code is released for academic use. TabPFN model weights carry a separate
non-commercial licence from Prior Labs and must be obtained through their
terms. Please cite the accompanying article when using this pipeline.
