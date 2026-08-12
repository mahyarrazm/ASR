# Machine-learning prediction of ASR expansion

Reproducible pipeline for predicting alkali–silica reaction (ASR) expansion
from measured mixture and exposure descriptors, with validation that accounts
for the repeated-measures structure of mortar-bar testing.

## The result that shapes this repository

Mortar-bar testing measures the same specimen repeatedly over time. In this
dataset **99.1% of rows share a mixture with another row** — 1,999 rows come
from only 210 distinct mixtures. Splitting those rows at random puts different
measurement ages of the *same specimen* into both training and test sets, so
the model interpolates a curve it has already seen rather than generalising to
a new mixture.

| Validation protocol | R² |
|---|---|
| Random-row cross-validation (conventional) | 0.987 |
| **Mixture-disjoint cross-validation** | **0.62** |
| Mixture-disjoint locked holdout | 0.82 |

Every score in this repository is mixture-disjoint by default. The random-row
protocol is still run and reported so the optimism it introduces is measured
rather than hidden.

A second effect matters as much. The dataset pools three different
experiments — the accelerated mortar-bar test (ASTM C1260/C1567, 14 days at
80 °C), the concrete prism test (ASTM C1293, one year at 38 °C), and a block
of non-standard tests. The same model scores **0.77 on the mortar-bar rows**
and **0.17 on the prism rows**. Pooling protocols is what produces the low
headline number, so the analysis scope is configurable.

## Contents

| File | Purpose |
|---|---|
| `asr_publication_pipeline.py` | **The deliverable.** Fifteen figures, thirty-seven tables, run manifest, checksummed archive. |
| `ASR_pipeline.ipynb` | The same code as a Colab notebook, one cell per stage. |
| `asr_local.py` | Scope comparison. Which test protocol should the paper report? Minutes on a laptop, no GPU. |
| `asr_model_search.py` | Model and hyperparameter search across sixteen families and four target transforms. |
| `asr_structure.py` | Compares problem formulations — pooled, per-standard, kinetic-curve, hybrid — with a variance decomposition and oracle ceilings. |
| `make_notebook.py` | Rebuilds the notebook from the pipeline, which is the single source of truth. |
| `requirements.txt`, `requirements-lock.txt` | Minimum bounds, and exact versions of a verified run. |

Run `python make_notebook.py` after editing the pipeline to regenerate the
notebook.

## Quick start — Google Colab

Select a **GPU runtime**. Add your TabPFN token as a Colab secret named
`TABPFN_TOKEN` (key icon, notebook access on), or set `REQUIRE_TABPFN = False`
to run everything else without it.

```python
!git clone -b claude/confident-wright-ju9c18 https://github.com/mahyarrazm/ASR /content/ASR
%cd /content/ASR

import os
os.environ["ASR_CSV_PATH"] = "/content/ASR_FinalE-ComCo.csv"
os.environ["ASR_SCOPE"] = "c1260"        # all | standard | c1260 | c1293

%run asr_publication_pipeline.py
```

Use `%run`, not `!python`: it renders every figure inline as the analysis
proceeds. A verified ZIP of all figures and tables is offered for download at
the end. Copy it to Drive (`!cp ASR_publication_outputs.zip
/content/drive/MyDrive/`) so a runtime restart cannot destroy it.

## Quick start — local, VS Code

Faster and immune to Colab disconnects. On Apple silicon the whole scope
comparison takes a few minutes.

```bash
git clone -b claude/confident-wright-ju9c18 https://github.com/mahyarrazm/ASR
cd ASR
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python asr_local.py                                  # which scope to report
ASR_SCOPE=c1260 python asr_publication_pipeline.py   # full figure package
```

`asr_local.py` locates the dataset by name if the path is wrong, or takes
`--csv "/path/with spaces/ASR_FinalE-ComCo.csv"`. Set `REQUIRE_TABPFN = False`
in the pipeline for a fast CPU run: TabPFN costs roughly fifty times the
runtime on CPU for about +0.03 R².

## Configuration

| Variable | Effect |
|---|---|
| `ASR_CSV_PATH` | Dataset location |
| `ASR_SCOPE` | `all`, `standard`, `c1260`, `c1293` |
| `ASR_FAST_MODE=1` | Reduced folds and repeats, for a smoke test |
| `ASR_SKIP_INSTALL=1` | Skip the dependency check |
| `TABPFN_TOKEN` | TabPFN licence token; also read from a Colab secret |

The chosen scope is written into `Table_02_factor_inventory.csv`,
`run_manifest.json` and `HEADLINE_RESULTS.csv`. Narrowing scope is legitimate
only when the reported result states its scope, so the outputs do that
automatically.

## Data format

One CSV row per measurement. Required columns are `x1`–`x29` (note the
capitalised `X21`), `Standard`, and the target `y` in expansion percent.
`x15`, `x28` and `Standard` are categorical; the rest are numeric. A
description row beneath the header is detected automatically — its text is
harvested as figure labels, then the row is dropped. Missing composition
values are imputed inside each fold using medians conditioned on whether the
corresponding supplementary cementitious material is present.

## Protocol

**Measured inputs only.** The five deterministic engineered terms used in
earlier versions (`total_SCM`, `age_x_temp`, `silica_x_alkali`, `log_age`,
`SCM_alkali`) are excluded before any model is fitted. `x8` is dropped because
relative humidity is constant across the analysed rows.

**Mixture grouping.** Each row is assigned to a mixture by hashing every
predictor except testing age. All folds and the locked holdout are
mixture-disjoint.

**Nothing crosses the holdout line.** Model selection, importance ranking and
the input-removal boundary use the development partition only; the holdout is
scored once per reported model.

| Stage | Output |
|---|---|
| Data audit: duplication, repeated-measures structure, distributions | Fig. 1, Tables 1–4 |
| Model comparison under both resampling schemes | Figs. 2–3, Tables 6–11 |
| Target-transform sensitivity | Table 12 |
| Grouped-permutation importance over folds | Fig. 4, Tables 13–14 |
| Cumulative least-important-first input removal | Fig. 5, Tables 15–18 |
| Locked holdout with paired bootstrap intervals | Figs. 6–7, Tables 19–24 |
| Split-conformal prediction intervals and coverage | Fig. 8, Table 25 |
| Reactivity classification at standard expansion limits | Fig. 9, Table 26 |
| y-randomization and learning-curve controls | Fig. 10, Tables 27–28 |
| Calibration and error stratification | Fig. 11, Tables 29–30 |
| SHAP attribution with additivity and stability audits | Figs. 12–15, Tables 31–34 |

## Models compared

Ridge, a multilayer perceptron, TabNet, LightGBM, XGBoost, CatBoost, a fixed
50/50 LightGBM–CatBoost blend, TabPFN, and a fold-pure stacked ensemble with a
BayesianRidge meta-learner. Hyperparameters are fixed in advance, so the
comparison is not confounded by unequal search effort. `asr_model_search.py`
explores a wider set; anything promising found there should be promoted into
the pipeline's fixed specifications rather than tuned inside it.

## Formulations tested and rejected

Recorded because negative results are worth reporting.

| Formulation | Grouped R² |
|---|---|
| Pooled row-wise regression | **0.734** |
| Per-standard models | 0.708 |
| Hybrid (kinetic curve as an input) | 0.536 |
| Kinetic curve, y = A(1 − e^(−kt)) | −1.97 |

The kinetic formulation follows ASR reaction kinetics and works on synthetic
data generated from that curve, but collapses here: mixtures span 14 days to a
year, most have about five measurement points, and predicting the rate
constant then exponentiating compounds the error.

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

## Known ceilings

Replicate rows bound the achievable accuracy at **R² ≤ 0.985**
(irreducible RMSE ≥ 0.0317%), so measurement noise is not the binding
constraint. The binding constraint is that **71% of the variance sits between
mixtures** and must come from composition alone; testing age explains the
other 29%. Reaching R² 0.9 would require explaining about 86% of the
between-mixture variance from composition, which no model architecture
supplies. The learning curve saturates near 100 mixtures, so more measurement
ages of existing mixtures will not help — only more distinct mixtures, or new
features, will.

## Outputs

`ASR_publication_outputs/figures/` holds every figure as 600-dpi PNG, vector
PDF and editable SVG. `ASR_publication_outputs/results/` holds every table as
CSV, plus `run_manifest.json`, `environment_lock.txt`,
`artefact_inventory.csv` with SHA-256 checksums, `HEADLINE_RESULTS.csv` and
`FIGURE_INDEX_AND_DRAFT_CAPTIONS.csv`.

## Reproducibility

All stochastic steps derive from `RANDOM_STATE`. Preprocessing, the target
transform and the stacking meta-learner are refitted inside every fold. The
run manifest records the resolved version of every dependency, the selected
model, the analysis scope, the exact removal order, and the stated
limitations.

## Licence and citation

Code released for academic use. TabPFN model weights carry a separate
non-commercial licence from Prior Labs and must be obtained through their
terms. Please cite the accompanying article when using this pipeline.
