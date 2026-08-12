# Handoff

Everything a new session needs to continue this project. The chat history is
not durable; this file and the code are.

## Where things stand

The analysis is complete and the code works. What remains is running the final
figure package and writing the paper.

**Last verified state:** `asr_local.py` ran on the real dataset and produced
the scope comparison below. `asr_publication_pipeline.py` runs end to end and
produces 15 figures and 37 tables. HistGradientBoosting was added to the
pipeline's candidates after the scope comparison showed it winning; whether it
also wins inside the pipeline on the `standard` scope has not yet been
confirmed on the real data.

## The two findings that define the project

### 1. Repeated-measure leakage

The dataset has **1,999 rows from 210 mixtures — 99.1% repeated measures**.
Mortar bars are measured repeatedly over time, so splitting rows at random
puts different ages of the same specimen into training and test.

| Protocol | R² |
|---|---|
| Random-row cross-validation (the conventional approach) | 0.987 |
| Mixture-disjoint cross-validation | 0.62 |

The optimism is concentrated in the strongest models: the stacked ensemble and
CatBoost overstate by 0.36 R² and understate RMSE by 56%, while Ridge and
TabNet overstate by only 0.13–0.18. Models that fit better exploit the leak
harder, which is what leakage predicts and noise does not.

No ASR paper found in a literature search reports this. It is the project's
methodological contribution.

### 2. Pooled protocols depress accuracy

The dataset mixes three incompatible experiments. Fitting them separately
under identical grouped folds (`asr_local.py`, real data):

| Scope | Best model | R² | RMSE | Rows | Mixtures |
|---|---|---|---|---|---|
| **standard** (both ASTM) | HistGBM | **0.770** | **0.0856** | 1,119 | 153 |
| all pooled | CatBoost | 0.738 | 0.1342 | 1,999 | 210 |
| c1260 only | CatBoost | 0.715 | 0.1127 | 661 | 131 |
| nonstandard | CatBoost | 0.631 | 0.1832 | 880 | 57 |
| c1293 only | RandomForest | 0.547 | 0.0660 | 458 | 22 |

**Use `ASR_SCOPE=standard`.** Both metrics improve together against the pooled
baseline, so it is a real gain, not an artefact of evaluating on a subset with
less variance. The 880 non-standard rows are what dragged the pool down; they
followed no standard protocol and cannot be relabelled.

C1293's low R² with the lowest RMSE of any scope is a variance artefact —
report RMSE alongside R² there or it looks worse than it is.

## Why R² 0.9 is not reachable

- Replicate rows bound achievable accuracy at **R² ≤ 0.985** (irreducible
  RMSE ≥ 0.0317%), so measurement noise is not the constraint.
- **71% of the variance sits between mixtures** and must come from composition
  alone; testing age explains the other 29%. Reaching 0.9 would need about 86%
  of the between-mixture variance explained.
- Fig. 1d shows why that is out of reach: the SCM composition blocks
  (x11–x14, x16–x21, x22–x27) are internally correlated near 1.0, meaning very
  few *distinct* SCM sources. Those columns carry far less information than
  their count suggests.
- The learning curve saturates near 100 mixtures, so more measurement ages of
  existing mixtures will not help. Only more distinct mixtures or genuinely
  new features would.

## Approaches tried and rejected

Sixteen model families, four target transforms, per-mixture sample weighting
and a monotone age constraint all converged near 0.62 on pooled data.

| Formulation | Grouped R² |
|---|---|
| Pooled row-wise regression | 0.734 |
| Per-standard models | 0.708 |
| Hybrid (kinetic curve as an extra input) | 0.536 |
| Kinetic curve, y = A(1 − e^(−kt)) | −1.97 |

The kinetic formulation follows ASR reaction kinetics and worked on synthetic
data generated from that curve, but collapsed on real data: mixtures span 14
days to a year, most have about five points, and predicting the rate constant
then exponentiating compounds the error.

## Running it

```bash
cd ~/"Smart Lab"/Lab/ML/ASR
source .venv/bin/activate
ASR_SCOPE=standard ASR_USE_TABPFN=0 ./run.sh pipeline
```

`run.sh` handles the directory, virtual environment and dependencies, and
works from any directory. See README.md for the configuration variables.

**Environment notes.** The laptop runs Python 3.9 with no Homebrew, so
LightGBM and XGBoost cannot load (no OpenMP runtime) and are skipped
automatically; HistGradientBoosting covers that ground and needs no OpenMP.
TabPFN needs Python 3.10+, a licence token, and roughly fifty times the CPU
runtime for about +0.03 R² — leave `ASR_USE_TABPFN=0` unless running on a
Colab GPU.

## Next steps

1. Run the pipeline at `ASR_SCOPE=standard` and confirm whether HistGBM beats
   CatBoost there as the scope comparison predicts.
2. Take the **locked mixture-disjoint holdout** row from the evaluation
   summary as the paper's headline. Grouped CV runs on the smaller development
   partition and reads lower; both belong in the paper, but the holdout is the
   final internal test.
3. Write the paper around the leakage finding, with the scoped result as the
   model contribution:

   > On a compiled ASR database that is 99.1% repeated measures, random-row
   > splitting yields R² 0.99 while mixture-disjoint validation yields ~0.77.
   > We provide the first leakage-free benchmark for ASR expansion, with
   > calibrated prediction intervals and reactivity classification against
   > ASTM limits.

   The 0.99 belongs in the paper as the demonstration of the problem, not as
   the result.

## Honest caveats to carry forward

- The holdout is internal. It is mixture-disjoint but drawn from the same
  compiled sources, so it does not establish external validity.
- Restricting scope is legitimate only because the outputs state the scope.
  `Table_02_factor_inventory.csv`, `run_manifest.json` and
  `HEADLINE_RESULTS.csv` all record it automatically.
- Paired model comparisons are descriptive: folds overlap and the winner is
  chosen on the same development results.
- SHAP values and the age-by-curing pattern describe the fitted model, not
  causal material effects.
