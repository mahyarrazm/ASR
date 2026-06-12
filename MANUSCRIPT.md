# Uncertainty-quantified prediction of alkali–silica reaction expansion with a tabular-foundation-model ensemble

**Authors:** [Author names and affiliations to be completed]

**Correspondence:** [corresponding author e-mail]

---

## Abstract

Alkali–silica reaction (ASR) is a leading cause of premature deterioration in concrete infrastructure, yet the standard laboratory tests used to anticipate it take from two weeks to more than a year per aggregate–binder combination. Here we present a stacked machine-learning ensemble that predicts mortar-bar ASR expansion from 21 mixture, binder-composition, and exposure descriptors, combining two gradient-boosting models with TabPFN, a tabular foundation model, under a Bayesian linear meta-learner. Trained on 1,999 published expansion measurements, the ensemble achieves a cross-validated R² of 0.986 ± 0.005 and a held-out test R² of 0.988 (RMSE 0.029 % expansion), with the foundation model carrying 77 % of the ensemble weight. Cross-conformal prediction intervals attain 95.2 % empirical coverage at a 95 % nominal level, and the model reproduces the reactivity classification implied by the 0.10 %/0.20 % expansion limits with 95.0 % accuracy (Cohen's κ = 0.92), never erring by more than one class. y-Randomization, multi-seed stability, learning-curve, and applicability-domain analyses support the validity of the result, and SHAP attribution recovers the established physics of ASR — time–temperature kinetics, reactive-aggregate size, alkali loading, and the mitigating role of supplementary cementitious materials — without their being imposed. The framework provides a calibrated, interpretable screening tool that complements accelerated laboratory testing.

---

## Introduction

Alkali–silica reaction — the deleterious reaction between alkali hydroxides in concrete pore solution and metastable silica phases in aggregates, first identified by Stanton more than eighty years ago [1] — produces a hygroscopic gel whose swelling cracks concrete from within. ASR affects dams, bridges, pavements, and nuclear containment structures worldwide, and once initiated it cannot be arrested economically; prevention at the mixture-design stage is therefore the only practical defence [2,3]. The principal preventive levers are well established: limiting the alkali loading of the mixture, avoiding the most reactive aggregate size fractions, and incorporating supplementary cementitious materials (SCMs) such as fly ash, metakaolin, or nano-silica that bind alkalis and densify the microstructure [4,5].

Whether a given combination of aggregate, binder, and exposure will expand deleteriously is, however, still decided empirically. The accelerated mortar-bar test (AMBT) requires 14–28 days at 80 °C in 1 N NaOH [6], and the concrete prism test (CPT), regarded as more representative of field behaviour, requires a full year at 38 °C [7]. Laboratory performance testing is further complicated by the sensitivity of measured expansion to specimen geometry, leaching, and storage conditions [8]. With thousands of candidate aggregate–SCM–exposure combinations of practical interest, exhaustive testing is infeasible, and empirical expansion limits — commonly 0.10 % and 0.20 % — must be applied to sparse data.

Machine learning offers a complementary route: learning the mapping from mixture and exposure descriptors to measured expansion across the accumulated published record, so that new combinations can be screened computationally before targeted laboratory confirmation. Data-driven models are now established tools for concrete property prediction [9,10], and several recent studies have applied neural networks, support-vector machines, and gradient-boosting ensembles to literature databases of roughly 1,900–2,000 ASR expansion measurements [11–13]. These studies demonstrate encouraging accuracy (test R² typically 0.95–0.98) but generally stop at point prediction: they rarely quantify predictive uncertainty in a calibrated way, rarely test the chance-performance (y-randomization) and applicability-domain criteria that are standard in molecular property modeling [14,15], and rarely evaluate the decision-level question practitioners actually face — will this mixture cross the regulatory expansion limit?

Here we address these gaps with a leakage-controlled ensemble framework evaluated to the standards of validated structure–property modeling. Three advances are combined. First, we incorporate TabPFN [16], a transformer-based tabular foundation model pre-trained on millions of synthetic datasets, as a base learner alongside LightGBM [17] and CatBoost [18] in a stacked ensemble [19] — to our knowledge among the first applications of a tabular foundation model to cementitious-materials property prediction — and find that the meta-learner assigns it the dominant weight (0.77). Second, we attach distribution-free uncertainty estimates to every prediction using cross-conformal calibration on out-of-fold residuals [20,21], achieving 95.2 % empirical coverage at the 95 % nominal level on a held-out test set. Third, we translate regression accuracy into decision-level reliability, showing 95.0 % agreement (Cohen's κ = 0.92 [22]) with the three-class reactivity grading implied by the 0.10 %/0.20 % expansion limits, with no specimen misclassified by more than one class. The complete pipeline — including fold-pure preprocessing, y-randomization, learning-curve, multi-seed, and Williams applicability-domain analyses [15] — is released as an open, single-command notebook.

## Results

### A curated expansion database and leakage-controlled modeling framework

The database comprises 1,999 mortar-bar expansion measurements assembled from the published ASR literature, each described by mixture proportions (aggregate-to-cement ratio, water-to-cement ratio, silica content of the sand), reactive-aggregate characteristics (average size), alkali loading (cement Na₂O-equivalent), SCM type and dosage (metakaolin, fly ash, and colloidal nano-silica, with their oxide compositions and alkali contents), specimen geometry (cross-section and length), and exposure (curing temperature and testing age). Measured expansions span 0–1.4 %, with the distribution concentrated below 0.2 % (Supplementary Fig. S1a) — the region in which the accept/reject decisions of practice are made. Missing composition values, which arise structurally when an SCM is absent from a mixture (Supplementary Fig. S1b), were imputed with conditional medians learned only from the training portion of each data split. After an ablation study, 16 base descriptors were retained and augmented with five engineered features (total SCM content, age × temperature, silica × alkali, log-transformed age, and combined SCM alkali content), giving 21 model inputs (Methods).

The model is a stacked ensemble: LightGBM, CatBoost, and TabPFN base learners are combined by a BayesianRidge meta-learner trained exclusively on inner out-of-fold predictions, so that no base learner ever scores a sample it was trained on. The target was variance-stabilized by a Box–Cox transform (λ = 0.85). All preprocessing — imputation statistics and the transform shift — was refit inside every fold of every evaluation ("fold-pure"), eliminating the train–test leakage that inflates apparent performance in many applied studies.

### Predictive performance

In 10-fold cross-validation stratified by expansion quantiles within the 80 % training partition, the ensemble achieved R² = 0.9862 ± 0.0045, RMSE = 0.0302 ± 0.0044 % expansion, MAE = 0.0144 ± 0.0014 %, and a median absolute percentage error of 4.1 ± 0.9 % (mean ± s.d. over folds; pooled out-of-fold R² = 0.9864; Supplementary Table S1). On the untouched 20 % holdout (n = 400), the final model reached R² = 0.9878, RMSE = 0.0293 %, and MAE = 0.0139 % (Fig. 1; Supplementary Table S4). The modest train–test gap (train R² = 0.9975) and the closeness of cross-validated and holdout estimates indicate limited overfitting. Performance was insensitive to the choice of data split: across five random 80/20 partitions with all model seeds varied jointly, test R² was 0.9863 ± 0.0024 (Fig. 10; Supplementary Table S3).

Under identical folds, preprocessing, and target transform, the ensemble was compared with five reference learners and its own components (Fig. 4; Supplementary Tables S2, S2b). A standardized linear ridge regression reached only R² = 0.51 ± 0.04, confirming that ASR expansion is a strongly non-linear function of its descriptors; k-nearest neighbours reached 0.69 ± 0.06 and random forest 0.93 ± 0.02. The tuned gradient-boosting learners performed strongly alone (LightGBM 0.9729 ± 0.0080; CatBoost 0.9784 ± 0.0071), and TabPFN — applied without any task-specific tuning — exceeded both (0.9859 ± 0.0052). The stacked ensemble was significantly better than every classical baseline and both gradient-boosting learners (two-sided paired Wilcoxon on per-fold R², p ≤ 0.0039), and statistically indistinguishable from TabPFN alone (p = 0.105) while showing lower fold-to-fold variance and the lowest RMSE overall. Consistently, the meta-learner assigned weights of 0.77 to TabPFN, 0.16 to CatBoost, and 0.08 to LightGBM.

A learning-curve analysis (Fig. 5; Supplementary Table S7) shows test R² rising from 0.904 with 320 training samples to 0.988 with the full 1,599, still improving at the largest size — indicating that the assembled database is large enough to support the reported accuracy, and that further data collection should yield further, diminishing, gains.

### Chance performance, residual structure, and applicability domain

Three analyses probe the validity of the result. First, y-randomization: retraining the full pipeline on ten random permutations of the training targets collapsed test R² to −0.003 ± 0.002 (Fig. 8; Supplementary Table S8), demonstrating that the model's performance cannot be an artifact of leakage, descriptor redundancy, or overfitting capacity. Second, residual diagnostics (Fig. 6): test-set residuals are centred on zero without trend against the predicted value; their distribution is sharply peaked with heavier-than-Gaussian tails (Fig. 6b, c), driven by a small number of specimens — a structure that motivates the distribution-free uncertainty quantification below rather than Gaussian error bands. Third, the Williams plot (Fig. 11; Supplementary Table S11): with leverage threshold h* = 0.043, 95.7 % of training and 96.8 % of test samples fall inside the applicability domain, and the high-leverage points are dispersed rather than clustered, indicating that the test set is overwhelmingly interpolative with respect to the training chemistry. Predictions for new mixtures outside this domain should be flagged and confirmed experimentally.

### Calibrated, distribution-free prediction intervals

Point predictions alone cannot support engineering decisions near a regulatory limit. We therefore attached cross-conformal prediction intervals [20,21] to every prediction: the symmetric half-width q is the finite-sample-corrected 95 % quantile of the absolute out-of-fold residuals from the fold-pure cross-validation — a calibration set that is leak-free by construction and costs no training data. The resulting intervals of ±0.062 % expansion achieved 95.2 % empirical coverage on the 400 held-out specimens against a 95 % nominal target (Fig. 9a; Supplementary Table S9), and coverage remained at the nominal level within each quartile of measured expansion (Fig. 9b), confirming that the intervals are not selectively failing at high expansions where the consequences of under-prediction are greatest.

### Decision-level reliability at reactivity limits

Mapping measured and predicted test-set expansions onto the three-class grading implied by the widely used 0.10 % and 0.20 % limits (innocuous / potentially reactive / reactive), the model agreed with the measured class for 95.0 % of specimens, with Cohen's κ = 0.923 (linearly weighted κ = 0.947; Fig. S3; Supplementary Table S10). Critically, every one of the 20 misclassifications fell into an adjacent class; no innocuous mixture was predicted reactive or vice versa. Combined with the conformal intervals — which allow a user to ask whether the entire 95 % interval lies below a limit — this supports the use of the model as a conservative computational pre-screen ahead of laboratory testing.

### Feature attribution recovers the physics of ASR

SHAP attribution over the gradient-boosting components of the fitted ensemble (Methods) identifies age × temperature, average reactive-aggregate size, cement alkali content (Na₂O-equivalent), and total SCM content as the four dominant drivers of predicted expansion (Figs. 2, 3; Supplementary Table S5), followed by the silica content of the sand and the metakaolin replacement level. The dependence structure of these effects (Fig. S2) is consistent with the established mechanistic picture, although none of it was imposed on the model. The age × temperature response rises steeply and saturates (Fig. S2a), reproducing the thermally activated, asymptotic expansion kinetics that underlie accelerated testing [8]. Expansion attribution decreases with increasing average reactive-aggregate size over the range covered (Fig. S2b), consistent with the greater specific surface of finer reactive particles and the size dependence documented in pessimum-effect studies [2,3]. Total SCM content shows a monotonic, dose-dependent mitigation (Fig. S2d), the central empirical fact of ASR prevention practice [5]. The cement-alkali effect (Fig. S2c) is positive at low test severity but interacts strongly with specimen geometry and test protocol, as expected when external alkali supply (as in the NaOH-immersed AMBT) overwhelms the internal alkali contribution [6,8] — a reminder that attribution on heterogeneous literature data reflects the testing landscape as well as the chemistry. The feature correlation structure (Fig. 7) confirms that the engineered interaction features carry the strongest direct associations with expansion while the underlying composition variables remain only weakly mutually correlated.

## Discussion

This work establishes that ASR mortar expansion — a property whose measurement occupies weeks to a year — can be predicted from tabulated mixture and exposure descriptors with an accuracy (test R² 0.988, RMSE 0.029 % expansion) at the upper end of what has been reported on comparable databases [11–13], while adding the elements those studies have generally lacked: calibrated uncertainty, applicability-domain delineation, chance-performance falsification, and decision-level evaluation against the expansion limits used in practice. The evaluation protocol is deliberately conservative — fold-pure preprocessing, stratified cross-validation inside the training partition only, an untouched holdout, and multi-seed replication — so the reported numbers are estimates of generalization to unseen specimens from the same literature population, not refits to it.

The strongest single finding is the performance of the tabular foundation model. TabPFN, used off-the-shelf with no task-specific tuning, outperformed both carefully tuned gradient-boosting models and received 77 % of the ensemble weight; the full stack improved upon it only marginally (and not significantly, p = 0.105), though with reduced variance and with the practical insurance of not depending on a single learner family. This suggests that in materials informatics — where datasets of 10²–10⁴ rows are the norm and per-dataset tuning is a recurring cost — foundation models pre-trained on synthetic tabular tasks [16] are already competitive with, and can exceed, the gradient-boosting state of the art. We expect this finding to generalize beyond ASR to other laboratory-property prediction problems in cementitious materials.

For practice, the framework functions as a screening instrument rather than a replacement for testing. A candidate mixture can be evaluated in milliseconds; if its 95 % conformal interval lies entirely below 0.10 % expansion at the design age and the mixture falls inside the applicability domain, laboratory confirmation can be prioritized elsewhere. The three-class analysis indicates the cost of this shortcut is small — 5 % adjacent-class disagreement, no two-class errors on 400 held-out specimens — and the conformal guarantee is distribution-free, an important property given the heavy-tailed residual structure we observe. Conversely, mixtures flagged outside the Williams domain (high leverage) are extrapolations and must be tested: the model's accuracy claim does not extend to them.

Several limitations bound the interpretation. First, the database aggregates accelerated laboratory tests of heterogeneous protocols; the model predicts laboratory expansion under the encoded conditions, not field expansion, and the protocol mix shapes some attributions (notably the cement-alkali interaction discussed above). Second, the curing-condition descriptor was categorical and uninformative after numeric encoding and was excluded; its information is partially captured by temperature, geometry, and age, but explicit protocol encoding is a target for future versions. Third, although the conformal intervals are marginally calibrated and quartile-uniform, they are constant-width; locally adaptive (normalized) conformal methods could narrow intervals for easy specimens at the price of methodological complexity. Fourth, percentage errors are necessarily large for near-zero expansions (median APE 4.1 %), which is why absolute metrics and class agreement are the appropriate yardsticks in the decision region. Finally, the three SCMs represented (metakaolin, fly ash, nano-silica) bound the chemical scope; slag- or natural-pozzolan systems require retraining or, at minimum, applicability-domain screening.

The complete pipeline — data loading with automatic descriptor labeling, fold-pure evaluation, all figures and supplementary tables — runs end-to-end in a single public notebook (Code availability), and is structured so that an enlarged database, additional SCMs, or alternative base learners can be substituted without altering the validation protocol. We anticipate its use both as a practical pre-screen for ASR-resistant mixture design and as a template for uncertainty-quantified property prediction in cementitious-materials research generally.

## Methods

### Database

The dataset comprises 1,999 mortar-bar ASR expansion measurements (single CSV; one row per specimen observation) compiled from the published literature, spanning measured expansions of 0–1.4 %. Each row records: silica content of the sand; aggregate-to-cement ratio; average size of the reactive aggregate; water-to-cement ratio; cement alkali content (Na₂O-equivalent); curing temperature; specimen cross-section area and length; metakaolin descriptors (SiO₂, Al₂O₃, replacement %, alkali content); fly-ash descriptors (oxide composition, class, replacement %, alkali content); colloidal-nano-silica descriptors (composition, replacement %, alkali content); testing age; and the measured expansion (%). A non-numeric descriptor (curing-condition category) was excluded from modeling. Rows with a non-numeric target (a units/description header) were dropped automatically at load time, and the descriptor names embedded in that header were harvested to label all figures.

### Features, engineering, and fold-pure imputation

An ablation study on the full descriptor set retained 16 base features; ten composition descriptors of low marginal value (fly-ash composition and class; nano-silica composition) were removed from the model input but remain in the loading pipeline because they feed the engineered features and imputation. Five engineered features were added: total SCM content (sum of the three replacement percentages); age × temperature; silica × alkali; log(1 + age); and combined SCM alkali content (sum of the three SCM alkali contents), giving 21 model inputs. Missing SCM-composition values, which occur structurally when the corresponding SCM is absent, were imputed with the median of the corresponding column computed over training-fold rows in which that SCM is present (conditional median); SCM content columns and all remaining features used training-fold medians; the fly-ash class used the training-fold mode among fly-ash-containing rows. All imputation statistics were refit inside every fold of every analysis.

### Target transform

The target was transformed by a Box–Cox power transform [23] with fixed exponent λ = 0.85, applied to (y − shift) where shift = min(y_train) − 10⁻⁴ is refit on each training fold so the argument is strictly positive; predictions were inverse-transformed before computing any metric. The exponent was fixed, before the evaluation reported here, on the training partition only.

### Base learners and stacked ensemble

The ensemble stacks three base regressors: LightGBM [17] (2,900 trees, maximum depth 8, learning rate 0.0561, subsample 0.624, column subsample 0.783, L1/L2 regularization 0.030/2.884, minimum 7 samples per leaf, 32 leaves), CatBoost [18] (2,600 iterations, depth 10, learning rate 0.0825, L2 leaf regularization 15.34, bagging temperature 0.242, random strength 0.414), and TabPFN v2 [16] (default settings, GPU inference, `ignore_pretraining_limits=True`). Hyperparameters of the gradient-boosting learners were selected before this evaluation using the training partition only; TabPFN was not tuned. Stacking follows Wolpert [19] as implemented in scikit-learn's `StackingRegressor`: each base learner produces out-of-fold predictions under an internal 5-fold split (fixed seed), a BayesianRidge meta-learner [24] is fitted to those leak-free predictions, and the base learners are then refit on the full training data for inference. When TabPFN or its weights are unavailable the pipeline falls back automatically to the two-learner stack.

### Evaluation protocol and metrics

The data were split 80/20 (random_state = 256) into training and holdout partitions. Model selection and all internal analyses used only the training partition. Cross-validation used 10 folds stratified by deciles of the target (via quantile binning) with preprocessing and target-transform refit per fold. Reported metrics are R², RMSE, MAE, and median absolute percentage error (MedAPE; percentage errors are undefined at zero expansion and dominated by near-zero denominators, so the median rather than the mean is reported). Split-robustness was assessed by repeating the entire split–fit–test cycle over five seeds (256, 7, 42, 101, 2024) with base-learner seeds varied jointly. The learning curve retrained the full pipeline on nested random subsets (20–100 %) of the training partition, evaluating on the fixed holdout.

### Baseline comparison and statistics

Ridge regression and k-nearest neighbours (k = 5, distance weighting), both on standardized features, random forest (500 trees, minimum leaf 2), the two gradient-boosting learners, and TabPFN alone were evaluated under the identical folds, fold-pure preprocessing, and target transform. Per-fold R² of each model was compared with the stacked ensemble by two-sided paired Wilcoxon signed-rank tests over the 10 common folds; because cross-validation folds share training data, these p-values are approximate [25] and are reported alongside effect sizes. y-Randomization [14] retrained the full pipeline on ten random permutations of the training targets and evaluated against the true holdout targets.

### Cross-conformal prediction intervals

Prediction intervals follow the cross-conformal construction [20]: the half-width q is the ⌈(n+1)(1−α)⌉-th order statistic (finite-sample correction) of the n absolute out-of-fold residuals from the fold-pure cross-validation, with α = 0.05. Intervals are prediction ± q. Marginal coverage was verified on the holdout set and conditional coverage within quartiles of measured expansion. The cross-conformal guarantee is approximate but mildly conservative, as the calibration models are trained on 9/10 of the training data [20,21].

### Applicability domain

The Williams plot [15] used leverages h_i = a_iᵀ(AᵀA)⁻¹a_i, with A the standardized training feature matrix augmented with an intercept (pseudo-inverse for numerical stability), warning threshold h* = 3p/n, and residuals standardized by the RMSE of their own split; samples with h < h* and |standardized residual| < 3 were counted inside the applicability domain.

### Expansion-limit class agreement

Measured and predicted holdout expansions were classified as < 0.10 %, 0.10–0.20 %, or > 0.20 %, the limits commonly applied in accelerated mortar-bar assessment [6]. Agreement was summarized by the confusion matrix, overall accuracy, and Cohen's κ [22] (unweighted and linearly weighted, the latter appropriate for ordinal classes).

### SHAP attribution

Exact TreeSHAP values [26,27] were computed for the LightGBM and CatBoost components on the holdout set and combined with the renormalized absolute meta-learner weights of those components; TabPFN has no exact SHAP algorithm and is excluded from attribution. Because base learners operate on the Box–Cox-transformed target, SHAP magnitudes are in transformed units; rankings and dependence shapes are unaffected. Dependence plots use SHAP's automatic selection of the strongest interacting feature for coloring.

### Software and reproducibility

Analyses used Python 3.12 with numpy 2.0, pandas 2.2, scikit-learn 1.6 [28], LightGBM 4.6, CatBoost 1.2, TabPFN 2.x, and SHAP. All stochastic steps are seeded; package versions are printed at run time; every figure (600-dpi PNG and vector PDF) and supplementary table (CSV) is regenerated by a single notebook execution.

## Data availability

The compiled expansion database is available from the corresponding author on reasonable request [or: is provided as Supplementary Data 1]. Source data for all figures are provided with this paper as CSV files (Supplementary Tables S0–S11).

## Code availability

The complete analysis pipeline (Google Colab notebook and equivalent Python script) is available at [repository URL/DOI to be inserted upon acceptance].

## References

1. Stanton, T. E. Expansion of concrete through reaction between cement and aggregate. *Proc. Am. Soc. Civ. Eng.* **66**, 1781–1811 (1940).
2. Rajabipour, F., Giannini, E., Dunant, C., Ideker, J. H. & Thomas, M. D. A. Alkali–silica reaction: current understanding of the reaction mechanisms and the knowledge gaps. *Cem. Concr. Res.* **76**, 130–146 (2015).
3. Fournier, B. & Bérubé, M.-A. Alkali–aggregate reaction in concrete: a review of basic concepts and engineering implications. *Can. J. Civ. Eng.* **27**, 167–191 (2000).
4. Nixon, P. J. & Sims, I. (eds) *RILEM Recommendations for the Prevention of Damage by Alkali-Aggregate Reactions in New Concrete Structures* (Springer, 2016).
5. Thomas, M. The effect of supplementary cementing materials on alkali–silica reaction: a review. *Cem. Concr. Res.* **41**, 1224–1231 (2011).
6. ASTM International. *ASTM C1260: Standard Test Method for Potential Alkali Reactivity of Aggregates (Mortar-Bar Method)* (ASTM International, West Conshohocken, PA).
7. ASTM International. *ASTM C1293: Standard Test Method for Determination of Length Change of Concrete Due to Alkali-Silica Reaction* (ASTM International, West Conshohocken, PA).
8. Lindgård, J. et al. Alkali–silica reactions (ASR): literature review on parameters influencing laboratory performance testing. *Cem. Concr. Res.* **42**, 223–243 (2012).
9. Ben Chaabene, W., Flah, M. & Nehdi, M. L. Machine learning prediction of mechanical properties of concrete: critical review. *Constr. Build. Mater.* **260**, 119889 (2020).
10. DeRousseau, M. A., Kasprzyk, J. R. & Srubar, W. V. III. Computational design optimization of concrete mixtures: a review. *Cem. Concr. Res.* **109**, 42–53 (2018).
11. Yang, L., Lai, B., Xu, R., Hu, X., Su, H., Cusatis, G. & Shi, C. Prediction of alkali-silica reaction expansion of concrete using artificial neural networks. *Cem. Concr. Compos.* (2023). [volume/pages to verify]
12. A new understanding of the alkali-silica reaction expansion in concrete using a hybrid ensemble model. *J. Build. Eng.* (2024). [authors/volume to verify]
13. Prediction of alkali-silica reaction expansion of concrete using explainable machine learning methods. *Discov. Appl. Sci.* (2025). https://doi.org/10.1007/s42452-025-06880-y [authors to verify]
14. Rücker, C., Rücker, G. & Meringer, M. y-Randomization and its variants in QSPR/QSAR. *J. Chem. Inf. Model.* **47**, 2345–2357 (2007).
15. Gramatica, P. Principles of QSAR models validation: internal and external. *QSAR Comb. Sci.* **26**, 694–701 (2007).
16. Hollmann, N. et al. Accurate predictions on small data with a tabular foundation model. *Nature* **637**, 319–326 (2025).
17. Ke, G. et al. LightGBM: a highly efficient gradient boosting decision tree. In *Advances in Neural Information Processing Systems* **30** (2017).
18. Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A. V. & Gulin, A. CatBoost: unbiased boosting with categorical features. In *Advances in Neural Information Processing Systems* **31** (2018).
19. Wolpert, D. H. Stacked generalization. *Neural Netw.* **5**, 241–259 (1992).
20. Vovk, V. Cross-conformal predictors. *Ann. Math. Artif. Intell.* **74**, 9–28 (2015).
21. Angelopoulos, A. N. & Bates, S. Conformal prediction: a gentle introduction. *Found. Trends Mach. Learn.* **16**, 494–591 (2023).
22. Cohen, J. A coefficient of agreement for nominal scales. *Educ. Psychol. Meas.* **20**, 37–46 (1960).
23. Box, G. E. P. & Cox, D. R. An analysis of transformations. *J. R. Stat. Soc. B* **26**, 211–252 (1964).
24. MacKay, D. J. C. Bayesian interpolation. *Neural Comput.* **4**, 415–447 (1992).
25. Nadeau, C. & Bengio, Y. Inference for the generalization error. *Mach. Learn.* **52**, 239–281 (2003).
26. Lundberg, S. M. & Lee, S.-I. A unified approach to interpreting model predictions. In *Advances in Neural Information Processing Systems* **30** (2017).
27. Lundberg, S. M. et al. From local explanations to global understanding with explainable AI for trees. *Nat. Mach. Intell.* **2**, 56–67 (2020).
28. Pedregosa, F. et al. Scikit-learn: machine learning in Python. *J. Mach. Learn. Res.* **12**, 2825–2830 (2011).

## Acknowledgements

[To be completed.]

## Author contributions

[To be completed.]

## Competing interests

The authors declare no competing interests.

---

## Figure legends

**Fig. 1 | Predicted versus measured ASR expansion.** Parity plots for the training set (**a**, n = 1,599) and the held-out test set (**b**, n = 400) of the stacked LightGBM + CatBoost + TabPFN ensemble. Dashed line, 1:1; insets give R², RMSE, MAE, and median absolute percentage error of each split.

**Fig. 2 | SHAP summary of feature effects.** Beeswarm of SHAP values (Box–Cox units) on the test set for the 15 most influential features, computed by exact TreeSHAP on the gradient-boosting components weighted by their meta-learner coefficients; TabPFN is excluded from attribution (no exact SHAP algorithm). Color encodes the feature value.

**Fig. 3 | Mean absolute SHAP feature importance.** Mean |SHAP| (Box–Cox units) of the 15 most influential features on the test set.

**Fig. 4 | Model comparison under identical evaluation.** Mean ± s.d. over the same 10 stratified, fold-pure cross-validation folds for R² (**a**) and RMSE (**b**). All models share identical folds, imputation, and target transform. Orange, stacked ensemble. Two-sided paired Wilcoxon tests versus the ensemble: p = 0.0020 (ridge, k-NN, random forest, LightGBM), p = 0.0039 (CatBoost), p = 0.1055 (TabPFN).

**Fig. 5 | Learning curve.** Test R² (blue, left axis) and RMSE (orange, right axis) of ensembles retrained on nested random subsets of the training partition and evaluated on the fixed holdout set.

**Fig. 6 | Residual diagnostics (test set).** **a**, Residuals versus predicted expansion. **b**, Residual distribution with Gaussian fit (line). **c**, Normal quantile–quantile plot showing heavier-than-Gaussian tails, motivating distribution-free uncertainty quantification.

**Fig. 7 | Feature correlation structure.** Pearson correlations among the 21 model features and measured expansion (training partition).

**Fig. 8 | y-Randomization test.** Distribution of holdout R² after retraining on ten random permutations of the training targets (grey; −0.003 ± 0.002) versus the true model (red line, R² = 0.988).

**Fig. 9 | Cross-conformal prediction intervals.** **a**, 95 % intervals (±0.062 % expansion; shaded) around predictions for the 400 test specimens sorted by predicted value; points outside the interval in red; empirical coverage 95.2 %. **b**, Coverage within quartiles of measured expansion; dashed line, 95 % nominal target.

**Fig. 10 | Performance stability.** **a**, Distribution of R² and RMSE (×10) over the ten cross-validation folds. **b**, Holdout R² across five random 80/20 splits with all model seeds varied jointly (dashed line, mean = 0.9863).

**Fig. 11 | Applicability domain (Williams plot).** Standardized residuals versus leverage for training (grey) and test (blue) samples; vertical dashed line, leverage threshold h* = 3p/n; horizontal dotted lines, ±3 s.d. 95.7 % of training and 96.8 % of test samples lie inside the domain.

## Supplementary information

**Supplementary Fig. S1 | Dataset overview.** **a**, Distribution of measured expansion. **b**, Missing-value fraction by descriptor (structural missingness of SCM composition when the SCM is absent).

**Supplementary Fig. S2 | SHAP dependence of the four dominant features.** **a**, age × temperature; **b**, average reactive-aggregate size; **c**, cement alkali content (Na₂O-equivalent); **d**, total SCM content. Color, strongest interacting feature (automatic selection).

**Supplementary Fig. S3 | Expansion-limit class agreement.** Confusion matrix of measured versus predicted reactivity class at the 0.10 %/0.20 % limits on the test set; accuracy 95.0 %, Cohen's κ = 0.923 (linear κ = 0.947). All errors are adjacent-class.

**Supplementary Tables S0–S11.** S0, descriptor summary statistics; S1, per-fold cross-validation metrics; S2/S2b, model comparison and per-fold R²; S3, multi-seed stability; S4, holdout metrics; S5, SHAP importances; S6, correlation matrix; S7, learning curve; S8, y-randomization; S9, conformal calibration and coverage; S10, expansion-limit confusion matrix and agreement statistics; S11, applicability-domain statistics.
