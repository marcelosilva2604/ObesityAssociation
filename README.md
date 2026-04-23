# Cross-sectional early-life features are insufficient for childhood obesity prediction: evidence from ENANI-2019

Code repository for the manuscript:

> Carvalho e Silva M, et al. *Cross-sectional early-life features from a nationally representative Brazilian survey are insufficient for individual-level childhood obesity prediction: evidence from ENANI-2019.* Preventive Medicine (submitted, 2026).

The prior companion study applying the same analytical approach to a distinct ENANI-2019 subsample (exclusive breastfeeding duration in infants under six months) is published at:

> Carvalho e Silva M, Delgado AF, Carvalho e Silva F. *Machine learning identification of factors associated with exclusive breastfeeding in Brazilian infants: Cross-sectional analysis of the ENANI-2019 survey.* International Journal of Medical Informatics. 2025. doi:10.1016/j.ijmedinf.2025.106126. PMID: 41218392.

## Research question

Whether the cross-sectional retrospective data format that dominates low- and middle-income country child-nutrition surveillance can support individual-level machine learning prediction of early-childhood obesity (BMI-for-age z-score >+3 SD, WHO Child Growth Standards) at a level compatible with screening deployment.

**Finding:** across 13 machine learning algorithms evaluated over three model configurations, maximum cross-validated AUC-ROC was 0.619; precision at the default classification threshold remained below 5% in every configuration. The ceiling persisted under class-imbalance correction (SMOTE), feature-set expansion (15–22× the baseline feature count), and inclusion of derived maternal pre-pregnancy BMI. The same analytical approach recovered AUC-ROC 0.865 on a different outcome (exclusive breastfeeding) in a distinct ENANI-2019 subsample, indicating that the ceiling here reflects the informational content of the cross-sectional feature space rather than limitations of the pipeline.

## Dataset

ENANI-2019 — Estudo Nacional de Alimentação e Nutrição Infantil (Brazilian National Survey on Child Nutrition), a probabilistic household survey conducted by the Federal University of Rio de Janeiro (UFRJ) between February 2019 and March 2020. Microdata are curated by UFRJ and available via the official data access pathway at https://enani.nutricao.ufrj.br/. Secondary analysis approved under CAAE 91352125.7.0000.0243.

This analysis uses the 2–4 years age stratum (n = 8,236). Raw data files are excluded from this repository by `.gitignore` because of the ENANI-2019 access terms and GitHub file-size limits.

## Pipeline

Sequential stages, each contained in a separate folder:

| Folder | Purpose |
|---|---|
| `A-Age Sample Filter/` | Filter ENANI-2019 to children aged 2–4 years (741 variables, n = 8,236). |
| `B-Featuring Removing/` | Remove columns with >30% missing values (741 → 419) and domain-based variable selection (419 → 56 variables focused on the first 24 months of life). |
| `C-Feature Engeneering/` | Feature engineering across five sequential notebooks (one-hot encoding, consolidation of rare categories, standardization). |
| `D-Train-Test Split/` | Construct the three model configurations stratified by missingness pattern; 80/20 stratified train–test split with `random_state=42`. |
| `E-FeatureSelection/` | L1/L2/Elastic-Net regularization and 13-algorithm evaluation under nested 5-fold stratified cross-validation; sensitivity analyses (SMOTE, expanded feature sets, maternal pre-pregnancy BMI). |
| `F-Modelo vencedor com holdOut/` | Hold-out validation with bootstrap 95% confidence intervals (n = 1,000). |

### Sensitivity-analysis scripts (under `E-FeatureSelection/`)

- `run_algorithm.py`, `run_all_expanded.py`, `run_holdout_new.py` — driver scripts for the 13-algorithm evaluation on restricted and expanded feature sets.
- `feature_reduction_sensitivity.py`, `feature_reduction_ml_sensitivity.py`, `feature_reduction_holdout_sensitivity.py` — feature-reduction sensitivity pipeline.
- `maternal_bmi_sensitivity.py` — maternal pre-pregnancy BMI sensitivity analysis (three models × three algorithms × baseline vs. extended).

## Reproducibility

- All analyses implemented in Python 3.
- Primary dependencies: `pandas`, `numpy`, `scikit-learn`, `scipy`, `statsmodels`, `xgboost`, `lightgbm`, `catboost`, `pytorch_tabnet`, `imbalanced-learn`, `python-docx`, `matplotlib`.
- All random processes use fixed seeds (`random_state=42`).
- `RobustScaler` normalization is fit inside each cross-validation fold to prevent data leakage.
- Class reweighting is applied uniformly: `class_weight='balanced'` for algorithms that support it, `scale_pos_weight=33` for XGBoost and LightGBM, `auto_class_weights='Balanced'` for CatBoost, sample weights for TabNet.
- Results (AUC-ROC point estimates and bootstrap 95% CIs) for all algorithm × model combinations are saved as JSON under `E-FeatureSelection/`.

## Conventions

- All code, comments, docstrings, variable names, and notebook markdown cells are in English.
- Pre-processing thresholds are pre-specified: >30% missing-value removal; feature-block selection restricted to thematic blocks relevant to the first 24 months of life (blocks A, B, D, H, I ≤24 months, J, K, Q, R, X) for the expanded sensitivity analysis.

## Ethics

Secondary analysis of de-identified ENANI-2019 data, approved by the Research Ethics Committee (CAAE: 91352125.7.0000.0243), conducted in accordance with the principles of the Declaration of Helsinki.

## Data availability

Source microdata are not redistributable under ENANI-2019 access terms; they must be requested directly from UFRJ (https://enani.nutricao.ufrj.br/). All analysis code, intermediate result files (JSON), and analysis scripts in this repository are available under the MIT licence (see `LICENSE`).

## Citation

If you use this code, please cite the companion paper:

```
Carvalho e Silva M, Delgado AF, Carvalho e Silva F.
Machine learning identification of factors associated with exclusive
breastfeeding in Brazilian infants: Cross-sectional analysis of the
ENANI-2019 survey.
International Journal of Medical Informatics. 2025.
doi:10.1016/j.ijmedinf.2025.106126
```

The obesity manuscript citation will be added here once accepted.

## Contact

Marcelo Carvalho e Silva — marcelo_carvalhosilva@hotmail.com
