# A Pre-Implementation Evaluation Framework for Machine Learning in Population Surveillance with Rare Outcomes: Validation in Childhood Obesity Classification

Code repository for the manuscript submitted to the *Journal of the American Medical Informatics Association* (JAMIA).

## Overview

This repository contains the complete analytical pipeline for a pre-implementation evaluation framework that determines whether machine learning classification warrants integration into population surveillance platforms with rare outcomes. The framework is validated using childhood obesity classification (BMI-for-age z-score >+3 SD, ~3% prevalence) in the Brazilian National Survey on Child Nutrition (ENANI-2019).

## Repository Structure

### Data Preprocessing

- **A-Age Sample Filter**: Filters ENANI-2019 (n=14,558) to children aged 2-4 years (n=8,236)
- **B-Featuring Removing**: Missing value analysis (>30% threshold: 741 to 419 features) and domain-based variable selection (419 to 56 variables focused on first 24 months of life)
- **C-Feature Engeneering**: Feature engineering across five sequential processing batches (one-hot encoding, consolidation, standardization)
- **D-Train-Test Split**: Missingness-stratified model construction (3 models) and 80/20 stratified train-test splits (random_state=42, stratify=y)

### Stage 2: Feature Selection and Regularization (Nested Cross-Validation)

- **E-FeatureSelection/FSMODEL1.ipynb**: Lasso (L1), Ridge (L2), Elastic Net regularization for Model 1 (27 features, n=8,236)
- **E-FeatureSelection/FSMODEL2.ipynb**: Regularization comparison for Model 2 (17 features, n=5,638)
- **E-FeatureSelection/FSMODEL3.ipynb**: Regularization comparison for Model 3 (44 features, n=5,638)

### Stage 2: Machine Learning Algorithm Evaluation

- **E-FeatureSelection/modelando1e2.ipynb**: Eight ML algorithms (Logistic Regression, Random Forest, Gradient Boosting, Decision Tree, KNN, Gaussian Naive Bayes, SVM Linear, SVM RBF) evaluated via nested 5x5 stratified cross-validation for Models 1 and 2
- **E-FeatureSelection/modelando3.ipynb**: Same eight algorithms for Model 3

### Stage 3: Hold-out Validation and Bootstrap Confidence Intervals

- **F-Modelo vencedor com holdOut/validacao.ipynb**: Hold-out feasibility assessment with bootstrap confidence intervals (n=1,000) for all performance metrics (AUC-ROC, Precision, Recall, F1, Accuracy, Specificity, NPV, PPV)

### Stage 4: SMOTE Sensitivity Analysis

- **E-FeatureSelection/smote_comparison.ipynb**: SMOTE oversampling comparison across all 3 models x 8 algorithms to distinguish informational failure from algorithmic failure (class imbalance)

### Feature Association Diagnostic

- **F-Modelo vencedor com holdOut/featureIC.ipynb**: Bootstrap confidence intervals (n=1,000) for logistic regression coefficients to identify statistically significant feature associations

## Reproducibility

- All analyses use Python 3.8 with pandas, NumPy, scikit-learn, imbalanced-learn, and SciPy
- All random processes use fixed seeds (random_state=42)
- RobustScaler normalization applied within each fold to prevent data leakage
- class_weight='balanced' applied to all algorithms that support it
- Large CSV data files are excluded via .gitignore due to GitHub file size limits

## Dataset

This project uses the ENANI-2019 dataset (Brazilian National Survey on Child Nutrition), a nationwide household survey conducted between February 2019 and March 2020. Secondary analysis approved by CAAE: 91352125.7.0000.0243.
