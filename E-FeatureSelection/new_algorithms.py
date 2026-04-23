#!/usr/bin/env python3
"""
Extended algorithm evaluation: XGBoost, LightGBM, CatBoost, MLP, and TabNet.
Runs nested 5-fold stratified CV on all 3 models, matching the existing pipeline.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from scipy import stats
import warnings
import json
import os

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================
BASE = '/Users/marcelosilva/project/early-obesity-prediction/D-Train-Test Split'

MODEL_CONFIGS = {
    'Model 1': {
        'name': 'Model 1 (Demographic/Perinatal)',
        'train': f'{BASE}/MODEL1TRAIN.csv',
        'test': f'{BASE}/MODEL1TEST.csv',
        'binary_fix': False,
    },
    'Model 2': {
        'name': 'Model 2 (Feeding Practices)',
        'train': f'{BASE}/MODEL2TRAIN.csv',
        'test': f'{BASE}/MODEL2TEST.csv',
        'binary_fix': True,
    },
    'Model 3': {
        'name': 'Model 3 (Combined)',
        'train': f'{BASE}/MODEL3TRAIN.csv',
        'test': f'{BASE}/MODEL3TEST.csv',
        'binary_fix': True,
    },
}

BINARY_VARS = [
    'doou_leite_banco', 'recebeu_leite_banco', 'amamentou_outra_crianca',
    'usou_mamadeira', 'deixou_amamentar_por_outra', 'busca_info_aleitamento',
    'deu_outros_liquidos', 'k15_recebeu', 'k11_amamentou', 'k03_prenatal',
    'usou_utensilios_amamentacao'
]

TARGET_COLS = ['desnutrido', 'eutrofico', 'sobrepeso', 'obeso']

OUTER_CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
INNER_CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


# =============================================================================
# NEW ALGORITHMS
# =============================================================================

# Calculate scale_pos_weight for XGBoost/LightGBM (approx 33:1 ratio)
SCALE_POS = 33.0

NEW_ALGORITHMS = {
    'XGBoost': {
        'model': XGBClassifier(
            random_state=42,
            scale_pos_weight=SCALE_POS,
            eval_metric='logloss',
            use_label_encoder=False,
            verbosity=0,
        ),
        'params': {
            'model__n_estimators': [50, 100, 200],
            'model__max_depth': [3, 5, 7],
            'model__learning_rate': [0.01, 0.1, 0.2],
        }
    },
    'LightGBM': {
        'model': LGBMClassifier(
            random_state=42,
            scale_pos_weight=SCALE_POS,
            verbose=-1,
        ),
        'params': {
            'model__n_estimators': [50, 100, 200],
            'model__max_depth': [3, 5, 7, -1],
            'model__learning_rate': [0.01, 0.1, 0.2],
        }
    },
    'CatBoost': {
        'model': CatBoostClassifier(
            random_state=42,
            auto_class_weights='Balanced',
            verbose=0,
        ),
        'params': {
            'model__iterations': [50, 100, 200],
            'model__depth': [3, 5, 7],
            'model__learning_rate': [0.01, 0.1, 0.2],
        }
    },
    'MLP': {
        'model': MLPClassifier(
            random_state=42,
            max_iter=1000,
            early_stopping=True,
            validation_fraction=0.15,
        ),
        'params': {
            'model__hidden_layer_sizes': [(64,), (128,), (64, 32), (128, 64), (128, 64, 32)],
            'model__activation': ['relu', 'tanh'],
            'model__alpha': [0.0001, 0.001, 0.01],
            'model__learning_rate': ['adaptive'],
        }
    },
}

# TabNet needs special handling (no sklearn Pipeline compatibility for GridSearchCV)
TABNET_CONFIG = {
    'name': 'TabNet',
    'param_grid': [
        {'n_d': 8, 'n_a': 8, 'n_steps': 3, 'gamma': 1.3},
        {'n_d': 16, 'n_a': 16, 'n_steps': 3, 'gamma': 1.3},
        {'n_d': 16, 'n_a': 16, 'n_steps': 5, 'gamma': 1.5},
        {'n_d': 32, 'n_a': 32, 'n_steps': 3, 'gamma': 1.3},
    ]
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_data(config):
    """Load and prepare train/test data for a model."""
    df_train = pd.read_csv(config['train'])
    df_test = pd.read_csv(config['test'])

    if config['binary_fix']:
        for var in BINARY_VARS:
            if var in df_train.columns:
                df_train[var] = df_train[var].astype(int)
            if var in df_test.columns:
                df_test[var] = df_test[var].astype(int)

    y_train = df_train['obeso'].copy()
    X_train = df_train.drop(TARGET_COLS + ['id_anon'], axis=1)
    y_test = df_test['obeso'].copy()
    X_test = df_test.drop(TARGET_COLS + ['id_anon'], axis=1)

    return X_train, y_train, X_test, y_test


def ci95(scores):
    """Calculate mean and 95% CI using t-distribution."""
    n = len(scores)
    if n <= 1:
        m = np.mean(scores)
        return m, m, m
    m = np.mean(scores)
    h = stats.sem(scores) * stats.t.ppf(0.975, n - 1)
    return m, m - h, m + h


def evaluate_sklearn_algorithm(X, y, alg_name, alg_config):
    """Evaluate a sklearn-compatible algorithm with nested CV."""
    pipeline = Pipeline([
        ('scaler', RobustScaler()),
        ('model', alg_config['model'])
    ])

    auc_scores, prec_scores, rec_scores, f1_scores = [], [], [], []

    for fold, (train_idx, val_idx) in enumerate(OUTER_CV.split(X, y)):
        X_train_fold = X.iloc[train_idx]
        y_train_fold = y.iloc[train_idx]
        X_val_fold = X.iloc[val_idx]
        y_val_fold = y.iloc[val_idx]

        grid = GridSearchCV(
            pipeline, alg_config['params'],
            cv=INNER_CV, scoring='roc_auc', n_jobs=-1, verbose=0
        )

        try:
            grid.fit(X_train_fold, y_train_fold)
            y_proba = grid.predict_proba(X_val_fold)[:, 1]
            y_pred = grid.predict(X_val_fold)

            auc_scores.append(roc_auc_score(y_val_fold, y_proba))
            prec_scores.append(precision_score(y_val_fold, y_pred, zero_division=0))
            rec_scores.append(recall_score(y_val_fold, y_pred, zero_division=0))
            f1_scores.append(f1_score(y_val_fold, y_pred, zero_division=0))
        except Exception as e:
            print(f"    Fold {fold} error ({alg_name}): {e}")
            auc_scores.append(0.5)
            prec_scores.append(0.0)
            rec_scores.append(0.0)
            f1_scores.append(0.0)

    auc_m, auc_lo, auc_hi = ci95(auc_scores)
    prec_m, prec_lo, prec_hi = ci95(prec_scores)
    rec_m, rec_lo, rec_hi = ci95(rec_scores)
    f1_m, f1_lo, f1_hi = ci95(f1_scores)

    return {
        'algorithm': alg_name,
        'auc': {'mean': auc_m, 'ci_lower': auc_lo, 'ci_upper': auc_hi},
        'precision': {'mean': prec_m, 'ci_lower': prec_lo, 'ci_upper': prec_hi},
        'recall': {'mean': rec_m, 'ci_lower': rec_lo, 'ci_upper': rec_hi},
        'f1': {'mean': f1_m, 'ci_lower': f1_lo, 'ci_upper': f1_hi},
    }


def evaluate_tabnet(X, y):
    """Evaluate TabNet with manual nested CV (not sklearn GridSearchCV compatible)."""
    from pytorch_tabnet.tab_model import TabNetClassifier

    auc_scores, prec_scores, rec_scores, f1_scores = [], [], [], []

    for fold, (train_idx, val_idx) in enumerate(OUTER_CV.split(X, y)):
        X_train_fold = X.iloc[train_idx].values
        y_train_fold = y.iloc[train_idx].values
        X_val_fold = X.iloc[val_idx].values
        y_val_fold = y.iloc[val_idx].values

        # Scale
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train_fold)
        X_val_scaled = scaler.transform(X_val_fold)

        # Inner CV to find best params
        best_inner_auc = -1
        best_params = TABNET_CONFIG['param_grid'][0]

        for params in TABNET_CONFIG['param_grid']:
            inner_aucs = []
            for inner_train_idx, inner_val_idx in INNER_CV.split(
                X_train_scaled, y_train_fold
            ):
                X_inner_train = X_train_scaled[inner_train_idx]
                y_inner_train = y_train_fold[inner_train_idx]
                X_inner_val = X_train_scaled[inner_val_idx]
                y_inner_val = y_train_fold[inner_val_idx]

                # Calculate class weights for TabNet
                n_pos = y_inner_train.sum()
                n_neg = len(y_inner_train) - n_pos
                weight_pos = n_neg / max(n_pos, 1)

                try:
                    model = TabNetClassifier(
                        n_d=params['n_d'],
                        n_a=params['n_a'],
                        n_steps=params['n_steps'],
                        gamma=params['gamma'],
                        seed=42,
                        verbose=0,
                    )
                    model.fit(
                        X_inner_train, y_inner_train,
                        eval_set=[(X_inner_val, y_inner_val)],
                        eval_metric=['auc'],
                        max_epochs=100,
                        patience=10,
                        batch_size=256,
                        weights={0: 1.0, 1: weight_pos},
                    )
                    y_proba = model.predict_proba(X_inner_val)[:, 1]
                    inner_aucs.append(roc_auc_score(y_inner_val, y_proba))
                except Exception:
                    inner_aucs.append(0.5)

            mean_inner = np.mean(inner_aucs)
            if mean_inner > best_inner_auc:
                best_inner_auc = mean_inner
                best_params = params

        # Train with best params on full outer training fold
        n_pos = y_train_fold.sum()
        n_neg = len(y_train_fold) - n_pos
        weight_pos = n_neg / max(n_pos, 1)

        try:
            model = TabNetClassifier(
                n_d=best_params['n_d'],
                n_a=best_params['n_a'],
                n_steps=best_params['n_steps'],
                gamma=best_params['gamma'],
                seed=42,
                verbose=0,
            )
            model.fit(
                X_train_scaled, y_train_fold,
                eval_set=[(X_val_scaled, y_val_fold)],
                eval_metric=['auc'],
                max_epochs=100,
                patience=10,
                batch_size=256,
                weights={0: 1.0, 1: weight_pos},
            )
            y_proba = model.predict_proba(X_val_scaled)[:, 1]
            y_pred = model.predict(X_val_scaled)

            auc_scores.append(roc_auc_score(y_val_fold, y_proba))
            prec_scores.append(precision_score(y_val_fold, y_pred, zero_division=0))
            rec_scores.append(recall_score(y_val_fold, y_pred, zero_division=0))
            f1_scores.append(f1_score(y_val_fold, y_pred, zero_division=0))
        except Exception as e:
            print(f"    TabNet fold {fold} error: {e}")
            auc_scores.append(0.5)
            prec_scores.append(0.0)
            rec_scores.append(0.0)
            f1_scores.append(0.0)

    auc_m, auc_lo, auc_hi = ci95(auc_scores)
    prec_m, prec_lo, prec_hi = ci95(prec_scores)
    rec_m, rec_lo, rec_hi = ci95(rec_scores)
    f1_m, f1_lo, f1_hi = ci95(f1_scores)

    return {
        'algorithm': 'TabNet',
        'auc': {'mean': auc_m, 'ci_lower': auc_lo, 'ci_upper': auc_hi},
        'precision': {'mean': prec_m, 'ci_lower': prec_lo, 'ci_upper': prec_hi},
        'recall': {'mean': rec_m, 'ci_lower': rec_lo, 'ci_upper': rec_hi},
        'f1': {'mean': f1_m, 'ci_lower': f1_lo, 'ci_upper': f1_hi},
    }


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    print("=" * 100)
    print("EXTENDED ALGORITHM EVALUATION: XGBoost, LightGBM, CatBoost, MLP, TabNet")
    print("Nested 5-fold stratified CV | 3 models | Same pipeline as original 8 algorithms")
    print("=" * 100)

    all_results = {}

    for model_key, config in MODEL_CONFIGS.items():
        print(f"\n{'=' * 80}")
        print(f"  {config['name'].upper()}")
        print(f"{'=' * 80}")

        X_train, y_train, X_test, y_test = load_data(config)
        print(f"  Train: {len(X_train):,} obs | {X_train.shape[1]} features | "
              f"obese={y_train.sum()} ({y_train.mean() * 100:.1f}%)")

        model_results = []

        # Sklearn-compatible algorithms
        for alg_name, alg_config in NEW_ALGORITHMS.items():
            print(f"    Running {alg_name}...", end=" ", flush=True)
            result = evaluate_sklearn_algorithm(X_train, y_train, alg_name, alg_config)
            result['model'] = model_key
            model_results.append(result)
            print(f"AUC={result['auc']['mean']:.3f} "
                  f"[{result['auc']['ci_lower']:.3f}-{result['auc']['ci_upper']:.3f}]")

        # TabNet (special handling)
        print(f"    Running TabNet...", end=" ", flush=True)
        tabnet_result = evaluate_tabnet(X_train, y_train)
        tabnet_result['model'] = model_key
        model_results.append(tabnet_result)
        print(f"AUC={tabnet_result['auc']['mean']:.3f} "
              f"[{tabnet_result['auc']['ci_lower']:.3f}-{tabnet_result['auc']['ci_upper']:.3f}]")

        all_results[model_key] = model_results

    # ==========================================================================
    # RESULTS SUMMARY
    # ==========================================================================
    print(f"\n{'=' * 100}")
    print("RESULTS SUMMARY - NEW ALGORITHMS")
    print(f"{'=' * 100}")

    for model_key in MODEL_CONFIGS:
        config = MODEL_CONFIGS[model_key]
        results = all_results[model_key]

        print(f"\n{config['name'].upper()}:")
        print(f"{'Algorithm':<15} {'AUC-ROC':<25} {'Precision':<25} "
              f"{'Recall':<25} {'F1-Score':<25}")
        print("-" * 115)

        for r in results:
            auc_str = (f"{r['auc']['mean']:.3f} "
                       f"[{r['auc']['ci_lower']:.3f}-{r['auc']['ci_upper']:.3f}]")
            prec_str = (f"{r['precision']['mean']:.3f} "
                        f"[{r['precision']['ci_lower']:.3f}-{r['precision']['ci_upper']:.3f}]")
            rec_str = (f"{r['recall']['mean']:.3f} "
                       f"[{r['recall']['ci_lower']:.3f}-{r['recall']['ci_upper']:.3f}]")
            f1_str = (f"{r['f1']['mean']:.3f} "
                      f"[{r['f1']['ci_lower']:.3f}-{r['f1']['ci_upper']:.3f}]")
            print(f"{r['algorithm']:<15} {auc_str:<25} {prec_str:<25} "
                  f"{rec_str:<25} {f1_str:<25}")

    # Save results to JSON for later use
    output_path = os.path.join(
        os.path.dirname(__file__), 'new_algorithms_results.json'
    )

    serializable = {}
    for model_key, results in all_results.items():
        serializable[model_key] = results

    with open(output_path, 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    # Overall ranking across all models
    print(f"\n{'=' * 80}")
    print("OVERALL RANKING - NEW ALGORITHMS (all models)")
    print(f"{'=' * 80}")

    flat = []
    for model_key, results in all_results.items():
        for r in results:
            flat.append(r)

    flat.sort(key=lambda x: x['auc']['mean'], reverse=True)

    print(f"{'Pos':<4} {'Algorithm':<15} {'Model':<10} {'AUC-ROC':<10} {'95% CI':<25}")
    print("-" * 65)
    for i, r in enumerate(flat, 1):
        ci = f"[{r['auc']['ci_lower']:.3f}-{r['auc']['ci_upper']:.3f}]"
        print(f"{i:<4} {r['algorithm']:<15} {r['model']:<10} "
              f"{r['auc']['mean']:<10.3f} {ci:<25}")

    # Comparison with original best results
    print(f"\n{'=' * 80}")
    print("COMPARISON WITH ORIGINAL 8 ALGORITHMS (best per model)")
    print(f"{'=' * 80}")

    original_best = {
        'Model 1': {'algorithm': 'Random Forest', 'auc': 0.570},
        'Model 2': {'algorithm': 'Logistic Regression', 'auc': 0.598},
        'Model 3': {'algorithm': 'Gradient Boosting', 'auc': 0.571},
    }

    for model_key in MODEL_CONFIGS:
        results = all_results[model_key]
        best_new = max(results, key=lambda x: x['auc']['mean'])
        orig = original_best[model_key]
        delta = best_new['auc']['mean'] - orig['auc']

        print(f"\n  {model_key}:")
        print(f"    Original best: {orig['algorithm']} (AUC={orig['auc']:.3f})")
        print(f"    New best:      {best_new['algorithm']} "
              f"(AUC={best_new['auc']['mean']:.3f} "
              f"[{best_new['auc']['ci_lower']:.3f}-{best_new['auc']['ci_upper']:.3f}])")
        print(f"    Delta:         {delta:+.3f}")

    print(f"\n{'=' * 80}")
    print("CONCLUSION")
    print(f"{'=' * 80}")
    print("If all new algorithms remain in the AUC 0.50-0.63 range,")
    print("this confirms the performance ceiling is informational (insufficient")
    print("discriminative signal) rather than algorithmic — even state-of-the-art")
    print("gradient boosting (XGBoost, LightGBM, CatBoost) and deep learning")
    print("(MLP, TabNet) cannot overcome the fundamental limitation of")
    print("cross-sectional early-life features for obesity prediction.")


if __name__ == "__main__":
    main()
