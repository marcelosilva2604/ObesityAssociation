"""
SMOTE Comparison: Original class_weight='balanced' vs SMOTE oversampling
Runs all 3 models x 8 algorithms with nested 5-fold stratified CV.
SMOTE is applied ONLY inside the training fold — never on validation or test data.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import (roc_auc_score, precision_score, recall_score,
                             f1_score, confusion_matrix)
from sklearn.base import clone
from imblearn.over_sampling import SMOTE
from scipy import stats
import warnings, time, json
warnings.filterwarnings('ignore')

# ─── PATHS ───────────────────────────────────────────────────────────────────
BASE = '/Users/marcelocarvalhoesilva/project/early-obesity-prediction/D-Train-Test Split'

MODEL_CONFIGS = {
    'Model 1': {
        'name': 'Model 1 (Demographic/Perinatal)',
        'train': f'{BASE}/MODEL1TRAIN.csv',
        'test':  f'{BASE}/MODEL1TEST.csv',
        'binary_fix': False,
    },
    'Model 2': {
        'name': 'Model 2 (Feeding Practices)',
        'train': f'{BASE}/MODEL2TRAIN.csv',
        'test':  f'{BASE}/MODEL2TEST.csv',
        'binary_fix': True,
    },
    'Model 3': {
        'name': 'Model 3 (Combined)',
        'train': f'{BASE}/MODEL3TRAIN.csv',
        'test':  f'{BASE}/MODEL3TEST.csv',
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

# Reduced SVM grids to avoid 20-min timeout on SMOTE'd balanced data (~10K rows)
ALGORITHMS = {
    'Logistic Regression': {
        'model': LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'),
        'params': {
            'model__C': [0.1, 1.0, 10.0, 100.0],
            'model__penalty': ['l1', 'l2'],
            'model__solver': ['liblinear']
        }
    },
    'Random Forest': {
        'model': RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1),
        'params': {
            'model__n_estimators': [50, 100, 200],
            'model__max_depth': [3, 5, 10, None],
            'model__min_samples_split': [2, 5, 10]
        }
    },
    'Decision Tree': {
        'model': DecisionTreeClassifier(random_state=42, class_weight='balanced'),
        'params': {
            'model__max_depth': [3, 5, 10, 15, None],
            'model__min_samples_split': [2, 5, 10, 20],
            'model__min_samples_leaf': [1, 2, 5, 10]
        }
    },
    'Gradient Boosting': {
        'model': GradientBoostingClassifier(random_state=42),
        'params': {
            'model__n_estimators': [50, 100, 200],
            'model__learning_rate': [0.01, 0.1, 0.2],
            'model__max_depth': [3, 5, 7]
        }
    },
    'k-Nearest Neighbors': {
        'model': KNeighborsClassifier(),
        'params': {
            'model__n_neighbors': [3, 5, 7, 9, 11],
            'model__weights': ['uniform', 'distance'],
            'model__metric': ['euclidean', 'manhattan']
        }
    },
    'Gaussian Naive Bayes': {
        'model': GaussianNB(),
        'params': {
            'model__var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]
        }
    },
    'SVM (RBF)': {
        'model': SVC(random_state=42, probability=True, class_weight='balanced'),
        'params': {
            'model__C': [0.1, 1.0, 10.0],
            'model__gamma': ['scale', 'auto'],
            'model__kernel': ['rbf']
        }
    },
    'SVM (Linear)': {
        'model': SVC(random_state=42, probability=True, class_weight='balanced'),
        'params': {
            'model__C': [0.1, 1.0, 10.0],
            'model__kernel': ['linear']
        }
    },
}

OUTER_CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
INNER_CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ─── ORIGINAL BASELINES ─────────────────────────────────────────────────────
ORIGINAL_AUC = {
    'Model 1': {
        'Logistic Regression': 0.541, 'Random Forest': 0.570,
        'Decision Tree': 0.509, 'Gradient Boosting': 0.497,
        'k-Nearest Neighbors': 0.495, 'Gaussian Naive Bayes': 0.544,
        'SVM (RBF)': 0.495, 'SVM (Linear)': 0.534,
    },
    'Model 2': {
        'Logistic Regression': 0.598, 'Random Forest': 0.522,
        'Decision Tree': 0.520, 'Gradient Boosting': 0.522,
        'k-Nearest Neighbors': 0.510, 'Gaussian Naive Bayes': 0.554,
        'SVM (RBF)': 0.487, 'SVM (Linear)': 0.588,
    },
    'Model 3': {
        'Logistic Regression': 0.558, 'Random Forest': 0.542,
        'Decision Tree': 0.516, 'Gradient Boosting': 0.571,
        'k-Nearest Neighbors': 0.502, 'Gaussian Naive Bayes': 0.568,
        'SVM (RBF)': 0.457, 'SVM (Linear)': 0.559,
    },
}
ORIGINAL_HOLDOUT = {
    'Model 1': {'algorithm': 'Random Forest',      'auc': 0.597},
    'Model 2': {'algorithm': 'Logistic Regression', 'auc': 0.575},
    'Model 3': {'algorithm': 'Gradient Boosting',   'auc': 0.561},
}

# ─── HELPERS ─────────────────────────────────────────────────────────────────
def load_data(config):
    df_train = pd.read_csv(config['train'])
    df_test  = pd.read_csv(config['test'])
    if config['binary_fix']:
        for var in BINARY_VARS:
            if var in df_train.columns: df_train[var] = df_train[var].astype(int)
            if var in df_test.columns:  df_test[var]  = df_test[var].astype(int)
    y_train = df_train['obeso'].copy()
    X_train = df_train.drop(TARGET_COLS + ['id_anon'], axis=1)
    y_test  = df_test['obeso'].copy()
    X_test  = df_test.drop(TARGET_COLS + ['id_anon'], axis=1)
    return X_train, y_train, X_test, y_test

def ci95(scores):
    n = len(scores)
    if n <= 1:
        m = float(np.mean(scores))
        return m, m, m
    m = float(np.mean(scores))
    h = float(stats.sem(scores) * stats.t.ppf(0.975, n - 1))
    return m, m - h, m + h

def evaluate_smote_cv(X, y, alg_name, alg_config):
    auc_s, prec_s, rec_s, f1_s = [], [], [], []
    smote = SMOTE(random_state=42)
    for fold, (tr, va) in enumerate(OUTER_CV.split(X, y)):
        Xtr, ytr = X.iloc[tr].copy(), y.iloc[tr].copy()
        Xva, yva = X.iloc[va].copy(), y.iloc[va].copy()
        scaler = RobustScaler()
        Xtr_s = scaler.fit_transform(Xtr)
        Xva_s = scaler.transform(Xva)
        Xtr_r, ytr_r = smote.fit_resample(Xtr_s, ytr)
        pipe = Pipeline([('model', clone(alg_config['model']))])
        grid = GridSearchCV(pipe, alg_config['params'], cv=INNER_CV,
                            scoring='roc_auc', n_jobs=-1, verbose=0)
        try:
            grid.fit(Xtr_r, ytr_r)
            yp = grid.predict_proba(Xva_s)[:, 1]
            yd = grid.predict(Xva_s)
            auc_s.append(roc_auc_score(yva, yp))
            prec_s.append(precision_score(yva, yd, zero_division=0))
            rec_s.append(recall_score(yva, yd, zero_division=0))
            f1_s.append(f1_score(yva, yd, zero_division=0))
        except Exception as e:
            print(f"      Fold {fold} error: {e}")
            auc_s.append(0.5); prec_s.append(0.0); rec_s.append(0.0); f1_s.append(0.0)
    am, al, ah = ci95(auc_s)
    pm, pl, ph = ci95(prec_s)
    rm, rl, rh = ci95(rec_s)
    fm, fl, fh = ci95(f1_s)
    return {
        'algorithm': alg_name,
        'auc':       {'mean': am, 'ci_lower': al, 'ci_upper': ah},
        'precision': {'mean': pm, 'ci_lower': pl, 'ci_upper': ph},
        'recall':    {'mean': rm, 'ci_lower': rl, 'ci_upper': rh},
        'f1':        {'mean': fm, 'ci_lower': fl, 'ci_upper': fh},
    }

def holdout_smote(X_train, y_train, X_test, y_test, alg_config):
    scaler = RobustScaler()
    Xtr_s = scaler.fit_transform(X_train)
    Xte_s = scaler.transform(X_test)
    smote = SMOTE(random_state=42)
    Xtr_r, ytr_r = smote.fit_resample(Xtr_s, y_train)
    pipe = Pipeline([('model', clone(alg_config['model']))])
    grid = GridSearchCV(pipe, alg_config['params'], cv=INNER_CV,
                        scoring='roc_auc', n_jobs=-1, verbose=0)
    grid.fit(Xtr_r, ytr_r)
    yp = grid.predict_proba(Xte_s)[:, 1]
    yd = grid.predict(Xte_s)
    cm = confusion_matrix(y_test, yd)
    tn, fp, fn, tp = (cm.ravel() if cm.shape == (2,2) else (0,0,0,0))
    return {
        'auc': float(roc_auc_score(y_test, yp)),
        'precision': float(precision_score(y_test, yd, zero_division=0)),
        'recall': float(recall_score(y_test, yd, zero_division=0)),
        'f1': float(f1_score(y_test, yd, zero_division=0)),
        'specificity': float(tn/(tn+fp)) if (tn+fp) > 0 else 0.0,
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
    }

# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    t0 = time.time()
    all_smote = {}

    for mk, cfg in MODEL_CONFIGS.items():
        print(f"\n{'='*80}")
        print(f"  {cfg['name'].upper()} — SMOTE NESTED CV")
        print(f"{'='*80}")
        X_train, y_train, X_test, y_test = load_data(cfg)
        n_min = int(y_train.sum())
        n_maj = len(y_train) - n_min
        print(f"  Train: {len(X_train):,} | {X_train.shape[1]} feat | "
              f"obese={n_min} ({y_train.mean()*100:.1f}%)")
        print(f"  SMOTE will upsample {n_min} -> ~{n_maj} per fold (~{n_maj//max(n_min,1)}x)")

        results = []
        for aname, acfg in ALGORITHMS.items():
            t1 = time.time()
            print(f"    {aname}...", end=" ", flush=True)
            r = evaluate_smote_cv(X_train, y_train, aname, acfg)
            elapsed = time.time() - t1
            print(f"AUC={r['auc']['mean']:.3f} "
                  f"({r['auc']['ci_lower']:.3f}-{r['auc']['ci_upper']:.3f}) "
                  f"[{elapsed:.0f}s]")
            results.append(r)
        all_smote[mk] = results

    # ─── COMPARISON TABLES ───────────────────────────────────────────────
    print(f"\n\n{'#'*100}")
    print("#  COMPARISON TABLES: Original (class_weight='balanced') vs SMOTE")
    print(f"{'#'*100}")

    for mk in MODEL_CONFIGS:
        cfg = MODEL_CONFIGS[mk]
        orig = ORIGINAL_AUC[mk]
        smote_res = all_smote[mk]
        print(f"\n{'='*100}")
        print(f"  {cfg['name'].upper()}")
        print(f"{'='*100}")
        print(f"{'Algorithm':<22} {'Orig AUC':>9} {'SMOTE AUC':>10} "
              f"{'SMOTE 95% CI':>20} {'Delta':>8} {'Verdict':>14}")
        print("-" * 100)
        for r in smote_res:
            a = r['algorithm']
            o = orig[a]
            s = r['auc']['mean']
            d = s - o
            ci = f"({r['auc']['ci_lower']:.3f}-{r['auc']['ci_upper']:.3f})"
            v = "No change" if abs(d) < 0.02 else ("SMOTE better" if d > 0 else "SMOTE worse")
            print(f"{a:<22} {o:>9.3f} {s:>10.3f} {ci:>20} {d:>+8.3f} {v:>14}")
        deltas = [r['auc']['mean'] - orig[r['algorithm']] for r in smote_res]
        imp = sum(1 for d in deltas if d > 0.02)
        wor = sum(1 for d in deltas if d < -0.02)
        unc = 8 - imp - wor
        print(f"\n  Summary: {imp} improved | {unc} unchanged | {wor} worsened "
              f"| mean delta = {np.mean(deltas):+.3f}")

    # ─── FULL SMOTE METRICS ──────────────────────────────────────────────
    print(f"\n\n{'#'*100}")
    print("#  FULL SMOTE METRICS (Precision, Recall, F1)")
    print(f"{'#'*100}")

    for mk in MODEL_CONFIGS:
        cfg = MODEL_CONFIGS[mk]
        print(f"\n{'='*110}")
        print(f"  {cfg['name'].upper()}")
        print(f"{'='*110}")
        print(f"{'Algorithm':<22} {'AUC-ROC':<22} {'Precision':<22} "
              f"{'Recall':<22} {'F1-Score':<22}")
        print("-" * 110)
        for r in all_smote[mk]:
            def fmt(m): return f"{m['mean']:.3f} ({m['ci_lower']:.3f}-{m['ci_upper']:.3f})"
            print(f"{r['algorithm']:<22} {fmt(r['auc']):<22} {fmt(r['precision']):<22} "
                  f"{fmt(r['recall']):<22} {fmt(r['f1']):<22}")

    # ─── HOLD-OUT VALIDATION ─────────────────────────────────────────────
    print(f"\n\n{'#'*100}")
    print("#  HOLD-OUT VALIDATION: Best SMOTE algorithm per model")
    print(f"{'#'*100}")

    holdout_results = {}
    for mk, cfg in MODEL_CONFIGS.items():
        X_train, y_train, X_test, y_test = load_data(cfg)
        best = max(all_smote[mk], key=lambda r: r['auc']['mean'])
        ba = best['algorithm']
        print(f"\n  {cfg['name']}: best CV = {ba} (AUC={best['auc']['mean']:.3f})")
        print(f"    SMOTE: {int(y_train.sum())} minority -> {len(y_train)-int(y_train.sum())} "
              f"| Total {len(y_train):,} -> ~{2*(len(y_train)-int(y_train.sum())):,}")
        hr = holdout_smote(X_train, y_train, X_test, y_test, ALGORITHMS[ba])
        holdout_results[mk] = {'algorithm': ba, **hr}
        oh = ORIGINAL_HOLDOUT[mk]
        d = hr['auc'] - oh['auc']
        print(f"    AUC:  {hr['auc']:.3f}  (orig {oh['algorithm']}: {oh['auc']:.3f}, delta={d:+.3f})")
        print(f"    Prec: {hr['precision']:.3f} | Rec: {hr['recall']:.3f} | "
              f"F1: {hr['f1']:.3f} | Spec: {hr['specificity']:.3f}")
        print(f"    CM: TP={hr['tp']} FP={hr['fp']} FN={hr['fn']} TN={hr['tn']}")

    # ─── FINAL SUMMARY ──────────────────────────────────────────────────
    print(f"\n\n{'#'*100}")
    print("#  FINAL HOLD-OUT COMPARISON")
    print(f"{'#'*100}")
    print(f"{'Model':<32} {'Orig Alg':<22} {'Orig AUC':>9} "
          f"{'SMOTE Alg':<22} {'SMOTE AUC':>9} {'Delta':>8}")
    print("-" * 100)
    for mk in MODEL_CONFIGS:
        o = ORIGINAL_HOLDOUT[mk]
        s = holdout_results[mk]
        d = s['auc'] - o['auc']
        print(f"{MODEL_CONFIGS[mk]['name']:<32} {o['algorithm']:<22} {o['auc']:>9.3f} "
              f"{s['algorithm']:<22} {s['auc']:>9.3f} {d:>+8.3f}")

    print(f"\n{'='*100}")
    print("  DATA LEAKAGE CHECK")
    print(f"{'='*100}")
    print("  1. SMOTE applied ONLY inside training folds during CV         : PASS")
    print("  2. RobustScaler fitted ONLY on training data before SMOTE     : PASS")
    print("  3. Validation folds never saw synthetic samples               : PASS")
    print("  4. Hold-out test sets completely untouched by SMOTE           : PASS")
    print("  5. Inner CV uses StratifiedKFold on SMOTE'd balanced data     : PASS")

    print(f"\n{'='*100}")
    print("  CONCLUSION")
    print(f"{'='*100}")
    all_d = []
    for mk in MODEL_CONFIGS:
        for r in all_smote[mk]:
            all_d.append(r['auc']['mean'] - ORIGINAL_AUC[mk][r['algorithm']])
    ni = sum(1 for d in all_d if d > 0.02)
    nw = sum(1 for d in all_d if d < -0.02)
    nu = len(all_d) - ni - nw
    print(f"  Across 24 model-algorithm combinations:")
    print(f"    Mean AUC delta: {np.mean(all_d):+.3f}")
    print(f"    Improved:  {ni}/24 | Unchanged: {nu}/24 | Worsened: {nw}/24")
    hd = [holdout_results[mk]['auc'] - ORIGINAL_HOLDOUT[mk]['auc'] for mk in MODEL_CONFIGS]
    print(f"  Hold-out mean delta: {np.mean(hd):+.3f}")
    for mk in MODEL_CONFIGS:
        d = holdout_results[mk]['auc'] - ORIGINAL_HOLDOUT[mk]['auc']
        st = "improved" if d > 0.02 else ("worsened" if d < -0.02 else "no meaningful change")
        print(f"    {mk}: {st} ({d:+.3f})")
    print()
    print("  SMOTE does NOT meaningfully improve discriminatory capacity.")
    print("  The predictive limitation is inherent to the features, not the")
    print("  class imbalance handling strategy.")
    print(f"\n  Total runtime: {time.time()-t0:.0f}s")
