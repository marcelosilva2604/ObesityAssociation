#!/usr/bin/env python3
"""
Run ONE algorithm at a time across all 3 models with controlled parallelism.

Memory-safe design after prior crash:
  - GridSearchCV n_jobs=1 for boosters (they parallelize internally with 4 threads).
  - GridSearchCV n_jobs=4 for MLP (sklearn MLP is single-threaded internally).
  - TabNet uses torch.set_num_threads(4), CPU only, instances deleted between fits.
  - gc.collect() between every fold and between models.
  - Results saved INCREMENTALLY to new_algorithms_results.json after each algorithm,
    so an interrupted run keeps prior algorithms.

Usage:
    python run_algorithm.py --alg XGBoost
    python run_algorithm.py --alg LightGBM
    python run_algorithm.py --alg CatBoost
    python run_algorithm.py --alg MLP
    python run_algorithm.py --alg TabNet
"""

import argparse
import gc
import json
import os
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings("ignore")

# Cap BLAS / OpenMP threads at the process level too (defensive).
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

# =============================================================================
# CONFIG
# =============================================================================
BASE = "/Users/marcelosilva/project/early-obesity-prediction/D-Train-Test Split"
RESULTS_PATH = os.path.join(
    os.path.dirname(__file__), "new_algorithms_results.json"
)

MODEL_CONFIGS = {
    "Model 1": {
        "name": "Model 1 (Demographic/Perinatal)",
        "train": f"{BASE}/MODEL1TRAIN.csv",
        "binary_fix": False,
    },
    "Model 2": {
        "name": "Model 2 (Feeding Practices)",
        "train": f"{BASE}/MODEL2TRAIN.csv",
        "binary_fix": True,
    },
    "Model 3": {
        "name": "Model 3 (Combined)",
        "train": f"{BASE}/MODEL3TRAIN.csv",
        "binary_fix": True,
    },
}

BINARY_VARS = [
    "doou_leite_banco", "recebeu_leite_banco", "amamentou_outra_crianca",
    "usou_mamadeira", "deixou_amamentar_por_outra", "busca_info_aleitamento",
    "deu_outros_liquidos", "k15_recebeu", "k11_amamentou", "k03_prenatal",
    "usou_utensilios_amamentacao",
]
TARGET_COLS = ["desnutrido", "eutrofico", "sobrepeso", "obeso"]

OUTER_CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
INNER_CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
SCALE_POS = 33.0  # ~33:1 negative:positive ratio for obesity


# =============================================================================
# DATA
# =============================================================================
def load_data(config):
    df = pd.read_csv(config["train"])
    if config["binary_fix"]:
        for var in BINARY_VARS:
            if var in df.columns:
                df[var] = df[var].astype(int)
    y = df["obeso"].copy()
    X = df.drop(TARGET_COLS + ["id_anon"], axis=1)
    return X, y


def ci95(scores):
    n = len(scores)
    if n <= 1:
        m = float(np.mean(scores))
        return m, m, m
    m = float(np.mean(scores))
    h = float(stats.sem(scores) * stats.t.ppf(0.975, n - 1))
    return m, m - h, m + h


# =============================================================================
# ALGORITHM FACTORIES (lazy imports — only the chosen one is loaded)
# =============================================================================
def get_sklearn_alg(name):
    """Return (sklearn_estimator, param_grid, gridsearch_n_jobs)."""
    if name == "XGBoost":
        from xgboost import XGBClassifier
        model = XGBClassifier(
            random_state=42,
            scale_pos_weight=SCALE_POS,
            eval_metric="logloss",
            verbosity=0,
            n_jobs=4,  # internal parallelism
            tree_method="hist",  # memory-efficient
        )
        params = {
            "model__n_estimators": [50, 100, 200],
            "model__max_depth": [3, 5, 7],
            "model__learning_rate": [0.01, 0.1, 0.2],
        }
        return model, params, 1  # GridSearch n_jobs=1, internal=4

    if name == "LightGBM":
        from lightgbm import LGBMClassifier
        model = LGBMClassifier(
            random_state=42,
            scale_pos_weight=SCALE_POS,
            verbose=-1,
            n_jobs=4,
        )
        params = {
            "model__n_estimators": [50, 100, 200],
            "model__max_depth": [3, 5, 7, -1],
            "model__learning_rate": [0.01, 0.1, 0.2],
        }
        return model, params, 1

    if name == "CatBoost":
        from catboost import CatBoostClassifier
        model = CatBoostClassifier(
            random_state=42,
            auto_class_weights="Balanced",
            verbose=0,
            thread_count=4,  # internal parallelism cap
            allow_writing_files=False,
        )
        params = {
            "model__iterations": [50, 100, 200],
            "model__depth": [3, 5, 7],
            "model__learning_rate": [0.01, 0.1, 0.2],
        }
        return model, params, 1

    if name == "MLP":
        from sklearn.neural_network import MLPClassifier
        model = MLPClassifier(
            random_state=42,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=10,
        )
        # Trimmed grid (was 30 combos → 12) — MLP is single-threaded.
        # GridSearch n_jobs=1 after repeated crashes: loky fork + BLAS state
        # segfaults sklearn MLP on macOS. BLAS still parallelizes inside each fit.
        params = {
            "model__hidden_layer_sizes": [(64,), (128,), (64, 32)],
            "model__activation": ["relu", "tanh"],
            "model__alpha": [0.0001, 0.01],
            "model__learning_rate": ["adaptive"],
        }
        return model, params, 1

    raise ValueError(f"Unknown sklearn algorithm: {name}")


def evaluate_sklearn(X, y, alg_name):
    model, param_grid, gs_njobs = get_sklearn_alg(alg_name)
    pipeline = Pipeline([("scaler", RobustScaler()), ("model", model)])

    aucs, precs, recs, f1s = [], [], [], []

    for fold, (tr_idx, va_idx) in enumerate(OUTER_CV.split(X, y), 1):
        X_tr = X.iloc[tr_idx]
        y_tr = y.iloc[tr_idx]
        X_va = X.iloc[va_idx]
        y_va = y.iloc[va_idx]

        grid = GridSearchCV(
            pipeline, param_grid,
            cv=INNER_CV, scoring="roc_auc",
            n_jobs=gs_njobs, verbose=0,
        )
        grid.fit(X_tr, y_tr)

        y_proba = grid.predict_proba(X_va)[:, 1]
        y_pred = grid.predict(X_va)

        a = roc_auc_score(y_va, y_proba)
        p = precision_score(y_va, y_pred, zero_division=0)
        r = recall_score(y_va, y_pred, zero_division=0)
        f = f1_score(y_va, y_pred, zero_division=0)

        aucs.append(a); precs.append(p); recs.append(r); f1s.append(f)
        print(f"      fold {fold}/5 AUC={a:.3f} P={p:.3f} R={r:.3f} F1={f:.3f}",
              flush=True)

        del grid, X_tr, y_tr, X_va, y_va
        gc.collect()

    return _summarize(alg_name, aucs, precs, recs, f1s)


def evaluate_tabnet(X, y):
    """TabNet with manual CV, controlled torch threads, explicit cleanup."""
    import torch
    from pytorch_tabnet.tab_model import TabNetClassifier

    torch.set_num_threads(4)

    param_grid = [
        {"n_d": 8,  "n_a": 8,  "n_steps": 3, "gamma": 1.3},
        {"n_d": 16, "n_a": 16, "n_steps": 3, "gamma": 1.3},
        {"n_d": 16, "n_a": 16, "n_steps": 5, "gamma": 1.5},
        {"n_d": 32, "n_a": 32, "n_steps": 3, "gamma": 1.3},
    ]

    aucs, precs, recs, f1s = [], [], [], []

    for fold, (tr_idx, va_idx) in enumerate(OUTER_CV.split(X, y), 1):
        X_tr = X.iloc[tr_idx].values
        y_tr = y.iloc[tr_idx].values
        X_va = X.iloc[va_idx].values
        y_va = y.iloc[va_idx].values

        scaler = RobustScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_va_s = scaler.transform(X_va)

        # Inner CV for hyperparameter search
        best_auc, best_params = -1.0, param_grid[0]
        for params in param_grid:
            inner_aucs = []
            for in_tr, in_va in INNER_CV.split(X_tr_s, y_tr):
                Xi_tr, yi_tr = X_tr_s[in_tr], y_tr[in_tr]
                Xi_va, yi_va = X_tr_s[in_va], y_tr[in_va]
                w_pos = (len(yi_tr) - yi_tr.sum()) / max(yi_tr.sum(), 1)

                m = TabNetClassifier(
                    n_d=params["n_d"], n_a=params["n_a"],
                    n_steps=params["n_steps"], gamma=params["gamma"],
                    seed=42, verbose=0, device_name="cpu",
                )
                m.fit(
                    Xi_tr, yi_tr,
                    eval_set=[(Xi_va, yi_va)], eval_metric=["auc"],
                    max_epochs=80, patience=10, batch_size=256,
                    weights={0: 1.0, 1: float(w_pos)},
                )
                proba = m.predict_proba(Xi_va)[:, 1]
                inner_aucs.append(roc_auc_score(yi_va, proba))

                del m
                gc.collect()

            mean_in = float(np.mean(inner_aucs))
            if mean_in > best_auc:
                best_auc, best_params = mean_in, params

        # Refit on full outer train fold with best params
        w_pos = (len(y_tr) - y_tr.sum()) / max(y_tr.sum(), 1)
        m = TabNetClassifier(
            n_d=best_params["n_d"], n_a=best_params["n_a"],
            n_steps=best_params["n_steps"], gamma=best_params["gamma"],
            seed=42, verbose=0, device_name="cpu",
        )
        m.fit(
            X_tr_s, y_tr,
            eval_set=[(X_va_s, y_va)], eval_metric=["auc"],
            max_epochs=80, patience=10, batch_size=256,
            weights={0: 1.0, 1: float(w_pos)},
        )
        proba = m.predict_proba(X_va_s)[:, 1]
        pred = m.predict(X_va_s)

        a = roc_auc_score(y_va, proba)
        p = precision_score(y_va, pred, zero_division=0)
        r = recall_score(y_va, pred, zero_division=0)
        f = f1_score(y_va, pred, zero_division=0)
        aucs.append(a); precs.append(p); recs.append(r); f1s.append(f)
        print(f"      fold {fold}/5 AUC={a:.3f} P={p:.3f} R={r:.3f} F1={f:.3f} "
              f"best={best_params}", flush=True)

        del m, scaler, X_tr_s, X_va_s, X_tr, X_va, y_tr, y_va
        gc.collect()

    return _summarize("TabNet", aucs, precs, recs, f1s)


def _summarize(name, aucs, precs, recs, f1s):
    am, alo, ahi = ci95(aucs)
    pm, plo, phi = ci95(precs)
    rm, rlo, rhi = ci95(recs)
    fm, flo, fhi = ci95(f1s)
    return {
        "algorithm": name,
        "auc":       {"mean": am, "ci_lower": alo, "ci_upper": ahi},
        "precision": {"mean": pm, "ci_lower": plo, "ci_upper": phi},
        "recall":    {"mean": rm, "ci_lower": rlo, "ci_upper": rhi},
        "f1":        {"mean": fm, "ci_lower": flo, "ci_upper": fhi},
        "fold_aucs": [float(x) for x in aucs],
    }


# =============================================================================
# INCREMENTAL RESULTS PERSISTENCE
# =============================================================================
def load_results():
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            return json.load(f)
    return {}


def save_results(results):
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)


# =============================================================================
# MAIN
# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--alg", required=True,
        choices=["XGBoost", "LightGBM", "CatBoost", "MLP", "TabNet"],
    )
    args = ap.parse_args()
    alg = args.alg

    print("=" * 90)
    print(f"  ALGORITHM: {alg}")
    print("=" * 90)

    all_results = load_results()

    for model_key, config in MODEL_CONFIGS.items():
        print(f"\n  >>> {config['name']}")
        X, y = load_data(config)
        print(f"      n={len(X):,}  features={X.shape[1]}  "
              f"obese={int(y.sum())} ({y.mean() * 100:.1f}%)")

        if alg == "TabNet":
            res = evaluate_tabnet(X, y)
        else:
            res = evaluate_sklearn(X, y, alg)
        res["model"] = model_key

        # Merge into results: replace any prior entry for this (model, alg)
        prior = all_results.get(model_key, [])
        prior = [r for r in prior if r.get("algorithm") != alg]
        prior.append(res)
        all_results[model_key] = prior
        save_results(all_results)

        a = res["auc"]
        print(f"      → {alg} AUC={a['mean']:.3f} "
              f"[{a['ci_lower']:.3f}-{a['ci_upper']:.3f}]")

        del X, y
        gc.collect()

    # Summary for this algorithm
    print(f"\n{'=' * 90}")
    print(f"  {alg} — SUMMARY")
    print(f"{'=' * 90}")
    print(f"{'Model':<10} {'AUC-ROC':<22} {'Precision':<22} "
          f"{'Recall':<22} {'F1':<22}")
    print("-" * 100)
    for model_key in MODEL_CONFIGS:
        r = next(x for x in all_results[model_key] if x["algorithm"] == alg)
        def s(m): return (f"{r[m]['mean']:.3f} "
                          f"[{r[m]['ci_lower']:.3f}-{r[m]['ci_upper']:.3f}]")
        print(f"{model_key:<10} {s('auc'):<22} {s('precision'):<22} "
              f"{s('recall'):<22} {s('f1'):<22}")

    print(f"\n  Saved → {RESULTS_PATH}")


if __name__ == "__main__":
    main()
