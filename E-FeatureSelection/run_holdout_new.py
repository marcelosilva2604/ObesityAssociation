#!/usr/bin/env python3
"""
Hold-out bootstrap (n=1,000) for the new algorithms whose CV 95% CI excluded 0.5:
TabNet on Models 1/2/3 and XGBoost on Model 3.

For each case:
  1. Fit the algorithm on the full training set with its grid-searched best hyperparameters
     (quick inner 5-fold CV for hyperparameter selection, matching run_algorithm.py).
  2. Predict probabilities on the held-out test set.
  3. Resample (y_test, y_proba) pairs 1,000 times with replacement; recompute AUC,
     precision, recall, F1 each time.
  4. Report mean and 95% percentile CI per metric.

Results are appended to holdout_new_algorithms_results.json next to this script.
"""

import gc
import json
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings("ignore")

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

BASE = "/Users/marcelosilva/project/early-obesity-prediction/D-Train-Test Split"
RESULTS_PATH = os.path.join(
    os.path.dirname(__file__), "holdout_new_algorithms_results.json"
)

BINARY_VARS = [
    "doou_leite_banco", "recebeu_leite_banco", "amamentou_outra_crianca",
    "usou_mamadeira", "deixou_amamentar_por_outra", "busca_info_aleitamento",
    "deu_outros_liquidos", "k15_recebeu", "k11_amamentou", "k03_prenatal",
    "usou_utensilios_amamentacao",
]
TARGET_COLS = ["desnutrido", "eutrofico", "sobrepeso", "obeso"]

MODEL_CONFIGS = {
    "Model 1": {
        "train": f"{BASE}/MODEL1TRAIN.csv",
        "test":  f"{BASE}/MODEL1TEST.csv",
        "binary_fix": False,
    },
    "Model 2": {
        "train": f"{BASE}/MODEL2TRAIN.csv",
        "test":  f"{BASE}/MODEL2TEST.csv",
        "binary_fix": True,
    },
    "Model 3": {
        "train": f"{BASE}/MODEL3TRAIN.csv",
        "test":  f"{BASE}/MODEL3TEST.csv",
        "binary_fix": True,
    },
}

INNER_CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
SCALE_POS = 33.0
N_BOOT = 1000
RNG = np.random.RandomState(42)


def load_data(config):
    df_tr = pd.read_csv(config["train"])
    df_te = pd.read_csv(config["test"])
    if config["binary_fix"]:
        for var in BINARY_VARS:
            if var in df_tr.columns:
                df_tr[var] = df_tr[var].astype(int)
            if var in df_te.columns:
                df_te[var] = df_te[var].astype(int)
    y_tr = df_tr["obeso"].copy()
    X_tr = df_tr.drop(TARGET_COLS + ["id_anon"], axis=1)
    y_te = df_te["obeso"].copy()
    X_te = df_te.drop(TARGET_COLS + ["id_anon"], axis=1)
    return X_tr, y_tr, X_te, y_te


def bootstrap_metrics(y_true, y_proba, y_pred, n_boot=N_BOOT):
    n = len(y_true)
    aucs, precs, recs, f1s = [], [], [], []
    specs, accs, npvs = [], [], []
    for _ in range(n_boot):
        idx = RNG.randint(0, n, n)
        yt = y_true[idx]
        yp = y_proba[idx]
        yh = y_pred[idx]
        if yt.sum() == 0 or yt.sum() == len(yt):
            continue
        aucs.append(roc_auc_score(yt, yp))
        precs.append(precision_score(yt, yh, zero_division=0))
        recs.append(recall_score(yt, yh, zero_division=0))
        f1s.append(f1_score(yt, yh, zero_division=0))

        tn, fp, fn, tp = confusion_matrix(yt, yh, labels=[0, 1]).ravel()
        specs.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)
        accs.append((tp + tn) / len(yt))
        npvs.append(tn / (tn + fn) if (tn + fn) > 0 else 0.0)

    def ci(arr):
        a = np.asarray(arr)
        return {
            "mean": float(a.mean()),
            "ci_lower": float(np.percentile(a, 2.5)),
            "ci_upper": float(np.percentile(a, 97.5)),
        }

    return {
        "auc": ci(aucs),
        "precision": ci(precs),
        "recall": ci(recs),
        "f1": ci(f1s),
        "specificity": ci(specs),
        "accuracy": ci(accs),
        "npv": ci(npvs),
        "n_boot_effective": len(aucs),
    }


def fit_xgboost(X_tr, y_tr):
    from xgboost import XGBClassifier
    model = XGBClassifier(
        random_state=42, scale_pos_weight=SCALE_POS,
        eval_metric="logloss", verbosity=0, n_jobs=4, tree_method="hist",
    )
    pipe = Pipeline([("scaler", RobustScaler()), ("model", model)])
    params = {
        "model__n_estimators": [50, 100, 200],
        "model__max_depth": [3, 5, 7],
        "model__learning_rate": [0.01, 0.1, 0.2],
    }
    grid = GridSearchCV(pipe, params, cv=INNER_CV, scoring="roc_auc",
                        n_jobs=1, verbose=0)
    grid.fit(X_tr, y_tr)
    return grid.best_estimator_


def fit_catboost(X_tr, y_tr):
    from catboost import CatBoostClassifier
    model = CatBoostClassifier(
        random_state=42, auto_class_weights="Balanced",
        verbose=0, thread_count=4, allow_writing_files=False,
    )
    pipe = Pipeline([("scaler", RobustScaler()), ("model", model)])
    params = {
        "model__iterations": [50, 100, 200],
        "model__depth": [3, 5, 7],
        "model__learning_rate": [0.01, 0.1, 0.2],
    }
    grid = GridSearchCV(pipe, params, cv=INNER_CV, scoring="roc_auc",
                        n_jobs=1, verbose=0)
    grid.fit(X_tr, y_tr)
    return grid.best_estimator_


def fit_tabnet(X_tr, y_tr):
    """TabNet: inner 5-fold CV to pick best of 4 configs, refit on full train."""
    import torch
    from pytorch_tabnet.tab_model import TabNetClassifier

    torch.set_num_threads(4)

    param_grid = [
        {"n_d": 8,  "n_a": 8,  "n_steps": 3, "gamma": 1.3},
        {"n_d": 16, "n_a": 16, "n_steps": 3, "gamma": 1.3},
        {"n_d": 16, "n_a": 16, "n_steps": 5, "gamma": 1.5},
        {"n_d": 32, "n_a": 32, "n_steps": 3, "gamma": 1.3},
    ]

    scaler = RobustScaler()
    X_tr_s = scaler.fit_transform(X_tr.values)
    y_tr_arr = y_tr.values

    best_auc, best_params = -1.0, param_grid[0]
    for params in param_grid:
        inner_aucs = []
        for in_tr, in_va in INNER_CV.split(X_tr_s, y_tr_arr):
            Xi_tr, yi_tr = X_tr_s[in_tr], y_tr_arr[in_tr]
            Xi_va, yi_va = X_tr_s[in_va], y_tr_arr[in_va]
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

    w_pos = (len(y_tr_arr) - y_tr_arr.sum()) / max(y_tr_arr.sum(), 1)
    final = TabNetClassifier(
        n_d=best_params["n_d"], n_a=best_params["n_a"],
        n_steps=best_params["n_steps"], gamma=best_params["gamma"],
        seed=42, verbose=0, device_name="cpu",
    )
    final.fit(
        X_tr_s, y_tr_arr,
        eval_metric=["auc"], max_epochs=80, patience=10, batch_size=256,
        weights={0: 1.0, 1: float(w_pos)},
    )
    return final, scaler, best_params


def holdout_sklearn(alg_name, model_key, X_tr, y_tr, X_te, y_te):
    if alg_name == "XGBoost":
        fitted = fit_xgboost(X_tr, y_tr)
    elif alg_name == "CatBoost":
        fitted = fit_catboost(X_tr, y_tr)
    else:
        raise ValueError(f"Unknown sklearn alg for holdout: {alg_name}")

    y_proba = fitted.predict_proba(X_te)[:, 1]
    y_pred = fitted.predict(X_te)
    boot = bootstrap_metrics(np.asarray(y_te), y_proba, y_pred)
    boot["algorithm"] = alg_name
    boot["model"] = model_key
    return boot


def holdout_tabnet(model_key, X_tr, y_tr, X_te, y_te):
    fitted, scaler, best_params = fit_tabnet(X_tr, y_tr)
    X_te_s = scaler.transform(X_te.values)
    y_proba = fitted.predict_proba(X_te_s)[:, 1]
    y_pred = fitted.predict(X_te_s)
    boot = bootstrap_metrics(np.asarray(y_te), y_proba, y_pred)
    boot["algorithm"] = "TabNet"
    boot["model"] = model_key
    boot["best_params"] = best_params
    return boot


def main():
    jobs = [
        ("Model 1", "TabNet"),
        ("Model 2", "TabNet"),
        ("Model 3", "TabNet"),
        ("Model 3", "XGBoost"),
        ("Model 2", "CatBoost"),
    ]

    results = {}
    if os.path.exists(RESULTS_PATH):
        results = json.load(open(RESULTS_PATH))

    for model_key, alg in jobs:
        key = f"{model_key}|{alg}"
        if key in results:
            print(f"  SKIP (already saved): {key}", flush=True)
            continue
        print("=" * 80, flush=True)
        print(f"  HOLD-OUT BOOTSTRAP: {alg} on {model_key}", flush=True)
        print("=" * 80, flush=True)

        X_tr, y_tr, X_te, y_te = load_data(MODEL_CONFIGS[model_key])
        print(f"  train n={len(X_tr):,}  test n={len(X_te):,}  "
              f"test obese={int(y_te.sum())} ({y_te.mean() * 100:.1f}%)",
              flush=True)

        if alg == "TabNet":
            out = holdout_tabnet(model_key, X_tr, y_tr, X_te, y_te)
        else:
            out = holdout_sklearn(alg, model_key, X_tr, y_tr, X_te, y_te)

        key = f"{model_key}|{alg}"
        results[key] = out

        with open(RESULTS_PATH, "w") as f:
            json.dump(results, f, indent=2)

        a = out["auc"]; p = out["precision"]; r = out["recall"]; f = out["f1"]
        sp = out["specificity"]; ac = out["accuracy"]; npv = out["npv"]
        print(f"  AUC         = {a['mean']:.3f} [{a['ci_lower']:.3f}-{a['ci_upper']:.3f}]")
        print(f"  Precision   = {p['mean']:.3f} [{p['ci_lower']:.3f}-{p['ci_upper']:.3f}]")
        print(f"  Sensitivity = {r['mean']:.3f} [{r['ci_lower']:.3f}-{r['ci_upper']:.3f}]")
        print(f"  Specificity = {sp['mean']:.3f} [{sp['ci_lower']:.3f}-{sp['ci_upper']:.3f}]")
        print(f"  F1          = {f['mean']:.3f} [{f['ci_lower']:.3f}-{f['ci_upper']:.3f}]")
        print(f"  Accuracy    = {ac['mean']:.3f} [{ac['ci_lower']:.3f}-{ac['ci_upper']:.3f}]")
        print(f"  NPV         = {npv['mean']:.3f} [{npv['ci_lower']:.3f}-{npv['ci_upper']:.3f}]")
        print(f"  saved \u2192 {RESULTS_PATH}", flush=True)

        del X_tr, y_tr, X_te, y_te
        gc.collect()


if __name__ == "__main__":
    main()
