#!/usr/bin/env python3
"""
Level-3 feature-reduction sensitivity: run all 13 algorithms on the expanded
feature set (Model 1: ~195 features, Model 2: ~179, Model 3: ~205), with the
same nested 5x5 cross-validation protocol used in the primary analysis.

Expanded feature set = each model's baseline features + all ENANI variables
from thematic blocks A, B, D, H, I (≤24 months), J, K, Q, R, X, after:
  - excluding derived composite indices (vd_ prefix)
  - excluding administrative timestamps and outcome measurements
  - median imputation for numerics, mode imputation for categoricals
  - one-hot encoding (max 20 categories) for categoricals

Results saved incrementally to expanded_algorithms_results.json.
"""
import gc
import json
import os
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings("ignore")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

BASE = "/Users/marcelosilva/project/early-obesity-prediction"
A_PATH = f"{BASE}/A-Age Sample Filter/data_age_filtered_2_3_4.csv"
DICT_PATH = f"{BASE}/4-Dicionario-ENANI-2019 (1).xlsx"
RESULTS_PATH = f"{BASE}/E-FeatureSelection/expanded_algorithms_results.json"

MODEL_CONFIGS = {
    "Model 1": {
        "train": f"{BASE}/D-Train-Test Split/MODEL1TRAIN.csv",
        "binary_fix": False,
    },
    "Model 2": {
        "train": f"{BASE}/D-Train-Test Split/MODEL2TRAIN.csv",
        "binary_fix": True,
    },
    "Model 3": {
        "train": f"{BASE}/D-Train-Test Split/MODEL3TRAIN.csv",
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

ALLOWED_BLOCOS = {"A", "B", "D", "H", "I", "J", "K", "Q", "R", "X"}
OUT_OF_WINDOW_SUBSTR = ("_30m", "_36m", "_48m", "_59m")
EXCLUDE_EXACT = {
    "id_anon", "vd_zimc", "obeso", "sobrepeso", "eutrofico", "desnutrido",
    "idade", "b05a_idade_em_meses",
}
MAX_CATEGORIES = 20

OUTER_CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
INNER_CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
SCALE_POS = 33.0


# ----------------------------------------------------------------------
# Expanded dataset construction
# ----------------------------------------------------------------------

def _load_bloco_map():
    d = pd.read_excel(DICT_PATH)
    return dict(zip(d["variavel"], d["bloco"]))


VAR2BLOCO = _load_bloco_map()


def is_valid_candidate(col: str) -> bool:
    if col in EXCLUDE_EXACT:
        return False
    if col.lower().startswith("vd_"):
        return False
    if VAR2BLOCO.get(col) not in ALLOWED_BLOCOS:
        return False
    if any(s in col.lower() for s in OUT_OF_WINDOW_SUBSTR):
        return False
    return True


def build_expanded(model_key):
    """Return (X, y) with imputed + one-hot encoded expanded feature set."""
    cfg = MODEL_CONFIGS[model_key]
    m = pd.read_csv(cfg["train"])
    if cfg["binary_fix"]:
        for var in BINARY_VARS:
            if var in m.columns:
                m[var] = m[var].astype(int)

    y = m["obeso"].astype(int).copy()
    base_feats = [c for c in m.columns
                  if c not in TARGET_COLS and c != "id_anon"]
    X = m[base_feats].copy()

    a = pd.read_csv(A_PATH, low_memory=False)
    a_aligned = a.set_index("id_anon").loc[m["id_anon"]].reset_index(drop=True)

    added_num, added_cat = [], []
    for col in a_aligned.columns:
        if col == "id_anon" or not is_valid_candidate(col):
            continue
        if col in X.columns:
            continue
        s = a_aligned[col]
        n_unique = s.nunique(dropna=True)
        if n_unique <= 1:
            continue
        if pd.api.types.is_numeric_dtype(s):
            X[col] = s.values
            added_num.append(col)
        else:
            if n_unique > MAX_CATEGORIES:
                continue
            X[col] = s.astype(str).values
            added_cat.append(col)

    # median impute numerics globally (simple — acceptable for sensitivity)
    num_cols = [c for c in X.columns
                if pd.api.types.is_numeric_dtype(X[c])]
    for c in num_cols:
        X[c] = X[c].fillna(X[c].median())
    # mode impute categoricals, then one-hot encode with max_categories cap
    cat_cols = [c for c in X.columns if c not in num_cols]
    for c in cat_cols:
        mode = X[c].mode(dropna=True)
        fill = mode.iloc[0] if len(mode) else "missing"
        X[c] = X[c].fillna(fill).astype(str)

    X_enc = pd.get_dummies(X, columns=cat_cols, drop_first=True,
                           dummy_na=False)
    # limit to top-20 categories per raw categorical if an explosion occurred
    X_enc = X_enc.loc[:, ~X_enc.columns.duplicated()]
    return X_enc.astype(float), y


# ----------------------------------------------------------------------
# Algorithm factories
# ----------------------------------------------------------------------

def factory(name):
    if name == "LogisticRegression":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(class_weight="balanced",
                                  max_iter=2000,
                                  random_state=42), {
            "model__C": [0.1, 1.0, 10.0],
        }, 1
    if name == "RandomForest":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(class_weight="balanced",
                                      random_state=42,
                                      n_jobs=4), {
            "model__n_estimators": [100, 200],
            "model__max_depth": [5, 10, None],
        }, 1
    if name == "GradientBoosting":
        from sklearn.ensemble import GradientBoostingClassifier
        return GradientBoostingClassifier(random_state=42), {
            "model__n_estimators": [100, 200],
            "model__max_depth": [3, 5],
            "model__learning_rate": [0.05, 0.1],
        }, 1
    if name == "DecisionTree":
        from sklearn.tree import DecisionTreeClassifier
        return DecisionTreeClassifier(class_weight="balanced",
                                      random_state=42), {
            "model__max_depth": [3, 5, 10, None],
        }, 1
    if name == "KNN":
        from sklearn.neighbors import KNeighborsClassifier
        return KNeighborsClassifier(n_jobs=4), {
            "model__n_neighbors": [5, 15, 31],
        }, 1
    if name == "GaussianNB":
        from sklearn.naive_bayes import GaussianNB
        return GaussianNB(), {}, 1
    if name == "SVMLinear":
        from sklearn.svm import SVC
        return SVC(kernel="linear", class_weight="balanced",
                   probability=True, random_state=42), {
            "model__C": [0.1, 1.0, 10.0],
        }, 1
    if name == "SVMRBF":
        from sklearn.svm import SVC
        return SVC(kernel="rbf", class_weight="balanced",
                   probability=True, random_state=42), {
            "model__C": [0.1, 1.0, 10.0],
            "model__gamma": ["scale", 0.01],
        }, 1
    if name == "XGBoost":
        from xgboost import XGBClassifier
        return XGBClassifier(random_state=42,
                             scale_pos_weight=SCALE_POS,
                             eval_metric="logloss", verbosity=0,
                             n_jobs=4,
                             tree_method="hist"), {
            "model__n_estimators": [100, 200],
            "model__max_depth": [3, 5, 7],
            "model__learning_rate": [0.05, 0.1],
        }, 1
    if name == "LightGBM":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(random_state=42,
                              scale_pos_weight=SCALE_POS,
                              verbose=-1, n_jobs=4), {
            "model__n_estimators": [100, 200],
            "model__max_depth": [-1, 5, 7],
            "model__learning_rate": [0.05, 0.1],
        }, 1
    if name == "CatBoost":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(random_state=42,
                                  auto_class_weights="Balanced",
                                  verbose=0, thread_count=4,
                                  allow_writing_files=False), {
            "model__iterations": [100, 200],
            "model__depth": [3, 5, 7],
            "model__learning_rate": [0.05, 0.1],
        }, 1
    if name == "MLP":
        from sklearn.neural_network import MLPClassifier
        return MLPClassifier(random_state=42, max_iter=500,
                             early_stopping=True,
                             validation_fraction=0.15,
                             n_iter_no_change=10), {
            "model__hidden_layer_sizes": [(64,), (128,), (64, 32)],
            "model__alpha": [0.0001, 0.01],
        }, 1
    raise ValueError(name)


SKLEARN_ALGS = [
    "LogisticRegression", "RandomForest", "GradientBoosting", "DecisionTree",
    "KNN", "GaussianNB", "SVMLinear", "SVMRBF",
    "XGBoost", "LightGBM", "CatBoost", "MLP",
]


def evaluate_sklearn(X, y, alg):
    model, grid, gs_njobs = factory(alg)
    pipe = Pipeline([("scaler", RobustScaler()), ("model", model)])
    aucs, precs, recs, f1s = [], [], [], []

    for fold, (tr, va) in enumerate(OUTER_CV.split(X, y), 1):
        X_tr, y_tr = X.iloc[tr], y.iloc[tr]
        X_va, y_va = X.iloc[va], y.iloc[va]

        if grid:
            gs = GridSearchCV(pipe, grid, cv=INNER_CV, scoring="roc_auc",
                              n_jobs=gs_njobs, verbose=0)
            gs.fit(X_tr, y_tr)
            est = gs.best_estimator_
        else:
            pipe.fit(X_tr, y_tr)
            est = pipe
        proba = est.predict_proba(X_va)[:, 1]
        pred = est.predict(X_va)
        a = roc_auc_score(y_va, proba)
        p = precision_score(y_va, pred, zero_division=0)
        r = recall_score(y_va, pred, zero_division=0)
        f = f1_score(y_va, pred, zero_division=0)
        aucs.append(a); precs.append(p); recs.append(r); f1s.append(f)
        print(f"      fold {fold}/5  AUC={a:.3f} P={p:.3f} R={r:.3f} F1={f:.3f}",
              flush=True)
        del est
        gc.collect()

    return summarize(alg, aucs, precs, recs, f1s)


def evaluate_tabnet(X, y):
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
    for fold, (tr, va) in enumerate(OUTER_CV.split(X, y), 1):
        X_tr = X.iloc[tr].values; y_tr = y.iloc[tr].values
        X_va = X.iloc[va].values; y_va = y.iloc[va].values

        scaler = RobustScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_va_s = scaler.transform(X_va)

        best_auc, best_params = -1.0, param_grid[0]
        for params in param_grid:
            inner_aucs = []
            for in_tr, in_va in INNER_CV.split(X_tr_s, y_tr):
                Xi_tr, yi_tr = X_tr_s[in_tr], y_tr[in_tr]
                Xi_va, yi_va = X_tr_s[in_va], y_tr[in_va]
                w_pos = (len(yi_tr) - yi_tr.sum()) / max(yi_tr.sum(), 1)
                m = TabNetClassifier(seed=42, verbose=0, device_name="cpu",
                                     **params)
                m.fit(Xi_tr, yi_tr,
                      eval_set=[(Xi_va, yi_va)], eval_metric=["auc"],
                      max_epochs=60, patience=10, batch_size=256,
                      weights={0: 1.0, 1: float(w_pos)})
                inner_aucs.append(roc_auc_score(yi_va,
                                                m.predict_proba(Xi_va)[:, 1]))
                del m; gc.collect()
            mean_in = float(np.mean(inner_aucs))
            if mean_in > best_auc:
                best_auc, best_params = mean_in, params

        w_pos = (len(y_tr) - y_tr.sum()) / max(y_tr.sum(), 1)
        m = TabNetClassifier(seed=42, verbose=0, device_name="cpu",
                             **best_params)
        m.fit(X_tr_s, y_tr,
              eval_set=[(X_va_s, y_va)], eval_metric=["auc"],
              max_epochs=60, patience=10, batch_size=256,
              weights={0: 1.0, 1: float(w_pos)})
        proba = m.predict_proba(X_va_s)[:, 1]
        pred = m.predict(X_va_s)
        a = roc_auc_score(y_va, proba)
        p = precision_score(y_va, pred, zero_division=0)
        r = recall_score(y_va, pred, zero_division=0)
        f = f1_score(y_va, pred, zero_division=0)
        aucs.append(a); precs.append(p); recs.append(r); f1s.append(f)
        print(f"      fold {fold}/5  AUC={a:.3f} best={best_params}",
              flush=True)
        del m, scaler
        gc.collect()
    return summarize("TabNet", aucs, precs, recs, f1s)


def summarize(name, aucs, precs, recs, f1s):
    def ci95(scores):
        n = len(scores); m = float(np.mean(scores))
        if n <= 1:
            return m, m, m
        h = float(stats.sem(scores) * stats.t.ppf(0.975, n - 1))
        return m, m - h, m + h
    am, alo, ahi = ci95(aucs)
    pm, plo, phi = ci95(precs)
    rm, rlo, rhi = ci95(recs)
    fm, flo, fhi = ci95(f1s)
    return {"algorithm": name,
            "auc": {"mean": am, "ci_lower": alo, "ci_upper": ahi},
            "precision": {"mean": pm, "ci_lower": plo, "ci_upper": phi},
            "recall": {"mean": rm, "ci_lower": rlo, "ci_upper": rhi},
            "f1": {"mean": fm, "ci_lower": flo, "ci_upper": fhi},
            "fold_aucs": [float(x) for x in aucs]}


def load_results():
    if os.path.exists(RESULTS_PATH):
        return json.load(open(RESULTS_PATH))
    return {}


def save_results(r):
    with open(RESULTS_PATH, "w") as f:
        json.dump(r, f, indent=2)


def main():
    all_results = load_results()

    for model_key in MODEL_CONFIGS:
        print("=" * 90)
        print(f"  {model_key} - EXPANDED feature set")
        print("=" * 90, flush=True)
        X, y = build_expanded(model_key)
        print(f"  shape: {X.shape}, obeso={int(y.sum())} ({y.mean()*100:.1f}%)",
              flush=True)

        prior = all_results.get(model_key, [])
        done_algs = {r["algorithm"] for r in prior}

        for alg in SKLEARN_ALGS + ["TabNet"]:
            if alg in done_algs:
                print(f"  SKIP {alg} (already saved)", flush=True)
                continue
            print(f"\n  >>> {alg}", flush=True)
            if alg == "TabNet":
                res = evaluate_tabnet(X, y)
            else:
                res = evaluate_sklearn(X, y, alg)
            res["model"] = model_key
            prior = [r for r in prior if r["algorithm"] != alg]
            prior.append(res)
            all_results[model_key] = prior
            save_results(all_results)
            a = res["auc"]
            print(f"      -> {alg} AUC={a['mean']:.3f} "
                  f"[{a['ci_lower']:.3f}-{a['ci_upper']:.3f}]",
                  flush=True)

        del X, y
        gc.collect()

    print("\n" + "=" * 90)
    print("  ALL DONE")
    print("=" * 90, flush=True)


if __name__ == "__main__":
    main()
