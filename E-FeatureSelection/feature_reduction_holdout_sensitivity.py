#!/usr/bin/env python3
"""
Hold-out sensitivity for the EXPANDED Model 1 feature set.

Fits Random Forest on the full Model 1 training split using both the
RESTRICTED (27-feature) and EXPANDED (27 + all valid candidates from A)
feature sets, and evaluates on the untouched hold-out test split with
1,000-sample bootstrap confidence intervals for AUC, precision, sensitivity,
specificity, F1, accuracy, and NPV.

Purpose: verify whether the CV-level AUC lift from feature expansion
(+0.069 vs restricted) generalizes to independent data.
"""
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

warnings.filterwarnings("ignore")
os.environ.setdefault("OMP_NUM_THREADS", "4")

BASE = "/Users/marcelosilva/project/early-obesity-prediction"
A_PATH = f"{BASE}/A-Age Sample Filter/data_age_filtered_2_3_4.csv"
DICT_PATH = f"{BASE}/4-Dicionario-ENANI-2019 (1).xlsx"
OUT = f"{BASE}/E-FeatureSelection/feature_reduction_holdout_sensitivity.csv"

# All three models use the same binary target (obeso) and the same A-derived
# feature pool; each has its own train/test split and baseline algorithm.
MODEL_CONFIGS = {
    "Model 1": {
        "train": f"{BASE}/D-Train-Test Split/MODEL1TRAIN.csv",
        "test":  f"{BASE}/D-Train-Test Split/MODEL1TEST.csv",
        "n_features": 27,
        "algorithm": "Random Forest",
    },
    "Model 2": {
        "train": f"{BASE}/D-Train-Test Split/MODEL2TRAIN.csv",
        "test":  f"{BASE}/D-Train-Test Split/MODEL2TEST.csv",
        "n_features": 17,
        "algorithm": "Logistic Regression",
    },
    "Model 3": {
        "train": f"{BASE}/D-Train-Test Split/MODEL3TRAIN.csv",
        "test":  f"{BASE}/D-Train-Test Split/MODEL3TEST.csv",
        "n_features": 44,
        "algorithm": "Gradient Boosting",  # noted for reference; sensitivity
                                           # uses RF for all 3 for consistency
    },
}

TARGET_COLS = {"desnutrido", "eutrofico", "sobrepeso", "obeso"}

# ENANI blocks thematically valid for first-24-months or time-invariant
# demographic/SES context. Excluded blocks cover current state of the 2-4y
# child, blood/urine collection metadata, interview administrative items,
# and derived/composite indices (vd_ prefix).
ALLOWED_BLOCOS = {"A", "B", "D", "H", "I", "J", "K", "Q", "R", "X"}

# Sub-exclusions applied on top of the bloco whitelist.
OUT_OF_WINDOW_SUBSTR = ("_30m", "_36m", "_48m", "_59m")  # bloco I has
                                                         # items at 30m+

EXCLUDE_EXACT = {
    "id_anon", "vd_zimc", "obeso", "sobrepeso", "eutrofico", "desnutrido",
    "idade", "b05a_idade_em_meses",  # age itself is a leak / included elsewhere
}

MAX_CATEGORIES = 20
N_BOOT = 1000


def _load_bloco_map():
    d = pd.read_excel(DICT_PATH)
    return dict(zip(d["variavel"], d["bloco"]))


VAR2BLOCO = _load_bloco_map()


def is_valid_candidate(col: str) -> bool:
    if col in EXCLUDE_EXACT:
        return False
    if col.lower().startswith("vd_"):
        return False  # derived / composite index
    bloco = VAR2BLOCO.get(col)
    if bloco not in ALLOWED_BLOCOS:
        return False
    low = col.lower()
    if any(s in low for s in OUT_OF_WINDOW_SUBSTR):
        return False
    return True


def build_pipe(num_cols, cat_cols):
    num = Pipeline([("impute", SimpleImputer(strategy="median")),
                    ("scale", RobustScaler())])
    cat = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore",
                                             max_categories=MAX_CATEGORIES,
                                             sparse_output=False))])
    prep = ColumnTransformer([("num", num, num_cols), ("cat", cat, cat_cols)],
                             remainder="drop")
    clf = RandomForestClassifier(
        n_estimators=100, max_depth=10, class_weight="balanced",
        random_state=42, n_jobs=4,
    )
    return Pipeline([("prep", prep), ("clf", clf)])


def bootstrap_metrics(y_true, y_proba, y_pred, n_boot=N_BOOT, seed=42):
    rng = np.random.RandomState(seed)
    y_true = np.asarray(y_true); y_proba = np.asarray(y_proba)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    aucs, precs, recs, specs, f1s, accs, npvs = [], [], [], [], [], [], []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        yt, yp, yh = y_true[idx], y_proba[idx], y_pred[idx]
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
        return dict(mean=float(a.mean()),
                    ci_lower=float(np.percentile(a, 2.5)),
                    ci_upper=float(np.percentile(a, 97.5)))
    return {"auc": ci(aucs), "precision": ci(precs),
            "recall": ci(recs), "specificity": ci(specs),
            "f1": ci(f1s), "accuracy": ci(accs), "npv": ci(npvs)}


def determine_expanded_features(a_df: pd.DataFrame):
    """Decide numeric/categorical candidates using the FULL dataset,
    so train/test share an identical schema."""
    added_num, added_cat = [], []
    for col in a_df.columns:
        if col == "id_anon" or not is_valid_candidate(col):
            continue
        s = a_df[col]
        if s.nunique(dropna=True) <= 1:
            continue
        if pd.api.types.is_numeric_dtype(s):
            added_num.append(col)
        else:
            if s.nunique(dropna=True) > MAX_CATEGORIES:
                continue
            added_cat.append(col)
    return added_num, added_cat


def prepare(m1_df: pd.DataFrame, a_df: pd.DataFrame,
            expand: bool,
            added_num=None, added_cat=None):
    """Return (X, y, numeric_cols, categorical_cols)."""
    feats = [c for c in m1_df.columns
             if c not in TARGET_COLS and c != "id_anon"]
    X = m1_df[feats].copy()
    num_cols = [c for c in feats if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in feats if c not in num_cols]

    if expand:
        a_aligned = (a_df.set_index("id_anon")
                     .loc[m1_df["id_anon"]]
                     .reset_index(drop=True))
        to_add_num = [c for c in (added_num or []) if c not in num_cols and c not in cat_cols]
        to_add_cat = [c for c in (added_cat or []) if c not in num_cols and c not in cat_cols]
        for col in to_add_num:
            X[col] = a_aligned[col].values
        for col in to_add_cat:
            X[col] = a_aligned[col].astype(str).values
        num_cols = num_cols + to_add_num
        cat_cols = cat_cols + to_add_cat

    y = m1_df["obeso"].astype(int).values
    return X, y, num_cols, cat_cols


def _bin_fix(df, binary_vars):
    for var in binary_vars:
        if var in df.columns:
            df[var] = df[var].astype(int)
    return df


BINARY_VARS = [
    "doou_leite_banco", "recebeu_leite_banco", "amamentou_outra_crianca",
    "usou_mamadeira", "deixou_amamentar_por_outra", "busca_info_aleitamento",
    "deu_outros_liquidos", "k15_recebeu", "k11_amamentou", "k03_prenatal",
    "usou_utensilios_amamentacao",
]


def run_one_model(model_key: str, cfg: dict, a: pd.DataFrame,
                  added_num, added_cat):
    print("=" * 80, flush=True)
    print(f"  {model_key} ({cfg['n_features']} features)", flush=True)
    print("=" * 80, flush=True)

    m_tr = pd.read_csv(cfg["train"])
    m_te = pd.read_csv(cfg["test"])
    if model_key in ("Model 2", "Model 3"):
        m_tr = _bin_fix(m_tr, BINARY_VARS)
        m_te = _bin_fix(m_te, BINARY_VARS)

    rows = []
    for label, expand in [("RESTRICTED", False), ("EXPANDED", True)]:
        X_tr, y_tr, num_cols, cat_cols = prepare(
            m_tr, a, expand, added_num, added_cat)
        X_te, y_te, _, _ = prepare(
            m_te, a, expand, added_num, added_cat)
        X_te = X_te[X_tr.columns]

        print(f"\n  [{model_key} | {label}] "
              f"{len(num_cols)} numeric + {len(cat_cols)} categorical",
              flush=True)
        pipe = build_pipe(num_cols, cat_cols)
        pipe.fit(X_tr, y_tr)
        y_proba = pipe.predict_proba(X_te)[:, 1]
        y_pred = pipe.predict(X_te)

        m = bootstrap_metrics(y_te, y_proba, y_pred)
        a_ = m["auc"]; p_ = m["precision"]; r_ = m["recall"]
        sp_ = m["specificity"]; f_ = m["f1"]
        ac_ = m["accuracy"]; np_ = m["npv"]
        print(f"  AUC         = {a_['mean']:.3f} "
              f"[{a_['ci_lower']:.3f}-{a_['ci_upper']:.3f}]")
        print(f"  Precision   = {p_['mean']:.3f} "
              f"[{p_['ci_lower']:.3f}-{p_['ci_upper']:.3f}]")
        print(f"  Sensitivity = {r_['mean']:.3f} "
              f"[{r_['ci_lower']:.3f}-{r_['ci_upper']:.3f}]")
        print(f"  Specificity = {sp_['mean']:.3f} "
              f"[{sp_['ci_lower']:.3f}-{sp_['ci_upper']:.3f}]")
        print(f"  F1          = {f_['mean']:.3f} "
              f"[{f_['ci_lower']:.3f}-{f_['ci_upper']:.3f}]")
        print(f"  Accuracy    = {ac_['mean']:.3f} "
              f"[{ac_['ci_lower']:.3f}-{ac_['ci_upper']:.3f}]")
        print(f"  NPV         = {np_['mean']:.3f} "
              f"[{np_['ci_lower']:.3f}-{np_['ci_upper']:.3f}]")

        rows.append({
            "model": model_key,
            "configuration": label,
            "n_features": len(num_cols) + len(cat_cols),
            "auc": f"{a_['mean']:.3f} [{a_['ci_lower']:.3f}-{a_['ci_upper']:.3f}]",
            "precision": f"{p_['mean']:.3f} [{p_['ci_lower']:.3f}-{p_['ci_upper']:.3f}]",
            "sensitivity": f"{r_['mean']:.3f} [{r_['ci_lower']:.3f}-{r_['ci_upper']:.3f}]",
            "specificity": f"{sp_['mean']:.3f} [{sp_['ci_lower']:.3f}-{sp_['ci_upper']:.3f}]",
            "f1": f"{f_['mean']:.3f} [{f_['ci_lower']:.3f}-{f_['ci_upper']:.3f}]",
            "accuracy": f"{ac_['mean']:.3f} [{ac_['ci_lower']:.3f}-{ac_['ci_upper']:.3f}]",
            "npv": f"{np_['mean']:.3f} [{np_['ci_lower']:.3f}-{np_['ci_upper']:.3f}]",
            "auc_mean": a_['mean'],
        })

    delta = rows[1]["auc_mean"] - rows[0]["auc_mean"]
    print(f"\n  Delta ({model_key}) = {delta:+.3f}", flush=True)
    return rows


def main():
    print("Loading ...", flush=True)
    a = pd.read_csv(A_PATH, low_memory=False)

    added_num, added_cat = determine_expanded_features(a)
    print(f"  Candidates (from valid blocos, after filtering): "
          f"{len(added_num)} numeric + {len(added_cat)} categorical",
          flush=True)

    all_rows = []
    for model_key, cfg in MODEL_CONFIGS.items():
        all_rows.extend(run_one_model(model_key, cfg, a, added_num, added_cat))

    df = pd.DataFrame(all_rows).drop(columns=["auc_mean"])
    df.to_csv(OUT, index=False)
    print(f"\nSaved -> {OUT}", flush=True)

    # Summary table
    print("\n" + "=" * 80)
    print("  SENSITIVITY SUMMARY (all 3 models, hold-out)")
    print("=" * 80)
    print(df[["model","configuration","n_features","auc","precision"]].to_string(index=False))


if __name__ == "__main__":
    main()
