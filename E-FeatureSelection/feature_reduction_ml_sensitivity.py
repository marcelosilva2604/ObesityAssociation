#!/usr/bin/env python3
"""
ML feature-reduction sensitivity analysis (Reviewer #2 response).

Direct empirical test of whether features removed during the reduction pipeline
(for high missingness, low variance, etc.) carry ML-usable signal that was lost.

Procedure:
  1. Start from the raw age-filtered dataset (8,236 × 741).
  2. Filter to valid candidate features (exclude tautological, administrative,
     and out-of-window-of-interest variables).
  3. Split candidates into numeric and low-cardinality categorical.
  4. Build an "expanded" feature matrix per Model 1's train/test partition.
  5. Run Random Forest (the only algorithm whose hold-out 95% CI excluded 0.5
     in the primary analysis) under identical 5-fold CV settings, on:
        (a) RESTRICTED: the current Model 1 feature set (27 features)
        (b) EXPANDED:  Model 1 feature set plus all valid candidate features
                       from A, with in-fold median/mode imputation.
  6. Compare CV AUC. Hypothesis: if removed features carried material signal,
     EXPANDED should substantially exceed RESTRICTED.
"""
import os
import warnings
import time

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

warnings.filterwarnings("ignore")
os.environ.setdefault("OMP_NUM_THREADS", "4")

BASE = "/Users/marcelosilva/project/early-obesity-prediction"
A_PATH = f"{BASE}/A-Age Sample Filter/data_age_filtered_2_3_4.csv"
M1_TRAIN = f"{BASE}/D-Train-Test Split/MODEL1TRAIN.csv"
M1_TEST = f"{BASE}/D-Train-Test Split/MODEL1TEST.csv"
OUT = f"{BASE}/E-FeatureSelection/feature_reduction_ml_sensitivity_results.csv"

TARGET_COLS = {"desnutrido", "eutrofico", "sobrepeso", "obeso"}
TAUTOLOGICAL = ("peso", "altura", "imc", "cintura")
ADMIN = ("datahora", "_codigo", "posestrato", "flag", "_id", "idh",
         "s01", "s02", "s03", "s04", "s05",
         "u27", "u28", "u29", "u30", "u31")
OUT_OF_WINDOW = ("_30m", "_36m", "_48m", "_59m")
EXCLUDE_EXACT = {"id_anon", "vd_zimc", "obeso", "sobrepeso", "eutrofico",
                 "desnutrido", "idade", "b05a_idade_em_meses"}
MAX_CATEGORIES = 20  # skip very-high-cardinality categoricals


def is_valid_candidate(col: str) -> bool:
    low = col.lower()
    if col in EXCLUDE_EXACT or low.startswith("vd_"):
        return False
    if any(s in low for s in TAUTOLOGICAL + ADMIN + OUT_OF_WINDOW):
        return False
    return True


def run_cv(X: pd.DataFrame, y: np.ndarray, numeric_cols, categorical_cols,
           label: str):
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", RobustScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore",
                                 max_categories=MAX_CATEGORIES,
                                 sparse_output=False)),
    ])
    preprocessor = ColumnTransformer(
        transformers=[("num", numeric, numeric_cols),
                      ("cat", categorical, categorical_cols)],
        remainder="drop",
    )

    clf = RandomForestClassifier(
        n_estimators=100, max_depth=10, class_weight="balanced",
        random_state=42, n_jobs=4,
    )
    pipe = Pipeline([("prep", preprocessor), ("clf", clf)])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []
    t0 = time.time()
    print(f"\n  [{label}]  features: "
          f"{len(numeric_cols)} numeric + {len(categorical_cols)} categorical",
          flush=True)
    for fold, (tr, va) in enumerate(cv.split(X, y), 1):
        X_tr, X_va = X.iloc[tr], X.iloc[va]
        y_tr, y_va = y[tr], y[va]
        pipe.fit(X_tr, y_tr)
        proba = pipe.predict_proba(X_va)[:, 1]
        auc = roc_auc_score(y_va, proba)
        aucs.append(auc)
        print(f"     fold {fold}/5  AUC={auc:.3f}", flush=True)

    arr = np.asarray(aucs)
    mean = float(arr.mean())
    se = float(arr.std(ddof=1) / np.sqrt(len(arr)))
    from scipy import stats
    ci_half = se * stats.t.ppf(0.975, len(arr) - 1)
    dt = time.time() - t0
    print(f"  [{label}]  mean AUC = {mean:.3f}  "
          f"95% CI [{mean-ci_half:.3f}, {mean+ci_half:.3f}]  "
          f"({dt:.0f}s)", flush=True)
    return {"label": label, "mean_auc": mean,
            "ci_lower": mean - ci_half, "ci_upper": mean + ci_half,
            "fold_aucs": aucs,
            "n_numeric": len(numeric_cols),
            "n_categorical": len(categorical_cols)}


def main():
    print("Loading data ...", flush=True)
    a = pd.read_csv(A_PATH, low_memory=False)
    m1_tr = pd.read_csv(M1_TRAIN)
    m1_te = pd.read_csv(M1_TEST)
    m1 = pd.concat([m1_tr, m1_te], ignore_index=True)

    # Target
    a["obeso_raw"] = (a["vd_zimc"] > 3).astype(int)
    a = a.merge(m1[["id_anon"]], on="id_anon", how="inner")  # same 8,236 ids

    y = a["obeso_raw"].values
    print(f"  A-filtered + M1 matched: {len(a):,} children, "
          f"obesity prevalence {y.mean()*100:.1f}%", flush=True)

    # ----- RESTRICTED: current Model 1 features -----
    m1_feats = [c for c in m1.columns if c not in TARGET_COLS and c != "id_anon"]
    X_restricted = m1.set_index("id_anon").loc[a["id_anon"]].reset_index(drop=True)
    X_restricted = X_restricted[m1_feats]
    # numeric vs categorical
    restricted_num = [c for c in m1_feats
                      if pd.api.types.is_numeric_dtype(X_restricted[c])]
    restricted_cat = [c for c in m1_feats if c not in restricted_num]

    res_restricted = run_cv(X_restricted, y, restricted_num, restricted_cat,
                            "RESTRICTED (Model 1 as published, 27 features)")

    # ----- EXPANDED: Model 1 + all valid candidates from A -----
    added_numeric, added_categorical = [], []
    X_add = pd.DataFrame(index=a.index)
    for col in a.columns:
        if col in ("obeso_raw", "id_anon"):
            continue
        if not is_valid_candidate(col):
            continue
        if col in restricted_num or col in restricted_cat:
            continue
        series = a[col]
        n_unique = series.nunique(dropna=True)
        if n_unique <= 1:
            continue
        if pd.api.types.is_numeric_dtype(series):
            X_add[col] = series
            added_numeric.append(col)
        else:
            if n_unique > MAX_CATEGORIES:
                continue  # skip high-cardinality categoricals
            X_add[col] = series.astype(str)
            added_categorical.append(col)

    print(f"\n  Candidates added: {len(added_numeric)} numeric "
          f"+ {len(added_categorical)} categorical", flush=True)

    X_expanded = pd.concat(
        [X_restricted.reset_index(drop=True), X_add.reset_index(drop=True)],
        axis=1,
    )
    exp_num = restricted_num + added_numeric
    exp_cat = restricted_cat + added_categorical

    res_expanded = run_cv(X_expanded, y, exp_num, exp_cat,
                          "EXPANDED (Model 1 + all valid candidates from A)")

    # ----- Summary -----
    print("\n" + "=" * 80)
    print("  SENSITIVITY ANALYSIS SUMMARY")
    print("=" * 80)
    delta = res_expanded["mean_auc"] - res_restricted["mean_auc"]
    print(f"  Restricted mean AUC = {res_restricted['mean_auc']:.3f} "
          f"[{res_restricted['ci_lower']:.3f}, {res_restricted['ci_upper']:.3f}]")
    print(f"  Expanded   mean AUC = {res_expanded['mean_auc']:.3f} "
          f"[{res_expanded['ci_lower']:.3f}, {res_expanded['ci_upper']:.3f}]")
    print(f"  Delta (Expanded - Restricted) = {delta:+.3f}")

    pd.DataFrame([res_restricted, res_expanded]).to_csv(OUT, index=False)
    print(f"\n  Saved -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
