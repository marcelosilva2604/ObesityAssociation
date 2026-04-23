"""
Maternal pre-pregnancy BMI sensitivity analysis.

Hypothesis: the informational ceiling observed with the primary feature set
could reflect omission of maternal pre-pregnancy BMI — the most consistently
predictive variable in the longitudinal-EHR comparator literature
(LeCroy 2021, Cheng 2019, Xue 2020).

Maternal BMI measured at the ENANI-2019 visit (vd_imc_mae, t02/t03 weight,
t05/t06 height) reflects maternal anthropometry when the index child was
already 2-4 years old and shares simultaneous household-level exposures with
the child. This constitutes reverse temporal leakage and these variables were
excluded from the primary analysis.

Maternal pre-pregnancy BMI temporally precedes the child and is computed here
as:
    IMC_pregravidez = k06_peso_engravidar (kg) / (t05_altura_medida1 / 100)**2

Maternal height is treated as a fixed adult characteristic, so pairing
height measured at interview with self-reported pre-pregnancy weight yields
a temporally valid pre-pregnancy BMI.

Design: apples-to-apples comparison on the complete-case subsample where
maternal pre-pregnancy BMI is computable, applied to each of the three
model configurations from the primary analysis.
- Baseline: original feature set of the model, restricted to the complete-
  case subsample.
- Extended: same + maternal pre-pregnancy BMI.
Three algorithms (Logistic Regression, Random Forest, TabNet) to cover the
linear, ensemble-tree, and deep-learning paradigms represented in the paper's
top CV performers. Evaluation: 5-fold stratified cross-validation with
bootstrap 95% CI on the mean fold AUC.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings("ignore")

try:
    from pytorch_tabnet.tab_model import TabNetClassifier
    import torch
    HAS_TABNET = True
except Exception:
    HAS_TABNET = False

PROJECT = Path("/Users/marcelosilva/project/early-obesity-prediction")
RAW = PROJECT / "A-Age Sample Filter" / "data_age_filtered_2_3_4.csv"
OUT_JSON = PROJECT / "E-FeatureSelection" / "maternal_bmi_sensitivity_results.json"

RNG = 42
N_BOOT = 1000


def load_maternal_bmi_feature():
    """Load raw ENANI data, compute maternal pre-pregnancy BMI, return DataFrame."""
    raw = pd.read_csv(RAW, low_memory=False,
                      usecols=["id_anon", "k06_peso_engravidar",
                               "t05_altura_medida1"])
    raw["k06_peso_engravidar"] = pd.to_numeric(
        raw["k06_peso_engravidar"], errors="coerce")
    raw["t05_altura_medida1"] = pd.to_numeric(
        raw["t05_altura_medida1"], errors="coerce")
    # Sentinel codes in ENANI (999.9 for refused/missing in self-reported weights)
    raw.loc[raw["k06_peso_engravidar"] >= 500, "k06_peso_engravidar"] = np.nan
    raw.loc[raw["t05_altura_medida1"] < 120, "t05_altura_medida1"] = np.nan
    raw["imc_materno_pregravidez"] = (
        raw["k06_peso_engravidar"] /
        (raw["t05_altura_medida1"] / 100.0) ** 2
    )
    # Physiologic plausibility filter
    raw.loc[~raw["imc_materno_pregravidez"].between(14, 60),
            "imc_materno_pregravidez"] = np.nan
    return raw[["id_anon", "imc_materno_pregravidez"]]


def merge_feature(dsmodel_path, bmi_df):
    dsm = pd.read_csv(dsmodel_path)
    merged = dsm.merge(bmi_df, on="id_anon", how="left")
    return merged


def evaluate_logreg(X, y, seed=RNG):
    aucs = []
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for tr, te in cv.split(X, y):
        scaler = RobustScaler()
        Xtr = scaler.fit_transform(X.iloc[tr])
        Xte = scaler.transform(X.iloc[te])
        clf = LogisticRegression(class_weight="balanced", max_iter=1000,
                                 random_state=seed)
        clf.fit(Xtr, y.iloc[tr])
        p = clf.predict_proba(Xte)[:, 1]
        aucs.append(roc_auc_score(y.iloc[te], p))
    return np.array(aucs)


def evaluate_rf(X, y, seed=RNG):
    aucs = []
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for tr, te in cv.split(X, y):
        clf = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                     random_state=seed, n_jobs=1)
        clf.fit(X.iloc[tr], y.iloc[tr])
        p = clf.predict_proba(X.iloc[te])[:, 1]
        aucs.append(roc_auc_score(y.iloc[te], p))
    return np.array(aucs)


def evaluate_tabnet(X, y, seed=RNG):
    if not HAS_TABNET:
        return None
    torch.manual_seed(seed)
    aucs = []
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for tr, te in cv.split(X, y):
        scaler = RobustScaler()
        Xtr = scaler.fit_transform(X.iloc[tr]).astype(np.float32)
        Xte = scaler.transform(X.iloc[te]).astype(np.float32)
        ytr = y.iloc[tr].values
        yte = y.iloc[te].values
        w = np.where(ytr == 1, float(len(ytr)) / (2 * ytr.sum()),
                     float(len(ytr)) / (2 * (len(ytr) - ytr.sum())))
        clf = TabNetClassifier(verbose=0, seed=seed,
                               optimizer_params=dict(lr=2e-2))
        clf.fit(Xtr, ytr, weights=w, max_epochs=50, patience=10,
                batch_size=256, virtual_batch_size=64)
        p = clf.predict_proba(Xte)[:, 1]
        aucs.append(roc_auc_score(yte, p))
    return np.array(aucs)


def bootstrap_ci(aucs, n_boot=N_BOOT, seed=RNG):
    rng = np.random.default_rng(seed)
    boots = rng.choice(aucs, size=(n_boot, len(aucs)), replace=True).mean(1)
    return float(np.mean(aucs)), float(np.percentile(boots, 2.5)), \
           float(np.percentile(boots, 97.5))


def run_model(model_name, dsmodel_path, bmi_df, results_store):
    print(f"\n{'='*70}\n{model_name}\n{'='*70}")
    df = merge_feature(dsmodel_path, bmi_df)
    cc = df.dropna(subset=["imc_materno_pregravidez"]).copy()
    n_cc = len(cc)
    n_obe_cc = int(cc["obeso"].sum())
    print(f"Complete cases: n={n_cc}, obese={n_obe_cc} "
          f"({100*n_obe_cc/n_cc:.2f}%)")

    # Univariate test on this subsample
    imc_obe = cc.loc[cc["obeso"] == 1, "imc_materno_pregravidez"]
    imc_non = cc.loc[cc["obeso"] == 0, "imc_materno_pregravidez"]
    _, p_mw = stats.mannwhitneyu(imc_obe, imc_non, alternative="two-sided")
    print(f"IMC materno pregravidez: obese mean={imc_obe.mean():.2f}, "
          f"non-obese mean={imc_non.mean():.2f}, MW p={p_mw:.4g}")

    target = "obeso"
    exclude = ["id_anon", "obeso", "desnutrido", "eutrofico", "sobrepeso"]
    base_feats = [c for c in cc.columns if c not in exclude
                  and c != "imc_materno_pregravidez"]
    ext_feats = base_feats + ["imc_materno_pregravidez"]
    print(f"Baseline features: {len(base_feats)}  |  "
          f"Extended features: {len(ext_feats)}")

    X_base = cc[base_feats]
    X_ext = cc[ext_feats]
    y = cc[target]

    model_results = {
        "n_complete_cases": int(n_cc),
        "n_obese": int(n_obe_cc),
        "prevalence_pct": round(100 * n_obe_cc / n_cc, 2),
        "n_baseline_features": len(base_feats),
        "n_extended_features": len(ext_feats),
        "imc_materno_mean_obese": float(imc_obe.mean()),
        "imc_materno_mean_nonobese": float(imc_non.mean()),
        "univariate_p_mannwhitney": float(p_mw),
        "algorithms": {},
    }

    algos = [("Logistic Regression", evaluate_logreg),
             ("Random Forest", evaluate_rf)]
    if HAS_TABNET:
        algos.append(("TabNet", evaluate_tabnet))

    for name, fn in algos:
        aucs_base = fn(X_base, y)
        m_base, lo_base, hi_base = bootstrap_ci(aucs_base)
        aucs_ext = fn(X_ext, y)
        m_ext, lo_ext, hi_ext = bootstrap_ci(aucs_ext)
        delta = m_ext - m_base
        print(f"  {name:20s}  baseline AUC={m_base:.3f} "
              f"[{lo_base:.3f}-{hi_base:.3f}]  |  "
              f"extended AUC={m_ext:.3f} [{lo_ext:.3f}-{hi_ext:.3f}]  |  "
              f"ΔAUC={delta:+.3f}")

        model_results["algorithms"][name] = {
            "baseline_auc": m_base,
            "baseline_ci": [lo_base, hi_base],
            "extended_auc": m_ext,
            "extended_ci": [lo_ext, hi_ext],
            "delta_auc": delta,
            "baseline_folds": aucs_base.tolist(),
            "extended_folds": aucs_ext.tolist(),
        }

    results_store[model_name] = model_results


def main():
    bmi_df = load_maternal_bmi_feature()

    results = {}
    for name, path in [
        ("DSModel1", PROJECT / "D-Train-Test Split" / "DSModel1.csv"),
        ("DSModel2", PROJECT / "D-Train-Test Split" / "DSModel2.csv"),
        ("DSModel3", PROJECT / "D-Train-Test Split" / "DSModel3.csv"),
    ]:
        run_model(name, path, bmi_df, results)

    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {OUT_JSON}")


if __name__ == "__main__":
    main()
