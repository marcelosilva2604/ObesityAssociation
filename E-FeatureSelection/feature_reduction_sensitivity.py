#!/usr/bin/env python3
"""
Feature-reduction sensitivity analysis.

Addresses JAMIA Reviewer #2: "columns with high degrees of missingness and
columns with low variance can be PARTICULARLY informative, especially in the
case of rare outcome prediction."

Tests empirically whether features removed during pipeline reduction carry
stronger univariate association with obesity than retained features.

For each column in the raw age-filtered dataset:
  - Compute univariate AUC-ROC against the binary obesity target
    (a standard, scale-free measure of discriminative signal that does not
     require imputation for missing values — we use only complete rows per
     feature for its own calculation)
  - Flag the feature as RETAINED (present in the final MODEL1 feature set
    after all reduction steps) or REMOVED
  - Summarize the distribution of |AUC - 0.5| for each group

Result: if the strongest univariate predictors are in the RETAINED set,
Reviewer #2's hypothesis is empirically rejected.

Output: feature_reduction_sensitivity_results.csv with per-feature AUC,
and a summary report printed to stdout.
"""
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

BASE = "/Users/marcelosilva/project/early-obesity-prediction"
A_PATH = f"{BASE}/A-Age Sample Filter/data_age_filtered_2_3_4.csv"
M1_TRAIN = f"{BASE}/D-Train-Test Split/MODEL1TRAIN.csv"
M1_TEST = f"{BASE}/D-Train-Test Split/MODEL1TEST.csv"
OUT_CSV = f"{BASE}/E-FeatureSelection/feature_reduction_sensitivity_results.csv"

NON_FEATURE_PREFIXES = (
    "vd_",      # derived/computed vars incl. the outcome itself
)
# Substrings that disqualify a var regardless of position in the name.
# These represent:
#   - measurements of the outcome itself (weight/height/BMI/waist)
#   - administrative / timestamp / internal IDs
#   - survey questionnaires asked at ages beyond the 0-24-month target window
TAUTOLOGICAL_SUBSTRINGS = ("peso", "altura", "imc", "cintura")
ADMIN_SUBSTRINGS = (
    "datahora", "_codigo", "posestrato", "flag", "_id", "idh",
    "s01", "s02", "s03", "s04", "s05",   # measurement session records
    "u27", "u28", "u29", "u30", "u31",   # urine collection administrative
)
OUT_OF_WINDOW_SUBSTRINGS = (
    "_30m", "_36m", "_48m", "_59m",  # questions asked at ages > 24 months
)
EXCLUDE_EXACT = {
    "id_anon", "vd_zimc", "obeso", "sobrepeso", "eutrofico", "desnutrido",
    "idade", "b05a_idade_em_meses",  # outcome is age-dependent; age itself leaks
}


def is_valid_candidate(col: str) -> bool:
    low = col.lower()
    if col in EXCLUDE_EXACT:
        return False
    if any(low.startswith(p) for p in NON_FEATURE_PREFIXES):
        return False
    if any(s in low for s in TAUTOLOGICAL_SUBSTRINGS):
        return False
    if any(s in low for s in ADMIN_SUBSTRINGS):
        return False
    if any(s in low for s in OUT_OF_WINDOW_SUBSTRINGS):
        return False
    return True


def compute_univariate_auc(col: pd.Series, y: pd.Series,
                           n_boot: int = 500,
                           rng: np.random.RandomState = None):
    """
    Return (mean_auc, ci_lower, ci_upper, n_complete, n_pos)
    using only rows where col is not NaN. Direction-agnostic (folds to >=0.5).
    """
    if rng is None:
        rng = np.random.RandomState(42)
    mask = col.notna() & y.notna()
    n = int(mask.sum())
    if n < 50:
        return np.nan, np.nan, np.nan, n, 0
    yv = y[mask].astype(int).values
    n_pos = int(yv.sum())
    if n_pos < 5 or n_pos == len(yv):
        return np.nan, np.nan, np.nan, n, n_pos
    cv = col[mask]

    if pd.api.types.is_numeric_dtype(cv):
        scores = cv.values.astype(float)
    else:
        means = pd.DataFrame({"cat": cv.astype(str), "y": yv}).groupby("cat")["y"].mean()
        scores = cv.astype(str).map(means).values.astype(float)

    try:
        base = roc_auc_score(yv, scores)
    except Exception:
        return np.nan, np.nan, np.nan, n, n_pos
    direction = 1.0 if base >= 0.5 else -1.0

    aucs = []
    N = len(yv)
    for _ in range(n_boot):
        idx = rng.randint(0, N, N)
        yb = yv[idx]; sb = scores[idx] * direction
        if yb.sum() == 0 or yb.sum() == N:
            continue
        try:
            aucs.append(roc_auc_score(yb, sb))
        except Exception:
            pass
    if not aucs:
        return max(base, 1 - base), np.nan, np.nan, n, n_pos
    arr = np.asarray(aucs)
    return (max(base, 1 - base),
            float(np.percentile(arr, 2.5)),
            float(np.percentile(arr, 97.5)),
            n, n_pos)


def main():
    a = pd.read_csv(A_PATH, low_memory=False)
    y = (a["vd_zimc"] > 3).astype(int)
    print(f"Dataset: {len(a):,} children, obesity prevalence {y.mean()*100:.1f}%",
          flush=True)

    retained_ids = set(pd.read_csv(M1_TRAIN)["id_anon"]) | \
                   set(pd.read_csv(M1_TEST)["id_anon"])
    # The final Model 1 feature set, AFTER all reduction and one-hot expansion,
    # comes from MODEL1TRAIN columns (minus target/id). To compare apples-to-
    # apples at the raw-feature level we match by feature-name prefix.
    m1_cols = list(pd.read_csv(M1_TRAIN).columns)
    target_cols = {"desnutrido", "eutrofico", "sobrepeso", "obeso", "id_anon"}
    m1_features = [c for c in m1_cols if c not in target_cols]
    # For one-hot encoded columns (e.g. "cor_Parda"), the underlying raw var
    # is the part before "_". So we infer the set of raw source variables kept.
    retained_raw_vars = set()
    for c in m1_features:
        # conservative: if a raw column with this exact name exists in A, use it
        if c in a.columns:
            retained_raw_vars.add(c)
        else:
            # one-hot artifact — map back to raw prefix
            prefix = c.split("_")[0]
            if prefix in a.columns:
                retained_raw_vars.add(prefix)

    print(f"Model 1 retained raw variables (matched in A): "
          f"{len(retained_raw_vars)}", flush=True)

    rng = np.random.RandomState(42)
    rows = []
    for col in a.columns:
        if not is_valid_candidate(col):
            continue

        miss_rate = a[col].isna().mean()
        n_unique = a[col].nunique(dropna=True)
        if n_unique <= 1:
            continue

        auc, lo, hi, n_comp, n_pos = compute_univariate_auc(a[col], y, rng=rng)
        if np.isnan(auc):
            continue

        status = "retained" if col in retained_raw_vars else "removed"
        if status == "removed":
            sub = "removed_high_missing" if miss_rate > 0.30 else "removed_low_missing"
        else:
            sub = "retained"

        ci_excludes_half = (not np.isnan(lo)) and (lo > 0.5)

        rows.append({
            "feature": col,
            "status": status,
            "substatus": sub,
            "missing_rate": round(miss_rate, 4),
            "n_complete": n_comp,
            "n_pos_complete": n_pos,
            "univariate_auc": round(auc, 4),
            "auc_ci_lower": round(lo, 4) if not np.isnan(lo) else np.nan,
            "auc_ci_upper": round(hi, 4) if not np.isnan(hi) else np.nan,
            "ci_excludes_half": ci_excludes_half,
        })

    df = pd.DataFrame(rows).sort_values("univariate_auc", ascending=False)
    df.to_csv(OUT_CSV, index=False)
    print(f"Saved per-feature results to: {OUT_CSV}\n", flush=True)

    print("=" * 80)
    print("  SUMMARY: univariate AUC distribution by status (all features)")
    print("=" * 80)
    for sub, g in df.groupby("substatus"):
        print(f"  {sub:26s} n={len(g):4d}  "
              f"mean AUC={g['univariate_auc'].mean():.3f}  "
              f"max={g['univariate_auc'].max():.3f}  "
              f"with CI>0.5: {int(g['ci_excludes_half'].sum())}")

    print("\n" + "=" * 80)
    print("  ROBUST SIGNALS: features whose bootstrap 95% CI for AUC excludes 0.5")
    print("=" * 80)
    robust = df[df["ci_excludes_half"]].sort_values("univariate_auc", ascending=False)
    if len(robust) == 0:
        print("  NONE.")
    else:
        print(robust[["feature","status","substatus","missing_rate",
                      "n_complete","univariate_auc","auc_ci_lower",
                      "auc_ci_upper"]].to_string(index=False))

    print("\n" + "=" * 80)
    print("  KEY COMPARISON (robust signals only)")
    print("=" * 80)
    ret_robust = robust[robust["status"] == "retained"]
    rem_robust = robust[robust["status"] == "removed"]
    print(f"  RETAINED with CI>0.5: {len(ret_robust):3d}  "
          f"max AUC {ret_robust['univariate_auc'].max() if len(ret_robust) else 0:.3f}")
    print(f"  REMOVED  with CI>0.5: {len(rem_robust):3d}  "
          f"max AUC {rem_robust['univariate_auc'].max() if len(rem_robust) else 0:.3f}")
    if len(rem_robust):
        thresh = ret_robust['univariate_auc'].max() if len(ret_robust) else 0.5
        stronger = rem_robust[rem_robust['univariate_auc'] > thresh]
        print(f"  REMOVED features with AUC > max retained robust: {len(stronger)}")


if __name__ == "__main__":
    main()
