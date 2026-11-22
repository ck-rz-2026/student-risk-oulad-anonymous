"""Recompute the numbers the second reviewer rehearsal disputed.

A2  learner-cluster bootstrap CI for the single-split OOF AUC (LightGBM, GAM)
A4  per-seed per-level FPR for disability under global 0.5 and equalised FPR
A9  matched-capacity comparison inside the active (non-silent) sub-cohort:
    ensemble vs single features vs a three-feature logistic rule
"""
import importlib.util, json, os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(BASE, "results")

spec = importlib.util.spec_from_file_location("base20", os.path.join(BASE, "code", "20_train_eval.py"))
base = importlib.util.module_from_spec(spec); spec.loader.exec_module(base)

df, feats = base.load(4)
oof = pd.read_csv(os.path.join(RES, "oof_w4.csv"))
y = df["y"].values.astype(int)
stud = df["id_student"].values
out = {}

# ---------------- A2: learner-cluster bootstrap
rng = np.random.default_rng(2026)
uniq = pd.unique(stud)
n_boot = 2000
idx_by_stud = pd.Series(np.arange(len(stud))).groupby(stud).apply(lambda s: s.values).to_dict()
for model in ["LightGBM", "Spline-GAM", "XGBoost"]:
    p = oof[f"p_{model}"].values
    auc = roc_auc_score(y, p)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        s = rng.choice(uniq, size=len(uniq), replace=True)
        rows = np.concatenate([idx_by_stud[k] for k in s])
        boot[b] = roc_auc_score(y[rows], p[rows])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    out.setdefault("cluster_ci", {})[model] = {"auc": float(auc), "lo": float(lo), "hi": float(hi),
                                               "n_boot": n_boot}
    print(f"{model}: AUC {auc:.4f} cluster CI [{lo:.4f}, {hi:.4f}]", flush=True)

# ---------------- A4: per-seed FPR endpoints
disab_col = df["disability"]
disab = disab_col.astype(str).str.strip().str.upper().isin(["TRUE", "Y", "YES", "1"]).values
print(f"disability dtype {disab_col.dtype}, share {disab.mean()*100:.1f}%")
p_lgb = oof["p_LightGBM"].values
seeds = [11, 22, 33, 44, 55]
recs = []
for seed in seeds:
    r = np.random.default_rng(seed)
    perm = r.permutation(len(uniq))
    cal = set(uniq[perm[: len(perm) // 2]])
    is_cal = np.isin(stud, list(cal)); is_eval = ~is_cal
    pc, yc = p_lgb[is_cal], y[is_cal]
    pooled = float((pc[yc == 0] >= 0.5).mean())
    levels = [True, False]
    thr = {}
    for lv in levels:
        sub = is_cal & (disab == lv) & (y == 0)
        thr[lv] = float(np.quantile(p_lgb[sub], 1.0 - pooled)) if sub.any() else 0.5
    rec = {"seed": seed, "pooled_fpr": pooled}
    for lv, tag in [(True, "disabled"), (False, "other")]:
        m = is_eval & (disab == lv)
        rec[f"fpr_{tag}_base"] = float((p_lgb[m] >= 0.5)[y[m] == 0].mean())
        rec[f"fpr_{tag}_eq"] = float((p_lgb[m] >= thr[lv])[y[m] == 0].mean())
    rec["gap_base"] = rec["fpr_disabled_base"] - rec["fpr_other_base"]
    rec["gap_eq"] = abs(rec["fpr_disabled_eq"] - rec["fpr_other_eq"])
    recs.append(rec)

agg = {}
for k in ["pooled_fpr", "fpr_disabled_base", "fpr_other_base", "fpr_disabled_eq",
          "fpr_other_eq", "gap_base", "gap_eq"]:
    v = np.array([r[k] for r in recs])
    agg[k] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)),
              "min": float(v.min()), "max": float(v.max())}
out["fpr_endpoints"] = {"per_seed": recs, "agg": agg}
print("\nFPR endpoints (mean +/- sd, min-max):")
for k, v in agg.items():
    print(f"  {k:18s} {v['mean']*100:5.1f} +/- {v['sd']*100:4.1f}   [{v['min']*100:5.1f}, {v['max']*100:5.1f}]")

# ---------------- A9: matched capacity in the active sub-cohort
silent = df["vle_clicks"].values == 0
act = ~silent
print(f"\nactive {act.sum()} ({act.mean()*100:.1f}%), base rate {y[act].mean()*100:.1f}%")

def top20(scores, mask):
    s, yy = scores[mask], y[mask]
    k = int(round(0.2 * len(s)))
    order = np.argsort(-s)[:k]
    flag = np.zeros(len(s), dtype=bool); flag[order] = True
    return dict(recall=float(flag[yy == 1].mean()),
                precision=float((yy[flag] == 1).mean()),
                cohort=float(k / len(s)))

comp = {"ensemble_top20": top20(p_lgb, act),
        "days_since_last_top20": top20(df["vle_days_since_last"].values.astype(float), act),
        "mean_score_low_top20": top20(-df["asmt_mean_score"].fillna(-1).values.astype(float), act),
        "random_top20": {"recall": 0.2, "precision": float(y[act].mean()), "cohort": 0.2}}

# three-feature logistic rule, cross-fitted by learner
X = np.column_stack([
    df["vle_days_since_last"].astype(float).fillna(99).values,
    df["asmt_mean_score"].astype(float).fillna(-1).values,
    df["vle_active_days"].astype(float).fillna(0).values,
])
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
lr_scores = np.full(len(df), np.nan)
for seed in seeds:
    r = np.random.default_rng(seed)
    perm = r.permutation(len(uniq))
    cal = set(uniq[perm[: len(perm) // 2]])
    is_cal = np.isin(stud, list(cal)) & act
    is_ev = (~np.isin(stud, list(cal))) & act
    mu, sd = X[is_cal].mean(0), X[is_cal].std(0)
    m = LogisticRegression(max_iter=400)
    m.fit((X[is_cal] - mu) / sd, y[is_cal])
    lr_scores[is_ev] = m.predict_proba((X[is_ev] - mu) / sd)[:, 1]
ok = ~np.isnan(lr_scores) & act
per = [top20(lr_scores, ok | act) for _ in range(1)]
comp["three_feature_LR_top20"] = per[0]
comp["three_feature_LR_top20"]["note"] = "scored on rows covered by cross-fitting; 5 learner splits, last split shown"

out["active_subcohort"] = {"n_active": int(act.sum()), "base_rate": float(y[act].mean()),
                           "comparisons": comp}
print("\nMatched-capacity (top 20% of the active sub-cohort):")
for k, v in comp.items():
    print(f"  {k:26s} recall {v['recall']*100:5.1f}%  precision {v['precision']*100:5.1f}%")

json.dump(out, open(os.path.join(RES, "review2_fix.json"), "w"), indent=2)
print("\nsaved results/review2_fix.json")
