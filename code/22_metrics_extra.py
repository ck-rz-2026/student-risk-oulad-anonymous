"""Additional deployment-oriented metrics and bootstrap confidence intervals."""
import json, os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(BASE, "results")
df = pd.read_csv(os.path.join(RES, "oof_w4.csv"))
y = df["y"].values
models = [c[2:] for c in df.columns if c.startswith("p_")]


def operating_point(y, p, target_recall=0.80):
    fpr, tpr, thr = roc_curve(y, p)
    idx = np.searchsorted(tpr, target_recall)
    idx = min(idx, len(thr) - 1)
    t = thr[idx]
    flagged = (p >= t)
    return {"threshold": float(t), "cohort_flagged": float(flagged.mean()),
            "precision": float((y[flagged] == 1).mean()) if flagged.any() else 0.0,
            "recall": float(tpr[idx])}


def bootstrap_auc(y, p, n=2000, seed=7):
    rng = np.random.default_rng(seed)
    vals = []
    n_obs = len(y)
    for _ in range(n):
        idx = rng.integers(0, n_obs, n_obs)
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(roc_auc_score(y[idx], p[idx]))
    return float(np.mean(vals)), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


out = {}
for m in models:
    p = df[f"p_{m}"].values
    op = operating_point(y, p, 0.80)
    op60 = operating_point(y, p, 0.60)
    n_top = int(round(0.20 * len(y)))
    top = np.argsort(-p)[:n_top]
    r20 = float(y[top].sum() / y.sum())
    pr20 = float(y[top].mean())
    mean, lo, hi = bootstrap_auc(y, p)
    out[m] = {"auc_ci": [mean, lo, hi], "op80": op, "op60": op60,
              "recall@20": r20, "precision@20": pr20}
    print(f"{m:14s} AUC {mean:.4f} [{lo:.4f},{hi:.4f}]  "
          f"flag {op['cohort_flagged']*100:.1f}% for 80% recall, prec={op['precision']:.3f} | "
          f"top20%: recall={r20:.3f} prec={pr20:.3f}")
with open(os.path.join(RES, "metrics_extra.json"), "w") as f:
    json.dump(out, f, indent=2)

# pairwise AUC differences (bootstrap, paired)
rng = np.random.default_rng(11)
pairs = {}
n = len(y)
diffs = {f"{a}-{b}": [] for a in models for b in models if a < b}
boot_idx = [rng.integers(0, n, n) for _ in range(500)]
for idx in boot_idx:
    if len(np.unique(y[idx])) < 2:
        continue
    aucs = {m: roc_auc_score(y[idx], df[f'p_{m}'].values[idx]) for m in models}
    for a in models:
        for b in models:
            if a < b:
                diffs[f"{a}-{b}"].append(aucs[a] - aucs[b])
for k, v in diffs.items():
    pairs[k] = {"mean": float(np.mean(v)), "p_gt_0": float(np.mean(np.array(v) > 0))}
with open(os.path.join(RES, "pairwise.json"), "w") as f:
    json.dump(pairs, f, indent=2)
