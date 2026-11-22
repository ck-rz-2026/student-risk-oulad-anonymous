"""
Fast vectorised version of stage [D] of 31_sensitivity_evolution.py.

Same estimand, same seed, same JSON schema -- only the resampling inner
loop is rewritten.  The original loop concatenated ~28k per-learner index
arrays on every replicate (O(reps * learners)); here one multinomial
draw per replicate is expanded with a single np.repeat over the sorted
row vector (O(reps * n_rows)), which is ~40x faster.

Run:
    python code/32_paired_tests_fast.py
"""
import importlib.util
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(BASE, "results")
N_BOOT = 2000
SEED = 2026


def _load_base():
    path = os.path.join(BASE, "code", "20_train_eval.py")
    spec = importlib.util.spec_from_file_location("base20", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    base = _load_base()
    oof = pd.read_csv(os.path.join(RES, "oof_w4.csv"))
    df, _ = base.load(4)
    y = oof["y"].values
    groups = np.asarray(df["id_student"].values)

    pairs = [("Spline-GAM", "LightGBM"), ("DecisionTree", "Spline-GAM"),
             ("RandomForest", "LightGBM"), ("Spline-GAM", "XGBoost")]
    models = sorted({m for pr in pairs for m in pr})
    P = {m: oof[f"p_{m}"].values for m in models}

    # ---- pre-build the block structure once
    order = np.argsort(groups, kind="stable")
    g_sorted = groups[order]
    # numpy returns (unique, index, counts) in the order of the flags
    uniq, starts, counts = np.unique(g_sorted, return_index=True,
                                     return_counts=True)
    n_clusters = len(uniq)
    rows_sorted = order                       # rows in learner-block order
    mult_template = np.repeat(np.arange(len(counts)), counts)  # block id per row

    rng = np.random.default_rng(SEED)
    boots = {pr: [] for pr in pairs}
    used = 0
    print(f"[D-fast] paired cluster bootstrap, {N_BOOT} reps, "
          f"{n_clusters} learners", flush=True)
    for b in range(N_BOOT):
        draw = rng.integers(0, n_clusters, n_clusters)      # learners w/ replacement
        w = np.bincount(draw, minlength=n_clusters)
        if w.sum() < 2:
            continue
        mult = w[mult_template]                              # multiplicity per row
        idx = rows_sorted[np.repeat(np.arange(len(rows_sorted)), mult)]
        yy = y[idx]
        if yy.min() == yy.max():
            continue
        auc = {m: roc_auc_score(yy, P[m][idx]) for m in models}
        for pr in pairs:
            boots[pr].append(auc[pr[0]] - auc[pr[1]])
        used += 1
        if (b + 1) % 500 == 0:
            print(f"  {b + 1}/{N_BOOT} reps", flush=True)

    out = {"n_boot": N_BOOT, "n_boot_used": used, "tests": {}}
    for a, b_ in pairs:
        obs = roc_auc_score(y, P[a]) - roc_auc_score(y, P[b_])
        bs = np.asarray(boots[(a, b_)])
        lo, hi = np.percentile(bs, [2.5, 97.5])
        p_neg = float((bs <= 0).mean())
        p_pos = float((bs >= 0).mean())
        out["tests"][f"{a}_minus_{b_}"] = {
            "delta_auc": float(obs), "ci95": [float(lo), float(hi)],
            "p_bootstrap": float(min(1.0, 2 * min(p_neg, p_pos))),
            "frac_boot_positive": p_pos,
            "significant": bool(lo > 0 or hi < 0)}

    names = list(out["tests"])
    ps = np.array([out["tests"][n]["p_bootstrap"] for n in names])
    order_p = np.argsort(ps)
    m = len(ps)
    adj = ps.copy()
    running = 0.0
    for rank, i in enumerate(order_p):
        running = max(running, min(1.0, (m - rank) * ps[i]))
        adj[i] = running
    for n, p_adj in zip(names, adj):
        t = out["tests"][n]
        t["p_holm"] = float(p_adj)
        t["significant_holm"] = bool(p_adj < 0.05)
        print(f"  {n:34s} dAUC={t['delta_auc']:+.4f} "
              f"CI[{t['ci95'][0]:+.4f},{t['ci95'][1]:+.4f}] "
              f"p_holm={t['p_holm']:.4f}", flush=True)

    json.dump(out, open(os.path.join(RES, "paired_tests.json"), "w"), indent=2)
    print("  -> results/paired_tests.json", flush=True)


if __name__ == "__main__":
    main()
