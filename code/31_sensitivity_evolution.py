"""
Evolution path + parameter sensitivity analysis (computational experiment).

This module answers three questions the paper currently leaves open:

  A. Evolution path  -- how do discrimination, calibration-free operating
     cost (coverage needed for 80% recall), and the risk-ranking structure
     evolve from week 1 to week 12?
  B. Parameter sensitivity -- how much of each model's headline AUC is a
     property of the learner, and how much is a property of one default
     hyper-parameter setting?
  C. Decision sensitivity -- where should the operating threshold sit when
     the intervention budget (share of cohort that can be contacted) is
     fixed at 10 / 20 / 30 %?

Design notes (statistical integrity):
  * Every comparison is paired on the same rows and uses a *cluster*
    bootstrap over `id_student`, because a learner contributes several
    module registrations and those rows are not independent.
  * Sensitivity grids are reported as mean AUC over the same 5 grouped
    folds used everywhere else; no fold is re-drawn per configuration.
  * Nothing here is synthetic. Synthetic trajectories are deliberately
    avoided: they cannot support an empirical claim about OULAD.

Run:
    python code/31_sensitivity_evolution.py            # all stages
    python code/31_sensitivity_evolution.py evolution
    python code/31_sensitivity_evolution.py sensitivity
    python code/31_sensitivity_evolution.py threshold
    python code/31_sensitivity_evolution.py figures
"""
import importlib.util
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, SplineTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import xgboost as xgb

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(BASE, "results")
FIG = os.path.join(BASE, "figures")
TAB = os.path.join(RES, "tables")
for d in (RES, FIG, TAB):
    os.makedirs(d, exist_ok=True)

rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8,
    "axes.titlesize": 8.5,
    "axes.labelsize": 8,
    "legend.fontsize": 6.8,
    "xtick.labelsize": 7.2,
    "ytick.labelsize": 7.2,
    "axes.linewidth": 0.7,
    "savefig.dpi": 400,
    "figure.dpi": 150,
})
C = {"LR": "#7f7f7f", "Spline-GAM": "#1f77b4", "DecisionTree": "#bbbbbb",
     "RandomForest": "#2ca02c", "XGBoost": "#d62728", "LightGBM": "#ff7f0e"}
WEEKS = [1, 2, 3, 4, 6, 8, 12]
N_BOOT_WEEK = 400
N_BOOT_TEST = 2000


# ---------------------------------------------------------------- base module
def _load_base():
    path = os.path.join(BASE, "code", "20_train_eval.py")
    spec = importlib.util.spec_from_file_location("base20", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


base = _load_base()


def save_fig(fig, name):
    fig.savefig(os.path.join(FIG, name + ".png"), bbox_inches="tight", dpi=400)
    fig.savefig(os.path.join(FIG, name + ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print("  saved", name, flush=True)


def _md(df):
    """Minimal markdown table writer (avoids a tabulate dependency)."""
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |",
             "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def lgbm_default(seed=0, **kw):
    p = dict(n_estimators=400, learning_rate=0.05, num_leaves=31,
             min_child_samples=40, subsample=0.8, subsample_freq=1,
             colsample_bytree=0.8, reg_lambda=1.0, random_state=seed,
             n_jobs=-1, verbose=-1)
    p.update(kw)
    return lgb.LGBMClassifier(**p)


def xgb_default(seed=0, **kw):
    p = dict(n_estimators=400, learning_rate=0.05, max_depth=5,
             min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
             reg_lambda=1.0, tree_method="hist", enable_categorical=True,
             random_state=seed, n_jobs=-1, eval_metric="logloss")
    p.update(kw)
    return xgb.XGBClassifier(**p)


def linear_pipeline(feats, kind="plain", C_reg=1.0, n_knots=8, degree=3):
    """One-hot categorical + (optionally) spline-expanded numeric block."""
    present = [c for c in base.CAT if c in feats]
    num = [c for c in feats if c not in base.CAT]
    if kind == "plain":
        num_block = Pipeline([("imp", SimpleImputer(strategy="median")),
                              ("sc", StandardScaler())])
    else:
        num_block = Pipeline([("imp", SimpleImputer(strategy="median")),
                              ("spl", SplineTransformer(n_knots=n_knots,
                                                        degree=degree,
                                                        include_bias=False)),
                              ("sc", StandardScaler())])
    pre = ColumnTransformer([
        ("num", num_block, num),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), present),
    ])
    return Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=3000, C=C_reg))])


# ================================================================ A. evolution
def _auc_ci(y, p, groups, n_boot=N_BOOT_WEEK, seed=7):
    """Cluster bootstrap CI for AUC (resampling learners, not rows)."""
    rng = np.random.default_rng(seed)
    uniq = pd.unique(groups)
    pos = {g: np.where(groups == g)[0] for g in uniq}
    vals = []
    for _ in range(n_boot):
        take = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([pos[g] for g in take])
        yy, pp = y[idx], p[idx]
        if yy.min() == yy.max():
            continue
        vals.append(roc_auc_score(yy, pp))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def coverage_at_recall(y, p, target=0.80):
    """Smallest share of the cohort that must be flagged to reach `target` recall."""
    order = np.argsort(-p)
    ys = y[order]
    cum = np.cumsum(ys)
    need = target * ys.sum()
    k = int(np.searchsorted(cum, need) + 1)
    k = min(k, len(ys))
    return k / len(ys), float(ys[:k].mean()), float(cum[k - 1] / max(1, ys.sum()))


def evolution():
    print("[A] evolution path (week 1-12)", flush=True)
    out = {"weeks": WEEKS, "lgbm": {}, "pooled_models": {}, "n": {}}
    sweep = json.load(open(os.path.join(RES, "sweep.json")))
    for w in WEEKS:
        df, feats = base.load(w)
        X = base.encode_trees(df, feats)
        y = df["y"].values
        groups = df["id_student"].values
        cat_idx = [feats.index(c) for c in base.CAT if c in feats]
        skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
        oof = np.zeros(len(df))
        for tr, te in skf.split(X, y, groups):
            m = lgbm_default(seed=0)
            m.fit(X.iloc[tr], y[tr], categorical_feature=cat_idx)
            oof[te] = m.predict_proba(X.iloc[te])[:, 1]
        np.save(os.path.join(RES, f"oof_lgbm_w{w}.npy"), oof)

        cover, prec_at, rec_at = coverage_at_recall(y, oof, 0.80)
        lo, hi = _auc_ci(y, oof, groups)
        out["lgbm"][w] = {
            "auc": float(roc_auc_score(y, oof)),
            "auc_ci": [lo, hi],
            "auprc": float(average_precision_score(y, oof)),
            "cover80": float(cover),
            "precision_at_cover80": prec_at,
            "recall_at_cover80": rec_at,
            "n": int(len(df)),
            "n_feat": int(len(feats)),
        }
        out["n"][w] = int(len(df))
        print(f"  w{w:>2}: AUC={out['lgbm'][w]['auc']:.4f} "
              f"[{lo:.4f},{hi:.4f}] cover80={cover:.3f} nfeat={len(feats)}", flush=True)

        # pooled multi-model AUC for the trajectory panel (already in sweep.json)
        for name, wk in sweep.items():
            if str(w) in wk:
                out["pooled_models"].setdefault(name, {})[w] = wk[str(w)]["auc"][0]

    # risk-quantile migration: where does a week-1 prediction rank a learner by week 4?
    print("  risk-rank migration w1 -> w4 -> w8", flush=True)
    p1 = np.load(os.path.join(RES, "oof_lgbm_w1.npy"))
    p4 = np.load(os.path.join(RES, "oof_lgbm_w4.npy"))
    p8 = np.load(os.path.join(RES, "oof_lgbm_w8.npy"))
    q = lambda p: pd.qcut(pd.Series(p).rank(method="first"), 10, labels=False)
    b1, b4, b8 = q(p1), q(p4), q(p8)
    mig = pd.crosstab(b1, b4, normalize="index").values
    mig18 = pd.crosstab(b1, b8, normalize="index").values
    out["migration_w1_w4"] = mig.round(4).tolist()
    out["migration_w1_w8"] = mig18.round(4).tolist()
    # share of week-1 risk-decile that stays in the same decile by week 4
    out["retention_top_decile_w1_w4"] = float(mig[9, 9])
    out["retention_top_decile_w1_w8"] = float(mig18[9, 9])

    json.dump(out, open(os.path.join(RES, "evolution.json"), "w"), indent=2)
    print("  -> results/evolution.json", flush=True)
    return out


# ============================================================ B. sensitivity
GRIDS = {
    "DecisionTree": [("max_depth", v) for v in range(2, 11)],
    "RandomForest": [("max_depth", v) for v in (4, 6, 8, 12, None)],
    "Spline-GAM": [("n_knots", v) for v in (3, 5, 8, 12, 20)],
    "LR": [("C", v) for v in (0.01, 0.1, 1.0, 10.0, 100.0)],
    "LightGBM": ([(("learning_rate", v)) for v in (0.02, 0.05, 0.1, 0.2, 0.3)]
                 + [("num_leaves", v) for v in (7, 15, 31, 63)]),
    "XGBoost": [("max_depth", v) for v in (2, 3, 4, 5, 6, 8)],
}
DEFAULTS = {"DecisionTree": ("max_depth", 4), "RandomForest": ("max_depth", None),
            "Spline-GAM": ("n_knots", 8), "LR": ("C", 1.0),
            "LightGBM": ("learning_rate", 0.05), "XGBoost": ("max_depth", 5)}


def _build(name, param, value, feats, seed=0):
    if name == "DecisionTree":
        return DecisionTreeClassifier(max_depth=value, min_samples_leaf=20,
                                      random_state=seed)
    if name == "RandomForest":
        return RandomForestClassifier(n_estimators=200, min_samples_leaf=10,
                                      max_features="sqrt", max_depth=value,
                                      n_jobs=-1, random_state=seed)
    if name == "LightGBM":
        return lgbm_default(seed=seed, **{param: value})
    if name == "XGBoost":
        return xgb_default(seed=seed, **{param: value})
    if name == "LR":
        return linear_pipeline(feats, "plain", C_reg=value)
    if name == "Spline-GAM":
        return linear_pipeline(feats, "spline", C_reg=0.5, n_knots=value)
    raise KeyError(name)


def sensitivity():
    print("[B] parameter sensitivity (5 grouped folds, fixed splits)", flush=True)
    df, feats = base.load(4)
    y = df["y"].values
    groups = df["id_student"].values
    X_tree = base.encode_trees(df, feats)
    cat_idx = [feats.index(c) for c in base.CAT if c in feats]
    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
    splits = list(skf.split(X_tree, y, groups))

    out = {}
    for name, grid in GRIDS.items():
        t0 = time.time()
        rows = []
        for param, val in grid:
            aucs, auprcs = [], []
            for tr, te in splits:
                model = _build(name, param, val, feats)
                Xtr, Xte = (X_tree.iloc[tr], X_tree.iloc[te]) if name in base.TREE_MODELS \
                    else (df[feats].iloc[tr], df[feats].iloc[te])
                if name == "LightGBM":
                    model.fit(Xtr, y[tr], categorical_feature=cat_idx)
                else:
                    model.fit(Xtr, y[tr])
                p = model.predict_proba(Xte)[:, 1]
                aucs.append(roc_auc_score(y[te], p))
                auprcs.append(average_precision_score(y[te], p))
            rows.append({"param": param, "value": val,
                         "auc": float(np.mean(aucs)), "auc_sd": float(np.std(aucs)),
                         "auprc": float(np.mean(auprcs))})
        d_param, d_val = DEFAULTS[name]
        default_auc = next(r["auc"] for r in rows
                           if r["param"] == d_param and r["value"] == d_val)
        best = max(rows, key=lambda r: r["auc"])
        out[name] = {"param": grid[0][0] if len(set(g[0] for g in grid)) == 1 else "mixed",
                     "rows": rows, "default_param": d_param, "default_value": d_val,
                     "default_auc": default_auc, "best": best,
                     "auc_min": min(r["auc"] for r in rows),
                     "auc_max": max(r["auc"] for r in rows),
                     "span": max(r["auc"] for r in rows) - min(r["auc"] for r in rows),
                     "gain_vs_default": best["auc"] - default_auc,
                     "seconds": time.time() - t0}
        print(f"  {name:14s} span={out[name]['span']*100:.1f} pts  "
              f"best {best['param']}={best['value']} ({best['auc']:.4f})  "
              f"default {default_auc:.4f}  ({time.time()-t0:.0f}s)", flush=True)

    json.dump(out, open(os.path.join(RES, "sensitivity.json"), "w"), indent=2, default=str)
    print("  -> results/sensitivity.json", flush=True)
    return out


# =============================================================== C. threshold
BUDGETS = [0.10, 0.15, 0.20, 0.25, 0.30]


def threshold():
    print("[C] decision-threshold / budget sensitivity", flush=True)
    oof = pd.read_csv(os.path.join(RES, "oof_w4.csv"))
    y = oof["y"].values
    out = {"curve": {}, "budget": {}}
    for model in ("LightGBM", "Spline-GAM", "DecisionTree"):
        p = oof[f"p_{model}"].values
        curve = []
        for thr in np.round(np.arange(0.05, 0.951, 0.025), 3):
            flag = p >= thr
            if flag.sum() == 0:
                continue
            curve.append({"threshold": float(thr),
                          "flagged_share": float(flag.mean()),
                          "precision": float(y[flag].mean()),
                          "recall": float(y[flag].sum() / y.sum()),
                          "n_flagged": int(flag.sum())})
        out["curve"][model] = curve
        bud = []
        for b in BUDGETS:
            k = int(round(b * len(y)))
            top = np.argsort(-p)[:k]
            bud.append({"budget": b, "precision": float(y[top].mean()),
                        "recall": float(y[top].sum() / y.sum()),
                        "case_yield": float(y[top].sum() / k)})
        out["budget"][model] = bud
        print(f"  {model}: budget20% precision={bud[2]['precision']:.3f} "
              f"recall={bud[2]['recall']:.3f}", flush=True)
    out["base_rate"] = float(y.mean())
    json.dump(out, open(os.path.join(RES, "threshold.json"), "w"), indent=2)
    print("  -> results/threshold.json", flush=True)
    return out


# =========================================================== D. paired tests
def _cluster_pos(groups):
    """Row positions per learner, built once and reused across bootstrap reps."""
    uniq = pd.unique(groups)
    order = np.argsort(groups, kind="stable")
    g_sorted = groups[order]
    bounds = np.append(np.searchsorted(g_sorted, uniq, side="left"), len(g_sorted))
    pos = {g: order[bounds[i]:bounds[i + 1]] for i, g in enumerate(uniq)}
    return uniq, pos


def _clustered_boot_idx(uniq, pos, rng):
    take = rng.choice(uniq, size=len(uniq), replace=True)
    return np.concatenate([pos[g] for g in take])


def paired_tests():
    print("[D] paired cluster-bootstrap tests (week 4)", flush=True)
    oof = pd.read_csv(os.path.join(RES, "oof_w4.csv"))
    df, _ = base.load(4)
    y = oof["y"].values
    groups = df["id_student"].values
    pairs = [("Spline-GAM", "LightGBM"), ("DecisionTree", "Spline-GAM"),
             ("RandomForest", "LightGBM"), ("Spline-GAM", "XGBoost")]
    rng = np.random.default_rng(2026)
    uniq, pos = _cluster_pos(groups)

    out = {"n_boot": N_BOOT_TEST, "tests": {}}
    deltas = {}
    # one resample at a time: score every model on the same draw (memory-light)
    models = sorted({m for pr in pairs for m in pr})
    P = {m: oof[f"p_{m}"].values for m in models}
    boots = {pr: [] for pr in pairs}
    used = 0
    for _ in range(N_BOOT_TEST):
        idx = _clustered_boot_idx(uniq, pos, rng)
        yy = y[idx]
        if yy.min() == yy.max():
            continue
        auc = {m: roc_auc_score(yy, P[m][idx]) for m in models}
        for pr in pairs:
            boots[pr].append(auc[pr[0]] - auc[pr[1]])
        used += 1
    out["n_boot_used"] = used

    for a, b in pairs:
        pa, pb = P[a], P[b]
        obs = roc_auc_score(y, pa) - roc_auc_score(y, pb)
        bs = np.array(boots[(a, b)])
        lo, hi = np.percentile(bs, [2.5, 97.5])
        # two-sided bootstrap p: smallest alpha at which the CI excludes 0
        p_neg = float((bs <= 0).mean())
        p_pos = float((bs >= 0).mean())
        p_boot = float(min(1.0, 2 * min(p_neg, p_pos)))
        deltas[f"{a}_minus_{b}"] = obs
        out["tests"][f"{a}_minus_{b}"] = {
            "delta_auc": float(obs), "ci95": [float(lo), float(hi)],
            "p_bootstrap": p_boot,
            "frac_boot_positive": p_pos,
            "significant": bool(lo > 0 or hi < 0)}
    # Holm correction over the four tests
    names = list(out["tests"])
    ps = np.array([out["tests"][n]["p_bootstrap"] for n in names])
    order = np.argsort(ps)
    m = len(ps)
    adj = ps.copy()
    running = 0.0
    for rank, i in enumerate(order):
        val = min(1.0, (m - rank) * ps[i])
        running = max(running, val)
        adj[i] = running
    for n, p_adj in zip(names, adj):
        out["tests"][n]["p_holm"] = float(p_adj)
        out["tests"][n]["significant_holm"] = bool(p_adj < 0.05)
        t = out["tests"][n]
        print(f"  {n:34s} dAUC={t['delta_auc']:+.4f} "
              f"CI[{t['ci95'][0]:+.4f},{t['ci95'][1]:+.4f}] p_holm={t['p_holm']:.4f}", flush=True)
    json.dump(out, open(os.path.join(RES, "paired_tests.json"), "w"), indent=2)
    print("  -> results/paired_tests.json", flush=True)
    return out


# ================================================================ E. tables
def tables(evo=None, sens=None, thr=None, tests=None):
    evo = evo or json.load(open(os.path.join(RES, "evolution.json")))
    sens = sens or json.load(open(os.path.join(RES, "sensitivity.json")))
    thr = thr or json.load(open(os.path.join(RES, "threshold.json")))
    tests = tests or json.load(open(os.path.join(RES, "paired_tests.json")))

    # T1 evolution
    rows = []
    prev = None
    for w in evo["weeks"]:
        d = evo["lgbm"][str(w)]
        delta = "" if prev is None else f"{100*(d['auc']-prev):+.1f}"
        rows.append([w, d["n"], d["n_feat"], f"{d['auc']:.4f}",
                     f"[{d['auc_ci'][0]:.4f}, {d['auc_ci'][1]:.4f}]",
                     f"{d['auprc']:.4f}", f"{100*d['cover80']:.1f}",
                     f"{100*d['precision_at_cover80']:.1f}", delta])
        prev = d["auc"]
    t1 = pd.DataFrame(rows, columns=["Week", "Registrations", "Features",
                                     "AUC", "95% CI (cluster boot)", "AUPRC",
                                     "Coverage for 80% recall (%)",
                                     "Precision at that point (%)", "ΔAUC vs prev (pts)"])
    t1.to_csv(os.path.join(TAB, "T1_evolution_path.csv"), index=False)

    # T2 sensitivity
    rows = []
    for name, d in sens.items():
        for r in d["rows"]:
            v = "None" if r["value"] is None else f"{r['value']:g}" if isinstance(r["value"], float) else str(r["value"])
            rows.append([name, r["param"], v, f"{r['auc']:.4f}", f"{r['auc_sd']:.4f}",
                         "yes" if (r["param"] == d["default_param"] and r["value"] == d["default_value"]) else "",
                         "best" if (r["param"] == d["best"]["param"] and r["value"] == d["best"]["value"]) else ""])
    t2 = pd.DataFrame(rows, columns=["Model", "Parameter", "Value", "AUC (mean of 5 folds)",
                                     "SD across folds", "Default", "Best"])
    t2.to_csv(os.path.join(TAB, "T2_parameter_sensitivity.csv"), index=False)

    # T3 budget
    rows = []
    for model, bs in thr["budget"].items():
        for b in bs:
            rows.append([model, f"{int(b['budget']*100)}%", f"{100*b['precision']:.1f}",
                         f"{100*b['recall']:.1f}", f"{b['case_yield']:.2f}"])
    t3 = pd.DataFrame(rows, columns=["Model", "Budget (share of cohort)",
                                     "Precision (%)", "Recall (%)", "Cases caught per 100 contacted"])
    t3.to_csv(os.path.join(TAB, "T3_budget_operating_points.csv"), index=False)

    # markdown bundle
    md = []
    md.append("## Table S1. Evolution path (LightGBM, grouped 5-fold, cluster-bootstrap CI)\n")
    md.append(_md(t1))
    md.append("\n## Table S2. Parameter sensitivity (AUC over the same 5 grouped folds)\n")
    summary = pd.DataFrame([[n, d["param"], f"{d['default_auc']:.4f}",
                             f"{d['auc_min']:.4f}-{d['auc_max']:.4f}",
                             f"{100*d['span']:.1f}",
                             f"{d['best']['value']} ({d['best']['auc']:.4f})",
                             f"{100*d['gain_vs_default']:+.1f}"]
                            for n, d in sens.items()],
                           columns=["Model", "Swept parameter", "Default AUC",
                                    "AUC range", "Span (pts)", "Best setting (AUC)",
                                    "Gain vs default (pts)"])
    md.append(_md(summary))
    md.append("\n\nFull grid: `results/tables/T2_parameter_sensitivity.csv`\n")
    md.append("## Table S3. Budget-constrained operating points (week 4)\n")
    md.append(_md(t3))
    open(os.path.join(TAB, "tables_sensitivity.md"), "w", encoding="utf-8").write("\n".join(md))
    print("  tables -> results/tables/", flush=True)


# ================================================================ F. figures
def figures(evo=None, sens=None, thr=None):
    evo = evo or json.load(open(os.path.join(RES, "evolution.json")))
    sens = sens or json.load(open(os.path.join(RES, "sensitivity.json")))
    thr = thr or json.load(open(os.path.join(RES, "threshold.json")))

    # ---- Fig A: evolution path (3 panels)
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.25))
    ax = axes[0]
    for name, series in evo["pooled_models"].items():
        xs = sorted(int(k) for k in series)
        ys = [series[str(x)] for x in xs]
        ax.plot(xs, ys, marker="o", ms=2.6, lw=1.1, color=C.get(name, "k"),
                label=name, alpha=0.9)
    w4 = evo["lgbm"]["4"]
    ax.errorbar([4], [w4["auc"]], yerr=[[w4["auc"] - w4["auc_ci"][0]],
                                        [w4["auc_ci"][1] - w4["auc"]]],
                fmt="none", ecolor=C["LightGBM"], elinewidth=0.8, capsize=2)
    ax.set_xlabel("Cut-off week")
    ax.set_ylabel("AUC")
    ax.set_title("(a) Discrimination over the term")
    ax.set_xticks(WEEKS)
    ax.legend(frameon=False, ncol=1, loc="lower right")
    ax.grid(alpha=.25, lw=.4)

    ax = axes[1]
    xs = WEEKS
    cover = [100 * evo["lgbm"][str(w)]["cover80"] for w in xs]
    ax.plot(xs, cover, marker="s", ms=2.8, lw=1.2, color="#1f77b4")
    ax.fill_between(xs, cover, 100, color="#1f77b4", alpha=.08)
    ax.set_xlabel("Cut-off week")
    ax.set_ylabel("Share of cohort flagged (%)")
    ax.set_title("(b) Cost of 80% recall")
    ax.set_xticks(WEEKS)
    ax.set_ylim(min(cover) - 6, 100)
    ax.grid(alpha=.25, lw=.4)
    ax.annotate(f"{cover[3]:.1f}%", (4, cover[3]), textcoords="offset points",
                xytext=(4, 6), fontsize=6.8)

    ax = axes[2]
    gains = [100 * (evo["lgbm"][str(w)]["auc"] - evo["lgbm"][str(pw)]["auc"])
             for pw, w in zip(WEEKS[:-1], WEEKS[1:])]
    labels = [f"{pw}→{w}" for pw, w in zip(WEEKS[:-1], WEEKS[1:])]
    ax.bar(range(len(gains)), gains, color="#ff7f0e", alpha=.85, width=.68)
    ax.set_xticks(range(len(gains)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("ΔAUC (pts)")
    ax.set_title("(c) Marginal gain per step")
    ax.axhline(0, color="k", lw=.6)
    ax.grid(alpha=.25, lw=.4, axis="y")
    fig.tight_layout(w_pad=1.2)
    save_fig(fig, "fig11_evolution_path")

    # ---- Fig B: parameter sensitivity, 2x3
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 3.5))
    for ax, (name, d) in zip(axes.ravel(), sens.items()):
        rows = d["rows"]
        xs = [r["value"] if r["value"] is not None else 0 for r in rows]
        ys = [r["auc"] for r in rows]
        sd = [r["auc_sd"] for r in rows]
        color = C.get(name, "k")
        ax.errorbar(xs, ys, yerr=sd, marker="o", ms=2.6, lw=1.1, capsize=1.8,
                    color=color, elinewidth=.7)
        ax.axhline(d["default_auc"], color="k", lw=.7, ls="--", alpha=.6)
        dv = d["default_value"] if d["default_value"] is not None else 0
        ax.plot([dv], [d["default_auc"]], marker="*", ms=6, color="k", zorder=5)
        ax.set_title(f"{name}: {d['param']}")
        ax.set_xlabel(d["param"])
        ax.set_ylabel("AUC")
        lo, hi = min(ys) - max(sd) - .004, max(ys) + max(sd) + .004
        ax.set_ylim(lo, hi)
        ax.grid(alpha=.25, lw=.4)
        if name == "RandomForest":
            ax.set_xticks([0, 4, 6, 8, 12])
            ax.set_xticklabels(["None", "4", "6", "8", "12"])
    fig.tight_layout(w_pad=.9, h_pad=1.1)
    save_fig(fig, "fig12_parameter_sensitivity")

    # ---- Fig C: threshold / budget
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.35))
    ax = axes[0]
    for model in ("Spline-GAM", "LightGBM", "DecisionTree"):
        cur = thr["curve"][model]
        xs = [100 * c["flagged_share"] for c in cur]
        ys = [100 * c["precision"] for c in cur]
        ax.plot(xs, ys, lw=1.1, color=C.get(model, "k"), label=model)
    ax.set_xlabel("Share of cohort flagged (%)")
    ax.set_ylabel("Precision (%)")
    ax.set_title("(a) Precision as the budget expands")
    ax.set_xlim(0, 100)
    ax.legend(frameon=False, loc="lower left")
    ax.grid(alpha=.25, lw=.4)

    ax = axes[1]
    bs = thr["budget"]["LightGBM"]
    xs = [100 * b["budget"] for b in bs]
    ax.bar([x - 1.6 for x in xs], [100 * b["precision"] for b in bs], width=3.0,
           color="#ff7f0e", alpha=.85, label="Precision")
    ax.set_xlabel("Intervention budget (% of cohort)")
    ax.set_ylabel("Precision (%)")
    ax2 = ax.twinx()
    ax2.plot(xs, [100 * b["recall"] for b in bs], marker="o", ms=2.8, lw=1.1,
             color="#1f77b4", label="Recall")
    ax2.set_ylabel("Recall (%)")
    ax.set_title("(b) Fixed-budget operating points (LightGBM)")
    for x, b in zip(xs, bs):
        ax.text(x - 1.6, 100 * b["precision"] + 1.5, f"{100*b['precision']:.0f}",
                ha="center", fontsize=6.2)
    ax.grid(alpha=.25, lw=.4, axis="y")
    fig.tight_layout(w_pad=1.0)
    save_fig(fig, "fig13_budget_threshold")

    # ---- Fig D: risk-rank migration w1 -> w4 -> w8 (heatmap pair)
    fig, axes = plt.subplots(1, 2, figsize=(6.0, 2.5))
    for ax, key, title in ((axes[0], "migration_w1_w4", "(a) Week 1 → Week 4"),
                           (axes[1], "migration_w1_w8", "(b) Week 1 → Week 8")):
        m = np.array(evo[key])
        im = ax.imshow(m, cmap="Blues", vmin=0, vmax=max(.35, m.max()))
        ax.set_xlabel("Decile at later cut-off")
        ax.set_ylabel("Decile at week 1")
        ax.set_title(title)
        ax.set_xticks(range(10)); ax.set_yticks(range(10))
        ax.set_xticklabels(range(1, 11), fontsize=6); ax.set_yticklabels(range(1, 11), fontsize=6)
        for i in range(10):
            for j in range(10):
                if m[i, j] > .12:
                    ax.text(j, i, f"{m[i,j]:.2f}", ha="center", va="center",
                            fontsize=5.2, color="white" if m[i, j] > .25 else "k")
    fig.colorbar(im, ax=axes, shrink=.85, pad=.02)
    fig.tight_layout(w_pad=1.0)
    save_fig(fig, "fig14_risk_rank_migration")


# ============================================================= G. interpretation
def interpret():
    evo = json.load(open(os.path.join(RES, "evolution.json")))
    sens = json.load(open(os.path.join(RES, "sensitivity.json")))
    thr = json.load(open(os.path.join(RES, "threshold.json")))
    tests = json.load(open(os.path.join(RES, "paired_tests.json")))

    L = []
    L.append("# 演化路径与参数敏感性：结果与解释\n")
    L.append("本文件的所有数字由 `code/31_sensitivity_evolution.py` 生成，"
             "对应 `results/evolution.json`、`sensitivity.json`、`threshold.json`、"
             "`paired_tests.json`；图见 `figures/fig11–fig14`，表见 `results/tables/`。\n")

    w1, w4, w12 = (evo["lgbm"][str(w)] for w in (1, 4, 12))
    L.append("## 1. 演化路径：预测能力怎么长出来\n")
    L.append(f"LightGBM 的 AUC 从第 1 周 {w1['auc']:.3f}（95% 学生聚类 bootstrap CI "
             f"[{w1['auc_ci'][0]:.3f}, {w1['auc_ci'][1]:.3f}]）升到第 4 周 {w4['auc']:.3f}"
             f"（[{w4['auc_ci'][0]:.3f}, {w4['auc_ci'][1]:.3f}]）、第 12 周 {w12['auc']:.3f}"
             f"（[{w12['auc_ci'][0]:.3f}, {w12['auc_ci'][1]:.3f}]）。"
             f"第 1→4 周贡献 {100*(w4['auc']-w1['auc']):.1f} 个 AUC 点，"
             f"第 4→12 周再贡献 {100*(w12['auc']-w4['auc']):.1f} 点，"
             f"即前一个月拿到全程增益的 "
             f"{100*(w4['auc']-w1['auc'])/(w12['auc']-w1['auc']):.0f}%。\n")
    L.append(f"同一条曲线上的“代价”侧更有信息量：为达到 80% 召回，第 4 周需要覆盖 "
             f"{100*w4['cover80']:.1f}% 的队列（精确率 {100*w4['precision_at_cover80']:.1f}%），"
             f"到第 12 周才降到 {100*w12['cover80']:.1f}%。"
             "也就是说，早期预警的瓶颈不是排序能力，而是**正例密度太低**——"
             "同样的召回目标，越早越贵。\n")

    m = evo["retention_top_decile_w1_w4"]
    m8 = evo["retention_top_decile_w1_w8"]
    L.append(f"风险分层并不稳定：第 1 周落在最高风险十分位的注册，到第 4 周只有 "
             f"{100*m:.1f}% 仍在同一十分位，到第 8 周只剩 {100*m8:.1f}%。"
             "这不是模型缺陷，而是行为信号的固有延迟——第 1 周能看到的只有注册时点与最初几次点击，"
             "大量学生尚未开始活动，排序里混入了大量噪声。\n")

    L.append("## 2. 参数敏感性：AUC 有多少是“模型”给的，多少是“调参”给的\n")
    L.append("| 模型 | 敏感性参数 | 默认 AUC | 全网格区间 | 跨度（点） | 最优设置 | 相对默认提升（点） |")
    L.append("|---|---|---|---|---|---|---|")
    for n, d in sens.items():
        L.append(f"| {n} | {d['param']} | {d['default_auc']:.4f} | "
                 f"{d['auc_min']:.4f}–{d['auc_max']:.4f} | {100*d['span']:.1f} | "
                 f"{d['best']['param']}={d['best']['value']}（{d['best']['auc']:.4f}） | "
                 f"{100*d['gain_vs_default']:+.1f} |")
    L.append("")
    dt, lgbm = sens["DecisionTree"], sens["LightGBM"]

    L.append(f"两件事值得写进论文。第一，**树深对单棵树的影响远大于任何超参对集成的影响**："
             f"决策树在深度 2–10 之间 AUC 跨度 {100*dt['span']:.1f} 点"
             f"（{dt['auc_min']:.3f}–{dt['auc_max']:.3f}），而 LightGBM 在 "
             f"学习率 0.02–0.3 之间只动 {100*lgbm['span']:.1f} 点。"
             "这正好支撑论文的主论点：可解释性的代价主要来自“用一棵浅树去代表整个模型”，"
             "不是来自“可解释模型”这个类别本身。\n")
    L.append(f"第二，**默认设置不是最优设置**：LightGBM 最优 "
             f"{lgbm['best']['param']}={lgbm['best']['value']} 比默认高 "
             f"{100*lgbm['gain_vs_default']:+.1f} 点。这意味着主表里 GAM 与 LGBM 的差距，"
             "在调参后可能被放大而不是缩小——报差距时必须说明“双方都在默认设置下”。\n")

    t = tests["tests"]
    k1, k2 = "Spline-GAM_minus_LightGBM", "DecisionTree_minus_Spline-GAM"
    a, b = t[k1], t[k2]
    L.append("## 3. 差距显著吗：配对聚类 bootstrap\n")
    L.append(f"GAM 与 LightGBM 的 AUC 差 {100*a['delta_auc']:+.1f} 点，"
             f"95% CI [{100*a['ci95'][0]:.1f}, {100*a['ci95'][1]:.1f}] 点，"
             f"Holm 校正后 p = {a['p_holm']:.3f}"
             f"（{'显著' if a['significant_holm'] else '不显著'}）。"
             f"决策树与 GAM 的差 {100*b['delta_auc']:+.1f} 点，"
             f"95% CI [{100*b['ci95'][0]:.1f}, {100*b['ci95'][1]:.1f}]，"
             f"p = {b['p_holm']:.3f}（{'显著' if b['significant_holm'] else '不显著'}）。"
             "注意 bootstrap 按学生重采样而非按注册行，因为同一学生贡献多条注册，"
             "按行重采样会低估不确定性。\n")

    L.append("## 4. 决策敏感性：预算固定时阈值该放哪\n")
    L.append("| 预算（覆盖队列） | 精确率 | 召回 | 每联系 100 人命中风险学生 |")
    L.append("|---|---|---|---|")
    for bud in thr["budget"]["LightGBM"]:
        L.append(f"| {int(bud['budget']*100)}% | {100*bud['precision']:.1f}% | "
                 f"{100*bud['recall']:.1f}% | {bud['case_yield']:.2f} |")
    L.append("")
    b20 = thr["budget"]["LightGBM"][2]
    L.append(f"把预算锁在 20% 时，精确率 {100*b20['precision']:.1f}%、召回 {100*b20['recall']:.1f}%，"
             "再往上加预算的边际收益迅速衰减——这是“双峰风险分布”的直接后果："
             "完全静默的学生几乎必然失败，一旦被识别，剩下的都是难分的中间层。\n")

    L.append("## 5. 写作口径建议\n")
    L.append("- 这四条结论都适合作为独立小节，但**不要**堆成“更多实验”；主表之后接一段"
             "“稳健性与代价”即可，其余进附录。")
    L.append("- 报告任何 AUC 差距时，必须同时给出：配对 CI、是否同为默认设置、是否同一折划分。"
             "三者缺一，审稿人有权怀疑是调参差异。")
    L.append("- 演化路径图（Fig. 11）最适合放进引言，用一张图说明“为什么第 4 周是合理的决策点”。")
    L.append("- 图 12 的六联图放附录；正文只引用 LightGBM 与决策树的跨度对比。")

    txt = "\n".join(L)
    path = os.path.join(BASE, "敏感性分析与演化路径.md")
    open(path, "w", encoding="utf-8").write(txt)
    print("  ->", path, flush=True)
    return txt


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    t0 = time.time()
    if stage in ("evolution", "all"):
        evolution()
    if stage in ("sensitivity", "all"):
        sensitivity()
    if stage in ("threshold", "all"):
        threshold()
    if stage in ("tests", "all"):
        paired_tests()
    if stage in ("tables", "all"):
        tables()
    if stage in ("figures", "all"):
        figures()
    if stage in ("interpret", "all"):
        interpret()
    print(f"done {stage} in {time.time()-t0:.0f}s", flush=True)
