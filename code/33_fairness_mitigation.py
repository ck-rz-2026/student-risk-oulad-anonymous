"""
Fairness mitigation at week 4 (post-processing, group-specific thresholds).

Baseline (paper, at p >= 0.5) shows a ~10-point false-positive gap between
disabled and non-disabled registrations, while AUC is essentially equal
(0.848 vs 0.849).  This script asks the obvious follow-up question:

    can the *decision*, rather than the score, be made fairer,
    and what does it cost in recall / coverage?

Protocol (cross-fitted, so no threshold is tuned on the rows it scores):
  1. split learners (not rows) 50/50 into calibration / evaluation halves,
     repeated over 5 seeds;
  2. on the calibration half compute (a) the global 0.5 baseline, (b) group
     thresholds that equalise FPR across the levels of a protected attribute
     to the pooled FPR, (c) group thresholds that equalise TPR likewise;
  3. apply those thresholds to the evaluation half, report per-level TPR/FPR,
     the max-min gap, and the pooled recall / precision / coverage.

Each protected attribute is handled as its own decision rule (never OR-ed
across attributes, which would silently over-flag every row that belongs to
several groups).

Outputs:
  results/fairness_mitigation.json
  results/tables/T4_fairness_mitigation.csv
  figures/fig15_fairness_mitigation.png / .pdf

Run: python code/33_fairness_mitigation.py
"""
import importlib.util
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(BASE, "results")
FIG = os.path.join(BASE, "figures")
TAB = os.path.join(RES, "tables")

rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "legend.fontsize": 6.8, "xtick.labelsize": 7.2, "ytick.labelsize": 7.2,
    "axes.linewidth": 0.7, "savefig.dpi": 400, "figure.dpi": 150,
})

MODELS = ["LightGBM", "Spline-GAM"]
PROTECTED = ["disability", "gender", "age_band", "imd_band"]
HEADLINE = "disability"
SEEDS = [11, 22, 33, 44, 55]
FPR_TARGETS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40]


def _load_base():
    path = os.path.join(BASE, "code", "20_train_eval.py")
    spec = importlib.util.spec_from_file_location("base20", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def rates(y, flag):
    y = np.asarray(y, dtype=int)
    flag = np.asarray(flag, dtype=bool)
    pos, neg = y == 1, y == 0
    return {
        "n": int(len(y)),
        "tpr": float(flag[pos].mean()) if pos.any() else np.nan,
        "fpr": float(flag[neg].mean()) if neg.any() else np.nan,
        "flag_rate": float(flag.mean()),
        "base_rate": float(y.mean()),
    }


def thr_at_fpr(neg_scores, target_fpr):
    if len(neg_scores) == 0:
        return 0.5
    return float(np.quantile(neg_scores, 1.0 - target_fpr))


def thr_at_tpr(pos_scores, target_tpr):
    if len(pos_scores) == 0:
        return 0.5
    return float(np.quantile(pos_scores, 1.0 - target_tpr))


def aggregate(recs):
    keys = [k for k in recs[0] if isinstance(recs[0][k], (int, float, np.floating))]
    out = {}
    for k in keys:
        v = np.asarray([r[k] for r in recs], dtype=float)
        out[k] = {"mean": float(np.nanmean(v)), "sd": float(np.nanstd(v, ddof=1))}
    return out


def gap(level_dict, key):
    vals = [d[key] for d in level_dict.values() if np.isfinite(d[key])]
    return float(max(vals) - min(vals)) if vals else 0.0


def main():
    base = _load_base()
    df, _ = base.load(4)
    oof = pd.read_csv(os.path.join(RES, "oof_w4.csv"))
    y_all = df["y"].values.astype(int)
    stud = df["id_student"].values
    uniq_stud = pd.unique(stud)
    attrs = {g: df[g].astype(str).values for g in PROTECTED}

    out = {"protocol": {
        "split": "learners 50/50 calibration/evaluation, 5 seeds",
        "thresholds": "estimated on the calibration half only",
        "baseline": "global p >= 0.5",
        "fpr_targets": FPR_TARGETS}}

    for model in MODELS:
        p_all = oof[f"p_{model}"].values
        rule_recs = {}          # rule -> list of per-seed records
        attr_recs = {}          # attribute -> list of per-seed records (FPR-eq)
        curve = {"fpr_target": FPR_TARGETS,
                 "global": {t: [] for t in FPR_TARGETS},
                 "groupwise": {t: [] for t in FPR_TARGETS}}

        def add_rule(name, flag_e, ye, masks_e, prefix=""):
            rec = {"recall": float(flag_e[ye == 1].mean()),
                   "precision": float((ye[flag_e] == 1).mean()) if flag_e.any() else 0.0,
                   "coverage": float(flag_e.mean())}
            for g in PROTECTED:
                sub = {lv: rates(ye[m], flag_e[m]) for lv, m in masks_e[g].items()}
                rec[f"fpr_gap_{g}"] = gap(sub, "fpr")
                rec[f"tpr_gap_{g}"] = gap(sub, "tpr")
            rule_recs.setdefault(prefix + name, []).append(rec)
            return rec

        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            perm = rng.permutation(len(uniq_stud))
            cal = set(uniq_stud[perm[: len(perm) // 2]])
            is_cal = np.isin(stud, list(cal))
            is_eval = ~is_cal
            pc, yc = p_all[is_cal], y_all[is_cal]
            pe, ye = p_all[is_eval], y_all[is_eval]

            masks_c = {g: {lv: (attrs[g][is_cal] == lv) for lv in np.unique(attrs[g])}
                       for g in PROTECTED}
            masks_e = {g: {lv: (attrs[g][is_eval] == lv) for lv in np.unique(attrs[g])}
                       for g in PROTECTED}

            pooled_fpr = float((pc[yc == 0] >= 0.5).mean())
            pooled_tpr = float((pc[yc == 1] >= 0.5).mean())

            # ---------- baseline
            add_rule("global_0.5", pe >= 0.5, ye, masks_e)

            # ---------- per-attribute equalised decision rules
            for g in PROTECTED:
                t_f, t_t = {}, {}
                for lv, m in masks_c[g].items():
                    t_f[lv] = thr_at_fpr(pc[m & (yc == 0)], pooled_fpr)
                    t_t[lv] = thr_at_tpr(pc[m & (yc == 1)], pooled_tpr)
                f_f = np.zeros(len(pe), dtype=bool)
                f_t = np.zeros(len(pe), dtype=bool)
                for lv, m in masks_e[g].items():
                    f_f[m] |= pe[m] >= t_f[lv]
                    f_t[m] |= pe[m] >= t_t[lv]
                rec_f = add_rule(f"fpr_eq:{g}", f_f, ye, masks_e)
                rec_t = add_rule(f"tpr_eq:{g}", f_t, ye, masks_e)
                if g == HEADLINE:
                    rule_recs.setdefault("fpr_eq", []).append(
                        rule_recs.pop(f"fpr_eq:{g}")[-1])
                    rule_recs.setdefault("tpr_eq", []).append(
                        rule_recs.pop(f"tpr_eq:{g}")[-1])
                # store per-level rates for the appendix view
                lv_rates_f = {lv: rates(ye[m], f_f[m]) for lv, m in masks_e[g].items()}
                lv_rates_t = {lv: rates(ye[m], f_t[m]) for lv, m in masks_e[g].items()}
                base_rates = {lv: rates(ye[m], (pe >= 0.5)[m]) for lv, m in masks_e[g].items()}
                attr_recs.setdefault(g, []).append({
                    "fpr_eq": {"levels": lv_rates_f, "coverage": float(f_f.mean()),
                               "recall": rec_f["recall"]},
                    "tpr_eq": {"levels": lv_rates_t, "coverage": float(f_t.mean()),
                               "recall": rec_t["recall"]},
                    "baseline_levels": base_rates})

            # ---------- cost curve on the headline attribute
            m_c = masks_c[HEADLINE]
            m_e = masks_e[HEADLINE]
            for t in FPR_TARGETS:
                g_thr = {lv: thr_at_fpr(pc[m & (yc == 0)], t) for lv, m in m_c.items()}
                flag_g = np.zeros(len(pe), dtype=bool)
                for lv, m in m_e.items():
                    flag_g[m] |= pe[m] >= g_thr[lv]
                flag_s = pe >= thr_at_fpr(pc[yc == 0], t)
                curve["global"][t].append(float(flag_s[ye == 1].mean()))
                curve["groupwise"][t].append(float(flag_g[ye == 1].mean()))

        out[model] = {
            "rules": {k: aggregate(v) for k, v in rule_recs.items()},
            "by_attribute": {g: {"fpr_eq": aggregate([r["fpr_eq"] for r in v]),
                                 "tpr_eq": aggregate([r["tpr_eq"] for r in v])}
                             for g, v in attr_recs.items()},
            "per_level": {g: {"baseline": {lv: aggregate([r["baseline_levels"][lv] for r in v])
                                           for lv in v[0]["baseline_levels"]},
                              "fpr_eq": {lv: aggregate([r["fpr_eq"]["levels"][lv] for r in v])
                                         for lv in v[0]["fpr_eq"]["levels"]},
                              "tpr_eq": {lv: aggregate([r["tpr_eq"]["levels"][lv] for r in v])
                                         for lv in v[0]["tpr_eq"]["levels"]}}
                         for g, v in attr_recs.items()},
            "cost_curve": {
                "fpr_target": FPR_TARGETS,
                "recall_global_thr": [float(np.mean(curve["global"][t])) for t in FPR_TARGETS],
                "recall_group_thr": [float(np.mean(curve["groupwise"][t])) for t in FPR_TARGETS]},
        }

        b = out[model]["rules"]["global_0.5"]
        f = out[model]["rules"].get("fpr_eq")
        t_ = out[model]["rules"].get("tpr_eq")
        print(f"[{model}] global 0.5 : recall={b['recall']['mean']:.3f} "
              f"cov={b['coverage']['mean']:.3f} "
              f"FPRgap(dis)={b['fpr_gap_disability']['mean']*100:.1f}pt", flush=True)
        if f:
            print(f"[{model}] FPR-equal. : recall={f['recall']['mean']:.3f} "
                  f"cov={f['coverage']['mean']:.3f} "
                  f"FPRgap(dis)={f['fpr_gap_disability']['mean']*100:.1f}pt "
                  f"TPRgap(dis)={f['tpr_gap_disability']['mean']*100:.1f}pt", flush=True)
        if t_:
            print(f"[{model}] TPR-equal. : recall={t_['recall']['mean']:.3f} "
                  f"cov={t_['coverage']['mean']:.3f} "
                  f"FPRgap(dis)={t_['fpr_gap_disability']['mean']*100:.1f}pt", flush=True)

    with open(os.path.join(RES, "fairness_mitigation.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print("  -> results/fairness_mitigation.json", flush=True)

    # ---------------------------------------------------------------- table T4
    rows = []
    for model in MODELS:
        for rule, label in (("global_0.5", "Global p >= 0.5"),
                            ("fpr_eq", "Equalised FPR (disability)"),
                            ("tpr_eq", "Equalised TPR (disability)")):
            r = out[model]["rules"][rule]
            rows.append({
                "Model": model, "Decision rule": label,
                "Recall": f"{r['recall']['mean']:.3f}",
                "Precision": f"{r['precision']['mean']:.3f}",
                "Coverage": f"{r['coverage']['mean']:.3f}",
                "FPR gap dis.": f"{r['fpr_gap_disability']['mean']*100:.1f} pt",
                "TPR gap dis.": f"{r['tpr_gap_disability']['mean']*100:.1f} pt",
                "FPR gap gender": f"{r['fpr_gap_gender']['mean']*100:.1f} pt",
                "FPR gap age": f"{r['fpr_gap_age_band']['mean']*100:.1f} pt",
            })
    t4 = pd.DataFrame(rows)
    os.makedirs(TAB, exist_ok=True)
    t4.to_csv(os.path.join(TAB, "T4_fairness_mitigation.csv"), index=False)
    print("  -> results/tables/T4_fairness_mitigation.csv", flush=True)

    # ---------------------------------------------------------------- fig 15
    # two stacked panels so the figure reads at single-column width
    fig, axes = plt.subplots(2, 1, figsize=(3.4, 3.5), constrained_layout=True)
    lv = ["False", "True"]
    cats = [("fpr", "False", "FPR\nno dis."), ("fpr", "True", "FPR\ndisabled"),
            ("tpr", "False", "TPR\nno dis."), ("tpr", "True", "TPR\ndisabled")]
    x = np.arange(len(cats))
    w = 0.36
    ax = axes[0]
    for i, (rule, col, lab) in enumerate((("baseline", "#bbbbbb", "global p>=0.5"),
                                          ("fpr_eq", "#1f77b4", "equalised FPR"))):
        vals = [out["LightGBM"]["per_level"]["disability"][rule][g][k]["mean"] * 100
                for (k, g, _) in cats]
        bars = ax.bar(x + (i - 0.5) * w, vals, w, color=col, label=lab,
                      yerr=[out["LightGBM"]["per_level"]["disability"][rule][g][k]["sd"] * 100
                            for (k, g, _) in cats], capsize=1.5,
                      error_kw={"elinewidth": 0.5, "ecolor": "#444444"})
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.1f}",
                    ha="center", va="bottom", fontsize=5.6)
    ax.set_xticks(x, [c[2] for c in cats])
    ax.set_ylabel("Rate (%)")
    ax.set_ylim(0, 95)
    ax.set_title("(a) Error rates, before vs after group thresholds")
    ax.legend(frameon=False, ncol=2, loc="upper left", fontsize=6)

    cc = out["LightGBM"]["cost_curve"]
    ax = axes[1]
    ax.plot(np.array(cc["fpr_target"]) * 100, np.array(cc["recall_global_thr"]) * 100,
            "o--", color="#555555", lw=0.9, ms=4, mfc="none", label="single global threshold")
    ax.plot(np.array(cc["fpr_target"]) * 100, np.array(cc["recall_group_thr"]) * 100,
            "s-", color="#1f77b4", lw=1.0, ms=3.4, label="equalised FPR")
    max_d = float(np.max(np.abs(np.subtract(cc["recall_global_thr"],
                                             cc["recall_group_thr"])) * 100))
    ax.text(0.97, 0.06, f"curves coincide (max diff {max_d:.1f} pt)",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=5.8,
            bbox=dict(fc="white", ec="none", alpha=0.8))
    ax.set_xlabel("Enforced FPR ceiling per group (%)")
    ax.set_ylabel("Pooled recall (%)")
    ax.set_title("(b) Price of equalising the decision")
    ax.legend(frameon=False, loc="upper left", fontsize=6)

    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"fig15_fairness_mitigation.{ext}"),
                    bbox_inches="tight", dpi=400)
    plt.close(fig)
    print("  saved fig15_fairness_mitigation", flush=True)


if __name__ == "__main__":
    main()
