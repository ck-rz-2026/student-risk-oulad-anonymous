"""Generate all paper figures from result files (300 dpi, serif, print-ready)."""
import json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib import rcParams

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(BASE, "results")
FIG = os.path.join(BASE, "figures")
os.makedirs(FIG, exist_ok=True)

rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8,
    "axes.titlesize": 8.5,
    "axes.labelsize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "axes.linewidth": 0.7,
    "savefig.dpi": 400,
    "figure.dpi": 150,
})
C = {"LR": "#7f7f7f", "Spline-GAM": "#1f77b4", "DecisionTree": "#bbbbbb",
     "RandomForest": "#2ca02c", "XGBoost": "#d62728", "LightGBM": "#ff7f0e"}


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".png"), bbox_inches="tight", dpi=400)
    fig.savefig(os.path.join(FIG, name + ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


# ---------------------------------------------------------------- framework
def framework():
    fig, ax = plt.subplots(figsize=(7.0, 2.35))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.2)
    ax.axis("off")
    boxes = [
        (0.15, 1.15, 1.55, 0.95, "OULAD\nraw tables"),
        (2.05, 1.15, 1.65, 0.95, "Cutoff filter\n$t\\leq 7W$"),
        (4.05, 1.15, 1.85, 0.95, "Feature blocks\nD | R | V | A"),
        (6.25, 1.15, 1.75, 0.95, "Model family\nLR / GAM / trees"),
        (8.35, 1.15, 1.50, 0.95, "Risk score\n$\\hat{p}$"),
    ]
    for x, y, w, h, txt in boxes:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06,rounding_size=0.12",
                                    linewidth=0.8, edgecolor="#333333", facecolor="#f2f2f2"))
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=7.6)
    for x0 in [1.72, 3.72, 5.92, 8.02]:
        ax.add_patch(FancyArrowPatch((x0, 1.62), (x0 + 0.30, 1.62), arrowstyle="-|>",
                                     mutation_scale=8, linewidth=0.8, color="#333333"))
    # lower row: evaluation + explanation
    low = [
        (2.05, 0.10, 2.35, 0.72, "Grouped CV / cross-module\ntransfer / subgroup audit"),
        (4.75, 0.10, 2.55, 0.72, "TreeSHAP: global, local,\nstability, what-if"),
        (7.55, 0.10, 2.35, 0.72, "Actionable risk profile\nfor advisors"),
    ]
    for x, y, w, h, txt in low:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06,rounding_size=0.12",
                                    linewidth=0.7, edgecolor="#666666", facecolor="#ffffff",
                                    linestyle="--"))
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=7.2, color="#222222")
    ax.annotate("", xy=(3.6, 1.12), xytext=(3.6, 0.84),
                arrowprops=dict(arrowstyle="-|>", lw=0.8, color="#666666"))
    ax.annotate("", xy=(7.1, 1.12), xytext=(7.1, 0.84),
                arrowprops=dict(arrowstyle="-|>", lw=0.8, color="#666666"))
    save(fig, "fig1_framework")


# ---------------------------------------------------------------- sweep
def sweep_curve():
    d = json.load(open(os.path.join(RES, "sweep.json")))
    weeks = [1, 2, 3, 4, 6, 8, 12]
    fig, ax = plt.subplots(figsize=(3.45, 2.25))
    for m in ["LR", "Spline-GAM", "DecisionTree", "XGBoost", "LightGBM"]:
        if m not in d:
            continue
        mu = [d[m][str(w)]["auc"][0] for w in weeks]
        sd = [d[m][str(w)]["auc"][1] for w in weeks]
        ax.plot(weeks, mu, marker="o", ms=2.8, lw=1.0, color=C[m], label=m)
        if m in ("LightGBM", "Spline-GAM"):
            ax.fill_between(weeks, np.array(mu) - np.array(sd), np.array(mu) + np.array(sd),
                            color=C[m], alpha=0.12, lw=0)
    ax.axvline(4, color="#999999", lw=0.6, ls=":")
    ax.text(4.15, 0.885, "operational\ncutoff", fontsize=6.4, color="#555555")
    ax.set_xlabel("Cutoff week $W$ after module start")
    ax.set_ylabel("AUC (out-of-fold)")
    ax.set_xticks(weeks)
    ax.set_ylim(0.725, 0.94)
    ax.legend(loc="lower right", frameon=False, ncol=1, handlelength=1.4,
              borderaxespad=0.3)
    ax.grid(alpha=0.18, lw=0.5)
    save(fig, "fig2_sweep")


# ---------------------------------------------------------------- model bars
def model_bars():
    main = json.load(open(os.path.join(RES, "main_w4.json")))
    extra = json.load(open(os.path.join(RES, "metrics_extra.json")))
    models = ["DecisionTree", "LR", "Spline-GAM", "RandomForest", "LightGBM", "XGBoost"]
    fig, ax = plt.subplots(figsize=(3.45, 1.85))
    ypos = np.arange(len(models))[::-1]
    vals = [main[m]["metrics"]["auc"][0] for m in models]
    lo = [extra[m]["auc_ci"][1] for m in models]
    hi = [extra[m]["auc_ci"][2] for m in models]
    xerr = [np.array(vals) - np.array(lo), np.array(hi) - np.array(vals)]
    ax.barh(ypos, vals, color=[C[m] for m in models], height=0.6, edgecolor="#333333",
            lw=0.4, xerr=xerr, error_kw={"elinewidth": 0.6, "capsize": 1.4, "ecolor": "#333333"})
    for y, v in zip(ypos, vals):
        ax.text(v + 0.002, y, f"{v:.3f}", va="center", ha="left", fontsize=6.2)
    ax.set_yticks(ypos)
    ax.set_yticklabels(models, fontsize=6.4)
    ax.set_xlabel("AUC (out-of-fold, 95% bootstrap CI)")
    ax.set_xlim(0.785, 0.868)
    ax.grid(axis="x", alpha=0.2, lw=0.4)
    ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig, "fig3_models")


# ---------------------------------------------------------------- beeswarm
def beeswarm():
    z = np.load(os.path.join(RES, "shap_values.npz"), allow_pickle=True)
    sv, X, feats = z["sv"], z["X"], z["feats"]
    imp = pd.read_csv(os.path.join(RES, "shap_global.csv"))
    top = imp.head(12)["feature"].tolist()
    idx = [np.where(feats == f)[0][0] for f in top]
    fig, ax = plt.subplots(figsize=(3.45, 2.9))
    rng = np.random.default_rng(0)
    for row, (f, j) in enumerate(zip(top[::-1], idx[::-1])):
        s = sv[:, j]
        order = np.argsort(X[:, j])
        # subsample for legibility
        take = order[rng.choice(len(order), size=min(1200, len(order)), replace=False)]
        xs, cs = s[take], X[take, j].astype(float)
        qlo, qhi = np.percentile(cs, [2, 98])
        cn = np.clip((cs - qlo) / (qhi - qlo + 1e-9), 0, 1)
        jit = rng.normal(0, 0.10, len(xs))
        ax.scatter(xs, np.full(len(xs), row) + jit, c=cn, cmap="coolwarm", s=2.2,
                   alpha=0.65, lw=0)
        ax.plot([np.percentile(s, 2), np.percentile(s, 98)], [row, row],
                color="#222222", lw=0.8, zorder=3)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top[::-1], fontsize=6.6)
    ax.axvline(0, color="#888888", lw=0.6, ls=":")
    ax.set_xlabel("SHAP value (impact on log-odds of unsuccessful outcome)")
    ax.grid(axis="x", alpha=0.15, lw=0.4)
    ax.set_axisbelow(True)
    sm = plt.cm.ScalarMappable(cmap="coolwarm")
    cb = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("feature value (low $\\rightarrow$ high)", fontsize=6.2)
    cb.ax.tick_params(labelsize=6)
    save(fig, "fig4_beeswarm")


# ---------------------------------------------------------------- dependence
def dependence():
    z = np.load(os.path.join(RES, "shap_values.npz"), allow_pickle=True)
    sv, X, feats = z["sv"], z["X"], z["feats"]
    panels = [("vle_active_days", "Active VLE days (first 4 weeks)"),
              ("asmt_mean_score", "Mean score of submitted assessments"),
              ("vle_days_since_last", "Days since last VLE activity"),
              ("vle_active_weeks", "Active weeks in the VLE")]
    fig, axes = plt.subplots(2, 2, figsize=(3.45, 3.05))
    for ax, (f, lab) in zip(axes.ravel(), panels):
        if f not in feats:
            ax.set_visible(False)
            continue
        j = int(np.where(feats == f)[0][0])
        x = X[:, j].astype(float)
        y = sv[:, j]
        m = np.ones(len(x), bool)
        if f == "asmt_mean_score":
            m = x >= 0
        ax.scatter(x[m], y[m], s=0.5, alpha=0.3, color="#1f77b4", lw=0, rasterized=True)
        # binned conditional mean
        q = np.quantile(x[m], np.linspace(0, 1, 21))
        q = np.unique(q)
        mids, means = [], []
        for a, b in zip(q[:-1], q[1:]):
            sel = (x[m] >= a) & (x[m] <= b)
            if sel.sum() > 20:
                mids.append(np.mean(x[m][sel]))
                means.append(np.mean(y[m][sel]))
        ax.plot(mids, means, color="#d62728", lw=1.0)
        ax.axhline(0, color="#999999", lw=0.5, ls=":")
        ax.set_xlabel(lab, fontsize=5.8)
        ax.set_ylabel("SHAP value", fontsize=5.8)
        ax.tick_params(labelsize=5.2, length=2.0, width=0.5, pad=1.5)
        ax.grid(alpha=0.15, lw=0.35)
        ax.set_axisbelow(True)
    fig.tight_layout(w_pad=1.2, h_pad=1.0)
    save(fig, "fig5_dependence")


# ---------------------------------------------------------------- waterfall
def waterfall():
    cases = json.load(open(os.path.join(RES, "local_cases.json")))
    order = [k for k in ["withdrawn_high_risk", "withdrawn_missed", "pass_borderline"]
             if k in cases]
    fig, axes = plt.subplots(len(order), 1, figsize=(3.45, 3.6))
    if len(order) == 1:
        axes = [axes]
    titles = {"withdrawn_high_risk": "(a) withdrawn, correctly flagged",
              "withdrawn_missed": "(b) withdrawn, missed",
              "pass_borderline": "(c) pass, borderline"}
    for ax, k in zip(np.atleast_1d(axes), order):
        c = cases[k]
        items = sorted(c["top_contrib"].items(), key=lambda kv: abs(kv[1]))[::-1][:8][::-1]
        names = [i[0] for i in items]
        vals = [i[1] for i in items]
        colors = ["#d62728" if v > 0 else "#1f77b4" for v in vals]
        ax.barh(range(len(vals)), vals, color=colors, height=0.6, edgecolor="#333333", lw=0.3)
        ax.set_yticks(range(len(vals)))
        ax.set_yticklabels(names, fontsize=6.2)
        ax.axvline(0, color="#333333", lw=0.5)
        ax.set_title(f"{titles.get(k, k)}   $\\hat{{p}}$={c['p']:.2f}, y={c['y']}",
                     fontsize=7.2, pad=2)
        ax.tick_params(axis="x", labelsize=6)
        ax.set_xlabel("SHAP contribution (log-odds)", fontsize=6.5)
    fig.tight_layout(h_pad=0.8)
    save(fig, "fig6_waterfall")


# ---------------------------------------------------------------- fairness
def fairness():
    fa = json.load(open(os.path.join(RES, "fairness.json")))
    fig, axes = plt.subplots(2, 1, figsize=(3.45, 3.4),
                             gridspec_kw={"height_ratios": [1.55, 1]})
    ax = axes[0]
    labels, vals, errs, colors = [], [], [], []
    palette = {"gender": "#1f77b4", "disability": "#ff7f0e", "age_band": "#2ca02c",
               "imd_band": "#d62728"}
    for key, sub in fa.items():
        if key == "module":
            continue
        for lv, s in sub.items():
            labels.append(f"{key}:{lv}")
            vals.append(s["auc"])
            errs.append(0)
            colors.append(palette.get(key, "#777777"))
    ypos = np.arange(len(labels))[::-1]
    ax.barh(ypos, vals, color=colors, height=0.65, edgecolor="#333333", lw=0.3, alpha=0.9)
    for y, v in zip(ypos, vals):
        ax.text(v + 0.002, y, f"{v:.3f}", va="center", fontsize=5.8)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=5.4)
    ax.tick_params(axis="x", labelsize=5.4, length=2.0, width=0.5, pad=1.5)
    ax.set_xlim(min(vals) - 0.02, max(vals) + 0.02)
    ax.set_xlabel("AUC within subgroup", fontsize=5.8)
    ax.grid(axis="x", alpha=0.15, lw=0.4)
    ax.set_axisbelow(True)

    # calibration
    ax2 = axes[1]
    oof = pd.read_csv(os.path.join(RES, "oof_w4.csv"))
    p, yv = oof["p_LightGBM"].values, oof["y"].values
    bins = np.linspace(0, 1, 11)
    idx = np.clip(np.digitize(p, bins) - 1, 0, 9)
    xs, ys = [], []
    for b in range(10):
        m = idx == b
        if m.sum() > 50:
            xs.append(p[m].mean())
            ys.append(yv[m].mean())
    ax2.plot([0, 1], [0, 1], color="#999999", lw=0.7, ls=":")
    ax2.plot(xs, ys, "o-", ms=2.2, lw=0.9, color="#1f77b4")
    ax2.set_xlabel("Predicted risk", fontsize=5.8)
    ax2.set_ylabel("Observed rate", fontsize=5.8)
    ax2.tick_params(labelsize=5.2, length=2.0, width=0.5, pad=1.5)
    ax2.grid(alpha=0.15, lw=0.35)
    ax2.set_axisbelow(True)
    fig.tight_layout(h_pad=1.0)
    save(fig, "fig7_fairness")


# ---------------------------------------------------------------- ablation
def ablation():
    ab = json.load(open(os.path.join(RES, "ablation.json")))
    blocks = [b for b in ["D", "D+R", "D+R+V", "D+R+V+A", "R+V+A"] if b in ab]
    labels = {"D": "D", "D+R": "D+R", "D+R+V": "D+R+V", "D+R+V+A": "D+R+V+A",
              "R+V+A": "R+V+A\n(no demo.)"}
    fig, ax = plt.subplots(figsize=(3.45, 1.95))
    w = 0.36
    for i, m in enumerate(["LR", "LightGBM"]):
        vals = [ab[b][m]["auc"][0] for b in blocks]
        ax.bar(np.arange(len(blocks)) + (i - 0.5) * w, vals, width=w, color=C[m],
               edgecolor="#333333", lw=0.3, label=m)
        for x, v in zip(np.arange(len(blocks)) + (i - 0.5) * w, vals):
            ax.text(x, v + 0.004, f"{v:.3f}", ha="center", fontsize=5.6, rotation=90)
    ax.set_xticks(range(len(blocks)))
    ax.set_xticklabels([labels[b] for b in blocks], fontsize=6.4)
    ax.set_ylim(0.55, 0.90)
    ax.set_ylabel("AUC")
    ax.legend(frameon=False, fontsize=6.5, loc="upper left")
    ax.grid(axis="y", alpha=0.15, lw=0.4)
    ax.set_axisbelow(True)
    save(fig, "fig8_ablation")


# ---------------------------------------------------------------- transfer
def transfer():
    tr = json.load(open(os.path.join(RES, "transfer.json")))
    mods = [k for k in tr if k != "overall"]
    fig, ax = plt.subplots(figsize=(3.45, 1.8))
    aucs = [tr[m]["auc"] for m in mods]
    base = [tr[m]["base_rate"] for m in mods]
    ax.bar(range(len(mods)), aucs, color="#2ca02c", edgecolor="#333333", lw=0.3, width=0.62)
    for x, v in enumerate(aucs):
        ax.text(x, v + 0.004, f"{v:.2f}", ha="center", fontsize=6)
    ax.set_xticks(range(len(mods)))
    ax.set_xticklabels(mods)
    ax.set_ylabel("AUC on unseen module")
    ax.set_ylim(0.63, 0.96)
    ax2 = ax.twinx()
    ax2.plot(range(len(mods)), base, "o--", color="#d62728", ms=2.5, lw=0.7, label="base rate")
    ax2.set_ylabel("at-risk rate", fontsize=7)
    ax2.set_ylim(0, 1)
    ax2.legend(frameon=False, fontsize=6, loc="upper left")
    save(fig, "fig9_transfer")


# ---------------------------------------------------------------- stability
def stability():
    st = json.load(open(os.path.join(RES, "stability.json")))
    d = st["cutoff_rank_corr"]
    weeks = [1, 2, 3, 4, 6, 8]
    M = np.ones((len(weeks), len(weeks)))
    for k, v in d.items():
        a, b = [int(x) for x in k.split("-")]
        i, j = weeks.index(a), weeks.index(b)
        M[i, j] = M[j, i] = v
    fig, ax = plt.subplots(figsize=(2.6, 2.3))
    im = ax.imshow(M, cmap="YlGnBu", vmin=0.5, vmax=1.0)
    ax.set_xticks(range(len(weeks)))
    ax.set_xticklabels(weeks)
    ax.set_yticks(range(len(weeks)))
    ax.set_yticklabels(weeks)
    ax.set_xlabel("Cutoff week")
    ax.set_ylabel("Cutoff week")
    for i in range(len(weeks)):
        for j in range(len(weeks)):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=5.6,
                    color="black" if M[i, j] > 0.75 else "white")
    cb = fig.colorbar(im, ax=ax, fraction=0.035)
    cb.ax.tick_params(labelsize=6)
    cb.set_label("Spearman rank corr.", fontsize=6.2)
    save(fig, "fig10_stability")


if __name__ == "__main__":
    framework()
    if os.path.exists(os.path.join(RES, "sweep.json")):
        sweep_curve()
    if os.path.exists(os.path.join(RES, "metrics_extra.json")):
        model_bars()
    if os.path.exists(os.path.join(RES, "shap_values.npz")):
        beeswarm()
        dependence()
    if os.path.exists(os.path.join(RES, "local_cases.json")):
        waterfall()
    if os.path.exists(os.path.join(RES, "fairness.json")):
        fairness()
    if os.path.exists(os.path.join(RES, "ablation.json")):
        ablation()
    if os.path.exists(os.path.join(RES, "transfer.json")):
        transfer()
    if os.path.exists(os.path.join(RES, "stability.json")):
        stability()
    print("figures done")
