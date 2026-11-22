import os
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import json, os
import numpy as np
import pandas as pd

RES = os.path.join(BASE, "results")
z = np.load(os.path.join(RES, "shap_values.npz"), allow_pickle=True)
sv, X, feats = z["sv"], z["X"], [str(f) for f in z["feats"]]
imp = pd.read_csv(os.path.join(RES, "shap_global.csv"))
top = imp.head(12)
rows = []
for _, r in top.iterrows():
    j = feats.index(r["feature"])
    x, s = X[:, j].astype(float), sv[:, j]
    corr = np.corrcoef(x, s)[0, 1] if x.std() > 0 else 0.0
    rows.append((r["feature"], r["mean_abs_shap"], r["group"], corr))
print(f"{'feature':22s} {'|SHAP|':>7s} {'group':14s} corr(x, shap)")
for f, v, g, c in rows:
    print(f"{f:22s} {v:7.3f} {g:14s} {c:+.3f}")
json.dump([{"feature": f, "mean_abs": float(v), "group": g, "corr": float(c)} for f, v, g, c in rows],
          open(os.path.join(RES, "shap_direction.json"), "w"), indent=2)

# group-level share
g = imp.groupby("group")["mean_abs_shap"].sum()
print("\nshare by group:")
print((g / g.sum()).round(3).to_string())

# conditional risk anchors from oof
oof = pd.read_csv(os.path.join(RES, "oof_w4.csv"))
df = pd.read_pickle(os.path.join(BASE, "data", "features", "feat_w4.pkl"))
anchor = {}
p = oof["p_LightGBM"].values
at_risk = df["final_result"].isin(["Fail", "Withdrawn"]).values
for name, m in [("inactive", df["vle_active_days"] <= 4),
                ("active", df["vle_active_days"] >= 23),
                ("nosub", (df["asmt_n_due"].fillna(0) > 0) & (df["asmt_has_sub"] == 0)),
                ("nodue", df["asmt_n_due"].fillna(0) == 0)]:
    anchor[name] = {"risk": float(p[m].mean()), "observed": float(at_risk[m].mean()), "n": int(m.sum())}
json.dump(anchor, open(os.path.join(RES, "risk_anchors.json"), "w"), indent=2)
print("\nanchors:", json.dumps(anchor, indent=2))
for lo, hi in [(0, 4), (5, 9), (10, 15), (16, 22), (23, 28)]:
    m = (df["vle_active_days"] >= lo) & (df["vle_active_days"] <= hi)
    print(f"  {lo:2d}-{hi:2d}: n={m.sum():5d} risk={oof['p_LightGBM'][m].mean():.3f} actual={df['final_result'].isin(['Fail','Withdrawn'])[m].mean():.3f}")
print("\nmean predicted risk by assessment status:")
nodue = df["asmt_n_due"].fillna(0) == 0
print("  no assessment due:", round(oof['p_LightGBM'][nodue].mean(), 3), int(nodue.sum()))
m = (df["asmt_n_due"] > 0) & (df["asmt_has_sub"] == 0)
print("  due but none submitted:", round(oof['p_LightGBM'][m].mean(), 3), int(m.sum()))
m = df["asmt_has_sub"] == 1
for lo, hi in [(0, 50), (50, 65), (65, 80), (80, 101)]:
    mm = m & (df["asmt_mean_score"] >= lo) & (df["asmt_mean_score"] < hi)
    print(f"  submitted, score {lo}-{hi}: risk={oof['p_LightGBM'][mm].mean():.3f} n={int(mm.sum())}")
