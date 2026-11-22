import pickle, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df = pickle.load(open(BASE + r"\data\features\feat_w4.pkl", "rb"))
p = np.load(BASE + r"\results\oof_w4.npy")
y = df["final_result"].isin(["Fail", "Withdrawn"]).astype(int).values
silent = (df["vle_clicks"].values == 0)
n = len(df)

print(f"n={n}  base rate={y.mean():.4f}  silent={silent.mean()*100:.2f}%  "
      f"silent at-risk={y[silent].mean()*100:.2f}%")

# overall AUC vs non-silent AUC
print(f"AUC all          = {roc_auc_score(y, p):.4f}")
print(f"AUC non-silent   = {roc_auc_score(y[~silent], p[~silent]):.4f}")
print(f"AUC silent only  = {roc_auc_score(y[silent], p[silent]):.4f}")

# top-quintile rule
k = int(round(0.20 * n))
thr = np.sort(p)[::-1][k - 1]
flag = p >= thr
print(f"\ntop-20%: cutoff {thr:.4f}  flagged {flag.sum()} ({flag.mean()*100:.1f}%)")
print(f"  precision {y[flag].mean()*100:.1f}%  recall {y[flag].sum()/y.sum()*100:.1f}%")
print(f"  silent inside flagged: {silent[flag].mean()*100:.1f}%  "
      f"(silent share of cohort {silent.mean()*100:.1f}%)")
ns = ~silent
print(f"  within non-silent: precision {y[flag&ns].mean()*100:.1f}%  "
      f"recall {y[flag&ns].sum()/y[ns].sum()*100:.1f}%  "
      f"(top-20% of non-silent cohort = {int(round(0.20*ns.sum()))} rows)")

# same rule applied only to the non-silent cohort
idx_ns = np.where(ns)[0]
k2 = int(round(0.20 * len(idx_ns)))
thr2 = np.sort(p[idx_ns])[::-1][k2 - 1]
flag2 = np.zeros(n, bool); flag2[idx_ns[p[idx_ns] >= thr2]] = True
print(f"  top-quintile OF the non-silent cohort: precision {y[flag2].mean()*100:.1f}%  "
      f"recall {y[flag2].sum()/y.sum()*100:.1f}% of all, "
      f"{y[flag2].sum()/y[ns].sum()*100:.1f}% of non-silent positives")

# simple rule baseline: silent => at risk
print(f"\nzero-click rule alone: precision {y[silent].mean()*100:.1f}%  "
      f"recall {y[silent].sum()/y.sum()*100:.1f}% of all positives")

# ---- fairness at deployment operating points (disability) ----
dis = df["disability"].astype(bool).values
print(f"\ndisability share {dis.mean()*100:.1f}%")


def fpr_tpr(mask, flagged):
    a, b = mask, ~mask
    fpr_a = ((flagged & a) & (y == 0)).sum() / max((a & (y == 0)).sum(), 1)
    fpr_b = ((flagged & b) & (y == 0)).sum() / max((b & (y == 0)).sum(), 1)
    tpr_a = ((flagged & a) & (y == 1)).sum() / max((a & (y == 1)).sum(), 1)
    tpr_b = ((flagged & b) & (y == 1)).sum() / max((b & (y == 1)).sum(), 1)
    return fpr_a, fpr_b, tpr_a, tpr_b


for name, f in [("p>=0.5", p >= 0.5), ("top-20%", flag)]:
    fa, fb, ta, tb = fpr_tpr(dis, f)
    print(f"{name:8s} FPR {fa*100:.1f} vs {fb*100:.1f} gap {(fa-fb)*100:+.1f} | "
          f"TPR {ta*100:.1f} vs {tb*100:.1f} gap {(ta-tb)*100:+.1f} | coverage {f.mean()*100:.1f}%")

# 80% recall operating point
order = np.argsort(p)[::-1]
cum = np.cumsum(y[order]) / y.sum()
k80 = int(np.searchsorted(cum, 0.80) + 1)
thr80 = p[order][k80 - 1]
f80 = p >= thr80
fa, fb, ta, tb = fpr_tpr(dis, f80)
print(f"80%-rec FPR {fa*100:.1f} vs {fb*100:.1f} gap {(fa-fb)*100:+.1f} | "
      f"TPR {ta*100:.1f} vs {tb*100:.1f} gap {(ta-tb)*100:+.1f} | coverage {f80.mean()*100:.1f}%")
