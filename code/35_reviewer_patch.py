"""Supplementary analyses requested by the adversarial review rehearsal.

1. fold-wise AUC spread for the reported week-4 predictions
2. ablation without module / presentation identifiers (LightGBM, Spline-GAM)
3. Spline-GAM hyperparameter sweep (knots, ridge strength)
4. MLP baseline (one hidden-layer pair) on the same folds
5. non-silent sub-cohort and deployment-threshold fairness are in 34_*.py

Writes results/review_patch.json
"""
import importlib.util, json, os, pickle, time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, SplineTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("te", os.path.join(BASE, "code", "20_train_eval.py"))
te = importlib.util.module_from_spec(spec)
spec.loader.exec_module(te)

OUT = {}
df, feats = te.load(4)
y = df["y"].values
groups = df["id_student"].values
p = np.load(os.path.join(BASE, "results", "oof_w4.npy"))


def fold_auc(pred, seed=0):
    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    a = []
    for tr, tes in skf.split(df[feats], y, groups):
        a.append(roc_auc_score(y[tes], pred[tes]))
    return [float(x) for x in a]


t0 = time.time()
OUT["fold_auc"] = fold_auc(p)
OUT["fold_auc_mean"] = float(np.mean(OUT["fold_auc"]))
OUT["fold_auc_range"] = float(max(OUT["fold_auc"]) - min(OUT["fold_auc"]))
print("fold AUC", [f"{x:.4f}" for x in OUT["fold_auc"]], f"range {OUT['fold_auc_range']:.4f}", flush=True)

# ---------------------------------------------------------- 2. no identifier
NOID = [c for c in feats if c not in ("code_module", "code_presentation")]
print(f"\nfeatures total {len(feats)} -> no-id {len(NOID)}", flush=True)
no = {}
for name, seeds in [("LightGBM", [0, 1, 2]), ("Spline-GAM", [0, 1])]:
    aucs = []
    for s in seeds:
        o = te.run_cv(df, NOID, name, s)
        aucs.append(roc_auc_score(y, o))
        print(f"  {name} seed{s} AUC {aucs[-1]:.4f} ({time.time()-t0:.0f}s)", flush=True)
    no[name] = {"auc_mean": float(np.mean(aucs)), "auc_seeds": [float(a) for a in aucs],
                "auc_sd": float(np.std(aucs))}
OUT["no_id"] = no

# also LightGBM without only code_presentation (module kept)
JUSTPRES = [c for c in feats if c != "code_presentation"]
o = te.run_cv(df, JUSTPRES, "LightGBM", 0)
OUT["no_presentation_id"] = {"auc": float(roc_auc_score(y, o))}
print(f"  no presentation id: AUC {OUT['no_presentation_id']['auc']:.4f}", flush=True)

# --------------------------------------------------- 3. GAM hyperparameter sweep
num = [c for c in feats if c not in te.CAT]
present = [c for c in te.CAT if c in feats]


def gam_model(knots, C, degree=3):
    pre = ColumnTransformer([
        ("spl", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("spl", SplineTransformer(n_knots=knots, degree=degree,
                                                    include_bias=False)),
                          ("sc", StandardScaler())]), num),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), present)])
    return Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=3000, C=C))])


skf = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0).split(df[feats], y, groups))
tr, tes = skf[0]
sweep = {}
for knots, C in [(5, 0.5), (8, 0.5), (12, 0.5), (8, 0.25), (8, 1.0), (8, 2.0)]:
    m = gam_model(knots, C)
    m.fit(df.iloc[tr][feats], y[tr])
    auc = roc_auc_score(y[tes], m.predict_proba(df.iloc[tes][feats])[:, 1])
    sweep[f"knots{knots}_C{C}"] = float(auc)
    print(f"  GAM {knots} knots C={C}: {auc:.4f} ({time.time()-t0:.0f}s)", flush=True)
OUT["gam_sweep"] = {"aucs": sweep,
                    "range": float(max(sweep.values()) - min(sweep.values()))}

# ------------------------------------------------------------ 4. MLP baseline
mlp_pre = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("sc", StandardScaler())]), num),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), present)])
mlp_aucs = []
for s in [0, 1]:
    oof = np.zeros(len(df))
    for tr, tes in skf:
        m = Pipeline([("pre", mlp_pre),
                      ("clf", MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-3,
                                            learning_rate_init=1e-3, max_iter=400,
                                            early_stopping=True, n_iter_no_change=20,
                                            random_state=s))])
        m.fit(df.iloc[tr][feats], y[tr])
        oof[tes] = m.predict_proba(df.iloc[tes][feats])[:, 1]
    mlp_aucs.append(roc_auc_score(y, oof))
    print(f"  MLP seed{s} AUC {mlp_aucs[-1]:.4f} ({time.time()-t0:.0f}s)", flush=True)
OUT["mlp"] = {"auc_mean": float(np.mean(mlp_aucs)), "auc_seeds": [float(a) for a in mlp_aucs]}

with open(os.path.join(BASE, "results", "review_patch.json"), "w") as f:
    json.dump(OUT, f, indent=2)
print("saved review_patch.json", f"{time.time()-t0:.0f}s total")
