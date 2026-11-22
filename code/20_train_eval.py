"""
Training / evaluation pipeline.

Protocol
--------
* Rows: 32,593 module registrations.
* Grouping: GroupKFold on id_student (a learner never appears in both
  train and test), repeated with several seeds.
* Metrics: AUC, AUPRC, Brier, ECE, F1, precision/recall at 0.5,
  recall within the top-10% flagged students (Recall@10), lift.
"""
import json, os, time, warnings
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, SplineTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
import lightgbm as lgb
import xgboost as xgb

warnings.filterwarnings("ignore")
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEAT = os.path.join(BASE, "data", "features")
RES = os.path.join(BASE, "results")
os.makedirs(RES, exist_ok=True)

CAT = ["gender", "region", "highest_education", "imd_band", "age_band",
       "disability", "code_module", "code_presentation"]
TARGET = "final_result"
POS = {"Fail", "Withdrawn"}


LEAKY = {"date_unregistration", "final_result", "id_student", "row_id",
         "y", "cutoff_week", "key"}


def load(w):
    df = pd.read_pickle(os.path.join(FEAT, f"feat_w{w}.pkl"))
    df["y"] = df[TARGET].isin(POS).astype(int)
    feats = [c for c in df.columns if df[c].dtype.kind in "ifb" and c not in LEAKY]
    for c in CAT:
        df[c] = df[c].astype("string").fillna("NA").astype("category")
        if c not in feats:
            feats.append(c)
    feats = [c for c in feats if c not in LEAKY]
    assert not (set(feats) & LEAKY), set(feats) & LEAKY
    return df, feats


def ece(y, p, bins=10):
    idx = np.clip((p * bins).astype(int), 0, bins - 1)
    e = 0.0
    for b in range(bins):
        m = idx == b
        if m.sum() == 0:
            continue
        e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)


def evaluate(y, p, thr=0.5, flag=0.10):
    out = {}
    out["auc"] = roc_auc_score(y, p)
    out["auprc"] = average_precision_score(y, p)
    out["brier"] = float(np.mean((p - y) ** 2))
    out["ece"] = ece(y, p)
    yhat = (p >= thr).astype(int)
    out["f1"] = f1_score(y, yhat, zero_division=0)
    n_top = max(1, int(round(flag * len(y))))
    top = np.argsort(-p)[:n_top]
    out["recall@10"] = float(y[top].sum() / max(1, y.sum()))
    out["precision@10"] = float(y[top].mean())
    out["lift@10"] = float(y[top].mean() / y.mean())
    return out


# ---------------------------------------------------------------- models
def make_tree_prep(feats):
    return feats  # trees consume categorical dtypes directly


def model_zoo(feats, seed=0):
    present = [c for c in CAT if c in feats]
    cat_idx = [feats.index(c) for c in present]
    lgbm = lgb.LGBMClassifier(
        n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=40,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
        reg_lambda=1.0, random_state=seed, n_jobs=-1, verbose=-1)
    xgbm = xgb.XGBClassifier(
        n_estimators=400, learning_rate=0.05, max_depth=5, min_child_weight=5,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        tree_method="hist", enable_categorical=True, random_state=seed,
        n_jobs=-1, eval_metric="logloss")
    rf = RandomForestClassifier(
        n_estimators=200, min_samples_leaf=10, max_features="sqrt",
        n_jobs=-1, random_state=seed)
    dt = DecisionTreeClassifier(max_depth=4, min_samples_leaf=20, random_state=seed)

    num = [c for c in feats if c not in CAT]
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), num),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), present),
    ])
    lr = Pipeline([("pre", pre),
                   ("clf", LogisticRegression(max_iter=2000, C=1.0))])
    spline_pre = ColumnTransformer([
        ("spl", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("spl", SplineTransformer(n_knots=8, degree=3,
                                                    include_bias=False)),
                          ("sc", StandardScaler())]), num),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), present),
    ])
    gam = Pipeline([("pre", spline_pre),
                    ("clf", LogisticRegression(max_iter=3000, C=0.5))])
    return {"LR": lr, "Spline-GAM": gam, "DecisionTree": dt,
            "RandomForest": rf, "XGBoost": xgbm, "LightGBM": lgbm}


PROB = {"LR": "predict_proba", "Spline-GAM": "predict_proba",
        "DecisionTree": "predict_proba", "RandomForest": "predict_proba",
        "XGBoost": "predict_proba", "LightGBM": "predict_proba"}


TREE_MODELS = ("DecisionTree", "RandomForest", "XGBoost", "LightGBM")


def encode_trees(df, feats):
    """Ordinal integer codes for categorical columns (tree consumers)."""
    X = df[feats].copy()
    for c in CAT:
        if c in X.columns:
            X[c] = X[c].cat.codes.astype("int16")
    return X


def run_cv(df, feats, model_name, seed, n_splits=5):
    y = df["y"].values
    groups = df["id_student"].values
    X_lin = df[feats]
    X_tree = encode_trees(df, feats) if model_name in TREE_MODELS else None
    X = X_tree if X_tree is not None else X_lin
    cat_idx = ([feats.index(c) for c in CAT if c in feats]
               if model_name in TREE_MODELS else None)
    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(df))
    for tr, te in skf.split(X, y, groups):
        m = model_zoo(feats, seed=seed)[model_name]
        if model_name == "LightGBM":
            m.fit(X.iloc[tr], y[tr], categorical_feature=cat_idx)
        else:
            m.fit(X.iloc[tr], y[tr])
        oof[te] = m.predict_proba(X.iloc[te])[:, 1]
    return oof


def main_table():
    df, feats = load(4)
    models = ["DecisionTree", "LR", "Spline-GAM", "RandomForest", "XGBoost", "LightGBM"]
    seeds = [0, 1, 2]
    out = {}
    preds = {}
    for name in models:
        t0 = time.time()
        rows = []
        oofs = []
        for s in seeds:
            oof = run_cv(df, feats, name, s)
            oofs.append(oof)
            rows.append(evaluate(df["y"].values, oof))
        preds[name] = np.mean(oofs, axis=0)
        agg = {k: (float(np.mean([r[k] for r in rows])),
                   float(np.std([r[k] for r in rows]))) for k in rows[0]}
        out[name] = {"metrics": agg, "seconds": time.time() - t0}
        print(f"{name:14s} AUC={agg['auc'][0]:.4f}+-{agg['auc'][1]:.4f} "
              f"AUPRC={agg['auprc'][0]:.4f} Brier={agg['brier'][0]:.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)
    np.save(os.path.join(RES, "oof_w4.npy"), preds["LightGBM"])
    pd.DataFrame({"y": df["y"].values,
                  **{f"p_{k}": v for k, v in preds.items()}}).to_csv(
        os.path.join(RES, "oof_w4.csv"), index=False)
    np.save(os.path.join(RES, "oof_all_w4.npy"), np.vstack([preds[m] for m in models]))
    with open(os.path.join(RES, "main_w4.json"), "w") as f:
        json.dump(out, f, indent=2)
    return out


def sweep():
    out = {}
    for w in [1, 2, 3, 4, 6, 8, 12]:
        df, feats = load(w)
        for name in ["LR", "Spline-GAM", "DecisionTree", "LightGBM", "XGBoost"]:
            oofs = [run_cv(df, feats, name, s) for s in (0,)]
            rows = [evaluate(df["y"].values, o) for o in oofs]
            agg = {k: (float(np.mean([r[k] for r in rows])),
                       float(np.std([r[k] for r in rows]))) for k in rows[0]}
            out.setdefault(name, {})[w] = agg
        print(f"week {w}: LGBM AUC={out['LightGBM'][w]['auc'][0]:.4f} "
              f"LR AUC={out['LR'][w]['auc'][0]:.4f}", flush=True)
    with open(os.path.join(RES, "sweep.json"), "w") as f:
        json.dump(out, f, indent=2)


BLOCKS = {
    "D": ["gender", "region", "highest_education", "imd_band", "age_band",
          "disability", "num_of_prev_attempts", "studied_credits"],
    "D+R": None, "D+R+V": None, "D+R+V+A": None,
}


def block_features(df, feats, block):
    if block == "D":
        return [c for c in feats if c in BLOCKS["D"]]
    if block == "D+R":
        return [c for c in feats if c in BLOCKS["D"] or
                c in ("date_registration", "registered_late")]
    if block == "D+R+V":
        return [c for c in feats if c in BLOCKS["D"] or
                c in ("date_registration", "registered_late", "code_module",
                      "code_presentation") or
                c.startswith(("clk_", "vle_", "wk_"))]
    if block == "R+V+A":
        return [c for c in feats if c not in BLOCKS["D"]]
    return feats


def ablation():
    df, feats = load(4)
    out = {}
    for block in ["D", "D+R", "D+R+V", "D+R+V+A", "R+V+A"]:
        f = block_features(df, feats, block)
        for name in ["LR", "LightGBM"]:
            oofs = [run_cv(df, f, name, s) for s in (0, 1)]
            rows = [evaluate(df["y"].values, o) for o in oofs]
            agg = {k: (float(np.mean([r[k] for r in rows])),
                       float(np.std([r[k] for r in rows]))) for k in rows[0]}
            out.setdefault(block, {})[name] = agg
        print(f"{block:8s} n_feat={len(f):3d} LGBM AUC={out[block]['LightGBM']['auc'][0]:.4f}",
              flush=True)
    with open(os.path.join(RES, "ablation.json"), "w") as f:
        json.dump(out, f, indent=2)


def transfer():
    """Train on six modules, test on the held-out module (unseen course)."""
    df, feats = load(4)
    out = {}
    oof = np.zeros(len(df))
    for mod in sorted(df["code_module"].astype(str).unique()):
        tr = df["code_module"].astype(str) != mod
        te = ~tr
        model = model_zoo(feats)["LightGBM"]
        model.fit(df.loc[tr, feats], df.loc[tr, "y"])
        oof[te.values] = model.predict_proba(df.loc[te, feats])[:, 1]
        met = evaluate(df.loc[te, "y"].values, oof[te.values])
        out[mod] = {"n": int(te.sum()), "auc": met["auc"],
                    "auprc": met["auprc"], "f1": met["f1"],
                    "base_rate": float(df.loc[te, "y"].mean())}
        print(f"module {mod}: n={te.sum():5d} base={out[mod]['base_rate']:.3f} "
              f"AUC={met['auc']:.4f}", flush=True)
    out["overall"] = evaluate(df["y"].values, oof)
    np.save(os.path.join(RES, "oof_transfer.npy"), oof)
    with open(os.path.join(RES, "transfer.json"), "w") as f:
        json.dump(out, f, indent=2)


def fairness():
    df, feats = load(4)
    p = np.load(os.path.join(RES, "oof_w4.npy"))
    y = df["y"].values
    out = {}
    groups = {
        "gender": df["gender"].astype(str),
        "disability": df["disability"].astype(str),
        "age_band": df["age_band"].astype(str),
        "imd_band": np.where(df["imd_band"].astype(str).isin(
            ["0-10%", "10-20%", "20-30%"]), "low(0-30%)",
            np.where(df["imd_band"].astype(str).isin(["NA", "nan"]), "NA", "high(30-100%)")),
        "module": df["code_module"].astype(str),
    }
    for name, g in groups.items():
        g = pd.Series(g)
        sub = {}
        for lv in sorted(g.unique()):
            m = (g == lv).values
            if m.sum() < 200 or len(np.unique(y[m])) < 2:
                continue
            pr = p[m]
            sub[lv] = {"n": int(m.sum()), "base_rate": float(y[m].mean()),
                       "auc": float(roc_auc_score(y[m], pr)),
                       "mean_pred": float(pr.mean()),
                       "tpr": float((pr[y[m] == 1] >= 0.5).mean()),
                       "fpr": float((pr[y[m] == 0] >= 0.5).mean()),
                       "calib_gap": float(abs(y[m].mean() - pr.mean()))}
        out[name] = sub
        best = max(sub.items(), key=lambda kv: kv[1]["auc"]) if sub else None
        print(f"{name}: groups={len(sub)} auc range "
              f"{min(v['auc'] for v in sub.values()):.3f}-{max(v['auc'] for v in sub.values()):.3f}",
              flush=True)
    with open(os.path.join(RES, "fairness.json"), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    import sys
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    if stage in ("main", "all"):
        main_table()
    if stage in ("sweep", "all"):
        sweep()
    if stage in ("ablation", "all"):
        ablation()
    if stage in ("transfer", "all"):
        transfer()
    if stage in ("fairness", "all"):
        fairness()
    print("done", stage)
