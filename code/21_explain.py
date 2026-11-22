"""
Interpretability analyses for the LightGBM week-4 model.

Produces:
  results/shap_global.csv      mean |SHAP| per feature + group
  results/shap_values.npz      SHAP matrix on a 8k sample
  results/surrogate_rules.txt  depth-3 surrogate rules + fidelity
  results/stability.json       rank stability across folds / cutoffs
  results/whatif.json          counterfactual sensitivity
  results/local_cases.json     three local explanation cases
"""
import json, os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.tree import DecisionTreeRegressor, export_text

import sys, importlib.util
_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("train_eval", os.path.join(_here, "20_train_eval.py"))
T = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(T)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(BASE, "results")

GROUPS = {
    "Demographic": ["gender", "region", "highest_education", "imd_band", "age_band",
                    "disability", "num_of_prev_attempts", "studied_credits"],
    "Registration": ["date_registration", "registered_late", "code_module",
                     "code_presentation"],
}
GROUP_MAP = {}
for g, cols in GROUPS.items():
    for c in cols:
        GROUP_MAP[c] = g


def group_of(f):
    if f in GROUP_MAP:
        return GROUP_MAP[f]
    if f.startswith("asmt_"):
        return "Assessment"
    if f.startswith(("clk_", "vle_", "wk_")):
        return "VLE behaviour"
    return "Other"


def fit_full(w=4):
    df, feats = T.load(w)
    X = T.encode_trees(df, feats)
    cat_idx = [feats.index(c) for c in T.CAT]
    m = T.model_zoo(feats, seed=0)["LightGBM"]
    m.fit(X, df["y"], categorical_feature=cat_idx)
    return df, feats, X, m


def shap_matrix(m, X):
    contrib = m.booster_.predict(X, pred_contrib=True)
    return contrib[:, :-1], contrib[:, -1]


def global_importance(df, feats, X, m, n_sample=8000, seed=3):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(n_sample, len(X)), replace=False)
    sv, base = shap_matrix(m, X.iloc[idx])
    sv_df = pd.DataFrame(sv, columns=feats)
    imp = (sv_df.abs().mean().sort_values(ascending=False)
           .rename("mean_abs_shap").reset_index().rename(columns={"index": "feature"}))
    imp["group"] = imp["feature"].map(group_of)
    imp["mean_shap"] = sv_df.mean().reindex(imp["feature"]).values
    imp.to_csv(os.path.join(RES, "shap_global.csv"), index=False)
    np.savez_compressed(os.path.join(RES, "shap_values.npz"),
                        sv=sv.astype(np.float32), X=X.iloc[idx].to_numpy(dtype=np.float32),
                        feats=np.array(feats), idx=idx, base=float(np.mean(base)))
    print(imp.head(15).to_string(index=False))
    return imp, sv_df, idx


def surrogate(df, feats, X, m):
    p = m.predict_proba(X)[:, 1]
    tree = DecisionTreeRegressor(max_depth=3, min_samples_leaf=200, random_state=0)
    tree.fit(X, p)
    pred = tree.predict(X)
    r2 = 1 - ((p - pred) ** 2).sum() / ((p - p.mean()) ** 2).sum()
    agree = float(np.mean(np.abs(p - pred) < 0.10))
    txt = export_text(tree, feature_names=list(feats), decimals=2)
    rules = []
    t = tree.tree_
    def walk(node, conds):
        if t.feature[node] == -2:
            rules.append({"conditions": conds, "value": float(t.value[node][0][0]),
                          "n": int(t.n_node_samples[node])})
            return
        f = feats[t.feature[node]]
        thr = t.threshold[node]
        walk(t.children_left[node], conds + [f"{f} <= {thr:.2f}"])
        walk(t.children_right[node], conds + [f"{f} > {thr:.2f}"])
    walk(0, [])
    out = {"r2": float(r2), "agreement_10pct": agree, "rules": rules,
           "text": txt}
    with open(os.path.join(RES, "surrogate_rules.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    with open(os.path.join(RES, "surrogate_rules.txt"), "w", encoding="utf-8") as fh:
        fh.write(f"R2={r2:.3f}  agreement(+-0.10)={agree:.3f}\n\n" + txt)
    print(f"surrogate R2={r2:.3f} agreement={agree:.3f}")
    return out


def stability(df, feats):
    out = {}
    # across folds: refit on 5 different 80% subsamples grouped by student
    ranks = []
    for s in range(5):
        df_s, feats_s = T.load(4)
        rng = np.random.default_rng(s)
        students = df_s["id_student"].unique()
        hold = rng.choice(students, size=int(0.2 * len(students)), replace=False)
        sub = df_s[~df_s["id_student"].isin(hold)]
        X = T.encode_trees(sub, feats_s)
        m = T.model_zoo(feats_s, seed=s)["LightGBM"]
        m.fit(X, sub["y"], categorical_feature=[feats_s.index(c) for c in T.CAT])
        sv, _ = shap_matrix(m, X.iloc[rng.choice(len(X), 4000, replace=False)])
        ranks.append(pd.Series(np.abs(sv).mean(axis=0), index=feats_s))
    rk = pd.DataFrame(ranks)
    corr = rk.rank(axis=1).T.corr(method="spearman")
    out["fold_rank_corr_mean"] = float(corr.values[np.triu_indices(5, 1)].mean())
    out["fold_rank_corr_min"] = float(corr.values[np.triu_indices(5, 1)].min())
    out["fold_top10_overlap"] = float(np.mean([
        len(set(rk.iloc[i].nlargest(10).index) & set(rk.iloc[j].nlargest(10).index)) / 10
        for i in range(5) for j in range(i + 1, 5)]))

    # across cutoffs
    per_week = {}
    for w in [1, 2, 3, 4, 6, 8]:
        df_w, feats_w = T.load(w)
        X = T.encode_trees(df_w, feats_w)
        m = T.model_zoo(feats_w, seed=0)["LightGBM"]
        m.fit(X, df_w["y"], categorical_feature=[feats_w.index(c) for c in T.CAT])
        sv, _ = shap_matrix(m, X.iloc[:4000])
        per_week[w] = pd.Series(np.abs(sv).mean(axis=0), index=feats_w)
    common = set.intersection(*[set(v.index) for v in per_week.values()])
    wk = pd.DataFrame({w: v.reindex(sorted(common)) for w, v in per_week.items()})
    cross = wk.rank().corr(method="spearman")
    out["cutoff_rank_corr"] = {f"{a}-{b}": float(cross.loc[a, b])
                              for a in cross.index for b in cross.columns if a < b}
    out["cutoff_top10_overlap"] = {
        f"{a}-{b}": len(set(wk[a].nlargest(10).index) & set(wk[b].nlargest(10).index)) / 10
        for a in wk.columns for b in wk.columns if a < b}
    with open(os.path.join(RES, "stability.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out, indent=2)[:400])
    return out


def local_cases(df, feats, X, m):
    y = df["y"].values
    p = m.predict_proba(X)[:, 1]
    cases = {}
    def pick(mask, name):
        ids = np.where(mask)[0]
        if len(ids) == 0:
            return None
        # choose the most "typical" high-confidence case
        i = ids[np.argmax(p[ids])] if "risk" in name else ids[np.argmin(np.abs(p[ids] - 0.5))]
        return int(i)
    sel = {
        "withdrawn_high_risk": pick((y == 1) & (p > 0.9), "risk"),
        "withdrawn_missed": pick((y == 1) & (p < 0.35), "missed"),
        "pass_borderline": pick((y == 0) & (np.abs(p - 0.45) < 0.05), "border"),
    }
    sv, base = shap_matrix(m, X)
    for name, i in sel.items():
        if i is None:
            continue
        contrib = pd.Series(sv[i], index=feats).sort_values(key=np.abs, ascending=False)
        top = contrib.head(8)
        cases[name] = {
            "row": i,
            "p": float(p[i]),
            "y": int(y[i]),
            "base_value": float(base[i]),
            "top_contrib": {k: float(v) for k, v in top.items()},
            "values": {k: (float(X.iloc[i][k]) if X.iloc[i][k] == X.iloc[i][k] else None)
                       for k in top.index},
            "profile": {
                "vle_clicks": float(X.iloc[i]["vle_clicks"]),
                "vle_active_days": float(X.iloc[i]["vle_active_days"]),
                "asmt_mean_score": float(X.iloc[i]["asmt_mean_score"]),
                "asmt_n_missed": float(X.iloc[i]["asmt_n_missed"]),
                "vle_slope": float(X.iloc[i]["vle_slope"]),
            },
        }
    with open(os.path.join(RES, "local_cases.json"), "w") as fh:
        json.dump(cases, fh, indent=2)
    print(json.dumps({k: (v["p"], v["y"]) for k, v in cases.items()}, indent=2))
    return cases


def whatif(df, feats, X, m):
    """Actionable counterfactuals for the uncertain middle of the risk distribution."""
    p0 = m.predict_proba(X)[:, 1]
    hi = (p0 >= 0.5) & (p0 <= 0.85)
    out = {"n_moderate_risk": int(hi.sum()), "mean_p_before": float(p0[hi].mean())}
    variants = {}
    vle_cols = [c for c in feats if c.startswith(("vle_clicks", "clk_", "wk_"))]
    v = X.copy()
    # coherent engagement boost: 50% more activity spread evenly
    for c in vle_cols:
        v[c] = v[c] * 1.5
    p1 = m.predict_proba(v)[:, 1]
    variants["engagement_+50pct"] = {
        "mean_p_after": float(p1[hi].mean()),
        "delta": float(p1[hi].mean() - p0[hi].mean()),
        "share_below_0.5": float((p1[hi] < 0.5).mean())}

    v2 = X.copy()
    for c in ["asmt_mean_score", "asmt_min_score", "asmt_max_score", "asmt_wscore"]:
        if c in v2.columns:
            v2[c] = np.where(v2[c] >= 0, np.minimum(v2[c] + 10, 100), v2[c])
    p2 = m.predict_proba(v2)[:, 1]
    variants["assessment_+10pts"] = {
        "mean_p_after": float(p2[hi].mean()),
        "delta": float(p2[hi].mean() - p0[hi].mean()),
        "share_below_0.5": float((p2[hi] < 0.5).mean())}

    v3 = v.copy()
    for c in ["asmt_mean_score", "asmt_min_score", "asmt_max_score", "asmt_wscore"]:
        if c in v3.columns:
            v3[c] = np.where(v3[c] >= 0, np.minimum(v3[c] + 10, 100), v3[c])
    p3 = m.predict_proba(v3)[:, 1]
    variants["both"] = {
        "mean_p_after": float(p3[hi].mean()),
        "delta": float(p3[hi].mean() - p0[hi].mean()),
        "share_below_0.5": float((p3[hi] < 0.5) .mean())}
    out["variants"] = variants
    with open(os.path.join(RES, "whatif.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    df, feats, X, m = fit_full(4)
    imp, sv_df, idx = global_importance(df, feats, X, m)
    surrogate(df, feats, X, m)
    local_cases(df, feats, X, m)
    whatif(df, feats, X, m)
    stability(df, feats)
    print("done explain")
