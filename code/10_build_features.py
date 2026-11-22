"""
Build leakage-controlled feature tables from OULAD at multiple early cutoffs.

Design principles
-----------------
1. Every row is a (module, presentation, student) registration.
2. At cutoff week W only records with date <= 7*W (VLE) and
   date_submitted <= 7*W (assessments) are used.
3. Anything that reveals the outcome after the cutoff is never used:
   date_unregistration, final_result (target only), assessment scores
   submitted later, VLE events later.
4. Group key (id_student) is stored for GroupKFold to avoid the same
   learner leaking across folds.
"""
import os
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(BASE, "data", "csv")
OUT = os.path.join(BASE, "data", "features")
os.makedirs(OUT, exist_ok=True)

CUTOFFS = [1, 2, 3, 4, 6, 8, 12]

print("loading ...")
stu = pd.read_csv(os.path.join(CSV, "student.csv"))
reg = pd.read_csv(os.path.join(CSV, "student_registration.csv"))

svle = pd.read_csv(
    os.path.join(CSV, "student_vle.csv"),
    dtype={"code_module": "category", "code_presentation": "category",
           "id_student": "int32", "id_site": "int32",
           "date": "int16", "sum_click": "int32"},
)
vle_meta = pd.read_csv(os.path.join(CSV, "vle.csv"),
                       dtype={"code_module": "category",
                              "code_presentation": "category",
                              "id_site": "int32"})
sass = pd.read_csv(os.path.join(CSV, "student_assessment.csv"),
                   dtype={"id_assessment": "int32", "id_student": "int32",
                          "date_submitted": "float32", "score": "float32",
                          "is_banked": "bool"})
ass = pd.read_csv(os.path.join(CSV, "assessment.csv"),
                  dtype={"code_module": "category",
                         "code_presentation": "category",
                         "id_assessment": "int32", "date": "float32",
                         "weight": "float32"})

print(f"  students={len(stu):,} vle={len(svle):,} assessment_rows={len(sass):,}")

# ---- registration / course start -------------------------------------------
base = stu.merge(reg, on=["code_module", "code_presentation", "id_student"], how="left")
base["row_id"] = np.arange(len(base))
base["key"] = (base["code_module"].astype(str) + "|" +
               base["code_presentation"].astype(str) + "|" +
               base["id_student"].astype(str))

# activity type map
site_type = (vle_meta.drop_duplicates(subset=["code_module", "code_presentation", "id_site"])
             [["code_module", "code_presentation", "id_site", "activity_type"]])
TOP_TYPES = (svle.merge(site_type, on=["code_module", "code_presentation", "id_site"], how="left")
             ["activity_type"].value_counts().head(14).index.tolist())
print("  top activity types:", TOP_TYPES)

svle = svle.merge(site_type, on=["code_module", "code_presentation", "id_site"], how="left")
svle["activity_type"] = svle["activity_type"].fillna("unknown").astype("category")

# assessment metadata incl. due date, type, weight
ass_meta = ass[["id_assessment", "code_module", "code_presentation",
                "assessment_type", "date", "weight"]].rename(
    columns={"date": "due_date"})
sass = sass.merge(ass_meta, on="id_assessment", how="left")
sass["delay"] = sass["date_submitted"] - sass["due_date"]


def add_vle_features(feat, cutoff):
    sub = svle[svle["date"] <= cutoff]
    g = sub.groupby(["code_module", "code_presentation", "id_student"], observed=True)
    agg = pd.DataFrame({
        "vle_clicks": g["sum_click"].sum(),
        "vle_active_days": g["date"].nunique(),
        "vle_distinct_sites": g["id_site"].nunique(),
        "vle_last_day": g["date"].max(),
        "vle_first_day": g["date"].min(),
    })
    agg["vle_clicks_per_active_day"] = agg["vle_clicks"] / agg["vle_active_days"].clip(lower=1)

    # clicks by activity type
    piv = (sub.groupby(["code_module", "code_presentation", "id_student", "activity_type"],
                       observed=True)["sum_click"].sum().unstack(fill_value=0))
    piv.columns = [f"clk_{c}" for c in piv.columns]

    # weekly click profile (weeks 0..W-1 relative to start); negatives -> week -1
    sub = sub.assign(week=np.clip((sub["date"] // 7), -1, cutoff // 7))
    wk = (sub.groupby(["code_module", "code_presentation", "id_student", "week"],
                      observed=True)["sum_click"].sum().unstack(fill_value=0))
    wk.columns = [f"wk_{c}" for c in wk.columns]
    wk_cols = [c for c in wk.columns if c.startswith("wk_") and c != "wk_-1"]
    if wk_cols:
        W = np.array([int(c.split("_")[1]) for c in wk_cols], dtype=float)
        vals = wk[wk_cols].to_numpy(dtype=float)
        wc = W - W.mean()
        slope = (vals * wc).sum(axis=1) / (wc ** 2).sum()
        wk["vle_slope"] = slope / (vals.mean(axis=1) + 1.0)
        wk["vle_last_week_share"] = vals[:, -1] / (vals.sum(axis=1) + 1)
        wk["vle_active_weeks"] = (vals > 0).sum(axis=1)

    out = agg.join(piv, how="outer").join(wk, how="outer")
    out = out.reset_index()
    res = feat.merge(out, on=["code_module", "code_presentation", "id_student"], how="left")
    res["vle_days_since_last"] = cutoff - res["vle_last_day"]
    for c in ["vle_clicks", "vle_active_days", "vle_distinct_sites"]:
        res[c] = res[c].fillna(0)
    res["vle_days_since_last"] = res["vle_days_since_last"].fillna(cutoff)
    res = res.drop(columns=["vle_last_day", "vle_first_day"])
    return res


def add_assessment_features(feat, cutoff):
    due = ass_meta[ass_meta["due_date"] <= cutoff]
    n_due = (due.groupby(["code_module", "code_presentation"], observed=True)
             .size().rename("asmt_n_due"))
    sub = sass[(sass["date_submitted"] <= cutoff) & (sass["due_date"] <= cutoff)]
    g = sub.groupby(["code_module", "code_presentation", "id_student"], observed=True)
    a = pd.DataFrame({
        "asmt_n_sub": g["score"].size(),
        "asmt_mean_score": g["score"].mean(),
        "asmt_min_score": g["score"].min(),
        "asmt_max_score": g["score"].max(),
        "asmt_std_score": g["score"].std(),
        "asmt_mean_delay": g["delay"].mean(),
        "asmt_n_late": g["delay"].apply(lambda s: int((s > 0).sum())),
    })
    wscore = (sub.assign(wx=sub["score"] * sub["weight"]).groupby(
        ["code_module", "code_presentation", "id_student"], observed=True)
        [["wx", "weight"]].sum())
    a["asmt_wscore"] = (wscore["wx"] / wscore["weight"].replace(0, np.nan)).fillna(0)
    a = a.reset_index()
    res = feat.merge(a, on=["code_module", "code_presentation", "id_student"], how="left")
    res = res.merge(n_due.reset_index(), on=["code_module", "code_presentation"], how="left")
    res["asmt_n_missed"] = (res["asmt_n_due"] - res["asmt_n_sub"]).clip(lower=0)
    res["asmt_has_sub"] = res["asmt_n_sub"].notna().astype(int)
    res["asmt_n_sub"] = res["asmt_n_sub"].fillna(0)
    for c in ["asmt_mean_score", "asmt_min_score", "asmt_max_score",
              "asmt_std_score", "asmt_mean_delay", "asmt_n_late"]:
        res[c] = res[c].fillna(-1)  # sentinel: "no submission yet"
    res["asmt_wscore"] = res["asmt_wscore"].fillna(-1)
    return res


def build(cutoff_week):
    cutoff = 7 * cutoff_week
    f = base.copy()
    f["date_registration"] = f["date_registration"].fillna(0)
    f["registered_late"] = (f["date_registration"] >= 0).astype(int)
    f = add_vle_features(f, cutoff)
    f = add_assessment_features(f, cutoff)
    f["cutoff_week"] = cutoff_week
    return f


if __name__ == "__main__":
    for w in CUTOFFS:
        f = build(w)
        f.to_pickle(os.path.join(OUT, f"feat_w{w}.pkl"))
        numeric_cols = [c for c in f.columns if f[c].dtype.kind in "ifb"]
        print(f"week {w:2d}: rows={len(f):,} cols={f.shape[1]} "
              f"n_features={len(numeric_cols)} at_risk={(f['final_result'].isin(['Fail','Withdrawn'])).mean():.3f}")
    print("done")
