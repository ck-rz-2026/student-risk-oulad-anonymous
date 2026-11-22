import json, os
import pandas as pd
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(BASE, "data", "csv")
RES = os.path.join(BASE, "results")

stu = pd.read_csv(os.path.join(CSV, "student.csv"))
svle = pd.read_csv(os.path.join(CSV, "student_vle.csv"),
                   usecols=["sum_click", "date", "id_site", "activity_type" if False else "id_site"])
vle = pd.read_csv(os.path.join(CSV, "vle.csv"))
sass = pd.read_csv(os.path.join(CSV, "student_assessment.csv"))
ass = pd.read_csv(os.path.join(CSV, "assessment.csv"))
course = pd.read_csv(os.path.join(CSV, "course.csv"))

stats = {
    "registrations": int(len(stu)),
    "unique_students": int(stu["id_student"].nunique()),
    "modules": int(stu["code_module"].nunique()),
    "presentations": int(stu.groupby(["code_module", "code_presentation"]).ngroups),
    "vle_events": int(len(svle)),
    "vle_clicks": int(svle["sum_click"].sum()),
    "vle_sites": int(vle["id_site"].nunique()),
    "activity_types": int(vle["activity_type"].nunique()),
    "assessments": int(ass["id_assessment"].nunique()),
    "submissions": int(len(sass)),
    "mean_presentation_length": float(course["module_presentation_length"].mean()),
    "result_dist": {k: float(v) for k, v in stu["final_result"].value_counts(normalize=True).items()},
    "gender_F": float((stu["gender"] == "F").mean()),
    "disability": float((stu["disability"] == True).mean()),
    "age": {k: float(v) for k, v in stu["age_band"].value_counts(normalize=True).items()},
    "edu": {k: float(v) for k, v in stu["highest_education"].value_counts(normalize=True).items()},
    "prev_attempts_zero": float((stu["num_of_prev_attempts"] == 0).mean()),
    "mean_credits": float(stu["studied_credits"].mean()),
    "imd_missing": float(stu["imd_band"].isna().mean()),
}
# VLE activity type distribution
site_type = vle.drop_duplicates(subset=["code_module", "code_presentation", "id_site"])
print(json.dumps(stats, indent=2))
with open(os.path.join(RES, "dataset_stats.json"), "w") as f:
    json.dump(stats, f, indent=2)

# average clicks per active day etc at week 4
feat = pd.read_pickle(os.path.join(BASE, "data", "features", "feat_w4.pkl"))
print("median clicks w4:", feat["vle_clicks"].median(),
      "median active days:", feat["vle_active_days"].median(),
      "median score:", feat["asmt_mean_score"][feat["asmt_mean_score"] >= 0].median())
print("share with no submission by w4:", (feat["asmt_has_sub"] == 0).mean())
print("n due by module:", feat.groupby("code_module")["asmt_n_due"].first().to_dict())
