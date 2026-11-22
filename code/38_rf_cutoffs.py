"""Random Forest AUC across cutoffs, to complete Table IV with all six families."""
import importlib.util, json, os
import numpy as np
from sklearn.metrics import roc_auc_score

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(BASE, "results")
spec = importlib.util.spec_from_file_location("base20", os.path.join(BASE, "code", "20_train_eval.py"))
base = importlib.util.module_from_spec(spec); spec.loader.exec_module(base)

out = {}
for w in [1, 2, 3, 4, 6, 8, 12]:
    df, feats = base.load(w)
    oof = base.run_cv(df, feats, "RandomForest", 0, n_splits=5)
    out[str(w)] = float(roc_auc_score(df["y"].values, oof))
    print(f"week {w}: RF AUC {out[str(w)]:.4f}", flush=True)

json.dump(out, open(os.path.join(RES, "rf_cutoffs.json"), "w"), indent=2)
print("saved results/rf_cutoffs.json")
