import os
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
"""Four-class variant: Distinction / Pass / Fail / Withdrawn at week four."""
import json, os, sys, importlib.util
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("train_eval", os.path.join(_here, "20_train_eval.py"))
T = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(T)

RES = os.path.join(BASE, "results")

df, feats = T.load(4)
order = ["Distinction", "Pass", "Fail", "Withdrawn"]
y = df["final_result"].astype(str).map({k: i for i, k in enumerate(order)}).values
X = T.encode_trees(df, feats)
cat_idx = [feats.index(c) for c in T.CAT if c in feats]

import lightgbm as lgb
oof = np.zeros((len(df), 4))
skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
for tr, te in skf.split(X, y, df["id_student"].values):
    m = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31,
                           min_child_samples=40, subsample=0.8, subsample_freq=1,
                           colsample_bytree=0.8, reg_lambda=1.0, objective="multiclass",
                           num_class=4, random_state=0, n_jobs=-1, verbose=-1)
    m.fit(X.iloc[tr], y[tr], categorical_feature=cat_idx)
    oof[te] = m.predict_proba(X.iloc[te])

pred = oof.argmax(1)
res = {
    "macro_f1": float(f1_score(y, pred, average="macro")),
    "accuracy": float(accuracy_score(y, pred)),
    "per_class_f1": {order[i]: float(f1_score(y == i, pred == i)) for i in range(4)},
    "recall": {order[i]: float(((pred == i) & (y == i)).sum() / (y == i).sum()) for i in range(4)},
    "confusion": confusion_matrix(y, pred).tolist(),
    "order": order,
}
json.dump(res, open(os.path.join(RES, "fourclass.json"), "w"), indent=2)
print(json.dumps(res, indent=2))
