"""Leave-one-module-out transfer for both the ensemble and the additive GAM.

Answers the unsupported conclusion sentence "steadier where data are scarce":
does the GAM degrade less than the ensemble on an unseen course?
"""
import importlib.util, json, os
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(BASE, "results")

spec = importlib.util.spec_from_file_location("base20", os.path.join(BASE, "code", "20_train_eval.py"))
base = importlib.util.module_from_spec(spec); spec.loader.exec_module(base)

df, feats = base.load(4)
out = {}
for model_name in ["LightGBM", "Spline-GAM"]:
    per = {}
    oof = np.zeros(len(df))
    for mod in sorted(df["code_module"].astype(str).unique()):
        tr = df["code_module"].astype(str) != mod
        te = ~tr
        model = base.model_zoo(feats)[model_name]
        model.fit(df.loc[tr, feats], df.loc[tr, "y"])
        oof[te.values] = model.predict_proba(df.loc[te, feats])[:, 1]
        met = base.evaluate(df.loc[te, "y"].values, oof[te.values])
        per[mod] = {"n": int(te.sum()), "auc": float(met["auc"]),
                    "base_rate": float(df.loc[te, "y"].mean())}
        print(f"{model_name} module {mod}: AUC {met['auc']:.4f}", flush=True)
    over = base.evaluate(df["y"].values, oof)
    out[model_name] = {"per_module": per, "overall_auc": float(over["auc"])}
    np.save(os.path.join(RES, f"oof_transfer_{model_name.replace('-', '')}.npy"), oof)

for k, v in out.items():
    aucs = [m["auc"] for m in v["per_module"].values()]
    print(k, "band", round(min(aucs), 3), "-", round(max(aucs), 3), "mean", round(np.mean(aucs), 3))

json.dump(out, open(os.path.join(RES, "transfer_models.json"), "w"), indent=2)
print("saved results/transfer_models.json")
