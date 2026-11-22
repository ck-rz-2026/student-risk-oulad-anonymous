# Reproducibility package

Anonymous package accompanying the submission *Interpretable Prediction of Student Academic
Performance on Open Educational Data*. Every number, table and figure in the manuscript is
produced by the scripts below from the public OULAD release; nothing is hard-coded by hand.

Author identity is removed: the manuscript source in `code/paper_content.py` carries the
placeholder author block, and no e-mail address, institution or personal path appears anywhere
in this package.

## Contents

```
code/            analysis, training, figure and typesetting scripts (Python)
results/         JSON/CSV outputs that the manuscript numbers are read from
figures/         figure PNGs used by the typesetting step
data/features/   cached feature tables, one per cutoff week (feat_w{1,2,3,4,6,8,12}.pkl)
```

## Environment

Python 3.13, with

```
pandas 3.1.1   numpy 2.5.5   scipy 1.18.1   scikit-learn 1.9.1
lightgbm 4.7.0   xgboost 3.4.1   matplotlib 3.11.2
reportlab 5.1.0   pypdf (any recent version)   shap (optional, only for 21_explain.py)
joblib
```

`pip install -r requirements.txt`.

## Data

Download the Open University Learning Analytics Dataset (CC-BY 4.0) and place the seven CSV
files in `data/csv/`:

`courses.csv`, `assessments.csv`, `vle.csv`, `studentInfo.csv`,
`studentRegistration.csv`, `studentAssessment.csv`, `studentVle.csv`.

The derived tables in `data/features/` are already provided, so the training, evaluation,
fairness, explanation and typesetting steps run without the raw CSVs. Only
`10_build_features.py` needs `data/csv/`.

## Order of execution

All scripts resolve paths relative to the package root, so run them from anywhere:

```bash
python code/10_build_features.py     # raw OULAD -> data/features/feat_w*.pkl
python code/11_stats.py              # cohort description (Table I)
python code/20_train_eval.py         # six models, three repeated grouped splits (Table II)
python code/22_metrics_extra.py      # operating points, ECE, precision/recall at top 20%
python code/30_figures.py            # feature importance and SHAP figures
python code/31_sensitivity_evolution.py  # per-cutoff sweep and hyperparameter sweep
python code/32_paired_tests_fast.py  # paired learner-cluster bootstrap (Table III)
python code/33_fairness_mitigation.py    # subgroup flags, equalised-FPR/TPR (Table V)
python code/34_nonsilent_fairness.py # fairness inside the non-silent sub-cohort
python code/36_review2_fix.py        # cluster CI, per-seed FPR endpoints, capacity-matched baselines
python code/37_transfer_models.py    # leave-one-module-out transfer for ensemble and GAM
python code/38_rf_cutoffs.py         # random forest AUC by cutoff (Table IV, RF column)
python code/40_build_paper.py        # rebuild the manuscript PDF from figures + results
```

Seeds are fixed (`11, 22, 33, 44, 55` for splits; `0-4` for repeated CV); folds are grouped by
`id_student` throughout. `36`, `37` and `38` take roughly 2, 4 and 15 minutes on a laptop;
the rest scale with available cores.

## Where each claim comes from

| Manuscript element | Script | Output |
| --- | --- | --- |
| Table I cohort description | `11_stats.py` | `results/dataset_stats.json` |
| Table II AUC / AP / recall at 80% | `20_train_eval.py` | `results/main_w4.json` |
| Operating points, top-20% precision, ECE | `22_metrics_extra.py` | `results/metrics_extra.json` |
| Fig. 1, Table IV (cutoff sweep, incl. RF) | `31_sensitivity_evolution.py`, `38_rf_cutoffs.py` | `results/sweep.json`, `results/rf_cutoffs.json` |
| Hyperparameter sensitivity (Fig. 2 discussion) | `31_sensitivity_evolution.py` | `results/sensitivity.json` |
| Table III paired comparisons | `32_paired_tests_fast.py` | `results/paired_tests.json` |
| Cluster CI quoted in V-A | `36_review2_fix.py` | `results/review2_fix.json` → `cluster_ci` |
| Table V, per-seed FPR endpoints | `33_fairness_mitigation.py` | `results/fairness_mitigation.json` |
| Same, per-split means and SD | `36_review2_fix.py` | `results/review2_fix.json` → `fpr_endpoints` |
| IMD / deprivation gaps | `33_fairness_mitigation.py` | `results/fairness_mitigation.json` |
| Non-silent sub-cohort and capacity-matched baselines | `34_nonsilent_fairness.py`, `36_review2_fix.py` | `results/nonsilent_fairness.json`, `results/review2_fix.json` |
| Module-level and demographic subgroup AUC | `33_fairness_mitigation.py` | `results/fairness.json` |
| Fig. 3 leave-one-module-out (ensemble) | `20_train_eval.py` / `35_reviewer_patch.py` | `results/transfer.json` |
| GAM transfer band and means | `37_transfer_models.py` | `results/transfer_models.json` |
| Feature-block ablation (Fig. 2) | `20_train_eval.py` | `results/ablation.json` |
| SHAP attributions and block shares | `21_explain.py` | `results/shap_global.csv`, `results/review_patch.json` |
| Rank stability across resamples | `21_explain.py` | `results/stability.json` |
| Four-class variant | `23_fourclass.py` | `results/fourclass.json` |
| Silent cohort statistics | `11_stats.py` | `results/silent.json` |
| Threshold-recall curve (Fig. 5b) | `31_sensitivity_evolution.py` | `results/threshold.json` |

The manuscript reads these through `code/paper_facts.py` (single entry point) and
`code/paper_content.py` (text assembly), so a rebuilt PDF always matches the stored results.

## Rebuilding the PDF

```bash
python code/40_build_paper.py
```

writes `paper/Interpretable_Prediction_of_Student_Academic_Performance.pdf`, a ten-page
two-column document. The typesetting uses reportlab rather than the IEEE class file, so the
PDF is a submission copy, not the camera-ready.

## License

Code: MIT. Derived feature tables and figures: CC-BY-4.0, following the licence of OULAD.
