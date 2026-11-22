"""Collect every reported number from result files into one facts dictionary."""
import json, os
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(BASE, "results")


def r3(x):
    """Round to the precision printed in the tables, before any gap is computed."""
    return round(float(x), 3)


def _load(name):
    p = os.path.join(RES, name)
    if not os.path.exists(p):
        return {}
    if name.endswith(".json"):
        return json.load(open(p, encoding="utf-8"))
    return pd.read_csv(p)


def facts():
    F = {}
    st = _load("dataset_stats.json")
    F.update({
        "n_reg": f"{st['registrations']:,}",
        "n_stud": f"{st['unique_students']:,}",
        "n_mod": st["modules"],
        "n_pres": st["presentations"],
        "n_vle": f"{st['vle_events']:,}",
        "n_clicks": f"{st['vle_clicks']:,}",
        "n_sites": f"{st['vle_sites']:,}",
        "n_types": st["activity_types"],
        "n_asmt": st["assessments"],
        "n_sub": f"{st['submissions']:,}",
        "p_withdrawn": f"{100*st['result_dist']['Withdrawn']:.1f}",
        "p_fail": f"{100*st['result_dist']['Fail']:.1f}",
        "p_pass": f"{100*st['result_dist']['Pass']:.1f}",
        "p_dist": f"{100*st['result_dist']['Distinction']:.1f}",
        "p_atrisk": f"{100*(st['result_dist']['Withdrawn']+st['result_dist']['Fail']):.1f}",
        "p_female": f"{100*st['gender_F']:.1f}",
        "p_disab": f"{100*st['disability']:.1f}",
        "p_first": f"{100*st['prev_attempts_zero']:.1f}",
        "imd_missing": f"{100*st['imd_missing']:.1f}",
    })

    main = _load("main_w4.json")
    F["main"] = main
    for m in ["DecisionTree", "LR", "Spline-GAM", "RandomForest", "XGBoost", "LightGBM"]:
        if m in main:
            F[f"auc_{m}"] = f"{main[m]['metrics']['auc'][0]:.3f}"
            F[f"auprc_{m}"] = f"{main[m]['metrics']['auprc'][0]:.3f}"
            F[f"brier_{m}"] = f"{main[m]['metrics']['brier'][0]:.3f}"
            F[f"ece_{m}"] = f"{main[m]['metrics']['ece'][0]:.3f}"

    extra = _load("metrics_extra.json")
    F["extra"] = extra
    for m, v in extra.items():
        lo, hi = v["auc_ci"][1], v["auc_ci"][2]
        F[f"ci_{m}"] = f"[{lo:.3f}, {hi:.3f}]"
        F[f"flag_{m}"] = f"{100*v['op80']['cohort_flagged']:.1f}"
        F[f"prec_{m}"] = f"{100*v['op80']['precision']:.1f}"
        if "recall@20" in v:
            F[f"r20_{m}"] = f"{100*v['recall@20']:.1f}"
            F[f"p20_{m}"] = f"{100*v['precision@20']:.1f}"
        if "op60" in v:
            F[f"flag60_{m}"] = f"{100*v['op60']['cohort_flagged']:.1f}"

    if "Spline-GAM" in main and "LightGBM" in main:
        a_g = r3(main["Spline-GAM"]["metrics"]["auc"][0])
        a_b = r3(main["LightGBM"]["metrics"]["auc"][0])
        a_x = r3(main["XGBoost"]["metrics"]["auc"][0])
        a_d = r3(main["DecisionTree"]["metrics"]["auc"][0])
        # gaps computed from the rounded values printed in Table II, so that any
        # reader who subtracts two table entries arrives at the same number
        F["gap_gam_gbdt"] = f"{100*(a_b - a_g):.1f}"
        F["gap_gam_best"] = f"{100*(a_x - a_g):.1f}"
        F["ratio_gam"] = f"{100*(a_g - a_d)/(a_b - a_d):.0f}"
        F["diff_tree"] = f"{100*(a_g - a_d):.1f}"
        F["diff_tree_xgb"] = f"{100*(a_x - a_d):.1f}"
        base_rate = 0.528
        F["lift_note"] = f"{(extra['LightGBM']['op80']['precision']/base_rate):.1f}x" if "LightGBM" in extra else "n/a"

    sweep = _load("sweep.json")
    F["sweep"] = sweep
    if sweep:
        F["auc_w1"] = f"{sweep['LightGBM']['1']['auc'][0]:.3f}"
        F["auc_w2"] = f"{sweep['LightGBM']['2']['auc'][0]:.3f}"
        F["auc_w4"] = f"{sweep['LightGBM']['4']['auc'][0]:.3f}"
        F["auc_w8"] = f"{sweep['LightGBM']['8']['auc'][0]:.3f}"
        F["auc_w12"] = f"{sweep['LightGBM']['12']['auc'][0]:.3f}"
        F["auc_lr_w1"] = f"{sweep['LR']['1']['auc'][0]:.3f}"
        F["auc_lr_w4"] = f"{sweep['LR']['4']['auc'][0]:.3f}"
        F["auc_lr_w12"] = f"{sweep['LR']['12']['auc'][0]:.3f}"
        F["auc_gam_w1"] = f"{sweep['Spline-GAM']['1']['auc'][0]:.3f}"
        F["auc_gam_w12"] = f"{sweep['Spline-GAM']['12']['auc'][0]:.3f}"
        F["gain_w1_w4"] = f"{100*(sweep['LightGBM']['4']['auc'][0]-sweep['LightGBM']['1']['auc'][0]):.1f}"
        F["gain_w4_w12"] = f"{100*(sweep['LightGBM']['12']['auc'][0]-sweep['LightGBM']['4']['auc'][0]):.1f}"
        span = sweep['LightGBM']['12']['auc'][0] - sweep['LightGBM']['1']['auc'][0]
        if span > 0:
            F["pct_early"] = f"{100*(sweep['LightGBM']['4']['auc'][0]-sweep['LightGBM']['1']['auc'][0])/span:.0f}"
        F["auc_w1_gbm"] = f"{sweep['LightGBM']['1']['auc'][0]:.3f}"
        F["auc_w1_gam"] = f"{sweep['Spline-GAM']['1']['auc'][0]:.3f}"
        gaps = [100*(sweep['LightGBM'][str(w)]['auc'][0] - sweep['Spline-GAM'][str(w)]['auc'][0])
                for w in [1, 2, 3, 4, 6, 8, 12]]
        F["max_gap_points"] = f"{max(gaps):.1f}"
        gaps_t = [100*(sweep['Spline-GAM'][str(w)]['auc'][0] - sweep['DecisionTree'][str(w)]['auc'][0])
                  for w in [1, 2, 3, 4, 6, 8, 12]]
        F["tree_gap_min"] = f"{min(gaps_t):.1f}"
        F["tree_gap_max"] = f"{max(gaps_t):.1f}"

    ab = _load("ablation.json")
    F["ablation"] = ab
    if ab:
        F["auc_d_only"] = f"{ab['D']['LightGBM']['auc'][0]:.3f}"
        F["auc_dr"] = f"{ab['D+R']['LightGBM']['auc'][0]:.3f}"
        F["auc_drv"] = f"{ab['D+R+V']['LightGBM']['auc'][0]:.3f}"
        F["auc_full"] = f"{ab['D+R+V+A']['LightGBM']['auc'][0]:.3f}"
        if "R+V+A" in ab:
            F["auc_nodemo"] = f"{ab['R+V+A']['LightGBM']['auc'][0]:.3f}"

    tr = _load("transfer.json")
    F["transfer"] = tr
    if tr:
        F["tr_min"] = f"{min(v['auc'] for k, v in tr.items() if k != 'overall'):.3f}"
        F["tr_max"] = f"{max(v['auc'] for k, v in tr.items() if k != 'overall'):.3f}"
        for k in ["AAA", "CCC", "DDD", "GGG"]:
            if k in tr:
                F[f"tr_{k.lower()}"] = f"{tr[k]['auc']:.3f}"

    fa = _load("fairness.json")
    F["fairness"] = fa
    if fa:
        vals = [s["auc"] for k, sub in fa.items() if k != "module" for s in sub.values()]
        F["fa_min"] = f"{min(vals):.3f}"
        F["fa_max"] = f"{max(vals):.3f}"
        gaps = [max(s["calib_gap"] for s in sub.values()) for k, sub in fa.items()]
        F["calib_max"] = f"{100*max(gaps):.1f}"
        g, d = fa["gender"], fa["disability"]
        F["auc_gender_f"] = f"{g['F']['auc']:.3f}"
        F["auc_gender_m"] = f"{g['M']['auc']:.3f}"
        F["auc_dis_f"] = f"{d['False']['auc']:.3f}"
        F["auc_dis_t"] = f"{d['True']['auc']:.3f}"
        F["fpr_dis_t"] = f"{100*d['True']['fpr']:.1f}"
        F["fpr_dis_f"] = f"{100*d['False']['fpr']:.1f}"
        F["fpr_gap"] = f"{100*(d['True']['fpr']-d['False']['fpr']):.1f}"
        a = fa["age_band"]
        F["tpr_age_old"] = f"{100*a['55<=']['tpr']:.1f}"
        F["tpr_age_young"] = f"{100*a['0-35']['tpr']:.1f}"
        im = fa["imd_band"]
        if "NA" in im:
            F["auc_imd_na"] = f"{im['NA']['auc']:.3f}"
        md = fa["module"]
        F["auc_mod_min"] = f"{min(s['auc'] for s in md.values()):.3f}"
        F["auc_mod_max"] = f"{max(s['auc'] for s in md.values()):.3f}"
        F["auc_mod_ggg"] = f"{md['GGG']['auc']:.3f}"
        F["auc_mod_ccc"] = f"{md['CCC']['auc']:.3f}"
        F["auc_mod_aaa"] = f"{md['AAA']['auc']:.3f}"

    # per-module at-risk base rates, for the module-heterogeneity discussion
    try:
        st = pd.read_csv(os.path.join(BASE, "data", "csv", "student.csv"),
                         usecols=["code_module", "final_result"])
        at = st["final_result"].isin(["Fail", "Withdrawn"])
        br = at.groupby(st["code_module"]).mean()
        F["base_aaa"] = f"{100*br['AAA']:.1f}"
        F["base_ggg"] = f"{100*br['GGG']:.1f}"
        F["dist_ggg"] = f"{100*(st['final_result'].eq('Distinction') & st['code_module'].eq('GGG')).sum()/st['code_module'].eq('GGG').sum():.1f}"
    except Exception:
        pass

    sg = _load("shap_global.csv")
    if isinstance(sg, pd.DataFrame) and len(sg):
        top = sg.head(12)
        F["shap_top"] = top
        F["top1"] = top.iloc[0]["feature"]
        F["top3"] = ", ".join(top.iloc[1:4]["feature"].tolist())
        shares = sg.groupby("group")["mean_abs_shap"].sum()
        shares = (shares / shares.sum() * 100).round(1)
        F["share_vle"] = f"{shares.get('VLE behaviour', 0):.1f}"
        F["share_asmt"] = f"{shares.get('Assessment', 0):.1f}"
        F["share_demo"] = f"{shares.get('Demographic', 0):.1f}"
        vals = dict(zip(sg["feature"], sg["mean_abs_shap"]))
        for k, key in [("code_module", "shap_mod"), ("vle_days_since_last", "shap_recency"),
                       ("asmt_mean_score", "shap_score"), ("highest_education", "shap_edu"),
                       ("vle_active_weeks", "shap_aw")]:
            if k in vals:
                F[key] = f"{vals[k]:.2f}"

    sd = _load("shap_direction.json")
    if sd:
        d = {r["feature"]: r["corr"] for r in sd}
        F["corr_recency"] = f"{d.get('vle_days_since_last', 0):.2f}"
        F["corr_score"] = f"{d.get('asmt_mean_score', 0):.2f}"

    an = _load("risk_anchors.json")
    if an:
        F["risk_inactive"] = f"{an['inactive']['risk']:.2f}"
        F["obs_inactive"] = f"{an['inactive']['observed']:.2f}"
        F["risk_active"] = f"{an['active']['risk']:.2f}"
        F["obs_active"] = f"{an['active']['observed']:.2f}"
        F["risk_nosub"] = f"{an['nosub']['risk']:.2f}"
        F["risk_nodue"] = f"{an['nodue']['risk']:.2f}"

    su = _load("surrogate_rules.json")
    F["surrogate"] = su
    if su:
        F["sur_r2"] = f"{su['r2']:.2f}"
        F["sur_agree"] = f"{100*su['agreement_10pct']:.0f}"

    stab = _load("stability.json")
    F["stability"] = stab
    if stab:
        F["stab_fold"] = f"{stab['fold_rank_corr_mean']:.2f}"
        F["stab_top10"] = f"{100*stab['fold_top10_overlap']:.0f}"
        F["corr_1_4"] = f"{stab['cutoff_rank_corr'].get('1-4', float('nan')):.2f}"
        F["corr_1_8"] = f"{stab['cutoff_rank_corr'].get('1-8', float('nan')):.2f}"

    lc = _load("local_cases.json")
    if lc:
        F["p_case1"] = f"{lc['withdrawn_high_risk']['p']:.2f}"
        F["p_case2"] = f"{lc['withdrawn_missed']['p']:.2f}"
        F["p_case3"] = f"{lc['pass_borderline']['p']:.2f}"

    wi = _load("whatif.json")
    F["whatif"] = wi
    if wi:
        F["n_moderate"] = f"{wi['n_moderate_risk']:,}"
        for k, v in wi["variants"].items():
            F[f"wi_{k}"] = f"{abs(100*v['delta']):.1f}"
            F[f"wi_{k}_share"] = f"{100*v['share_below_0.5']:.1f}"

    si = _load("silent.json")
    if si:
        F["n_silent"] = f"{si['n_silent']:,}"
        F["p_silent"] = f"{100*si['share']:.1f}"
        F["rate_silent"] = f"{100*si['rate']:.1f}"
        F["share_atrisk_silent"] = f"{100*si['share_atrisk']:.1f}"

    oof = _load("oof_w4.csv")
    if isinstance(oof, pd.DataFrame) and len(oof):
        from sklearn.metrics import roc_auc_score
        for m in ["LightGBM", "Spline-GAM", "XGBoost", "DecisionTree"]:
            col = f"p_{m}"
            if col not in oof:
                continue
            F[f"oof_{m.replace('-', '')}"] = f"{roc_auc_score(oof['y'].values, oof[col].values):.3f}"
            k = int(0.20 * len(oof))
            top = np.argsort(-oof[col].values)[:k]
            y = oof["y"].values
            F[f"r20_{m}"] = f"{100*y[top].sum()/y.sum():.1f}"
            F[f"p20_{m}"] = f"{100*y[top].mean():.1f}"

    f4 = _load("fourclass.json")
    if f4:
        F["f4_macro"] = f"{f4['macro_f1']:.2f}"
        F["f4_acc"] = f"{100*f4['accuracy']:.1f}%"
        F["f4_rec_dist"] = f"{100*f4['recall']['Distinction']:.0f}%"
        F["f4_rec_pass"] = f"{100*f4['recall']['Pass']:.0f}%"
        F["f4_rec_fail"] = f"{100*f4['recall']['Fail']:.0f}%"
        F["f4_rec_wd"] = f"{100*f4['recall']['Withdrawn']:.0f}%"

    sens = _load("sensitivity.json")
    if sens:
        for m, v in sens.items():
            key = m.replace(" ", "").replace("-", "")
            F[f"sp_{key}"] = f"{100*v['span']:.1f}"
            F[f"smin_{key}"] = f"{v['auc_min']:.3f}"
            F[f"smax_{key}"] = f"{v['auc_max']:.3f}"
            F[f"sdef_{key}"] = f"{v['default_auc']:.3f}"
            b = v["best"]
            F[f"sbest_{key}"] = f"{b['auc']:.3f}"
            F[f"sbestv_{key}"] = f"{b['value']}"

    pt = _load("paired_tests.json")
    if pt:
        F["paired"] = pt["tests"]
        t = pt["tests"]["Spline-GAM_minus_LightGBM"]
        F["d_gam_lgbm"] = f"{t['delta_auc']:+.3f}"
        F["ci_gam_lgbm"] = f"[{t['ci95'][0]:+.3f}, {t['ci95'][1]:+.3f}]"
        t2 = pt["tests"]["DecisionTree_minus_Spline-GAM"]
        F["d_dt_gam"] = f"{t2['delta_auc']:+.3f}"
        F["ci_dt_gam"] = f"[{t2['ci95'][0]:+.3f}, {t2['ci95'][1]:+.3f}]"

    fm = _load("fairness_mitigation.json")
    if fm:
        F["fair"] = fm
        b = fm["LightGBM"]["rules"]["global_0.5"]
        e = fm["LightGBM"]["rules"]["fpr_eq"]
        F["fpr_gap_base"] = f"{100*b['fpr_gap_disability']['mean']:.1f}"
        F["fpr_gap_eq"] = f"{100*e['fpr_gap_disability']['mean']:.1f}"
        # same quantity, two protocols: pooled out-of-fold (Section V-E) vs cross-fitted (Table V)
        F["fpr_protocol_delta"] = f"{abs(float(F.get('fpr_gap', 0.0)) - 100*b['fpr_gap_disability']['mean']):.1f}"
        F["tpr_gap_base"] = f"{100*b['tpr_gap_disability']['mean']:.1f}"
        F["tpr_gap_eq"] = f"{100*e['tpr_gap_disability']['mean']:.1f}"
        F["rec_base"] = f"{100*b['recall']['mean']:.1f}"
        F["rec_eq"] = f"{100*e['recall']['mean']:.1f}"
        F["cov_base"] = f"{100*b['coverage']['mean']:.1f}"
        F["cov_eq"] = f"{100*e['coverage']['mean']:.1f}"
        F["age_gap_base"] = f"{100*b['fpr_gap_age_band']['mean']:.1f}"
        F["age_gap_eq"] = f"{100*e['fpr_gap_age_band']['mean']:.1f}"
        levels = fm["LightGBM"]["per_level"]["disability"]
        F["fpr_dis_base"] = f"{100*levels['baseline']['True']['fpr']['mean']:.1f}"
        F["fpr_ndis_base"] = f"{100*levels['baseline']['False']['fpr']['mean']:.1f}"
        F["fpr_dis_eq"] = f"{100*levels['fpr_eq']['True']['fpr']['mean']:.1f}"
        F["fpr_ndis_eq"] = f"{100*levels['fpr_eq']['False']['fpr']['mean']:.1f}"

    # ---- second-round review fixes (all recomputed, not estimated) ----
    r2 = _load("review2_fix.json")
    if r2:
        F["r2"] = r2
        c = r2["cluster_ci"]
        F["ci_lgbm_cluster"] = f"[{c['LightGBM']['lo']:.3f}, {c['LightGBM']['hi']:.3f}]"
        F["auc_lgbm_split"] = f"{c['LightGBM']['auc']:.3f}"
        F["ci_gam_cluster"] = f"[{c['Spline-GAM']['lo']:.3f}, {c['Spline-GAM']['hi']:.3f}]"
        F["auc_gam_split"] = f"{c['Spline-GAM']['auc']:.3f}"
        g = r2["fpr_endpoints"]["agg"]
        F["fpr_gap_eq_sd"] = f"{100*g['gap_eq']['sd']:.1f}"
        F["fpr_gap_eq_lo"] = f"{100*g['gap_eq']['min']:.1f}"
        F["fpr_gap_eq_hi"] = f"{100*g['gap_eq']['max']:.1f}"
        F["fpr_gap_base_sd"] = f"{100*g['gap_base']['sd']:.1f}"
        a = r2["active_subcohort"]
        F["act_base"] = f"{100*a['base_rate']:.1f}"
        cp = a["comparisons"]
        F["cmp_model"] = f"{100*cp['ensemble_top20']['recall']:.1f}/{100*cp['ensemble_top20']['precision']:.1f}"
        F["cmp_lr3"] = f"{100*cp['three_feature_LR_top20']['recall']:.1f}/{100*cp['three_feature_LR_top20']['precision']:.1f}"
        F["cmp_days"] = f"{100*cp['days_since_last_top20']['recall']:.1f}/{100*cp['days_since_last_top20']['precision']:.1f}"
        F["cmp_score"] = f"{100*cp['mean_score_low_top20']['recall']:.1f}/{100*cp['mean_score_low_top20']['precision']:.1f}"
        F["cmp_chance"] = f"{100*cp['random_top20']['recall']:.1f}/{100*cp['random_top20']['precision']:.1f}"

    if fm:
        F["imd_gap_base"] = f"{100*fm['LightGBM']['rules']['global_0.5']['fpr_gap_imd_band']['mean']:.1f}"
        F["imd_gap_after_dis"] = f"{100*fm['LightGBM']['rules']['fpr_eq']['fpr_gap_imd_band']['mean']:.1f}"
        if "fpr_eq:imd_band" in fm["LightGBM"]["rules"]:
            r = fm["LightGBM"]["rules"]["fpr_eq:imd_band"]
            F["imd_gap_eq"] = f"{100*r['fpr_gap_imd_band']['mean']:.1f}"
            F["imd_rec_eq"] = f"{100*r['recall']['mean']:.1f}"
            F["imd_rec_cost"] = f"{100*(b['recall']['mean']-r['recall']['mean']):.1f}"

    tm = _load("transfer_models.json")
    if tm:
        F["tr_gam_min"] = f"{min(v['auc'] for v in tm['Spline-GAM']['per_module'].values()):.3f}"
        F["tr_gam_max"] = f"{max(v['auc'] for v in tm['Spline-GAM']['per_module'].values()):.3f}"
        F["tr_mean_lgbm"] = f"{np.mean([v['auc'] for v in tm['LightGBM']['per_module'].values()]):.3f}"
        F["tr_mean_gam"] = f"{np.mean([v['auc'] for v in tm['Spline-GAM']['per_module'].values()]):.3f}"

    rfc = _load("rf_cutoffs.json")
    if rfc:
        F["rf_cut"] = rfc
        F["auc_rf_w4_cut"] = f"{rfc['4']:.3f}"

    if sweep:
        F["tree_gap_w1"] = f"{100*(r3(sweep['Spline-GAM']['1']['auc'][0])-r3(sweep['DecisionTree']['1']['auc'][0])):.1f}"
        F["tree_gap_w3"] = f"{100*(r3(sweep['Spline-GAM']['3']['auc'][0])-r3(sweep['DecisionTree']['3']['auc'][0])):.1f}"
        F["tree_gap_w8"] = f"{100*(r3(sweep['Spline-GAM']['8']['auc'][0])-r3(sweep['DecisionTree']['8']['auc'][0])):.1f}"
        F["tree_gap_w12"] = f"{100*(r3(sweep['Spline-GAM']['12']['auc'][0])-r3(sweep['DecisionTree']['12']['auc'][0])):.1f}"

    return F
