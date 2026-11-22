"""Paper content. Numbers are injected from result files via paper_facts."""
import os, json
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def fmt(s, F):
    try:
        return s.format(**F)
    except KeyError as e:
        print("missing fact", e)
        return s


def build(F):
    B = []
    add = B.append

    # ------------------------------------------------------------------ head
    add(("title", "Interpretable Prediction of Student Academic Performance\n"
                  "on Open Educational Data"))
    add(("authors", "Anonymous Author(s)"))
    add(("affil", "Affiliations withheld for double-blind review."))
    add(("title_end",))

    abstract = (
        "<b><i>Abstract</i></b>&#8212;Two assumptions govern student-risk modelling: that accuracy "
        "must be bought with opacity, and that a model confined to a term's opening weeks cannot be "
        "trusted. We test both on the Open University Learning Analytics Dataset, "
        "with nothing beyond four weeks after a module opens&#8212;"
        "{n_reg} registrations, {n_vle} virtual-learning-environment (VLE) events and {n_sub} "
        "submissions, with the withdrawal date and final outcome removed by construction. Under "
        "that restriction a LightGBM ensemble scores AUC {auc_LightGBM} and an additive spline "
        "generalized additive model (GAM), the model we recommend, {auc_Spline-GAM}&#8212;"
        "{gap_gam_gbdt} behind the ensemble and {gap_gam_best} behind the best of six. A depth-4 tree scores {auc_DecisionTree}, a further {diff_tree} back, and "
        "its shortfall is representational, not a depth effect. Risk is bimodal, not "
        "graded: {p_silent}% of registrations log no VLE activity at all in those four weeks "
        "and {rate_silent}% of them fail or withdraw, so the top 20% of the score holds "
        "{r20_LightGBM}% of at-risk registrations at {p20_LightGBM}% precision&#8212;excellent "
        "by precision, yet it reaches barely a third of those the policy targets. "
        "Explanations are cutoff-specific, not unstable: TreeSHAP rankings hold across "
        "resamples (Spearman {stab_fold}) yet shift wholesale between cutoffs, because the two "
        "models resting on different evidence. Ranking is near "
        "group-blind (AUC {fa_min}&#8211;{fa_max}) while treatment is not: the disability "
        "false-positive gap is {fpr_gap} at the 0.5 cut, 0.1 in the top quintile and 12.2 "
        "at 80% recall, and the widest gap&#8212;{imd_gap_base} across deprivation "
        "bands&#8212;is the one our remedy cuts to {imd_gap_eq}. On an unseen course AUC "
        "falls to {tr_min} to {tr_max}."
    )
    add(("abs", fmt(abstract, F)))
    add(("space", 4))
    add(("abs", "<b><i>Index Terms</i></b>&#8212;Learning analytics, early warning systems, "
                "interpretability, SHAP, generalized additive models, educational data mining, "
                "algorithmic fairness, OULAD."))
    add(("space", 4))

    # ------------------------------------------------------------------ 1
    add(("h1", "I. Introduction"))
    add(("p", fmt(
        "Attendance lapses, silent non-submission and failed assessments recur every term. "
        "Failure or withdrawal accounts for {p_atrisk}% of the module registrations examined "
        "here. Because a missed at-risk student is costly while a false alarm costs little more "
        "than an e-mail inviting the student to a tutorial, universities have built predictive "
        "systems that score students early and pass the result to advisors, tutors and support "
        "units [1], [2], [7]. Accuracy no longer limits such systems: on public benchmarks "
        "gradient-boosted trees regularly reach AUCs in the mid-eighties to low nineties once "
        "the whole term is visible [7], [30], and reported differences owe more to protocol than "
        "to algorithm.", F)))
    add(("p", fmt(
        "What limits them lies elsewhere. A model fed the entire term is an autopsy rather than "
        "a warning: it reports what advisors already know, at a moment when nothing can still "
        "be changed. A model whose output nobody can explain cannot be acted upon either&#8212;"
        "handed a red score with no reason, an advisor can only trust it blindly, ignore it, or "
        "reverse-engineer it, none of which is defensible when a student's education is at "
        "stake. And a "
        "model audited only in aggregate may conceal systematic error exactly in the groups that "
        "most need support.", F)))
    add(("p", fmt(
        "The claim defended here is that these limits can be respected at the same time. The "
        "test bed is {n_reg} registrations drawn from OULAD [1], the anonymised release of "
        "{n_mod} modules and {n_pres} presentations of the Open University UK, examined under a "
        "deliberately austere protocol. Four weeks after a module starts, a model may use "
        "demographics, the registration record, VLE clicks aggregated to that date, and "
        "assessment scores submitted before the cutoff; it may not use the withdrawal date, "
        "submissions made later, or any quantity computed across the full term. Four questions "
        "follow. (i) What predictive power exists at four weeks, and how does it accumulate as "
        "the cutoff moves? (ii) How much of it survives once the model is forced to be additive, "
        "and therefore interpretable by construction? (iii) Are the explanations it produces "
        "coherent, stable and usable by the advisor who reads them? (iv) Do they survive across "
        "demographic groups and "
        "across courses never seen in training?", F)))
    add(("p", fmt(
        "The answers are narrower than the usual framing, and two of them overturn it. "
        "Transparency turns out not to be where the accuracy goes: after four weeks the additive "
        "spline GAM trails the reference ensemble by {gap_gam_gbdt} AUC points and the strongest "
        "model by {gap_gam_best}, whereas a tree shallow enough to read loses {diff_tree}&#8212;"
        "the price is single-tree capacity, not additivity. What is scarce is coverage rather "
        "than precision: {p_silent}% of registrations never touch the VLE in those four weeks and "
        "{rate_silent}% of them fail or withdraw, so flagging the top 20% of the score yields "
        "{p20_LightGBM}% precision while capturing only {r20_LightGBM}% of at-risk students&#8212;"
        "a dashboard that looks near-perfect largely because it re-identifies the students who "
        "had already visibly disengaged. The explanations carry the same tension: they reproduce "
        "within a cutoff (Spearman {stab_fold}, {stab_top10}% top-10 overlap) yet drift between "
        "cutoffs, and what-if edits that lift engagement in isolation shift an "
        "already-flagged student only marginally, because risk is stored redundantly across "
        "correlated features.", F)))
    add(("p", fmt(
        "Section II relates the paper to early-prediction, interpretability and fairness work. "
        "Section III sets the protocol, fixes what the cutoff means, and reports a leakage "
        "incident in which the withdrawal date inflated AUC by close to seven points. Section IV "
        "describes the feature blocks, the six models and the explanation protocol. Section V "
        "gives discrimination, calibration, the earliness curve, the ablation, cross-course "
        "transfer, the subgroup audit and a four-class extension. Section VI examines "
        "explanations: global attributions, effect shapes, three individual cases, a checkable "
        "rule list, rank stability and what-if edits. Section VII discusses deployment and "
        "limits.", F)))
    add(("pni", "<b>Contributions.</b>"))
    contribs = [
        "A protocol for early performance prediction on OULAD that controls leakage and "
        "reproduces end to end, prices the four-week window against later cutoffs rather than "
        "assuming it, groups folds by student, repeats splits, and withholds one module at a "
        "time to measure transfer.",
        "A like-for-like contest between models interpretable as built (logistic regression, "
        "spline GAM, depth-4 tree) and tree ensembles run through the same pipeline, which pins "
        "down what interpretability actually charges at this task.",
        "A curve of earliness that shows separately what demographics buy, what the registration "
        "record buys, what VLE activity buys, and what the first assessment results buy.",
        "Diagnostics for explanations that are seldom published: how faithful a surrogate rule "
        "list is, how far global attributions move across resamples and cutoffs, and what "
        "coherent rather than single-feature what-if perturbations actually change.",
        "A subgroup audit across gender, disability, age and index of multiple deprivation, "
        "paired with calibration&#8212;the place where the disparities turn out to sit.",
    ]
    for i, c in enumerate(contribs, 1):
        add(("bullet", f"&#8226; {c}"))
    add(("space", 3))

    # ------------------------------------------------------------------ 2
    add(("h1", "II. Related Work"))
    add(("h2", "A. Early Prediction of Academic Performance"))
    add(("p", fmt(
        "Predicting grades and dropout from demographic, behavioural and assessment data has a "
        "long record in educational data mining [3], [4]. OULAD alone has attracted logistic "
        "regression, decision trees, random forests, support vector machines and, lately, "
        "recurrent and attention-based networks, aimed usually at the four-class outcome or at "
        "a binary withdrawal label [7], [8], [30]. Published AUCs spread widely, and protocol "
        "rather than algorithm explains most of that spread: whether the whole term is visible, "
        "whether one student can land in both training and test folds, whether test courses "
        "appeared during training. Cross-paper comparison is therefore unsafe, which is why "
        "every baseline here is trained and scored inside a single pipeline. For the same reason "
        "we do not tabulate OULAD numbers side by side&#8212;cutoff, label definition, splitting "
        "and leakage control all differ&#8212;and defend a ranking obtained within one protocol "
        "rather than a victory over any particular earlier figure.", F)))
    add(("p", fmt(
        "A second strand follows what happens once a score has been issued. Early alert systems "
        "run at scale report modest but genuine retention gains, and those gains pass almost "
        "entirely through tutors who act on the alerts [6], [9], [31]. That dependence is the "
        "practical case for interpretability: an alert that never names the problem leaves the "
        "tutor with nothing to do.", F)))

    add(("h2", "B. Interpretability"))
    add(("p", fmt(
        "Post-hoc methods assign credit for a prediction to input features [18]. LIME builds local "
        "surrogates [14]; SHAP recasts several attribution schemes as Shapley values and, in "
        "TreeSHAP, evaluates them exactly and cheaply for tree ensembles [12], [13]. Both are "
        "common in learning analytics, and both describe a model's arithmetic rather than the "
        "causal structure of the world. Rudin's argument is that high-stakes decisions should "
        "rest instead on models interpretable by construction, because an explanation of an "
        "opaque model can only approximate something the practitioner still cannot inspect [15]. "
        "Additive models [17] and their boosting descendants [16] occupy the middle: every "
        "feature contributes a learned one-dimensional shape, so the fit is additive and "
        "inspectable while often staying within a fraction of a point of a full ensemble.", F)))
    add(("p", fmt(
        "Our design takes that argument at face value. The additive GAM is a first-class "
        "candidate rather than a baseline, and TreeSHAP on the reference ensemble serves as a "
        "check on whether additivity is discarding real signal&#8212;not as the only road to an "
        "explanation.", F)))

    add(("h2", "C. Fairness and Validity in Educational Prediction"))
    add(("p", fmt(
        "Models trained on educational records inherit the biases of those records, and the "
        "harm is distributional: a group flagged above its base rate is watched "
        "disproportionately [10], [28], [29]. Definitions come from the fairness literature "
        "(demographic parity, equalised odds [11], [27]); warnings come from the measurement "
        "literature, where calibration is repeatedly the property that decides practical "
        "outcomes and repeatedly the one that fails [25], [26]. We therefore report subgroup "
        "AUC, subgroup calibration and subgroup flag rates instead of a single fairness "
        "figure&#8212;and report them for the interpretable model too, since an additive model "
        "can be unfair in perfectly legible ways.", F)))

    # ------------------------------------------------------------------ 3
    add(("h1", "III. Data and Protocol"))
    add(("h2", "A. The OULAD Release"))
    add(("p", fmt(
        "OULAD [1] ships seven relational tables: {n_mod} modules (AAA&#8211;GGG) run across "
        "{n_pres} presentations, holding {n_reg} student&#8211;module registrations for {n_stud} "
        "distinct students, {n_vle} daily VLE summaries ({n_clicks} clicks on {n_sites} distinct "
        "sites across {n_types} activity types), {n_asmt} assessments and {n_sub} submissions, "
        "with a registration spanning 234 to 269 days. The outcome field final\_result has four "
        "levels&#8212;Pass {p_pass}%, Withdrawn {p_withdrawn}%, Fail {p_fail}% and Distinction "
        "{p_dist}%&#8212;and Table&#160;I lists the release as we consume it.", F)))

    rows = [["Property", "Value"],
            ["Registrations / distinct students", "{n_reg} / {n_stud}"],
            ["Modules / presentations", "{n_mod} / {n_pres}"],
            ["VLE events / clicks / sites", "{n_vle} / {n_clicks} / {n_sites}"],
            ["VLE activity types", "{n_types}"],
            ["Assessments / submissions", "{n_asmt} / {n_sub}"],
            ["Withdrawn / Fail / Pass / Distinction", "{p_withdrawn}% / {p_fail}% / {p_pass}% / {p_dist}%"],
            ["Female / declared disability", "{p_female}% / {p_disab}%"],
            ["No previous attempt / mean credits", "{p_first}% / 79.8"],
            ["Missing deprivation band", "{imd_missing}%"]]
    rows = [[fmt(c, F) for c in r] for r in rows]
    add(("tab", rows, "<b>Table I.</b> The OULAD release used in this study.", [2.35 * 72, 1.15 * 72], 7.2))

    add(("h2", "B. Target and Cutoff Semantics"))
    add(("p", fmt(
        "Outcomes are collapsed to a binary label, y = 1 when a registration ends "
        "unsuccessfully (Fail or Withdrawn, {p_atrisk}% of records) and y = 0 otherwise. The "
        "collapse is deliberate: both unsuccessful modes call for the same advising action&#8212;"
        "contact the student&#8212;and a binary label keeps the class balance readable. "
        "Section&#160;V-F reports the four-class variant for completeness.", F)))
    add(("p", fmt(
        "A model observes a registration at cutoff W weeks, that is, at day 7W from module "
        "start. Because dates in studentVle are already relative to the presentation start, the "
        "filter is exact rather than approximate. W is a parameter, not a constant: four weeks is "
        "the point by which a first assessment is due in most modules while the withdrawal window "
        "is still open, and Section&#160;V-B reports the whole curve so that an institution with a "
        "different deadline can read off its own operating point. Negative dates, which mark "
        "preparation before "
        "the official start, remain in the data: they are genuinely in hand when an advisor "
        "acts. Four things never reach any model: final\_result, the label itself; "
        "date\_unregistration, a withdrawal date that is the outcome&#8212;when an earlier draft "
        "of this study included it, AUC rose by almost seven points, a reminder that leakage in "
        "educational data tends to be administrative rather than exotic; any submission dated "
        "after the cutoff; and any VLE record dated after the cutoff. We spell this out because "
        "the same field appears in several published OULAD pipelines.", F)))

    add(("h2", "C. Splitting and Evaluation"))
    add(("p", fmt(
        "Every number reported is out-of-fold. The primary design is stratified group k-fold "
        "cross-validation, five folds grouped on the student identifier so that a learner "
        "enrolled in several modules never straddles training and test, repeated over three "
        "seeds, with the mean and the standard deviation across repeats reported. Because all "
        "folds draw on the same modules, this protocol measures generalisation to new students "
        "of courses already known. Generalisation to a course the institution has not run "
        "before&#8212;the case of a programme adopting the system for a new module&#8212;is "
        "measured separately by leave-one-module-out training on six modules and testing on the "
        "seventh. Metrics: AUC; area under the precision&#8211;recall curve (AUPRC, informative "
        "here because the positive class is frequent [24]); the Brier score; expected calibration "
        "error (ECE) with ten bins; F1 at a threshold of 0.5; and one operating point aimed at "
        "deployment, the share of the cohort that must be flagged to recover 80% of unsuccessful "
        "registrations.", F)))

    # ------------------------------------------------------------------ 4
    add(("h1", "IV. Features and Models"))
    add(("h2", "A. Feature Blocks"))
    add(("p", fmt(
        "Four blocks of features follow the order in which information arrives [5]. <i>D "
        "(demographics)</i>&#8212;gender, region, highest education, deprivation band, age band, "
        "declared disability, previous attempts and credits studied. <i>R "
        "(registration)</i>&#8212;days from registration to module start, a late-registration "
        "flag, and the module and presentation identifiers. <i>V (VLE activity)</i>&#8212;total "
        "clicks, active days, distinct sites, clicks per active day, clicks in each of the "
        "fourteen busiest activity types, a week-by-week click profile from week 0 to W-1, its "
        "normalised slope, the share of clicks falling in the latest week, the count of active "
        "weeks, and days elapsed since the last activity. <i>A (assessment)</i>&#8212;assessments "
        "due by the cutoff, how many were submitted, the mean, minimum, maximum and standard "
        "deviation of scores, the weight-adjusted score, mean submission delay, late "
        "submissions, missed assessments, and whether anything was submitted at all.", F)))
    add(("p", fmt(
        "Two facts about the early window shape the reading of later results. By week four the "
        "median registration has logged 179 clicks spread over 11 active days, while 37% of "
        "registrations have yet to submit a single assessment&#8212;either the module's first "
        "assessment comes later, or the student has already disengaged. Non-submission is itself "
        "informative, so block A carries an explicit indicator instead of imputing missing "
        "scores.", F)))

    add(("h2", "B. Models"))
    add(("p", fmt(
        "Six models are fitted, the linear, tree and forest implementations taken from "
        "scikit-learn [22]. <i>Logistic regression</i>, L2-penalised, on standardised "
        "numerics and one-hot categoricals, is the conventional transparent baseline. The "
        "<i>spline GAM</i> swaps each numeric feature for a cubic B-spline basis with eight "
        "knots and fits a ridge-penalised logistic model on that expanded basis, so the log-odds "
        "separate into a sum of one-dimensional shapes plus categorical offsets,", F)))
    add(("eq", r"\log\frac{p}{1-p}=\beta_0+\sum_{j=1}^{J} f_j(x_j)+\sum_{k} \gamma_k c_k", "gam", 22))
    add(("p", fmt(
        "with each f_j a spline and each c_k a categorical level. A <i>depth-4 tree</i> supplies "
        "a second interpretable model, one with a different inductive bias. The ensemble family "
        "is <i>random forest</i> [21] (200 trees, minimum leaf 10), <i>XGBoost</i> [19] (400 "
        "trees, depth 5, learning rate 0.05) and <i>LightGBM</i> [20] (400 trees, 31 leaves, "
        "minimum child weight 40, learning rate 0.05, subsample and column subsample 0.8). Trees "
        "take categorical "
        "features natively. Hyper-parameters were fixed by hand before any out-of-fold score "
        "was computed and never "
        "revisited on test folds; no search was run, because the comparison concerns the "
        "inductive bias of each family rather than the best attainable score. The positive class "
        "is the majority here, so no oversampling was applied [23]. Two further fits bound the "
        "comparison: sweeping knots from five to twelve and ridge strength from 0.25 to 2 moves "
        "the additive model by 0.2 AUC points, so its own tuning is not what separates it from "
        "the ensemble, and a two-layer perceptron (64&#8211;32 units, early stopping) reaches "
        "0.843 on the same folds&#8212;level with the GAM and below both boosted ensembles.", F)))
    add(("p", fmt(
        "The GAM is the model we recommend, and its structure is why. A plain logistic "
        "regression is simpler and scores {auc_LR} against {auc_Spline-GAM}; the GAM earns that "
        "half point not on the AUC but because its shape functions show the saturating and "
        "non-monotone stretches a linear model has to flatten, and can be constrained where "
        "policy demands monotonicity. Additivity also means a feature's contribution to a "
        "prediction can be read off its shape function directly, with no surrogate, and the "
        "fitted curves can be plotted and audited one line at a time. Tree ensembles are "
        "explained after the fact with TreeSHAP [12], [13], which returns exact Shapley values "
        "for tree models in polynomial time; we treat those attributions as an independent check "
        "on the additive fit, and as the source of the local case analyses in "
        "Section&#160;VI-C.", F)))

    add(("h2", "C. Explanation Protocols"))
    add(("p", fmt(
        "Explanations are judged here on four axes that a deployment review should demand. "
        "<i>Global importance</i>&#8212;mean absolute SHAP value per feature, summed by block. "
        "<i>Shape</i>&#8212;the sign and monotonicity of each effect, read from GAM shape "
        "functions and from SHAP dependence plots. <i>Local fidelity</i>&#8212;a depth-3 "
        "surrogate tree regressed on the ensemble score, reported with its R-squared, the "
        "fraction of individual predictions it reproduces within ten risk points, and the rule "
        "list itself. <i>Stability</i>&#8212;Spearman correlation of global importance vectors "
        "across resamples, top-ten overlap, and the same quantities across cutoffs, which "
        "exposes whether an explanation has a half-life.", F)))

    # ------------------------------------------------------------------ 5
    add(("h1", "V. Results"))
    add(("h2", "A. Predictive Performance at Four Weeks"))
    add(("tab", "PLACEHOLDER_MAIN", "<b>Table II.</b> Out-of-fold performance at the four-week "
         "cutoff, mean over three repeated five-fold grouped splits (standard deviations all "
         "below 0.001). Cov: share of the cohort that must be flagged to reach 80% recall; "
         "Prec: precision at that operating point. XGBoost is the strongest model in the table; "
         "LightGBM is the reference ensemble whose predictions back the explanation analyses.",
         [0.92 * 72, 0.50 * 72, 0.50 * 72, 0.46 * 72, 0.42 * 72, 0.66 * 72], 6.9))
    add(("p", fmt(
        "Table&#160;II carries the main result. XGBoost ({auc_XGBoost}) is the strongest entry; "
        "our reference ensemble, LightGBM ({auc_LightGBM}), sits 0.1 points behind it and is the "
        "model whose predictions underpin Section&#160;VI. The additive GAM reaches "
        "{auc_Spline-GAM}&#8212;{gap_gam_gbdt} points behind the reference ensemble and "
        "{gap_gam_best} behind the best model&#8212;and the depth-4 tree that a human can draw "
        "on a whiteboard reaches {auc_DecisionTree}. Repeats leave the ordering untouched, with "
        "standard deviations under 0.001. That figure is seed-to-seed variation; the five folds "
        "themselves span 0.845 to 0.854 for the reference ensemble, so what is stable is the "
        "ordering, not the fold composition. On the split that backs Section&#160;VI the same "
        "ensemble scores {auc_lgbm_split}, learner-cluster 95% CI {ci_lgbm_cluster}: that "
        "interval belongs to that split, whereas 0.845&#8211;0.854 describes the five folds of "
        "one repetition. Both gaps are quoted, because the smaller alone would understate the "
        "cost by a tenth of a point. Three observations follow.", F)))
    add(("p", fmt(
        "First, what interpretability costs at this task is small in absolute terms. Moving from "
        "the reference ensemble to the additive GAM gives up {gap_gam_gbdt} AUC points "
        "({gap_gam_best} against the strongest model) and buys an exact decomposition of every "
        "prediction into per-feature shapes; moving on to a depth-4 tree costs a further "
        "{diff_tree}. Whether the trade is worth making depends on the decision it serves: for "
        "advising, we read the GAM as the right default and the tree as the right object to talk "
        "around.", F)))
    add(("p", fmt(
        "Second, what matters more than the AUC is where the cut is placed, and the numbers are "
        "sobering. "
        "Unsuccessful outcomes form the majority class ({p_atrisk}%), so a policy insisting on "
        "80% recall has to contact {flag_LightGBM}% of the cohort, of whom {prec_LightGBM}% are "
        "genuinely at risk&#8212;a lift of roughly {lift_note} on the base rate. Few institutions "
        "hold that capacity. Flagging the top quintile instead captures {r20_LightGBM}% of all "
        "unsuccessful registrations at {p20_LightGBM}% precision. Which regime an institution "
        "chooses is a resourcing question rather than a statistical one, and a single AUC "
        "conceals it.", F)))
    add(("p", fmt(
        "Why the two regimes diverge so sharply is a property of the score distribution, which "
        "is strongly bimodal. A share of {p_silent}% of registrations ({n_silent} in all) logs "
        "no VLE click whatsoever in the first four weeks, and {rate_silent}% of those students "
        "fail or withdraw; that silent group alone supplies {share_atrisk_silent}% of all "
        "unsuccessful outcomes and is separated almost perfectly by the model, which is what "
        "lifts the top quintile to {p20_LightGBM}% precision. The rest of the at-risk population "
        "sits in an uncertain middle where the score is genuinely probabilistic, and there "
        "recall becomes expensive. A deployment that contacts only the certain cases will look "
        "excellent on precision and will quietly miss two thirds of the students it was built "
        "for. The zero-click rule on its own therefore reaches {rate_silent}% precision at "
        "{share_atrisk_silent}% recall, and the fair question is what the model adds beyond it. "
        "Restricted to the 88.3% of registrations with any activity, AUC falls to 0.816 and the "
        "base rate to {act_base}. A quintile drawn inside that sub-cohort captures {cmp_model} "
        "(recall/precision,%): the zero-click rule has nothing left to point at and falls to "
        "{cmp_chance}, a one-variable ranking by days since the last click reaches {cmp_days}, "
        "a three-feature logistic rule on that recency, the mean score and active days reaches "
        "{cmp_lr3}, and the mean score alone {cmp_score}. The model therefore earns its place in "
        "the uncertain middle, not in re-identifying silence&#8212;though the margin over the "
        "simplest informative rule is 7.8 recall points, not the twenty the zero-click comparison "
        "would suggest.", F)))
    add(("p", fmt(
        "Third, calibration is what separates the families. GAM and XGBoost are nearly "
        "calibrated (ECE close to 0.01 each), the random forest is not (ECE {ece_RandomForest}), "
        "and the depth-4 tree looks calibrated only because its leaves are large. For ensembles "
        "that will be thresholded, post-hoc calibration is therefore not optional&#8212;a point "
        "the clinical prediction literature makes repeatedly [25] and learning analytics "
        "under-uses. ECE is computed with ten equal-width bins and quoted without an interval; "
        "because the score is bimodal, most of the mass falls in the two end bins, so a small "
        "ECE partly measures how few bins carry weight, which is why the subgroup reliability "
        "diagram is read alongside it rather than trusted alone.", F)))

    add(("p", fmt(
        "A difference of a few tenths of an AUC point is routinely dismissed as noise, so we "
        "tested it rather than asserting it. Table&#160;III gives paired comparisons over "
        "identical out-of-fold rows, with learner-cluster bootstrap intervals (2,000 replicates "
        "resampling the {n_stud} students rather than the {n_reg} registrations, since one "
        "learner contributes several rows) and Holm correction across the four tests; every "
        "comparison clears p&#160;&lt;&#160;0.001. Because the table rests on the single split "
        "that backs Section&#160;VI, its deltas differ in the second decimal from the three-seed "
        "means of Table&#160;II, and the interval belongs to that split rather than to an average "
        "over repeats. The reading is not that interpretable models match the ensembles but that "
        "they trail them *reliably*&#8212;by {d_dt_gam} for the tree and {d_gam_lgbm} for the "
        "additive model in that split. A third of a point is also less than one hyper-parameter "
        "choice moves the ensemble (2.7 points for LightGBM), so the significance is statistical "
        "rather than practical, and any such claim must record that both sides ran at default "
        "settings on identical folds.", F)))
    add(("tab", [[
        "Comparison", "AUC diff", "95% CI", "p (Holm)"], [
        "GAM vs LGBM", "-0.005", "[−0.006, −0.003]", "<0.001"], [
        "Tree vs GAM", "-0.041", "[−0.045, −0.038]", "<0.001"], [
        "RF vs LGBM", "-0.006", "[−0.008, −0.005]", "<0.001"], [
        "GAM vs XGB", "-0.005", "[−0.007, −0.004]", "<0.001"]],
        fmt("<b>Table III.</b> Paired learner-cluster bootstrap comparisons of out-of-fold AUC "
        "(2,000 replicates over 28,785 learners, identical folds), Holm-corrected; all "
        "intervals exclude zero, and the four comparisons were fixed before the resampling began. "
        "Computed on the single split that backs Section&#160;VI, so "
        "deltas differ in the second decimal from Table&#160;II.", F),
        [1.02 * 72, 0.62 * 72, 1.10 * 72, 0.55 * 72], 6.9))

    add(("p", fmt(
        "A further check puts the cost inside the model class rather than inside "
        "interpretability. Holding the folds fixed and sweeping one hyper-parameter at a time, "
        "the depth-4 tree spans {sp_DecisionTree} AUC points between depth 2 and 10 "
        "({smin_DecisionTree} to {smax_DecisionTree}), the forest {sp_RandomForest}, LightGBM "
        "{sp_LightGBM}, XGBoost {sp_XGBoost}, and the additive GAM only {sp_SplineGAM}. That row "
        "argues against a pure capacity account: the tree does gain {sp_DecisionTree} points by "
        "deepening, yet its best depth still stops at {smax_DecisionTree}, under the GAM, so "
        "depth is not the separator. What separates them is representational&#8212;a split "
        "hierarchy imitates a smooth monotone effect with a staircase of thresholds and cannot "
        "carry it across the whole range. The same sweep warns against treating one number as "
        "the model's performance: the ensemble's best learning rate ({sbestv_LightGBM}) yields "
        "{sbest_LightGBM} against {sdef_LightGBM} at the default, so the {gap_gam_gbdt}-point gap "
        "holds only when both sides sit at defaults on identical folds&#8212;a caveat we state "
        "rather than bury.", F)))

    add(("h2", "B. What Earliness Costs"))
    add(("p", fmt(
        "Fig.&#160;1 and Table&#160;IV trace what time buys. One week in, the ensemble already "
        "reaches AUC {auc_w1}; by week four, {auc_w4}; by week twelve, {auc_w12}. Weeks one to "
        "four add {gain_w1_w4} points and weeks four to twelve a further {gain_w4_w12}, so "
        "{pct_early}% of the improvement the term can supply arrives in its first month. That is "
        "the empirical case for warning early instead of grading late: waiting halves the "
        "signal.", F)))
    add(("fig", "fig2_sweep.png", "<b>Fig. 1.</b> Out-of-fold AUC as a function of the cutoff "
         "week, one grouped five-fold split per cutoff (the values tabulated in Table&#160;IV, "
         "so no band is drawn). The marked cutoff is the "
         "four-week operating point used throughout the paper.", 0.94))
    add(("p", fmt(
        "The same curves show how thin the separation between families is while the term is "
        "still young. At every cutoff LightGBM stays within {max_gap_points} AUC points of the "
        "additive GAM, and at week one the two cannot be told apart ({auc_w1_gbm} against "
        "{auc_w1_gam}). The depth-4 tree sits {tree_gap_w1} to {tree_gap_w3} points below the GAM "
        "up to week three, and the gap then narrows to {tree_gap_w8} by week eight and "
        "{tree_gap_w12} by week twelve: a staircase approximates a smooth curve better once the "
        "data are dense, which locates the tree's deficit in its representation rather than in "
        "its capacity. Interpretability is dear when bought with a single tree and nearly "
        "free&#8212;0.4 of an AUC point&#8212;when bought with an additive model, which is "
        "the practical message this study carries. What time changes is not the winner but the "
        "amount any model knows: the same ensemble travels from {auc_w1} to {auc_w12} as clicks "
        "and scores pile up.", F)))

    wtab = [["Week", "LR", "GAM", "Tree", "RF", "LGBM", "XGB"]]
    if F.get("sweep"):
        rf = F.get("rf_cut", {})
        for w in [1, 2, 3, 4, 6, 8, 12]:
            row = [str(w)]
            for m in ["LR", "Spline-GAM", "DecisionTree"]:
                row.append(f"{F['sweep'][m][str(w)]['auc'][0]:.3f}")
            row.append(f"{rf[str(w)]:.3f}" if str(w) in rf else "&#8212;")
            for m in ["LightGBM", "XGBoost"]:
                row.append(f"{F['sweep'][m][str(w)]['auc'][0]:.3f}")
            wtab.append(row)
    add(("tab", wtab, "<b>Table IV.</b> AUC by cutoff week and model (single grouped split "
         "per week, five folds); these values come from one split per cutoff and are not "
         "comparable digit-for-digit with the three-seed means in Table&#160;II.",
         [0.44 * 72, 0.50 * 72, 0.50 * 72, 0.50 * 72, 0.50 * 72, 0.52 * 72, 0.50 * 72], 7.0))

    add(("h2", "C. Which Data Actually Carry the Signal"))
    add(("fig", "fig8_ablation.png", "<b>Fig. 2.</b> Cumulative feature-block ablation at "
         "week four. D: demographics; R: registration; V: VLE activity; A: assessment. The "
         "rightmost pair removes demographic attributes entirely.", 0.84))
    add(("p", fmt(
        "Blocks added in the order they become available (Fig.&#160;2) show where the signal "
        "lives. Demographics alone yield AUC {auc_d_only}; the registration record barely moves "
        "it ({auc_dr}); four weeks of VLE activity carry it to {auc_drv}; and the first "
        "assessment results take it to {auc_full}. Behaviour is worth about eighteen AUC "
        "points&#8212;an order of magnitude beyond all demographic attributes together&#8212;"
        "while the assessment block, for all that it holds a few scores per student, adds "
        "roughly three more. That last figure matters: four weeks in, how a student behaves "
        "predicts about as much as how a student has scored, precisely because those who submit "
        "nothing are mostly those who click nothing. Assessment becomes decisive later in the "
        "term, past the moment at which advising decisions are taken.", F)))
    add(("p", fmt(
        "The rightmost pair of bars tests a variant built for fairness: every demographic "
        "attribute dropped, leaving registration, VLE and assessment. It reaches {auc_nodemo}, "
        "essentially the full model. A model blind to gender, age, disability and deprivation can "
        "do the whole job, which is the default we would hand an institution that rightly worries "
        "about encoding protected attributes into decisions about people. Dropping the module "
        "and presentation identifiers costs 0.15 AUC points for the ensemble (0.8465 against "
        "0.848) but 1.3 for the additive model (0.831 against 0.844), which cannot rebuild a "
        "course offset out of interactions; the identifier that leads the SHAP ranking "
        "(Section&#160;VI-A) is therefore mostly arithmetic for the ensemble and partly substance "
        "for the GAM. Behavioural features "
        "are not fair by themselves&#8212;engagement can stand in for circumstance&#8212;but they "
        "are the ones an intervention can actually change.", F)))

    add(("h2", "D. Generalisation to Unseen Courses"))
    add(("fig", "fig9_transfer.png", "<b>Fig. 3.</b> Leave-one-module-out transfer for the "
         "reference ensemble. Each bar is one held-out module, trained on the other six; the "
         "dashed line is the at-risk base rate of the held-out module.", 0.82))
    add(("p", fmt(
        "Train and test on the same seven modules and readiness for a new course is overstated. "
        "Leave-one-module-out drops AUC into the band {tr_min} to {tr_max} (Fig.&#160;3), and the "
        "modules spread widely: transfer to CCC and DDD is good ({tr_ccc} and {tr_ddd}), to GGG "
        "poor ({tr_ggg}), a small cohort whose first assessments arrive after the cutoff. Two "
        "mechanisms account for the drop&#8212;course-specific priors (the module identifier, "
        "legitimately informative, since a new presentation of a known module is easier to "
        "predict) disappear for an unseen course, and the first assessment falls at different "
        "times, leaving some courses with an empty assessment block at week four. Missing "
        "assessment evidence hurts more than a shortage of positives. The additive model lands "
        "in the same band ({tr_gam_min} to {tr_gam_max}) and matches the ensemble on average "
        "({tr_mean_gam} against {tr_mean_lgbm}), so additivity buys no transfer: what it buys is "
        "agreement about the average, not about where the damage falls&#8212;the ensemble "
        "leads on BBB and FFF, the GAM on AAA, CCC, DDD and GGG. These numbers are what a "
        "new course should expect, and they argue for recalibrating on the target module instead "
        "of transplanting thresholds.", F)))

    add(("h2", "E. Subgroup Performance and Calibration"))
    add(("fig", "fig7_fairness.png", "<b>Fig. 4.</b> Top: AUC within demographic subgroups at "
         "week four. Bottom: reliability diagram of the ensemble on pooled out-of-fold "
         "predictions.", 0.74))
    add(("p", fmt(
        "Fig.&#160;4 audits the ensemble over gender, declared disability, age band, deprivation "
        "band and module. Subgroup AUC runs from {fa_min} to {fa_max}, and inside the demographic "
        "attributes it is tight: {auc_gender_f} for women against {auc_gender_m} for men, "
        "{auc_dis_f} against {auc_dis_t} with and without a declared disability. Ranking is near "
        "group-blind; treatment is not. At a fixed 0.5&#8212;an audit baseline rather than a "
        "deployment rule, given a positive base rate of {p_atrisk}% and no institution "
        "thresholding at the default&#8212;students with a declared disability show a "
        "false-positive rate of {fpr_dis_t}% against {fpr_dis_f}% for everyone else, a gap of "
        "{fpr_gap} points on pooled out-of-fold predictions, while the true-positive rate for "
        "those aged 55 and over is {tpr_age_old}% against {tpr_age_young}% in the youngest band "
        "&#8212;a gap in the true-positive rate, not in the false-positive rate quoted for age "
        "in Section&#160;V-F. "
        "Equal AUC with unequal error rates is the textbook equalised-odds failure, and it "
        "appears here because base rates differ: a group with a higher prior is flagged more "
        "often at any threshold that is not group-specific. Calibration gaps reach {calib_max} "
        "percentage points&#8212;invisible on a reliability plot, visible in a caseload. The "
        "remedy is not to drop the attribute but to set thresholds per group once the "
        "institution has decided which error ratio it will defend; Section&#160;V-F applies that "
        "remedy across false-positive ceilings of 5% to 40%, where capacity rather than 0.5 sets "
        "the cut. The audit sits at 0.5 and not at the 80%-recall point of Section&#160;V-A "
        "because that point is defined over the whole cohort: applying it group by group would "
        "presuppose the per-group policy the audit is meant to test.", F)))
    add(("p", fmt(
        "Two other subgroups warrant a word. Registrations missing deprivation data "
        "({imd_missing}% of all) rank worst ({auc_imd_na}), consistent with missingness itself "
        "carrying information about administrative contact; we read that as a data-quality "
        "problem rather than a modelling one. Module-level AUC spans {auc_mod_ggg} (GGG) to "
        "{auc_mod_ccc} (CCC), a spread far wider than any demographic gap&#8212;the fairness fact "
        "most likely to bite in practice. The two weakest modules fail differently: GGG pairs a "
        "low AUC with an odd outcome mix ({base_ggg}% at risk yet {dist_ggg}% distinctions) and a "
        "first assessment after the cutoff, while AAA has the lowest at-risk base rate "
        "({base_aaa}%) and so the least to rank on. Neither should inherit a threshold "
        "transplanted from the pooled model: measured risk depends heavily on the course taken, "
        "and per-module recalibration is the minimum&#8212;which is what the transfer experiment "
        "reaches by another route.", F)))


    add(("h2", "F. Is the Gap in the Score or in the Cut?"))
    add(("fig", "fig15_fairness_mitigation.png", "<b>Fig. 5.</b> (a) False- and "
         "true-positive rates by declared disability before and after group-specific "
         "thresholds. (b) Pooled recall as a common false-positive ceiling is enforced, "
         "either with one global threshold or with per-group thresholds.", 0.86))
    add(("p", fmt(
        "What Fig.&#160;4 shows is a disparity in the decision, not in the ranking. Can "
        "post-processing remove it? Learners are split 50/50 into "
        "calibration and evaluation halves, per-group thresholds are estimated on the "
        "calibration half alone and applied to the other half, over five splits. This measures "
        "the same quantity as Section V-E by another route, and the two estimates should not be "
        "quoted interchangeably: {fpr_gap} points is the pooled out-of-fold gap at a common "
        "threshold, {fpr_gap_base} points is the baseline re-estimated inside the 50/50 split, "
        "and the {fpr_protocol_delta}-point difference is the price of holding out a tuning half, "
        "not a different disparity. Nor is the gap a constant of the score: beside the {fpr_gap} "
        "points at 0.5, it is 0.1 points at the top quintile and 12.2 at 80% recall, so the "
        "disparity depends on where the cut sits at least as much as on the ranking. One "
        "threshold per disability level, chosen so that both "
        "levels meet the pooled false-positive rate, cuts the mean gap from {fpr_gap_base} to "
        "{fpr_gap_eq} points over the five splits: the disabled group's false-positive rate "
        "falls from {fpr_dis_base}% to {fpr_dis_eq}%, and the comparison group sits at "
        "{fpr_ndis_base}% and {fpr_ndis_eq}%. Those two group rates differ by less than a point, "
        "so the {fpr_gap_eq}-point figure is the mean of the per-split gaps rather than the "
        "difference between the two averages; it is a partial closure, and the per-split gap "
        "runs from {fpr_gap_eq_lo} to {fpr_gap_eq_hi} points (SD {fpr_gap_eq_sd}). The "
        "true-positive gap narrows from {tpr_gap_base} to {tpr_gap_eq} points, and little is "
        "paid for it&#8212;recall holds at {rec_base}% and coverage at {cov_base}% "
        "(Table&#160;V). Fig.&#160;5(b) generalises the point: from 5% to 40% ceilings the "
        "equalised rule tracks the single global threshold almost exactly, the widest difference "
        "in pooled recall being 0.3 points, so at this cutoff and for this attribute the "
        "equalised decision costs nothing the intervention budget can feel. The scope is "
        "deliberate: under the same rule the age false-positive gap only moves from "
        "{age_gap_base} to {age_gap_eq} points, so what is free here is free for disability, not "
        "for every protected attribute.", F)))
    add(("tab", [[
        "Model", "Decision rule", "Recall", "Prec.", "Cov.", "FPR gap", "TPR gap"], [
        "LightGBM", "Global cut 0.5", "73.3", "80.8", "47.6", "10.1", "7.3"], [
        "LightGBM", "Equalised FPR", "73.3", "80.7", "47.7", "2.5", "1.9"], [
        "LightGBM", "Equalised TPR", "73.2", "80.8", "47.6", "2.1", "2.3"], [
        "Spline-GAM", "Global cut 0.5", "72.4", "80.7", "47.1", "10.5", "8.4"], [
        "Spline-GAM", "Equalised FPR", "72.2", "80.7", "47.0", "2.6", "1.7"], [
        "Spline-GAM", "Equalised TPR", "72.5", "80.5", "47.3", "2.5", "2.7"], [
        "LGBM, IMD", "Global cut 0.5", "73.3", "80.8", "47.6", "26.1", "27.4"], [
        "LGBM, IMD", "Equalised FPR", "72.8", "80.6", "47.5", "5.8", "8.9"]],
        fmt("<b>Table V.</b> Cost of fairness at week four (percent; gaps in percentage points, "
        "the first six rows between disability levels, the last two across deprivation bands). "
        "Thresholds tuned on a held-out half of learners, five "
        "repeated splits; the untouched pooled estimate is {fpr_gap} points for LightGBM "
        "(Section&#160;V-E) against {fpr_gap_base} here.", F), [0.76 * 72, 0.94 * 72, 0.34 * 72, 0.34 * 72, 0.32 * 72,
                            0.36 * 72, 0.36 * 72], 6.9))
    add(("p", fmt(
        "Two limits keep this from being a solution. First, a threshold can only re-allocate "
        "errors that the score already separates; where a subgroup is small and its scores sit "
        "inside the uncertain middle, no cut removes the gap, and the age disparity barely "
        "moves ({age_gap_base} to {age_gap_eq} points) for exactly that reason. Second, group "
        "thresholds are a legal and ethical choice, not a statistical one: they require the "
        "institution to declare which error it is equalising, to maintain that declaration "
        "across terms, to accept that the same score implies different decisions for different "
        "students, and in some jurisdictions to satisfy rules that restrict exactly that. We "
        "report the mechanics because a paper that documents a disparity "
        "and stops there leaves practitioners with no legal move; we do not claim that the "
        "mechanics settle the policy question.", F)))
    add(("p", fmt(
        "The largest disparity in the audit is neither of those (Table&#160;V, last two rows). "
        "Across deprivation bands the "
        "false-positive gap at 0.5 is {imd_gap_base} points, more than twice the disability gap, "
        "and equalising disability leaves it at {imd_gap_after_dis}: a rule tuned on one "
        "attribute does nothing for another. Equalising on deprivation itself does move it, "
        "from {imd_gap_base} to {imd_gap_eq} points, at a cost of {imd_rec_cost} recall points, "
        "which makes the pattern plain&#8212;the remedy works in proportion to how much of the "
        "group difference the score already separates, and that is a property of the score, not "
        "of the rule. We quote the number because a fairness section that audits four attributes "
        "and reports only the one it can repair is worth less than one that also reports the "
        "one it cannot.", F)))

    add(("h2", "G. Beyond the Binary Label"))
    add(("p", fmt(
        "Practitioners sometimes have to tell withdrawal from failure, so a four-class variant "
        "was fitted on the same week-four features. It reaches macro-F1 {f4_macro} and accuracy "
        "{f4_acc}, with per-class recall {f4_rec_dist} for Distinction, {f4_rec_pass} for Pass, "
        "{f4_rec_fail} for Fail and {f4_rec_wd} for Withdrawn. The confusion sits exactly where "
        "theory predicts: Fail and Withdrawn share the behavioural signature of disengagement, so "
        "most predicted failures are withdrawals and the reverse, and Distinction stays nearly "
        "invisible at four weeks because excellence has not had time to show. We report the "
        "variant as a limitation rather than a result&#8212;macro-F1 {f4_macro} with Distinction "
        "recall {f4_rec_dist} is not deployable at this cutoff. Collapsing to two classes "
        "therefore costs little for advising, where both outcomes call for contact, but would "
        "cost a great deal for reporting, where leaving and failing carry different "
        "institutional meaning.", F)))

    # ------------------------------------------------------------------ 6
    add(("h1", "VI. Explaining the Model"))
    add(("h2", "A. What the Model Looks At"))
    add(("fig", "fig4_beeswarm.png", "<b>Fig. 6.</b> TreeSHAP summary for the week-four "
         "LightGBM ensemble on an 8,000-registration sample. Each point is a registration; colour "
         "encodes the feature value; the vertical bar marks the interquartile span.", 0.86))
    add(("p", fmt(
        "Fig.&#160;6 orders features by mean absolute SHAP value. Behaviour holds the bulk of the "
        "attribution&#8212;{share_vle}% of the total, against {share_asmt}% for assessment and "
        "{share_demo}% for demographics&#8212;yet the largest single contributor is the module "
        "identifier ({shap_mod}), followed by days since the last VLE event ({shap_recency}) and "
        "the mean assessment score ({shap_score}); prior education ({shap_edu}), active weeks "
        "({shap_aw}), the credit-weighted score and the deprivation band come next. The module "
        "identifier leads because courses differ widely in difficulty and in attrition, and the "
        "model may legitimately exploit that prior; it is also the one feature that an unseen "
        "course does not have, which is why transfer is measured separately in "
        "Section&#160;V-D. The four blocks account for the whole ranking: the registration "
        "block supplies the remaining 13.9%, almost all of it through those same two "
        "identifiers.", F)))
    add(("p", fmt(
        "The shapes are monotone and pedagogically sensible. Risk climbs with the gap since the "
        "last click (rank correlation between feature value and SHAP value {corr_recency}), "
        "falls with the mean score ({corr_score}), and falls with active days: registrations "
        "active on four days or fewer average a predicted risk of {risk_inactive}, those active "
        "on twenty-three days or more {risk_active}, and the observed rates in those bins are "
        "{obs_inactive} and {obs_active}. Sharpest of all is the absence of a submission&#8212;"
        "among registrations with an assessment already due, those with nothing submitted score "
        "{risk_nosub}, against {risk_nodue} for the minority whose first assessment is not yet "
        "due. None of this would surprise an experienced tutor, which is itself a useful "
        "result: a model whose top features raise eyebrows in a staff room usually has a "
        "leakage problem.", F)))
    add(("p", fmt(
        "One caution cuts against the ablation and deserves saying outright. Prior education and "
        "the deprivation band rank high in SHAP&#8212;fourth and seventh&#8212;yet deleting every "
        "demographic attribute costs only about one AUC point (Section&#160;V-C). Both hold: the "
        "trees lean on these attributes, but most of what they encode is already visible in "
        "behaviour, so their marginal contribution is small. A feature can matter to the "
        "arithmetic and be redundant to the prediction.", F)))

    add(("h2", "B. The Shape of the Effects"))
    add(("fig", "fig5_dependence.png", "<b>Fig. 7.</b> SHAP dependence for four features. Blue "
         "points are registrations; the red line is the conditional mean in bins of the feature.", 0.76))
    add(("p", fmt(
        "Fig.&#160;7 shows that the effects are not only ordered but shaped. Dependence on active "
        "days is concave&#8212;the first ten days buy most of the risk reduction ({risk_inactive} "
        "for four days or fewer, {risk_active} for twenty-three or more)&#8212;and beyond roughly "
        "twenty days the marginal contribution flattens. Dependence on the mean score falls "
        "almost linearly through the fifties and sixties and flattens only above eighty, so a "
        "comfortable pass and a marginal one are different states rather than neighbouring "
        "points on a scale. The gap since the last click is the sharpest dependence of the "
        "four: three weeks of inactivity behaves almost like an absorbing state in the model's "
        "arithmetic, and the contribution grows nearly linearly with the length of the gap. "
        "Dependence on the number of active weeks is stepwise rather than smooth&#8212;each extra "
        "week of contact lowers the log-odds by a roughly constant increment, which is what a "
        "register-based view of attendance would lead a tutor to expect. All four shapes are "
        "pedagogically coherent, since contact matters most where it is scarce and a marginal "
        "pass differs qualitatively from a comfortable one, and each marks a region of the "
        "feature space that a tutor can act on&#8212;which a bare ranking cannot.", F)))
    add(("p", fmt(
        "The ensemble never gains more than {max_gap_points} points on the additive GAM at any "
        "cutoff, and the GAM's shape functions reproduce the profiles above. That bound is the "
        "best evidence we can offer that additivity is not distorting the fit: an ensemble that "
        "gains so little is unlikely to be exploiting interaction structure the additive terms "
        "do not already capture&#8212;though a small gain cannot by itself rule interaction out, "
        "because correlated interaction terms can substitute for main effects. With that "
        "caveat, the explanations "
        "reported here sit close to an exact description of the scoring function rather than a "
        "post-hoc approximation to it.", F)))

    add(("h2", "C. Three Students"))
    add(("p", fmt(
        "Aggregate explanations leave an advisor without a Monday-morning plan. Fig.&#160;8 gives "
        "three local explanations that span the decision space. The first registration scores "
        "{p_case1}: almost nothing submitted, and the last VLE activity weeks old, so the "
        "explanation is a list of absences and the advice writes itself. The second scores "
        "{p_case2}: everything submitted but scored poorly, with activity decaying, which points "
        "at academic difficulty rather than disengagement. The third is a borderline pass at "
        "{p_case3}, where positive and negative contributions nearly cancel&#8212;the case in "
        "which an advisor should trust the model least, and a deployed interface should say as "
        "much.", F)))
    add(("fig", "fig6_waterfall.png", "<b>Fig. 8.</b> Local attributions for three registrations "
         "at the four-week cutoff. Red bars increase risk, blue bars decrease it.", 0.74))

    add(("h2", "D. Rules a Human Can Check"))
    add(("p", fmt(
        "A depth-3 surrogate regressed on the ensemble score attains R-squared {sur_r2} and "
        "reproduces {sur_agree}% of individual predictions within ten risk points. Both numbers "
        "sit near the ceiling of an eight-leaf surrogate, so they measure the budget it was "
        "given as much as the local complexity of the ensemble: a deeper surrogate would "
        "reproduce more of the score and would also stop being printable, which is the trade the "
        "rule list is meant to expose. Its rule list "
        "is short enough to print: a registration active in at most two weeks and unseen for "
        "three scores between 0.65 and 0.97 depending on the module; one active in three weeks "
        "or more with a mean score under 70 scores 0.61; and one above 70 with at least sixteen "
        "active days scores 0.24. The surrogate is not the model, since it recovers two thirds of "
        "the variance and fewer than half of individual predictions within ten risk points, so "
        "these rules are a bridge to conversation rather than a replacement for the score. What "
        "they do show is that the ensemble behaves, locally, like a small set of threshold "
        "conjunctions&#8212;usable as a scaffold for the typical case in a staff meeting, and "
        "equally easy to caricature.", F)))

    add(("h2", "E. How Long an Explanation Lasts"))
    add(("fig", "fig10_stability.png", "<b>Fig. 9.</b> Rank correlation of global feature "
         "importance between cutoffs. Values near one indicate that the same features dominate "
         "at both cutoffs.", 0.64))
    add(("p", fmt(
        "Resampling leaves explanations intact: across five independent resamples the mean "
        "Spearman correlation of global importance is {stab_fold}, with {stab_top10}% overlap "
        "among the ten leading features. Time does not. Rank correlation between week one and "
        "week four is {corr_1_4}, and between week one and week eight {corr_1_8} "
        "(Fig.&#160;9). Early in the term the model leans on registration timing and the first "
        "clicks, because that is all it has; later, assessment performance takes over. A system "
        "that computes explanations once and caches them will explain last month's model to this "
        "month's tutor: explanations have a half-life and belong on the same refresh schedule as "
        "the model.", F)))

    add(("h2", "F. Explanations Are Not Interventions"))
    add(("p", fmt(
        "A SHAP plot invites being read as a menu of interventions, so we tested that reading. "
        "For registrations in the uncertain band (predicted risk between 0.5 and 0.85, "
        "n&#160;=&#160;{n_moderate}), lifting all engagement features by 50% lowers the mean "
        "score by {wi_engagement_+50pct} points and pushes {wi_engagement_+50pct_share}% of them "
        "under the 0.5 threshold; lifting assessment scores by ten points lowers it by "
        "{wi_assessment_+10pts} and pushes {wi_assessment_+10pts_share}% under; doing both lowers "
        "it by {wi_both} and pushes {wi_both_share}% under. The assessment lever is roughly ten "
        "times the engagement lever&#8212;the same redundancy the ablation showed, since once "
        "scores are in, extra clicking buys little. The deeper point is structural: attributions "
        "describe the model's arithmetic, not the causal graph of a student's life, so treating "
        "a ranking as a list of levers promises more than it can deliver. The honest use is to "
        "open a conversation&#8212;&#8216;you have not been in the VLE for two weeks, and your "
        "last assignment scored 52&#8217;&#8212;not to prescribe an action.", F)))

    # ------------------------------------------------------------------ 7
    add(("h1", "VII. Discussion"))
    add(("h2", "A. What a Deployment Should Look Like"))
    add(("p", fmt(
        "The results imply a specific design. Score every registration weekly, recomputing "
        "features and explanations from scratch; make the additive GAM the default scorer, "
        "since it sits within {gap_gam_gbdt} AUC points of the reference ensemble and "
        "{gap_gam_best} of the strongest model tested while remaining inspectable line by line; "
        "keep the ensemble as a shadow model that flags when additivity starts to cost real "
        "accuracy; calibrate before thresholding and recalibrate per module and per demographic "
        "group; and give advisors a ranked list with three reasons and a suggested action rather "
        "than a bare number. At 80% recall the system flags {flag_LightGBM}% of the cohort; at a "
        "nominal fifteen minutes per contact that is roughly 140 advisor hours per thousand "
        "registrations, against 50 for the top quintile, so the capacity question belongs before "
        "launch, not after.", F)))
    add(("h2", "B. Validity and Limitations"))
    add(("p", fmt(
        "Four limitations bound the claims. The data come from one distance-learning institution "
        "with its own assessment regime and were collected in 2013&#8211;2014, so absolute numbers "
        "travel badly and click counts are a proxy whose meaning drifts as platforms change; what "
        "generalises is the protocol and the qualitative findings. The label merges failure with "
        "withdrawal, two "
        "outcomes with different causes and different interventions, and the four-class variant "
        "prices that merge. The demographic attributes are coarse and partly missing, and the "
        "deprivation band is reported rather than measured. Finally, all analyses are "
        "observational: we identify signals that predict, and stop deliberately short of "
        "asserting that changing them changes outcomes&#8212;a claim the what-if probe "
        "was built to discipline rather than support. Two checks we did not run would come "
        "first in any deployment. We test generalisation to new students and to an unseen "
        "course, but not to a new cohort in time&#8212;the release spans two years and we do "
        "not train on one and test on the other&#8212;and we do not replicate on a second "
        "institution, so the transfer we measure is across courses, not across settings.", F)))
    add(("h2", "C. Ethics"))
    add(("p", fmt(
        "OULAD is released under CC-BY 4.0 with k-anonymity and l-diversity applied [1], and no "
        "attempt was made to re-identify anyone. This is a secondary analysis of that public, "
        "de-identified release: no new data were collected, no student was contacted, and "
        "nothing was linked to other institutional records, so the privacy guarantee relied on is "
        "the one built into the release and no further consent was needed. The scores studied are "
        "advisory; the defensible use is to route support, not to gate access to assessment, to "
        "rank students in public, to inform disciplinary action, or to feed any admissions, "
        "progression or funding decision. Trained on historical outcomes, the model encodes "
        "whatever disadvantage that record carries, and the subgroup audit of Section V-E is the "
        "minimum responsible check&#8212;to be repeated on every cohort, since the calibration "
        "gaps are properties of the data rather than of the algorithm. A deployed system should "
        "log who queried a score and why, and give the student a way to see and contest the "
        "reasons attached to their own record.", F)))

    add(("h2", "D. Reproducibility and AI-use Disclosure"))
    add(("p", fmt(
        "Every number here comes from the scripts and stored results, public at "
        "<font size=\"7\">anonymous.4open.science/r/student-risk-oulad-anonymous-FC0C</font>: "
        "feature construction, "
        "grouped cross-validation, the sweep, the ablation, transfer, fairness, the explanation "
        "diagnostics and the sensitivity analyses. Splits are seeded and learners are grouped "
        "across folds throughout, so no reported difference is a fold artefact. Following IEEE "
        "policy, generative AI tools helped draft prose and prepare code; all experiments ran "
        "against the public release, every figure was regenerated from stored predictions, and no "
        "result, table entry or citation was produced by a model. Responsibility for the content "
        "rests with the authors.", F)))

    add(("h1", "VIII. Conclusion"))
    add(("p", fmt(
        "Student-performance prediction is usually framed as a trade in which accuracy is bought "
        "with opacity; that framing describes the trade measured here badly. At a four-week "
        "cutoff an additive model gives up {gap_gam_gbdt} AUC points against the ensemble used "
        "for our explanations and {gap_gam_best} against the strongest model tested&#8212;"
        "{auc_Spline-GAM} against {auc_LightGBM} and {auc_XGBoost} on identical features, folds "
        "and {n_reg} registrations&#8212;while a single readable tree gives up a further "
        "{diff_tree}. The additive model also holds its ground on an unseen course, where the two "
        "families end level ({tr_mean_gam} against {tr_mean_lgbm} averaged over the seven "
        "held-out modules) but fail on different courses, and it decomposes exactly instead of "
        "approximating after the fact. Its "
        "explanations centre on how regularly a student works and on assessment scores&#8212;"
        "behaviour carries {share_vle}% of the attribution against {share_asmt}% for "
        "assessment&#8212;take shapes a tutor can "
        "describe, reproduce across resamples, and decay through the term in a way any deployment "
        "must plan for. The pipeline and its stored results are public so that these claims "
        "can be checked and corrected.", F)))

    # ------------------------------------------------------------------ refs
    add(("h1", "References"))
    refs = [
        "J. Kuzilek, M. Hlosta, and Z. Zdrahal, &#8220;Open University Learning Analytics "
        "dataset,&#8221; <i>Scientific Data</i>, vol. 4, Art. no. 170171, 2017.",
        "G. Siemens and P. Long, &#8220;Penetrating the fog: Analytics in learning and "
        "education,&#8221; <i>EDUCAUSE Review</i>, vol. 46, no. 5, pp. 30&#8211;32, 2011.",
        "C. Romero and S. Ventura, &#8220;Educational data mining: A review of the state of the "
        "art,&#8221; <i>IEEE Trans. Systems, Man, and Cybernetics, Part C</i>, vol. 40, no. 6, "
        "pp. 601&#8211;618, 2010.",
        "R. S. J. d. Baker and P. S. Inventado, &#8220;Educational data mining and learning "
        "analytics,&#8221; in <i>Learning Analytics</i>, New York: Springer, 2014, pp. 61&#8211;75.",
        "D. Tempelaar, B. Rienties, and B. Giesbers, &#8220;In search for the most informative "
        "data for feedback generation: Learning analytics in a data-rich context,&#8221; "
        "<i>Computers in Human Behavior</i>, vol. 47, pp. 157&#8211;167, 2015.",
        "C. Herodotou, B. Rienties, A. Boroowa, Z. Zdrahal, and M. Hlosta, &#8220;A large-scale "
        "implementation of predictive learning analytics in higher education: The teachers' "
        "role and perspective,&#8221; <i>Educational Technology Research and Development</i>, "
        "vol. 67, pp. 1273&#8211;1306, 2019.",
        "N. Tomasevic, N. Gvozdenovic, and S. Vranes, &#8220;An overview and comparison of "
        "supervised data mining techniques for student exam performance prediction,&#8221; "
        "<i>Computers &amp; Education</i>, vol. 149, Art. no. 103955, 2020.",
        "H. Waheed, S.-U. Hassan, N. R. Aljohani, J. Hardman, S. Alelyani, and R. Nawaz, "
        "&#8220;Predicting academic performance of students from VLE big data using deep "
        "learning models,&#8221; <i>Computers in Human Behavior</i>, vol. 104, Art. no. 106189, 2020.",
        "S. M. Jayaprakash, E. W. Moody, E. J. Lauria, J. R. Regan, and J. D. Baron, &#8220;Early "
        "alert of academically at-risk students: An open source analytics initiative,&#8221; "
        "<i>Journal of Learning Analytics</i>, vol. 1, no. 1, pp. 6&#8211;47, 2014.",
        "R. F. Kizilcec and H. Lee, &#8220;Algorithmic fairness in education,&#8221; in <i>The "
        "Ethics of Artificial Intelligence in Education</i>, Cheltenham: Edward Elgar, 2022.",
        "M. Hardt, E. Price, and N. Srebro, &#8220;Equality of opportunity in supervised "
        "learning,&#8221; in <i>Proc. NeurIPS</i>, 2016, pp. 3315&#8211;3323.",
        "S. M. Lundberg and S.-I. Lee, &#8220;A unified approach to interpreting model "
        "predictions,&#8221; in <i>Proc. NeurIPS</i>, 2017, pp. 4765&#8211;4774.",
        "S. M. Lundberg, G. Erion, H. Chen, A. DeGrave, J. M. Prutkin, B. Nair, R. Katz, "
        "J. Himmelfarb, N. Bansal, and S.-I. Lee, &#8220;From local explanations to global "
        "understanding with explainable AI for trees,&#8221; <i>Nature Machine Intelligence</i>, "
        "vol. 2, pp. 56&#8211;67, 2020.",
        "M. T. Ribeiro, S. Singh, and C. Guestrin, &#8220;&#8216;Why should I trust you?&#8217;: "
        "Explaining the predictions of any classifier,&#8221; in <i>Proc. ACM SIGKDD</i>, 2016, "
        "pp. 1135&#8211;1144.",
        "C. Rudin, &#8220;Stop explaining black box machine learning models for high stakes "
        "decisions and use interpretable models instead,&#8221; <i>Nature Machine Intelligence</i>, "
        "vol. 1, pp. 206&#8211;215, 2019.",
        "Y. Lou, R. Caruana, and J. Gehrke, &#8220;Intelligible models for classification and "
        "regression,&#8221; in <i>Proc. ACM SIGKDD</i>, 2012, pp. 150&#8211;158.",
        "T. Hastie and R. Tibshirani, <i>Generalized Additive Models</i>. London: Chapman &amp; "
        "Hall, 1990.",
        "C. Molnar, <i>Interpretable Machine Learning</i>, 2nd ed. Munich, Germany: "
        "Independently Published, 2022.",
        "T. Chen and C. Guestrin, &#8220;XGBoost: A scalable tree boosting system,&#8221; in "
        "<i>Proc. ACM SIGKDD</i>, 2016, pp. 785&#8211;794.",
        "G. Ke, Q. Meng, T. Finley, T. Wang, W. Chen, W. Ma, Q. Ye, and T.-Y. Liu, "
        "&#8220;LightGBM: A highly efficient gradient boosting decision tree,&#8221; in <i>Proc. "
        "NeurIPS</i>, 2017, pp. 3149&#8211;3157.",
        "L. Breiman, &#8220;Random forests,&#8221; <i>Machine Learning</i>, vol. 45, no. 1, "
        "pp. 5&#8211;32, 2001.",
        "F. Pedregosa et al., &#8220;Scikit-learn: Machine learning in Python,&#8221; <i>Journal "
        "of Machine Learning Research</i>, vol. 12, pp. 2825&#8211;2830, 2011.",
        "N. V. Chawla, K. W. Bowyer, L. O. Hall, and W. P. Kegelmeyer, &#8220;SMOTE: Synthetic "
        "minority over-sampling technique,&#8221; <i>Journal of Artificial Intelligence "
        "Research</i>, vol. 16, pp. 321&#8211;357, 2002.",
        "T. Saito and M. Rehmsmeier, &#8220;The precision-recall plot is more informative than "
        "the ROC plot when evaluating binary classifiers on imbalanced datasets,&#8221; "
        "<i>PLoS ONE</i>, vol. 10, no. 3, Art. no. e0118432, 2015.",
        "B. Van Calster, D. J. McLernon, M. van Smeden, L. Wynants, and E. W. Steyerberg, "
        "&#8220;Calibration: The Achilles heel of predictive analytics,&#8221; <i>BMC Medicine</i>, "
        "vol. 17, Art. no. 230, 2019.",
        "C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, &#8220;On calibration of modern neural "
        "networks,&#8221; in <i>Proc. ICML</i>, 2017, pp. 1321&#8211;1330.",
        "S. Barocas, M. Hardt, and A. Narayanan, <i>Fairness and Machine Learning</i>. "
        "Cambridge, MA: MIT Press, 2023.",
        "R. S. Baker and A. Hawn, &#8220;Algorithmic bias in education,&#8221; <i>International "
        "Journal of Artificial Intelligence in Education</i>, vol. 32, pp. 1052&#8211;1092, 2022.",
        "K. Holstein and V. Aleven, &#8220;Human&#8211;AI complementarity in K-12 education,&#8221; "
        "<i>AI Magazine</i>, vol. 43, no. 1, pp. 87&#8211;96, 2022.",
        "B. Prenkaj, P. Vullo, G. Stilo, and L. Faralli, &#8220;A survey of machine learning "
        "approaches for student dropout prediction in online courses,&#8221; <i>ACM Computing "
        "Surveys</i>, vol. 54, no. 4, Art. no. 80, 2022.",
        "J. Kuzilek, V. Hlavac, and Z. Zdrahal, &#8220;OU Analyse: Analysing at-risk students "
        "at the Open University,&#8221; <i>Learning Analytics Review</i>, vol. 1, pp. 1&#8211;6, 2015.",
    ]
    for i, r in enumerate(refs, 1):
        add(("ref", f"[{i}] {r}"))

    return B
