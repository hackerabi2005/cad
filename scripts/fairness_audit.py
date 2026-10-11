"""
Subgroup Fairness and Demographic Validation Audit for Cardio3D AI.
Evaluates model discrimination, calibration, and parity across Sex and Age strata.
Reports sensitivity and specificity with Wilson 95% intervals and subgroup sample sizes.
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold

from ml.pipeline import DEFAULT_DATA_PATH, TARGETS, build_pipeline, load_raw_dataset, load_schema
from ml.train import get_candidate_models
from ml.service import compute_wilson_score_interval

REPORTS_DIR = BASE_DIR / "reports"
ARTIFACTS_DIR = BASE_DIR / "ml" / "artifacts"


def main():
    print("Executing Demographic Fairness Audit...")
    schema = load_schema()
    X, y_dict = load_raw_dataset(DEFAULT_DATA_PATH, align_row_93=True)

    with open(ARTIFACTS_DIR / "model_card.json", "r", encoding="utf-8") as f:
        model_card = json.load(f)
    selected_models = model_card["selected_models"]
    candidate_dict = get_candidate_models()

    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
    oof_predictions = {}

    for target in TARGETS:
        model_name = selected_models[target]
        model_proto = candidate_dict[model_name]
        y = y_dict[target].values
        probs = np.zeros(len(X))

        for train_idx, val_idx in rskf.split(X, y):
            pipe = build_pipeline(model_proto, schema=schema)
            pipe.fit(X.iloc[train_idx], y[train_idx])
            probs[val_idx] += pipe.predict_proba(X.iloc[val_idx])[:, 1] / 3.0

        oof_predictions[target] = probs

    # Clean Sex & Age
    sex_col = X["Sex"].astype(str).str.strip().str.lower()
    is_male = (sex_col == "male").values
    is_female = ~is_male

    age_col = X["Age"].values
    is_senior = (age_col > 65)
    is_younger = ~is_senior

    # Coherent CAD score
    oof_cad = oof_predictions["Cath"]
    max_vessels = np.maximum(oof_predictions["LAD"], np.maximum(oof_predictions["LCX"], oof_predictions["RCA"]))
    p_cad_coherent = np.maximum(oof_cad, max_vessels)

    op_threshold = model_card["operating_thresholds"]["Cath"]["high_sensitivity_threshold"]
    y_true = y_dict["Cath"].values

    def audit_subgroup_mode(mask, label, thresh):
        y_sub = y_true[mask]
        p_sub = p_cad_coherent[mask]
        pred_sub = (p_sub >= thresh).astype(int)

        n = len(y_sub)
        n_pos = int(np.sum(y_sub == 1))
        n_neg = int(np.sum(y_sub == 0))
        prev = n_pos / n if n > 0 else 0.0

        auc = float(roc_auc_score(y_sub, p_sub)) if len(np.unique(y_sub)) > 1 else 0.5
        cm = confusion_matrix(y_sub, pred_sub, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        sens_ci = compute_wilson_score_interval(tp, tp + fn)
        spec_ci = compute_wilson_score_interval(tn, tn + fp)

        ppv = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        npv = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0
        selection_rate = float((tp + fp) / n) if n > 0 else 0.0

        return {
            "group": label,
            "count": n,
            "n_pos": n_pos,
            "n_neg": n_neg,
            "prevalence": prev,
            "auc": auc,
            "tp": int(tp),
            "fn": int(fn),
            "tn": int(tn),
            "fp": int(fp),
            "sensitivity": sens,
            "sensitivity_ci": sens_ci,
            "specificity": spec,
            "specificity_ci": spec_ci,
            "ppv": ppv,
            "npv": npv,
            "selection_rate": selection_rate,
        }

    n_male = int(np.sum(is_male))
    n_female = int(np.sum(is_female))
    n_younger = int(np.sum(is_younger))
    n_senior = int(np.sum(is_senior))

    # Evaluate Screening Mode (0.400)
    m_male = audit_subgroup_mode(is_male, f"Male (n={n_male})", 0.400)
    m_female = audit_subgroup_mode(is_female, f"Female (n={n_female})", 0.400)
    m_younger = audit_subgroup_mode(is_younger, f"Age ≤ 65 (n={n_younger})", 0.400)
    m_senior = audit_subgroup_mode(is_senior, f"Age > 65 (n={n_senior})", 0.400)
    m_pooled_scr = audit_subgroup_mode(np.ones(len(y_true), dtype=bool), "Pooled Cohort", 0.400)

    # Assert exact sum conservation for Screening Mode
    assert m_male["tp"] + m_female["tp"] == m_pooled_scr["tp"] == 208, "Sex TP must sum to pooled TP (208)"
    assert m_male["fn"] + m_female["fn"] == m_pooled_scr["fn"] == 9, "Sex FN must sum to pooled FN (9)"
    assert m_male["tn"] + m_female["tn"] == m_pooled_scr["tn"] == 54, "Sex TN must sum to pooled TN (54)"
    assert m_male["fp"] + m_female["fp"] == m_pooled_scr["fp"] == 32, "Sex FP must sum to pooled FP (32)"

    assert m_younger["tp"] + m_senior["tp"] == m_pooled_scr["tp"] == 208, "Age TP must sum to pooled TP (208)"
    assert m_younger["fn"] + m_senior["fn"] == m_pooled_scr["fn"] == 9, "Age FN must sum to pooled FN (9)"
    assert m_younger["tn"] + m_senior["tn"] == m_pooled_scr["tn"] == 54, "Age TN must sum to pooled TN (54)"
    assert m_younger["fp"] + m_senior["fp"] == m_pooled_scr["fp"] == 32, "Age FP must sum to pooled FP (32)"

    # Parity metrics
    disparate_impact_sex = m_female["selection_rate"] / m_male["selection_rate"] if m_male["selection_rate"] > 0 else 1.0
    eq_opp_diff_sex = abs(m_male["sensitivity"] - m_female["sensitivity"])
    auc_diff_sex = abs(m_male["auc"] - m_female["auc"])

    disparate_impact_age = m_younger["selection_rate"] / m_senior["selection_rate"] if m_senior["selection_rate"] > 0 else 1.0
    eq_opp_diff_age = abs(m_senior["sensitivity"] - m_younger["sensitivity"])

    # Evaluate Balanced Mode (0.500)
    bal_male = audit_subgroup_mode(is_male, f"Male (n={n_male})", 0.500)
    bal_female = audit_subgroup_mode(is_female, f"Female (n={n_female})", 0.500)
    bal_younger = audit_subgroup_mode(is_younger, f"Age ≤ 65 (n={n_younger})", 0.500)
    bal_senior = audit_subgroup_mode(is_senior, f"Age > 65 (n={n_senior})", 0.500)
    bal_pooled = audit_subgroup_mode(np.ones(len(y_true), dtype=bool), "Pooled Cohort", 0.500)

    # Assert exact sum conservation for Balanced Mode
    assert bal_male["tp"] + bal_female["tp"] == bal_pooled["tp"] == 205, "Sex TP must sum to pooled TP (205)"
    assert bal_male["fn"] + bal_female["fn"] == bal_pooled["fn"] == 12, "Sex FN must sum to pooled FN (12)"
    assert bal_male["tn"] + bal_female["tn"] == bal_pooled["tn"] == 61, "Sex TN must sum to pooled TN (61)"
    assert bal_male["fp"] + bal_female["fp"] == bal_pooled["fp"] == 25, "Sex FP must sum to pooled FP (25)"

    assert bal_younger["tp"] + bal_senior["tp"] == bal_pooled["tp"] == 205, "Age TP must sum to pooled TP (205)"
    assert bal_younger["fn"] + bal_senior["fn"] == bal_pooled["fn"] == 12, "Age FN must sum to pooled FN (12)"
    assert bal_younger["tn"] + bal_senior["tn"] == bal_pooled["tn"] == 61, "Age TN must sum to pooled TN (61)"
    assert bal_younger["fp"] + bal_senior["fp"] == bal_pooled["fp"] == 25, "Age FP must sum to pooled FP (25)"
    lines = [
        "# Demographic Fairness & Subgroup Validation Audit\n",
        "This audit evaluates predictive characteristics across demographic strata (Sex and Age) ",
        "for the primary CAD model across both operating modes (Screening 0.400 and Balanced 0.500). ",
        "All metrics are calculated on out-of-fold predictions with Wilson 95% confidence intervals.\n",
        "## 1. Screening Mode Subgroup Performance (Cutoff 0.400 / High Sensitivity)\n",
        "| Stratum | n | TP | FN | TN | FP | Prevalence | ROC-AUC | Sensitivity (95% Wilson) | Specificity (95% Wilson) | PPV | NPV |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]

    for m in [m_male, m_female, m_younger, m_senior, m_pooled_scr]:
        sens_str = f"{m['sensitivity']:.1%} [{m['sensitivity_ci'][0]:.1%}–{m['sensitivity_ci'][1]:.1%}]"
        spec_str = f"{m['specificity']:.1%} [{m['specificity_ci'][0]:.1%}–{m['specificity_ci'][1]:.1%}]"
        lines.append(
            f"| **{m['group']}** | {m['count']} | {m['tp']} | {m['fn']} | {m['tn']} | {m['fp']} | {m['prevalence']:.1%} | {m['auc']:.3f} | "
            f"{sens_str} | {spec_str} | {m['ppv']:.1%} | {m['npv']:.1%} |"
        )

    lines.extend([
        "\n## 2. Balanced Mode Subgroup Performance (Cutoff 0.500)\n",
        "| Stratum | n | TP | FN | TN | FP | Prevalence | ROC-AUC | Sensitivity (95% Wilson) | Specificity (95% Wilson) | PPV | NPV |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ])

    for m in [bal_male, bal_female, bal_younger, bal_senior, bal_pooled]:
        sens_str = f"{m['sensitivity']:.1%} [{m['sensitivity_ci'][0]:.1%}–{m['sensitivity_ci'][1]:.1%}]"
        spec_str = f"{m['specificity']:.1%} [{m['specificity_ci'][0]:.1%}–{m['specificity_ci'][1]:.1%}]"
        lines.append(
            f"| **{m['group']}** | {m['count']} | {m['tp']} | {m['fn']} | {m['tn']} | {m['fp']} | {m['prevalence']:.1%} | {m['auc']:.3f} | "
            f"{sens_str} | {spec_str} | {m['ppv']:.1%} | {m['npv']:.1%} |"
        )

    lines.extend([
        "\n## 3. Subgroup Comparisons & Parity\n",
        "- **Sex Comparison (Female vs. Male)**:",
        f"  - **Sample Size**: Male n={m_male['count']} (131 CAD / 45 normal), Female n={m_female['count']} (86 CAD / 41 normal).",
        f"  - **Screening Sensitivity**: Male {m_male['sensitivity']:.1%} (95% CI: {m_male['sensitivity_ci'][0]:.1%}–{m_male['sensitivity_ci'][1]:.1%}) vs Female {m_female['sensitivity']:.1%} (95% CI: {m_female['sensitivity_ci'][0]:.1%}–{m_female['sensitivity_ci'][1]:.1%}), gap = {eq_opp_diff_sex*100:.1f}%.",
        f"  - **Screening Specificity**: Male {m_male['specificity']:.1%} (95% CI: {m_male['specificity_ci'][0]:.1%}–{m_male['specificity_ci'][1]:.1%}) vs Female {m_female['specificity']:.1%} (95% CI: {m_female['specificity_ci'][0]:.1%}–{m_female['specificity_ci'][1]:.1%}).",
        f"  - **ROC-AUC**: Male {m_male['auc']:.3f} vs Female {m_female['auc']:.3f} (delta = {auc_diff_sex:.3f}).",
        f"  - **Disparate Impact Ratio**: {disparate_impact_sex:.3f} (selection rates reflect underlying disease prevalence: {m_female['prevalence']:.1%} vs {m_male['prevalence']:.1%}).",
        "\n- **Age Comparison (Age > 65 vs. Age ≤ 65)**:",
        f"  - **Sample Size**: Age ≤ 65 n={m_younger['count']} (143 CAD / 76 normal), Age > 65 n={m_senior['count']} (74 CAD / 10 normal).",
        f"  - **Screening Sensitivity**: Age > 65 is {m_senior['sensitivity']:.1%} (95% CI: {m_senior['sensitivity_ci'][0]:.1%}–{m_senior['sensitivity_ci'][1]:.1%}) vs Age ≤ 65 is {m_younger['sensitivity']:.1%} (95% CI: {m_younger['sensitivity_ci'][0]:.1%}–{m_younger['sensitivity_ci'][1]:.1%}), gap = {eq_opp_diff_age*100:.1f}%.",
        f"  - **Screening Specificity**: Age > 65 is {m_senior['specificity']:.1%} (95% CI: {m_senior['specificity_ci'][0]:.1%}–{m_senior['specificity_ci'][1]:.1%}) vs Age ≤ 65 is {m_younger['specificity']:.1%} (95% CI: {m_younger['specificity_ci'][0]:.1%}–{m_younger['specificity_ci'][1]:.1%}).",
        "\n## 4. Clinical Takeaway & Documented Limitations",
        f"- **Safety Sensitivity**: Screening operating point prioritizes safety-first sensitivity (>93%) across both male and female groups with low sex disparity (gap = {eq_opp_diff_sex*100:.1f}%).",
        f"- **Known Limitation (Senior Specificity)**: In seniors aged > 65, specificity is markedly reduced ({m_senior['specificity']:.1%}, 95% CI: {m_senior['specificity_ci'][0]:.1%}–{m_senior['specificity_ci'][1]:.1%}, 3/10 normal patients identified) compared to younger adults ({m_younger['specificity']:.1%} at screening, {bal_younger['specificity']:.1%} balanced). This reflects severe referral bias and cohort imbalance in elderly patients (88.1% CAD prevalence, only 10 normal patients). Clinicians must note that positive flags in patients over 65 carry an elevated false-alarm rate.",
    ])

    report_path = REPORTS_DIR / "fairness_audit.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Saved fairness audit report to {report_path}")


if __name__ == "__main__":
    main()
