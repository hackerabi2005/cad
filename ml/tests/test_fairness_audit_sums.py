"""
Tests for Demographic Fairness Audit subgroup sum conservation.
Verifies that subgroup confusion matrix counts (Male/Female, Age <=65 / Age >65)
strictly sum to the pooled confusion matrix in both Screening (0.400) and Balanced (0.500) modes.
"""

import json
import numpy as np
import pytest
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import RepeatedStratifiedKFold

from ml.pipeline import DEFAULT_DATA_PATH, TARGETS, build_pipeline, load_raw_dataset, load_schema
from ml.train import get_candidate_models


@pytest.fixture(scope="module")
def oof_predictions_and_dataset():
    schema = load_schema()
    X, y_dict = load_raw_dataset(DEFAULT_DATA_PATH, align_row_93=True)
    with open("ml/artifacts/model_card.json", "r", encoding="utf-8") as f:
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

    oof_cad = oof_predictions["Cath"]
    max_vessels = np.maximum(oof_predictions["LAD"], np.maximum(oof_predictions["LCX"], oof_predictions["RCA"]))
    p_cad_coherent = np.maximum(oof_cad, max_vessels)

    return X, y_dict["Cath"].values, p_cad_coherent


def test_fairness_subgroups_sum_to_pooled_screening(oof_predictions_and_dataset):
    X, y_true, p_cad_coherent = oof_predictions_and_dataset

    sex_col = X["Sex"].astype(str).str.strip().str.lower()
    is_male = (sex_col == "male").values
    is_female = ~is_male

    age_col = X["Age"].values
    is_senior = (age_col > 65)
    is_younger = ~is_senior

    def get_cm(mask, thresh=0.400):
        pred = (p_cad_coherent[mask] >= thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true[mask], pred, labels=[0, 1]).ravel()
        return int(tp), int(fn), int(tn), int(fp)

    m_tp, m_fn, m_tn, m_fp = get_cm(is_male)
    f_tp, f_fn, f_tn, f_fp = get_cm(is_female)
    y_tp, y_fn, y_tn, y_fp = get_cm(is_younger)
    s_tp, s_fn, s_tn, s_fp = get_cm(is_senior)
    p_tp, p_fn, p_tn, p_fp = get_cm(np.ones(len(y_true), dtype=bool))

    # Pooled must equal 208 TP, 9 FN, 54 TN, 32 FP (Screening 0.400)
    assert (p_tp, p_fn, p_tn, p_fp) == (208, 9, 54, 32)

    # Sex strata sum exactly to pooled
    assert m_tp + f_tp == p_tp == 208
    assert m_fn + f_fn == p_fn == 9
    assert m_tn + f_tn == p_tn == 54
    assert m_fp + f_fp == p_fp == 32

    # Age strata sum exactly to pooled
    assert y_tp + s_tp == p_tp == 208
    assert y_fn + s_fn == p_fn == 9
    assert y_tn + s_tn == p_tn == 54
    assert y_fp + s_fp == p_fp == 32


def test_fairness_subgroups_sum_to_pooled_balanced(oof_predictions_and_dataset):
    X, y_true, p_cad_coherent = oof_predictions_and_dataset

    sex_col = X["Sex"].astype(str).str.strip().str.lower()
    is_male = (sex_col == "male").values
    is_female = ~is_male

    age_col = X["Age"].values
    is_senior = (age_col > 65)
    is_younger = ~is_senior

    def get_cm(mask, thresh=0.500):
        pred = (p_cad_coherent[mask] >= thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true[mask], pred, labels=[0, 1]).ravel()
        return int(tp), int(fn), int(tn), int(fp)

    m_tp, m_fn, m_tn, m_fp = get_cm(is_male)
    f_tp, f_fn, f_tn, f_fp = get_cm(is_female)
    y_tp, y_fn, y_tn, y_fp = get_cm(is_younger)
    s_tp, s_fn, s_tn, s_fp = get_cm(is_senior)
    p_tp, p_fn, p_tn, p_fp = get_cm(np.ones(len(y_true), dtype=bool))

    # Pooled must equal 205 TP, 12 FN, 61 TN, 25 FP (Balanced 0.500)
    assert (p_tp, p_fn, p_tn, p_fp) == (205, 12, 61, 25)

    # Sex strata sum exactly to pooled
    assert m_tp + f_tp == p_tp == 205
    assert m_fn + f_fn == p_fn == 12
    assert m_tn + f_tn == p_tn == 61
    assert m_fp + f_fp == p_fp == 25

    # Age strata sum exactly to pooled
    assert y_tp + s_tp == p_tp == 205
    assert y_fn + s_fn == p_fn == 12
    assert y_tn + s_tn == p_tn == 61
    assert y_fp + s_fp == p_fp == 25


def test_report_confusion_tuples_conservation():
    """
    Asserts that every confusion tuple in report documentation and generator script
    strictly conserves class balances: Positives sum to 217, Negatives sum to 86, Total 303.
    """
    import re
    from pathlib import Path

    base_dir = Path(__file__).resolve().parent.parent.parent

    # Canonical model confusion matrices
    cath_screening = (208, 9, 54, 32)  # TP, FN, TN, FP
    cath_balanced = (205, 12, 61, 25)   # TP, FN, TN, FP
    or_rule = (209, 8, 49, 37)          # TP, FN, TN, FP

    for name, (tp, fn, tn, fp) in [
        ("Cath Screening", cath_screening),
        ("Cath Balanced", cath_balanced),
        ("OR Rule", or_rule),
    ]:
        assert tp + fn == 217, f"{name}: TP ({tp}) + FN ({fn}) must equal 217 CAD positives"
        assert tn + fp == 86, f"{name}: TN ({tn}) + FP ({fp}) must equal 86 Normal negatives"
        assert tp + fn + tn + fp == 303, f"{name}: Total must equal 303 cohort size"

    # Regex search for confusion count patterns in report.md and generate_pdf_report.py
    target_files = [
        base_dir / "docs" / "report.md",
        base_dir / "scripts" / "generate_pdf_report.py",
        base_dir / "reports" / "fairness_audit.md",
    ]

    # Pattern: 209 TP / 8 FN / 49 TN / 37 FP or 208 TP / 54 TN / 9 FN / 32 FP
    p1 = re.compile(r"(\d+)\s*TP\s*/\s*(\d+)\s*FN\s*/\s*(\d+)\s*TN\s*/\s*(\d+)\s*FP")
    p2 = re.compile(r"(\d+)\s*TP\s*/\s*(\d+)\s*TN\s*/\s*(\d+)\s*FN\s*/\s*(\d+)\s*FP")

    for fpath in target_files:
        if not fpath.exists():
            continue
        text = fpath.read_text(encoding="utf-8")
        
        for m in p1.finditer(text):
            tp, fn, tn, fp = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
            assert tp + fn == 217, f"In {fpath.name}: {m.group(0)} positives {tp}+{fn} != 217"
            assert tn + fp == 86, f"In {fpath.name}: {m.group(0)} negatives {tn}+{fp} != 86"

        for m in p2.finditer(text):
            tp, tn, fn, fp = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
            assert tp + fn == 217, f"In {fpath.name}: {m.group(0)} positives {tp}+{fn} != 217"
            assert tn + fp == 86, f"In {fpath.name}: {m.group(0)} negatives {tn}+{fp} != 86"

