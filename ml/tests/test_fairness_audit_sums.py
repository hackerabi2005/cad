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
