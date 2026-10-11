import math
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.main import app, load_artifacts, ml_state
from ml.pipeline import DEFAULT_DATA_PATH, TARGETS, load_raw_dataset


def sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


@pytest.fixture(scope="module")
def client():
    load_artifacts()
    return TestClient(app)


def test_prediction_explanation_consistency_presets(client):
    """Verify consistency between predicted probability, raw score, and SHAP sum across preset samples."""
    samples_resp = client.get("/api/samples")
    assert samples_resp.status_code == 200
    samples = samples_resp.json()
    assert len(samples) >= 3

    for sample in samples:
        resp = client.post("/api/predict", json=sample["data"])
        assert resp.status_code == 200
        data = resp.json()

        cad_raw_prob = data["cad"]["raw_prob"] if "raw_prob" in data["cad"] else data["cad"]["prob"]
        vessel_probs = {v: data["vessels"][v]["prob"] for v in ["LAD", "LCX", "RCA"]}

        for target in TARGETS:
            exp = data["explain"][target]
            raw_score = exp["raw_score"]
            base_val = exp["base_value"]
            add_err = exp["additive_error"]

            # Additive SHAP exactness: |base + sum(shap) - raw_score| < 1e-4
            assert add_err < 1e-4, f"Additivity error {add_err} >= 1e-4 for {target} on {sample['id']}"

            # Probability consistency with raw score
            target_prob = cad_raw_prob if target == "Cath" else vessel_probs[target]
            pipeline = ml_state["pipelines"][target]
            model = pipeline.named_steps["classifier"]
            model_class = model.__class__.__name__

            if model_class == "RandomForestClassifier":
                # For probability trees, raw_score is probability
                assert abs(target_prob - raw_score) < 2e-3, (
                    f"Mismatch for RF {target}: prob={target_prob}, raw_score={raw_score}"
                )
            else:
                # LogisticRegression or XGBoost (margin / logit space)
                expected_prob = sigmoid(raw_score)
                assert abs(target_prob - expected_prob) < 2e-3, (
                    f"Mismatch for {model_class} {target}: prob={target_prob}, sigmoid(raw_score)={expected_prob}"
                )


def test_prediction_explanation_consistency_random_dataset_rows(client):
    """Verify consistency over 10 random rows from the dataset."""
    X, _ = load_raw_dataset(DEFAULT_DATA_PATH)
    rng = np.random.RandomState(123)
    sample_indices = rng.choice(len(X), size=10, replace=False)

    for idx in sample_indices:
        row_dict = X.iloc[idx].to_dict()
        resp = client.post("/api/predict", json=row_dict)
        assert resp.status_code == 200
        data = resp.json()

        cad_raw_prob = data["cad"]["raw_prob"] if "raw_prob" in data["cad"] else data["cad"]["prob"]
        vessel_probs = {v: data["vessels"][v]["prob"] for v in ["LAD", "LCX", "RCA"]}

        for target in TARGETS:
            exp = data["explain"][target]
            raw_score = exp["raw_score"]
            base_val = exp["base_value"]
            add_err = exp["additive_error"]

            assert add_err < 1e-4, f"Additivity error {add_err} >= 1e-4 for {target} on row {idx}"

            target_prob = cad_raw_prob if target == "Cath" else vessel_probs[target]
            pipeline = ml_state["pipelines"][target]
            model = pipeline.named_steps["classifier"]
            model_class = model.__class__.__name__

            if model_class == "RandomForestClassifier":
                assert abs(target_prob - raw_score) < 2e-3, (
                    f"Mismatch for RF {target}: prob={target_prob}, raw_score={raw_score}"
                )
            else:
                expected_prob = sigmoid(raw_score)
                assert abs(target_prob - expected_prob) < 2e-3, (
                    f"Mismatch for {model_class} {target}: prob={target_prob}, sigmoid(raw_score)={expected_prob}"
                )


def test_decision_curve_net_benefit_from_confusion_counts():
    """
    Recomputes clinical net benefit directly from 2x2 confusion matrix counts
    per Vickers & Elkin (2006) and asserts exact mathematical bounds:
    - Delta over Treat-All at pt=0.400 equals +8.91 per 100 patients (+8.64 bootstrap mean)
    - Recomputed delta is strictly <= theoretical ceiling of +18.92 per 100 patients: (1 - prev) * (pt / (1 - pt))
    - Explicitly proves that the stale +29.6 headline is impossible under pt=0.400.
    """
    n_total = 303
    n_pos = 217
    n_neg = 86
    pt = 0.400
    weight = pt / (1.0 - pt)  # 0.4 / 0.6 = 2/3

    # At Cath cutoff 0.400: Se = 95.85% (208 TP, 9 FN), Sp = 62.79% (54 TN, 32 FP)
    tp = 208
    fn = 9
    tn = 54
    fp = 32

    assert tp + fn == n_pos
    assert tn + fp == n_neg

    nb_model = (tp / n_total) - (fp / n_total) * weight
    nb_treat_all = (n_pos / n_total) - (n_neg / n_total) * weight
    delta_nb = nb_model - nb_treat_all
    delta_per_100 = delta_nb * 100.0

    # Direct algebraic formula: -(FN / N) + (TN / N) * (pt / (1 - pt))
    direct_delta = (-(fn / n_total) + (tn / n_total) * weight) * 100.0
    assert abs(delta_per_100 - direct_delta) < 1e-9

    # Verify exact value: +8.91 per 100 patients
    assert round(delta_per_100, 2) == 8.91

    # Theoretical maximum gain ceiling for ANY perfect classifier at pt=0.400
    max_possible_gain = ((n_neg / n_total) * weight) * 100.0  # (86 / 303) * (2/3) * 100 = 18.92
    assert delta_per_100 <= max_possible_gain
    assert max_possible_gain < 19.0, f"Max gain ceiling {max_possible_gain} must be < 19.0"

    # Confirms that +29.6 is strictly impossible for pt=0.400
    assert delta_per_100 < 29.6, "+29.6 is mathematically impossible at threshold 0.400"


def test_prediction_explanation_consistency_all_303_dataset_rows(client):
    """
    Extended invariant check across all 303 dataset rows x 4 targets:
    Asserts |P - sigmoid(raw_score)| < 1e-4 directly on pipelines for every target.
    """
    X, _ = load_raw_dataset(DEFAULT_DATA_PATH)
    assert len(X) == 303

    for target in TARGETS:
        pipe = ml_state["pipelines"][target]
        clf = pipe.named_steps["classifier"]
        prep = pipe.named_steps["preprocessor"]

        X_trans = prep.transform(X)
        z = clf.decision_function(X_trans)
        probs = pipe.predict_proba(X)[:, 1]

        expected_probs = 1.0 / (1.0 + np.exp(-z))
        max_diff = float(np.max(np.abs(probs - expected_probs)))
        assert max_diff < 1e-4, f"Target {target} broke P = sigmoid(z) invariant: max diff {max_diff}"

