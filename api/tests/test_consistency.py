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
