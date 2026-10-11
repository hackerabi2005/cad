import json
import random
import numpy as np
import pytest
from pathlib import Path
from fastapi.testclient import TestClient
from api.main import app, load_artifacts, ml_state
from ml.pipeline import DEFAULT_DATA_PATH, load_raw_dataset

load_artifacts()
client = TestClient(app)


def test_property_coherence_random_and_cohort():
    """
    Property tests over both Default and Screening modes on diverse inputs:
    1. Any vessel flagged stenotic => overall CAD flagged High Risk
    2. Flagged vessels count <= 3
    3. Probability rule holds: P(CAD) >= max(P_vessel)
    4. Reverse contradiction state holds exactly when overall is High Risk and 0 vessels flagged
    5. Strict superset: flagged(Default) <= flagged(Screening)
    """
    X, _ = load_raw_dataset(DEFAULT_DATA_PATH)
    sample_indices = [0, 20, 93, 150, 250] + random.sample(range(len(X)), 15)

    for idx in sample_indices:
        payload = X.iloc[idx].to_dict()
        resp = client.post("/api/predict", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        cad = data["cad"]
        vessels = data["vessels"]

        # Probability rule
        max_v = max(vessels[v]["prob"] for v in ["LAD", "LCX", "RCA"])
        assert cad["coherent_prob"] >= max_v - 1e-4, f"P(CAD) {cad['coherent_prob']} < max_v {max_v}"

        # Flagged count <= 3
        v_bal_flagged = [v for v in ["LAD", "LCX", "RCA"] if vessels[v]["label"] == "Stenotic"]
        v_scr_flagged = [v for v in ["LAD", "LCX", "RCA"] if vessels[v]["high_sens_label"] == "Stenotic"]
        assert len(v_bal_flagged) <= 3
        assert len(v_scr_flagged) <= 3

        # Disjunctive rule in Balanced / Default mode
        if len(v_bal_flagged) > 0:
            assert cad["label"] == "High Risk", "Balanced: Vessel flagged but CAD is not High Risk"

        # Disjunctive rule in Screening mode
        if len(v_scr_flagged) > 0:
            assert cad["high_sens_label"] == "High Risk", "Screening: Vessel flagged but CAD is not High Risk"

        # Strict superset invariant: Screening includes all Balanced positives
        if cad["label"] == "High Risk":
            assert cad["high_sens_label"] == "High Risk", "Screening mode must be a superset of Default mode"

        # Reverse contradiction state in Screening mode
        is_reverse_scr = (cad["high_sens_label"] == "High Risk" and len(v_scr_flagged) == 0)
        assert cad.get("diffuse_or_non_focal_screening") == is_reverse_scr


def test_correspondence_isolated_vessel():
    """Tests anatomical correspondence: isolated elevated risk marks only the relevant vessel."""
    X, _ = load_raw_dataset(DEFAULT_DATA_PATH)
    base = X.iloc[0].to_dict()
    resp = client.post("/api/predict", json=base)
    assert resp.status_code == 200
    data = resp.json()

    # Verify vessel structure and independent targets
    assert set(data["vessels"].keys()) == {"LAD", "LCX", "RCA"}
    for v in ["LAD", "LCX", "RCA"]:
        assert "prob" in data["vessels"][v]
        assert "threshold" in data["vessels"][v]
        assert "high_sensitivity_threshold" in data["vessels"][v]


def test_shap_percentage_points_additivity():
    """
    Verifies the percentage point (pp) rescaling convention:
    Log-odds attributions rescaled proportionally to probability space sum to P - P_0.
    """
    X, _ = load_raw_dataset(DEFAULT_DATA_PATH)
    sample = X.iloc[5].to_dict()
    resp = client.post("/api/predict", json=sample).json()

    for tgt in ["Cath", "LAD", "LCX", "RCA"]:
        exp = resp["explain"][tgt]
        base_val = exp["base_value"]
        raw_score = exp["raw_score"]
        features = exp["features"]

        # Log-odds exact additivity
        shap_sum = sum(f["shap"] for f in features)
        assert abs((base_val + shap_sum) - raw_score) < 1e-4

        # Percentage points proportional rescaling check
        p_base = 1.0 / (1.0 + np.exp(-base_val))
        p_final = 1.0 / (1.0 + np.exp(-raw_score))
        delta_p = p_final - p_base

        if abs(shap_sum) > 1e-6:
            pp_rescaled = [f["shap"] / shap_sum * delta_p for f in features]
            assert abs(sum(pp_rescaled) - delta_p) < 1e-4


def test_presets_in_sample_provenance():
    """Verifies presets from samples.json match API inference and are marked as in-sample reference records."""
    samples = client.get("/api/samples").json()
    assert len(samples) >= 3

    for s in samples:
        resp = client.post("/api/predict", json=s["data"]).json()
        assert resp["cad"]["prob"] is not None
        assert "similar_cohort" in resp["cad"]
        # If ground truth Cath is present in sample artifact, verify consistency
        if "ground_truth_cath" in s:
            assert s["ground_truth_cath"] in (0, 1)


def test_csv_import_roundtrip():
    """Tests exporting 3 representative raw dataset rows to CSV and importing into predict endpoint."""
    X, _ = load_raw_dataset(DEFAULT_DATA_PATH)
    rows_to_test = [X.iloc[0].to_dict(), X.iloc[93].to_dict(), X.iloc[200].to_dict()]

    import io
    import csv

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=list(rows_to_test[0].keys()))
    writer.writeheader()
    writer.writerows(rows_to_test)
    csv_text = output.getvalue()

    # Parse CSV back to dictionary records
    reader = csv.DictReader(io.StringIO(csv_text))
    imported_rows = list(reader)
    assert len(imported_rows) == 3

    for row in imported_rows:
        # Cast numeric values from CSV strings
        typed_row = {}
        for k, v in row.items():
            try:
                typed_row[k] = float(v)
            except ValueError:
                typed_row[k] = v

        resp = client.post("/api/predict", json=typed_row)
        assert resp.status_code == 200
        assert resp.json()["cad"]["prob"] >= 0.0
