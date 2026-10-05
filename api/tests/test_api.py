"""
API Contract and Performance Tests.
Validates:
1. GET /api/schema returns 54 valid schema items.
2. GET /api/metrics returns CV evaluation and global SHAP.
3. GET /api/samples returns 3 preset patients.
4. POST /api/predict returns CAD status, vessels, and exact SHAP explanations.
5. Out-of-range input returns HTTP 422 naming the invalid field.
6. CPU latency benchmark: p95 /predict < 300 ms over 50 calls.
"""

import time
from pathlib import Path
import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.main import app, load_artifacts

# Ensure artifacts are loaded for test client
load_artifacts()
client = TestClient(app)


def test_get_schema():
    response = client.get("/api/schema")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 54
    assert "Age" in data
    assert "BP" in data
    assert "EF-TTE" in data


def test_get_metrics():
    response = client.get("/api/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "metrics" in data
    assert "Cath" in data["metrics"]
    assert "LAD" in data["metrics"]
    assert "shap_global" in data
    assert "model_card" in data


def test_get_samples():
    response = client.get("/api/samples")
    assert response.status_code == 200
    samples = response.json()
    assert len(samples) == 3
    assert any(s["id"] == "sample-low-risk" for s in samples)
    assert any(s["id"] == "sample-high-risk" for s in samples)


def test_predict_contract():
    # Use sample patient data
    sample_resp = client.get("/api/samples")
    sample_patient = sample_resp.json()[0]["data"]

    response = client.post("/api/predict", json=sample_patient)
    assert response.status_code == 200
    data = response.json()

    # Structure checks
    assert "cad" in data
    assert "prob" in data["cad"]
    assert "raw_prob" in data["cad"]
    assert "coherent_prob" in data["cad"]
    assert "label" in data["cad"]

    # Coherence rule assertion: displayed CAD probability >= every vessel probability
    max_v_prob = max(data["vessels"][v]["prob"] for v in ["LAD", "LCX", "RCA"])
    assert data["cad"]["prob"] >= max_v_prob - 1e-4, (
        f"Displayed CAD prob {data['cad']['prob']} is less than max vessel prob {max_v_prob}"
    )

    assert "vessels" in data
    for vessel in ["LAD", "LCX", "RCA"]:
        assert vessel in data["vessels"]
        assert "prob" in data["vessels"][vessel]
        assert "label" in data["vessels"][vessel]

    assert "explain" in data
    for target in ["Cath", "LAD", "LCX", "RCA"]:
        assert target in data["explain"]
        exp = data["explain"][target]
        assert "base_value" in exp
        assert "features" in exp
        assert len(exp["features"]) > 0

    assert "disclaimer" in data
    assert "Decision support" in data["disclaimer"]


def test_out_of_range_422():
    # Age > 200 should trigger 422
    invalid_patient = {"Age": 250.0}
    response = client.post("/api/predict", json=invalid_patient)
    assert response.status_code == 422
    assert "outside plausible physiological range" in response.json()["detail"]
    assert "Age" in response.json()["detail"]


def test_prediction_latency_p95():
    sample_resp = client.get("/api/samples")
    sample_patient = sample_resp.json()[0]["data"]

    # Warmup
    for _ in range(3):
        client.post("/api/predict", json=sample_patient)

    # Benchmark 50 calls
    latencies = []
    for _ in range(50):
        t0 = time.perf_counter()
        resp = client.post("/api/predict", json=sample_patient)
        t1 = time.perf_counter()
        assert resp.status_code == 200
        latencies.append((t1 - t0) * 1000.0)

    p95 = float(np.percentile(latencies, 95))
    mean_lat = float(np.mean(latencies))
    print(f"\nLatency Benchmark (50 calls): Mean={mean_lat:.1f}ms, p95={p95:.1f}ms (Budget < 300ms)")
    assert p95 < 300.0, f"p95 latency {p95:.1f}ms exceeds 300ms budget!"
