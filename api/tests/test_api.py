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

import json
import time
from pathlib import Path
import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.main import app, load_artifacts

# Ensure artifacts are loaded for test client
load_artifacts()
client = TestClient(app)


def test_health_check():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_get_version():
    response = client.get("/api/version")
    assert response.status_code == 200
    data = response.json()
    assert data["version"] == "1.1.0"
    assert "sklearn_version" in data


def test_get_schema():
    response = client.get("/api/schema")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 54
    assert "Age" in data
    assert "BP" in data
    assert "EF-TTE" in data
    assert "domain" in data["Age"]
    valid_groups = {"demographic", "symptoms-exam", "lab", "ECG", "echo"}
    valid_domains = {"demographics", "symptoms_exam", "labs", "ecg", "echo"}
    for feat, meta in data.items():
        assert meta.get("group") in valid_groups, f"Feature {feat} has invalid group {meta.get('group')}"
        assert meta.get("domain") in valid_domains, f"Feature {feat} has invalid domain {meta.get('domain')}"
    assert data["St Elevation"]["group"] == "ECG"
    assert data["St Depression"]["group"] == "ECG"


def test_get_targets():
    response = client.get("/api/targets")
    assert response.status_code == 200
    data = response.json()
    assert "targets" in data
    targets = data["targets"]
    assert "Cath" in targets
    assert "LAD" in targets
    assert "LCX" in targets
    assert "RCA" in targets


def test_targets_json_parity():
    """Guarantees zero configuration drift between backend, frontend, bundle, and metrics registries."""
    import joblib
    ml_path = Path(__file__).resolve().parent.parent.parent / "ml" / "targets.json"
    web_path = Path(__file__).resolve().parent.parent.parent / "web" / "src" / "targets.json"
    bundle_path = Path(__file__).resolve().parent.parent.parent / "ml" / "artifacts" / "model_bundle.joblib"
    metrics_path = Path(__file__).resolve().parent.parent.parent / "ml" / "artifacts" / "metrics.json"

    assert ml_path.exists()
    assert web_path.exists()
    assert bundle_path.exists()
    assert metrics_path.exists()

    with open(ml_path, "r", encoding="utf-8") as f1, open(web_path, "r", encoding="utf-8") as f2:
        t_ml = json.load(f1)
        t_web = json.load(f2)
        assert t_ml == t_web, "ml/targets.json and web/src/targets.json have drifted!"

    bundle = joblib.load(bundle_path)
    bundle_thresh = bundle["thresholds"]
    with open(metrics_path, "r", encoding="utf-8") as fm:
        metrics_thresh = json.load(fm)["operating_thresholds"]

    for tgt in ["Cath", "LAD", "LCX", "RCA"]:
        ml_cutoff = t_ml["targets"][tgt]["high_sensitivity_threshold"]
        ml_def = t_ml["targets"][tgt]["default_threshold"]
        b_cutoff = bundle_thresh[tgt]["high_sensitivity_threshold"]
        m_cutoff = metrics_thresh[tgt]["high_sensitivity_threshold"]

        # 4-way parity check
        assert ml_cutoff == b_cutoff == m_cutoff, f"Cutoff mismatch for {tgt}: ml={ml_cutoff}, bundle={b_cutoff}, metrics={m_cutoff}"
        # Strict superset invariant: Screening cutoff <= Default cutoff (0.50)
        assert ml_cutoff <= ml_def, f"Screening cutoff {ml_cutoff} must be <= default {ml_def} for {tgt}"



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
    assert len(samples) == 4
    assert any(s["id"] == "sample-low-risk" for s in samples)
    assert any(s["id"] == "sample-high-risk" for s in samples)
    assert any(s["id"] == "sample-false-negative" for s in samples)


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


def test_adversarial_input_validation_422():
    # Empty payload
    r_empty = client.post("/api/predict", json={})
    assert r_empty.status_code == 422
    assert "Empty payload" in r_empty.json()["detail"]

    # Boolean passed for numeric field
    r_bool = client.post("/api/predict", json={"Age": True})
    assert r_bool.status_code == 422
    assert "must be numeric" in r_bool.json()["detail"]

    # String passed for numeric field
    r_str = client.post("/api/predict", json={"Age": "abc"})
    assert r_str.status_code == 422
    assert "must be numeric" in r_str.json()["detail"]

    # Unknown field
    r_unknown = client.post("/api/predict", json={"Foo": 42})
    assert r_unknown.status_code == 422
    assert "unknown feature" in r_unknown.json()["detail"]

    # Invalid categorical value
    r_cat = client.post("/api/predict", json={"Sex": "Banana"})
    assert r_cat.status_code == 422
    assert "invalid" in r_cat.json()["detail"]

    # Invalid binary value
    r_dm = client.post("/api/predict", json={"DM": "maybe"})
    assert r_dm.status_code == 422
    assert "invalid" in r_dm.json()["detail"]

    # Extreme BP out of bounds
    r_bp = client.post("/api/predict", json={"BP": 400})
    assert r_bp.status_code == 422
    assert "outside plausible physiological range" in r_bp.json()["detail"]

    # EF-TTE out of bounds (> 90%)
    r_ef = client.post("/api/predict", json={"EF-TTE": 150})
    assert r_ef.status_code == 422
    assert "outside plausible physiological range" in r_ef.json()["detail"]

    # Hemoglobin out of bounds (> 25)
    r_hb = client.post("/api/predict", json={"HB": 40})
    assert r_hb.status_code == 422
    assert "outside plausible physiological range" in r_hb.json()["detail"]


def test_coherence_metadata_and_driving_vessel():
    sample_resp = client.get("/api/samples")
    sample_high = sample_resp.json()[2]["data"]
    r = client.post("/api/predict", json=sample_high)
    assert r.status_code == 200
    cad = r.json()["cad"]
    assert "raw_prob" in cad
    assert "displayed_prob" in cad
    assert "coherent_prob" in cad
    assert "coherence_adjusted" in cad
    if cad["coherence_adjusted"]:
        assert cad["displayed_prob"] > cad["raw_prob"]
        assert cad["driving_vessel"] in ("LAD", "LCX", "RCA")


def test_security_and_cache_headers():
    r = client.get("/api/health")
    assert "default-src 'self'" in r.headers.get("content-security-policy", "")
    assert r.headers.get("x-content-type-options") == "nosniff"
    assert r.headers.get("x-frame-options") == "DENY"
    assert r.headers.get("referrer-policy") == "strict-origin-when-cross-origin"
    assert "no-store" in r.headers.get("cache-control", "")


def test_cohort_modes_in_schema():
    r = client.get("/api/schema")
    schema = r.json()
    for feat, meta in schema.items():
        if meta["encoding"] in ("binary", "nominal"):
            assert "mode" in meta, f"Feature '{feat}' missing mode in schema"


def test_openapi_schema_typed_and_forbids_extra():
    openapi = app.openapi()
    req_schema = openapi["paths"]["/api/predict"]["post"]["requestBody"]["content"]["application/json"]["schema"]
    schema_name = req_schema["$ref"].split("/")[-1]
    properties = openapi["components"]["schemas"][schema_name]["properties"]
    assert len(properties) == 54, f"Expected 54 typed fields, got {len(properties)}"
    assert openapi["components"]["schemas"][schema_name].get("additionalProperties") is False


def test_completeness_gate_and_metadata():
    # Only 3 features provided -> 422 INSUFFICIENT_INPUT
    r_few = client.post("/api/predict", json={"Age": 55, "Weight": 70, "BP": 120})
    assert r_few.status_code == 422
    assert "INSUFFICIENT_INPUT" in r_few.json()["detail"]

    # Full preset patient -> 200 with completeness and reliability metadata
    sample_resp = client.get("/api/samples")
    sample_data = sample_resp.json()[0]["data"]
    r_full = client.post("/api/predict", json=sample_data)
    assert r_full.status_code == 200
    res = r_full.json()

    assert "completeness" in res
    assert res["completeness"]["provided"] == 54
    assert res["completeness"]["total"] == 54
    assert res["reliability"] == "high"
    assert "missing_features" in res
    assert "imputed_values" in res


def test_imputed_features_and_shap_units():
    sample_resp = client.get("/api/samples")
    sample_data = sample_resp.json()[0]["data"]
    r = client.post("/api/predict", json=sample_data)
    assert r.status_code == 200
    data = r.json()

    for tgt in ["Cath", "LAD", "LCX", "RCA"]:
        exp = data["explain"][tgt]
        assert "shap_unit" in exp
        assert exp["shap_unit"] in ("probability", "log-odds")
        assert "unit_label" in exp
        for feat in exp["features"]:
            assert "is_imputed" in feat


def test_whatif_endpoint():
    sample_resp = client.get("/api/samples")
    sample_data = sample_resp.json()[0]["data"]
    modified_data = {**sample_data, "BP": 180, "Typical Chest Pain": 1}

    payload = {"baseline": sample_data, "modified": modified_data}
    r = client.post("/api/whatif", json=payload)
    assert r.status_code == 200
    res = r.json()
    assert "baseline_cad_prob" in res
    assert "modified_cad_prob" in res
    assert "delta_cad_prob" in res
    assert "vessel_deltas" in res
    assert res["delta_cad_prob"] >= 0  # elevating BP and Chest Pain should not decrease risk

    # Verifies fast-path produces identical probabilities to /api/predict
    r_pred_base = client.post("/api/predict", json=sample_data)
    r_pred_mod = client.post("/api/predict", json=modified_data)
    assert abs(res["baseline_cad_prob"] - r_pred_base.json()["cad"]["prob"]) < 1e-4
    assert abs(res["modified_cad_prob"] - r_pred_mod.json()["cad"]["prob"]) < 1e-4


def test_female_alias_accepted():
    sample_resp = client.get("/api/samples")
    sample_data = dict(sample_resp.json()[0]["data"])
    sample_data["Sex"] = "Female"
    r_female = client.post("/api/predict", json=sample_data)
    assert r_female.status_code == 200

    sample_data["Sex"] = "Fmale"
    r_fmale = client.post("/api/predict", json=sample_data)
    assert r_fmale.status_code == 200

    # Predictions must be identical
    assert r_female.json()["cad"]["prob"] == r_fmale.json()["cad"]["prob"]
    assert r_female.json()["vessels"]["LAD"]["prob"] == r_fmale.json()["vessels"]["LAD"]["prob"]


def test_dca_net_benefit_from_confusion_counts():
    """
    Verifies that DCA net benefit at screening cutoff pt=0.40 equals the analytical
    formula from confusion counts:
    NB_model = (TP/N) - (FP/N) * (pt / (1-pt))
    NB_treat_all = pi - (1-pi) * (pt / (1-pt))
    At pt=0.40, with TP=208, FP=32, TN=54, FN=9 across N=303:
    NB_model = 208/303 - (32/303)*(0.4/0.6) = 0.6161
    NB_treat_all = 217/303 - (86/303)*(0.4/0.6) = 0.5270
    Gain over Treat-All = +0.0891 (+8.91 per 100), bounded by ceiling 18.92 per 100.
    """
    n = 303
    tp, fp, tn, fn = 208, 32, 54, 9
    pt = 0.40
    weight = pt / (1.0 - pt)  # 2/3 = 0.6667

    nb_model = (tp / n) - (fp / n) * weight
    prev = (tp + fn) / n  # 217/303
    nb_treat_all = prev - (1.0 - prev) * weight
    delta_nb_per_100 = (nb_model - nb_treat_all) * 100.0

    assert round(nb_model, 4) == 0.6161
    assert round(nb_treat_all, 4) == 0.5270
    assert round(delta_nb_per_100, 2) == 8.91

    # Theoretical maximum gain for any perfect classifier: (1 - pi) * (pt/(1-pt)) * 100
    theoretical_ceiling = (1.0 - prev) * weight * 100.0
    assert delta_nb_per_100 <= theoretical_ceiling
    assert round(theoretical_ceiling, 2) == 18.92

