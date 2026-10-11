import pytest
from fastapi.testclient import TestClient
from api.main import app, load_artifacts, ml_state

load_artifacts()
client = TestClient(app)


def test_patient_b_coherence_and_vessel_flagging():
    """Verify that Patient B (where LAD is flagged) flags overall CAD without contradiction."""
    samples = client.get("/api/samples").json()
    patient_b = next(s for s in samples if "sample-mid-risk" in s["id"] or "Patient B" in s["name"])
    
    resp = client.post("/api/predict", json=patient_b["data"])
    assert resp.status_code == 200
    data = resp.json()

    cad = data["cad"]
    vessels = data["vessels"]

    # In screening mode, if any vessel is flagged stenotic, CAD high_sens_label must be High Risk
    any_vessel_flagged_screening = any(v["high_sens_label"] == "Stenotic" for v in vessels.values())
    if any_vessel_flagged_screening:
        assert cad["high_sens_label"] == "High Risk", (
            f"Contradiction: vessel flagged but CAD high_sens_label is {cad['high_sens_label']}"
        )

    # In balanced mode, if any vessel is flagged stenotic, CAD label must be High Risk
    any_vessel_flagged_balanced = any(v["label"] == "Stenotic" for v in vessels.values())
    if any_vessel_flagged_balanced:
        assert cad["label"] == "High Risk", (
            f"Contradiction: vessel flagged but CAD label is {cad['label']}"
        )

    # Coherence rule ensures P(CAD) >= max(P_vessel)
    max_v_prob = max(v["prob"] for v in vessels.values())
    assert cad["coherent_prob"] >= max_v_prob - 1e-4

    # Verify plain language explanation summary is present
    assert "plain_language_summary" in cad
    assert len(cad["plain_language_summary"]) > 0


def test_direction_probes_monotonicity():
    """Verify that clinical risk drivers (LDL, BP, FBS, smoking) monotonically increase or preserve CAD risk."""
    samples = client.get("/api/samples").json()
    for s in samples:
        base_data = s["data"]
        base_resp = client.post("/api/predict", json=base_data).json()
        base_prob = base_resp["cad"]["raw_prob"]

        # Increase BP by 30 mmHg
        d_bp = dict(base_data, BP=base_data["BP"] + 30)
        p_bp = client.post("/api/predict", json=d_bp).json()["cad"]["raw_prob"]
        assert p_bp >= base_prob - 1e-5, f"BP increase lowered risk on {s['name']}: {p_bp} < {base_prob}"

        # Increase FBS by 50 mg/dL
        d_fbs = dict(base_data, FBS=base_data["FBS"] + 50)
        p_fbs = client.post("/api/predict", json=d_fbs).json()["cad"]["raw_prob"]
        assert p_fbs >= base_prob - 1e-5, f"FBS increase lowered risk on {s['name']}: {p_fbs} < {base_prob}"

        # Increase LDL by 40 mg/dL
        d_ldl = dict(base_data, LDL=base_data["LDL"] + 40)
        p_ldl = client.post("/api/predict", json=d_ldl).json()["cad"]["raw_prob"]
        assert p_ldl >= base_prob - 1e-5, f"LDL increase lowered risk on {s['name']}: {p_ldl} < {base_prob}"

        # Turn smoking on
        d_smk = dict(base_data, **{"Current Smoker": 1, "EX-Smoker": 0})
        p_smk = client.post("/api/predict", json=d_smk).json()["cad"]["raw_prob"]
        assert p_smk >= base_prob - 1e-5, f"Smoking turned on lowered risk on {s['name']}: {p_smk} < {base_prob}"

        # Probe physiological bounds on BMI coupling (verify prediction remains bounded in [0, 1])
        d_bmi = dict(base_data, BMI=float(base_data["BMI"]) + 5.0)
        if d_bmi["BMI"] > 25.0:
            d_bmi["Obesity"] = "Y"
        p_bmi = client.post("/api/predict", json=d_bmi).json()["cad"]["raw_prob"]
        assert 0.0 <= p_bmi <= 1.0, f"BMI coupled probability out of bounds: {p_bmi}"

