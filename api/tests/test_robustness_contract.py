"""Additive contract plugs — NEW file, does not edit api/tests/test_api.py.

Covers DoD 1-6 in a conflict-free way (other AI owns test_api.py).
Run: pytest api/tests/test_plugs_contract.py -q
"""
from fastapi.testclient import TestClient

from api.main import app, load_artifacts

load_artifacts()
client = TestClient(app, raise_server_exceptions=False)


def test_fuzz_zero_500s():
    cases = [{}, {"Age": 200}, {"Age": "abc"}, {"Sex": "Banana"},
             {"Age": 55, "BP": 400}, {"Age": 55, "Foo": 1}]
    for body in cases:
        r = client.post("/api/predict", json=body)
        assert r.status_code != 500, body
        assert r.status_code == 422, (body, r.status_code)


def test_openapi_54_typed_fields():
    spec = app.openapi()
    props = spec["components"]["schemas"]["PatientInput"]["properties"]
    assert len(props) == 54


def test_raw_displayed_driving_present():
    from ml.pipeline import DEFAULT_DATA_PATH, load_raw_dataset
    X, _ = load_raw_dataset(DEFAULT_DATA_PATH)
    r = client.post("/api/predict", json=X.iloc[0].to_dict()).json()
    assert "raw_prob" in r["cad"]
    assert "displayed_prob" in r["cad"]
    assert "driving_vessel" in r["cad"]
    assert "missing_features" in r
    assert "reliability" in r
    assert r["reliability"] in ("high", "moderate", "low")


def test_imputed_flag_consistency():
    """Minimal payload must list imputed features; provided count drives reliability."""
    from ml.pipeline import load_schema
    schema = load_schema()
    # 10+ valid fields to pass completeness gate, rest missing -> imputed
    payload = {"Age": 55, "Sex": "Male", "BP": 120, "Weight": 70, "Length": 170,
               "BMI": 24.0, "DM": 0, "HTN": 1, "FH": 0, "Obesity": "N",
               "BP": 120, "PR": 70}
    r = client.post("/api/predict", json=payload)
    assert r.status_code == 200, r.text
    j = r.json()
    assert len(j["missing_features"]) == len(schema) - len(payload)
    assert j["completeness"]["provided"] == len(payload)
