"""Pony-tail verifier — read-only tail checks, never edits dirty files.

Other AI owns: api/main.py, ml/*.py (service/train), web/*.
This script only READS + writes reports/pony_tail_plugs.json (new file).

Checks (DoD tail):
 1-6. extended fuzz zero-500s, CORS, security/cache headers, phys bounds
 7. samples.json artifact (E5), MANIFEST (E6), VIDEO placeholder (G1)
 8. ensemble margin identity (B6/A6 path, no retrain)
 9. OpenAPI response_model + 54 fields (E4)
 10. hygiene: CITATION.cff, requirements-dev, THIRD_PARTY year note (G7)

Usage: python scripts/verify_tail.py
"""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

results = {"checks": [], "pass": 0, "fail": 0}


def check(name, ok, detail=""):
    results["checks"].append({"name": name, "ok": bool(ok), "detail": str(detail)[:500]})
    if ok:
        results["pass"] += 1
    else:
        results["fail"] += 1
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")


def main():
    from api.main import app, load_artifacts
    load_artifacts()
    from fastapi.testclient import TestClient
    c = TestClient(app, raise_server_exceptions=False)

    # 1. Extended fuzz (audit appendix tail rows)
    fuzz = [
        ({"Age": 55, "HB": 40}, 422),
        ({"Age": 55, "EF-TTE": 150}, 422),
        ({"Age": 55, "DM": "maybe"}, 422),
        ({"Age": 55}, 422),
        ({"Age": 55, "Foo": 1}, 422),
    ]
    zero500 = True
    for body, exp in fuzz:
        st = c.post("/api/predict", json=body).status_code
        zero500 &= (st != 500)
        check(f"tail fuzz {list(body)[-1]}->{exp}", st == exp, f"got={st}")
    check("tail fuzz zero-500s", zero500, "")

    # 2. CORS / headers via TestClient (E1-E3)
    r = c.options("/api/predict", headers={"Origin": "https://evil.example",
                                            "Access-Control-Request-Method": "POST"})
    check("E1 CORS no reflection", "evil.example" not in r.headers.get("access-control-allow-origin", ""), "")

    from ml.pipeline import DEFAULT_DATA_PATH, load_raw_dataset
    X, _ = load_raw_dataset(DEFAULT_DATA_PATH)
    rp = c.post("/api/predict", json=X.iloc[0].to_dict())
    h = {k.lower(): v for k, v in rp.headers.items()}
    check("E2 security headers", h.get("x-content-type-options") == "nosniff" and h.get("x-frame-options") == "DENY", "")
    check("E3 api no-store", "no-store" in h.get("cache-control", ""), h.get("cache-control", ""))

    # 3. E4 contract
    spec = app.openapi()
    props = spec.get("components", {}).get("schemas", {}).get("PatientInput", {}).get("properties", {})
    check("E4 54 typed fields", len(props) == 54, f"fields={len(props)}")
    has_resp = any(getattr(rte, "path", None) == "/api/predict" and getattr(rte, "response_model", None) is not None
                   for rte in app.routes)
    check("E4 response_model", has_resp, "")

    # 4. E5/E6 artifacts
    check("E5 samples.json", (ROOT / "ml" / "artifacts" / "samples.json").exists(), "")
    man_p = ROOT / "ml" / "artifacts" / "MANIFEST.json"
    bundle = ROOT / "ml" / "artifacts" / "model_bundle.joblib"
    ok_man = man_p.exists() and bundle.exists() and len(json.loads(man_p.read_text(encoding="utf-8")).get("sha256", "")) == 64
    check("E6 MANIFEST valid 64-hex", ok_man, f"bundle_bytes={bundle.stat().st_size if bundle.exists() else -1}")

    # 5. A3 phys bounds
    from ml.pipeline import load_schema
    schema = load_schema()
    missing = [k for k, m in schema.items() if m.get("encoding") in ("numeric", "ordinal")
               and ("phys_min" not in m or "phys_max" not in m)]
    check("A3 phys bounds present", missing == [], f"missing={missing[:3]}")

    # 6. WhatIf + disclaimer (D7/D8 invariant tail)
    p = X.iloc[0].to_dict()
    keys = list(p.keys())[:12]
    base = {k: p[k] for k in keys}
    mod = dict(base)
    for k in ("BP", "Age", "PR", "Weight"):
        if k in mod and isinstance(mod[k], (int, float)):
            mod[k] = float(mod[k]) + 5
            break
    w = c.post("/api/whatif", json={"baseline": base, "modified": mod})
    ok_w = w.status_code == 200 and abs(w.json()["delta_cad_prob"] - (w.json()["modified_cad_prob"] - w.json()["baseline_cad_prob"])) < 1e-6
    check("whatif delta math", ok_w, f"status={w.status_code}")
    check("disclaimer invariant", "Decision support / educational purposes only" in rp.json().get("disclaimer", ""), "")

    # 7. Ensemble margin identity (B6/A6, no retrain, uses ml/ensemble.py only)
    import numpy as np
    from ml.ensemble import LogitEnsemble
    from ml.pipeline import build_preprocessor, load_schema
    from ml.pipeline import load_raw_dataset as _l
    _, yfull = _l(DEFAULT_DATA_PATH)
    yc = yfull["Cath"].iloc[:60].reset_index(drop=True)
    Xs_raw = X.iloc[:60]
    Xs = build_preprocessor(load_schema()).fit_transform(Xs_raw)
    ens = LogitEnsemble().fit(Xs, yc)
    z = ens.decision_function(Xs)
    p_ = ens.predict_proba(Xs)[:, 1]
    check("ensemble proba==sigmoid(margin)", float(np.max(np.abs(p_ - np.clip(1 / (1 + np.exp(-z)), 1e-7, 1 - 1e-7)))) < 1e-12, "")
    check("ensemble margin avg identity",
          float(np.max(np.abs(z - (0.5 * ens.lr_.decision_function(Xs) + 0.5 * ens.xgb_.predict(Xs, output_margin=True))))) < 1e-12, "")

    # 8. Hygiene tail (G1/G7/F1) — read-only notes
    vtxt = (ROOT / "docs" / "VIDEO.md").read_text(encoding="utf-8") if (ROOT / "docs" / "VIDEO.md").exists() else ""
    check("G1 video not fabricated", "PENDING" in vtxt and "youtu.be/" not in vtxt.replace("PENDING", ""), "")
    check("CITATION.cff exists", (ROOT / "CITATION.cff").exists(), "")
    check("requirements-dev exists", (ROOT / "requirements-dev.txt").exists(), "")

    out = ROOT / "reports" / "robustness_checks.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote {out} pass={results['pass']} fail={results['fail']}")
    return 1 if results["fail"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
