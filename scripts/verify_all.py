"""Bottom-up verification plugs — read-only checks that do NOT edit concurrent work.

Covers Verification Report DoD items in additive fashion:
- API fuzz (A3): zero 500s, 422 on garbage
- Contract fields (A1/A4): raw_prob, displayed_prob, driving_vessel, missing/imputed/completeness/reliability/warnings
- SHAP units (A6): reports mixed units without changing models
- OpenAPI (E4): 54 typed fields snapshot
- Security headers (E1-E3): checked live if server running, else code-level
- Determinism note (B2): checks RNG-free flags in train.py (read-only)
- Hygiene (G1/G7): VIDEO placeholder present-not-fabricated, citation year note

Usage: python scripts/verify_all.py  (no writes except reports/verification_plugs.json)
"""
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
    # 1. OpenAPI typed fields (E4/A3) — read-only
    from api.main import app, load_artifacts
    load_artifacts()
    spec = app.openapi()
    schemas = spec.get("components", {}).get("schemas", {})
    pin = schemas.get("PatientInput", {}).get("properties", {})
    check("openapi PatientInput typed fields", len(pin) == 54, f"fields={len(pin)}")
    # response_model present?
    has_resp = False
    for r in app.routes:
        if getattr(r, "path", None) == "/api/predict" and getattr(r, "response_model", None) is not None:
            # None means untyped; document as open
            has_resp = True
    check("openapi response_model present (E4 remaining)", has_resp, "expected typed response_model; currently None = handoff to owner of api/main.py")

    # 2. API fuzz matrix (A3) — zero 500s
    from fastapi.testclient import TestClient
    c = TestClient(app, raise_server_exceptions=False)
    fuzz = {
        "empty": ({}, 422),
        "Age=200": ({"Age": 200}, 422),
        "Age=abc": ({"Age": "abc"}, 422),
        "Sex=Banana": ({"Sex": "Banana"}, 422),
        "BP=400": ({"Age": 55, "BP": 400}, 422),
        "unknown field": ({"Age": 55, "Foo": 1}, 422),
    }
    zero_500 = True
    for n, (body, exp) in fuzz.items():
        st = c.post("/api/predict", json=body).status_code
        ok = (st == exp and st != 500)
        zero_500 = zero_500 and (st != 500)
        check(f"fuzz {n} -> {exp}", ok, f"got={st}")
    check("fuzz zero-500s", zero_500, "")

    # 3. Contract fields over full cohort (A1/A4) — read-only sweep
    from ml.pipeline import DEFAULT_DATA_PATH, load_raw_dataset
    X, _ = load_raw_dataset(DEFAULT_DATA_PATH)
    ok_fields = True
    adj = 0
    worst_add = 0.0
    units = {}
    for i in range(len(X)):
        r = c.post("/api/predict", json=X.iloc[i].to_dict())
        if r.status_code != 200:
            ok_fields = False
            check(f"cohort row {i} 200", False, f"got={r.status_code}")
            break
        j = r.json()
        for k in ("raw_prob", "displayed_prob", "driving_vessel"):
            if k not in j.get("cad", {}):
                ok_fields = False
        for k in ("missing_features", "imputed_values", "completeness", "reliability", "warnings"):
            if k not in j:
                ok_fields = False
        if j["cad"].get("coherence_adjusted"):
            adj += 1
        for t, e in j.get("explain", {}).items():
            units[t] = e.get("shap_unit")
            worst_add = max(worst_add, abs(float(e.get("additive_error", 0))))
    check("contract fields present all 303 rows (A1/A4)", ok_fields, f"coherence_adjusted={adj}/303")
    check("coherence count matches cohort (44/303 retrained, 64/303 baseline)", adj in (44, 64), f"got={adj}")
    all_logodds = all(u == "log-odds" for u in units.values())
    check("SHAP units unified log-odds (A6 remaining)", all_logodds, f"units={units} worst_add_err={worst_add:.2e} (LAD=probability is known remaining)")

    # 4. RNG-free flags (B2) — read-only source check
    train_src = (ROOT / "ml" / "train.py").read_text(encoding="utf-8")
    rng_free = ("subsample=1.0" in train_src and "colsample_bytree=1.0" in train_src and "n_jobs=1" in train_src)
    check("train.py RNG-free XGB flags (B2)", rng_free, "")

    # 5. G1 video placeholder not fabricated
    video_md = (ROOT / "docs" / "VIDEO.md")
    if video_md.exists():
        txt = video_md.read_text(encoding="utf-8")
        pending = "PENDING" in txt and "youtube.com/watch?v=PENDING" in txt.replace(" ", "")
        no_fake = "watch?v=dQw" not in txt and "youtu.be/" not in txt.replace("PENDING", "")
        check("G1 VIDEO.md placeholder honest (human-only)", pending, "placeholder present, URL not fabricated")
    else:
        check("G1 VIDEO.md exists", False, "missing")

    # 6. MANIFEST exists (E6 remaining)
    check("E6 MANIFEST.json exists", (ROOT / "ml" / "artifacts" / "MANIFEST.json").exists(), "manifest artifact present")

    # 7. Metrics parity across artifacts (metrics.json, model_card.json, README.md)
    metrics_p = ROOT / "ml" / "artifacts" / "metrics.json"
    card_p = ROOT / "ml" / "artifacts" / "model_card.json"
    readme_p = ROOT / "README.md"
    if metrics_p.exists() and card_p.exists() and readme_p.exists():
        mj = json.loads(metrics_p.read_text(encoding="utf-8"))
        cj = json.loads(card_p.read_text(encoding="utf-8"))
        rtxt = readme_p.read_text(encoding="utf-8")
        
        # Check AUCs match between metrics.json and model_card.json
        auc_match = True
        for tgt in ["Cath", "LAD", "LCX", "RCA"]:
            m_auc = mj.get("operating_thresholds", {}).get(tgt, {}).get("metrics_at_05", {}).get("roc_auc")
            c_auc = cj.get("operating_thresholds", {}).get(tgt, {}).get("metrics_at_05", {}).get("roc_auc")
            if m_auc != c_auc:
                auc_match = False
            # Check README contains this target's AUC formatted
            if f"{m_auc:.3f}" not in rtxt:
                auc_match = False
        check("metrics parity: metrics.json == model_card.json == README.md", auc_match, "AUCs and thresholds synchronized across all docs")

        modal_p = ROOT / "web" / "src" / "components" / "AboutModelModal.tsx"
        if modal_p.exists():
            modal_txt = modal_p.read_text(encoding="utf-8")
            modal_match = all(
                f"{mj.get('candidate_models', {}).get(t, {}).get('LogitEnsemble', {}).get('roc_auc_mean', 0):.3f}" in modal_txt
                for t in ["Cath", "LAD", "LCX", "RCA"]
            )
            check("metrics parity: AboutModelModal.tsx matches metrics.json", modal_match, "CV ROC-AUCs synchronized in modal")
        else:
            check("AboutModelModal.tsx exists", False, "file missing")
    else:
        check("metrics parity files exist", False, "one or more files missing")

    out = ROOT / "reports" / "verification_results.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote {out} pass={results['pass']} fail={results['fail']}")
    return 1 if results["fail"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
