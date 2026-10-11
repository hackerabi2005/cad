"""
FastAPI Backend for Cardiovascular Risk Prediction and SHAP Explainability.
Serves:
- GET /api/health (container health check)
- GET /api/version (service version & bundle SHA-256)
- GET /api/schema (54-feature metadata & physiological boundaries)
- GET /api/targets (target artery registry & operating thresholds)
- GET /api/metrics (cross-validation metrics & decision curve analysis)
- GET /api/samples (pre-configured patient case presets)
- POST /api/predict (end-to-end inference & additive SHAP attribution)
- POST /api/whatif (counterfactual sensitivity exploration)
- Static build of web frontend in production
"""

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, create_model, Field, ConfigDict

from ml.pipeline import DEFAULT_DATA_PATH, SCHEMA_PATH, TARGETS, load_raw_dataset, load_schema
from ml.service import ValidationError, predict_patient

BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BASE_DIR / "ml" / "artifacts"
WEB_DIST = BASE_DIR / "web" / "dist"

# State container for loaded artifacts
ml_state: Dict[str, Any] = {}


def load_artifacts():
    print("Loading ML artifacts at startup...")
    bundle_path = ARTIFACTS_DIR / "model_bundle.joblib"
    if not bundle_path.exists():
        raise RuntimeError(f"Artifact {bundle_path} not found. Run 'python -m ml.train' first.")

    bundle = joblib.load(bundle_path)
    ml_state["pipelines"] = bundle["pipelines"]
    ml_state["explainers"] = bundle["explainers"]
    ml_state["parent_mappings"] = bundle["parent_mappings"]
    ml_state["thresholds"] = bundle["thresholds"]
    ml_state["coherence_fix"] = bundle["coherence_fix"]

    with open(ARTIFACTS_DIR / "metrics.json", "r", encoding="utf-8") as f:
        ml_state["metrics"] = json.load(f)

    with open(ARTIFACTS_DIR / "shap_global.json", "r", encoding="utf-8") as f:
        ml_state["shap_global"] = json.load(f)

    with open(ARTIFACTS_DIR / "model_card.json", "r", encoding="utf-8") as f:
        ml_state["model_card"] = json.load(f)

    schema = load_schema()
    ml_state["schema"] = schema

    targets_path = BASE_DIR / "ml" / "targets.json"
    if targets_path.exists():
        with open(targets_path, "r", encoding="utf-8") as f:
            ml_state["targets"] = json.load(f)
    else:
        ml_state["targets"] = {"targets": []}

    dca_path = ARTIFACTS_DIR / "decision_curve.json"
    if dca_path.exists():
        with open(dca_path, "r", encoding="utf-8") as f:
            ml_state["dca"] = json.load(f)
    else:
        ml_state["dca"] = {}

    # Load 3 representative sample patients from artifact or compute once
    samples_file = ARTIFACTS_DIR / "samples.json"
    if samples_file.exists():
        with open(samples_file, "r", encoding="utf-8") as f:
            ml_state["samples"] = json.load(f)
    else:
        X, _ = load_raw_dataset(DEFAULT_DATA_PATH)
        cad_pipe = ml_state["pipelines"]["Cath"]
        probs = cad_pipe.predict_proba(X)[:, 1]

        low_idx = int(np.argmin(probs))
        high_idx = int(np.argmax(probs))
        mid_idx = int(np.argsort(np.abs(probs - 0.50))[0])

        ml_state["samples"] = [
            {
                "id": "sample-low-risk",
                "name": "Patient A — Low Risk Profile",
                "description": f"Predicted CAD probability: {probs[low_idx]:.1%}. Low clinical risk profile, absence of typical chest pain.",
                "data": X.iloc[low_idx].to_dict(),
            },
            {
                "id": "sample-mid-risk",
                "name": "Patient B — Moderate / Borderline Risk",
                "description": f"Predicted CAD probability: {probs[mid_idx]:.1%}. Moderate risk markers.",
                "data": X.iloc[mid_idx].to_dict(),
            },
            {
                "id": "sample-high-risk",
                "name": "Patient C — High Risk Stenosis",
                "description": f"Predicted CAD probability: {probs[high_idx]:.1%}. High clinical and ECG risk markers (typical chest pain, T-inversion, reduced EF).",
                "data": X.iloc[high_idx].to_dict(),
            },
        ]
        with open(samples_file, "w", encoding="utf-8") as f:
            json.dump(ml_state["samples"], f, indent=2)

    print("All ML artifacts loaded successfully.")


# Dynamic Pydantic schema validation for patient inputs
def create_patient_input_model():
    schema = load_schema()
    fields: Dict[str, Any] = {}

    for name, meta in schema.items():
        enc = meta["encoding"]
        if enc in ("numeric", "ordinal"):
            pmin = meta.get("phys_min", meta.get("min", float("-inf")) * 0.2)
            pmax = meta.get("phys_max", meta.get("max", float("inf")) * 2.5)
            field_def = (
                Optional[float],
                Field(default=None, ge=pmin, le=pmax, strict=True, allow_inf_nan=False),
            )
        elif enc in ("binary", "nominal"):
            cats = list(meta.get("categories", []))
            if name == "Sex" and "Fmale" in cats and "Female" not in cats:
                cats.append("Female")
            if cats:
                field_def = (Optional[Literal[tuple(cats)]], Field(default=None))
            else:
                field_def = (Optional[str], Field(default=None))
        else:
            field_def = (Optional[Union[str, int, float]], Field(default=None))

        fields[name] = field_def

    return create_model("PatientInput", __config__=ConfigDict(extra="forbid"), **fields)


PatientInput = create_patient_input_model()


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_artifacts()
    yield


app = FastAPI(
    title="Cardiovascular Risk & 3D Coronary Viewer API",
    description="Clinical CAD risk visualization & multi-vessel stenosis prediction system",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(GZipMiddleware, minimum_size=1024)

cors_origins_env = os.environ.get("CORS_ORIGINS", "")
allowed_origins = [o.strip() for o in cors_origins_env.split(",") if o.strip()] if cors_origins_env else [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.middleware("http")
async def security_and_cache_middleware(request: Request, call_next):
    response = await call_next(request)
    # E2 Security headers
    response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; connect-src 'self'; font-src 'self'; object-src 'none'; base-uri 'self'; form-action 'self'; frame-ancestors 'none';"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"

    # E3 Cache headers
    path = request.url.path
    if path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
        response.headers["Pragma"] = "no-cache"
    elif path.startswith("/assets/"):
        response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
    elif path.startswith("/models/"):
        response.headers["Cache-Control"] = "public, max-age=86400"
    elif path in ("/", "/index.html"):
        response.headers["Cache-Control"] = "no-cache"
    return response


@app.get("/api/health")
def health_check():
    return {"status": "healthy", "service": "cad-prediction-api"}


@app.get("/api/version")
def version_info():
    import sklearn
    manifest_path = ARTIFACTS_DIR / "MANIFEST.json"
    bundle_hash = None
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            bundle_hash = json.load(f).get("model_bundle_sha256")
    return {
        "version": "1.1.0",
        "service": "Cardio3D AI",
        "sklearn_version": sklearn.__version__,
        "bundle_sha256": bundle_hash,
    }


@app.get("/api/schema")
def get_schema():
    return ml_state["schema"]


@app.get("/api/targets")
def get_targets():
    return ml_state.get("targets", {"targets": []})


@app.get("/api/metrics")
def get_metrics():
    return {
        "metrics": ml_state["metrics"],
        "shap_global": ml_state["shap_global"],
        "model_card": ml_state["model_card"],
        "dca": ml_state.get("dca", {}),
        "vessel_count_confusion": ml_state["metrics"].get("vessel_count_confusion"),
    }


@app.get("/api/samples")
def get_samples():
    return ml_state["samples"]


HTTP_422 = getattr(status, "HTTP_422_UNPROCESSABLE_CONTENT", 422)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    error_messages = []
    structured_errors = []
    schema = ml_state.get("schema") or load_schema()

    for err in exc.errors():
        loc = err.get("loc", [])
        field = str(loc[-1]) if loc else "unknown"
        msg = err.get("msg", "")
        err_type = err.get("type", "")
        inp = err.get("input", None)

        if "extra_forbidden" in err_type:
            readable = f"Field '{field}' is an unknown feature not in schema."
        elif (
            "float_type" in err_type
            or "int_type" in err_type
            or "int_parsing" in err_type
            or "float_parsing" in err_type
            or "bool_type" in err_type
        ):
            readable = f"Field '{field}' must be numeric, got {type(inp).__name__ if inp is not None else 'invalid'}"
        elif "greater_than" in err_type or "less_than" in err_type:
            meta = schema.get(field, {})
            pmin = meta.get("phys_min", meta.get("min", "min"))
            pmax = meta.get("phys_max", meta.get("max", "max"))
            readable = f"Field '{field}' value {inp} is outside plausible physiological range [{pmin}, {pmax}]"
        elif "literal_error" in err_type:
            meta = schema.get(field, {})
            cats = meta.get("categories", [])
            readable = f"Field '{field}' value '{inp}' is invalid; expected one of {cats}"
        else:
            readable = f"Field '{field}': {msg}"

        error_messages.append(readable)
        structured_errors.append({"field": field, "code": err_type, "message": readable})

    detail_str = "; ".join(error_messages)
    return JSONResponse(
        status_code=HTTP_422,
        content={"detail": detail_str, "errors": structured_errors},
    )


class CadPredictionModel(BaseModel):
    prob: float
    raw_prob: Optional[float] = None
    displayed_prob: Optional[float] = None
    coherent_prob: float
    driving_vessel: Optional[str] = None
    similar_cohort: Optional[Dict[str, Any]] = None
    label: str
    high_sens_label: str
    threshold: float
    high_sensitivity_threshold: float
    nested_sensitivity: Optional[float] = None
    coherence_adjusted: bool
    flagged_by_vessel: Optional[bool] = None
    flagged_vessels_screening: Optional[List[str]] = None
    flagged_vessels_balanced: Optional[List[str]] = None
    flagged_vessels_count: Optional[int] = None
    diseased_vessels_count: Optional[int] = None
    diffuse_or_non_focal: Optional[bool] = None
    diffuse_or_non_focal_screening: Optional[bool] = None
    diffuse_or_non_focal_balanced: Optional[bool] = None
    diffuse_cad_message: Optional[str] = None
    plain_language_summary: Optional[str] = None


class VesselPredictionModel(BaseModel):
    prob: float
    label: str
    high_sens_label: str
    threshold: float
    high_sensitivity_threshold: float
    nested_sensitivity: Optional[float] = None


class PredictionResponse(BaseModel):
    cad: CadPredictionModel
    vessels: Dict[str, VesselPredictionModel]
    explain: Dict[str, Any]
    cohort_defaults_applied: List[str]
    missing_features: Optional[List[str]] = None
    imputed_values: Optional[Dict[str, Any]] = None
    completeness: Optional[Dict[str, Any]] = None
    reliability: Optional[str] = None
    warnings: Optional[List[str]] = None
    disclaimer: str


@app.post("/api/predict", response_model=PredictionResponse)
def predict_cardiac_risk(payload: PatientInput):
    data = {k: v for k, v in payload.model_dump().items() if v is not None}
    if not data:
        raise HTTPException(
            status_code=HTTP_422,
            detail="Empty payload: at least physiological parameters must be provided.",
        )
    if data.get("Sex") == "Female":
        data["Sex"] = "Fmale"
    try:
        return predict_patient(data, ml_state)
    except ValidationError as e:
        raise HTTPException(
            status_code=HTTP_422,
            detail=str(e),
        )


class WhatIfRequest(BaseModel):
    baseline: PatientInput
    modified: PatientInput


class WhatIfResponse(BaseModel):
    baseline_cad_prob: float
    modified_cad_prob: float
    delta_cad_prob: float
    vessel_deltas: Dict[str, float]
    baseline_prediction: PredictionResponse
    modified_prediction: PredictionResponse


@app.post("/api/whatif", response_model=WhatIfResponse)
def simulate_whatif(payload: WhatIfRequest):
    base_data = {k: v for k, v in payload.baseline.model_dump().items() if v is not None}
    mod_data = {k: v for k, v in payload.modified.model_dump().items() if v is not None}
    if not base_data or not mod_data:
        raise HTTPException(
            status_code=HTTP_422,
            detail="Both baseline and modified patient data must be provided.",
        )
    if base_data.get("Sex") == "Female":
        base_data["Sex"] = "Fmale"
    if mod_data.get("Sex") == "Female":
        mod_data["Sex"] = "Fmale"
    try:
        base_res = predict_patient(base_data, ml_state, compute_shap=False)
        mod_res = predict_patient(mod_data, ml_state, compute_shap=False)
        delta_cad = round(mod_res["cad"]["prob"] - base_res["cad"]["prob"], 4)
        vessel_deltas = {
            v: round(mod_res["vessels"][v]["prob"] - base_res["vessels"][v]["prob"], 4)
            for v in ("LAD", "LCX", "RCA")
        }
        return {
            "baseline_cad_prob": base_res["cad"]["prob"],
            "modified_cad_prob": mod_res["cad"]["prob"],
            "delta_cad_prob": delta_cad,
            "vessel_deltas": vessel_deltas,
            "baseline_prediction": base_res,
            "modified_prediction": mod_res,
        }
    except ValidationError as e:
        raise HTTPException(status_code=HTTP_422, detail=str(e))



# Serve React build in production if built
if WEB_DIST.exists():
    app.mount("/", StaticFiles(directory=str(WEB_DIST), html=True), name="static")
