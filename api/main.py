"""
FastAPI Backend for Cardiovascular Risk Prediction and SHAP Explainability.
Serves:
- GET /api/schema
- GET /api/metrics
- GET /api/samples
- POST /api/predict
- Static build of web frontend in production
"""

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, create_model, Field

from ml.explain import explain_sample
from ml.pipeline import DEFAULT_DATA_PATH, SCHEMA_PATH, TARGETS, load_raw_dataset, load_schema

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

    # Load 3 representative sample patients (low, mid, high risk)
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
            "description": f"Predicted CAD probability: {probs[low_idx]:.1%}. Normal coronaries, normal vitals.",
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
            "description": f"Predicted CAD probability: {probs[high_idx]:.1%}. Strong angiographic and ECG risk indicators.",
            "data": X.iloc[high_idx].to_dict(),
        },
    ]

    print("All ML artifacts loaded successfully.")


# Dynamic Pydantic schema validation for patient inputs
def create_patient_input_model():
    schema = load_schema()
    fields: Dict[str, Any] = {}

    for name, meta in schema.items():
        enc = meta["encoding"]
        if enc in ("numeric", "ordinal"):
            min_val = meta.get("min", -999999.0)
            max_val = meta.get("max", 999999.0)
            # Allow reasonable buffer for real patient inputs while catching gross errors
            field_def = (
                Optional[float],
                Field(default=None, ge=min_val * 0.5 if min_val >= 0 else min_val * 2.0, le=max_val * 1.5),
            )
        elif enc in ("binary", "nominal"):
            cats = meta.get("categories", [0, 1])
            field_def = (Optional[Union[str, int, float]], Field(default=None))
        else:
            field_def = (Optional[Union[str, int, float]], Field(default=None))

        fields[name] = field_def

    return create_model("PatientInput", **fields)


PatientInput = create_patient_input_model()


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_artifacts()
    yield


app = FastAPI(
    title="Cardiovascular Risk & 3D Coronary Viewer API",
    description="Multimodal CAD risk visualization & multi-vessel stenosis prediction system",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_no_cache_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


@app.get("/api/health")
def health_check():
    return {"status": "healthy", "service": "cad-prediction-api"}


@app.get("/api/schema")
def get_schema():
    return ml_state["schema"]


@app.get("/api/metrics")
def get_metrics():
    return {
        "metrics": ml_state["metrics"],
        "shap_global": ml_state["shap_global"],
        "model_card": ml_state["model_card"],
    }


@app.get("/api/samples")
def get_samples():
    return ml_state["samples"]


@app.post("/api/predict")
def predict_cardiac_risk(payload: Dict[str, Any]):
    schema = ml_state["schema"]

    # Validate numeric ranges to guarantee 422 if out of bounds
    out_of_bounds = []
    for k, v in payload.items():
        if v is not None and k in schema:
            meta = schema[k]
            if meta["encoding"] in ("numeric", "ordinal") and isinstance(v, (int, float)):
                min_v = meta.get("min", -1e9)
                max_v = meta.get("max", 1e9)
                # Bounds check
                if v < min_v * 0.2 or v > max_v * 2.5:
                    out_of_bounds.append(
                        f"Field '{k}' value {v} is outside plausible physiological range [{min_v}, {max_v}]"
                    )

    if out_of_bounds:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="; ".join(out_of_bounds),
        )

    # Convert payload into DataFrame row matching schema features
    row_dict: Dict[str, Any] = {}
    cohort_defaults_applied = []

    for feat_name, meta in schema.items():
        if feat_name in payload and payload[feat_name] is not None:
            row_dict[feat_name] = payload[feat_name]
        else:
            # Leave as None or default median/mode
            if meta["encoding"] in ("numeric", "ordinal"):
                row_dict[feat_name] = meta.get("median", np.nan)
            else:
                row_dict[feat_name] = meta.get("categories", ["N"])[0]
            cohort_defaults_applied.append(feat_name)

    sample_df = pd.DataFrame([row_dict])

    # Run predictions across the 4 targets
    probs: Dict[str, float] = {}
    for tgt in TARGETS:
        pipe = ml_state["pipelines"][tgt]
        p = float(pipe.predict_proba(sample_df)[0, 1])
        probs[tgt] = round(p, 4)

    # Coherence check & adjustment
    raw_cad_prob = probs["Cath"]
    max_vessel_prob = max(probs["LAD"], probs["LCX"], probs["RCA"])

    if ml_state["coherence_fix"]:
        coherent_cad_prob = round(max(raw_cad_prob, max_vessel_prob), 4)
    else:
        coherent_cad_prob = raw_cad_prob

    # Compute labels using thresholds
    thresholds = ml_state["thresholds"]

    cad_high_thresh = thresholds["Cath"]["high_sensitivity_threshold"]
    cad_label = "High Risk" if coherent_cad_prob >= 0.50 else "Low Risk"
    cad_high_sens_label = "High Risk" if coherent_cad_prob >= cad_high_thresh else "Low Risk"

    vessel_results = {}
    for v in ["LAD", "LCX", "RCA"]:
        p = probs[v]
        v_thresh = thresholds[v]["high_sensitivity_threshold"]
        vessel_results[v] = {
            "prob": p,
            "label": "Stenotic" if p >= 0.50 else "Normal",
            "high_sens_label": "Stenotic" if p >= v_thresh else "Normal",
            "threshold": 0.50,
            "high_sensitivity_threshold": v_thresh,
        }

    # Generate SHAP explanations for all 4 targets
    explanations = {}
    for tgt in TARGETS:
        exp = explain_sample(
            pipeline=ml_state["pipelines"][tgt],
            explainer=ml_state["explainers"][tgt],
            parent_mapping=ml_state["parent_mappings"][tgt],
            schema=schema,
            sample_df=sample_df,
        )
        explanations[tgt] = exp

    return {
        "cad": {
            "prob": raw_cad_prob,
            "coherent_prob": coherent_cad_prob,
            "label": cad_label,
            "high_sens_label": cad_high_sens_label,
            "threshold": 0.50,
            "high_sensitivity_threshold": cad_high_thresh,
            "coherence_adjusted": coherent_cad_prob > raw_cad_prob,
        },
        "vessels": vessel_results,
        "explain": explanations,
        "cohort_defaults_applied": cohort_defaults_applied,
        "disclaimer": "Decision support / educational use only — not a substitute for formal diagnostic imaging.",
    }


# Serve React build in production if built
if WEB_DIST.exists():
    app.mount("/", StaticFiles(directory=str(WEB_DIST), html=True), name="static")
