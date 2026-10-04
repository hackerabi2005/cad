"""
Python ML Sidecar Service for Cardio3D AI.
Runs on internal port 8001.
Provides fast JSON inference and exact additive SHAP explanations to the Rust Axum web engine.
"""

import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from ml.explain import explain_sample
from ml.pipeline import DEFAULT_DATA_PATH, TARGETS, load_raw_dataset, load_schema

BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BASE_DIR / "ml" / "artifacts"

ml_state: Dict[str, Any] = {}


def load_artifacts():
    bundle_path = ARTIFACTS_DIR / "model_bundle.joblib"
    if not bundle_path.exists():
        raise RuntimeError(f"Artifact {bundle_path} not found.")

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

    # Load 3 representative sample patients
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
            "description": f"Predicted CAD probability: {probs[low_idx]:.1%}. Normal coronaries.",
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
            "description": f"Predicted CAD probability: {probs[high_idx]:.1%}. Strong stenosis indicators.",
            "data": X.iloc[high_idx].to_dict(),
        },
    ]


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_artifacts()
    yield


app = FastAPI(title="Cardio3D ML Sidecar", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/health")
def health():
    return {"status": "healthy", "service": "cardio3d-ml-sidecar"}


@app.get("/schema")
def get_schema():
    return ml_state["schema"]


@app.get("/samples")
def get_samples():
    return ml_state["samples"]


@app.get("/metrics")
def get_metrics():
    return {
        "metrics": ml_state["metrics"],
        "shap_global": ml_state["shap_global"],
        "model_card": ml_state["model_card"],
    }


@app.post("/predict")
def predict(payload: Dict[str, Any]):
    schema = ml_state["schema"]

    # Bounds check
    for k, v in payload.items():
        if v is not None and k in schema:
            meta = schema[k]
            if meta["encoding"] in ("numeric", "ordinal") and isinstance(v, (int, float)):
                min_v = meta.get("min", -1e9)
                max_v = meta.get("max", 1e9)
                if v < min_v * 0.2 or v > max_v * 2.5:
                    raise HTTPException(
                        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                        detail=f"Field '{k}' value {v} outside plausible range [{min_v}, {max_v}]",
                    )

    # Convert to DataFrame row
    row_dict: Dict[str, Any] = {}
    for feat_name, meta in schema.items():
        if feat_name in payload and payload[feat_name] is not None:
            row_dict[feat_name] = payload[feat_name]
        else:
            if meta["encoding"] in ("numeric", "ordinal"):
                row_dict[feat_name] = meta.get("median", np.nan)
            else:
                row_dict[feat_name] = meta.get("categories", ["N"])[0]

    sample_df = pd.DataFrame([row_dict])

    probs: Dict[str, float] = {}
    for tgt in TARGETS:
        pipe = ml_state["pipelines"][tgt]
        probs[tgt] = round(float(pipe.predict_proba(sample_df)[0, 1]), 4)

    raw_cad_prob = probs["Cath"]
    max_vessel_prob = max(probs["LAD"], probs["LCX"], probs["RCA"])
    coherent_cad_prob = round(max(raw_cad_prob, max_vessel_prob), 4)

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

    explanations = {}
    for tgt in TARGETS:
        explanations[tgt] = explain_sample(
            pipeline=ml_state["pipelines"][tgt],
            explainer=ml_state["explainers"][tgt],
            parent_mapping=ml_state["parent_mappings"][tgt],
            schema=schema,
            sample_df=sample_df,
        )

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
        "disclaimer": "Decision support / educational use only — not a substitute for formal diagnostic imaging.",
    }


if __name__ == "__main__":
    uvicorn.run("ml.sidecar:app", host="127.0.0.1", port=8001, log_level="info")
