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

from ml.pipeline import DEFAULT_DATA_PATH, TARGETS, load_raw_dataset, load_schema
from ml.service import ValidationError, predict_patient

BASE_DIR = Path(__file__).resolve().parent.parent.parent
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
    try:
        return predict_patient(payload, ml_state)
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
        )


if __name__ == "__main__":
    uvicorn.run("ml.sidecar:app", host="127.0.0.1", port=8001, log_level="info")
