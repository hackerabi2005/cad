"""
Unified cardiac risk prediction and SHAP explanation service.
Centralizes validation, model scoring, coherence enforcement, thresholding,
and explanation generation in exactly one place.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from ml.explain import explain_sample
from ml.pipeline import TARGETS


class ValidationError(ValueError):
    """Raised when patient input feature values are outside plausible physiological limits."""
    pass


def validate_patient_payload(payload: Dict[str, Any], schema: Dict[str, Any]) -> List[str]:
    """Validates patient numeric features against physiological bounds in schema."""
    out_of_bounds = []
    for k, v in payload.items():
        if v is not None and k in schema:
            meta = schema[k]
            if meta["encoding"] in ("numeric", "ordinal") and isinstance(v, (int, float)):
                min_v = meta.get("min", -1e9)
                max_v = meta.get("max", 1e9)
                # Flag if value is outside plausible boundary (20% below min or 2.5x max)
                if v < min_v * 0.2 or v > max_v * 2.5:
                    out_of_bounds.append(
                        f"Field '{k}' value {v} is outside plausible physiological range [{min_v}, {max_v}]"
                    )
    return out_of_bounds


def predict_patient(payload: Dict[str, Any], ml_state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Executes end-to-end inference and SHAP attribution for a single patient record.
    Enforces risk coherence P(CAD) >= max(P_vessels) and sensitivity-first thresholding.
    """
    schema = ml_state["schema"]

    # 1. Bounds validation
    errors = validate_patient_payload(payload, schema)
    if errors:
        raise ValidationError("; ".join(errors))

    # 2. Build single-row DataFrame with defaults for unprovided features
    row_dict: Dict[str, Any] = {}
    cohort_defaults_applied: List[str] = []

    for feat_name, meta in schema.items():
        if feat_name in payload and payload[feat_name] is not None:
            row_dict[feat_name] = payload[feat_name]
        else:
            if meta["encoding"] in ("numeric", "ordinal"):
                row_dict[feat_name] = meta.get("median", np.nan)
            else:
                row_dict[feat_name] = meta.get("categories", ["N"])[0]
            cohort_defaults_applied.append(feat_name)

    sample_df = pd.DataFrame([row_dict])

    # 3. Model probability scoring across targets
    probs: Dict[str, float] = {}
    for tgt in TARGETS:
        pipe = ml_state["pipelines"][tgt]
        p = float(pipe.predict_proba(sample_df)[0, 1])
        probs[tgt] = round(p, 4)

    # 4. Logical risk coherence: P(CAD_coherent) = max(P(CAD), max(vessel_P))
    raw_cad_prob = probs["Cath"]
    max_vessel_prob = max(probs["LAD"], probs["LCX"], probs["RCA"])

    if ml_state.get("coherence_fix", True):
        coherent_cad_prob = round(max(raw_cad_prob, max_vessel_prob), 4)
    else:
        coherent_cad_prob = raw_cad_prob

    # 5. Threshold evaluation
    thresholds = ml_state["thresholds"]
    cad_high_thresh = thresholds["Cath"]["high_sensitivity_threshold"]

    cad_label = "High Risk" if coherent_cad_prob >= 0.50 else "Low Risk"
    cad_high_sens_label = "High Risk" if coherent_cad_prob >= cad_high_thresh else "Low Risk"

    vessel_results: Dict[str, Any] = {}
    for v in ["LAD", "LCX", "RCA"]:
        p = probs[v]
        v_thresh = thresholds[v]["high_sensitivity_threshold"]
        v_nest_sens = thresholds[v].get("nested_cv_operating_metrics", {}).get("sensitivity_mean", 0.90)
        vessel_results[v] = {
            "prob": p,
            "label": "Stenotic" if p >= 0.50 else "Normal",
            "high_sens_label": "Stenotic" if p >= v_thresh else "Normal",
            "threshold": 0.50,
            "high_sensitivity_threshold": v_thresh,
            "nested_sensitivity": v_nest_sens,
        }

    # 6. Exact additive SHAP explanations
    explanations: Dict[str, Any] = {}
    for tgt in TARGETS:
        exp = explain_sample(
            pipeline=ml_state["pipelines"][tgt],
            explainer=ml_state["explainers"][tgt],
            parent_mapping=ml_state["parent_mappings"][tgt],
            schema=schema,
            sample_df=sample_df,
        )
        explanations[tgt] = exp

    cad_nest_sens = thresholds["Cath"].get("nested_cv_operating_metrics", {}).get("sensitivity_mean", 0.90)

    return {
        "cad": {
            "prob": coherent_cad_prob,
            "raw_prob": raw_cad_prob,
            "coherent_prob": coherent_cad_prob,
            "label": cad_label,
            "high_sens_label": cad_high_sens_label,
            "threshold": 0.50,
            "high_sensitivity_threshold": cad_high_thresh,
            "nested_sensitivity": cad_nest_sens,
            "coherence_adjusted": coherent_cad_prob > raw_cad_prob,
        },
        "vessels": vessel_results,
        "explain": explanations,
        "cohort_defaults_applied": cohort_defaults_applied,
        "disclaimer": "Decision support / educational use only — not a substitute for formal diagnostic imaging.",
    }
