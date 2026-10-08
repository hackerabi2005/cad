import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from ml.pipeline import TARGETS
from ml.explain import explain_sample


class ValidationError(Exception):
    pass


def validate_patient_payload(payload: Dict[str, Any], schema: Dict[str, Any]) -> List[str]:
    """Validates patient input features against physiological boundaries."""
    out_of_bounds = []
    for k, v in payload.items():
        if k in schema and v is not None:
            meta = schema[k]
            if meta["encoding"] in ("numeric", "ordinal") and isinstance(v, (int, float)):
                min_v = meta.get("min", float("-inf"))
                max_v = meta.get("max", float("inf"))
                if v < min_v * 0.2 or v > max_v * 2.5:
                    out_of_bounds.append(
                        f"Field '{k}' value {v} is outside plausible physiological range [{min_v}, {max_v}]"
                    )
    return out_of_bounds


def compute_wilson_score_interval(k: int, n: int) -> Tuple[float, float]:
    """Computes two-sided Wilson score interval for binomial proportion k/n at 95% confidence."""
    if n <= 0:
        return (0.0, 1.0)
    # Using z for 95% confidence level: norm.ppf(0.975) = 1.959963984540054
    z_val = 1.959963984540054
    p_hat = k / n
    denom = 1.0 + (z_val * z_val) / n
    center = (p_hat + (z_val * z_val) / (2.0 * n)) / denom
    margin = (z_val * np.sqrt((p_hat * (1.0 - p_hat)) / n + (z_val * z_val) / (4.0 * n * n))) / denom
    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)
    return (round(float(lower), 4), round(float(upper), 4))


def compute_similar_cohort_summary(
    score: float,
    tgt: str,
    ml_state: Dict[str, Any],
    k_neighbors: int = 50,
) -> Optional[Dict[str, Any]]:
    """
    Finds nearest out-of-fold predictions to score, computing observed positive rate
    and 95% Wilson score interval over the empirical reference cohort.
    """
    oof_bundle = ml_state.get("oof_predictions", {}).get(tgt)
    if not oof_bundle:
        return None

    oof_scores = np.asarray(oof_bundle["probs"], dtype=float)
    oof_labels = np.asarray(oof_bundle["labels"], dtype=int)

    # Nearest out-of-fold scores
    diffs = np.abs(oof_scores - score)
    nearest_idx = np.argsort(diffs)[:k_neighbors]
    sub_labels = oof_labels[nearest_idx]

    n_count = len(sub_labels)
    if n_count == 0:
        return None

    k_pos = int(np.sum(sub_labels))
    obs_rate = float(k_pos / n_count)
    w_low, w_high = compute_wilson_score_interval(k_pos, n_count)

    outcome_desc = "confirmed CAD" if tgt == "Cath" else "confirmed stenosis"
    pct_str = f"{int(round(obs_rate * 100))}%"
    summary_text = f"Among {n_count} similar-scoring patients, {pct_str} had {outcome_desc}"

    return {
        "n": n_count,
        "observed_rate": round(obs_rate, 4),
        "wilson_interval": [w_low, w_high],
        "summary": summary_text,
    }


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
            "similar_cohort": compute_similar_cohort_summary(p, v, ml_state),
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
            "similar_cohort": compute_similar_cohort_summary(coherent_cad_prob, "Cath", ml_state),
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
