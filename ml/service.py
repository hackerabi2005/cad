import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from ml.pipeline import TARGETS
from ml.explain import explain_sample


class ValidationError(Exception):
    pass


def validate_patient_payload(
    payload: Dict[str, Any], schema: Dict[str, Any]
) -> Tuple[List[str], List[str]]:
    """Validates patient input features against physiological boundaries and types. Returns (errors, warnings)."""
    if not payload:
        return (["Empty payload: at least physiological parameters must be provided."], [])

    errors = []
    warnings = []
    provided_count = 0

    if payload.get("Sex") == "Female":
        payload["Sex"] = "Fmale"

    for k, v in payload.items():
        if k not in schema:
            errors.append(f"Field '{k}' is an unknown feature not in schema.")
            continue
        if v is not None:
            provided_count += 1
            meta = schema[k]
            enc = meta["encoding"]
            if enc in ("numeric", "ordinal"):
                if type(v) is bool or not isinstance(v, (int, float)):
                    errors.append(f"Field '{k}' must be numeric, got {type(v).__name__}")
                    continue
                pmin = meta.get("phys_min", meta.get("min", float("-inf")) * 0.2)
                pmax = meta.get("phys_max", meta.get("max", float("inf")) * 2.5)
                if v < pmin or v > pmax:
                    errors.append(
                        f"Field '{k}' value {v} is outside plausible physiological range [{pmin}, {pmax}]"
                    )
                elif "min" in meta and "max" in meta and (v < meta["min"] or v > meta["max"]):
                    warnings.append(
                        f"OUTSIDE_TRAINING_RANGE: Field '{k}' value {v} is outside dataset distribution [{meta['min']}, {meta['max']}]."
                    )
            elif enc in ("binary", "nominal"):
                cats = meta.get("categories", [])
                if k == "Sex" and str(v).lower() in ("female", "fmale", "male"):
                    pass
                elif cats and (v not in cats and str(v) not in [str(c) for c in cats]):
                    errors.append(f"Field '{k}' value '{v}' is invalid; expected one of {cats}")

    # Completeness gate: if no syntax/bounds errors, verify minimum features provided
    if not errors and provided_count < 10:
        errors.append(
            f"INSUFFICIENT_INPUT: Insufficient patient data: provided {provided_count} features, minimum required is 10."
        )

    return errors, warnings


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

    # Nearest out-of-fold scores excluding exact self-match (diff == 0.0) if enough neighbors exist
    diffs = np.abs(oof_scores - score)
    valid_mask = diffs > 1e-5
    if np.sum(valid_mask) >= k_neighbors:
        candidate_indices = np.where(valid_mask)[0]
        sorted_order = np.argsort(diffs[candidate_indices])
        nearest_idx = candidate_indices[sorted_order[:k_neighbors]]
    else:
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
    w_low_pct = int(round(w_low * 100))
    w_high_pct = int(round(w_high * 100))
    summary_text = (
        f"Observed in {n_count} similar patients: {pct_str} (95% CI {w_low_pct}%–{w_high_pct}%). "
        f"Among {n_count} similar-scoring patients, {pct_str} had {outcome_desc}"
    )

    return {
        "n": n_count,
        "observed_rate": round(obs_rate, 4),
        "wilson_interval": [w_low, w_high],
        "summary": summary_text,
    }


def format_feature_phrase(f: Dict[str, Any]) -> str:
    name = f.get("feature", "")
    label = f.get("label", name)
    val = f.get("value")
    unit = f.get("unit", "")

    if str(val) in ("N", "0", "0.0", "False", "false"):
        if "Chest Pain" in label or "Angina" in label:
            return "no typical chest pain"
        if "Smoker" in label:
            return "non-smoking status"
        if "HTN" in name or "Hypertension" in label:
            return "absence of hypertension"
        if "DM" in name or "Diabetes" in label:
            return "absence of diabetes"
        if "RWMA" in name or "Regional Wall" in label:
            return "normal regional wall motion"
        if "Tinversion" in name or "T-wave" in label:
            return "no T-wave inversion"
        if "StDepression" in name or "ST" in label:
            return "no ST-segment depression"
        return f"absence of {label.lower()}"
    elif str(val) in ("Y", "1", "1.0", "True", "true"):
        if "Chest Pain" in label or "Angina" in label:
            return "typical anginal chest pain"
        if "Tinversion" in name or "T-wave" in label:
            return "ECG T-wave inversion"
        if "StDepression" in name or "ST" in label:
            return "ECG ST-segment depression"
        if "Smoker" in label:
            return "smoking history"
        if "HTN" in name or "Hypertension" in label:
            return "hypertension"
        if "DM" in name or "Diabetes" in label:
            return "diabetes mellitus"
        return f"{label.lower()}"
    elif isinstance(val, (int, float)):
        if "Age" in name or "Age" in label:
            return f"age ({int(val)} years)"
        if "EF" in name or "Ejection" in label:
            return f"low ejection fraction ({int(val)}%)" if val < 50 else f"preserved ejection fraction ({int(val)}%)"
        if "BP" in name:
            return f"blood pressure ({int(val)} mmHg)"
        if "FBS" in name or "Glucose" in label:
            return f"fasting blood sugar ({int(val)} mg/dl)"
        if unit:
            return f"{label.lower()} ({val:g} {unit})"
        return f"{label.lower()} ({val:g})"
    return label


def predict_patient(
    payload: Dict[str, Any],
    ml_state: Dict[str, Any],
    compute_shap: bool = True,
) -> Dict[str, Any]:
    """
    Executes end-to-end inference and SHAP attribution for a single patient record.
    Enforces risk coherence P(CAD) >= max(P_vessels) and sensitivity-first thresholding.
    """
    schema = ml_state["schema"]
    payload = dict(payload)
    sex_raw = str(payload.get("Sex", "")).strip().lower()
    if sex_raw in ("female", "fmale", "f"):
        payload["Sex"] = "Fmale"
    elif sex_raw in ("male", "m"):
        payload["Sex"] = "Male"

    # 1. Bounds validation
    errors, warnings = validate_patient_payload(payload, schema)
    if errors:
        raise ValidationError("; ".join(errors))

    # 2. Build single-row DataFrame with defaults for unprovided features
    row_dict: Dict[str, Any] = {}
    cohort_defaults_applied: List[str] = []
    missing_features: List[str] = []
    imputed_values: Dict[str, Any] = {}
    provided_count = 0
    group_stats: Dict[str, Dict[str, int]] = {}

    for feat_name, meta in schema.items():
        grp = meta.get("group", "other")
        if grp not in group_stats:
            group_stats[grp] = {"provided": 0, "total": 0}
        group_stats[grp]["total"] += 1

        if feat_name in payload and payload[feat_name] is not None:
            val = payload[feat_name]
            if feat_name == "Sex" and str(val).strip().lower() == "female":
                val = "Fmale"
            row_dict[feat_name] = val
            provided_count += 1
            group_stats[grp]["provided"] += 1
        else:
            if meta["encoding"] in ("numeric", "ordinal"):
                val = meta.get("median", np.nan)
            else:
                val = meta.get("mode", meta.get("categories", ["N"])[0])
            row_dict[feat_name] = val
            cohort_defaults_applied.append(feat_name)
            missing_features.append(feat_name)
            imputed_values[feat_name] = val

    sample_df = pd.DataFrame([row_dict])

    input_completeness = "high" if provided_count >= 35 else ("moderate" if provided_count >= 20 else "low")
    completeness = {
        "provided": provided_count,
        "total": len(schema),
        "by_group": group_stats,
    }

    # 3. Model probability scoring across targets
    probs: Dict[str, float] = {}
    for tgt in TARGETS:
        pipe = ml_state["pipelines"][tgt]
        p = float(pipe.predict_proba(sample_df)[0, 1])
        probs[tgt] = round(p, 4)

    # 4. Logical risk coherence: P(CAD_coherent) = max(P(CAD), max(vessel_P))
    raw_cad_prob = probs["Cath"]
    vessel_items = [("LAD", probs["LAD"]), ("LCX", probs["LCX"]), ("RCA", probs["RCA"])]
    driving_vessel, max_vessel_prob = max(vessel_items, key=lambda x: x[1])

    if ml_state.get("coherence_fix", True) and max_vessel_prob > raw_cad_prob:
        coherent_cad_prob = round(max_vessel_prob, 4)
        is_adjusted = True
    else:
        coherent_cad_prob = raw_cad_prob
        is_adjusted = False
        driving_vessel = None

    # 5. Threshold evaluation
    thresholds = ml_state["thresholds"]
    cad_high_thresh = thresholds["Cath"]["high_sensitivity_threshold"]

    vessel_results: Dict[str, Any] = {}
    any_vessel_balanced_flagged = False
    any_vessel_screening_flagged = False
    flagged_vessels_screening = []
    flagged_vessels_balanced = []

    for v in ["LAD", "LCX", "RCA"]:
        p = probs[v]
        v_thresh = thresholds[v]["high_sensitivity_threshold"]
        v_nest_sens = thresholds[v].get("nested_cv_operating_metrics", {}).get(
            "sensitivity_pooled", thresholds[v].get("nested_cv_operating_metrics", {}).get("sensitivity_mean", 0.90)
        )
        is_stenotic_bal = p >= 0.50
        is_stenotic_scr = p >= v_thresh
        if is_stenotic_bal:
            any_vessel_balanced_flagged = True
            flagged_vessels_balanced.append(v)
        if is_stenotic_scr:
            any_vessel_screening_flagged = True
            flagged_vessels_screening.append(v)

        vessel_results[v] = {
            "prob": p,
            "similar_cohort": compute_similar_cohort_summary(p, v, ml_state),
            "label": "Stenotic" if is_stenotic_bal else "Normal",
            "high_sens_label": "Stenotic" if is_stenotic_scr else "Normal",
            "threshold": 0.50,
            "high_sensitivity_threshold": v_thresh,
            "nested_sensitivity": v_nest_sens,
        }

    # Logical disjunctive rule: flag overall CAD if P_CAD >= cutoff OR any vessel is flagged
    cad_screening_thresh = min(0.50, cad_high_thresh)
    cad_label = "High Risk" if (coherent_cad_prob >= 0.50 or any_vessel_balanced_flagged) else "Low Risk"
    cad_high_sens_label = "High Risk" if (cad_label == "High Risk" or coherent_cad_prob >= cad_screening_thresh or any_vessel_screening_flagged) else "Low Risk"

    # Reverse contradiction detection: overall CAD flagged, but 0 vessels flagged
    diffuse_or_non_focal_screening = bool(cad_high_sens_label == "High Risk" and len(flagged_vessels_screening) == 0)
    diffuse_or_non_focal_balanced = bool(cad_label == "High Risk" and len(flagged_vessels_balanced) == 0)
    diffuse_cad_message = (
        "Diffuse or Non-Focal CAD Risk: Overall model above cutoff; no vessel model above its cutoff; location not determined."
        if (diffuse_or_non_focal_screening or diffuse_or_non_focal_balanced)
        else None
    )

    # 6. Exact additive SHAP explanations
    explanations: Dict[str, Any] = {}
    if compute_shap:
        for tgt in TARGETS:
            exp = explain_sample(
                pipeline=ml_state["pipelines"][tgt],
                explainer=ml_state["explainers"][tgt],
                parent_mapping=ml_state["parent_mappings"][tgt],
                schema=schema,
                sample_df=sample_df,
                imputed_features=missing_features,
            )
            explanations[tgt] = exp

        # Plain language explanation synthesis
        cath_features = explanations.get("Cath", {}).get("features", [])
        elevating_feats = [f for f in cath_features if f.get("shap", 0) > 0][:2]
        lowering_feats = [f for f in cath_features if f.get("shap", 0) < 0][:2]
        elevating_phrases = [format_feature_phrase(f) for f in elevating_feats]
        lowering_phrases = [format_feature_phrase(f) for f in lowering_feats]

        if elevating_phrases and lowering_phrases:
            plain_language = f"Driven mainly by {', '.join(elevating_phrases)}; partly offset by {' and '.join(lowering_phrases)}."
        elif elevating_phrases:
            plain_language = f"Driven mainly by {', '.join(elevating_phrases)}."
        elif lowering_phrases:
            plain_language = f"Risk lowered by {', '.join(lowering_phrases)}."
        else:
            plain_language = "Risk is consistent with baseline population prevalence."
    else:
        plain_language = "Fast what-if sensitivity screening without full SHAP attribution decomposition."

    cad_nest_sens = thresholds["Cath"].get("nested_cv_operating_metrics", {}).get(
        "sensitivity_pooled", thresholds["Cath"].get("nested_cv_operating_metrics", {}).get("sensitivity_mean", 0.90)
    )

    return {
        "cad": {
            "prob": coherent_cad_prob,
            "raw_prob": raw_cad_prob,
            "displayed_prob": coherent_cad_prob,
            "driving_vessel": driving_vessel,
            "similar_cohort": compute_similar_cohort_summary(coherent_cad_prob, "Cath", ml_state),
            "label": cad_label,
            "high_sens_label": cad_high_sens_label,
            "threshold": 0.50,
            "high_sensitivity_threshold": cad_high_thresh,
            "nested_sensitivity": cad_nest_sens,
            "coherence_adjusted": is_adjusted,
            "flagged_vessels_screening": flagged_vessels_screening,
            "flagged_vessels_balanced": flagged_vessels_balanced,
            "flagged_vessels_count": len(flagged_vessels_screening),
            "diffuse_or_non_focal": diffuse_or_non_focal_screening,
            "diffuse_or_non_focal_screening": diffuse_or_non_focal_screening,
            "diffuse_or_non_focal_balanced": diffuse_or_non_focal_balanced,
            "diffuse_cad_message": diffuse_cad_message,
            "plain_language_summary": plain_language,
        },
        "vessels": vessel_results,
        "explain": explanations,
        "cohort_defaults_applied": cohort_defaults_applied,
        "missing_features": missing_features,
        "imputed_values": imputed_values,
        "completeness": completeness,
        "input_completeness": input_completeness,
        "reliability": input_completeness,
        "warnings": warnings,
        "disclaimer": "Decision support / educational purposes only — not a substitute for formal diagnostic imaging.",
    }
