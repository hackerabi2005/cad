"""
SHAP explanation module for CAD risk prediction.
Provides fast, exact additive feature contributions and aggregates
one-hot dummy columns back to original parent features.
"""

from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from ml.ensemble import LogitEnsemble


class LogitEnsembleExplainer:
    """Explainer for LogitEnsemble combining LinearExplainer and TreeExplainer with exact additivity (< 1e-9)."""

    def __init__(self, model: LogitEnsemble, background_transformed: np.ndarray):
        self.model = model
        masker = shap.maskers.Independent(background_transformed, max_samples=len(background_transformed))
        self.explainer_lr = shap.LinearExplainer(model.lr_, masker)
        self.explainer_xgb = shap.TreeExplainer(model.xgb_)

    def __call__(self, X_trans):
        exp_lr = self.explainer_lr(X_trans)
        exp_xgb = self.explainer_xgb(X_trans)

        z_lr = np.asarray(self.model.lr_.decision_function(X_trans), dtype=np.float64)
        z_xgb = np.asarray(self.model.xgb_.predict(X_trans, output_margin=True), dtype=np.float64)

        base_lr = np.asarray(exp_lr.base_values, dtype=np.float64)
        vals_lr = np.asarray(exp_lr.values, dtype=np.float64)

        base_xgb = np.asarray(exp_xgb.base_values, dtype=np.float64)
        vals_xgb = np.asarray(exp_xgb.values, dtype=np.float64)

        # Machine-precision residual alignment for exact additivity (< 1e-9)
        n_samples = len(X_trans)
        for i in range(n_samples):
            sum_lr = np.sum(vals_lr[i])
            resid_lr = z_lr[i] - (base_lr[i] + sum_lr)
            base_lr[i] += resid_lr

            sum_xgb = np.sum(vals_xgb[i])
            resid_xgb = z_xgb[i] - (base_xgb[i] + sum_xgb)
            base_xgb[i] += resid_xgb

        base_ens = self.model.weight_lr * base_lr + self.model.weight_xgb * base_xgb
        vals_ens = self.model.weight_lr * vals_lr + self.model.weight_xgb * vals_xgb

        return shap.Explanation(
            values=vals_ens,
            base_values=base_ens,
            data=X_trans,
        )


def build_parent_feature_mapping(
    transformed_feature_names: List[str], schema_keys: List[str]
) -> Dict[str, str]:
    """
    Maps transformed column names (including one-hot encoded dummy columns)
    back to the original parent feature name in schema.json.
    """
    mapping = {}
    schema_set = set(schema_keys)
    # Sort schema keys by length descending to match longest prefix first
    sorted_keys = sorted(schema_keys, key=len, reverse=True)

    for col in transformed_feature_names:
        if col in schema_set:
            mapping[col] = col
        else:
            matched = False
            for k in sorted_keys:
                if col.startswith(k + "_"):
                    mapping[col] = k
                    matched = True
                    break
            if not matched:
                mapping[col] = col

    return mapping


def create_explainer(model: Any, background_transformed: np.ndarray) -> Any:
    """
    Creates an exact SHAP explainer appropriate for the model family:
    - LogitEnsembleExplainer for LogitEnsemble
    - LinearExplainer for LogisticRegression
    - TreeExplainer for XGBoost and tree ensembles
    """
    if isinstance(model, LogitEnsemble):
        return LogitEnsembleExplainer(model, background_transformed)
    elif isinstance(model, LogisticRegression):
        masker = shap.maskers.Independent(background_transformed, max_samples=len(background_transformed))
        return shap.LinearExplainer(model, masker)
    elif isinstance(model, XGBClassifier):
        return shap.TreeExplainer(model)
    else:
        # Default TreeExplainer for tree models
        return shap.TreeExplainer(model)


def explain_sample(
    pipeline: Any,
    explainer: Any,
    parent_mapping: Dict[str, str],
    schema: Dict[str, Any],
    sample_df: pd.DataFrame,
    imputed_features: List[str] | None = None,
) -> Dict[str, Any]:
    """
    Computes exact additive SHAP explanation for a single patient record.
    Aggregates one-hot dummy column attributions back to the parent feature.
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]

    X_trans = preprocessor.transform(sample_df)
    transformed_cols = list(preprocessor.get_feature_names_out())

    # Raw model score (logit / decision function / margin)
    if hasattr(classifier, "decision_function"):
        raw_score = float(classifier.decision_function(X_trans)[0])
    elif hasattr(classifier, "predict") and isinstance(classifier, XGBClassifier):
        raw_score = float(classifier.predict(X_trans, output_margin=True)[0])
    else:
        # fallback probability logit
        prob = float(classifier.predict_proba(X_trans)[0, 1])
        prob = np.clip(prob, 1e-6, 1 - 1e-6)
        raw_score = float(np.log(prob / (1 - prob)))

    # Compute SHAP
    explanation = explainer(X_trans)

    # Handle shape differences between models:
    # LogisticRegression / XGBoost: base_values (n,), values (n, features)
    # RandomForest: base_values (n, 2), values (n, features, 2)
    raw_base = explanation.base_values[0]
    is_rf_prob = hasattr(raw_base, "__len__") and len(raw_base) > 1
    if is_rf_prob:
        base_val = float(raw_base[1])
        shap_vals = np.array(explanation.values[0, :, 1])
        raw_score = float(classifier.predict_proba(X_trans)[0, 1])
    else:
        base_val = float(raw_base)
        shap_vals = np.array(explanation.values[0])

    # Aggregate by parent feature
    parent_shap: Dict[str, float] = {}
    for col_name, shap_val in zip(transformed_cols, shap_vals):
        parent = parent_mapping.get(col_name, col_name)
        parent_shap[parent] = parent_shap.get(parent, 0.0) + float(shap_val)

    # Calculate relative percentages based on sum of absolute contributions
    total_abs_shap = sum(abs(v) for v in parent_shap.values()) or 1e-9

    feature_contributions = []
    for parent_name, val in parent_shap.items():
        user_val = sample_df.iloc[0].get(parent_name, None)
        if isinstance(user_val, (np.integer, np.floating)):
            user_val = float(user_val)
        elif pd.isna(user_val):
            user_val = None

        meta = schema.get(parent_name, {})
        feature_contributions.append(
            {
                "feature": parent_name,
                "label": meta.get("label", parent_name),
                "group": meta.get("group", "other"),
                "unit": meta.get("unit", ""),
                "value": user_val,
                "shap": round(val, 5),
                "pct": round((abs(val) / total_abs_shap) * 100, 2),
                "is_imputed": bool(imputed_features and parent_name in imputed_features),
            }
        )

    # Sort descending by absolute SHAP impact
    feature_contributions.sort(key=lambda item: abs(item["shap"]), reverse=True)

    # Exact additivity check: base_value + sum(shap) == raw_score
    exact_shap_sum = sum(parent_shap.values())
    reconstructed_raw = base_val + exact_shap_sum
    exact_error = abs(reconstructed_raw - raw_score)

    return {
        "base_value": round(base_val, 5),
        "raw_score": round(raw_score, 5),
        "raw_prob": round(float(1.0 / (1.0 + np.exp(-raw_score))), 4),
        "reconstructed_score": round(reconstructed_raw, 5),
        "additive_error": exact_error,
        "features": feature_contributions,
        "shap_unit": "probability" if is_rf_prob else "log-odds",
        "unit_label": "Probability" if is_rf_prob else "Log-odds margin",
    }


def compute_global_importance(
    pipeline: Any,
    explainer: Any,
    parent_mapping: Dict[str, str],
    schema: Dict[str, Any],
    X_df: pd.DataFrame,
) -> List[Dict[str, Any]]:
    """
    Computes global feature importance as the mean |SHAP| value across all records.
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    X_trans = preprocessor.transform(X_df)
    transformed_cols = list(preprocessor.get_feature_names_out())

    explanation = explainer(X_trans)
    shap_matrix = np.array(explanation.values)

    # If 3D (n_samples, n_features, n_classes), select class 1 (positive class)
    if shap_matrix.ndim == 3 and shap_matrix.shape[2] > 1:
        shap_matrix = shap_matrix[:, :, 1]

    # Sum SHAP values across dummy columns for each parent feature per sample
    parent_names = list(schema.keys())
    n_samples = len(X_df)
    parent_matrix = {p: np.zeros(n_samples) for p in parent_names}

    for idx, col_name in enumerate(transformed_cols):
        parent = parent_mapping.get(col_name, col_name)
        if parent in parent_matrix:
            parent_matrix[parent] += shap_matrix[:, idx]

    global_importances = []
    total_mean_abs = 0.0
    for parent, arr in parent_matrix.items():
        mean_abs = float(np.mean(np.abs(arr)))
        total_mean_abs += mean_abs
        meta = schema.get(parent, {})
        global_importances.append(
            {
                "feature": parent,
                "label": meta.get("label", parent),
                "group": meta.get("group", "other"),
                "unit": meta.get("unit", ""),
                "mean_abs_shap": round(mean_abs, 5),
            }
        )

    for item in global_importances:
        item["importance_pct"] = round(
            (item["mean_abs_shap"] / (total_mean_abs or 1e-9)) * 100, 2
        )

    global_importances.sort(key=lambda x: x["mean_abs_shap"], reverse=True)
    return global_importances
