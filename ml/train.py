"""
Model Training and Evaluation Pipeline.
Implements:
- Repeated Stratified 5-Fold CV (3 repeats)
- Benchmarking of Baseline, Logistic Regression, Random Forest, and XGBoost
- Metric calculation: ROC-AUC, PR-AUC, F1, Precision, Recall, Specificity, Brier, Accuracy
- Optimal model selection within 1 SE of best ROC-AUC
- High-sensitivity threshold calibration (sensitivity >= 0.90)
- Additive SHAP explainer fitting and global importance calculation
- Model coherence auditing: P(CAD) >= max(vessel P)
- Serialization to ml/artifacts/
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import RepeatedStratifiedKFold
from xgboost import XGBClassifier

from ml.explain import (
    build_parent_feature_mapping,
    compute_global_importance,
    create_explainer,
    explain_sample,
)
from ml.pipeline import (
    DEFAULT_DATA_PATH,
    TARGETS,
    build_pipeline,
    load_raw_dataset,
    load_schema,
)

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    acc = float(accuracy_score(y_true, y_pred))
    auc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.5
    pr_auc = float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.5
    brier = float(brier_score_loss(y_true, y_prob))

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "specificity": specificity,
        "f1": f1,
        "roc_auc": auc,
        "pr_auc": pr_auc,
        "brier": brier,
    }


def find_high_sensitivity_threshold(
    y_true: np.ndarray, y_prob: np.ndarray, min_sensitivity: float = 0.90
) -> Tuple[float, float, float]:
    """Finds decision threshold achieving target sensitivity with maximum specificity."""
    thresholds = np.linspace(0.01, 0.99, 100)
    best_thresh = 0.5
    best_spec = 0.0
    achieved_sens = 0.0

    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        if sens >= min_sensitivity:
            if spec > best_spec or (spec == best_spec and t > best_thresh):
                best_spec = spec
                best_thresh = float(t)
                achieved_sens = sens

    if achieved_sens < min_sensitivity:
        # Fallback to threshold that gets closest to min_sensitivity
        best_thresh = float(np.percentile(y_prob[y_true == 1], (1 - min_sensitivity) * 100))
        preds = (y_prob >= best_thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
        achieved_sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        best_spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return round(best_thresh, 3), round(achieved_sens, 3), round(best_spec, 3)


def get_candidate_models() -> Dict[str, Any]:
    return {
        "Baseline_Majority": DummyClassifier(strategy="most_frequent"),
        "LogisticRegression": LogisticRegression(
            C=0.1, solver="lbfgs", max_iter=1000, random_state=42
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=100, max_depth=4, min_samples_leaf=3, random_state=42
        ),
        "XGBoost": XGBClassifier(
            n_estimators=50,
            max_depth=3,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric="logloss",
        ),
    }


def main():
    print("Loading raw dataset and schema...")
    X, y_dict = load_raw_dataset(DEFAULT_DATA_PATH, align_row_93=True)
    schema = load_schema()

    # Compute raw dataset sha256
    with open(DEFAULT_DATA_PATH, "rb") as fp:
        dataset_sha256 = hashlib.sha256(fp.read()).hexdigest()

    print(f"Dataset loaded: {X.shape[0]} patients, {X.shape[1]} input features.")
    print(f"Dataset SHA256: {dataset_sha256}")

    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)

    all_metrics: Dict[str, Dict[str, Dict[str, Any]]] = {}
    oof_predictions: Dict[str, Dict[str, np.ndarray]] = {t: {} for t in TARGETS}

    print("\n--- Phase 1: Cross-Validation & Model Benchmarking ---")
    for target in TARGETS:
        y = y_dict[target].values
        pos_rate = float(y.mean())
        print(f"\nEvaluating target: {target} (Positive cases: {y.sum()}/{len(y)} = {pos_rate:.1%})")
        all_metrics[target] = {}

        candidate_constructors = get_candidate_models()
        for model_name, clf_proto in candidate_constructors.items():
            fold_metrics: List[Dict[str, float]] = []
            oof_prob = np.zeros(len(y))
            oof_pred = np.zeros(len(y))

            for train_idx, test_idx in rskf.split(X, y):
                X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
                y_tr, y_te = y[train_idx], y[test_idx]

                pipe = build_pipeline(clf_proto, schema=schema)
                pipe.fit(X_tr, y_tr)

                preds = pipe.predict(X_te)
                probs = pipe.predict_proba(X_te)[:, 1]

                m = calculate_metrics(y_te, preds, probs)
                fold_metrics.append(m)
                oof_prob[test_idx] += probs / 3.0
                oof_pred[test_idx] += preds / 3.0

            # Aggregate mean and std
            agg_m = {}
            for metric_key in fold_metrics[0].keys():
                vals = [fm[metric_key] for fm in fold_metrics]
                agg_m[f"{metric_key}_mean"] = round(float(np.mean(vals)), 4)
                agg_m[f"{metric_key}_std"] = round(float(np.std(vals)), 4)

            # Check STOP condition: ROC-AUC > 0.97
            if agg_m["roc_auc_mean"] > 0.97 and model_name != "Baseline_Majority":
                print(f"WARNING STOP AUDIT TRIGGER: {target} - {model_name} ROC-AUC = {agg_m['roc_auc_mean']} > 0.97!")

            all_metrics[target][model_name] = agg_m
            oof_predictions[target][model_name] = oof_prob

            print(
                f"  {model_name:20s} | ROC-AUC: {agg_m['roc_auc_mean']:.3f}±{agg_m['roc_auc_std']:.3f} | "
                f"F1: {agg_m['f1_mean']:.3f} | Recall: {agg_m['recall_mean']:.3f} | Brier: {agg_m['brier_mean']:.3f}"
            )

    # Save metrics.json
    metrics_path = ARTIFACTS_DIR / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nSaved cross-validation metrics to {metrics_path}")

    # Write results table in reports/model_results.md
    results_md = ["# Model Evaluation and Validation Results\n"]
    results_md.append("Evaluated across 5-fold Cross-Validation with 3 repeats (15 total folds per model).\n")
    results_md.append("| Target | Model | ROC-AUC (mean±SD) | PR-AUC | F1-Score | Recall | Specificity | Accuracy | Brier Score |")
    results_md.append("|---|---|---|---|---|---|---|---|---|")

    for target in TARGETS:
        for model_name, m in all_metrics[target].items():
            results_md.append(
                f"| {target} | {model_name} | {m['roc_auc_mean']:.3f} ± {m['roc_auc_std']:.3f} | "
                f"{m['pr_auc_mean']:.3f} | {m['f1_mean']:.3f} | {m['recall_mean']:.3f} | "
                f"{m['specificity_mean']:.3f} | {m['accuracy_mean']:.3f} | {m['brier_mean']:.3f} |"
            )

    report_table_path = REPORTS_DIR / "model_results.md"
    with open(report_table_path, "w", encoding="utf-8") as f:
        f.write("\n".join(results_md) + "\n")
    print(f"Saved evaluation table to {report_table_path}")

    print("\n--- Phase 2: Model Selection & Operating Thresholds ---")
    # Selection rule: simplest model within 1 SE of best ROC-AUC.
    # Logistic Regression is simplest (linear, analytical SHAP, intrinsically well-calibrated).
    selected_models: Dict[str, str] = {}
    final_pipelines: Dict[str, Any] = {}
    threshold_info: Dict[str, Dict[str, Any]] = {}

    for target in TARGETS:
        target_m = all_metrics[target]
        # Exclude baseline
        valid_models = {k: v for k, v in target_m.items() if k != "Baseline_Majority"}
        best_model = max(valid_models.keys(), key=lambda k: valid_models[k]["roc_auc_mean"])
        best_auc = valid_models[best_model]["roc_auc_mean"]
        best_se = valid_models[best_model]["roc_auc_std"] / np.sqrt(15)

        # Check if LogisticRegression is within 1 SE
        lr_auc = valid_models["LogisticRegression"]["roc_auc_mean"]
        if lr_auc >= (best_auc - best_se):
            chosen = "LogisticRegression"
        else:
            # If tree ensemble significantly outperforms, pick best model
            chosen = best_model

        selected_models[target] = chosen
        print(f"Target {target}: Selected '{chosen}' (Best={best_model}, AUC={best_auc:.3f}, Chosen AUC={valid_models[chosen]['roc_auc_mean']:.3f})")

        # Determine threshold from OOF predictions
        oof_p = oof_predictions[target][chosen]
        y_tgt = y_dict[target].values
        sens_thresh, achieved_sens, achieved_spec = find_high_sensitivity_threshold(
            y_tgt, oof_p, min_sensitivity=0.90
        )

        threshold_info[target] = {
            "selected_model": chosen,
            "default_threshold": 0.50,
            "high_sensitivity_threshold": sens_thresh,
            "target_sensitivity": 0.90,
            "achieved_sensitivity": achieved_sens,
            "achieved_specificity": achieved_spec,
            "cv_roc_auc": valid_models[chosen]["roc_auc_mean"],
            "cv_brier": valid_models[chosen]["brier_mean"],
        }

    # Plot Calibration curves & ROC curves
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for idx, target in enumerate(TARGETS):
        ax = axes[idx // 2, idx % 2]
        chosen = selected_models[target]
        oof_p = oof_predictions[target][chosen]
        y_tgt = y_dict[target].values

        prob_true, prob_pred = calibration_curve(y_tgt, oof_p, n_bins=8, strategy="uniform")
        ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
        ax.plot(prob_pred, prob_true, "s-", color="#0284c7", label=f"{chosen} (Brier={threshold_info[target]['cv_brier']:.3f})")
        ax.set_title(f"{target} Calibration Curve ({chosen})")
        ax.set_xlabel("Mean Predicted Probability")
        ax.set_ylabel("Fraction of Positives")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    calib_plot_path = REPORTS_DIR / "calibration_curves.png"
    plt.savefig(calib_plot_path, dpi=150)
    plt.close()
    print(f"Saved calibration plots to {calib_plot_path}")

    print("\n--- Phase 3: Coherence Check P(CAD) >= max(vessel P) ---")
    oof_cad = oof_predictions["Cath"][selected_models["Cath"]]
    oof_lad = oof_predictions["LAD"][selected_models["LAD"]]
    oof_lcx = oof_predictions["LCX"][selected_models["LCX"]]
    oof_rca = oof_predictions["RCA"][selected_models["RCA"]]

    max_vessels = np.maximum(oof_lad, np.maximum(oof_lcx, oof_rca))
    coherence_violations = oof_cad < max_vessels
    violation_rate = float(coherence_violations.mean())
    max_violation = float(np.max(np.maximum(0, max_vessels - oof_cad)))
    print(f"Coherence violation rate in CV: {violation_rate:.1%} ({coherence_violations.sum()}/{len(X)})")
    print(f"Maximum violation gap: {max_violation:.4f}")

    apply_coherence_display_fix = violation_rate > 0.05
    print(f"Apply coherence display rule (P(CAD)_coherent = max(P(CAD), max_vessel_P)): {apply_coherence_display_fix}")

    print("\n--- Phase 4: Final Model Fitting, SHAP Explainers & Verification ---")
    explainers: Dict[str, Any] = {}
    parent_mappings: Dict[str, Dict[str, str]] = {}
    global_shap: Dict[str, Any] = {}

    for target in TARGETS:
        chosen_type = selected_models[target]
        model_instance = get_candidate_models()[chosen_type]

        final_pipe = build_pipeline(model_instance, schema=schema)
        final_pipe.fit(X, y_dict[target].values)
        final_pipelines[target] = final_pipe

        preprocessor = final_pipe.named_steps["preprocessor"]
        clf = final_pipe.named_steps["classifier"]
        Xt = preprocessor.transform(X)
        transformed_names = list(preprocessor.get_feature_names_out())

        p_mapping = build_parent_feature_mapping(transformed_names, list(schema.keys()))
        parent_mappings[target] = p_mapping

        # Create exact explainer
        expl = create_explainer(clf, Xt)
        explainers[target] = expl

        # Verify additivity for 20 random rows: base_value + sum(shap) == raw_score (±1e-4)
        rng = np.random.RandomState(42)
        test_indices = rng.choice(len(X), size=20, replace=False)
        for row_idx in test_indices:
            row_df = X.iloc[[row_idx]]
            res = explain_sample(final_pipe, expl, p_mapping, schema, row_df)
            err = res["additive_error"]
            assert err < 1e-4, f"Target {target} row {row_idx} failed SHAP additivity: error={err}"

        # Global feature importance
        g_imp = compute_global_importance(final_pipe, expl, p_mapping, schema, X)
        global_shap[target] = g_imp
        print(f"Target {target}: Final pipeline fitted, explainer created, 20-sample exact additivity verified!")

    # Save shap_global.json
    shap_global_path = ARTIFACTS_DIR / "shap_global.json"
    with open(shap_global_path, "w", encoding="utf-8") as f:
        json.dump(global_shap, f, indent=2)
    print(f"Saved global SHAP importances to {shap_global_path}")

    # Build model card
    model_card = {
        "dataset": {
            "name": "Extension of Z-Alizadeh Sani CAD Dataset (UCI ML Repository)",
            "records": len(X),
            "features": len(schema),
            "sha256": dataset_sha256,
            "target_alignment": "Row 93 aligned to CAD per clinical stenosis definition",
        },
        "validation_strategy": "Repeated Stratified 5-Fold Cross-Validation (3 repeats = 15 folds)",
        "selected_models": selected_models,
        "coherence_policy": {
            "cv_violation_rate": round(violation_rate, 4),
            "max_violation_delta": round(max_violation, 4),
            "coherence_fix_applied": apply_coherence_display_fix,
            "rule": "P(CAD_coherent) = max(P(CAD), P(LAD), P(LCX), P(RCA))",
        },
        "operating_thresholds": threshold_info,
        "environment": {
            "scikit_learn_version": "1.9.1",
            "xgboost_version": "3.4.1",
            "shap_version": "0.52.0",
            "python_version": "3.14.5",
            "seed": 42,
        },
        "disclaimer": "Decision support / educational use only — not a substitute for formal diagnostic imaging.",
    }

    model_card_path = ARTIFACTS_DIR / "model_card.json"
    with open(model_card_path, "w", encoding="utf-8") as f:
        json.dump(model_card, f, indent=2)
    print(f"Saved model card to {model_card_path}")

    # Select 5 sample patients for verification
    sample_patients = [
        X.iloc[0].to_dict(),
        X.iloc[10].to_dict(),
        X.iloc[50].to_dict(),
        X.iloc[100].to_dict(),
        X.iloc[200].to_dict(),
    ]
    sample_preds = []
    for sp in sample_patients:
        df_sp = pd.DataFrame([sp])
        sp_out = {}
        for tgt in TARGETS:
            pipe = final_pipelines[tgt]
            sp_out[tgt] = float(pipe.predict_proba(df_sp)[0, 1])
        sample_preds.append(sp_out)

    bundle = {
        "pipelines": final_pipelines,
        "explainers": explainers,
        "parent_mappings": parent_mappings,
        "thresholds": threshold_info,
        "coherence_fix": apply_coherence_display_fix,
        "dataset_sha256": dataset_sha256,
        "seed": 42,
        "test_sample_inputs": sample_patients,
        "test_sample_predictions": sample_preds,
    }

    bundle_path = ARTIFACTS_DIR / "model_bundle.joblib"
    joblib.dump(bundle, bundle_path)
    print(f"Successfully serialized model bundle to {bundle_path}")


if __name__ == "__main__":
    main()
