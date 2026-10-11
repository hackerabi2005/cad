# Model Selection & Architectural Rationale

## 1. Candidate Model Benchmarking (Repeated 5-Fold CV, 15 Total Folds)

| Target | Model | CV ROC-AUC | PR-AUC | F1-Score | Sensitivity | Specificity | Brier Score |
|---|---|---|---|---|---|---|---|
| Cath | Baseline_Majority | 0.500 ± 0.000 | 0.716 | 0.835 | 1.000 | 0.000 | 0.284 |
| Cath | LogisticRegression | 0.929 ± 0.024 | 0.971 | 0.909 | 0.934 | 0.698 | 0.098 |
| Cath | RandomForest | 0.929 ± 0.029 | 0.971 | 0.910 | 0.946 | 0.663 | 0.113 |
| Cath | XGBoost | 0.912 ± 0.029 | 0.965 | 0.903 | 0.923 | 0.694 | 0.107 |
| Cath | LogitEnsemble | 0.932 ± 0.021 | 0.973 | 0.919 | 0.945 | 0.721 | 0.096 |
| LAD | Baseline_Majority | 0.500 ± 0.000 | 0.584 | 0.738 | 1.000 | 0.000 | 0.416 |
| LAD | LogisticRegression | 0.829 ± 0.054 | 0.874 | 0.793 | 0.816 | 0.664 | 0.165 |
| LAD | RandomForest | 0.846 ± 0.049 | 0.883 | 0.819 | 0.874 | 0.635 | 0.168 |
| LAD | XGBoost | 0.821 ± 0.049 | 0.861 | 0.796 | 0.821 | 0.664 | 0.169 |
| LAD | LogitEnsemble | 0.840 ± 0.050 | 0.882 | 0.809 | 0.840 | 0.667 | 0.161 |
| LCX | Baseline_Majority | 0.500 ± 0.000 | 0.393 | 0.000 | 0.000 | 1.000 | 0.393 |
| LCX | LogisticRegression | 0.704 ± 0.073 | 0.608 | 0.510 | 0.451 | 0.795 | 0.216 |
| LCX | RandomForest | 0.727 ± 0.079 | 0.615 | 0.391 | 0.280 | 0.908 | 0.210 |
| LCX | XGBoost | 0.718 ± 0.071 | 0.600 | 0.526 | 0.468 | 0.799 | 0.209 |
| LCX | LogitEnsemble | 0.724 ± 0.068 | 0.610 | 0.536 | 0.479 | 0.801 | 0.206 |
| RCA | Baseline_Majority | 0.500 ± 0.000 | 0.376 | 0.000 | 0.000 | 1.000 | 0.376 |
| RCA | LogisticRegression | 0.733 ± 0.045 | 0.617 | 0.498 | 0.451 | 0.799 | 0.202 |
| RCA | RandomForest | 0.732 ± 0.042 | 0.628 | 0.408 | 0.304 | 0.907 | 0.204 |
| RCA | XGBoost | 0.710 ± 0.061 | 0.617 | 0.512 | 0.436 | 0.845 | 0.207 |
| RCA | LogitEnsemble | 0.741 ± 0.056 | 0.636 | 0.510 | 0.442 | 0.836 | 0.198 |

## 2. Decision: LogitEnsemble Architecture
- **Unified Model Family**: Every target (Cath, LAD, LCX, RCA) uses the identical `LogitEnsemble` combining Logistic Regression and RNG-free XGBoost in log-odds margin space.
- **Cross-Platform Determinism**: By setting `subsample=1.0, colsample_bytree=1.0, colsample_bylevel=1.0, colsample_bynode=1.0, n_jobs=1`, XGBoost operates with zero stochasticity, guaranteeing bit-identical predictions across Linux and Windows.
- **Exact Additive SHAP (< 1e-9)**: Because log-odds margins are linear under addition, LinearExplainer (for LR) and TreeExplainer (for XGBoost) combine with mathematical precision. Residual alignment guarantees exact log-odds additivity (< 1e-9 error).
- **Clinical Balance**: The ensemble pairs the robust linear regularization of L2 Logistic Regression with XGBoost's nonlinear boundary detection.
