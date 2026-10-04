# Model Evaluation and Validation Results

Evaluated across 5-fold Cross-Validation with 3 repeats (15 total folds per model).

| Target | Model | ROC-AUC (mean±SD) | PR-AUC | F1-Score | Recall | Specificity | Accuracy | Brier Score |
|---|---|---|---|---|---|---|---|---|
| Cath | Baseline_Majority | 0.500 ± 0.000 | 0.716 | 0.835 | 1.000 | 0.000 | 0.716 | 0.284 |
| Cath | LogisticRegression | 0.929 ± 0.024 | 0.971 | 0.909 | 0.934 | 0.698 | 0.867 | 0.098 |
| Cath | RandomForest | 0.929 ± 0.029 | 0.971 | 0.910 | 0.946 | 0.663 | 0.866 | 0.113 |
| Cath | XGBoost | 0.924 ± 0.024 | 0.970 | 0.910 | 0.929 | 0.713 | 0.868 | 0.101 |
| LAD | Baseline_Majority | 0.500 ± 0.000 | 0.584 | 0.738 | 1.000 | 0.000 | 0.584 | 0.416 |
| LAD | LogisticRegression | 0.829 ± 0.054 | 0.874 | 0.793 | 0.816 | 0.664 | 0.752 | 0.165 |
| LAD | RandomForest | 0.846 ± 0.049 | 0.883 | 0.819 | 0.874 | 0.635 | 0.775 | 0.168 |
| LAD | XGBoost | 0.834 ± 0.047 | 0.873 | 0.813 | 0.844 | 0.675 | 0.773 | 0.164 |
| LCX | Baseline_Majority | 0.500 ± 0.000 | 0.393 | 0.000 | 0.000 | 1.000 | 0.607 | 0.393 |
| LCX | LogisticRegression | 0.704 ± 0.073 | 0.608 | 0.510 | 0.451 | 0.795 | 0.660 | 0.216 |
| LCX | RandomForest | 0.727 ± 0.079 | 0.615 | 0.391 | 0.280 | 0.908 | 0.661 | 0.210 |
| LCX | XGBoost | 0.735 ± 0.060 | 0.615 | 0.541 | 0.482 | 0.806 | 0.679 | 0.204 |
| RCA | Baseline_Majority | 0.500 ± 0.000 | 0.376 | 0.000 | 0.000 | 1.000 | 0.624 | 0.376 |
| RCA | LogisticRegression | 0.733 ± 0.045 | 0.617 | 0.498 | 0.451 | 0.799 | 0.668 | 0.202 |
| RCA | RandomForest | 0.732 ± 0.042 | 0.628 | 0.408 | 0.304 | 0.907 | 0.680 | 0.204 |
| RCA | XGBoost | 0.711 ± 0.057 | 0.609 | 0.517 | 0.450 | 0.833 | 0.689 | 0.208 |
