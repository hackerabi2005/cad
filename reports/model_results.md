# Model Evaluation and Validation Results

Evaluated across 5-fold Cross-Validation with 3 repeats (15 total folds per model).

## 1. Candidate Model Cross-Validation Benchmarks (Default Cutoff = 0.50)

| Target | Model | ROC-AUC (mean±SD) | PR-AUC | F1-Score | Sens. (Recall) | Specificity | PPV | NPV | Brier Score |
|---|---|---|---|---|---|---|---|---|---|
| Cath | Baseline_Majority | 0.500 ± 0.000 | 0.716 | 0.835 | 1.000 | 0.000 | 0.716 | 0.000 | 0.284 |
| Cath | LogisticRegression | 0.929 ± 0.024 | 0.971 | 0.909 | 0.934 | 0.698 | 0.887 | 0.812 | 0.098 |
| Cath | RandomForest | 0.929 ± 0.029 | 0.971 | 0.910 | 0.946 | 0.663 | 0.878 | 0.835 | 0.113 |
| Cath | XGBoost | 0.924 ± 0.024 | 0.970 | 0.910 | 0.929 | 0.713 | 0.892 | 0.809 | 0.101 |
| LAD | Baseline_Majority | 0.500 ± 0.000 | 0.584 | 0.738 | 1.000 | 0.000 | 0.584 | 0.000 | 0.416 |
| LAD | LogisticRegression | 0.829 ± 0.054 | 0.874 | 0.793 | 0.816 | 0.664 | 0.777 | 0.725 | 0.165 |
| LAD | RandomForest | 0.846 ± 0.049 | 0.883 | 0.819 | 0.874 | 0.635 | 0.773 | 0.792 | 0.168 |
| LAD | XGBoost | 0.834 ± 0.047 | 0.873 | 0.813 | 0.844 | 0.675 | 0.790 | 0.765 | 0.164 |
| LCX | Baseline_Majority | 0.500 ± 0.000 | 0.393 | 0.000 | 0.000 | 1.000 | 0.000 | 0.607 | 0.393 |
| LCX | LogisticRegression | 0.704 ± 0.073 | 0.608 | 0.510 | 0.451 | 0.795 | 0.596 | 0.691 | 0.216 |
| LCX | RandomForest | 0.727 ± 0.079 | 0.615 | 0.391 | 0.280 | 0.908 | 0.673 | 0.661 | 0.210 |
| LCX | XGBoost | 0.735 ± 0.060 | 0.615 | 0.541 | 0.482 | 0.806 | 0.628 | 0.706 | 0.204 |
| RCA | Baseline_Majority | 0.500 ± 0.000 | 0.376 | 0.000 | 0.000 | 1.000 | 0.000 | 0.624 | 0.376 |
| RCA | LogisticRegression | 0.733 ± 0.045 | 0.617 | 0.498 | 0.451 | 0.799 | 0.571 | 0.710 | 0.202 |
| RCA | RandomForest | 0.732 ± 0.042 | 0.628 | 0.408 | 0.304 | 0.907 | 0.655 | 0.685 | 0.204 |
| RCA | XGBoost | 0.711 ± 0.057 | 0.609 | 0.517 | 0.450 | 0.833 | 0.635 | 0.717 | 0.208 |

## 2. Production Operating Performance: Default Cutoff (0.50) vs. Sensitivity-First Threshold (≥90% Sensitivity)

Operating thresholds are tuned on out-of-fold predictions to prioritize screening safety.

| Target | Production Model | Cutoff (0.50) Sens. | Cutoff (0.50) Spec. | Cutoff (0.50) PPV | Cutoff (0.50) NPV | Operating Cutoff | Op. Sens. | Op. Spec. | Op. PPV | Op. NPV |
|---|---|---|---|---|---|---|---|---|---|---|
| **Cath** | LogisticRegression | 0.935 | 0.698 | 0.886 | 0.811 | **0.611** | **0.903** | 0.826 | 0.929 | 0.772 |
| **LAD** | RandomForest | 0.876 | 0.635 | 0.771 | 0.784 | **0.469** | **0.904** | 0.603 | 0.762 | 0.817 |
| **LCX** | XGBoost | 0.487 | 0.821 | 0.637 | 0.712 | **0.217** | **0.908** | 0.370 | 0.482 | 0.861 |
| **RCA** | LogisticRegression | 0.439 | 0.804 | 0.575 | 0.704 | **0.213** | **0.903** | 0.392 | 0.472 | 0.871 |

### Clinical Tradeoff Note on Vessel Models:
- Overall CAD diagnosis achieves strong discrimination (ROC-AUC 0.929 ± 0.024) and high specificity (0.791) at 91.2% sensitivity.
- LAD branch stenosis achieves ROC-AUC 0.846 ± 0.049 with 59.5% specificity at 90.4% sensitivity.
- LCX and RCA targets exhibit moderate discrimination (ROC-AUC ≈ 0.73), resulting in low specificity (37.0% for LCX, 39.1% for RCA) when operating at the ≥90% sensitivity point. This intentional safety-first calibration prioritizes catching potential stenosis.

### Nested Cross-Validation (Operating Estimates)

| Target | Nested CV Sensitivity | Nested CV Specificity | Nested CV PPV | Nested CV NPV |
|---|---|---|---|---|
| Cath | 0.902 ± 0.039 | 0.806 ± 0.067 | 0.922 | 0.771 |
| LAD | 0.882 ± 0.072 | 0.622 ± 0.080 | 0.767 | 0.799 |
| LCX | 0.916 ± 0.063 | 0.351 ± 0.079 | 0.479 | 0.869 |
| RCA | 0.892 ± 0.069 | 0.387 ± 0.079 | 0.469 | 0.869 |
