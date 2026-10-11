# Nested Cross-Validation Operating Threshold Evaluation

Inner thresholds are derived strictly from inner out-of-fold predictions 
within each training fold (5-fold inner CV) to eliminate optimistic overfitting bias on tree models.

## 1. Summary of Nested Cross-Validation (Operating Estimates)

| Target | Selected Model | Operating Cutoff | Nested-CV Sens. (Mean ± SD) | Nested-CV Spec. (Mean ± SD) | Pooled Sens. | Pooled Spec. |
|---|---|---|---|---|---|---|
| **Cath** | LogitEnsemble | 0.400 | **0.954 ± 0.030** | 0.648 ± 0.091 | **0.954** | **0.647** |
| **LAD** | LogitEnsemble | 0.420 | **0.885 ± 0.067** | 0.609 ± 0.099 | **0.885** | **0.609** |
| **LCX** | LogitEnsemble | 0.248 | **0.916 ± 0.073** | 0.371 ± 0.108 | **0.916** | **0.371** |
| **RCA** | LogitEnsemble | 0.232 | **0.895 ± 0.071** | 0.367 ± 0.104 | **0.895** | **0.367** |
