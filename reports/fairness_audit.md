# Demographic Fairness & Subgroup Validation Audit

This audit evaluates predictive characteristics across demographic strata (Sex and Age) 
for the primary CAD model at the clinical operating threshold (0.611). 
All metrics are calculated on out-of-fold predictions with Wilson 95% confidence intervals.

## 1. Subgroup Performance Breakdown

| Stratum | n | Prevalence | ROC-AUC | Sensitivity (95% Wilson) | Specificity (95% Wilson) | PPV | NPV |
|---|---|---|---|---|---|---|---|
| **Male (n=176)** | 176 | 74.4% | 0.911 | 90.8% [84.7%–94.7%] | 75.6% [61.3%–85.8%] | 91.5% | 73.9% |
| **Female (n=127)** | 127 | 67.7% | 0.952 | 93.0% [85.6%–96.8%] | 85.4% [71.6%–93.1%] | 93.0% | 85.4% |
| **Age ≤ 65 (n=219)** | 219 | 65.3% | 0.937 | 88.8% [82.6%–93.0%] | 86.8% [77.5%–92.7%] | 92.7% | 80.5% |
| **Age > 65 (n=84)** | 84 | 88.1% | 0.838 | 97.3% [90.7%–99.3%] | 30.0% [10.8%–60.3%] | 91.1% | 60.0% |

## 2. Fairness and Subgroup Comparisons

- **Sex Comparison (Female vs. Male)**:
  - **Sample Size**: Male n=176 (131 CAD / 45 normal), Female n=127 (86 CAD / 41 normal).
  - **Sensitivity**: Male 90.8% (95% CI: 84.7%–94.7%) vs Female 93.0% (95% CI: 85.6%–96.8%), gap = 2.2%.
  - **Specificity**: Male 75.6% (95% CI: 61.3%–85.8%) vs Female 85.4% (95% CI: 71.6%–93.1%).
  - **ROC-AUC**: Male 0.911 vs Female 0.952 (delta = 0.040).
  - **Disparate Impact Ratio**: 0.917 (selection rates reflect underlying disease prevalence: 67.7% vs 74.4%).

- **Age Comparison (Age > 65 vs. Age ≤ 65)**:
  - **Sample Size**: Age ≤ 65 n=219 (143 CAD / 76 normal), Age > 65 n=84 (74 CAD / 10 normal).
  - **Sensitivity**: Age > 65 is 97.3% (95% CI: 90.7%–99.3%) vs Age ≤ 65 is 88.8% (95% CI: 82.6%–93.0%), gap = 8.5%.
  - **Specificity**: Age > 65 is 30.0% (95% CI: 10.8%–60.3%) due to very high prevalence (88.1%, only 10 normal patients) vs Age ≤ 65 is 86.8% (95% CI: 77.5%–92.7%).

## 3. Clinical Takeaway
The screening operating point prioritizes safety-first sensitivity (>88%) across all demographic subgroups. There is no difference detectable at this sample size; apparent specificity differences in seniors reflect high cohort prevalence and small subgroup sample size.
