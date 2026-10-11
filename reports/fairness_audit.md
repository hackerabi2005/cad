# Demographic Fairness & Subgroup Validation Audit

This audit evaluates predictive characteristics across demographic strata (Sex and Age) 
for the primary CAD model across both operating modes (Screening 0.400 and Balanced 0.500). 
All metrics are calculated on out-of-fold predictions with Wilson 95% confidence intervals.

## 1. Screening Mode Subgroup Performance (Cutoff 0.400 / High Sensitivity)

| Stratum | n | TP | FN | TN | FP | Prevalence | ROC-AUC | Sensitivity (95% Wilson) | Specificity (95% Wilson) | PPV | NPV |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **Male (n=176)** | 176 | 125 | 6 | 26 | 19 | 74.4% | 0.912 | 95.4% [90.4%–97.9%] | 57.8% [43.3%–71.0%] | 86.8% | 81.2% |
| **Female (n=127)** | 127 | 83 | 3 | 28 | 13 | 67.7% | 0.952 | 96.5% [90.2%–98.8%] | 68.3% [53.0%–80.4%] | 86.5% | 90.3% |
| **Age ≤ 65 (n=219)** | 219 | 134 | 9 | 51 | 25 | 65.3% | 0.936 | 93.7% [88.5%–96.7%] | 67.1% [55.9%–76.6%] | 84.3% | 85.0% |
| **Age > 65 (n=84)** | 84 | 74 | 0 | 3 | 7 | 88.1% | 0.826 | 100.0% [95.1%–100.0%] | 30.0% [10.8%–60.3%] | 91.4% | 100.0% |
| **Pooled Cohort** | 303 | 208 | 9 | 54 | 32 | 71.6% | 0.930 | 95.9% [92.3%–97.8%] | 62.8% [52.2%–72.2%] | 86.7% | 85.7% |

## 2. Balanced Mode Subgroup Performance (Cutoff 0.500)

| Stratum | n | TP | FN | TN | FP | Prevalence | ROC-AUC | Sensitivity (95% Wilson) | Specificity (95% Wilson) | PPV | NPV |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **Male (n=176)** | 176 | 122 | 9 | 31 | 14 | 74.4% | 0.912 | 93.1% [87.5%–96.3%] | 68.9% [54.3%–80.5%] | 89.7% | 77.5% |
| **Female (n=127)** | 127 | 83 | 3 | 30 | 11 | 67.7% | 0.952 | 96.5% [90.2%–98.8%] | 73.2% [58.1%–84.3%] | 88.3% | 90.9% |
| **Age ≤ 65 (n=219)** | 219 | 132 | 11 | 58 | 18 | 65.3% | 0.936 | 92.3% [86.8%–95.7%] | 76.3% [65.6%–84.5%] | 88.0% | 84.1% |
| **Age > 65 (n=84)** | 84 | 73 | 1 | 3 | 7 | 88.1% | 0.826 | 98.6% [92.7%–99.8%] | 30.0% [10.8%–60.3%] | 91.2% | 75.0% |
| **Pooled Cohort** | 303 | 205 | 12 | 61 | 25 | 71.6% | 0.930 | 94.5% [90.6%–96.8%] | 70.9% [60.6%–79.5%] | 89.1% | 83.6% |

## 3. Subgroup Comparisons & Parity

- **Sex Comparison (Female vs. Male)**:
  - **Sample Size**: Male n=176 (131 CAD / 45 normal), Female n=127 (86 CAD / 41 normal).
  - **Screening Sensitivity**: Male 95.4% (95% CI: 90.4%–97.9%) vs Female 96.5% (95% CI: 90.2%–98.8%), gap = 1.1%.
  - **Screening Specificity**: Male 57.8% (95% CI: 43.3%–71.0%) vs Female 68.3% (95% CI: 53.0%–80.4%).
  - **ROC-AUC**: Male 0.912 vs Female 0.952 (delta = 0.040).
  - **Disparate Impact Ratio**: 0.924 (selection rates reflect underlying disease prevalence: 67.7% vs 74.4%).

- **Age Comparison (Age > 65 vs. Age ≤ 65)**:
  - **Sample Size**: Age ≤ 65 n=219 (143 CAD / 76 normal), Age > 65 n=84 (74 CAD / 10 normal).
  - **Screening Sensitivity**: Age > 65 is 100.0% (95% CI: 95.1%–100.0%) vs Age ≤ 65 is 93.7% (95% CI: 88.5%–96.7%), gap = 6.3%.
  - **Screening Specificity**: Age > 65 is 30.0% (95% CI: 10.8%–60.3%) vs Age ≤ 65 is 67.1% (95% CI: 55.9%–76.6%).

## 4. Clinical Takeaway & Documented Limitations
- **Safety Sensitivity**: Screening operating point prioritizes safety-first sensitivity (>93%) across both male and female groups with low sex disparity (gap = 1.1%).
- **Known Limitation (Senior Specificity)**: In seniors aged > 65, specificity is markedly reduced (30.0%, 95% CI: 10.8%–60.3%, 3/10 normal patients identified) compared to younger adults (67.1% at screening, 76.3% balanced). This reflects severe referral bias and cohort imbalance in elderly patients (88.1% CAD prevalence, only 10 normal patients). Clinicians must note that positive flags in patients over 65 carry an elevated false-alarm rate.
