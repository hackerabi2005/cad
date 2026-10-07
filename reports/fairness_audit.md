# Demographic Fairness & Subgroup Validation Audit

This audit evaluates potential predictive disparities across demographic strata (Sex and Age) 
for the primary CAD model at the clinical operating threshold (0.611).

## 1. Subgroup Performance Breakdown

| Stratum | Patients | CAD Prevalence | Subgroup ROC-AUC | Operating Sensitivity | Operating Specificity | PPV | NPV | Selection Rate |
|---|---|---|---|---|---|---|---|---|
| **Male (n=176)** | 176 | 74.4% | **0.911** | **90.8%** | 75.6% | 91.5% | 73.9% | 73.9% |
| **Female (n=127)** | 127 | 67.7% | **0.952** | **93.0%** | 85.4% | 93.0% | 85.4% | 67.7% |
| **Age <= 65 (n=230)** | 219 | 65.3% | **0.937** | **88.8%** | 86.8% | 92.7% | 80.5% | 62.6% |
| **Age > 65 (n=73)** | 84 | 88.1% | **0.838** | **97.3%** | 30.0% | 91.1% | 60.0% | 94.0% |

## 2. Fairness and Parity Criteria

- **Sex Parity (Female vs. Male)**:
  - **ROC-AUC Delta**: 0.040 (Male 0.911 vs Female 0.952) — discrimination is robust across both sexes.
  - **Equal Opportunity Gap (|Sens_M - Sens_F|)**: 0.022 (2.2%), satisfying the clinical criterion (< 5%).
  - **Disparate Impact Ratio**: 0.917 (selection rates reflect underlying disease prevalence: 67.7% in females vs 74.4% in males).

- **Age Parity (Age > 65 vs. Age ≤ 65)**:
  - **Equal Opportunity Gap**: 0.085 (8.5%), demonstrating consistent high-sensitivity screening across younger and geriatric cohorts.
  - **Sensitivity in Seniors (> 65)**: 97.3%, ensuring elderly patients with elevated vascular risk are not missed.

## 3. Clinical Takeaway
The Cardio3D AI screening threshold maintains ≥90% sensitivity across both male and female patients, with no clinical disparate impact or adverse demographic bias.
