# Cardio3D AI: Multimodal Cardiovascular Risk & 3D Coronary Stenosis Viewer
**Multimodal AI Hackathon 2026 — Track A Technical Report**

---

## 1. Executive Summary & Clinical Context
Coronary Artery Disease (CAD) remains the leading contributor to cardiovascular mortality worldwide. Standard risk scores (e.g., Framingham, ASCVD) estimate population-level statistical risk but provide no anatomical localization of impending vascular catastrophe. Conversely, invasive coronary angiography reveals precise lesion anatomy but carries procedural risk and expense.

**Cardio3D AI** bridges this critical gap. It combines rigorous machine learning predicting both overall CAD status and vessel-specific stenosis (LAD, LCX, RCA) with a responsive, client-side 3D interactive anatomical heart model. Clinicians and patients can rotate, zoom, and inspect localized risk, backed by exact additive SHAP (SHapley Additive exPlanations) attributions translating physiological features into intuitive visual explanations.

---

## 2. Dataset Preprocessing & Zero-Leakage Architecture
The system is developed on the **UCI Extension of Z-Alizadeh Sani CAD Dataset** (303 patient records, 59 columns):
- **Accounting for 59 Columns vs. 54 Published Features**:
  - **4 Target Columns**: `Cath` (Overall CAD status), `LAD`, `LCX`, `RCA` (stenosis $\ge 50\%$).
  - **1 Constant / Zero-Variance Column**: `Exertional CP` (100% `'N'`, offering zero predictive entropy).
  - **54 Active Input Features**: 5 Demographic, 25 Symptoms/History/Exam, 7 ECG, 14 Laboratory Blood biomarkers, and 3 Echocardiographic metrics.
- **Target Consistency & Row 93 Audit**:
  In the raw dataset, row 93 contains `LAD='Stenotic'`, `LCX='Normal'`, `RCA='Normal'`, but `Cath='Normal'`. By clinical definition, $\ge 50\%$ stenosis in any major coronary artery constitutes CAD. To ensure complete target consistency without modifying raw source files, row 93's target was aligned to `CAD` in the derived training pipeline, achieving 100% consistency (`Cath == (LAD | LCX | RCA)` with 0 mismatches across 303 patients).
- **Leakage Prevention**:
  `Cath`, `LAD`, `LCX`, `RCA`, and `Exertional CP` are strictly excluded from the feature matrix `X`. All imputation (median for continuous, mode for categorical), standard scaling, and one-hot encoding execute strictly inside scikit-learn `Pipeline` objects fitted only on training folds. Label-shuffled cross-validation confirmed random-chance ROC-AUC ($0.50 \pm 0.02$).

---

## 3. Predictive Modeling & Evaluation Methodology
Validation was conducted using **Repeated Stratified 5-Fold Cross-Validation with 3 Repeats (15 total folds per candidate model)**. Candidates included Dummy Majority Baseline, L2-Regularized Logistic Regression, Random Forest, and XGBoost.

### Summary Cross-Validation Results (Mean ± SD)
| Target Vessel / Condition | Selected Model | ROC-AUC | PR-AUC | F1-Score | Recall (Sens.) | Specificity | Brier Score |
|---|---|---|---|---|---|---|---|
| **Cath (Overall CAD)** | **LogisticRegression** | **0.929 ± 0.024** | 0.965 | 0.909 | 0.934 | 0.701 | 0.098 |
| **LAD (Anterior)** | **RandomForest** | **0.846 ± 0.049** | 0.884 | 0.819 | 0.874 | 0.638 | 0.168 |
| **LCX (Circumflex)** | **XGBoost** | **0.735 ± 0.060** | 0.652 | 0.541 | 0.482 | 0.803 | 0.204 |
| **RCA (Right Coronary)** | **LogisticRegression** | **0.733 ± 0.045** | 0.655 | 0.498 | 0.451 | 0.804 | 0.202 |

*Note: All models passed the strict audit ceiling (no CV ROC-AUC > 0.97).*

### Decision Thresholds & Sensitivity-First Calibration
In cardiac clinical screening, false negatives are catastrophic. Operating points were determined on out-of-fold cross-validation predictions under a sensitivity-first objective ($\ge 0.90$ sensitivity):
- **Cath**: High-sensitivity threshold = **0.380** (Achieved Sensitivity: 96.8%, Specificity: 62.8%).
- **LAD**: High-sensitivity threshold = **0.420** (Achieved Sensitivity: 92.1%, Specificity: 57.1%).
- **LCX**: High-sensitivity threshold = **0.250** (Achieved Sensitivity: 90.8%, Specificity: 45.1%).
- **RCA**: High-sensitivity threshold = **0.260** (Achieved Sensitivity: 90.4%, Specificity: 44.4%).

---

## 4. Logical Risk Coherence Policy
By medical logic, CAD is the logical union of stenosis across the major vessels ($P(\text{CAD}) \ge \max(P(\text{LAD}), P(\text{LCX}), P(\text{RCA}))$).
- In out-of-fold CV predictions, independent target models exhibited a **21.5% coherence violation rate** (maximum violation delta: 0.288).
- Because the violation rate exceeded the 5% threshold, the system enforces runtime coherence:
  $$P(\text{CAD})_{\text{coherent}} = \max\left(P(\text{CAD})_{\text{raw}}, \max(P(\text{LAD}), P(\text{LCX}), P(\text{RCA}))\right)$$
Both raw and coherent scores are transparently displayed in the clinician dashboard.

---

## 5. Explainable AI (SHAP) & Feature Attribution
The system delivers exact, instantaneous feature attribution using **TreeSHAP** (for XGBoost and Random Forest) and **LinearSHAP** (for Logistic Regression):
- **Additive Exactness**: For every prediction, the sum of baseline value plus all feature SHAP attributions identically equals the raw model margin score ($\text{error} < 10^{-5}$).
- **Parent Feature Aggregation**: Categorical dummy columns generated by one-hot encoding are aggregated back to their parent physiological measurement via the additive property of Shapley values:
  $$\text{SHAP}(\text{Parent}) = \sum_{d \in \text{Dummies}} \text{SHAP}(d)$$
- **Global Population Hierarchy**: Across the cohort, the strongest global predictors of CAD are **Typical Chest Pain**, **Age**, **Fasting Blood Sugar (FBS)**, **Region RWMA count**, **Ejection Fraction (EF-TTE)**, and **Dyslipidemia (DLP)**.

---

## 6. 3D Anatomical Visualization Pipeline
The 3D interactive viewer is built on **Three.js** and **React Three Fiber (R3F)** using the open-source **BodyParts3D** anatomical model (Branch A: Separate Artery Meshes):
- **Segment Extraction**: High-resolution anatomical meshes for the LAD (9 branches/trunks), LCX (4 branches/trunks), RCA (10 branches/trunks), and Ascending Aorta were combined and transformed into standard Three.js coordinates centered at the origin.
- **Myocardium Shell & Decimation**: The 49 chamber walls and septa were decimated using quadric decimation by 70%, reducing the total triangle count from >160k to **83,600 triangles** (comfortably within the $\le 100,000$ triangle budget).
- **Rendering Performance**: Under pure software rendering (`--disable-gpu`), the 3D viewer maintains a **median frame time of 16.5 ms (~60.6 FPS)**, well above the 20 FPS minimum budget.
- **Dynamic Risk Mapping**: Arteries interpolate across a continuous spectrum (Green $\to$ Lime $\to$ Amber $\to$ Orange $\to$ Red) with floating 3D numeric risk pills and interactive click-to-focus hit meshes.

---

## 7. Architectural Decisions: TimesFM & TabPFN
- **TimesFM Considered & Rejected**: TimesFM 3.0 is a foundation model architected strictly for ordered, time-aligned sequential time series. The dataset consists of 303 independent static clinical snapshots with no longitudinal temporal dimension. Imposing a pseudo-time axis over tabular clinical rows creates spurious correlations and degrades predictive validity.
- **TabPFN Consideration**: TabPFN requires non-standard runtime dependencies and authentication tokens unsuitable for zero-friction deployment. Standard tuned tree and linear models achieved high discriminative power (Cath AUC 0.929, LAD AUC 0.846) with deterministic sub-millisecond inference and exact SHAP guarantees.

---

## 8. Limitations & Clinical Safety Disclaimer
- **Cohort Scale & Demographics**: The dataset comprises 303 patients from a single tertiary cardiovascular center in Tehran. Multi-center external validation on racially and geographically diverse cohorts is essential prior to prospective clinical adoption.
- **RWMA Non-Spatial Nature**: While `Region RWMA` (number of abnormal motion regions) is an informative numerical feature, the dataset provides no spatial Cartesian lesion coordinates. RWMA is treated as a clinical biomarker, not an anatomical texture map.
- **Safety Disclaimer**: The UI features an indelible, prominent disclaimer banner and footer stating that predictions are for **decision support and educational purposes only**, and are not a substitute for formal diagnostic angiography.
- **Open-Source Attribution**: Dataset licensed under **CC BY 4.0**; BodyParts3D anatomical assets licensed under **CC BY-SA 2.1 Japan** (© DBCLS).
