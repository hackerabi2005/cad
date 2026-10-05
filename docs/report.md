# Cardio3D AI: Cardiovascular Risk & 3D Coronary Stenosis Viewer
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
  - **1 Constant / Zero-Variance Column**: `Exertional CP` (100% `'N'`, offering zero predictive entropy and dropped).
  - **54 Active Input Features**: 5 Demographic, 25 Symptoms/History/Exam, 7 ECG, 14 Laboratory Blood biomarkers, and 3 Echocardiographic metrics.
- **Target Consistency & Row 93 Alignment**:
  In the raw dataset, row index 93 (spreadsheet row 95) contains `LAD='Stenotic'`, `LCX='Normal'`, `RCA='Normal'`, but `Cath='Normal'`. By clinical definition, $\ge 50\%$ stenosis in any major coronary artery constitutes CAD. To match the dataset's own definition (CAD = $\ge 1$ stenotic vessel), row 93's target was aligned to `CAD` in derived data (`Cath_aligned = Cath | LAD | LCX | RCA`), while leaving the raw data file untouched and verified by SHA256. This aligns the released distribution from **216 CAD / 87 Normal $\to$ 217 CAD / 86 Normal**, achieving 100% logical consistency across all 303 rows. The data cannot show which recorded label was erroneous.
- **Leakage Prevention**:
  `Cath`, `LAD`, `LCX`, `RCA`, and `Exertional CP` are strictly excluded from the feature matrix `X`. All imputation (median for continuous, mode for categorical), standard scaling, and one-hot encoding execute strictly inside scikit-learn `Pipeline` objects fitted only on training folds. Label-shuffled cross-validation confirmed random-chance ROC-AUC ($0.50 \pm 0.02$).

---

## 3. Predictive Modeling & Operating Thresholds
Validation was conducted using **Repeated Stratified 5-Fold Cross-Validation with 3 Repeats (15 total folds per candidate model)**. Candidates included Dummy Majority Baseline, L2-Regularized Logistic Regression, Random Forest, and XGBoost.

### Summary Cross-Validation Results at Default Cutoff (0.50)
| Target Vessel / Condition | Selected Model | ROC-AUC (mean±SD) | PR-AUC | F1-Score | Sens. (Recall) | Specificity | PPV | NPV | Brier Score |
|---|---|---|---|---|---|---|---|---|---|
| **Cath (Overall CAD)** | **LogisticRegression** | **0.929 ± 0.024** | 0.971 | 0.909 | 0.934 | 0.698 | 0.887 | 0.812 | 0.098 |
| **LAD (Anterior)** | **RandomForest** | **0.846 ± 0.049** | 0.883 | 0.819 | 0.874 | 0.635 | 0.773 | 0.792 | 0.168 |
| **LCX (Circumflex)** | **XGBoost** | **0.735 ± 0.060** | 0.615 | 0.541 | 0.482 | 0.806 | 0.628 | 0.706 | 0.204 |
| **RCA (Right Coronary)** | **LogisticRegression** | **0.733 ± 0.045** | 0.617 | 0.498 | 0.451 | 0.799 | 0.571 | 0.710 | 0.202 |

*All models passed the strict audit ceiling (no CV ROC-AUC > 0.97).*

### Production Operating Performance: Default Cutoff (0.50) vs. Sensitivity-First Operating Threshold (≥90% Sensitivity)
Operating decision thresholds were tuned on out-of-fold predictions to prioritize screening safety by enforcing $\ge 90\%$ sensitivity:

| Target | Production Model | Cutoff (0.50) Sens. | Cutoff (0.50) Spec. | Cutoff (0.50) PPV | Cutoff (0.50) NPV | Operating Cutoff | Op. Sens. | Op. Spec. | Op. PPV | Op. NPV |
|---|---|---|---|---|---|---|---|---|---|---|
| **Cath** | LogisticRegression | 0.935 | 0.698 | 0.886 | 0.811 | **0.611** | **0.903** | 0.826 | 0.929 | 0.772 |
| **LAD** | RandomForest | 0.876 | 0.635 | 0.771 | 0.784 | **0.469** | **0.904** | 0.603 | 0.762 | 0.817 |
| **LCX** | XGBoost | 0.487 | 0.821 | 0.637 | 0.712 | **0.217** | **0.908** | 0.370 | 0.482 | 0.861 |
| **RCA** | LogisticRegression | 0.439 | 0.804 | 0.575 | 0.704 | **0.213** | **0.903** | 0.392 | 0.472 | 0.871 |

> **Clinical Operating Tradeoff**: Because LCX and RCA targets have moderate discrimination (ROC-AUC $\approx 0.73$), prioritizing $\ge 90\%$ screening sensitivity yields lower specificity ($37.0\%$ for LCX, $39.2\%$ for RCA). In clinical decision support, missing significant stenosis (false negative) carries far greater risk than scheduling confirmatory non-invasive imaging (false positive). Derived from inner out-of-fold tuning, nested cross-validation sensitivity estimates are **$90.2\% \pm 3.9\%$ for Cath**, **$88.2\% \pm 7.2\%$ for LAD**, **$91.6\% \pm 6.3\%$ for LCX**, and **$89.2\% \pm 6.9\%$ for RCA** (detailed in `reports/threshold_nested.md`). LCX labels exhibit lower precision under high-sensitivity tuning, reflecting moderate discrimination and low-confidence single-vessel boundaries.

---

## 4. Logical Risk Coherence Evaluation & Decision
By medical domain logic, CAD is the logical union of stenosis across major coronary branches:
$$P(\text{CAD}) \ge \max(P(\text{LAD}), P(\text{LCX}), P(\text{RCA}))$$

We quantitatively audited out-of-fold cross-validation predictions to verify whether applying this post-processing rule preserves model calibration:
- **Raw CAD Model**: ROC-AUC = 0.9302, Brier score = 0.0967.
- **Coherent CAD ($P_{\text{coherent}} = \max(P_{\text{CAD}}, \max(P_{\text{vessels}}))$)**: ROC-AUC = 0.9291, Brier score = 0.1017.
- **Criterion Evaluation**: AUC drop of $0.0012 \le 0.010$, Brier change of $+0.0050 \le +0.010$. The criteria are satisfied.
- **Policy Decision**: The `max()` coherence rule is retained. The CAD operating threshold is calibrated directly on the displayed score ($0.611$). The API transparently returns both `raw_prob` and `prob`, and discrete label overrides ("any vessel High $\implies$ overall High") are rejected to avoid compounding false positives.

---

## 5. Explainable AI (SHAP) & Feature Attribution
The system delivers exact, instantaneous feature attribution using **TreeSHAP** (for XGBoost and Random Forest) and **LinearSHAP** (for Logistic Regression):
- **Additive Exactness**: For every prediction, the sum of baseline value plus all feature SHAP attributions identically equals the raw model margin score ($\text{error} < 10^{-4}$).
- **Parent Feature Aggregation**: Categorical dummy columns generated by one-hot encoding are aggregated back to their parent physiological measurement via the additive property of Shapley values:
  $$\text{SHAP}(\text{Parent}) = \sum_{d \in \text{Dummies}} \text{SHAP}(d)$$
- **Global Population Hierarchy**: Across the cohort, the strongest global predictors of CAD are **Typical Chest Pain**, **Age**, **Fasting Blood Sugar (FBS)**, **Region RWMA count**, **Ejection Fraction (EF-TTE)**, and **Dyslipidemia (DLP)**.

---

## 6. 3D Anatomical Visualization Pipeline
The 3D interactive viewer is built on **Three.js** and **React Three Fiber (R3F)** using the open-source **BodyParts3D** anatomical model (Branch A: Separate Artery Meshes):
- **Segment Extraction**: High-resolution anatomical meshes for the LAD (9 branches/trunks), LCX (4 branches/trunks), RCA (10 branches/trunks), and Ascending Aorta were combined and transformed into standard Three.js coordinates centered at the origin.
- **Myocardium Shell & Decimation**: The 49 chamber walls and septa were decimated using quadric decimation by 70%, reducing the total triangle count from >160k to **83,600 triangles** (comfortably within the $\le 100,000$ triangle budget).
- **Rendering Performance**: Under pure software rendering (`--disable-gpu` at 1280×720 on 13th Gen Intel Core i5-13420H), the 3D viewer benchmarks at **49.3 ms mean frame time (~20.3 FPS continuous orbit, p95 = 53.7 ms)**, recorded in `reports/fps_benchmark.json`. On standard hardware acceleration, it renders at the display refresh rate (not benchmarked).
- **Dynamic Risk Mapping**: Arteries interpolate across a continuous spectrum (Green $\to$ Lime $\to$ Amber $\to$ Orange $\to$ Red) with floating 3D numeric risk pills and interactive click-to-focus hit meshes.

---

## 7. Architectural Decisions: TimesFM & TabPFN
- **TimesFM Considered & Rejected**: TimesFM 3.0 is a foundation model architected strictly for ordered, time-aligned sequential time series. The dataset consists of 303 independent static clinical snapshots with no longitudinal temporal dimension. Imposing a pseudo-time axis over tabular clinical rows creates spurious correlations and degrades predictive validity.
- **TabPFN Not Evaluated**: TabPFN (v2.5+) imposes non-commercial licensing constraints, requires interactive browser authentication tokens for pre-trained weights, and introduces heavyweight specialized dependencies. Standard tuned tree and regularized linear models achieved high discriminative power (Cath AUC 0.929, LAD AUC 0.846) with sub-millisecond local inference and exact SHAP guarantees.

---

## 8. Limitations & Clinical Safety Disclaimer
- **Cohort Scale & Demographics**: The dataset comprises 303 patients from a single tertiary cardiovascular center in Tehran. Multi-center external validation on racially and geographically diverse cohorts is essential prior to prospective clinical adoption.
- **RWMA Non-Spatial Nature**: While `Region RWMA` (number of abnormal motion regions) is an informative numerical feature, the dataset provides no spatial Cartesian lesion coordinates. RWMA is treated as a clinical biomarker, not an anatomical texture map.
- **Safety Disclaimer**: The UI features an indelible, prominent disclaimer banner and footer stating that predictions are for **decision support and educational purposes only**, and are not a substitute for formal diagnostic angiography.
- **Open-Source Attribution & Licenses**:
  - Code: MIT License (see `LICENSE`).
  - Third-Party Assets & Data: See `THIRD_PARTY_NOTICES.md` for BodyParts3D mesh (CC BY-SA 2.1 JP) and UCI CAD dataset (CC BY 4.0).

