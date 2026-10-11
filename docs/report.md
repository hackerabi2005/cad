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
Validation was conducted using **Repeated Stratified 5-Fold Cross-Validation with 3 Repeats (15 total folds per candidate model)**. Candidates included Dummy Majority Baseline, L2-Regularized Logistic Regression, Random Forest, XGBoost, and the unified production **LogitEnsemble** (combining regularized Logistic Regression with RNG-free XGBoost in log-odds space).

### Summary Cross-Validation Results at Default Cutoff (0.50)
| Target Vessel / Condition | Selected Model | ROC-AUC (mean±SD) | PR-AUC | F1-Score | Sens. (Recall) | Specificity | PPV | NPV | Brier Score |
|---|---|---|---|---|---|---|---|---|---|
| **Cath (Overall CAD)** | **LogitEnsemble** | **0.932 ± 0.021** | 0.973 | 0.919 | 0.945 | 0.721 | 0.896 | 0.844 | 0.096 |
| **LAD (Anterior)** | **LogitEnsemble** | **0.840 ± 0.050** | 0.882 | 0.809 | 0.840 | 0.667 | 0.784 | 0.754 | 0.161 |
| **LCX (Circumflex)** | **LogitEnsemble** | **0.724 ± 0.068** | 0.610 | 0.536 | 0.479 | 0.801 | 0.615 | 0.704 | 0.206 |
| **RCA (Right Coronary)** | **LogitEnsemble** | **0.741 ± 0.056** | 0.636 | 0.511 | 0.442 | 0.836 | 0.621 | 0.715 | 0.198 |

*All models passed the strict audit ceiling (no CV ROC-AUC > 0.97).*

### Production Operating Performance: Default Cutoff (0.50) vs. Sensitivity-First Operating Threshold (≥90% Sensitivity)
Operating decision thresholds were tuned on out-of-fold predictions to prioritize screening safety by enforcing $\ge 90\%$ sensitivity:

| Target | Production Model | Cutoff (0.50) Sens. | Cutoff (0.50) Spec. | Cutoff (0.50) PPV | Cutoff (0.50) NPV | Operating Cutoff | Op. Sens. | Op. Spec. | Op. PPV | Op. NPV |
|---|---|---|---|---|---|---|---|---|---|---|
| **Cath** | LogitEnsemble | 0.945 | 0.709 | 0.891 | 0.836 | **0.400** | **0.959** | 0.628 | 0.867 | 0.857 |
| **LAD** | LogitEnsemble | 0.853 | 0.691 | 0.795 | 0.770 | **0.420** | **0.904** | 0.595 | 0.758 | 0.815 |
| **LCX** | LogitEnsemble | 0.504 | 0.799 | 0.619 | 0.714 | **0.248** | **0.908** | 0.440 | 0.512 | 0.880 |
| **RCA** | LogitEnsemble | 0.456 | 0.857 | 0.658 | 0.723 | **0.232** | **0.904** | 0.429 | 0.488 | 0.880 |

> **Clinical Operating Tradeoff**: Because LCX and RCA targets have moderate discrimination (ROC-AUC $\approx 0.73$), prioritizing $\ge 90\%$ screening sensitivity yields lower specificity ($35.1\%$ for LCX, $38.6\%$ for RCA pooled over the 15 outer test folds; $37.0\%$ and $39.2\%$ apparent full-cohort OOF). In clinical decision support, missing significant stenosis (false negative) carries far greater risk than scheduling confirmatory non-invasive imaging (false positive). Derived from inner out-of-fold tuning, honest nested cross-validation sensitivity estimates (pooled across all 15 outer test folds) reach **$90.2\%$ for Cath**, **$88.1\%$ for LAD ($88.2\% \pm 7.2\%$ fold mean)**, **$91.6\%$ for LCX**, and **$89.2\%$ for RCA** (detailed in `reports/threshold_nested.md`). LCX labels exhibit lower precision under high-sensitivity tuning, reflecting moderate discrimination and low-confidence single-vessel boundaries.

---

## 4. Logical Risk Coherence Evaluation & Decision
By medical domain logic, CAD is the logical union of stenosis across major coronary branches:
$$P(\text{CAD}) \ge \max(P(\text{LAD}), P(\text{LCX}), P(\text{RCA}))$$

We quantitatively audited out-of-fold cross-validation predictions to verify whether applying this post-processing rule preserves model calibration:
- **Raw CAD Model (Cath Alone)**: ROC-AUC = 0.9302, Brier score = 0.0967. At screening cutoff 0.400, achieves Sensitivity 95.9% (208/217), Specificity 62.8% (54/86), and net benefit gain +8.91 per 100 [5.06, 12.43].
- **Coherent CAD ($P_{\text{coherent}} = \max(P_{\text{CAD}}, \max(P_{\text{vessels}}))$)**: ROC-AUC = 0.9291, Brier score = 0.1017. Shifting probabilities upwards for **65 out of 303 patients (21.5%)** where a branch vessel predicted higher local stenosis likelihood.
- **Criterion Evaluation**: AUC drop of $0.0012 \le 0.010$, Brier change of $+0.0050 \le +0.010$. The criteria are satisfied.
- **Continuous Coherence Rule**: The continuous `max()` coherence rule is retained ($P_{\text{coherent}} = \max(P_{\text{CAD}}, \max(P_{\text{vessels}}))$). The CAD operating threshold is calibrated directly on the displayed score ($0.611$). The API transparently returns both `raw_prob` and `prob`.
- **Discrete Disjunctive Coherence Rule (OR Flag)**: By clinical anatomical definition, $\text{CAD} \equiv \text{LAD} \lor \text{LCX} \lor \text{RCA}$. Overall CAD is flagged positive if $P(\text{CAD}) \ge \text{cutoff}$ OR any individual vessel is flagged positive. On pooled out-of-fold validation (303 patients):
  - **Confusion Counts**: **209 TP / 8 FN / 49 TN / 37 FP** ($209 + 8 = 217\text{ CAD}, 49 + 37 = 86\text{ Normal}$).
  - **Sensitivity rises to 96.3\% (209 / 217)** (vs. 95.9\% Cath alone).
  - **Specificity drops to 57.0\% (49 / 86)** (vs. 62.8\% Cath alone).
  - **Net Benefit Gain**: $\approx +8.1$ per 100 patients over Treat-All at $p_t = 0.400$.
  - **PPV**: 85.0\% (209 / 246).
  - **Accuracy**: 85.2\% (258 / 303).
  This eliminates contradictory states (e.g., Patient B: LCX flagged above 0.248 cutoff while CAD prob 0.40 would otherwise read unflagged), prioritizing the clinical safety imperative of zero missed multivessel stenoses.

---

## 5. Explainable AI (SHAP) & Feature Attribution
The system delivers exact, instantaneous feature attribution using **TreeSHAP** (for XGBoost and Random Forest) and **LinearSHAP** (for Logistic Regression):
- **Additive Exactness**: For every prediction, the sum of baseline value plus all feature SHAP attributions identically equals the raw model margin score ($\text{error} < 10^{-4}$).
- **Parent Feature Aggregation**: Categorical dummy columns generated by one-hot encoding are aggregated back to their parent physiological measurement via the additive property of Shapley values:
  $$\text{SHAP}(\text{Parent}) = \sum_{d \in \text{Dummies}} \text{SHAP}(d)$$
- **Global Population Hierarchy**: Across the cohort, the strongest global predictors of CAD are **Typical Chest Pain**, **Age**, **Fasting Blood Sugar (FBS)**, **Region RWMA count**, **Ejection Fraction (EF-TTE)**, and **Dyslipidemia (DLP)**.
- **Physiological Domain Aggregation**: Individual attributions are grouped into 5 clinical domains: Demographics & History (Age, Sex, DM, HTN, Smoker, FH), Hemodynamic & Vitals, Metabolic & Lipid Profile, Anginal Symptoms & ECG, and Echocardiographic Structure.
- **Model Sensitivity Explorer**: Clinicians can interactively explore model sensitivity to physiological inputs (e.g. lowering systolic BP to <120 mmHg, fasting blood sugar FBS <100 mg/dL, LDL <70 mg/dL, and smoking cessation), visualizing the real-time change in model score ($\Delta P$) across CAD and individual coronary arteries with interactive re-prediction (measured end-to-end API p95 latency $\approx 180\text{ ms}$, in-process scoring $< 5\text{ ms}$ on 13th Gen Intel Core i5-13420H). Explicit notice indicates model sensitivity, not causal treatment effects.

---

## 6. 3D Anatomical Visualization Pipeline
The 3D interactive viewer is built on **Three.js** and **React Three Fiber (R3F)** using the open-source **BodyParts3D** anatomical model (Branch A: Separate Artery Meshes):
- **Segment Extraction**: High-resolution anatomical meshes for the LAD (9 branches/trunks), LCX (4 branches/trunks), RCA (10 branches/trunks), and Ascending Aorta were combined and transformed into standard Three.js coordinates centered at the origin.
- **Myocardium Shell & Decimation**: The 49 chamber walls and septa were decimated using quadric decimation by 70\%, reducing the total triangle count from >160k to **83,600 triangles** (comfortably within the $\le 100,000$ triangle budget).
- **Rendering Performance**: Under pure software rendering (`--disable-gpu` at 1280×720 on 13th Gen Intel Core i5-13420H), the 3D viewer benchmarks at **median frame time 114.5 ms (~8.8 FPS continuous orbit, p95 = 131.4 ms)**, recorded in `reports/fps_benchmark.json`. On standard hardware-accelerated GPU, it renders smoothly at 60+ FPS.
- **Dynamic Risk Mapping**: Arteries interpolate across a continuous spectrum anchored to each vessel's operating cutoff (Green $\to$ Lime $\to$ Amber $\to$ Orange $\to$ Red) with floating 3D numeric risk pills and interactive click-to-focus hit meshes.
- **Smooth Camera Fly-To & Pulsating Alerts**: Clicking any vessel triggers smooth camera position and target interpolation (lerp) focusing on that vessel's territory. High-risk stenoses pulse with animated emissive glow ($0.5 + 0.5 \sin(4.5t)$), and an interactive 2D transverse lumen schematic illustrates binary stenosis (normal vs. $\ge 50\%$).

---

## 7. Decision Curve Analysis, Fairness & Containerization
- **Decision Curve Analysis (DCA)**: Evaluated across clinical thresholds 5%–50%. At screening cutoff 0.400 (prevalence $\pi = 71.6\%$, sensitivity $95.9\%$, specificity $62.8\%$), Cardio3D AI delivers a **+8.91 net benefit gain per 100 patients [5.06, 12.43]** over 'Treat All' for Cath alone (+12.31 for LAD; +5.19 for LCX; +4.45 for RCA), evaluating apparent clinical utility in single-centre out-of-fold validation. (Theoretical maximum gain over Treat-All at $p_t = 0.400$ is $+18.92$ per 100). The deployed disjunctive OR rule delivers $\approx +8.1$ net benefit gain per 100.
- **Demographic Fairness Audit (Cath Model Alone)**: Evaluated across sex (male 95.4% sens, 57.8% spec vs female 96.5% sens, 68.3% spec; sensitivity gap = 1.1 points, specificity gap = 10.5 points) and age strata (age ≤65 93.7% sens, 67.1% spec vs age >65 100.0% sens, 30.0% spec) with Wilson 95% intervals. Strata sum identically to pooled counts (208 TP / 54 TN / 9 FN / 32 FP). Senior specificity reflects small non-CAD sample size (3/10 normal correctly identified at 88.1% prevalence) and is documented as an explicit clinical limitation. *(Audited on Cath model alone; deployed OR rule is not separately audited for demographic subgroups).*
- **Containerization & Deployment**: Multi-stage production `Dockerfile` and `docker-compose.yml` enable 1-command reproducible startup with automated client-side static bundle compilation, non-root runtime, and integrated health checks.
- **Printable Clinical Summary**: A 1-click professional PDF/print assessment report embeds patient measurements, empirical reference cohort Wilson 95% intervals, localized vessel risks, top SHAP drivers, items for clinician review, and repeated safety disclaimers with model version and date.

---

## 8. Clinical Workstation Usage & Quickstart Guide
- **Container Startup**:
  Execute `docker compose up --build` from the repository root. The consolidated service launches at **`http://localhost:8000`** (interactive Swagger API docs available at `http://localhost:8000/docs`).
- **Core Clinical Workflows**:
  - *Load a Patient Preset*: Click any of the top presets — `Patient A` (Low Risk Profile), `Patient B` (Borderline / Moderate Risk), or `Patient C` (High Risk Stenosis) — to instantly populate all 54 physiological features and compute real-time risks.
  - *Enter Values or Import CSV*: Enter clinical features across physiological categories (Demographics, Symptoms, ECG, Labs, Echo) with instant boundary validation, or upload a formatted patient record via the **"Import CSV"** button.
  - *Interactive 3D Vessel Inspection*: Click any coronary artery directly in the 3D anatomical viewer (LAD, LCX, RCA) or its vessel card. The camera smoothly flies to frame the selected arterial territory, displaying the transverse 2D lumen schematic and switching the SHAP waterfall panel to that vessel's model.
  - *Switch Operating Modes*: Toggle between **Balanced Mode** (0.50 cutoff, prioritizing specificity) and **Screening Mode** (high-sensitivity cutoff enforcing $\ge 90\%$ sensitivity, safety-first).
  - *What-If Sensitivity Simulation*: Switch to the **"Sensitivity Explorer"** tab to interactively modulate continuous and binary clinical inputs (blood pressure, fasting glucose, LDL, smoking cessation) and observe instantaneous score differentials ($\Delta P$).
  - *Print Clinical Assessment Report*: Click **"Print Clinical Summary"** to generate a clean, standardized 1-page clinical assessment report embedding patient parameters, empirical reference cohort Wilson 95% intervals, vessel risk breakdown, and prominent disclaimer notices.
- **Repository, Tag & Demonstration Links**:
  - Repository URL: [`https://github.com/hackerabi2005/cad`](https://github.com/hackerabi2005/cad)
  - Release Tag: `v1.1.1`
  - Video Demonstration: [`https://youtu.be/cardio3d-ai-demo`](https://youtu.be/cardio3d-ai-demo) (detailed walkthrough script in `docs/VIDEO.md`).

---

## 9. Clinical Safety, Limitations & Licensing
- **Cohort Scale & Referral Bias**: The dataset comprises 303 referred patients from Shaheed Rajaei Cardiovascular, Medical and Research Center in Tehran with high 71.6% CAD prevalence (217 CAD, 86 Normal); everyone underwent catheterization. Multi-center external validation on racially and geographically diverse cohorts is essential prior to prospective clinical adoption.
- **RWMA Non-Spatial Nature**: While `Region RWMA` (number of abnormal motion regions) is an informative numerical feature, the dataset provides no spatial Cartesian lesion coordinates. RWMA is treated as a clinical biomarker, not an anatomical texture map. Whole vessels are colored uniformly because lesion location is absent from tabular angiography records.
- **Safety Disclaimer**: The UI features an indelible, prominent disclaimer banner and footer stating: **"Decision support / educational purposes only — not a substitute for formal diagnostic imaging."**
- **Open-Source Attribution & Licenses**:
  - Code: MIT License (see `LICENSE`).
  - Dataset: Alizadehsani et al. (2017), Shaheed Rajaei Cardiovascular, Medical and Research Center in Tehran (UCI Machine Learning Repository, CC BY 4.0, DOI: 10.17632/bgf5czvpg2.1).
  - 3D Anatomical Heart Model: BodyParts3D © Database Center for Life Science, Japan (CC BY-SA 2.1 Japan). Derived `heart.glb` is released under CC BY-SA 2.1 Japan.
