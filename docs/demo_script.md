# Cardio3D AI: Demonstration Video Script & Shot List
**Multimodal AI Hackathon 2026 — Track A (Cardiovascular Risk Visualization & Prediction)**
*Estimated Video Runtime: 7 Minutes 15 Seconds (Target Budget: 3–10 Minutes)*

---

### Shot 1: Title & Clinical Safety Disclaimer (0:00 – 0:45)
- **Visual**: Screen capture of browser opening `http://localhost:8000/`. Highlight the amber **Clinical Safety Disclaimer Banner** at the top.
- **Narrator**:
  > "Welcome to this demonstration of Cardio3D AI, built for the Multimodal AI Hackathon 2026 Track A. Before we begin, please note the persistent clinical safety disclaimer displayed across the system: this software is designed strictly for clinical decision support and educational exploration. It is not an automated diagnostic tool or a substitute for certified coronary angiography.
  >
  > Cardiovascular disease remains the leading cause of death globally. While statistical algorithms can output risk percentages, numbers alone cannot convey *where* and *how* critical stenosis is developing. Today, we demonstrate a unified system that bridges rigorous machine learning with real-time 3D anatomical visualization."

---

### Shot 2: Dataset, Preprocessing & Zero Leakage (0:45 – 1:45)
- **Visual**: Switch to terminal or code view showing `ml/schema.json` and `ml/pipeline.py`.
- **Narrator**:
  > "Our modeling pipeline is built on the UCI Extension of the Z-Alizadeh Sani CAD dataset containing 303 patient records.
  >
  > We performed a thorough data audit: while the raw spreadsheet lists 59 columns, published literature cites 54 features. We reconciled this difference: 4 columns are target labels (`Cath`, `LAD`, `LCX`, `RCA`), 1 column (`Exertional CP`) has zero variance and was dropped, leaving exactly 54 active physiological predictors.
  >
  > Critically, we enforce zero target leakage: all four target columns are strictly quarantined from the input matrix `X`. All imputation, standard scaling, and one-hot encoding execute strictly inside scikit-learn Pipeline objects. Under label-shuffled cross-validation, our ROC-AUC drops to exactly 0.50, proving zero leakage."

---

### Shot 3: Interactive 3D Anatomical Viewer (1:45 – 3:15)
- **Visual**: Switch to the live 3D Canvas. Rotate, zoom in on the anterior interventricular groove (LAD), pan around the circumflex branch (LCX), and right coronary artery (RCA). Toggle the Myocardium opacity slider.
- **Narrator**:
  > "Here is our 3D anatomical viewer, rendered client-side using Three.js and React Three Fiber.
  >
  > We utilized segmented 3D meshes from BodyParts3D, extracting the exact anatomical branches for the Left Anterior Descending artery (LAD), Left Circumflex (LCX), Right Coronary Artery (RCA), and Ascending Aorta.
  >
  > By applying quadric decimation to the ventricular myocardium shell, we optimized the entire asset down to 83,600 triangles—well under our 100,000 triangle budget. Under pure CPU software rendering with GPU disabled, the viewer benchmarks at 49.3 ms mean frame time (≈ 20.3 FPS); the GPU path is not benchmarked.
  >
  > Notice the dynamic risk color-coding: each artery interpolates across a continuous green-to-red spectrum according to predicted stenosis probability, accompanied by 3D floating numeric risk badges."

---

### Shot 4: Real Patient Presets & Live Debounced Prediction (3:15 – 4:30)
- **Visual**: Click "Patient A — Low Risk Profile" preset. Show the heart turning green and overall risk dropping to 3%. Then click "Patient C — High Risk Stenosis" preset. Show the LAD and overall gauge turning crimson within 50 milliseconds.
- **Narrator**:
  > "Let's test the interactive clinical workflow. The dashboard includes presets from actual patients in the cohort.
  >
  > When we load Patient A—a low-risk profile—our model calculates a 3% CAD probability; the 3D heart vessels remain emerald green.
  >
  > Now, let's load Patient C—a high-risk patient with typical chest pain, elevated fasting blood sugar, and echocardiographic wall motion abnormalities. Within 50 milliseconds, our debounced prediction updates the dashboard: overall CAD risk rises to 99%, the LAD stenosis probability spikes to 92%, and the anterior artery pulses in high-risk crimson."

---

### Shot 5: Exact Additive SHAP Explainability (4:30 – 5:45)
- **Visual**: Click on the LAD card or 3D badge. Show the SHAP waterfall panel switching to LAD. Hover over the red bars and switch to the Table View.
- **Narrator**:
  > "A black-box prediction is unacceptable in medicine. Cardio3D AI integrates exact additive SHAP explanations.
  >
  > Clicking the LAD artery immediately re-indexes the explainability panel to the vessel-specific LAD model.
  >
  > In the waterfall chart, red bars indicate features pushing risk upward—here, typical anginal chest pain (+0.38) and regional wall motion abnormalities (+0.24). Green bars show protective factors.
  >
  > Crucially, our explainer maintains exact mathematical additivity: the base value plus the sum of all SHAP values equals the raw score within a numerical tolerance of 10^-5. Dummy variables from one-hot encoding are aggregated back into their parent physiological measurements so clinicians see clean, interpretable clinical biomarkers."

---

### Shot 6: Rigorous Cross-Validation & Coherence Policy (5:45 – 6:30)
- **Visual**: Click the "Model CV Validation" tab. Show the 15-fold cross-validation table and metrics.
- **Narrator**:
  > "Under our 'Model CV Validation' tab, we display full transparency into our 5-fold cross-validation with 3 repeats—15 folds total.
  >
  > Our selected models achieve 0.929 ROC-AUC for overall CAD and 0.846 for LAD stenosis. To minimize dangerous false negatives in screening, we calibrated high-sensitivity operating cutoffs: 0.611 for Cath, 0.469 for LAD, 0.217 for LCX, and 0.213 for RCA.
  >
  > Under rigorous nested cross-validation where thresholds are tuned strictly out-of-fold, honest sensitivity reaches 90.2% for Cath, 88.1% for LAD, 91.6% for LCX, and 89.2% for RCA (pooled over the 15 outer test folds). We are completely transparent about the tradeoff: catching 90%+ of branch disease means LCX and RCA nested specificity drop to 35.1% and 38.6% (pooled over the 15 outer test folds; 37.0% and 39.2% apparent OOF)—an intentional clinical screening-style stance.
  >
  > Furthermore, our architecture audits logical coherence: because CAD is the union of individual vessels, P(CAD) must be at least the maximum vessel risk. Our system logs this check and enforces coherent risk display."

---

### Shot 7: Extensibility & Reproducibility (6:30 – 7:15)
- **Visual**: Show `web/src/vessels.json` and terminal running `pytest`.
- **Narrator**:
  > "Cardio3D AI is built for extensibility. Adding a new clinical feature or vessel requires only updating the declarative `schema.json` and `vessels.json` registries—no UI redesign required.
  >
  > The entire project is reproducible with standard commands: `python -m ml.train` (which runs in ~2.5–3 minutes on CPU), `pytest`, and `npm run dev`.
  >
  > Thank you for reviewing Cardio3D AI, bridging statistical machine learning and 3D human anatomy for enhanced cardiovascular decision support."
