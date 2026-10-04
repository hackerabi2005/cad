# Cardio3D AI: Multimodal Cardiovascular Risk & 3D Coronary Stenosis Viewer
**Multimodal AI Hackathon 2026 — Track A (Cardiovascular Risk Visualization & Prediction)**

> **Clinical Safety Disclaimer**: This software is designed exclusively for decision support and educational exploration. It is **not** a substitute for certified coronary angiography, formal imaging, or physician diagnostic judgment.

---

## 🌟 Highlights
- **Predictive ML Pipeline**: Multi-task classification predicting overall CAD status ($\text{ROC-AUC} = 0.929$) alongside target stenosis for the **LAD** ($\text{AUC} = 0.846$), **LCX** ($\text{AUC} = 0.735$), and **RCA** ($\text{AUC} = 0.733$) with zero target leakage.
- **Interactive 3D Heart Anatomy**: Real-time WebGL anatomical heart rendered with Three.js & React Three Fiber using BodyParts3D segment meshes (83,600 triangles total, running at **>60 FPS** without a discrete GPU).
- **Exact Additive SHAP Explanations**: Instantaneous feature attribution with mathematical additivity ($\text{error} < 10^{-5}$) mapping dummy encoded columns back to parent physiological biomarkers.
- **Logical Risk Coherence**: Automatically enforces $P(\text{CAD}) \ge \max(P(\text{LAD}), P(\text{LCX}), P(\text{RCA}))$ while transparently presenting both raw and coherent scores.
- **Sensitivity-First Calibration**: Calibrated decision thresholds ensuring $\ge 90\%$ sensitivity to prioritize clinical safety.

---

## 🚀 Quickstart (5 Commands to Running App)

```bash
# 1. Install ML & backend dependencies
pip install -r ml/requirements.txt

# 2. Build the web frontend
cd web && npm install && npm run build && cd ..

# 3. Assemble and optimize 3D heart GLB asset
python scripts/build_heart_glb.py

# 4. Train models & generate SHAP artifacts (takes < 30 seconds on CPU)
python -m ml.train

# 5. Launch the unified application server
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```
Open **[http://localhost:8000](http://localhost:8000)** in your browser.

---

## 🧪 Verification & Testing

Run the automated test suite verifying leakage prevention, contract validation, and UI performance:

```bash
# 1. Verify zero target leakage and chance-level shuffled label baseline
pytest ml/tests/test_leakage.py -v

# 2. Verify API contract and CPU latency benchmark (p95 < 300ms)
pytest api/tests/test_api.py -v

# 3. Verify end-to-end browser interactions and software rendering FPS
python scripts/test_ui_playwright.py

# 4. Compile technical report PDF and verify page count (<= 6 pages)
python scripts/generate_pdf_report.py
```

---

## 📊 Model Evaluation Summary

Evaluated across **Repeated Stratified 5-Fold Cross-Validation with 3 Repeats (15 total folds per model)**:

| Target | Production Model | CV ROC-AUC | PR-AUC | F1-Score | Recall (Sens.) | Specificity | Brier Score |
|---|---|---|---|---|---|---|---|
| **Cath (Overall CAD)** | **LogisticRegression** | **0.929 ± 0.024** | 0.965 | 0.909 | 0.934 | 0.701 | 0.098 |
| **LAD (Anterior)** | **RandomForest** | **0.846 ± 0.049** | 0.884 | 0.819 | 0.874 | 0.638 | 0.168 |
| **LCX (Circumflex)** | **XGBoost** | **0.735 ± 0.060** | 0.652 | 0.541 | 0.482 | 0.803 | 0.204 |
| **RCA (Right Coronary)** | **LogisticRegression** | **0.733 ± 0.045** | 0.655 | 0.498 | 0.451 | 0.804 | 0.202 |

---

## 🫀 3D Anatomical Heart Model

- **Source**: BodyParts3D (Database Center for Life Science, Japan).
- **Extracted Nodes**:
  - `LAD`: Trunk of anterior interventricular branch + diagonal & septal branches (13,574 triangles).
  - `LCX`: Trunk of circumflex branch + posterior ventricular branches (19,160 triangles).
  - `RCA`: Trunk of right coronary artery + marginal, conus, AV nodal branches (13,764 triangles).
  - `Aorta`: Ascending aorta and bulb (6,572 triangles).
  - `Heart_Muscle`: Decimated myocardium shell (30,530 triangles).
- **Total Asset Size**: 1.67 MB GLB, 83,600 triangles (budget $\le 100,000$).
- **Software Rendering Benchmark**: 16.5 ms frame time (~60.6 FPS) under `--disable-gpu`.

---

## 🧩 Architectural Extensibility

Adding a new clinical biomarker or vessel requires zero frontend redesign:
- **`ml/schema.json`**: Feature registry with units, encoding, ranges, and groups.
- **`web/src/vessels.json`**: Declarative mapping of anatomical mesh nodes, territories, 3D badge coordinates, and risk color scales.

---

## 📜 Licenses & Citations
- **Dataset**: UCI Machine Learning Repository — *Extension of Z-Alizadeh Sani CAD Dataset* (CC BY 4.0).
- **3D Anatomical Mesh**: BodyParts3D, © The Database Center for Life Science (CC BY-SA 2.1 JP).
- **Code**: MIT License.
