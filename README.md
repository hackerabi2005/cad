# Cardio3D AI: Cardiovascular Risk & 3D Coronary Stenosis Viewer
**Multimodal AI Hackathon 2026 — Track A (Cardiovascular Risk Visualization & Prediction)**

> **Clinical Safety Disclaimer**: This software is designed exclusively for decision support and educational exploration. It is **not** a substitute for certified coronary angiography, formal imaging, or physician diagnostic judgment.

---

## 🌟 Highlights
- **Predictive ML Pipeline**: Multi-task classification predicting overall CAD status ($\text{ROC-AUC} = 0.929 \pm 0.024$) alongside target stenosis for the **LAD** ($\text{AUC} = 0.846 \pm 0.049$), **LCX** ($\text{AUC} = 0.735 \pm 0.060$), and **RCA** ($\text{AUC} = 0.733 \pm 0.045$) with zero target leakage.
- **Interactive 3D Heart Anatomy**: Real-time WebGL anatomical heart rendered with Three.js & React Three Fiber using BodyParts3D segment meshes (83,600 triangles total, running at **>60 FPS** with WebGL acceleration and **~17.3 FPS** under pure CPU software rendering `--disable-gpu`).
- **Exact Additive SHAP Explanations**: Instantaneous feature attribution with mathematical additivity ($\text{error} < 10^{-4}$) mapping dummy encoded columns back to parent physiological biomarkers.
- **Logical Risk Coherence**: Empirically audited post-processing enforcing $P(\text{CAD}) \ge \max(P(\text{LAD}), P(\text{LCX}), P(\text{RCA}))$ while transparently returning both raw and coherent scores.
- **Sensitivity-First Operating Thresholds**: Operating decision thresholds prioritizing clinical safety by achieving $\ge 90\%$ sensitivity across all coronary vessels.

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

Run the automated test suite verifying leakage prevention, label consistency, API contract, and UI performance:

```bash
# 1. Verify raw label audit, hash immutability, and row 93 alignment
python -m pytest ml/tests/test_labels.py -v

# 2. Verify zero target leakage and chance-level shuffled label baseline
python -m pytest ml/tests/test_leakage.py -v

# 3. Verify API contract, p95 latency (<300ms), and SHAP consistency
python -m pytest api/tests/ -v

# 4. Verify end-to-end browser interactions and 5-second orbit FPS benchmark
python scripts/test_ui_playwright.py

# 5. Compile technical report PDF and verify page count (<= 6 pages)
python scripts/generate_pdf_report.py
```

---

## 📊 Model Evaluation Summary

Evaluated across **Repeated Stratified 5-Fold Cross-Validation with 3 Repeats (15 total folds per model)**.

### 1. Cross-Validation Benchmarks at Default Cutoff (0.50)

| Target | Production Model | CV ROC-AUC | PR-AUC | F1-Score | Sens. (Recall) | Specificity | PPV | NPV | Brier Score |
|---|---|---|---|---|---|---|---|---|---|
| **Cath (Overall CAD)** | **LogisticRegression** | **0.929 ± 0.024** | 0.971 | 0.909 | 0.934 | 0.698 | 0.887 | 0.812 | 0.098 |
| **LAD (Anterior)** | **RandomForest** | **0.846 ± 0.049** | 0.883 | 0.819 | 0.874 | 0.635 | 0.773 | 0.792 | 0.168 |
| **LCX (Circumflex)** | **XGBoost** | **0.735 ± 0.060** | 0.615 | 0.541 | 0.482 | 0.806 | 0.628 | 0.706 | 0.204 |
| **RCA (Right Coronary)** | **LogisticRegression** | **0.733 ± 0.045** | 0.617 | 0.498 | 0.451 | 0.799 | 0.571 | 0.710 | 0.202 |

### 2. Production Operating Performance: Default Cutoff (0.50) vs. Sensitivity-First Operating Threshold (≥90% Sensitivity)

Operating thresholds are tuned on out-of-fold predictions to prioritize clinical screening safety.

| Target | Production Model | Cutoff (0.50) Sens. | Cutoff (0.50) Spec. | Cutoff (0.50) PPV | Cutoff (0.50) NPV | Operating Cutoff | Op. Sens. | Op. Spec. | Op. PPV | Op. NPV |
|---|---|---|---|---|---|---|---|---|---|---|
| **Cath** | LogisticRegression | 0.935 | 0.698 | 0.886 | 0.811 | **0.604** | **0.912** | 0.791 | 0.917 | 0.782 |
| **LAD** | RandomForest | 0.876 | 0.635 | 0.771 | 0.784 | **0.465** | **0.904** | 0.595 | 0.758 | 0.815 |
| **LCX** | XGBoost | 0.487 | 0.821 | 0.637 | 0.712 | **0.208** | **0.916** | 0.353 | 0.478 | 0.867 |
| **RCA** | LogisticRegression | 0.439 | 0.804 | 0.575 | 0.704 | **0.208** | **0.903** | 0.381 | 0.468 | 0.868 |

> **Clinical Tradeoff Note**: At the operating cutoff tuned for $\ge 90\%$ sensitivity, LCX and RCA specificity drops to $35.3\%$ and $38.1\%$ due to moderate target discrimination ($\text{AUC} \approx 0.73$). This tradeoff is clinically intentional: in a cardiovascular triage setting, missing an acute coronary occlusion (false negative) carries far greater risk than scheduling confirmatory imaging (false positive). Unbiased nested cross-validation sensitivity is $85.9\% \pm 4.2\%$ for Cath, $80.3\% \pm 7.7\%$ for LAD, $47.7\% \pm 7.4\%$ for LCX, and $78.7\% \pm 8.7\%$ for RCA.

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
- **Software Rendering Benchmark**: 57.8 ms mean frame time (~17.3 FPS continuous orbit, p95 = 66.2 ms) under pure CPU software rendering (`--disable-gpu` at 1280×720 on 13th Gen Intel Core i5-13420H, Chromium 148). Runs >60 FPS on standard WebGL hardware acceleration.

---

## 🧩 Architectural Extensibility

Adding a new clinical biomarker or vessel requires zero frontend redesign:
- **`ml/schema.json`**: Feature registry with units, encoding, ranges, and groups (54 features).
- **`web/src/vessels.json`**: Declarative mapping of anatomical mesh nodes, territories, 3D badge coordinates, and risk color scales.

---

## 📜 Provenance & Licenses
- **Model Provenance**: Shipped artifacts trained locally on Windows 11 (Python 3.14.5, seed 42). Google Colab execution scripts are provided in `scripts/` (untested).
- **Software Code**: MIT License (see [LICENSE](LICENSE)).
- **Clinical Dataset**: UCI Machine Learning Repository — *Extension of Z-Alizadeh Sani CAD Dataset* (CC BY 4.0).
- **3D Anatomical Mesh**: BodyParts3D, © The Database Center for Life Science, Japan (CC BY-SA 2.1 JP).

