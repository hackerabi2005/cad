# Cardio3D AI — Demonstration Video & Presentation Guide

> **Planned Public Video URL**: `https://youtube.com/watch?v=PENDING` [Walkthrough recording PENDING final upload]

---

## 1. Video Presentation Overview

- **Format**: 5 to 7 minute technical walkthrough and clinical demonstration.
- **Presenter**: Engineering & Clinical Informatics Lead.
- **Scope**: Demonstrates end-to-end functionality across 5 core evaluation criteria:
  1. Multi-vessel stenosis risk prediction (LAD, LCX, RCA, and overall CAD).
  2. Exact additive SHAP feature explainability in log-odds and probability space.
  3. Interactive 3D coronary anatomy viewer (WebGL + 2D schematic fallback).
  4. Decision Curve Analysis (DCA) and clinical utility at screening thresholds.
  5. What-If sensitivity explorer, empirical Wilson cohort confidence intervals, and clinical print summary.

---

## 2. Timed Presentation Shot List

| Timestamp | Screen View | Demonstration Action | Key Narrative Points |
|---|---|---|---|
| **0:00 – 0:45** | Initial Dashboard | Overview of 3D coronary anatomy, disclaimer, and architecture. | Single-center Z-Alizadeh Sani cohort (303 patients, 54 features). Zero-leakage pipeline design. |
| **0:45 – 1:30** | Patient Presets | Click Patient A (Low Risk) &rarr; Patient C (High Risk). | Live debounced inference (<50ms model compute). Logical risk coherence rule ($P(\text{CAD}) \ge \max(P_{\text{vessels}})$). |
| **1:30 – 2:30** | 3D Heart Viewer | Rotate 3D mesh, click LAD, LCX, and RCA branches. | Vessel-level anatomical color mapping from `vessels.json`. Camera fly-to focus and normal-inflated hit-proxy. |
| **2:30 – 3:30** | SHAP Attribution | Inspect local waterfall for Cath and individual vessels. | Exact additive feature decomposition. Top drivers (Typical Chest Pain, Age, HTN). Empirical Wilson intervals. |
| **3:30 – 4:30** | Sensitivity & DCA | Toggle Balanced (0.50) vs Screening (cutoff 0.400) mode. Switch to Model Validation tab for DCA net-benefit. | Sensitivity-first thresholding enforcing $\ge 95\%$ screening sensitivity. Net-benefit superiority over treat-all/none. |
| **4:30 – 5:30** | What-If & Report | Modify systolic BP / Typical Chest Pain in What-If explorer. Open Clinical Report modal. | Local gradient exploration without causal claims. Exportable publication-grade clinical assessment summary. |
| **5:30 – 6:00** | Conclusion & GitHub | Show GitHub repository, Docker container, and test suite. | 100% test pass rate, strict bounds validation, and reproducible nested-CV protocol. |

---

## 3. Recording Instructions for Presenters

1. Start local production server:
   ```bash
   python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
   ```
2. Open browser in 1080p (1920x1080) at `http://localhost:8000`.
3. Follow the timed shot list above with clear clinical diction.
4. Export video in 1080p60 MP4, upload to YouTube (Unlisted or Public), and update this file and the README badge with the live URL.
