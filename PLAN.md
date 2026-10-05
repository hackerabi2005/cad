# PLAN.md — CAD Risk Prediction & Interactive 3D Coronary Viewer

## 0. Core Decisions (FIX_PLAN.md Alignment)

1. **Row 93 Alignment (Option 2)**:
   - In the raw dataset, row index 93 (spreadsheet row 95) contains `LAD='Stenotic'`, `LCX='Normal'`, `RCA='Normal'`, but `Cath='Normal'`.
   - By definition ($\ge 50\%$ stenosis in $\ge 1$ major coronary artery constitutes CAD), this row is a clinical contradiction.
   - We derive `Cath_aligned = Cath | LAD | LCX | RCA` in derived training data, asserting exactly 1 row changes.
   - Raw input Excel file remains completely untouched and verified via SHA256 checksum.
   - Aligns the cohort distribution from **216 CAD / 87 Normal &rarr; 217 CAD / 86 Normal**.
   - Dataset cannot determine whether `Cath` or vessel labels were misrecorded.

2. **Primary Application Stack: FastAPI + React 18 SPA**:
   - The primary supported application is the unified FastAPI REST API (`api/main.py`) serving the compiled React 18 SPA + React Three Fiber 3D viewer (`web/dist/`) on port 8000.
   - The Rust (Axum) + HTMX implementation is preserved as an optional lightweight extra in `extras/rust-htmx/`.
   - All inference, range validation, coherence, and SHAP attribution are centralized in `ml/service.py` to prevent logic duplication.

3. **Logical Risk Coherence Decision**:
   - The coherence display rule ($P(\text{CAD})_{\text{coherent}} = \max(P(\text{CAD}), P(\text{LAD}), P(\text{LCX}), P(\text{RCA}))$) was quantitatively evaluated on out-of-fold predictions.
   - Raw CAD: ROC-AUC = 0.9302, Brier = 0.0967.
   - Coherent CAD: ROC-AUC = 0.9291, Brier = 0.1017 (AUC drop = 0.0012 $\le 0.010$, Brier change = $+0.0050 \le 0.010$).
   - Rule is **retained**: displayed Brier $\le$ raw $+ 0.01$ and AUC drop $\le 0.01$.
   - CAD operating threshold is calibrated directly on the displayed score ($0.611$).
   - API returns both `raw_prob` and `prob`.
   - Discrete label overrides ("any vessel High &implies; overall High") are rejected to avoid compounding false positives.

4. **Provenance & Reproducibility**:
   - Shipped model weights and artifacts were trained locally on Windows 11 (Python 3.14.5, seed 42) and verified on Python 3.12 (Linux/WSL2).
   - Dependencies strictly pinned in `ml/requirements.txt` and `.python-version`.

5. **TimesFM & TabPFN Decisions**:
   - TimesFM 3.0: Considered and rejected; strictly designed for sequential time series, inappropriate for static cross-sectional clinical observations.
   - TabPFN: Not evaluated; v2.5+ imposes non-commercial licensing constraints, interactive browser login tokens, and heavy specialized dependencies.

6. **Operating Decision Thresholds**:
   - Operating thresholds are tuned on out-of-fold predictions to enforce $\ge 90\%$ screening sensitivity.
   - Operating cutoffs: Cath ($0.611$, Sens $90.3\%$, Spec $82.6\%$), LAD ($0.469$, Sens $90.4\%$, Spec $60.3\%$), LCX ($0.217$, Sens $90.8\%$, Spec $37.0\%$), RCA ($0.213$, Sens $90.3\%$, Spec $39.2\%$).
   - Derived from inner out-of-fold tuning, nested cross-validation sensitivity estimates are **$90.2\% \pm 3.9\%$ for Cath**, **$88.2\% \pm 7.2\%$ for LAD**, **$91.6\% \pm 6.3\%$ for LCX**, and **$89.2\% \pm 6.9\%$ for RCA** (detailed in `reports/threshold_nested.md`).

---

## 1. Directory Structure

```text
CAD/
├── .gitignore
├── .python-version               # Pinned Python 3.14.5
├── CLAUDE.md                     # Development invariants and guidelines
├── LICENSE                       # MIT + CC BY-SA 2.1 JP + CC BY 4.0
├── Makefile                      # Setup, train, test, test-ui, serve, report
├── PLAN.md                       # Architectural decisions and roadmap
├── README.md                     # Quickstart, benchmarks, usage
│
├── api/                          # Primary FastAPI application
│   ├── main.py                   # REST endpoints & React static SPA serving
│   └── tests/                    # Integration & consistency test suite
│
├── data/raw/                     # Canonical raw data & SHA256 checksums
│   ├── CHECKSUMS
│   ├── extention of Z-Alizadeh sani dataset.xlsx
│   └── BP51782_.../              # 129 BodyParts3D OBJ meshes
│
├── docs/                         # Clinical & technical documentation
│   ├── demo_script.md
│   ├── report.md                 # Technical report markdown
│   └── report.pdf                # Compiled publication-grade PDF report (<= 6 pages)
│
├── extras/rust-htmx/             # Optional Rust Axum + HTMX stack
│   ├── server_rust/
│   ├── sidecar.py
│   └── test_rust_ui_playwright.py
│
├── ml/                           # Core ML pipeline
│   ├── artifacts/                # Serialized bundles, metrics, model card
│   ├── explain.py                # LinearSHAP & TreeSHAP exact additivity
│   ├── pipeline.py               # Preprocessing, row 93 alignment, leakage prevention
│   ├── requirements.txt          # Pinned package versions
│   ├── schema.json               # 54-feature registry
│   ├── service.py                # Centralized prediction, validation, coherence logic
│   ├── train.py                  # 15-fold CV, calibration, threshold tuning
│   └── tests/                    # Label consistency & leakage unit tests
│
├── reports/                      # Validation audits & figures
│   ├── calibration_curves.png
│   ├── coherence_eval.md
│   ├── data_audit.md
│   ├── mesh_audit.md
│   ├── model_results.md
│   └── screenshots/
│
├── scripts/                      # Build & verification utilities
│   ├── build_heart_glb.py
│   ├── generate_pdf_report.py
│   └── test_ui_playwright.py
│
└── web/                          # Primary React 18 SPA + React Three Fiber viewer
    ├── src/
    └── dist/                     # Compiled frontend bundle
```

---

## 2. Verification Protocol

1. **Labels & Consistency**:
   - `python -m pytest ml/tests/test_labels.py` &rarr; verify raw mismatch is exactly index 93; raw hash untouched; aligned Cath == OR(vessels) on all 303 rows.
2. **Leakage & Baseline**:
   - `python -m pytest ml/tests/test_leakage.py` &rarr; verify no targets in X, 54 features match schema, shuffled label CV drops to chance ($0.50 \pm 0.08$).
3. **API & SHAP Consistency**:
   - `python -m pytest api/tests/` &rarr; verify schema, metrics, samples, HTTP 422 bounds validation, p95 latency (<300ms), and $|base + \sum shap - raw\_score| < 10^{-4}$.
4. **End-to-End & 3D Viewer Benchmark**:
   - `python scripts/test_ui_playwright.py` &rarr; 5-second continuous orbit at 1280&times;720 under `--disable-gpu` (verifies $\ge 15$ FPS software rendering).
5. **Technical Report**:
   - `python scripts/generate_pdf_report.py` &rarr; asserts compiled PDF is $\le 6$ pages and numbers match `metrics.json`.
