# Cardio3D AI: System Architecture & Technical Specifications

---

## 1. PROJECT OVERVIEW

### Core Purpose
**Cardio3D AI** is a clinical decision-support and educational visualization platform for coronary artery disease (CAD) assessment (developed for the Multimodal AI Hackathon 2026). Given 54 patient demographic, symptom, examination, ECG, echocardiographic, and laboratory biomarkers, the system concurrently predicts overall CAD status alongside vessel-specific stenosis ($\ge 50\%$ diameter reduction) across the three primary coronary arteries: the Left Anterior Descending (**LAD**), the Left Circumflex (**LCX**), and the Right Coronary Artery (**RCA**). Predictions are coupled with exact additive local feature attributions via SHAP (SHapley Additive exPlanations) and mapped dynamically onto an interactive 3D anatomical coronary heart model derived from BodyParts3D meshes.

### Technology Stack
- **Primary Languages**:
  - **Python 3.14.5**: Core ML pipelines, training, SHAP explainers, dataset ingestion, and IPC sidecar microservice.
  - **Rust (2021 edition)**: High-concurrency, sub-millisecond asynchronous Axum web server and hypermedia router.
  - **TypeScript 5.7 / JavaScript (ES2022)**: Browser interaction layers, WebGL 3D rendering, and client component logic.
  - **HTML5 & CSS3**: Modern dark glassmorphic medical workstation design system.
- **Machine Learning & Data Processing**:
  - `scikit-learn 1.6+`: Data transformers, Pipelines, LogisticRegression, RandomForest, cross-validation metrics.
  - `xgboost 2.0+`: Gradient-boosted decision trees for non-linear vessel stenosis estimation.
  - `shap 0.45+`: `LinearExplainer` and `TreeExplainer` providing mathematically exact additive attribution ($| \text{error} | < 10^{-4}$).
  - `pandas 2.2+` & `numpy 1.26+`: Tabular feature wrangling and matrix operations.
  - `joblib 1.4+`: Zero-copy artifact serialization and deserialization.
  - `trimesh 4.0+`: 3D OBJ parsing, coordinate centering, node classification, and binary GLB compilation.
- **Web Engines & Backends**:
  - **Rust Backend**: `axum 0.7`, `tokio 1.0` (asynchronous runtime), `tower-http 0.5`, `tera 1.19` (Jinja2-compatible templating), `reqwest 0.12`.
  - **Python Backend**: `fastapi 0.110+`, `uvicorn 0.28+`, `pydantic 2.6+`, `starlette 0.36+`.
  - **Hypermedia / Client**: `HTMX 2.0` (reactive AJAX partial swapping and event-driven trigger dispatch).
  - **3D Graphics**: `Three.js 0.170` (Vanilla WebGL 3D viewer) and React Three Fiber / Drei (`@react-three/fiber 8.17`, `@react-three/drei 9.120`).
  - **React SPA**: `React 18.3`, `Vite 6.2`, `Tailwind CSS 3.4`, `Lucide React 0.475`.
- **Runtime Versions**:
  - Python: 3.14.5 (x86_64 Windows)
  - Rust: rustc 1.84.0 / Cargo 1.84.0
  - Node.js: v20+ / npm 10+

### Project Type
Cardio3D AI is a **clinical decision-support workstation and ML inference monorepo**.
1. **Primary Application (FastAPI + React)**: Unified FastAPI REST API server + React 18 SPA + React Three Fiber 3D interactive viewer (compiled into `web/dist/`, served on port 8000).
2. **Optional Extra Stack (Rust + HTMX)**: High-performance Rust (Axum) web server + HTMX reactive swapping + Three.js + shared Python ML service (located in `extras/rust-htmx/`).

---

## 2. FULL DIRECTORY TREE

Below is the complete inventory of all files and folders in the repository (excluding generated build artifacts, virtual environments, and `.git`):

```text
CAD/
│
├── .gitignore                                         # Git ignore specifications for Python, Node, OS, and build artifacts
├── .python-version                                    # Pinned Python version (3.14.5)
├── CLAUDE.md                                          # Development invariants, coding rules, and zero-leakage constraints
├── LICENSE                                            # MIT (software) + CC BY-SA 2.1 JP (mesh) + CC BY 4.0 (dataset)
├── Makefile                                           # Orchestration tasks: setup, local/colab training, testing, 3D builds, serving
├── PLAN.md                                            # Technical specifications, assumptions, and architectural roadmap
├── README.md                                          # Project overview, quickstart instructions, evaluation tables, and citations
│
├── extras/                                            # Optional auxiliary stacks and tools
│   └── rust-htmx/                                     # Optional Rust (Axum) + HTMX implementation and Python sidecar
│
├── api/                                               # FastAPI monolithic REST API and web application server
│   ├── main.py                                        # FastAPI app serving /api routes, validating inputs, and serving static SPA
│   └── tests/                                         # API integration test suite
│       ├── test_api.py                                # Pytest validating schema, metrics, samples, predict contract, HTTP 422, p95 latency
│       └── test_consistency.py                        # Mathematical consistency tests between predictions, raw scores, and SHAP sum
│
├── data/                                              # Structured project data directory
│   └── raw/                                           # Canonical raw immutable inputs
│       ├── CHECKSUMS                                  # SHA256 integrity hashes for dataset and all 129 BodyParts3D OBJ files
│       ├── extention of Z-Alizadeh sani dataset.xlsx  # Canonical input dataset spreadsheet
│       └── BP51782_FMA3_2_1_inference_isa_FMA67135_Postnatal_anatomical_structure/ # Canonical mirror of 129 BodyParts3D OBJ files
│
├── docs/                                              # Clinical and technical documentation
│   ├── demo_script.md                                 # Timed live demo script for clinical presentations and hackathon judging
│   ├── report.md                                      # Full technical report source detailing data audit, ML methodology, and 3D modeling
│   └── report.pdf                                     # Compiled publication-grade PDF technical report
│
├── ml/                                                # Machine Learning pipeline, training, and explainability modules
│   ├── artifacts/                                     # Serialized ML models, explainers, and metrics
│   │   ├── metrics.json                               # Repeated Stratified 5-Fold CV metrics across models and targets
│   │   ├── model_bundle.joblib                        # Serialized dictionary containing fitted pipelines, explainers, thresholds
│   │   ├── model_card.json                            # Model provenance, evaluation conditions, hyperparameters, and environment specs
│   │   └── shap_global.json                           # Dataset-wide global feature importance rankings per clinical target
│   ├── explain.py                                     # Additive SHAP explanation logic, dummy-to-parent aggregation, and additivity checks
│   ├── pipeline.py                                    # Zero-leakage data loading, column transformers, and pipeline construction
│   ├── requirements.txt                               # Pinned Python package dependencies for reproducible environments
│   ├── schema.json                                    # Authoritative metadata registry for 54 features (ranges, categories, groups, units)
│   ├── service.py                                     # Centralized patient inference, validation, coherence, and SHAP service
│   ├── train.py                                       # End-to-end training script: CV benchmark, model selection, calibration, serialization
│   └── tests/                                         # ML integrity and validation unit tests
│       ├── test_labels.py                             # Pytest verifying raw label audit, hash immutability, and row 93 alignment
│       └── test_leakage.py                            # Pytest verifying strict exclusion of targets from X and chance shuffled baseline
│
├── reports/                                           # Technical audits, validation figures, and visual evidence
│   ├── calibration_curves.png                         # Matplotlib reliability calibration plots for Cath, LAD, LCX, and RCA
│   ├── coherence_eval.md                              # Quantitative evaluation of risk coherence rule P(CAD) >= max(P_vessel)
│   ├── data_audit.md                                  # In-depth clinical data audit: distributions, missingness, Row 93 alignment
│   ├── mesh_audit.md                                  # 3D mesh node catalog, polygon budgets, and coronary vessel extraction mapping
│   ├── model_results.md                               # Formatted markdown table comparing CV ROC-AUC, PR-AUC, F1, Recall, and Brier
│   └── screenshots/                                   # High-resolution visual proof of UI states and workflows
│       ├── 01_initial_dashboard.png                   # React UI: Initial dashboard state with 3D anatomical heart
│       ├── 02_mobile_responsive_375px.png             # React UI: Mobile 375px responsive layout
│       ├── 03_high_risk_patient.png                   # React UI: Patient C loaded showing high CAD risk
│       ├── 04_low_risk_patient.png                    # React UI: Patient A loaded showing low CAD risk
│       ├── 05_lad_shap_waterfall.png                  # React UI: Localized LAD SHAP waterfall chart
│       ├── 06_lcx_shap_waterfall.png                  # React UI: Localized LCX SHAP waterfall chart
│       ├── 07_shap_table_view.png                     # React UI: Tabular SHAP feature contributions
│       ├── 08_global_importance_tab.png               # React UI: Dataset-wide global feature importance view
│       └── 09_model_validation_tab.png                # React UI: 15-fold cross-validation performance metrics
│
├── scripts/                                           # Automation, asset processing, and verification scripts
│   ├── build_heart_glb.py                             # Converts 129 BodyParts3D OBJ meshes into an optimized 83,600-triangle binary GLB
│   ├── colab_launcher.py                              # Remote launcher script for execution inside Google Colab
│   ├── generate_pdf_report.py                         # Generates docs/report.pdf using ReportLab and matplotlib figures
│   ├── test_ui_playwright.py                          # Playwright E2E test verifying React UI responsiveness, 3D FPS, and presets
│   └── train_colab.sh                                 # Bash script executing automated training run inside Google Colab
└── web/                                               # React 18 + Vite + TypeScript frontend monorepo package
    ├── index.html                                     # Single Page Application HTML entry shell
    ├── package-lock.json                              # Locked NPM dependency tree
    ├── package.json                                   # NPM package manifest (React, Three, Lucide, Tailwind, Vite)
    ├── postcss.config.js                              # PostCSS configuration loading Tailwind CSS and Autoprefixer
    ├── tailwind.config.js                             # Tailwind CSS design system configuration
    ├── tsconfig.json                                  # TypeScript compiler options
    ├── vite.config.ts                                 # Vite bundler configuration with dev server proxy to port 8000
    ├── public/                                        # Static assets served by Vite
    │   └── models/                                    # 3D assets
    │       └── heart.glb                              # Mirror of compiled 1.67 MB binary GLB model
    └── src/                                           # React source code
        ├── App.tsx                                    # Top-level application component coordinating state, tabs, and API calls
        ├── index.css                                  # Global CSS rules and Tailwind utility directives
        ├── main.tsx                                   # React DOM root render mount
        ├── types.ts                                   # TypeScript interfaces (PredictResponse, ExplainResult, SamplePatient, etc.)
        ├── vessels.json                               # Declarative coronary anatomy mapping (FMA IDs, coordinates, branch definitions)
        └── components/                                # Reusable UI components
            ├── DisclaimerBanner.tsx                   # Prominent amber clinical safety disclaimer header and footer
            ├── GlobalImportance.tsx                   # Dataset-wide global feature importance visualization
            ├── HeartViewer.tsx                        # React Three Fiber 3D coronary anatomy scene with floating Drei HTML badges
            ├── MetricsTab.tsx                         # 15-fold cross-validation performance metrics dashboard
            ├── PatientForm.tsx                        # Debounced biomarker input controls with preset patient selectors
            ├── RiskOverview.tsx                       # Overall CAD status card with animated SVG circle gauge
            ├── ShapWaterfall.tsx                      # Local SHAP waterfall attribution bar chart
            └── VesselCards.tsx                        # Interactive LAD, LCX, RCA cards with stenosis badges
```

---

## 3. ENTRY POINTS

The system supports a primary production entry point (FastAPI + React 18 SPA) and an optional lightweight hypermedia stack (Rust Axum + HTMX in `extras/rust-htmx/`).

### Main Execution Files
1. **Primary Unified FastAPI Application Entry Point**:
   - **Path**: `api/main.py`
   - **Command**: `python -m uvicorn api.main:app --host 127.0.0.1 --port 8000`
   - **Role**: Monolithic server that exposes REST API endpoints under `/api/*` and serves the precompiled React SPA from `web/dist/` under `/`.
2. **React Client Application Entry Point**:
   - **Path**: `web/src/main.tsx` (built via `web/index.html`)
   - **Command**: `npm run dev` (Vite dev server on port 5173) or `npm run build` (outputs to `web/dist/`)
   - **Role**: Client-side SPA entry point initializing React DOM and mounting `App.tsx` with React Three Fiber 3D viewer.
3. **Machine Learning Training & Artifact Pipeline**:
   - **Path**: `ml/train.py`
   - **Command**: `python -m ml.train`
   - **Role**: Reads the raw Excel dataset, executes 15-fold Repeated Stratified CV, benchmarks models, fits final production pipelines and SHAP explainers, verifies additivity, and serializes `ml/artifacts/`.
4. **Centralized ML Inference Service**:
   - **Path**: `ml/service.py`
   - **Role**: Single source of truth for patient validation, multi-task model inference, coherence adjustment ($P(\text{CAD}) \ge \max(P(\text{vessels}))$), operating thresholding, and additive SHAP attribution.
5. **3D Asset Pipeline Tool**:
   - **Path**: `scripts/build_heart_glb.py`
   - **Command**: `python scripts/build_heart_glb.py`
   - **Role**: Parses 129 BodyParts3D OBJ meshes from `data/raw/`, extracts coronary branches, decodes myocardium, centers bounding boxes, and outputs binary GLB to `web/public/models/heart.glb` (and `extras/rust-htmx/server_rust/static/models/heart.glb`).
6. **Optional Rust Server Workstation Entry Point**:
   - **Path**: `extras/rust-htmx/server_rust/src/main.rs`
   - **Command**: `cargo run --manifest-path extras/rust-htmx/server_rust/Cargo.toml`
   - **Role**: Optional extra stack. Listens on `http://127.0.0.1:8000`. Serves hypermedia HTML partials via Tera, streams static assets, and communicates asynchronously with the Python ML sidecar.
7. **Optional Python ML Sidecar Entry Point**:
   - **Path**: `extras/rust-htmx/sidecar.py`
   - **Command**: `python -m uvicorn extras.rust-htmx.sidecar:app --host 127.0.0.1 --port 8001`
   - **Role**: IPC bridge microservice on port 8001 importing `ml.service` for the optional Rust Axum engine.

### System Boot Sequences

#### Architecture A: Python FastAPI + React SPA (Primary Monolithic Application)
```mermaid
sequenceDiagram
    autonumber
    participant FastAPI as FastAPI Server (Port 8000)
    participant Client as Browser (React 18 SPA)

    FastAPI->>FastAPI: lifespan(): load_artifacts() (model_bundle.joblib, metrics.json, shap_global.json)
    FastAPI->>FastAPI: Mount web/dist/ as StaticFiles (if compiled)
    
    Client->>FastAPI: GET / (Loads index.html, JS, CSS bundle)
    Client->>FastAPI: GET /api/schema, GET /api/samples, GET /api/metrics (Promise.all)
    FastAPI-->>Client: Returns JSON metadata & patient presets
    Client->>Client: Initialize React state with Patient A (Low Risk)
    Client->>FastAPI: POST /api/predict (Initial Patient A JSON)
    FastAPI-->>Client: Returns JSON {cad, vessels, explain, disclaimer}
    Client->>Client: React Three Fiber renders 3D heart with animated materials
```

#### Architecture B: Rust (Axum) + HTMX + Three.js (Optional Workstation in `extras/rust-htmx/`)
```mermaid
sequenceDiagram
    autonumber
    participant Sidecar as Python ML Sidecar (Port 8001)
    participant Rust as Rust Axum Server (Port 8000)
    participant Client as Browser (HTMX + Three.js)

    Note over Sidecar: python -m uvicorn extras.rust-htmx.sidecar:app --port 8001
    Sidecar->>Sidecar: lifespan(): load_artifacts() into memory
    
    Note over Rust: cargo run (in extras/rust-htmx/server_rust)
    Rust->>Rust: Compile Tera templates ("templates/**/*")
    Rust->>Rust: Bind TcpListener to 127.0.0.1:8000
    
    Client->>Rust: GET / (Initial Page Load)
    Rust->>Sidecar: GET /samples
    Sidecar-->>Rust: Return 3 preset patient records
    Rust->>Sidecar: POST /predict (Payload: Patient A initial sample)
    Sidecar-->>Rust: Return CAD & vessel probabilities, labels, and SHAP
    Rust->>Rust: Render overview.html, form.html, shap.html into index.html
    Rust-->>Client: Return complete HTML document + /static assets
    
    Client->>Client: heart_viewer.js loads /models/heart.glb into Three.js WebGL scene
    Client->>Client: Apply risk colors & attach HTMX custom event listeners
```

---

## 4. MODULE / COMPONENT BREAKDOWN

### 4.1. Machine Learning Module (`ml/`)
**Responsibility**: Enforces clinical invariants, loads raw data without target leakage, trains multi-task models across 15 CV folds, computes exact SHAP explanations, and runs the internal inference sidecar.

- **`ml/schema.json`**:
  - Declarative dictionary of 54 features with metadata: `label`, `group` (`demographic`, `symptoms-exam`, `lab`, `ECG`, `echo`), `encoding` (`numeric`, `binary`, `ordinal`, `nominal`), `min`, `max`, `mean`, `median`, `categories`, and `unit`.
- **`ml/pipeline.py`**:
  - `load_schema() -> Dict[str, Any]`: Loads and returns `schema.json`.
  - `get_feature_groups(schema: Dict[str, Any]) -> Tuple[List[str], List[str]]`: Splits features into numeric/ordinal vs. categorical/binary.
  - `build_preprocessor(schema: Dict[str, Any]) -> ColumnTransformer`: Assembles a `ColumnTransformer` with `SimpleImputer(strategy="median")` + `StandardScaler()` for numeric columns, and `SimpleImputer(strategy="most_frequent")` + `OneHotEncoder(handle_unknown="ignore")` for categorical columns.
  - `build_pipeline(estimator: Any, schema: Dict[str, Any] | None = None) -> Pipeline`: Wraps the preprocessor and a given classifier into an end-to-end `sklearn.pipeline.Pipeline`.
  - `load_raw_dataset(excel_path: str | Path, align_row_93: bool = True) -> Tuple[pd.DataFrame, Dict[str, pd.Series]]`: Reads the Excel spreadsheet, binarizes targets (`LAD`, `LCX`, `RCA` to $0/1$, `Cath` to $0/1$), aligns Row 93 to CAD per clinical guidelines, and strictly drops all target columns (`DROP_COLS = ["Cath", "LAD", "LCX", "RCA", "Exertional CP"]`) from $X$.
- **`ml/explain.py`**:
  - `build_parent_feature_mapping(transformed_feature_names: List[str], schema_keys: List[str]) -> Dict[str, str]`: Maps one-hot dummy columns (e.g. `Sex_Male`, `Sex_Fmale`) back to their parent physiological feature (`Sex`).
  - `create_explainer(model: Any, background_transformed: np.ndarray) -> shap.Explainer`: Instantiates `shap.LinearExplainer` for LogisticRegression or `shap.TreeExplainer` for XGBoost / RandomForest.
  - `explain_sample(pipeline: Any, explainer: Any, parent_mapping: Dict[str, str], schema: Dict[str, Any], sample_df: pd.DataFrame) -> Dict[str, Any]`: Computes raw log-odds or margin scores, extracts SHAP values, sums one-hot attributions into parent features, verifies additivity ($\text{error} = | \text{base\_value} + \sum \text{shap} - \text{raw\_score} | < 10^{-4}$), and calculates percentage contributions.
  - `compute_global_importance(pipeline: Any, explainer: Any, parent_mapping: Dict[str, str], schema: Dict[str, Any], X_df: pd.DataFrame) -> List[Dict[str, Any]]`: Calculates dataset-wide mean absolute SHAP values for global ranking.
- **`ml/train.py`**:
  - `calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]`: Computes Accuracy, Precision, Recall, Specificity, F1, ROC-AUC, PR-AUC, and Brier Score.
  - `find_high_sensitivity_threshold(y_true: np.ndarray, y_prob: np.ndarray, min_sensitivity: float = 0.90) -> Tuple[float, float, float]`: Grid-searches thresholds on out-of-fold predictions to find the cut-point maximizing specificity subject to sensitivity $\ge 90\%$.
  - `get_candidate_models() -> Dict[str, Any]`: Returns candidate constructors: `Baseline_Majority`, `LogisticRegression(C=0.1)`, `RandomForest(n_estimators=100, max_depth=4)`, and `XGBoost(n_estimators=50, max_depth=3)`.
  - `main()`: Orchestrates the 15-fold Repeated Stratified CV evaluation, generates calibration plots, selects models within 1 SE of best ROC-AUC, verifies 20-sample SHAP additivity, and serializes `metrics.json`, `model_card.json`, `shap_global.json`, and `model_bundle.joblib`.
- **`ml/service.py`**:
  - Centralized clinical prediction service used across both FastAPI and the optional Rust sidecar.
  - Implements `predict_patient()`, physiological range validation (`ValidationError`), model inference, coherence adjustment ($P(\text{CAD}) \ge \max(P(\text{vessels}))$), sensitivity-first thresholding, and additive SHAP generation.
- **`ml/tests/test_leakage.py`**:
  - Unit tests verifying: (1) no targets exist in feature matrix $X$; (2) schema matches $X$ columns exactly ($54$ features); (3) label-shuffled CV ROC-AUC drops to chance ($0.50 \pm 0.08$).

---

### 4.2. Optional Rust Axum Engine (`extras/rust-htmx/server_rust/`)
**Responsibility**: Serves as an optional lightweight hypermedia web workstation on port 8000. Provides sub-millisecond route handling, server-side template rendering, hypermedia partial updates, static model streaming, and HTTP IPC bridging to the ML sidecar (`extras/rust-htmx/sidecar.py`).

- **`server_rust/src/main.rs`**:
  - `struct AppState`: Contains shared `reqwest::Client` connection pool and `Arc<Tera>` template instance.
  - `main()`: Tokio entry point. Initializes tracing subscriber, compiles Tera templates, builds Axum router, mounts `/static` and `/models` static directories, and binds to `127.0.0.1:8000`.
  - `render_index(State(state): State<Arc<AppState>>) -> Result<Html<String>, StatusCode>`: Fetches default patient preset, predicts baseline cardiac risk, renders `overview.html`, `form.html`, and `shap.html`, and outputs `index.html`.
  - `handle_predict(State(state): State<Arc<AppState>>, Form(form_data): Form<HashMap<String, String>>) -> Result<Response, StatusCode>`: Parses form fields, converts strings to floats/strings, dispatches `POST /predict` to Python sidecar on port 8001, renders `overview.html` and `shap.html`, constructs `HX-Trigger: {"vesselRiskUpdated": {"LAD": ..., "LCX": ..., "RCA": ...}}`, and returns combined HTML with an out-of-band swap (`hx-swap-oob="innerHTML"`) for the SHAP container.
  - `handle_get_sample(State(state): State<Arc<AppState>>, AxumPath(sample_id): AxumPath<String>) -> Result<Response, StatusCode>`: Fetches requested preset (Patient A, B, or C) from sidecar, extracts all 54 features into hidden inputs, renders `form.html`, and sends `HX-Trigger: formUpdated`.
  - `handle_get_shap(State(state): State<Arc<AppState>>, Query(query): Query<ShapQuery>) -> Result<Html<String>, StatusCode>`: Renders isolated `shap.html` partial for a requested vessel (`Cath`, `LAD`, `LCX`, `RCA`).
  - `handle_get_tab(State(state): State<Arc<AppState>>, AxumPath(tab_name): AxumPath<String>) -> Result<Html<String>, StatusCode>`: Swaps main tabs (`risk-shap`, `global-factors`, `cv-metrics`).
  - `render_overview`, `render_form`, `render_shap`: Internal Tera helper rendering functions.
- **`server_rust/templates/`**:
  - `index.html`: Base layout, navigation header, and main split viewport grid.
  - `overview.html`: CAD assessment card, circular SVG gauge with stroke-dashoffset animation, raw vs. coherent score metrics, and 3 clickable vessel cards.
  - `form.html`: Two-column grid of 12 clinical biomarker controls, preset buttons, and hidden inputs for the remaining 42 features.
  - `shap.html`: Explainer card with base value indicator and horizontal bars for top 8 features colored by sign (red = risk, green = protective).
  - `cv_metrics.html`: Cross-validation validation table showing ROC-AUC, PR-AUC, F1, Sensitivity, Specificity, and Brier score.
  - `global_factors.html`: Dataset-wide global feature importance list.
- **`server_rust/static/js/heart_viewer.js`**:
  - Three.js r170 script initializing WebGLRenderer, PerspectiveCamera, OrbitControls, and DirectionalLight.
  - Loads `/models/heart.glb` via GLTFLoader.
  - Isolates meshes: `LAD`, `LCX`, `RCA`, `Aorta`, and `Heart_Muscle`.
  - Renders floating HTML 3D badges projected from 3D object space to 2D screen space.
  - Implements Raycaster click detection: clicking coronary vessels spotlights them with emissive glow and calls `htmx.ajax` to switch the SHAP target.
  - Listens for `vesselRiskUpdated` custom events dispatched by HTMX to smoothly interpolate mesh material colors.

---

### 4.3. Primary React Web Application (`web/`)
**Responsibility**: Primary interactive Single Page Application (SPA) providing React Three Fiber 3D rendering, responsive clinician controls, and client-side reactive state management.

- **`web/src/App.tsx`**:
  - Coordinates top-level state: `patientData`, `prediction`, `selectedVessel`, `activeTarget`, `activeTab`.
  - Implements 250ms debounced input dispatch to `/api/predict`.
- **`web/src/types.ts`**:
  - TypeScript interfaces: `PredictResponse`, `CadResult`, `VesselResult`, `ExplainResult`, `FeatureContribution`, `SamplePatient`, `SchemaRegistry`, `ModelMetricsResponse`.
- **`web/src/vessels.json`**:
  - Maps vessel identifiers to mesh node names (`MM513`, `MM635`, `MM545`), clinical descriptions, camera target coordinates, and badge positions.
- **`web/src/components/`**:
  - `DisclaimerBanner.tsx`: Renders persistent top warning banner and footer notice.
  - `HeartViewer.tsx`: Canvas container with `@react-three/fiber`, OrbitControls, and Drei `<Html>` floating badges.
  - `RiskOverview.tsx`: Displays the animated SVG circular gauge and operating threshold indicators.
  - `VesselCards.tsx`: Renders clickable vessel cards for LAD, LCX, RCA.
  - `ShapWaterfall.tsx`: Local feature attribution bar chart with additivity verification indicator.
  - `PatientForm.tsx`: Inputs for physiological biomarkers and preset patient switches.
  - `GlobalImportance.tsx`: Displays global feature rankings.
  - `MetricsTab.tsx`: Displays the 15-fold cross-validation table.

---

### 4.4. 3D Model Asset Pipeline (`scripts/build_heart_glb.py`)
**Responsibility**: Converts heterogeneous clinical mesh collections into an optimized, web-ready 3D anatomical asset.

- **Inputs**: 129 BodyParts3D `.obj` files in `3D MODAL/...`.
- **Processing**:
  - Parses vertices and triangle faces using `trimesh`.
  - Classifies nodes by FMA anatomical naming conventions:
    - **LAD**: Trunk of anterior interventricular branch + diagonal & septal branches (13,574 triangles).
    - **LCX**: Trunk of circumflex branch + posterior ventricular branches (19,160 triangles).
    - **RCA**: Trunk of right coronary artery + marginal, conus, AV nodal branches (13,764 triangles).
    - **Aorta**: Ascending aorta and bulb (6,572 triangles).
    - **Heart_Muscle**: Merged and decimated myocardium chambers (30,530 triangles).
  - Centers bounding box to $(0, 0, 0)$ and normalizes scale.
- **Output**: Binary GLB asset (`heart.glb`), 1.67 MB, 83,600 total triangles (well within the $\le 100,000$ budget).

---

## 5. DATA FLOW

### Detailed Request / Action Execution Sequence

```mermaid
flowchart TD
    subgraph Client["Clinician Workstation (Browser)"]
        A[Clinician edits Biomarker or clicks Patient Preset] --> B{Interaction Type}
        B -- "Preset Click (Patient C)" --> C["HTMX GET /api/sample/sample-high-risk"]
        B -- "Form Input (Debounced 250ms)" --> D["HTMX POST /api/predict (Form Encoded)"]
        B -- "Vessel Mesh Click in 3D" --> E["HTMX GET /api/shap?target=LAD"]
    end

    subgraph RustServer["Rust Axum Server (Port 8000)"]
        C --> F["handle_get_sample()"]
        D --> G["handle_predict()"]
        E --> H["handle_get_shap()"]
        
        F -- "GET /samples" --> I["Sidecar Client"]
        G -- "POST /predict (JSON Payload)" --> I
        H -- "Internal Logic / Cache" --> J["Render Tera Partial"]
    end

    subgraph Sidecar["Python ML Sidecar (Port 8001)"]
        I -- "HTTP Requests" --> K["FastAPI sidecar.py"]
        K --> L["Plausible Range Bounds Validation (422)"]
        L --> M["Impute Missing Features via Medians/Modes"]
        M --> N["Scikit-Learn Preprocessor (ColumnTransformer)"]
        N --> O["Predict Proba: LR (Cath, RCA), RF (LAD), XGB (LCX)"]
        O --> P["Compute Coherent Risk: max(Cath, LAD, LCX, RCA)"]
        P --> Q["SHAP Explainers: Exact Additive Feature Decomposition"]
        Q --> R["Aggregate One-Hot Dummy Attributions to Parent Biomarkers"]
        R --> S["Return Complete JSON Response"]
    end

    S --> RustServer
    RustServer --> T["Render overview.html + shap.html via Tera"]
    T --> U["Attach Header: HX-Trigger: vesselRiskUpdated"]
    U --> Client

    subgraph ClientUpdates["Reactive Client Rendering"]
        U -- "HTML Swapped into DOM" --> V["Update Circular Gauge, Risk Badges & SHAP Chart"]
        U -- "Event: vesselRiskUpdated" --> W["Three.js heart_viewer.js updates Vessel Materials"]
        W --> X["LAD/LCX/RCA Mesh Color Interpolation (Green -> Red)"]
    end
```

### Step-by-Step Data Transformation
1. **User Action**: Clinician selects "Patient C (High Risk)" or adjusts Systolic BP.
2. **Client Dispatch**:
   - HTMX captures the event, serializes form parameters (or queries preset endpoint), and fires an asynchronous HTTP request to `http://127.0.0.1:8000`.
3. **Rust Axum Processing**:
   - `handle_predict` extracts form fields into a `HashMap<String, String>`, attempts numerical parsing (`f64`), and packages it into a JSON map.
   - Rust dispatches an asynchronous HTTP POST request to `http://127.0.0.1:8001/predict` via `reqwest`.
4. **Python ML Sidecar Execution**:
   - Checks biomarker inputs against physiological bounds defined in `schema.json` (throws HTTP 422 if values exceed plausible bounds).
   - Constructs a 1-row `pandas.DataFrame`. Features missing from visible form inputs are populated from patient state or cohort medians/modes.
   - Feeds the row through the 4 production pipelines:
     - `Cath`: LogisticRegression ($\text{ROC-AUC} = 0.929$)
     - `LAD`: RandomForest ($\text{ROC-AUC} = 0.846$)
     - `LCX`: XGBoost ($\text{ROC-AUC} = 0.735$)
     - `RCA`: LogisticRegression ($\text{ROC-AUC} = 0.733$)
   - Calculates coherent CAD probability: $P(\text{CAD}_\text{coherent}) = \max(P(\text{Cath}), P(\text{LAD}), P(\text{LCX}), P(\text{RCA}))$.
   - Invokes exact SHAP explainers for all 4 endpoints. One-hot dummy column contributions are grouped back to the parent feature. Verifies additivity: $| \text{base\_value} + \sum \text{shap} - \text{raw\_score} | < 10^{-4}$.
   - Returns full inference JSON payload back to Rust in $\sim 28\text{ ms}$.
5. **Hypermedia Assembly & Dispatch**:
   - Rust renders `overview.html` (SVG circular gauge, probability percentages, vessel status cards).
   - Rust renders `shap.html` (feature bars and impact labels).
   - Sets HTTP header: `HX-Trigger: {"vesselRiskUpdated": {"LAD": 0.87, "LCX": 0.79, "RCA": 0.81}}`.
   - Sends HTML payload to browser with `hx-swap-oob="innerHTML"` for the `#shap-container`.
6. **Browser DOM & WebGL Updates**:
   - HTMX seamlessly updates the DOM without full-page reloads.
   - `heart_viewer.js` catches the `vesselRiskUpdated` custom event, calculates HSL color transitions on the GPU, sets emissive highlights, and updates 3D floating badge coordinates.

---

## 6. API / ROUTES

### 6.1. Primary FastAPI Application Server (`http://127.0.0.1:8000` via `api/main.py`)

| Method | Path | Handler Function | Handler File | Purpose | Auth & Middleware |
|---|---|---|---|---|---|
| `GET` | `/api/health` | `health_check` | `api/main.py` | Health check endpoint | None; `CORSMiddleware`, No-Cache headers |
| `GET` | `/api/schema` | `get_schema` | `api/main.py` | Serves schema registry JSON | None; `CORSMiddleware`, No-Cache headers |
| `GET` | `/api/metrics` | `get_metrics` | `api/main.py` | Serves CV metrics, global SHAP, model card | None; `CORSMiddleware`, No-Cache headers |
| `GET` | `/api/samples` | `get_samples` | `api/main.py` | Serves preset patient records | None; `CORSMiddleware`, No-Cache headers |
| `POST` | `/api/predict` | `predict_cardiac_risk` | `api/main.py` | Ingests JSON patient biomarkers, returns multi-task predictions & SHAP via `ml.service` | None; `CORSMiddleware`, Range bounds validation |
| `GET` | `/*` | `StaticFiles` | `web/dist/` | Serves precompiled React Single Page Application | None; Static file streaming |

---

### 6.2. Optional Rust Axum Workstation Server (`http://127.0.0.1:8000` in `extras/rust-htmx/server_rust/`)

| Method | Path | Handler Function | Handler File | Purpose | Auth & Middleware |
|---|---|---|---|---|---|
| `GET` | `/` | `render_index` | `server_rust/src/main.rs` | Renders full master HTML workstation shell | None; `CorsLayer::permissive()` |
| `POST` | `/api/predict` | `handle_predict` | `server_rust/src/main.rs` | Receives form data, calls sidecar, returns overview + SHAP HTML partials with `HX-Trigger` | None; `CorsLayer::permissive()` |
| `GET` | `/api/sample/:id` | `handle_get_sample` | `server_rust/src/main.rs` | Loads preset patient (A, B, C) and returns prefilled form partial with hidden inputs | None; `CorsLayer::permissive()` |
| `GET` | `/api/shap` | `handle_get_shap` | `server_rust/src/main.rs` | Returns isolated SHAP waterfall HTML partial for requested vessel target (`?target=LAD`) | None; `CorsLayer::permissive()` |
| `GET` | `/api/tab/:name` | `handle_get_tab` | `server_rust/src/main.rs` | Renders and swaps tab contents (`risk-shap`, `global-factors`, `cv-metrics`) | None; `CorsLayer::permissive()` |
| `GET` | `/static/*` | `ServeDir` | `server_rust/static/` | Serves CSS stylesheets, Three.js client scripts | None; Static file streaming |
| `GET` | `/models/*` | `ServeDir` | `server_rust/static/models/`| Serves binary 3D anatomical GLB mesh files | None; Static file streaming |

---

### 6.3. Optional Python ML Sidecar Microservice (`http://127.0.0.1:8001` in `extras/rust-htmx/sidecar.py`)

| Method | Path | Handler Function | Handler File | Purpose | Auth & Middleware |
|---|---|---|---|---|---|
| `GET` | `/health` | `health` | `extras/rust-htmx/sidecar.py` | Health-check status endpoint | None; `CORSMiddleware` |
| `GET` | `/schema` | `get_schema` | `extras/rust-htmx/sidecar.py` | Returns the 54-feature metadata registry JSON | None; `CORSMiddleware` |
| `GET` | `/samples` | `get_samples` | `extras/rust-htmx/sidecar.py` | Returns 3 representative patient presets (Low, Mid, High risk) | None; `CORSMiddleware` |
| `GET` | `/metrics` | `get_metrics` | `extras/rust-htmx/sidecar.py` | Returns CV metrics, global SHAP rankings, and model card | None; `CORSMiddleware` |
| `POST` | `/predict` | `predict` | `extras/rust-htmx/sidecar.py` | Computes CAD/vessel risks, coherence fix, and SHAP via `ml.service` | None; `CORSMiddleware`; Range bounds validation |

---

## 7. DATABASE & DATA MODELS

### Database Architecture
Cardio3D AI does not utilize an external relational SQL or NoSQL database server. Because patient inference operates in a point-of-care, privacy-preserving clinical sandbox, state persistence is handled entirely via **in-memory immutable model bundles** and **declarative static JSON registries**.

### Data Models & Schemas

#### 1. Input Clinical Biomarkers (`ml/schema.json`)
The canonical feature registry defines 54 physiological biomarkers across 5 clinical categories:
- **Demographics** (5 features): `Age` (years), `Sex` (Male/Fmale), `Weight` (kg), `Length` (cm), `BMI` (kg/m²).
- **Symptoms & Physical Exam** (25 features): `DM` (Diabetes Mellitus, 0/1), `HTN` (Hypertension, 0/1), `Current Smoker` (0/1), `EX-Smoker` (0/1), `FH` (Family History, 0/1), `Obesity` (Y/N), `CRF` (Chronic Renal Failure, Y/N), `CVA` (Stroke history, Y/N), `Airway disease` (Y/N), `Thyroid Disease` (Y/N), `CHF` (Heart failure, Y/N), `DLP` (Dyslipidemia, Y/N), `BP` (Systolic BP, mmHg), `PR` (Pulse rate, bpm), `Edema` (0/1), `Weak Peripheral Pulse` (Y/N), `Lung rales` (Y/N), `Systolic Murmur` (Y/N), `Diastolic Murmur` (Y/N), `Typical Chest Pain` (0/1), `Dyspnea` (Y/N), `Function Class` (NYHA class 0-3), `Atypical` (Y/N), `Nonanginal` (Y/N), `LowTH Ang` (Y/N).
- **ECG Features** (7 features): `Q Wave` (0/1), `St Elevation` (0/1), `St Depression` (0/1), `Tinversion` (0/1), `LVH` (Left ventricular hypertrophy, Y/N), `Poor R Progression` (Y/N), `BBB` (Bundle branch block, N/LBBB/RBBB).
- **Laboratory Analyses** (14 features): `FBS` (Fasting blood sugar, mg/dL), `CR` (Serum creatinine, mg/dL), `TG` (Triglycerides, mg/dL), `LDL` (mg/dL), `HDL` (mg/dL), `BUN` (Blood urea nitrogen, mg/dL), `ESR` (mm/hr), `HB` (Hemoglobin, g/dL), `K` (Serum potassium, mEq/L), `Na` (Serum sodium, mEq/L), `WBC` (White blood cells, /µL), `Lymph` (%), `Neut` (%), `PLT` (Platelets, 10³/µL).
- **Echocardiography** (3 features): `EF-TTE` (Ejection fraction, %), `Region RWMA` (Regional wall motion abnormality count, 0-4), `VHD` (Valvular heart disease, N/mild/Moderate/Severe).
*Total active features*: 5 + 25 + 7 + 14 + 3 = 54.

#### 2. Target Labels
- `Cath`: Binary catheterization diagnosis ($1 = \text{CAD}, 0 = \text{Normal}$).
- `LAD`: Left Anterior Descending artery stenosis $\ge 50\%$ ($1 = \text{Stenotic}, 0 = \text{Normal}$).
- `LCX`: Left Circumflex artery stenosis $\ge 50\%$ ($1 = \text{Stenotic}, 0 = \text{Normal}$).
- `RCA`: Right Coronary Artery stenosis $\ge 50\%$ ($1 = \text{Stenotic}, 0 = \text{Normal}$).

#### 3. Prediction Output Schema (`PredictResponse`)
```json
{
  "cad": {
    "prob": 0.2351,
    "raw_prob": 0.0261,
    "coherent_prob": 0.2351,
    "label": "Low Risk",
    "high_sens_label": "Low Risk",
    "threshold": 0.50,
    "high_sensitivity_threshold": 0.604,
    "coherence_adjusted": true
  },
  "vessels": {
    "LAD": { "prob": 0.2351, "label": "Normal", "high_sens_label": "Normal", "threshold": 0.5, "high_sensitivity_threshold": 0.465 },
    "LCX": { "prob": 0.0604, "label": "Normal", "high_sens_label": "Normal", "threshold": 0.5, "high_sensitivity_threshold": 0.208 },
    "RCA": { "prob": 0.0182, "label": "Normal", "high_sens_label": "Normal", "threshold": 0.5, "high_sensitivity_threshold": 0.208 }
  },
  "explain": {
    "Cath": {
      "base_value": 1.685,
      "raw_score": -3.621,
      "reconstructed_score": -3.621,
      "additive_error": 0.000016,
      "features": [
        { "feature": "Typical Chest Pain", "label": "Typical Anginal Chest Pain", "group": "symptoms-exam", "unit": "", "value": 0, "shap": -0.749, "pct": 18.2 }
      ]
    },
    "LAD": { ... },
    "LCX": { ... },
    "RCA": { ... }
  },
  "disclaimer": "Decision support / educational use only — not a substitute for formal diagnostic imaging."
}
```

#### 4. Model Bundle Serialization (`ml/artifacts/model_bundle.joblib`)
A binary dictionary structure containing:
- `pipelines`: Dict mapping target strings to trained `sklearn.pipeline.Pipeline` objects.
- `explainers`: Dict mapping target strings to fitted `shap.Explainer` objects.
- `parent_mappings`: Dict mapping transformed column strings back to schema keys.
- `thresholds`: Operating threshold dictionaries including high-sensitivity cut-points ($\ge 90\%$).
- `coherence_fix`: Boolean indicating whether logical risk coherence enforcement is active.
- `dataset_sha256`: Hash signature of training data for reproducibility validation.

---

## 8. STATE MANAGEMENT & EXTERNAL SERVICES

### State Management Strategy

#### 1. Rust + HTMX Hypermedia State
- **Hypermedia as the Engine of Application State (HATEOAS)**:
  - Form state is held directly in the DOM inside `<form id="patient-input-form">`.
  - When switching presets, `GET /api/sample/:id` renders `form.html` populated with the new patient's visible controls and injects all remaining 42 features into hidden `<input type="hidden">` fields.
  - Form modifications trigger debounced POST requests (`hx-trigger="input delay:250ms, change, formUpdated from:body"`).
  - Out-of-band updates (`hx-swap-oob="innerHTML"`) allow a single HTTP response to simultaneously refresh the clinical gauge, the vessel summary cards, and the SHAP attribution waterfall.
- **3D Scene Material State**:
  - The Three.js WebGL canvas maintains its own rendering loop independent of DOM mutations.
  - The Rust backend emits custom HTTP triggers via `HX-Trigger: {"vesselRiskUpdated": {"LAD": ..., "LCX": ..., "RCA": ...}}`.
  - Client-side event listeners in `heart_viewer.js` receive the probabilities and update Three.js `MeshStandardMaterial` emissive and base colors without re-rendering or reloading the GLB mesh.

#### 2. React SPA State (`web/src/App.tsx`)
- Centralized React Hooks:
  - `useState<Record<string, any>>({})` holds the 54-feature patient biomarker dictionary.
  - `useState<PredictResponse | null>(null)` stores current ML inference.
  - `useState<string | null>(null)` tracks 3D-selected vessel spotlighting.
  - `useRef<any>(null)` coordinates a 250ms debounce timer for keystroke input.

### External Services & Integrations
- **Zero Cloud / External Telemetry**: The primary FastAPI + React application is fully self-contained in local bundles (`web/dist/`) and operates in air-gapped clinical environments. No external analytics, CDNs, or third-party APIs are called during execution.
- **IPC Communication**: Local HTTP socket IPC between the optional Rust server (`http://127.0.0.1:8000`) and the Python ML Sidecar (`http://127.0.0.1:8001`) in `extras/rust-htmx/`.

---

## 9. CONFIGURATION & ENVIRONMENT

### Configuration Files
- **`extras/rust-htmx/server_rust/Cargo.toml`**: Configures Rust edition (`2021`), optimization levels, and crate dependencies (`axum`, `tokio`, `tower-http`, `tera`, `reqwest`, `serde`).
- **`web/package.json`**: Configures Node.js build scripts, runtime dependencies (`three`, `@react-three/fiber`, `lucide-react`), and dev dependencies (`vite`, `tailwindcss`, `typescript`).
- **`web/vite.config.ts`**: Configures Vite development server, port, and proxy rules forwarding `/api` to port 8000.
- **`web/tailwind.config.js`**: Defines the dark medical workstation theme, cyan/emerald/rose/amber risk color tokens, and font families.
- **`web/postcss.config.js`**: Connects Tailwind CSS and Autoprefixer to the Vite CSS bundling pipeline.
- **`web/tsconfig.json`**: TypeScript configuration enabling strict type checking, JSX runtime, and ES2022 module resolution.
- **`Makefile`**: Target orchestrator for environment setup, model training, test suites, and report generation.
- **`.gitignore`**: Excludes Python virtual environments, compiled `.pyc`, `node_modules`, Vite `.vite`, Cargo `target/`, and operating system artifacts.

### Environment Variables
The application relies on predictable sensible defaults and does not require complex environment configuration:
- `PORT`: (Optional) Port for the primary HTTP server (default: `8000`).
- `HOST`: (Optional) Network interface binding (default: `127.0.0.1`).
- `SIDECAR_URL`: Internal URL for the Python ML sidecar (default: `http://127.0.0.1:8001`).
- `PYTHONIOENCODING`: Recommended `utf-8` on Windows systems to support medical symbols ($\ge, \mu$) in console logs.

---

## 10. DEPENDENCIES

### Python Dependencies (`ml/requirements.txt`)
- `scikit-learn>=1.5.0`: Core machine learning pipelines, standardizers, imputers, LogisticRegression, RandomForestClassifier, and cross-validation metrics.
- `xgboost>=2.0.0`: High-performance gradient boosted decision trees for non-linear LCX stenosis prediction.
- `shap>=0.45.0`: Exact `LinearExplainer` and `TreeExplainer` algorithms for mathematical feature attribution.
- `pandas>=2.2.0`: DataFrame manipulation and spreadsheet ingestion.
- `numpy>=1.26.0`: Numerical matrix operations, array slicing, and metric aggregations.
- `openpyxl>=3.1.0`: Engine for parsing the raw Excel clinical dataset.
- `joblib>=1.4.0`: Fast persistence of trained pipelines and explainers.
- `scipy>=1.13.0`: Statistical utilities and probability distributions.
- `matplotlib>=3.8.0`: Headless (`Agg`) plot generation for reliability calibration curves and PDF reporting.
- `fastapi>=0.110.0`: Asynchronous REST API framework powering `api/main.py` (and `extras/rust-htmx/sidecar.py`).
- `uvicorn>=0.28.0`: High-performance ASGI server for hosting FastAPI applications.
- `pydantic>=2.6.0`: Data validation and schema enforcement.
- `trimesh>=4.0.0`: 3D geometry processing library used to parse, center, and export BodyParts3D OBJ meshes into binary GLTF/GLB.

### Rust Dependencies (`extras/rust-htmx/server_rust/Cargo.toml`)
- `axum 0.7`: Ergonomic, asynchronous web framework built on Tokio and Tower.
- `tokio 1`: Multi-threaded asynchronous runtime.
- `tower-http 0.5`: Middleware for static directory serving (`ServeDir`), CORS, and tracing.
- `serde 1.0` & `serde_json 1.0`: High-speed JSON serialization and deserialization.
- `reqwest 0.12`: Asynchronous HTTP client for IPC requests to the Python ML sidecar.
- `tera 1.19`: Template engine based on Jinja2 / Django templates for server-side HTML rendering.
- `tracing 0.1` & `tracing-subscriber 0.3`: Structured asynchronous diagnostics logging.

### Frontend Dependencies (`web/package.json`)
- `react 18.3.1` & `react-dom 18.3.1`: Declarative UI component library.
- `three 0.170.0`: WebGL 3D graphics rendering engine.
- `@react-three/fiber 8.17.10`: React reconciler for Three.js.
- `@react-three/drei 9.120.4`: Helpers and abstractions for React Three Fiber (OrbitControls, Html).
- `lucide-react 0.475.0`: Clean, accessible icon system for medical user interfaces.
- `tailwindcss 3.4.19`: Utility-first CSS framework.
- `vite 6.2.0`: High-speed development server and production bundler.
- `typescript 5.7.3`: Static type checker.

---

## 11. BUILD, RUN & DEPLOY

### Local Execution Instructions

#### 1. Setup Environment
```bash
# Install Python dependencies
pip install -r ml/requirements.txt
```

#### 2. Build 3D Anatomical Heart Asset
```bash
# Assembles 129 BodyParts3D meshes into optimized 83,600-triangle heart.glb
python scripts/build_heart_glb.py
```

#### 3. Train Models & Generate Artifacts
```bash
# Executes 15-fold CV, verifies SHAP additivity, and writes ml/artifacts/
python -m ml.train
```

#### 4. Launching the Application

##### Option A: Unified FastAPI + React Stack (Primary Application)
```bash
# Build React frontend
cd web && npm install && npm run build && cd ..

# Launch FastAPI server
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```
Open **[http://127.0.0.1:8000](http://127.0.0.1:8000)** in any modern browser.

##### Option B: Rust + HTMX Stack (Optional Extra in `extras/rust-htmx/`)
```bash
# Terminal 1: Start Python ML Sidecar on port 8001
python -m uvicorn extras.rust-htmx.sidecar:app --host 127.0.0.1 --port 8001

# Terminal 2: Start Rust Axum Workstation on port 8000
cd extras/rust-htmx/server_rust
cargo run --release
```
Open **[http://127.0.0.1:8000](http://127.0.0.1:8000)** in any modern browser.

---

## 12. TESTING & QUALITY ASSURANCE

### Test Architecture

```text
CAD/
├── ml/tests/
│   └── test_leakage.py               # Data leakage & statistical baseline tests (Pytest)
├── api/tests/
│   └── test_api.py                   # API contract, validation & CPU latency tests (Pytest)
└── scripts/
    ├── test_rust_ui_playwright.py    # Rust + HTMX end-to-end browser test (Playwright)
    └── test_ui_playwright.py         # React UI end-to-end browser test (Playwright)
```

### Test Coverage Details

#### 1. Data Leakage & Sanity Testing (`ml/tests/test_leakage.py`)
- **Target Exclusion**: Verifies that `Cath`, `LAD`, `LCX`, and `RCA` are strictly absent from feature matrix $X$ ($54$ features verified).
- **Schema Parity**: Asserts that `schema.json` defines all 54 features present in $X$ with zero missing entries.
- **Chance-Level Shuffled Baseline**: Shuffles target labels and runs 5-fold cross-validation. Asserts that the resulting ROC-AUC drops to chance ($0.50 \pm 0.08$), confirming that no hidden target proxies exist.

#### 2. API Contract & Performance Testing (`api/tests/test_api.py`)
- **Schema Verification**: Confirms `GET /api/schema` returns all 54 items.
- **Preset Availability**: Confirms `GET /api/samples` provides Patient A, B, and C presets.
- **Contract Verification**: Validates structure of `POST /api/predict`, confirming presence of `cad`, `vessels`, `explain`, and clinical disclaimer.
- **Range Bounds Validation**: Asserts that out-of-range clinical parameters (e.g. $\text{Age} = 250$) trigger HTTP 422 with informative error messages.
- **Latency Benchmark**: Runs 50 consecutive calls to `/api/predict` on CPU. Asserts that $p95$ latency is well within budget ($p95 = 28.5\text{ ms} < 300\text{ ms}$).

#### 3. End-to-End Browser Testing (`scripts/test_rust_ui_playwright.py`)
- **Clinical Safety Disclaimer**: Asserts visibility of the prominent amber disclaimer banner.
- **3D Canvas Mount**: Verifies WebGL context and Three.js canvas initialization.
- **Preset Interactivity**: Tests clicking "Patient C (High Risk)" and asserts dynamic update to "High Risk" ($100\%$ CAD risk), then tests "Patient A (Low Risk)" and asserts dynamic update to "Low Risk" ($24\%$ CAD risk).
- **Tab Navigation**: Verifies swapping to the "Model CV Validation" tab and asserts presence of 15-fold CV metrics.
- **Vessel Spotlight & SHAP Switching**: Clicks the LAD vessel card and verifies that the SHAP waterfall dynamically switches to the "Left Anterior Descending (LAD) Explainer".

---

## 13. ARCHITECTURAL DIAGRAMS

### 13.1. High-Level System Architecture

```mermaid
graph TB
    subgraph Browser["User Browser / Clinical Workstation"]
        UI["Hypermedia UI (HTMX + HTML5)"]
        ThreeCanvas["3D WebGL Canvas (Three.js r170)"]
    end

    subgraph RustServer["Rust Axum Web Server (Port 8000)"]
        Router["Axum Asynchronous Router"]
        TeraEngine["Tera Template Engine (Jinja2)"]
        StaticServer["Tower ServeDir (Static & 3D GLB)"]
        IPCClient["Reqwest Async HTTP Client"]
    end

    subgraph MLService["Python ML Sidecar (Port 8001)"]
        FastAPISidecar["FastAPI Microservice (sidecar.py)"]
        Validator["Plausible Range Bounds Validator"]
        CoherenceEngine["Coherence & Threshold Engine"]
        
        subgraph ModelBundle["Loaded ML Artifacts (In-Memory)"]
            PipeCath["LogisticRegression (Cath)"]
            PipeLAD["RandomForest (LAD)"]
            PipeLCX["XGBoost (LCX)"]
            PipeRCA["LogisticRegression (RCA)"]
            ShapCath["LinearExplainer (Cath)"]
            ShapLAD["TreeExplainer (LAD)"]
            ShapLCX["TreeExplainer (LCX)"]
            ShapRCA["LinearExplainer (RCA)"]
        end
    end

    UI -- "HTTP GET /" --> Router
    UI -- "HTMX POST /api/predict (Debounced form)" --> Router
    UI -- "HTMX GET /api/sample/:id (Preset click)" --> Router
    UI -- "HTMX GET /api/shap?target=LAD" --> Router
    ThreeCanvas -- "Fetch /models/heart.glb" --> StaticServer

    Router --> TeraEngine
    Router --> IPCClient
    
    IPCClient -- "JSON over HTTP" --> FastAPISidecar
    FastAPISidecar --> Validator --> ModelBundle
    ModelBundle --> CoherenceEngine --> FastAPISidecar

    TeraEngine -- "Render HTML Partials" --> UI
    Router -- "HX-Trigger: vesselRiskUpdated" --> ThreeCanvas
```

### 13.2. Detailed Request & Data Flow Diagram

```mermaid
sequenceDiagram
    autonumber
    actor Clinician as Clinician / Researcher
    participant DOM as Browser DOM (HTMX)
    participant ThreeJS as WebGL Viewer (Three.js)
    participant Rust as Rust Axum Server (:8000)
    participant Sidecar as Python Sidecar (:8001)

    Clinician->>DOM: Selects "Patient C (High Risk)"
    DOM->>Rust: GET /api/sample/sample-high-risk
    Rust->>Sidecar: GET /samples
    Sidecar-->>Rust: Return Patient C 54-feature record
    Rust->>Rust: Render form.html with visible inputs & hidden fields
    Rust-->>DOM: Swap #patient-form-container & trigger form change
    
    DOM->>Rust: POST /api/predict (Form-encoded patient parameters)
    Rust->>Sidecar: POST /predict (JSON dictionary)
    Sidecar->>Sidecar: Range validation & Median imputation
    Sidecar->>Sidecar: Execute 4 ML Pipelines (Cath, LAD, LCX, RCA)
    Sidecar->>Sidecar: Enforce Coherence: max(Cath, LAD, LCX, RCA)
    Sidecar->>Sidecar: Compute exact additive SHAP for all targets
    Sidecar-->>Rust: Return JSON {cad, vessels, explain, disclaimer}
    
    Rust->>Rust: Render overview.html & shap.html via Tera
    Rust-->>DOM: HTML Partials + HX-Trigger: {"vesselRiskUpdated": {LAD: 0.87, LCX: 0.79, RCA: 0.81}}
    
    DOM->>DOM: Update Circular Gauge (>99%) & Vessel Cards (Red)
    DOM->>ThreeJS: Dispatch vesselRiskUpdated event
    ThreeJS->>ThreeJS: Interpolate mesh materials to Crimson Red (#ef4444)
    ThreeJS->>ThreeJS: Reposition floating 3D screen badges
```

### 13.3. Clinical Data & Schema Entity Model

```mermaid
erDiagram
    PATIENT_RECORD {
        int record_id PK
        float Age "Demographic (20-95)"
        string Sex "Male / Fmale"
        float BMI "Derived kg/m2"
        float BP "Systolic mmHg (80-220)"
        float PR "Pulse Rate bpm (40-150)"
        float FBS "Fasting Blood Sugar mg/dL"
        float CR "Serum Creatinine mg/dL"
        float TG "Triglycerides mg/dL"
        float LDL "LDL Cholesterol mg/dL"
        float HDL "HDL Cholesterol mg/dL"
        float EF_TTE "Ejection Fraction % (15-75)"
        int Region_RWMA "Wall motion abnormalities (0-5)"
        int Typical_Chest_Pain "Anginal symptoms (0/1)"
        int Q_Wave "ECG abnormality (0/1)"
        int St_Elevation "ECG abnormality (0/1)"
        int St_Depression "ECG abnormality (0/1)"
    }

    TARGET_OUTCOMES {
        int record_id FK
        int Cath "Overall CAD Diagnosis (0/1)"
        int LAD "LAD Stenosis >= 50% (0/1)"
        int LCX "LCX Stenosis >= 50% (0/1)"
        int RCA "RCA Stenosis >= 50% (0/1)"
    }

    PREDICTION_RESULTS {
        int record_id FK
        float Raw_CAD_Prob
        float Coherent_CAD_Prob
        string CAD_Label "Low Risk / High Risk"
        float LAD_Prob
        string LAD_Label "Normal / Stenotic"
        float LCX_Prob
        string LCX_Label "Normal / Stenotic"
        float RCA_Prob
        string RCA_Label "Normal / Stenotic"
    }

    SHAP_EXPLANATION {
        string target PK "Cath / LAD / LCX / RCA"
        float base_value "Model log-odds bias"
        float raw_score "Logit margin"
        float additive_error "Math error < 1e-4"
        json feature_contributions "Parent-aggregated values"
    }

    PATIENT_RECORD ||--|| TARGET_OUTCOMES : "evaluated_against"
    PATIENT_RECORD ||--|| PREDICTION_RESULTS : "generates"
    PREDICTION_RESULTS ||--|{ SHAP_EXPLANATION : "decomposed_by"
```

---

## 14. OBSERVATIONS & RECOMMENDATIONS

### Architectural Observations

#### 1. Dual Web Application Stacks
- **Observation**: The repository contains two parallel web applications:
  1. The **Python FastAPI + React 18 SPA** application (`api/` and `web/`).
  2. The **Rust (Axum) + HTMX + Three.js** workstation (`extras/rust-htmx/`).
- **Analysis**: The FastAPI + React SPA serves as the primary production workstation matching the quickstart workflow. It pairs a typed REST API with modular React Three Fiber 3D rendering. The Rust + HTMX stack offers an ultra-lightweight alternative.
- **Recommendation**: Standardize on the FastAPI + React application as the primary deployment target. The Rust + HTMX implementation is maintained under `extras/rust-htmx/` as an optional extra.

#### 2. Mesh Directory Redundancy (Resolved)
- **Status**: Deduplication completed. `3D MODAL/` has been removed. All asset compilation scripts (`scripts/build_heart_glb.py`) point strictly to canonical meshes in `data/raw/`, verified against `data/raw/CHECKSUMS`.

#### 3. Excel Dataset Redundancy (Resolved)
- **Status**: Deduplication completed. `DATASET/` has been removed. `data/raw/extention of Z-Alizadeh sani dataset.xlsx` is the single canonical dataset file.

#### 4. Model Selection & Non-Temporal Modeling Choice
- **Observation**: The initial prompt inquired about TimesFM 3.0.
- **Analysis**: Confirmed **TimesFM 3.0 was properly rejected**. The dataset comprises 303 independent, static cross-sectional patient observations. Imposing an autoregressive temporal foundation model like TimesFM on static patient rows would invent fictitious temporal dependencies. The chosen per-target model suite—LogisticRegression for Cath ($\text{AUC} = 0.929$) and RCA ($\text{AUC} = 0.733$), RandomForest for LAD ($\text{AUC} = 0.846$), and XGBoost for LCX ($\text{AUC} = 0.735$)—strictly adheres to clinical epidemiology principles.

#### 5. Security & Authentication Considerations
- **Observation**: The API endpoints currently operate without authentication and use permissive CORS (`allow_origins=["*"]`).
- **Analysis**: This is appropriate for a local desktop clinical workstation or research prototype. However, if deployed to a production hospital intranet:
  - Role-based access control (RBAC) should be introduced.
  - Endpoints should enforce mTLS or session JWTs.
  - Rate limiting should be added to `/api/predict`.

---
*Document generated automatically after thorough codebase exploration and verification. Reference: [ARCHITECTURE.md](file:///c:/Users/Abish/Desktop/CAD/ARCHITECTURE.md).*
