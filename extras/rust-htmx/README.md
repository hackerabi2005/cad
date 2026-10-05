# Optional Stack: Rust (Axum) + HTMX + Three.js Medical Workstation

This directory contains an optional, ultra-lightweight alternative architecture for Cardio3D AI built with **Rust (Axum)**, **HTMX 2.0**, **Vanilla Three.js**, and a **Python ML Sidecar**.

---

## 1. Architectural Overview

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

## 2. Directory Inventory

```text
extras/rust-htmx/
├── README.md                     # This documentation file
├── sidecar.py                    # Lightweight Python FastAPI sidecar on port 8001
├── test_rust_ui_playwright.py    # Playwright browser verification for Rust + HTMX
├── screenshots/                  # High-resolution screenshots of the Rust UI
└── server_rust/                  # Rust Axum project
    ├── Cargo.toml                # Axum, Tokio, Tower, Tera, Reqwest dependencies
    ├── Cargo.lock
    ├── src/main.rs               # Axum server routes, Tera rendering, IPC handlers
    ├── templates/                # Tera HTML partial templates
    │   ├── index.html            # Main dashboard shell with persistent disclaimer
    │   ├── overview.html         # CAD gauge & LAD/LCX/RCA vessel risk cards
    │   ├── form.html             # Dynamic 54-biomarker input form with preset buttons
    │   ├── shap.html             # Top-8 SHAP local feature attribution waterfall
    │   ├── global_factors.html   # Cohort-wide global feature importance ranking
    │   └── cv_metrics.html       # 15-fold cross-validation performance table
    └── static/
        ├── css/style.css         # Dark glassmorphic medical workstation styling
        ├── js/heart_viewer.js    # Vanilla Three.js 3D viewer & color-mapping engine
        └── models/heart.glb      # 83,600-triangle 3D anatomical GLB model
```

---

## 3. Running the Rust + HTMX Stack

```bash
# Terminal 1: Launch Python ML sidecar (port 8001)
python -m uvicorn extras.rust-htmx.sidecar:app --host 127.0.0.1 --port 8001

# Terminal 2: Launch Rust Axum server (port 8000)
cd extras/rust-htmx/server_rust
cargo run --release
```

Open **[http://127.0.0.1:8000](http://127.0.0.1:8000)** in your browser.

---

## 4. Key Endpoints & IPC Contracts

- `GET /`: Renders initial dashboard with Patient A defaults pre-populated.
- `POST /predict`: Receives form submissions, queries `http://127.0.0.1:8001/predict` (which imports `predict_patient()` from `ml/service.py`), and returns multi-swap HTML partials (`#overview-container` and `#shap-container`).
- `GET /api/sample/:id`: Fetches preset patient data (`sample-low-risk`, `sample-mid-risk`, `sample-high-risk`), updates input fields, and triggers HTMX recalculation.
- `GET /tab/global-factors`: Renders cohort-wide global SHAP ranking partial.
- `GET /tab/cv-metrics`: Renders 15-fold CV metrics table partial.
