use axum::{
    extract::{Path as AxumPath, Query, State},
    http::{header, HeaderMap, HeaderValue, StatusCode},
    response::{Html, IntoResponse, Response},
    routing::{get, post},
    Form, Router,
};
use serde::Deserialize;
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use tera::{Context, Tera};
use tower_http::cors::CorsLayer;
use tower_http::services::ServeDir;

const SIDECAR_URL: &str = "http://127.0.0.1:8001";

#[derive(Clone)]
struct AppState {
    http_client: reqwest::Client,
    tera: Arc<Tera>,
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    let tera = match Tera::new("templates/**/*") {
        Ok(t) => Arc::new(t),
        Err(e) => {
            eprintln!("Tera template compilation error: {}", e);
            std::process::exit(1);
        }
    };

    let state = AppState {
        http_client: reqwest::Client::new(),
        tera,
    };

    let app = Router::new()
        .route("/", get(render_index))
        .route("/api/predict", post(handle_predict))
        .route("/api/sample/:id", get(handle_get_sample))
        .route("/api/shap", get(handle_get_shap))
        .route("/api/tab/:name", get(handle_get_tab))
        .nest_service("/static", ServeDir::new("static"))
        .nest_service("/models", ServeDir::new("static/models"))
        .layer(CorsLayer::permissive())
        .with_state(Arc::new(state));

    let addr = SocketAddr::from(([127, 0, 0, 1], 8000));
    println!("Cardio3D Rust (Axum) Server listening on http://{}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

async fn render_index(State(state): State<Arc<AppState>>) -> Result<Html<String>, StatusCode> {
    let samples_res = state
        .http_client
        .get(format!("{}/samples", SIDECAR_URL))
        .send()
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    let samples: Vec<serde_json::Value> = samples_res
        .json()
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    let initial_patient = samples
        .first()
        .and_then(|s| s.get("data"))
        .cloned()
        .unwrap_or(serde_json::json!({}));

    let pred_res = state
        .http_client
        .post(format!("{}/predict", SIDECAR_URL))
        .json(&initial_patient)
        .send()
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    let prediction: serde_json::Value = pred_res
        .json()
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    let overview_html = render_overview(&state.tera, &prediction)?;
    let form_html = render_form(&state.tera, &initial_patient)?;
    let shap_html = render_shap(&state.tera, &prediction, "Cath")?;

    let mut ctx = Context::new();
    ctx.insert("initial_overview_html", &overview_html);
    ctx.insert("initial_form_html", &form_html);
    ctx.insert("initial_shap_html", &shap_html);

    let html = state
        .tera
        .render("index.html", &ctx)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    Ok(Html(html))
}

async fn handle_predict(
    State(state): State<Arc<AppState>>,
    Form(form_data): Form<HashMap<String, String>>,
) -> Result<Response, StatusCode> {
    let mut payload = serde_json::Map::new();
    for (k, v) in form_data {
        if let Ok(num) = v.parse::<f64>() {
            payload.insert(k, serde_json::Value::Number(serde_json::Number::from_f64(num).unwrap()));
        } else {
            payload.insert(k, serde_json::Value::String(v));
        }
    }

    let sidecar_res = state
        .http_client
        .post(format!("{}/predict", SIDECAR_URL))
        .json(&payload)
        .send()
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    let prediction: serde_json::Value = sidecar_res
        .json()
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    let lad_prob = prediction["vessels"]["LAD"]["prob"].as_f64().unwrap_or(0.2);
    let lcx_prob = prediction["vessels"]["LCX"]["prob"].as_f64().unwrap_or(0.2);
    let rca_prob = prediction["vessels"]["RCA"]["prob"].as_f64().unwrap_or(0.2);

    let trigger_json = serde_json::json!({
        "vesselRiskUpdated": {
            "LAD": lad_prob,
            "LCX": lcx_prob,
            "RCA": rca_prob,
        }
    })
    .to_string();

    let overview_html = render_overview(&state.tera, &prediction)?;
    let shap_html = render_shap(&state.tera, &prediction, "Cath")?;

    let combined_html = format!(
        "{}\n<div id=\"shap-container\" hx-swap-oob=\"innerHTML\">{}</div>",
        overview_html, shap_html
    );

    let mut headers = HeaderMap::new();
    headers.insert("HX-Trigger", HeaderValue::from_str(&trigger_json).unwrap());
    headers.insert(header::CONTENT_TYPE, HeaderValue::from_static("text/html"));

    Ok((headers, combined_html).into_response())
}

async fn handle_get_sample(
    State(state): State<Arc<AppState>>,
    AxumPath(sample_id): AxumPath<String>,
) -> Result<Response, StatusCode> {
    let samples_res = state
        .http_client
        .get(format!("{}/samples", SIDECAR_URL))
        .send()
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    let samples: Vec<serde_json::Value> = samples_res
        .json()
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    let sample = samples
        .iter()
        .find(|s| s["id"] == sample_id)
        .and_then(|s| s.get("data"))
        .cloned()
        .unwrap_or(serde_json::json!({}));

    let form_html = render_form(&state.tera, &sample)?;

    let mut headers = HeaderMap::new();
    headers.insert("HX-Trigger", HeaderValue::from_static("formUpdated"));
    headers.insert(header::CONTENT_TYPE, HeaderValue::from_static("text/html"));

    Ok((headers, Html(form_html)).into_response())
}

#[derive(Deserialize)]
struct ShapQuery {
    target: Option<String>,
}

async fn handle_get_shap(
    State(state): State<Arc<AppState>>,
    Query(query): Query<ShapQuery>,
) -> Result<Html<String>, StatusCode> {
    let target = query.target.unwrap_or_else(|| "Cath".to_string());

    let samples_res = state
        .http_client
        .get(format!("{}/samples", SIDECAR_URL))
        .send()
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    let samples: Vec<serde_json::Value> = samples_res
        .json()
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    let sample_data = samples.first().and_then(|s| s.get("data")).unwrap();

    let pred_res = state
        .http_client
        .post(format!("{}/predict", SIDECAR_URL))
        .json(&sample_data)
        .send()
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    let prediction: serde_json::Value = pred_res
        .json()
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    let shap_html = render_shap(&state.tera, &prediction, &target)?;
    Ok(Html(shap_html))
}

async fn handle_get_tab(
    State(state): State<Arc<AppState>>,
    AxumPath(tab_name): AxumPath<String>,
) -> Result<Html<String>, StatusCode> {
    match tab_name.as_str() {
        "risk-shap" => {
            let samples_res = state.http_client.get(format!("{}/samples", SIDECAR_URL)).send().await.map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
            let samples: Vec<serde_json::Value> = samples_res.json().await.map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
            let initial_patient = samples.first().and_then(|s| s.get("data")).cloned().unwrap_or(serde_json::json!({}));
            let pred_res = state.http_client.post(format!("{}/predict", SIDECAR_URL)).json(&initial_patient).send().await.map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
            let prediction: serde_json::Value = pred_res.json().await.map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

            let overview = render_overview(&state.tera, &prediction)?;
            let form = render_form(&state.tera, &initial_patient)?;
            let shap = render_shap(&state.tera, &prediction, "Cath")?;

            let mut ctx = Context::new();
            ctx.insert("overview_html", &overview);
            ctx.insert("form_html", &form);
            ctx.insert("shap_html", &shap);

            let html = state.tera.render("risk_shap.html", &ctx).map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
            Ok(Html(html))
        }
        "global-factors" => {
            let metrics_res = state.http_client.get(format!("{}/metrics", SIDECAR_URL)).send().await.map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
            let metrics_data: serde_json::Value = metrics_res.json().await.map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
            let factors_arr = metrics_data["shap_global"]["Cath"]["top_features"].as_array();

            let mut factors = Vec::new();
            if let Some(list) = factors_arr {
                for (idx, item) in list.iter().take(12).enumerate() {
                    let feat = item["feature"].as_str().unwrap_or("");
                    let val = item["mean_abs_shap"].as_f64().unwrap_or(0.0);
                    let pct = (val / 0.85 * 100.0).min(100.0);
                    factors.push(serde_json::json!({
                        "rank": format!("{:02}", idx + 1),
                        "feature": feat,
                        "pct": format!("{:.1}", pct),
                        "val": format!("{:.4}", val),
                    }));
                }
            }

            let mut ctx = Context::new();
            ctx.insert("factors", &factors);
            let html = state.tera.render("global_factors.html", &ctx).map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
            Ok(Html(html))
        }
        "cv-metrics" => {
            let ctx = Context::new();
            let html = state.tera.render("cv_metrics.html", &ctx).map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
            Ok(Html(html))
        }
        _ => Err(StatusCode::NOT_FOUND),
    }
}

// Tera rendering helper functions

fn render_overview(tera: &Tera, prediction: &serde_json::Value) -> Result<String, StatusCode> {
    let cad = &prediction["cad"];
    let prob = cad["coherent_prob"].as_f64().unwrap_or(0.24);
    let pct = (prob * 100.0).round() as i64;
    let is_high = cad["label"].as_str().unwrap_or("") == "High Risk";
    let color = get_risk_hex(prob);

    let radius = 54.0;
    let circumference = 2.0 * std::f64::consts::PI * radius;
    let stroke_offset = circumference - (pct as f64 / 100.0) * circumference;

    let vessels = &prediction["vessels"];
    let lad_p = vessels["LAD"]["prob"].as_f64().unwrap_or(0.2);
    let lcx_p = vessels["LCX"]["prob"].as_f64().unwrap_or(0.2);
    let rca_p = vessels["RCA"]["prob"].as_f64().unwrap_or(0.2);

    let mut ctx = Context::new();
    ctx.insert("cad_label", cad["label"].as_str().unwrap_or("Low Risk"));
    ctx.insert("is_high", &is_high);
    ctx.insert("status_bg", if is_high { "rgba(239, 68, 68, 0.15)" } else { "rgba(16, 185, 129, 0.15)" });
    ctx.insert("status_color", if is_high { "#f87171" } else { "#34d399" });
    ctx.insert("status_border", if is_high { "rgba(239, 68, 68, 0.3)" } else { "rgba(16, 185, 129, 0.3)" });
    ctx.insert("cad_color", color);
    ctx.insert("cad_pct", &pct);
    ctx.insert("raw_cad_pct", &format!("{:.1}", cad["prob"].as_f64().unwrap_or(0.0) * 100.0));
    ctx.insert("circumference", &format!("{:.2}", circumference));
    ctx.insert("stroke_offset", &format!("{:.2}", stroke_offset));

    ctx.insert("lad_prob", &lad_p);
    ctx.insert("lad_pct", &format!("{:.0}", lad_p * 100.0));
    ctx.insert("lad_color", get_risk_hex(lad_p));

    ctx.insert("lcx_prob", &lcx_p);
    ctx.insert("lcx_pct", &format!("{:.0}", lcx_p * 100.0));
    ctx.insert("lcx_color", get_risk_hex(lcx_p));

    ctx.insert("rca_prob", &rca_p);
    ctx.insert("rca_pct", &format!("{:.0}", rca_p * 100.0));
    ctx.insert("rca_color", get_risk_hex(rca_p));

    tera.render("overview.html", &ctx).map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)
}

fn render_form(tera: &Tera, patient: &serde_json::Value) -> Result<String, StatusCode> {
    let get_num = |k: &str, def: f64| -> f64 { patient.get(k).and_then(|v| v.as_f64()).unwrap_or(def) };
    let get_str = |k: &str, def: &str| -> String { patient.get(k).and_then(|v| v.as_str()).unwrap_or(def).to_string() };

    let mut ctx = Context::new();
    ctx.insert("age", &format!("{:.0}", get_num("Age", 55.0)));
    ctx.insert("bp", &format!("{:.0}", get_num("BP", 120.0)));
    ctx.insert("pr", &format!("{:.0}", get_num("PR", 75.0)));
    ctx.insert("fbs", &format!("{:.0}", get_num("FBS", 100.0)));
    ctx.insert("cr", &format!("{:.1}", get_num("CR", 1.0)));
    ctx.insert("tg", &format!("{:.0}", get_num("TG", 150.0)));
    ctx.insert("ldl", &format!("{:.0}", get_num("LDL", 100.0)));
    ctx.insert("hdl", &format!("{:.0}", get_num("HDL", 45.0)));
    ctx.insert("ef", &format!("{:.0}", get_num("EF-TTE", 55.0)));
    ctx.insert("rwma", &format!("{:.0}", get_num("Region RWMA", 0.0)));
    ctx.insert("sex", &get_str("Sex", "Male"));

    let typ_cp_val = patient
        .get("Typical Chest Pain")
        .and_then(|v| v.as_i64().or_else(|| v.as_str().and_then(|s| s.parse().ok())))
        .unwrap_or(0);
    ctx.insert("typ_cp", &typ_cp_val);

    let explicit_keys = [
        "Age", "BP", "PR", "FBS", "CR", "TG", "LDL", "HDL", "EF-TTE", "Region RWMA", "Sex", "Typical Chest Pain"
    ];
    let mut hidden_fields = Vec::new();
    if let Some(obj) = patient.as_object() {
        for (k, v) in obj {
            if !explicit_keys.contains(&k.as_str()) {
                let val_str = if let Some(s) = v.as_str() {
                    s.to_string()
                } else if let Some(n) = v.as_f64() {
                    n.to_string()
                } else if let Some(i) = v.as_i64() {
                    i.to_string()
                } else if let Some(b) = v.as_bool() {
                    b.to_string()
                } else {
                    v.to_string()
                };
                hidden_fields.push(serde_json::json!({
                    "key": k,
                    "value": val_str,
                }));
            }
        }
    }
    ctx.insert("hidden_fields", &hidden_fields);

    tera.render("form.html", &ctx).map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)
}

fn render_shap(tera: &Tera, prediction: &serde_json::Value, target: &str) -> Result<String, StatusCode> {
    let expl = &prediction["explain"][target];
    let base_val = expl["base_value"].as_f64().unwrap_or(0.0);
    let features_arr = expl["features"].as_array();

    let mut features = Vec::new();
    if let Some(list) = features_arr {
        for feat in list.iter().take(8) {
            let name = feat["feature"].as_str().unwrap_or("");
            let shap_val = feat["shap"].as_f64().unwrap_or(0.0);
            let pct = feat["pct"].as_f64().unwrap_or(0.0);
            let val_str = feat["value"].to_string();

            let is_pos = shap_val >= 0.0;
            let bar_color = if is_pos { "#f87171" } else { "#34d399" };
            let bar_width = (pct * 2.5).min(100.0);

            features.push(serde_json::json!({
                "feature": name,
                "value": val_str,
                "shap_str": format!("{:+0.3}", shap_val),
                "pct_str": format!("{:.1}", pct),
                "bar_width": format!("{:.1}", bar_width),
                "color": bar_color,
            }));
        }
    }

    let title = match target {
        "Cath" => "Overall CAD",
        "LAD" => "Left Anterior Descending (LAD)",
        "LCX" => "Left Circumflex (LCX)",
        "RCA" => "Right Coronary Artery (RCA)",
        _ => target,
    };

    let mut ctx = Context::new();
    ctx.insert("target_title", title);
    ctx.insert("base_val", &format!("{:.3}", base_val));
    ctx.insert("features", &features);

    tera.render("shap.html", &ctx).map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)
}

fn get_risk_hex(prob: f64) -> &'static str {
    if prob < 0.25 {
        "#10b981"
    } else if prob < 0.50 {
        "#84cc16"
    } else if prob < 0.75 {
        "#f59e0b"
    } else {
        "#ef4444"
    }
}
