use std::process::{Command, Stdio};

use tauri::Emitter;

use crate::api_client::{api_get, fetch_from_api};
use crate::error::AppError;
use crate::process::{ensure_distllm, validate_host, validate_model_name};
use crate::state::AppState;
use crate::tray::update_tray_icon;
use crate::types::ClusterStatus;

/// Create a new cluster. Async because it spawns a process and polls for readiness.
#[tauri::command]
pub async fn create_cluster(
    app: tauri::AppHandle,
    state: tauri::State<'_, AppState>,
    port: Option<u16>,
    model: Option<String>,
) -> Result<ClusterStatus, AppError> {
    let port = port.unwrap_or(8000);
    let mut coord = state.coordinator.lock().map_err(|e| AppError::Internal(e.to_string()))?;
    if coord.is_some() {
        return Err(AppError::ClusterAlreadyRunning);
    }

    let python = ensure_distllm()?;

    let mut args = vec![
        "-m".into(),
        "distllm.cli.main".into(),
        "cluster".into(),
        "start".into(),
        "--port".into(),
        port.to_string(),
    ];
    if let Some(ref m) = model {
        validate_model_name(m)?;
        args.extend(["--model".into(), m.clone()]);
    }

    let child = Command::new(python)
        .args(&args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| AppError::SpawnFailed(format!("coordinator: {}", e)))?;

    *coord = Some(child);
    let cid = format!("cluster-{}", &uuid::Uuid::new_v4().to_string()[..8]);
    let token = uuid::Uuid::new_v4().to_string();
    *state.cluster_id.lock().map_err(|e| AppError::Internal(e.to_string()))? = Some(cid);
    *state.api_port.lock().map_err(|e| AppError::Internal(e.to_string()))? = Some(port);
    *state.auth_token.lock().map_err(|e| AppError::Internal(e.to_string()))? = Some(token.clone());

    // Async poll for readiness instead of blocking the main thread
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    while std::time::Instant::now() < deadline {
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
        if api_get(port, "/health", Some(&token)).is_ok() {
            break;
        }
    }
    update_tray_icon(&app, true);

    let status = fetch_from_api(port, Some(&token));
    Ok(status)
}

/// Join an existing cluster. Async because it spawns a process.
#[tauri::command]
pub async fn join_cluster(
    app: tauri::AppHandle,
    state: tauri::State<'_, AppState>,
    host: String,
    port: u16,
) -> Result<ClusterStatus, AppError> {
    validate_host(&host)?;

    let mut worker = state.worker.lock().map_err(|e| AppError::Internal(e.to_string()))?;
    if worker.is_some() {
        return Err(AppError::AlreadyConnected);
    }

    let python = ensure_distllm()?;

    let child = Command::new(python)
        .args([
            "-m",
            "distllm.cli.main",
            "cluster",
            "join",
            "--coordinator",
            &format!("{}:{}", host, port),
            "--discover",
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| AppError::SpawnFailed(format!("worker: {}", e)))?;

    *worker = Some(child);
    update_tray_icon(&app, true);

    *state.api_port.lock().map_err(|e| AppError::Internal(e.to_string()))? = Some(port);

    Ok(ClusterStatus {
        running: true,
        node_id: Some("worker".into()),
        role: Some("worker".into()),
        coordinator_addr: Some(format!("http://{}:{}", host, port)),
        nodes: vec![],
        uptime_secs: 0,
    })
}

#[tauri::command]
pub fn leave_cluster(app: tauri::AppHandle, state: tauri::State<'_, AppState>) -> Result<(), AppError> {
    for child_opt in [&state.coordinator, &state.worker] {
        if let Ok(mut guard) = child_opt.lock() {
            if let Some(ref mut c) = *guard {
                let _ = c.kill();
                let _ = c.wait();
            }
            *guard = None;
        }
    }

    *state.cluster_id.lock().map_err(|e| AppError::Internal(e.to_string()))? = None;
    *state.api_port.lock().map_err(|e| AppError::Internal(e.to_string()))? = None;
    *state.auth_token.lock().map_err(|e| AppError::Internal(e.to_string()))? = None;

    update_tray_icon(&app, false);
    Ok(())
}

#[tauri::command]
pub fn get_cluster_status(state: tauri::State<'_, AppState>) -> Result<ClusterStatus, AppError> {
    let coord_running = state
        .coordinator
        .lock()
        .map_err(|e| AppError::Internal(e.to_string()))?
        .is_some();
    let worker_running = state
        .worker
        .lock()
        .map_err(|e| AppError::Internal(e.to_string()))?
        .is_some();
    let port = *state.api_port.lock().map_err(|e| AppError::Internal(e.to_string()))?;
    let token = state.auth_token.lock().map_err(|e| AppError::Internal(e.to_string()))?;

    if !coord_running && !worker_running {
        return Ok(ClusterStatus {
            running: false,
            node_id: None,
            role: None,
            coordinator_addr: None,
            nodes: vec![],
            uptime_secs: 0,
        });
    }

    match port {
        Some(p) => Ok(fetch_from_api(p, token.as_deref())),
        None => Ok(ClusterStatus {
            running: true,
            node_id: None,
            role: None,
            coordinator_addr: None,
            nodes: vec![],
            uptime_secs: 0,
        }),
    }
}

/// Check if a coordinator is running on the given host:port.
#[tauri::command]
pub async fn check_coordinator(host: String, port: u16) -> Result<bool, AppError> {
    match crate::api_client::api_get_to(&host, port, "/health", None) {
        Ok(_) => Ok(true),
        Err(_) => Ok(false),
    }
}
