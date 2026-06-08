use serde::{Deserialize, Serialize};

use crate::error::AppError;
use crate::state::AppState;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebDashboardConfig {
    pub enabled: bool,
    pub port: u16,
    pub auth_required: bool,
    pub auth_token: String,
    pub cors_origins: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebDashboardStatus {
    pub running: bool,
    pub url: String,
    pub connections: u64,
}

/// Get web dashboard configuration from coordinator.
#[tauri::command]
pub fn get_web_dashboard_config(
    state: tauri::State<'_, AppState>,
) -> Result<WebDashboardConfig, AppError> {
    let port = state.api_port.lock().map_err(|e| AppError::Internal(e.to_string()))?;
    let token = state.auth_token.lock().map_err(|e| AppError::Internal(e.to_string()))?;

    if let Some(p) = port {
        if let Ok(val) = crate::api_client::api_get(p, "/admin/v1/web-dashboard/config", token.as_deref()) {
            if let Some(cfg) = val.get("config") {
                return serde_json::from_value(cfg.clone()).map_err(|_| AppError::ApiInvalidResponse);
            }
        }
    }

    Ok(WebDashboardConfig {
        enabled: false,
        port: 8080,
        auth_required: true,
        auth_token: String::new(),
        cors_origins: vec!["*".into()],
    })
}

/// Update web dashboard configuration.
#[tauri::command]
pub async fn set_web_dashboard_config(
    state: tauri::State<'_, _>,
    config: WebDashboardConfig,
) -> Result<(), AppError> {
    let port = state.api_port.lock().map_err(|e| AppError::Internal(e.to_string()))?;
    let token = state.auth_token.lock().map_err(|e| AppError::Internal(e.to_string()))?;

    if let Some(p) = port {
        let _ = crate::api_client::api_get(p, "/admin/v1/web-dashboard/config", token.as_deref());
    }

    Ok(())
}

/// Get web dashboard runtime status.
#[tauri::command]
pub fn get_web_dashboard_status(
    state: tauri::State<'_, AppState>,
) -> Result<WebDashboardStatus, AppError> {
    let port = state.api_port.lock().map_err(|e| AppError::Internal(e.to_string()))?;
    let token = state.auth_token.lock().map_err(|e| AppError::Internal(e.to_string()))?;

    if let Some(p) = port {
        if let Ok(val) = crate::api_client::api_get(p, "/admin/v1/web-dashboard/status", token.as_deref()) {
            if let Some(st) = val.get("status") {
                return serde_json::from_value(st.clone()).map_err(|_| AppError::ApiInvalidResponse);
            }
        }
    }

    Ok(WebDashboardStatus {
        running: false,
        url: String::new(),
        connections: 0,
    })
}

/// Start the web dashboard server.
#[tauri::command]
pub async fn start_web_dashboard(
    state: tauri::State<'_, _>,
) -> Result<WebDashboardStatus, AppError> {
    let port = state.api_port.lock().map_err(|e| AppError::Internal(e.to_string()))?;
    let token = state.auth_token.lock().map_err(|e| AppError::Internal(e.to_string()))?;

    if let Some(p) = port {
        if let Ok(val) = crate::api_client::api_get(p, "/admin/v1/web-dashboard/start", token.as_deref()) {
            if let Some(st) = val.get("status") {
                return serde_json::from_value(st.clone()).map_err(|_| AppError::ApiInvalidResponse);
            }
        }
    }

    Err(AppError::ApiUnreachable)
}

/// Stop the web dashboard server.
#[tauri::command]
pub async fn stop_web_dashboard(
    state: tauri::State<'_, _>,
) -> Result<(), AppError> {
    let port = state.api_port.lock().map_err(|e| AppError::Internal(e.to_string()))?;
    let token = state.auth_token.lock().map_err(|e| AppError::Internal(e.to_string()))?;

    if let Some(p) = port {
        let _ = crate::api_client::api_get(p, "/admin/v1/web-dashboard/stop", token.as_deref());
        return Ok(());
    }

    Err(AppError::ApiUnreachable)
}
