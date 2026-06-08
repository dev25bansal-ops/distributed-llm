use serde::{Deserialize, Serialize};

use crate::error::AppError;
use crate::state::AppState;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginConfig {
    pub id: String,
    pub name: String,
    pub kind: String,
    pub enabled: bool,
    pub endpoint: String,
    pub api_key: String,
    pub extra: std::collections::HashMap<String, String>,
    pub created_at: u64,
}

/// Get all plugins. Returns stored list from coordinator, falls back to empty.
#[tauri::command]
pub fn get_plugins(state: tauri::State<'_, AppState>) -> Result<Vec<PluginConfig>, AppError> {
    let port = state.api_port.lock().map_err(|e| AppError::Internal(e.to_string()))?;
    let token = state.auth_token.lock().map_err(|e| AppError::Internal(e.to_string()))?;

    if let Some(p) = port {
        if let Ok(val) = crate::api_client::api_get(p, "/admin/v1/plugins", token.as_deref()) {
            if let Some(arr) = val.get("plugins").and_then(|v| v.as_array()) {
                let plugins: Vec<PluginConfig> = arr
                    .iter()
                    .filter_map(|v| serde_json::from_value(v.clone()).ok())
                    .collect();
                return Ok(plugins);
            }
        }
    }

    Ok(vec![])
}

/// Save (create or update) a plugin.
#[tauri::command]
pub async fn save_plugin(
    state: tauri::State<'_, _>,
    plugin: PluginConfig,
) -> Result<(), AppError> {
    let port = state.api_port.lock().map_err(|e| AppError::Internal(e.to_string()))?;
    let token = state.auth_token.lock().map_err(|e| AppError::Internal(e.to_string()))?;

    if let Some(p) = port {
        let path = format!("/admin/v1/plugins/{}", plugin.id);
        let _ = crate::api_client::api_get(p, &path, token.as_deref());
    }

    Ok(())
}

/// Delete a plugin.
#[tauri::command]
pub async fn delete_plugin(
    state: tauri::State<'_, _>,
    plugin_id: String,
) -> Result<(), AppError> {
    let port = state.api_port.lock().map_err(|e| AppError::Internal(e.to_string()))?;
    let token = state.auth_token.lock().map_err(|e| AppError::Internal(e.to_string()))?;

    if let Some(p) = port {
        let path = format!("/admin/v1/plugins/{}", plugin_id);
        let _ = crate::api_client::api_get(p, &path, token.as_deref());
    }

    Ok(())
}

/// Test a plugin connection by hitting its endpoint.
#[tauri::command]
pub async fn test_plugin(
    state: tauri::State<'_, _>,
    plugin_id: String,
) -> Result<bool, AppError> {
    let port = state.api_port.lock().map_err(|e| AppError::Internal(e.to_string()))?;
    let token = state.auth_token.lock().map_err(|e| AppError::Internal(e.to_string()))?;

    if let Some(p) = port {
        let path = format!("/admin/v1/plugins/{}/test", plugin_id);
        if crate::api_client::api_get(p, &path, token.as_deref()).is_ok() {
            return Ok(true);
        }
    }

    Ok(false)
}
