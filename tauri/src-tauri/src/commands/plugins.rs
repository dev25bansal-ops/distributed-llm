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

/// Get all plugins from the coordinator's real `/v1/plugins` endpoint.
///
/// The previous implementation hit `/admin/v1/plugins`, which does not exist
/// in the backend, so this always returned an empty list. The real endpoint
/// reports installed plugins as {name, version, description, state, hooks};
/// enable/disable are separate POST actions keyed by name.
#[tauri::command]
pub fn get_plugins(state: tauri::State<'_, AppState>) -> Result<Vec<PluginConfig>, AppError> {
    let port = *state.api_port.lock().map_err(|e| AppError::Internal(e.to_string()))?;
    let token = state.auth_token.lock().map_err(|e| AppError::Internal(e.to_string()))?;

    if let Some(p) = port {
        if let Ok(val) = crate::api_client::api_get(p, "/v1/plugins", token.as_deref()) {
            if let Some(arr) = val.get("plugins").and_then(|v| v.as_array()) {
                let plugins: Vec<PluginConfig> = arr
                    .iter()
                    .filter_map(|v| {
                        let name = v.get("name").and_then(|x| x.as_str())?.to_string();
                        Some(PluginConfig {
                            id: name.clone(),
                            name,
                            kind: "backend".to_string(),
                            enabled: v
                                .get("state")
                                .and_then(|x| x.as_str())
                                .map(|s| s == "active")
                                .unwrap_or(false),
                            endpoint: v
                                .get("description")
                                .and_then(|x| x.as_str())
                                .unwrap_or("")
                                .to_string(),
                            api_key: String::new(),
                            extra: std::collections::HashMap::new(),
                            created_at: 0,
                        })
                    })
                    .collect();
                return Ok(plugins);
            }
        }
    }

    Ok(vec![])
}

/// Save (create or update) a plugin. The coordinator's plugin API can only
/// enable or disable existing plugins by name; it cannot create or edit them.
#[tauri::command]
pub async fn save_plugin(
    state: tauri::State<'_, AppState>,
    plugin: PluginConfig,
) -> Result<(), AppError> {
    let port = *state.api_port.lock().map_err(|e| AppError::Internal(e.to_string()))?;
    let token = state.auth_token.lock().map_err(|e| AppError::Internal(e.to_string()))?;

    if let Some(p) = port {
        let action = if plugin.enabled { "enable" } else { "disable" };
        let path = format!("/v1/plugins/{}/{}", plugin.id, action);
        crate::api_client::api_post_empty(p, &path, token.as_deref())
            .map_err(|_| AppError::ApiUnreachable)?;
        return Ok(());
    }

    Err(AppError::ApiUnreachable)
}

/// Delete a plugin. Not supported by the coordinator — there is no delete
/// endpoint; plugins are built in and can only be enabled or disabled.
#[tauri::command]
pub async fn delete_plugin(
    _state: tauri::State<'_, AppState>,
    _plugin_id: String,
) -> Result<(), AppError> {
    Err(AppError::Internal(
        "Deleting plugins is not supported by the coordinator; disable it instead.".into(),
    ))
}

/// Test a plugin. The coordinator exposes no per-plugin health check, so
/// this reports the plugin as absent rather than pretending it passed.
#[tauri::command]
pub async fn test_plugin(
    state: tauri::State<'_, AppState>,
    plugin_id: String,
) -> Result<bool, AppError> {
    let port = *state.api_port.lock().map_err(|e| AppError::Internal(e.to_string()))?;
    let token = state.auth_token.lock().map_err(|e| AppError::Internal(e.to_string()))?;

    if let Some(p) = port {
        if let Ok(val) = crate::api_client::api_get(p, "/v1/plugins", token.as_deref()) {
            return Ok(val
                .get("plugins")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter().any(|x| {
                        x.get("name").and_then(|n| n.as_str()) == Some(plugin_id.as_str())
                            && x.get("state").and_then(|s| s.as_str()) == Some("active")
                    })
                })
                .unwrap_or(false));
        }
    }

    Ok(false)
}
