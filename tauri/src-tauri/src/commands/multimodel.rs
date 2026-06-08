use serde::{Deserialize, Serialize};

use crate::api_client::api_get;
use crate::error::AppError;
use crate::state::AppState;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelSlot {
    pub id: String,
    pub model_id: String,
    pub model_name: String,
    pub status: String,
    pub vram_allocated_mb: u64,
    pub max_context: u32,
    pub requests_served: u64,
    pub avg_tokens_per_sec: f64,
    pub error_message: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelRoutingRule {
    pub id: String,
    pub pattern: String,
    pub target_slot: String,
    pub priority: i32,
}

/// Get model slots from coordinator API, with fallback to local defaults.
#[tauri::command]
pub fn get_model_slots(state: tauri::State<'_, AppState>) -> Result<Vec<ModelSlot>, AppError> {
    let port = state.api_port.lock().map_err(|e| AppError::Internal(e.to_string()))?;
    let token = state.auth_token.lock().map_err(|e| AppError::Internal(e.to_string()))?;

    if let Some(p) = port {
        if let Ok(val) = api_get(p, "/admin/v1/models/slots", token.as_deref()) {
            if let Some(arr) = val.get("slots").and_then(|v| v.as_array()) {
                let slots: Vec<ModelSlot> = arr
                    .iter()
                    .map(|s| ModelSlot {
                        id: s.get("id").and_then(|v| v.as_str()).unwrap_or("").into(),
                        model_id: s.get("model_id").and_then(|v| v.as_str()).unwrap_or("").into(),
                        model_name: s.get("model_name").and_then(|v| v.as_str()).unwrap_or("").into(),
                        status: s.get("status").and_then(|v| v.as_str()).unwrap_or("unloaded").into(),
                        vram_allocated_mb: s.get("vram_allocated_mb").and_then(|v| v.as_u64()).unwrap_or(0),
                        max_context: s.get("max_context").and_then(|v| v.as_u64()).unwrap_or(2048) as u32,
                        requests_served: s.get("requests_served").and_then(|v| v.as_u64()).unwrap_or(0),
                        avg_tokens_per_sec: s.get("avg_tokens_per_sec").and_then(|v| v.as_f64()).unwrap_or(0.0),
                        error_message: s.get("error_message").and_then(|v| v.as_str()).map(|s| s.into()),
                    })
                    .collect();
                return Ok(slots);
            }
        }
    }

    // Fallback: return default empty slots
    Ok(vec![
        ModelSlot {
            id: "slot-1".into(),
            model_id: String::new(),
            model_name: String::new(),
            status: "unloaded".into(),
            vram_allocated_mb: 0,
            max_context: 2048,
            requests_served: 0,
            avg_tokens_per_sec: 0.0,
            error_message: None,
        },
        ModelSlot {
            id: "slot-2".into(),
            model_id: String::new(),
            model_name: String::new(),
            status: "unloaded".into(),
            vram_allocated_mb: 0,
            max_context: 2048,
            requests_served: 0,
            avg_tokens_per_sec: 0.0,
            error_message: None,
        },
    ])
}

/// Load a model into a slot via coordinator API.
#[tauri::command]
pub async fn load_model_slot(
    state: tauri::State<'_, _>,
    slot_id: String,
    model_id: String,
) -> Result<ModelSlot, AppError> {
    let port = state.api_port.lock().map_err(|e| AppError::Internal(e.to_string()))?;
    let token = state.auth_token.lock().map_err(|e| AppError::Internal(e.to_string()))?;

    if let Some(p) = port {
        let path = format!("/admin/v1/models/slots/{}/load?model={}", slot_id, model_id);
        if api_get(p, &path, token.as_deref()).is_ok() {
            return Ok(ModelSlot {
                id: slot_id,
                model_id,
                model_name: String::new(),
                status: "loading".into(),
                vram_allocated_mb: 0,
                max_context: 2048,
                requests_served: 0,
                avg_tokens_per_sec: 0.0,
                error_message: None,
            });
        }
    }

    Err(AppError::ApiUnreachable)
}

/// Unload a model from a slot.
#[tauri::command]
pub async fn unload_model_slot(
    state: tauri::State<'_, _>,
    slot_id: String,
) -> Result<(), AppError> {
    let port = state.api_port.lock().map_err(|e| AppError::Internal(e.to_string()))?;
    let token = state.auth_token.lock().map_err(|e| AppError::Internal(e.to_string()))?;

    if let Some(p) = port {
        let path = format!("/admin/v1/models/slots/{}/unload", slot_id);
        if api_get(p, &path, token.as_deref()).is_ok() {
            return Ok(());
        }
    }

    Err(AppError::ApiUnreachable)
}

/// Get routing rules from coordinator.
#[tauri::command]
pub fn get_routing_rules(state: tauri::State<'_, AppState>) -> Result<Vec<ModelRoutingRule>, AppError> {
    let port = state.api_port.lock().map_err(|e| AppError::Internal(e.to_string()))?;
    let token = state.auth_token.lock().map_err(|e| AppError::Internal(e.to_string()))?;

    if let Some(p) = port {
        if let Ok(val) = api_get(p, "/admin/v1/models/rules", token.as_deref()) {
            if let Some(arr) = val.get("rules").and_then(|v| v.as_array()) {
                let rules: Vec<ModelRoutingRule> = arr
                    .iter()
                    .map(|r| ModelRoutingRule {
                        id: r.get("id").and_then(|v| v.as_str()).unwrap_or("").into(),
                        pattern: r.get("pattern").and_then(|v| v.as_str()).unwrap_or("").into(),
                        target_slot: r.get("target_slot").and_then(|v| v.as_str()).unwrap_or("").into(),
                        priority: r.get("priority").and_then(|v| v.as_i64()).unwrap_or(0) as i32,
                    })
                    .collect();
                return Ok(rules);
            }
        }
    }

    Ok(vec![])
}

/// Set (create or update) a routing rule.
#[tauri::command]
pub async fn set_routing_rule(
    state: tauri::State<'_, _>,
    rule: ModelRoutingRule,
) -> Result<(), AppError> {
    let port = state.api_port.lock().map_err(|e| AppError::Internal(e.to_string()))?;
    let token = state.auth_token.lock().map_err(|e| AppError::Internal(e.to_string()))?;

    if let Some(p) = port {
        let path = format!(
            "/admin/v1/models/rules/{}?pattern={}&target={}&priority={}",
            rule.id, rule.pattern, rule.target_slot, rule.priority
        );
        if api_get(p, &path, token.as_deref()).is_ok() {
            return Ok(());
        }
    }

    Err(AppError::ApiUnreachable)
}

/// Delete a routing rule.
#[tauri::command]
pub async fn delete_routing_rule(
    state: tauri::State<'_, _>,
    rule_id: String,
) -> Result<(), AppError> {
    let port = state.api_port.lock().map_err(|e| AppError::Internal(e.to_string()))?;
    let token = state.auth_token.lock().map_err(|e| AppError::Internal(e.to_string()))?;

    if let Some(p) = port {
        let path = format!("/admin/v1/models/rules/{}", rule_id);
        if api_get(p, &path, token.as_deref()).is_ok() {
            return Ok(());
        }
    }

    Err(AppError::ApiUnreachable)
}
