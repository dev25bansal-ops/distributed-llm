use serde::Serialize;
use tauri::{AppHandle, Emitter};

use crate::api_client::api_get;
use crate::error::AppError;
use crate::process::validate_model_name;
use crate::state::AppState;
use crate::types::ModelInfo;

#[derive(Clone, Serialize)]
pub struct DownloadProgress {
    pub model_id: String,
    pub status: String,
    pub detail: String,
}

/// Guard to remove model from active_downloads on scope exit.
struct DownloadGuard<'a> {
    state: &'a AppState,
    model_id: String,
}

impl Drop for DownloadGuard<'_> {
    fn drop(&mut self) {
        if let Ok(mut downloads) = self.state.active_downloads.lock() {
            downloads.remove(&self.model_id);
        }
    }
}

/// List models from the coordinator's OpenAI-compatible `/v1/models` endpoint.
///
/// The old implementation shelled out to `distllm models list --json`. That
/// command does not exist — the CLI group is `model` (singular) and its `list`
/// takes no `--json` flag — so it returned exit code 2 and the command failed
/// unconditionally. The coordinator API is the real source of truth.
#[tauri::command]
pub fn list_models(state: tauri::State<'_, AppState>) -> Result<Vec<ModelInfo>, AppError> {
    let port = *state.api_port.lock().map_err(|e| AppError::Internal(e.to_string()))?;
    let token = state
        .auth_token
        .lock()
        .map_err(|e| AppError::Internal(e.to_string()))?
        .clone();

    let Some(p) = port else {
        return Ok(vec![]);
    };

    let val = api_get(p, "/v1/models", token.as_deref())?;
    let arr = val
        .get("data")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();

    Ok(arr
        .iter()
        .filter_map(|m| {
            let id = m.get("id").and_then(|v| v.as_str())?.to_string();
            Some(ModelInfo {
                name: id.rsplit('/').next().unwrap_or(&id).to_string(),
                id,
                size: String::new(),
                downloaded: true,
                quantization: vec![],
                gpu_required: String::new(),
            })
        })
        .collect())
}

/// Request that the coordinator load (download + initialise) a model.
///
/// The old implementation invoked `distllm models download --model <id>`,
/// which does not exist and exited with a "No such command 'models'" error.
/// Model acquisition lives on `POST /api/models/load`, which requires the
/// `model-admin` role and streams nothing, so there is no per-token progress
/// to emit; we report started/completed/failed around the request instead.
#[tauri::command]
pub async fn download_model(
    app: AppHandle,
    state: tauri::State<'_, AppState>,
    model_id: String,
) -> Result<String, AppError> {
    validate_model_name(&model_id)?;

    {
        let mut downloads = state
            .active_downloads
            .lock()
            .map_err(|e| AppError::Internal(e.to_string()))?;
        if downloads.contains(&model_id) {
            return Err(AppError::DownloadInProgress(model_id));
        }
        downloads.insert(model_id.clone());
    }

    let _guard = DownloadGuard {
        state: state.inner(),
        model_id: model_id.clone(),
    };

    let port = (*state
        .api_port
        .lock()
        .map_err(|e| AppError::Internal(e.to_string()))?)
    .ok_or(AppError::NotConnected)?;
    let token = state
        .auth_token
        .lock()
        .map_err(|e| AppError::Internal(e.to_string()))?
        .clone();

    let _ = app.emit(
        "download-progress",
        DownloadProgress {
            model_id: model_id.clone(),
            status: "started".into(),
            detail: "Requesting model load from coordinator...".into(),
        },
    );

    let body = serde_json::json!({ "model": model_id });
    // Blocking ureq call, matching the existing pattern in cluster.rs's async
    // create_cluster. (`tokio::task::spawn_blocking` is unavailable here: it
    // needs tokio's "rt" feature, which Cargo.toml does not enable.)
    let result = crate::api_client::api_post(port, "/api/models/load", token.as_deref(), &body);

    match result {
        Ok(val) if val.get("error").is_none() => {
            let _ = app.emit(
                "download-progress",
                DownloadProgress {
                    model_id: model_id.clone(),
                    status: "completed".into(),
                    detail: format!("Model {} load requested successfully", model_id),
                },
            );
            Ok(format!("Model {} load requested", model_id))
        }
        Ok(val) => {
            let detail = val
                .get("error")
                .and_then(|v| v.as_str())
                .unwrap_or("coordinator rejected the load request")
                .to_string();
            let _ = app.emit(
                "download-progress",
                DownloadProgress {
                    model_id: model_id.clone(),
                    status: "failed".into(),
                    detail: detail.clone(),
                },
            );
            Err(AppError::DownloadFailed(detail))
        }
        Err(e) => {
            let detail = e.to_string();
            let _ = app.emit(
                "download-progress",
                DownloadProgress {
                    model_id: model_id.clone(),
                    status: "failed".into(),
                    detail: detail.clone(),
                },
            );
            Err(AppError::DownloadFailed(detail))
        }
    }
}
