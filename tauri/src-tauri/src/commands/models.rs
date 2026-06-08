use std::io::{BufRead, BufReader};
use std::process::{Command, Stdio};

use serde::Serialize;
use tauri::{AppHandle, Emitter};

use crate::error::AppError;
use crate::process::ensure_distllm;
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

/// List available models. Async because it spawns a Python subprocess.
#[tauri::command]
pub async fn list_models() -> Result<Vec<ModelInfo>, AppError> {
    let python = ensure_distllm()?;
    let output = tokio::task::spawn_blocking(move || {
        Command::new(python)
            .args(["-m", "distllm.cli.main", "models", "list", "--json"])
            .output()
    })
    .await
    .map_err(|e| AppError::Internal(e.to_string()))?
    .map_err(|e| AppError::SpawnFailed(format!("models list: {}", e)))?;

    if output.status.success() {
        let stdout = String::from_utf8_lossy(&output.stdout);
        if stdout.trim().is_empty() {
            return Ok(vec![]);
        }
        serde_json::from_str(&stdout).map_err(|_| AppError::ApiInvalidResponse)
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr);
        Err(AppError::CliError(stderr.trim().to_string()))
    }
}

/// Download model with progress streaming. Async because it runs a long subprocess.
#[tauri::command]
pub async fn download_model(
    app: AppHandle,
    state: tauri::State<'_, AppState>,
    model_id: String,
) -> Result<String, AppError> {
    // Prevent duplicate downloads
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
        state: &state,
        model_id: model_id.clone(),
    };

    let python = ensure_distllm()?;
    let mut child = Command::new(python)
        .args([
            "-m",
            "distllm.cli.main",
            "models",
            "download",
            "--model",
            &model_id,
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| AppError::SpawnFailed(format!("model download: {}", e)))?;

    let _ = app.emit(
        "download-progress",
        DownloadProgress {
            model_id: model_id.clone(),
            status: "started".into(),
            detail: "Starting download...".into(),
        },
    );

    // Stream stderr for progress updates
    let stderr = child.stderr.take();
    let app_clone = app.clone();
    let model_clone = model_id.clone();

    let progress_handle = stderr.map(|stderr| {
        std::thread::spawn(move || {
            let reader = BufReader::new(stderr);
            for line in reader.lines() {
                if let Ok(line) = line {
                    let _ = app_clone.emit(
                        "download-progress",
                        DownloadProgress {
                            model_id: model_clone.clone(),
                            status: "downloading".into(),
                            detail: line,
                        },
                    );
                }
            }
        })
    });

    // Wait for process in a blocking thread to not block the async runtime
    let output = tokio::task::spawn_blocking(move || child.wait_with_output())
        .await
        .map_err(|e| AppError::Internal(e.to_string()))?
        .map_err(|e| AppError::DownloadFailed(e.to_string()))?;

    if let Some(handle) = progress_handle {
        let _ = handle.join();
    }

    if output.status.success() {
        let _ = app.emit(
            "download-progress",
            DownloadProgress {
                model_id: model_id.clone(),
                status: "completed".into(),
                detail: format!("Model {} downloaded successfully", model_id),
            },
        );
        Ok(format!("Model {} downloaded successfully", model_id))
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let _ = app.emit(
            "download-progress",
            DownloadProgress {
                model_id: model_id.clone(),
                status: "failed".into(),
                detail: stderr.trim().to_string(),
            },
        );
        Err(AppError::DownloadFailed(stderr.trim().to_string()))
    }
}
