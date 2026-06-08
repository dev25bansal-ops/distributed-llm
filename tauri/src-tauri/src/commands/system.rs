use std::process::Command;

use crate::error::AppError;
use crate::process::find_python;
use crate::state::AppState;
use crate::types::{InviteInfo, SystemInfo};

#[tauri::command]
pub fn generate_invite(state: tauri::State<'_, AppState>) -> Result<InviteInfo, AppError> {
    let code = uuid::Uuid::new_v4().to_string();
    let addr = state.api_port.lock().map_err(|e| AppError::Internal(e.to_string()))?;
    let port = addr.unwrap_or(8000);
    // Embed host:port in the invite link for deep link parsing
    // Default to localhost; real deployments should use actual IP
    let link = format!("distllm://connect/127.0.0.1:{}/{}", port, code);
    Ok(InviteInfo {
        code,
        link,
        qr_base64: String::new(),
    })
}

/// Get system info. Does NOT include GPU metrics — frontend fetches those separately
/// via `get_gpu_metrics()` to avoid double-querying NVML every poll cycle.
#[tauri::command]
pub fn get_system_info() -> Result<SystemInfo, AppError> {
    let python = find_python();
    let py_version = Command::new(python)
        .arg("--version")
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string());

    let distllm_version = Command::new(python)
        .args(["-m", "pip", "show", "distributed-llm"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| {
            let stdout = String::from_utf8_lossy(&o.stdout);
            stdout
                .lines()
                .find(|line| line.starts_with("Version:"))
                .map(|line| line.trim_start_matches("Version:").trim().to_string())
        })
        .unwrap_or_else(|| "unknown".into());

    let sys = sysinfo::System::new_all();
    let cpu = sys
        .cpus()
        .first()
        .map(|c| c.brand().to_string())
        .unwrap_or_else(|| "Unknown".into());
    let ram_gb = sys.total_memory() / (1024 * 1024 * 1024);

    Ok(SystemInfo {
        os: format!(
            "{} {}",
            std::env::consts::OS,
            sysinfo::System::long_os_version().unwrap_or_default()
        ),
        cpu,
        ram_gb,
        python_version: py_version,
        distllm_version,
        gpus: vec![], // Frontend fetches GPU metrics separately
    })
}
