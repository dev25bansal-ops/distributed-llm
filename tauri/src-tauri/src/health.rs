use std::sync::Arc;
use std::time::Duration;

use tauri::{AppHandle, Emitter};

use crate::state::AppState;

/// Background health monitor that checks child process status every 5 seconds.
/// Emits "process-crashed" event if a managed process exits unexpectedly.
pub fn start_health_monitor(app: AppHandle, state: Arc<AppState>) {
    std::thread::spawn(move || {
        loop {
            std::thread::sleep(Duration::from_secs(5));

            // Check coordinator process
            if let Ok(mut guard) = state.coordinator.lock() {
                if let Some(ref mut child) = *guard {
                    match child.try_wait() {
                        Ok(Some(status)) => {
                            eprintln!("[health] Coordinator exited unexpectedly: {}", status);
                            *guard = None;
                            let _ = app.emit("process-crashed", "coordinator");
                        }
                        Ok(None) => {} // Still running
                        Err(e) => {
                            eprintln!("[health] Failed to check coordinator: {}", e);
                        }
                    }
                }
            }

            // Check worker process
            if let Ok(mut guard) = state.worker.lock() {
                if let Some(ref mut child) = *guard {
                    match child.try_wait() {
                        Ok(Some(status)) => {
                            eprintln!("[health] Worker exited unexpectedly: {}", status);
                            *guard = None;
                            let _ = app.emit("process-crashed", "worker");
                        }
                        Ok(None) => {} // Still running
                        Err(e) => {
                            eprintln!("[health] Failed to check worker: {}", e);
                        }
                    }
                }
            }
        }
    });
}
