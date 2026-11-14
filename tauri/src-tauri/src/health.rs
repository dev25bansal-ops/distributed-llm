use std::time::Duration;

use tauri::{AppHandle, Emitter, Manager};

use crate::state::AppState;

/// Background health monitor that checks child process status every 5 seconds.
/// Emits "process-crashed" event if a managed process exits unexpectedly.
///
/// Resolves the managed [`AppState`] from the handle on every tick. An earlier
/// version took an `Arc<AppState>` built fresh in `setup()`, which shared no
/// mutexes with the state commands mutate, so it could never see a live child.
pub fn start_health_monitor(app: AppHandle) {
    std::thread::spawn(move || {
        loop {
            std::thread::sleep(Duration::from_secs(5));

            // Scope the (borrowing) State so it is released before the next
            // iteration — otherwise its MutexGuard temporaries outlive it.
            let alive = {
                let Some(state) = app.try_state::<AppState>() else {
                    // State was torn down (app shutting down) — stop watching.
                    break;
                };

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
                true
            };
            let _ = alive;
        }
    });
}
