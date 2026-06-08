mod api_client;
mod commands;
pub mod error;
mod health;
mod process;
mod state;
mod tray;
mod types;

use std::sync::Arc;
use tauri::Manager;

pub use error::AppError;
pub use state::AppState;
pub use types::*;

pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_deep_link::init())
        .manage(AppState::new())
        .setup(|app| {
            tray::setup_tray(app)?;

            // 5.8: Register deep link handler for distllm:// URLs
            let handle = app.handle().clone();
            app.listen("deep-link://new-url", move |event| {
                if let Some(urls) = event.payload().as_str() {
                    // Payload is a JSON array of URLs, parse the first one
                    if let Ok(parsed) = serde_json::from_str::<Vec<String>>(urls) {
                        if let Some(url) = parsed.first() {
                            let _ = handle.emit("deep-link-connect", url.as_str());
                            if let Some(window) = handle.get_webview_window("main") {
                                let _ = window.show();
                                let _ = window.set_focus();
                            }
                        }
                    }
                }
            });

            // 3.5: Start background health monitor for process crash detection
            let state = app.state::<AppState>();
            let state_arc = Arc::new(AppState::new());
            health::start_health_monitor(app.handle().clone(), state_arc);

            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            commands::cluster::create_cluster,
            commands::cluster::join_cluster,
            commands::cluster::leave_cluster,
            commands::cluster::get_cluster_status,
            commands::cluster::check_coordinator,
            commands::gpu::get_gpu_metrics,
            commands::models::list_models,
            commands::models::download_model,
            commands::multimodel::get_model_slots,
            commands::multimodel::load_model_slot,
            commands::multimodel::unload_model_slot,
            commands::multimodel::get_routing_rules,
            commands::multimodel::set_routing_rule,
            commands::multimodel::delete_routing_rule,
            commands::plugins::get_plugins,
            commands::plugins::save_plugin,
            commands::plugins::delete_plugin,
            commands::plugins::test_plugin,
            commands::webdashboard::get_web_dashboard_config,
            commands::webdashboard::set_web_dashboard_config,
            commands::webdashboard::get_web_dashboard_status,
            commands::webdashboard::start_web_dashboard,
            commands::webdashboard::stop_web_dashboard,
            commands::discovery::get_discovered_services,
            commands::discovery::start_discovery,
            commands::discovery::stop_discovery,
            commands::discovery::get_discovery_status,
            commands::ollama::get_ollama_config,
            commands::ollama::check_ollama,
            commands::ollama::list_ollama_models,
            commands::ollama::ollama_chat,
            commands::ollama::pull_ollama_model,
            commands::system::generate_invite,
            commands::system::get_system_info,
            commands::tray::update_tray_status,
            commands::tray::add_recent_cluster,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
