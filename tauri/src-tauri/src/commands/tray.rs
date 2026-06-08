use crate::tray;

/// Update the tray tooltip and icon from the frontend.
#[tauri::command]
pub fn update_tray_status(
    app: tauri::AppHandle,
    running: bool,
    node_count: usize,
    addr: Option<String>,
) {
    tray::update_tray_icon(&app, running);
    tray::update_tray_tooltip(&app, running, node_count, &addr);
    tray::rebuild_tray_menu(&app, running);
}

/// Push a cluster address to the recent clusters list.
#[tauri::command]
pub fn add_recent_cluster(addr: String) {
    tray::push_recent_cluster(addr);
}
