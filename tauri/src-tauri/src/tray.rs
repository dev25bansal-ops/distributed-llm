use std::collections::VecDeque;

use tauri::{
    image::Image,
    menu::{Menu, MenuItem, PredefinedMenuItem, Submenu},
    tray::TrayIconBuilder,
    Emitter, Manager,
};

use crate::state::AppState;

/// Recent clusters stored for quick access.
pub static RECENT_CLUSTERS: once_cell::sync::Lazy<std::sync::Mutex<VecDeque<String>>> =
    once_cell::sync::Lazy::new(|| std::sync::Mutex::new(VecDeque::new()));

const MAX_RECENT: usize = 5;

pub fn push_recent_cluster(addr: String) {
    if let Ok(mut queue) = RECENT_CLUSTERS.lock() {
        queue.retain(|a| a != &addr);
        queue.push_front(addr);
        if queue.len() > MAX_RECENT {
            queue.pop_back();
        }
    }
}

/// Generate RGBA pixel data for a circular tray icon.
pub fn make_tray_rgba(size: u32, r: u8, g: u8, b: u8) -> Vec<u8> {
    let mut rgba = vec![0u8; (size * size * 4) as usize];
    let center = (size / 2) as f32;
    let radius = center - 2.0;
    for y in 0..size {
        for x in 0..size {
            let dx = x as f32 - center;
            let dy = y as f32 - center;
            let dist = (dx * dx + dy * dy).sqrt();
            let idx = ((y * size + x) * 4) as usize;
            if dist <= radius {
                rgba[idx] = r;
                rgba[idx + 1] = g;
                rgba[idx + 2] = b;
                rgba[idx + 3] = 255;
            } else if dist <= radius + 1.0 {
                let alpha = ((radius + 1.0 - dist) * 255.0) as u8;
                rgba[idx] = r;
                rgba[idx + 1] = g;
                rgba[idx + 2] = b;
                rgba[idx + 3] = alpha;
            }
        }
    }
    rgba
}

/// Update the tray icon color based on cluster state.
pub fn update_tray_icon(app: &tauri::AppHandle, active: bool) {
    let (r, g, b) = if active {
        (0x22, 0xcc, 0x66)
    } else {
        (0x66, 0x66, 0x66)
    };
    let rgba = make_tray_rgba(32, r, g, b);
    if let Some(tray) = app.tray_by_id("main-tray") {
        let _ = tray.set_icon(Some(Image::new(&rgba, 32, 32)));
    }
}

/// Update the tray tooltip with cluster status info.
pub fn update_tray_tooltip(
    app: &tauri::AppHandle,
    running: bool,
    node_count: usize,
    addr: &Option<String>,
) {
    let tooltip = if running {
        let base = format!("Distributed LLM — Running ({} node{})", node_count, if node_count == 1 { "" } else { "s" });
        if let Some(addr) = addr {
            format!("{}\n{}", base, addr)
        } else {
            base
        }
    } else {
        "Distributed LLM — Idle".to_string()
    };
    if let Some(tray) = app.tray_by_id("main-tray") {
        let _ = tray.set_tooltip(Some(&tooltip));
    }
}

/// Build the system tray context menu.
pub fn build_tray_menu(app: &tauri::AppHandle, running: bool) -> Result<Menu<tauri::Wry>, tauri::Error> {
    let show = MenuItem::with_id(app, "show", "Show Window", true, None::<&str>)?;
    let create =
        MenuItem::with_id(app, "create_cluster", "Create Cluster", true, None::<&str>)?;
    let join = MenuItem::with_id(
        app,
        "join_cluster",
        "Join Cluster...",
        true,
        None::<&str>,
    )?;
    let leave = MenuItem::with_id(
        app,
        "leave_cluster",
        "Leave Cluster",
        true,
        Some(!running),
    )?;

    // Recent clusters submenu
    let recent_menu = if let Ok(queue) = RECENT_CLUSTERS.lock() {
        if queue.is_empty() {
            None
        } else {
            let submenu = Submenu::with_id(app, "recent_clusters", "Recent Clusters", true)?;
            for addr in queue.iter() {
                let item = MenuItem::with_id(app, format!("recent_{}", addr), addr.as_str(), true, None::<&str>)?;
                submenu.append(&item)?;
            }
            Some(submenu)
        }
    } else {
        None
    };

    let sep = PredefinedMenuItem::separator(app)?;
    let quit = MenuItem::with_id(
        app,
        "quit",
        "Quit Distributed LLM",
        true,
        None::<&str>,
    )?;

    let mut items: Vec<&dyn tauri::menu::IsMenuItem<tauri::Wry>> = vec![
        &show,
        &create,
        &join,
        &leave,
    ];
    if let Some(ref recent) = recent_menu {
        items.push(recent);
    }
    items.push(&sep);
    items.push(&quit);

    Menu::with_items(app, &items)
}

/// Handle tray menu item clicks.
pub fn handle_tray_menu(app: &tauri::AppHandle, event_id: &str) {
    if event_id.starts_with("recent_") {
        let addr = event_id.strip_prefix("recent_").unwrap_or("");
        let _ = app.emit("deep-link-connect", addr);
        if let Some(window) = app.get_webview_window("main") {
            let _ = window.show();
            let _ = window.set_focus();
        }
        return;
    }

    match event_id {
        "show" | "create_cluster" | "join_cluster" => {
            let _ = app.emit("navigate", event_id);
            if let Some(window) = app.get_webview_window("main") {
                let _ = window.show();
                let _ = window.set_focus();
            }
        }
        "leave_cluster" => {
            let state = app.state::<AppState>();
            for child_opt in [&state.coordinator, &state.worker] {
                if let Ok(mut guard) = child_opt.lock() {
                    if let Some(ref mut c) = *guard {
                        let _ = c.kill();
                        let _ = c.wait();
                    }
                    *guard = None;
                }
            }
            if let Ok(mut guard) = state.cluster_id.lock() {
                *guard = None;
            }
            if let Ok(mut guard) = state.api_port.lock() {
                *guard = None;
            }
            if let Ok(mut guard) = state.auth_token.lock() {
                *guard = None;
            }
            update_tray_icon(app, false);
            update_tray_tooltip(app, false, 0, &None);
            let _ = app.emit("cluster-stopped", ());
        }
        "quit" => {
            let state = app.state::<AppState>();
            for child_opt in [&state.coordinator, &state.worker] {
                if let Ok(mut guard) = child_opt.lock() {
                    if let Some(ref mut c) = *guard {
                        let _ = c.kill();
                        let _ = c.wait();
                    }
                }
            }
            app.exit(0);
        }
        _ => {}
    }
}

/// Rebuild the tray menu (call after cluster state changes).
pub fn rebuild_tray_menu(app: &tauri::AppHandle, running: bool) {
    if let Some(tray) = app.tray_by_id("main-tray") {
        if let Ok(menu) = build_tray_menu(app, running) {
            let _ = tray.set_menu(Some(menu));
        }
    }
}

/// Set up the system tray icon and menu.
pub fn setup_tray(app: &mut tauri::App) -> Result<(), Box<dyn std::error::Error>> {
    let menu = build_tray_menu(app.handle(), false)?;

    let _tray = TrayIconBuilder::new()
        .icon(app.default_window_icon().unwrap().clone())
        .tooltip("Distributed LLM — Idle")
        .menu(&menu)
        .on_menu_event(|app, event| handle_tray_menu(app, event.id.as_ref()))
        .on_tray_icon_event(|tray, event| {
            if let tauri::tray::TrayIconEvent::Click {
                button: tauri::tray::MouseButton::Left,
                button_state: tauri::tray::MouseButtonState::Up,
                ..
            } = event
            {
                let app = tray.app_handle();
                if let Some(window) = app.get_webview_window("main") {
                    let _ = window.show();
                    let _ = window.set_focus();
                }
            }
        })
        .build(app)?;

    update_tray_icon(app.handle(), false);
    Ok(())
}
