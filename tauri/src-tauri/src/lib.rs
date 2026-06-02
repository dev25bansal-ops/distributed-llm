use serde::{Deserialize, Serialize};
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;
use tauri::{
    image::Image, menu::{Menu, MenuItem, PredefinedMenuItem},
    tray::TrayIconBuilder, Emitter, Manager,
};

// ===== Types =====

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuInfo {
    pub index: u32,
    pub name: String,
    pub temperature: f64,
    pub utilization: f64,
    pub memory_total: u64,
    pub memory_used: u64,
    pub memory_free: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterStatus {
    pub running: bool,
    pub node_id: Option<String>,
    pub role: Option<String>,
    pub coordinator_addr: Option<String>,
    pub nodes: Vec<PeerInfo>,
    pub uptime_secs: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeerInfo {
    pub node_id: String,
    pub host: String,
    pub port: u16,
    pub healthy: bool,
    pub gpu_name: String,
    pub gpu_utilization: f64,
    pub layers: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub id: String,
    pub name: String,
    pub size: String,
    pub downloaded: bool,
    pub quantization: Vec<String>,
    pub gpu_required: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InviteInfo {
    pub code: String,
    pub link: String,
    pub qr_base64: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemInfo {
    pub os: String,
    pub cpu: String,
    pub ram_gb: u64,
    pub python_version: Option<String>,
    pub distllm_version: String,
    pub gpus: Vec<GpuInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiError {
    pub message: String,
}

impl From<String> for ApiError {
    fn from(message: String) -> Self {
        Self { message }
    }
}

impl From<ApiError> for String {
    fn from(err: ApiError) -> Self {
        err.message
    }
}

// ===== App State =====

pub struct AppState {
    pub coordinator: Mutex<Option<Child>>,
    pub worker: Mutex<Option<Child>>,
    pub cluster_id: Mutex<Option<String>>,
    pub api_port: Mutex<Option<u16>>,
}

impl AppState {
    pub fn new() -> Self {
        Self {
            coordinator: Mutex::new(None),
            worker: Mutex::new(None),
            cluster_id: Mutex::new(None),
            api_port: Mutex::new(None),
        }
    }
}

// ===== Python discovery =====

fn find_python() -> String {
    for name in &["python3", "python", "py"] {
        if Command::new(name)
            .arg("--version")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .is_ok()
        {
            return name.to_string();
        }
    }
    "python".to_string()
}

fn ensure_distllm() -> Result<String, ApiError> {
    let python = find_python();
    let check = Command::new(&python)
        .args(["-m", "distllm.cli.main", "--help"])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status();
    if check.is_ok() {
        Ok(python)
    } else {
        Err(ApiError::from(
            "distllm Python package not found.\n\nInstall it:\n  pip install distributed-llm\n\nOr activate the virtual environment where it's installed.".to_string(),
        ))
    }
}

fn api_get(port: u16, path: &str) -> Result<serde_json::Value, String> {
    let url = format!("http://127.0.0.1:{}{}", port, path);
    let resp = ureq::get(&url)
        .timeout(std::time::Duration::from_secs(3))
        .call()
        .map_err(|e| format!("API request failed: {}", e))?;
    let body = resp.into_string().map_err(|e| format!("Read error: {}", e))?;
    serde_json::from_str(&body).map_err(|e| format!("Parse error: {}", e))
}

// ===== Tray icon rendering =====

fn make_tray_rgba(size: u32, r: u8, g: u8, b: u8) -> Vec<u8> {
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

fn update_tray_icon(app: &tauri::AppHandle, active: bool) {
    let (r, g, b) = if active { (0x22, 0xcc, 0x66) } else { (0x66, 0x66, 0x66) };
    let rgba = make_tray_rgba(32, r, g, b);
    if let Some(tray) = app.tray_by_id("main-tray") {
        let _ = tray.set_icon(Some(Image::new(&rgba, 32, 32)));
    }
}

// ===== Tauri Commands =====

#[tauri::command]
fn create_cluster(
    app: tauri::AppHandle,
    state: tauri::State<AppState>,
    port: Option<u16>,
    model: Option<String>,
) -> Result<ClusterStatus, String> {
    let port = port.unwrap_or(8000);
    let mut coord = state.coordinator.lock().map_err(|e| e.to_string())?;
    if coord.is_some() {
        return Err("Cluster is already running".to_string());
    }

    let python = ensure_distllm()?;

    let mut args = vec![
        "-m".into(),
        "distllm.cli.main".into(),
        "cluster".into(),
        "start".into(),
        "--port".into(),
        port.to_string(),
    ];
    if let Some(m) = model {
        args.extend(["--model".into(), m]);
    }

    let child = Command::new(&python)
        .args(&args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("Failed to start coordinator: {}", e))?;

    *coord = Some(child);
    let cid = format!("cluster-{}", &uuid::Uuid::new_v4().to_string()[..8]);
    *state.cluster_id.lock().map_err(|e| e.to_string())? = Some(cid);
    *state.api_port.lock().map_err(|e| e.to_string())? = Some(port);

    std::thread::sleep(std::time::Duration::from_secs(3));
    update_tray_icon(&app, true);

    // Fetch initial status
    let status = fetch_from_api(port);
    Ok(status)
}

#[tauri::command]
fn join_cluster(
    app: tauri::AppHandle,
    state: tauri::State<AppState>,
    host: String,
    port: u16,
) -> Result<ClusterStatus, String> {
    let mut worker = state.worker.lock().map_err(|e| e.to_string())?;
    if worker.is_some() {
        return Err("Already connected to a cluster".to_string());
    }

    let python = ensure_distllm()?;

    let child = Command::new(&python)
        .args([
            "-m",
            "distllm.cli.main",
            "cluster",
            "join",
            "--coordinator",
            &format!("{}:{}", host, port),
            "--discover",
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("Failed to join cluster: {}", e))?;

    *worker = Some(child);
    update_tray_icon(&app, true);

    // Store the API port so the dashboard can query the coordinator
    *state.api_port.lock().map_err(|e| e.to_string())? = Some(port);

    Ok(ClusterStatus {
        running: true,
        node_id: Some("worker".into()),
        role: Some("worker".into()),
        coordinator_addr: Some(format!("http://{}:{}", host, port)),
        nodes: vec![],
        uptime_secs: 0,
    })
}

#[tauri::command]
fn leave_cluster(app: tauri::AppHandle, state: tauri::State<AppState>) -> Result<(), String> {
    for child_opt in [&state.coordinator, &state.worker] {
        if let Ok(mut guard) = child_opt.lock() {
            if let Some(ref mut c) = *guard {
                let _ = c.kill();
                let _ = c.wait();
            }
            *guard = None;
        }
    }

    *state.cluster_id.lock().map_err(|e| e.to_string())? = None;
    *state.api_port.lock().map_err(|e| e.to_string())? = None;

    update_tray_icon(&app, false);
    Ok(())
}

fn fetch_from_api(port: u16) -> ClusterStatus {
    let mut nodes = vec![];
    // Try the admin nodes endpoint first
    if let Ok(val) = api_get(port, "/admin/v1/nodes") {
        if let Some(list) = val.get("nodes").and_then(|n| n.as_array()) {
            for n in list {
                nodes.push(PeerInfo {
                    node_id: n.get("node_id").and_then(|v| v.as_str()).unwrap_or("?").into(),
                    host: n.get("host").and_then(|v| v.as_str()).unwrap_or("").into(),
                    port: n.get("port").and_then(|v| v.as_u64()).unwrap_or(0) as u16,
                    healthy: n.get("healthy").and_then(|v| v.as_bool()).unwrap_or(false),
                    gpu_name: n.get("gpu_name").and_then(|v| v.as_str()).unwrap_or("").into(),
                    gpu_utilization: n.get("gpu_utilization").and_then(|v| v.as_f64()).unwrap_or(0.0),
                    layers: format!(
                        "{}-{}",
                        n.get("start_layer").and_then(|v| v.as_u64()).unwrap_or(0),
                        n.get("end_layer").and_then(|v| v.as_u64()).unwrap_or(0)
                    ),
                });
            }
        }
    }

    // Get model info from health endpoint
    let _model = if let Ok(val) = api_get(port, "/health") {
        val.get("model").and_then(|v| v.as_str()).unwrap_or("").to_string()
    } else {
        String::new()
    };

    ClusterStatus {
        running: true,
        node_id: None,
        role: Some("coordinator".into()),
        coordinator_addr: Some(format!("http://127.0.0.1:{}", port)),
        nodes,
        uptime_secs: 0,
    }
}

#[tauri::command]
fn get_cluster_status(state: tauri::State<AppState>) -> Result<ClusterStatus, String> {
    let coord_running = state.coordinator.lock().map_err(|e| e.to_string())?.is_some();
    let worker_running = state.worker.lock().map_err(|e| e.to_string())?.is_some();
    let port = *state.api_port.lock().map_err(|e| e.to_string())?;

    if !coord_running && !worker_running {
        return Ok(ClusterStatus {
            running: false,
            node_id: None,
            role: None,
            coordinator_addr: None,
            nodes: vec![],
            uptime_secs: 0,
        });
    }

    match port {
        Some(p) => Ok(fetch_from_api(p)),
        None => Ok(ClusterStatus {
            running: true,
            node_id: None,
            role: None,
            coordinator_addr: None,
            nodes: vec![],
            uptime_secs: 0,
        }),
    }
}

#[tauri::command]
fn get_gpu_metrics() -> Result<Vec<GpuInfo>, String> {
    let mut gpus = vec![];

    #[cfg(any(target_os = "linux", target_os = "windows"))]
    {
        use nvml_wrapper::enum_wrappers::device::TemperatureSensor;
        match nvml_wrapper::Nvml::init() {
            Ok(nvml) => {
                let count = nvml.device_count().unwrap_or(0);
                for i in 0..count {
                    if let Ok(device) = nvml.device_by_index(i) {
                        let name = device.name().unwrap_or_else(|_| "Unknown".into());
                        let temp = device.temperature(TemperatureSensor::Gpu).unwrap_or(0);
                        let util = device.utilization_rates().ok();
                        let mem = device.memory_info().ok();
                        let mem_total = mem.as_ref().map(|m| m.total).unwrap_or(0);
                        let mem_used = mem.as_ref().map(|m| m.used).unwrap_or(0);
                        let mem_free = mem.as_ref().map(|m| m.free).unwrap_or(0);
                        gpus.push(GpuInfo {
                            index: i,
                            name: name.trim().to_string(),
                            temperature: temp as f64,
                            utilization: util.map(|u| u.gpu as f64).unwrap_or(0.0),
                            memory_total: mem_total,
                            memory_used: mem_used,
                            memory_free: mem_free,
                        });
                    }
                }
                let _ = nvml.shutdown();
            }
            Err(_) => {
                // NVML not available — try lspci fallback on Linux
                #[cfg(target_os = "linux")]
                {
                    if let Ok(output) = std::process::Command::new("lspci")
                        .args(["-v", "-s", "VGA"])
                        .output()
                    {
                        if let Ok(text) = String::from_utf8(output.stdout) {
                            for line in text.lines() {
                                if line.contains("VGA") || line.contains("3D") {
                                    let name = line.split(':').nth(1).unwrap_or("Unknown GPU").trim();
                                    gpus.push(GpuInfo {
                                        index: gpus.len() as u32,
                                        name: name.to_string(),
                                        temperature: 0.0,
                                        utilization: 0.0,
                                        memory_total: 0,
                                        memory_used: 0,
                                        memory_free: 0,
                                    });
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[cfg(target_os = "macos")]
    {
        // macOS: use system_profiler to detect GPU hardware
        // NVML is not available on macOS, so we provide basic GPU info
        if let Ok(output) = std::process::Command::new("system_profiler")
            .arg("SPDisplaysDataType")
            .arg("-json")
            .output()
        {
            if let Ok(json_str) = String::from_utf8(output.stdout) {
                if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&json_str) {
                    if let Some(displays) = parsed["SPDisplaysDataType"].as_array() {
                        for (i, display) in displays.iter().enumerate() {
                            let name = display["_name"]
                                .as_str()
                                .unwrap_or("Apple GPU")
                                .to_string();
                            let mem_str = display["sppci_model"]
                                .as_str()
                                .unwrap_or("");
                            // Extract VRAM if available (e.g., "spdisplays_vram: 8192 MB")
                            let mem_total: u64 = display["spdisplays_vram"]
                                .as_str()
                                .and_then(|s| s.split_whitespace().next())
                                .and_then(|s| s.parse::<u64>().ok())
                                .map(|mb| mb * 1024 * 1024)
                                .unwrap_or(0);
                            gpus.push(GpuInfo {
                                index: i as u32,
                                name,
                                temperature: 0.0,  // Not available via system_profiler
                                utilization: 0.0,   // Not available via system_profiler
                                memory_total: mem_total,
                                memory_used: 0,
                                memory_free: mem_total,
                            });
                        }
                    }
                }
            }
        }
        // Fallback if system_profiler fails: report at least one Apple GPU
        if gpus.is_empty() {
            gpus.push(GpuInfo {
                index: 0,
                name: "Apple GPU".to_string(),
                temperature: 0.0,
                utilization: 0.0,
                memory_total: 0,
                memory_used: 0,
                memory_free: 0,
            });
        }
    }

    Ok(gpus)
}

#[tauri::command]
fn list_models() -> Result<Vec<ModelInfo>, String> {
    let python = ensure_distllm()?;
    let output = Command::new(&python)
        .args(["-m", "distllm.cli.main", "models", "list", "--json"])
        .output()
        .map_err(|e| format!("Failed to list models: {}", e))?;

    if output.status.success() {
        let stdout = String::from_utf8_lossy(&output.stdout);
        if stdout.trim().is_empty() {
            return Ok(vec![]);
        }
        serde_json::from_str(&stdout).map_err(|e| format!("Parse error: {}", e))
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr);
        Err(format!("CLI error: {}", stderr.trim()))
    }
}

#[tauri::command]
fn download_model(model_id: String) -> Result<String, String> {
    let python = ensure_distllm()?;
    let child = Command::new(&python)
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
        .map_err(|e| format!("Failed to start download: {}", e))?;

    let output = child.wait_with_output().map_err(|e| format!("Download failed: {}", e))?;

    if output.status.success() {
        Ok(format!("Model {} downloaded successfully", model_id))
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr);
        Err(format!("Download failed: {}", stderr.trim()))
    }
}

#[tauri::command]
fn generate_invite() -> Result<InviteInfo, String> {
    let code = uuid::Uuid::new_v4().to_string();
    let short = code.split('-').next().unwrap_or(&code).to_string();
    let link = format!("distllm://connect/{}", short);
    Ok(InviteInfo {
        code: short,
        link,
        qr_base64: String::new(),
    })
}

#[tauri::command]
fn get_system_info() -> Result<SystemInfo, String> {
    let python = find_python();
    let py_version = Command::new(&python)
        .arg("--version")
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string());

    let gpus = get_gpu_metrics().unwrap_or_default();

    Ok(SystemInfo {
        os: std::env::consts::OS.to_string(),
        cpu: "Unknown".into(),
        ram_gb: 0,
        python_version: py_version,
        distllm_version: "0.4.0".into(),
        gpus,
    })
}

// ===== App entry =====

fn build_tray_menu(app: &tauri::AppHandle) -> Result<Menu<tauri::Wry>, tauri::Error> {
    let show = MenuItem::with_id(app, "show", "Show Window", true, None::<&str>)?;
    let create = MenuItem::with_id(app, "create_cluster", "Create Cluster", true, None::<&str>)?;
    let join = MenuItem::with_id(app, "join_cluster", "Join Cluster...", true, None::<&str>)?;
    let leave = MenuItem::with_id(app, "leave_cluster", "Leave Cluster", true, None::<&str>)?;
    let sep = PredefinedMenuItem::separator(app)?;
    let quit = MenuItem::with_id(app, "quit", "Quit Distributed LLM", true, None::<&str>)?;
    Menu::with_items(app, &[&show, &create, &join, &leave, &sep, &quit])
}

fn handle_tray_menu(app: &tauri::AppHandle, event_id: &str) {
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
            *state.cluster_id.lock().unwrap() = None;
            *state.api_port.lock().unwrap() = None;
            update_tray_icon(app, false);
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

pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .manage(AppState::new())
        .setup(|app| {
            let menu = build_tray_menu(app.handle())?;

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
        })
        .invoke_handler(tauri::generate_handler![
            create_cluster,
            join_cluster,
            leave_cluster,
            get_cluster_status,
            get_gpu_metrics,
            list_models,
            download_model,
            generate_invite,
            get_system_info,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
