use crate::error::AppError;
use crate::types::GpuInfo;

// Cache NVML handle to avoid re-initialization on every poll
#[cfg(any(target_os = "linux", target_os = "windows"))]
static NVML_HANDLE: std::sync::OnceLock<nvml_wrapper::Nvml> = std::sync::OnceLock::new();

#[tauri::command]
pub fn get_gpu_metrics() -> Result<Vec<GpuInfo>, AppError> {
    let mut gpus = vec![];

    #[cfg(any(target_os = "linux", target_os = "windows"))]
    {
        use nvml_wrapper::enum_wrappers::device::TemperatureSensor;
        let nvml_result = NVML_HANDLE.get_or_try_init(nvml_wrapper::Nvml::init);
        match nvml_result {
            Ok(nvml) => {
                let count = nvml.device_count().unwrap_or(0);
                for i in 0..count {
                    if let Ok(device) = nvml.device_by_index(i) {
                        let name = device.name().unwrap_or_else(|_| "Unknown".into());
                        let temp = device
                            .temperature(TemperatureSensor::Gpu)
                            .unwrap_or(0);
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
            }
            Err(_) => {
                // NVML not available — try lspci fallback on Linux
                #[cfg(target_os = "linux")]
                {
                    if let Ok(output) =
                        std::process::Command::new("lspci")
                            .args(["-v", "-s", "VGA"])
                            .output()
                    {
                        if let Ok(text) = String::from_utf8(output.stdout) {
                            for line in text.lines() {
                                if line.contains("VGA") || line.contains("3D") {
                                    let name = line
                                        .split(':')
                                        .nth(1)
                                        .unwrap_or("Unknown GPU")
                                        .trim();
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
                            let mem_total: u64 = display["spdisplays_vram"]
                                .as_str()
                                .and_then(|s| s.split_whitespace().next())
                                .and_then(|s| s.parse::<u64>().ok())
                                .map(|mb| mb * 1024 * 1024)
                                .unwrap_or(0);
                            gpus.push(GpuInfo {
                                index: i as u32,
                                name,
                                temperature: 0.0,
                                utilization: 0.0,
                                memory_total: mem_total,
                                memory_used: 0,
                                memory_free: mem_total,
                            });
                        }
                    }
                }
            }
        }
        // Fallback if system_profiler fails
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
