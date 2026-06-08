use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use mdns_sd::{ServiceDaemon, ServiceInfo};
use serde::{Deserialize, Serialize};
use tauri::Emitter;

use crate::error::AppError;
use crate::state::AppState;

const SERVICE_TYPE: &str = "_distllm._tcp.local.";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoveredService {
    pub name: String,
    pub host: String,
    pub port: u16,
    pub properties: HashMap<String, String>,
    pub discovered_at: u64,
}

struct DiscoveryState {
    mdns: Option<ServiceDaemon>,
    receiver: Option<mdns_sd::Receiver>,
    services: Vec<DiscoveredService>,
}

static DISCOVERY: once_cell::sync::Lazy<Arc<Mutex<DiscoveryState>>> =
    once_cell::sync::Lazy::new(|| {
        Arc::new(Mutex::new(DiscoveryState {
            mdns: None,
            receiver: None,
            services: vec![],
        }))
    });

/// Start mDNS discovery for Distributed LLM services on the LAN.
#[tauri::command]
pub async fn start_discovery(app: tauri::AppHandle) -> Result<(), AppError> {
    let mut state = DISCOVERY.lock().map_err(|e| AppError::Internal(e.to_string()))?;

    if state.mdns.is_some() {
        return Ok(());
    }

    let mdns = ServiceDaemon::new().map_err(|e| AppError::Internal(e.to_string()))?;

    let receiver = mdns
        .browse(SERVICE_TYPE)
        .map_err(|e| AppError::Internal(e.to_string()))?;

    state.mdns = Some(mdns);
    state.receiver = Some(receiver);
    state.services.clear();

    // Spawn background task to process mDNS events
    let app_handle = app.clone();
    tokio::spawn(async move {
        loop {
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;

            let receiver = {
                let mut s = DISCOVERY.lock().unwrap();
                s.receiver.take()
            };

            if let Some(recv) = receiver {
                // Try to receive without blocking
                match recv.try_recv() {
                    Ok(mdns_sd::Event::ServiceFound(info)) => {
                        let svc = DiscoveredService {
                            name: info.get_fullname().to_string(),
                            host: info.get_hostname().to_string(),
                            port: info.get_port(),
                            properties: HashMap::new(),
                            discovered_at: std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .unwrap_or_default()
                                .as_millis() as u64,
                        };
                        let mut s = DISCOVERY.lock().unwrap();
                        // Deduplicate
                        if !s.services.iter().any(|x| x.host == svc.host && x.port == svc.port) {
                            s.services.push(svc);
                            let _ = app_handle.emit("discovery-updated", ());
                        }
                        // Put receiver back
                        s.receiver = Some(recv);
                    }
                    Ok(mdns_sd::Event::ServiceResolved(info)) => {
                        // Update existing service with resolved addresses
                        let mut s = DISCOVERY.lock().unwrap();
                        let addr = info.get_addresses().iter().next().map(|a| a.to_string());
                        if let Some(addr) = addr {
                            if let Some(svc) = s.services.iter_mut().find(|x| x.port == info.get_port()) {
                                svc.host = addr;
                            }
                        }
                        s.receiver = Some(recv);
                    }
                    Ok(mdns_sd::Event::ServiceRemoved(info)) => {
                        let mut s = DISCOVERY.lock().unwrap();
                        let name = info.get_fullname();
                        s.services.retain(|x| x.name != name);
                        s.receiver = Some(recv);
                    }
                    Err(_) => {
                        // Channel closed or empty — put receiver back
                        let mut s = DISCOVERY.lock().unwrap();
                        s.receiver = Some(recv);
                    }
                }
            } else {
                break;
            }
        }
    });

    Ok(())
}

/// Stop mDNS discovery.
#[tauri::command]
pub async fn stop_discovery() -> Result<(), AppError> {
    let mut state = DISCOVERY.lock().map_err(|e| AppError::Internal(e.to_string()))?;

    if let Some(mdns) = state.mdns.take() {
        let _ = mdns.shutdown();
    }
    state.receiver = None;
    state.services.clear();

    Ok(())
}

/// Get the list of discovered services.
#[tauri::command]
pub fn get_discovered_services() -> Result<Vec<DiscoveredService>, AppError> {
    let state = DISCOVERY.lock().map_err(|e| AppError::Internal(e.to_string()))?;
    Ok(state.services.clone())
}

/// Get discovery status.
#[tauri::command]
pub fn get_discovery_status() -> Result<serde_json::Value, AppError> {
    let state = DISCOVERY.lock().map_err(|e| AppError::Internal(e.to_string()))?;
    Ok(serde_json::json!({
        "active": state.mdns.is_some(),
        "service_count": state.services.len(),
    }))
}

/// Publish this coordinator as an mDNS service.
pub fn publish_service(port: u16) -> Result<(), AppError> {
    let mdns = ServiceDaemon::new().map_err(|e| AppError::Internal(e.to_string()))?;

    let hostname = hostname::get()
        .map(|h| h.to_string_lossy().to_string())
        .unwrap_or_else(|_| "localhost".into());

    let mut props = HashMap::new();
    props.insert("version".into(), "0.4.0".into());

    let service_info = ServiceInfo::new(
        SERVICE_TYPE,
        &format!("distllm-{}", &hostname[..hostname.len().min(16)]),
        &hostname,
        "",
        port,
        props,
    )
    .map_err(|e| AppError::Internal(e.to_string()))?;

    mdns.register(service_info)
        .map_err(|e| AppError::Internal(e.to_string()))?;

    // Keep the daemon alive
    std::mem::forget(mdns);

    Ok(())
}
