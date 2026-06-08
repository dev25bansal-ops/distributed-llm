use std::sync::OnceLock;
use std::time::Duration;

use crate::error::AppError;
use crate::types::{ClusterStatus, PeerInfo};

/// Shared HTTP agent with connection pooling — initialized once.
static AGENT: OnceLock<ureq::Agent> = OnceLock::new();

fn get_agent() -> &'static ureq::Agent {
    AGENT.get_or_init(|| {
        ureq::AgentBuilder::new()
            .timeout(Duration::from_secs(3))
            .max_idle_connections(2)
            .max_idle_connections_per_host(2)
            .build()
    })
}

/// Make a GET request to the coordinator API using pooled connections.
pub fn api_get(port: u16, path: &str, token: Option<&str>) -> Result<serde_json::Value, AppError> {
    api_get_to("127.0.0.1", port, path, token)
}

/// Make a GET request to a specific host using pooled connections.
pub fn api_get_to(
    host: &str,
    port: u16,
    path: &str,
    token: Option<&str>,
) -> Result<serde_json::Value, AppError> {
    let url = format!("http://{}:{}{}", host, port, path);
    let mut req = get_agent().get(&url);
    if let Some(t) = token {
        req = req.set("Authorization", &format!("Bearer {}", t));
    }
    let resp = req.call().map_err(|e| {
        eprintln!("[api_get] Request failed for {}: {}", url, e);
        AppError::ApiUnreachable
    })?;
    let body = resp.into_string().map_err(|e| {
        eprintln!("[api_get] Read error for {}: {}", url, e);
        AppError::ApiInvalidResponse
    })?;
    serde_json::from_str(&body).map_err(|e| {
        eprintln!("[api_get] Parse error for {}: {}", url, e);
        AppError::ApiInvalidResponse
    })
}

/// Fetch cluster status from the coordinator API.
pub fn fetch_from_api(port: u16, token: Option<&str>) -> ClusterStatus {
    let mut nodes = vec![];
    let api_ok = if let Ok(val) = api_get(port, "/admin/v1/nodes", token) {
        if let Some(list) = val.get("nodes").and_then(|n| n.as_array()) {
            for n in list {
                nodes.push(PeerInfo {
                    node_id: n
                        .get("node_id")
                        .and_then(|v| v.as_str())
                        .unwrap_or("?")
                        .into(),
                    host: n
                        .get("host")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .into(),
                    port: n.get("port").and_then(|v| v.as_u64()).unwrap_or(0) as u16,
                    healthy: n.get("healthy").and_then(|v| v.as_bool()).unwrap_or(false),
                    gpu_name: n
                        .get("gpu_name")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .into(),
                    gpu_utilization: n
                        .get("gpu_utilization")
                        .and_then(|v| v.as_f64())
                        .unwrap_or(0.0),
                    layers: format!(
                        "{}-{}",
                        n.get("start_layer").and_then(|v| v.as_u64()).unwrap_or(0),
                        n.get("end_layer").and_then(|v| v.as_u64()).unwrap_or(0)
                    ),
                });
            }
        }
        true
    } else {
        false
    };

    ClusterStatus {
        running: api_ok,
        node_id: None,
        role: if api_ok {
            Some("coordinator".into())
        } else {
            None
        },
        coordinator_addr: Some(format!("http://127.0.0.1:{}", port)),
        nodes,
        uptime_secs: 0,
    }
}
