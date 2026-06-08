use serde::{Deserialize, Serialize};

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
