export interface GpuInfo {
  index: number;
  name: string;
  temperature: number;
  utilization: number;
  memory_total: number;
  memory_used: number;
  memory_free: number;
}

export interface ClusterStatus {
  running: boolean;
  node_id: string | null;
  role: string | null;
  coordinator_addr: string | null;
  nodes: PeerInfo[];
  uptime_secs: number;
}

export interface PeerInfo {
  node_id: string;
  host: string;
  port: number;
  healthy: boolean;
  gpu_name: string;
  gpu_utilization: number;
  layers: string;
}

export interface ModelInfo {
  id: string;
  name: string;
  size: string;
  downloaded: boolean;
  quantization: string[];
  gpu_required: string;
}

export interface InviteInfo {
  code: string;
  link: string;
  qr_base64: string;
}

export interface SystemInfo {
  os: string;
  cpu: string;
  ram_gb: number;
  python_version: string | null;
  distllm_version: string;
  gpus: GpuInfo[];
}

export type Page = "dashboard" | "cluster" | "models" | "friends";
