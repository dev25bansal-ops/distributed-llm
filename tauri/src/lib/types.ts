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

export type Page = "dashboard" | "cluster" | "models" | "friends" | "chat" | "benchmark" | "topology" | "multimodel" | "plugins" | "webdashboard" | "discovery" | "settings" | "logs";

export interface ChatMessage {
  id: string;
  role: "system" | "user" | "assistant";
  content: string;
  timestamp: number;
}

export interface ChatOptions {
  temperature: number;
  top_p: number;
  max_tokens: number;
  system_prompt: string;
}

export interface InferenceMetrics {
  ttft: number | null;
  tokens_per_sec: number;
  inter_token_latency: number;
  total_tokens: number;
  total_time: number;
}

// 4.3 Benchmark types
export interface BenchmarkRun {
  id: string;
  model: string;
  prompt_tokens: number;
  completion_tokens: number;
  tokens_per_sec: number;
  ttft: number;
  inter_token_latency: number;
  total_time: number;
  nodes_used: number;
  quantization: string;
  timestamp: number;
}

export interface BenchmarkConfig {
  model: string;
  prompt_length: number;
  max_tokens: number;
  num_runs: number;
  quantization: string;
}

// 4.4 Topology types
export interface TopologyNode {
  id: string;
  label: string;
  type: "coordinator" | "worker";
  gpu_name: string;
  gpu_utilization: number;
  layers: { start: number; end: number };
  healthy: boolean;
  host: string;
  port: number;
}

export interface TopologyLink {
  source: string;
  target: string;
  active: boolean;
  throughput: number;
}

// 4.5 Multi-model types
export interface ModelSlot {
  id: string;
  model_id: string;
  model_name: string;
  status: "loading" | "ready" | "error" | "unloaded";
  vram_allocated_mb: number;
  max_context: number;
  requests_served: number;
  avg_tokens_per_sec: number;
  error_message: string | null;
}

export interface ModelRoutingRule {
  id: string;
  pattern: string;
  target_slot: string;
  priority: number;
}

// 4.6 Plugin types
export type PluginKind = "backend" | "auth" | "monitoring";

export interface PluginConfig {
  id: string;
  name: string;
  kind: PluginKind;
  enabled: boolean;
  endpoint: string;
  api_key: string;
  extra: Record<string, string>;
  created_at: number;
}

// 4.7 Web Dashboard types
export interface WebDashboardConfig {
  enabled: boolean;
  port: number;
  auth_required: boolean;
  auth_token: string;
  cors_origins: string[];
}

export interface WebDashboardStatus {
  running: boolean;
  url: string;
  connections: number;
}

// 4.8 Discovery types
export interface DiscoveredService {
  name: string;
  host: string;
  port: number;
  properties: Record<string, string>;
  discovered_at: number;
}

// 4.9 Ollama Compatibility types
export interface OllamaConfig {
  host: string;
  port: number;
  enabled: boolean;
}

export interface OllamaModel {
  name: string;
  size: number;
  digest: string;
  modified_at: string;
}
