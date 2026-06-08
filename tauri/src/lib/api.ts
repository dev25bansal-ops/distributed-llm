import { invoke } from "@tauri-apps/api/core";
import type {
  ClusterStatus,
  GpuInfo,
  ModelInfo,
  InviteInfo,
  SystemInfo,
  ChatMessage,
  ChatOptions,
  BenchmarkRun,
  BenchmarkConfig,
  ModelSlot,
  ModelRoutingRule,
  PluginConfig,
  WebDashboardConfig,
  WebDashboardStatus,
  DiscoveredService,
  OllamaConfig,
  OllamaModel,
} from "./types";

export async function createCluster(
  port?: number,
  model?: string,
): Promise<ClusterStatus> {
  return invoke<ClusterStatus>("create_cluster", { port, model });
}

export async function joinCluster(
  host: string,
  port: number,
): Promise<ClusterStatus> {
  return invoke<ClusterStatus>("join_cluster", { host, port });
}

export async function leaveCluster(): Promise<void> {
  return invoke<void>("leave_cluster");
}

export async function getClusterStatus(): Promise<ClusterStatus> {
  return invoke<ClusterStatus>("get_cluster_status");
}

export async function getGpuMetrics(): Promise<GpuInfo[]> {
  return invoke<GpuInfo[]>("get_gpu_metrics");
}

export async function listModels(): Promise<ModelInfo[]> {
  return invoke<ModelInfo[]>("list_models");
}

export async function downloadModel(modelId: string): Promise<string> {
  return invoke<string>("download_model", { modelId });
}

export async function generateInvite(): Promise<InviteInfo> {
  return invoke<InviteInfo>("generate_invite");
}

export async function getSystemInfo(): Promise<SystemInfo> {
  return invoke<SystemInfo>("get_system_info");
}

export async function checkCoordinator(
  host: string,
  port: number,
): Promise<boolean> {
  return invoke<boolean>("check_coordinator", { host, port });
}

export interface StreamCallbacks {
  onToken: (token: string) => void;
  onDone: (metrics: {
    ttft: number;
    tokens_per_sec: number;
    inter_token_latency: number;
    total_tokens: number;
    total_time: number;
  }) => void;
  onError: (error: string) => void;
}

export async function streamChatCompletion(
  baseUrl: string,
  messages: ChatMessage[],
  options: ChatOptions,
  callbacks: StreamCallbacks,
  signal?: AbortSignal,
): Promise<void> {
  const apiMessages = messages.map((m) => ({
    role: m.role,
    content: m.content,
  }));

  const body = {
    model: "local",
    messages: apiMessages,
    stream: true,
    temperature: options.temperature,
    top_p: options.top_p,
    max_tokens: options.max_tokens,
  };

  const startTime = performance.now();
  let ttft: number | null = null;
  let tokenCount = 0;
  let lastTokenTime = startTime;
  const tokenTimes: number[] = [];

  try {
    const response = await fetch(`${baseUrl}/v1/chat/completions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal,
    });

    if (!response.ok) {
      const errorText = await response.text();
      callbacks.onError(`HTTP ${response.status}: ${errorText}`);
      return;
    }

    const reader = response.body?.getReader();
    if (!reader) {
      callbacks.onError("No response body");
      return;
    }

    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() || "";

      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed || !trimmed.startsWith("data: ")) continue;

        const data = trimmed.slice(6);
        if (data === "[DONE]") {
          const totalTime = (performance.now() - startTime) / 1000;
          callbacks.onDone({
            ttft: ttft,
            tokens_per_sec: tokenCount / totalTime,
            inter_token_latency:
              tokenTimes.length > 1
                ? tokenTimes.reduce((a, b) => a + b, 0) / tokenTimes.length
                : 0,
            total_tokens: tokenCount,
            total_time: totalTime,
          });
          return;
        }

        try {
          const parsed = JSON.parse(data);
          const delta = parsed.choices?.[0]?.delta?.content;
          if (delta) {
            const now = performance.now();
            if (ttft === null) {
              ttft = now - startTime;
            }
            if (tokenCount > 0) {
              tokenTimes.push(now - lastTokenTime);
            }
            lastTokenTime = now;
            tokenCount++;
            callbacks.onToken(delta);
          }
        } catch {
          // Skip malformed JSON lines
        }
      }
    }

    // Stream ended without [DONE]
    const totalTime = (performance.now() - startTime) / 1000;
    callbacks.onDone({
      ttft: ttft,
      tokens_per_sec: tokenCount / totalTime,
      inter_token_latency:
        tokenTimes.length > 1
          ? tokenTimes.reduce((a, b) => a + b, 0) / tokenTimes.length
          : 0,
      total_tokens: tokenCount,
      total_time: totalTime,
    });
  } catch (e: unknown) {
    if (e instanceof DOMException && e.name === "AbortError") {
      return;
    }
    callbacks.onError(String(e));
  }
}

// 4.3 Benchmark API
export async function runBenchmark(
  baseUrl: string,
  config: BenchmarkConfig,
  signal?: AbortSignal,
): Promise<BenchmarkRun[]> {
  const results: BenchmarkRun[] = [];
  const prompt = "Explain the theory of relativity in detail. ".repeat(
    Math.ceil(config.prompt_length / 10),
  );

  for (let i = 0; i < config.num_runs; i++) {
    const startTime = performance.now();
    let ttft: number | null = null;
    let tokenCount = 0;
    let lastTokenTime = startTime;
    const tokenTimes: number[] = [];

    try {
      const response = await fetch(`${baseUrl}/v1/chat/completions`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: config.model,
          messages: [
            { role: "system", content: "You are a helpful assistant." },
            { role: "user", content: prompt },
          ],
          stream: true,
          temperature: 0.7,
          max_tokens: config.max_tokens,
        }),
        signal,
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      const reader = response.body?.getReader();
      if (!reader) throw new Error("No response body");

      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed || !trimmed.startsWith("data: ")) continue;
          const data = trimmed.slice(6);
          if (data === "[DONE]") break;

          try {
            const parsed = JSON.parse(data);
            const delta = parsed.choices?.[0]?.delta?.content;
            if (delta) {
              const now = performance.now();
              if (ttft === null) ttft = now - startTime;
              if (tokenCount > 0) tokenTimes.push(now - lastTokenTime);
              lastTokenTime = now;
              tokenCount++;
            }
          } catch {
            // skip
          }
        }
      }

      const totalTime = (performance.now() - startTime) / 1000;
      results.push({
        id: `bench-${Date.now()}-${i}`,
        model: config.model,
        prompt_tokens: Math.floor(config.prompt_length / 4),
        completion_tokens: tokenCount,
        tokens_per_sec: tokenCount / totalTime,
        ttft: ttft ?? 0,
        inter_token_latency:
          tokenTimes.length > 1
            ? tokenTimes.reduce((a, b) => a + b, 0) / tokenTimes.length
            : 0,
        total_time: totalTime,
        nodes_used: 1,
        quantization: config.quantization,
        timestamp: Date.now(),
      });
    } catch (e: unknown) {
      if (e instanceof DOMException && e.name === "AbortError") break;
      results.push({
        id: `bench-${Date.now()}-${i}`,
        model: config.model,
        prompt_tokens: Math.floor(config.prompt_length / 4),
        completion_tokens: 0,
        tokens_per_sec: 0,
        ttft: 0,
        inter_token_latency: 0,
        total_time: 0,
        nodes_used: 1,
        quantization: config.quantization,
        timestamp: Date.now(),
      });
    }
  }

  return results;
}

// 4.4 Topology API — derives topology from cluster status
export function buildTopologyFromCluster(
  cluster: ClusterStatus | null,
): { nodes: { id: string; label: string; type: "coordinator" | "worker"; gpu_name: string; gpu_utilization: number; layers: { start: number; end: number }; healthy: boolean; host: string; port: number }[]; links: { source: string; target: string; active: boolean; throughput: number }[] } {
  if (!cluster || !cluster.running) {
    return { nodes: [], links: [] };
  }

  const nodes = [
    {
      id: "coordinator",
      label: "Coordinator",
      type: "coordinator" as const,
      gpu_name: "",
      gpu_utilization: 0,
      layers: { start: 0, end: 0 },
      healthy: true,
      host: "127.0.0.1",
      port: 8000,
    },
    ...cluster.nodes.map((n) => {
      const layerParts = n.layers.split("-").map(Number);
      return {
        id: n.node_id,
        label: n.node_id.slice(0, 12),
        type: "worker" as const,
        gpu_name: n.gpu_name,
        gpu_utilization: n.gpu_utilization,
        layers: { start: layerParts[0] || 0, end: layerParts[1] || 0 },
        healthy: n.healthy,
        host: n.host,
        port: n.port,
      };
    }),
  ];

  const links = cluster.nodes.map((n) => ({
    source: "coordinator",
    target: n.node_id,
    active: n.healthy,
    throughput: n.gpu_utilization,
  }));

  return { nodes, links };
}

// 4.5 Multi-model API
export async function getModelSlots(): Promise<ModelSlot[]> {
  return invoke<ModelSlot[]>("get_model_slots");
}

export async function loadModelSlot(
  slotId: string,
  modelId: string,
): Promise<ModelSlot> {
  return invoke<ModelSlot>("load_model_slot", { slotId, modelId });
}

export async function unloadModelSlot(slotId: string): Promise<void> {
  return invoke<void>("unload_model_slot", { slotId });
}

export async function getRoutingRules(): Promise<ModelRoutingRule[]> {
  return invoke<ModelRoutingRule[]>("get_routing_rules");
}

export async function setRoutingRule(
  rule: ModelRoutingRule,
): Promise<void> {
  return invoke<void>("set_routing_rule", { rule });
}

export async function deleteRoutingRule(ruleId: string): Promise<void> {
  return invoke<void>("delete_routing_rule", { ruleId });
}

// 4.6 Plugin API
export async function getPlugins(): Promise<PluginConfig[]> {
  return invoke<PluginConfig[]>("get_plugins");
}

export async function savePlugin(plugin: PluginConfig): Promise<void> {
  return invoke<void>("save_plugin", { plugin });
}

export async function deletePlugin(pluginId: string): Promise<void> {
  return invoke<void>("delete_plugin", { pluginId });
}

export async function testPlugin(pluginId: string): Promise<boolean> {
  return invoke<boolean>("test_plugin", { pluginId });
}

// 4.7 Web Dashboard API
export async function getWebDashboardConfig(): Promise<WebDashboardConfig> {
  return invoke<WebDashboardConfig>("get_web_dashboard_config");
}

export async function setWebDashboardConfig(config: WebDashboardConfig): Promise<void> {
  return invoke<void>("set_web_dashboard_config", { config });
}

export async function getWebDashboardStatus(): Promise<WebDashboardStatus> {
  return invoke<WebDashboardStatus>("get_web_dashboard_status");
}

export async function startWebDashboard(): Promise<WebDashboardStatus> {
  return invoke<WebDashboardStatus>("start_web_dashboard");
}

export async function stopWebDashboard(): Promise<void> {
  return invoke<void>("stop_web_dashboard");
}

// 4.8 Discovery API
export async function getDiscoveredServices(): Promise<DiscoveredService[]> {
  return invoke<DiscoveredService[]>("get_discovered_services");
}

export async function startDiscovery(): Promise<void> {
  return invoke<void>("start_discovery");
}

export async function stopDiscovery(): Promise<void> {
  return invoke<void>("stop_discovery");
}

export async function getDiscoveryStatus(): Promise<{ active: boolean; service_count: number }> {
  return invoke<{ active: boolean; service_count: number }>("get_discovery_status");
}

// 4.9 Ollama Compatibility API
export async function getOllamaConfig(): Promise<OllamaConfig> {
  return invoke<OllamaConfig>("get_ollama_config");
}

export async function checkOllama(config: OllamaConfig): Promise<boolean> {
  return invoke<boolean>("check_ollama", { config });
}

export async function listOllamaModels(config: OllamaConfig): Promise<OllamaModel[]> {
  return invoke<OllamaModel[]>("list_ollama_models", { config });
}

export async function ollamaChat(
  config: OllamaConfig,
  model: string,
  messages: { role: string; content: string }[],
  stream: boolean,
): Promise<any> {
  return invoke("ollama_chat", { config, model, messages, stream });
}

export async function pullOllamaModel(config: OllamaConfig, modelName: string): Promise<string> {
  return invoke<string>("pull_ollama_model", { config, modelName });
}

// 5.7: Tray commands
export async function updateTrayStatus(
  running: boolean,
  nodeCount: number,
  addr?: string,
): Promise<void> {
  return invoke<void>("update_tray_status", { running, nodeCount, addr: addr ?? null });
}

export async function addRecentCluster(addr: string): Promise<void> {
  return invoke<void>("add_recent_cluster", { addr });
}
