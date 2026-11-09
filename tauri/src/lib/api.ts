import { invoke } from "@tauri-apps/api/core";
import type {
  ClusterStatus,
  PeerInfo,
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
import {
  isTauri,
  apiBaseUrl,
  authHeaders,
  BrowserUnsupportedError,
  DEFAULT_API_BASE,
} from "./transport";

export { BrowserUnsupportedError };

// ── Browser-mode helpers ──────────────────────────────────────────
// Every exported function below has two branches: `invoke(...)` in the Tauri
// shell (unchanged behaviour) and direct HTTP (or a typed error / no-op) in a
// plain browser, where `window.__TAURI_INTERNALS__` is undefined.

/** Read a failed response body and surface the server's message verbatim. */
async function httpError(res: Response): Promise<Error> {
  const text = await res.text().catch(() => "");
  let detail = text;
  try {
    const parsed = JSON.parse(text);
    detail = parsed?.error?.message ?? parsed?.detail ?? text;
  } catch {
    // not JSON — keep raw text
  }
  return new Error(`HTTP ${res.status}: ${detail || res.statusText}`);
}

/** GET JSON with the coordinator Authorization header. */
async function getJson<T>(path: string, authenticated = true): Promise<T> {
  const res = await fetch(`${apiBaseUrl()}${path}`, {
    headers: authenticated ? authHeaders() : undefined,
  });
  if (!res.ok) throw await httpError(res);
  return (await res.json()) as T;
}

interface NodeApiInfo {
  node_id?: string;
  host?: string;
  port?: number;
  healthy?: boolean;
  gpu_name?: string;
  start_layer?: number;
  end_layer?: number;
  /** Server computes this from free/total memory; it never sends "gpu_utilization". */
  gpu_memory_util_pct?: number;
}

function mapNodes(nodes: NodeApiInfo[]): PeerInfo[] {
  return nodes.map((n) => ({
    node_id: String(n.node_id ?? ""),
    host: n.host ?? "",
    port: n.port ?? 0,
    healthy: n.healthy === true,
    gpu_name: n.gpu_name ?? "",
    gpu_utilization: n.gpu_memory_util_pct ?? 0,
    layers: `${n.start_layer ?? 0}-${n.end_layer ?? 0}`,
  }));
}

/**
 * Resolve a chat/benchmark model id against GET /v1/models.
 * A caller-supplied id is used only if the server advertises it; otherwise
 * the first advertised id wins. Returns null when the list is unavailable
 * and no candidate was supplied.
 */
async function resolveChatModel(
  base: string,
  candidate?: string,
): Promise<string | null> {
  const trimmed = candidate?.trim();
  try {
    const res = await fetch(`${base}/v1/models`, { headers: authHeaders() });
    if (res.ok) {
      const body = (await res.json()) as { data?: { id?: string }[] };
      const ids = (body.data ?? [])
        .map((m) => m.id)
        .filter((id): id is string => typeof id === "string" && id.length > 0);
      if (ids.length > 0) {
        return trimmed && ids.includes(trimmed) ? trimmed : ids[0];
      }
    }
  } catch {
    // Network-level failure: tell the user *where* we tried, instead of
    // surfacing a bare TypeError from fetch. (Stale apiEndpoint values in
    // localStorage are the usual cause.)
    const where = base || (typeof window !== "undefined" ? window.location.origin : "");
    throw new Error(
      `Backend unreachable at ${where || DEFAULT_API_BASE}. ` +
        "Is the DistLLM server running? Check the API Endpoint in Settings.",
    );
  }
  return trimmed && trimmed.length > 0 ? trimmed : null;
}

// ── 4.1 Cluster API ───────────────────────────────────────────────

export async function createCluster(
  port?: number,
  model?: string,
): Promise<ClusterStatus> {
  if (!isTauri()) {
    throw new BrowserUnsupportedError("Creating a cluster");
  }
  return invoke<ClusterStatus>("create_cluster", { port, model });
}

export async function joinCluster(
  host: string,
  port: number,
): Promise<ClusterStatus> {
  if (!isTauri()) {
    throw new BrowserUnsupportedError("Joining a cluster");
  }
  return invoke<ClusterStatus>("join_cluster", { host, port });
}

export async function leaveCluster(): Promise<void> {
  if (!isTauri()) {
    throw new BrowserUnsupportedError("Leaving a cluster");
  }
  return invoke<void>("leave_cluster");
}

export async function getClusterStatus(): Promise<ClusterStatus> {
  if (isTauri()) {
    return invoke<ClusterStatus>("get_cluster_status");
  }

  // Browser: the coordinator's /health endpoint is the liveness signal.
  // Never throw — cluster-store polls this every 3 s and expects a value.
  const base = apiBaseUrl();
  let running = false;
  try {
    const res = await fetch(`${base}/health`);
    running = res.ok;
  } catch {
    running = false;
  }

  if (!running) {
    return {
      running: false,
      node_id: null,
      role: null,
      coordinator_addr: null,
      nodes: [],
      uptime_secs: 0,
    };
  }

  let nodes: PeerInfo[] = [];
  try {
    const body = await getJson<{ nodes?: NodeApiInfo[] }>("/admin/v1/nodes");
    nodes = mapNodes(body.nodes ?? []);
  } catch {
    // Missing/invalid API key or older server — report the coordinator
    // itself, just without worker detail.
  }

  return {
    running: true,
    node_id: null,
    role: "coordinator",
    // Chat/Benchmark treat a falsy coordinator_addr as "no cluster running".
    // In same-origin (Vite proxy) mode `base` is "", so make it explicit.
    coordinator_addr:
      base || (typeof window !== "undefined" ? window.location.origin : DEFAULT_API_BASE),
    nodes,
    uptime_secs: 0,
  };
}

export async function getGpuMetrics(): Promise<GpuInfo[]> {
  if (isTauri()) {
    return invoke<GpuInfo[]>("get_gpu_metrics");
  }

  // Browser: scrape the node GPU gauges from the Prometheus text endpoint.
  // The server exposes only utilization and used-memory per node; anything
  // it does not send (temperature, total memory) stays 0 rather than being
  // invented.
  const base = apiBaseUrl();
  let text: string;
  try {
    const res = await fetch(`${base}/metrics`, { headers: authHeaders() });
    if (!res.ok) return [];
    text = await res.text();
  } catch {
    return [];
  }

  const utilRe = /^distllm_node_gpu_utilization_percent\{[^}]*node_id="([^"]*)"[^}]*\}\s+(\S+)/;
  const memRe = /^distllm_node_gpu_memory_bytes\{[^}]*node_id="([^"]*)"[^}]*\}\s+(\S+)/;
  const byNode = new Map<string, { utilization: number; memoryUsed: number }>();

  for (const line of text.split("\n")) {
    const u = utilRe.exec(line);
    if (u) {
      const value = Number(u[2]);
      if (Number.isFinite(value)) {
        const entry = byNode.get(u[1]) ?? { utilization: 0, memoryUsed: 0 };
        entry.utilization = value;
        byNode.set(u[1], entry);
      }
      continue;
    }
    const m = memRe.exec(line);
    if (m) {
      const value = Number(m[2]);
      if (Number.isFinite(value)) {
        const entry = byNode.get(m[1]) ?? { utilization: 0, memoryUsed: 0 };
        entry.memoryUsed = value;
        byNode.set(m[1], entry);
      }
    }
  }

  return [...byNode.entries()].map(([nodeId, v], index) => ({
    index,
    name: nodeId,
    temperature: 0,
    utilization: v.utilization,
    memory_total: 0,
    memory_used: v.memoryUsed,
    memory_free: 0,
  }));
}

// ── Models ────────────────────────────────────────────────────────

interface RegistryModel {
  id?: string;
  loaded?: boolean;
  dtype?: string | null;
  memory_used_mb?: number;
  gpu_name?: string | null;
}

export async function listModels(): Promise<ModelInfo[]> {
  if (isTauri()) {
    return invoke<ModelInfo[]>("list_models");
  }

  const base = apiBaseUrl();
  // Primary: the registry knows dtype/loaded state.
  try {
    const body = await getJson<{ models?: RegistryModel[] }>("/api/models/registry");
    if (Array.isArray(body.models)) {
      return body.models.map((m) => {
        const id = String(m.id ?? "");
        return {
          id,
          name: id.split("/").pop() ?? id,
          size:
            typeof m.memory_used_mb === "number" && m.memory_used_mb > 0
              ? `${Math.round(m.memory_used_mb)} MB`
              : "—",
          downloaded: m.loaded === true,
          quantization: m.dtype ? [m.dtype] : [],
          gpu_required: m.gpu_name ?? "unknown",
        };
      });
    }
  } catch {
    // Registry unavailable (e.g. missing key) — try the OpenAI-style list.
  }

  // Fallback: /v1/models carries ids only.
  const body = await getJson<{ data?: { id?: string; owned_by?: string }[] }>(
    "/v1/models",
  );
  return (body.data ?? []).map((m) => {
    const id = String(m.id ?? "");
    return {
      id,
      name: id.split("/").pop() ?? id,
      size: "—",
      downloaded: false,
      quantization: [],
      gpu_required: "unknown",
    };
  });
}

export async function downloadModel(modelId: string): Promise<string> {
  if (isTauri()) {
    return invoke<string>("download_model", { modelId });
  }

  // Browser: REST model "load" is the closest equivalent to the desktop
  // download command — it asks the coordinator to fetch/initialise the model.
  const res = await fetch(`${apiBaseUrl()}/api/models/load`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...authHeaders() },
    body: JSON.stringify({ model: modelId, dtype: "float16" }),
  });
  if (!res.ok) throw await httpError(res);
  const body = (await res.json().catch(() => ({}))) as {
    status?: string;
    model?: string;
  };
  return `Load requested: ${body.model ?? modelId} (${body.status ?? "accepted"})`;
}

// ── Invites & system ──────────────────────────────────────────────

export async function generateInvite(): Promise<InviteInfo> {
  if (!isTauri()) {
    throw new BrowserUnsupportedError("Generating a cluster invite");
  }
  return invoke<InviteInfo>("generate_invite");
}

export async function getSystemInfo(): Promise<SystemInfo> {
  if (isTauri()) {
    return invoke<SystemInfo>("get_system_info");
  }

  // Browser: honest partial view. CPU/RAM/Python are host facts only the
  // desktop shell can read; /metrics carries no version label. Never throw —
  // this is polled alongside the cluster status.
  return {
    os: navigator.userAgent,
    cpu: "unknown",
    ram_gb: 0,
    python_version: null,
    distllm_version: "unknown",
    gpus: [],
  };
}

export async function checkCoordinator(
  host: string,
  port: number,
): Promise<boolean> {
  if (isTauri()) {
    return invoke<boolean>("check_coordinator", { host, port });
  }
  try {
    const res = await fetch(`http://${host}:${port}/health`);
    return res.ok;
  } catch {
    return false;
  }
}

// ── 4.2 Chat / inference ──────────────────────────────────────────

export interface StreamCallbacks {
  onToken: (token: string) => void;
  onDone: (metrics: {
    /** null when the stream produced no tokens before finishing. */
    ttft: number | null;
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
  const base = baseUrl || apiBaseUrl();
  const apiMessages = messages.map((m) => ({
    role: m.role,
    content: m.content,
  }));

  // Resolve the real model id against /v1/models. The server rejects the old
  // hard-coded "local" with HTTP 400 in both Tauri and browser modes.
  // resolveChatModel throws on network failure (unreachable backend); route
  // that to onError so callers that don't await (chat-store) never see an
  // unhandled rejection.
  let model: string | null;
  try {
    model = await resolveChatModel(base, options.model);
  } catch (e: unknown) {
    callbacks.onError(e instanceof Error ? e.message : String(e));
    return;
  }
  if (!model) {
    callbacks.onError(
      "Could not resolve a model: GET /v1/models returned no models (check the API endpoint and API key in Settings).",
    );
    return;
  }

  const body = {
    model,
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
    const response = await fetch(`${base}/v1/chat/completions`, {
      method: "POST",
      headers: { "Content-Type": "application/json", ...authHeaders() },
      body: JSON.stringify(body),
      signal,
    });

    if (!response.ok) {
      const errorText = await response.text();
      let detail = errorText;
      try {
        const parsed = JSON.parse(errorText);
        detail = parsed?.error?.message ?? parsed?.detail ?? errorText;
      } catch {
        // not JSON
      }
      callbacks.onError(`HTTP ${response.status}: ${detail}`);
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
            ttft: ttft ?? 0,
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
      ttft: ttft ?? 0,
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
  const base = baseUrl || apiBaseUrl();
  const results: BenchmarkRun[] = [];
  const prompt = "Explain the theory of relativity in detail. ".repeat(
    Math.ceil(config.prompt_length / 10),
  );

  // Same model-id resolution as chat: use config.model only when the server
  // advertises it, else the first advertised id.
  const model = (await resolveChatModel(base, config.model)) ?? config.model;

  for (let i = 0; i < config.num_runs; i++) {
    const startTime = performance.now();
    let ttft: number | null = null;
    let tokenCount = 0;
    let lastTokenTime = startTime;
    const tokenTimes: number[] = [];

    try {
      const response = await fetch(`${base}/v1/chat/completions`, {
        method: "POST",
        headers: { "Content-Type": "application/json", ...authHeaders() },
        body: JSON.stringify({
          model,
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
        const errorText = await response.text().catch(() => "");
        throw new Error(`HTTP ${response.status}: ${errorText}`);
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
        model,
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
        model,
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
// Model slots are Rust-local process state — there is no REST equivalent.
export async function getModelSlots(): Promise<ModelSlot[]> {
  if (!isTauri()) {
    throw new BrowserUnsupportedError("Model slots");
  }
  return invoke<ModelSlot[]>("get_model_slots");
}

export async function loadModelSlot(
  slotId: string,
  modelId: string,
): Promise<ModelSlot> {
  if (!isTauri()) {
    throw new BrowserUnsupportedError("Loading a model slot");
  }
  return invoke<ModelSlot>("load_model_slot", { slotId, modelId });
}

export async function unloadModelSlot(slotId: string): Promise<void> {
  if (!isTauri()) {
    throw new BrowserUnsupportedError("Unloading a model slot");
  }
  return invoke<void>("unload_model_slot", { slotId });
}

interface RouterRuleApi {
  name?: string;
  match_type?: string;
  pattern?: string;
  target_model?: string;
  priority?: number;
}

export async function getRoutingRules(): Promise<ModelRoutingRule[]> {
  if (isTauri()) {
    return invoke<ModelRoutingRule[]>("get_routing_rules");
  }

  const body = await getJson<{ rules?: RouterRuleApi[] }>("/v1/router/rules");
  return (body.rules ?? []).map((r) => ({
    id: String(r.name ?? ""),
    pattern: r.pattern ?? "",
    target_slot: r.target_model ?? "",
    priority: r.priority ?? 0,
  }));
}

export async function setRoutingRule(
  rule: ModelRoutingRule,
): Promise<void> {
  if (isTauri()) {
    return invoke<void>("set_routing_rule", { rule });
  }

  // The REST rule schema keys on name/match_type/target_model. The UI's
  // "pattern" field is documented as regex-or-model-name, so match_type
  // "regex" is the faithful mapping; the rule id doubles as the rule name.
  const res = await fetch(`${apiBaseUrl()}/v1/router/rules`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...authHeaders() },
    body: JSON.stringify({
      name: rule.id,
      match_type: "regex",
      pattern: rule.pattern,
      target_model: rule.target_slot,
      priority: rule.priority,
    }),
  });
  if (!res.ok) throw await httpError(res);
}

export async function deleteRoutingRule(ruleId: string): Promise<void> {
  if (isTauri()) {
    return invoke<void>("delete_routing_rule", { ruleId });
  }

  const res = await fetch(
    `${apiBaseUrl()}/v1/router/rules/${encodeURIComponent(ruleId)}`,
    { method: "DELETE", headers: authHeaders() },
  );
  if (!res.ok) throw await httpError(res);
}

// 4.6 Plugin API
interface PluginRegistryEntry {
  description?: string;
  config?: string;
  hooks?: string;
}

function pluginKind(name: string): PluginConfig["kind"] {
  if (name.includes("auth")) return "auth";
  if (name.includes("rate") || name.includes("limit")) return "backend";
  return "monitoring";
}

export async function getPlugins(): Promise<PluginConfig[]> {
  if (isTauri()) {
    return invoke<PluginConfig[]>("get_plugins");
  }

  const body = await getJson<{
    builtin?: Record<string, PluginRegistryEntry>;
    installed?: string[];
  }>("/v1/plugins/registry");
  const installed = new Set(body.installed ?? []);
  return Object.entries(body.builtin ?? {}).map(([name, doc]) => ({
    id: name,
    name,
    kind: pluginKind(name),
    enabled: installed.has(name),
    endpoint: "",
    api_key: "",
    extra: {
      description: doc.description ?? "",
      hooks: doc.hooks ?? "",
      config: doc.config ?? "",
    },
    created_at: 0,
  }));
}

// The registry is read-only documentation; there is no REST create/delete/test.
export async function savePlugin(plugin: PluginConfig): Promise<void> {
  if (!isTauri()) {
    throw new BrowserUnsupportedError("Saving a plugin");
  }
  return invoke<void>("save_plugin", { plugin });
}

export async function deletePlugin(pluginId: string): Promise<void> {
  if (!isTauri()) {
    throw new BrowserUnsupportedError("Deleting a plugin");
  }
  return invoke<void>("delete_plugin", { pluginId });
}

export async function testPlugin(pluginId: string): Promise<boolean> {
  if (!isTauri()) {
    throw new BrowserUnsupportedError("Testing a plugin");
  }
  return invoke<boolean>("test_plugin", { pluginId });
}

// 4.7 Web Dashboard API — the dashboard server is a Rust-hosted process.
export async function getWebDashboardConfig(): Promise<WebDashboardConfig> {
  if (!isTauri()) {
    throw new BrowserUnsupportedError("The web dashboard");
  }
  return invoke<WebDashboardConfig>("get_web_dashboard_config");
}

export async function setWebDashboardConfig(config: WebDashboardConfig): Promise<void> {
  if (!isTauri()) {
    throw new BrowserUnsupportedError("The web dashboard");
  }
  return invoke<void>("set_web_dashboard_config", { config });
}

export async function getWebDashboardStatus(): Promise<WebDashboardStatus> {
  if (!isTauri()) {
    throw new BrowserUnsupportedError("The web dashboard");
  }
  return invoke<WebDashboardStatus>("get_web_dashboard_status");
}

export async function startWebDashboard(): Promise<WebDashboardStatus> {
  if (!isTauri()) {
    throw new BrowserUnsupportedError("The web dashboard");
  }
  return invoke<WebDashboardStatus>("start_web_dashboard");
}

export async function stopWebDashboard(): Promise<void> {
  if (!isTauri()) {
    throw new BrowserUnsupportedError("The web dashboard");
  }
  return invoke<void>("stop_web_dashboard");
}

// 4.8 Discovery API — mDNS discovery runs in the Rust shell.
export async function getDiscoveredServices(): Promise<DiscoveredService[]> {
  if (!isTauri()) {
    throw new BrowserUnsupportedError("Service discovery");
  }
  return invoke<DiscoveredService[]>("get_discovered_services");
}

export async function startDiscovery(): Promise<void> {
  if (!isTauri()) {
    throw new BrowserUnsupportedError("Service discovery");
  }
  return invoke<void>("start_discovery");
}

export async function stopDiscovery(): Promise<void> {
  if (!isTauri()) {
    throw new BrowserUnsupportedError("Service discovery");
  }
  return invoke<void>("stop_discovery");
}

export async function getDiscoveryStatus(): Promise<{ active: boolean; service_count: number }> {
  if (!isTauri()) {
    throw new BrowserUnsupportedError("Service discovery");
  }
  return invoke<{ active: boolean; service_count: number }>("get_discovery_status");
}

// 4.9 Ollama Compatibility API — Ollama has its own HTTP API, so these work
// in a plain browser too. Caveat: Ollama only serves CORS headers when
// OLLAMA_ORIGINS includes this origin (it allows localhost by default).
export async function getOllamaConfig(): Promise<OllamaConfig> {
  if (!isTauri()) {
    return { host: "127.0.0.1", port: 11434, enabled: false };
  }
  return invoke<OllamaConfig>("get_ollama_config");
}

function ollamaBase(config: OllamaConfig): string {
  return `http://${config.host}:${config.port}`;
}

export async function checkOllama(config: OllamaConfig): Promise<boolean> {
  if (isTauri()) {
    return invoke<boolean>("check_ollama", { config });
  }
  try {
    const res = await fetch(`${ollamaBase(config)}/api/tags`);
    return res.ok;
  } catch {
    return false;
  }
}

export async function listOllamaModels(config: OllamaConfig): Promise<OllamaModel[]> {
  if (isTauri()) {
    return invoke<OllamaModel[]>("list_ollama_models", { config });
  }
  const res = await fetch(`${ollamaBase(config)}/api/tags`);
  if (!res.ok) throw await httpError(res);
  const body = (await res.json()) as {
    models?: { name?: string; size?: number; digest?: string; modified_at?: string }[];
  };
  return (body.models ?? []).map((m) => ({
    name: m.name ?? "",
    size: m.size ?? 0,
    digest: m.digest ?? "",
    modified_at: m.modified_at ?? "",
  }));
}

export async function ollamaChat(
  config: OllamaConfig,
  model: string,
  messages: { role: string; content: string }[],
  stream: boolean,
): Promise<any> {
  if (isTauri()) {
    return invoke("ollama_chat", { config, model, messages, stream });
  }
  const res = await fetch(`${ollamaBase(config)}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model, messages, stream }),
  });
  if (!res.ok) throw await httpError(res);
  return stream ? res.body : res.json();
}

export async function pullOllamaModel(config: OllamaConfig, modelName: string): Promise<string> {
  if (isTauri()) {
    return invoke<string>("pull_ollama_model", { config, modelName });
  }
  const res = await fetch(`${ollamaBase(config)}/api/pull`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name: modelName, stream: false }),
  });
  if (!res.ok) throw await httpError(res);
  const body = (await res.json().catch(() => ({}))) as { status?: string };
  return body.status ?? "success";
}

// 5.7: Tray commands — pure Tauri chrome; resolve to no-ops in the browser
// because callers (App.svelte) invoke them without awaiting or catching.
export async function updateTrayStatus(
  running: boolean,
  nodeCount: number,
  addr?: string,
): Promise<void> {
  if (!isTauri()) return;
  return invoke<void>("update_tray_status", { running, nodeCount, addr: addr ?? null });
}

export async function addRecentCluster(addr: string): Promise<void> {
  if (!isTauri()) return;
  return invoke<void>("add_recent_cluster", { addr });
}

// ── Notifications ──────────────────────────────────────────────
import { isPermissionGranted, requestPermission, sendNotification } from "@tauri-apps/plugin-notification";

export async function initNotifications(): Promise<boolean> {
  if (!isTauri()) {
    // Browser Notification API. Requesting permission is only allowed from a
    // user gesture in some browsers; failures degrade to "not granted".
    if (typeof Notification === "undefined") return false;
    if (Notification.permission === "granted") return true;
    if (Notification.permission === "denied") return false;
    try {
      return (await Notification.requestPermission()) === "granted";
    } catch {
      return false;
    }
  }
  let granted = await isPermissionGranted();
  if (!granted) {
    const permission = await requestPermission();
    granted = permission === "granted";
  }
  return granted;
}

export function notify(title: string, body: string): void {
  if (!isTauri()) {
    try {
      if (typeof Notification !== "undefined" && Notification.permission === "granted") {
        new Notification(title, { body });
      } else {
        console.info(`[notification] ${title}: ${body}`);
      }
    } catch {
      console.info(`[notification] ${title}: ${body}`);
    }
    return;
  }
  sendNotification({ title, body });
}

// ── Updater ────────────────────────────────────────────────────
import { check } from "@tauri-apps/plugin-updater";
import type { Update } from "@tauri-apps/plugin-updater";

export async function checkForUpdates(): Promise<Update | null> {
  if (!isTauri()) return null;
  try {
    const update = await check();
    return update;
  } catch {
    return null;
  }
}

export async function installUpdate(): Promise<void> {
  if (!isTauri()) return;
  const update = await check();
  if (update?.available) {
    await update.downloadAndInstall();
  }
}

export interface UpdateCheckResult {
  available: boolean;
  version?: string;
  body?: string;
  date?: string;
}
