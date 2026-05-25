import * as vscode from "vscode";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface HealthResponse {
  status: string;
  model?: string;
  nodes?: number;
  node_health?: Record<string, { healthy: boolean }>;
}

interface CompletionsResponse {
  choices?: Array<{ text: string }>;
  usage?: { completion_tokens?: number };
}

interface ChatCompletionsResponse {
  choices?: Array<{ message: { content: string } }>;
  usage?: { completion_tokens?: number };
}

interface MetricsResponse {
  latency?: { avg?: number };
  throughput?: { tokens_per_sec_avg?: number };
  [key: string]: unknown;
}

interface CollectorResponse {
  latency?: { avg?: number; p50?: number; p95?: number; p99?: number };
  throughput?: { tokens_per_sec_avg?: number };
  [key: string]: unknown;
}

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

interface ClusterState {
  connected: boolean;
  model: string;
  totalNodes: number;
  healthyNodes: number;
  tokPerSec: number;
  avgLatencyMs: number;
}

let state: ClusterState = {
  connected: false,
  model: "",
  totalNodes: 0,
  healthyNodes: 0,
  tokPerSec: 0,
  avgLatencyMs: 0,
};

// ---------------------------------------------------------------------------
// Status bar
// ---------------------------------------------------------------------------

let _statusModel: vscode.StatusBarItem;
let _statusHealth: vscode.StatusBarItem;
let _statusThroughput: vscode.StatusBarItem;
let _pollTimer: ReturnType<typeof setInterval> | undefined;

function createStatusBar(): void {
  _statusModel = vscode.window.createStatusBarItem(
    vscode.StatusBarAlignment.Left,
    100,
  );
  _statusModel.tooltip = "DistLLM — click to open dashboard";
  _statusModel.command = "distllm.openDashboard";
  _statusModel.show();

  _statusHealth = vscode.window.createStatusBarItem(
    vscode.StatusBarAlignment.Left,
    99,
  );
  _statusHealth.tooltip = "Cluster node health";
  _statusHealth.show();

  _statusThroughput = vscode.window.createStatusBarItem(
    vscode.StatusBarAlignment.Left,
    98,
  );
  _statusThroughput.tooltip = "Live inference throughput";
  _statusThroughput.show();
}

function updateStatusBar(): void {
  const cfg = vscode.workspace.getConfiguration("distllm");
  const apiUrl = cfg.get<string>("apiUrl", "http://localhost:8000");

  if (!state.connected) {
    _statusModel.text = `$(server) Disconnected`;
    _statusModel.backgroundColor = new vscode.ThemeColor(
      "statusBarItem.warningBackground",
    );
    _statusModel.tooltip = `DistLLM — click to open ${apiUrl}`;
    _statusHealth.text = `$(circuit-board) ---`;
    _statusThroughput.text = `$(zap) --- tok/s`;
    _statusThroughput.backgroundColor = undefined;
    return;
  }

  _statusModel.text = `$(server) ${state.model || "unknown"}`;
  _statusModel.backgroundColor = undefined;
  _statusModel.tooltip = `DistLLM: ${state.model} @ ${apiUrl}`;

  const allHealthy = state.totalNodes > 0 && state.healthyNodes === state.totalNodes;
  const healthIcon = allHealthy ? "$(check)" : "$(warning)";
  const healthColor = allHealthy ? "" : `${state.healthyNodes}/${state.totalNodes}`;
  _statusHealth.text = `$(circuit-board) ${healthColor} ${state.healthyNodes}/${state.totalNodes}`;
  _statusHealth.color = allHealthy ? undefined : new vscode.ThemeColor("errorForeground");

  const tokText = state.tokPerSec > 0 ? `${state.tokPerSec.toFixed(1)} tok/s` : "--- tok/s";
  _statusThroughput.text = `$(zap) ${tokText}`;

  // Color coding: green > 50 tok/s, yellow > 20, red <= 20
  if (state.tokPerSec > 50) {
    _statusThroughput.color = new vscode.ThemeColor("terminal.ansiGreen");
  } else if (state.tokPerSec > 20) {
    _statusThroughput.color = new vscode.ThemeColor("terminal.ansiYellow");
  } else {
    _statusThroughput.color = new vscode.ThemeColor("terminal.ansiRed");
  }
}

// ---------------------------------------------------------------------------
// API calls
// ---------------------------------------------------------------------------

async function fetchHealth(cfg: vscode.WorkspaceConfiguration): Promise<void> {
  const apiUrl = cfg.get<string>("apiUrl", "http://localhost:8000");
  try {
    const resp = await fetch(`${apiUrl}/health`, { signal: AbortSignal.timeout(5000) });
    if (!resp.ok) {
      if (resp.status === 503) {
        // Coordinator exists but no model loaded
        state.connected = true;
        state.model = "(no model)";
        state.totalNodes = 0;
        state.healthyNodes = 0;
        updateStatusBar();
        return;
      }
      throw new Error(`HTTP ${resp.status}`);
    }
    const data: HealthResponse = await resp.json();
    state.connected = true;
    state.model = data.model || cfg.get<string>("model", "") || "unknown";

    const nh = data.node_health || {};
    const entries = Object.entries(nh);
    state.totalNodes = data.nodes || entries.length;
    state.healthyNodes = entries.filter(([, v]) => v.healthy).length;
  } catch {
    state.connected = false;
    state.model = "";
    state.totalNodes = 0;
    state.healthyNodes = 0;
  }
  updateStatusBar();
}

async function fetchMetrics(cfg: vscode.WorkspaceConfiguration): Promise<void> {
  const apiUrl = cfg.get<string>("apiUrl", "http://localhost:8000");
  try {
    const resp = await fetch(`${apiUrl}/api/metrics/collector`, {
      signal: AbortSignal.timeout(5000),
    });
    if (!resp.ok) return;
    const data: CollectorResponse = await resp.json();
    state.tokPerSec = data.throughput?.tokens_per_sec_avg ?? 0;
    state.avgLatencyMs = data.latency?.avg ?? 0;
  } catch {
    // non-critical — leave previous values
  }
  updateStatusBar();
}

async function refreshAll(): Promise<void> {
  const cfg = vscode.workspace.getConfiguration("distllm");
  await fetchHealth(cfg);
  await fetchMetrics(cfg);
}

// ---------------------------------------------------------------------------
// Send to DistLLM
// ---------------------------------------------------------------------------

async function sendToModel(
  apiUrl: string,
  modelName: string,
  text: string,
  maxTokens: number,
  temperature: number,
): Promise<{ text: string; tokens: number; elapsed: string }> {
  // Try completions endpoint first, fall back to chat completions
  const endpoints = [
    {
      url: `${apiUrl}/v1/completions`,
      body: { model: modelName, prompt: text, max_tokens: maxTokens, temperature },
      extract: (d: CompletionsResponse) => d.choices?.[0]?.text ?? "",
    },
    {
      url: `${apiUrl}/v1/chat/completions`,
      body: {
        model: modelName,
        messages: [{ role: "user", content: text }],
        max_tokens: maxTokens,
        temperature,
      },
      extract: (d: ChatCompletionsResponse) => d.choices?.[0]?.message?.content ?? "",
    },
  ];

  for (const ep of endpoints) {
    const startTime = Date.now();
    try {
      const resp = await fetch(ep.url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(ep.body),
        signal: AbortSignal.timeout(120_000),
      });
      if (!resp.ok && resp.status >= 400 && resp.status < 500) continue; // try next
      if (!resp.ok) {
        const errText = await resp.text().catch(() => "unknown error");
        return { text: "", tokens: 0, elapsed: "0" };
      }
      const data = await resp.json();
      const elapsed = ((Date.now() - startTime) / 1000).toFixed(2);
      const usage = data.usage || {};
      const tokens = usage.completion_tokens || 0;
      const output = ep.extract(data) || "(empty response)";
      return { text: output, tokens, elapsed };
    } catch {
      continue;
    }
  }
  throw new Error("Both /v1/completions and /v1/chat/completions failed");
}

async function sendSelection(): Promise<void> {
  const editor = vscode.window.activeTextEditor;
  if (!editor) {
    vscode.window.showErrorMessage("No active editor");
    return;
  }

  const selection = editor.selection;
  const text = editor.document.getText(selection);
  if (!text || text.trim().length === 0) {
    vscode.window.showErrorMessage("No text selected");
    return;
  }

  const cfg = vscode.workspace.getConfiguration("distllm");
  const apiUrl = cfg.get<string>("apiUrl", "http://localhost:8000");
  const modelOverride = cfg.get<string>("model", "");
  const maxTokens = cfg.get<number>("maxTokens", 256);
  const temperature = cfg.get<number>("temperature", 0.7);
  const modelName = modelOverride || state.model || "default";

  // Create output channel for the result
  const channel = vscode.window.createOutputChannel("DistLLM Inference");
  channel.show();
  channel.appendLine(`[DistLLM] Sending to ${modelName} @ ${apiUrl}...`);
  channel.appendLine(`[DistLLM] Prompt (${text.length} chars):`);
  channel.appendLine("─".repeat(60));
  channel.appendLine(text);
  channel.appendLine("─".repeat(60));

  try {
    const result = await sendToModel(apiUrl, modelName, text, maxTokens, temperature);
    const tokRate = result.elapsed && parseFloat(result.elapsed) > 0
      ? (result.tokens / parseFloat(result.elapsed)).toFixed(1)
      : "?";

    channel.appendLine(`[DistLLM] Response (${result.tokens} tokens, ${result.elapsed}s, ${tokRate} tok/s):`);
    channel.appendLine("─".repeat(60));
    channel.appendLine(result.text.trimEnd());
    channel.appendLine("─".repeat(60));
    channel.appendLine(`[DistLLM] Done.`);
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    channel.appendLine(`[DistLLM] ERROR: ${msg}`);
    vscode.window.showErrorMessage(`DistLLM request failed: ${msg}`);
  }
}

// ---------------------------------------------------------------------------
// Dashboard
// ---------------------------------------------------------------------------

async function openDashboard(): Promise<void> {
  const cfg = vscode.workspace.getConfiguration("distllm");
  const apiUrl = cfg.get<string>("apiUrl", "http://localhost:8000");
  const dashboardUrl = `${apiUrl}/dashboard`;
  vscode.env.openExternal(vscode.Uri.parse(dashboardUrl));
}

// ---------------------------------------------------------------------------
// Activate / deactivate
// ---------------------------------------------------------------------------

export function activate(context: vscode.ExtensionContext): void {
  createStatusBar();

  context.subscriptions.push(
    vscode.commands.registerCommand("distllm.sendSelection", sendSelection),
    vscode.commands.registerCommand("distllm.openDashboard", openDashboard),
    vscode.commands.registerCommand("distllm.refreshStatus", refreshAll),
  );

  // Initial fetch
  refreshAll();

  // Polling
  const cfg = vscode.workspace.getConfiguration("distllm");
  const intervalSec = cfg.get<number>("refreshInterval", 10);
  _pollTimer = setInterval(refreshAll, intervalSec * 1000);

  // Re-trigger on config change
  context.subscriptions.push(
    vscode.workspace.onDidChangeConfiguration((e) => {
      if (e.affectsConfiguration("distllm")) {
        clearInterval(_pollTimer);
        const newInterval = vscode.workspace
          .getConfiguration("distllm")
          .get<number>("refreshInterval", 10);
        _pollTimer = setInterval(refreshAll, newInterval * 1000);
        refreshAll();
      }
    }),
  );

  // Clean up
  context.subscriptions.push({
    dispose: () => {
      if (_pollTimer) clearInterval(_pollTimer);
      _statusModel?.dispose();
      _statusHealth?.dispose();
      _statusThroughput?.dispose();
    },
  });
}

export function deactivate(): void {
  if (_pollTimer) clearInterval(_pollTimer);
  _statusModel?.dispose();
  _statusHealth?.dispose();
  _statusThroughput?.dispose();
}
