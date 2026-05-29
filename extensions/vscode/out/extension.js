"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || (function () {
    var ownKeys = function(o) {
        ownKeys = Object.getOwnPropertyNames || function (o) {
            var ar = [];
            for (var k in o) if (Object.prototype.hasOwnProperty.call(o, k)) ar[ar.length] = k;
            return ar;
        };
        return ownKeys(o);
    };
    return function (mod) {
        if (mod && mod.__esModule) return mod;
        var result = {};
        if (mod != null) for (var k = ownKeys(mod), i = 0; i < k.length; i++) if (k[i] !== "default") __createBinding(result, mod, k[i]);
        __setModuleDefault(result, mod);
        return result;
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
exports.activate = activate;
exports.deactivate = deactivate;
const vscode = __importStar(require("vscode"));
let state = {
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
let _statusModel;
let _statusHealth;
let _statusThroughput;
let _pollTimer;
function createStatusBar() {
    _statusModel = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Left, 100);
    _statusModel.tooltip = "DistLLM — click to open dashboard";
    _statusModel.command = "distllm.openDashboard";
    _statusModel.show();
    _statusHealth = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Left, 99);
    _statusHealth.tooltip = "Cluster node health";
    _statusHealth.show();
    _statusThroughput = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Left, 98);
    _statusThroughput.tooltip = "Live inference throughput";
    _statusThroughput.show();
}
function updateStatusBar() {
    const cfg = vscode.workspace.getConfiguration("distllm");
    const apiUrl = cfg.get("apiUrl", "http://localhost:8000");
    if (!state.connected) {
        _statusModel.text = `$(server) Disconnected`;
        _statusModel.backgroundColor = new vscode.ThemeColor("statusBarItem.warningBackground");
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
    }
    else if (state.tokPerSec > 20) {
        _statusThroughput.color = new vscode.ThemeColor("terminal.ansiYellow");
    }
    else {
        _statusThroughput.color = new vscode.ThemeColor("terminal.ansiRed");
    }
}
// ---------------------------------------------------------------------------
// API calls
// ---------------------------------------------------------------------------
async function fetchHealth(cfg) {
    const apiUrl = cfg.get("apiUrl", "http://localhost:8000");
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
        const data = await resp.json();
        state.connected = true;
        state.model = data.model || cfg.get("model", "") || "unknown";
        const nh = data.node_health || {};
        const entries = Object.entries(nh);
        state.totalNodes = data.nodes || entries.length;
        state.healthyNodes = entries.filter(([, v]) => v.healthy).length;
    }
    catch {
        state.connected = false;
        state.model = "";
        state.totalNodes = 0;
        state.healthyNodes = 0;
    }
    updateStatusBar();
}
async function fetchMetrics(cfg) {
    const apiUrl = cfg.get("apiUrl", "http://localhost:8000");
    try {
        const resp = await fetch(`${apiUrl}/api/metrics/collector`, {
            signal: AbortSignal.timeout(5000),
        });
        if (!resp.ok)
            return;
        const data = await resp.json();
        state.tokPerSec = data.throughput?.tokens_per_sec_avg ?? 0;
        state.avgLatencyMs = data.latency?.avg ?? 0;
    }
    catch {
        // non-critical — leave previous values
    }
    updateStatusBar();
}
async function refreshAll() {
    const cfg = vscode.workspace.getConfiguration("distllm");
    await fetchHealth(cfg);
    await fetchMetrics(cfg);
}
// ---------------------------------------------------------------------------
// Send to DistLLM
// ---------------------------------------------------------------------------
async function sendToModel(apiUrl, modelName, text, maxTokens, temperature) {
    // Try completions endpoint first, fall back to chat completions
    const endpoints = [
        {
            url: `${apiUrl}/v1/completions`,
            body: { model: modelName, prompt: text, max_tokens: maxTokens, temperature },
            extract: (d) => d.choices?.[0]?.text ?? "",
        },
        {
            url: `${apiUrl}/v1/chat/completions`,
            body: {
                model: modelName,
                messages: [{ role: "user", content: text }],
                max_tokens: maxTokens,
                temperature,
            },
            extract: (d) => d.choices?.[0]?.message?.content ?? "",
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
            if (!resp.ok && resp.status >= 400 && resp.status < 500)
                continue; // try next
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
        }
        catch {
            continue;
        }
    }
    throw new Error("Both /v1/completions and /v1/chat/completions failed");
}
async function sendSelection() {
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
    const apiUrl = cfg.get("apiUrl", "http://localhost:8000");
    const modelOverride = cfg.get("model", "");
    const maxTokens = cfg.get("maxTokens", 256);
    const temperature = cfg.get("temperature", 0.7);
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
    }
    catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        channel.appendLine(`[DistLLM] ERROR: ${msg}`);
        vscode.window.showErrorMessage(`DistLLM request failed: ${msg}`);
    }
}
// ---------------------------------------------------------------------------
// Dashboard
// ---------------------------------------------------------------------------
async function openDashboard() {
    const cfg = vscode.workspace.getConfiguration("distllm");
    const apiUrl = cfg.get("apiUrl", "http://localhost:8000");
    const dashboardUrl = `${apiUrl}/dashboard`;
    vscode.env.openExternal(vscode.Uri.parse(dashboardUrl));
}
// ---------------------------------------------------------------------------
// Activate / deactivate
// ---------------------------------------------------------------------------
function activate(context) {
    createStatusBar();
    context.subscriptions.push(vscode.commands.registerCommand("distllm.sendSelection", sendSelection), vscode.commands.registerCommand("distllm.openDashboard", openDashboard), vscode.commands.registerCommand("distllm.refreshStatus", refreshAll));
    // Initial fetch
    refreshAll();
    // Polling
    const cfg = vscode.workspace.getConfiguration("distllm");
    const intervalSec = cfg.get("refreshInterval", 10);
    _pollTimer = setInterval(refreshAll, intervalSec * 1000);
    // Re-trigger on config change
    context.subscriptions.push(vscode.workspace.onDidChangeConfiguration((e) => {
        if (e.affectsConfiguration("distllm")) {
            clearInterval(_pollTimer);
            const newInterval = vscode.workspace
                .getConfiguration("distllm")
                .get("refreshInterval", 10);
            _pollTimer = setInterval(refreshAll, newInterval * 1000);
            refreshAll();
        }
    }));
    // Clean up
    context.subscriptions.push({
        dispose: () => {
            if (_pollTimer)
                clearInterval(_pollTimer);
            _statusModel?.dispose();
            _statusHealth?.dispose();
            _statusThroughput?.dispose();
        },
    });
}
function deactivate() {
    if (_pollTimer)
        clearInterval(_pollTimer);
    _statusModel?.dispose();
    _statusHealth?.dispose();
    _statusThroughput?.dispose();
}
//# sourceMappingURL=extension.js.map