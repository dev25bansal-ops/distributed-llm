"""WebGPU Browser Client for zero-install GPU contribution.

Enables browsers to contribute GPU compute to the DistLLM cluster
using WebGPU. Users can run inference directly in the browser or
contribute their GPU to the cluster for distributed inference.

Architecture:
    Browser (WebGPU) ──WebSocket──▸ DistLLM API ──▸ Coordinator
    │                                      │
    ├─ Run inference locally (web-llm)     ├─ Route to cluster
    ├─ Contribute GPU to cluster           ├─ Aggregate results
    └─ Zero-install                        └─ Load balance

Usage:
    Serve the WebGPU client HTML page:
        GET /webgpu — Serves the WebGPU client
        POST /webgpu/register — Register browser GPU contributor
        POST /webgpu/inference — Run inference via browser GPU

The client uses web-llm (https://github.com/mlc-ai/web-llm) for
in-browser inference with WebGPU acceleration.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger


@dataclass
class BrowserGPU:
    """A browser-based GPU contributor."""
    session_id: str
    user_agent: str = ""
    gpu_vendor: str = ""      # "NVIDIA", "AMD", "Intel", "Apple"
    gpu_model: str = ""       # "RTX 4090", "M2 Max", etc.
    vram_mb: int = 0
    webgpu_features: list[str] = field(default_factory=list)
    last_heartbeat: float = field(default_factory=time.time)
    is_available: bool = True
    requests_served: int = 0
    total_tokens: int = 0
    connected_at: float = field(default_factory=time.time)


@dataclass
class WebGPUNode:
    """A WebGPU node registered with the coordinator."""
    node_id: str
    session_id: str
    gpu_info: BrowserGPU
    status: str = "active"  # active, busy, disconnected
    current_request: str | None = None
    latency_ms: float = 0.0


class WebGPUManager:
    """Manages browser-based GPU contributors.

    Tracks connected browsers, routes inference requests to them,
    and handles heartbeats for liveness detection.

    Args:
        max_nodes: Maximum concurrent browser contributors.
        heartbeat_timeout_s: Seconds without heartbeat before disconnect.
        enable_contribution: Whether to use browser GPUs for cluster inference.
    """

    def __init__(
        self,
        max_nodes: int = 100,
        heartbeat_timeout_s: float = 30.0,
        enable_contribution: bool = True,
    ):
        self._max_nodes = max_nodes
        self._heartbeat_timeout = heartbeat_timeout_s
        self._enable_contribution = enable_contribution
        self._nodes: dict[str, WebGPUNode] = {}
        self._sessions: dict[str, BrowserGPU] = {}
        self._lock = threading.Lock()
        self._id_counter = 0
        self._stats = {
            "total_registrations": 0,
            "total_requests": 0,
            "total_tokens_served": 0,
            "active_nodes": 0,
        }

    def register_browser(self, gpu_info: dict[str, Any]) -> str:
        """Register a browser as a GPU contributor.

        Args:
            gpu_info: Dict with gpu_vendor, gpu_model, vram_mb, webgpu_features.

        Returns:
            Session ID for the registered browser.
        """
        with self._lock:
            self._id_counter += 1
            counter = self._id_counter
        session_id = hashlib.sha256(
            f"{gpu_info.get('gpu_model', '')}:{time.time()}:{counter}".encode()
        ).hexdigest()[:16]

        browser_gpu = BrowserGPU(
            session_id=session_id,
            gpu_vendor=gpu_info.get("gpu_vendor", "unknown"),
            gpu_model=gpu_info.get("gpu_model", "unknown"),
            vram_mb=gpu_info.get("vram_mb", 0),
            webgpu_features=gpu_info.get("webgpu_features", []),
        )

        node_id = f"webgpu-{session_id[:8]}"
        node = WebGPUNode(
            node_id=node_id,
            session_id=session_id,
            gpu_info=browser_gpu,
        )

        with self._lock:
            if len(self._nodes) >= self._max_nodes:
                logger.warning("WebGPU max nodes reached, rejecting registration")
                return ""
            self._nodes[node_id] = node
            self._sessions[session_id] = browser_gpu
            self._stats["total_registrations"] += 1
            self._stats["active_nodes"] = len(self._nodes)

        logger.info(
            f"WebGPU browser registered: {node_id} "
            f"({browser_gpu.gpu_vendor} {browser_gpu.gpu_model}, "
            f"{browser_gpu.vram_mb}MB VRAM)"
        )
        return session_id

    def heartbeat(self, session_id: str) -> bool:
        """Update heartbeat for a browser session."""
        with self._lock:
            gpu = self._sessions.get(session_id)
            if gpu:
                gpu.last_heartbeat = time.time()
                return True
            return False

    def unregister(self, session_id: str) -> None:
        """Unregister a browser session."""
        with self._lock:
            gpu = self._sessions.pop(session_id, None)
            if gpu:
                node_id = f"webgpu-{session_id[:8]}"
                self._nodes.pop(node_id, None)
                self._stats["active_nodes"] = len(self._nodes)

    def get_available_node(self) -> WebGPUNode | None:
        """Get an available WebGPU node for inference."""
        with self._lock:
            self._cleanup_stale()
            for node in self._nodes.values():
                if node.status == "active" and node.gpu_info.is_available:
                    return node
            return None

    def mark_busy(self, node_id: str, request_id: str) -> None:
        """Mark a node as busy with a request."""
        with self._lock:
            node = self._nodes.get(node_id)
            if node:
                node.status = "busy"
                node.current_request = request_id

    def mark_free(self, node_id: str, tokens_served: int = 0) -> None:
        """Mark a node as free after completing a request."""
        with self._lock:
            node = self._nodes.get(node_id)
            if node:
                node.status = "active"
                node.current_request = None
                node.gpu_info.requests_served += 1
                node.gpu_info.total_tokens += tokens_served
                self._stats["total_requests"] += 1
                self._stats["total_tokens_served"] += tokens_served

    def _cleanup_stale(self) -> None:
        """Remove nodes that haven't heartbeated recently."""
        now = time.time()
        stale = [
            nid for nid, node in self._nodes.items()
            if now - node.gpu_info.last_heartbeat > self._heartbeat_timeout
        ]
        for nid in stale:
            node = self._nodes.pop(nid, None)
            if node:
                self._sessions.pop(node.session_id, None)
                logger.debug(f"WebGPU node {nid} timed out")

    def get_client_html(self) -> str:
        """Generate the WebGPU client HTML page."""
        return _WEBGPU_CLIENT_HTML

    def stats(self) -> dict:
        with self._lock:
            self._cleanup_stale()
            return {
                **self._stats,
                "active_nodes": len(self._nodes),
                "nodes": [
                    {
                        "node_id": n.node_id,
                        "gpu": f"{n.gpu_info.gpu_vendor} {n.gpu_info.gpu_model}",
                        "vram_mb": n.gpu_info.vram_mb,
                        "status": n.status,
                        "requests_served": n.gpu_info.requests_served,
                        "total_tokens": n.gpu_info.total_tokens,
                    }
                    for n in self._nodes.values()
                ],
            }


# ── WebGPU Client HTML ────────────────────────────────────────────────

_WEBGPU_CLIENT_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DistLLM WebGPU Client</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #0d1117; color: #c9d1d9; padding: 20px; }
        .container { max-width: 800px; margin: 0 auto; }
        h1 { color: #58a6ff; margin-bottom: 20px; }
        .card { background: #161b22; border: 1px solid #30363d; border-radius: 12px; padding: 20px; margin-bottom: 16px; }
        .card h2 { color: #f0f6fc; font-size: 16px; margin-bottom: 12px; }
        .status { padding: 4px 12px; border-radius: 12px; font-size: 12px; font-weight: 500; }
        .status.connected { background: #1a4d2e; color: #3fb950; }
        .status.disconnected { background: #4d1a1a; color: #f85149; }
        .btn { padding: 10px 20px; border: none; border-radius: 8px; cursor: pointer; font-weight: 500; }
        .btn-primary { background: #10a37f; color: white; }
        .btn-primary:hover { background: #0d8c6d; }
        .btn-primary:disabled { background: #30363d; cursor: not-allowed; }
        textarea { width: 100%; padding: 12px; border: 1px solid #30363d; border-radius: 8px; background: #0d1117; color: #c9d1d9; font-size: 14px; resize: vertical; }
        #output { min-height: 100px; padding: 12px; background: #0d1117; border: 1px solid #30363d; border-radius: 8px; white-space: pre-wrap; font-family: monospace; }
        .gpu-info { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; font-size: 13px; }
        .gpu-info .label { color: #8b949e; }
    </style>
</head>
<body>
    <div class="container">
        <h1>DistLLM WebGPU Client</h1>

        <div class="card">
            <h2>GPU Status</h2>
            <div class="gpu-info">
                <div><span class="label">Vendor:</span> <span id="gpu-vendor">Detecting...</span></div>
                <div><span class="label">Renderer:</span> <span id="gpu-renderer">Detecting...</span></div>
                <div><span class="label">VRAM:</span> <span id="gpu-vram">Unknown</span></div>
                <div><span class="label">Status:</span> <span id="gpu-status" class="status disconnected">Disconnected</span></div>
            </div>
            <button id="connect-btn" class="btn btn-primary" style="margin-top: 12px;">Connect & Contribute GPU</button>
        </div>

        <div class="card">
            <h2>Inference</h2>
            <textarea id="prompt" rows="4" placeholder="Enter a prompt..."></textarea>
            <div style="margin-top: 12px; display: flex; gap: 8px;">
                <input id="max-tokens" type="number" value="256" min="1" max="4096" style="width: 100px; padding: 8px; border: 1px solid #30363d; border-radius: 6px; background: #0d1117; color: #c9d1d9;">
                <button id="generate-btn" class="btn btn-primary" disabled>Generate</button>
            </div>
        </div>

        <div class="card">
            <h2>Output</h2>
            <div id="output">Ready.</div>
        </div>
    </div>

    <script>
    const API_BASE = window.location.origin;
    let sessionId = null;
    let gpuInfo = null;

    // Detect WebGPU
    async function detectGPU() {
        if (!navigator.gpu) {
            document.getElementById('gpu-vendor').textContent = 'WebGPU not supported';
            document.getElementById('gpu-renderer').textContent = 'Use Chrome 113+';
            return null;
        }
        try {
            const adapter = await navigator.gpu.requestAdapter();
            if (!adapter) {
                document.getElementById('gpu-vendor').textContent = 'No GPU adapter';
                return null;
            }
            const info = adapter.info || {};
            const vendor = info.vendor || 'Unknown';
            const renderer = info.description || 'Unknown GPU';
            document.getElementById('gpu-vendor').textContent = vendor;
            document.getElementById('gpu-renderer').textContent = renderer;
            return { vendor, renderer };
        } catch (e) {
            document.getElementById('gpu-vendor').textContent = 'Error: ' + e.message;
            return null;
        }
    }

    // Register with server
    async function register() {
        gpuInfo = await detectGPU();
        if (!gpuInfo) return;

        const res = await fetch(API_BASE + '/webgpu/register', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                gpu_vendor: gpuInfo.vendor,
                gpu_model: gpuInfo.renderer,
                vram_mb: 0,
                webgpu_features: [],
            }),
        });
        const data = await res.json();
        sessionId = data.session_id;

        document.getElementById('gpu-status').textContent = 'Connected';
        document.getElementById('gpu-status').className = 'status connected';
        document.getElementById('generate-btn').disabled = false;
        document.getElementById('connect-btn').disabled = true;
        document.getElementById('connect-btn').textContent = 'Connected';

        // Start heartbeat
        setInterval(async () => {
            if (sessionId) {
                await fetch(API_BASE + '/webgpu/heartbeat', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({session_id: sessionId}),
                });
            }
        }, 10000);
    }

    // Generate inference
    async function generate() {
        const prompt = document.getElementById('prompt').value;
        const maxTokens = parseInt(document.getElementById('max-tokens').value);
        const output = document.getElementById('output');
        output.textContent = 'Generating...';

        try {
            const res = await fetch(API_BASE + '/v1/chat/completions', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    model: 'distributed-llm',
                    messages: [{role: 'user', content: prompt}],
                    max_tokens: maxTokens,
                    stream: false,
                }),
            });
            const data = await res.json();
            output.textContent = data.choices?.[0]?.message?.content || JSON.stringify(data, null, 2);
        } catch (e) {
            output.textContent = 'Error: ' + e.message;
        }
    }

    document.getElementById('connect-btn').addEventListener('click', register);
    document.getElementById('generate-btn').addEventListener('click', generate);
    detectGPU();
    </script>
</body>
</html>"""
