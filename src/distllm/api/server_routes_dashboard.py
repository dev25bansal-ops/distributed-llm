"""Dashboard/UI WebSocket and page endpoints for the distributed LLM API."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from loguru import logger

from distllm.api.server_config import _extract_prom_counter, _extract_prom_gauge
from distllm.api.server_state import state
from distllm.dashboard.ws_handler import get_collector, manager, parse_client_message


def register_dashboard_routes(app: FastAPI) -> None:
    """Register all dashboard/UI WebSocket and page endpoints on *app*."""

    @app.websocket("/ws")
    async def dashboard_websocket(websocket: WebSocket) -> None:
        """WebSocket endpoint for real-time dashboard metrics.

        Client may send JSON commands:
          - ``{"type":"subscribe","metrics":["latency","gpu"],"interval":2.0}``
          - ``{"type":"ping"}``

        Supported metric categories: latency, ttft, throughput, tokens_per_sec,
        kv_cache, speculative, cost, queue_depth, active_requests, scheduler,
        nodes, gpu, prefix_cache, spec_decoder, topology, tenants.
        """
        # SECURITY FIX: Removed query param token support — tokens in URLs are logged
        # in server access logs, proxy logs, browser history, and analytics.
        # Token from Authorization header, or (browser WS clients) from a
        # Sec-WebSocket-Protocol subprotocol pair: ["Bearer", "<key>"].
        auth_token = None
        auth_header = websocket.headers.get("authorization", "")
        if auth_header.startswith("Bearer "):
            auth_token = auth_header[7:]
        else:
            sec_ws = websocket.headers.get("sec-websocket-protocol", "")
            parts = [p.strip() for p in sec_ws.split(",") if p.strip()]
            if len(parts) >= 2 and parts[0] == "Bearer":
                auth_token = parts[1]

        # Validate API key if auth is configured
        from distllm.core.api_key_store import get_api_key_store

        store = get_api_key_store()
        if store.get_key_count() > 0:
            if not auth_token:
                logger.warning("WebSocket connection rejected: missing API key")
                await websocket.close(code=4001, reason="API key required")
                return
            result = store.authenticate(auth_token)
            if result is None:
                await websocket.close(code=4001, reason="Invalid API key")
                return

        await manager.connect(websocket)
        try:
            while True:
                raw = await websocket.receive_text()
                cmd = parse_client_message(raw)
                cmd_type = cmd.get("type", "error")

                if cmd_type == "subscribe":
                    manager.subscribe(
                        websocket,
                        metric_types=cmd.get("metrics"),
                        interval=cmd.get("interval", 1.0),
                    )
                    await manager.send_to(
                        websocket,
                        {
                            "type": "subscribed",
                            "metrics": cmd.get("metrics"),
                            "interval": cmd.get("interval", 1.0),
                        },
                    )
                elif cmd_type == "ping":
                    await manager.send_to(websocket, {"type": "pong", "timestamp": time.time()})
                elif cmd_type == "error":
                    await manager.send_to(websocket, {"type": "error", "detail": cmd.get("detail", "Unknown error")})
        except WebSocketDisconnect:
            manager.disconnect(websocket)

    @app.websocket("/ws/metrics")
    async def metrics_websocket(websocket: WebSocket) -> None:
        """Dedicated WebSocket endpoint for live metrics streaming.

        Unlike /ws (which requires subscribe commands), this endpoint
        auto-streams all metrics at a configurable interval.

        SECURITY: Requires API key authentication (same as /ws endpoint).

        Query params:
            interval: Stream interval in seconds (default: 1.0)
            categories: Comma-separated metric categories to include
            token: API key for authentication
        """
        # SECURITY: Authenticate WebSocket connection via header only
        # Query param tokens are insecure (logged in URLs, proxies, browser history)
        from distllm.core.api_key_store import get_api_key_store

        store = get_api_key_store()
        auth_token = None
        auth_header = websocket.headers.get("authorization", "")
        if auth_header.startswith("Bearer "):
            auth_token = auth_header[7:]

        if store.get_key_count() > 0:
            if not auth_token:
                logger.warning("Metrics WebSocket rejected: missing API key")
                await websocket.close(code=4001, reason="API key required")
                return
            result = store.authenticate(auth_token)
            if result is None:
                await websocket.close(code=4001, reason="Invalid API key")
                return

        await websocket.accept()
        # Clamp interval to safe range (0.2s - 10.0s) to prevent DoS
        interval = max(0.2, min(float(websocket.query_params.get("interval", "1.0")), 10.0))
        categories = websocket.query_params.get("categories", "")
        requested = [c.strip() for c in categories.split(",") if c.strip()] or None

        collector = get_collector()
        try:
            while True:
                coord = state.coordinator
                if coord is None:
                    await websocket.send_json({"type": "status", "coordinator": "not_loaded"})
                    await asyncio.sleep(interval)
                    continue

                # Build metrics snapshot
                snapshot = {"type": "metrics", "timestamp": time.time()}

                # Coordinator metrics
                try:
                    coord_metrics = coord.get_metrics()
                    if not requested or "coordinator" in requested:
                        snapshot["coordinator"] = coord_metrics
                except Exception:
                    logger.exception("Failed to collect coordinator metrics")
                    if coord.scheduler:
                        sched_stats = coord.scheduler.stats()
                        if not requested or "scheduler" in requested:
                            snapshot["scheduler"] = sched_stats
                except Exception:
                    logger.debug("Failed to collect scheduler snapshot")

                # Prometheus metrics snapshot (if available) — cached with 200ms TTL
                # to avoid CPU-heavy generate_latest() on every tick.
                try:
                    from prometheus_client import generate_latest

                    now = time.time()
                    if (
                        not hasattr(state, "_prom_cache_ts")
                        or not hasattr(state, "_prom_cache")
                        or now - getattr(state, "_prom_cache_ts", 0) > 0.2
                    ):
                        prom_data = generate_latest()
                        state._prom_cache = prom_data
                        state._prom_cache_ts = now
                    else:
                        prom_data = state._prom_cache
                    if prom_data and (not requested or "prometheus" in requested):
                        snapshot["prometheus"] = {
                            "gpu_util": _extract_prom_gauge(prom_data, "distllm_gpu_utilization"),
                            "requests_active": _extract_prom_gauge(prom_data, "distllm_active_requests"),
                            "tokens_total": _extract_prom_counter(prom_data, "distllm_tokens_total"),
                        }
                except Exception:
                    logger.exception("Failed to collect Prometheus metrics")

                # Collector metrics
                if collector:
                    if not requested or "latency" in requested:
                        snapshot["latency"] = collector.summary()
                    if not requested or "kv_cache" in requested:
                        snapshot["kv_cache"] = {"hit_rate": collector.kv_hit_rate()}
                    if not requested or "speculative" in requested:
                        snapshot["speculative"] = {"acceptance_rate": collector.spec_acceptance_rate()}

                # GPU metrics
                try:
                    mon = state.monitor
                    if mon and (not requested or "gpu" in requested):
                        sys_metrics = mon.collect()
                        snapshot["gpu"] = sys_metrics.get("gpu", {})
                        snapshot["cpu"] = sys_metrics.get("cpu", {})
                except Exception:
                    logger.exception("Failed to collect GPU/system metrics")

                await websocket.send_json(snapshot)
                await asyncio.sleep(interval)
        except WebSocketDisconnect:
            pass

    @app.get(
        "/dashboard",
        response_class=HTMLResponse,
        summary="Dashboard page",
        description="Serve the real-time monitoring dashboard HTML page. The dashboard displays live metrics, request throughput, latency charts, and system health via WebSocket connection.",
        response_description="Dashboard HTML page",
        include_in_schema=False,
    )
    async def dashboard_page() -> HTMLResponse:
        """Serve the real-time dashboard HTML."""
        html_path = Path(__file__).parent.parent / "dashboard" / "static_v2" / "index.html"
        if html_path.exists():
            return HTMLResponse(content=html_path.read_text())
        return HTMLResponse(content="<h1>Dashboard not found</h1>")

    @app.get(
        "/dashboard/leaderboard",
        response_class=HTMLResponse,
        summary="Benchmark leaderboard page",
        description="Serve the benchmark leaderboard HTML page for comparing results across models, hardware, and frameworks.",
        include_in_schema=False,
    )
    async def dashboard_leaderboard_page() -> HTMLResponse:
        """Serve the benchmark leaderboard HTML."""
        html_path = Path(__file__).parent.parent / "dashboard" / "static_v2" / "leaderboard.html"
        if html_path.exists():
            return HTMLResponse(content=html_path.read_text())
        return HTMLResponse(content="<h1>Leaderboard not found</h1>")

    @app.get(
        "/dashboard/models",
        response_class=HTMLResponse,
        summary="Model Registry page",
        include_in_schema=False,
    )
    async def model_registry_page() -> HTMLResponse:
        """Serve the model registry dashboard HTML."""
        html_path = Path(__file__).parent.parent / "dashboard" / "static_v2" / "models.html"
        if html_path.exists():
            return HTMLResponse(content=html_path.read_text())
        return HTMLResponse(content="<h1>Model Registry not found</h1>")

