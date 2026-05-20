"""Web dashboard for DistLLM monitoring and management.

Features:
- System overview with real-time metrics
- Node health monitoring
- WebSocket for live updates (v2 dashboard)
- Interactive API for configuration
"""

import argparse
import asyncio
import json
import os
import time
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
from loguru import logger

from distllm.dashboard.ws_handler import manager, metrics_broadcaster, parse_client_message

dashboard_app = FastAPI(title="DistLLM Dashboard", version="0.4.0")

# Global coordinator reference (set by main())
coordinator: object | None = None
monitor: object | None = None
_broadcast_task = None


@dashboard_app.on_event("startup")
async def startup_event():
    """Start the metrics broadcaster background task."""
    global coordinator, _broadcast_task
    if coordinator is not None:
        _broadcast_task = asyncio.create_task(metrics_broadcaster(coordinator))


@dashboard_app.on_event("shutdown")
async def shutdown_event():
    """Cancel the metrics broadcaster."""
    global _broadcast_task
    if _broadcast_task:
        _broadcast_task.cancel()


# --- HTML Pages ---

@dashboard_app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Serve the v2 dashboard HTML."""
    html_path = Path(__file__).parent / "static_v2" / "index.html"
    if html_path.exists():
        return html_path.read_text()
    # Fallback to v1
    html_path = Path(__file__).parent / "static" / "index.html"
    return html_path.read_text()


# --- WebSocket ---

@dashboard_app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time metric streaming.

    Client may send JSON commands:
      - ``{"type":"subscribe","metrics":["latency","gpu"],"interval":2.0}``
      - ``{"type":"ping"}``
    """
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
                await manager.send_to(websocket, {
                    "type": "subscribed",
                    "metrics": cmd.get("metrics"),
                    "interval": cmd.get("interval", 1.0),
                })
            elif cmd_type == "ping":
                await manager.send_to(websocket, {"type": "pong", "timestamp": time.time()})
            elif cmd_type == "error":
                await manager.send_to(websocket, {"type": "error", "detail": cmd.get("detail", "Unknown error")})
    except WebSocketDisconnect:
        manager.disconnect(websocket)


# --- REST API ---

@dashboard_app.get("/api/status")
async def api_status():
    """Return current system status."""
    global coordinator
    if coordinator is None:
        return {"error": "No coordinator connected"}

    result = {
        "model": getattr(coordinator, "model_name", "unknown"),
        "nodes": len(getattr(coordinator, "nodes", {})),
        "uptime": time.time(),
        "ws_connections": manager.connection_count,
    }

    try:
        result["metrics"] = coordinator.get_metrics()
    except (AttributeError, RuntimeError) as e:
        logger.debug(f"Metrics unavailable: {e}")
        result["metrics"] = {}

    try:
        if coordinator.scheduler:
            result["scheduler"] = coordinator.scheduler.stats()
    except (AttributeError, RuntimeError) as e:
        logger.debug(f"Scheduler stats unavailable: {e}")

    try:
        if coordinator.prefix_cache:
            result["prefix_cache"] = coordinator.prefix_cache.stats()
    except (AttributeError, RuntimeError) as e:
        logger.debug(f"Prefix cache stats unavailable: {e}")

    return result


@dashboard_app.get("/api/nodes")
async def api_nodes():
    """Return node health information."""
    global coordinator
    if coordinator is None:
        return []

    nodes = getattr(coordinator, "nodes", {})
    result = []
    for node_id, reg in nodes.items():
        result.append({
            "node_id": node_id,
            "host": reg.host,
            "port": reg.port,
            "healthy": reg.healthy,
            "layers": f"{reg.start_layer}-{reg.end_layer}",
            "role": getattr(reg, "role", "auto"),
        })
    return result


@dashboard_app.get("/api/metrics")
async def api_metrics():
    """Return detailed metrics."""
    global coordinator
    if coordinator is None:
        return {}
    return coordinator.get_metrics()


@dashboard_app.get("/api/requests/waterfall")
async def api_waterfall(limit: int = Query(50, ge=1, le=200)):
    """Return recent request waterfall data showing request lifecycle phases."""
    global coordinator
    if coordinator is None:
        return []

    try:
        scheduler = getattr(coordinator, "scheduler", None)
        if scheduler is None:
            return []

        tracker = getattr(scheduler, "latency_tracker", None)
        if tracker is None:
            return []

        return tracker.get_recent_metrics(limit=limit)
    except (AttributeError, RuntimeError) as e:
        logger.debug(f"Waterfall data unavailable: {e}")
        return []


@dashboard_app.post("/api/config")
async def api_update_config(config: dict):
    """Update runtime configuration (requires coordinator).

    Supported keys: batch_size, max_tokens, temperature
    """
    global coordinator
    if coordinator is None:
        raise HTTPException(status_code=503, detail="No coordinator connected")

    updated = {}
    try:
        if "batch_size" in config and coordinator.scheduler:
            coordinator.scheduler.max_batch_size = int(config["batch_size"])
            updated["batch_size"] = config["batch_size"]
    except (ValueError, TypeError) as e:
        logger.debug(f"Config update error: {e}")

    return {"status": "ok", "updated": updated}


def main():
    """CLI entry point for the dashboard."""
    parser = argparse.ArgumentParser(description="DistLLM Web Dashboard")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8501)
    parser.add_argument("--api-url", type=str, default=None,
                        help="URL of the DistLLM API server (default: localhost:8000)")

    args = parser.parse_args()

    logger.info(f"Starting dashboard on {args.host}:{args.port}")
    uvicorn.run(dashboard_app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
