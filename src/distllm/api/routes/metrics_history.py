"""Metrics history, health trends, alert thresholds, and topology graph routes.

Provides four endpoint groups for the real-time monitoring dashboard:

1. ``GET /api/metrics/history`` — Time-series of node health metrics at 10s
   intervals over a configurable lookback window (default 1h).
2. ``GET /api/metrics/trends`` — Last 60 data points per node for sparkline
   charts (latency, GPU util, memory).
3. ``POST/GET /api/monitor/thresholds`` — Client-side alert thresholds.
4. ``GET /api/topology/graph`` — Interactive topology canvas with nodes and
   pipeline edges.

A background daemon thread polls the coordinator and system monitor every
10 seconds and stores the records in an in-memory ring buffer (last 360
entries = 1 hour at 10s).
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any

from fastapi import APIRouter, Depends, Query, Request
from pydantic import BaseModel, Field

from ..api_state import g
from distllm.api.auth_deps import require_role

router = APIRouter(tags=["monitoring"])

# ── Constants ─────────────────────────────────────────────────────────────

_MAX_ENTRIES = 360          # Ring buffer capacity (1 hour at 10s intervals)
_POLL_INTERVAL = 10.0       # Seconds between health data collection
_DEFAULT_WINDOW = 3600      # Default history lookback (seconds)
_TREND_POINTS = 60          # Number of data points per sparkline trend

# ── Pydantic Models ───────────────────────────────────────────────────────


class HealthRecord(BaseModel):
    """A single health data point for one node at one timestamp."""

    timestamp: float = Field(description="Unix timestamp of the measurement")
    node_id: str = Field(description="Unique node identifier")
    gpu_util: float = Field(default=0.0, description="GPU utilization percentage")
    memory_used: float = Field(default=0.0, description="GPU memory used in MB")
    request_count: int = Field(default=0, description="Active + pending requests")
    avg_latency: float = Field(default=0.0, description="Average request latency in ms")


class ThresholdSet(BaseModel):
    """Alert threshold values that the dashboard client checks against."""

    gpu_util_min: float | None = Field(
        default=None, ge=0, le=100,
        description="Minimum GPU utilization before alert (percent)",
    )
    memory_max: float | None = Field(
        default=None, ge=0,
        description="Maximum GPU memory before alert (MB)",
    )
    latency_max: float | None = Field(
        default=None, ge=0,
        description="Maximum request latency before alert (ms)",
    )
    error_rate_max: float | None = Field(
        default=None, ge=0, le=100,
        description="Maximum error rate before alert (percent)",
    )


# ── In-Memory Ring Buffer ────────────────────────────────────────────────


class HealthRingBuffer:
    """Thread-safe ring buffer for node health time-series data.

    Stores the most recent *maxlen* records.  Old records are evicted
    automatically when the capacity is reached.
    """

    def __init__(self, maxlen: int = _MAX_ENTRIES) -> None:
        self._maxlen = maxlen
        self._buffer: deque[HealthRecord] = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def append(self, record: HealthRecord) -> None:
        """Append a single health record (thread-safe)."""
        with self._lock:
            self._buffer.append(record)

    def get_window(self, window: float = _DEFAULT_WINDOW) -> list[dict[str, Any]]:
        """Return all records whose timestamp is within *window* seconds of now."""
        cutoff = time.time() - window
        with self._lock:
            return [
                r.model_dump()
                for r in self._buffer
                if r.timestamp >= cutoff
            ]

    def get_trends(
        self,
        node_ids: list[str] | None = None,
        count: int = _TREND_POINTS,
    ) -> dict[str, dict[str, list[float]]]:
        """Return the last *count* data points per metric per requested node.

        Returns a dict keyed by ``node_id``, each containing ``latency``,
        ``gpu``, and ``memory`` lists suitable for sparkline rendering.
        """
        with self._lock:
            data = list(self._buffer)

        by_node: dict[str, dict[str, list[float]]] = {}
        for rec in data:
            if node_ids is not None and rec.node_id not in node_ids:
                continue
            bucket = by_node.setdefault(rec.node_id, {
                "latency": [],
                "gpu": [],
                "memory": [],
            })
            bucket["latency"].append(rec.avg_latency)
            bucket["gpu"].append(rec.gpu_util)
            bucket["memory"].append(rec.memory_used)

        # Trim to the last *count* entries per node
        for nid in by_node:
            for key in ("latency", "gpu", "memory"):
                by_node[nid][key] = by_node[nid][key][-count:]

        return by_node


# Module-level singleton ring buffer shared across all handlers.
_ring = HealthRingBuffer()


# ── Alert Thresholds Store ────────────────────────────────────────────────

_default_thresholds: dict[str, float] = {
    "gpu_util_min": 10.0,
    "memory_max": 80_000.0,         # 80 GB
    "latency_max": 5_000.0,         # 5 seconds
    "error_rate_max": 5.0,          # 5 %
}

_thresholds_lock = threading.Lock()
_current_thresholds: dict[str, float] = dict(_default_thresholds)


# ── Background Data Collector ─────────────────────────────────────────────

_collector_running = threading.Event()
_collector_running.set()


def _collect_and_store() -> None:
    """Background daemon: poll coordinator and monitor every 10s.

    Reads ``g.coordinator`` and ``g.monitor`` from the shared application
    state (see ``api_state.py``).  Gracefully handles the case where the
    coordinator has not been initialised yet.
    """
    while _collector_running.is_set():
        try:
            coord = g.coordinator
            if coord is None:
                time.sleep(_POLL_INTERVAL)
                continue

            now = time.time()

            # ── system-level GPU metrics ──
            gpu_util = 0.0
            memory_used = 0.0
            mon = g.monitor
            if mon is not None:
                try:
                    sys_m = mon.collect()
                    gpu_info = sys_m.get("gpu", {})
                    gpu_util = float(gpu_info.get("utilization_gpu", 0))
                    memory_used = float(gpu_info.get("memory_used_mb", 0))
                except Exception:
                    pass

            # ── scheduler request metrics ──
            req_count = 0
            if coord.scheduler is not None:
                try:
                    stats = coord.scheduler.stats()
                    req_count = int(stats.get("active_requests", 0)) + int(
                        stats.get("pending_requests", 0)
                    )
                except Exception:
                    pass

            # ── average latency from pipeline tracker ──
            avg_latency = 0.0
            pipeline = getattr(coord, "_pipeline", None)
            if pipeline is not None:
                lt = getattr(pipeline, "_latency_tracker", None)
                if lt is not None and hasattr(lt, "get_avg"):
                    try:
                        avg_latency = float(lt.get_avg("all") or 0)
                    except Exception:
                        pass

            # ── per-node health data ──
            node_health = coord.health_check()
            nodes_data = node_health.get("nodes", {})

            if not nodes_data:
                # Fallback: single coordinator-level entry
                _ring.append(HealthRecord(
                    timestamp=now,
                    node_id="coordinator",
                    gpu_util=gpu_util,
                    memory_used=memory_used,
                    request_count=req_count,
                    avg_latency=avg_latency,
                ))
            else:
                for node_id, info in nodes_data.items():
                    _ring.append(HealthRecord(
                        timestamp=now,
                        node_id=str(node_id),
                        gpu_util=float(info.get("gpu_util", gpu_util)),
                        memory_used=float(info.get("memory_used", memory_used)),
                        request_count=int(info.get("request_count", req_count)),
                        avg_latency=float(info.get("latency_ms", avg_latency)),
                    ))
        except Exception:
            # Swallow so the collector never crashes.
            pass

        time.sleep(_POLL_INTERVAL)


# Start the collector as a daemon thread so it dies with the process.
_collector_thread = threading.Thread(target=_collect_and_store, daemon=True)
_collector_thread.start()


# ── Endpoints ─────────────────────────────────────────────────────────────


@router.get(
    "/api/metrics/history",
    summary="Node health history",
    description="Return time-series of node health metrics at 10s intervals "
                "within the given lookback window (default 1 hour).  Data is "
                "served from an in-memory ring buffer (last 360 entries).",
    response_description="List of health record dicts with timestamp, node_id, "
                         "gpu_util, memory_used, request_count, avg_latency",
    dependencies=[Depends(require_role("auditor"))],
)
async def get_health_history(
    request: Request,
    window: float = Query(
        default=_DEFAULT_WINDOW,
        ge=60,
        le=86400,
        description="Lookback window in seconds (60-86400)",
    ),
) -> list[dict[str, Any]]:
    """Return the health history ring-buffer contents filtered by *window*."""
    return _ring.get_window(window=window)


@router.get(
    "/api/metrics/trends",
    summary="Health trend sparklines",
    description="Return the last 60 data points per node for latency, GPU "
                "utilization, and memory — intended for dashboard mini "
                "sparkline charts (10-minute window at 10s intervals).",
    response_description="Per-node dict keyed by node_id, each containing "
                         "latency, gpu, and memory arrays",
    dependencies=[Depends(require_role("auditor"))],
)
async def get_health_trends(
    request: Request,
    nodes: str = Query(
        default="all",
        description="Comma-separated node IDs, or 'all' for every node",
    ),
) -> dict[str, dict[str, list[float]]]:
    """Return sparkline trend data for the requested nodes."""
    node_list: list[str] | None = None
    if nodes != "all":
        node_list = [n.strip() for n in nodes.split(",") if n.strip()]
    return _ring.get_trends(node_ids=node_list)


@router.post(
    "/api/monitor/thresholds",
    summary="Set alert thresholds",
    description="Set client-side alert thresholds for GPU utilization, "
                "GPU memory, request latency, and error rate.  Only "
                "included fields are updated; omitted fields keep their "
                "current values.",
    response_description="The full current thresholds after the update",
    dependencies=[Depends(require_role("admin"))],
)
async def set_thresholds(
    request: Request,
    thresholds: ThresholdSet,
) -> dict[str, float]:
    """Update one or more alert thresholds."""
    updates = thresholds.model_dump(exclude_none=True)
    with _thresholds_lock:
        _current_thresholds.update(updates)
        return dict(_current_thresholds)


@router.get(
    "/api/monitor/thresholds",
    summary="Get alert thresholds",
    description="Return the current alert thresholds for GPU utilization, "
                "GPU memory, request latency, and error rate.",
    response_description="Current thresholds dict",
    dependencies=[Depends(require_role("auditor"))],
)
async def get_thresholds(
    request: Request,
) -> dict[str, float]:
    """Return the current alert thresholds."""
    with _thresholds_lock:
        return dict(_current_thresholds)


@router.get(
    "/api/topology/graph",
    summary="Interactive topology graph",
    description="Return the cluster node topology as a graph structure "
                "(nodes + edges) suitable for rendering an interactive "
                "canvas in the dashboard.  Nodes include GPU info and "
                "health status; edges connect sequential pipeline stages.",
    response_description="Topology graph with nodes and edges arrays",
    dependencies=[Depends(require_role("auditor"))],
)
async def get_topology_graph(
    request: Request,
) -> dict[str, list[dict[str, Any]]]:
    """Build a topology graph from the coordinator's registered nodes.

    Nodes are sorted by their pipeline layer assignment.  Edges connect
    consecutive nodes in the pipeline order.
    """
    coord = g.coordinator
    if coord is None or not coord.nodes:
        return {"nodes": [], "edges": []}

    nodes: list[dict[str, Any]] = []
    layer_map: dict[str, int] = {}

    for node_id, node in coord.nodes.items():
        if isinstance(node, dict):
            label = node.get("host", str(node_id))
            gpu_name = node.get("gpu_name", "")
            status = "healthy" if node.get("healthy", False) else "unhealthy"
            start_layer = node.get("start_layer", 0)
        else:
            label = getattr(node, "host", str(node_id))
            gpu_name = getattr(node, "gpu_name", "")
            status = "healthy" if getattr(node, "healthy", False) else "unhealthy"
            start_layer = getattr(node, "start_layer", 0)

        layer_map[node_id] = start_layer
        nodes.append({
            "id": str(node_id),
            "label": label,
            "gpu": gpu_name,
            "status": status,
        })

    # Sort by pipeline layer order
    sorted_ids = sorted(coord.nodes.keys(), key=lambda nid: layer_map.get(nid, 0))

    # Build edges between consecutive pipeline stages
    edges: list[dict[str, Any]] = []
    for i in range(len(sorted_ids) - 1):
        edges.append({
            "source": str(sorted_ids[i]),
            "target": str(sorted_ids[i + 1]),
            "latency": 0.0,
            "bandwidth": 0.0,
        })

    return {"nodes": nodes, "edges": edges}


# ── Registration Instructions ─────────────────────────────────────────────
#
# To register this router in ``server.py``:
#
# 1. Add the import to the existing block near the top of the file::
#
#        from distllm.api.routes import (
#            ...
#            metrics_history_router,   # <-- add this line
#        )
#
# 2. Add ``include_router`` after the existing ones (around line 907)::
#
#        app.include_router(metrics_history_router)
#
# 3. (Optional) If the background collector should be stopped cleanly,
#    cancel the event in the shutdown handler::
#
#        _collector_running.clear()
