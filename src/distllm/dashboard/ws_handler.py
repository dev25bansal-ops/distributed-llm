"""WebSocket and SSE handler for real-time dashboard metrics.

Broadcasts node health, GPU utilization, KV cache hit rates,
latency histograms, speculative acceptance rates, and cost per request.
Supports per-client metric subscriptions and SSE streaming.
"""

import asyncio
import json
import time
from collections import defaultdict, deque
from typing import Any, AsyncGenerator

from fastapi import WebSocket
from fastapi.responses import StreamingResponse
from loguru import logger


# ---------------------------------------------------------------------------
# Connection manager with per-client subscriptions
# ---------------------------------------------------------------------------

class ConnectionManager:
    """Manages WebSocket connections with per-client metric subscriptions."""

    def __init__(self):
        self.active_connections: set[WebSocket] = set()
        self._subscriptions: dict[WebSocket, dict[str, Any]] = {}
        self._last_sent: dict[WebSocket, float] = {}

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)
        now = time.time()
        self._subscriptions[websocket] = {
            "metrics": None,  # None = all metrics
            "interval": 1.0,
        }
        self._last_sent[websocket] = now
        logger.info(f"WebSocket connected. Total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.discard(websocket)
        self._subscriptions.pop(websocket, None)
        self._last_sent.pop(websocket, None)
        logger.info(f"WebSocket disconnected. Total: {len(self.active_connections)}")

    def subscribe(self, websocket: WebSocket, metric_types: list[str] | None = None, interval: float = 1.0):
        """Set subscription filters for a client. None = all metrics."""
        sub = self._subscriptions.get(websocket)
        if sub is not None:
            sub["metrics"] = set(metric_types) if metric_types else None
            sub["interval"] = max(0.2, min(interval, 10.0))

    def get_interval(self, websocket: WebSocket) -> float:
        sub = self._subscriptions.get(websocket)
        return sub["interval"] if sub else 1.0

    def wants_metric(self, websocket: WebSocket, metric_name: str) -> bool:
        """Check whether a client wants a particular metric category."""
        sub = self._subscriptions.get(websocket)
        if sub is None or sub["metrics"] is None:
            return True
        return metric_name in sub["metrics"]

    async def broadcast_filtered(self, message: dict, metric_category: str = ""):
        """Send message only to clients that subscribe to *metric_category*."""
        data = json.dumps(message)
        disconnected = set()
        for conn in self.active_connections:
            if metric_category and not self.wants_metric(conn, metric_category):
                continue
            try:
                await conn.send_text(data)
            except Exception:
                disconnected.add(conn)
        self.active_connections -= disconnected

    async def broadcast(self, message: dict):
        """Send message to every connected client."""
        await self.broadcast_filtered(message, metric_category="")

    async def send_to(self, websocket: WebSocket, message: dict):
        """Send a message to a single client."""
        try:
            await websocket.send_text(json.dumps(message))
        except Exception:
            self.disconnect(websocket)

    def clients_due(self, now: float) -> list[WebSocket]:
        """Return clients whose subscription interval has elapsed since last send."""
        due = []
        for conn in list(self.active_connections):
            sub = self._subscriptions.get(conn)
            if sub is None:
                continue
            interval = sub.get("interval", 1.0)
            last = self._last_sent.get(conn, 0)
            if now - last >= interval:
                due.append(conn)
        return due

    def mark_sent(self, websocket: WebSocket, now: float):
        self._last_sent[websocket] = now

    @property
    def connection_count(self) -> int:
        return len(self.active_connections)


manager = ConnectionManager()


# ---------------------------------------------------------------------------
# In-memory sliding-window metrics collector
# ---------------------------------------------------------------------------

class MetricsCollector:
    """Collects and tracks real-time metrics for dashboard display."""

    _instance: "MetricsCollector | None" = None

    def __init__(self, max_history: int = 300):
        self.max_history = max_history
        self._latencies: deque[float] = deque(maxlen=max_history)
        self._ttft: deque[float] = deque(maxlen=max_history)
        self._tokens_per_sec: deque[float] = deque(maxlen=max_history)
        self._kv_hits: int = 0
        self._kv_misses: int = 0
        self._spec_drafts: int = 0
        self._spec_accepted: int = 0
        self._costs: deque[float] = deque(maxlen=max_history)
        self._gpu_util: dict[str, deque] = defaultdict(lambda: deque(maxlen=120))
        self._requests_by_model: dict[str, int] = defaultdict(int)
        self._requests_by_endpoint: dict[str, int] = defaultdict(int)
        self._request_queue_depth: deque[int] = deque(maxlen=120)
        self._active_requests: deque[int] = deque(maxlen=120)
        self._last_collected: float = time.time()
        self._pipeline_topology: dict = {}

    def record_request(self, latency_ms: float, ttft_ms: float = 0,
                       tokens_per_sec: float = 0, model: str = "",
                       endpoint: str = "", cost: float = 0):
        self._latencies.append(latency_ms)
        self._ttft.append(ttft_ms)
        self._tokens_per_sec.append(tokens_per_sec)
        self._costs.append(cost)
        if model:
            self._requests_by_model[model] += 1
        if endpoint:
            self._requests_by_endpoint[endpoint] += 1

    def record_kv_cache(self, hit: bool):
        if hit:
            self._kv_hits += 1
        else:
            self._kv_misses += 1

    def record_speculative(self, draft_count: int, accepted_count: int):
        self._spec_drafts += draft_count
        self._spec_accepted += accepted_count

    def record_gpu_util(self, node_id: str, util_pct: float):
        self._gpu_util[node_id].append(util_pct)

    def record_queue_depth(self, depth: int):
        self._request_queue_depth.append(depth)

    def record_active_requests(self, count: int):
        self._active_requests.append(count)

    def update_pipeline_topology(self, topology: dict):
        self._pipeline_topology = topology

    def kv_hit_rate(self) -> float:
        total = self._kv_hits + self._kv_misses
        return self._kv_hits / max(total, 1)

    def spec_acceptance_rate(self) -> float:
        return self._spec_accepted / max(self._spec_drafts, 1)

    def latency_histogram(self, buckets: list[float] | None = None) -> dict:
        if buckets is None:
            buckets = [10, 25, 50, 100, 200, 500, 1000, 2000, 5000]
        if not self._latencies:
            return {str(b): 0 for b in buckets}
        hist = {str(b): 0 for b in buckets}
        for lat in self._latencies:
            for b in buckets:
                if lat <= b:
                    hist[str(b)] += 1
                    break
            else:
                hist[str(buckets[-1])] += 1
        return hist

    def summary(self) -> dict:
        latencies = list(self._latencies)
        ttft = list(self._ttft)
        tokens = list(self._tokens_per_sec)
        costs = list(self._costs)
        queue = list(self._request_queue_depth)
        active = list(self._active_requests)

        def pct(data, p):
            if not data:
                return 0
            s = sorted(data)
            return s[int(len(s) * p / 100)]

        return {
            "latency": {
                "p50": pct(latencies, 50),
                "p95": pct(latencies, 95),
                "p99": pct(latencies, 99),
                "avg": sum(latencies) / max(len(latencies), 1),
                "histogram": self.latency_histogram(),
            },
            "ttft": {
                "p50": pct(ttft, 50),
                "p95": pct(ttft, 95),
                "avg": sum(ttft) / max(len(ttft), 1),
            },
            "throughput": {
                "tokens_per_sec_avg": sum(tokens) / max(len(tokens), 1),
            },
            "kv_cache": {
                "hit_rate": self.kv_hit_rate(),
                "hits": self._kv_hits,
                "misses": self._kv_misses,
            },
            "speculative": {
                "acceptance_rate": self.spec_acceptance_rate(),
                "drafts": self._spec_drafts,
                "accepted": self._spec_accepted,
            },
            "cost": {
                "total": sum(costs),
                "avg_per_request": sum(costs) / max(len(costs), 1),
            },
            "queue_depth": {
                "current": queue[-1] if queue else 0,
                "avg": sum(queue) / max(len(queue), 1) if queue else 0,
                "max": max(queue) if queue else 0,
            },
            "active_requests": {
                "current": active[-1] if active else 0,
                "avg": sum(active) / max(len(active), 1) if active else 0,
            },
            "requests_by_model": dict(self._requests_by_model),
            "requests_by_endpoint": dict(self._requests_by_endpoint),
        }


_collector = MetricsCollector()


def get_collector() -> MetricsCollector:
    return _collector


# ---------------------------------------------------------------------------
# Structured metrics snapshot — pulled from coordinator once per tick
# ---------------------------------------------------------------------------

async def collect_metrics_snapshot(coordinator) -> dict:
    """Collect a complete structured snapshot of all metrics from coordinator.

    This is the single source of truth for both WebSocket and SSE streams.
    """
    now = time.time()
    collector = get_collector()
    data: dict[str, Any] = {
        "timestamp": now,
        "model": getattr(coordinator, "model_name", "unknown"),
        "nodes": len(getattr(coordinator, "nodes", {})),
        "uptime": now - getattr(coordinator, "_startup_time", now),
        "ws_connections": manager.connection_count,
    }

    # Collector summary (latency percentiles, TTFT, throughput, KV, cost)
    data["metrics_summary"] = collector.summary()

    # Scheduler stats (active_requests, pending_requests, queue depth)
    try:
        if coordinator.scheduler:
            sched = coordinator.scheduler.stats()
            data["scheduler"] = sched
            collector.record_queue_depth(sched.get("pending_requests", 0))
            collector.record_active_requests(sched.get("active_requests", 0))
    except Exception as e:
        logger.warning(f"Failed to collect scheduler stats: {e}")

    # Node health with GPU utilization
    try:
        nodes = {}
        for node_id, reg in getattr(coordinator, "nodes", {}).items():
            gpu_util = getattr(reg, "gpu_utilization", 0)
            if callable(gpu_util):
                try:
                    gpu_util = gpu_util()
                except Exception:
                    gpu_util = 0
            collector.record_gpu_util(node_id, float(gpu_util))
            node_info = {
                "healthy": reg.healthy,
                "host": reg.host,
                "port": reg.port,
                "layers": f"{reg.start_layer}-{reg.end_layer}",
                "gpu_utilization": float(gpu_util),
                "role": getattr(reg, "role", "auto"),
            }
            kv = getattr(reg, "kv_cache_stats", None)
            if kv:
                node_info["kv_cache"] = kv
            nodes[node_id] = node_info
        data["nodes"] = nodes
        data["gpu_utilization"] = {
            nid: ninfo["gpu_utilization"]
            for nid, ninfo in nodes.items()
        }
    except Exception as e:
        logger.warning(f"Failed to collect node health data: {e}")

    # Prefix / KV cache stats
    try:
        if coordinator.prefix_cache:
            data["prefix_cache"] = (
                coordinator.prefix_cache.stats()
                if hasattr(coordinator.prefix_cache, "stats")
                else {}
            )
    except Exception as e:
        logger.warning(f"Failed to collect prefix cache stats: {e}")

    # Speculative decoder
    try:
        sd = getattr(coordinator, "_spec_decoder", None)
        if sd:
            data["spec_decoder"] = (
                sd.get_metrics()
                if hasattr(sd, "get_metrics")
                else {}
            )
    except Exception as e:
        logger.warning(f"Failed to collect speculative decoder metrics: {e}")

    # Pipeline topology
    try:
        topology = {
            "model": getattr(coordinator, "model_name", "unknown"),
            "nodes": len(getattr(coordinator, "nodes", {})),
            "pipeline_parallel": len(getattr(coordinator, "nodes", {})) > 1,
        }
        collector.update_pipeline_topology(topology)
        data["topology"] = topology
    except Exception as e:
        logger.warning(f"Failed to collect pipeline topology: {e}")

    # Tenant info
    try:
        from distllm.api.server import tenant_store
        if tenant_store:
            tenants = tenant_store.list_tenants()
            data["tenants"] = [
                {"tenant_id": t.tenant_id, "name": t.name, "tier": t.tier.value}
                for t in tenants
            ]
    except Exception as e:
        logger.warning(f"Failed to collect tenant info: {e}")

    # Request latency tracker percentiles (per-request granularity)
    try:
        from distllm.core.request_latency import RequestLatencyTracker
        ltracker = getattr(coordinator.scheduler, "_latency_tracker", None) if coordinator.scheduler else None
        if ltracker and hasattr(ltracker, "get_recent_metrics"):
            recent = ltracker.get_recent_metrics(limit=100)
            if recent:
                latencies = [r.get("elapsed_ms", 0) for r in recent if r.get("elapsed_ms")]
                ttfts = [r.get("ttft_ms", 0) for r in recent if r.get("ttft_ms")]
                if latencies:
                    sorted_lats = sorted(latencies)
                    n = len(sorted_lats)
                    data["per_request_latency"] = {
                        "p50": sorted_lats[int(n * 0.5)],
                        "p95": sorted_lats[int(n * 0.95)],
                        "p99": sorted_lats[int(n * 0.99)],
                        "avg": sum(sorted_lats) / n,
                        "count": n,
                    }
                if ttfts:
                    data["per_request_ttft"] = {
                        "avg": sum(ttfts) / len(ttfts),
                    }
    except Exception as e:
        logger.warning(f"Failed to collect request latency metrics: {e}")

    return data


# ---------------------------------------------------------------------------
# WebSocket broadcaster
# ---------------------------------------------------------------------------

async def metrics_broadcaster(coordinator, interval: float = 0.2):
    """Periodically broadcast rich metrics to all connected WebSocket clients.

    Respects per-client subscription intervals — only sends to clients
    whose configured interval has elapsed since their last update.
    The base tick *interval* controls the wake-up frequency (default 0.2s
    to support sub-second client intervals); each client receives data
    at most once per their own configured interval.
    """
    if coordinator is None:
        return

    while True:
        try:
            now = time.time()
            due = manager.clients_due(now)
            if due:
                snapshot = await collect_metrics_snapshot(coordinator)
                message = {"type": "metrics", "timestamp": snapshot["timestamp"], "data": snapshot}
                for conn in due:
                    try:
                        await conn.send_text(json.dumps(message))
                    except Exception:
                        manager.disconnect(conn)
                    else:
                        manager.mark_sent(conn, now)
        except Exception as e:
            logger.debug(f"Broadcast error: {e}")
        await asyncio.sleep(interval)


# ---------------------------------------------------------------------------
# SSE metrics stream (async generator)
# ---------------------------------------------------------------------------

KNOWN_METRIC_CATEGORIES = {
    "latency", "ttft", "throughput", "tokens_per_sec",
    "kv_cache", "speculative", "cost", "queue_depth",
    "active_requests", "scheduler", "nodes", "gpu",
    "prefix_cache", "spec_decoder", "topology", "tenants",
    "per_request_latency",
}

_CATEGORY_MAP = {
    "tokens_per_sec": "throughput",
    "gpu": "nodes",
}


async def stream_metrics_sse(
    coordinator,
    requested_metrics: set[str] | None = None,
    interval: float = 1.0,
) -> AsyncGenerator[str, None]:
    """Async generator yielding SSE ``event: metric`` lines.

    Yields complete JSON snapshots filtered to *requested_metrics* categories.
    When *requested_metrics* is *None* all metrics are included.
    """
    if coordinator is None:
        yield "event: error\ndata: {\"detail\":\"No coordinator available\"}\n\n"
        return

    # Send an initial heartbeat
    yield f"event: connected\ndata: {json.dumps({'status': 'streaming', 'interval': interval, 'categories': list(KNOWN_METRIC_CATEGORIES)})}\n\n"

    while True:
        try:
            snapshot = await collect_metrics_snapshot(coordinator)
            # Filter to requested categories
            if requested_metrics:
                filtered: dict[str, Any] = {"timestamp": snapshot["timestamp"]}
                for cat in requested_metrics:
                    mapped = _CATEGORY_MAP.get(cat, cat)
                    if mapped == "nodes":
                        filtered["nodes"] = snapshot.get("nodes", {})
                    elif mapped == "gpu":
                        filtered["nodes"] = snapshot.get("nodes", {})
                    elif mapped == "all":
                        filtered.update(snapshot)
                    else:
                        val = snapshot.get(mapped)
                        if val is not None:
                            filtered[mapped] = val
                payload = filtered
            else:
                payload = snapshot

            yield f"event: metric\ndata: {json.dumps(payload)}\n\n"
        except Exception as e:
            yield f"event: error\ndata: {json.dumps({'detail': str(e)})}\n\n"
            logger.debug(f"SSE stream error: {e}")

        await asyncio.sleep(interval)


# ---------------------------------------------------------------------------
# WebSocket client message parsing
# ---------------------------------------------------------------------------

def parse_client_message(text: str) -> dict:
    """Parse an incoming WebSocket text message from a dashboard client.

    Supported commands:

    - ``{"type":"subscribe","metrics":["latency","gpu"],"interval":2}``
    - ``{"type":"ping"}``
    - ``{"type":"pong"}``
    """
    try:
        msg = json.loads(text)
    except json.JSONDecodeError:
        return {"type": "error", "detail": "invalid JSON"}

    msg_type = msg.get("type", "")

    if msg_type == "subscribe":
        raw = msg.get("metrics")
        if raw is not None and not isinstance(raw, list):
            return {"type": "error", "detail": "metrics must be a list of category names"}
        return {
            "type": "subscribe",
            "metrics": raw,
            "interval": float(msg.get("interval", 1.0)),
        }

    if msg_type in ("ping", "pong"):
        return {"type": msg_type}

    return {"type": "error", "detail": f"unknown command: {msg_type}"}
