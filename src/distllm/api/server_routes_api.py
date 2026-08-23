"""Ad-hoc operational API endpoints extracted from server.py.

Contains all ``@app.get`` and ``@app.post`` endpoints that are NOT part
of the ``routes/`` package.  Imported by ``server.py`` via
``register_api_routes(app)``.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import time

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from loguru import logger
from pydantic import BaseModel, Field, field_validator

from distllm.api.auth_deps import require_role
from distllm.api.server_state import state
from distllm.dashboard.ws_handler import (
    KNOWN_METRIC_CATEGORIES,
    get_collector,
    stream_metrics_sse,
)


def register_api_routes(app: FastAPI) -> None:
    """Register all ad-hoc operational API endpoints on *app*."""

    # ── Cluster ────────────────────────────────────────────────────────

    @app.get(
        "/api/cluster/nodes",
        summary="Cluster node topology",
        description="Return all registered worker nodes with their GPU info, health status, and layer assignments.",
        response_description="List of cluster nodes with capabilities",
        dependencies=[Depends(require_role("auditor"))],
    )
    async def api_cluster_nodes(request: Request) -> dict:
        """Return current cluster node topology."""
        coord = state.coordinator
        if coord is None:
            return {"nodes": []}
        nodes_list = []
        for node_id, node in coord.nodes.items():
            if isinstance(node, dict):
                nodes_list.append({
                    "node_id": node_id,
                    "host": node.get("host", ""),
                    "port": node.get("port", 0),
                    "healthy": node.get("healthy", False),
                    "start_layer": node.get("start_layer", 0),
                    "end_layer": node.get("end_layer", 0),
                    "gpu_name": node.get("gpu_name", ""),
                })
            else:
                nodes_list.append({
                    "node_id": node_id,
                    "host": getattr(node, "host", ""),
                    "port": getattr(node, "port", 0),
                    "healthy": getattr(node, 'healthy', False),
                    "start_layer": getattr(node, 'start_layer', 0),
                    "end_layer": getattr(node, 'end_layer', 0),
                    "gpu_name": getattr(node, 'gpu_name', ''),
                    "gpu_memory_free": getattr(node, 'gpu_memory_free', 0),
                    "gpu_memory_total": getattr(node, 'gpu_memory_total', 0),
                    "gpu_sm_count": getattr(node, 'gpu_sm_count', 0),
                })
        return {"nodes": nodes_list, "total_layers": coord.total_layers}

    # ── Federation ─────────────────────────────────────────────────────

    @app.post(
        "/v1/federation/heartbeat",
        summary="Federation heartbeat",
        description="Receive heartbeat from a federated peer coordinator with load metrics. "
                    "Authenticated via cluster key in X-Cluster-Key header.",
        include_in_schema=False,
    )
    async def federation_heartbeat(request: Request) -> dict:
        """Receive and store heartbeat from a federated peer.

        Requires a valid ``X-Cluster-Key`` header matching the local coordinator's
        cluster key.  Prevents spoofed heartbeats from untrusted sources.
        """
        coord = state.coordinator
        if coord is None:
            return JSONResponse(status_code=503, content={"status": "unavailable"})

        # SECURITY: Cluster key is required
        local_key = getattr(coord.config, 'cluster_key', None)
        if not local_key:
            return JSONResponse(
                status_code=503,
                content={"status": "unavailable", "error": "Federation disabled: no cluster_key configured"},
            )
        received_key = request.headers.get("X-Cluster-Key", "")
        if not received_key:
            return JSONResponse(
                status_code=401,
                content={"status": "unauthorized", "error": "missing cluster key"},
            )

        # Accept current key or pending old key (grace period during rotation)
        key_valid = hmac.compare_digest(received_key, local_key)
        if not key_valid:
            pending_old = getattr(coord, "_pending_old_cluster_key", None)
            rotation_time = getattr(coord, "_key_rotation_time", 0)
            grace_expiry = rotation_time + 300  # 5-minute grace period
            if pending_old and time.time() < grace_expiry:
                key_valid = hmac.compare_digest(received_key, pending_old)
        if not key_valid:
            return JSONResponse(
                status_code=401,
                content={"status": "unauthorized", "error": "invalid cluster key"},
            )

        # Validate heartbeat body with Pydantic
        class FederationHeartbeat(BaseModel):
            active_requests: int = Field(default=0, ge=0)
            pending_requests: int = Field(default=0, ge=0)
            gpu_utilization: float = Field(default=0.0, ge=0.0, le=100.0)
            queue_depth: int | None = Field(default=None, ge=0)

            @field_validator("gpu_utilization")
            @classmethod
            def validate_utilization_range(cls, v: float) -> float:
                return max(0.0, min(100.0, v))

        raw_body = await request.json()
        heartbeat = FederationHeartbeat(**raw_body)

        federation = getattr(coord, 'federation', None)
        if federation is not None:
            try:
                for pid, peer in federation._peers.items():
                    if pid != federation.config.cluster_id:
                        federation._load_balancer.report_load(
                            cluster_id=pid,
                            active_requests=heartbeat.active_requests,
                            pending_requests=heartbeat.pending_requests,
                            gpu_utilization=heartbeat.gpu_utilization,
                            queue_depth=heartbeat.queue_depth or heartbeat.pending_requests,
                        )
            except Exception:
                logger.opt(exception=True).debug("Federation heartbeat processing failed")
        return {"status": "ok"}

    # ── Admin ──────────────────────────────────────────────────────────

    @app.post(
        "/api/cluster/rotate-key",
        summary="Rotate cluster key",
        description="Rotate the cluster authentication key. The new key is applied immediately "
                    "and must be distributed to all nodes. Supports a grace period during which "
                    "both the old and new key are accepted.",
        dependencies=[Depends(require_role("admin"))],
    )
    async def rotate_cluster_key(request: Request) -> dict:
        """Rotate the cluster authentication key.

        Generates a new cryptographically random key, applies it with a
        configurable grace period during which both old and new keys are
        accepted for HMAC verification.

        Rate-limited: max 1 rotation per 60 seconds to prevent
        rolling-DoS attacks (CWE-799).
        """
        # Rate limit: max 1 rotation per 60 seconds
        now = time.time()
        last_rotation = getattr(state, "_last_key_rotation_time", 0.0)
        if now - last_rotation < 60.0:
            remaining = int(60.0 - (now - last_rotation))
            return JSONResponse(
                status_code=429,
                content={
                    "status": "rate_limited",
                    "detail": f"Key rotation rate limited. Retry in {remaining}s.",
                    "retry_after_s": remaining,
                },
            )
        state._last_key_rotation_time = now

        new_key = secrets.token_urlsafe(32)

        coord = state.coordinator
        if coord is None:
            return {"status": "error", "detail": "coordinator not available"}

        old_key = getattr(coord.config, 'cluster_key', None)
        # Store old key as pending_old_key for grace period validation
        if old_key:
            coord.pending_old_cluster_key = old_key
            coord.key_rotation_time = time.time()

        coord.config.cluster_key = new_key
        logger.warning(
            f"Cluster key rotated by admin. "
            f"New key fingerprint: {hashlib.sha256(new_key.encode()).hexdigest()[:16]}"
        )
        return {
            "status": "ok",
            "new_key": new_key,
            "detail": "Save this key and distribute it to all nodes. "
                      "The previous key will be accepted for 5 minutes.",
        }

    @app.post(
        "/api/v1/ha/snapshot",
        summary="HA state snapshot",
        description="Receive a coordinator state snapshot from the leader for HA standby replication.",
        include_in_schema=False,
    )
    async def ha_state_snapshot(request: Request) -> dict:
        """Receive and apply a coordinator state snapshot from the HA leader.

        Authenticated via a shared HA secret header that both leader and
        standby coordinators use, preventing arbitrary state injection.
        """
        coord = state.coordinator
        if coord is None:
            return {"status": "error", "detail": "coordinator not available"}

        # SECURITY: Require HA shared secret (fail closed when not configured).
        # NOTE: this module is currently not wired into the app (dead code) —
        # kept fail-closed so it cannot silently reopen the CVE if ever used.
        expected_secret = os.environ.get("DISTLLM_HA_SECRET", "")
        if not expected_secret:
            return JSONResponse(
                status_code=403,
                content={"status": "error", "detail": "HA shared secret not configured"},
            )
        received_secret = request.headers.get("X-HA-Secret", "")
        if not hmac.compare_digest(received_secret, expected_secret):
            return JSONResponse(
                status_code=403,
                content={"status": "error", "detail": "invalid HA secret"},
            )

        try:
            class HASnapshot(BaseModel):
                nodes: dict = {}
                metadata: dict = {}

            raw = await request.json()
            snapshot = HASnapshot(**raw)
            coord.apply_state_snapshot(raw)
            return {"status": "ok", "applied_nodes": len(snapshot.nodes)}
        except Exception as e:
            return {"status": "error", "detail": str(e)}

    # ── Pipeline ───────────────────────────────────────────────────────

    @app.get(
        "/api/pipeline/health",
        summary="Pipeline orchestrator health and metrics",
        description="Return pipeline health status, node execution metrics, transport info, and configuration.",
        response_description="Pipeline health and metrics",
        dependencies=[Depends(require_role("auditor"))],
    )
    async def api_pipeline_health(request: Request) -> dict:
        """Return pipeline orchestrator health and metrics."""
        coord = state.coordinator
        if coord is None:
            return {"status": "no_coordinator"}
        pipeline = getattr(coord, '_pipeline', None)
        if pipeline is None:
            return {"status": "no_pipeline"}
        metrics = pipeline.get_pipeline_metrics()

        # Add per-node health and latency
        nodes = {}
        latency_tracker = getattr(pipeline, '_latency_tracker', None)
        for node_id, node in pipeline.nodes.items():
            node_latency = None
            if latency_tracker is not None:
                node_latency = latency_tracker.get_avg(node_id) if hasattr(latency_tracker, 'get_avg') else None
            nodes[node_id] = {
                "healthy": getattr(node, 'healthy', False),
                "latency_ms": node_latency,
                "start_layer": getattr(node, 'start_layer', 0),
                "end_layer": getattr(node, 'end_layer', 0),
                "gpu_name": getattr(node, 'gpu_name', ''),
            }
        metrics["nodes"] = nodes
        return metrics

    # ── Reputation ─────────────────────────────────────────────────────

    @app.get(
        "/api/cluster/reputation",
        summary="Node reputation scores",
        description="Return reputation scores for all registered nodes based on reliability, speed, uptime, and health.",
        response_description="Reputation scores per node",
        dependencies=[Depends(require_role("auditor"))],
    )
    async def api_cluster_reputation(request: Request) -> dict:
        """Return node reputation scores."""
        coord = state.coordinator
        if coord is None or not hasattr(coord, '_reputation'):
            return {"reputation": {}}
        reputation = getattr(coord, 'reputation', None)
        if reputation is None:
            return {"error": "Reputation system not available"}
        return reputation.get_summary()

    # ── Metrics ────────────────────────────────────────────────────────

    @app.get(
        "/api/metrics/collector",
        summary="Metrics collector snapshot",
        description="Return a snapshot of all collected metrics from the observability collector, including raw counters and gauges for instrumentation debugging.",
        response_description="Collector metrics snapshot",
        dependencies=[Depends(require_role("auditor"))],
    )
    async def api_collector_metrics(request: Request) -> dict:
        """Return current collector metrics snapshot."""
        return get_collector().summary()

    @app.get(
        "/api/metrics/stream",
        summary="Metrics SSE stream",
        description="Subscribe to a real-time metrics stream via Server-Sent Events. Use query parameters to filter metric categories and set update interval.",
        response_description="Event stream of structured metrics JSON.",
        dependencies=[Depends(require_role("auditor"))],
    )
    async def api_metrics_stream(
        request: Request,
        metrics: str = "",
        interval: float = 1.0,
    ) -> StreamingResponse:
        """SSE endpoint for real-time dashboard metrics.

        Query parameters:
          - ``metrics``: Comma-separated list of categories to subscribe to
                         (omit for all). e.g. ``metrics=latency,gpu,nodes``
          - ``interval``: Update interval in seconds (0.2-10.0, default 1.0)

        Supported categories: latency, ttft, throughput, tokens_per_sec,
        kv_cache, speculative, cost, queue_depth, active_requests, scheduler,
        nodes, gpu, prefix_cache, spec_decoder, topology, tenants.
        """
        interval = max(0.2, min(interval, 10.0))
        requested = None
        if metrics:
            requested = {m.strip() for m in metrics.split(",") if m.strip()}
            unknown = requested - KNOWN_METRIC_CATEGORIES
            if unknown:
                return JSONResponse(
                    status_code=400,
                    content={
                        "detail": f"Unknown metric categories: {', '.join(sorted(unknown))}",
                        "valid_categories": sorted(KNOWN_METRIC_CATEGORIES),
                    },
                )

        return StreamingResponse(
            stream_metrics_sse(state.coordinator, requested_metrics=requested, interval=interval),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # ── Waterfall ──────────────────────────────────────────────────────

    @app.get(
        "/api/requests/waterfall",
        summary="Recent request waterfall data",
        description="Return recent request lifecycle data (queue -> prefill -> decode) for the dashboard waterfall chart.",
        response_description="List of request timing entries with elapsed_ms, ttft_ms, and request_id",
        dependencies=[Depends(require_role("auditor"))],
    )
    async def api_waterfall(request: Request, limit: int = 50) -> list:
        """Return recent request waterfall entries showing lifecycle phases."""
        coord = state.coordinator
        if coord is None:
            return []
        try:
            scheduler = getattr(coord, "scheduler", None)
            if scheduler is None:
                return []
            tracker = getattr(scheduler, "latency_tracker", None) or getattr(scheduler, "_latency_tracker", None)
            if tracker is None:
                return []
            return tracker.get_recent_metrics(limit=limit)
        except (AttributeError, RuntimeError):
            return []

    # ── Continuum ──────────────────────────────────────────────────────

    @app.get(
        "/api/continuum/stats",
        summary="Edge-to-cloud continuum statistics",
        description="Return statistics about the edge-to-cloud device continuum including device types, transports, and layer assignments.",
        response_description="Continuum statistics",
        dependencies=[Depends(require_role("auditor"))],
    )
    async def api_continuum_stats(request: Request) -> dict:
        """Return edge-to-cloud continuum statistics."""
        continuum = getattr(state, "continuum", None)
        if continuum is None:
            return {"status": "not_initialized"}
        return continuum.get_stats()

    # ── Cost ───────────────────────────────────────────────────────────

    @app.get(
        "/api/cost/summary",
        summary="Cost tracking summary",
        description="Return cost tracking summary including per-request costs, savings vs cloud APIs, and throughput metrics.",
        response_description="Cost summary",
        dependencies=[Depends(require_role("auditor"))],
    )
    async def api_cost_summary(request: Request, tenant_id: str = "") -> dict:
        """Return cost tracking summary."""
        try:
            from distllm.core.cost_tracker import get_cost_tracker
            return get_cost_tracker().get_cost_summary(tenant_id)
        except ImportError:
            return {"status": "not_available"}

    @app.get(
        "/api/cost/history",
        summary="Cost history",
        description="Return recent cost tracking history.",
        response_description="Cost history entries",
        dependencies=[Depends(require_role("auditor"))],
    )
    async def api_cost_history(request: Request, limit: int = 100) -> list:
        """Return recent cost history."""
        try:
            from distllm.core.cost_tracker import get_cost_tracker
            return get_cost_tracker().get_history(limit)
        except ImportError:
            return []

    @app.get(
        "/api/streaming-cost/stats",
        summary="Streaming cost statistics",
        description="Return real-time streaming cost tracker statistics.",
        response_description="Streaming cost stats",
        dependencies=[Depends(require_role("auditor"))],
    )
    async def api_streaming_cost_stats(request: Request) -> dict:
        """Return streaming cost tracker statistics."""
        try:
            from distllm.core.streaming_cost import get_streaming_cost_tracker
            return get_streaming_cost_tracker().get_stats()
        except ImportError:
            return {"status": "not_available"}
