"""Health, readiness, liveness, metrics, and model management routes."""

import time

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse, Response
from pydantic import BaseModel, ConfigDict, Field

from ..api_state import g
from distllm.api.auth_deps import require_role
from distllm.api.errors import error_response_from_request, error_openapi_entry
from distllm.api import server as _server_state


router = APIRouter(tags=["system"])

# Shorthand used in route ``responses={...}`` dicts so Swagger UI renders the
# concrete error envelope every failure path returns.
ERR_ENTRY = error_openapi_entry


class ModelInfo(BaseModel):
    id: str = Field(..., description="Model identifier")
    object: str = "model"
    created: int = Field(..., description="Unix timestamp of model creation")
    owned_by: str = "distributed-llm"
    root: str | None = Field(default=None, description="Root model for fine-tuned models")
    archived: bool = Field(default=False, description="Whether the model is archived")


class ModelList(BaseModel):
    object: str = "list"
    data: list[ModelInfo]


class ParamUpdateRequest(BaseModel):
    """Request to update generation parameters mid-stream."""
    temperature: float | None = Field(default=None, ge=0, le=2.0, description="New sampling temperature")
    top_p: float | None = Field(default=None, ge=0, le=1.0, description="New nucleus sampling threshold")
    top_k: int | None = Field(default=None, ge=0, description="New top-k sampling value")


# ── Docs-only response schemas ────────────────────────────────────────────────
# Referenced from ``responses={...}`` to document probe payloads in Swagger UI
# without enforcing them at runtime (handlers return heterogeneous dicts that
# may carry extra monitor keys). ``extra="allow"`` keeps those keys visible.


class HealthStatusResponse(BaseModel):
    """Health check payload."""
    model_config = ConfigDict(extra="allow")
    status: str = Field(default="healthy", description='"healthy" when a coordinator is loaded')
    model: str = Field(default="", description="Name of the loaded model")
    nodes: int = Field(default=0, description="Number of connected worker nodes")
    node_health: dict = Field(default_factory=dict, description="Per-node health details")


class ReadinessResponse(BaseModel):
    """Successful readiness payload."""
    status: str = Field(default="ready", description='"ready" when the service accepts traffic')


class LivenessResponse(BaseModel):
    """Liveness probe payload."""
    status: str = Field(default="alive", description='"alive" while the process is responsive')
    uptime_seconds: float = Field(default=0.0, description="Seconds since server start")


class PluginReadinessResponse(BaseModel):
    """Plugin-based readiness payload (adds diagnostic reasons on failure)."""
    model_config = ConfigDict(extra="allow")
    status: str = Field(default="ready", description='"ready" or "not_ready"')
    reasons: list[str] = Field(default_factory=list, description="Present on 503: why the service is not ready")


@router.get(
    "/v1/models",
    summary="List available models",
    description="Return all registered and available models with metadata including model ID and creation timestamp. Returns an empty list if no coordinator is loaded. Requires API-key authentication.",
    response_description="List of model identifiers and metadata",
    response_model=ModelList,
    responses={
        401: ERR_ENTRY("Missing or invalid API key (`Authorization: Bearer <key>`)", type_="auth_error", code="authentication_error"),
        429: ERR_ENTRY("Rate limit or auth-failure limit exceeded; retry after the interval in the error body", type_="rate_limit_error", code="rate_limit_exceeded"),
    },
)
async def list_models():
    """List available models."""
    coord = g.coordinator
    if coord is None:
        return ModelList(data=[])
    if hasattr(coord, 'list_models'):
        model_names = coord.list_models()
    else:
        model_names = [coord.model_name]
    return ModelList(
        data=[
            ModelInfo(id=name, created=int(time.time()))
            for name in model_names
        ]
    )


@router.get(
    "/v1/health",
    summary="Health check (OpenAI-compatible)",
    description="OpenAI-compatible health check endpoint. Returns the current health status of the service, including model name, connected nodes, per-node health, and optional monitor metrics. Returns 503 if no model is loaded. Unauthenticated (safe for load balancers).",
    response_description="Health status with node and model information",
    responses={
        200: {"model": HealthStatusResponse},
        503: ERR_ENTRY("No model loaded or coordinator not initialized", type_="health_error", code="503"),
    },
)
@router.get(
    "/health",
    summary="Health check (legacy)",
    include_in_schema=False,
)
async def health_check(request: Request):
    """Health check endpoint (supports /health and /v1/health)."""
    coord = g.coordinator
    if coord is None:
        return error_response_from_request(503, "Service Unavailable", "No model loaded", "health_error", request)

    node_health = coord.health_check() if coord.nodes else {}
    health = {
        "status": "healthy",
        "model": coord.model_name,
        "nodes": len(coord.nodes),
        "node_health": node_health,
    }

    mon = g.monitor
    if mon and coord.scheduler:
        health.update(mon.health_check(coord.scheduler))

    return health


@router.get(
    "/v1/health/readiness",
    summary="Kubernetes readiness probe (versioned alias of /ready)",
    include_in_schema=False,
)
async def readiness_check_v1(request: Request):
    """Versioned alias of :func:`readiness_check`."""
    return await readiness_check(request)


@router.get(
    "/v1/health/liveness",
    summary="Kubernetes liveness probe (versioned alias of /live)",
    include_in_schema=False,
)
async def liveness_check_v1():
    """Versioned alias of :func:`liveness_check`."""
    return await liveness_check()


@router.get(
    "/ready",
    summary="Kubernetes readiness probe",
    description="Kubernetes readiness probe. Returns 200 only when the service can accept traffic (model loaded, not shutting down, healthy nodes available). Designed for container orchestration health checks.",
    response_description="Readiness status with 'ready' or 503 error",
    responses={
        200: {"model": ReadinessResponse},
        503: ERR_ENTRY("No model loaded, service shutting down, or no healthy nodes available", type_="health_error", code="503"),
    },
)
async def readiness_check(request: Request):
    """Kubernetes readiness probe.

    Returns 200 only when the service can accept traffic.
    """
    coord = g.coordinator
    if coord is None:
        return error_response_from_request(503, "Service Unavailable", "No model loaded", "health_error", request)

    if getattr(coord, "_shutting_down", False):
        return error_response_from_request(503, "Service Unavailable", "Service is shutting down", "health_error", request)

    # Check if at least one node is healthy (for distributed mode)
    if coord.nodes:
        node_health = coord.health_check()
        healthy_nodes = sum(1 for h in node_health.values() if h.get("healthy"))
        if healthy_nodes == 0:
            return error_response_from_request(503, "Service Unavailable", "No healthy nodes available", "health_error", request)

    return {"status": "ready"}


@router.get(
    "/live",
    summary="Kubernetes liveness probe",
    description="Kubernetes liveness probe. Returns 200 with uptime in seconds if the process is alive and not deadlocked. Designed for container orchestration health checks.",
    response_description="Liveness status with uptime",
    responses={200: {"model": LivenessResponse}},
)
async def liveness_check():
    """Kubernetes liveness probe.

    Returns 200 if the process is alive and not deadlocked.
    """
    return {"status": "alive", "uptime_seconds": time.time() - g.startup_time}


@router.get(
    "/healthz",
    summary="Plugin-based liveness probe",
    description="Liveness probe powered by HealthPlugin. Returns 200 when the process is responsive. "
                "Kubernetes restarts the pod only when this endpoint fails. Falls back to a basic "
                "alive response if the HealthPlugin is not enabled.",
    response_description="Liveness status",
    responses={200: {"model": LivenessResponse}},
)
async def healthz():
    """Plugin-based liveness probe (/healthz).

    Delegates to HealthPlugin.liveness() when available.
    Always returns 200 — Kubernetes restarts the pod only on failure.
    """
    plugin = _get_health_plugin()
    if plugin is not None:
        return plugin.liveness()
    # Fallback: basic alive check when plugin is not loaded
    return {"status": "alive", "uptime_seconds": time.time() - g.startup_time}


@router.get(
    "/readyz",
    summary="Plugin-based readiness probe",
    description="Readiness probe powered by HealthPlugin. Returns 200 only when the service can "
                "accept traffic: model loaded, error rate below threshold, GPU and system memory "
                "within limits, and circuit breaker not open. Returns 503 with reasons otherwise.",
    response_description="Readiness status with diagnostic details",
    responses={
        200: {"model": PluginReadinessResponse},
        503: {
            "description": "Service not ready — body lists the failing checks",
            "content": {
                "application/json": {
                    "example": {"status": "not_ready", "reasons": ["no_coordinator"]},
                },
            },
        },
    },
)
async def readyz():
    """Plugin-based readiness probe (/readyz).

    Delegates to HealthPlugin.readiness() which checks:
    - Coordinator is loaded
    - Error rate is below threshold
    - GPU memory is within limits
    - System memory is within limits
    - Circuit breaker is not open
    """
    plugin = _get_health_plugin()
    if plugin is not None:
        body, status_code = plugin.readiness()
        if status_code != 200:
            return JSONResponse(status_code=status_code, content=body)
        return body
    # Fallback: basic readiness when plugin is not loaded
    coord = g.coordinator
    if coord is None:
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "reasons": ["no_coordinator"]},
        )
    return {"status": "ready"}


def _get_health_plugin():
    """Resolve HealthPlugin from the plugin system, if loaded."""
    try:
        ps = getattr(_server_state, "state", None)
        if ps is None:
            return None
        plugin_sys = getattr(ps, "plugin_system", None)
        if plugin_sys is None:
            return None
        inst = plugin_sys.get_plugin("health")
        if inst and inst.instance is not None:
            return inst.instance
    except (AttributeError, Exception):
        pass
    return None


@router.post(
    "/v1/models/{model_id}/warmup",
    summary="Warm up a model (CUDA graphs, caches)",
    description="Send dummy tokens through the model to warm CUDA graphs, kernel caches, and memory pools before production traffic arrives."
                " This reduces first-token latency (TTFT) for subsequent requests. Restricted to model-admin or higher.",
    response_description="Warmup status",
    responses={503: {"description": "Model not loaded or coordinator not initialized"}},
    dependencies=[Depends(require_role("model-admin"))],
)
async def warmup_model(model_id: str, request: Request):
    """Warm up a model by running a dummy forward pass."""
    coord = g.coordinator
    if coord is None:
        return error_response_from_request(503, "Service Unavailable", "No coordinator", "warmup_error", request)
    if coord.tokenizer is None:
        return error_response_from_request(503, "Service Unavailable", "Tokenizer not loaded", "warmup_error", request)

    dummy_prompt = "Hello"
    warmup_tokens = 16
    try:
        import asyncio
        result = await asyncio.to_thread(
            coord.generate, dummy_prompt,
            max_new_tokens=warmup_tokens, temperature=0.0,
        )
        return {"status": "ok", "model": model_id, "warmup_tokens": warmup_tokens, "length": len(result)}
    except Exception as e:
        return error_response_from_request(503, "Warmup Failed", str(e), "warmup_error", request)


@router.get(
    "/metrics",
    summary="Prometheus metrics",
    description="Expose Prometheus-compatible metrics including service status, coordinator state, scheduler stats, prefix cache performance, GPU utilization, and system health. Returns plaintext in Prometheus exposition format. Unauthenticated (safe for Prometheus scrapers).",
    response_description="Prometheus-formatted metrics text",
    responses={
        200: {
            "description": "Prometheus text exposition format (version=0.0.4)",
            "content": {
                "text/plain": {
                    "schema": {"type": "string"},
                    "example": (
                        "# TYPE distllm_service_up gauge\n"
                        "distllm_service_up 1\n"
                        "# TYPE distllm_active_requests gauge\n"
                        "distllm_active_requests 3\n"
                    ),
                },
            },
        },
    },
)
async def metrics():
    """Prometheus-compatible metrics endpoint."""
    from prometheus_client import generate_latest

    coord = g.coordinator
    mon = g.monitor
    lines = []

    # Include the API request-layer RED metrics recorded by
    # ObservabilityMiddleware into the shared exporter registry, so /metrics
    # reflects request rate/errors/duration in addition to coordinator gauges.
    api_exporter = getattr(g, "metrics_exporter", None)
    api_registry = getattr(api_exporter, "registry", None)
    api_metrics = generate_latest(api_registry).decode() if api_registry is not None else ""

    if coord is None:
        # Return service status even when not initialized
        lines.append("# TYPE distllm_service_up gauge")
        lines.append("distllm_service_up 0")
        lines.append("# TYPE distllm_coordinator_loaded gauge")
        lines.append("distllm_coordinator_loaded 0")
        return api_metrics + "\n" + "\n".join(lines) if api_metrics else "\n".join(lines)

    # Use Prometheus exporter if available
    if coord.metrics_exporter:
        coord.metrics_exporter.populate_gauges(coordinator=coord)
        data = coord.metrics_exporter.generate_metrics()
        combined = api_metrics + "\n" + data if api_metrics else data
        return Response(content=combined, media_type="text/plain; version=0.0.4; charset=utf-8")

    # Fallback: dict-based text format
    m = coord.get_metrics()

    # Add service status
    lines.append("# TYPE distllm_service_up gauge")
    lines.append("distllm_service_up 1")
    lines.append("# TYPE distllm_coordinator_loaded gauge")
    lines.append("distllm_coordinator_loaded 1")

    for name, value in m.items():
        if isinstance(value, (int, float)):
            lines.append(f"# TYPE {name} gauge")
            lines.append(f"{name} {value}")
        else:
            lines.append(f"# {name} {value}")

    # Add scheduler metrics
    if coord.scheduler:
        stats = coord.scheduler.stats()
        lines.append("# TYPE distllm_active_requests gauge")
        lines.append(f"distllm_active_requests {stats['active_requests']}")
        lines.append("# TYPE distllm_pending_requests gauge")
        lines.append(f"distllm_pending_requests {stats['pending_requests']}")

    # Add prefix cache metrics
    if coord.prefix_cache:
        pc_stats = coord.prefix_cache.stats()
        for name, value in pc_stats.items():
            if isinstance(value, (int, float)):
                lines.append(f"# TYPE distllm_{name} gauge")
                lines.append(f"distllm_{name} {value}")

    # Add system metrics
    if mon:
        sys_metrics = mon.collect()
        if "cpu" in sys_metrics:
            lines.append("# TYPE distllm_cpu_percent gauge")
            lines.append(f"distllm_cpu_percent {sys_metrics['cpu']['percent']}")
        if "gpu" in sys_metrics:
            gpu = sys_metrics["gpu"]
            if "memory_percent" in gpu:
                lines.append("# TYPE distllm_gpu_memory_percent gauge")
                lines.append(f"distllm_gpu_memory_percent {gpu['memory_percent']}")
            if "temperature_c" in gpu:
                lines.append("# TYPE distllm_gpu_temperature_c gauge")
                lines.append(f"distllm_gpu_temperature_c {gpu['temperature_c']}")

    # Add readiness status
    lines.append("# TYPE distllm_ready gauge")
    lines.append(f"distllm_ready {1 if not getattr(coord, '_shutting_down', False) else 0}")

    body = "\n".join(lines)
    return api_metrics + "\n" + body if api_metrics else body


@router.post(
    "/v1/update-params/{request_id}",
    summary="Update generation parameters",
    description="Dynamically update generation parameters (temperature, top_p, top_k) for an in-progress streaming request. Enables real-time steering of model output without restarting generation.",
    response_description="Updated parameter values for the request",
    responses={
        404: {"description": "Request ID not found or already completed"},
        503: {"description": "Coordinator not initialized"},
    },
    dependencies=[Depends(require_role("inference-only"))],
)
async def update_generation_params(request_id: str, params: ParamUpdateRequest):
    """Update generation parameters for an in-progress request.

    Allows changing temperature, top_p, and top_k mid-generation for streaming requests.
    """
    coord = g.coordinator
    if coord is None:
        raise HTTPException(status_code=503, detail="Coordinator not initialized")

    puc = getattr(coord, '_param_update_channel', None)
    if puc is None:
        return {"status": "error", "detail": "param_update_channel not available"}
    updated = puc.update(
        request_id,
        temperature=params.temperature if params.temperature is not None else None,
        top_p=params.top_p if params.top_p is not None else None,
        top_k=params.top_k if params.top_k is not None else None,
    )
    if updated is None:
        raise HTTPException(status_code=404, detail=f"Request {request_id} not found or already completed")

    return {
        "request_id": request_id,
        "temperature": updated.temperature,
        "top_p": updated.top_p,
        "top_k": updated.top_k,
    }
