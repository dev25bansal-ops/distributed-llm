"""Health, readiness, liveness, metrics, and model management routes."""

import time

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ..api_state import g


router = APIRouter(tags=["system"])


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


@router.get("/v1/models")
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


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    coord = g.coordinator
    if coord is None:
        return {"status": "unhealthy", "reason": "No model loaded"}

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


@router.get("/ready")
async def readiness_check():
    """Kubernetes readiness probe.

    Returns 200 only when the service can accept traffic.
    """
    coord = g.coordinator
    if coord is None:
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "reason": "No model loaded"},
        )

    if getattr(coord, "_shutting_down", False):
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "reason": "Service is shutting down"},
        )

    # Check if at least one node is healthy (for distributed mode)
    if coord.nodes:
        node_health = coord.health_check()
        healthy_nodes = sum(1 for h in node_health.values() if h.get("healthy"))
        if healthy_nodes == 0:
            return JSONResponse(
                status_code=503,
                content={
                    "status": "not_ready",
                    "reason": "No healthy nodes available",
                    "healthy_nodes": 0,
                    "total_nodes": len(coord.nodes),
                },
            )

    return {"status": "ready"}


@router.get("/live")
async def liveness_check():
    """Kubernetes liveness probe.

    Returns 200 if the process is alive and not deadlocked.
    """
    return {"status": "alive", "uptime_seconds": time.time() - g._startup_time}


@router.get("/metrics")
async def metrics():
    """Prometheus-compatible metrics endpoint."""
    coord = g.coordinator
    mon = g.monitor
    lines = []

    if coord is None:
        # Return service status even when not initialized
        lines.append("# TYPE distllm_service_up gauge")
        lines.append("distllm_service_up 0")
        lines.append("# TYPE distllm_coordinator_loaded gauge")
        lines.append("distllm_coordinator_loaded 0")
        return "\n".join(lines)

    # Use Prometheus exporter if available
    if coord.metrics_exporter:
        return coord.metrics_exporter.generate_metrics()

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

    return "\n".join(lines)


@router.post("/v1/update-params/{request_id}")
async def update_generation_params(request_id: str, params: ParamUpdateRequest):
    """Update generation parameters for an in-progress request.

    Allows changing temperature, top_p, and top_k mid-generation for streaming requests.
    """
    coord = g.coordinator
    if coord is None:
        raise HTTPException(status_code=503, detail="Coordinator not initialized")

    updated = coord._param_update_channel.update(
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
