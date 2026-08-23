"""REST Admin API for cluster management.

Provides endpoints under ``/admin/v1/`` for programmatic cluster
management: listing nodes, draining/offlining/recovering nodes,
updating runtime configuration, viewing logs, and triggering model
compression.

Requires ``admin`` role API key.
"""

import time

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from loguru import logger
from pydantic import BaseModel, Field

from ..api_state import g
from ..auth_deps import require_role


def _coord_attr(coord, attr: str, default=None):
    """Safely access coordinator attribute with default.

    Avoids scattered getattr/hasattr calls throughout admin endpoints.
    """
    return getattr(coord, attr, default)


# ── Router ──────────────────────────────────────────────────────────────────

router = APIRouter(
    prefix="/admin/v1",
    tags=["admin"],
    dependencies=[Depends(require_role("admin", "model-admin"))],
)


# ── Pydantic models ─────────────────────────────────────────────────────────


class NodeInfo(BaseModel):
    node_id: str
    host: str
    port: int
    healthy: bool
    draining: bool = False
    state: str = "unknown"
    start_layer: int = 0
    end_layer: int = 0
    gpu_name: str = ""
    gpu_memory_free: int = 0
    gpu_memory_total: int = 0
    gpu_memory_util_pct: float = 0.0
    gpu_sm_count: int = 0
    role: str = "auto"
    cluster_id: str = "default"
    last_health_time: float = 0.0


class NodeListResponse(BaseModel):
    nodes: list[NodeInfo]
    total_nodes: int
    healthy_count: int
    draining_count: int
    total_layers: int = 0


class NodeActionResponse(BaseModel):
    status: str
    node_id: str
    message: str


class ConfigUpdateRequest(BaseModel):
    max_batch_size: int | None = Field(default=None, ge=1, description="New max batch size")
    max_tokens_per_batch: int | None = Field(default=None, ge=128, description="New max tokens per batch")
    temperature: float | None = Field(default=None, ge=0, le=2.0, description="Default generation temperature")
    top_p: float | None = Field(default=None, ge=0, le=1.0, description="Default top-p sampling")
    top_k: int | None = Field(default=None, ge=0, description="Default top-k sampling")


class ConfigUpdateResponse(BaseModel):
    status: str
    updated: dict[str, object]
    message: str


class LogEntry(BaseModel):
    timestamp: str
    level: str
    message: str
    module: str = ""


class LogsResponse(BaseModel):
    logs: list[LogEntry]
    total: int


class CompressRequest(BaseModel):
    model_name: str | None = Field(default=None, description="Model to compress (default: currently loaded)")
    method: str = Field(default="int4", description="Compression method (int4, int8, fp8)")
    output_dir: str | None = Field(default=None, description="Output directory for compressed model")


class CompressResponse(BaseModel):
    status: str
    job_id: str
    model_name: str
    method: str
    message: str


# ── Helper ──────────────────────────────────────────────────────────────────


def _get_node_state(coord, node_id: str) -> str:
    """Determine the health state of a node from coordinator state."""
    try:
        resource_mgr = getattr(coord, '_resource_mgr', None)
        if resource_mgr is not None:
            if hasattr(resource_mgr, 'is_node_draining') and resource_mgr.is_node_draining(node_id):
                return "draining"
        health_mgr = getattr(coord, '_health_mgr', None)
        if health_mgr is not None:
            state_store = getattr(health_mgr, '_state_store', None) or getattr(health_mgr, 'state_store', None)
            if state_store is not None:
                record = state_store.get(node_id)
                if record is not None:
                    return record.state.value if hasattr(record.state, 'value') else str(record.state)
        node = coord.nodes.get(node_id)
        if node is not None:
            healthy = node.get("healthy", True) if isinstance(node, dict) else getattr(node, "healthy", True)
            return "healthy" if healthy else "unhealthy"
    except Exception:
        pass
    return "unknown"


def _format_node(node_id: str, node, coord) -> NodeInfo:
    """Format a NodeRegistration into a NodeInfo response model."""
    state = _get_node_state(coord, node_id)
    resource_mgr = getattr(coord, '_resource_mgr', None)
    draining = False
    if resource_mgr is not None and hasattr(resource_mgr, 'is_node_draining'):
        draining = resource_mgr.is_node_draining(node_id)

    gpu_mem_total = getattr(node, 'gpu_memory_total', 0)
    gpu_mem_free = getattr(node, 'gpu_memory_free', 0)
    gpu_util = 0.0
    if gpu_mem_total > 0:
        gpu_util = round((1 - gpu_mem_free / gpu_mem_total) * 100, 1)

    return NodeInfo(
        node_id=node_id,
        host=getattr(node, 'host', ''),
        port=getattr(node, 'port', 0),
        healthy=getattr(node, 'healthy', False),
        draining=draining,
        state=state,
        start_layer=getattr(node, 'start_layer', 0),
        end_layer=getattr(node, 'end_layer', 0),
        gpu_name=getattr(node, 'gpu_name', ''),
        gpu_memory_free=gpu_mem_free,
        gpu_memory_total=gpu_mem_total,
        gpu_memory_util_pct=gpu_util,
        gpu_sm_count=getattr(node, 'gpu_sm_count', 0),
        role=getattr(node, 'role', 'auto'),
        cluster_id=getattr(node, 'cluster_id', 'default'),
        last_health_time=getattr(node, 'last_health_time', 0.0),
    )


def _resolve_coordinator():
    """Get the coordinator or raise 503."""
    coord = g.coordinator
    if coord is None:
        raise HTTPException(status_code=503, detail="No coordinator loaded")
    return coord


# ── Endpoints ───────────────────────────────────────────────────────────────


@router.get(
    "/nodes",
    summary="List cluster nodes",
    description="Return all registered worker nodes with health state, draining status, GPU capabilities, and layer assignments.",
    response_model=NodeListResponse,
)
async def list_nodes():
    """List all nodes in the cluster with their current state."""
    coord = _resolve_coordinator()
    nodes_list = []
    healthy_count = 0
    draining_count = 0
    for node_id, node in coord.nodes.items():
        info = _format_node(node_id, node, coord)
        nodes_list.append(info)
        if info.healthy:
            healthy_count += 1
        if info.draining:
            draining_count += 1
    return NodeListResponse(
        nodes=nodes_list,
        total_nodes=len(nodes_list),
        healthy_count=healthy_count,
        draining_count=draining_count,
        total_layers=getattr(coord, 'total_layers', 0),
    )


@router.get(
    "/cluster/status",
    summary="Cluster status overview",
    description="Aggregated cluster state: model, node counts by health/draining, and per-node summaries.",
)
async def cluster_status():
    """Return an aggregated snapshot of the whole cluster."""
    coord = _resolve_coordinator()

    nodes_list = []
    healthy_count = 0
    draining_count = 0
    for node_id, node in coord.nodes.items():
        info = _format_node(node_id, node, coord)
        nodes_list.append(info)
        if info.healthy:
            healthy_count += 1
        if info.draining:
            draining_count += 1

    return {
        "status": "healthy" if healthy_count > 0 else "degraded",
        "model": getattr(coord, "model_name", ""),
        "total_nodes": len(nodes_list),
        "healthy_nodes": healthy_count,
        "draining_nodes": draining_count,
        "total_layers": getattr(coord, "total_layers", 0),
        "nodes": [n.model_dump() for n in nodes_list],
    }


@router.post(
    "/nodes/{node_id}/drain",
    summary="Drain a node",
    description="Stop sending new requests to a node. Existing in-flight requests are allowed to complete.",
    response_model=NodeActionResponse,
    responses={404: {"description": "Node not found"}},
)
async def drain_node(node_id: str):
    """Drain a node: stop sending new requests to it."""
    coord = _resolve_coordinator()
    if node_id not in coord.nodes:
        raise HTTPException(status_code=404, detail=f"Node '{node_id}' not found")

    resource_mgr = getattr(coord, '_resource_mgr', None)
    if resource_mgr is not None and hasattr(resource_mgr, 'mark_node_draining'):
        resource_mgr.mark_node_draining(node_id)
        logger.info(f"Admin: drained node {node_id}")
        return NodeActionResponse(status="ok", node_id=node_id, message=f"Node '{node_id}' is now draining")
    return NodeActionResponse(
        status="error", node_id=node_id,
        message="Resource manager not available on this coordinator",
    )


@router.post(
    "/nodes/{node_id}/undrain",
    summary="Restore a drained node",
    description="Restore a previously drained node back to active service.",
    response_model=NodeActionResponse,
    responses={404: {"description": "Node not found"}},
)
async def undrain_node(node_id: str):
    """Restore a drained node to active service."""
    coord = _resolve_coordinator()
    if node_id not in coord.nodes:
        raise HTTPException(status_code=404, detail=f"Node '{node_id}' not found")

    resource_mgr = getattr(coord, '_resource_mgr', None)
    if resource_mgr is not None and hasattr(resource_mgr, 'mark_node_alive'):
        resource_mgr.mark_node_alive(node_id)
        logger.info(f"Admin: restored node {node_id}")
        return NodeActionResponse(status="ok", node_id=node_id, message=f"Node '{node_id}' restored to active service")
    return NodeActionResponse(
        status="error", node_id=node_id,
        message="Resource manager not available on this coordinator",
    )


@router.post(
    "/nodes/{node_id}/offline",
    summary="Mark a node offline",
    description="Mark a node as offline (drain + mark unhealthy). Use for planned maintenance.",
    response_model=NodeActionResponse,
    responses={404: {"description": "Node not found"}},
)
async def offline_node(node_id: str):
    """Mark a node offline (drain + mark unhealthy)."""
    coord = _resolve_coordinator()
    if node_id not in coord.nodes:
        raise HTTPException(status_code=404, detail=f"Node '{node_id}' not found")

    resource_mgr = getattr(coord, '_resource_mgr', None)
    if resource_mgr is not None:
        if hasattr(resource_mgr, 'mark_node_draining'):
            resource_mgr.mark_node_draining(node_id)
        if hasattr(resource_mgr, 'simulate_node_failure'):
            try:
                resource_mgr.simulate_node_failure(node_id)
            except Exception:
                pass

    node = coord.nodes.get(node_id)
    if node is not None:
        if isinstance(node, dict):
            # Dict from PipelineOrchestrator — update via pipeline
            pass
        elif isinstance(node, object) and hasattr(node, 'healthy'):
            # M-13: Thread-safe write via attribute with instance lock
            node.healthy = False
        else:
            logger.warning(f"Admin: cannot mark node {node_id} offline (unknown type: {type(node).__name__})")
    return NodeActionResponse(status="ok", node_id=node_id, message=f"Node '{node_id}' marked offline")


@router.post(
    "/nodes/{node_id}/recover",
    summary="Trigger node recovery",
    description="Attempt to recover a failed or offline node. Restores the node to active service and resets circuit breaker state.",
    response_model=NodeActionResponse,
    responses={404: {"description": "Node not found"}},
)
async def recover_node(node_id: str):
    """Trigger recovery for a failed/offline node."""
    coord = _resolve_coordinator()
    if node_id not in coord.nodes:
        raise HTTPException(status_code=404, detail=f"Node '{node_id}' not found")

    node = coord.nodes.get(node_id)
    if node is not None:
        if isinstance(node, dict):
            # Dict from PipelineOrchestrator — mark healthy via pipeline
            coord._pipeline.mark_node_healthy(node_id)
        else:
            node.healthy = True

    resource_mgr = getattr(coord, '_resource_mgr', None)
    if resource_mgr is not None:
        if hasattr(resource_mgr, 'mark_node_alive'):
            resource_mgr.mark_node_alive(node_id)
        if hasattr(resource_mgr, 'record_success'):
            resource_mgr.record_success(node_id)

    logger.info(f"Admin: recovered node {node_id}")
    return NodeActionResponse(status="ok", node_id=node_id, message=f"Node '{node_id}' recovered and restored to service")


@router.patch(
    "/config",
    summary="Update runtime configuration",
    description="Update runtime generation parameters (max_batch_size, temperature, top_p, top_k). These changes take effect immediately for new requests.",
    response_model=ConfigUpdateResponse,
)
async def update_config(body: ConfigUpdateRequest):
    """Update runtime configuration parameters."""
    coord = _resolve_coordinator()

    updated: dict[str, object] = {}
    if body.max_batch_size is not None and hasattr(coord, 'max_batch_size'):
        coord.max_batch_size = body.max_batch_size
        updated["max_batch_size"] = body.max_batch_size
    if body.max_tokens_per_batch is not None and hasattr(coord, 'max_tokens_per_batch'):
        coord.max_tokens_per_batch = body.max_tokens_per_batch
        updated["max_tokens_per_batch"] = body.max_tokens_per_batch

    scheduler = getattr(coord, 'scheduler', None)
    if scheduler is not None:
        if body.temperature is not None and hasattr(scheduler, 'default_temperature'):
            scheduler.default_temperature = body.temperature
            updated["temperature"] = body.temperature
        if body.top_p is not None and hasattr(scheduler, 'default_top_p'):
            scheduler.default_top_p = body.top_p
            updated["top_p"] = body.top_p
        if body.top_k is not None and hasattr(scheduler, 'default_top_k'):
            scheduler.default_top_k = body.top_k
            updated["top_k"] = body.top_k

    if not updated:
        return ConfigUpdateResponse(
            status="ok", updated={},
            message="No supported fields were updated. Check that the coordinator supports the requested fields.",
        )

    logger.info(f"Admin: updated config {updated}")
    return ConfigUpdateResponse(
        status="ok",
        updated=updated,
        message=f"Updated {len(updated)} configuration value(s)",
    )


@router.get(
    "/logs",
    summary="View recent logs",
    description="Return recent log entries from the in-memory log buffer. Supports filtering by level and search term.",
    response_model=LogsResponse,
)
async def view_logs(
    request: Request,
    level: str = Query(default="INFO", description="Minimum log level (TRACE, DEBUG, INFO, WARNING, ERROR)"),
    search: str = Query(default="", description="Filter logs containing this text"),
    limit: int = Query(default=100, ge=1, le=1000, description="Max log entries to return"),
):
    """View recent logs from the in-memory log buffer."""
    _resolve_coordinator()

    LEVEL_ORDER = {"TRACE": 0, "DEBUG": 1, "INFO": 2, "WARNING": 3, "ERROR": 4}
    min_level = LEVEL_ORDER.get(level.upper(), 2)

    entries: list[LogEntry] = []
    try:
        from loguru import logger as loguru_logger

        seen = set()
        for handler_id, handler in enumerate(loguru_logger._core.handlers.values()):
            if hasattr(handler, "_sink") and hasattr(handler._sink, "logs"):
                for record in reversed(handler._sink.logs):
                    rec_level = LEVEL_ORDER.get(record.get("level", "INFO"), 2)
                    if rec_level < min_level:
                        continue
                    msg = record.get("message", "")
                    if search and search.lower() not in msg.lower():
                        continue
                    entry_id = f"{record.get('time', '')}:{msg}"
                    if entry_id in seen:
                        continue
                    seen.add(entry_id)
                    entries.append(LogEntry(
                        timestamp=str(record.get("time", "")),
                        level=record.get("level", "INFO"),
                        message=msg,
                        module=record.get("module", ""),
                    ))
                    if len(entries) >= limit:
                        break
            if len(entries) >= limit:
                break
    except Exception:
        entries.append(LogEntry(
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
            level="INFO",
            message="Log retrieval via in-memory buffer is not available on this runtime. Check log files directly.",
            module="admin",
        ))

    return LogsResponse(logs=entries, total=len(entries))


@router.post(
    "/compress",
    summary="Compress model",
    description="Trigger model compression (quantization). Compresses the currently loaded model or a specified model using the given method.",
    response_model=CompressResponse,
)
async def compress_model(body: CompressRequest):
    """Trigger model compression."""
    coord = _resolve_coordinator()

    model_name = body.model_name or coord.model_name
    model_path = model_name

    compression_mgr = getattr(coord, '_adaptive_compression_mgr', None)
    if compression_mgr is not None and hasattr(compression_mgr, '_run_compression'):
        job_id = f"compress-{model_name}-{int(time.time())}"
        import threading
        thread = threading.Thread(
            target=compression_mgr._run_compression,
            args=(model_name, model_path),
            kwargs={"job": type("Job", (), {"model_name": model_name, "model_path": model_path, "compressed_path": "", "method": body.method, "started_at": time.time()})()},
            name=job_id,
            daemon=True,
        )
        thread.start()
        logger.info(f"Admin: started compression of {model_name} (method={body.method})")
        return CompressResponse(
            status="started",
            job_id=job_id,
            model_name=model_name,
            method=body.method,
            message=f"Compression of '{model_name}' started with method '{body.method}'",
        )

    try:
        from distllm.core.adaptive_compression import SimpleCompressor
        output_dir = body.output_dir or f"/tmp/distllm-compress/{model_name}"
        compressor = SimpleCompressor(
            output_base=output_dir,
            method=body.method,
            trust_remote_code=getattr(coord, 'trust_remote_code', False),
        )
        job_id = f"compress-{model_name}-{int(time.time())}"
        import threading
        thread = threading.Thread(
            target=lambda: compressor.compress(model_name, model_path),
            name=job_id,
            daemon=True,
        )
        thread.start()
        logger.info(f"Admin: started compression of {model_name} (method={body.method})")
        return CompressResponse(
            status="started",
            job_id=job_id,
            model_name=model_name,
            method=body.method,
            message=f"Compression of '{model_name}' started with method '{body.method}'",
        )
    except Exception as e:
        logger.exception("Admin compress failed")
        raise HTTPException(status_code=500, detail=f"Failed to start compression: {e}")


class RegisterNodeRequest(BaseModel):
    """Request to register a new worker node."""
    node_id: str = Field(..., description="Unique node ID")
    host: str = Field(..., description="Node hostname or IP")
    port: int = Field(..., description="gRPC port")
    start_layer: int = Field(..., description="First layer index")
    end_layer: int = Field(..., description="Last layer index")
    total_layers: int = Field(..., description="Total model layers")
    device: str = Field("cpu", description="Device type")
    gpu_name: str = Field("", description="GPU name")
    ready: bool = Field(False, description="Whether the worker has finished loading and is ready to serve")


@router.post(
    "/nodes/register",
    summary="Register a worker node",
    description="Register a new worker node with the coordinator. Called automatically by distllm cluster join.",
)
async def register_node(body: RegisterNodeRequest):
    """Register a worker node.

    If the worker isn't reachable yet (still loading model), the node
    is still registered and the health manager will reconnect later.

    If a node at the same host:port already exists, it is replaced
    (the old node was likely restarted with a new node_id).
    """
    coord = _resolve_coordinator()

    # Remove stale node at the same host:port if it exists
    stale_node_id = None
    for nid, node in coord.nodes.items():
        node_host = node.get("host") if isinstance(node, dict) else getattr(node, "host", None)
        node_port = node.get("port") if isinstance(node, dict) else getattr(node, "port", None)
        if node_host == body.host and node_port == body.port:
            stale_node_id = nid
            break
    if stale_node_id and stale_node_id != body.node_id:
        logger.info(f"Removing stale node {stale_node_id} at {body.host}:{body.port} for re-registration")
        try:
            coord._pipeline.unregister_node(stale_node_id)
        except Exception as e:
            logger.warning(f"Could not remove stale node {stale_node_id}: {e}")

    try:
        coord.manual_register(
            node_id=body.node_id,
            host=body.host,
            port=body.port,
            start_layer=body.start_layer,
            end_layer=body.end_layer,
            total_layers=body.total_layers,
        )
        logger.info(f"Node {body.node_id} registered (host={body.host}, port={body.port}, ready={body.ready})")

        # If the node is not ready yet, mark it as unhealthy so the coordinator
        # won't route requests to it until health probes confirm it's alive
        if not body.ready:
            node = coord._pipeline.get_node(body.node_id)
            if node:
                node.is_healthy = False
                logger.info(f"Node {body.node_id} registered but not ready — marked unhealthy until health check passes")

        return {"status": "registered", "node_id": body.node_id}
    except HTTPException:
        raise
    except Exception as e:
        err_str = str(e)
        if "overlap" in err_str.lower():
            logger.info(f"Node {body.node_id} already registered")
            raise HTTPException(status_code=409, detail=f"Node already registered: {e}")
        if "unreachable" in err_str.lower():
            logger.warning(f"Node {body.node_id} unreachable (worker loading?) — registered as pending")
            return {"status": "registered_pending", "node_id": body.node_id, "warning": "worker not yet reachable"}
        logger.error(f"Failed to register node {body.node_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to register node: {e}")
