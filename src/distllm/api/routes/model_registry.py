"""Model Registry API routes — aggregated view of loaded models, versions, and cache."""

import threading
import time

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from ..api_state import g
from ..auth_deps import require_role

router = APIRouter(tags=["models"])


class LoadModelRequest(BaseModel):
    model: str
    dtype: str = "float16"


class UnloadModelRequest(BaseModel):
    model: str


def _build_registry():
    """Build the full model registry response from available data sources."""
    coord = g.coordinator
    models_list = []
    total_gpu_memory = 0.0
    used_gpu_memory = 0.0
    cluster_nodes = 0

    # ── Coordinator data ──
    loaded_models: dict[str, dict] = {}
    if coord is not None:
        cluster_nodes = len(getattr(coord, "nodes", {}) or {})

        # Aggregate GPU memory from nodes
        for node in getattr(coord, "nodes", {}).values():
            mem_total = getattr(node, "gpu_memory_total", 0) or 0
            mem_free = getattr(node, "gpu_memory_free", 0) or 0
            total_gpu_memory += mem_total / (1024 * 1024)
            used_gpu_memory += (mem_total - mem_free) / (1024 * 1024)

        # Models from coordinator
        model_names = []
        try:
            if hasattr(coord, "list_models"):
                model_names = coord.list_models() or []
        except Exception:
            model_names = [coord.model_name] if getattr(coord, "model_name", None) else []

        for m_name in model_names:
            partition = None
            if hasattr(coord, "nodes") and coord.nodes:
                nodes_sorted = sorted(coord.nodes.values(), key=lambda n: getattr(n, "start_layer", 0) or 0)
                if nodes_sorted:
                    partition = {
                        "start_layer": getattr(nodes_sorted[0], "start_layer", 0),
                        "end_layer": getattr(nodes_sorted[-1], "end_layer", 0),
                    }

            gpu_name = None
            gpu_free = None
            gpu_total = None
            if hasattr(coord, "nodes") and coord.nodes:
                first_node = next(iter(coord.nodes.values()), None)
                if first_node:
                    gpu_name = getattr(first_node, "gpu_name", None) or None
                    gpu_free = getattr(first_node, "gpu_memory_free", None)
                    gpu_total = getattr(first_node, "gpu_memory_total", None)

            loaded_models[m_name] = {
                "loaded": True,
                "dtype": getattr(coord, "dtype", "float16") or "float16",
                "memory_used_mb": used_gpu_memory,
                "memory_peak_mb": used_gpu_memory,
                "partition": partition,
                "gpu_name": gpu_name,
                "gpu_memory_free": gpu_free,
                "gpu_memory_total": gpu_total,
            }

    # ── Version manager data ──
    version_data: dict[str, dict] = {}
    try:
        from distllm.core.model_version_manager import ModelVersionManager

        version_mgr: ModelVersionManager | None = getattr(g, "_version_manager", None)
        if version_mgr is None and coord is not None:
            version_mgr = getattr(coord, "_version_manager", None) or getattr(coord, "version_manager", None)
        if version_mgr is not None:
            for ver in version_mgr.list_versions():
                version_data[ver.model_name] = {
                    "version_id": ver.version_id,
                    "status": ver.status.value if hasattr(ver.status, "value") else str(ver.status),
                    "deployed_at": ver.deployed_at,
                }
    except ImportError:
        pass
    except Exception:
        pass

    # ── Model Hub cached data ──
    cached_data: dict[str, dict] = {}
    try:
        from distllm.models.model_hub import ModelHub

        hub: ModelHub | None = getattr(g, "_model_hub", None)
        if hub is None and coord is not None:
            hub = getattr(coord, "_model_hub", None) or getattr(coord, "model_hub", None)
        if hub is not None:
            for cached in hub.list_cached():
                cached_data[cached.model_id] = {
                    "size_bytes": cached.size_bytes,
                    "downloaded_at": cached.downloaded_at,
                }
    except ImportError:
        pass
    except Exception:
        pass

    # ── Merge into final model list ──
    all_ids = set(loaded_models.keys()) | set(version_data.keys()) | set(cached_data.keys())

    for mid in all_ids:
        lm = loaded_models.get(mid, {})
        vd = version_data.get(mid)
        cd = cached_data.get(mid)

        models_list.append({
            "id": mid,
            "loaded": lm.get("loaded", False),
            "dtype": lm.get("dtype", ""),
            "memory_used_mb": lm.get("memory_used_mb", 0.0),
            "memory_peak_mb": lm.get("memory_peak_mb", 0.0),
            "partition": lm.get("partition"),
            "version": vd,
            "cached": cd,
            "gpu_name": lm.get("gpu_name"),
            "gpu_memory_free": lm.get("gpu_memory_free"),
            "gpu_memory_total": lm.get("gpu_memory_total"),
            "created": int(time.time()),
        })

    total_loaded = sum(1 for m in models_list if m["loaded"])
    total_cached = len(cached_data)

    return {
        "models": models_list,
        "cluster_nodes": cluster_nodes,
        "total_gpu_memory_mb": total_gpu_memory,
        "used_gpu_memory_mb": used_gpu_memory,
        "models_cached": total_cached,
        "models_loaded_count": total_loaded,
    }


_registry_cache: dict | None = None
_cache_timestamp: float = 0
_cache_lock = threading.Lock()
_CACHE_TTL = 2.0


def _get_registry(force: bool = False) -> dict:
    global _registry_cache, _cache_timestamp
    now = time.time()
    if force or _registry_cache is None or (now - _cache_timestamp) > _CACHE_TTL:
        with _cache_lock:
            if force or _registry_cache is None or (now - _cache_timestamp) > _CACHE_TTL:
                _registry_cache = _build_registry()
                _cache_timestamp = now
    return _registry_cache


@router.get(
    "/api/models/registry",
    summary="Model Registry",
    description="Return detailed model registry info combining loaded models, versions, and cached models.",
)
async def get_model_registry():
    """Return the full model registry."""
    return _get_registry()


@router.get(
    "/api/models/registry/{model_id}",
    summary="Model Registry Detail",
    description="Return detailed info for a single model from the registry.",
)
async def get_model_registry_detail(model_id: str):
    """Return registry detail for a single model."""
    registry = _get_registry()
    for m in registry["models"]:
        if m["id"] == model_id:
            return m
    return {"error": "model not found", "id": model_id}


@router.post(
    "/api/models/registry/reload",
    summary="Reload Registry",
    description="Force a refresh of the model registry data.",
)
async def reload_model_registry():
    """Force-refresh the registry cache."""
    data = _get_registry(force=True)
    return {"status": "ok", "models": len(data["models"])}


@router.post(
    "/api/models/load",
    summary="Load Model",
    description="Load a model onto the coordinator.",
    dependencies=[Depends(require_role("model-admin"))],
)
async def load_model(body: LoadModelRequest):
    """Load a model via the coordinator."""
    coord = g.coordinator
    if coord is None:
        return {"error": "coordinator not available"}
    try:
        if hasattr(coord, "load_model"):
            result = coord.load_model(body.model, body.dtype)
        elif hasattr(coord, "hot_swap_load"):
            result = coord.hot_swap_load(body.model)
        else:
            return {"error": "no load_model method available"}
        _get_registry(force=True)
        return {"status": "ok", "model": body.model, "result": result}
    except Exception as e:
        return {"error": str(e)}


@router.post(
    "/api/models/unload",
    summary="Unload Model",
    description="Unload a model from the coordinator.",
    dependencies=[Depends(require_role("model-admin"))],
)
async def unload_model(body: UnloadModelRequest):
    """Unload a model via the coordinator."""
    coord = g.coordinator
    if coord is None:
        return {"error": "coordinator not available"}
    try:
        if hasattr(coord, "unload_model"):
            result = coord.unload_model(body.model)
        elif hasattr(coord, "hot_swap_unload"):
            result = coord.hot_swap_unload(body.model)
        else:
            return {"error": "no unload_model method available"}
        _get_registry(force=True)
        return {"status": "ok", "model": body.model, "result": result}
    except Exception as e:
        return {"error": str(e)}
