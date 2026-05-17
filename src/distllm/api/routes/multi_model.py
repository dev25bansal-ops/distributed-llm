"""Multi-model hot-swap routes: POST/DELETE /v1/models/{model_id}/load."""

from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..api_state import g


router = APIRouter(tags=["multi-model"])


class ModelLoadRequest(BaseModel):
    model_path: str = Field(..., description="Path or HuggingFace ID for the model")
    total_layers: int = Field(default=0, description="Number of transformer layers (0 = auto-detect)")
    memory_budget_gb: float = Field(default=0.0, ge=0, description="GPU memory budget in GB (0 = auto)")


class ModelUnloadRequest(BaseModel):
    model_id: str = Field(..., description="Model identifier to unload")


class ModelInfo(BaseModel):
    name: str
    path: str
    loaded_at: float
    last_used_at: float
    request_count: int
    memory_gb: float
    is_loading: bool


@router.post("/v1/models/{model_id}/load")
async def load_model(model_id: str, body: ModelLoadRequest):
    """Hot-load a model without restarting the server.

    Registers the model and loads it into GPU memory.
    If memory is insufficient, the LRU model is evicted automatically.
    """
    coord = g.coordinator
    if coord is None:
        raise HTTPException(status_code=503, detail="No model loaded")

    msm = getattr(coord, "_model_hotswap", None)
    if msm is None:
        raise HTTPException(status_code=503, detail="Multi-model hot-swap not enabled")

    # Determine total layers
    total_layers = body.total_layers
    if total_layers <= 0:
        try:
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(body.model_path)
            total_layers = getattr(config, "num_hidden_layers", 0)
            if total_layers <= 0:
                raise ValueError("Could not determine num_hidden_layers")
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Could not auto-detect layers for '{body.model_path}': {e}",
            )

    # Register with memory budget
    msm.register_model(
        name=model_id,
        path=body.model_path,
        total_layers=total_layers,
        memory_budget_gb=body.memory_budget_gb if body.memory_budget_gb > 0 else 0.0,
    )

    # Load the model
    if msm.load_model(model_id):
        return {"status": "loaded", "model_id": model_id, "path": body.model_path}
    raise HTTPException(
        status_code=507,
        detail=f"Failed to load model '{model_id}': insufficient memory or loading error",
    )


@router.post("/v1/models/{model_id}/unload")
async def unload_model(model_id: str):
    """Unload a model from GPU memory (keeps it registered)."""
    coord = g.coordinator
    if coord is None:
        raise HTTPException(status_code=503, detail="No model loaded")

    msm = getattr(coord, "_model_hotswap", None)
    if msm is None:
        raise HTTPException(status_code=503, detail="Multi-model hot-swap not enabled")

    if msm.unload_model(model_id):
        return {"status": "unloaded", "model_id": model_id}
    raise HTTPException(status_code=404, detail=f"Model '{model_id}' not loaded")


@router.delete("/v1/models/{model_id}")
async def remove_model(model_id: str):
    """Fully remove a model (unload + unregister)."""
    coord = g.coordinator
    if coord is None:
        raise HTTPException(status_code=503, detail="No model loaded")

    msm = getattr(coord, "_model_hotswap", None)
    if msm is None:
        raise HTTPException(status_code=503, detail="Multi-model hot-swap not enabled")

    if msm.remove_model(model_id):
        return {"status": "removed", "model_id": model_id}
    raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")


@router.get("/v1/models/loaded")
async def list_loaded_models():
    """List all currently loaded models with memory usage."""
    coord = g.coordinator
    if coord is None:
        raise HTTPException(status_code=503, detail="No model loaded")

    msm = getattr(coord, "_model_hotswap", None)
    if msm is None:
        raise HTTPException(status_code=503, detail="Multi-model hot-swap not enabled")

    return {
        "loaded_models": msm.list_loaded_models(),
        "memory": msm.memory_budget.stats(),
        "stats": {
            "total_loads": msm._total_loads,
            "total_unloads": msm._total_unloads,
            "total_evictions": msm._total_evictions,
        },
    }


@router.get("/v1/models/{model_id}/status")
async def get_model_status(model_id: str):
    """Get the status of a specific model (loaded, registered, memory usage)."""
    coord = g.coordinator
    if coord is None:
        raise HTTPException(status_code=503, detail="No model loaded")

    msm = getattr(coord, "_model_hotswap", None)
    if msm is None:
        raise HTTPException(status_code=503, detail="Multi-model hot-swap not enabled")

    instance = msm.get_model(model_id)
    if instance is None:
        # Check if it's registered but not loaded
        entry = msm.registry.get(model_id)
        if entry:
            return {
                "model_id": model_id,
                "status": "registered",
                "path": entry.path,
                "loaded": False,
            }
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")

    return {
        "model_id": model_id,
        "status": "loaded",
        "path": instance.path,
        "loaded": True,
        "loaded_at": instance.loaded_at,
        "last_used_at": instance.last_used_at,
        "request_count": instance.request_count,
        "memory_gb": round(instance.actual_memory_gb, 2),
        "is_loading": instance.is_loading,
    }


@router.post("/v1/models/memory/budget")
async def set_memory_budget(model_id: str, body: dict):
    """Set the GPU memory budget for a model."""
    coord = g.coordinator
    if coord is None:
        raise HTTPException(status_code=503, detail="No model loaded")

    msm = getattr(coord, "_model_hotswap", None)
    if msm is None:
        raise HTTPException(status_code=503, detail="Multi-model hot-swap not enabled")

    budget_gb = body.get("budget_gb")
    if budget_gb is None or budget_gb <= 0:
        raise HTTPException(status_code=400, detail="budget_gb must be positive")

    msm.memory_budget.set_budget(model_id, budget_gb)
    return {"status": "updated", "model_id": model_id, "budget_gb": budget_gb}
