"""Optimization API routes for self-optimizing engine."""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from ..api_state import g

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/optimization", tags=["optimization"])


@router.get(
    "/status",
    summary="Get optimization status",
    description="Return the current status of the self-optimizing engine, including whether it is enabled, running or stopped, and current optimization statistics (latency, throughput, batch size, KV cache settings).",
    response_description="Optimization engine status and stats",
    responses={
        503: {"description": "Coordinator not available"},
    },
)
async def optimization_status():
    """Return self-optimizing engine status and suggestions."""
    coord = g.coordinator
    if coord is None:
        raise HTTPException(status_code=503, detail="Coordinator not available")
    engine = getattr(coord, "_self_optimizing", None)
    if engine is None:
        return {"enabled": False, "status": "not_initialized"}

    stats = engine.stats()
    return {
        "enabled": True,
        "status": "running" if engine._running else "stopped",
        "stats": stats,
    }


@router.get(
    "/suggestions",
    summary="Get optimization suggestions",
    description="Return actionable optimization suggestions from the self-optimizing engine, including recommended batch size, KV cache quantization settings, and speculative decoding parameters. These suggestions can be applied to improve performance.",
    response_description="Optimization suggestions with tunable parameters",
    responses={
        503: {"description": "Coordinator not available or engine not initialized"},
    },
)
async def optimization_suggestions():
    """Return optimization suggestions from the self-optimizing engine."""
    coord = g.coordinator
    if coord is None:
        raise HTTPException(status_code=503, detail="Coordinator not available")
    engine = getattr(coord, "_self_optimizing", None)
    if engine is None:
        raise HTTPException(status_code=503, detail="Self-optimizing engine not initialized")

    suggestions = engine.get_suggestions()
    return {
        "suggestions": suggestions,
        "tunable_params": {
            "batch_size": suggestions.get("batch_size"),
            "kv_cache_quantization": suggestions.get("kv_cache_quantization"),
            "speculative_decoding": suggestions.get("speculative_decoding"),
        },
    }
