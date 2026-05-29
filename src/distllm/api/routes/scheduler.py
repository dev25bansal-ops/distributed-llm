"""Scheduler tuning API — live configuration and monitoring.

Provides endpoints for live scheduler parameter tuning and
real-time scheduling metrics.
"""

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from ..auth_deps import require_role
from ..api_state import g
from loguru import logger


router = APIRouter(
    prefix="/v1/scheduler",
    tags=["scheduler"],
    dependencies=[Depends(require_role("admin"))],
)


# ── Request/Response Models ──────────────────────────────────────────────

class SchedulerConfigUpdate(BaseModel):
    """Live scheduler configuration update."""
    max_batch_size: int | None = Field(default=None, ge=1, le=512)
    max_tokens_per_batch: int | None = Field(default=None, ge=256, le=1048576)
    prefill_slack_ratio: float | None = Field(default=None, ge=0.0, le=1.0)
    aging_interval_s: float | None = Field(default=None, ge=1.0, le=600.0)
    aging_max_boost: int | None = Field(default=None, ge=0, le=10)
    aging_enabled: bool | None = None
    max_preempted: int | None = Field(default=None, ge=0, le=64)
    starvation_threshold_s: float | None = Field(default=None, ge=10.0, le=3600.0)


class SchedulerStatsResponse(BaseModel):
    """Scheduler statistics response."""
    active_requests: int = 0
    pending_requests: int = 0
    preempted_requests: int = 0
    max_batch_size: int = 0
    max_tokens_per_batch: int = 0
    iteration: int = 0
    total_prefill_tokens: int = 0
    total_decode_tokens: int = 0
    chunked_prefill_active: int = 0
    adaptive_batching: bool = False
    advanced: dict = Field(default_factory=dict)


# ── Endpoints ────────────────────────────────────────────────────────────

@router.get("/stats", response_model=SchedulerStatsResponse)
async def get_scheduler_stats(request: Request):
    """Get current scheduler statistics.

    Returns active/pending/preempted counts, token budgets,
    iteration count, and advanced scheduling feature stats.
    """
    coord = g.coordinator
    if coord is None:
        raise HTTPException(status_code=503, detail="Coordinator not initialized")

    scheduler = getattr(coord, '_batch_scheduler', None)
    if scheduler is None:
        raise HTTPException(status_code=503, detail="Batch scheduler not configured")

    try:
        stats = scheduler.stats()
        return SchedulerStatsResponse(**stats)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/config")
async def update_scheduler_config(
    update: SchedulerConfigUpdate,
    request: Request,
):
    """Live-update scheduler configuration.

    Only updates fields that are explicitly provided (non-None).
    All changes take effect on the next scheduling iteration.

    Example::

        PATCH /v1/scheduler/config
        {"max_batch_size": 64, "aging_enabled": false}
    """
    coord = g.coordinator
    if coord is None:
        raise HTTPException(status_code=503, detail="Coordinator not initialized")

    scheduler = getattr(coord, '_batch_scheduler', None)
    if scheduler is None:
        raise HTTPException(status_code=503, detail="Batch scheduler not configured")

    updated = {}
    try:
        with scheduler._lock:
            if update.max_batch_size is not None:
                scheduler.max_batch_size = update.max_batch_size
                scheduler._budget.max_batch_size = update.max_batch_size
                updated["max_batch_size"] = update.max_batch_size

            if update.max_tokens_per_batch is not None:
                scheduler.max_tokens_per_batch = update.max_tokens_per_batch
                scheduler._budget.max_total_tokens = update.max_tokens_per_batch
                updated["max_tokens_per_batch"] = update.max_tokens_per_batch

            if update.prefill_slack_ratio is not None:
                scheduler._budget.prefill_slack_ratio = update.prefill_slack_ratio
                updated["prefill_slack_ratio"] = update.prefill_slack_ratio

            if update.aging_interval_s is not None:
                scheduler._aging_interval_s = update.aging_interval_s
                updated["aging_interval_s"] = update.aging_interval_s

            if update.aging_max_boost is not None:
                scheduler._aging_max_boost = update.aging_max_boost
                updated["aging_max_boost"] = update.aging_max_boost

            if update.aging_enabled is not None:
                scheduler._aging_enabled = update.aging_enabled
                updated["aging_enabled"] = update.aging_enabled

            if update.max_preempted is not None:
                scheduler._max_preempted = update.max_preempted
                updated["max_preempted"] = update.max_preempted

            if update.starvation_threshold_s is not None:
                scheduler._starvation_threshold_s = update.starvation_threshold_s
                updated["starvation_threshold_s"] = update.starvation_threshold_s

        logger.info(f"Scheduler config updated: {updated}")
        return {"status": "ok", "updated": updated}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/config")
async def get_scheduler_config(request: Request):
    """Get current scheduler configuration.

    Returns all configurable parameters and their current values.
    """
    coord = g.coordinator
    if coord is None:
        raise HTTPException(status_code=503, detail="Coordinator not initialized")

    scheduler = getattr(coord, '_batch_scheduler', None)
    if scheduler is None:
        raise HTTPException(status_code=503, detail="Batch scheduler not configured")

    return {
        "max_batch_size": scheduler.max_batch_size,
        "max_tokens_per_batch": scheduler.max_tokens_per_batch,
        "prefill_slack_ratio": scheduler._budget.prefill_slack_ratio,
        "aging_interval_s": scheduler._aging_interval_s,
        "aging_max_boost": scheduler._aging_max_boost,
        "aging_enabled": scheduler._aging_enabled,
        "max_preempted": scheduler._max_preempted,
        "starvation_threshold_s": scheduler._starvation_threshold_s,
        "enable_chunked_prefill": scheduler._enable_chunked_prefill,
        "adapt_prefill_budget": scheduler._adapt_prefill_budget,
        "priority_weights": scheduler._priority_weights,
    }
