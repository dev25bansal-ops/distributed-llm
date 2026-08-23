"""GPU memory defragmentation API routes."""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from ..api_state import g
from ..auth_deps import require_role


router = APIRouter(prefix="/v1/defrag", tags=["defrag"])


class DefragResponse(BaseModel):
    enabled: bool
    policy: str | None = None
    fragmentation_ratio: float | None = None
    predictive_fragmentation: float | None = None
    stats: dict | None = None
    config: dict | None = None


class DefragRunResponse(BaseModel):
    """Per-backend defragmentation results."""


@router.get(
    "/status",
    summary="Get defragmentation status",
    description="Returns current fragmentation ratio, policy, and statistics.",
)
async def defrag_status():
    coord = g.coordinator
    if coord is None:
        raise HTTPException(status_code=503, detail="Coordinator not available")
    if hasattr(coord, "defrag_status"):
        return coord.defrag_status()
    return DefragResponse(enabled=False)


@router.post(
    "/run",
    summary="Trigger defragmentation pass",
    description="Runs an immediate defragmentation on all KV cache backends. Restricted to admin users.",
    dependencies=[Depends(require_role("admin"))],
)
async def defrag_run():
    coord = g.coordinator
    if coord is None:
        raise HTTPException(status_code=503, detail="Coordinator not available")
    if hasattr(coord, "defrag_run_now"):
        return coord.defrag_run_now()
    return {"error": "Defragmenter not available"}


@router.get(
    "/stats",
    summary="Get defragmentation statistics",
    description="Historical statistics and fragmentation history.",
)
async def defrag_stats():
    coord = g.coordinator
    if coord is None:
        raise HTTPException(status_code=503, detail="Coordinator not available")
    if hasattr(coord, "defrag_stats"):
        return coord.defrag_stats()
    return {"enabled": False}
