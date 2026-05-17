"""LoRA adapter management routes: GET/POST /v1/adapters."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..api_state import g
from ..validation import validate_adapter_path


router = APIRouter(tags=["adapters"])


class AdapterLoadRequest(BaseModel):
    action: str = Field(..., description="Action: 'load', 'set', 'list', 'warmup', 'unload', 'rank'")
    id: str | None = None
    path: str | None = None
    adapters: dict[str, str] | None = None  # For warmup: {id: path}
    rank: int | None = None  # For load/rank: priority rank
    tenant_id: str | None = None  # For load: tenant identifier


class AdapterInfo(BaseModel):
    adapter_id: str
    path: str
    use_count: int = 0
    rank: int = 0
    tenant_id: str = ""
    vram_bytes: int = 0


class AdapterListResponse(BaseModel):
    active: str | None = None
    adapters: list[str] = []
    stats: dict = {}
    ranking: list[dict] | None = None


def _check_adapter_enabled():
    """Raise HTTPException if adapter manager is not available."""
    coord = g.coordinator
    if coord is None or not hasattr(coord, 'adapter_manager') or coord.adapter_manager is None:
        raise HTTPException(status_code=503, detail="LoRA not enabled")
    return coord.adapter_manager


@router.get("/v1/adapters")
async def list_adapters():
    """List loaded LoRA adapters with stats and ranking."""
    adapter_mgr = _check_adapter_enabled()

    # Get ranking
    ranking = None
    try:
        ranked = adapter_mgr.rank_adapters()
        ranking = [
            {
                "adapter_id": a.adapter_id,
                "rank": a.rank,
                "use_count": a.use_count,
                "tenant_id": a.tenant_id,
            }
            for a in ranked
        ]
    except Exception:
        pass

    return AdapterListResponse(
        active=adapter_mgr.active_adapter,
        adapters=adapter_mgr.list_adapters(),
        stats=adapter_mgr.get_stats(),
        ranking=ranking,
    )


@router.post("/v1/adapters")
async def manage_adapters(request: AdapterLoadRequest):
    """Manage LoRA adapters: load, set, list, warmup, unload, rank."""
    adapter_mgr = _check_adapter_enabled()

    if request.action == "load":
        if not request.id or not request.path:
            raise HTTPException(status_code=400, detail="id and path required for load action")

        # Security: Validate adapter path to prevent path traversal
        try:
            validate_adapter_path(request.path)
        except ValueError as e:
            raise HTTPException(status_code=403, detail=str(e))

        adapter_mgr.load_adapter(
            request.id,
            request.path,
            rank=request.rank or 0,
            tenant_id=request.tenant_id or "",
        )
        return {"status": "loaded", "id": request.id}

    elif request.action == "warmup":
        """Pre-load multiple adapters before traffic arrives."""
        if not request.adapters:
            raise HTTPException(status_code=400, detail="adapters dict required for warmup")

        rank_map = {}
        tenant_map = {}
        for adapter_id in request.adapters:
            if request.rank is not None:
                rank_map[adapter_id] = request.rank
            if request.tenant_id:
                tenant_map[adapter_id] = request.tenant_id

        loaded = adapter_mgr.warmup_adapters(
            request.adapters,
            rank_map=rank_map,
            tenant_map=tenant_map,
        )
        return {"status": "warmed", "loaded": loaded}

    elif request.action == "set":
        if not request.id:
            raise HTTPException(status_code=400, detail="id required for set action")
        adapter_mgr.set_active(request.id)
        return {"status": "active", "id": request.id}

    elif request.action == "unload":
        if not request.id:
            raise HTTPException(status_code=400, detail="id required for unload action")
        success = adapter_mgr.unload_adapter(request.id)
        if not success:
            raise HTTPException(status_code=404, detail=f"Adapter '{request.id}' not found")
        return {"status": "unloaded", "id": request.id}

    elif request.action == "rank":
        """Update adapter ranking for multi-tenant priority."""
        if not request.id or request.rank is None:
            raise HTTPException(status_code=400, detail="id and rank required for rank action")
        info = adapter_mgr.get_adapter_info(request.id)
        if info is None:
            raise HTTPException(status_code=404, detail=f"Adapter '{request.id}' not found")
        info.rank = request.rank
        return {"status": "ranked", "id": request.id, "rank": request.rank}

    elif request.action == "list":
        ranked = adapter_mgr.rank_adapters()
        return {
            "active": adapter_mgr.active_adapter,
            "adapters": adapter_mgr.list_adapters(),
            "stats": adapter_mgr.get_stats(),
            "ranking": [
                {"adapter_id": a.adapter_id, "rank": a.rank, "use_count": a.use_count}
                for a in ranked
            ],
        }
    else:
        raise HTTPException(status_code=400, detail=f"Unknown action: {request.action}")
