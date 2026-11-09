"""Federated training API routes — distributed LoRA training and merging."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..api_state import g
from ..auth_deps import require_role
from ..errors import FeatureDisabledError, error_openapi_entry

# Shorthand used in route ``responses={...}`` dicts so Swagger UI renders the
# concrete error envelope every failure path returns.
ERR_ENTRY = error_openapi_entry


router = APIRouter(tags=["federated"], prefix="/v1/federated")


# ── Request/Response Models ─────────────────────────────────────────────────

class RegisterNodeRequest(BaseModel):
    node_id: str = Field(..., description="Node identifier")
    dataset_size: int = Field(default=0, ge=0, description="Local dataset size")
    local_epochs: int = Field(default=3, ge=1, description="Training epochs per round")
    learning_rate: float = Field(default=2e-4, gt=0, description="Learning rate")


class SubmitAdapterRequest(BaseModel):
    node_id: str = Field(..., description="Submitting node ID")
    adapter_path: str = Field(..., description="Path to trained adapter weights")
    loss: float = Field(..., description="Final training loss")
    dataset_size: int = Field(default=0, ge=0, description="Dataset size used")


class FederatedStatsResponse(BaseModel):
    total_rounds: int
    registered_nodes: int
    active_nodes: int
    total_versions: int
    merge_strategy: str
    current_round: int | None = None
    current_round_status: str | None = None
    avg_loss_last_round: float | None = None


# ── Endpoints ───────────────────────────────────────────────────────────────

@router.post(
    "/nodes",
    dependencies=[Depends(require_role("admin"))],
    summary="Register a node for federated training",
    responses={
        401: ERR_ENTRY("Missing or invalid API key (`Authorization: Bearer <key>`)", type_="auth_error", code="authentication_error"),
        403: ERR_ENTRY("Authenticated but not permitted (requires the admin role)", type_="auth_error", code="forbidden"),
        422: ERR_ENTRY("Invalid registration payload (e.g. non-positive learning rate)", type_="invalid_request_error", code="invalid_request"),
        503: ERR_ENTRY("Federated training not available", type_="service_unavailable", code="503"),
    },
)
async def register_node(req: RegisterNodeRequest):
    """Register a node for federated training."""
    coordinator = g.get("federated_merge")
    if not coordinator:
        raise FeatureDisabledError("federated-training", "Federated training not available")

    state = coordinator.register_node(
        node_id=req.node_id,
        dataset_size=req.dataset_size,
        local_epochs=req.local_epochs,
        learning_rate=req.learning_rate,
    )
    return {
        "node_id": state.node_id,
        "status": state.status,
        "dataset_size": state.dataset_size,
    }


@router.delete(
    "/nodes/{node_id}",
    dependencies=[Depends(require_role("admin"))],
    summary="Remove a node from federated training",
    responses={
        401: ERR_ENTRY("Missing or invalid API key (`Authorization: Bearer <key>`)", type_="auth_error", code="authentication_error"),
        403: ERR_ENTRY("Authenticated but not permitted (requires the admin role)", type_="auth_error", code="forbidden"),
        503: ERR_ENTRY("Federated training not available", type_="service_unavailable", code="503"),
    },
)
async def unregister_node(node_id: str):
    """Remove a node from federated training."""
    coordinator = g.get("federated_merge")
    if not coordinator:
        raise FeatureDisabledError("federated-training", "Federated training not available")

    coordinator.unregister_node(node_id)
    return {"status": "removed", "node_id": node_id}


@router.post(
    "/rounds",
    dependencies=[Depends(require_role("admin"))],
    summary="Start a new federated training round",
    responses={
        400: ERR_ENTRY("Not enough registered nodes to start a round", type_="invalid_request_error", code="invalid_request"),
        401: ERR_ENTRY("Missing or invalid API key (`Authorization: Bearer <key>`)", type_="auth_error", code="authentication_error"),
        403: ERR_ENTRY("Authenticated but not permitted (requires the admin role)", type_="auth_error", code="forbidden"),
        503: ERR_ENTRY("Federated training not available", type_="service_unavailable", code="503"),
    },
)
async def start_round():
    """Start a new federated training round."""
    coordinator = g.get("federated_merge")
    if not coordinator:
        raise FeatureDisabledError("federated-training", "Federated training not available")

    round_info = coordinator.start_round()
    if not round_info:
        raise HTTPException(status_code=400, detail="Not enough nodes to start round")

    return {
        "round_id": round_info.round_id,
        "round_number": round_info.round_number,
        "participating_nodes": round_info.participating_nodes,
        "status": round_info.status,
    }


@router.post(
    "/rounds/submit",
    dependencies=[Depends(require_role("admin"))],
    summary="Submit a locally trained adapter for merging",
    responses={
        400: ERR_ENTRY("Adapter not accepted for the current round", type_="invalid_request_error", code="invalid_request"),
        401: ERR_ENTRY("Missing or invalid API key (`Authorization: Bearer <key>`)", type_="auth_error", code="authentication_error"),
        403: ERR_ENTRY("Authenticated but not permitted (requires the admin role)", type_="auth_error", code="forbidden"),
        422: ERR_ENTRY("Invalid submission payload (e.g. missing adapter_path)", type_="invalid_request_error", code="invalid_request"),
        503: ERR_ENTRY("Federated training not available", type_="service_unavailable", code="503"),
    },
)
async def submit_adapter(req: SubmitAdapterRequest):
    """Submit a locally trained adapter for merging."""
    coordinator = g.get("federated_merge")
    if not coordinator:
        raise FeatureDisabledError("federated-training", "Federated training not available")

    accepted = coordinator.submit_node_adapter(
        node_id=req.node_id,
        adapter_path=req.adapter_path,
        loss=req.loss,
        dataset_size=req.dataset_size,
    )
    if not accepted:
        raise HTTPException(status_code=400, detail="Adapter not accepted")
    return {"status": "submitted", "node_id": req.node_id}


@router.post(
    "/rounds/merge",
    dependencies=[Depends(require_role("admin"))],
    summary="Merge all submitted adapters",
    responses={
        400: ERR_ENTRY("Merge failed (no adapters to merge or incompatible adapters)", type_="invalid_request_error", code="invalid_request"),
        401: ERR_ENTRY("Missing or invalid API key (`Authorization: Bearer <key>`)", type_="auth_error", code="authentication_error"),
        403: ERR_ENTRY("Authenticated but not permitted (requires the admin role)", type_="auth_error", code="forbidden"),
        503: ERR_ENTRY("Federated training not available", type_="service_unavailable", code="503"),
    },
)
async def merge_adapters():
    """Merge all submitted adapters."""
    coordinator = g.get("federated_merge")
    if not coordinator:
        raise FeatureDisabledError("federated-training", "Federated training not available")

    merged_path = coordinator.merge_adapters()
    if not merged_path:
        raise HTTPException(status_code=400, detail="Merge failed")
    return {"status": "merged", "path": merged_path}


@router.get(
    "/stats",
    response_model=FederatedStatsResponse,
    summary="Get federated training statistics",
    responses={
        401: ERR_ENTRY("Missing or invalid API key (`Authorization: Bearer <key>`)", type_="auth_error", code="authentication_error"),
        503: ERR_ENTRY("Federated training not available", type_="service_unavailable", code="503"),
    },
)
async def get_federated_stats():
    """Get federated training statistics."""
    coordinator = g.get("federated_merge")
    if not coordinator:
        raise FeatureDisabledError("federated-training", "Federated training not available")

    stats = coordinator.get_stats()
    return FederatedStatsResponse(**stats)


@router.get(
    "/versions",
    summary="List adapter versions from federated training",
    responses={
        401: ERR_ENTRY("Missing or invalid API key (`Authorization: Bearer <key>`)", type_="auth_error", code="authentication_error"),
        503: ERR_ENTRY("Federated training not available", type_="service_unavailable", code="503"),
    },
)
async def list_versions():
    """List adapter versions from federated training."""
    coordinator = g.get("federated_merge")
    if not coordinator:
        raise FeatureDisabledError("federated-training", "Federated training not available")

    versions = coordinator.get_versions()
    return [
        {
            "version_id": v.version_id,
            "round_number": v.round_number,
            "path": v.path,
            "metrics": v.metrics,
            "created_at": v.created_at,
        }
        for v in versions
    ]
