"""Disaggregated serving API routes for prefill/decode separation."""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..api_state import g
from distllm.core.disagg_serving import PoolStatus

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/disagg", tags=["disagg"])


def _get_coordinator():
    """Get the coordinator instance from the app state."""
    coord = g.coordinator
    if coord is None:
        raise HTTPException(status_code=503, detail="Coordinator not available")
    return coord


class DisaggGenerateRequest(BaseModel):
    prompt_tokens: list[int]
    max_new_tokens: int = 128


class DisaggGenerateResponse(BaseModel):
    request_id: str
    status: str


class DisaggResultResponse(BaseModel):
    request_id: str
    tokens: list[int] | None
    status: str


class AddNodeRequest(BaseModel):
    node_id: str
    host: str
    port: int
    capacity: int = 1


class DisaggHealthResponse(BaseModel):
    status: str
    prefill_nodes: int
    decode_nodes: int
    nodes: list[dict]


@router.post(
    "/generate",
    response_model=DisaggGenerateResponse,
    summary="Submit disaggregated generation",
    description="Submit a token generation request through the disaggregated orchestrator, which separates prefill and decode phases across independent node pools for improved throughput and resource utilization.",
    response_description="Submission confirmation with request ID",
    responses={
        503: {"description": "Coordinator not available or orchestrator not initialized"},
    },
)
async def disagg_generate(request: DisaggGenerateRequest):
    """Submit a generation request through the disaggregated orchestrator."""
    coord = _get_coordinator()
    orchestrator = getattr(coord, "_disagg_orchestrator", None)
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="Disaggregated orchestrator not initialized")

    request_id = await orchestrator.submit(
        prompt_tokens=request.prompt_tokens,
        max_new_tokens=request.max_new_tokens,
    )
    return DisaggGenerateResponse(
        request_id=request_id,
        status="submitted",
    )


@router.get(
    "/result/{request_id}",
    response_model=DisaggResultResponse,
    summary="Get disaggregated result",
    description="Retrieve the result of a previously submitted disaggregated generation request by its ID. Returns generated tokens once processing is complete.",
    response_description="Generated tokens with status",
    responses={
        404: {"description": "Request ID not found"},
        503: {"description": "Coordinator not available or orchestrator not initialized"},
    },
)
async def disagg_result(request_id: str):
    """Get the result of a disaggregated generation request."""
    coord = _get_coordinator()
    orchestrator = getattr(coord, "_disagg_orchestrator", None)
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="Disaggregated orchestrator not initialized")

    tokens = await orchestrator.get_result(request_id)
    if tokens is None:
        raise HTTPException(status_code=404, detail=f"Request {request_id} not found")

    return DisaggResultResponse(
        request_id=request_id,
        tokens=list(tokens) if tokens else [],
        status="completed",
    )


@router.post(
    "/nodes/prefill",
    summary="Add prefill node",
    description="Register a new prefill node with the disaggregated serving router. Prefill nodes handle prompt processing and KV cache computation in the separated prefill/decode architecture.",
    response_description="Node addition confirmation with role",
    responses={
        503: {"description": "Coordinator not available or orchestrator not initialized"},
    },
)
async def disagg_add_prefill(request: AddNodeRequest):
    """Add a prefill node to the disaggregated router."""
    coord = _get_coordinator()
    orchestrator = getattr(coord, "_disagg_orchestrator", None)
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="Disaggregated orchestrator not initialized")

    await orchestrator.router.add_prefill_node(
        node_id=request.node_id,
        host=request.host,
        port=request.port,
        capacity=request.capacity,
    )
    return {"status": "added", "node_id": request.node_id, "role": "prefill"}


@router.post(
    "/nodes/decode",
    summary="Add decode node",
    description="Register a new decode node with the disaggregated serving router. Decode nodes handle autoregressive token generation in the separated prefill/decode architecture.",
    response_description="Node addition confirmation with role",
    responses={
        503: {"description": "Coordinator not available or orchestrator not initialized"},
    },
)
async def disagg_add_decode(request: AddNodeRequest):
    """Add a decode node to the disaggregated router."""
    coord = _get_coordinator()
    orchestrator = getattr(coord, "_disagg_orchestrator", None)
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="Disaggregated orchestrator not initialized")

    await orchestrator.router.add_decode_node(
        node_id=request.node_id,
        host=request.host,
        port=request.port,
        capacity=request.capacity,
    )
    return {"status": "added", "node_id": request.node_id, "role": "decode"}


@router.get(
    "/health",
    response_model=DisaggHealthResponse,
    summary="Disaggregated health check",
    description="Return health status of all disaggregated serving nodes, including prefill and decode pool active node counts and per-node health status.",
    response_description="Disaggregated serving health status",
    responses={
        503: {"description": "Coordinator not available or orchestrator not initialized"},
    },
)
async def disagg_health():
    """Return health status of disaggregated nodes."""
    coord = _get_coordinator()
    orchestrator = getattr(coord, "_disagg_orchestrator", None)
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="Disaggregated orchestrator not initialized")

    health = orchestrator.health_check()

    nodes = []
    for n in orchestrator.router.prefill_pool._nodes.values():
        nodes.append({"node_id": n.node_id, "role": "prefill", "healthy": n.status == PoolStatus.ACTIVE})
    for n in orchestrator.router.decode_pool._nodes.values():
        nodes.append({"node_id": n.node_id, "role": "decode", "healthy": n.status == PoolStatus.ACTIVE})

    return DisaggHealthResponse(
        status="healthy" if health["healthy"] else "degraded",
        prefill_nodes=health["prefill_pool"]["active_nodes"],
        decode_nodes=health["decode_pool"]["active_nodes"],
        nodes=nodes,
    )
