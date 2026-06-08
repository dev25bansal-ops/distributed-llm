"""Debug and replay routes: GET /v1/debug/recent, POST /v1/debug/replay."""

import hmac
import os

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from pydantic import BaseModel, Field

from ..api_state import g

router = APIRouter(tags=["debug"])


def _verify_debug_access(request: Request, authorization: str = Header("")) -> None:
    enabled = os.environ.get("DISTLLM_ENABLE_DEBUG_ROUTES") == "1"
    if not enabled:
        raise HTTPException(status_code=404, detail="Not found")

    admin_key = os.environ.get("DISTLLM_ADMIN_KEY")
    if not admin_key:
        raise HTTPException(status_code=503, detail="Debug admin API not configured")
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing authorization")
    if not hmac.compare_digest(authorization[7:], admin_key):
        raise HTTPException(status_code=401, detail="Invalid admin key")


class ReplayRequest(BaseModel):
    request_id: str = Field(..., description="ID of the request to replay")


class DeterministicModeRequest(BaseModel):
    enabled: bool = Field(..., description="Enable or disable deterministic mode")
    seed: int = Field(default=42, description="Random seed for deterministic generation")


@router.get(
    "/v1/debug/recent",
    summary="Get recent requests",
    description="Retrieve the N most recent requests stored in the replay buffer for debugging and analysis. Returns truncated prompts, timestamps, duration, model, and error information.",
    response_description="List of recent request entries",
    responses={
        503: {"description": "No coordinator available or replay buffer not available"},
    },
)
async def get_recent_requests(n: int = 10, _admin=Depends(_verify_debug_access)):  # noqa: B008
    """Get the N most recent requests stored in the replay buffer."""
    coord = g.coordinator
    if coord is None:
        raise HTTPException(status_code=503, detail="No coordinator available")
    if not hasattr(coord, '_replay_buffer'):
        raise HTTPException(status_code=503, detail="Replay buffer not available")

    requests = coord.get_recent_requests(n)
    return {
        "requests": [
            {
                "request_id": r.request_id,
                "prompt": r.prompt[:200] + "..." if len(r.prompt) > 200 else r.prompt,
                "timestamp": r.timestamp,
                "duration_ms": r.duration_ms,
                "model": r.model,
                "error": r.error,
                "replay_count": r.replay_count,
            }
            for r in requests
        ],
        "total": len(requests),
    }


@router.get(
    "/v1/debug/request/{request_id}",
    summary="Get request details",
    description="Get full details of a stored request from the replay buffer, including the complete prompt, generation parameters, response text, error info, duration, and replay count.",
    response_description="Full request details including prompt, params, and response",
    responses={
        404: {"description": "Request ID not found"},
        503: {"description": "No coordinator available"},
    },
)
async def get_request_detail(request_id: str, _admin=Depends(_verify_debug_access)):  # noqa: B008
    """Get full details of a stored request."""
    coord = g.coordinator
    if coord is None:
        raise HTTPException(status_code=503, detail="No coordinator available")

    entry = coord._replay_buffer.get(request_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Request {request_id} not found")

    return {
        "request_id": entry.request_id,
        "prompt": entry.prompt,
        "params": entry.params,
        "response": entry.response,
        "error": entry.error,
        "duration_ms": entry.duration_ms,
        "timestamp": entry.timestamp,
        "replay_count": entry.replay_count,
        "model": entry.model,
    }


@router.post(
    "/v1/debug/replay",
    summary="Replay request",
    description="Replay a previously stored request through the model with the same prompt and parameters. Useful for debugging non-deterministic behavior and reproducing issues.",
    response_description="Replayed response text",
    responses={
        404: {"description": "Request ID not found or replay failed"},
        503: {"description": "No coordinator available"},
    },
)
async def replay_request(body: ReplayRequest, _admin=Depends(_verify_debug_access)):  # noqa: B008
    """Replay a stored request through the model."""
    coord = g.coordinator
    if coord is None:
        raise HTTPException(status_code=503, detail="No coordinator available")

    response = coord.replay_request(body.request_id)
    if response is None:
        raise HTTPException(status_code=404, detail=f"Request {body.request_id} not found or replay failed")

    return {"request_id": body.request_id, "response": response}


@router.post(
    "/v1/debug/deterministic",
    summary="Set deterministic mode",
    description="Enable or disable deterministic generation mode for debugging. When enabled, all generations use a fixed seed for reproducible outputs, facilitating debugging and testing.",
    response_description="Deterministic mode status with seed",
    responses={
        503: {"description": "No coordinator available"},
    },
)
async def set_deterministic_mode(body: DeterministicModeRequest, _admin=Depends(_verify_debug_access)):  # noqa: B008
    """Enable or disable deterministic debug mode."""
    coord = g.coordinator
    if coord is None:
        raise HTTPException(status_code=503, detail="No coordinator available")

    coord.set_deterministic_mode(enabled=body.enabled, seed=body.seed)
    return {"status": "enabled" if body.enabled else "disabled", "seed": body.seed}


@router.get(
    "/v1/debug/buffer/export",
    summary="Export replay buffer",
    description="Export recent requests from the replay buffer for external analysis and debugging. Returns up to max_entries (default 50, max 200) buffer contents as JSON.",
    response_description="Replay buffer contents with entry count",
    responses={
        503: {"description": "No coordinator available"},
    },
)
async def export_replay_buffer(
    max_entries: int = 50,
    _admin=Depends(_verify_debug_access),  # noqa: B008
):
    """Export recent requests from the replay buffer for external debugging."""
    coord = g.coordinator
    if coord is None:
        raise HTTPException(status_code=503, detail="No coordinator available")

    clamped = max(1, min(max_entries, 200))
    all_entries = coord._replay_buffer.export()
    entries = all_entries[-clamped:]
    return {"entries": _truncate_sensitive_fields(entries), "count": len(entries)}


def _truncate_sensitive_fields(entries: list) -> list:
    """Truncate sensitive fields (prompts, responses) in exported entries.

    Prevents accidental leakage of full prompt/response data through the
    debug export endpoint.
    """
    result = []
    for entry in entries:
        safe_entry = dict(entry)
        if "prompt" in safe_entry and isinstance(safe_entry["prompt"], str) and len(safe_entry["prompt"]) > 200:
            safe_entry["prompt"] = safe_entry["prompt"][:200] + "..."
        if "response" in safe_entry and isinstance(safe_entry["response"], str) and len(safe_entry["response"]) > 500:
            safe_entry["response"] = safe_entry["response"][:500] + "..."
        result.append(safe_entry)
    return result
