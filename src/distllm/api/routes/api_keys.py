"""API Key Management — create, list, revoke API keys.

Endpoints under ``/v1/api-keys`` for programmatic key management.
Requires ``admin`` role.

All key material is stored via ``ApiKeyStore.add_key`` (PBKDF2-HMAC-SHA256
with a fresh random salt — the same format ``authenticate`` verifies
against).  The raw key is returned exactly once, on creation; it is never
logged or persisted by this module.
"""

from __future__ import annotations

import secrets
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Request
from loguru import logger
from pydantic import BaseModel, Field

from distllm.core.api_key_store import get_api_key_store

from ..auth_deps import require_coordinator, require_role

router = APIRouter(
    prefix="/v1/api-keys",
    tags=["api-keys"],
    dependencies=[Depends(require_coordinator), Depends(require_role("admin"))],
)


class CreateAPIKeyRequest(BaseModel):
    label: str = Field(..., min_length=1, max_length=64, description="Human-readable label")
    role: str = Field(default="inference-only", description="Role: admin, user-admin, model-admin, inference-only, read-only, auditor")


class CreateAPIKeyResponse(BaseModel):
    id: str
    label: str
    role: str
    key: str  # Only returned on creation
    created_at: str


class APIKeyObject(BaseModel):
    id: str
    label: str
    role: str
    created_at: str
    last_used_at: str | None = None


class APIKeyListResponse(BaseModel):
    object: str = "list"
    data: list[APIKeyObject]


def _iso(ts: float | None) -> str:
    """Format a unix timestamp as an ISO-8601 UTC string."""
    if ts is None:
        ts = 0.0
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def _created_at_for(store, key_id: str) -> float | None:
    """Best-effort creation timestamp of the newest entry with *key_id*.

    Reads metadata only (list_keys never exposes key material).
    Returns None when the entry cannot be found or has no timestamp.
    """
    created_at: float | None = None
    for entry in store.list_keys():
        if entry["key_id"] == key_id:
            ts = entry.get("created_at")
            if isinstance(ts, (int, float)):
                created_at = float(ts)
    return created_at


@router.post("", response_model=CreateAPIKeyResponse, status_code=201)
async def create_api_key(body: CreateAPIKeyRequest):
    """Create a new API key.

    The raw key is returned exactly once in the response body; only its
    salted PBKDF2 hash is stored, so it cannot be retrieved again.
    """
    store = get_api_key_store()
    raw_key = f"sk-{secrets.token_urlsafe(40)}"
    try:
        # add_key validates the role and hashes with a fresh random salt via
        # the same code path authenticate() verifies against — never build
        # StoredKey entries by hand here.
        key_id = store.add_key(raw_key, role=body.role, label=body.label)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    logger.info(f"API key created: {key_id} (role={body.role}, label={body.label})")
    return CreateAPIKeyResponse(
        id=key_id,
        label=body.label,
        role=body.role,
        key=raw_key,
        created_at=_iso(_created_at_for(store, key_id)),
    )


@router.get("", response_model=APIKeyListResponse)
async def list_api_keys():
    """List all API keys (key values are NOT returned)."""
    store = get_api_key_store()
    keys = store.list_keys()
    return APIKeyListResponse(
        data=[
            APIKeyObject(
                id=k["key_id"],
                label=k["label"],
                role=k["role"],
                created_at=_iso(k.get("created_at")),
            )
            for k in keys
        ],
    )


@router.delete("/{key_id}")
async def revoke_api_key(key_id: str, request: Request):
    """Revoke an API key by ID (every stored entry with that ID).

    Revoking the caller's own key is rejected — it would lock the admin out
    mid-session with no remaining management credential.
    """
    caller_key_id = getattr(request.state, "api_key_id", None)
    if caller_key_id is not None and caller_key_id == key_id:
        raise HTTPException(
            status_code=400,
            detail="Cannot revoke the API key used for this request",
        )

    store = get_api_key_store()
    if not store.remove_key(key_id):
        raise HTTPException(status_code=404, detail=f"API key '{key_id}' not found")
    logger.info(f"API key revoked: {key_id}")
    return {"status": "revoked", "id": key_id}
