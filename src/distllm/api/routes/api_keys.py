"""API Key Management — create, list, revoke API keys.

Endpoints under ``/v1/api-keys`` for programmatic key management.
Requires ``admin`` role.
"""

from __future__ import annotations

import secrets
import time
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
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


@router.post("", response_model=CreateAPIKeyResponse, status_code=201)
async def create_api_key(body: CreateAPIKeyRequest):
    """Create a new API key."""
    store = get_api_key_store()
    raw_key = f"sk-{secrets.token_urlsafe(40)}"
    key_id = f"key-{secrets.token_hex(8)}"

    # Use the key store's internal _load mechanism by re-reading API_KEYS
    # Not ideal — see below for the direct approach
    import hashlib

    from distllm.core.api_key_store import StoredKey

    new_key = StoredKey(
        key=hashlib.sha256(raw_key.encode()).hexdigest(),
        role=body.role,
        label=body.label,
        key_id=key_id,
        created_at=time.time(),
    )
    store._keys.append(new_key)

    logger.info(f"API key created: {key_id} (role={body.role}, label={body.label})")
    return CreateAPIKeyResponse(
        id=key_id,
        label=body.label,
        role=body.role,
        key=raw_key,
        created_at=datetime.utcnow().isoformat(),
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
                created_at=datetime.utcfromtimestamp(k.get("created_at", 0)).isoformat()
                if isinstance(k.get("created_at"), (int, float))
                else datetime.utcnow().isoformat(),
            )
            for k in keys
        ],
    )


@router.delete("/{key_id}")
async def revoke_api_key(key_id: str):
    """Revoke an API key by ID."""
    store = get_api_key_store()
    removed = [k for k in store._keys if k.key_id == key_id]
    if not removed:
        raise HTTPException(status_code=404, detail=f"API key '{key_id}' not found")
    store._keys[:] = [k for k in store._keys if k.key_id != key_id]
    logger.info(f"API key revoked: {key_id} (label={removed[0].label})")
    return {"status": "revoked", "id": key_id}
