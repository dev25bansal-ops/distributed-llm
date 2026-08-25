"""Webhook registration and management API.

Endpoints::

    POST   /v1/webhooks           — Register a webhook endpoint
    GET    /v1/webhooks           — List registered webhooks
    GET    /v1/webhooks/{id}      — Get webhook details
    PUT    /v1/webhooks/{id}      — Update a webhook
    DELETE /v1/webhooks/{id}      — Unregister a webhook

    POST   /v1/webhooks/{id}/test — Send a test event

Supported event types:
- ``batch.completed`` — Batch processing finished
- ``model.loaded`` — A model was loaded or unloaded
- ``quota.warning`` — A tenant is approaching or has exceeded quota
- ``job.completed`` — A fine-tuning or eval job finished
"""

from __future__ import annotations

import threading
import time
import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from loguru import logger
from pydantic import BaseModel, Field

from ..auth_deps import require_coordinator, require_role
from ..persistent_store import get_store
from distllm.core.webhook_manager import is_safe_webhook_url

# Lazy import — the webhook engine is only needed for the test endpoint
# and the delivery is triggered by event producers, not by this route file.

router = APIRouter(
    prefix="/v1/webhooks",
    tags=["webhooks"],
    dependencies=[
        Depends(require_coordinator),
        # Webhook registration is a privileged operation: a webhook turns the
        # coordinator into an authenticated outbound POSTer of signed event
        # payloads to caller-chosen URLs.  Without this gate any low-privilege
        # key could aim it at internal services (SEC-A4).
        Depends(require_role("admin", "user-admin")),
    ],
)


# ── Valid event types ────────────────────────────────────────────────────

VALID_EVENTS = frozenset({
    "batch.completed",
    "model.loaded",
    "quota.warning",
    "job.completed",
})


# ── Pydantic models ─────────────────────────────────────────────────────


class WebhookCreate(BaseModel):
    """Request to register a webhook endpoint."""
    url: str = Field(..., description="Callback URL (must be HTTPS in production)")
    secret: str = Field(..., min_length=16, description="HMAC signing secret (min 16 chars)")
    events: list[str] = Field(..., min_length=1, description="Event types to subscribe to")
    description: str | None = Field(default=None, description="Optional description")


class WebhookUpdate(BaseModel):
    """Request to update a webhook endpoint."""
    url: str | None = None
    secret: str | None = Field(default=None, min_length=16)
    events: list[str] | None = None
    description: str | None = None
    active: bool | None = None


class WebhookObject(BaseModel):
    """A registered webhook endpoint."""
    id: str
    url: str
    events: list[str]
    description: str | None = None
    active: bool = True
    created_at: float
    last_success_at: float | None = None
    last_failure_at: float | None = None
    consecutive_failures: int = 0


class WebhookListResponse(BaseModel):
    object: str = "list"
    data: list[WebhookObject]


# ── In-memory webhook store (ephemeral, backed by PersistentStore) ────────

# Webhooks are stored in the PersistentStore under a synthetic batch entry
# keyed by "webhook:{id}".  A module-level cache reduces SQLite reads on
# the hot path (dispatch_webhook_to_all is called on every event).

_webhook_cache: dict[str, dict] = {}
_cache_lock = threading.Lock()


def _load_webhooks() -> None:
    """Populate the in-memory cache from the persistent store."""
    global _cache_lock
    store = get_store()
    # The store's batch mechanism is repurposed: we store webhooks as
    # batches with a "webhook:" prefix to reuse the existing persistence.
    all_batches = store.list_batches(limit=1000)
    with _cache_lock:
        for b in all_batches:
            if b.get("id", "").startswith("webhook:"):
                _webhook_cache[b["id"]] = b


def _save_webhook(webhook_id: str, data: dict) -> None:
    store = get_store()
    store.save_batch(webhook_id, data)
    with _cache_lock:
        _webhook_cache[webhook_id] = data


def _delete_webhook(webhook_id: str) -> None:
    store = get_store()
    store.delete_batch(webhook_id)
    with _cache_lock:
        _webhook_cache.pop(webhook_id, None)


def _get_cached_webhooks() -> list[dict]:
    with _cache_lock:
        return list(_webhook_cache.values())


def _get_cached_webhook(webhook_id: str) -> dict | None:
    with _cache_lock:
        return _webhook_cache.get(webhook_id)


# ── Endpoints ────────────────────────────────────────────────────────────


def _validate_webhook_url(url: str) -> str:
    """Reject webhook URLs that fail the SSRF guard (SEC-A4).

    Blocks cloud metadata endpoints, loopback, private/link-local ranges,
    non-http(s) schemes, and hosts resolving to those — unless explicitly
    allowlisted via ``DISTLLM_WEBHOOK_ALLOWLIST``.  Returns the URL or
    raises HTTP 400.
    """
    if not is_safe_webhook_url(url):
        raise HTTPException(
            status_code=400,
            detail="Webhook URL rejected: private, loopback, link-local, or "
                   "metadata targets are not allowed unless the host is in "
                   "DISTLLM_WEBHOOK_ALLOWLIST.",
        )
    return url


@router.post(
    "",
    response_model=WebhookObject,
    status_code=201,
    summary="Register a webhook",
    description="Register a new webhook endpoint that receives HTTP callbacks for specified events.",
)
async def create_webhook(body: WebhookCreate):
    """Register a new webhook endpoint."""
    _validate_webhook_url(body.url)

    # Validate events
    for ev in body.events:
        if ev not in VALID_EVENTS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid event '{ev}'. Valid events: {sorted(VALID_EVENTS)}",
            )

    webhook_id = f"webhook:{uuid.uuid4().hex[:20]}"
    now = time.time()

    data = {
        "id": webhook_id,
        "url": body.url,
        "secret": body.secret,
        "events": body.events,
        "description": body.description,
        "active": True,
        "created_at": now,
        "last_success_at": None,
        "last_failure_at": None,
        "consecutive_failures": 0,
    }

    _save_webhook(webhook_id, data)
    logger.info(f"Webhook registered: {webhook_id} -> {body.url} ({len(body.events)} events)")
    return WebhookObject(**{k: v for k, v in data.items() if k != "secret"})


@router.get(
    "",
    response_model=WebhookListResponse,
    summary="List webhooks",
    description="List all registered webhook endpoints.",
)
async def list_webhooks():
    """List all registered webhooks."""
    webhooks = _get_cached_webhooks()
    return WebhookListResponse(
        data=[WebhookObject(**{k: v for k, v in w.items() if k != "secret"}) for w in webhooks],
    )


@router.get(
    "/{webhook_id}",
    response_model=WebhookObject,
    summary="Get webhook details",
    description="Get details of a specific webhook endpoint.",
)
async def get_webhook(webhook_id: str):
    """Get a webhook by ID."""
    data = _get_cached_webhook(webhook_id)
    if data is None:
        raise HTTPException(status_code=404, detail=f"Webhook '{webhook_id}' not found")
    return WebhookObject(**{k: v for k, v in data.items() if k != "secret"})


@router.put(
    "/{webhook_id}",
    response_model=WebhookObject,
    summary="Update a webhook",
    description="Update a webhook endpoint's URL, secret, events, or active status.",
)
async def update_webhook(webhook_id: str, body: WebhookUpdate):
    """Update a webhook endpoint."""
    data = _get_cached_webhook(webhook_id)
    if data is None:
        raise HTTPException(status_code=404, detail=f"Webhook '{webhook_id}' not found")

    update_dict: dict[str, Any] = {}
    if body.url is not None:
        _validate_webhook_url(body.url)
        update_dict["url"] = body.url
    if body.secret is not None:
        update_dict["secret"] = body.secret
    if body.events is not None:
        for ev in body.events:
            if ev not in VALID_EVENTS:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid event '{ev}'. Valid events: {sorted(VALID_EVENTS)}",
                )
        update_dict["events"] = body.events
    if body.description is not None:
        update_dict["description"] = body.description
    if body.active is not None:
        update_dict["active"] = body.active

    data.update(update_dict)
    _save_webhook(webhook_id, data)
    logger.info(f"Webhook updated: {webhook_id}")
    return WebhookObject(**{k: v for k, v in data.items() if k != "secret"})


@router.delete(
    "/{webhook_id}",
    summary="Delete a webhook",
    description="Unregister a webhook endpoint.",
)
async def delete_webhook(webhook_id: str):
    """Unregister a webhook."""
    data = _get_cached_webhook(webhook_id)
    if data is None:
        raise HTTPException(status_code=404, detail=f"Webhook '{webhook_id}' not found")
    _delete_webhook(webhook_id)
    logger.info(f"Webhook deleted: {webhook_id}")
    return {"status": "deleted", "id": webhook_id}


@router.post(
    "/{webhook_id}/test",
    summary="Send a test event",
    description="Send a test webhook event to verify the endpoint is reachable.",
)
async def test_webhook(webhook_id: str):
    """Send a test event to a webhook endpoint."""
    from distllm.api.webhooks import dispatch_webhook

    data = _get_cached_webhook(webhook_id)
    if data is None:
        raise HTTPException(status_code=404, detail=f"Webhook '{webhook_id}' not found")
    if not data.get("active", True):
        raise HTTPException(status_code=400, detail="Webhook is not active")

    ok = await dispatch_webhook(
        url=data["url"],
        secret=data["secret"],
        event="webhook.test",
        data={"message": "This is a test webhook event from DistLLM."},
    )

    if ok:
        data["last_success_at"] = time.time()
        data["consecutive_failures"] = 0
    else:
        data["last_failure_at"] = time.time()
        data["consecutive_failures"] = data.get("consecutive_failures", 0) + 1
    _save_webhook(webhook_id, data)

    return {"success": ok, "webhook_id": webhook_id}


# ── Event dispatch helper (imported by other modules) ────────────────────


async def dispatch_event(event: str, data: dict) -> int:
    """Deliver *event* with *data* to all registered webhooks.

    Returns the number of successful deliveries.

    Call this from event producers like::

        from distllm.api.routes.webhooks import dispatch_event
        await dispatch_event("batch.completed", {"batch_id": "...", "status": "succeeded"})
    """
    import asyncio

    from distllm.api.webhooks import dispatch_webhook

    webhooks = _get_cached_webhooks()
    active = [w for w in webhooks if w.get("active", True) and event in w.get("events", [])]
    if not active:
        return 0

    tasks = []
    for wh in active:
        tasks.append(
            dispatch_webhook(wh["url"], wh["secret"], event, data)
        )

    results = await asyncio.gather(*tasks)
    successes = sum(1 for r in results if r)

    # Update delivery stats
    for wh, ok in zip(active, results):
        if ok:
            wh["last_success_at"] = time.time()
            wh["consecutive_failures"] = 0
        else:
            wh["last_failure_at"] = time.time()
            wh["consecutive_failures"] = wh.get("consecutive_failures", 0) + 1
        _save_webhook(wh["id"], wh)

    return successes


import threading  # noqa: E402 — needed for cache lock
