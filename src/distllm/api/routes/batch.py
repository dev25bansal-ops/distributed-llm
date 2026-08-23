"""Batch API endpoint for bulk processing.

Allows submitting multiple requests in a single API call for
higher throughput. Processes requests concurrently and returns
all results together.

Supports both synchronous batch processing (POST /v1/batch) and
asynchronous background submission with status polling, SSE streaming,
per-item timeout isolation, webhook notifications, and cost tracking
(POST /v1/batch/submit, GET /v1/batch/{batch_id}/status,
 GET /v1/batch/{batch_id}/stream, POST /v1/batch/{batch_id}/cancel).

Usage::

    # Synchronous batch (legacy)
    POST /v1/batch
    {
      "requests": [
        {"method": "chat", "body": {"messages": [{"role": "user", "content": "Hello"}]}},
        {"method": "chat", "body": {"messages": [{"role": "user", "content": "World"}]}}
      ]
    }

    # Background batch submission
    POST /v1/batch/submit
    {
      "items": [
        {"request_id": "r1", "prompt": "Hello", "max_tokens": 128}
      ],
      "webhook_url": "https://example.com/callback"
    }
    -> {"batch_id": "abc-123", "status": "pending"}

    # Poll status
    GET /v1/batch/abc-123/status

    # SSE stream of individual results
    GET /v1/batch/abc-123/stream

    # Cancel
    POST /v1/batch/abc-123/cancel
"""

from __future__ import annotations

import asyncio
import ipaddress
import json
import math
import socket
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, AsyncGenerator
from urllib.parse import urlparse

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from loguru import logger


# ─── Router ────────────────────────────────────────────────────────────────

router = APIRouter(tags=["batch"], prefix="/v1/batch")


# ─── Per-token cost rates for self-hosted inference (USD) ──────────────────
# Rough estimates based on typical GPU costs (A100-80GB ~$2-3/hr at 1000 tok/s
# output = ~$0.0008/tok; input cheaper because it is prefill-dominated).
_COST_PER_INPUT_TOKEN = 0.000002    # ~$2/1M tokens
_COST_PER_OUTPUT_TOKEN = 0.000008   # ~$8/1M tokens


# ─── Legacy synchronous batch models ───────────────────────────────────────

class BatchRequestItem(BaseModel):
    """A single request within a batch."""
    method: str = Field(..., description="Request method: 'chat', 'completion', 'embedding'")
    body: dict = Field(..., description="Request body (same as individual endpoint)")
    request_id: str = Field(default="", description="Optional client-assigned request ID")


class BatchRequest(BaseModel):
    """A batch of requests to process concurrently."""
    requests: list[BatchRequestItem] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="List of requests (max 100)",
    )


class BatchResultItem(BaseModel):
    """Result of a single request within a batch."""
    request_id: str
    status: str  # "success" or "error"
    result: Any = None
    error: str | None = None
    latency_ms: float = 0.0


class BatchResponse(BaseModel):
    """Response containing all batch results."""
    batch_id: str
    total_requests: int
    successful: int
    failed: int
    total_latency_ms: float
    results: list[BatchResultItem]


# ─── New background-batch models ───────────────────────────────────────────

class BatchItemStatus(str, Enum):
    """Status of a single batch item."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


class BatchStatus(str, Enum):
    """Status of an entire batch."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PARTIALLY_COMPLETED = "partially_completed"


class BatchSubmitItem(BaseModel):
    """A single item to process within a background batch."""
    request_id: str = Field(default="", description="Optional client-assigned request ID")
    prompt: str = Field(..., description="Prompt text to send to the model")
    max_tokens: int = Field(default=128, ge=1, le=32768, description="Maximum tokens to generate")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")


class BatchSubmitRequest(BaseModel):
    """Request body for submitting a background batch."""
    items: list[BatchSubmitItem] = Field(
        ...,
        min_length=1,
        max_length=500,
        description="List of items to process (max 500)",
    )
    webhook_url: str | None = Field(
        default=None,
        description="URL to POST batch results to on completion",
    )
    webhook_token: str | None = Field(
        default=None,
        description="Bearer token for the webhook call (X-Webhook-Token header)",
    )
    item_timeout: float = Field(
        default=30.0,
        ge=1.0,
        le=600.0,
        description="Per-item timeout in seconds (default 30s, max 600s)",
    )


class BatchSubmitResponse(BaseModel):
    """Response after submitting a background batch."""
    batch_id: str
    status: BatchStatus
    total_items: int
    created_at: float
    message: str = "Batch submitted successfully. Use GET /v1/batch/{batch_id}/status to poll."


class BatchItemResult(BaseModel):
    """Result of a single item within a background batch."""
    request_id: str
    prompt: str = ""
    max_tokens: int = 0
    temperature: float = 0.0
    status: BatchItemStatus
    result: Any = None
    error: str | None = None
    cost: float = 0.0
    started_at: float | None = None
    completed_at: float | None = None


class BatchStatusResponse(BaseModel):
    """Status response for a background batch."""
    batch_id: str
    status: BatchStatus
    total_items: int
    pending: int
    running: int
    completed: int
    failed: int
    timed_out: int
    cancelled: int
    total_cost: float = 0.0
    created_at: float
    completed_at: float | None = None
    items: list[BatchItemResult] = []


class BatchCancelResponse(BaseModel):
    """Response after cancelling a batch."""
    batch_id: str
    status: str
    cancelled_items: int
    message: str


# ─── Internal batch state (in-memory) ─────────────────────────────────────
# Items and batches are stored in plain dicts with an asyncio lock for thread
# safety.  This is ephemeral -- data is lost on server restart.  Future
# versions could persist to Redis / SQLite.

@dataclass
class _BatchItemState:
    """Internal mutable state for a single batch item."""
    request_id: str
    prompt: str
    max_tokens: int
    temperature: float
    status: BatchItemStatus = BatchItemStatus.PENDING
    result: Any = None
    error: str | None = None
    cost: float = 0.0
    started_at: float | None = None
    completed_at: float | None = None
    _task: asyncio.Task | None = None  # reference to the processing coroutine


@dataclass
class _BatchState:
    """Internal mutable state for an entire batch."""
    batch_id: str
    status: BatchStatus
    items: list[_BatchItemState]
    created_at: float
    completed_at: float | None = None
    webhook_url: str | None = None
    webhook_token: str | None = None
    item_timeout: float = 30.0
    owner: str | None = None  # api_key_id that submitted the batch (IDOR guard)
    _changed: asyncio.Event = field(default_factory=asyncio.Event)
    _cancelled: bool = False


# Thread-safe in-memory store.  Using asyncio.Lock because all access is
# from asyncio coroutines running on the same event loop.
_batch_store: dict[str, _BatchState] = {}
_batch_lock = asyncio.Lock()

# Background tasks registry so we can await / cancel them on shutdown
_background_tasks: set[asyncio.Task] = set()


async def _store_get(batch_id: str) -> _BatchState | None:
    """Thread-safe read from the batch store."""
    async with _batch_lock:
        return _batch_store.get(batch_id)


async def _store_upsert(batch: _BatchState) -> None:
    """Thread-safe write to the batch store."""
    async with _batch_lock:
        _batch_store[batch.batch_id] = batch


async def _store_delete(batch_id: str) -> None:
    """Thread-safe delete from the batch store."""
    async with _batch_lock:
        _batch_store.pop(batch_id, None)


# ─── Legacy synchronous batch endpoint ─────────────────────────────────────

@router.post("", response_model=BatchResponse)
async def process_batch(batch: BatchRequest, request: Request):
    """Process multiple requests concurrently.

    Each request is dispatched to the appropriate handler (chat, completion,
    embedding) and processed in parallel. Results are collected and returned
    together.

    Maximum batch size: 100 requests.
    """
    batch_id = str(uuid.uuid4())
    t0 = time.monotonic()

    # Get coordinator reference
    from distllm.api.api_state import g as _g
    coordinator = _g.coordinator
    if coordinator is None:
        raise HTTPException(status_code=503, detail="Coordinator not initialized")

    # Process all requests concurrently
    tasks = []
    for item in batch.requests:
        req_id = item.request_id or str(uuid.uuid4())
        tasks.append(_process_single(item, req_id, coordinator))

    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Collect results
    batch_results = []
    successful = 0
    failed = 0

    for i, result in enumerate(results):
        req_id = batch.requests[i].request_id or f"item-{i}"
        if isinstance(result, Exception):
            batch_results.append(BatchResultItem(
                request_id=req_id,
                status="error",
                error=str(result),
            ))
            failed += 1
        else:
            batch_results.append(result)
            if result.status == "success":
                successful += 1
            else:
                failed += 1

    total_ms = (time.monotonic() - t0) * 1000

    logger.info(
        f"Batch {batch_id}: {successful}/{len(batch.requests)} succeeded, "
        f"{total_ms:.0f}ms total"
    )

    return BatchResponse(
        batch_id=batch_id,
        total_requests=len(batch.requests),
        successful=successful,
        failed=failed,
        total_latency_ms=round(total_ms, 2),
        results=batch_results,
    )


async def _process_single(
    item: BatchRequestItem,
    request_id: str,
    coordinator: Any,
) -> BatchResultItem:
    """Process a single request within a batch."""
    t0 = time.monotonic()

    try:
        method = item.method.lower()
        body = item.body

        if method == "chat":
            result = await _process_chat(body, coordinator)
        elif method == "completion":
            result = await _process_completion(body, coordinator)
        elif method == "embedding":
            result = await _process_embedding(body, coordinator)
        else:
            return BatchResultItem(
                request_id=request_id,
                status="error",
                error=f"Unknown method: {method}",
            )

        latency_ms = (time.monotonic() - t0) * 1000
        return BatchResultItem(
            request_id=request_id,
            status="success",
            result=result,
            latency_ms=round(latency_ms, 2),
        )

    except Exception as e:
        latency_ms = (time.monotonic() - t0) * 1000
        logger.warning(f"Batch item {request_id} failed: {e}")
        return BatchResultItem(
            request_id=request_id,
            status="error",
            error=str(e),
            latency_ms=round(latency_ms, 2),
        )


async def _process_chat(body: dict, coordinator: Any) -> dict:
    """Process a chat completion request."""
    messages = body.get("messages", [])
    if not messages:
        raise ValueError("messages is required")

    # Extract the last user message as the prompt
    prompt = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            prompt = msg.get("content", "")
            break

    if not prompt:
        raise ValueError("No user message found")

    max_tokens = body.get("max_tokens", 128)
    temperature = body.get("temperature", 0.7)

    result = await asyncio.to_thread(
        coordinator.generate,
        prompt=prompt,
        max_new_tokens=max_tokens,
        temperature=temperature,
    )

    return {
        "choices": [{"message": {"role": "assistant", "content": result}}],
        "usage": {"prompt_tokens": len(prompt.split()), "completion_tokens": len(result.split())},
    }


async def _process_completion(body: dict, coordinator: Any) -> dict:
    """Process a text completion request."""
    prompt = body.get("prompt", "")
    if not prompt:
        raise ValueError("prompt is required")

    max_tokens = body.get("max_tokens", 128)
    temperature = body.get("temperature", 0.7)

    result = await asyncio.to_thread(
        coordinator.generate,
        prompt=prompt,
        max_new_tokens=max_tokens,
        temperature=temperature,
    )

    return {
        "choices": [{"text": result}],
        "usage": {"prompt_tokens": len(prompt.split()), "completion_tokens": len(result.split())},
    }


async def _process_embedding(body: dict, coordinator: Any) -> dict:
    """Process an embedding request."""
    input_text = body.get("input", "")
    if not input_text:
        raise ValueError("input is required")

    # Placeholder -- embedding support requires embedding model
    raise HTTPException(status_code=501, detail="Embedding not yet supported in batch mode")


# ═══════════════════════════════════════════════════════════════════════════
# NEW BACKGROUND BATCH ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════

# ─── Background batch: submit ──────────────────────────────────────────────

@router.post("/submit", response_model=BatchSubmitResponse, status_code=202)
async def submit_batch(batch_req: BatchSubmitRequest, request: Request) -> BatchSubmitResponse:
    """Submit a batch for background processing.

    Returns immediately with a ``batch_id`` that can be used to poll status,
    stream results via SSE, or cancel the batch.  Each item has an independent
    timeout: a slow or hung item does not delay other items.

    Optional ``webhook_url`` receives a POST with the full results when the
    batch completes (all items finished, cancelled, or timed out).
    """
    # SECURITY: reject SSRF-prone webhook targets before creating any state.
    if batch_req.webhook_url:
        _validate_webhook_url(batch_req.webhook_url)

    batch_id = _generate_batch_id()

    # Build internal state
    items = [
        _BatchItemState(
            request_id=item.request_id or f"item-{i}",
            prompt=item.prompt,
            max_tokens=item.max_tokens,
            temperature=item.temperature,
        )
        for i, item in enumerate(batch_req.items)
    ]

    batch_state = _BatchState(
        batch_id=batch_id,
        status=BatchStatus.PENDING,
        items=items,
        created_at=time.time(),
        webhook_url=batch_req.webhook_url,
        webhook_token=batch_req.webhook_token,
        item_timeout=batch_req.item_timeout,
        owner=getattr(request.state, "api_key_id", None),
    )
    await _store_upsert(batch_state)

    # Retrieve coordinator reference
    coordinator = _get_coordinator()
    if coordinator is None:
        await _store_delete(batch_id)
        raise HTTPException(status_code=503, detail="Coordinator not initialized")

    logger.info(
        f"Batch {batch_id} submitted: {len(items)} items, "
        f"timeout={batch_req.item_timeout}s, "
        f"webhook={'yes' if batch_req.webhook_url else 'no'}"
    )

    # Kick off background processing (fire-and-forget via task)
    task = asyncio.create_task(
        _process_batch_background(batch_state, coordinator)
    )
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)

    return BatchSubmitResponse(
        batch_id=batch_id,
        status=BatchStatus.PENDING,
        total_items=len(items),
        created_at=batch_state.created_at,
    )


# ─── Background batch: status polling ──────────────────────────────────────

@router.get("/{batch_id}/status", response_model=BatchStatusResponse)
async def get_batch_status(batch_id: str, request: Request) -> BatchStatusResponse:
    """Poll the current status of a background batch.

    Returns per-item results for items that have finished
    (completed / failed / timed out / cancelled).
    """
    state = await _store_get(batch_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Batch {batch_id} not found")
    _require_owner(state, request)

    return _build_status_response(state)


# ─── Background batch: SSE stream ──────────────────────────────────────────

@router.get("/{batch_id}/stream")
async def stream_batch_results(batch_id: str, request: Request) -> StreamingResponse:
    """Server-Sent Events stream of batch item results.

    Events::

        event: item_complete
        data: {"request_id": "...", "status": "completed", ...}

        event: item_complete
        data: {"request_id": "...", "status": "failed", "error": "..."}

        event: batch_complete
        data: {"batch_id": "...", "status": "completed", ...}

        event: keepalive
        data: {}

    The stream stays open until all items have finished or the client
    disconnects.
    """
    state = await _store_get(batch_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Batch {batch_id} not found")
    _require_owner(state, request)

    return StreamingResponse(
        _stream_events(state, request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


async def _stream_events(
    state: _BatchState,
    request: Request,
) -> AsyncGenerator[bytes, None]:
    """Generate SSE event bytes for a batch stream."""
    # Track which items we have already emitted
    emitted_seq: set[int] = set()

    # Send initial batch-start event
    start_payload = json.dumps({
        "event": "batch_start",
        "batch_id": state.batch_id,
        "total_items": len(state.items),
        "created_at": state.created_at,
    })
    yield f"event: batch_start\ndata: {start_payload}\n\n".encode("utf-8")

    try:
        while True:
            # Check for client disconnect
            if await request.is_disconnected():
                logger.debug(f"Client disconnected from batch {state.batch_id} stream")
                break

            # Emit newly completed items
            all_done = True
            for i, item in enumerate(state.items):
                if i in emitted_seq:
                    continue
                if item.status in (
                    BatchItemStatus.COMPLETED,
                    BatchItemStatus.FAILED,
                    BatchItemStatus.TIMEOUT,
                    BatchItemStatus.CANCELLED,
                ):
                    emitted_seq.add(i)
                    event_data = _build_item_result(item)
                    payload = json.dumps({
                        "event": "item_complete",
                        "item": event_data.model_dump(),
                    })
                    yield f"event: item_complete\ndata: {payload}\n\n".encode("utf-8")

                if item.status in (
                    BatchItemStatus.PENDING,
                    BatchItemStatus.RUNNING,
                ):
                    all_done = False

            # Check if the batch itself is done
            if state.status in (
                BatchStatus.COMPLETED,
                BatchStatus.FAILED,
                BatchStatus.CANCELLED,
                BatchStatus.PARTIALLY_COMPLETED,
            ):
                complete_payload = json.dumps({
                    "event": "batch_complete",
                    "batch_id": state.batch_id,
                    "status": state.status.value,
                    "total_items": len(state.items),
                    "completed": sum(
                        1 for it in state.items if it.status == BatchItemStatus.COMPLETED
                    ),
                    "failed": sum(
                        1 for it in state.items if it.status in (
                            BatchItemStatus.FAILED,
                            BatchItemStatus.TIMEOUT,
                        )
                    ),
                    "total_cost": sum(it.cost for it in state.items),
                })
                yield f"event: batch_complete\ndata: {complete_payload}\n\n".encode("utf-8")
                break

            if all_done and not _batch_has_live_items(state):
                # All items terminated but batch status not updated yet — wait briefly
                pass

            # Wait for state change or send keepalive
            try:
                await asyncio.wait_for(state._changed.wait(), timeout=15.0)
                state._changed.clear()
            except asyncio.TimeoutError:
                # Send keepalive to prevent proxy timeouts
                yield b"event: keepalive\ndata: {}\n\n"

        # Final flush: emit any remaining items we might have missed
        for i, item in enumerate(state.items):
            if i not in emitted_seq and item.status in (
                BatchItemStatus.COMPLETED,
                BatchItemStatus.FAILED,
                BatchItemStatus.TIMEOUT,
                BatchItemStatus.CANCELLED,
            ):
                emitted_seq.add(i)
                event_data = _build_item_result(item)
                payload = json.dumps({
                    "event": "item_complete",
                    "item": event_data.model_dump(),
                })
                yield f"event: item_complete\ndata: {payload}\n\n".encode("utf-8")

    except asyncio.CancelledError:
        logger.debug(f"Batch {state.batch_id} stream cancelled")
        raise


# ─── Background batch: cancel ──────────────────────────────────────────────

@router.post("/{batch_id}/cancel", response_model=BatchCancelResponse)
async def cancel_batch(batch_id: str, request: Request) -> BatchCancelResponse:
    """Cancel an in-progress background batch.

    Pending items are marked cancelled.  Running items are cancelled via
    their asyncio task.  Already-finished items are left untouched.
    """
    state = await _store_get(batch_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Batch {batch_id} not found")
    _require_owner(state, request)

    if state.status in (BatchStatus.COMPLETED, BatchStatus.CANCELLED, BatchStatus.FAILED):
        raise HTTPException(
            status_code=400,
            detail=f"Batch {batch_id} is already {state.status.value}",
        )

    state._cancelled = True
    cancelled_count = 0

    for item in state.items:
        if item.status == BatchItemStatus.PENDING:
            item.status = BatchItemStatus.CANCELLED
            item.completed_at = time.time()
            cancelled_count += 1
        elif item.status == BatchItemStatus.RUNNING:
            # Cancel the underlying asyncio task
            if item._task is not None and not item._task.done():
                item._task.cancel()
            item.status = BatchItemStatus.CANCELLED
            item.completed_at = time.time()
            cancelled_count += 1

    state.status = BatchStatus.CANCELLED
    state.completed_at = time.time()
    state._changed.set()

    await _store_upsert(state)

    logger.info(f"Batch {batch_id} cancelled: {cancelled_count} items")
    return BatchCancelResponse(
        batch_id=batch_id,
        status="cancelled",
        cancelled_items=cancelled_count,
        message=f"Batch cancelled. {cancelled_count} item(s) marked cancelled.",
    )


# ═══════════════════════════════════════════════════════════════════════════
# BACKGROUND PROCESSING
# ═══════════════════════════════════════════════════════════════════════════

async def _process_batch_background(state: _BatchState, coordinator: Any) -> None:
    """Process all items in a batch concurrently with per-item timeouts.

    Each item runs in its own asyncio task with an independent timeout.
    A slow or hung item does not affect other items.  When all items
    have finished, the webhook (if configured) is fired.
    """
    state.status = BatchStatus.RUNNING
    state._changed.set()
    await _store_upsert(state)

    # Fire off one task per item
    tasks: list[asyncio.Task] = []
    for item_state in state.items:
        task = asyncio.create_task(
            _process_single_background_item(item_state, coordinator, state)
        )
        item_state._task = task
        tasks.append(task)

    # Wait for all items to finish (each handles its own timeout internally)
    await asyncio.gather(*tasks, return_exceptions=True)

    # Determine final batch status
    completed = sum(1 for it in state.items if it.status == BatchItemStatus.COMPLETED)
    failed = sum(1 for it in state.items if it.status == BatchItemStatus.FAILED)
    timed_out = sum(1 for it in state.items if it.status == BatchItemStatus.TIMEOUT)
    cancelled = sum(1 for it in state.items if it.status == BatchItemStatus.CANCELLED)
    total = len(state.items)

    if cancelled == total:
        state.status = BatchStatus.CANCELLED
    elif completed == total:
        state.status = BatchStatus.COMPLETED
    elif completed > 0:
        state.status = BatchStatus.PARTIALLY_COMPLETED
    elif failed + timed_out == total:
        state.status = BatchStatus.FAILED
    else:
        state.status = BatchStatus.PARTIALLY_COMPLETED

    state.completed_at = time.time()
    state._changed.set()
    await _store_upsert(state)

    logger.info(
        f"Batch {state.batch_id} finished: status={state.status.value}, "
        f"{completed}/{total} completed, {failed} failed, "
        f"{timed_out} timed out, {cancelled} cancelled"
    )

    # Fire webhook (best-effort, non-blocking)
    if state.webhook_url and not state._cancelled:
        try:
            await _send_webhook(state)
        except Exception as exc:
            logger.warning(f"Batch {state.batch_id} webhook failed: {exc}")


async def _process_single_background_item(
    item_state: _BatchItemState,
    coordinator: Any,
    batch_state: _BatchState,
) -> None:
    """Process a single batch item with its own timeout.

    The item is marked RUNNING at start, then COMPLETED / FAILED / TIMEOUT
    depending on the outcome.  The batch's ``_changed`` event is set so that
    any SSE stream listener picks up the update.
    """
    if batch_state._cancelled:
        item_state.status = BatchItemStatus.CANCELLED
        item_state.completed_at = time.time()
        batch_state._changed.set()
        return

    item_state.status = BatchItemStatus.RUNNING
    item_state.started_at = time.time()
    batch_state._changed.set()

    # The coordinator.generate is CPU-bound and may block; we wrap it in
    # asyncio.to_thread.  The timeout is applied on the await so that a
    # stuck coordinator thread does not hang the item forever.
    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(
                coordinator.generate,
                prompt=item_state.prompt,
                max_new_tokens=item_state.max_tokens,
                temperature=item_state.temperature,
            ),
            timeout=batch_state.item_timeout,
        )

        # Estimate tokens and cost
        prompt_tokens = max(1, len(item_state.prompt.split()))
        completion_tokens = max(1, len(result.split()))

        item_state.result = result
        item_state.status = BatchItemStatus.COMPLETED
        item_state.cost = _estimate_cost(prompt_tokens, completion_tokens)

    except asyncio.TimeoutError:
        logger.warning(
            f"Batch item {item_state.request_id} timed out "
            f"after {batch_state.item_timeout}s"
        )
        item_state.status = BatchItemStatus.TIMEOUT
        item_state.error = f"Item timed out after {batch_state.item_timeout}s"

    except asyncio.CancelledError:
        item_state.status = BatchItemStatus.CANCELLED
        item_state.error = "Item cancelled"

    except Exception as exc:
        logger.warning(f"Batch item {item_state.request_id} failed: {exc}")
        item_state.status = BatchItemStatus.FAILED
        item_state.error = str(exc)

    finally:
        item_state.completed_at = time.time()
        batch_state._changed.set()


# ─── Webhook delivery ──────────────────────────────────────────────────────

async def _send_webhook(state: _BatchState) -> None:
    """POST batch results to the configured webhook URL.

    Uses httpx.AsyncClient with a 10-second timeout.  The webhook is
    best-effort: failures are logged but do not affect batch processing.
    """
    # SECURITY: fail closed if the URL is (or has become) an SSRF target since
    # submission — never POST to private/loopback/metadata addresses.
    try:
        _validate_webhook_url(state.webhook_url)
    except HTTPException:
        logger.warning(
            f"Batch {state.batch_id} webhook URL rejected by SSRF guard; skipping"
        )
        return

    payload = _build_webhook_payload(state)

    headers = {
        "Content-Type": "application/json",
        "User-Agent": "DistLLM-Batch/1.0",
        "X-Batch-ID": state.batch_id,
    }
    if state.webhook_token:
        headers["Authorization"] = f"Bearer {state.webhook_token}"

    import httpx

    async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
        response = await client.post(
            state.webhook_url,  # type: ignore[arg-type]
            json=payload,
            headers=headers,
        )
        response.raise_for_status()

    logger.info(
        f"Batch {state.batch_id} webhook delivered to "
        f"{state.webhook_url} (status={response.status_code})"
    )


# ─── Webhook / ownership security guards ───────────────────────────────────

def _is_private_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """Return True for loopback / private / link-local / metadata addresses."""
    return (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
        or (
            isinstance(ip, ipaddress.IPv4Address)
            and ip == ipaddress.ip_address("169.254.169.254")
        )
    )


def _validate_webhook_url(url: str | None) -> None:
    """Reject webhook URLs that enable SSRF.

    Only public HTTPS endpoints are allowed: the host must resolve exclusively
    to public addresses (no loopback / private / link-local / cloud-metadata /
    multicast ranges).  Raises ``HTTPException(400)`` on any violation.
    """
    if not url:
        raise HTTPException(status_code=400, detail="webhook_url is required")
    parsed = urlparse(url)
    if parsed.scheme != "https":
        raise HTTPException(
            status_code=400, detail="webhook_url must use the https scheme"
        )
    host = (parsed.hostname or "").rstrip(".")
    if not host:
        raise HTTPException(status_code=400, detail="webhook_url must include a host")

    port = parsed.port or 443
    try:
        infos = socket.getaddrinfo(host, port)
    except socket.gaierror:
        raise HTTPException(
            status_code=400, detail=f"webhook_url host {host!r} does not resolve"
        )

    for info in infos:
        try:
            ip = ipaddress.ip_address(info[4][0])
        except ValueError:
            continue
        if _is_private_ip(ip):
            raise HTTPException(
                status_code=400,
                detail="webhook_url must not target a loopback/private/link-local "
                       "address",
            )


def _require_owner(state: _BatchState, request: Request) -> None:
    """Reject access to a batch that the requesting API key does not own.

    Returns 404 (not 403) so batch existence is not leaked to other tenants.
    """
    requester = getattr(request.state, "api_key_id", None)
    if state.owner is not None and requester != state.owner:
        raise HTTPException(status_code=404, detail="Batch not found")


# ═══════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _generate_batch_id() -> str:
    """Generate a unique, URL-safe batch ID."""
    return f"batch_{uuid.uuid4().hex[:16]}"


def _get_coordinator() -> Any:
    """Retrieve the current coordinator from shared application state."""
    from distllm.api.api_state import g as _g
    return _g.coordinator


def _estimate_cost(prompt_tokens: int, completion_tokens: int) -> float:
    """Estimate the USD cost of an inference request.

    Uses per-token rates for self-hosted inference (GPU compute only).
    """
    return round(
        prompt_tokens * _COST_PER_INPUT_TOKEN
        + completion_tokens * _COST_PER_OUTPUT_TOKEN,
        8,
    )


def _build_status_response(state: _BatchState) -> BatchStatusResponse:
    """Build a BatchStatusResponse from internal state."""
    counts: dict[BatchItemStatus, int] = {}
    for s in BatchItemStatus:
        counts[s] = 0
    for it in state.items:
        counts[it.status] = counts.get(it.status, 0) + 1

    return BatchStatusResponse(
        batch_id=state.batch_id,
        status=state.status,
        total_items=len(state.items),
        pending=counts.get(BatchItemStatus.PENDING, 0),
        running=counts.get(BatchItemStatus.RUNNING, 0),
        completed=counts.get(BatchItemStatus.COMPLETED, 0),
        failed=counts.get(BatchItemStatus.FAILED, 0),
        timed_out=counts.get(BatchItemStatus.TIMEOUT, 0),
        cancelled=counts.get(BatchItemStatus.CANCELLED, 0),
        total_cost=sum(it.cost for it in state.items),
        created_at=state.created_at,
        completed_at=state.completed_at,
        items=[_build_item_result(it) for it in state.items],
    )


def _build_item_result(item: _BatchItemState) -> BatchItemResult:
    """Build a public BatchItemResult from internal item state."""
    return BatchItemResult(
        request_id=item.request_id,
        prompt=item.prompt,
        max_tokens=item.max_tokens,
        temperature=item.temperature,
        status=item.status,
        result=item.result,
        error=item.error,
        cost=item.cost,
        started_at=item.started_at,
        completed_at=item.completed_at,
    )


def _build_webhook_payload(state: _BatchState) -> dict[str, Any]:
    """Build the JSON payload for a webhook POST."""
    return {
        "event": "batch_complete",
        "batch_id": state.batch_id,
        "status": state.status.value,
        "total_items": len(state.items),
        "completed": sum(
            1 for it in state.items if it.status == BatchItemStatus.COMPLETED
        ),
        "failed": sum(
            1 for it in state.items if it.status == BatchItemStatus.FAILED
        ),
        "timed_out": sum(
            1 for it in state.items if it.status == BatchItemStatus.TIMEOUT
        ),
        "cancelled": sum(
            1 for it in state.items if it.status == BatchItemStatus.CANCELLED
        ),
        "total_cost": sum(it.cost for it in state.items),
        "created_at": state.created_at,
        "completed_at": state.completed_at,
        "items": [
            {
                "request_id": it.request_id,
                "status": it.status.value,
                "result": it.result,
                "error": it.error,
                "cost": it.cost,
                "started_at": it.started_at,
                "completed_at": it.completed_at,
            }
            for it in state.items
        ],
    }


def _batch_has_live_items(state: _BatchState) -> bool:
    """Check if any items are still pending or running."""
    return any(
        it.status in (BatchItemStatus.PENDING, BatchItemStatus.RUNNING)
        for it in state.items
    )
