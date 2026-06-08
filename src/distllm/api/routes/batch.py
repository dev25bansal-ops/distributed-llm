"""Batch API endpoint for bulk processing.

Allows submitting multiple requests in a single API call for
higher throughput. Processes requests concurrently and returns
all results together.

Usage::

    POST /v1/batch
    {
      "requests": [
        {"method": "chat", "body": {"messages": [{"role": "user", "content": "Hello"}]}},
        {"method": "chat", "body": {"messages": [{"role": "user", "content": "World"}]}}
      ]
    }
"""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from loguru import logger


router = APIRouter(tags=["batch"], prefix="/v1/batch")


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

    # Placeholder — embedding support requires embedding model
    raise HTTPException(status_code=501, detail="Embedding not yet supported in batch mode")
