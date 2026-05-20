"""Async batch processing: POST /v1/batch.

OpenAI-compatible batch API for async processing of multiple requests.
Clients submit a JSONL file of requests, poll for completion, and retrieve results.
"""

import json
import os
import time
import uuid
from pathlib import Path
from enum import Enum
from typing import Literal

import asyncio
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from loguru import logger

from ..api_state import g
from ..persistent_store import get_data_dir, get_store

router = APIRouter(tags=["batch"])

# Persistent storage via SQLite
_store = get_store()

# In-memory task tracking (lost on restart, but batches persist)
_batch_tasks: dict[str, asyncio.Task] = {}
_batch_tasks_lock = asyncio.Lock()


class BatchStatus(str, Enum):
    validating = "validating"
    failed = "failed"
    in_progress = "in_progress"
    finalizing = "finalizing"
    completed = "completed"
    expired = "expired"
    cancelled = "cancelled"


class BatchRequest(BaseModel):
    input_file_id: str = Field(
        ...,
        min_length=6,
        max_length=128,
        pattern=r"^file[-_][A-Za-z0-9_-]+$",
        description="ID of uploaded JSONL file with requests",
    )
    endpoint: Literal["/v1/chat/completions", "/v1/completions"] = Field(
        ...,
        description="API endpoint",
    )
    completion_window: Literal["1h", "8h", "24h"] = Field(
        default="24h",
        description="Max processing time",
    )
    metadata: dict | None = Field(default=None, description="Optional metadata")


class BatchResponse(BaseModel):
    id: str
    object: str = "batch"
    endpoint: str
    errors: dict | None = None
    input_file_id: str
    completion_window: str
    status: str
    output_file_id: str | None = None
    error_file_id: str | None = None
    created_at: int
    in_progress_at: int | None = None
    expires_at: int | None = None
    finalizing_at: int | None = None
    completed_at: int | None = None
    failed_at: int | None = None
    expired_at: int | None = None
    cancelled_at: int | None = None
    request_counts: dict = Field(default_factory=lambda: {"total": 0, "completed": 0, "failed": 0})
    metadata: dict | None = None


class BatchListResponse(BaseModel):
    object: str = "list"
    data: list[BatchResponse]
    first_id: str | None = None
    last_id: str | None = None
    has_more: bool = False


@router.post(
    "/v1/batches",
    response_model=BatchResponse,
    summary="Create batch job",
    description="Create a new asynchronous batch processing job. Submit a JSONL file of requests for non-real-time processing. Supports chat completions and text completions endpoints. Jobs process asynchronously and results are stored for retrieval.",
    response_description="Created batch job with status and metadata",
    responses={
        404: {"description": "Input file not found"},
        503: {"description": "No model loaded"},
    },
)
async def create_batch(body: BatchRequest):
    """Create a new batch job."""
    coord = g.coordinator
    if coord is None:
        raise HTTPException(status_code=503, detail="No model loaded")

    # Verify input file exists
    file_path = _get_file_path(body.input_file_id)
    if file_path is None or not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Input file '{body.input_file_id}' not found. Upload via POST /v1/files first.",
        )

    batch_id = f"batch_{uuid.uuid4().hex[:12]}"
    now = int(time.time())

    # Parse completion window
    window_hours = _parse_window(body.completion_window)

    batch = {
        "id": batch_id,
        "endpoint": body.endpoint,
        "input_file_id": body.input_file_id,
        "completion_window": body.completion_window,
        "status": BatchStatus.validating.value,
        "created_at": now,
        "in_progress_at": None,
        "expires_at": now + window_hours * 3600,
        "finalizing_at": None,
        "completed_at": None,
        "failed_at": None,
        "expired_at": None,
        "cancelled_at": None,
        "output_file_id": None,
        "error_file_id": None,
        "errors": None,
        "metadata": body.metadata,
        "request_counts": {"total": 0, "completed": 0, "failed": 0},
    }
    _store.save_batch(batch_id, batch)

    # Start async processing
    task = asyncio.create_task(_process_batch(batch_id))
    task.add_done_callback(lambda _task: asyncio.create_task(_forget_batch_task(batch_id)))
    async with _batch_tasks_lock:
        _batch_tasks[batch_id] = task

    return BatchResponse(**batch)


@router.get(
    "/v1/batches/{batch_id}",
    response_model=BatchResponse,
    summary="Get batch status",
    description="Retrieve the current status, progress, and metadata of a batch processing job by its ID.",
    response_description="Batch job status and metadata",
    responses={
        404: {"description": "Batch not found"},
    },
)
async def get_batch(batch_id: str):
    """Get batch status."""
    batch = _store.get_batch(batch_id)
    if not batch:
        raise HTTPException(status_code=404, detail=f"Batch '{batch_id}' not found")
    return BatchResponse(**batch)


@router.get(
    "/v1/batches",
    response_model=BatchListResponse,
    summary="List batch jobs",
    description="List all batch processing jobs with pagination support. Returns batch metadata including status and timestamps.",
    response_description="Paginated list of batch jobs",
)
async def list_batches(
    limit: int = Query(default=20, ge=1, le=100),
    after: str | None = Query(default=None, min_length=1, max_length=128),
):
    """List all batches."""
    batches = _store.list_batches(limit=limit + 1, after=after)

    has_more = len(batches) > limit
    batches = batches[:limit]
    data = [BatchResponse(**b) for b in batches]

    return BatchListResponse(
        data=data,
        first_id=data[0].id if data else None,
        last_id=data[-1].id if data else None,
        has_more=has_more,
    )


@router.post(
    "/v1/batches/{batch_id}/cancel",
    summary="Cancel batch job",
    description="Cancel a batch processing job that is in progress. Only batches in validating or in_progress states can be cancelled. Completed, failed, or already-cancelled batches cannot be cancelled.",
    response_description="Cancelled batch job status",
    responses={
        400: {"description": "Cannot cancel batch in current state"},
        404: {"description": "Batch not found"},
    },
)
async def cancel_batch(batch_id: str):
    """Cancel a batch job."""
    batch = _store.get_batch(batch_id)
    if not batch:
        raise HTTPException(status_code=404, detail=f"Batch '{batch_id}' not found")

    if batch["status"] in (BatchStatus.completed.value, BatchStatus.failed.value, BatchStatus.cancelled.value):
        raise HTTPException(status_code=400, detail=f"Cannot cancel batch in state: {batch['status']}")

    batch["status"] = BatchStatus.cancelled.value
    batch["cancelled_at"] = int(time.time())
    _store.save_batch(batch_id, batch)

    # Cancel async task
    async with _batch_tasks_lock:
        task = _batch_tasks.get(batch_id)
    if task and not task.done():
        task.cancel()

    return BatchResponse(**batch)


async def _process_batch(batch_id: str):
    """Process a batch asynchronously."""
    batch = _store.get_batch(batch_id)
    if batch is None:
        return
    coord = g.coordinator

    try:
        # Get input file
        file_path = _get_file_path(batch["input_file_id"])
        if file_path is None:
            raise ValueError(f"Input file not found: {batch['input_file_id']}")

        # Read and parse requests
        requests = []
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    requests.append(json.loads(line))

        batch["request_counts"]["total"] = len(requests)
        batch["status"] = BatchStatus.in_progress.value
        batch["in_progress_at"] = int(time.time())
        _store.save_batch(batch_id, batch)

        results = []
        errors = []

        # Process each request
        for req_data in requests:
            try:
                custom_id = req_data.get("custom_id", "")
                body = req_data.get("body", {})

                # Generate response using the coordinator
                result = await _process_single_request(coord, batch["endpoint"], body)

                results.append({
                    "id": custom_id,
                    "custom_id": custom_id,
                    "response": {
                        "status_code": 200,
                        "body": result,
                    },
                    "error": None,
                })
                batch["request_counts"]["completed"] += 1
            except Exception as e:
                errors.append({
                    "id": req_data.get("custom_id", ""),
                    "custom_id": req_data.get("custom_id", ""),
                    "error": {"message": str(e), "code": "internal_error"},
                })
                batch["request_counts"]["failed"] += 1

        # Write results
        batch["status"] = BatchStatus.finalizing.value
        batch["finalizing_at"] = int(time.time())
        _store.save_batch(batch_id, batch)

        output_file_id = f"file_output_{uuid.uuid4().hex[:8]}"
        error_file_id = f"file_errors_{uuid.uuid4().hex[:8]}" if errors else None

        _store_results(output_file_id, results)
        if errors:
            _store_results(error_file_id, errors)

        batch["status"] = BatchStatus.completed.value
        batch["completed_at"] = int(time.time())
        batch["output_file_id"] = output_file_id
        if error_file_id:
            batch["error_file_id"] = error_file_id
        _store.save_batch(batch_id, batch)

    except asyncio.CancelledError:
        batch["status"] = BatchStatus.cancelled.value
        batch["cancelled_at"] = int(time.time())
        _store.save_batch(batch_id, batch)
    except Exception as e:
        batch["status"] = BatchStatus.failed.value
        batch["failed_at"] = int(time.time())
        batch["errors"] = {"message": str(e)}
        _store.save_batch(batch_id, batch)

    finally:
        async with _batch_tasks_lock:
            current = _batch_tasks.get(batch_id)
            if current is asyncio.current_task():
                _batch_tasks.pop(batch_id, None)


async def _process_single_request(coord, endpoint: str, body: dict) -> dict:
    """Process a single request within a batch."""
    if endpoint == "/v1/chat/completions":
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError("chat batch request body must include non-empty messages")
        prompt = "\n".join(
            f"{msg['role']}: {msg['content']}"
            for msg in messages
            if isinstance(msg, dict) and "role" in msg and "content" in msg
        )
        if not prompt:
            raise ValueError("chat batch messages must include role and content")
        max_tokens = _bounded_int(body.get("max_tokens", 256), "max_tokens", 1, 4096)
        temperature = _bounded_float(body.get("temperature", 0.7), "temperature", 0.0, 2.0)

        # Simple generation
        result_text = await _generate_text(coord, prompt, max_tokens, temperature)

        return {
            "id": f"chatcmpl-batch_{uuid.uuid4().hex[:8]}",
            "object": "chat.completion",
            "model": body.get("model", "distributed-llm"),
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": result_text},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": max_tokens, "total_tokens": max_tokens},
        }
    else:
        prompt = body.get("prompt", "")
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("completion batch request body must include a non-empty prompt")
        max_tokens = _bounded_int(body.get("max_tokens", 16), "max_tokens", 1, 4096)
        temperature = _bounded_float(body.get("temperature", 1.0), "temperature", 0.0, 2.0)

        result_text = await _generate_text(coord, prompt, max_tokens, temperature)

        return {
            "id": f"cmpl-batch_{uuid.uuid4().hex[:8]}",
            "object": "text_completion",
            "model": body.get("model", "distributed-llm"),
            "choices": [{
                "index": 0,
                "text": result_text,
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": max_tokens, "total_tokens": max_tokens},
        }


async def _generate_text(coord, prompt: str, max_tokens: int, temperature: float) -> str:
    """Generate text for a batch request."""
    if not coord or not coord.local_partitioner or not coord.tokenizer:
        return "[Error: No model loaded]"

    import torch
    from distllm.core.token_generator import TokenGenerator

    token_gen = TokenGenerator()
    model = coord.local_partitioner.full_model
    tokenizer = coord.tokenizer
    device = next(model.parameters()).device

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    past_key_values = None

    with torch.no_grad():
        outputs = model(input_ids, use_cache=True)
        generated = []
        for _ in range(max_tokens):
            logits = outputs.logits[:, -1, :]
            next_token, _ = token_gen.sample(logits, temperature=temperature)
            if next_token.dim() == 0:
                next_token = next_token.unsqueeze(0).unsqueeze(0)
            elif next_token.dim() == 1:
                next_token = next_token.unsqueeze(0)
            if next_token.item() == tokenizer.eos_token_id:
                break
            generated.append(next_token.item())
            outputs = model(next_token, past_key_values=outputs.past_key_values, use_cache=True)

    return tokenizer.decode(generated, skip_special_tokens=True)


def _parse_window(window_str: str) -> int:
    """Parse completion window string to hours."""
    if window_str.endswith('h'):
        return int(window_str[:-1])
    return 24


async def _forget_batch_task(batch_id: str) -> None:
    async with _batch_tasks_lock:
        task = _batch_tasks.get(batch_id)
        if task is not None and task.done():
            _batch_tasks.pop(batch_id, None)


def _bounded_int(value, field_name: str, min_value: int, max_value: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer") from exc
    if parsed < min_value or parsed > max_value:
        raise ValueError(f"{field_name} must be between {min_value} and {max_value}")
    return parsed


def _bounded_float(value, field_name: str, min_value: float, max_value: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a number") from exc
    if parsed < min_value or parsed > max_value:
        raise ValueError(f"{field_name} must be between {min_value} and {max_value}")
    return parsed


def _get_base_dir() -> Path:
    """Get durable batch artifact storage directory."""
    base = Path(os.environ.get("DISTLLM_BATCH_DIR", str(get_data_dir() / "batches"))).expanduser()
    base.mkdir(parents=True, exist_ok=True)
    return base


def _get_file_path(file_id: str) -> Path | None:
    """Get the path for an uploaded file."""
    # Check persistent store first
    file_meta = _store.get_file(file_id)
    if file_meta is not None:
        storage_path = file_meta.get("storage_path")
        if storage_path:
            return Path(storage_path)
        filename = file_meta.get("filename", "")
        return _get_base_dir() / f"{file_id}_{filename}"

    # Fallback: check disk by common naming patterns for legacy records.
    base = _get_base_dir()
    for p in base.glob(f"{file_id}*"):
        return p
    return None


def _store_results(file_id: str, results: list[dict]) -> None:
    """Store batch results to disk."""
    base = _get_base_dir()
    storage_path = base / f"{file_id}.jsonl"
    content = "".join(json.dumps(r) + "\n" for r in results)
    storage_path.write_text(content, encoding="utf-8")
    _store.save_file(
        file_id,
        {
            "id": file_id,
            "bytes": storage_path.stat().st_size,
            "created_at": int(time.time()),
            "filename": storage_path.name,
            "purpose": "batch_results",
            "status": "uploaded",
            "storage_path": str(storage_path),
        },
    )
