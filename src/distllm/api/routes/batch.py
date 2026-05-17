"""Async batch processing: POST /v1/batch.

OpenAI-compatible batch API for async processing of multiple requests.
Clients submit a JSONL file of requests, poll for completion, and retrieve results.
"""

import json
import os
import time
import uuid
import warnings
from pathlib import Path
from enum import Enum

import asyncio
from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel, Field

from ..api_state import g

router = APIRouter(tags=["batch"])

# In-memory batch storage (production: database)
_batches: dict[str, dict] = {}
_batch_results: dict[str, list[dict]] = {}
_batch_tasks: dict[str, asyncio.Task] = {}


class BatchStatus(str, Enum):
    validating = "validating"
    failed = "failed"
    in_progress = "in_progress"
    finalizing = "finalizing"
    completed = "completed"
    expired = "expired"
    cancelled = "cancelled"


class BatchRequest(BaseModel):
    input_file_id: str = Field(..., description="ID of uploaded JSONL file with requests")
    endpoint: str = Field(..., description="API endpoint: /v1/chat/completions or /v1/completions")
    completion_window: str = Field(default="24h", description="Max processing time: 1h, 8h, 24h")
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


@router.post("/v1/batches", response_model=BatchResponse)
async def create_batch(body: BatchRequest):
    """Create a new batch job."""
    coord = g.coordinator
    if coord is None:
        raise HTTPException(status_code=503, detail="No model loaded")

    # Validate endpoint
    if body.endpoint not in ("/v1/chat/completions", "/v1/completions"):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported endpoint: {body.endpoint}. Must be /v1/chat/completions or /v1/completions",
        )

    # Verify input file exists (check file registry, not batch results)
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
        "expires_at": now + window_hours * 3600,
        "metadata": body.metadata,
        "request_counts": {"total": 0, "completed": 0, "failed": 0},
    }
    _batches[batch_id] = batch

    # Start async processing
    task = asyncio.create_task(_process_batch(batch_id))
    _batch_tasks[batch_id] = task

    return BatchResponse(**batch)


@router.get("/v1/batches/{batch_id}", response_model=BatchResponse)
async def get_batch(batch_id: str):
    """Get batch status."""
    batch = _batches.get(batch_id)
    if not batch:
        raise HTTPException(status_code=404, detail=f"Batch '{batch_id}' not found")
    return BatchResponse(**batch)


@router.get("/v1/batches", response_model=BatchListResponse)
async def list_batches(limit: int = 20, after: str | None = None):
    """List all batches."""
    batches = list(_batches.values())

    # Pagination
    if after:
        idx = next((i for i, b in enumerate(batches) if b["id"] == after), -1)
        batches = batches[idx + 1:] if idx >= 0 else []

    batches = batches[:limit]
    data = [BatchResponse(**b) for b in batches]

    return BatchListResponse(
        data=data,
        first_id=data[0].id if data else None,
        last_id=data[-1].id if data else None,
        has_more=len(batches) > limit,
    )


@router.post("/v1/batches/{batch_id}/cancel")
async def cancel_batch(batch_id: str):
    """Cancel a batch job."""
    batch = _batches.get(batch_id)
    if not batch:
        raise HTTPException(status_code=404, detail=f"Batch '{batch_id}' not found")

    if batch["status"] in (BatchStatus.completed.value, BatchStatus.failed.value, BatchStatus.cancelled.value):
        raise HTTPException(status_code=400, detail=f"Cannot cancel batch in state: {batch['status']}")

    batch["status"] = BatchStatus.cancelled.value
    batch["cancelled_at"] = int(time.time())

    # Cancel async task
    task = _batch_tasks.get(batch_id)
    if task and not task.done():
        task.cancel()

    return BatchResponse(**batch)


async def _process_batch(batch_id: str):
    """Process a batch asynchronously."""
    batch = _batches[batch_id]
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

    except asyncio.CancelledError:
        batch["status"] = BatchStatus.cancelled.value
        batch["cancelled_at"] = int(time.time())
    except Exception as e:
        batch["status"] = BatchStatus.failed.value
        batch["failed_at"] = int(time.time())
        batch["errors"] = {"message": str(e)}


async def _process_single_request(coord, endpoint: str, body: dict) -> dict:
    """Process a single request within a batch."""
    if endpoint == "/v1/chat/completions":
        prompt = "\n".join(
            f"{msg['role']}: {msg['content']}"
            for msg in body.get("messages", [])
        )
        max_tokens = body.get("max_tokens", 256)
        temperature = body.get("temperature", 0.7)

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
        max_tokens = body.get("max_tokens", 16)
        temperature = body.get("temperature", 1.0)

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
            probs = torch.softmax(logits / max(temperature, 1e-6), dim=-1)
            next_token = torch.multinomial(probs, 1)
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


def _get_base_dir() -> Path:
    """Get batch storage directory, with persistent storage warning."""
    base = Path(os.environ.get("DISTLLM_BATCH_DIR", "/tmp/distllm/batches"))
    if "DISTLLM_BATCH_DIR" not in os.environ:
        warnings.warn(
            "DISTLLM_BATCH_DIR not set, defaulting to /tmp which may not persist "
            "across container restarts. Set DISTLLM_BATCH_DIR to a persistent volume path."
        )
    return base


def _get_file_path(file_id: str) -> Path | None:
    """Get the path for an uploaded file."""
    base = _get_base_dir()
    return base / f"{file_id}.jsonl"


def _store_results(file_id: str, results: list[dict]) -> None:
    """Store batch results to disk."""
    base = _get_base_dir()
    base.mkdir(parents=True, exist_ok=True)
    with open(base / f"{file_id}.jsonl", 'w') as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
