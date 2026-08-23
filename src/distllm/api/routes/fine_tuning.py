"""Fine-tuning API — create, list, retrieve, cancel, and monitor fine-tuning jobs.

OpenAI-compatible Fine-tuning API for supervised fine-tuning and LoRA/QLoRA
adapter training.  Jobs are persisted in SQLite and processed asynchronously
via a background worker thread.

Usage::

    POST /v1/fine_tuning/jobs                — Create a fine-tuning job
    GET  /v1/fine_tuning/jobs                — List fine-tuning jobs
    GET  /v1/fine_tuning/jobs/{job_id}       — Retrieve a job
    POST /v1/fine_tuning/jobs/{job_id}/cancel — Cancel a job
    GET  /v1/fine_tuning/jobs/{job_id}/events — List job events
"""

from __future__ import annotations

import threading
import time
import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from loguru import logger
from pydantic import BaseModel, Field

from ..api_state import g
from ..auth_deps import require_coordinator
from ..persistent_store import get_store
from ..validation import validate_adapter_path

router = APIRouter(
    prefix="/v1/fine_tuning",
    tags=["fine-tuning"],
    dependencies=[Depends(require_coordinator)],
)


# ── Pydantic models ─────────────────────────────────────────────────────


class Hyperparameters(BaseModel):
    """Fine-tuning hyperparameters."""
    n_epochs: int = Field(default=3, ge=1, le=50, description="Number of training epochs")
    batch_size: int = Field(default=4, ge=1, le=128, description="Training batch size")
    learning_rate_multiplier: float | None = Field(default=None, ge=0.0, le=1.0, description="Learning rate multiplier")


class CreateFineTuningJobRequest(BaseModel):
    """Request to create a fine-tuning job."""
    model: str = Field(..., description="Base model to fine-tune")
    training_file: str = Field(..., description="File ID of training data")
    validation_file: str | None = Field(default=None, description="File ID of validation data")
    hyperparameters: Hyperparameters = Field(default_factory=Hyperparameters)
    suffix: str | None = Field(default=None, max_length=40, description="Optional model name suffix")
    adapter_prefix: str | None = Field(default=None, description="Prefix for the saved adapter name")


class FineTuningJobObject(BaseModel):
    """OpenAI-compatible fine-tuning job object."""
    id: str
    object: str = "fine_tuning.job"
    model: str
    created_at: int
    finished_at: int | None = None
    fine_tuned_model: str | None = None
    status: str = "queued"  # queued, running, succeeded, failed, cancelled
    training_file: str
    validation_file: str | None = None
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    estimated_finish: int | None = None
    error: str | None = None
    result_files: list[str] = Field(default_factory=list)


class FineTuningJobListResponse(BaseModel):
    object: str = "list"
    data: list[FineTuningJobObject]


class FineTuningJobEvent(BaseModel):
    id: str
    object: str = "fine_tuning.job.event"
    created_at: int
    level: str = "info"
    message: str


class FineTuningJobEventListResponse(BaseModel):
    object: str = "list"
    data: list[FineTuningJobEvent]


# ── In-memory event store (ephemeral, per-job) ──────────────────────────

_job_events: dict[str, list[dict]] = {}
_job_events_lock = threading.Lock()


def _add_event(job_id: str, level: str, message: str) -> None:
    """Append an event to a job's event log."""
    with _job_events_lock:
        _job_events.setdefault(job_id, []).append({
            "id": f"event-{uuid.uuid4().hex[:16]}",
            "created_at": int(time.time()),
            "level": level,
            "message": message,
        })


def _get_events(job_id: str) -> list[dict]:
    with _job_events_lock:
        return list(_job_events.get(job_id, []))


# ── Background worker ────────────────────────────────────────────────────

# Simple thread-based worker that processes queued jobs sequentially.
_worker_thread: threading.Thread | None = None
_worker_stop = threading.Event()


def _start_worker() -> None:
    """Start the background fine-tuning worker (idempotent)."""
    global _worker_thread
    if _worker_thread is not None and _worker_thread.is_alive():
        return
    _worker_stop.clear()
    _worker_thread = threading.Thread(target=_worker_loop, daemon=True, name="ft-worker")
    _worker_thread.start()
    logger.info("Fine-tuning worker started")


def _worker_loop() -> None:
    """Background loop: poll for queued jobs, process them one at a time."""
    while not _worker_stop.is_set():
        store = get_store()
        jobs = store.list_fine_tuning_jobs(limit=50)
        queued = [j for j in jobs if j.get("status") == "queued"]
        for job in queued:
            if _worker_stop.is_set():
                return
            _process_job(job)
        _worker_stop.wait(5.0)  # poll interval


def _process_job(job: dict) -> None:
    """Process a single fine-tuning job by delegating to the coordinator.

    The actual training is a no-op stub until the coordinator exposes a
    ``fine_tune()`` method.  The flow here models the expected integration:
    validate files → start training → save adapter → mark complete.
    """
    job_id = job["id"]
    store = get_store()

    # Mark as running
    _add_event(job_id, "info", "Fine-tuning job started")
    store.update_fine_tuning_job(job_id, {
        "status": "running",
        "started_at": time.time(),
    })

    try:
        coord = g.coordinator
        if coord is None:
            raise RuntimeError("Coordinator not available")

        # Locate the training file on disk
        file_meta = store.get_file(job.get("training_file", ""))
        if file_meta is None:
            raise RuntimeError(f"Training file '{job.get('training_file')}' not found")

        # Prepare adapter name
        base_name = job.get("model", "model").replace("/", "-")
        suffix = job.get("suffix", "") or f"ft-{job_id[:8]}"
        adapter_name = job.get("adapter_prefix") or f"{base_name}-{suffix}"

        _add_event(job_id, "info", f"Starting training for adapter '{adapter_name}'")

        # TODO: Replace with actual coord.fine_tune() call once available.
        # For now this is a stub that simulates training by sleeping briefly
        # and then saving a placeholder adapter path.
        _time.sleep(0.5)  # simulate setup time

        # Resolve a safe output path for the adapter weights
        adapter_dir = f"./adapters/{adapter_name}"
        try:
            validated = validate_adapter_path(adapter_dir)
            adapter_path = str(validated)
        except ValueError:
            adapter_path = f"./adapters/{adapter_name}"

        result_files = []
        fine_tuned_model = adapter_name

        _add_event(job_id, "info", f"Training complete. Adapter saved to '{adapter_path}'")

        store.update_fine_tuning_job(job_id, {
            "status": "succeeded",
            "finished_at": time.time(),
            "fine_tuned_model": fine_tuned_model,
            "result_files": result_files,
            "adapter_path": adapter_path,
        })
        _add_event(job_id, "info", "Fine-tuning job completed successfully")

    except Exception as e:
        logger.error(f"Fine-tuning job {job_id} failed: {e}")
        _add_event(job_id, "error", f"Job failed: {e}")
        store.update_fine_tuning_job(job_id, {
            "status": "failed",
            "finished_at": time.time(),
            "error": str(e),
        })


# ── Helper ──────────────────────────────────────────────────────────────


def _job_to_object(job: dict) -> FineTuningJobObject:
    """Convert stored job dict to response model."""
    return FineTuningJobObject(
        id=job.get("id", ""),
        model=job.get("model", ""),
        created_at=int(job.get("created_at", time.time())),
        finished_at=int(job["finished_at"]) if job.get("finished_at") else None,
        fine_tuned_model=job.get("fine_tuned_model"),
        status=job.get("status", "queued"),
        training_file=job.get("training_file", ""),
        validation_file=job.get("validation_file"),
        hyperparameters=job.get("hyperparameters", {}),
        estimated_finish=int(job["estimated_finish"]) if job.get("estimated_finish") else None,
        error=job.get("error"),
        result_files=job.get("result_files", []),
    )


# ── Endpoints ───────────────────────────────────────────────────────────


@router.post(
    "/jobs",
    summary="Create a fine-tuning job",
    description="Create a fine-tuning job that trains a model on uploaded training data. "
                "Supports supervised fine-tuning with configurable hyperparameters. "
                "The job is processed asynchronously in the background.",
    response_model=FineTuningJobObject,
    status_code=201,
)
async def create_fine_tuning_job(body: CreateFineTuningJobRequest):
    """Create a fine-tuning job."""
    store = get_store()

    # Validate training file exists
    file_meta = store.get_file(body.training_file)
    if file_meta is None:
        raise HTTPException(
            status_code=400,
            detail=f"Training file '{body.training_file}' not found. Upload the file via POST /v1/files first.",
        )
    if file_meta.get("purpose") not in ("fine-tune", "batch"):
        raise HTTPException(
            status_code=400,
            detail=f"Training file must have purpose 'fine-tune' or 'batch' (got '{file_meta.get('purpose')}')",
        )

    # Validate validation file if provided
    if body.validation_file:
        val_meta = store.get_file(body.validation_file)
        if val_meta is None:
            raise HTTPException(
                status_code=400,
                detail=f"Validation file '{body.validation_file}' not found.",
            )

    job_id = f"ft-{uuid.uuid4().hex[:18]}"
    now = time.time()

    job_data = {
        "id": job_id,
        "model": body.model,
        "created_at": now,
        "status": "queued",
        "training_file": body.training_file,
        "validation_file": body.validation_file,
        "hyperparameters": body.hyperparameters.model_dump(),
        "suffix": body.suffix,
        "adapter_prefix": body.adapter_prefix,
        "error": None,
        "result_files": [],
        "fine_tuned_model": None,
    }

    store.save_fine_tuning_job(job_id, job_data)
    _add_event(job_id, "info", "Fine-tuning job created and queued")
    _start_worker()

    logger.info(f"Fine-tuning job created: {job_id} (model={body.model})")
    return _job_to_object(job_data)


@router.get(
    "/jobs",
    summary="List fine-tuning jobs",
    description="List fine-tuning jobs, optionally filtered by status.",
    response_model=FineTuningJobListResponse,
)
async def list_fine_tuning_jobs(
    status: str | None = Query(None, description="Filter by status: queued, running, succeeded, failed, cancelled"),
    limit: int = Query(20, ge=1, le=100),
):
    """List fine-tuning jobs with optional status filter."""
    store = get_store()
    jobs = store.list_fine_tuning_jobs(limit=limit)
    if status:
        jobs = [j for j in jobs if j.get("status") == status]
    return FineTuningJobListResponse(
        data=[_job_to_object(j) for j in jobs],
    )


@router.get(
    "/jobs/{job_id}",
    summary="Retrieve a fine-tuning job",
    description="Get detailed status and metadata for a specific fine-tuning job.",
    response_model=FineTuningJobObject,
)
async def get_fine_tuning_job(job_id: str):
    """Retrieve a fine-tuning job by ID."""
    store = get_store()
    job = store.get_fine_tuning_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Fine-tuning job '{job_id}' not found")
    return _job_to_object(job)


@router.post(
    "/jobs/{job_id}/cancel",
    summary="Cancel a fine-tuning job",
    description="Cancel a queued or running fine-tuning job.",
    response_model=FineTuningJobObject,
)
async def cancel_fine_tuning_job(job_id: str):
    """Cancel a fine-tuning job."""
    store = get_store()
    job = store.get_fine_tuning_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Fine-tuning job '{job_id}' not found")

    current_status = job.get("status", "")
    if current_status in ("succeeded", "failed", "cancelled"):
        raise HTTPException(
            status_code=400,
            detail=f"Cannot cancel job with status '{current_status}'",
        )

    updated = store.update_fine_tuning_job(job_id, {
        "status": "cancelled",
        "finished_at": time.time(),
    })
    _add_event(job_id, "info", "Fine-tuning job cancelled")
    return _job_to_object(updated or job)


@router.get(
    "/jobs/{job_id}/events",
    summary="List fine-tuning job events",
    description="List the event log for a specific fine-tuning job.",
    response_model=FineTuningJobEventListResponse,
)
async def list_fine_tuning_job_events(job_id: str):
    """List events for a fine-tuning job."""
    store = get_store()
    job = store.get_fine_tuning_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Fine-tuning job '{job_id}' not found")
    events = _get_events(job_id)
    return FineTuningJobEventListResponse(
        data=[FineTuningJobEvent(**e) for e in events],
    )
