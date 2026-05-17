"""Fine-tuning: POST /v1/fine_tuning/jobs.

OpenAI-compatible fine-tuning job management.
"""

import asyncio
import json
import os
import time
import uuid
from pathlib import Path
from typing import List, Optional, Literal
from enum import Enum

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..api_state import g

router = APIRouter(tags=["fine-tuning"])

# In-memory job storage
_jobs: dict = {}
_job_tasks: dict = {}


class FineTuningStatus(str, Enum):
    validating_files = "validating_files"
    queued = "queued"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"
    cancelled = "cancelled"


class Hyperparams(BaseModel):
    n_epochs: int = Field(default=3, ge=1, le=50)
    batch_size: Optional[int] = Field(default=None, ge=1)
    learning_rate_multiplier: float = Field(default=1.0, ge=0.1, le=10.0)


class FineTuningRequest(BaseModel):
    model: str = Field(..., description="Base model to fine-tune")
    training_file: str = Field(..., description="ID of uploaded training file")
    validation_file: Optional[str] = Field(default=None, description="ID of validation file")
    hyperparameters: Optional[Hyperparams] = Field(default=None)
    suffix: Optional[str] = Field(default=None, description="Model name suffix")
    integrations: Optional[List[dict]] = Field(default=None)
    seed: Optional[int] = Field(default=None)


class FineTuningJobEvent(BaseModel):
    object: str = "fine_tuning.job.event"
    id: str
    created_at: int
    level: str
    message: str
    data: Optional[dict] = None
    type: str


class FineTuningJob(BaseModel):
    id: str
    object: str = "fine_tuning.job"
    model: str
    created_at: int
    finished_at: Optional[int] = None
    fine_tuned_model: Optional[str] = None
    hyperparameters: dict
    organization_id: Optional[str] = None
    result_files: List[str] = Field(default_factory=list)
    status: str
    trained_tokens: Optional[int] = None
    training_file: str
    validation_file: Optional[str] = None
    error: Optional[dict] = None
    estimated_finish: Optional[int] = None
    integrations: Optional[List[dict]] = None
    seed: Optional[int] = None


class FineTuningJobList(BaseModel):
    object: str = "list"
    data: List[FineTuningJob]
    has_more: bool = False


@router.post("/v1/fine_tuning/jobs", response_model=FineTuningJob)
async def create_fine_tuning_job(body: FineTuningRequest):
    """Create a fine-tuning job."""
    coord = g.coordinator
    if coord is None:
        raise HTTPException(status_code=503, detail="No model loaded")

    # Verify training file exists
    from .files import _files
    if body.training_file not in _files:
        raise HTTPException(
            status_code=404,
            detail=f"Training file '{body.training_file}' not found. Upload via POST /v1/files first.",
        )

    training_file = _files[body.training_file]
    if training_file["purpose"] not in ("fine-tune", "batch"):
        raise HTTPException(
            status_code=400,
            detail=f"File purpose '{training_file['purpose']}' is not valid for fine-tuning.",
        )

    hyperparams = body.hyperparameters or Hyperparams()
    job_id = f"ftjob-{uuid.uuid4().hex[:10]}"
    now = int(time.time())
    model_name = body.model.replace("/", "-")
    if body.suffix:
        model_name += f"-{body.suffix}"
    fine_tuned_model = f"{model_name}:{job_id}"

    # Estimate finish time (rough: 1 hour per epoch)
    estimated_finish = now + hyperparams.n_epochs * 3600

    job = {
        "id": job_id,
        "model": body.model,
        "created_at": now,
        "finished_at": None,
        "fine_tuned_model": fine_tuned_model,
        "hyperparameters": {
            "n_epochs": hyperparams.n_epochs,
            "batch_size": hyperparams.batch_size,
            "learning_rate_multiplier": hyperparams.learning_rate_multiplier,
        },
        "status": FineTuningStatus.validating_files.value,
        "training_file": body.training_file,
        "validation_file": body.validation_file,
        "error": None,
        "estimated_finish": estimated_finish,
        "integrations": body.integrations,
        "seed": body.seed,
    }
    _jobs[job_id] = job

    # Start async training
    task = asyncio.create_task(_run_fine_tuning(job_id))
    _job_tasks[job_id] = task

    return FineTuningJob(**job)


@router.get("/v1/fine_tuning/jobs/{job_id}", response_model=FineTuningJob)
async def get_fine_tuning_job(job_id: str):
    """Get fine-tuning job status."""
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    return FineTuningJob(**job)


@router.get("/v1/fine_tuning/jobs", response_model=FineTuningJobList)
async def list_fine_tuning_jobs(limit: int = 20, after: Optional[str] = None, model: Optional[str] = None):
    """List fine-tuning jobs."""
    jobs = list(_jobs.values())

    if model:
        jobs = [j for j in jobs if j["model"] == model]

    if after:
        idx = next((i for i, j in enumerate(jobs) if j["id"] == after), -1)
        jobs = jobs[idx + 1:] if idx >= 0 else []

    jobs = jobs[:limit]
    data = [FineTuningJob(**j) for j in jobs]

    return FineTuningJobList(data=data, has_more=len(jobs) > limit)


@router.post("/v1/fine_tuning/jobs/{job_id}/cancel", response_model=FineTuningJob)
async def cancel_fine_tuning_job(job_id: str):
    """Cancel a fine-tuning job."""
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

    if job["status"] in (FineTuningStatus.succeeded.value, FineTuningStatus.failed.value, FineTuningStatus.cancelled.value):
        raise HTTPException(status_code=400, detail=f"Cannot cancel job in state: {job['status']}")

    job["status"] = FineTuningStatus.cancelled.value
    job["finished_at"] = int(time.time())

    task = _job_tasks.get(job_id)
    if task and not task.done():
        task.cancel()

    return FineTuningJob(**job)


@router.get("/v1/fine_tuning/jobs/{job_id}/events")
async def list_fine_tuning_job_events(job_id: str, after: Optional[str] = None, limit: int = 20):
    """Get events for a fine-tuning job."""
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

    # Generate events based on job status
    events = []
    base_event = {
        "object": "fine_tuning.job.event",
        "created_at": job["created_at"],
        "level": "info",
        "data": {},
    }

    events.append(FineTuningJobEvent(
        **base_event,
        id=f"ftevent-{uuid.uuid4().hex[:8]}",
        message="Job created and queued for processing.",
        type="message",
    ))

    if job["status"] in ("running", "succeeded"):
        events.append(FineTuningJobEvent(
            **base_event,
            id=f"ftevent-{uuid.uuid4().hex[:8]}",
            created_at=job["created_at"] + 60,
            message="Validating training file...",
            type="message",
        ))

    if job["status"] in ("running", "succeeded"):
        events.append(FineTuningJobEvent(
            **base_event,
            id=f"ftevent-{uuid.uuid4().hex[:8]}",
            created_at=job["created_at"] + 120,
            message=f"Training started with {job['hyperparameters'].get('n_epochs', 3)} epochs.",
            type="message",
        ))

    if job["status"] == "succeeded":
        events.append(FineTuningJobEvent(
            **base_event,
            id=f"ftevent-{uuid.uuid4().hex[:8]}",
            created_at=job["finished_at"] or job["created_at"],
            message=f"Training completed. Fine-tuned model: {job['fine_tuned_model']}",
            type="message",
        ))

    if job["status"] == "failed" and job.get("error"):
        events.append(FineTuningJobEvent(
            **base_event,
            id=f"ftevent-{uuid.uuid4().hex[:8]}",
            created_at=job["finished_at"] or job["created_at"],
            level="error",
            message=job["error"].get("message", "Training failed"),
            type="error",
        ))

    if after:
        idx = next((i for i, e in enumerate(events) if e.id == after), -1)
        events = events[idx + 1:] if idx >= 0 else []

    events = events[:limit]

    return {"object": "list", "data": events, "has_more": False}


async def _run_fine_tuning(job_id: str):
    """Run the fine-tuning process asynchronously."""
    job = _jobs[job_id]
    coord = g.coordinator

    try:
        # Phase 1: Validate files
        job["status"] = FineTuningStatus.validating_files.value
        await asyncio.sleep(1)  # Simulate validation time

        # Phase 2: Queue
        job["status"] = FineTuningStatus.queued.value
        await asyncio.sleep(1)

        # Phase 3: Train
        job["status"] = FineTuningStatus.running.value

        # Get training file path
        from .files import get_file_path
        file_path = get_file_path(job["training_file"])
        if file_path is None or not file_path.exists():
            raise ValueError(f"Training file not found: {job['training_file']}")

        # Count training examples
        with open(file_path, 'r') as f:
            num_examples = sum(1 for line in f if line.strip())

        # In production, this would run actual training:
        # from transformers import Trainer, TrainingArguments
        # training_args = TrainingArguments(
        #     output_dir=f"/tmp/finetune/{job_id}",
        #     num_train_epochs=job["hyperparameters"]["n_epochs"],
        #     ...
        # )
        # trainer = Trainer(model=..., args=training_args, train_dataset=...)
        # trainer.train()

        # Simulate training time
        n_epochs = job["hyperparameters"].get("n_epochs", 3)
        await asyncio.sleep(n_epochs * 2)  # 2 seconds per epoch (simulated)

        # Success
        job["status"] = FineTuningStatus.succeeded.value
        job["finished_at"] = int(time.time())
        job["trained_tokens"] = num_examples * n_epochs * 100  # Rough estimate

    except asyncio.CancelledError:
        job["status"] = FineTuningStatus.cancelled.value
        job["finished_at"] = int(time.time())
    except Exception as e:
        job["status"] = FineTuningStatus.failed.value
        job["finished_at"] = int(time.time())
        job["error"] = {"message": str(e), "code": "training_error"}
