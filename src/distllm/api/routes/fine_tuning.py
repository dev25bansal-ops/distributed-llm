"""Fine-tuning: POST /v1/fine_tuning/jobs.

OpenAI-compatible fine-tuning job management.
"""

import asyncio
import time
import uuid
from enum import Enum

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..api_state import g
from ..persistent_store import get_data_dir, get_store

router = APIRouter(tags=["fine-tuning"])

# Persistent storage via SQLite
_store = get_store()

# In-memory task tracking (lost on restart, but jobs persist)
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
    batch_size: int | None = Field(default=None, ge=1)
    learning_rate_multiplier: float = Field(default=1.0, ge=0.1, le=10.0)


class FineTuningRequest(BaseModel):
    model: str = Field(..., description="Base model to fine-tune")
    training_file: str = Field(..., description="ID of uploaded training file")
    validation_file: str | None = Field(default=None, description="ID of validation file")
    hyperparameters: Hyperparams | None = Field(default=None)
    suffix: str | None = Field(default=None, description="Model name suffix")
    integrations: list[dict] | None = Field(default=None)
    seed: int | None = Field(default=None)


class FineTuningJobEvent(BaseModel):
    object: str = "fine_tuning.job.event"
    id: str
    created_at: int
    level: str
    message: str
    data: dict | None = None
    type: str


class FineTuningJob(BaseModel):
    id: str
    object: str = "fine_tuning.job"
    model: str
    created_at: int
    finished_at: int | None = None
    fine_tuned_model: str | None = None
    hyperparameters: dict
    organization_id: str | None = None
    result_files: list[str] = Field(default_factory=list)
    status: str
    trained_tokens: int | None = None
    training_file: str
    validation_file: str | None = None
    error: dict | None = None
    estimated_finish: int | None = None
    integrations: list[dict] | None = None
    seed: int | None = None


class FineTuningJobList(BaseModel):
    object: str = "list"
    data: list[FineTuningJob]
    has_more: bool = False


@router.post(
    "/v1/fine_tuning/jobs",
    response_model=FineTuningJob,
    summary="Create fine-tuning job",
    description="Create a new fine-tuning job to train a model on custom data. Requires a previously uploaded training file in JSONL format. Supports configurable hyperparameters (epochs, batch size, learning rate), validation files, and model naming suffixes.",
    response_description="Created fine-tuning job with status and metadata",
    responses={
        400: {"description": "Invalid file purpose or file content"},
        404: {"description": "Training or validation file not found"},
        503: {"description": "No model loaded"},
    },
)
async def create_fine_tuning_job(body: FineTuningRequest):
    """Create a fine-tuning job."""
    coord = g.coordinator
    if coord is None:
        raise HTTPException(status_code=503, detail="No model loaded")

    # Verify training file exists
    from .files import _store as file_store
    training_file = file_store.get_file(body.training_file)
    if training_file is None:
        raise HTTPException(
            status_code=404,
            detail=f"Training file '{body.training_file}' not found. Upload via POST /v1/files first.",
        )

    if training_file["purpose"] not in ("fine-tune", "batch"):
        raise HTTPException(
            status_code=400,
            detail=f"File purpose '{training_file['purpose']}' is not valid for fine-tuning.",
        )

    if body.validation_file is not None:
        validation_file = file_store.get_file(body.validation_file)
        if validation_file is None:
            raise HTTPException(
                status_code=404,
                detail=f"Validation file '{body.validation_file}' not found. Upload via POST /v1/files first.",
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
    _store.save_fine_tuning_job(job_id, job)

    # Start async training
    task = asyncio.create_task(_run_fine_tuning(job_id))
    _job_tasks[job_id] = task

    return FineTuningJob(**job)


@router.get(
    "/v1/fine_tuning/jobs/{job_id}",
    response_model=FineTuningJob,
    summary="Get fine-tuning job",
    description="Get the current status and metadata of a fine-tuning job by its ID. Reports progress, trained tokens, and completion status.",
    response_description="Fine-tuning job with status and progress",
    responses={
        404: {"description": "Job not found"},
    },
)
async def get_fine_tuning_job(job_id: str):
    """Get fine-tuning job status."""
    job = _store.get_fine_tuning_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    return FineTuningJob(**job)


@router.get(
    "/v1/fine_tuning/jobs",
    response_model=FineTuningJobList,
    summary="List fine-tuning jobs",
    description="List all fine-tuning jobs with optional filtering by model name and cursor-based pagination using the after parameter.",
    response_description="Paginated list of fine-tuning jobs",
)
async def list_fine_tuning_jobs(limit: int = 20, after: str | None = None, model: str | None = None):
    """List fine-tuning jobs."""
    jobs = _store.list_fine_tuning_jobs(limit=limit + 1)

    if model:
        jobs = [j for j in jobs if j["model"] == model]

    if after:
        idx = next((i for i, j in enumerate(jobs) if j["id"] == after), -1)
        jobs = jobs[idx + 1:] if idx >= 0 else []

    has_more = len(jobs) > limit
    jobs = jobs[:limit]
    data = [FineTuningJob(**j) for j in jobs]

    return FineTuningJobList(data=data, has_more=has_more)


@router.post(
    "/v1/fine_tuning/jobs/{job_id}/cancel",
    response_model=FineTuningJob,
    summary="Cancel fine-tuning job",
    description="Cancel a running or queued fine-tuning job. Cannot cancel jobs that have already succeeded, failed, or been cancelled.",
    response_description="Cancelled fine-tuning job",
    responses={
        400: {"description": "Cannot cancel job in current state"},
        404: {"description": "Job not found"},
    },
)
async def cancel_fine_tuning_job(job_id: str):
    """Cancel a fine-tuning job."""
    job = _store.get_fine_tuning_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

    if job["status"] in (FineTuningStatus.succeeded.value, FineTuningStatus.failed.value, FineTuningStatus.cancelled.value):
        raise HTTPException(status_code=400, detail=f"Cannot cancel job in state: {job['status']}")

    job["status"] = FineTuningStatus.cancelled.value
    job["finished_at"] = int(time.time())
    _store.save_fine_tuning_job(job_id, job)

    task = _job_tasks.get(job_id)
    if task and not task.done():
        task.cancel()

    return FineTuningJob(**job)


@router.get(
    "/v1/fine_tuning/jobs/{job_id}/events",
    summary="List fine-tuning events",
    description="Get the event stream for a fine-tuning job. Events include validation progress, training start/end status, and error messages. Supports cursor-based pagination.",
    response_description="Paginated list of fine-tuning events",
    responses={
        404: {"description": "Job not found"},
    },
)
async def list_fine_tuning_job_events(job_id: str, after: str | None = None, limit: int = 20):
    """Get events for a fine-tuning job."""
    job = _store.get_fine_tuning_job(job_id)
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
    job = _store.get_fine_tuning_job(job_id)
    if job is None:
        return
    coord = g.coordinator

    try:
        # Phase 1: Validate files
        job["status"] = FineTuningStatus.validating_files.value
        _store.save_fine_tuning_job(job_id, job)

        # Phase 2: Queue
        job["status"] = FineTuningStatus.queued.value
        _store.save_fine_tuning_job(job_id, job)

        # Phase 3: Train
        job["status"] = FineTuningStatus.running.value
        _store.save_fine_tuning_job(job_id, job)

        # Get training file path
        from .files import get_file_path
        file_path = get_file_path(job["training_file"])
        if file_path is None or not file_path.exists():
            raise ValueError(f"Training file not found: {job['training_file']}")

        n_epochs = job["hyperparameters"].get("n_epochs", 3)
        learning_rate_multiplier = job["hyperparameters"].get("learning_rate_multiplier", 1.0)
        batch_size = job["hyperparameters"].get("batch_size") or 4

        trained_tokens = await _train_job(
            coord=coord,
            job=job,
            file_path=file_path,
            n_epochs=n_epochs,
            batch_size=batch_size,
            learning_rate_multiplier=learning_rate_multiplier,
        )

        # Success
        job["status"] = FineTuningStatus.succeeded.value
        job["finished_at"] = int(time.time())
        job["trained_tokens"] = trained_tokens
        _store.save_fine_tuning_job(job_id, job)

    except asyncio.CancelledError:
        job["status"] = FineTuningStatus.cancelled.value
        job["finished_at"] = int(time.time())
        _store.save_fine_tuning_job(job_id, job)
    except Exception as e:
        job["status"] = FineTuningStatus.failed.value
        job["finished_at"] = int(time.time())
        job["error"] = {"message": str(e), "code": "training_error"}
        _store.save_fine_tuning_job(job_id, job)


async def _train_job(
    coord,
    job: dict,
    file_path,
    n_epochs: int,
    batch_size: int,
    learning_rate_multiplier: float,
) -> int:
    """Run a real fine-tuning backend and return trained token count."""
    backend = getattr(coord, "fine_tuning_backend", None) or getattr(coord, "_fine_tuning_backend", None)
    if backend is not None:
        train = getattr(backend, "train", None)
        if train is None:
            raise RuntimeError("Configured fine-tuning backend does not expose train()")
        result = train(job=job, training_file=str(file_path))
        if asyncio.iscoroutine(result):
            result = await result
        return int(result.get("trained_tokens", 0)) if isinstance(result, dict) else int(result or 0)

    local_partitioner = getattr(coord, "local_partitioner", None)
    full_model = getattr(local_partitioner, "full_model", None)
    if full_model is None:
        raise RuntimeError(
            "Fine-tuning backend unavailable: load a local model or configure coord.fine_tuning_backend"
        )

    return await asyncio.to_thread(
        _train_with_transformers,
        coord,
        full_model,
        file_path,
        job,
        n_epochs,
        batch_size,
        learning_rate_multiplier,
    )


def _train_with_transformers(
    coord,
    full_model,
    file_path,
    job: dict,
    n_epochs: int,
    batch_size: int,
    learning_rate_multiplier: float,
) -> int:
    """Fine-tune a local Transformers model."""
    try:
        import torch
        from torch.nn.utils.rnn import pad_sequence
        from torch.utils.data import Dataset
        from transformers import AutoTokenizer, Trainer, TrainingArguments
    except ImportError as exc:
        raise RuntimeError(
            "Fine-tuning requires torch and transformers to be installed"
        ) from exc

    class TextDataset(Dataset):
        def __init__(self, path, tokenizer, max_len=512):
            self.examples = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    tokens = tokenizer.encode(line, truncation=True, max_length=max_len)
                    if tokens:
                        self.examples.append(torch.tensor(tokens, dtype=torch.long))

        def __len__(self):
            return len(self.examples)

        def __getitem__(self, idx):
            return self.examples[idx]

    tokenizer = getattr(coord, "tokenizer", None)
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(coord.model_name)
    if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token", None):
        tokenizer.pad_token = tokenizer.eos_token

    dataset = TextDataset(file_path, tokenizer)
    if len(dataset) == 0:
        raise RuntimeError("Training file contains no tokenizable examples")

    pad_token_id = getattr(tokenizer, "pad_token_id", 0) or 0

    def collate(batch):
        padded = pad_sequence(batch, batch_first=True, padding_value=pad_token_id)
        labels = padded.clone()
        labels[padded == pad_token_id] = -100
        return {"input_ids": padded, "labels": labels}

    output_dir = get_data_dir() / "fine_tuning" / job["id"]
    output_dir.mkdir(parents=True, exist_ok=True)
    base_lr = 5e-5
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=n_epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=base_lr * learning_rate_multiplier,
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
    )
    trainer = Trainer(
        model=full_model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collate,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    return int(sum(len(example) for example in dataset.examples) * n_epochs)
