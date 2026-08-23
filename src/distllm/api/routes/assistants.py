"""OpenAI-compatible Assistants API — threads, runs, messages, vector stores.

Endpoints:

- ``POST /v1/threads`` — create conversation thread
- ``GET /v1/threads/{id}`` — retrieve thread
- ``DELETE /v1/threads/{id}`` — delete thread (cascades to messages + runs)
- ``POST /v1/threads/{id}/messages`` — add message to thread
- ``GET /v1/threads/{id}/messages`` — list thread messages
- ``POST /v1/threads/{id}/runs`` — create a run (invoke coordinator on thread)
- ``GET /v1/threads/{id}/runs/{run_id}`` — get run status
- ``POST /v1/vector_stores`` — create vector store
- ``GET /v1/vector_stores/{id}`` — get vector store details
- ``DELETE /v1/vector_stores/{id}`` — delete vector store
- ``POST /v1/vector_stores/{id}/file_search`` — search within vector store

Backed by ``PersistentStore`` for durability and the coordinator for inference.
"""

from __future__ import annotations

import time
import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from loguru import logger

from ..api_state import g
from ..auth_deps import require_role
from ..persistent_store import get_store

router = APIRouter(tags=["assistants"], prefix="/v1",
                   dependencies=[Depends(require_role("inference-only"))])


# ── Pydantic models ─────────────────────────────────────────────────────────


class ThreadCreateRequest(BaseModel):
    messages: list[dict] = Field(default_factory=list, description="Initial messages")
    metadata: dict[str, Any] = Field(default_factory=dict)


class ThreadResponse(BaseModel):
    id: str
    object: str = "thread"
    created_at: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class MessageRequest(BaseModel):
    role: str = Field("user", pattern=r"^(user|assistant|system)$")
    content: str = Field(..., min_length=1, max_length=131072)
    metadata: dict[str, Any] = Field(default_factory=dict)


class MessageResponse(BaseModel):
    id: str
    object: str = "thread.message"
    created_at: int = 0
    thread_id: str
    role: str
    content: list[dict]


class RunCreateRequest(BaseModel):
    assistant_id: str = Field(..., description="Assistant model to run")
    instructions: str = Field("", max_length=65536)
    model: str = Field("", description="Override model name")
    temperature: float | None = Field(None, ge=0.0, le=2.0)
    max_completion_tokens: int | None = Field(None, ge=1, le=524288)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RunResponse(BaseModel):
    id: str
    object: str = "thread.run"
    created_at: int = 0
    thread_id: str
    assistant_id: str
    status: str  # queued | in_progress | completed | failed | cancelled
    model: str = ""
    instructions: str = ""
    temperature: float | None = None
    max_completion_tokens: int | None = None
    completed_at: int | None = None
    failed_at: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class VectorStoreCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=256)
    metadata: dict[str, Any] = Field(default_factory=dict)


class VectorStoreResponse(BaseModel):
    id: str
    object: str = "vector_store"
    name: str = ""
    created_at: int = 0
    file_count: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class FileSearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4096)
    max_results: int = Field(5, ge=1, le=50)


class FileSearchResponse(BaseModel):
    object: str = "list"
    data: list[dict] = Field(default_factory=list)


# ── Helpers ─────────────────────────────────────────────────────────────────


def _now() -> int:
    return int(time.time())


def _coord() -> Any:
    """Resolve the coordinator or raise 503."""
    coord = g.coordinator
    if coord is None:
        raise HTTPException(status_code=503, detail="Coordinator not available")
    return coord


# ── Thread endpoints ────────────────────────────────────────────────────────


@router.post("/threads", response_model=ThreadResponse, status_code=201)
async def create_thread(body: ThreadCreateRequest):
    """Create a conversation thread.

    Optionally seeds the thread with initial messages.
    """
    store = get_store()
    thread_id = f"thread_{uuid.uuid4().hex[:16]}"
    now = _now()
    data = {
        "id": thread_id,
        "object": "thread",
        "created_at": now,
        "metadata": body.metadata,
    }
    store.save_thread(thread_id, data)

    # Seed initial messages
    for msg in body.messages[:50]:
        msg_id = f"msg_{uuid.uuid4().hex[:12]}"
        msg_data = {
            "id": msg_id,
            "object": "thread.message",
            "created_at": now,
            "thread_id": thread_id,
            "role": msg.get("role", "user"),
            "content": [{"type": "text", "text": msg.get("content", "")}],
        }
        store.save_message(msg_id, thread_id, msg_data)

    logger.info(f"Thread created: {thread_id}")
    return ThreadResponse(id=thread_id, created_at=now, metadata=body.metadata)


@router.get("/threads/{thread_id}", response_model=ThreadResponse)
async def get_thread(thread_id: str):
    """Retrieve a thread by ID."""
    store = get_store()
    data = store.get_thread(thread_id)
    if data is None:
        raise HTTPException(status_code=404, detail="Thread not found")
    return ThreadResponse(
        id=data["id"],
        created_at=data.get("created_at", 0),
        metadata=data.get("metadata", {}),
    )


@router.delete("/threads/{thread_id}")
async def delete_thread(thread_id: str):
    """Delete a thread and all its messages and runs."""
    store = get_store()
    if not store.delete_thread(thread_id):
        raise HTTPException(status_code=404, detail="Thread not found")
    return {"id": thread_id, "object": "thread.deleted", "deleted": True}


@router.get("/threads", response_model=list[ThreadResponse])
async def list_threads(limit: int = 20):
    """List all threads."""
    store = get_store()
    threads = store.list_threads(limit=limit)
    return [
        ThreadResponse(id=t["id"], created_at=t.get("created_at", 0), metadata=t.get("metadata", {}))
        for t in threads
    ]


# ── Message endpoints ───────────────────────────────────────────────────────


@router.post("/threads/{thread_id}/messages", response_model=MessageResponse, status_code=201)
async def create_message(thread_id: str, body: MessageRequest):
    """Add a message to a thread."""
    store = get_store()
    if store.get_thread(thread_id) is None:
        raise HTTPException(status_code=404, detail="Thread not found")

    msg_id = f"msg_{uuid.uuid4().hex[:12]}"
    now = _now()
    msg_data = {
        "id": msg_id,
        "object": "thread.message",
        "created_at": now,
        "thread_id": thread_id,
        "role": body.role,
        "content": [{"type": "text", "text": body.content}],
        "metadata": body.metadata,
    }
    store.save_message(msg_id, thread_id, msg_data)
    return MessageResponse(
        id=msg_id, created_at=now, thread_id=thread_id,
        role=body.role, content=msg_data["content"],
    )


@router.get("/threads/{thread_id}/messages", response_model=list[MessageResponse])
async def list_messages(thread_id: str, limit: int = 50):
    """List messages in a thread (chronological order)."""
    store = get_store()
    if store.get_thread(thread_id) is None:
        raise HTTPException(status_code=404, detail="Thread not found")
    msgs = store.list_messages(thread_id, limit=limit)
    return [
        MessageResponse(
            id=m["id"], created_at=m.get("created_at", 0),
            thread_id=thread_id, role=m.get("role", "user"),
            content=m.get("content", []),
        )
        for m in msgs
    ]


# ── Run endpoints ───────────────────────────────────────────────────────────


@router.post("/threads/{thread_id}/runs", response_model=RunResponse, status_code=201)
async def create_run(thread_id: str, body: RunCreateRequest):
    """Create a run — invokes the coordinator on the thread's messages.

    The run is created in ``queued`` status, then executed synchronously
    (for now — future versions will use a background worker).
    """
    store = get_store()
    thread = store.get_thread(thread_id)
    if thread is None:
        raise HTTPException(status_code=404, detail="Thread not found")

    coord = _coord()
    run_id = f"run_{uuid.uuid4().hex[:12]}"
    now = _now()

    # Collect thread messages as context
    messages = store.list_messages(thread_id, limit=50)
    conversation = "\n".join(
        f"{m.get('role', 'user')}: {m.get('content', [{}])[0].get('text', '') if isinstance(m.get('content'), list) else m.get('content', '')}"
        for m in messages
    )

    run_data = {
        "id": run_id,
        "object": "thread.run",
        "created_at": now,
        "thread_id": thread_id,
        "assistant_id": body.assistant_id,
        "status": "in_progress",
        "model": body.model or coord.model_name,
        "instructions": body.instructions,
        "temperature": body.temperature,
        "max_completion_tokens": body.max_completion_tokens,
        "metadata": body.metadata,
    }
    store.save_run(run_id, thread_id, run_data)

    try:
        prompt = f"{body.instructions}\n\n{conversation}" if body.instructions else conversation
        result = await coord.generate_async(
            prompt=prompt,
            model=body.model or None,
            max_new_tokens=body.max_completion_tokens or 1024,
            temperature=body.temperature or 0.7,
        )

        # Save assistant response as a message
        resp_id = f"msg_{uuid.uuid4().hex[:12]}"
        resp_msg = {
            "id": resp_id,
            "object": "thread.message",
            "created_at": _now(),
            "thread_id": thread_id,
            "role": "assistant",
            "content": [{"type": "text", "text": str(result)}],
        }
        store.save_message(resp_id, thread_id, resp_msg)

        completed_at = _now()
        run_data["status"] = "completed"
        run_data["completed_at"] = completed_at
        store.update_run(run_id, thread_id, {"status": "completed", "completed_at": completed_at})
    except Exception as e:
        failed_at = _now()
        run_data["status"] = "failed"
        run_data["failed_at"] = failed_at
        store.update_run(run_id, thread_id, {"status": "failed", "failed_at": failed_at, "error": str(e)})
        logger.error(f"Run {run_id} failed: {e}")

    return RunResponse(
        id=run_id, created_at=now, thread_id=thread_id,
        assistant_id=body.assistant_id, status=run_data["status"],
        model=run_data["model"], instructions=body.instructions,
        temperature=body.temperature,
        max_completion_tokens=body.max_completion_tokens,
        completed_at=run_data.get("completed_at"),
        failed_at=run_data.get("failed_at"),
        metadata=body.metadata,
    )


@router.get("/threads/{thread_id}/runs/{run_id}", response_model=RunResponse)
async def get_run(thread_id: str, run_id: str):
    """Get the status of a run."""
    store = get_store()
    run = store.get_run(run_id, thread_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return RunResponse(
        id=run["id"], created_at=run.get("created_at", 0),
        thread_id=thread_id, assistant_id=run.get("assistant_id", ""),
        status=run.get("status", "unknown"),
        model=run.get("model", ""),
        instructions=run.get("instructions", ""),
        temperature=run.get("temperature"),
        max_completion_tokens=run.get("max_completion_tokens"),
        completed_at=run.get("completed_at"),
        failed_at=run.get("failed_at"),
        metadata=run.get("metadata", {}),
    )


@router.get("/threads/{thread_id}/runs", response_model=list[RunResponse])
async def list_runs(thread_id: str, limit: int = 20):
    """List all runs for a thread."""
    store = get_store()
    if store.get_thread(thread_id) is None:
        raise HTTPException(status_code=404, detail="Thread not found")
    runs = store.list_runs(thread_id, limit=limit)
    return [
        RunResponse(
            id=r["id"], created_at=r.get("created_at", 0),
            thread_id=thread_id, assistant_id=r.get("assistant_id", ""),
            status=r.get("status", "unknown"), model=r.get("model", ""),
            instructions=r.get("instructions", ""),
            temperature=r.get("temperature"),
            max_completion_tokens=r.get("max_completion_tokens"),
            completed_at=r.get("completed_at"), failed_at=r.get("failed_at"),
            metadata=r.get("metadata", {}),
        )
        for r in runs
    ]


# ── Vector store endpoints ──────────────────────────────────────────────────


@router.post("/vector_stores", response_model=VectorStoreResponse, status_code=201)
async def create_vector_store(body: VectorStoreCreateRequest):
    """Create a vector store for RAG."""
    store = get_store()
    vs_id = f"vs_{uuid.uuid4().hex[:16]}"
    now = _now()
    data = {
        "id": vs_id,
        "object": "vector_store",
        "name": body.name,
        "created_at": now,
        "file_ids": [],
        "metadata": body.metadata,
    }
    store.save_vector_store(vs_id, data)
    logger.info(f"Vector store created: {vs_id} ({body.name})")
    return VectorStoreResponse(id=vs_id, name=body.name, created_at=now, metadata=body.metadata)


@router.get("/vector_stores/{vs_id}", response_model=VectorStoreResponse)
async def get_vector_store(vs_id: str):
    """Get vector store details."""
    store = get_store()
    data = store.get_vector_store(vs_id)
    if data is None:
        raise HTTPException(status_code=404, detail="Vector store not found")
    return VectorStoreResponse(
        id=data["id"], name=data.get("name", ""),
        created_at=data.get("created_at", 0),
        file_count=len(data.get("file_ids", [])),
        metadata=data.get("metadata", {}),
    )


@router.delete("/vector_stores/{vs_id}")
async def delete_vector_store(vs_id: str):
    """Delete a vector store."""
    store = get_store()
    if not store.delete_vector_store(vs_id):
        raise HTTPException(status_code=404, detail="Vector store not found")
    return {"id": vs_id, "object": "vector_store.deleted", "deleted": True}


@router.post("/vector_stores/{vs_id}/file_search", response_model=FileSearchResponse)
async def vector_store_file_search(vs_id: str, body: FileSearchRequest):
    """Search within a vector store using the coordinator's retriever."""
    store = get_store()
    vs_data = store.get_vector_store(vs_id)
    if vs_data is None:
        raise HTTPException(status_code=404, detail="Vector store not found")

    coord = _coord()
    try:
        results = await coord.search(
            query=body.query,
            collection=vs_id,
            top_k=body.max_results,
        )
    except (AttributeError, Exception) as e:
        # Fallback: return empty results if retriever not available
        logger.debug(f"Vector search not available: {e}")
        results = []

    return FileSearchResponse(data=results if isinstance(results, list) else [])
