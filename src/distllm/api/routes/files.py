"""Files API — upload, list, retrieve, and delete files.

OpenAI-compatible Files API for fine-tuning data, batch inputs, and
assistant-style workflows.  File content is stored on disk under the
configured data directory; metadata is persisted in SQLite.

Usage::

    POST /v1/files              — Upload a file (multipart/form-data)
    GET  /v1/files              — List uploaded files
    GET  /v1/files/{file_id}    — Retrieve file metadata
    GET  /v1/files/{file_id}/content  — Download file content
    DELETE /v1/files/{file_id}  — Delete a file
"""

from __future__ import annotations

import shutil
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile
from fastapi.params import File, Form, Query
from fastapi.responses import FileResponse
from loguru import logger
from pydantic import BaseModel

from ..auth_deps import require_coordinator
from ..persistent_store import get_data_dir, get_store

router = APIRouter(prefix="/v1/files", tags=["files"], dependencies=[Depends(require_coordinator)])


# ── Pydantic models ─────────────────────────────────────────────────────


class FileObject(BaseModel):
    """OpenAI-compatible file object."""
    id: str
    object: str = "file"
    bytes: int = 0
    created_at: int = 0
    filename: str = ""
    purpose: str = ""
    status: str = "uploaded"


class FileListResponse(BaseModel):
    object: str = "list"
    data: list[FileObject]


class FileDeleteResponse(BaseModel):
    id: str
    object: str = "file"
    deleted: bool = True


# ── Helpers ──────────────────────────────────────────────────────────────


_MAX_FILE_SIZE = 512 * 1024 * 1024  # 512 MB
_ALLOWED_PURPOSES = {"fine-tune", "batch", "assistants", "vision"}


def _ensure_upload_dir() -> Path:
    """Return the file upload directory, creating it if necessary."""
    data_dir = Path(get_data_dir())
    upload_dir = data_dir / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    return upload_dir


def _file_metadata_to_object(meta: dict) -> FileObject:
    """Convert a stored file metadata dict to a FileObject."""
    return FileObject(
        id=meta.get("id", ""),
        object="file",
        bytes=meta.get("bytes", 0),
        created_at=int(meta.get("created_at", time.time())),
        filename=meta.get("filename", ""),
        purpose=meta.get("purpose", ""),
        status=meta.get("status", "uploaded"),
    )


# ── Endpoints ────────────────────────────────────────────────────────────


@router.post(
    "",
    summary="Upload a file",
    description="Upload a file for fine-tuning, batch processing, or assistant usage. "
                "Accepts multipart/form-data with the file content and purpose.",
    response_model=FileObject,
    status_code=201,
)
async def upload_file(
    file: UploadFile = File(..., description="File to upload"),
    purpose: str = Form("batch", description="Purpose: fine-tune, batch, assistants, vision"),
):
    """Upload a file to the server.

    The file content is written to ``{data_dir}/uploads/{file_id}/{filename}``
    and the metadata is recorded in the persistent store.
    """
    # Validate purpose
    if purpose not in _ALLOWED_PURPOSES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid purpose '{purpose}'. Allowed: {sorted(_ALLOWED_PURPOSES)}",
        )

    # Validate file
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")

    # Read content (with size limit)
    try:
        content = await file.read()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to read file: {exc}")

    if len(content) > _MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds maximum size of {_MAX_FILE_SIZE // (1024*1024)} MB",
        )

    file_id = f"file-{uuid.uuid4().hex[:24]}"
    now = time.time()
    upload_dir = _ensure_upload_dir() / file_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    file_path = upload_dir / (file.filename or "unnamed")

    try:
        file_path.write_bytes(content)
    except OSError as exc:
        logger.error(f"Failed to write uploaded file {file_id}: {exc}")
        raise HTTPException(status_code=500, detail="Failed to store file")

    metadata = {
        "id": file_id,
        "bytes": len(content),
        "created_at": now,
        "filename": file.filename,
        "purpose": purpose,
        "status": "uploaded",
        "path": str(file_path),
        "object": "file",
    }

    store = get_store()
    store.save_file(file_id, metadata)
    logger.info(f"File uploaded: {file_id} ({file.filename}, {len(content)} bytes, purpose={purpose})")

    return _file_metadata_to_object(metadata)


@router.get(
    "",
    summary="List files",
    description="List all uploaded files, optionally filtered by purpose.",
    response_model=FileListResponse,
)
async def list_files(
    purpose: str | None = Query(None, description="Filter by purpose: fine-tune, batch, assistants, vision"),
):
    """List uploaded files, optionally filtered by purpose."""
    store = get_store()
    files = store.list_files(purpose=purpose)
    return FileListResponse(
        data=[_file_metadata_to_object(f) for f in files],
    )


@router.get(
    "/{file_id}",
    summary="Retrieve file metadata",
    description="Return metadata for a specific uploaded file.",
    response_model=FileObject,
)
async def get_file(file_id: str):
    """Retrieve metadata for a specific file."""
    store = get_store()
    meta = store.get_file(file_id)
    if meta is None:
        raise HTTPException(status_code=404, detail=f"File '{file_id}' not found")
    return _file_metadata_to_object(meta)


@router.get(
    "/{file_id}/content",
    summary="Download file content",
    description="Download the raw content of an uploaded file.",
    response_class=FileResponse,
)
async def get_file_content(file_id: str):
    """Download the raw content of an uploaded file."""
    store = get_store()
    meta = store.get_file(file_id)
    if meta is None:
        raise HTTPException(status_code=404, detail=f"File '{file_id}' not found")

    file_path = Path(meta["path"])
    if not file_path.exists():
        logger.error(f"File content missing: {file_id} (expected at {file_path})")
        raise HTTPException(status_code=500, detail="File content not found on disk")

    return FileResponse(
        path=str(file_path),
        filename=meta.get("filename", "unnamed"),
        media_type="application/octet-stream",
    )


@router.delete(
    "/{file_id}",
    summary="Delete a file",
    description="Delete a file and its content from the server.",
    response_model=FileDeleteResponse,
)
async def delete_file(file_id: str):
    """Delete a file and its content from disk."""
    store = get_store()
    meta = store.get_file(file_id)
    if meta is None:
        raise HTTPException(status_code=404, detail=f"File '{file_id}' not found")

    # Remove from disk
    file_path = Path(meta["path"])
    if file_path.exists():
        file_path.unlink()
    # Remove parent directory (the file-id folder)
    parent = file_path.parent
    if parent.exists():
        shutil.rmtree(parent, ignore_errors=True)

    store.delete_file(file_id)
    logger.info(f"File deleted: {file_id} ({meta.get('filename', 'unnamed')})")

    return FileDeleteResponse(id=file_id)
