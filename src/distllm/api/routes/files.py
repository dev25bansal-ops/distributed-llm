"""File management: POST /v1/files, GET /v1/files/{file_id}.

OpenAI-compatible file upload endpoint for fine-tuning and RAG.
"""

import json
import os
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, Field
from loguru import logger

from ..api_state import g
from ..persistent_store import get_data_dir, get_store

router = APIRouter(tags=["files"])

# Persistent storage via SQLite
_store = get_store()


class FileObject(BaseModel):
    id: str
    object: str = "file"
    bytes: int
    created_at: int
    filename: str
    purpose: str
    status: str = "uploaded"
    status_details: str | None = None


class FileListResponse(BaseModel):
    object: str = "list"
    data: list[FileObject]


class FileDeleteResponse(BaseModel):
    id: str
    object: str = "file"
    deleted: bool


@router.post(
    "/v1/files",
    response_model=FileObject,
    summary="Upload file",
    description="Upload a file for use with fine-tuning, batch processing, or RAG. Supports purposes: fine-tune, fine-tune-results, batch, batch_results. Files up to 100 MB. JSONL validation is performed for fine-tuning uploads.",
    response_description="Uploaded file metadata with file ID",
    responses={
        400: {"description": "Invalid purpose, invalid JSONL format, or empty file"},
        413: {"description": "File too large (max 100 MB)"},
        503: {"description": "No model loaded"},
    },
)
async def create_file(
    file: UploadFile = File(...),
    purpose: str = Form(default="fine-tune"),
):
    """Upload a file for use with fine-tuning, RAG, or batch processing.

    Supported purposes: fine-tune, fine-tune-results, batch, batch_results.
    """
    coord = g.coordinator
    if coord is None:
        raise HTTPException(status_code=503, detail="No model loaded")

    # Validate purpose
    valid_purposes = {"fine-tune", "fine-tune-results", "batch", "batch_results"}
    if purpose not in valid_purposes:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid purpose: {purpose}. Must be one of: {', '.join(sorted(valid_purposes))}",
        )

    # Validate file size (100 MB limit)
    MAX_FILE_SIZE = 100 * 1024 * 1024
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large (max 100MB)")
    filename = _safe_filename(file.filename or f"upload_{uuid.uuid4().hex[:8]}")

    # For fine-tuning, validate JSONL format
    if purpose == "fine-tune":
        _validate_jsonl(content, filename)

    # Store file
    file_id = f"file-{uuid.uuid4().hex[:12]}"
    storage_path = _get_storage_path(file_id, filename)

    storage_path.parent.mkdir(parents=True, exist_ok=True)
    storage_path.write_bytes(content)

    file_obj = {
        "id": file_id,
        "bytes": len(content),
        "created_at": int(time.time()),
        "filename": filename,
        "purpose": purpose,
        "status": "uploaded",
        "storage_path": str(storage_path),
    }
    _store.save_file(file_id, file_obj)

    return FileObject(**file_obj)


@router.get(
    "/v1/files",
    response_model=FileListResponse,
    summary="List files",
    description="List all uploaded files with optional filtering by purpose and limit. Returns file metadata including size, purpose, and creation timestamp.",
    response_description="List of file metadata objects",
)
async def list_files(purpose: str | None = None, limit: int = 100):
    """List all uploaded files."""
    files = _store.list_files(purpose=purpose)
    files = files[:limit]
    data = [FileObject(**f) for f in files]

    return FileListResponse(data=data)


@router.get(
    "/v1/files/{file_id}",
    response_model=FileObject,
    summary="Get file info",
    description="Get detailed information about an uploaded file, including its size, purpose, status, and creation timestamp.",
    response_description="File metadata object",
    responses={
        404: {"description": "File not found"},
    },
)
async def get_file(file_id: str):
    """Get information about a file."""
    file_obj = _store.get_file(file_id)
    if not file_obj:
        raise HTTPException(status_code=404, detail=f"File '{file_id}' not found")
    return FileObject(**file_obj)


@router.delete(
    "/v1/files/{file_id}",
    response_model=FileDeleteResponse,
    summary="Delete file",
    description="Delete an uploaded file by its ID. Removes the file from disk and the persistent store.",
    response_description="Deletion confirmation with file ID",
    responses={
        404: {"description": "File not found"},
    },
)
async def delete_file(file_id: str):
    """Delete a file."""
    file_obj = _store.get_file(file_id)
    if not file_obj:
        raise HTTPException(status_code=404, detail=f"File '{file_id}' not found")

    # Delete from disk
    storage_path = _resolve_storage_path(file_id, file_obj)
    if storage_path.exists():
        storage_path.unlink()

    _store.delete_file(file_id)

    return FileDeleteResponse(id=file_id, deleted=True)


@router.get(
    "/v1/files/{file_id}/content",
    summary="Download file content",
    description="Download the raw content of an uploaded file. Returns the file as an octet-stream attachment with the original filename.",
    response_description="File content as binary download",
    responses={
        404: {"description": "File not found or content not found on disk"},
    },
)
async def get_file_content(file_id: str):
    """Download a file's content."""
    from fastapi.responses import FileResponse

    file_obj = _store.get_file(file_id)
    if not file_obj:
        raise HTTPException(status_code=404, detail=f"File '{file_id}' not found")

    storage_path = _resolve_storage_path(file_id, file_obj)
    if not storage_path.exists():
        raise HTTPException(status_code=404, detail="File content not found on disk")

    return FileResponse(
        path=str(storage_path),
        filename=file_obj["filename"],
        media_type="application/octet-stream",
    )


def _validate_jsonl(content: bytes, filename: str) -> None:
    """Validate that content is valid JSONL format."""
    try:
        text = content.decode('utf-8')
    except UnicodeDecodeError:
        raise HTTPException(
            status_code=400,
            detail="File must be UTF-8 encoded.",
        )

    lines = text.strip().split('\n')
    if not lines:
        raise HTTPException(
            status_code=400,
            detail="File is empty. Upload a JSONL file with at least one line.",
        )

    # Check first line for basic structure
    try:
        first_line = json.loads(lines[0])
        if not isinstance(first_line, dict):
            raise HTTPException(
                status_code=400,
                detail="Each line must be a JSON object (not an array or primitive).",
            )
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid JSON on line 1: {e}",
        )


def _safe_filename(filename: str) -> str:
    """Strip any client path components from an uploaded filename."""
    safe = Path(filename).name.strip()
    return safe or f"upload_{uuid.uuid4().hex[:8]}"


def _get_storage_path(file_id: str, filename: str) -> Path:
    """Get the storage path for a file."""
    base = Path(os.environ.get("DISTLLM_FILE_DIR", str(get_data_dir() / "files"))).expanduser()
    base.mkdir(parents=True, exist_ok=True)
    return base / f"{file_id}_{_safe_filename(filename)}"


def _resolve_storage_path(file_id: str, file_obj: dict) -> Path:
    """Resolve a file object's durable content path."""
    storage_path = file_obj.get("storage_path")
    if storage_path:
        return Path(storage_path)
    return _get_storage_path(file_id, file_obj["filename"])


def get_file_path(file_id: str) -> Path | None:
    """Get the storage path for a file by its ID.

    Args:
        file_id: The file ID (e.g., "file-abc123").

    Returns:
        Path to the file, or None if the file ID is not found.
    """
    file_obj = _store.get_file(file_id)
    if file_obj is None:
        return None
    return _resolve_storage_path(file_id, file_obj)
