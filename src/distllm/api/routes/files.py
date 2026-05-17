"""File management: POST /v1/files, GET /v1/files/{file_id}.

OpenAI-compatible file upload endpoint for fine-tuning and RAG.
"""

import os
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, Field

from ..api_state import g

router = APIRouter(tags=["files"])

# In-memory file registry (production: database)
_files: dict = {}


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


@router.post("/v1/files", response_model=FileObject)
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

    # Validate file size (max 512MB)
    MAX_FILE_SIZE = 512 * 1024 * 1024
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large (max 512MB)")
    filename = file.filename or f"upload_{uuid.uuid4().hex[:8]}"

    # Validate file size (100 MB limit)
    if len(content) > 100 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large. Maximum size is 100 MB.")

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
    }
    _files[file_id] = file_obj

    return FileObject(**file_obj)


@router.get("/v1/files", response_model=FileListResponse)
async def list_files(purpose: str | None = None, limit: int = 100):
    """List all uploaded files."""
    files = list(_files.values())

    if purpose:
        files = [f for f in files if f["purpose"] == purpose]

    files = files[:limit]
    data = [FileObject(**f) for f in files]

    return FileListResponse(data=data)


@router.get("/v1/files/{file_id}", response_model=FileObject)
async def get_file(file_id: str):
    """Get information about a file."""
    file_obj = _files.get(file_id)
    if not file_obj:
        raise HTTPException(status_code=404, detail=f"File '{file_id}' not found")
    return FileObject(**file_obj)


@router.delete("/v1/files/{file_id}", response_model=FileDeleteResponse)
async def delete_file(file_id: str):
    """Delete a file."""
    file_obj = _files.get(file_id)
    if not file_obj:
        raise HTTPException(status_code=404, detail=f"File '{file_id}' not found")

    # Delete from disk
    storage_path = _get_storage_path(file_id, file_obj["filename"])
    if storage_path.exists():
        storage_path.unlink()

    del _files[file_id]

    return FileDeleteResponse(id=file_id, deleted=True)


@router.get("/v1/files/{file_id}/content")
async def get_file_content(file_id: str):
    """Download a file's content."""
    from fastapi.responses import FileResponse

    file_obj = _files.get(file_id)
    if not file_obj:
        raise HTTPException(status_code=404, detail=f"File '{file_id}' not found")

    storage_path = _get_storage_path(file_id, file_obj["filename"])
    if not storage_path.exists():
        raise HTTPException(status_code=404, detail="File content not found on disk")

    return FileResponse(
        path=str(storage_path),
        filename=file_obj["filename"],
        media_type="application/octet-stream",
    )


def _validate_jsonl(content: bytes, filename: str) -> None:
    """Validate that content is valid JSONL format."""
    import json

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


def _get_storage_path(file_id: str, filename: str) -> Path:
    """Get the storage path for a file."""
    base = Path(os.environ.get("DISTLLM_FILE_DIR", "/tmp/distllm/files"))
    if "DISTLLM_FILE_DIR" not in os.environ:
        warnings.warn(
            "DISTLLM_FILE_DIR not set, defaulting to /tmp which may not persist "
            "across container restarts. Set DISTLLM_FILE_DIR to a persistent volume path."
        )
    return base / f"{file_id}_{filename}"


def get_file_path(file_id: str) -> Path | None:
    """Get the storage path for a file by its ID.

    Args:
        file_id: The file ID (e.g., "file-abc123").

    Returns:
        Path to the file, or None if the file ID is not found.
    """
    file_obj = _files.get(file_id)
    if file_obj is None:
        return None
    return _get_storage_path(file_id, file_obj["filename"])
