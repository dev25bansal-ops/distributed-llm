"""Prompt template library API with sharing and versioning."""

from __future__ import annotations

import threading
import uuid
from dataclasses import asdict
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from distllm.api.auth_deps import require_role
from distllm.api.errors import error_openapi_entry
from distllm.prompts.engine import TemplateEngine
from distllm.prompts.library import SYSTEM_PROMPTS, SystemPromptDef
from distllm.prompts.templates import BUILTIN_TEMPLATES

# Shorthand used in route ``responses={...}`` dicts so Swagger UI renders the
# concrete error envelope every failure path returns.
ERR_ENTRY = error_openapi_entry

router = APIRouter(prefix="/v1/prompts", tags=["prompts"])


# ── Pydantic models ──────────────────────────────────────────────────────────


class PromptCreate(BaseModel):
    name: str
    category: str = "general"
    description: str = ""
    prompt: str
    tags: list[str] = Field(default_factory=list)


class PromptUpdate(BaseModel):
    name: str
    category: str = "general"
    description: str = ""
    prompt: str
    tags: list[str] = Field(default_factory=list)


class PromptImport(BaseModel):
    id: str | None = None
    name: str
    category: str = "general"
    description: str = ""
    prompt: str
    tags: list[str] = Field(default_factory=list)
    version: int = 1


class ForkRequest(BaseModel):
    name: str


class PromptSummary(BaseModel):
    id: str
    name: str
    category: str
    description: str
    tags: list[str]
    version: int


class PromptDetail(BaseModel):
    id: str
    name: str
    category: str
    description: str
    prompt: str
    tags: list[str]
    version: int


class ShareResponse(BaseModel):
    share_token: str
    prompt_id: str


class TemplateApplyRequest(BaseModel):
    template: str
    messages: list[dict[str, str]]
    add_generation_prompt: bool = True


class TemplateApplyResponse(BaseModel):
    result: str


class VersionInfo(BaseModel):
    version: int
    name: str
    category: str
    description: str
    prompt: str
    tags: list[str]


# ── In-memory data stores ────────────────────────────────────────────────────

_prompts: dict[str, SystemPromptDef] = {}
_share_tokens: dict[str, str] = {}
_version_history: dict[str, list[dict[str, Any]]] = {}
_seeded: bool = False
_seeded_lock = threading.Lock()


def _ensure_seeded() -> None:
    global _seeded
    if not _seeded:
        with _seeded_lock:
            if not _seeded:
                _seeded = True
                for pid, pdef in SYSTEM_PROMPTS.items():
                    _prompts[pid] = pdef
                    _version_history[pid] = [asdict(pdef)]


# ── Helpers ──────────────────────────────────────────────────────────────────


def _prompt_to_detail(p: SystemPromptDef) -> PromptDetail:
    return PromptDetail(
        id=p.id, name=p.name, category=p.category,
        description=p.description, prompt=p.prompt,
        tags=list(p.tags), version=p.version,
    )


def _prompt_to_summary(p: SystemPromptDef) -> PromptSummary:
    return PromptSummary(
        id=p.id, name=p.name, category=p.category,
        description=p.description, tags=list(p.tags),
        version=p.version,
    )


def _get_prompt_or_404(prompt_id: str) -> SystemPromptDef:
    _ensure_seeded()
    p = _prompts.get(prompt_id)
    if p is None:
        raise HTTPException(status_code=404, detail=f"Prompt '{prompt_id}' not found")
    return p


_SNAPSHOT_CONTENT_FIELDS = ("name", "category", "description", "prompt", "tags")


def _snapshot(prompt_id: str, p: SystemPromptDef) -> None:
    """Record *p* in the version history unless that state is already stored.

    Rule: a snapshot is skipped when any existing entry has the same
    ``version`` AND equal content fields. Comparing against every entry (not
    just the last one) matters after ``restore_version``: the history may
    already contain the state being replaced, and the restored version can
    legitimately re-appear later in the list. This keeps one snapshot per
    distinct version state instead of re-appending on every update/restore.
    """
    snapshot = asdict(p)
    history = _version_history.setdefault(prompt_id, [])
    for existing in history:
        if existing.get("version") == snapshot["version"] and all(
            existing.get(field) == snapshot[field]
            for field in _SNAPSHOT_CONTENT_FIELDS
        ):
            return
    history.append(snapshot)


# ── Static-path endpoints (BEFORE /{prompt_id} wildcard) ─────────────────────


@router.get(
    "/templates",
    summary="List names of built-in prompt templates available for apply",
)
async def list_templates():
    """List names of built-in prompt templates available for apply."""
    return {"templates": list(BUILTIN_TEMPLATES.keys())}


@router.post(
    "/templates/apply",
    response_model=TemplateApplyResponse,
    summary="Render a built-in or inline chat template against messages",
    responses={
        422: ERR_ENTRY("Invalid apply payload (template or messages malformed)", type_="invalid_request_error", code="invalid_request"),
    },
)
async def apply_template(body: TemplateApplyRequest):
    """Render a built-in or inline chat template against a list of messages."""
    engine = TemplateEngine(template=body.template)
    result = engine.apply(
        messages=body.messages,
        add_generation_prompt=body.add_generation_prompt,
    )
    return TemplateApplyResponse(result=result)


@router.post(
    "/import",
    response_model=PromptDetail,
    status_code=201,
    summary="Import a prompt with a caller-chosen ID",
    responses={
        409: ERR_ENTRY("Prompt with the given ID already exists", type_="invalid_request_error", code="conflict"),
        422: ERR_ENTRY("Invalid import payload (name and prompt are required)", type_="invalid_request_error", code="invalid_request"),
    },
)
async def import_prompt(body: PromptImport):
    """Import a prompt with a caller-chosen ID; rejects duplicate IDs with 409."""
    _ensure_seeded()
    prompt_id = body.id or str(uuid.uuid4())
    if prompt_id in _prompts:
        raise HTTPException(
            status_code=409,
            detail=f"Prompt with id '{prompt_id}' already exists",
        )
    pdef = SystemPromptDef(
        id=prompt_id,
        name=body.name,
        category=body.category,
        description=body.description,
        prompt=body.prompt,
        tags=list(body.tags),
        version=body.version,
    )
    _prompts[prompt_id] = pdef
    _version_history[prompt_id] = [asdict(pdef)]
    return _prompt_to_detail(pdef)


@router.get(
    "/shared/{share_token}",
    response_model=PromptDetail,
    summary="Fetch a prompt by its share token",
    responses={
        404: ERR_ENTRY("Share token not found, expired, or original prompt no longer exists", type_="invalid_request_error", code="not_found"),
    },
)
async def get_shared_prompt(share_token: str):
    """Fetch a prompt by its share token (from POST /{prompt_id}/share)."""
    _ensure_seeded()
    prompt_id = _share_tokens.get(share_token)
    if prompt_id is None:
        raise HTTPException(status_code=404, detail="Share token not found or expired")
    p = _prompts.get(prompt_id)
    if p is None:
        raise HTTPException(status_code=404, detail="Original prompt no longer exists")
    return _prompt_to_detail(p)


# ── List & Create (root path) ────────────────────────────────────────────────


@router.get(
    "",
    response_model=list[PromptSummary],
    summary="List prompts with optional category, text-search, and tag filters",
)
async def list_prompts(
    category: str | None = Query(None),
    search: str | None = Query(None),
    tag: str | None = Query(None),
):
    """List prompts with optional category, text-search, and tag filters."""
    _ensure_seeded()
    results = list(_prompts.values())
    if category:
        results = [p for p in results if p.category == category]
    if search:
        q = search.lower()
        results = [
            p for p in results
            if q in p.name.lower() or q in p.description.lower()
            or any(q in t.lower() for t in p.tags)
        ]
    if tag:
        q = tag.lower()
        results = [p for p in results if any(q == t.lower() for t in p.tags)]
    return [_prompt_to_summary(p) for p in results]


@router.post(
    "",
    response_model=PromptDetail,
    status_code=201,
    summary="Create a new prompt (version history starts at version 1)",
    responses={
        422: ERR_ENTRY("Invalid create payload (name and prompt are required)", type_="invalid_request_error", code="invalid_request"),
    },
)
async def create_prompt(body: PromptCreate):
    """Create a new prompt; version history starts at version 1."""
    _ensure_seeded()
    prompt_id = str(uuid.uuid4())
    pdef = SystemPromptDef(
        id=prompt_id,
        name=body.name,
        category=body.category,
        description=body.description,
        prompt=body.prompt,
        tags=list(body.tags),
        version=1,
    )
    _prompts[prompt_id] = pdef
    _version_history[prompt_id] = [asdict(pdef)]
    return _prompt_to_detail(pdef)


# ── Wildcard prompt_id routes ────────────────────────────────────────────────


@router.get(
    "/{prompt_id}",
    response_model=PromptDetail,
    summary="Fetch full prompt detail by ID",
    responses={
        404: ERR_ENTRY("Prompt not found", type_="invalid_request_error", code="not_found"),
    },
)
async def get_prompt_detail(prompt_id: str):
    """Fetch full prompt detail by ID."""
    return _prompt_to_detail(_get_prompt_or_404(prompt_id))


@router.put(
    "/{prompt_id}",
    response_model=PromptDetail,
    summary="Update a prompt (prior state is snapshotted as a new version)",
    responses={
        404: ERR_ENTRY("Prompt not found", type_="invalid_request_error", code="not_found"),
        422: ERR_ENTRY("Invalid update payload (name and prompt are required)", type_="invalid_request_error", code="invalid_request"),
    },
)
async def update_prompt(prompt_id: str, body: PromptUpdate):
    """Update a prompt; the prior state is snapshotted as a new version."""
    p = _get_prompt_or_404(prompt_id)
    _snapshot(prompt_id, p)
    new_version = p.version + 1
    p.name = body.name
    p.category = body.category
    p.description = body.description
    p.prompt = body.prompt
    p.tags = list(body.tags)
    p.version = new_version
    return _prompt_to_detail(p)


@router.delete(
    "/{prompt_id}",
    status_code=204,
    summary="Delete a prompt and its version history (admin only)",
    responses={
        401: ERR_ENTRY("Missing or invalid API key (`Authorization: Bearer <key>`)", type_="auth_error", code="authentication_error"),
        403: ERR_ENTRY("Authenticated but not permitted (requires the admin role)", type_="auth_error", code="forbidden"),
        404: ERR_ENTRY("Prompt not found", type_="invalid_request_error", code="not_found"),
    },
)
async def delete_prompt(
    prompt_id: str,
    _admin=Depends(require_role("admin")),
):
    """Delete a prompt and its version history (admin only)."""
    _ensure_seeded()
    if prompt_id not in _prompts:
        raise HTTPException(status_code=404, detail=f"Prompt '{prompt_id}' not found")
    del _prompts[prompt_id]
    _version_history.pop(prompt_id, None)
    stale = [t for t, pid in _share_tokens.items() if pid == prompt_id]
    for t in stale:
        del _share_tokens[t]


@router.post(
    "/{prompt_id}/fork",
    response_model=PromptDetail,
    status_code=201,
    summary="Copy an existing prompt into a new prompt under a new ID and name",
    responses={
        404: ERR_ENTRY("Prompt not found", type_="invalid_request_error", code="not_found"),
        422: ERR_ENTRY("Invalid fork payload (name is required)", type_="invalid_request_error", code="invalid_request"),
    },
)
async def fork_prompt(prompt_id: str, body: ForkRequest):
    """Copy an existing prompt into a new prompt under a new ID and name."""
    p = _get_prompt_or_404(prompt_id)
    new_id = str(uuid.uuid4())
    new_pdef = SystemPromptDef(
        id=new_id,
        name=body.name,
        category=p.category,
        description=p.description,
        prompt=p.prompt,
        tags=list(p.tags),
        version=1,
    )
    _prompts[new_id] = new_pdef
    _version_history[new_id] = [asdict(new_pdef)]
    return _prompt_to_detail(new_pdef)


@router.post(
    "/{prompt_id}/share",
    response_model=ShareResponse,
    summary="Create an opaque share token that resolves back to this prompt",
    responses={
        404: ERR_ENTRY("Prompt not found", type_="invalid_request_error", code="not_found"),
    },
)
async def share_prompt(prompt_id: str):
    """Create an opaque share token that resolves back to this prompt."""
    _get_prompt_or_404(prompt_id)
    share_token = str(uuid.uuid4())
    _share_tokens[share_token] = prompt_id
    return ShareResponse(share_token=share_token, prompt_id=prompt_id)


@router.post(
    "/{prompt_id}/export",
    response_model=PromptDetail,
    summary="Export the current prompt detail (portable representation)",
    responses={
        404: ERR_ENTRY("Prompt not found", type_="invalid_request_error", code="not_found"),
    },
)
async def export_prompt(prompt_id: str):
    """Export the current prompt detail (portable representation)."""
    return _prompt_to_detail(_get_prompt_or_404(prompt_id))


@router.get(
    "/{prompt_id}/versions",
    response_model=list[VersionInfo],
    summary="List all stored versions of a prompt, oldest first",
    responses={
        404: ERR_ENTRY("Prompt not found", type_="invalid_request_error", code="not_found"),
    },
)
async def list_versions(prompt_id: str):
    """List all stored versions of a prompt, oldest first."""
    _get_prompt_or_404(prompt_id)
    versions = _version_history.get(prompt_id, [])
    return [
        VersionInfo(
            version=snap["version"],
            name=snap["name"],
            category=snap["category"],
            description=snap["description"],
            prompt=snap["prompt"],
            tags=list(snap["tags"]),
        )
        for snap in versions
    ]


@router.post(
    "/{prompt_id}/versions/{version}",
    response_model=PromptDetail,
    summary="Restore a prompt to a previous version",
    responses={
        404: ERR_ENTRY("Prompt not found, or the requested version does not exist", type_="invalid_request_error", code="not_found"),
        422: ERR_ENTRY("Invalid version path parameter", type_="invalid_request_error", code="invalid_request"),
    },
)
async def restore_version(prompt_id: str, version: int):
    """Restore a prompt to a previous version; the current state is snapshotted first."""
    p = _get_prompt_or_404(prompt_id)
    versions = _version_history.get(prompt_id, [])
    target: dict[str, Any] | None = None
    for snap in versions:
        if snap["version"] == version:
            target = snap
            break
    if target is None:
        raise HTTPException(
            status_code=404,
            detail=f"Version {version} not found for prompt '{prompt_id}'",
        )
    # Keep the state being replaced unless it is already recorded, so
    # restoring away from the current version never loses it.
    _snapshot(prompt_id, p)
    p.name = target["name"]
    p.category = target["category"]
    p.description = target["description"]
    p.prompt = target["prompt"]
    p.tags = list(target["tags"])
    p.version = version
    return _prompt_to_detail(p)
