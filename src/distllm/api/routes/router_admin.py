"""Admin REST API for model router configuration and diagnostics.

Provides endpoints under ``/v1/router/`` for:
- Listing and managing routing rules
- Dry-run routing tests
- Viewing routing statistics and capabilities

Requires ``admin`` role API key for write operations.
"""

from __future__ import annotations

import time

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..api_state import g
from ..auth_deps import require_role
from loguru import logger


router = APIRouter(tags=["router"])


# ── Pydantic models ────────────────────────────────────────────────────────


class RouterRuleCreate(BaseModel):
    name: str = Field(..., description="Rule name")
    match_type: str = Field("keyword", description="Match type: keyword, regex, or workload")
    pattern: str = Field(..., description="Pattern to match")
    target_model: str = Field(..., description="Target model name")
    priority: int = Field(0, ge=0, description="Rule priority")


class RouterTestRequest(BaseModel):
    query: str = Field(..., description="Query text to test routing against")


class RouterTestResponse(BaseModel):
    model: str
    rule_name: str
    confidence: float
    latency_ms: float


# ── Public endpoints ────────────────────────────────────────────────────────


@router.get(
    "/v1/router/capabilities",
    summary="Router capabilities",
    description="Returns routing rules, workload patterns, hybrid models, and statistics.",
)
async def router_capabilities():
    """Return full router capabilities and state."""
    coord = g.coordinator
    model_router = getattr(coord, "_model_router", None)

    if model_router is None:
        return {
            "enabled": False,
            "rules": [],
            "workload_patterns": {},
            "hybrid_models": [],
            "stats": {},
        }

    return {
        "enabled": True,
        "default_model": model_router._default_model,
        "rules": model_router.to_dict().get("rules", []),
        "workload_patterns": {
            name: {
                "keywords": pats.get("keywords", []),
                "regex": pats.get("regex", []),
            }
            for name, pats in model_router._workload_patterns.items()
        },
        "hybrid_models": model_router.list_hybrid_models(),
        "stats": model_router.stats,
    }


@router.post(
    "/v1/router/test",
    response_model=RouterTestResponse,
    summary="Dry-run routing",
    description="Test routing against a query without dispatching inference.",
)
async def router_test(body: RouterTestRequest):
    """Test routing against a query."""
    coord = g.coordinator
    model_router = getattr(coord, "_model_router", None)

    if model_router is None:
        raise HTTPException(status_code=404, detail="Model router not configured")

    match = model_router.route(
        [{"role": "user", "content": body.query}],
        available_models=coord.list_models() if hasattr(coord, "list_models") else None,
    )
    return RouterTestResponse(
        model=match.model,
        rule_name=match.rule_name,
        confidence=match.confidence,
        latency_ms=match.latency_ms,
    )


# ── Admin endpoints (require admin role) ────────────────────────────────────


@router.get(
    "/v1/router/rules",
    summary="List routing rules",
    description="Returns all configured routing rules.",
)
async def list_rules():
    """List all routing rules."""
    coord = g.coordinator
    model_router = getattr(coord, "_model_router", None)
    if model_router is None:
        return {"rules": []}
    return {"rules": model_router.to_dict().get("rules", [])}


@router.post(
    "/v1/router/rules",
    summary="Add routing rule",
    description="Add a new routing rule at runtime.",
    dependencies=[Depends(require_role("admin"))],
)
async def add_rule(body: RouterRuleCreate):
    """Add a routing rule."""
    coord = g.coordinator
    model_router = getattr(coord, "_model_router", None)
    if model_router is None:
        raise HTTPException(status_code=404, detail="Model router not configured")

    from distllm.core.model_router import RouteRule
    model_router.add_rule(RouteRule(
        name=body.name,
        match_type=body.match_type,
        pattern=body.pattern,
        target_model=body.target_model,
        priority=body.priority,
    ))
    logger.info(f"Router rule added: {body.name} -> {body.target_model}")
    return {"status": "added", "name": body.name}


@router.delete(
    "/v1/router/rules/{name}",
    summary="Remove routing rule",
    description="Remove a routing rule by name.",
    dependencies=[Depends(require_role("admin"))],
)
async def remove_rule(name: str):
    """Remove a routing rule by name."""
    coord = g.coordinator
    model_router = getattr(coord, "_model_router", None)
    if model_router is None:
        raise HTTPException(status_code=404, detail="Model router not configured")

    original_count = len(model_router._rules)
    model_router._rules = [r for r in model_router._rules if r.name != name]
    if len(model_router._rules) == original_count:
        raise HTTPException(status_code=404, detail=f"Rule '{name}' not found")

    logger.info(f"Router rule removed: {name}")
    return {"status": "removed", "name": name}


@router.get(
    "/v1/router/stats",
    summary="Routing statistics",
    description="Returns routing decision statistics.",
)
async def router_stats():
    """Return routing statistics."""
    coord = g.coordinator
    model_router = getattr(coord, "_model_router", None)
    if model_router is None:
        return {"stats": {}}
    return {"stats": model_router.stats}
