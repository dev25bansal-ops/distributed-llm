"""Prompt Exchange API — community prompt marketplace with token-gating."""

from __future__ import annotations

import contextlib
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from loguru import logger
from pydantic import BaseModel, Field

from ..api_state import g
from ..auth_deps import require_role
from ..errors import FeatureDisabledError, error_openapi_entry

# Shorthand used in route ``responses={...}`` dicts so Swagger UI renders the
# concrete error envelope every failure path returns.
ERR_ENTRY = error_openapi_entry


router = APIRouter(tags=["prompt-exchange"], prefix="/v1/exchange")


# ── Request/Response Models ─────────────────────────────────────────────────

class PublishPromptRequest(BaseModel):
    author_id: str = Field(..., description="Publisher user ID")
    name: str = Field(..., description="Prompt display name")
    description: str = Field(..., description="What the prompt does")
    category: str = Field(..., description="Category (code, writing, analysis, etc.)")
    system_prompt: str = Field(..., description="The actual system prompt text")
    tags: list[str] = Field(default_factory=list)
    license: str = Field(default="free", description="License: free, attribution, premium, subscription")
    price_tokens: int = Field(default=0, ge=0, description="Cost in tokens (0 = free)")
    examples: list[dict[str, str]] = Field(default_factory=list)
    parent_id: str = Field(default="", description="Parent prompt ID for forks")


class PromptResponse(BaseModel):
    prompt_id: str
    author_id: str
    name: str
    description: str
    category: str
    system_prompt: str
    tags: list[str]
    license: str
    price_tokens: int
    status: str
    version: int
    total_uses: int
    avg_rating: float
    rating_count: int
    avg_throughput_tok_s: float
    avg_latency_ms: float


class ReviewRequest(BaseModel):
    user_id: str
    rating: int = Field(..., ge=1, le=5)
    comment: str = ""


class UsageRecordRequest(BaseModel):
    throughput_tok_s: float = 0.0
    latency_ms: float = 0.0
    tokens_generated: int = 0
    cost_usd: float = 0.0
    quality_score: float = 0.0


class ExchangeStatsResponse(BaseModel):
    total_prompts: int
    free_prompts: int
    premium_prompts: int
    total_uses: int
    total_tokens_generated: int
    total_wallets: int
    total_reviews: int
    categories: dict[str, int]


# ── Endpoints ───────────────────────────────────────────────────────────────

@router.post(
    "/prompts",
    response_model=PromptResponse,
    dependencies=[Depends(require_role("admin"))],
    summary="Publish a prompt to the exchange",
    responses={
        401: ERR_ENTRY("Missing or invalid API key (`Authorization: Bearer <key>`)", type_="auth_error", code="authentication_error"),
        403: ERR_ENTRY("Authenticated but not permitted (requires the admin role)", type_="auth_error", code="forbidden"),
        422: ERR_ENTRY("Invalid publish payload (e.g. missing name or system_prompt)", type_="invalid_request_error", code="invalid_request"),
        503: ERR_ENTRY("Prompt exchange not available", type_="service_unavailable", code="503"),
    },
)
async def publish_prompt(req: PublishPromptRequest):
    """Publish a prompt to the exchange."""
    exchange = g.get("prompt_exchange")
    if not exchange:
        raise FeatureDisabledError("prompt-exchange", "Prompt exchange not available")

    from distllm.prompts.exchange import PromptLicense
    try:
        lic = PromptLicense(req.license)
    except ValueError:
        lic = PromptLicense.FREE

    prompt = exchange.publish_prompt(
        author_id=req.author_id,
        name=req.name,
        description=req.description,
        category=req.category,
        system_prompt=req.system_prompt,
        tags=req.tags,
        license=lic,
        price_tokens=req.price_tokens,
        examples=req.examples,
        parent_id=req.parent_id,
    )
    return _prompt_to_response(prompt)


@router.get(
    "/prompts",
    response_model=list[PromptResponse],
    summary="Browse published prompts",
    responses={
        401: ERR_ENTRY("Missing or invalid API key (`Authorization: Bearer <key>`)", type_="auth_error", code="authentication_error"),
        422: ERR_ENTRY("Invalid filter value (e.g. unrecognized license)", type_="invalid_request_error", code="invalid_request"),
        503: ERR_ENTRY("Prompt exchange not available", type_="service_unavailable", code="503"),
    },
)
async def browse_prompts(
    category: str = "",
    tag: str = "",
    license: str = "",
    author_id: str = "",
    sort_by: str = Query("popular", description="Sort: popular, rating, newest, trending"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    """Browse published prompts."""
    exchange = g.get("prompt_exchange")
    if not exchange:
        raise FeatureDisabledError("prompt-exchange", "Prompt exchange not available")

    from distllm.prompts.exchange import PromptLicense
    lic = None
    if license:
        # An unrecognised license filter is ignored rather than rejecting the
        # whole browse request -- the listings still come back unfiltered.
        with contextlib.suppress(ValueError):
            lic = PromptLicense(license)

    tags = [tag] if tag else None
    prompts = exchange.browse(
        category=category,
        tags=tags,
        license=lic,
        author_id=author_id,
        sort_by=sort_by,
        limit=limit,
        offset=offset,
    )
    return [_prompt_to_response(p) for p in prompts]


@router.get(
    "/prompts/search",
    response_model=list[PromptResponse],
    summary="Search prompts by name, description, tags, or content",
    responses={
        401: ERR_ENTRY("Missing or invalid API key (`Authorization: Bearer <key>`)", type_="auth_error", code="authentication_error"),
        422: ERR_ENTRY("Missing required search query parameter `q`", type_="invalid_request_error", code="invalid_request"),
        503: ERR_ENTRY("Prompt exchange not available", type_="service_unavailable", code="503"),
    },
)
async def search_prompts(
    q: str = Query(..., description="Search query"),
    limit: int = Query(20, ge=1, le=100),
):
    """Search prompts by name, description, tags, or content."""
    exchange = g.get("prompt_exchange")
    if not exchange:
        raise FeatureDisabledError("prompt-exchange", "Prompt exchange not available")

    prompts = exchange.search(q, limit=limit)
    return [_prompt_to_response(p) for p in prompts]


@router.get(
    "/prompts/{prompt_id}",
    response_model=PromptResponse,
    summary="Get a specific prompt",
    responses={
        401: ERR_ENTRY("Missing or invalid API key (`Authorization: Bearer <key>`)", type_="auth_error", code="authentication_error"),
        404: ERR_ENTRY("Prompt not found", type_="invalid_request_error", code="not_found"),
        503: ERR_ENTRY("Prompt exchange not available", type_="service_unavailable", code="503"),
    },
)
async def get_prompt(prompt_id: str):
    """Get a specific prompt."""
    exchange = g.get("prompt_exchange")
    if not exchange:
        raise FeatureDisabledError("prompt-exchange", "Prompt exchange not available")

    prompt = exchange.get_prompt(prompt_id)
    if not prompt:
        raise HTTPException(status_code=404, detail="Prompt not found")
    return _prompt_to_response(prompt)


@router.post(
    "/prompts/{prompt_id}/acquire",
    summary="Acquire a prompt (free or token-gated)",
    responses={
        402: ERR_ENTRY("Insufficient tokens or prompt not found", type_="invalid_request_error", code="payment_required"),
        401: ERR_ENTRY("Missing or invalid API key, or authentication required", type_="auth_error", code="authentication_error"),
        403: ERR_ENTRY("user_id does not match the authenticated identity", type_="auth_error", code="forbidden"),
        404: ERR_ENTRY("Prompt not found", type_="invalid_request_error", code="not_found"),
        503: ERR_ENTRY("Prompt exchange not available", type_="service_unavailable", code="503"),
    },
)
async def acquire_prompt(prompt_id: str, request: Request, user_id: str = Query(...)):
    """Acquire a prompt (free or token-gated)."""
    exchange = g.get("prompt_exchange")
    if not exchange:
        raise FeatureDisabledError("prompt-exchange", "Prompt exchange not available")

    # SECURITY: Bind user_id to the authenticated identity so a caller cannot
    # act as another user (IDOR protection). The API-key owner is authoritative
    # when present; without it, fall back to the (legacy) audit-only binding.
    api_key_id = getattr(request.state, "api_key_id", None)
    if not api_key_id:
        raise HTTPException(status_code=401, detail="Authentication required")
    owner = getattr(request.state, "api_key_owner", None)
    if owner is not None and user_id != owner:
        raise HTTPException(
            status_code=403,
            detail="user_id does not match the authenticated identity",
        )
    logger.debug(f"User {user_id} acquiring prompt via API key {api_key_id[:8]}...")
    prompt = exchange.acquire_prompt(user_id, prompt_id)
    if not prompt:
        raise HTTPException(status_code=402, detail="Insufficient tokens or prompt not found")
    return {
        "prompt_id": prompt.prompt_id,
        "system_prompt": prompt.system_prompt,
        "acquired": True,
    }


@router.get(
    "/prompts/{prompt_id}/access",
    summary="Check if a user has access to a prompt",
    responses={
        401: ERR_ENTRY("Missing or invalid API key (`Authorization: Bearer <key>`)", type_="auth_error", code="authentication_error"),
        422: ERR_ENTRY("Missing required query parameter `user_id`", type_="invalid_request_error", code="invalid_request"),
        503: ERR_ENTRY("Prompt exchange not available", type_="service_unavailable", code="503"),
    },
)
async def check_access(prompt_id: str, user_id: str = Query(...)):
    """Check if a user has access to a prompt."""
    exchange = g.get("prompt_exchange")
    if not exchange:
        raise FeatureDisabledError("prompt-exchange", "Prompt exchange not available")

    has = exchange.has_access(user_id, prompt_id)
    return {"prompt_id": prompt_id, "user_id": user_id, "has_access": has}


@router.post(
    "/prompts/{prompt_id}/review",
    summary="Add a review for a prompt",
    responses={
        400: ERR_ENTRY("Review cannot be added (e.g. duplicate review for this user)", type_="invalid_request_error", code="invalid_request"),
        401: ERR_ENTRY("Missing or invalid API key (`Authorization: Bearer <key>`)", type_="auth_error", code="authentication_error"),
        422: ERR_ENTRY("Invalid review payload (rating must be 1-5)", type_="invalid_request_error", code="invalid_request"),
        503: ERR_ENTRY("Prompt exchange not available", type_="service_unavailable", code="503"),
    },
)
async def add_review(prompt_id: str, req: ReviewRequest):
    """Add a review for a prompt."""
    exchange = g.get("prompt_exchange")
    if not exchange:
        raise FeatureDisabledError("prompt-exchange", "Prompt exchange not available")

    review = exchange.add_review(req.user_id, prompt_id, req.rating, req.comment)
    if not review:
        raise HTTPException(status_code=400, detail="Cannot add review")
    return {
        "review_id": review.review_id,
        "prompt_id": prompt_id,
        "rating": review.rating,
    }


@router.get(
    "/prompts/{prompt_id}/reviews",
    summary="Get reviews for a prompt",
    responses={
        401: ERR_ENTRY("Missing or invalid API key (`Authorization: Bearer <key>`)", type_="auth_error", code="authentication_error"),
        503: ERR_ENTRY("Prompt exchange not available", type_="service_unavailable", code="503"),
    },
)
async def get_reviews(prompt_id: str, limit: int = 20):
    """Get reviews for a prompt."""
    exchange = g.get("prompt_exchange")
    if not exchange:
        raise FeatureDisabledError("prompt-exchange", "Prompt exchange not available")

    reviews = exchange.get_reviews(prompt_id, limit=limit)
    return [
        {
            "review_id": r.review_id,
            "user_id": r.user_id,
            "rating": r.rating,
            "comment": r.comment,
            "created_at": r.created_at,
            "helpful_count": r.helpful_count,
        }
        for r in reviews
    ]


@router.post(
    "/prompts/{prompt_id}/usage",
    summary="Record usage metrics for a prompt",
    responses={
        401: ERR_ENTRY("Missing or invalid API key (`Authorization: Bearer <key>`)", type_="auth_error", code="authentication_error"),
        422: ERR_ENTRY("Invalid usage payload (e.g. negative token counts)", type_="invalid_request_error", code="invalid_request"),
        503: ERR_ENTRY("Prompt exchange not available", type_="service_unavailable", code="503"),
    },
)
async def record_usage(prompt_id: str, req: UsageRecordRequest):
    """Record usage metrics for a prompt."""
    exchange = g.get("prompt_exchange")
    if not exchange:
        raise FeatureDisabledError("prompt-exchange", "Prompt exchange not available")

    exchange.record_usage(
        prompt_id=prompt_id,
        throughput_tok_s=req.throughput_tok_s,
        latency_ms=req.latency_ms,
        tokens_generated=req.tokens_generated,
        cost_usd=req.cost_usd,
        quality_score=req.quality_score,
    )
    return {"recorded": True, "prompt_id": prompt_id}


@router.post(
    "/prompts/{prompt_id}/fork",
    response_model=PromptResponse,
    dependencies=[Depends(require_role("admin"))],
    summary="Fork an existing prompt",
    responses={
        401: ERR_ENTRY("Missing or invalid API key (`Authorization: Bearer <key>`)", type_="auth_error", code="authentication_error"),
        403: ERR_ENTRY("Authenticated but not permitted (requires the admin role)", type_="auth_error", code="forbidden"),
        404: ERR_ENTRY("Original prompt not found", type_="invalid_request_error", code="not_found"),
        422: ERR_ENTRY("Missing required query parameter `user_id`", type_="invalid_request_error", code="invalid_request"),
        503: ERR_ENTRY("Prompt exchange not available", type_="service_unavailable", code="503"),
    },
)
async def fork_prompt(prompt_id: str, user_id: str = Query(...), new_name: str = ""):
    """Fork an existing prompt."""
    exchange = g.get("prompt_exchange")
    if not exchange:
        raise FeatureDisabledError("prompt-exchange", "Prompt exchange not available")

    fork = exchange.fork_prompt(user_id, prompt_id, new_name=new_name)
    if not fork:
        raise HTTPException(status_code=404, detail="Original prompt not found")
    return _prompt_to_response(fork)


@router.get(
    "/leaderboard",
    summary="Get prompt leaderboard",
    responses={
        401: ERR_ENTRY("Missing or invalid API key (`Authorization: Bearer <key>`)", type_="auth_error", code="authentication_error"),
        422: ERR_ENTRY("Invalid metric value (expected uses, rating, throughput, or quality)", type_="invalid_request_error", code="invalid_request"),
        503: ERR_ENTRY("Prompt exchange not available", type_="service_unavailable", code="503"),
    },
)
async def get_leaderboard(
    category: str = "",
    metric: str = Query("uses", description="Metric: uses, rating, throughput, quality"),
    limit: int = 20,
):
    """Get prompt leaderboard."""
    exchange = g.get("prompt_exchange")
    if not exchange:
        raise FeatureDisabledError("prompt-exchange", "Prompt exchange not available")

    return {"leaderboard": exchange.get_leaderboard(category=category, metric=metric, limit=limit)}


@router.get(
    "/stats",
    response_model=ExchangeStatsResponse,
    summary="Get exchange statistics",
    responses={
        401: ERR_ENTRY("Missing or invalid API key (`Authorization: Bearer <key>`)", type_="auth_error", code="authentication_error"),
        503: ERR_ENTRY("Prompt exchange not available", type_="service_unavailable", code="503"),
    },
)
async def get_exchange_stats():
    """Get exchange statistics."""
    exchange = g.get("prompt_exchange")
    if not exchange:
        raise FeatureDisabledError("prompt-exchange", "Prompt exchange not available")

    return exchange.get_exchange_stats()


@router.get(
    "/wallet/{user_id}",
    summary="Get user's token wallet",
    responses={
        401: ERR_ENTRY("Missing or invalid API key (`Authorization: Bearer <key>`)", type_="auth_error", code="authentication_error"),
        503: ERR_ENTRY("Prompt exchange not available", type_="service_unavailable", code="503"),
    },
)
async def get_wallet(user_id: str):
    """Get user's token wallet."""
    exchange = g.get("prompt_exchange")
    if not exchange:
        raise FeatureDisabledError("prompt-exchange", "Prompt exchange not available")

    wallet = exchange.get_wallet(user_id)
    return {
        "user_id": wallet.user_id,
        "balance_tokens": wallet.balance_tokens,
        "total_earned": wallet.total_earned,
        "total_spent": wallet.total_spent,
    }


@router.post(
    "/wallet/{user_id}/topup",
    dependencies=[Depends(require_role("admin"))],
    summary="Top up user's token wallet",
    responses={
        401: ERR_ENTRY("Missing or invalid API key (`Authorization: Bearer <key>`)", type_="auth_error", code="authentication_error"),
        403: ERR_ENTRY("Authenticated but not permitted (requires the admin role)", type_="auth_error", code="forbidden"),
        422: ERR_ENTRY("Invalid topup amount (must be a positive integer)", type_="invalid_request_error", code="invalid_request"),
        503: ERR_ENTRY("Prompt exchange not available", type_="service_unavailable", code="503"),
    },
)
async def top_up_wallet(user_id: str, amount: int = Query(..., gt=0)):
    """Top up user's token wallet."""
    exchange = g.get("prompt_exchange")
    if not exchange:
        raise FeatureDisabledError("prompt-exchange", "Prompt exchange not available")

    wallet = exchange.top_up(user_id, amount)
    return {
        "user_id": wallet.user_id,
        "balance_tokens": wallet.balance_tokens,
        "added": amount,
    }


@router.get(
    "/library/{user_id}",
    response_model=list[PromptResponse],
    summary="Get all prompts a user has access to",
    responses={
        401: ERR_ENTRY("Missing or invalid API key (`Authorization: Bearer <key>`)", type_="auth_error", code="authentication_error"),
        503: ERR_ENTRY("Prompt exchange not available", type_="service_unavailable", code="503"),
    },
)
async def get_user_library(user_id: str):
    """Get all prompts a user has access to."""
    exchange = g.get("prompt_exchange")
    if not exchange:
        raise FeatureDisabledError("prompt-exchange", "Prompt exchange not available")

    prompts = exchange.get_user_library(user_id)
    return [_prompt_to_response(p) for p in prompts]


# ── Helpers ─────────────────────────────────────────────────────────────────

def _prompt_to_response(p) -> PromptResponse:
    return PromptResponse(
        prompt_id=p.prompt_id,
        author_id=p.author_id,
        name=p.name,
        description=p.description,
        category=p.category,
        system_prompt=p.system_prompt,
        tags=p.tags,
        license=p.license.value,
        price_tokens=p.price_tokens,
        status=p.status.value,
        version=p.version,
        total_uses=p.metrics.total_uses,
        avg_rating=p.metrics.avg_rating,
        rating_count=p.metrics.rating_count,
        avg_throughput_tok_s=p.metrics.avg_throughput_tok_s,
        avg_latency_ms=p.metrics.avg_latency_ms,
    )
