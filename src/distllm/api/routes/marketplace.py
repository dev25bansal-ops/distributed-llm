"""Marketplace API routes — GPU listing, job posting, and matching."""

from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from ..api_state import g
from ..auth_deps import require_role


router = APIRouter(tags=["marketplace"], prefix="/v1/marketplace")


# ── Request/Response Models ─────────────────────────────────────────────────

class GPUListingRequest(BaseModel):
    provider_id: str = Field(..., description="Provider identifier")
    gpu_name: str = Field(..., description="GPU model name", examples=["NVIDIA A100 80GB"])
    gpu_memory_bytes: int = Field(..., description="GPU memory in bytes")
    price_per_hour: float = Field(..., ge=0, description="Price per GPU-hour in USD")
    gpu_count: int = Field(default=1, ge=1, description="Number of GPUs")
    region: str = Field(default="", description="Deployment region")
    supported_models: list[str] = Field(default_factory=list, description="Supported model names")
    max_batch_size: int = Field(default=8, ge=1, description="Max concurrent batch size")
    supports_streaming: bool = Field(default=True)
    supports_quantization: bool = Field(default=False)
    supports_lora: bool = Field(default=False)
    tags: list[str] = Field(default_factory=list)


class GPUListingResponse(BaseModel):
    listing_id: str
    provider_id: str
    gpu_name: str
    gpu_memory_bytes: int
    gpu_count: int
    price_per_hour: float
    region: str
    status: str
    reputation_score: float
    total_jobs_completed: int
    is_available: bool


class JobPostRequest(BaseModel):
    requester_id: str = Field(..., description="Requester identifier")
    model_name: str = Field(..., description="Model to run")
    min_gpu_memory_bytes: int = Field(default=0, description="Minimum GPU memory required")
    max_price_per_hour: float = Field(default=0, ge=0, description="Max price per hour")
    max_latency_ms: float = Field(default=5000, ge=0, description="Max acceptable latency")
    min_reputation: float = Field(default=0.3, ge=0, le=1, description="Min provider reputation")
    preferred_regions: list[str] = Field(default_factory=list)
    priority: int = Field(default=2, ge=0, le=3)
    requires_streaming: bool = Field(default=True)
    requires_quantization: bool = Field(default=False)
    requires_lora: bool = Field(default=False)
    max_budget_total: float = Field(default=0, ge=0, description="Max total budget")


class JobResponse(BaseModel):
    job_id: str
    requester_id: str
    model_name: str
    status: str
    matched_listing_id: str = ""
    matched_provider_id: str = ""
    tokens_generated: int = 0
    cost_accumulated: float = 0.0


class MarketplaceStatsResponse(BaseModel):
    total_listings: int
    active_listings: int
    total_jobs: int
    open_jobs: int
    running_jobs: int
    completed_jobs: int
    total_volume_usd: float
    total_tokens_served: int
    avg_price_per_hour: float


# ── Endpoints ───────────────────────────────────────────────────────────────

@router.post("/listings", response_model=GPUListingResponse, dependencies=[Depends(require_role("admin"))])
async def create_listing(req: GPUListingRequest):
    """Create a new GPU listing in the marketplace."""
    marketplace = g.get("marketplace")
    if not marketplace:
        raise HTTPException(status_code=503, detail="Marketplace not available")

    listing = marketplace.create_listing(
        provider_id=req.provider_id,
        gpu_name=req.gpu_name,
        gpu_memory_bytes=req.gpu_memory_bytes,
        price_per_hour=req.price_per_hour,
        gpu_count=req.gpu_count,
        region=req.region,
        supported_models=req.supported_models,
        max_batch_size=req.max_batch_size,
        supports_streaming=req.supports_streaming,
        supports_quantization=req.requires_quantization,
        supports_lora=req.requires_lora,
        tags=req.tags,
    )
    return GPUListingResponse(
        listing_id=listing.listing_id,
        provider_id=listing.provider_id,
        gpu_name=listing.gpu_name,
        gpu_memory_bytes=listing.gpu_memory_bytes,
        gpu_count=listing.gpu_count,
        price_per_hour=listing.price_per_hour,
        region=listing.region,
        status=listing.status.value,
        reputation_score=listing.reputation_score,
        total_jobs_completed=listing.total_jobs_completed,
        is_available=listing.is_available,
    )


@router.get("/listings", response_model=list[GPUListingResponse])
async def list_listings(
    min_gpu_memory: int = 0,
    max_price: float = 0.0,
    region: str = "",
):
    """List available GPU listings with optional filters."""
    marketplace = g.get("marketplace")
    if not marketplace:
        raise HTTPException(status_code=503, detail="Marketplace not available")

    listings = marketplace.list_listings(
        min_gpu_memory=min_gpu_memory,
        max_price=max_price,
        region=region,
    )
    return [
        GPUListingResponse(
            listing_id=l.listing_id,
            provider_id=l.provider_id,
            gpu_name=l.gpu_name,
            gpu_memory_bytes=l.gpu_memory_bytes,
            gpu_count=l.gpu_count,
            price_per_hour=l.price_per_hour,
            region=l.region,
            status=l.status.value,
            reputation_score=l.reputation_score,
            total_jobs_completed=l.total_jobs_completed,
            is_available=l.is_available,
        )
        for l in listings
    ]


@router.get("/listings/{listing_id}", response_model=GPUListingResponse)
async def get_listing(listing_id: str):
    """Get a specific GPU listing."""
    marketplace = g.get("marketplace")
    if not marketplace:
        raise HTTPException(status_code=503, detail="Marketplace not available")

    listing = marketplace.get_listing(listing_id)
    if not listing:
        raise HTTPException(status_code=404, detail="Listing not found")
    return GPUListingResponse(
        listing_id=listing.listing_id,
        provider_id=listing.provider_id,
        gpu_name=listing.gpu_name,
        gpu_memory_bytes=listing.gpu_memory_bytes,
        gpu_count=listing.gpu_count,
        price_per_hour=listing.price_per_hour,
        region=listing.region,
        status=listing.status.value,
        reputation_score=listing.reputation_score,
        total_jobs_completed=listing.total_jobs_completed,
        is_available=listing.is_available,
    )


@router.delete("/listings/{listing_id}", dependencies=[Depends(require_role("admin"))])
async def remove_listing(listing_id: str):
    """Remove a GPU listing."""
    marketplace = g.get("marketplace")
    if not marketplace:
        raise HTTPException(status_code=503, detail="Marketplace not available")

    if not marketplace.remove_listing(listing_id):
        raise HTTPException(status_code=404, detail="Listing not found")
    return {"status": "removed", "listing_id": listing_id}


@router.post("/jobs", response_model=JobResponse, dependencies=[Depends(require_role("admin"))])
async def post_job(req: JobPostRequest):
    """Post a compute job to the marketplace."""
    marketplace = g.get("marketplace")
    if not marketplace:
        raise HTTPException(status_code=503, detail="Marketplace not available")

    job = marketplace.post_job(
        requester_id=req.requester_id,
        model_name=req.model_name,
        min_gpu_memory_bytes=req.min_gpu_memory_bytes,
        max_price_per_hour=req.max_price_per_hour,
        max_latency_ms=req.max_latency_ms,
        min_reputation=req.min_reputation,
        preferred_regions=req.preferred_regions,
        priority=req.priority,
        requires_streaming=req.requires_streaming,
        requires_quantization=req.requires_quantization,
        requires_lora=req.requires_lora,
        max_budget_total=req.max_budget_total,
    )

    # Auto-match
    listing = marketplace.match_job(job.job_id)
    if listing:
        marketplace.start_job(job.job_id)

    return JobResponse(
        job_id=job.job_id,
        requester_id=job.requester_id,
        model_name=job.model_name,
        status=job.status.value,
        matched_listing_id=job.matched_listing_id,
        matched_provider_id=job.matched_provider_id,
    )


@router.get("/jobs", response_model=list[JobResponse])
async def list_jobs(
    requester_id: str = "",
    status: str = "",
):
    """List marketplace jobs."""
    marketplace = g.get("marketplace")
    if not marketplace:
        raise HTTPException(status_code=503, detail="Marketplace not available")

    from distllm.dist.marketplace import JobStatus
    job_status = None
    if status:
        try:
            job_status = JobStatus(status)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid status: {status}")

    jobs = marketplace.list_jobs(requester_id=requester_id, status=job_status)
    return [
        JobResponse(
            job_id=j.job_id,
            requester_id=j.requester_id,
            model_name=j.model_name,
            status=j.status.value,
            matched_listing_id=j.matched_listing_id,
            matched_provider_id=j.matched_provider_id,
            tokens_generated=j.tokens_generated,
            cost_accumulated=j.cost_accumulated,
        )
        for j in jobs
    ]


@router.post("/jobs/{job_id}/complete", dependencies=[Depends(require_role("admin"))])
async def complete_job(job_id: str, tokens_generated: int = 0):
    """Mark a job as completed."""
    marketplace = g.get("marketplace")
    if not marketplace:
        raise HTTPException(status_code=503, detail="Marketplace not available")

    if not marketplace.complete_job(job_id, tokens_generated):
        raise HTTPException(status_code=400, detail="Cannot complete job")
    return {"status": "completed", "job_id": job_id}


@router.post("/jobs/{job_id}/cancel", dependencies=[Depends(require_role("admin"))])
async def cancel_job(job_id: str):
    """Cancel a job."""
    marketplace = g.get("marketplace")
    if not marketplace:
        raise HTTPException(status_code=503, detail="Marketplace not available")

    if not marketplace.cancel_job(job_id):
        raise HTTPException(status_code=400, detail="Cannot cancel job")
    return {"status": "cancelled", "job_id": job_id}


@router.get("/providers/{provider_id}/earnings")
async def get_provider_earnings(provider_id: str):
    """Get earnings summary for a GPU provider."""
    marketplace = g.get("marketplace")
    if not marketplace:
        raise HTTPException(status_code=503, detail="Marketplace not available")

    earnings = marketplace.get_provider_earnings(provider_id)
    if not earnings:
        raise HTTPException(status_code=404, detail="Provider not found")
    return {
        "provider_id": earnings.provider_id,
        "total_earnings": earnings.total_earnings,
        "total_gpu_hours": earnings.total_gpu_hours,
        "total_tokens_served": earnings.total_tokens_served,
        "total_jobs": earnings.total_jobs,
        "current_month_earnings": earnings.current_month_earnings,
        "pending_payout": earnings.pending_payout,
    }


@router.get("/stats", response_model=MarketplaceStatsResponse)
async def get_marketplace_stats():
    """Get overall marketplace statistics."""
    marketplace = g.get("marketplace")
    if not marketplace:
        raise HTTPException(status_code=503, detail="Marketplace not available")

    stats = marketplace.get_marketplace_stats()
    return MarketplaceStatsResponse(**stats)
