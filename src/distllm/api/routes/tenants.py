"""Tenant management REST API routes.

Provides CRUD for tenants, usage reports, and billing data.
All routes require an admin API key (set via ADMIN_API_KEY env var).
"""

import os
import time
import hmac

from fastapi import APIRouter, Depends, HTTPException, Header, Query, Request
from loguru import logger

from distllm.tenant.store import TenantStore
from distllm.tenant.models import TenantTier, ResourceQuota
from distllm.tenant.billing import UsageMeter

router = APIRouter(prefix="/v1/tenants", tags=["tenants"])


def _verify_admin(authorization: str = Header("")) -> None:
    admin_key = os.environ.get("ADMIN_API_KEY", "")
    if not admin_key:
        raise HTTPException(status_code=503, detail="Admin API not configured")
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing authorization")
    key = authorization[7:]
    if not hmac.compare_digest(key, admin_key):
        raise HTTPException(status_code=401, detail="Invalid admin key")


def _get_store(request: Request) -> TenantStore:
    store = getattr(request.app.state, "tenant_store", None)
    if store is None:
        raise HTTPException(status_code=503, detail="Tenant system not initialized")
    return store


def _get_meter(request: Request) -> UsageMeter:
    meter = getattr(request.app.state, "usage_meter", None)
    if meter is None:
        raise HTTPException(status_code=503, detail="Usage metering not initialized")
    return meter


@router.post(
    "",
    summary="Create tenant",
    description="Create a new tenant with specified name, tier (free, pro, enterprise), and optional resource quota. Returns the tenant ID and generated API key. Requires admin authentication.",
    response_description="Created tenant with ID, API key, and quota",
    responses={
        400: {"description": "Invalid tier value"},
        401: {"description": "Invalid or missing admin API key"},
        503: {"description": "Tenant system not initialized"},
    },
)
async def create_tenant(
    body: dict,
    request: Request,
    _admin=Depends(_verify_admin),
):
    store = _get_store(request)
    name = body.get("name", "unnamed")
    tier_str = body.get("tier", "free")
    try:
        tier = TenantTier(tier_str)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid tier: {tier_str}")
    quota_dict = body.get("quota", {})
    quota = ResourceQuota(**quota_dict) if quota_dict else None
    tenant = store.create_tenant(name=name, tier=tier, quota=quota)
    return {
        "tenant_id": tenant.tenant_id,
        "name": tenant.name,
        "tier": tenant.tier.value,
        "api_key": tenant.api_key,
        "quota": {
            "max_rpm": tenant.quota.max_rpm,
            "max_tpm": tenant.quota.max_tpm,
            "max_concurrent_requests": tenant.quota.max_concurrent_requests,
            "max_context_length": tenant.quota.max_context_length,
            "kv_cache_size_mb": tenant.quota.kv_cache_size_mb,
            "allowed_models": tenant.quota.allowed_models,
        },
    }


@router.get(
    "",
    summary="List tenants",
    description="List all registered tenants with their ID, name, tier, active status, and creation timestamp. Requires admin authentication.",
    response_description="List of tenant summaries",
    responses={
        401: {"description": "Invalid or missing admin API key"},
        503: {"description": "Tenant system not initialized"},
    },
)
async def list_tenants(request: Request, _admin=Depends(_verify_admin)):
    store = _get_store(request)
    return [
        {
            "tenant_id": t.tenant_id,
            "name": t.name,
            "tier": t.tier.value,
            "is_active": t.is_active,
            "created_at": t.created_at,
        }
        for t in store.list_tenants()
    ]


@router.get(
    "/{tenant_id}",
    summary="Get tenant details",
    description="Get detailed information about a specific tenant, including resource quota and active status. Requires admin authentication.",
    response_description="Tenant details with quota configuration",
    responses={
        401: {"description": "Invalid or missing admin API key"},
        404: {"description": "Tenant not found"},
        503: {"description": "Tenant system not initialized"},
    },
)
async def get_tenant(tenant_id: str, request: Request, _admin=Depends(_verify_admin)):
    store = _get_store(request)
    tenant = store.get_tenant(tenant_id)
    if not tenant:
        raise HTTPException(status_code=404, detail="Tenant not found")
    return {
        "tenant_id": tenant.tenant_id,
        "name": tenant.name,
        "tier": tenant.tier.value,
        "is_active": tenant.is_active,
        "created_at": tenant.created_at,
        "quota": {
            "max_rpm": tenant.quota.max_rpm,
            "max_tpm": tenant.quota.max_tpm,
            "max_concurrent_requests": tenant.quota.max_concurrent_requests,
            "max_context_length": tenant.quota.max_context_length,
            "kv_cache_size_mb": tenant.quota.kv_cache_size_mb,
            "allowed_models": tenant.quota.allowed_models,
        },
    }


@router.put(
    "/{tenant_id}",
    summary="Update tenant",
    description="Update a tenant's configuration including name, tier, and active status. Requires admin authentication.",
    response_description="Update confirmation with tenant ID",
    responses={
        400: {"description": "Invalid tier value"},
        401: {"description": "Invalid or missing admin API key"},
        404: {"description": "Tenant not found"},
        503: {"description": "Tenant system not initialized"},
    },
)
async def update_tenant(
    tenant_id: str, body: dict, request: Request, _admin=Depends(_verify_admin)
):
    store = _get_store(request)
    updates = {}
    if "name" in body:
        updates["name"] = body["name"]
    if "tier" in body:
        try:
            updates["tier"] = TenantTier(body["tier"])
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid tier: {body['tier']}")
    if "is_active" in body:
        updates["is_active"] = bool(body["is_active"])
    tenant = store.update_tenant(tenant_id, **updates)
    if not tenant:
        raise HTTPException(status_code=404, detail="Tenant not found")
    return {"status": "ok", "tenant_id": tenant_id}


@router.delete(
    "/{tenant_id}",
    summary="Delete tenant",
    description="Permanently delete a tenant and its associated data. Requires admin authentication.",
    response_description="Deletion confirmation",
    responses={
        401: {"description": "Invalid or missing admin API key"},
        404: {"description": "Tenant not found"},
    },
)
async def delete_tenant(tenant_id: str, request: Request, _admin=Depends(_verify_admin)):
    store = _get_store(request)
    if not store.delete_tenant(tenant_id):
        raise HTTPException(status_code=404, detail="Tenant not found")
    return {"status": "ok"}


@router.post(
    "/{tenant_id}/regenerate-key",
    summary="Regenerate API key",
    description="Regenerate a tenant's API key. The previous key is immediately invalidated. Requires admin authentication.",
    response_description="New API key",
    responses={
        401: {"description": "Invalid or missing admin API key"},
        404: {"description": "Tenant not found"},
        503: {"description": "Tenant system not initialized"},
    },
)
async def regenerate_key(tenant_id: str, request: Request, _admin=Depends(_verify_admin)):
    store = _get_store(request)
    new_key = store.regenerate_api_key(tenant_id)
    if not new_key:
        raise HTTPException(status_code=404, detail="Tenant not found")
    return {"api_key": new_key}


@router.get(
    "/{tenant_id}/usage",
    summary="Get tenant usage report",
    description="Get a usage report for a tenant, including total requests, input/output tokens, cost, and average latency. Supports filtering by time range. Requires admin authentication.",
    response_description="Usage report with token counts and cost",
    responses={
        401: {"description": "Invalid or missing admin API key"},
        503: {"description": "Usage metering not initialized"},
    },
)
async def get_usage(
    tenant_id: str,
    request: Request,
    since: float = Query(0, description="Unix timestamp (default: all time)"),
    _admin=Depends(_verify_admin),
):
    meter = _get_meter(request)
    report = meter.get_report(tenant_id, since=since)
    return {
        "tenant_id": report.tenant_id,
        "tier": report.tier.value,
        "period_start": report.period_start,
        "period_end": report.period_end,
        "total_requests": report.total_requests,
        "total_input_tokens": report.total_input_tokens,
        "total_output_tokens": report.total_output_tokens,
        "total_cost": round(report.total_cost, 6),
        "avg_latency_ms": round(report.avg_latency_ms, 2),
    }


@router.get(
    "/{tenant_id}/billing",
    summary="Get tenant billing report",
    description="Get a detailed billing report for a tenant, including cost breakdown by period and request category. Supports filtering by time range. Requires admin authentication.",
    response_description="Billing report with cost details",
    responses={
        401: {"description": "Invalid or missing admin API key"},
        503: {"description": "Tenant system or usage metering not initialized"},
    },
)
async def get_billing(
    tenant_id: str,
    request: Request,
    since: float = Query(0),
    _admin=Depends(_verify_admin),
):
    store = _get_store(request)
    meter = _get_meter(request)
    from distllm.tenant.billing import BillingReport
    br = BillingReport(meter)
    return br.generate_report(tenant_id, since=since)


@router.get(
    "/{tenant_id}/live",
    summary="Get live tenant usage",
    description="Get real-time usage snapshot for a tenant over a configurable time window (5-3600 seconds). Returns current rate and request counts. Public endpoint (no admin auth required).",
    response_description="Live usage snapshot with current metrics",
    responses={
        404: {"description": "Tenant not found"},
        503: {"description": "Tenant system or usage metering not initialized"},
    },
)
async def get_live_usage(
    tenant_id: str,
    request: Request,
    window: int = Query(60, ge=5, le=3600),
):
    store = _get_store(request)
    tenant = store.get_tenant(tenant_id)
    if not tenant:
        raise HTTPException(status_code=404, detail="Tenant not found")
    meter = _get_meter(request)
    return meter.get_live_snapshot(tenant_id, window_seconds=window)
