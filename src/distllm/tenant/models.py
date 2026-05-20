"""Tenant data models for multi-tenant SaaS management."""

import uuid
import time
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional


class TenantTier(str, Enum):
    FREE = "free"
    STARTER = "starter"
    BUSINESS = "business"
    ENTERPRISE = "enterprise"


@dataclass
class ResourceQuota:
    max_rpm: int = 60
    max_tpm: int = 10000
    max_concurrent_requests: int = 5
    max_context_length: int = 4096
    max_batch_size: int = 1
    max_model_slots: int = 1
    kv_cache_size_mb: int = 512
    allowed_models: list[str] = field(default_factory=lambda: ["default"])
    allowed_adapters: int = 1


TIER_QUOTAS: dict[TenantTier, ResourceQuota] = {
    TenantTier.FREE: ResourceQuota(
        max_rpm=10, max_tpm=1000, max_concurrent_requests=1,
        max_context_length=2048, kv_cache_size_mb=128,
        allowed_models=["default"],
    ),
    TenantTier.STARTER: ResourceQuota(
        max_rpm=60, max_tpm=10000, max_concurrent_requests=5,
        max_context_length=4096, kv_cache_size_mb=512,
        allowed_models=["default", "fast"],
    ),
    TenantTier.BUSINESS: ResourceQuota(
        max_rpm=300, max_tpm=100000, max_concurrent_requests=20,
        max_context_length=8192, kv_cache_size_mb=2048,
        allowed_models=["default", "fast", "premium"],
    ),
    TenantTier.ENTERPRISE: ResourceQuota(
        max_rpm=3000, max_tpm=1000000, max_concurrent_requests=100,
        max_context_length=32768, kv_cache_size_mb=8192,
        allowed_models=["default", "fast", "premium", "enterprise"],
    ),
}


@dataclass
class TenantModelConfig:
    model_id: str
    tier_access: list[TenantTier] = field(default_factory=lambda: list(TenantTier))
    cost_per_1k_input_tokens: float = 0.0
    cost_per_1k_output_tokens: float = 0.0


MODEL_TIER_MATRIX: list[TenantModelConfig] = [
    TenantModelConfig("default", [TenantTier.FREE, TenantTier.STARTER, TenantTier.BUSINESS, TenantTier.ENTERPRISE], 0.001, 0.002),
    TenantModelConfig("fast", [TenantTier.STARTER, TenantTier.BUSINESS, TenantTier.ENTERPRISE], 0.002, 0.004),
    TenantModelConfig("premium", [TenantTier.BUSINESS, TenantTier.ENTERPRISE], 0.005, 0.010),
    TenantModelConfig("enterprise", [TenantTier.ENTERPRISE], 0.010, 0.020),
]


@dataclass
class Tenant:
    tenant_id: str
    name: str
    tier: TenantTier = TenantTier.FREE
    api_key: str = ""
    quota: ResourceQuota = field(default_factory=ResourceQuota)
    is_active: bool = True
    created_at: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.tenant_id:
            self.tenant_id = f"tnt_{uuid.uuid4().hex[:12]}"
        if self.tier and self.tier in TIER_QUOTAS:
            base = TIER_QUOTAS[self.tier]
            merged = ResourceQuota(
                max_rpm=self.quota.max_rpm if self.quota.max_rpm != 60 else base.max_rpm,
                max_tpm=self.quota.max_tpm if self.quota.max_tpm != 10000 else base.max_tpm,
                max_concurrent_requests=self.quota.max_concurrent_requests if self.quota.max_concurrent_requests != 5 else base.max_concurrent_requests,
                max_context_length=self.quota.max_context_length if self.quota.max_context_length != 4096 else base.max_context_length,
                max_batch_size=self.quota.max_batch_size if self.quota.max_batch_size != 1 else base.max_batch_size,
                max_model_slots=self.quota.max_model_slots if self.quota.max_model_slots != 1 else base.max_model_slots,
                kv_cache_size_mb=self.quota.kv_cache_size_mb if self.quota.kv_cache_size_mb != 512 else base.kv_cache_size_mb,
                allowed_models=self.quota.allowed_models if self.quota.allowed_models != ["default"] else base.allowed_models,
                allowed_adapters=self.quota.allowed_adapters if self.quota.allowed_adapters != 1 else base.allowed_adapters,
            )
            self.quota = merged


@dataclass
class TenantUsageRecord:
    tenant_id: str
    timestamp: float = field(default_factory=time.time)
    input_tokens: int = 0
    output_tokens: int = 0
    requests: int = 0
    model: str = ""
    endpoint: str = ""
    latency_ms: float = 0.0
    cost: float = 0.0


@dataclass
class TenantUsageReport:
    tenant_id: str
    tier: TenantTier
    period_start: float
    period_end: float
    total_requests: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost: float = 0.0
    avg_latency_ms: float = 0.0
    model_breakdown: dict[str, dict] = field(default_factory=dict)
    endpoint_breakdown: dict[str, dict] = field(default_factory=dict)
