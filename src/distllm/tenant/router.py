"""Tenant-aware model routing: selects models based on tenant tier."""

from typing import Optional

from distllm.tenant.models import Tenant, TenantTier, MODEL_TIER_MATRIX


class TenantModelRouter:
    """Routes requests to appropriate models based on tenant tier.

    Higher-tier tenants get access to premium models.
    Falls back to `default` if the requested model is not in the tenant's tier.
    """

    def __init__(self, coordinator=None):
        self.coordinator = coordinator

    def get_available_models(self, tenant: Optional[Tenant]) -> list[str]:
        if tenant is None:
            return ["default"]
        return list(tenant.quota.allowed_models)

    def resolve_model(self, requested_model: str, tenant: Optional[Tenant]) -> str:
        if tenant is None:
            return requested_model if requested_model in ("default",) else "default"

        allowed = tenant.quota.allowed_models
        if requested_model in allowed:
            return requested_model

        if "default" in allowed:
            return "default"

        return allowed[0] if allowed else "default"

    def get_tier_for_model(self, model_id: str) -> list[str]:
        for mc in MODEL_TIER_MATRIX:
            if mc.model_id == model_id:
                return [t.value for t in mc.tier_access]
        return ["free"]
