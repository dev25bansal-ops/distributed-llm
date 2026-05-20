"""Usage metering and billing for multi-tenant SaaS."""

import time
from typing import Optional

from loguru import logger

from distllm.tenant.models import (
    Tenant,
    TenantUsageRecord,
    TenantUsageReport,
    MODEL_TIER_MATRIX,
)
from distllm.tenant.store import TenantStore


def _lookup_model_cost(model: str) -> tuple[float, float]:
    for mc in MODEL_TIER_MATRIX:
        if mc.model_id == model:
            return mc.cost_per_1k_input_tokens, mc.cost_per_1k_output_tokens
    return 0.001, 0.002


class UsageMeter:
    """Records per-request usage and computes costs.

    Thread-safe when used with TenantStore (lock-free append).
    """

    def __init__(self, store: TenantStore):
        self.store = store

    def record(
        self,
        tenant_id: str,
        input_tokens: int,
        output_tokens: int,
        model: str = "",
        endpoint: str = "",
        latency_ms: float = 0.0,
    ) -> TenantUsageRecord:
        input_cost, output_cost = _lookup_model_cost(model)
        cost = (input_tokens / 1000) * input_cost + (output_tokens / 1000) * output_cost

        record = TenantUsageRecord(
            tenant_id=tenant_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            requests=1,
            model=model,
            endpoint=endpoint,
            latency_ms=latency_ms,
            cost=cost,
        )
        self.store.record_usage(record)
        return record

    def get_report(self, tenant_id: str, since: float = 0) -> TenantUsageReport:
        return self.store.get_usage_report(tenant_id, since=since)

    def get_live_snapshot(self, tenant_id: str, window_seconds: int = 60) -> dict:
        return self.store.get_tenant_usage_snapshot(tenant_id, window_seconds=window_seconds)


class BillingReport:
    """Generates formatted billing reports for tenants."""

    def __init__(self, meter: UsageMeter):
        self.meter = meter

    def generate_report(self, tenant_id: str, since: float) -> dict:
        report = self.meter.get_report(tenant_id, since=since)
        return {
            "tenant_id": report.tenant_id,
            "tier": report.tier.value,
            "period_start": report.period_start,
            "period_end": report.period_end,
            "summary": {
                "total_requests": report.total_requests,
                "total_input_tokens": report.total_input_tokens,
                "total_output_tokens": report.total_output_tokens,
                "total_cost": round(report.total_cost, 6),
                "avg_latency_ms": round(report.avg_latency_ms, 2),
            },
            "cost_breakdown": {
                "total": round(report.total_cost, 6),
                "by_model": {
                    model: {
                        "requests": data["requests"],
                        "input_tokens": data["input_tokens"],
                        "output_tokens": data["output_tokens"],
                        "cost": round(data["cost"], 6),
                    }
                    for model, data in report.model_breakdown.items()
                },
                "by_endpoint": {
                    ep: {"requests": data["requests"], "cost": round(data["cost"], 6)}
                    for ep, data in report.endpoint_breakdown.items()
                },
            },
        }
