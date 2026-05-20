"""In-memory tenant store with optional persistent backend."""

import threading
import time
import secrets
from collections import defaultdict
from typing import Optional

from loguru import logger

from distllm.tenant.models import (
    Tenant,
    TenantTier,
    ResourceQuota,
    TenantUsageRecord,
    TenantUsageReport,
)


class TenantStore:
    """Thread-safe in-memory store for tenants and usage records.

    Can be extended with a persistent backend (SQLite/PostgreSQL).
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._tenants: dict[str, Tenant] = {}
        self._api_key_map: dict[str, str] = {}
        self._usage: list[TenantUsageRecord] = []
        self._max_usage_records = 100000

    # --- Tenant CRUD ---

    def create_tenant(self, name: str, tier: TenantTier = TenantTier.FREE, quota: Optional[ResourceQuota] = None) -> Tenant:
        api_key = f"tnt_{secrets.token_hex(24)}"
        tenant = Tenant(
            tenant_id=f"tnt_{secrets.token_hex(8)}",
            name=name,
            tier=tier,
            api_key=api_key,
            quota=quota or ResourceQuota(),
        )
        with self._lock:
            self._tenants[tenant.tenant_id] = tenant
            self._api_key_map[api_key] = tenant.tenant_id
        logger.info(f"Created tenant {tenant.tenant_id} ({tier.value}): {name}")
        return tenant

    def get_tenant(self, tenant_id: str) -> Optional[Tenant]:
        with self._lock:
            return self._tenants.get(tenant_id)

    def get_tenant_by_api_key(self, api_key: str) -> Optional[Tenant]:
        with self._lock:
            tenant_id = self._api_key_map.get(api_key)
            if tenant_id:
                return self._tenants.get(tenant_id)
            return None

    def get_tenant_id_by_api_key(self, api_key: str) -> Optional[str]:
        with self._lock:
            return self._api_key_map.get(api_key)

    def update_tenant(self, tenant_id: str, **updates) -> Optional[Tenant]:
        with self._lock:
            tenant = self._tenants.get(tenant_id)
            if not tenant:
                return None
            for key, value in updates.items():
                if hasattr(tenant, key):
                    setattr(tenant, key, value)
            return tenant

    def delete_tenant(self, tenant_id: str) -> bool:
        with self._lock:
            tenant = self._tenants.pop(tenant_id, None)
            if tenant:
                self._api_key_map.pop(tenant.api_key, None)
                return True
            return False

    def list_tenants(self) -> list[Tenant]:
        with self._lock:
            return list(self._tenants.values())

    def regenerate_api_key(self, tenant_id: str) -> Optional[str]:
        with self._lock:
            tenant = self._tenants.get(tenant_id)
            if not tenant:
                return None
            old_key = tenant.api_key
            new_key = f"tnt_{secrets.token_hex(24)}"
            tenant.api_key = new_key
            self._api_key_map.pop(old_key, None)
            self._api_key_map[new_key] = tenant_id
            return new_key

    # --- Usage Metering ---

    def record_usage(self, record: TenantUsageRecord) -> None:
        self._usage.append(record)
        if len(self._usage) > self._max_usage_records:
            self._usage = self._usage[-self._max_usage_records // 2:]

    def get_usage_report(self, tenant_id: str, since: float = 0) -> TenantUsageReport:
        tenant = self.get_tenant(tenant_id)
        if not tenant:
            raise ValueError(f"Tenant {tenant_id} not found")

        now = time.time()
        records = [r for r in self._usage if r.tenant_id == tenant_id and r.timestamp >= since]

        model_breakdown: dict[str, dict] = {}
        endpoint_breakdown: dict[str, dict] = {}
        total_input = total_output = total_req = total_cost = total_latency = 0

        for r in records:
            total_input += r.input_tokens
            total_output += r.output_tokens
            total_req += r.requests
            total_cost += r.cost
            total_latency += r.latency_ms * r.requests

            if r.model:
                mb = model_breakdown.setdefault(r.model, {"requests": 0, "input_tokens": 0, "output_tokens": 0, "cost": 0.0})
                mb["requests"] += r.requests
                mb["input_tokens"] += r.input_tokens
                mb["output_tokens"] += r.output_tokens
                mb["cost"] += r.cost

            if r.endpoint:
                eb = endpoint_breakdown.setdefault(r.endpoint, {"requests": 0, "cost": 0.0})
                eb["requests"] += r.requests
                eb["cost"] += r.cost

        return TenantUsageReport(
            tenant_id=tenant_id,
            tier=tenant.tier,
            period_start=since,
            period_end=now,
            total_requests=total_req,
            total_input_tokens=total_input,
            total_output_tokens=total_output,
            total_cost=total_cost,
            avg_latency_ms=total_latency / max(total_req, 1),
            model_breakdown=model_breakdown,
            endpoint_breakdown=endpoint_breakdown,
        )

    def get_tenant_usage_snapshot(self, tenant_id: str, window_seconds: int = 60) -> dict:
        now = time.time()
        cutoff = now - window_seconds
        records = [r for r in self._usage if r.tenant_id == tenant_id and r.timestamp >= cutoff]
        return {
            "requests_1m": sum(r.requests for r in records),
            "input_tokens_1m": sum(r.input_tokens for r in records),
            "output_tokens_1m": sum(r.output_tokens for r in records),
            "cost_1m": sum(r.cost for r in records),
        }
