"""Multi-Tenant Cost Attribution — per-user cost tracking and routing metadata.

Extends the CostTracker with:
- Per-tenant (user/team/org) cost breakdown
- Routing decision metadata in cost records
- Response headers with routing attribution
- Cost analytics by compute source (cloud vs peer)
- Billing period management (hourly/daily/monthly)
"""

from __future__ import annotations

import time
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any

from loguru import logger


@dataclass
class RoutingAttribution:
    """Metadata about the routing decision that served a request."""
    compute_source: str = ""      # "cloud", "peer", "federated"
    provider: str = ""            # "aws", "gcp", "azure", "peer-abc123"
    instance_type: str = ""
    region: str = ""
    spot_used: bool = False
    price_per_hour: float = 0.0
    carbon_intensity: float = 0.0  # gCO2/kWh
    routing_reason: str = ""
    routing_score: float = 0.0
    alternatives_considered: int = 0

    def to_headers(self) -> dict[str, str]:
        """Convert to HTTP response headers."""
        headers: dict[str, str] = {}
        if self.compute_source:
            headers["X-DistLLM-Source"] = self.compute_source
        if self.provider:
            headers["X-DistLLM-Provider"] = self.provider
        if self.instance_type:
            headers["X-DistLLM-Instance"] = self.instance_type
        if self.region:
            headers["X-DistLLM-Region"] = self.region
        if self.spot_used:
            headers["X-DistLLM-Spot"] = "true"
        if self.price_per_hour > 0:
            headers["X-DistLLM-Price-Hour"] = f"{self.price_per_hour:.4f}"
        if self.carbon_intensity > 0:
            headers["X-DistLLM-Carbon"] = f"{self.carbon_intensity:.0f}"
        if self.routing_reason:
            headers["X-DistLLM-Route-Reason"] = self.routing_reason
        return headers


@dataclass
class TenantCostRecord:
    """A single cost record for a tenant."""
    tenant_id: str
    request_id: str
    timestamp: float = field(default_factory=time.time)

    # Token counts
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    # Cost
    estimated_cost_usd: float = 0.0
    actual_cost_usd: float = 0.0

    # Routing
    routing: RoutingAttribution = field(default_factory=RoutingAttribution)

    # Performance
    latency_ms: float = 0.0
    ttft_ms: float = 0.0
    tokens_per_second: float = 0.0


@dataclass
class TenantSummary:
    """Aggregated cost summary for a tenant over a period."""
    tenant_id: str
    period_start: float = 0.0
    period_end: float = 0.0
    total_requests: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    avg_cost_per_request: float = 0.0
    avg_latency_ms: float = 0.0
    avg_tokens_per_second: float = 0.0

    # By source
    cloud_cost_usd: float = 0.0
    peer_cost_usd: float = 0.0
    cloud_requests: int = 0
    peer_requests: int = 0

    # Carbon
    total_carbon_kg: float = 0.0

    # Top providers
    provider_costs: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tenant_id": self.tenant_id,
            "period": f"{self.period_start:.0f} - {self.period_end:.0f}",
            "total_requests": self.total_requests,
            "total_tokens": self.total_tokens,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "avg_cost_per_request": round(self.avg_cost_per_request, 8),
            "avg_latency_ms": round(self.avg_latency_ms, 1),
            "avg_tokens_per_second": round(self.avg_tokens_per_second, 1),
            "cloud_cost_usd": round(self.cloud_cost_usd, 6),
            "peer_cost_usd": round(self.peer_cost_usd, 6),
            "cloud_requests": self.cloud_requests,
            "peer_requests": self.peer_requests,
            "total_carbon_kg": round(self.total_carbon_kg, 4),
            "provider_costs": {k: round(v, 6) for k, v in self.provider_costs.items()},
        }


class TenantCostAttribution:
    """Multi-tenant cost attribution and billing.

    Tracks per-request costs with routing metadata and generates
    per-tenant summaries with cloud/peer breakdowns.

    Usage::

        attr = TenantCostAttribution()
        attr.record(
            tenant_id="team-alpha",
            request_id="req-123",
            input_tokens=100,
            output_tokens=200,
            routing=RoutingAttribution(
                compute_source="cloud",
                provider="aws",
                instance_type="p4d.24xlarge",
                spot_used=True,
                price_per_hour=14.40,
            ),
        )
        summary = attr.get_summary("team-alpha")
    """

    def __init__(self, max_history_per_tenant: int = 10000):
        self._max_history = max_history_per_tenant
        self._records: dict[str, deque[TenantCostRecord]] = defaultdict(
            lambda: deque(maxlen=max_history_per_tenant)
        )
        self._hourly_costs: dict[str, float] = defaultdict(float)
        self._daily_costs: dict[str, float] = defaultdict(float)
        self._monthly_costs: dict[str, float] = defaultdict(float)
        self._hourly_reset: float = time.time()
        self._daily_reset: float = time.time()
        self._monthly_reset: float = time.time()
        self._lock = threading.Lock()
        self._total_records = 0

    def record(
        self,
        tenant_id: str,
        request_id: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        estimated_cost_usd: float = 0.0,
        actual_cost_usd: float = 0.0,
        routing: RoutingAttribution | None = None,
        latency_ms: float = 0.0,
        ttft_ms: float = 0.0,
        tokens_per_second: float = 0.0,
    ) -> TenantCostRecord:
        """Record a completed request with full attribution."""
        cost = actual_cost_usd if actual_cost_usd > 0 else estimated_cost_usd
        record = TenantCostRecord(
            tenant_id=tenant_id,
            request_id=request_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            estimated_cost_usd=estimated_cost_usd,
            actual_cost_usd=cost,
            routing=routing or RoutingAttribution(),
            latency_ms=latency_ms,
            ttft_ms=ttft_ms,
            tokens_per_second=tokens_per_second,
        )
        with self._lock:
            self._records[tenant_id].append(record)
            self._hourly_costs[tenant_id] += cost
            self._daily_costs[tenant_id] += cost
            self._monthly_costs[tenant_id] += cost
            self._total_records += 1
            self._reset_periods_if_needed()
        return record

    def get_cost_headers(self, tenant_id: str, record: TenantCostRecord) -> dict[str, str]:
        """Generate all cost-related HTTP headers for a response."""
        headers: dict[str, str] = {}
        headers["X-DistLLM-Cost"] = f"{record.estimated_cost_usd:.8f}"
        headers["X-DistLLM-Tokens"] = f"{record.input_tokens}/{record.output_tokens}/{record.total_tokens}"
        if record.tokens_per_second > 0:
            headers["X-DistLLM-Tokens-Per-Second"] = f"{record.tokens_per_second:.1f}"
        if record.ttft_ms > 0:
            headers["X-DistLLM-TTFT"] = f"{record.ttft_ms:.1f}"
        if record.latency_ms > 0:
            headers["X-DistLLM-Latency"] = f"{record.latency_ms:.1f}"
        headers.update(record.routing.to_headers())
        with self._lock:
            hourly = self._hourly_costs.get(tenant_id, 0.0)
            daily = self._daily_costs.get(tenant_id, 0.0)
        headers["X-DistLLM-Tenant-Hourly-Cost"] = f"{hourly:.6f}"
        headers["X-DistLLM-Tenant-Daily-Cost"] = f"{daily:.6f}"
        return headers

    def get_summary(self, tenant_id: str, period_hours: float = 24.0) -> TenantSummary:
        """Get an aggregated cost summary for a tenant."""
        with self._lock:
            records = list(self._records.get(tenant_id, []))
        if not records:
            return TenantSummary(tenant_id=tenant_id)
        cutoff = time.time() - (period_hours * 3600)
        recent = [r for r in records if r.timestamp >= cutoff]
        if not recent:
            recent = records[-100:]
        summary = TenantSummary(
            tenant_id=tenant_id,
            period_start=recent[0].timestamp,
            period_end=recent[-1].timestamp,
            total_requests=len(recent),
            total_tokens=sum(r.total_tokens for r in recent),
            total_cost_usd=sum(r.actual_cost_usd for r in recent),
            avg_latency_ms=sum(r.latency_ms for r in recent) / len(recent),
            avg_tokens_per_second=sum(r.tokens_per_second for r in recent) / len(recent),
        )
        summary.avg_cost_per_request = summary.total_cost_usd / max(summary.total_requests, 1)
        provider_costs: dict[str, float] = defaultdict(float)
        for r in recent:
            src = r.routing.compute_source
            provider = r.routing.provider
            if src == "cloud":
                summary.cloud_cost_usd += r.actual_cost_usd
                summary.cloud_requests += 1
            elif src == "peer":
                summary.peer_cost_usd += r.actual_cost_usd
                summary.peer_requests += 1
            if provider:
                provider_costs[provider] += r.actual_cost_usd
            if r.routing.carbon_intensity > 0:
                energy_kwh = r.routing.carbon_intensity * 0.001
                summary.total_carbon_kg += energy_kwh * (r.latency_ms / 3600000)
        summary.provider_costs = dict(provider_costs)
        return summary

    def get_all_tenants(self) -> list[str]:
        """Get all tenant IDs with recorded costs."""
        with self._lock:
            return list(self._records.keys())

    def get_cost_breakdown(self, tenant_id: str) -> dict[str, Any]:
        """Get detailed cost breakdown by provider and source."""
        with self._lock:
            records = list(self._records.get(tenant_id, []))
        if not records:
            return {}
        by_source: dict[str, float] = defaultdict(float)
        by_provider: dict[str, float] = defaultdict(float)
        by_region: dict[str, float] = defaultdict(float)
        spot_cost = 0.0
        on_demand_cost = 0.0
        for r in records:
            by_source[r.routing.compute_source] += r.actual_cost_usd
            by_provider[r.routing.provider] += r.actual_cost_usd
            by_region[r.routing.region] += r.actual_cost_usd
            if r.routing.spot_used:
                spot_cost += r.actual_cost_usd
            else:
                on_demand_cost += r.actual_cost_usd
        return {
            "tenant_id": tenant_id,
            "total_records": len(records),
            "by_source": dict(by_source),
            "by_provider": dict(by_provider),
            "by_region": dict(by_region),
            "spot_cost_usd": round(spot_cost, 6),
            "on_demand_cost_usd": round(on_demand_cost, 6),
        }

    def _reset_periods_if_needed(self) -> None:
        now = time.time()
        if now - self._hourly_reset >= 3600:
            self._hourly_costs.clear()
            self._hourly_reset = now
        if now - self._daily_reset >= 86400:
            self._daily_costs.clear()
            self._daily_reset = now
        if now - self._monthly_reset >= 2592000:
            self._monthly_costs.clear()
            self._monthly_reset = now

    @property
    def total_records(self) -> int:
        return self._total_records

    @property
    def active_tenants(self) -> int:
        with self._lock:
            return len(self._records)
