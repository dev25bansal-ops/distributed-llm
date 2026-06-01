"""Cost-Aware Scheduling — per-request cost tracking and API headers.

Adds cost-per-token estimation, real-time cost tracking, and
exposes cost information in API response headers for enterprise
budgeting and transparency.

Features:
- Per-request GPU-time cost calculation
- Cost-per-token estimation based on model and hardware
- X-DistLLM-Cost and X-DistLLM-Tokens response headers
- Cost budget enforcement with pre-request checks
- Integration with UsageMeter for billing
- Historical cost analytics
- Multi-tenant cost attribution with routing metadata
"""

from __future__ import annotations

import re
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from distllm.core.tenant_cost_attribution import TenantCostAttribution, RoutingAttribution


# ── Pricing Models ──────────────────────────────────────────────────────────

# Cost per GPU-hour for different hardware (self-hosted cost)
GPU_COST_PER_HOUR: dict[str, float] = {
    "H100": 2.50,
    "A100-80GB": 1.80,
    "A100-40GB": 1.20,
    "A6000": 0.80,
    "RTX-4090": 0.60,
    "RTX-3090": 0.40,
    "RTX-3080": 0.30,
    "RTX-3070": 0.20,
    "Apple-M2-Ultra": 0.50,
    "Apple-M2-Pro": 0.30,
    "Apple-M1": 0.20,
    "CPU": 0.05,
}

# Cloud API pricing per million tokens (for comparison) — C8: Updated prices
CLOUD_API_COST_PER_M_TOKENS: dict[str, dict[str, float]] = {
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    "claude-3.5-sonnet": {"input": 3.00, "output": 15.00},
    "claude-3-opus": {"input": 15.00, "output": 75.00},
    "claude-3-sonnet": {"input": 3.00, "output": 15.00},
    "claude-3-haiku": {"input": 0.25, "output": 1.25},
    "llama-3.1-70b": {"input": 0.90, "output": 0.90},
    "llama-3.1-8b": {"input": 0.20, "output": 0.20},
    "mixtral-8x7b": {"input": 0.60, "output": 0.60},
    "deepseek-v3": {"input": 0.27, "output": 1.10},
}


@dataclass
class CostEstimate:
    """Cost estimate for an inference request."""
    model_name: str = ""
    gpu_type: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    # Self-hosted cost
    gpu_cost_per_hour: float = 0.0
    estimated_gpu_seconds: float = 0.0
    estimated_cost_usd: float = 0.0

    # Cloud comparison
    cloud_api_name: str = ""
    cloud_input_cost: float = 0.0
    cloud_output_cost: float = 0.0
    cloud_total_cost: float = 0.0
    savings_vs_cloud: float = 0.0
    savings_pct: float = 0.0

    # Time tracking
    start_time: float = 0.0
    end_time: float = 0.0
    ttft_ms: float = 0.0
    total_duration_ms: float = 0.0
    tokens_per_second: float = 0.0


@dataclass
class CostBudget:
    """Cost budget for a tenant or request class."""
    tenant_id: str = ""
    max_cost_per_request: float = 0.0
    max_cost_per_hour: float = 0.0
    max_cost_per_day: float = 0.0
    max_cost_per_month: float = 0.0
    current_hour_cost: float = 0.0
    current_day_cost: float = 0.0
    current_month_cost: float = 0.0
    alert_threshold_pct: float = 80.0


class CostTracker:
    """Tracks per-request and aggregate costs for inference.

    Integrates with the usage meter and provides cost headers
    for API responses. Supports multi-tenant cost attribution
    with routing metadata.
    """

    def __init__(
        self,
        default_gpu_type: str = "A100-80GB",
        default_model: str = "",
        tenant_attribution: "TenantCostAttribution | None" = None,
    ):
        self._default_gpu = default_gpu_type
        self._default_model = default_model
        # C1+C2: Track period boundaries for automatic reset
        self._hourly_costs: dict[str, tuple[float, float]] = {}  # tenant_id -> (cost, period_start)
        self._daily_costs: dict[str, tuple[float, float]] = {}
        self._monthly_costs: dict[str, tuple[float, float]] = {}
        self._budgets: dict[str, CostBudget] = {}
        self._history: deque[CostEstimate] = deque(maxlen=10000)
        self._tenant_attribution = tenant_attribution
        import threading
        self._lock = threading.Lock()

        # E1: Running aggregates for O(1) summary queries
        self._total_cost: float = 0.0
        self._total_cloud_cost: float = 0.0
        self._total_savings: float = 0.0
        self._total_requests: int = 0
        self._total_tps: float = 0.0

    def _reset_period_if_needed(self, tenant_id: str) -> None:
        """C2: Reset hourly/daily/monthly costs at period boundaries."""
        now = time.time()

        # Hourly reset
        cost, period_start = self._hourly_costs.get(tenant_id, (0.0, now))
        if now - period_start >= 3600:
            self._hourly_costs[tenant_id] = (0.0, now)
        else:
            self._hourly_costs[tenant_id] = (cost, period_start)

        # Daily reset
        cost, period_start = self._daily_costs.get(tenant_id, (0.0, now))
        if now - period_start >= 86400:
            self._daily_costs[tenant_id] = (0.0, now)
        else:
            self._daily_costs[tenant_id] = (cost, period_start)

        # Monthly reset (30 days)
        cost, period_start = self._monthly_costs.get(tenant_id, (0.0, now))
        if now - period_start >= 2592000:
            self._monthly_costs[tenant_id] = (0.0, now)
        else:
            self._monthly_costs[tenant_id] = (cost, period_start)

    def set_budget(self, tenant_id: str, budget: CostBudget) -> None:
        """Set a cost budget for a tenant."""
        with self._lock:
            self._budgets[tenant_id] = budget

    def check_budget(self, tenant_id: str, estimated_cost: float) -> tuple[bool, str]:
        """Check if a request would exceed the tenant's budget.

        Returns:
            (allowed, reason) — True if request is within budget.
        """
        with self._lock:
            budget = self._budgets.get(tenant_id)
            if not budget:
                return True, ""

            # C15: Validate input
            if estimated_cost < 0:
                return False, f"Invalid estimated cost: ${estimated_cost:.6f}"

            if budget.max_cost_per_request > 0 and estimated_cost > budget.max_cost_per_request:
                return False, f"Request cost ${estimated_cost:.6f} exceeds limit ${budget.max_cost_per_request:.6f}"

            # C2: Reset periods before checking
            self._reset_period_if_needed(tenant_id)

            hour_cost = self._hourly_costs.get(tenant_id, (0.0, time.time()))[0]
            if budget.max_cost_per_hour > 0 and hour_cost + estimated_cost > budget.max_cost_per_hour:
                return False, f"Hourly budget ${budget.max_cost_per_hour:.4f} would be exceeded"

            day_cost = self._daily_costs.get(tenant_id, (0.0, time.time()))[0]
            if budget.max_cost_per_day > 0 and day_cost + estimated_cost > budget.max_cost_per_day:
                return False, f"Daily budget ${budget.max_cost_per_day:.2f} would be exceeded"

            # C10: Check monthly limit (was missing)
            month_cost = self._monthly_costs.get(tenant_id, (0.0, time.time()))[0]
            if budget.max_cost_per_month > 0 and month_cost + estimated_cost > budget.max_cost_per_month:
                return False, f"Monthly budget ${budget.max_cost_per_month:.2f} would be exceeded"

            return True, ""

    def estimate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        model_name: str = "",
        gpu_type: str = "",
        cloud_api: str = "",
    ) -> CostEstimate:
        """Estimate the cost of an inference request.

        Args:
            input_tokens: Number of input/prompt tokens.
            output_tokens: Number of output/completion tokens.
            model_name: Model name for pricing lookup.
            gpu_type: GPU type for self-hosted cost.
            cloud_api: Cloud API name for comparison pricing.

        Returns:
            CostEstimate with self-hosted and cloud comparison costs.
        """
        # C15: Validate inputs
        input_tokens = max(0, input_tokens)
        output_tokens = max(0, output_tokens)

        gpu = gpu_type or self._default_gpu
        model = model_name or self._default_model
        total = input_tokens + output_tokens

        # Self-hosted cost calculation
        gpu_cost_per_hour = GPU_COST_PER_HOUR.get(gpu, 1.0)

        # Estimate GPU time based on token count and hardware
        tokens_per_second = _estimate_throughput(gpu, model)
        gpu_seconds = total / max(tokens_per_second, 1)
        estimated_cost = (gpu_seconds / 3600) * gpu_cost_per_hour

        estimate = CostEstimate(
            model_name=model,
            gpu_type=gpu,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total,
            gpu_cost_per_hour=gpu_cost_per_hour,
            estimated_gpu_seconds=gpu_seconds,
            estimated_cost_usd=estimated_cost,
        )

        # Cloud API comparison
        if cloud_api or model:
            cloud_name = cloud_api or _match_cloud_api(model)
            if cloud_name and cloud_name in CLOUD_API_COST_PER_M_TOKENS:
                pricing = CLOUD_API_COST_PER_M_TOKENS[cloud_name]
                estimate.cloud_api_name = cloud_name
                estimate.cloud_input_cost = (input_tokens / 1_000_000) * pricing["input"]
                estimate.cloud_output_cost = (output_tokens / 1_000_000) * pricing["output"]
                estimate.cloud_total_cost = estimate.cloud_input_cost + estimate.cloud_output_cost
                if estimate.cloud_total_cost > 0:
                    estimate.savings_vs_cloud = estimate.cloud_total_cost - estimated_cost
                    estimate.savings_pct = (
                        estimate.savings_vs_cloud / estimate.cloud_total_cost * 100
                    )

        return estimate

    def record_request(
        self,
        tenant_id: str,
        input_tokens: int,
        output_tokens: int,
        duration_ms: float,
        model_name: str = "",
        gpu_type: str = "",
        ttft_ms: float = 0.0,
        routing: "RoutingAttribution | None" = None,
    ) -> CostEstimate:
        """Record a completed request and update cost tracking.

        Args:
            tenant_id: Tenant identifier.
            input_tokens: Input token count.
            output_tokens: Output token count.
            duration_ms: Total request duration in ms.
            model_name: Model name.
            gpu_type: GPU type used.
            ttft_ms: Time to first token in ms.
            routing: Optional routing attribution metadata.

        Returns:
            CostEstimate for the completed request.
        """
        estimate = self.estimate_cost(input_tokens, output_tokens, model_name, gpu_type)
        # C16: Use actual end_time, not retroactive start_time
        estimate.end_time = time.time()
        estimate.start_time = estimate.end_time - (duration_ms / 1000) if duration_ms > 0 else estimate.end_time
        estimate.ttft_ms = ttft_ms
        estimate.total_duration_ms = duration_ms
        estimate.tokens_per_second = (
            output_tokens / (duration_ms / 1000) if duration_ms > 0 else 0
        )

        with self._lock:
            self._history.append(estimate)

            # E1: Update running aggregates
            self._total_cost += estimate.estimated_cost_usd
            self._total_cloud_cost += estimate.cloud_total_cost
            self._total_savings += estimate.savings_vs_cloud
            self._total_requests += 1
            self._total_tps += estimate.tokens_per_second

            # C2: Reset periods before updating
            self._reset_period_if_needed(tenant_id)

            # Update aggregate costs
            hour_cost, hour_start = self._hourly_costs.get(tenant_id, (0.0, time.time()))
            self._hourly_costs[tenant_id] = (hour_cost + estimate.estimated_cost_usd, hour_start)

            day_cost, day_start = self._daily_costs.get(tenant_id, (0.0, time.time()))
            self._daily_costs[tenant_id] = (day_cost + estimate.estimated_cost_usd, day_start)

            month_cost, month_start = self._monthly_costs.get(tenant_id, (0.0, time.time()))
            self._monthly_costs[tenant_id] = (month_cost + estimate.estimated_cost_usd, month_start)

            # Update budget tracking
            budget = self._budgets.get(tenant_id)
            if budget:
                budget.current_hour_cost = self._hourly_costs.get(tenant_id, (0.0, 0))[0]
                budget.current_day_cost = self._daily_costs.get(tenant_id, (0.0, 0))[0]
                budget.current_month_cost = self._monthly_costs.get(tenant_id, (0.0, 0))[0]

        # Record in tenant attribution if available
        if self._tenant_attribution and routing:
            try:
                self._tenant_attribution.record(
                    tenant_id=tenant_id,
                    request_id=f"req-{int(time.time() * 1000)}",
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    estimated_cost_usd=estimate.estimated_cost_usd,
                    routing=routing,
                    latency_ms=duration_ms,
                    ttft_ms=ttft_ms,
                    tokens_per_second=estimate.tokens_per_second,
                )
            except Exception as e:
                logger.debug(f"Tenant attribution recording failed: {e}")

        return estimate

    def get_cost_headers(self, estimate: CostEstimate, routing: "RoutingAttribution | None" = None) -> dict[str, str]:
        """Generate HTTP headers for cost information.

        Returns headers to include in API responses:
        - X-DistLLM-Cost: Estimated cost in USD
        - X-DistLLM-Tokens: Input/output/total tokens
        - X-DistLLM-Savings: Savings vs cloud API
        - X-DistLLM-GPU-Time: GPU seconds used
        - X-DistLLM-Tokens-Per-Second: Throughput
        - X-DistLLM-Source: Compute source (cloud/peer/federated)
        - X-DistLLM-Provider: Provider name
        - X-DistLLM-Region: Region used
        - X-DistLLM-Carbon: Carbon intensity (gCO2/kWh)
        """
        headers = {
            "X-DistLLM-Cost": f"{estimate.estimated_cost_usd:.8f}",
            "X-DistLLM-Tokens": f"{estimate.input_tokens}/{estimate.output_tokens}/{estimate.total_tokens}",
            "X-DistLLM-GPU-Time": f"{estimate.estimated_gpu_seconds:.4f}",
            "X-DistLLM-Tokens-Per-Second": f"{estimate.tokens_per_second:.1f}",
        }
        if estimate.cloud_total_cost > 0:
            headers["X-DistLLM-Savings"] = f"{estimate.savings_vs_cloud:.8f}"
            headers["X-DistLLM-Savings-Pct"] = f"{estimate.savings_pct:.1f}"
        if estimate.ttft_ms > 0:
            headers["X-DistLLM-TTFT"] = f"{estimate.ttft_ms:.1f}"
        if estimate.total_duration_ms > 0:
            headers["X-DistLLM-Latency"] = f"{estimate.total_duration_ms:.1f}"
        if routing:
            headers.update(routing.to_headers())
        return headers

    def get_cost_summary(self, tenant_id: str = "") -> dict[str, Any]:
        """Get cost summary for a tenant or overall.

        E1: Uses running aggregates for O(1) performance instead of
        iterating the full history deque.
        """
        with self._lock:
            if tenant_id:
                # C2: Reset periods before returning
                self._reset_period_if_needed(tenant_id)
                return {
                    "tenant_id": tenant_id,
                    "cost_last_hour": self._hourly_costs.get(tenant_id, (0.0, 0))[0],
                    "cost_last_day": self._daily_costs.get(tenant_id, (0.0, 0))[0],
                    "cost_last_month": self._monthly_costs.get(tenant_id, (0.0, 0))[0],
                    "budget": self._budgets.get(tenant_id, CostBudget()).__dict__,
                }
            # E1: O(1) summary using running aggregates
            return {
                "total_requests_tracked": self._total_requests,
                "total_cost_usd": self._total_cost,
                "total_cloud_cost_usd": self._total_cloud_cost,
                "total_savings_usd": self._total_savings,
                "avg_cost_per_request": (
                    self._total_cost / max(self._total_requests, 1)
                ),
                "avg_tokens_per_second": (
                    self._total_tps / max(self._total_requests, 1)
                ),
            }

    def get_history(self, limit: int = 100) -> list[dict[str, Any]]:
        """Get recent cost history."""
        with self._lock:
            recent = self._history[-limit:]
            return [
                {
                    "model": e.model_name,
                    "input_tokens": e.input_tokens,
                    "output_tokens": e.output_tokens,
                    "cost_usd": e.estimated_cost_usd,
                    "cloud_cost_usd": e.cloud_total_cost,
                    "savings_pct": e.savings_pct,
                    "duration_ms": e.total_duration_ms,
                    "tokens_per_sec": e.tokens_per_second,
                    "ttft_ms": e.ttft_ms,
                }
                for e in recent
            ]


def _estimate_throughput(gpu_type: str, model_name: str) -> float:
    """Estimate tokens/second based on GPU type and model size.

    C6: Uses regex-based model size extraction instead of fragile substring matching.
    E4: Config-driven model→throughput mapping with proper regex support.
    """
    # Base throughput for a 7B model
    base_tps: dict[str, float] = {
        "H100": 3000,
        "A100-80GB": 2000,
        "A100-40GB": 1500,
        "A6000": 1000,
        "RTX-4090": 1200,
        "RTX-3090": 800,
        "RTX-3080": 600,
        "RTX-3070": 400,
        "Apple-M2-Ultra": 500,
        "Apple-M2-Pro": 300,
        "Apple-M1": 200,
        "CPU": 20,
    }
    tps = base_tps.get(gpu_type, 100)

    model_lower = model_name.lower()

    # Check for special architectures first (before generic size extraction)
    if "8x7b" in model_lower or "8x22b" in model_lower:
        return tps * 0.25

    # C6: Extract model size using regex for robust matching
    # Match patterns like "70b", "70B", "70-billion", "70B-Instruct"
    size_match = re.search(r'(\d+)\s*[bB]', model_lower)
    if size_match:
        size_billions = int(size_match.group(1))
        if size_billions >= 65:
            tps *= 0.15
        elif size_billions >= 30:
            tps *= 0.3
        elif size_billions >= 13:
            tps *= 0.5
        # 7-12b models use base throughput (no scaling)

    return tps


def _match_cloud_api(model_name: str) -> str:
    """Match a model name to the closest cloud API for cost comparison.

    C7: Improved matching with explicit model family detection.
    """
    model_lower = model_name.lower()

    # OpenAI models
    if "gpt-4o-mini" in model_lower:
        return "gpt-4o-mini"
    if "gpt-4o" in model_lower:
        return "gpt-4o"
    if "gpt-4-turbo" in model_lower or "gpt-4-1106" in model_lower:
        return "gpt-4-turbo"
    if "gpt-4" in model_lower:
        return "gpt-4-turbo"
    if "gpt-3.5" in model_lower:
        return "gpt-3.5-turbo"

    # Anthropic models
    if "claude-3.5-sonnet" in model_lower or "claude-3-5-sonnet" in model_lower:
        return "claude-3.5-sonnet"
    if "claude-3-opus" in model_lower or "claude-opus" in model_lower:
        return "claude-3-opus"
    if "claude-3-haiku" in model_lower or "claude-haiku" in model_lower:
        return "claude-3-haiku"
    if "claude-3-sonnet" in model_lower or "claude-sonnet" in model_lower:
        return "claude-3-sonnet"

    # C7: Explicit model family detection before size-based matching
    if "deepseek" in model_lower:
        return "deepseek-v3"
    if "llama" in model_lower or "meta" in model_lower:
        size_match = re.search(r'(\d+)[bB]', model_lower)
        if size_match and int(size_match.group(1)) >= 30:
            return "llama-3.1-70b"
        return "llama-3.1-8b"
    if "mixtral" in model_lower:
        return "mixtral-8x7b"
    if "mistral" in model_lower:
        # Mistral: use size-based matching (not all mistral models are 8x7b)
        size_match = re.search(r'(\d+)[bB]', model_lower)
        if size_match and int(size_match.group(1)) >= 30:
            return "llama-3.1-70b"
        return "mixtral-8x7b"
    if "qwen" in model_lower:
        size_match = re.search(r'(\d+)[bB]', model_lower)
        if size_match and int(size_match.group(1)) >= 30:
            return "llama-3.1-70b"
        return "llama-3.1-8b"

    # Fallback: size-based matching for unknown families
    size_match = re.search(r'(\d+)[bB]', model_lower)
    if size_match:
        size = int(size_match.group(1))
        if size >= 30:
            return "llama-3.1-70b"
        return "llama-3.1-8b"

    return ""


# ── Module-level singleton ──────────────────────────────────────────────────

_tracker: CostTracker | None = None
_tracker_lock = threading.Lock()


def get_cost_tracker() -> CostTracker:
    """Get or create the module-level CostTracker singleton."""
    global _tracker
    if _tracker is None:
        with _tracker_lock:
            if _tracker is None:
                _tracker = CostTracker()
    return _tracker


def reset_cost_tracker() -> None:
    """C14: Reset the singleton for testing."""
    global _tracker
    _tracker = None
