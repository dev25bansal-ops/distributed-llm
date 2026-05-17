"""Cost-aware auto-scaling for distributed LLM inference.

Provides:
- GPU-hour tracking per request
- Spot instance preemption prediction
- Budget-aware scheduling: prefer cheaper nodes when latency SLA allows
- Per-tenant cost tracking and reporting
- Auto-scaling decisions based on cost/latency tradeoffs
"""

import time
from dataclasses import dataclass, field
from loguru import logger


@dataclass
class RequestCost:
    """Cost breakdown for a single request."""
    request_id: str
    tenant_id: str
    gpu_hours: float = 0.0
    compute_cost: float = 0.0
    memory_cost: float = 0.0
    total_cost: float = 0.0
    start_time: float = field(default_factory=time.time)
    end_time: float = 0.0
    tokens_generated: int = 0
    node_id: str = ""


@dataclass
class TenantCostReport:
    """Cost report for a tenant over a time period."""
    tenant_id: str
    period_start: float
    period_end: float
    total_requests: int = 0
    total_gpu_hours: float = 0.0
    total_compute_cost: float = 0.0
    total_memory_cost: float = 0.0
    total_cost: float = 0.0
    total_tokens: int = 0
    avg_cost_per_token: float = 0.0
    avg_cost_per_request: float = 0.0
    cost_by_node: dict[str, float] = field(default_factory=dict)


@dataclass
class PreemptionRisk:
    """Spot instance preemption risk assessment."""
    node_id: str
    risk_score: float = 0.0  # 0.0-1.0
    current_price: float = 0.0
    on_demand_price: float = 0.0
    price_history: list[float] = field(default_factory=list)
    interruption_history: int = 0
    uptime_hours: float = 0.0
    recommendation: str = "keep"  # keep, drain, replace


class GPUCostTracker:
    """Tracks GPU-hour consumption per request and per tenant.

    GPU-hours = (GPU time in seconds / 3600) * GPU count
    Cost = GPU-hours * GPU hourly rate
    """

    def __init__(self, gpu_hourly_rate: float = 1.0):
        self._gpu_hourly_rate = gpu_hourly_rate
        self._requests: dict[str, RequestCost] = {}
        self._tenant_requests: dict[str, list[str]] = {}
        self._completed_requests: list[RequestCost] = []

    def start_request(
        self,
        request_id: str,
        tenant_id: str,
        node_id: str,
        gpu_count: int = 1,
    ) -> None:
        """Start tracking cost for a new request.

        Args:
            request_id: Unique request identifier.
            tenant_id: Tenant identifier.
            node_id: Node executing the request.
            gpu_count: Number of GPUs used.
        """
        self._requests[request_id] = RequestCost(
            request_id=request_id,
            tenant_id=tenant_id,
            node_id=node_id,
        )
        if tenant_id not in self._tenant_requests:
            self._tenant_requests[tenant_id] = []
        self._tenant_requests[tenant_id].append(request_id)

    def record_tokens(self, request_id: str, token_count: int) -> None:
        """Record generated tokens for cost attribution.

        Args:
            request_id: Request identifier.
            token_count: Number of tokens generated in this step.
        """
        req = self._requests.get(request_id)
        if req:
            req.tokens_generated += token_count

    def complete_request(self, request_id: str) -> RequestCost | None:
        """Finalize cost tracking for a request.

        Args:
            request_id: Request identifier.

        Returns:
            Finalized RequestCost, or None if not found.
        """
        req = self._requests.pop(request_id, None)
        if req is None:
            return None

        req.end_time = time.time()
        duration_hours = (req.end_time - req.start_time) / 3600.0
        req.gpu_hours = duration_hours

        # Compute cost breakdown
        req.compute_cost = req.gpu_hours * self._gpu_hourly_rate
        # Memory cost: ~10% of compute cost (rough estimate)
        req.memory_cost = req.compute_cost * 0.1
        req.total_cost = req.compute_cost + req.memory_cost

        self._completed_requests.append(req)
        return req

    def get_tenant_cost(self, tenant_id: str) -> float:
        """Get total cost for a tenant (completed + in-progress)."""
        total = 0.0
        now = time.time()

        # Completed requests
        for req in self._completed_requests:
            if req.tenant_id == tenant_id:
                total += req.total_cost

        # In-progress requests (estimated)
        for req in self._requests.values():
            if req.tenant_id == tenant_id:
                elapsed = (now - req.start_time) / 3600.0
                total += elapsed * self._gpu_hourly_rate * 1.1  # + memory

        return total

    def get_tenant_report(
        self,
        tenant_id: str,
        period_start: float | None = None,
        period_end: float | None = None,
    ) -> TenantCostReport:
        """Generate a detailed cost report for a tenant.

        Args:
            tenant_id: Tenant identifier.
            period_start: Start of reporting period (Unix timestamp).
            period_end: End of reporting period.

        Returns:
            TenantCostReport with full breakdown.
        """
        now = time.time()
        start = period_start or 0.0
        end = period_end or now

        report = TenantCostReport(
            tenant_id=tenant_id,
            period_start=start,
            period_end=end,
        )

        for req in self._completed_requests:
            if req.tenant_id != tenant_id:
                continue
            if req.end_time < start or req.start_time > end:
                continue

            report.total_requests += 1
            report.total_gpu_hours += req.gpu_hours
            report.total_compute_cost += req.compute_cost
            report.total_memory_cost += req.memory_cost
            report.total_cost += req.total_cost
            report.total_tokens += req.tokens_generated

            # Per-node breakdown
            if req.node_id:
                report.cost_by_node[req.node_id] = (
                    report.cost_by_node.get(req.node_id, 0.0) + req.total_cost
                )

        if report.total_tokens > 0:
            report.avg_cost_per_token = report.total_cost / report.total_tokens
        if report.total_requests > 0:
            report.avg_cost_per_request = report.total_cost / report.total_requests

        return report

    def get_all_tenant_reports(self) -> dict[str, TenantCostReport]:
        """Generate cost reports for all tenants."""
        tenant_ids = set()
        for req in self._completed_requests:
            tenant_ids.add(req.tenant_id)
        for req in self._requests.values():
            tenant_ids.add(req.tenant_id)

        return {
            tid: self.get_tenant_report(tid) for tid in tenant_ids
        }

    def stats(self) -> dict:
        """Get overall tracking statistics."""
        return {
            "active_requests": len(self._requests),
            "completed_requests": len(self._completed_requests),
            "gpu_hourly_rate": self._gpu_hourly_rate,
        }


class PreemptionPredictor:
    """Predicts spot instance preemption risk based on price history.

    Uses simple heuristics:
    - Rising spot price relative to on-demand = higher risk
    - Frequent past interruptions = higher risk
    - Low uptime on spot = higher risk (new instances more likely preempted)
    """

    # Thresholds
    HIGH_RISK_RATIO = 0.7  # Spot price > 70% of on-demand
    MEDIUM_RISK_RATIO = 0.5  # Spot price > 50% of on-demand

    def __init__(self, price_history_window: int = 24):
        # hours of price history to consider
        self._price_history_window = price_history_window
        self._node_history: dict[str, PreemptionRisk] = {}

    def update_price(
        self,
        node_id: str,
        spot_price: float,
        on_demand_price: float,
    ) -> None:
        """Record a price update for a spot node.

        Args:
            node_id: Spot node identifier.
            spot_price: Current spot price per hour.
            on_demand_price: On-demand price per hour.
        """
        if node_id not in self._node_history:
            self._node_history[node_id] = PreemptionRisk(
                node_id=node_id,
                on_demand_price=on_demand_price,
            )

        risk = self._node_history[node_id]
        risk.current_price = spot_price
        risk.on_demand_price = on_demand_price
        risk.price_history.append(spot_price)

        # Keep only recent history
        if len(risk.price_history) > self._price_history_window:
            risk.price_history = risk.price_history[-self._price_history_window:]

    def record_interruption(self, node_id: str) -> None:
        """Record that a node was interrupted."""
        if node_id in self._node_history:
            self._node_history[node_id].interruption_history += 1

    def set_uptime(self, node_id: str, hours: float) -> None:
        """Set the current uptime of a node."""
        if node_id in self._node_history:
            self._node_history[node_id].uptime_hours = hours

    def assess_risk(self, node_id: str) -> PreemptionRisk:
        """Assess preemption risk for a node.

        Args:
            node_id: Node identifier.

        Returns:
            PreemptionRisk with score and recommendation.
        """
        risk = self._node_history.get(node_id)
        if risk is None:
            return PreemptionRisk(
                node_id=node_id,
                risk_score=0.0,
                recommendation="keep",
            )

        score = 0.0

        # Factor 1: Price ratio (0-0.4)
        if risk.on_demand_price > 0:
            ratio = risk.current_price / risk.on_demand_price
            if ratio >= self.HIGH_RISK_RATIO:
                score += 0.4
            elif ratio >= self.MEDIUM_RISK_RATIO:
                score += 0.2

        # Factor 2: Price trend (0-0.2)
        if len(risk.price_history) >= 3:
            recent = risk.price_history[-3:]
            if recent[-1] > recent[0] * 1.1:  # 10% increase
                score += 0.2

        # Factor 3: Interruption history (0-0.2)
        if risk.interruption_history > 3:
            score += 0.2
        elif risk.interruption_history > 1:
            score += 0.1

        # Factor 4: Low uptime (0-0.2)
        if risk.uptime_hours < 1.0:
            score += 0.2
        elif risk.uptime_hours < 4.0:
            score += 0.1

        risk.risk_score = min(score, 1.0)

        # Recommendation
        if score >= 0.7:
            risk.recommendation = "replace"
        elif score >= 0.4:
            risk.recommendation = "drain"
        else:
            risk.recommendation = "keep"

        return risk

    def get_risky_nodes(self, threshold: float = 0.5) -> list[str]:
        """Get list of nodes with preemption risk above threshold.

        Args:
            threshold: Minimum risk score (0.0-1.0).

        Returns:
            List of high-risk node IDs.
        """
        risky = []
        for node_id in self._node_history:
            risk = self.assess_risk(node_id)
            if risk.risk_score >= threshold:
                risky.append(node_id)
        return risky


class CostAwareScaler:
    """Orchestrates cost-aware auto-scaling decisions.

    Combines GPU cost tracking, preemption prediction, and budget
    constraints to make scaling decisions that minimize cost while
    meeting latency SLAs.
    """

    def __init__(
        self,
        gpu_hourly_rate: float = 1.0,
        budget_per_hour: float = 0.0,
        cost_per_token: float = 0.0,
    ):
        self._gpu_tracker = GPUCostTracker(gpu_hourly_rate=gpu_hourly_rate)
        self._preemption = PreemptionPredictor()
        self._budget_per_hour = budget_per_hour
        self._cost_per_token = cost_per_token
        self._scaling_events: list[dict] = []

    # --- GPU cost tracking ---
    def start_request(
        self, request_id: str, tenant_id: str, node_id: str, gpu_count: int = 1
    ) -> None:
        """Start cost tracking for a request."""
        self._gpu_tracker.start_request(request_id, tenant_id, node_id, gpu_count)

    def record_tokens(self, request_id: str, token_count: int) -> None:
        """Record generated tokens."""
        self._gpu_tracker.record_tokens(request_id, token_count)

    def complete_request(self, request_id: str) -> RequestCost | None:
        """Finalize cost tracking for a request."""
        return self._gpu_tracker.complete_request(request_id)

    # --- Tenant cost reporting ---
    def get_tenant_cost(self, tenant_id: str) -> float:
        """Get total cost for a tenant."""
        return self._gpu_tracker.get_tenant_cost(tenant_id)

    def get_tenant_report(
        self,
        tenant_id: str,
        period_start: float | None = None,
        period_end: float | None = None,
    ) -> TenantCostReport:
        """Get detailed cost report for a tenant."""
        return self._gpu_tracker.get_tenant_report(tenant_id, period_start, period_end)

    def get_all_tenant_reports(self) -> dict[str, TenantCostReport]:
        """Get cost reports for all tenants."""
        return self._gpu_tracker.get_all_tenant_reports()

    # --- Preemption prediction ---
    def update_node_price(
        self, node_id: str, spot_price: float, on_demand_price: float
    ) -> None:
        """Update spot price for a node."""
        self._preemption.update_price(node_id, spot_price, on_demand_price)

    def assess_node_risk(self, node_id: str) -> PreemptionRisk:
        """Assess preemption risk for a node."""
        return self._preemption.assess_risk(node_id)

    def get_risky_nodes(self, threshold: float = 0.5) -> list[str]:
        """Get nodes at high preemption risk."""
        return self._preemption.get_risky_nodes(threshold)

    # --- Budget-aware scheduling ---
    def can_afford_node(self, hourly_cost: float) -> bool:
        """Check if adding a node fits within budget.

        Args:
            hourly_cost: Hourly cost of the new node.

        Returns:
            True if within budget.
        """
        if self._budget_per_hour <= 0:
            return True  # No budget limit
        current_spend = self._gpu_tracker.stats().get("active_requests", 0) * self._gpu_tracker._gpu_hourly_rate
        return current_spend + hourly_cost <= self._budget_per_hour

    def select_node_for_sla(
        self,
        candidates: list[dict],
        target_latency_ms: float = 200.0,
    ) -> dict | None:
        """Select the best node balancing cost and latency SLA.

        Args:
            candidates: List of node dicts with keys:
                - node_id, cost_per_hour, is_spot, avg_latency_ms
            target_latency_ms: Maximum acceptable latency.

        Returns:
            Selected node dict, or None.
        """
        if not candidates:
            return None

        # Filter nodes that meet latency SLA
        sla_compliant = [
            n for n in candidates if n.get("avg_latency_ms", float("inf")) <= target_latency_ms
        ]

        # If no SLA-compliant nodes, relax constraint
        pool = sla_compliant if sla_compliant else candidates

        # Filter by budget
        if self._budget_per_hour > 0:
            pool = [n for n in pool if n.get("cost_per_hour", 0) <= self._budget_per_hour]

        if not pool:
            return None

        # Score each node: prefer cheaper + lower risk
        scored = []
        for node in pool:
            node_id = node.get("node_id", "")
            cost = node.get("cost_per_hour", 0)
            is_spot = node.get("is_spot", False)

            # Risk assessment
            risk = self.assess_node_risk(node_id)
            risk_penalty = risk.risk_score * 0.5  # Up to 50% penalty

            # Score: lower is better
            score = cost * (1 + risk_penalty)
            if is_spot:
                score *= 0.8  # Spot discount preference

            scored.append((score, node))

        scored.sort(key=lambda x: x[0])
        return scored[0][1]

    # --- Auto-scaling decisions ---
    def should_scale_up(
        self,
        current_qps: float,
        avg_latency_ms: float,
        target_latency_ms: float,
        current_nodes: int,
        node_capacity_qps: float = 10.0,
    ) -> bool:
        """Determine if we should add more nodes.

        Args:
            current_qps: Current queries per second.
            avg_latency_ms: Current average latency.
            target_latency_ms: Target latency threshold.
            current_nodes: Number of active nodes.
            node_capacity_qps: QPS capacity per node.

        Returns:
            True if scaling up is recommended.
        """
        # Latency-driven scaling
        if avg_latency_ms > target_latency_ms * 1.5:
            return self.can_afford_node(self._gpu_tracker._gpu_hourly_rate)

        # Capacity-driven scaling
        total_capacity = current_nodes * node_capacity_qps
        if current_qps > total_capacity * 0.8:  # 80% utilization threshold
            return self.can_afford_node(self._gpu_tracker._gpu_hourly_rate)

        return False

    def should_scale_down(
        self,
        current_qps: float,
        current_nodes: int,
        node_capacity_qps: float = 10.0,
        min_nodes: int = 1,
    ) -> bool:
        """Determine if we can remove nodes to save cost.

        Args:
            current_qps: Current queries per second.
            current_nodes: Number of active nodes.
            node_capacity_qps: QPS capacity per node.
            min_nodes: Minimum nodes to keep.

        Returns:
            True if scaling down is recommended.
        """
        if current_nodes <= min_nodes:
            return False

        total_capacity = current_nodes * node_capacity_qps
        utilization = current_qps / total_capacity if total_capacity > 0 else 0

        # Scale down if utilization < 30%
        return utilization < 0.3

    def stats(self) -> dict:
        """Get comprehensive scaler statistics."""
        return {
            "gpu_tracker": self._gpu_tracker.stats(),
            "budget_per_hour": self._budget_per_hour,
            "risky_nodes": self._preemption.get_risky_nodes(),
            "scaling_events": len(self._scaling_events),
        }
