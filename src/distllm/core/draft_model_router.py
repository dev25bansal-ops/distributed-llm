"""Heterogeneous Draft Fleet Router — SLA-based draft model selection.

Routes draft requests across a fleet of heterogeneous remote draft
models (different sizes, quantization levels, hardware) based on
latency SLA, cost budget, accuracy requirements, and current load.

This is the key differentiator: competitors (vLLM, SGLang) only
support local draft models.  DistLLM can route across *multiple*
remote draft endpoints with intelligent selection.

Usage::

    fleet = DraftModelFleet()
    fleet.register(DraftModelSpec(
        endpoint_url="http://cpu-node:8000/v1/completions",
        model_name="SmolLM-135M",
        hardware="cpu",
        cost_per_hour=0.05,
        avg_latency_ms=45.0,
    ))
    fleet.register(DraftModelSpec(
        endpoint_url="http://gpu-node:8001/v1/completions",
        model_name="SmolLM-360M",
        hardware="cuda:0",
        cost_per_hour=0.60,
        avg_latency_ms=8.0,
    ))

    router = DraftModelRouter(fleet)
    best = router.select(
        workload_type="code",
        max_latency_ms=50.0,
        max_cost_per_hour=1.0,
    )
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from loguru import logger


class WorkloadType(str, Enum):
    CODE = "code"
    INSTRUCTION = "instruction"
    REPETITIVE = "repetitive"
    DIVERSE = "diverse"
    UNKNOWN = "unknown"


@dataclass
class DraftModelSpec:
    """Specification for a single remote draft model endpoint."""
    endpoint_url: str
    model_name: str = ""
    api_key: str = ""
    hardware: str = "cpu"  # "cpu", "cuda:0", "mps", "edge"
    transport: str = "http"  # "http" or "grpc"
    cost_per_hour: float = 0.0
    avg_latency_ms: float = 0.0
    avg_acceptance_rate: float = 0.0
    max_concurrent: int = 10
    timeout_seconds: float = 30.0
    max_retries: int = 2
    verify_ssl: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DraftModelHealth:
    """Runtime health metrics for a draft model endpoint."""
    endpoint_url: str
    is_healthy: bool = True
    current_concurrent: int = 0
    total_calls: int = 0
    total_errors: int = 0
    total_latency_s: float = 0.0
    total_tokens: int = 0
    recent_latency_ms: float = 0.0
    recent_acceptance_rate: float = 0.0
    last_error: str = ""
    last_error_time: float = 0.0
    consecutive_failures: int = 0

    @property
    def avg_latency_ms(self) -> float:
        if self.total_calls == 0:
            return 0.0
        return (self.total_latency_s / self.total_calls) * 1000

    @property
    def error_rate(self) -> float:
        if self.total_calls == 0:
            return 0.0
        return self.total_errors / self.total_calls

    @property
    def tokens_per_second(self) -> float:
        if self.total_latency_s == 0:
            return 0.0
        return self.total_tokens / self.total_latency_s


@dataclass
class RoutingConstraints:
    """Constraints for draft model selection."""
    max_latency_ms: float = 100.0
    max_cost_per_hour: float = 10.0
    min_acceptance_rate: float = 0.0
    preferred_hardware: list[str] = field(default_factory=list)
    workload_type: str = "unknown"
    max_concurrent: int = 0  # 0 = no limit


@dataclass
class RoutingDecision:
    """Result of a routing decision."""
    selected_url: str
    selected_model: str
    selection_reason: str
    score: float
    candidates_evaluated: int
    candidates_qualified: int
    fallback_used: bool = False


class DraftModelFleet:
    """Registry of available remote draft model endpoints.

    Tracks specs, health, and routing state for a heterogeneous
    fleet of draft models across different hardware.
    """

    def __init__(self) -> None:
        self._specs: dict[str, DraftModelSpec] = {}
        self._health: dict[str, DraftModelHealth] = {}
        self._lock = threading.Lock()

    def register(self, spec: DraftModelSpec) -> None:
        """Register a draft model endpoint in the fleet."""
        with self._lock:
            self._specs[spec.endpoint_url] = spec
            if spec.endpoint_url not in self._health:
                self._health[spec.endpoint_url] = DraftModelHealth(
                    endpoint_url=spec.endpoint_url,
                )
            logger.info(
                f"Registered draft model: {spec.model_name} at {spec.endpoint_url} "
                f"({spec.hardware}, ${spec.cost_per_hour:.2f}/hr)"
            )

    def unregister(self, endpoint_url: str) -> None:
        """Remove a draft model endpoint from the fleet."""
        with self._lock:
            self._specs.pop(endpoint_url, None)
            self._health.pop(endpoint_url, None)
            logger.info(f"Unregistered draft model: {endpoint_url}")

    def get_spec(self, endpoint_url: str) -> DraftModelSpec | None:
        return self._specs.get(endpoint_url)

    def get_health(self, endpoint_url: str) -> DraftModelHealth | None:
        return self._health.get(endpoint_url)

    def get_all_specs(self) -> list[DraftModelSpec]:
        return list(self._specs.values())

    def get_all_health(self) -> dict[str, DraftModelHealth]:
        return dict(self._health)

    def record_success(
        self,
        endpoint_url: str,
        latency_s: float,
        tokens_generated: int,
        acceptance_rate: float = 0.0,
    ) -> None:
        """Record a successful draft model call."""
        with self._lock:
            health = self._health.get(endpoint_url)
            if health is None:
                return
            health.total_calls += 1
            health.total_latency_s += latency_s
            health.total_tokens += tokens_generated
            health.recent_latency_ms = latency_s * 1000
            health.recent_acceptance_rate = acceptance_rate
            health.consecutive_failures = 0
            health.is_healthy = True

    def record_error(self, endpoint_url: str, error: str) -> None:
        """Record a failed draft model call."""
        with self._lock:
            health = self._health.get(endpoint_url)
            if health is None:
                return
            health.total_errors += 1
            health.total_calls += 1
            health.last_error = error
            health.last_error_time = time.time()
            health.consecutive_failures += 1
            if health.consecutive_failures >= 3:
                health.is_healthy = False
                logger.warning(
                    f"Draft model {endpoint_url} marked unhealthy "
                    f"after {health.consecutive_failures} failures"
                )

    def mark_healthy(self, endpoint_url: str) -> None:
        with self._lock:
            health = self._health.get(endpoint_url)
            if health:
                health.is_healthy = True
                health.consecutive_failures = 0

    @property
    def healthy_endpoints(self) -> list[str]:
        return [
            url for url, h in self._health.items()
            if h.is_healthy and url in self._specs
        ]

    @property
    def size(self) -> int:
        return len(self._specs)


class DraftModelRouter:
    """Intelligent router for selecting draft model endpoints.

    Uses a weighted scoring function that balances:
    - Latency (lower is better)
    - Cost (lower is better)
    - Historical acceptance rate (higher is better)
    - Current load (lower is better)
    - Hardware preference (matching preferred hardware gets a bonus)

    Selection algorithm:
    1. Filter by constraints (latency, cost, health, concurrency)
    2. Score remaining candidates
    3. Pick highest-scored candidate
    4. If no candidates qualify, use relaxed fallback
    """

    def __init__(
        self,
        fleet: DraftModelFleet,
        latency_weight: float = 0.35,
        cost_weight: float = 0.20,
        acceptance_weight: float = 0.30,
        load_weight: float = 0.15,
    ) -> None:
        self._fleet = fleet
        self._latency_w = latency_weight
        self._cost_w = cost_weight
        self._acceptance_w = acceptance_weight
        self._load_w = load_weight
        self._last_decision: RoutingDecision | None = None

    def select(
        self,
        constraints: RoutingConstraints | None = None,
    ) -> RoutingDecision:
        """Select the best draft model endpoint for the given constraints.

        Returns a ``RoutingDecision`` with the selected endpoint and
        scoring details.  Never raises — falls back to the least-bad
        option if no endpoint fully qualifies.
        """
        if constraints is None:
            constraints = RoutingConstraints()

        specs = self._fleet.get_all_specs()
        if not specs:
            return RoutingDecision(
                selected_url="",
                selected_model="",
                selection_reason="no endpoints registered",
                score=0.0,
                candidates_evaluated=0,
                candidates_qualified=0,
                fallback_used=True,
            )

        evaluated = 0
        qualified: list[tuple[float, DraftModelSpec]] = []
        relaxed: list[tuple[float, DraftModelSpec]] = []

        for spec in specs:
            evaluated += 1
            health = self._fleet.get_health(spec.endpoint_url)
            if health is None:
                continue

            # Hard filter: must be healthy
            if not health.is_healthy:
                continue

            # Hard filter: concurrency
            if spec.max_concurrent > 0 and health.current_concurrent >= spec.max_concurrent:
                continue

            score = self._score(spec, health, constraints)

            # Check if this candidate meets all constraints
            meets_all = (
                health.recent_latency_ms <= constraints.max_latency_ms
                and spec.cost_per_hour <= constraints.max_cost_per_hour
                and health.recent_acceptance_rate >= constraints.min_acceptance_rate
            )

            if meets_all:
                qualified.append((score, spec))
            else:
                relaxed.append((score, spec))

        # Pick from qualified candidates
        candidates = qualified if qualified else relaxed
        fallback_used = not qualified and bool(relaxed)

        if not candidates:
            return RoutingDecision(
                selected_url="",
                selected_model="",
                selection_reason="no healthy endpoints available",
                score=0.0,
                candidates_evaluated=evaluated,
                candidates_qualified=0,
                fallback_used=True,
            )

        candidates.sort(key=lambda x: x[0], reverse=True)
        best_score, best_spec = candidates[0]

        reason = "best_score" if not fallback_used else "fallback_relaxed_constraints"
        decision = RoutingDecision(
            selected_url=best_spec.endpoint_url,
            selected_model=best_spec.model_name,
            selection_reason=reason,
            score=best_score,
            candidates_evaluated=evaluated,
            candidates_qualified=len(qualified),
            fallback_used=fallback_used,
        )
        self._last_decision = decision

        logger.debug(
            f"Draft router selected {best_spec.model_name} "
            f"(score={best_score:.3f}, {len(qualified)}/{evaluated} qualified)"
        )
        return decision

    def _score(
        self,
        spec: DraftModelSpec,
        health: DraftModelHealth,
        constraints: RoutingConstraints,
    ) -> float:
        """Score a draft model endpoint. Higher is better."""
        # Latency score: inverse of latency (lower latency = higher score)
        latency_ms = health.recent_latency_ms or spec.avg_latency_ms
        max_lat = max(constraints.max_latency_ms, 1.0)
        latency_score = max(0.0, 1.0 - (latency_ms / (max_lat * 2)))

        # Cost score: inverse of cost (lower cost = higher score)
        max_cost = max(constraints.max_cost_per_hour, 0.01)
        cost_score = max(0.0, 1.0 - (spec.cost_per_hour / (max_cost * 2)))

        # Acceptance rate score
        acceptance = health.recent_acceptance_rate or spec.avg_acceptance_rate
        acceptance_score = min(acceptance, 1.0)

        # Load score: lower concurrent = higher score
        max_conc = max(spec.max_concurrent, 1)
        load_score = max(0.0, 1.0 - (health.current_concurrent / max_conc))

        # Hardware preference bonus
        hw_bonus = 0.0
        if constraints.preferred_hardware:
            if spec.hardware in constraints.preferred_hardware:
                hw_bonus = 0.1

        total = (
            self._latency_w * latency_score
            + self._cost_w * cost_score
            + self._acceptance_w * acceptance_score
            + self._load_w * load_score
            + hw_bonus
        )
        return total

    @property
    def last_decision(self) -> RoutingDecision | None:
        return self._last_decision

    def fleet_stats(self) -> dict[str, Any]:
        """Return aggregate fleet statistics."""
        specs = self._fleet.get_all_specs()
        health_map = self._fleet.get_all_health()
        healthy = self._fleet.healthy_endpoints

        total_calls = sum(h.total_calls for h in health_map.values())
        total_errors = sum(h.total_errors for h in health_map.values())
        avg_latency = 0.0
        if total_calls > 0:
            total_lat = sum(h.total_latency_s for h in health_map.values())
            avg_latency = (total_lat / total_calls) * 1000

        return {
            "total_endpoints": len(specs),
            "healthy_endpoints": len(healthy),
            "total_calls": total_calls,
            "total_errors": total_errors,
            "error_rate": total_errors / max(total_calls, 1),
            "avg_latency_ms": round(avg_latency, 2),
            "endpoints": [
                {
                    "url": s.endpoint_url,
                    "model": s.model_name,
                    "hardware": s.hardware,
                    "cost_per_hour": s.cost_per_hour,
                    "healthy": health_map.get(s.endpoint_url, DraftModelHealth("")).is_healthy,
                    "calls": health_map.get(s.endpoint_url, DraftModelHealth("")).total_calls,
                    "avg_latency_ms": round(
                        health_map.get(s.endpoint_url, DraftModelHealth("")).avg_latency_ms, 2
                    ),
                }
                for s in specs
            ],
        }
