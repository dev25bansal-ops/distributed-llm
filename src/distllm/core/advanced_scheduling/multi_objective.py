"""Pareto-optimal multi-objective scheduler.

Unifies cost-aware, energy-aware, latency, and throughput objectives into
a single scheduling policy with dynamic weight tuning based on system state.

Instead of static weights that conflict, this uses a scalarised
Pareto-front approach: each request gets a composite score that is a
weighted sum of normalised sub-scores, and the weights shift dynamically
as system state (load, temperature, cost, latency SLO attainment) changes.

Usage::

    scheduler = MultiObjectiveScheduler()
    scheduler.update_cost_profile("node-1", cost_per_hour=2.50)
    scheduler.update_energy_profile("node-1", EnergyProfile(...), 65.0)
    score = scheduler.score_request(request, system_state)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from loguru import logger


# ── Objective weights (adapted dynamically) ──────────────────────────────────

@dataclass
class ObjectiveWeights:
    """Current weights for each objective.

    Weights are normalised (sum to 1.0) and shifted dynamically based
    on system state.
    """
    latency: float = 0.30
    cost: float = 0.25
    energy: float = 0.20
    throughput: float = 0.25

    def normalize(self) -> ObjectiveWeights:
        total = self.latency + self.cost + self.energy + self.throughput
        if total > 0:
            return ObjectiveWeights(
                latency=self.latency / total,
                cost=self.cost / total,
                energy=self.energy / total,
                throughput=self.throughput / total,
            )
        return ObjectiveWeights(
            latency=0.25, cost=0.25, energy=0.25, throughput=0.25,
        )


@dataclass
class SystemState:
    """Snapshot of current system state for weight tuning."""
    avg_latency_p50_ms: float = 100.0
    avg_latency_p99_ms: float = 500.0
    latency_slo_ms: float = 1000.0
    node_cost_per_hour: float = 0.0
    gpu_temperature_c: float = 60.0
    gpu_thermal_threshold_c: float = 83.0
    cluster_utilization: float = 0.5  # 0.0-1.0
    power_draw_watts: float = 100.0
    power_capacity_watts: float = 1000.0
    slo_attainment_rate: float = 1.0  # 0.0-1.0 (1.0 = all requests meet SLO)


# ── Multi-Objective Scheduler ───────────────────────────────────────────────

class MultiObjectiveScheduler:
    """Pareto-optimal scheduler combining cost, energy, latency, throughput.

    Each objective produces a normalised score (0.0 = worst, 1.0 = best).
    The composite score is the weighted sum, and weights shift dynamically
    to reflect system state.

    Pareto front interpretation::
        Requests with similar composite scores are Pareto-equivalent —
        the scheduler picks the one with the best individual sub-score
        for the most-constrained objective.
    """

    def __init__(
        self,
        latency_weight: float = 0.30,
        cost_weight: float = 0.25,
        energy_weight: float = 0.20,
        throughput_weight: float = 0.25,
        weight_adaptation_rate: float = 0.1,
    ):
        self._weights = ObjectiveWeights(
            latency=latency_weight,
            cost=cost_weight,
            energy=energy_weight,
            throughput=throughput_weight,
        ).normalize()

        self._adapt_rate = weight_adaptation_rate
        self._total_requests = 0

        # Cost profiles: node_id -> cost_per_hour
        self._node_costs: dict[str, float] = {}
        # Energy profiles: node_id -> (EnergyProfile, current_temp)
        self._energy_profiles: dict[str, Any] = {}

    def update_cost_profile(self, node_id: str, cost_per_hour: float) -> None:
        """Update the cost profile for a node."""
        self._node_costs[node_id] = cost_per_hour

    def update_energy_profile(self, node_id: str, temperature_c: float) -> None:
        """Update the thermal profile for a node."""
        self._energy_profiles[node_id] = temperature_c

    # ── Per-objective score functions ─────────────────────────────────

    def _latency_score(self, req: Any, state: SystemState) -> float:
        """Score based on predicted latency vs SLO.

        1.0 = well within SLO, 0.0 = exceeds SLO.
        """
        est_latency = getattr(req, 'estimated_latency_ms', state.avg_latency_p50_ms)
        if est_latency <= 0 or state.latency_slo_ms <= 0:
            return 0.5
        ratio = est_latency / state.latency_slo_ms
        if ratio <= 0.5:
            return 1.0
        if ratio >= 1.0:
            return 0.0
        return 1.0 - (ratio - 0.5) / 0.5

    def _cost_score(self, req: Any, state: SystemState) -> float:
        """Score based on estimated cost.

        1.0 = cheapest node, 0.0 = most expensive.
        """
        node_id = getattr(req, 'node_id', None) or getattr(req, 'target_node', '')
        if not node_id or node_id not in self._node_costs:
            return 0.5
        cost = self._node_costs[node_id]
        if cost <= 0:
            return 1.0
        max_cost = max(self._node_costs.values()) if self._node_costs else 10.0
        return max(0.0, 1.0 - cost / max_cost)

    def _energy_score(self, req: Any, state: SystemState) -> float:
        """Score based on thermal pressure.

        1.0 = cool (high score), 0.0 = at thermal limit (low score).
        """
        max_temp = max(self._energy_profiles.values()) if self._energy_profiles else state.gpu_temperature_c
        threshold = state.gpu_thermal_threshold_c
        if max_temp <= 0 or threshold <= 0:
            return 0.5
        if max_temp >= threshold:
            return 0.0
        return max(0.0, 1.0 - max_temp / threshold)

    def _throughput_score(self, req: Any, state: SystemState) -> float:
        """Score based on token throughput potential.

        1.0 = high throughput, 0.0 = low throughput.
        """
        util = state.cluster_utilization
        # Prefer throughput when utilization is low (room to grow)
        if util < 0.3:
            return 1.0
        if util > 0.9:
            return 0.0
        return 1.0 - (util - 0.3) / 0.6

    # ── Dynamic weight tuning ────────────────────────────────────────

    def _adapt_weights(self, state: SystemState) -> ObjectiveWeights:
        """Shift weights based on current system state.

        - High latency vs SLO → increase latency weight
        - High temperature → increase energy weight
        - High cost → increase cost weight
        - Low SLO attainment → increase latency weight
        """
        w = self._weights

        # Latency pressure
        latency_ratio = state.avg_latency_p99_ms / max(state.latency_slo_ms, 1)
        if latency_ratio > 0.8:
            w.latency += self._adapt_rate * (latency_ratio - 0.8) * 2

        # SLO attainment pressure
        if state.slo_attainment_rate < 0.95:
            w.latency += self._adapt_rate * (1.0 - state.slo_attainment_rate) * 3

        # Thermal pressure
        temp_ratio = state.gpu_temperature_c / max(state.gpu_thermal_threshold_c, 1)
        if temp_ratio > 0.7:
            w.energy += self._adapt_rate * (temp_ratio - 0.7) * 2

        # Cost pressure (when cost is high)
        if state.node_cost_per_hour > 0:
            w.cost += self._adapt_rate * min(state.node_cost_per_hour / 10.0, 1.0)

        # Utilization: prefer throughput when low, latency when high
        if state.cluster_utilization < 0.3:
            w.throughput += self._adapt_rate * 0.5
        elif state.cluster_utilization > 0.8:
            w.latency += self._adapt_rate * 0.5

        self._weights = w.normalize()

        # Enforce minimum weight of 0.05 for each objective.
        # Redistribute the excess from the largest weight(s) so the
        # total remains 1.0 without re-normalization drift.
        min_w = 0.05
        under = sum(
            max(0, min_w - getattr(self._weights, attr))
            for attr in ('latency', 'cost', 'energy', 'throughput')
        )
        if under > 0:
            for attr in ('latency', 'cost', 'energy', 'throughput'):
                setattr(self._weights, attr, max(min_w, getattr(self._weights, attr)))
            # Subtract the excess from the largest weight
            largest = max(
                ('latency', 'cost', 'energy', 'throughput'),
                key=lambda a: getattr(self._weights, a),
            )
            setattr(
                self._weights, largest,
                max(0, getattr(self._weights, largest) - under),
            )

        return self._weights

    # ── Public API ───────────────────────────────────────────────────

    def score_request(
        self,
        request: Any,
        system_state: SystemState | None = None,
    ) -> float:
        """Compute the composite multi-objective score for a request.

        Args:
            request: Request object with optional attributes
                ``estimated_latency_ms``, ``node_id``, ``priority``.
            system_state: Current system state snapshot.  If None,
                default (neutral) state is used.

        Returns:
            Composite score (0.0 = worst, 1.0 = best).
        """
        state = system_state or SystemState()
        self._total_requests += 1

        # Adapt weights every N requests
        if self._total_requests % 10 == 0:
            self._adapt_weights(state)

        w = self._weights
        score = (
            w.latency * self._latency_score(request, state)
            + w.cost * self._cost_score(request, state)
            + w.energy * self._energy_score(request, state)
            + w.throughput * self._throughput_request_score(request, state)
        )
        return score

    def _throughput_request_score(self, req: Any, state: SystemState) -> float:
        """Per-request throughput estimate."""
        priority = getattr(req, 'priority', 2)
        # Higher priority requests get a throughput bonus
        base = self._throughput_score(req, state)
        priority_bonus = max(0.0, 1.0 - priority / 3.0) * 0.2  # priority 0 → +0.2, 3 → +0.0
        return min(1.0, base + priority_bonus)

    def select_best(
        self,
        requests: list[Any],
        system_state: SystemState | None = None,
    ) -> Any | None:
        """Select the best request from a list using multi-objective scoring.

        For requests with similar composite scores (within 5%), picks
        the one with the best score in the most-constrained objective.

        Args:
            requests: List of request objects.
            system_state: Current system state.

        Returns:
            The highest-scoring request, or None if the list is empty.
        """
        if not requests:
            return None

        scored = [(self.score_request(r, system_state), r) for r in requests]
        scored.sort(key=lambda x: -x[0])

        best_score, best_req = scored[0]
        if len(scored) > 1:
            second_score = scored[1][0]
            if second_score > best_score * 0.95:
                # Pareto-tie: pick by most-constrained objective
                state = system_state or SystemState()
                constrained = self._most_constrained_objective(state)
                best_req = self._resolve_pareto_tie(
                    scored[:3], constrained, state,
                )

        return best_req

    def _most_constrained_objective(self, state: SystemState) -> str:
        """Identify the most constrained objective."""
        if state.gpu_temperature_c > state.gpu_thermal_threshold_c * 0.8:
            return "energy"
        if state.avg_latency_p99_ms > state.latency_slo_ms * 0.8:
            return "latency"
        if state.node_cost_per_hour > 5.0:
            return "cost"
        return "throughput"

    def _resolve_pareto_tie(
        self,
        candidates: list[tuple[float, Any]],
        objective: str,
        state: SystemState,
    ) -> Any:
        """Resolve a Pareto tie by picking the best sub-score."""
        best_req = candidates[0][1]
        best_sub = -1.0

        for _score, req in candidates:
            if objective == "latency":
                sub = self._latency_score(req, state)
            elif objective == "cost":
                sub = self._cost_score(req, state)
            elif objective == "energy":
                sub = self._energy_score(req, state)
            else:
                sub = self._throughput_request_score(req, state)

            if sub > best_sub:
                best_sub = sub
                best_req = req

        return best_req

    @property
    def weights(self) -> ObjectiveWeights:
        return ObjectiveWeights(
            latency=self._weights.latency,
            cost=self._weights.cost,
            energy=self._weights.energy,
            throughput=self._weights.throughput,
        )

    @property
    def stats(self) -> dict:
        return {
            "weights": {
                "latency": round(self._weights.latency, 3),
                "cost": round(self._weights.cost, 3),
                "energy": round(self._weights.energy, 3),
                "throughput": round(self._weights.throughput, 3),
            },
            "total_requests": self._total_requests,
            "profiled_nodes": {
                "cost": len(self._node_costs),
                "energy": len(self._energy_profiles),
            },
        }
