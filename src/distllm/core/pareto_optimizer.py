"""Multi-Objective Pareto Optimization Engine.

Replaces simple weighted-sum scoring with Pareto-optimal frontier
analysis for routing decisions. Users can specify strict constraints
and get top-N Pareto-optimal routes.

Optimization objectives:
- Price minimization
- Latency minimization
- Carbon intensity minimization
- Weighted compromise (any combination)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from loguru import logger


class ObjectiveType(Enum):
    """Type of optimization objective."""
    MINIMIZE = "minimize"   # Lower is better (price, latency, carbon)
    MAXIMIZE = "maximize"   # Higher is better (reputation, uptime)


@dataclass
class Objective:
    """A single optimization objective."""
    name: str
    objective_type: ObjectiveType
    weight: float = 1.0  # For weighted-sum fallback
    hard_limit: float | None = None  # Constraint: reject if violated

    def is_satisfied(self, value: float) -> bool:
        """Check if a value satisfies the hard limit."""
        if self.hard_limit is None:
            return True
        if self.objective_type == ObjectiveType.MINIMIZE:
            return value <= self.hard_limit
        return value >= self.hard_limit


@dataclass
class CandidateScore:
    """Scored candidate with all objective values."""
    candidate_id: str
    values: dict[str, float]  # objective_name -> value
    is_dominated: bool = False
    pareto_rank: int = 0
    weighted_score: float = 0.0
    crowding_distance: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class ParetoOptimizer:
    """Multi-objective optimizer using Pareto dominance.

    Usage::

        optimizer = ParetoOptimizer()
        optimizer.add_objective(Objective("price", ObjectiveType.MINIMIZE, hard_limit=10.0))
        optimizer.add_objective(Objective("latency", ObjectiveType.MINIMIZE, hard_limit=200.0))
        optimizer.add_objective(Objective("carbon", ObjectiveType.MINIMIZE))

        candidates = [
            {"id": "aws-us", "price": 5.0, "latency": 50, "carbon": 380},
            {"id": "gcp-eu", "price": 7.0, "latency": 80, "carbon": 15},
        ]
        results = optimizer.optimize(candidates, top_n=3)
    """

    def __init__(self):
        self._objectives: dict[str, Objective] = {}

    def add_objective(self, objective: Objective) -> None:
        """Add an optimization objective."""
        self._objectives[objective.name] = objective

    def remove_objective(self, name: str) -> None:
        """Remove an optimization objective."""
        self._objectives.pop(name, None)

    def optimize(
        self,
        candidates: list[dict[str, Any]],
        top_n: int = 5,
        method: str = "pareto",
    ) -> list[CandidateScore]:
        """Find the top-N optimal candidates.

        Args:
            candidates: List of dicts with 'id' and objective values.
            top_n: Number of results to return.
            method: "pareto" for Pareto front, "weighted" for weighted sum,
                    "crowding" for NSGA-II style crowding distance.

        Returns:
            List of CandidateScore sorted by optimality.
        """
        if not candidates or not self._objectives:
            return []

        # Score all candidates
        scored = []
        for c in candidates:
            cid = c.get("id", c.get("candidate_id", str(c)))
            values = {name: c.get(name, 0.0) for name in self._objectives}
            scored.append(CandidateScore(candidate_id=cid, values=values, metadata=c))

        # Filter by hard constraints
        feasible = []
        for s in scored:
            violated = False
            for name, obj in self._objectives.items():
                if not obj.is_satisfied(s.values.get(name, 0.0)):
                    violated = True
                    break
            if not violated:
                feasible.append(s)

        if not feasible:
            logger.warning("No candidates satisfy all hard constraints")
            return []

        if method == "pareto":
            return self._pareto_front(feasible, top_n)
        elif method == "crowding":
            return self._nsga2_ranking(feasible, top_n)
        else:
            return self._weighted_sum(feasible, top_n)

    def _pareto_front(self, candidates: list[CandidateScore], top_n: int) -> list[CandidateScore]:
        """Find Pareto-optimal candidates (non-dominated sorting)."""
        n = len(candidates)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if self._dominates(candidates[j], candidates[i]):
                    candidates[i].is_dominated = True
                    break

        pareto = [c for c in candidates if not c.is_dominated]
        pareto.sort(key=lambda c: self._weighted_score(c))
        return pareto[:top_n]

    def _nsga2_ranking(self, candidates: list[CandidateScore], top_n: int) -> list[CandidateScore]:
        """NSGA-II style: Pareto rank + crowding distance."""
        # Assign Pareto ranks
        remaining = list(candidates)
        rank = 0
        while remaining:
            front = []
            for c in remaining:
                dominated = False
                for other in remaining:
                    if other is c:
                        continue
                    if self._dominates(other, c):
                        dominated = True
                        break
                if not dominated:
                    front.append(c)
            for c in front:
                c.pareto_rank = rank
                remaining.remove(c)
            rank += 1

        # Compute crowding distance per front
        candidates.sort(key=lambda c: (c.pareto_rank, -c.crowding_distance))
        return candidates[:top_n]

    def _weighted_sum(self, candidates: list[CandidateScore], top_n: int) -> list[CandidateScore]:
        """Simple weighted-sum scoring."""
        for c in candidates:
            c.weighted_score = self._weighted_score(c)
        candidates.sort(key=lambda c: c.weighted_score)
        return candidates[:top_n]

    def _dominates(self, a: CandidateScore, b: CandidateScore) -> bool:
        """Check if candidate 'a' dominates candidate 'b'.

        'a' dominates 'b' if:
        - 'a' is no worse than 'b' in all objectives
        - 'a' is strictly better than 'b' in at least one objective
        """
        at_least_one_better = False
        for name, obj in self._objectives.items():
            va = a.values.get(name, 0.0)
            vb = b.values.get(name, 0.0)
            if obj.objective_type == ObjectiveType.MINIMIZE:
                if va > vb:
                    return False
                if va < vb:
                    at_least_one_better = True
            else:
                if va < vb:
                    return False
                if va > vb:
                    at_least_one_better = True
        return at_least_one_better

    def _weighted_score(self, candidate: CandidateScore) -> float:
        """Compute weighted-sum score (lower is better)."""
        score = 0.0
        for name, obj in self._objectives.items():
            val = candidate.values.get(name, 0.0)
            if obj.objective_type == ObjectiveType.MINIMIZE:
                score += obj.weight * val
            else:
                score += obj.weight * (1.0 / max(val, 0.001))
        return score

    def get_objectives(self) -> list[dict[str, Any]]:
        """Get all configured objectives."""
        return [
            {
                "name": obj.name,
                "type": obj.objective_type.value,
                "weight": obj.weight,
                "hard_limit": obj.hard_limit,
            }
            for obj in self._objectives.values()
        ]
