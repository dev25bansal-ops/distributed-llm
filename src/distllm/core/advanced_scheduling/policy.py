"""Core scheduling policy protocol and basic implementations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from distllm.core.batch_scheduler import IterationBudget, Sequence


@runtime_checkable
class SchedulingPolicy(Protocol):
    """Protocol for pluggable scheduling policies."""

    def compute_budget(self, base_budget: IterationBudget) -> IterationBudget:
        """Compute the iteration budget for this step."""
        ...

    def on_before_schedule(self, sequences: list[Sequence]) -> list[Sequence]:
        """Called before scheduling to allow priority modifications."""
        ...


@dataclass
class DefaultPolicy:
    """Passthrough policy — returns the base budget unchanged."""

    def compute_budget(self, base_budget: IterationBudget) -> IterationBudget:
        return base_budget

    def on_before_schedule(self, sequences: list[Sequence]) -> list[Sequence]:
        return sequences


@dataclass
class SarathiPolicy:
    """Sarathi-Serve style adaptive scheduling policy.

    Dynamically adjusts the prefill/decode split based on decode
    pipeline pressure.
    """

    pressure_threshold: float = 0.8
    prefill_scale_under_pressure: float = 0.5

    def compute_budget(self, base_budget: IterationBudget) -> IterationBudget:
        return base_budget

    def on_before_schedule(self, sequences: list[Sequence]) -> list[Sequence]:
        return sequences

    def should_disable_pressure_adaptation(self) -> bool:
        return False


@dataclass
class CompositePolicy:
    """Compose multiple policies — first wins on budget, all modify priorities."""

    policies: list[SchedulingPolicy] = None

    def __post_init__(self):
        if self.policies is None:
            self.policies = []

    def compute_budget(self, base_budget: IterationBudget) -> IterationBudget:
        for p in self.policies:
            base_budget = p.compute_budget(base_budget)
        return base_budget

    def on_before_schedule(self, sequences: list[Sequence]) -> list[Sequence]:
        for p in self.policies:
            sequences = p.on_before_schedule(sequences)
        return sequences
