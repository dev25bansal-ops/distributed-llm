"""Cost-aware provisioning via the digital twin / WhatIfEngine.

Connects real cloud :class:`~distllm.cloud.common.PriceQuote` data into the
what-if / digital-twin simulator so provisioning scenarios become
**cost-aware**: each candidate plan (provider / instance type / node count /
GPU type) is priced from its ``PriceQuote``, fed into the
:class:`~distllm.dist.simulation.digital_twin.WhatIfEngine`, and evaluated for
projected performance.  The optimiser then returns the *cheapest* plan whose
projected performance satisfies a supplied SLA constraint (e.g. throughput
>= target, latency_p99 <= budget).

No live cloud API calls are made here — ``PriceQuote`` instances are injected
by the caller (fetched elsewhere, or mocked in tests).  The
``PriceQuote`` -> ``WhatIfEngine`` wiring and the optimisation logic are real.

Usage::

    from distllm.cloud.common import PriceQuote
    from distllm.dist.simulation.digital_twin import DigitalTwin
    from distllm.dist.simulation.cost_aware_provisioning import (
        CostAwareProvisioner, ProvisioningPlan, SLAConstraint,
    )

    plans = [
        ProvisioningPlan("aws-a100", provider="aws",
                         instance_type="p4d.24xlarge", gpu_type="A100",
                         node_count=2, gpu_count=8,
                         quote=PriceQuote("aws", "p4d.24xlarge", "us-east-1",
                                          on_demand_hourly=32.77)),
        ...
    ]
    provisioner = CostAwareProvisioner(DigitalTwin())
    report = provisioner.optimize(plans, SLAConstraint(min_throughput=5.0))
    print(report.chosen.plan_id, report.chosen.hourly_cost)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from distllm.cloud.common import PriceQuote
from distllm.dist.simulation.digital_twin import (
    DigitalTwin,
    SimulationResult,
    WhatIfEngine,
)

# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


@dataclass
class ProvisioningPlan:
    """A candidate provisioning plan to evaluate.

    Attributes:
        plan_id: Human-readable identifier for the plan.
        provider: Cloud provider (``"aws"``, ``"gcp"``, ``"azure"``).
        instance_type: Provider instance type (e.g. ``"p4d.24xlarge"``).
        gpu_type: GPU hardware type used by the twin's throughput model
            (e.g. ``"A100"``, ``"H100"``).
        node_count: Number of nodes/instances to provision.
        gpu_count: GPUs per node.
        region: Cloud region.
        quote: The :class:`PriceQuote` for this instance type.  Injected by
            the caller (real fetch or mock) — never fetched live here.
        pricing_mode: Which field of the quote to bill from:
            ``"on_demand"`` (default), ``"spot"``, ``"reserved_1yr"`` or
            ``"reserved_3yr"``.
    """

    plan_id: str
    provider: str
    instance_type: str
    gpu_type: str
    node_count: int
    quote: PriceQuote
    gpu_count: int = 8
    region: str = ""
    pricing_mode: str = "on_demand"

    def unit_hourly(self) -> float:
        """Per-node hourly price selected from the quote by ``pricing_mode``."""
        mapping = {
            "on_demand": self.quote.on_demand_hourly,
            "spot": self.quote.spot_hourly,
            "reserved_1yr": self.quote.reserved_1yr_hourly,
            "reserved_3yr": self.quote.reserved_3yr_hourly,
        }
        price = mapping.get(self.pricing_mode, self.quote.on_demand_hourly)
        # Fall back to on-demand if the requested tier is unpriced (0.0).
        if price <= 0.0:
            price = self.quote.on_demand_hourly
        return price

    def fleet_hourly(self) -> float:
        """Total fleet cost per hour = per-node price * node_count."""
        return self.unit_hourly() * self.node_count


@dataclass
class SLAConstraint:
    """Performance / SLA constraint a plan must satisfy.

    All bounds are optional; ``None`` means "not constrained".

    Attributes:
        min_throughput: Minimum required completed requests/second.
        max_latency_p99: Maximum acceptable p99 latency in milliseconds.
        max_failures: Maximum acceptable number of failed requests.
    """

    min_throughput: float | None = None
    max_latency_p99: float | None = None
    max_failures: int | None = None

    def check(self, result: SimulationResult) -> tuple[bool, list[str]]:
        """Return ``(satisfied, violations)`` for a simulation result."""
        violations: list[str] = []
        if self.min_throughput is not None and result.throughput < self.min_throughput:
            violations.append(
                f"throughput {result.throughput} < min {self.min_throughput}"
            )
        if self.max_latency_p99 is not None and result.latency_p99 > self.max_latency_p99:
            violations.append(
                f"latency_p99 {result.latency_p99} > max {self.max_latency_p99}"
            )
        if self.max_failures is not None and result.failures > self.max_failures:
            violations.append(
                f"failures {result.failures} > max {self.max_failures}"
            )
        return (not violations, violations)


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------


@dataclass
class PlanEvaluation:
    """Result of evaluating a single provisioning plan through the twin.

    Attributes:
        plan: The evaluated :class:`ProvisioningPlan`.
        hourly_cost: Projected fleet cost per hour (from the ``PriceQuote``).
        result: The :class:`SimulationResult` from the digital twin.
        meets_sla: Whether the plan satisfied the SLA constraint.
        violations: Human-readable list of SLA violations (empty if satisfied).
        cost_per_throughput: ``hourly_cost / throughput`` — dollars per
            request/second; ``inf`` when throughput is zero.
    """

    plan: ProvisioningPlan
    hourly_cost: float
    result: SimulationResult
    meets_sla: bool
    violations: list[str] = field(default_factory=list)

    @property
    def cost_per_throughput(self) -> float:
        if self.result.throughput <= 0:
            return float("inf")
        return round(self.hourly_cost / self.result.throughput, 4)

    def as_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan.plan_id,
            "provider": self.plan.provider,
            "instance_type": self.plan.instance_type,
            "gpu_type": self.plan.gpu_type,
            "node_count": self.plan.node_count,
            "pricing_mode": self.plan.pricing_mode,
            "hourly_cost": round(self.hourly_cost, 4),
            "throughput": self.result.throughput,
            "latency_p99": self.result.latency_p99,
            "failures": self.result.failures,
            "meets_sla": self.meets_sla,
            "violations": list(self.violations),
            "cost_per_throughput": self.cost_per_throughput,
        }


@dataclass
class ProvisioningReport:
    """Full comparison across all candidate plans plus the chosen plan.

    Attributes:
        evaluations: Per-plan evaluations (input order preserved).
        chosen: The selected cost-optimal plan meeting the SLA, or ``None``
            if no plan satisfied the constraint.
        rationale: Human-readable explanation of the choice.
        constraint: The :class:`SLAConstraint` applied.
    """

    evaluations: list[PlanEvaluation]
    chosen: PlanEvaluation | None
    rationale: str
    constraint: SLAConstraint

    @property
    def feasible(self) -> list[PlanEvaluation]:
        """Evaluations that satisfied the SLA, sorted cheapest first."""
        return sorted(
            (e for e in self.evaluations if e.meets_sla),
            key=lambda e: e.hourly_cost,
        )

    def comparison_table(self) -> list[dict[str, Any]]:
        """Per-plan cost + performance comparison as a list of dicts."""
        return [e.as_dict() for e in self.evaluations]

    def as_dict(self) -> dict[str, Any]:
        return {
            "chosen": self.chosen.as_dict() if self.chosen else None,
            "rationale": self.rationale,
            "comparison": self.comparison_table(),
        }


# ---------------------------------------------------------------------------
# Provisioner
# ---------------------------------------------------------------------------


class CostAwareProvisioner:
    """Pick the cost-optimal provisioning plan under an SLA constraint.

    Each candidate plan is turned into a what-if scenario whose
    ``hourly_cost`` is derived from the plan's :class:`PriceQuote`, run
    through the :class:`WhatIfEngine`, and scored.  The cheapest plan whose
    projected performance meets the SLA is selected.
    """

    def __init__(
        self,
        twin: DigitalTwin | None = None,
        *,
        duration_s: float = 3600.0,
        load_multiplier: float | None = None,
        seed: int | None = 1234,
    ) -> None:
        """Initialise the provisioner.

        Args:
            twin: Base :class:`DigitalTwin`.  A fresh empty twin is used if
                ``None``.  Each plan is evaluated by replacing the topology,
                so the base twin's nodes do not leak into scenarios.
            duration_s: Simulation duration per scenario.
            load_multiplier: Optional load scaling applied to every scenario.
            seed: Random seed for reproducible simulations.  A fixed seed
                keeps plan comparisons apples-to-apples.
        """
        self._twin = twin if twin is not None else DigitalTwin()
        self._engine = WhatIfEngine(self._twin)
        self._duration_s = duration_s
        self._load_multiplier = load_multiplier
        self._seed = seed

    def evaluate_plan(
        self, plan: ProvisioningPlan, constraint: SLAConstraint
    ) -> PlanEvaluation:
        """Evaluate a single plan: feed its PriceQuote into the twin, simulate."""
        hourly_cost = plan.fleet_hourly()

        # Per-node price fed into the twin so the scenario is COST-AWARE.
        params: dict[str, Any] = {
            "replace": True,
            "count": plan.node_count,
            "gpu_type": plan.gpu_type,
            "gpu_count": plan.gpu_count,
            "region": plan.region,
            "hourly_cost": plan.unit_hourly(),
            "duration_s": self._duration_s,
        }
        if self._load_multiplier is not None:
            params["load_multiplier"] = self._load_multiplier

        # The scenario carries cost: hourly_cost flows into WhatIfEngine ->
        # DigitalTwin.add_nodes -> SimClusterNode.hourly_cost.
        result = self._engine.query(params, seed=self._seed)

        meets, violations = constraint.check(result)
        return PlanEvaluation(
            plan=plan,
            hourly_cost=hourly_cost,
            result=result,
            meets_sla=meets,
            violations=violations,
        )

    def optimize(
        self, plans: list[ProvisioningPlan], constraint: SLAConstraint
    ) -> ProvisioningReport:
        """Evaluate all plans and select the cheapest that meets the SLA.

        Args:
            plans: Candidate provisioning plans (each with a ``PriceQuote``).
            constraint: The SLA / performance constraint to satisfy.

        Returns:
            A :class:`ProvisioningReport` with per-plan cost+perf comparison,
            the chosen plan, and a rationale.

        Raises:
            ValueError: If ``plans`` is empty.
        """
        if not plans:
            raise ValueError("No candidate provisioning plans supplied")

        evaluations = [self.evaluate_plan(p, constraint) for p in plans]

        feasible = sorted(
            (e for e in evaluations if e.meets_sla),
            key=lambda e: e.hourly_cost,
        )

        if feasible:
            chosen = feasible[0]
            cheapest_overall = min(evaluations, key=lambda e: e.hourly_cost)
            if chosen is cheapest_overall:
                rationale = (
                    f"Selected '{chosen.plan.plan_id}' — the cheapest plan "
                    f"(${chosen.hourly_cost:.2f}/hr) and it satisfies the SLA "
                    f"(throughput={chosen.result.throughput}, "
                    f"latency_p99={chosen.result.latency_p99}ms)."
                )
            else:
                cheaper_failed = [
                    e for e in evaluations
                    if not e.meets_sla and e.hourly_cost < chosen.hourly_cost
                ]
                skipped = ", ".join(
                    f"'{e.plan.plan_id}' (${e.hourly_cost:.2f}/hr; "
                    f"{'; '.join(e.violations)})"
                    for e in sorted(cheaper_failed, key=lambda e: e.hourly_cost)
                )
                rationale = (
                    f"Selected '{chosen.plan.plan_id}' at "
                    f"${chosen.hourly_cost:.2f}/hr — the cheapest plan that "
                    f"meets the SLA. Skipped cheaper but non-compliant "
                    f"plan(s): {skipped}."
                )
        else:
            chosen = None
            rationale = (
                "No candidate plan satisfies the SLA constraint "
                f"({_describe_constraint(constraint)}). Returning full "
                "comparison for review."
            )

        return ProvisioningReport(
            evaluations=evaluations,
            chosen=chosen,
            rationale=rationale,
            constraint=constraint,
        )


def _describe_constraint(c: SLAConstraint) -> str:
    parts: list[str] = []
    if c.min_throughput is not None:
        parts.append(f"throughput>={c.min_throughput}")
    if c.max_latency_p99 is not None:
        parts.append(f"latency_p99<={c.max_latency_p99}ms")
    if c.max_failures is not None:
        parts.append(f"failures<={c.max_failures}")
    return ", ".join(parts) if parts else "no constraints"
