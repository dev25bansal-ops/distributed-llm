"""Regression test N5 — cost-aware provisioning x digital twin.

Verifies that real cloud PriceQuotes feed into the WhatIfEngine and drive a
cost-optimal provisioning decision under an SLA constraint.

All PriceQuotes are injected mocks — NO live cloud billing API is called.
The PriceQuote -> WhatIfEngine wiring and the optimisation logic are real.
"""

from __future__ import annotations

import pytest

from distllm.cloud.common import PriceQuote
from distllm.dist.simulation.digital_twin import DigitalTwin, WhatIfEngine
from distllm.dist.simulation.cost_aware_provisioning import (
    CostAwareProvisioner,
    PlanEvaluation,
    ProvisioningPlan,
    ProvisioningReport,
    SLAConstraint,
)


def _quote(provider: str, instance_type: str, on_demand: float) -> PriceQuote:
    """Build a mock PriceQuote (no live cloud call)."""
    return PriceQuote(
        provider=provider,
        instance_type=instance_type,
        region="us-east-1",
        on_demand_hourly=on_demand,
    )


def _candidate_plans() -> list[ProvisioningPlan]:
    """Three candidate plans spanning cheap/slow -> expensive/fast.

    - cheap-t4:  $12/hr fleet, low GPU throughput -> high queuing latency.
    - mid-a100:  $40/hr fleet, mid latency.
    - big-h100:  $192/hr fleet, lowest latency.
    """
    return [
        ProvisioningPlan(
            plan_id="cheap-t4",
            provider="aws",
            instance_type="g4dn.12xlarge",
            gpu_type="T4",
            node_count=2,
            gpu_count=4,
            region="us-east-1",
            quote=_quote("aws", "g4dn.12xlarge", 6.0),
        ),
        ProvisioningPlan(
            plan_id="mid-a100",
            provider="gcp",
            instance_type="a2-highgpu-8g",
            gpu_type="A100",
            node_count=2,
            gpu_count=8,
            region="us-central1",
            quote=_quote("gcp", "a2-highgpu-8g", 20.0),
        ),
        ProvisioningPlan(
            plan_id="big-h100",
            provider="azure",
            instance_type="ND-H100-v5",
            gpu_type="H100",
            node_count=4,
            gpu_count=8,
            region="eastus",
            quote=_quote("azure", "ND-H100-v5", 48.0),
        ),
    ]


def _provisioner() -> CostAwareProvisioner:
    # Fixed seed + load keep plan comparisons apples-to-apples & reproducible.
    return CostAwareProvisioner(
        DigitalTwin(), duration_s=3600.0, load_multiplier=6.0, seed=42
    )


# ---------------------------------------------------------------------------
# (1) Cheapest plan that meets the SLA is chosen
# ---------------------------------------------------------------------------


def test_picks_cheapest_meeting_sla():
    plans = _candidate_plans()
    prov = _provisioner()

    # A loose SLA that every plan satisfies -> the absolute cheapest wins.
    loose = SLAConstraint(min_throughput=1.0)
    report = prov.optimize(plans, loose)

    assert isinstance(report, ProvisioningReport)
    assert report.chosen is not None
    assert report.chosen.plan.plan_id == "cheap-t4"
    # cheapest fleet cost = 6.0/node * 2 nodes
    assert report.chosen.hourly_cost == pytest.approx(12.0)
    assert report.chosen.meets_sla
    # cheapest overall is also the chosen one -> rationale says so
    assert "cheapest" in report.rationale.lower()


# ---------------------------------------------------------------------------
# (2) When cheapest violates SLA, next-cheapest compliant plan is chosen
# ---------------------------------------------------------------------------


def test_skips_cheapest_when_it_violates_sla():
    plans = _candidate_plans()
    prov = _provisioner()

    # Tight latency SLA: the cheap T4 plan queues badly and blows the p99
    # budget, while the A100 plan stays under it. -> pick mid-a100.
    tight = SLAConstraint(min_throughput=1.0, max_latency_p99=100_000.0)
    report = prov.optimize(plans, tight)

    # Sanity: cheapest plan really does violate the SLA.
    by_id = {e.plan.plan_id: e for e in report.evaluations}
    assert not by_id["cheap-t4"].meets_sla
    assert by_id["cheap-t4"].violations  # non-empty explanation

    assert report.chosen is not None
    assert report.chosen.plan.plan_id == "mid-a100"
    assert report.chosen.hourly_cost == pytest.approx(40.0)
    assert report.chosen.meets_sla
    # It is NOT the cheapest overall -> rationale explains the skip.
    assert "skipped" in report.rationale.lower()
    assert "cheap-t4" in report.rationale


# ---------------------------------------------------------------------------
# (3) PriceQuote flows into the WhatIfEngine (scenario carries cost)
# ---------------------------------------------------------------------------


def test_pricequote_flows_into_whatif_scenario():
    """The per-node PriceQuote price must reach the twin's SimClusterNode."""
    twin = DigitalTwin()
    engine = WhatIfEngine(twin)

    quoted_hourly = 33.33
    result = engine.query(
        {
            "replace": True,
            "count": 3,
            "gpu_type": "A100",
            "gpu_count": 8,
            "hourly_cost": quoted_hourly,
            "duration_s": 3600.0,
        },
        seed=7,
    )
    # total_cost is derived from node hourly_cost -> non-zero means the quote
    # propagated (default table would give a different, non-quote value; the
    # key assertion is that our injected value is what drives billing).
    assert result.total_cost > 0.0

    # Directly assert the quote landed on the simulated nodes.
    scenario_twin = engine._make_twin(
        {"replace": True, "count": 3, "gpu_type": "A100", "hourly_cost": quoted_hourly}
    )
    node_costs = {n.hourly_cost for n in scenario_twin._nodes.values()}
    assert node_costs == {quoted_hourly}

    # And end-to-end through the provisioner: cost is quote-driven, not the
    # twin's built-in default table.
    prov = CostAwareProvisioner(DigitalTwin(), load_multiplier=6.0, seed=42)
    plan = ProvisioningPlan(
        plan_id="q-check",
        provider="aws",
        instance_type="p4d.24xlarge",
        gpu_type="A100",
        node_count=2,
        gpu_count=8,
        quote=_quote("aws", "p4d.24xlarge", 25.0),
    )
    ev = prov.evaluate_plan(plan, SLAConstraint())
    assert ev.hourly_cost == pytest.approx(50.0)  # 25.0 * 2 nodes, from quote


# ---------------------------------------------------------------------------
# (4) Output includes a per-plan cost + performance comparison
# ---------------------------------------------------------------------------


def test_report_includes_cost_and_perf_comparison():
    plans = _candidate_plans()
    prov = _provisioner()
    report = prov.optimize(plans, SLAConstraint(min_throughput=1.0))

    table = report.comparison_table()
    assert len(table) == len(plans)

    ids = {row["plan_id"] for row in table}
    assert ids == {"cheap-t4", "mid-a100", "big-h100"}

    for row in table:
        # cost fields
        assert "hourly_cost" in row and row["hourly_cost"] > 0
        assert "cost_per_throughput" in row
        # perf fields (from the twin)
        assert "throughput" in row
        assert "latency_p99" in row
        assert "failures" in row
        # SLA verdict
        assert "meets_sla" in row

    # Costs must match quote-derived fleet pricing.
    cost_by_id = {row["plan_id"]: row["hourly_cost"] for row in table}
    assert cost_by_id["cheap-t4"] == pytest.approx(12.0)
    assert cost_by_id["mid-a100"] == pytest.approx(40.0)
    assert cost_by_id["big-h100"] == pytest.approx(192.0)

    # as_dict() surfaces both the chosen plan and the full comparison.
    d = report.as_dict()
    assert d["chosen"] is not None
    assert len(d["comparison"]) == len(plans)
    assert d["rationale"]


# ---------------------------------------------------------------------------
# Edge / robustness
# ---------------------------------------------------------------------------


def test_no_feasible_plan_returns_none_with_rationale():
    plans = _candidate_plans()
    prov = _provisioner()
    # Impossible SLA -> no plan qualifies.
    impossible = SLAConstraint(min_throughput=1e9)
    report = prov.optimize(plans, impossible)
    assert report.chosen is None
    assert "no candidate plan" in report.rationale.lower()
    # Comparison is still emitted for review.
    assert len(report.comparison_table()) == len(plans)


def test_empty_plans_raises():
    with pytest.raises(ValueError):
        _provisioner().optimize([], SLAConstraint())


def test_spot_pricing_mode_used_when_available():
    q = PriceQuote(
        "aws", "p4d.24xlarge", "us-east-1",
        on_demand_hourly=32.0, spot_hourly=9.6,
    )
    plan = ProvisioningPlan(
        plan_id="spot",
        provider="aws",
        instance_type="p4d.24xlarge",
        gpu_type="A100",
        node_count=2,
        quote=q,
        pricing_mode="spot",
    )
    assert plan.unit_hourly() == pytest.approx(9.6)
    assert plan.fleet_hourly() == pytest.approx(19.2)
