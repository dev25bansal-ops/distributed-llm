"""Regression tests for B1: advanced_scheduling policies are out-of-contract
with their batch_scheduler consumer.

Enabling WAN/energy/cost/heterogeneous scheduling (or calling
``scheduler.get_stats()``) used to raise TypeError/AttributeError because the
policy classes exposed different APIs than ``batch_scheduler`` constructs and
uses.  These tests exercise the exact calls batch_scheduler makes.
"""

import pytest

from distllm.core.advanced_scheduling import (
    CostAwarePriorityAdjuster,
    EnergyAwareScheduler,
    HeterogeneousBudgetComputer,
    NodeCapabilityInfo,
    WANConfig,
    WANSchedulingPolicy,
)


def _node(node_id="n1", tflops=82.0, bandwidth=1008.0) -> NodeCapabilityInfo:
    return NodeCapabilityInfo(
        node_id=node_id, gpu_tflops=tflops, bandwidth_gbps=bandwidth,
    )


def test_heterogeneous_set_nodes_and_stats():
    """batch_scheduler calls set_nodes() then stats()."""
    comp = HeterogeneousBudgetComputer()
    comp.set_nodes({"n1": _node()})
    stats = comp.stats()
    assert isinstance(stats, dict)
    assert stats["node_count"] == 1


def test_cost_adjuster_batch_scheduler_kwargs():
    """batch_scheduler.set_cost_awareness constructs with these kwargs."""
    adjuster = CostAwarePriorityAdjuster(
        cost_per_hour_by_node={"n1": 0.60, "n2": 0.10},
        max_cost_per_request=0.0,
        prefer_cheap_for_low_priority=True,
    )
    new_pri, cost = adjuster.adjust_priority(2, 1000, node_id="n1")
    assert isinstance(new_pri, int)
    assert isinstance(cost, float)
    stats = adjuster.stats()
    assert stats["priced_nodes"] == 2


def test_wan_policy_batch_scheduler_contract():
    """batch_scheduler.set_wan_mode passes these WANConfig fields."""
    policy = WANSchedulingPolicy(WANConfig(
        enabled=True,
        chunk_multiplier=2.0,
        batch_multiplier=1.5,
        rtt_threshold_ms=10.0,
        prefetch_kv=True,
    ))
    # detect_wan_mode(nodes) must exist and not crash.
    assert policy.detect_wan_mode({"n1": _node(bandwidth=1008.0)}) is True
    stats = policy.stats()
    assert "wan_active" in stats


def test_wan_detect_low_bandwidth_node():
    """A low-bandwidth node is detected as WAN and does not crash."""
    policy = WANSchedulingPolicy(WANConfig(enabled=False, rtt_threshold_ms=10.0))
    assert policy.detect_wan_mode({"n1": _node(bandwidth=1.0)}) is True
    assert policy.stats()["wan_active"] is True


def test_energy_scheduler_batch_scheduler_kwargs():
    """batch_scheduler.set_energy_monitor constructs with these kwargs."""
    sched = EnergyAwareScheduler(max_power_watts=500.0, energy_cost_per_kwh=0.10)
    stats = sched.stats()
    assert stats["max_power_watts"] == 500.0
    assert stats["energy_cost_per_kwh"] == 0.10


def test_batch_scheduler_toggle_all_features_and_stats():
    """End-to-end: toggling every advanced feature + get_stats() must not crash."""
    from distllm.core.batch_scheduler import BatchScheduler

    sched = BatchScheduler(max_batch_size=4, max_tokens_per_batch=1024)
    sched.set_wan_mode(
        enabled=True, chunk_multiplier=2.0, batch_multiplier=1.5,
        rtt_threshold_ms=10.0, prefetch_kv=True,
    )
    sched.set_energy_monitor(max_power_watts=500.0, energy_cost_per_kwh=0.10)
    sched.set_cost_awareness(
        node_costs={"n1": 0.60, "n2": 0.10},
        max_cost_per_request=0.0,
        prefer_cheap_for_low_priority=True,
    )
    # set_node_capabilities calls set_nodes() + detect_wan_mode(nodes).
    sched.set_node_capabilities({
        "n1": _node("n1", tflops=82.0),
        "n2": _node("n2", tflops=20.0, bandwidth=1.0),
    })

    stats = sched.stats()
    for key in ("heterogeneous", "cost_aware", "wan", "energy"):
        assert key in stats, f"get_stats missing {key!r}"
        assert isinstance(stats[key], dict)
