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


# ===========================================================================
# C4 CONTRACT TESTS
#
# budget_computer.get_iteration_budget() calls each policy with an exact
# signature; any drift on either side used to raise AttributeError/TypeError
# inside schedule() and crash the serving loop on first use of a policy.
# These tests lock BOTH sides: introspection asserts the exact parameter
# names the consumer passes, behavioral asserts the semantics.
# ===========================================================================

import inspect

from distllm.core.scheduler.budget import IterationBudget
from distllm.core.scheduler.budget_computer import BudgetComputer


def _params(func) -> set[str]:
    return {
        name for name, p in inspect.signature(func).parameters.items()
        if name != "self" and p.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    }


class TestBudgetComputerCallContract:
    """Each policy must expose EXACTLY what budget_computer passes."""

    def test_heterogeneous_compute_budget_signature(self):
        """budget_computer.py: het.compute_budget(base_prefill_tokens=, ...)."""
        params = _params(HeterogeneousBudgetComputer.compute_budget)
        assert params == {
            "base_prefill_tokens", "base_decode_tokens",
            "base_batch_size", "base_total_tokens",
        }

    def test_wan_is_wan_active_and_adjust_signature(self):
        """budget_computer.py reads wan_policy.is_wan_active then calls
        adjust_budget_for_wan(base_prefill_tokens=, base_batch_size=,
        base_total_tokens=)."""
        assert isinstance(WANSchedulingPolicy().is_wan_active, bool)
        assert "is_wan_active" in dir(WANSchedulingPolicy)
        params = _params(WANSchedulingPolicy.adjust_budget_for_wan)
        assert params == {"base_prefill_tokens", "base_batch_size", "base_total_tokens"}

    def test_energy_adjust_for_energy_signature(self):
        """budget_computer.py: energy.adjust_for_energy(base_batch_size=,
        base_prefill_tokens=)."""
        params = _params(EnergyAwareScheduler.adjust_for_energy)
        assert params == {"base_batch_size", "base_prefill_tokens"}

    def test_cost_adjuster_priority_signature(self):
        """batch_scheduler._promote_pending calls adjust_priority(
        base_priority=, est_tokens=, node_id=...)."""
        from distllm.core.advanced_scheduling import CostAwarePriorityAdjuster
        params = _params(CostAwarePriorityAdjuster.adjust_priority)
        assert params == {"base_priority", "est_tokens", "node_id"}


class TestBudgetComputerPolicyChainBehavior:
    """Real policies through get_iteration_budget: no exception, right math."""

    def _computer(self) -> BudgetComputer:
        return BudgetComputer(
            kv_cache_mgr=None, pressure_tracker=None, adapt_prefill_budget=False,
        )

    def _base(self) -> IterationBudget:
        return IterationBudget(
            max_prefill_tokens=4096, max_decode_tokens=512,
            max_batch_size=32, max_total_tokens=32768,
            enable_chunked_prefill=True,
        )

    def test_no_policies_passthrough(self):
        result = self._computer().get_iteration_budget(
            base_budget=self._base(), enable_chunked_prefill=True,
            max_tokens_per_batch=32768, max_batch_size=32,
        )
        assert result.max_prefill_tokens == 4096
        assert result.max_decode_tokens == 512
        assert result.max_batch_size == 32

    def test_heterogeneous_scales_prefill(self):
        nodes = {
            "fast": NodeCapabilityInfo(node_id="fast", gpu_tflops=400.0),
            "slow": NodeCapabilityInfo(node_id="slow", gpu_tflops=100.0),
        }
        result = self._computer().get_iteration_budget(
            base_budget=self._base(), enable_chunked_prefill=True,
            max_tokens_per_batch=32768, max_batch_size=32,
            het_budget=HeterogeneousBudgetComputer(nodes),
        )
        assert result.max_prefill_tokens == int(4096 * 0.25)

    def test_wan_multipliers_applied(self):
        policy = WANSchedulingPolicy(WANConfig(enabled=True))
        base = self._base()
        result = self._computer().get_iteration_budget(
            base_budget=base, enable_chunked_prefill=True,
            max_tokens_per_batch=32768, max_batch_size=32,
            wan_policy=policy,
        )
        assert result.max_prefill_tokens == 8192   # 4096 * chunk_multiplier
        assert result.max_batch_size == 48          # 32 * batch_multiplier
        assert result.max_total_tokens == 98304     # 32768 * both multipliers

    def test_energy_passthrough_without_telemetry(self):
        result = self._computer().get_iteration_budget(
            base_budget=self._base(), enable_chunked_prefill=True,
            max_tokens_per_batch=32768, max_batch_size=32,
            energy_scheduler=EnergyAwareScheduler(max_power_watts=500.0),
        )
        assert result.max_batch_size == 32
        assert result.max_prefill_tokens == 4096

    def test_energy_throttles_over_power_budget(self):
        energy = EnergyAwareScheduler(max_power_watts=300.0)
        energy.update_power_draw("n1", 600.0)
        result = self._computer().get_iteration_budget(
            base_budget=self._base(), enable_chunked_prefill=True,
            max_tokens_per_batch=32768, max_batch_size=32,
            energy_scheduler=energy,
        )
        assert result.max_batch_size < 32
        assert result.max_prefill_tokens < 4096

    def test_all_four_policies_together(self):
        """The exact C4 repro: every policy enabled at once."""
        het = HeterogeneousBudgetComputer({
            "n1": NodeCapabilityInfo(node_id="n1", gpu_tflops=82.0),
            "n2": NodeCapabilityInfo(node_id="n2", gpu_tflops=20.0),
        })
        result = self._computer().get_iteration_budget(
            base_budget=self._base(), enable_chunked_prefill=True,
            max_tokens_per_batch=32768, max_batch_size=32,
            het_budget=het,
            wan_policy=WANSchedulingPolicy(WANConfig(enabled=True)),
            energy_scheduler=EnergyAwareScheduler(max_power_watts=500.0),
        )
        # het scales prefill by 20/82, WAN then doubles it.
        assert result.max_prefill_tokens == int(int(4096 * (20 / 82)) * 2)
        assert result.max_batch_size == 48

    def test_persistent_base_budget_not_mutated(self):
        """Policies mutate their input; the caller's budget must survive."""
        comp = self._computer()
        base = self._base()
        het = HeterogeneousBudgetComputer({
            "fast": NodeCapabilityInfo(node_id="fast", gpu_tflops=400.0),
            "slow": NodeCapabilityInfo(node_id="slow", gpu_tflops=100.0),
        })
        for _ in range(5):
            comp.get_iteration_budget(
                base_budget=base, enable_chunked_prefill=True,
                max_tokens_per_batch=32768, max_batch_size=32,
                het_budget=het,
                wan_policy=WANSchedulingPolicy(WANConfig(enabled=True)),
                energy_scheduler=EnergyAwareScheduler(max_power_watts=500.0),
            )
        # Repeated invocations never compound onto the caller's object.
        assert base.max_prefill_tokens == 4096
        assert base.max_batch_size == 32


class TestBatchSchedulerRuntimeContract:
    """Runtime update paths batch_scheduler exercises on the policies."""

    def test_update_node_power_reaches_energy_scheduler(self):
        sched = EnergyAwareScheduler(max_power_watts=1000.0)
        sched.update_power_draw("gpu-0", 450.0)
        assert sched.get_total_power_draw() == pytest.approx(450.0)
        stats = sched.stats()
        assert stats["total_power_watts"] == pytest.approx(450.0)
        assert "power_utilization_pct" in stats

    def test_record_energy_usage_accumulates(self):
        sched = EnergyAwareScheduler(max_power_watts=1000.0, energy_cost_per_kwh=0.10)
        sched.update_power_draw("gpu-0", 500.0)
        sched.record_energy_usage(duration_seconds=3600.0)
        assert sched.stats()["total_energy_wh"] == pytest.approx(500.0)
        assert sched.stats()["total_energy_cost_usd"] > 0

    def test_schedule_with_every_policy_enabled(self):
        """The C4 repro end-to-end: toggle all four, then schedule()."""
        from distllm.core.batch_scheduler import BatchScheduler, Sequence

        scheduler = BatchScheduler(max_batch_size=4, max_tokens_per_batch=1024)
        scheduler.set_wan_mode(enabled=True)
        scheduler.set_energy_monitor(max_power_watts=500.0)
        scheduler.set_cost_awareness(node_costs={"n1": 0.60})
        scheduler.set_node_capabilities({
            "n1": _node("n1", tflops=82.0),
            "n2": _node("n2", tflops=20.0, bandwidth=1.0),
        })

        seq = Sequence(request_id="r1", prompt_tokens=list(range(50)), max_new_tokens=2)
        scheduler.add(seq)
        batch = scheduler.schedule()
        assert batch is not None
        assert seq.request_id in scheduler.active

        # The cost-aware priority hook runs inside promote with its pinned kwargs.
        effective_pri, cost = scheduler._cost_adjuster.adjust_priority(
            base_priority=2, est_tokens=seq.total_len, node_id="n1")
        assert isinstance(effective_pri, int)
        assert cost >= 0.0

    def test_update_node_latency_paths(self):
        """update_node_latency mutates NodeCapabilityInfo and re-runs detection."""
        from distllm.core.batch_scheduler import BatchScheduler

        scheduler = BatchScheduler(max_batch_size=4, max_tokens_per_batch=1024)
        scheduler.set_wan_mode(enabled=False)
        scheduler.set_node_capabilities({"n1": _node("n1")})
        scheduler.update_node_latency("n1", latency_ms=250.0)
        assert scheduler._wan_policy.is_wan_active is True
        assert scheduler._het_budget is not None
