"""Tests for DisaggregatedBudget and DisaggregatedBatchScheduler."""

from __future__ import annotations

from types import SimpleNamespace

from tests._import_helper import bootstrap_fake_packages, load_module

bootstrap_fake_packages()

_disagg = load_module("distllm/core/advanced_scheduling/disaggregated.py")
DisaggregatedBudget = _disagg.DisaggregatedBudget
DisaggregatedBatchScheduler = _disagg.DisaggregatedBatchScheduler
RequestPhase = _disagg.RequestPhase
NodePoolState = _disagg.NodePoolState


class TestDisaggregatedBudget:
    """Test suite for DisaggregatedBudget."""

    def test_default_construction(self) -> None:
        budget = DisaggregatedBudget()
        assert budget.prefill_max_tokens == 4096
        assert budget.decode_max_tokens == 512
        assert budget.prefill_batch_size == 8
        assert budget.decode_batch_size == 32
        assert budget.prefill_fraction == 0.7
        assert budget.kv_transfer_bandwidth_gbps == 10.0

    def test_custom_values(self) -> None:
        budget = DisaggregatedBudget(
            prefill_max_tokens=8192,
            decode_max_tokens=1024,
            prefill_batch_size=16,
            decode_batch_size=64,
            prefill_fraction=0.5,
            kv_transfer_bandwidth_gbps=25.0,
        )
        assert budget.prefill_max_tokens == 8192
        assert budget.prefill_fraction == 0.5


class TestNodePoolState:
    """Test suite for NodePoolState."""

    def test_default_construction(self) -> None:
        pool = NodePoolState()
        assert pool.node_ids == []
        assert pool.active_requests == {}
        assert pool.total_processed == 0
        assert pool.avg_latency_ms == 0.0

    def test_available_nodes_empty(self) -> None:
        pool = NodePoolState()
        assert pool.available_nodes == []

    def test_available_nodes_below_capacity(self) -> None:
        pool = NodePoolState(node_ids=["a", "b", "c"], active_requests={"a": 2, "b": 8, "c": 3})
        available = pool.available_nodes
        assert "a" in available
        assert "b" not in available  # at capacity (8)
        assert "c" in available

    def test_least_loaded_empty(self) -> None:
        pool = NodePoolState()
        assert pool.least_loaded is None

    def test_least_loaded_returns_min(self) -> None:
        pool = NodePoolState(node_ids=["a", "b"], active_requests={"a": 5, "b": 1})
        assert pool.least_loaded == "b"

    def test_least_loaded_ties_first_by_sort(self) -> None:
        pool = NodePoolState(node_ids=["x", "y"], active_requests={"x": 3, "y": 3})
        # Both equal, min picks alphabetically first
        assert pool.least_loaded == "x"


class TestDisaggregatedBatchScheduler:
    """Test suite for DisaggregatedBatchScheduler."""

    def test_default_construction(self) -> None:
        scheduler = DisaggregatedBatchScheduler()
        assert scheduler._prefill_fraction == 0.7
        assert scheduler._min_prefill_tokens == 64
        assert scheduler._enable_kv_transfer is True
        assert scheduler._kv_transfer_bandwidth_gbps == 10.0
        assert scheduler._stats == {
            "prefill_scheduled": 0,
            "decode_scheduled": 0,
            "kv_transfers": 0,
            "fallback_to_local": 0,
        }

    def test_custom_construction(self) -> None:
        scheduler = DisaggregatedBatchScheduler(
            prefill_fraction=0.5,
            min_prefill_tokens=128,
            enable_kv_transfer=False,
            kv_transfer_bandwidth_gbps=25.0,
        )
        assert scheduler._prefill_fraction == 0.5
        assert scheduler._min_prefill_tokens == 128
        assert scheduler._enable_kv_transfer is False
        assert scheduler._kv_transfer_bandwidth_gbps == 25.0

    def test_set_prefill_nodes(self) -> None:
        scheduler = DisaggregatedBatchScheduler()
        scheduler.set_prefill_nodes(["gpu-0", "gpu-1"])
        assert "gpu-0" in scheduler._prefill_pool.node_ids
        assert "gpu-1" in scheduler._prefill_pool.node_ids
        assert scheduler._prefill_pool.active_requests == {"gpu-0": 0, "gpu-1": 0}

    def test_set_decode_nodes(self) -> None:
        scheduler = DisaggregatedBatchScheduler()
        scheduler.set_decode_nodes(["gpu-2", "gpu-3"])
        assert "gpu-2" in scheduler._decode_pool.node_ids
        assert scheduler._decode_pool.active_requests == {"gpu-2": 0, "gpu-3": 0}

    def test_classify_request_no_tokens_is_prefill(self) -> None:
        scheduler = DisaggregatedBatchScheduler()
        seq = SimpleNamespace(generated_tokens=[])
        assert scheduler.classify_request(seq) == RequestPhase.PREFILL

    def test_classify_request_with_tokens_is_decode(self) -> None:
        scheduler = DisaggregatedBatchScheduler()
        seq = SimpleNamespace(generated_tokens=[1, 2, 3])
        assert scheduler.classify_request(seq) == RequestPhase.DECODE

    def test_route_prefill_selects_least_loaded(self) -> None:
        scheduler = DisaggregatedBatchScheduler()
        scheduler.set_prefill_nodes(["gpu-0", "gpu-1"])

        # Make gpu-0 more loaded
        scheduler._prefill_pool.active_requests["gpu-0"] = 3

        # First route should pick gpu-1 (least loaded)
        node = scheduler.route(RequestPhase.PREFILL)
        assert node == "gpu-1"
        assert scheduler._prefill_pool.active_requests["gpu-1"] == 1

    def test_route_decode_selects_least_loaded(self) -> None:
        scheduler = DisaggregatedBatchScheduler()
        scheduler.set_decode_nodes(["gpu-2", "gpu-3"])

        node = scheduler.route(RequestPhase.DECODE)
        assert node in ("gpu-2", "gpu-3")

    def test_route_returns_none_when_no_nodes(self) -> None:
        scheduler = DisaggregatedBatchScheduler()
        assert scheduler.route(RequestPhase.PREFILL) is None
        assert scheduler.route(RequestPhase.DECODE) is None

    def test_release_decrements_active_count(self) -> None:
        scheduler = DisaggregatedBatchScheduler()
        scheduler.set_prefill_nodes(["gpu-0"])
        scheduler.route(RequestPhase.PREFILL)

        assert scheduler._prefill_pool.active_requests["gpu-0"] == 1
        scheduler.release("gpu-0", RequestPhase.PREFILL)
        assert scheduler._prefill_pool.active_requests["gpu-0"] == 0
        assert scheduler._prefill_pool.total_processed == 1

    def test_release_decode(self) -> None:
        scheduler = DisaggregatedBatchScheduler()
        scheduler.set_decode_nodes(["gpu-2"])
        scheduler.route(RequestPhase.DECODE)
        assert scheduler._decode_pool.active_requests["gpu-2"] == 1

        scheduler.release("gpu-2", RequestPhase.DECODE)
        assert scheduler._decode_pool.active_requests["gpu-2"] == 0
        assert scheduler._decode_pool.total_processed == 1

    def test_release_clamps_to_zero(self) -> None:
        scheduler = DisaggregatedBatchScheduler()
        scheduler.set_prefill_nodes(["gpu-0"])
        scheduler.release("gpu-0", RequestPhase.PREFILL)
        assert scheduler._prefill_pool.active_requests["gpu-0"] == 0

    def test_compute_budget_from_base_budget(self) -> None:
        scheduler = DisaggregatedBatchScheduler(prefill_fraction=0.7)
        base_budget = SimpleNamespace(max_total_tokens=10_000, max_batch_size=16)

        budget = scheduler.compute_budget(base_budget)
        assert budget.prefill_max_tokens == 7000
        assert budget.decode_max_tokens == 3000
        assert budget.prefill_batch_size == 16
        # decode_batch_size = max_batch_size * 4 = 64
        assert budget.decode_batch_size == 64

    def test_compute_budget_defaults_when_no_attrs(self) -> None:
        scheduler = DisaggregatedBatchScheduler()
        budget = scheduler.compute_budget(SimpleNamespace())
        assert budget.prefill_max_tokens == int(8192 * 0.7)
        assert budget.decode_max_tokens == int(8192 * 0.3)

    def test_kv_transfer_bandwidth_property(self) -> None:
        scheduler = DisaggregatedBatchScheduler(kv_transfer_bandwidth_gbps=25.0)
        assert scheduler.kv_transfer_bandwidth_gbps == 25.0

    def test_kv_transfer_bandwidth_setter(self) -> None:
        scheduler = DisaggregatedBatchScheduler()
        scheduler.kv_transfer_bandwidth_gbps = 40.0
        assert scheduler._kv_transfer_bandwidth_gbps == 40.0
        assert scheduler.kv_transfer_bandwidth_gbps == 40.0

    def test_get_transfer_estimate(self) -> None:
        scheduler = DisaggregatedBatchScheduler(kv_transfer_bandwidth_gbps=10.0)
        # 10 Gbps = 1.25 GB/s = 1250 MB/s
        # 1 MB = 1_000_000 bytes
        # time = (bytes) / (10 * 1e9 / 8) * 1000
        estimate = scheduler.get_transfer_estimate(10_000_000)
        expected_ms = (10_000_000 / (10.0 * 1e9 / 8)) * 1000
        assert estimate == expected_ms

    def test_allocate_decode_blocks(self) -> None:
        scheduler = DisaggregatedBatchScheduler()
        result = scheduler.allocate_decode_blocks("req-1", 1024, "gpu-2")
        assert result is True

    def test_rebalance_pools_decode_pressure(self) -> None:
        scheduler = DisaggregatedBatchScheduler()
        scheduler.set_prefill_nodes(["p0", "p1", "p2"])
        scheduler.set_decode_nodes(["d0"])

        # decode demand > decode_capacity * 1.3
        result = scheduler.rebalance_pools(
            prefill_demand=5,
            decode_demand=100,
            prefill_capacity=30,
            decode_capacity=10,
        )
        assert len(result["to_decode"]) == 1
        # Verify one node moved from prefill to decode
        assert len(scheduler._decode_pool.node_ids) == 2

    def test_rebalance_pools_prefill_pressure(self) -> None:
        scheduler = DisaggregatedBatchScheduler()
        scheduler.set_prefill_nodes(["p0"])
        scheduler.set_decode_nodes(["d0", "d1", "d2"])

        result = scheduler.rebalance_pools(
            prefill_demand=100,
            decode_demand=5,
            prefill_capacity=10,
            decode_capacity=30,
        )
        assert len(result["to_prefill"]) == 1
        assert len(scheduler._prefill_pool.node_ids) == 2

    def test_rebalance_pools_no_action_when_balanced(self) -> None:
        scheduler = DisaggregatedBatchScheduler()
        scheduler.set_prefill_nodes(["p0", "p1"])
        scheduler.set_decode_nodes(["d0", "d1"])

        result = scheduler.rebalance_pools(
            prefill_demand=10,
            decode_demand=10,
            prefill_capacity=20,
            decode_capacity=20,
        )
        assert result == {"to_decode": [], "to_prefill": []}

    def test_stats(self) -> None:
        scheduler = DisaggregatedBatchScheduler()
        scheduler.set_prefill_nodes(["gpu-0"])
        scheduler.set_decode_nodes(["gpu-1"])
        scheduler.route(RequestPhase.PREFILL)

        stats = scheduler.stats()
        assert stats["prefill_scheduled"] == 1
        assert stats["decode_scheduled"] == 0
        assert stats["prefill_nodes"] == 1
        assert stats["decode_nodes"] == 1
        assert stats["prefill_active"] == 1
        assert stats["decode_active"] == 0
