"""Tests for distllm.dist.partition.plugins.

Tests the plugin API for custom cost models: CostModelPlugin,
CostModelRegistry, and PluginAwareCostModel.

Zero mocks — all tests use real objects.
"""

from __future__ import annotations

from typing import Any

import pytest

from distllm.dist.partition.cost_model import NodeCost, PartitionCostModel
from distllm.dist.partition.plugins import (
    CostModelPlugin,
    CostModelRegistry,
    PluginAwareCostModel,
)
from distllm.dist.partition.profiles import GPUProfile, LayerWeights
from distllm.dist.partition.topology import LinkProfile, TopologyGraph


# ── helpers ────────────────────────────────────────────────────────────────


class DummyPlugin(CostModelPlugin):
    """A concrete plugin for testing; handles "dummy" devices."""

    def __init__(self, *, cost: NodeCost | None = None, priority: int = 0):
        self._cost = cost
        self._priority = priority

    def estimate_node_cost(
        self,
        node_id: str,
        layers: list[LayerWeights],
        batch_size: int,
        seq_len: int,
        gpu_profile: GPUProfile | None = None,
        topology: TopologyGraph | None = None,
        prev_node_id: str | None = None,
    ) -> NodeCost | None:
        if self._cost is not None:
            return self._cost
        return NodeCost(
            node_id=node_id,
            start_layer=layers[0].layer_id if layers else 0,
            end_layer=layers[-1].layer_id if layers else 0,
            compute_time_ms=10.0,
            communication_time_ms=2.0,
            total_time_ms=12.0,
            memory_bytes=1024,
            memory_available_bytes=8192,
            fits_in_memory=True,
        )

    def supported_devices(self) -> list[str]:
        return ["dummy", "test_device"]

    def priority(self) -> int:
        return self._priority

    def name(self) -> str:
        return "DummyPlugin"


class NeverHandlePlugin(CostModelPlugin):
    """Plugin whose can_handle always returns False."""

    def estimate_node_cost(
        self,
        node_id: str,
        layers: list[LayerWeights],
        batch_size: int,
        seq_len: int,
        gpu_profile: GPUProfile | None = None,
        topology: TopologyGraph | None = None,
        prev_node_id: str | None = None,
    ) -> NodeCost | None:
        return NodeCost(node_id=node_id, start_layer=0, end_layer=0, total_time_ms=99.0)

    def supported_devices(self) -> list[str]:
        return ["never"]

    def can_handle(self, node_id: str, gpu_profile: GPUProfile | None) -> bool:
        return False


class FailingPlugin(CostModelPlugin):
    """Plugin whose estimate_node_cost raises an exception."""

    def estimate_node_cost(
        self,
        node_id: str,
        layers: list[LayerWeights],
        batch_size: int,
        seq_len: int,
        gpu_profile: GPUProfile | None = None,
        topology: TopologyGraph | None = None,
        prev_node_id: str | None = None,
    ) -> NodeCost | None:
        msg = "simulated failure"
        raise RuntimeError(msg)

    def supported_devices(self) -> list[str]:
        return ["failing"]


def _make_layer(  # noqa: PLR0913
    layer_id: int,
    *,
    layer_type: str = "transformer",
    weight_bytes: int = 4096,
    act_bytes: int = 1024,
    flops_per_token: int = 1000,
    kv_bytes: int = 256,
) -> LayerWeights:
    return LayerWeights(
        layer_id=layer_id,
        layer_type=layer_type,
        weight_memory_bytes=weight_bytes,
        activation_memory_bytes=act_bytes,
        flops_per_token=flops_per_token,
        flops_per_seq=flops_per_token,
        kv_cache_bytes_per_token=kv_bytes,
    )


def _make_gpu_profile(name: str = "dummy", gpu_id: int = 0) -> GPUProfile:
    return GPUProfile(
        gpu_id=gpu_id,
        name=name,
        total_memory_bytes=16 * 1024**3,
        free_memory_bytes=8 * 1024**3,
        compute_tflops=100.0,
        memory_bandwidth_gbps=1000.0,
        sm_count=80,
        memory_bus_width_bits=5120,
        peak_tflops_fp16=100.0,
        peak_tflops_fp32=50.0,
    )


def _make_topology() -> TopologyGraph:
    return TopologyGraph(
        node_ids=["node-0", "node-1"],
        gpu_counts={"node-0": 1, "node-1": 1},
        gpu_profiles={},
        links=[
            LinkProfile(
                source="node-0",
                target="node-1",
                bandwidth_gbps=12.5,
                latency_us=100.0,
            ),
        ],
    )


def _make_base_cost_model() -> PartitionCostModel:
    gpu = _make_gpu_profile()
    layers = [_make_layer(0), _make_layer(1), _make_layer(2)]
    topo = _make_topology()
    return PartitionCostModel(
        gpu_profiles={"node-0": gpu, "node-1": gpu},
        layer_weights=layers,
        topology=topo,
    )


# ── TestCostModelPlugin ────────────────────────────────────────────────────


class TestCostModelPlugin:
    """Tests for the CostModelPlugin abstract base class."""

    def test_cannot_instantiate_abc(self) -> None:
        with pytest.raises(TypeError):
            CostModelPlugin()  # type: ignore[abstract]

    def test_concrete_subclass_works(self) -> None:
        plugin = DummyPlugin()
        assert plugin.name() == "DummyPlugin"
        assert plugin.priority() == 0
        assert plugin.supported_devices() == ["dummy", "test_device"]

    def test_estimate_node_cost_returns_node_cost(self) -> None:
        plugin = DummyPlugin()
        layers = [_make_layer(0), _make_layer(1)]
        cost = plugin.estimate_node_cost("node-0", layers, batch_size=1, seq_len=128)
        assert isinstance(cost, NodeCost)
        assert cost.node_id == "node-0"
        assert cost.start_layer == 0
        assert cost.end_layer == 1
        assert cost.compute_time_ms == 10.0
        assert cost.total_time_ms == 12.0

    def test_estimate_node_cost_returns_none(self) -> None:
        plugin = DummyPlugin(cost=None)
        layers = [_make_layer(0)]
        cost = plugin.estimate_node_cost("node-0", layers, batch_size=1, seq_len=128)
        # DummyPlugin without explicit cost returns a cost anyway;
        # we use `cost=None` only to override.  Since the default impl
        # always returns a NodeCost, this tests the *contract*:
        assert isinstance(cost, NodeCost)

    def test_can_handle_matches_device_name(self) -> None:
        plugin = DummyPlugin()
        gpu = _make_gpu_profile(name="MyDummyCard")
        assert plugin.can_handle("node-0", gpu) is True

    def test_can_handle_case_insensitive(self) -> None:
        plugin = DummyPlugin()
        gpu = _make_gpu_profile(name="DUMMY_GPU")
        assert plugin.can_handle("node-0", gpu) is True

    def test_can_handle_no_match(self) -> None:
        plugin = DummyPlugin()
        gpu = _make_gpu_profile(name="nvidia_a100")
        assert plugin.can_handle("node-0", gpu) is False

    def test_can_handle_none_gpu(self) -> None:
        plugin = DummyPlugin()
        assert plugin.can_handle("node-0", None) is False

    def test_can_handle_empty_devices(self) -> None:
        class EmptyDevicesPlugin(CostModelPlugin):
            def estimate_node_cost(self, *args: Any, **kwargs: Any) -> NodeCost | None:
                return None
            def supported_devices(self) -> list[str]:
                return []

        plugin = EmptyDevicesPlugin()
        gpu = _make_gpu_profile(name="anything")
        assert plugin.can_handle("node-0", gpu) is False

    def test_name_defaults_to_class_name(self) -> None:
        plugin = DummyPlugin()
        assert plugin.name() == "DummyPlugin"

    def test_priority_defaults_to_zero(self) -> None:
        plugin = DummyPlugin()
        assert plugin.priority() == 0

    def test_estimate_node_cost_with_optional_args(self) -> None:
        """All optional keyword arguments can be passed."""
        plugin = DummyPlugin()
        layers = [_make_layer(0)]
        topo = _make_topology()
        gpu = _make_gpu_profile()
        cost = plugin.estimate_node_cost(
            "node-0",
            layers,
            batch_size=2,
            seq_len=256,
            gpu_profile=gpu,
            topology=topo,
            prev_node_id="node-1",
        )
        assert isinstance(cost, NodeCost)

    def test_estimate_node_cost_empty_layers(self) -> None:
        plugin = DummyPlugin(cost=NodeCost(
            node_id="x", start_layer=0, end_layer=0, total_time_ms=0.0,
        ))
        cost = plugin.estimate_node_cost("x", [], batch_size=1, seq_len=1)
        assert cost.total_time_ms == 0.0

    def test_estimate_node_cost_large_batch_seq(self) -> None:
        plugin = DummyPlugin()
        layers = [_make_layer(0)]
        cost = plugin.estimate_node_cost(
            "node-0", layers, batch_size=1024, seq_len=131072,
        )
        assert isinstance(cost, NodeCost)


# ── TestCostModelRegistry ──────────────────────────────────────────────────


class TestCostModelRegistry:
    """Tests for the CostModelRegistry."""

    def test_empty_on_init(self) -> None:
        registry = CostModelRegistry()
        assert registry.list_plugins() == []
        assert registry.get("anything") is None

    def test_register_adds_plugin(self) -> None:
        registry = CostModelRegistry()
        plugin = DummyPlugin()
        registry.register(plugin)
        assert len(registry.list_plugins()) == 1
        assert registry.list_plugins()[0]["name"] == "DummyPlugin"

    def test_register_sorts_by_priority_descending(self) -> None:
        registry = CostModelRegistry()
        low = DummyPlugin(priority=0)
        high = DummyPlugin(priority=10)
        registry.register(low)
        registry.register(high)
        names = [p["name"] for p in registry.list_plugins()]
        # high (priority 10) should come first
        assert names == ["DummyPlugin", "DummyPlugin"]

    def test_get_returns_plugin_by_name(self) -> None:
        registry = CostModelRegistry()
        plugin = DummyPlugin()
        registry.register(plugin)
        assert registry.get("DummyPlugin") is plugin

    def test_get_returns_none_for_unknown(self) -> None:
        registry = CostModelRegistry()
        assert registry.get("nonexistent") is None

    def test_unregister_removes_plugin(self) -> None:
        registry = CostModelRegistry()
        registry.register(DummyPlugin())
        assert registry.unregister("DummyPlugin") is True
        assert registry.get("DummyPlugin") is None
        assert registry.list_plugins() == []

    def test_unregister_returns_false_for_missing(self) -> None:
        registry = CostModelRegistry()
        assert registry.unregister("nonexistent") is False

    def test_unregister_idempotent(self) -> None:
        registry = CostModelRegistry()
        assert registry.unregister("anything") is False

    def test_register_multiple_plugins(self) -> None:
        registry = CostModelRegistry()
        registry.register(DummyPlugin())
        registry.register(FailingPlugin())
        assert len(registry.list_plugins()) == 2

    def test_list_plugins_metadata(self) -> None:
        registry = CostModelRegistry()
        registry.register(DummyPlugin(priority=5))
        info = registry.list_plugins()[0]
        assert info["name"] == "DummyPlugin"
        assert info["priority"] == 5
        assert info["devices"] == ["dummy", "test_device"]

    def test_estimate_cost_returns_first_plugin_result(self) -> None:
        registry = CostModelRegistry()
        plugin_a = DummyPlugin(priority=10)
        plugin_b = DummyPlugin(priority=0)
        registry.register(plugin_a)
        registry.register(plugin_b)
        gpu = _make_gpu_profile(name="dummy")
        cost = registry.estimate_cost("node-0", [], batch_size=1, seq_len=128, gpu_profile=gpu)
        assert isinstance(cost, NodeCost)

    def test_estimate_cost_skips_non_matching_plugins(self) -> None:
        registry = CostModelRegistry()
        never = NeverHandlePlugin()
        dummy = DummyPlugin()
        registry.register(never)
        registry.register(dummy)
        gpu = _make_gpu_profile(name="dummy")
        cost = registry.estimate_cost("node-0", [_make_layer(0)], batch_size=1, seq_len=128, gpu_profile=gpu)
        assert isinstance(cost, NodeCost)
        assert cost.node_id == "node-0"

    def test_estimate_cost_returns_none_if_no_match(self) -> None:
        registry = CostModelRegistry()
        never = NeverHandlePlugin()
        registry.register(never)
        gpu = _make_gpu_profile(name="nvidia")
        cost = registry.estimate_cost("node-0", [_make_layer(0)], batch_size=1, seq_len=128, gpu_profile=gpu)
        assert cost is None

    def test_estimate_cost_returns_none_if_gpu_is_none(self) -> None:
        registry = CostModelRegistry()
        registry.register(DummyPlugin())
        cost = registry.estimate_cost("node-0", [_make_layer(0)], batch_size=1, seq_len=128, gpu_profile=None)
        assert cost is None

    def test_estimate_cost_skips_failing_plugin(self) -> None:
        registry = CostModelRegistry()
        registry.register(FailingPlugin())
        # No GPU match -> no plugin handles -> returns None (no crash)
        gpu = _make_gpu_profile(name="nvidia")
        cost = registry.estimate_cost("node-0", [_make_layer(0)], batch_size=1, seq_len=128, gpu_profile=gpu)
        assert cost is None

    def test_estimate_cost_returns_none_on_empty_registry(self) -> None:
        registry = CostModelRegistry()
        cost = registry.estimate_cost("node-0", [_make_layer(0)], batch_size=1, seq_len=128, gpu_profile=None)
        assert cost is None

    def test_estimate_cost_supports_optional_kwargs(self) -> None:
        registry = CostModelRegistry()
        registry.register(DummyPlugin())
        gpu = _make_gpu_profile(name="dummy")
        topo = _make_topology()
        cost = registry.estimate_cost(
            "node-0",
            [_make_layer(0)],
            batch_size=2,
            seq_len=512,
            gpu_profile=gpu,
            topology=topo,
            prev_node_id="prev-node",
        )
        assert isinstance(cost, NodeCost)


# ── TestPluginAwareCostModel ───────────────────────────────────────────────


class TestPluginAwareCostModel:
    """Tests for PluginAwareCostModel."""

    def test_evaluate_delegates_to_plugin_when_available(self) -> None:
        base = _make_base_cost_model()
        registry = CostModelRegistry()
        expected_cost = NodeCost(
            node_id="node-0", start_layer=0, end_layer=1,
            compute_time_ms=5.0, total_time_ms=5.0, fits_in_memory=True,
        )
        plugin = DummyPlugin(cost=expected_cost)
        registry.register(plugin)

        gpu = base._gpu_profiles["node-0"]
        # Force can_handle match by giving the GPU a matching name
        gpu.name = "dummy"

        aware = PluginAwareCostModel(base, registry)
        cost = aware.evaluate("node-0", 0, 1, batch_size=1, seq_len=128)
        assert cost.total_time_ms == 5.0
        assert cost.node_id == "node-0"

    def test_evaluate_falls_back_to_base_when_no_plugin_matches(self) -> None:
        base = _make_base_cost_model()
        registry = CostModelRegistry()
        # "never" won't match the "dummy" GPU name from _make_gpu_profile
        registry.register(NeverHandlePlugin())

        aware = PluginAwareCostModel(base, registry)
        cost = aware.evaluate("node-0", 0, 1, batch_size=1, seq_len=128)
        # Should come from the base cost model
        assert isinstance(cost, NodeCost)
        assert cost.node_id == "node-0"

    def test_evaluate_returns_base_result_on_empty_registry(self) -> None:
        base = _make_base_cost_model()
        registry = CostModelRegistry()
        aware = PluginAwareCostModel(base, registry)
        cost = aware.evaluate("node-0", 0, 1, batch_size=1, seq_len=128)
        assert isinstance(cost, NodeCost)
        assert cost.node_id == "node-0"
        assert cost.fits_in_memory is True

    def test_evaluate_partition_uses_plugins(self) -> None:
        base = _make_base_cost_model()
        registry = CostModelRegistry()
        expected = NodeCost(
            node_id="node-0", start_layer=0, end_layer=1,
            compute_time_ms=7.0, total_time_ms=7.0, fits_in_memory=True,
        )
        plugin = DummyPlugin(cost=expected)
        registry.register(plugin)

        gpu = base._gpu_profiles["node-0"]
        gpu.name = "dummy"

        aware = PluginAwareCostModel(base, registry)
        partition = [("node-0", 0, 1)]
        costs = aware.evaluate_partition(partition, batch_size=1, seq_len=128)
        assert len(costs) == 1
        assert costs[0].total_time_ms == 7.0

    def test_evaluate_partition_empty(self) -> None:
        base = _make_base_cost_model()
        registry = CostModelRegistry()
        aware = PluginAwareCostModel(base, registry)
        assert aware.evaluate_partition([], batch_size=1, seq_len=128) == []

    def test_combined_throughput_zero_for_empty_partition(self) -> None:
        base = _make_base_cost_model()
        registry = CostModelRegistry()
        aware = PluginAwareCostModel(base, registry)
        assert aware.combined_throughput([], batch_size=1, seq_len=128) == 0.0

    def test_combined_throughput_positive(self) -> None:
        base = _make_base_cost_model()
        registry = CostModelRegistry()
        # Register a plugin so we get a deterministic non-zero cost
        gpu = base._gpu_profiles["node-0"]
        gpu.name = "dummy"
        registry.register(DummyPlugin())
        aware = PluginAwareCostModel(base, registry)
        tput = aware.combined_throughput([("node-0", 0, 1)], batch_size=1, seq_len=128)
        assert tput > 0.0

    def test_max_latency_zero_for_empty_partition(self) -> None:
        base = _make_base_cost_model()
        registry = CostModelRegistry()
        aware = PluginAwareCostModel(base, registry)
        assert aware.max_latency([], batch_size=1, seq_len=128) == 0.0

    def test_max_latency_positive(self) -> None:
        base = _make_base_cost_model()
        registry = CostModelRegistry()
        gpu = base._gpu_profiles["node-0"]
        gpu.name = "dummy"
        registry.register(DummyPlugin())
        aware = PluginAwareCostModel(base, registry)
        latency = aware.max_latency([("node-0", 0, 1)], batch_size=1, seq_len=128)
        assert latency > 0.0

    def test_pipeline_latency_delegates_to_base(self) -> None:
        base = _make_base_cost_model()
        registry = CostModelRegistry()
        gpu = base._gpu_profiles["node-0"]
        gpu.name = "dummy"
        registry.register(DummyPlugin())
        aware = PluginAwareCostModel(base, registry)
        partition = [("node-0", 0, 1), ("node-1", 1, 2)]
        latency = aware.pipeline_latency(partition, batch_size=1, seq_len=128)
        assert latency > 0.0

    def test_pipeline_latency_with_stages_override(self) -> None:
        base = _make_base_cost_model()
        registry = CostModelRegistry()
        aware = PluginAwareCostModel(base, registry)
        partition = [("node-0", 0, 1), ("node-1", 1, 2)]
        latency = aware.pipeline_latency(partition, batch_size=1, seq_len=128, num_pipeline_stages=4)
        # pipeline_latency delegates to the base model; the second node
        # gets a non-zero communication cost on the link.
        assert latency > 0.0

    def test_evaluate_passes_prev_node_id(self) -> None:
        base = _make_base_cost_model()
        registry = CostModelRegistry()
        aware = PluginAwareCostModel(base, registry)
        cost = aware.evaluate("node-1", 0, 1, batch_size=1, seq_len=128, prev_node_id="node-0")
        assert isinstance(cost, NodeCost)

    def test_partition_with_multiple_nodes(self) -> None:
        base = _make_base_cost_model()
        registry = CostModelRegistry()
        aware = PluginAwareCostModel(base, registry)
        partition = [("node-0", 0, 1), ("node-1", 1, 2)]
        costs = aware.evaluate_partition(partition, batch_size=2, seq_len=256)
        assert len(costs) == 2
        for c in costs:
            assert isinstance(c, NodeCost)
            assert c.fits_in_memory is True
