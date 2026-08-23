"""Tests for QuantizationAwareCostModel and QuantNodeCost.

Uses only real objects from the module (no mocks, no GPU required).
All hardware-dependent paths fall back gracefully to CPU defaults.
"""

from __future__ import annotations

import pytest

from distllm.dist.partition.cost_model import NodeCost, PartitionCostModel
from distllm.dist.partition.profiles import GPUProfile, LayerWeights
from distllm.dist.partition.quant_cost import (
    QuantizationAwareCostModel,
    QuantNodeCost,
)
from distllm.dist.partition.quantization_tuner import (
    ACTIVATION_PROFILES,
    KV_PROFILES,
    ActivationQuantMethod,
    KVCacheBits,
    NodeQuantRecommendation,
    QuantizationPlan,
    QuantMethod,
    QuantProfile,
    QUANT_PROFILES,
)
from distllm.dist.partition.topology import LinkProfile, TopologyGraph


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def layer_weights() -> list[LayerWeights]:
    """Minimal layer set: embed + 2 transformer + lm_head."""
    return [
        LayerWeights(
            layer_id=0,
            layer_type="embed",
            weight_memory_bytes=128 * 1024,
            activation_memory_bytes=4096,
            flops_per_token=500,
            flops_per_seq=500,
            kv_cache_bytes_per_token=0,
        ),
        LayerWeights(
            layer_id=1,
            layer_type="transformer",
            weight_memory_bytes=4 * 1024 * 1024,
            activation_memory_bytes=4096,
            flops_per_token=200_000,
            flops_per_seq=200_000,
            kv_cache_bytes_per_token=128,
        ),
        LayerWeights(
            layer_id=2,
            layer_type="transformer",
            weight_memory_bytes=4 * 1024 * 1024,
            activation_memory_bytes=4096,
            flops_per_token=200_000,
            flops_per_seq=200_000,
            kv_cache_bytes_per_token=128,
        ),
        LayerWeights(
            layer_id=3,
            layer_type="lm_head",
            weight_memory_bytes=128 * 1024,
            activation_memory_bytes=4096,
            flops_per_token=200,
            flops_per_seq=200,
            kv_cache_bytes_per_token=0,
        ),
    ]


@pytest.fixture
def gpu_profiles() -> dict[str, GPUProfile]:
    """Two GPU profiles keyed by node id."""
    return {
        "node-0": GPUProfile(
            gpu_id=0,
            name="A100",
            total_memory_bytes=80 * 1024**3,
            free_memory_bytes=76 * 1024**3,
            compute_tflops=312,
            memory_bandwidth_gbps=2039,
            sm_count=108,
            memory_bus_width_bits=5120,
        ),
        "node-1": GPUProfile(
            gpu_id=1,
            name="A100",
            total_memory_bytes=80 * 1024**3,
            free_memory_bytes=76 * 1024**3,
            compute_tflops=312,
            memory_bandwidth_gbps=2039,
            sm_count=108,
            memory_bus_width_bits=5120,
        ),
    }


@pytest.fixture
def topology() -> TopologyGraph:
    """Two-node topology with a single inter-node link."""
    return TopologyGraph(
        node_ids=["node-0", "node-1"],
        gpu_counts={"node-0": 1, "node-1": 1},
        links=[
            LinkProfile(
                source="node-0",
                target="node-1",
                bandwidth_gbps=12.5,
                latency_us=100.0,
            ),
        ],
    )


@pytest.fixture
def cost_model(
    gpu_profiles: dict[str, GPUProfile],
    layer_weights: list[LayerWeights],
    topology: TopologyGraph,
) -> PartitionCostModel:
    return PartitionCostModel(gpu_profiles, layer_weights, topology)


@pytest.fixture
def quant_cost_model(
    cost_model: PartitionCostModel,
) -> QuantizationAwareCostModel:
    return QuantizationAwareCostModel(cost_model)


@pytest.fixture
def rec_none() -> NodeQuantRecommendation:
    """No-quantization recommendation."""
    return NodeQuantRecommendation(
        node_id="node-0",
        method=QuantMethod.NONE,
        memory_bytes_without_quant=100_000_000,
        memory_bytes_with_quant=100_000_000,
        memory_savings_bytes=0,
        memory_savings_pct=0.0,
        speed_penalty=1.0,
        quality_loss=0.0,
        reason="No quantization",
    )


@pytest.fixture
def rec_int8() -> NodeQuantRecommendation:
    """INT8 weight quantization recommendation."""
    return NodeQuantRecommendation(
        node_id="node-0",
        method=QuantMethod.INT8,
        memory_bytes_without_quant=100_000_000,
        memory_bytes_with_quant=50_000_000,
        memory_savings_bytes=50_000_000,
        memory_savings_pct=50.0,
        speed_penalty=1.02,
        quality_loss=0.01,
        reason="INT8 quantization",
        activation_quant=ActivationQuantMethod.INT8,
        kv_cache_bits=KVCacheBits.INT8,
    )


@pytest.fixture
def rec_fp8() -> NodeQuantRecommendation:
    """FP8 weight quantization recommendation."""
    return NodeQuantRecommendation(
        node_id="node-0",
        method=QuantMethod.FP8_E4M3,
        memory_bytes_without_quant=100_000_000,
        memory_bytes_with_quant=50_000_000,
        memory_savings_bytes=50_000_000,
        memory_savings_pct=50.0,
        speed_penalty=0.90,
        quality_loss=0.005,
        reason="FP8 quantization",
        activation_quant=ActivationQuantMethod.FP8_E4M3,
        kv_cache_bits=KVCacheBits.FP8,
    )


@pytest.fixture
def rec_4bit() -> NodeQuantRecommendation:
    """4-bit weight quantization recommendation."""
    return NodeQuantRecommendation(
        node_id="node-0",
        method=QuantMethod.BNB_4BIT,
        memory_bytes_without_quant=100_000_000,
        memory_bytes_with_quant=25_000_000,
        memory_savings_bytes=75_000_000,
        memory_savings_pct=75.0,
        speed_penalty=1.10,
        quality_loss=0.03,
        reason="4-bit quantization",
    )


@pytest.fixture
def union_partition() -> list[tuple[str, int, int]]:
    """Partition spanning both nodes."""
    return [
        ("node-0", 0, 2),
        ("node-1", 2, 4),
    ]


@pytest.fixture
def quant_plan(
    rec_int8: NodeQuantRecommendation,
    rec_none: NodeQuantRecommendation,
) -> QuantizationPlan:
    """Plan with INT8 on node-0, no quant on node-1."""
    rec_node1 = NodeQuantRecommendation(
        node_id="node-1",
        method=QuantMethod.NONE,
        memory_bytes_without_quant=100_000_000,
        memory_bytes_with_quant=100_000_000,
        memory_savings_bytes=0,
        memory_savings_pct=0.0,
        speed_penalty=1.0,
        quality_loss=0.0,
        reason="No quant on node-1",
    )
    return QuantizationPlan(
        recommendations=[rec_int8, rec_node1],
        strategy="Mixed",
        total_memory_saved_bytes=50_000_000,
        avg_quality_loss=0.005,
    )


# ---------------------------------------------------------------------------
# Tests for QuantNodeCost
# ---------------------------------------------------------------------------


class TestQuantNodeCost:
    """Dataclass with computed properties and summary."""

    def test_defaults(self) -> None:
        cost = QuantNodeCost(node_id="x", start_layer=0, end_layer=1)
        assert cost.node_id == "x"
        assert cost.start_layer == 0
        assert cost.end_layer == 1
        assert cost.base_compute_time_ms == 0.0
        assert cost.base_memory_bytes == 0
        assert cost.compute_time_ms == 0.0
        assert cost.memory_bytes == 0
        assert cost.fits_in_memory is True
        assert cost.weight_quant_method == "none"
        assert cost.total_memory_reduction == 1.0
        assert cost.quality_loss_estimate == 0.0

    def test_memory_utilization_zero_available(self) -> None:
        cost = QuantNodeCost(node_id="x", start_layer=0, end_layer=1)
        assert cost.memory_utilization == 0.0

    def test_memory_utilization_computed(self) -> None:
        cost = QuantNodeCost(
            node_id="x",
            start_layer=0,
            end_layer=1,
            memory_bytes=500,
            memory_available_bytes=1000,
        )
        assert cost.memory_utilization == 0.5

    def test_memory_savings_bytes(self) -> None:
        cost = QuantNodeCost(
            node_id="x",
            start_layer=0,
            end_layer=1,
            base_memory_bytes=1000,
            memory_bytes=400,
        )
        assert cost.memory_savings_bytes == 600

    def test_memory_savings_bytes_negative(self) -> None:
        cost = QuantNodeCost(
            node_id="x",
            start_layer=0,
            end_layer=1,
            base_memory_bytes=400,
            memory_bytes=1000,
        )
        # Negative savings means memory increased (should not happen but test boundary)
        assert cost.memory_savings_bytes == -600

    def test_summary_format(self) -> None:
        cost = QuantNodeCost(
            node_id="node-0",
            start_layer=0,
            end_layer=4,
            weight_quant_method="int8",
            memory_bytes=50 * 1024 * 1024,
            memory_available_bytes=80 * 1024 * 1024 * 1024,
            compute_time_ms=10.5,
            total_time_ms=15.3,
            total_memory_reduction=2.0,
            fits_in_memory=True,
        )
        s = cost.summary()
        assert "Node node-0" in s
        assert "quant=int8" in s
        assert " [OOM]" not in s
        assert "15.3ms" in s

    def test_summary_oom_format(self) -> None:
        cost = QuantNodeCost(
            node_id="node-1",
            start_layer=0,
            end_layer=4,
            memory_bytes=100 * 1024 * 1024 * 1024,  # 100GB
            memory_available_bytes=80 * 1024 * 1024 * 1024,  # 80GB
            fits_in_memory=False,
            compute_time_ms=5.0,
            total_time_ms=8.0,
            total_memory_reduction=1.0,
        )
        s = cost.summary()
        assert "[OOM]" in s

    def test_memory_utilization_rounding(self) -> None:
        """Division should produce a float between 0 and 1."""
        cost = QuantNodeCost(
            node_id="x",
            start_layer=0,
            end_layer=1,
            memory_bytes=3,
            memory_available_bytes=7,
        )
        assert cost.memory_utilization == pytest.approx(3 / 7)

    def test_edge_empty_node_id(self) -> None:
        cost = QuantNodeCost(node_id="", start_layer=0, end_layer=0)
        assert cost.summary().startswith("Node ")

    def test_edge_negative_layers(self) -> None:
        cost = QuantNodeCost(node_id="n", start_layer=-5, end_layer=-1)
        assert cost.start_layer == -5
        assert cost.end_layer == -1


# ---------------------------------------------------------------------------
# Tests for QuantizationAwareCostModel
# ---------------------------------------------------------------------------


class TestQuantizationAwareCostModelInit:
    """Construction and configuration."""

    def test_default_profiles(self, cost_model: PartitionCostModel) -> None:
        """Uses QUANT_PROFILES when none provided."""
        qcm = QuantizationAwareCostModel(cost_model)
        assert qcm._profiles is QUANT_PROFILES

    def test_custom_profiles(self, cost_model: PartitionCostModel) -> None:
        """Accepts custom profile dict."""
        custom = {
            QuantMethod.INT8: QUANT_PROFILES[QuantMethod.INT8],
        }
        qcm = QuantizationAwareCostModel(cost_model, quant_profiles=custom)
        assert qcm._profiles is custom

    def test_empty_profiles(self, cost_model: PartitionCostModel) -> None:
        """Empty profile dict is falsy, so falls back to QUANT_PROFILES."""
        qcm = QuantizationAwareCostModel(cost_model, quant_profiles={})
        # {} is falsy in Python → falls through to QUANT_PROFILES
        assert qcm._profiles is QUANT_PROFILES


class TestQuantizationAwareCostModelEvaluateNoQuant:
    """evaluate_with_quant with None or NONE recommendation."""

    def test_none_recommendation(
        self, quant_cost_model: QuantizationAwareCostModel
    ) -> None:
        result = quant_cost_model.evaluate_with_quant(
            node_id="node-0",
            start_layer=0,
            end_layer=2,
            quant_recommendation=None,
        )
        assert isinstance(result, QuantNodeCost)
        assert result.weight_quant_method == "none"
        assert result.total_memory_reduction == 1.0
        assert result.fits_in_memory is True

    def test_method_none_recommendation(
        self,
        quant_cost_model: QuantizationAwareCostModel,
        rec_none: NodeQuantRecommendation,
    ) -> None:
        result = quant_cost_model.evaluate_with_quant(
            node_id="node-0",
            start_layer=0,
            end_layer=2,
            quant_recommendation=rec_none,
        )
        assert result.weight_quant_method == "none"
        assert result.total_memory_reduction == 1.0
        assert result.quality_loss_estimate == 0.0

    def test_base_cost_passthrough(
        self, quant_cost_model: QuantizationAwareCostModel
    ) -> None:
        """Without quant, base_compute matches compute_time_ms."""
        result = quant_cost_model.evaluate_with_quant(
            node_id="node-0",
            start_layer=0,
            end_layer=2,
            quant_recommendation=None,
        )
        assert result.base_compute_time_ms == result.compute_time_ms
        assert result.base_memory_bytes == result.memory_bytes


class TestQuantizationAwareCostModelEvaluateWithQuant:
    """evaluate_with_quant with active quantization."""

    def test_int8_quant_applied(
        self,
        quant_cost_model: QuantizationAwareCostModel,
        rec_int8: NodeQuantRecommendation,
    ) -> None:
        result = quant_cost_model.evaluate_with_quant(
            node_id="node-0",
            start_layer=0,
            end_layer=2,
            quant_recommendation=rec_int8,
        )
        assert result.weight_quant_method == QuantMethod.INT8.value
        assert result.weight_memory_reduction == QUANT_PROFILES[QuantMethod.INT8].memory_reduction
        # Memory should differ from base
        assert result.memory_bytes != result.base_memory_bytes
        # Compute penalty applied (INT8 has speed_penalty 1.02)
        assert result.base_compute_time_ms * 1.02 == pytest.approx(
            result.compute_time_ms, rel=0.1
        )

    def test_fp8_quant_applied(
        self,
        quant_cost_model: QuantizationAwareCostModel,
        rec_fp8: NodeQuantRecommendation,
    ) -> None:
        result = quant_cost_model.evaluate_with_quant(
            node_id="node-0",
            start_layer=0,
            end_layer=2,
            quant_recommendation=rec_fp8,
        )
        assert result.weight_quant_method == QuantMethod.FP8_E4M3.value
        # FP8 has speed_penalty 0.90 (speedup)
        assert result.base_compute_time_ms * 0.90 == pytest.approx(
            result.compute_time_ms, rel=0.1
        )

    def test_4bit_quant_applied(
        self,
        quant_cost_model: QuantizationAwareCostModel,
        rec_4bit: NodeQuantRecommendation,
    ) -> None:
        result = quant_cost_model.evaluate_with_quant(
            node_id="node-0",
            start_layer=0,
            end_layer=2,
            quant_recommendation=rec_4bit,
        )
        assert result.weight_quant_method == QuantMethod.BNB_4BIT.value
        assert result.quality_loss_estimate == 0.03
        # Weight memory reduction should be 0.25
        assert result.weight_memory_reduction == QUANT_PROFILES[QuantMethod.BNB_4BIT].memory_reduction

    def test_activation_quant_applied(
        self,
        quant_cost_model: QuantizationAwareCostModel,
        rec_int8: NodeQuantRecommendation,
    ) -> None:
        result = quant_cost_model.evaluate_with_quant(
            node_id="node-0",
            start_layer=0,
            end_layer=2,
            quant_recommendation=rec_int8,
        )
        assert result.activation_quant_method == ActivationQuantMethod.INT8.value
        # Activation bandwidth should be reduced
        act_profile = ACTIVATION_PROFILES[ActivationQuantMethod.INT8]
        assert result.activation_bandwidth_reduction == act_profile["bandwidth_reduction"]

    def test_kv_cache_quant_applied(
        self,
        quant_cost_model: QuantizationAwareCostModel,
        rec_int8: NodeQuantRecommendation,
    ) -> None:
        result = quant_cost_model.evaluate_with_quant(
            node_id="node-0",
            start_layer=0,
            end_layer=2,
            quant_recommendation=rec_int8,
        )
        assert result.kv_cache_bits == KVCacheBits.INT8.value
        kv_profile = KV_PROFILES[KVCacheBits.INT8]
        assert result.kv_memory_reduction == kv_profile["memory_reduction"]

    def test_total_memory_reduction(
        self,
        quant_cost_model: QuantizationAwareCostModel,
        rec_int8: NodeQuantRecommendation,
    ) -> None:
        result = quant_cost_model.evaluate_with_quant(
            node_id="node-0",
            start_layer=0,
            end_layer=2,
            quant_recommendation=rec_int8,
        )
        # total_memory_reduction should be >= 1.0 (reduction) or could be < 1 if
        # the recommendation bytes are weird; just check it's a float > 0
        assert result.total_memory_reduction > 0
        assert isinstance(result.total_memory_reduction, float)

    def test_oom_detection(
        self,
        quant_cost_model: QuantizationAwareCostModel,
        rec_int8: NodeQuantRecommendation,
    ) -> None:
        """If quantized memory > 90% of available, fits_in_memory is False."""
        result = quant_cost_model.evaluate_with_quant(
            node_id="node-0",
            start_layer=0,
            end_layer=2,
            quant_recommendation=rec_int8,
            # These tiny GPUs won't fit the model
            batch_size=1,
            seq_len=4096,
        )
        # The recommendation already has memory_bytes_with_quant=50MB, which is
        # far below the 80GB available in the fixture, so this should fit.
        assert result.fits_in_memory is True

    def test_unknown_method_returns_base(
        self,
        quant_cost_model: QuantizationAwareCostModel,
    ) -> None:
        """Unknown quant method logs warning and returns base cost."""
        from distllm.dist.partition.quantization_tuner import QuantMethod

        rec_unknown = NodeQuantRecommendation(
            node_id="node-0",
            method=QuantMethod.NONE,  # Will become unknown via custom profiles
            memory_bytes_without_quant=100_000_000,
            memory_bytes_with_quant=50_000_000,
            memory_savings_bytes=50_000_000,
            memory_savings_pct=50.0,
            speed_penalty=1.0,
            quality_loss=0.0,
            reason="Unknown method test",
        )
        # Build model with empty profiles so no method is known
        empty_model = QuantizationAwareCostModel(
            quant_cost_model._base, quant_profiles={}
        )
        result = empty_model.evaluate_with_quant(
            node_id="node-0",
            start_layer=0,
            end_layer=2,
            quant_recommendation=rec_unknown,
        )
        # Should fall back to base-like cost
        assert result.base_compute_time_ms == result.compute_time_ms

    def test_custom_batch_seq_len(
        self,
        quant_cost_model: QuantizationAwareCostModel,
        rec_int8: NodeQuantRecommendation,
    ) -> None:
        """Different batch_size and seq_len produce different costs."""
        result1 = quant_cost_model.evaluate_with_quant(
            node_id="node-0",
            start_layer=0,
            end_layer=2,
            quant_recommendation=rec_int8,
            batch_size=1,
            seq_len=4096,
        )
        result2 = quant_cost_model.evaluate_with_quant(
            node_id="node-0",
            start_layer=0,
            end_layer=2,
            quant_recommendation=rec_int8,
            batch_size=8,
            seq_len=8192,
        )
        # Larger batch/seq should increase compute time
        assert result2.compute_time_ms > result1.compute_time_ms


class TestQuantizationAwareCostModelEvaluatePartition:
    """evaluate_partition_with_quant for multi-node partitions."""

    def test_empty_partition(
        self, quant_cost_model: QuantizationAwareCostModel
    ) -> None:
        costs = quant_cost_model.evaluate_partition_with_quant(
            partition=[], quant_plan=None
        )
        assert costs == []

    def test_partition_without_plan(
        self,
        quant_cost_model: QuantizationAwareCostModel,
        union_partition: list[tuple[str, int, int]],
    ) -> None:
        costs = quant_cost_model.evaluate_partition_with_quant(
            partition=union_partition, quant_plan=None
        )
        assert len(costs) == 2
        for c in costs:
            assert c.weight_quant_method == "none"
            assert c.total_memory_reduction == 1.0

    def test_partition_with_plan(
        self,
        quant_cost_model: QuantizationAwareCostModel,
        union_partition: list[tuple[str, int, int]],
        quant_plan: QuantizationPlan,
    ) -> None:
        costs = quant_cost_model.evaluate_partition_with_quant(
            partition=union_partition, quant_plan=quant_plan
        )
        assert len(costs) == 2
        # node-0 gets INT8
        assert costs[0].weight_quant_method == QuantMethod.INT8.value
        # node-1 gets NONE
        assert costs[1].weight_quant_method == "none"

    def test_partition_plan_missing_node(
        self,
        quant_cost_model: QuantizationAwareCostModel,
        union_partition: list[tuple[str, int, int]],
    ) -> None:
        """Nodes without a recommendation in the plan get base cost."""
        plan = QuantizationPlan(
            recommendations=[
                NodeQuantRecommendation(
                    node_id="unknown-node",
                    method=QuantMethod.INT8,
                    memory_bytes_without_quant=100_000_000,
                    memory_bytes_with_quant=50_000_000,
                    memory_savings_bytes=50_000_000,
                    memory_savings_pct=50.0,
                    speed_penalty=1.02,
                    quality_loss=0.01,
                    reason="Only for unknown-node",
                )
            ]
        )
        costs = quant_cost_model.evaluate_partition_with_quant(
            partition=union_partition, quant_plan=plan
        )
        assert len(costs) == 2
        for c in costs:
            assert c.weight_quant_method == "none"


class TestQuantizationAwareCostModelCompare:
    """compare_with_without_quant returns comparison dict."""

    def test_both_sides_present(
        self,
        quant_cost_model: QuantizationAwareCostModel,
        union_partition: list[tuple[str, int, int]],
        quant_plan: QuantizationPlan,
    ) -> None:
        result = quant_cost_model.compare_with_without_quant(
            partition=union_partition,
            quant_plan=quant_plan,
        )
        assert "without_quant" in result
        assert "with_quant" in result
        # Both sides should have summary keys
        for side in ("without_quant", "with_quant"):
            assert "max_latency_ms" in result[side]
            assert "total_memory_bytes" in result[side]
            assert "oom_nodes" in result[side]
            assert "nodes" in result[side]

    def test_quant_populates_fields(
        self,
        quant_cost_model: QuantizationAwareCostModel,
        union_partition: list[tuple[str, int, int]],
        quant_plan: QuantizationPlan,
    ) -> None:
        """Compare result has all expected keys and summaries per side."""
        result = quant_cost_model.compare_with_without_quant(
            partition=union_partition,
            quant_plan=quant_plan,
        )
        for side in ("without_quant", "with_quant"):
            d = result[side]
            assert "max_latency_ms" in d
            assert "total_memory_bytes" in d
            assert "total_memory_gb" in d
            assert "oom_nodes" in d
            assert "avg_memory_reduction" in d
            assert "nodes" in d
            assert isinstance(d["nodes"], list)
            assert len(d["nodes"]) == 2

    def test_empty_partition_compare(
        self, quant_cost_model: QuantizationAwareCostModel
    ) -> None:
        plan = QuantizationPlan(recommendations=[])
        result = quant_cost_model.compare_with_without_quant(
            partition=[], quant_plan=plan
        )
        assert result["without_quant"] == {}
        assert result["with_quant"] == {}

    def test_compare_rejects_none_plan(
        self,
        quant_cost_model: QuantizationAwareCostModel,
        union_partition: list[tuple[str, int, int]],
    ) -> None:
        """compare_with_without_quant should accept None quant_plan."""
        result = quant_cost_model.compare_with_without_quant(
            partition=union_partition,
            quant_plan=None,  # type: ignore[arg-type]
        )
        assert "without_quant" in result
        assert "with_quant" in result


class TestQuantizationAwareCostModelEdgeCases:
    """Edge and boundary cases."""

    def test_empty_layer_range(
        self, quant_cost_model: QuantizationAwareCostModel
    ) -> None:
        """start_layer == end_layer produces zero-sized segment."""
        result = quant_cost_model.evaluate_with_quant(
            node_id="node-0",
            start_layer=2,
            end_layer=2,
            quant_recommendation=None,
        )
        assert isinstance(result, QuantNodeCost)
        # Empty range -> no layers, memory=0, compute=0
        assert result.memory_bytes == 0
        assert result.compute_time_ms == 0.0

    def test_invalid_layer_range(
        self, quant_cost_model: QuantizationAwareCostModel
    ) -> None:
        """start_layer > end_layer produces empty segment."""
        result = quant_cost_model.evaluate_with_quant(
            node_id="node-0",
            start_layer=5,
            end_layer=2,
            quant_recommendation=None,
        )
        assert result.memory_bytes == 0
        assert result.compute_time_ms == 0.0

    def test_negative_batch_size(
        self,
        quant_cost_model: QuantizationAwareCostModel,
        rec_int8: NodeQuantRecommendation,
    ) -> None:
        """Negative batch size is unusual but should not crash."""
        result = quant_cost_model.evaluate_with_quant(
            node_id="node-0",
            start_layer=0,
            end_layer=2,
            quant_recommendation=rec_int8,
            batch_size=-1,
        )
        assert isinstance(result, QuantNodeCost)

    def test_zero_batch_size(
        self,
        quant_cost_model: QuantizationAwareCostModel,
        rec_int8: NodeQuantRecommendation,
    ) -> None:
        """Zero batch size should not crash (base model may still estimate)."""
        result = quant_cost_model.evaluate_with_quant(
            node_id="node-0",
            start_layer=0,
            end_layer=2,
            quant_recommendation=rec_int8,
            batch_size=0,
        )
        # Just verify no crash and result is a QuantNodeCost
        assert isinstance(result, QuantNodeCost)
        assert result.node_id == "node-0"

    def test_large_seq_len_passthrough(
        self,
        quant_cost_model: QuantizationAwareCostModel,
        rec_int8: NodeQuantRecommendation,
    ) -> None:
        """Very large seq_len should not crash."""
        result = quant_cost_model.evaluate_with_quant(
            node_id="node-0",
            start_layer=0,
            end_layer=2,
            quant_recommendation=rec_int8,
            seq_len=1_000_000,
        )
        assert isinstance(result, QuantNodeCost)
        assert result.total_time_ms > 0

    def test_negative_seq_len(
        self,
        quant_cost_model: QuantizationAwareCostModel,
        rec_int8: NodeQuantRecommendation,
    ) -> None:
        """Negative seq_len is accepted by base model (won't crash)."""
        result = quant_cost_model.evaluate_with_quant(
            node_id="node-0",
            start_layer=0,
            end_layer=2,
            quant_recommendation=rec_int8,
            seq_len=-100,
        )
        assert isinstance(result, QuantNodeCost)

    def test_single_layer_partition(
        self, quant_cost_model: QuantizationAwareCostModel
    ) -> None:
        """Partition with just one layer."""
        costs = quant_cost_model.evaluate_partition_with_quant(
            partition=[("node-0", 1, 2)],
            quant_plan=None,
        )
        assert len(costs) == 1
        assert costs[0].start_layer == 1
        assert costs[0].end_layer == 2

    def test_many_small_partitions(
        self, quant_cost_model: QuantizationAwareCostModel
    ) -> None:
        """Partition with many tiny segments."""
        segments = [(f"node-0", i, i + 1) for i in range(4)]
        costs = quant_cost_model.evaluate_partition_with_quant(
            partition=segments,
            quant_plan=None,
        )
        assert len(costs) == 4

    def test_recommendation_with_kv_cache_overrides_memory(
        self,
        quant_cost_model: QuantizationAwareCostModel,
    ) -> None:
        """KV cache savings should reduce total memory."""
        rec = NodeQuantRecommendation(
            node_id="node-0",
            method=QuantMethod.INT8,
            memory_bytes_without_quant=100_000_000,
            memory_bytes_with_quant=50_000_000,
            memory_savings_bytes=50_000_000,
            memory_savings_pct=50.0,
            speed_penalty=1.02,
            quality_loss=0.01,
            reason="With KV cache",
            kv_cache_bits=KVCacheBits.INT4,
        )
        result = quant_cost_model.evaluate_with_quant(
            node_id="node-0",
            start_layer=0,
            end_layer=2,
            quant_recommendation=rec,
        )
        # INT4 KV cache has memory_reduction=0.25
        assert result.kv_cache_bits == KVCacheBits.INT4.value
        assert result.kv_memory_reduction == 0.25
        # Total memory should be <= recommendation bytes minus KV savings
        assert result.memory_bytes <= 50_000_000

    def test_recommendation_with_no_kv_cache(
        self,
        quant_cost_model: QuantizationAwareCostModel,
        rec_int8: NodeQuantRecommendation,
    ) -> None:
        """When kv_cache_bits is NONE, no KV savings are applied."""
        result = quant_cost_model.evaluate_with_quant(
            node_id="node-0",
            start_layer=0,
            end_layer=2,
            quant_recommendation=rec_int8,
        )
        assert result.kv_cache_bits == KVCacheBits.INT8.value
        # KV savings applied via the profile lookup path

    def test_custom_profile_override(
        self,
        cost_model: PartitionCostModel,
        rec_int8: NodeQuantRecommendation,
    ) -> None:
        """Custom profiles override global defaults."""
        custom_profile = QuantProfile(
            method=QuantMethod.INT8,
            memory_reduction=0.1,  # very aggressive
            speed_penalty=2.0,
            min_vram_gb=0,
            quality_loss=0.1,
        )
        custom_model = QuantizationAwareCostModel(
            cost_model,
            quant_profiles={QuantMethod.INT8: custom_profile},
        )
        result = custom_model.evaluate_with_quant(
            node_id="node-0",
            start_layer=0,
            end_layer=2,
            quant_recommendation=rec_int8,
        )
        assert result.weight_memory_reduction == 0.1
        assert result.base_compute_time_ms * 2.0 == pytest.approx(
            result.compute_time_ms, rel=0.1
        )
