"""Regression tests: quantization-aware cost model must not undercount memory.

These run against the REAL ``PartitionCostModel`` (real GPU profile, real
``LayerWeights`` via ``GPUProfiler.estimate_layer_weights``, real topology) —
no test doubles.

The previous ``QuantizationAwareCostModel.evaluate_with_quant`` computed
``memory_bytes`` as the quantized *weights* minus KV savings, which approved
partitions that OOM at inference.  The correct recomposition is:

    quantized weights + quantized KV cache + (unquantized) activation memory
"""

from __future__ import annotations

from distllm.dist.partition.cost_model import PartitionCostModel
from distllm.dist.partition.profiles import GPUProfile, GPUProfiler
from distllm.dist.partition.quant_cost import QuantizationAwareCostModel
from distllm.dist.partition.quantization_tuner import (
    ActivationQuantMethod,
    KVCacheBits,
    NodeQuantRecommendation,
    QuantMethod,
)
from distllm.dist.partition.topology import TopologyGraph


def _real_model(gpu_total_bytes: int) -> QuantizationAwareCostModel:
    """Build a QuantizationAwareCostModel over a REAL cost model."""
    profiler = GPUProfiler()
    layer_weights = profiler.estimate_layer_weights(
        hidden_size=1024, intermediate_size=4096,
        num_layers=4, num_heads=8, head_dim=64, vocab_size=10000,
    )
    profile = GPUProfile(
        gpu_id=0, name="TestGPU", total_memory_bytes=gpu_total_bytes,
        compute_tflops=50.0, memory_bandwidth_gbps=200.0, sm_count=40,
        peak_tflops_fp16=50.0,
    )
    topo = TopologyGraph(node_ids=["gpu-0"], gpu_counts={"gpu-0": 1}, links=[])
    base = PartitionCostModel({"gpu-0": profile}, layer_weights, topo)
    return QuantizationAwareCostModel(base)


def _rec(quant_bytes: int, without_bytes: int) -> NodeQuantRecommendation:
    return NodeQuantRecommendation(
        node_id="gpu-0", method=QuantMethod.BNB_4BIT,
        memory_bytes_without_quant=without_bytes,
        memory_bytes_with_quant=quant_bytes,
        memory_savings_bytes=without_bytes - quant_bytes,
        memory_savings_pct=round((1 - quant_bytes / without_bytes) * 100, 1),
        speed_penalty=1.1, quality_loss=0.02, reason="test",
        activation_quant=ActivationQuantMethod.NONE,
        kv_cache_bits=KVCacheBits.INT4,
    )


class TestQuantCostMemoryRecomposition:
    def test_memory_includes_kv_and_activations(self) -> None:
        """Real model: estimate is quantized weights + quantized KV +
        activations, not just quantized weights (or weights minus KV)."""
        model = _real_model(8 * 1024**3)
        weights = model._base.weights_memory_bytes(0, 4)
        rec = _rec(quant_bytes=int(weights * 0.25), without_bytes=weights)

        cost = model.evaluate_with_quant(
            "gpu-0", 0, 4, rec, batch_size=1, seq_len=4096,
        )

        # Never below the quantized weights (the old bug subtracted from it).
        assert cost.memory_bytes >= rec.memory_bytes_with_quant
        # Includes KV + activation memory, so it is strictly larger than the
        # quantized-weights-only figure.
        assert cost.memory_bytes > rec.memory_bytes_with_quant
        # Still a real reduction vs the unquantized footprint, and monotonic.
        assert cost.memory_bytes < cost.base_memory_bytes
        assert cost.memory_bytes <= cost.base_memory_bytes

    def test_oom_partition_is_no_longer_approved(self) -> None:
        """Real model on a tiny GPU: even after 4-bit weights the honest
        footprint does not fit, so the partition must be rejected."""
        model = _real_model(4 * 1024**2)  # 4MB node
        weights = model._base.weights_memory_bytes(0, 4)
        rec = _rec(quant_bytes=int(weights * 0.25), without_bytes=weights)

        cost = model.evaluate_with_quant("gpu-0", 0, 4, rec)

        assert cost.memory_bytes >= rec.memory_bytes_with_quant
        assert cost.fits_in_memory is False  # old math could approve this

    def test_no_quant_returns_base_cost(self) -> None:
        model = _real_model(8 * 1024**3)
        cost = model.evaluate_with_quant("gpu-0", 0, 4, quant_recommendation=None)
        assert cost.memory_bytes == cost.base_memory_bytes
        assert cost.weight_quant_method == "none"

    def test_total_memory_never_exceeds_unquantized(self) -> None:
        model = _real_model(8 * 1024**3)
        weights = model._base.weights_memory_bytes(0, 4)
        cost = model.evaluate_with_quant(
            "gpu-0", 0, 4, _rec(int(weights * 0.25), weights),
        )
        assert cost.memory_bytes <= cost.base_memory_bytes
