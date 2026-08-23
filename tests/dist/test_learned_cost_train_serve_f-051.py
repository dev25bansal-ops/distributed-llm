"""Regression tests for audit finding F-051.

LearnedCostModel train/serve feature skew: FeatureExtractor.extract (the
serving path) used to emit intermediate_size == 0.0 always (feature index 14)
and memory/flops proxy formulas that disagreed with the training path
(_observation_to_features).  The fix makes both paths derive their features
from the same parametric layer layout:

* profiles.LayerWeights now carries ground-truth hidden_size /
  intermediate_size, populated by GPUProfiler.estimate_layer_weights;
* FeatureExtractor.extract reads those annotations (falling back to the
  shared _estimate_hidden_intermediate estimator for unannotated weights);
* _observation_to_features rebuilds the same aggregates from the
  observation's dims using the same _single_layer_* helpers.

The load-bearing invariant pinned here: for a canonical node/config, the
training-time feature vector must EQUAL the serving-time feature vector.
"""
from __future__ import annotations

import math

import pytest

from distllm.dist.partition.cost_model import NodeCost, PartitionCostModel
from distllm.dist.partition.learned_cost import (
    FeatureExtractor,
    LearnedCostModel,
    RuntimeObservation,
    _KV_CACHE_BYTES_PER_TOKEN,
    _estimate_hidden_intermediate,
    _single_layer_flops,
    _single_layer_weight_bytes,
)
from distllm.dist.partition.profiles import GPUProfile, GPUProfiler, LayerWeights
from distllm.dist.partition.topology import LinkProfile, TopologyGraph

HIDDEN = 4096
INTERMEDIATE = 11008
NUM_LAYERS = 8


def _canonical_gpu_profiles() -> dict[str, GPUProfile]:
    return {
        "node-0": GPUProfile(
            gpu_id=0,
            name="A100",
            total_memory_bytes=80 * 1024**3,
            compute_tflops=312.0,
            memory_bandwidth_gbps=2039.0,
        ),
        "node-1": GPUProfile(
            gpu_id=1,
            name="A100",
            total_memory_bytes=80 * 1024**3,
            compute_tflops=312.0,
            memory_bandwidth_gbps=2039.0,
        ),
    }


def _canonical_topology() -> TopologyGraph:
    return TopologyGraph(
        node_ids=["node-0", "node-1"],
        gpu_counts={"node-0": 1, "node-1": 1},
        links=[
            LinkProfile(
                source="node-0",
                target="node-1",
                bandwidth_gbps=600.0,
                latency_us=5.0,
                is_nvlink=True,
            ),
        ],
    )


def _transformer_only_layer_weights() -> list[LayerWeights]:
    """Real generator output, restricted to the transformer blocks a
    partition slice actually contains (embed/lm_head are never mid-model)."""
    all_layers = GPUProfiler().estimate_layer_weights(
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        num_layers=NUM_LAYERS,
    )
    return [l for l in all_layers if l.layer_type == "transformer"]


def _canonical_observation(num_layers: int = 4) -> RuntimeObservation:
    return RuntimeObservation(
        node_id="node-1",
        num_layers=num_layers,
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        batch_size=1,
        seq_len=4096,
        gpu_tflops=312.0,
        gpu_mem_bw_gbps=2039.0,
        gpu_mem_bytes=80 * 1024**3,
        is_nvlink=True,
        is_infiniband=False,
        comm_bandwidth_gbps=600.0,
        measured_latency_ms=25.0,
    )


class TestTrainServeFeatureAlignment:
    def test_intermediate_size_feature_is_not_always_zero(self):
        """Core F-051 regression: serving feature[14] used to be hard 0.0."""
        features = FeatureExtractor.extract(
            node_id="node-1",
            start_layer=0,
            end_layer=4,
            batch_size=1,
            seq_len=4096,
            gpu_profiles=_canonical_gpu_profiles(),
            layer_weights=_transformer_only_layer_weights(),
            topology=_canonical_topology(),
        )
        assert features[14] == float(INTERMEDIATE)
        assert features[13] == float(HIDDEN)

    def test_train_features_equal_serve_features_canonical(self):
        """The finding's recommended invariant: train-time features ==
        serve-time features for a canonical node/config.

        Uses an interior 4-layer slice: mid-model partitions contain
        homogeneous transformer blocks, whereas the generator tags the very
        first/last transformer layers with a small boundary weight bonus
        (+2*h*dtype) that an observation cannot express.
        """
        layer_weights = _transformer_only_layer_weights()
        train_feats = LearnedCostModel._observation_to_features(
            object.__new__(LearnedCostModel),  # method needs no instance state
            _canonical_observation(num_layers=4),
        )
        serve_feats = FeatureExtractor.extract(
            node_id="node-1",
            start_layer=1,
            end_layer=5,
            batch_size=1,
            seq_len=4096,
            gpu_profiles=_canonical_gpu_profiles(),
            layer_weights=layer_weights,
            topology=_canonical_topology(),
        )
        assert len(train_feats) == len(serve_feats) == 15
        for idx, (tr, sv) in enumerate(zip(train_feats, serve_feats)):
            assert tr == sv, f"feature[{idx}] train={tr} != serve={sv}"

    def test_aligned_via_model_helpers(self):
        """Same invariant exercised through a real LearnedCostModel instance."""
        model = LearnedCostModel(
            base_cost_model=PartitionCostModel(
                gpu_profiles=_canonical_gpu_profiles(),
                layer_weights=_transformer_only_layer_weights(),
                topology=_canonical_topology(),
            ),
            gpu_profiles=_canonical_gpu_profiles(),
        )
        train_feats = model._observation_to_features(_canonical_observation(4))
        serve_feats = model._extract_features("node-1", 1, 5, 1, 4096)
        assert train_feats == serve_feats


class TestSharedEstimator:
    def test_estimate_hidden_intermediate_positive(self):
        """Fallback estimator must recover BOTH dims (old code gave i == 0)."""
        w = _single_layer_weight_bytes(HIDDEN, INTERMEDIATE)
        h, i = _estimate_hidden_intermediate(w)
        assert h > 0
        assert i > 0
        # Within a few percent of the true dims.
        assert abs(h - HIDDEN) / HIDDEN < 0.05
        assert abs(i - INTERMEDIATE) / INTERMEDIATE < 0.05

    def test_estimate_roundtrip_matches_generator_weights(self):
        """Generator-produced weight bytes -> estimator -> same dims back."""
        layers = _transformer_only_layer_weights()
        sample = layers[0]
        h, i = _estimate_hidden_intermediate(sample.weight_memory_bytes)
        assert h == pytest.approx(HIDDEN, rel=0.05)
        assert i == pytest.approx(INTERMEDIATE, rel=0.05)

    def test_estimate_degenerate_inputs(self):
        assert _estimate_hidden_intermediate(0) == (0, 0)
        assert _estimate_hidden_intermediate(-5) == (0, 0)

    def test_unannotated_weights_still_get_nonzero_intermediate(self):
        """Hand-built LayerWeights without dim annotations take the fallback
        path and must still yield a nonzero intermediate_size feature."""
        raw = [
            LayerWeights(
                layer_id=k,
                layer_type="transformer",
                weight_memory_bytes=_single_layer_weight_bytes(HIDDEN, INTERMEDIATE),
                activation_memory_bytes=2 * HIDDEN,
                flops_per_token=_single_layer_flops(HIDDEN, INTERMEDIATE),
                flops_per_seq=_single_layer_flops(HIDDEN, INTERMEDIATE),
                kv_cache_bytes_per_token=_KV_CACHE_BYTES_PER_TOKEN,
            )
            for k in range(NUM_LAYERS)
        ]
        features = FeatureExtractor.extract(
            "node-1", 0, 4, 1, 4096,
            _canonical_gpu_profiles(), raw, _canonical_topology(),
        )
        assert features[13] > 0
        assert features[14] > 0


class TestEndToEndTrainedEvaluate:
    def test_record_train_evaluate_uses_aligned_features(self):
        """Full loop: record real observations, train, evaluate — the learned
        path must run on one consistent feature distribution."""
        layer_weights = _transformer_only_layer_weights()
        gpu_profiles = _canonical_gpu_profiles()
        topology = _canonical_topology()
        model = LearnedCostModel(
            base_cost_model=PartitionCostModel(
                gpu_profiles=gpu_profiles,
                layer_weights=layer_weights,
                topology=topology,
            ),
            gpu_profiles=gpu_profiles,
            min_samples_to_train=5,
            retrain_interval=9999.0,
        )
        for k in range(6):
            obs = _canonical_observation(num_layers=4)
            obs.measured_latency_ms = 20.0 + k
            model.record(obs)

        metrics = model.train()
        assert model.is_learned
        assert metrics.num_samples == 6

        cost = model.evaluate("node-1", 0, 4, batch_size=1, seq_len=4096)
        assert isinstance(cost, NodeCost)
        assert cost.total_time_ms >= 0
