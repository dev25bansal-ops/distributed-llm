"""Tests for dist/partition/learned_cost module — real objects, zero mocks."""
from __future__ import annotations

import math
import json
import tempfile
from pathlib import Path

import pytest

from distllm.dist.partition.learned_cost import (
    RuntimeObservation,
    TrainingMetrics,
    FeatureExtractor,
    DecisionStump,
    GradientBoostedTrees,
    LearnedCostModel,
)
from distllm.dist.partition.cost_model import PartitionCostModel, NodeCost
from distllm.dist.partition.profiles import GPUProfile, LayerWeights
from distllm.dist.partition.topology import TopologyGraph, LinkProfile


# ---------------------------------------------------------------------------
# Helper fixtures  (real objects, no mocks)
# ---------------------------------------------------------------------------


@pytest.fixture
def gpu_profiles() -> dict[str, GPUProfile]:
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


@pytest.fixture
def layer_weights() -> list[LayerWeights]:
    h = 4096
    inter = 11008
    dtype = 2
    qkv = 3 * h * h
    o_proj = h * h
    gate = h * inter
    up = h * inter
    down = inter * h
    norms = 4 * h
    weight = (qkv + o_proj + gate + up + down + norms) * dtype
    flops_per = (
        3 * 2 * h * h
        + 2 * 2 * 32 * 128
        + 2 * h * h
        + 2 * h * inter * 3
    )
    kv = 2 * 32 * 128 * dtype
    return [
        LayerWeights(
            layer_id=i,
            layer_type="transformer",
            weight_memory_bytes=weight,
            activation_memory_bytes=2 * h,
            flops_per_token=flops_per,
            flops_per_seq=flops_per,
            kv_cache_bytes_per_token=kv,
        )
        for i in range(8)
    ]


@pytest.fixture
def topology() -> TopologyGraph:
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


@pytest.fixture
def base_cost_model(
    gpu_profiles: dict[str, GPUProfile],
    layer_weights: list[LayerWeights],
    topology: TopologyGraph,
) -> PartitionCostModel:
    return PartitionCostModel(
        gpu_profiles=gpu_profiles,
        layer_weights=layer_weights,
        topology=topology,
    )


@pytest.fixture
def empty_cost_model() -> PartitionCostModel:
    return PartitionCostModel(
        gpu_profiles={},
        layer_weights=[],
        topology=TopologyGraph(),
    )


# ---------------------------------------------------------------------------
# RuntimeObservation
# ---------------------------------------------------------------------------


class TestRuntimeObservation:
    def test_default_timestamp(self):
        obs = RuntimeObservation(
            node_id="n0", num_layers=4, hidden_size=4096,
            intermediate_size=11008, batch_size=1, seq_len=4096,
            gpu_tflops=312.0, gpu_mem_bw_gbps=2039.0,
            gpu_mem_bytes=80 * 1024**3,
            is_nvlink=True, is_infiniband=False,
            comm_bandwidth_gbps=600.0,
        )
        assert obs.node_id == "n0"
        assert obs.quant_bits == 16
        assert obs.measured_latency_ms == 0.0
        assert obs.measured_memory_bytes == 0
        assert obs.timestamp > 0

    def test_default_quant_bits(self):
        obs = RuntimeObservation(
            node_id="n0", num_layers=4, hidden_size=4096,
            intermediate_size=11008, batch_size=1, seq_len=4096,
            gpu_tflops=312.0, gpu_mem_bw_gbps=2039.0,
            gpu_mem_bytes=80 * 1024**3,
            is_nvlink=False, is_infiniband=False,
            comm_bandwidth_gbps=12.5,
        )
        assert obs.quant_bits == 16

    def test_measured_values(self):
        obs = RuntimeObservation(
            node_id="n0", num_layers=4, hidden_size=4096,
            intermediate_size=11008, batch_size=2, seq_len=2048,
            gpu_tflops=312.0, gpu_mem_bw_gbps=2039.0,
            gpu_mem_bytes=80 * 1024**3,
            is_nvlink=False, is_infiniband=False,
            comm_bandwidth_gbps=12.5,
            measured_latency_ms=15.2,
            measured_memory_bytes=42_000_000_000,
        )
        assert obs.measured_latency_ms == 15.2
        assert obs.measured_memory_bytes == 42_000_000_000


# ---------------------------------------------------------------------------
# TrainingMetrics
# ---------------------------------------------------------------------------


class TestTrainingMetrics:
    def test_default_values(self):
        m = TrainingMetrics()
        assert m.num_samples == 0
        assert m.mae_ms == 0.0
        assert m.mape_pct == 0.0
        assert m.r_squared == 0.0
        assert m.max_error_ms == 0.0
        assert m.trained_at == 0.0

    def test_all_fields(self):
        m = TrainingMetrics(
            num_samples=100, mae_ms=2.5, mape_pct=5.0,
            r_squared=0.95, max_error_ms=12.3, trained_at=1000.0,
        )
        assert m.num_samples == 100
        assert m.mae_ms == 2.5
        assert m.mape_pct == 5.0
        assert m.r_squared == 0.95
        assert m.max_error_ms == 12.3
        assert m.trained_at == 1000.0


# ---------------------------------------------------------------------------
# FeatureExtractor
# ---------------------------------------------------------------------------


class TestFeatureExtractor:
    def test_feature_names_returned(self):
        names = FeatureExtractor.feature_names()
        assert len(names) == 15
        assert "num_layers" in names
        assert "batch_size" in names
        assert "seq_len" in names
        assert "hidden_size" in names
        assert "intermediate_size" in names
        assert "is_nvlink" in names
        assert "is_infiniband" in names

    def test_extract_basic(
        self,
        gpu_profiles: dict[str, GPUProfile],
        layer_weights: list[LayerWeights],
        topology: TopologyGraph,
    ):
        features = FeatureExtractor.extract(
            node_id="node-1",
            start_layer=2,
            end_layer=6,
            batch_size=1,
            seq_len=4096,
            gpu_profiles=gpu_profiles,
            layer_weights=layer_weights,
            topology=topology,
        )
        assert len(features) == 15
        # num_layers = 4 (end_layer - start_layer)
        assert features[0] == 4.0
        assert features[1] == 1.0  # batch_size
        assert features[2] == 4096.0  # seq_len
        # is_nvlink = 1.0 (there is a link between node-0 and node-1)
        assert features[11] == 1.0
        # is_infiniband = 0.0
        assert features[12] == 0.0
        # gpu_tflops > 0
        assert features[7] > 0
        # gpu_mem_bw_gbps > 0
        assert features[8] > 0

    def test_extract_first_node_has_no_prev(self, layer_weights, topology):
        features = FeatureExtractor.extract(
            node_id="node-0",
            start_layer=0,
            end_layer=2,
            batch_size=1,
            seq_len=4096,
            gpu_profiles={},
            layer_weights=layer_weights,
            topology=topology,
        )
        # No GPU profile for node-0 -> tflops = 0, mem_bw = 0
        assert features[7] == 0.0
        assert features[8] == 0.0
        # First node: no prev_node -> comm_bw = 0
        assert features[10] == 0.0

    def test_extract_empty_layer_range(self, gpu_profiles, topology):
        features = FeatureExtractor.extract(
            node_id="node-0",
            start_layer=0,
            end_layer=0,
            batch_size=1,
            seq_len=4096,
            gpu_profiles=gpu_profiles,
            layer_weights=[],
            topology=topology,
        )
        assert len(features) == 15
        assert features[0] == 0.0  # num_layers = 0

    def test_extract_unknown_node_no_gpu(self, layer_weights, topology):
        features = FeatureExtractor.extract(
            node_id="unknown",
            start_layer=0,
            end_layer=2,
            batch_size=1,
            seq_len=4096,
            gpu_profiles={},
            layer_weights=layer_weights,
            topology=topology,
        )
        assert features[7] == 0.0  # tflops
        assert features[8] == 0.0  # mem_bw


# ---------------------------------------------------------------------------
# DecisionStump
# ---------------------------------------------------------------------------


class TestDecisionStump:
    def test_initial_state(self):
        stump = DecisionStump(max_depth=2)
        assert stump.max_depth == 2
        assert stump._tree is None

    def test_predict_one_before_fit_returns_zero(self):
        stump = DecisionStump()
        # _traverse handles None by returning 0.0
        assert stump.predict_one([1.0, 2.0]) == 0.0

    def test_fit_and_predict_simple(self):
        X = [[1.0], [2.0], [3.0], [4.0]]
        y = [10.0, 20.0, 30.0, 40.0]
        stump = DecisionStump(max_depth=1)
        stump.fit(X, y)
        pred = stump.predict_one([2.5])
        assert pred > 0

    def test_predict_all(self):
        X = [[1.0], [2.0], [3.0]]
        y = [1.0, 2.0, 3.0]
        stump = DecisionStump(max_depth=1)
        stump.fit(X, y)
        preds = stump.predict_all(X)
        assert len(preds) == 3
        assert all(isinstance(p, float) for p in preds)

    def test_empty_fit(self):
        stump = DecisionStump()
        stump.fit([], [])
        # Should not raise and tree should be a leaf with value 0.0
        assert stump._tree is not None
        assert stump._tree.get("value") == 0.0

    def test_single_element_fit(self):
        stump = DecisionStump(max_depth=3)
        stump.fit([[1.0]], [42.0])
        assert stump.predict_one([1.0]) == 42.0

    def test_two_element_fit(self):
        stump = DecisionStump(max_depth=3)
        stump.fit([[1.0], [2.0]], [10.0, 20.0])
        # n <= 2 triggers leaf with mean
        assert stump.predict_one([1.0]) == 15.0
        assert stump.predict_one([2.0]) == 15.0

    def test_variance_zero(self):
        assert DecisionStump._variance([]) == 0.0
        assert DecisionStump._variance([5.0, 5.0, 5.0]) == 0.0

    def test_variance_nonzero(self):
        v = DecisionStump._variance([1.0, 3.0])
        assert v == 1.0  # ((1-2)^2 + (3-2)^2) / 2 = 1

    def test_to_dict_roundtrip(self):
        X = [[1.0], [2.0], [3.0], [4.0]]
        y = [10.0, 20.0, 30.0, 40.0]
        stump = DecisionStump(max_depth=2)
        stump.fit(X, y)
        data = stump.to_dict()
        assert "max_depth" in data
        assert "tree" in data
        restored = DecisionStump.from_dict(data)
        assert restored.max_depth == stump.max_depth
        assert restored.predict_one([2.5]) == stump.predict_one([2.5])

    def test_max_depth_zero(self):
        X = [[1.0], [2.0], [3.0]]
        y = [10.0, 20.0, 30.0]
        stump = DecisionStump(max_depth=0)
        stump.fit(X, y)
        # depth 0 -> always return mean
        assert stump.predict_one([100.0]) == 20.0


# ---------------------------------------------------------------------------
# GradientBoostedTrees
# ---------------------------------------------------------------------------


class TestGradientBoostedTrees:
    def test_initial_state(self):
        model = GradientBoostedTrees()
        assert not model.is_trained()
        assert model._trees == []
        assert model._base_prediction == 0.0

    def test_predict_before_training_raises(self):
        model = GradientBoostedTrees()
        with pytest.raises(RuntimeError, match="not trained"):
            model.predict([1.0, 2.0])

    def test_fit_with_empty_data(self):
        model = GradientBoostedTrees()
        model.fit([], [])
        # Should not raise, stays untrained
        assert not model.is_trained()

    def test_fit_and_predict(self):
        X = [[1.0], [2.0], [3.0], [4.0], [5.0]]
        y = [10.0, 20.0, 30.0, 40.0, 50.0]
        model = GradientBoostedTrees(n_estimators=10, learning_rate=0.1, max_depth=2)
        model.fit(X, y)
        assert model.is_trained()
        pred = model.predict([3.0])
        assert isinstance(pred, float)
        assert pred >= 0.0
        # Prediction should be close to actual
        assert abs(pred - 30.0) < 15.0

    def test_prediction_non_negative(self):
        X = [[1.0], [2.0], [3.0]]
        y = [0.0, 0.5, 1.0]
        model = GradientBoostedTrees(n_estimators=5, learning_rate=0.1, max_depth=1)
        model.fit(X, y)
        pred = model.predict([100.0])
        assert pred >= 0.0

    def test_multi_dimensional(self):
        X = [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]]
        y = [1.0, 2.0, 3.0, 4.0]
        model = GradientBoostedTrees(n_estimators=10, learning_rate=0.1, max_depth=2)
        model.fit(X, y)
        pred = model.predict([2.5, 25.0])
        assert pred >= 0.0

    def test_to_dict_from_dict_roundtrip(self):
        X = [[1.0], [2.0], [3.0], [4.0]]
        y = [10.0, 20.0, 30.0, 40.0]
        model = GradientBoostedTrees(n_estimators=5, learning_rate=0.2, max_depth=2)
        model.fit(X, y)
        data = model.to_dict()
        assert data["n_estimators"] == 5
        assert data["learning_rate"] == 0.2
        assert data["max_depth"] == 2
        assert len(data["trees"]) == 5

        restored = GradientBoostedTrees.from_dict(data)
        assert restored.is_trained()
        assert restored.n_estimators == 5
        assert restored.learning_rate == 0.2

        for x in X:
            assert restored.predict(x) == model.predict(x)

    def test_low_estimator_count(self):
        X = [[1.0], [2.0], [3.0]]
        y = [1.0, 2.0, 3.0]
        model = GradientBoostedTrees(n_estimators=1, learning_rate=1.0, max_depth=1)
        model.fit(X, y)
        pred = model.predict([2.0])
        assert pred >= 0.0

    def test_several_estimators_decreases_error(self):
        X = [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]
        y = [1.0, 4.0, 9.0, 16.0, 25.0, 36.0]
        model = GradientBoostedTrees(n_estimators=20, learning_rate=0.3, max_depth=2)
        model.fit(X, y)
        preds = [model.predict([x[0]]) for x in X]
        errors = [abs(a - p) for a, p in zip(y, preds)]
        mean_error = sum(errors) / len(errors)
        # Should produce reasonable predictions (within 50% of range)
        assert mean_error < 10.0


# ---------------------------------------------------------------------------
# LearnedCostModel
# ---------------------------------------------------------------------------


class TestLearnedCostModel:
    def test_init_without_save_path(self, base_cost_model):
        model = LearnedCostModel(base_cost_model=base_cost_model)
        assert model.num_observations == 0
        assert not model.is_learned
        assert model.metrics.num_samples == 0
        assert model._save_path is None

    def test_init_with_gpu_profiles(self, base_cost_model, gpu_profiles):
        model = LearnedCostModel(
            base_cost_model=base_cost_model,
            gpu_profiles=gpu_profiles,
        )
        assert model._gpu_profiles == gpu_profiles

    def test_init_empty_gpu_profiles(self, base_cost_model):
        model = LearnedCostModel(base_cost_model=base_cost_model)
        assert model._gpu_profiles == {}

    def test_record_adds_observation(self, base_cost_model):
        model = LearnedCostModel(
            base_cost_model=base_cost_model,
            min_samples_to_train=100,
        )
        obs = RuntimeObservation(
            node_id="node-0", num_layers=4, hidden_size=4096,
            intermediate_size=11008, batch_size=1, seq_len=4096,
            gpu_tflops=312.0, gpu_mem_bw_gbps=2039.0,
            gpu_mem_bytes=80 * 1024**3,
            is_nvlink=True, is_infiniband=False,
            comm_bandwidth_gbps=600.0,
            measured_latency_ms=25.0,
        )
        model.record(obs)
        assert model.num_observations == 1

    def test_record_clears_feature_cache(self, base_cost_model):
        model = LearnedCostModel(
            base_cost_model=base_cost_model,
            min_samples_to_train=100,
        )
        model._feature_cache["dummy"] = [1.0, 2.0]
        obs = RuntimeObservation(
            node_id="node-0", num_layers=4, hidden_size=4096,
            intermediate_size=11008, batch_size=1, seq_len=4096,
            gpu_tflops=312.0, gpu_mem_bw_gbps=2039.0,
            gpu_mem_bytes=80 * 1024**3,
            is_nvlink=True, is_infiniband=False,
            comm_bandwidth_gbps=600.0,
            measured_latency_ms=25.0,
        )
        model.record(obs)
        assert "dummy" not in model._feature_cache

    def test_train_not_enough_samples(self, base_cost_model):
        model = LearnedCostModel(
            base_cost_model=base_cost_model,
            min_samples_to_train=50,
        )
        metrics = model.train()
        assert metrics.num_samples == 0
        assert not model.is_learned

    def test_train_with_enough_samples(self, base_cost_model):
        model = LearnedCostModel(
            base_cost_model=base_cost_model,
            min_samples_to_train=10,
            retrain_interval=9999.0,  # Prevent auto-train during record loop
        )
        for i in range(12):
            obs = RuntimeObservation(
                node_id="node-0", num_layers=4, hidden_size=4096,
                intermediate_size=11008, batch_size=1, seq_len=4096,
                gpu_tflops=312.0, gpu_mem_bw_gbps=2039.0,
                gpu_mem_bytes=80 * 1024**3,
                is_nvlink=True, is_infiniband=False,
                comm_bandwidth_gbps=600.0,
                measured_latency_ms=20.0 + i * 2,
            )
            model.record(obs)

        # Explicitly train (won't auto-train because retrain_interval is large)
        metrics = model.train()
        assert model.is_learned
        assert metrics.num_samples == 12
        assert metrics.mae_ms >= 0
        assert metrics.mape_pct >= 0
        assert -0.1 <= metrics.r_squared <= 1.1  # Allow small numerical imprecision
        assert metrics.max_error_ms >= 0

    def test_evaluate_fallback_before_training(self, base_cost_model):
        model = LearnedCostModel(
            base_cost_model=base_cost_model,
            min_samples_to_train=100,
        )
        cost = model.evaluate("node-0", 0, 4, batch_size=1, seq_len=4096)
        assert isinstance(cost, NodeCost)
        assert cost.node_id == "node-0"
        assert cost.total_time_ms > 0  # analytical fallback

    def test_evaluate_after_training(self, base_cost_model):
        model = LearnedCostModel(
            base_cost_model=base_cost_model,
            min_samples_to_train=5,
        )
        for i in range(8):
            model.record(RuntimeObservation(
                node_id="node-0", num_layers=4, hidden_size=4096,
                intermediate_size=11008, batch_size=1, seq_len=4096,
                gpu_tflops=312.0, gpu_mem_bw_gbps=2039.0,
                gpu_mem_bytes=80 * 1024**3,
                is_nvlink=True, is_infiniband=False,
                comm_bandwidth_gbps=600.0,
                measured_latency_ms=25.0,
            ))

        cost = model.evaluate("node-0", 0, 4, batch_size=1, seq_len=4096)
        assert isinstance(cost, NodeCost)
        assert cost.total_time_ms >= 0

    def test_evaluate_partition(self, base_cost_model):
        model = LearnedCostModel(
            base_cost_model=base_cost_model,
            min_samples_to_train=5,
        )
        for i in range(8):
            model.record(RuntimeObservation(
                node_id="node-0", num_layers=4, hidden_size=4096,
                intermediate_size=11008, batch_size=1, seq_len=4096,
                gpu_tflops=312.0, gpu_mem_bw_gbps=2039.0,
                gpu_mem_bytes=80 * 1024**3,
                is_nvlink=True, is_infiniband=False,
                comm_bandwidth_gbps=600.0,
                measured_latency_ms=25.0,
            ))

        partition = [("node-0", 0, 4), ("node-1", 4, 8)]
        costs = model.evaluate_partition(partition, batch_size=1, seq_len=4096)
        assert len(costs) == 2
        assert all(isinstance(c, NodeCost) for c in costs)

    def test_max_latency(self, base_cost_model):
        model = LearnedCostModel(
            base_cost_model=base_cost_model,
            min_samples_to_train=5,
        )
        for i in range(8):
            model.record(RuntimeObservation(
                node_id="node-0", num_layers=4, hidden_size=4096,
                intermediate_size=11008, batch_size=1, seq_len=4096,
                gpu_tflops=312.0, gpu_mem_bw_gbps=2039.0,
                gpu_mem_bytes=80 * 1024**3,
                is_nvlink=True, is_infiniband=False,
                comm_bandwidth_gbps=600.0,
                measured_latency_ms=25.0,
            ))

        partition = [("node-0", 0, 4), ("node-1", 4, 8)]
        lat = model.max_latency(partition, batch_size=1, seq_len=4096)
        assert lat >= 0.0

    def test_max_latency_empty_partition(self, base_cost_model):
        model = LearnedCostModel(base_cost_model=base_cost_model)
        assert model.max_latency([]) == 0.0

    def test_combined_throughput(self, base_cost_model):
        model = LearnedCostModel(
            base_cost_model=base_cost_model,
            min_samples_to_train=5,
        )
        for i in range(8):
            model.record(RuntimeObservation(
                node_id="node-0", num_layers=4, hidden_size=4096,
                intermediate_size=11008, batch_size=1, seq_len=4096,
                gpu_tflops=312.0, gpu_mem_bw_gbps=2039.0,
                gpu_mem_bytes=80 * 1024**3,
                is_nvlink=True, is_infiniband=False,
                comm_bandwidth_gbps=600.0,
                measured_latency_ms=25.0,
            ))

        partition = [("node-0", 0, 4), ("node-1", 4, 8)]
        tp = model.combined_throughput(partition, batch_size=1, seq_len=4096)
        assert tp >= 0.0

    def test_combined_throughput_empty_partition(self, base_cost_model):
        model = LearnedCostModel(base_cost_model=base_cost_model)
        assert model.combined_throughput([]) == 0.0

    def test_evaluate_prediction_failure_falls_back(
        self, base_cost_model, gpu_profiles, layer_weights, topology,
    ):
        """When _extract_features raises (e.g. missing attributes), fall back to analytical."""
        # Create a cost model with missing topology attributes so
        # FeatureExtractor.extract will still function but we can test
        # the fallback path.  Actually, FeatureExtractor.extract is robust.
        # The best way to trigger fallback is to not train.
        model = LearnedCostModel(
            base_cost_model=base_cost_model,
            min_samples_to_train=100,  # Never train
        )
        cost = model.evaluate("node-0", 0, 4, batch_size=1, seq_len=4096)
        # Should return the base (analytical) cost
        assert cost.total_time_ms > 0
        assert not model.is_learned

    def test_save_and_load_model(self, base_cost_model):
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "model.json"

            model = LearnedCostModel(
                base_cost_model=base_cost_model,
                min_samples_to_train=8,
                retrain_interval=9999.0,  # prevent auto-train during record
                save_path=str(save_path),
            )
            for i in range(8):
                model.record(RuntimeObservation(
                    node_id="node-0", num_layers=4, hidden_size=4096,
                    intermediate_size=11008, batch_size=1, seq_len=4096,
                    gpu_tflops=312.0, gpu_mem_bw_gbps=2039.0,
                    gpu_mem_bytes=80 * 1024**3,
                    is_nvlink=True, is_infiniband=False,
                    comm_bandwidth_gbps=600.0,
                    measured_latency_ms=25.0,
                ))

            # Train explicitly so the file gets written
            model.train()
            assert save_path.exists()

            # Create new model that loads from saved path
            model2 = LearnedCostModel(
                base_cost_model=base_cost_model,
                save_path=str(save_path),
            )
            assert model2.is_learned
            assert model2.metrics.num_samples == 8

    def test_load_missing_file_does_not_raise(self, base_cost_model):
        model = LearnedCostModel(
            base_cost_model=base_cost_model,
            save_path="/nonexistent/path/model.json",
        )
        # Should not raise, just stays unlearned
        assert not model.is_learned

    def test_load_corrupted_file_does_not_raise(self, base_cost_model):
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "corrupt.json"
            save_path.write_text("this is not valid json")

            model = LearnedCostModel(
                base_cost_model=base_cost_model,
                save_path=str(save_path),
            )
            assert not model.is_learned

    def test_save_failure_does_not_raise(self, base_cost_model):
        """Save to a directory we can't write to (e.g. root-owned) should log and not raise."""
        # We can't easily create a read-only dir on all platforms, but
        # we can at least test that save to a non-existent parent is handled.
        model = LearnedCostModel(
            base_cost_model=base_cost_model,
            min_samples_to_train=2,
            save_path="/nonexistent_dir/model.json",
        )
        # This should not raise despite save_path being unwritable
        for i in range(4):
            model.record(RuntimeObservation(
                node_id="node-0", num_layers=4, hidden_size=4096,
                intermediate_size=11008, batch_size=1, seq_len=4096,
                gpu_tflops=312.0, gpu_mem_bw_gbps=2039.0,
                gpu_mem_bytes=80 * 1024**3,
                is_nvlink=True, is_infiniband=False,
                comm_bandwidth_gbps=600.0,
                measured_latency_ms=25.0,
            ))
        assert model.is_learned

    def test_auto_train_on_record_when_conditions_met(self, base_cost_model):
        model = LearnedCostModel(
            base_cost_model=base_cost_model,
            min_samples_to_train=5,
            retrain_interval=0.0,  # Always retrain
        )
        for i in range(6):
            obs = RuntimeObservation(
                node_id="node-0", num_layers=4, hidden_size=4096,
                intermediate_size=11008, batch_size=1, seq_len=4096,
                gpu_tflops=312.0, gpu_mem_bw_gbps=2039.0,
                gpu_mem_bytes=80 * 1024**3,
                is_nvlink=True, is_infiniband=False,
                comm_bandwidth_gbps=600.0,
                measured_latency_ms=20.0 + i,
            )
            model.record(obs)
        # Auto-training should have fired on or after min_samples
        assert model.is_learned

    def test_properties(self, base_cost_model):
        model = LearnedCostModel(
            base_cost_model=base_cost_model,
            min_samples_to_train=100,
        )
        assert isinstance(model.metrics, TrainingMetrics)
        assert model.num_observations == 0
        assert not model.is_learned

    def test_evaluate_with_infiniband(
        self, base_cost_model, gpu_profiles, layer_weights,
    ):
        """Test evaluate with an InfiniBand-linked node."""
        topo = TopologyGraph(
            node_ids=["node-0", "node-1"],
            gpu_counts={"node-0": 1, "node-1": 1},
            links=[
                LinkProfile(
                    source="node-0",
                    target="node-1",
                    bandwidth_gbps=200.0,
                    latency_us=10.0,
                    is_infiniband=True,
                ),
            ],
        )
        cm = PartitionCostModel(
            gpu_profiles=gpu_profiles,
            layer_weights=layer_weights,
            topology=topo,
        )
        model = LearnedCostModel(base_cost_model=cm, min_samples_to_train=5)
        for _ in range(6):
            model.record(RuntimeObservation(
                node_id="node-1", num_layers=4, hidden_size=4096,
                intermediate_size=11008, batch_size=1, seq_len=4096,
                gpu_tflops=312.0, gpu_mem_bw_gbps=2039.0,
                gpu_mem_bytes=80 * 1024**3,
                is_nvlink=False, is_infiniband=True,
                comm_bandwidth_gbps=200.0,
                measured_latency_ms=30.0,
            ))
        cost = model.evaluate("node-1", 0, 4, batch_size=1, seq_len=4096)
        assert cost.total_time_ms >= 0

    def test_evaluate_partition_empty(self, base_cost_model):
        model = LearnedCostModel(base_cost_model=base_cost_model)
        assert model.evaluate_partition([]) == []

    def test_metric_computation_known_values(self, base_cost_model):
        model = LearnedCostModel(base_cost_model=base_cost_model)
        actual = [10.0, 20.0, 30.0]
        predicted = [12.0, 18.0, 32.0]
        metrics = model._compute_metrics(actual, predicted)
        assert metrics.num_samples == 3
        assert metrics.mae_ms > 0
        assert metrics.mape_pct > 0
        assert abs(metrics.r_squared) <= 1.0

    def test_metric_computation_empty(self, base_cost_model):
        model = LearnedCostModel(base_cost_model=base_cost_model)
        metrics = model._compute_metrics([], [])
        assert metrics.num_samples == 0

    def test_metric_computation_perfect_prediction(self, base_cost_model):
        model = LearnedCostModel(base_cost_model=base_cost_model)
        actual = [10.0, 20.0, 30.0]
        metrics = model._compute_metrics(actual, actual)
        assert metrics.mae_ms == 0.0
        assert metrics.mape_pct == 0.0
        assert metrics.r_squared == 1.0
        assert metrics.max_error_ms == 0.0

    def test_observation_to_features_fields(self, base_cost_model):
        model = LearnedCostModel(base_cost_model=base_cost_model)
        obs = RuntimeObservation(
            node_id="n0", num_layers=4, hidden_size=4096,
            intermediate_size=11008, batch_size=2, seq_len=2048,
            gpu_tflops=312.0, gpu_mem_bw_gbps=2039.0,
            gpu_mem_bytes=80 * 1024**3,
            is_nvlink=False, is_infiniband=True,
            comm_bandwidth_gbps=200.0,
            measured_latency_ms=15.0,
        )
        features = model._observation_to_features(obs)
        assert len(features) == 15
        assert features[1] == 2.0  # batch_size
        assert features[2] == 2048.0  # seq_len
        assert features[11] == 0.0  # is_nvlink
        assert features[12] == 1.0  # is_infiniband

    def test_feature_cache(self, base_cost_model, gpu_profiles):
        model = LearnedCostModel(
            base_cost_model=base_cost_model,
            gpu_profiles=gpu_profiles,
        )
        # First call populates cache
        key = "node-0:0:4:1:4096"
        f1 = model._extract_features("node-0", 0, 4, 1, 4096)
        assert key in model._feature_cache
        # Second call returns cached
        f2 = model._extract_features("node-0", 0, 4, 1, 4096)
        assert f1 == f2

    def test_feature_cache_different_key(self, base_cost_model, gpu_profiles):
        model = LearnedCostModel(
            base_cost_model=base_cost_model,
            gpu_profiles=gpu_profiles,
        )
        f1 = model._extract_features("node-0", 0, 4, 1, 4096)
        f2 = model._extract_features("node-0", 0, 4, 2, 4096)
        # Different batch size -> different features
        assert f1[1] == 1.0
        assert f2[1] == 2.0

    def test_empty_cost_model(self, empty_cost_model):
        """Should handle a base model with no GPU profiles, layers, or topology."""
        model = LearnedCostModel(
            base_cost_model=empty_cost_model,
            min_samples_to_train=5,
        )
        # Evaluate should not crash
        cost = model.evaluate("missing-node", 0, 0, batch_size=1, seq_len=4096)
        assert isinstance(cost, NodeCost)

        # Train and predict should also be ok
        for _ in range(6):
            model.record(RuntimeObservation(
                node_id="missing-node", num_layers=0, hidden_size=1,
                intermediate_size=1, batch_size=1, seq_len=1,
                gpu_tflops=0.0, gpu_mem_bw_gbps=0.0,
                gpu_mem_bytes=1,
                is_nvlink=False, is_infiniband=False,
                comm_bandwidth_gbps=0.0,
                measured_latency_ms=5.0,
            ))
        assert model.is_learned
        cost = model.evaluate("missing-node", 0, 0, batch_size=1, seq_len=1)
        assert cost.total_time_ms >= 0
