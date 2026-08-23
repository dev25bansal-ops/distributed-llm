"""Tests for the competitive advantage features.

Tests: ParetoPartitionOptimizer, LearnedCostModel,
QuantAwarePartitionSolver, AdaptiveRepartitioner,
CloudArbitrageEngine, PartitionBenchmarkSuite.
"""

from __future__ import annotations

import math
import time
from unittest.mock import MagicMock, patch

import pytest

from distllm.dist.partition.cost_model import PartitionCostModel
from distllm.dist.partition.profiles import GPUProfile, GPUProfiler, LayerWeights
from distllm.dist.partition.topology import LinkProfile, TopologyGraph

# Pareto optimizer
from distllm.dist.partition.pareto_optimizer import (
    ObjectiveVector,
    ParetoFrontier,
    ParetoPartitionOptimizer,
    ParetoSolution,
)

# Learned cost model
from distllm.dist.partition.learned_cost import (
    DecisionStump,
    FeatureExtractor,
    GradientBoostedTrees,
    LearnedCostModel,
    RuntimeObservation,
    TrainingMetrics,
)

# Quantization-aware partitioning
from distllm.dist.partition.quant_partition import (
    QuantAwarePartitionSolver,
    QuantAwareSolution,
)

# Adaptive re-partitioning
from distllm.dist.partition.adaptive import (
    AdaptiveRepartitioner,
    LatencySample,
    RepartitionTrigger,
    StragglerDetector,
    StragglerReport,
)

# Cloud arbitrage
from distllm.dist.partition.cloud_arbitrage import (
    ArbitragePlan,
    CloudArbitrageEngine,
    CloudNode,
    CloudProvider,
    InstanceType,
    PricingTier,
)

# Benchmark suite
from distllm.dist.partition.benchmark_suite import (
    BenchmarkResult,
    BenchmarkSuiteResult,
    PartitionBenchmarkSuite,
)


# ── Shared fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def gpu_profiles():
    return {
        "gpu-0": GPUProfile(gpu_id=0, name="H100", total_memory_bytes=80 * 1024**3, compute_tflops=989.0, memory_bandwidth_gbps=3350.0),
        "gpu-1": GPUProfile(gpu_id=1, name="A100", total_memory_bytes=80 * 1024**3, compute_tflops=312.0, memory_bandwidth_gbps=2039.0),
        "gpu-2": GPUProfile(gpu_id=2, name="L4", total_memory_bytes=24 * 1024**3, compute_tflops=121.0, memory_bandwidth_gbps=300.0),
    }


@pytest.fixture
def layer_weights():
    profiler = GPUProfiler()
    return profiler.estimate_layer_weights(
        hidden_size=4096, intermediate_size=11008,
        num_layers=32, num_heads=32, head_dim=128, vocab_size=32000,
    )


@pytest.fixture
def topology():
    return TopologyGraph(
        node_ids=["gpu-0", "gpu-1", "gpu-2"],
        gpu_counts={"gpu-0": 1, "gpu-1": 1, "gpu-2": 1},
        links=[
            LinkProfile(source="gpu-0", target="gpu-1", bandwidth_gbps=25.0, latency_us=500.0),
            LinkProfile(source="gpu-1", target="gpu-2", bandwidth_gbps=25.0, latency_us=500.0),
            LinkProfile(source="gpu-0", target="gpu-2", bandwidth_gbps=25.0, latency_us=500.0),
        ],
    )


@pytest.fixture
def cost_model(gpu_profiles, layer_weights, topology):
    return PartitionCostModel(gpu_profiles, layer_weights, topology)


# ── Pareto Optimizer ─────────────────────────────────────────────────────────


class TestObjectiveVector:
    def test_defaults(self):
        v = ObjectiveVector()
        assert v.latency_ms == 0.0
        assert v.throughput_tok_s == 0.0
        assert v.cost_per_hour == 0.0

    def test_to_dict(self):
        v = ObjectiveVector(latency_ms=10.0, throughput_tok_s=100.0, cost_per_hour=5.0)
        d = v.to_dict()
        assert d["latency_ms"] == 10.0
        assert d["cost_per_hour"] == 5.0

    def test_get(self):
        v = ObjectiveVector(latency_ms=10.0, throughput_tok_s=100.0)
        assert v.get("latency") == 10.0
        assert v.get("throughput") == 100.0
        assert v.get("unknown") == 0.0

    def test_dominance(self):
        a = ObjectiveVector(latency_ms=10.0, memory_utilization=0.5, cost_per_hour=1.0)
        b = ObjectiveVector(latency_ms=20.0, memory_utilization=0.8, cost_per_hour=2.0)
        assert a.dominates(b)
        assert not b.dominates(a)

    def test_no_dominance(self):
        a = ObjectiveVector(latency_ms=10.0, throughput_tok_s=50.0)
        b = ObjectiveVector(latency_ms=20.0, throughput_tok_s=100.0)
        assert not a.dominates(b)
        assert not b.dominates(a)


class TestParetoFrontier:
    def test_empty(self):
        f = ParetoFrontier()
        assert f.size == 0
        assert f.best_by("latency") is None

    def test_best_by(self):
        s1 = ParetoSolution(
            points=[], vector=ObjectiveVector(latency_ms=10.0, throughput_tok_s=100.0),
        )
        s2 = ParetoSolution(
            points=[], vector=ObjectiveVector(latency_ms=20.0, throughput_tok_s=200.0),
        )
        f = ParetoFrontier(solutions=[s1, s2])
        assert f.best_by("latency") is s1
        assert f.best_by("throughput") is s2

    def test_weighted_select(self):
        s1 = ParetoSolution(
            points=[], vector=ObjectiveVector(latency_ms=10.0, cost_per_hour=10.0),
        )
        s2 = ParetoSolution(
            points=[], vector=ObjectiveVector(latency_ms=20.0, cost_per_hour=2.0),
        )
        f = ParetoFrontier(solutions=[s1, s2])

        best = f.weighted_select({"latency": 0.9, "cost": 0.1})
        assert best is s1

        best = f.weighted_select({"latency": 0.1, "cost": 0.9})
        assert best is s2


class TestParetoPartitionOptimizer:
    def test_no_nodes(self, cost_model):
        opt = ParetoPartitionOptimizer(cost_model, [])
        frontier = opt.solve(32)
        assert frontier.size == 0

    def test_single_node(self, cost_model, layer_weights):
        opt = ParetoPartitionOptimizer(cost_model, ["gpu-0"])
        frontier = opt.solve(len(layer_weights))
        assert frontier.size >= 1

    def test_two_nodes(self, cost_model, layer_weights):
        opt = ParetoPartitionOptimizer(cost_model, ["gpu-0", "gpu-1"])
        frontier = opt.solve(len(layer_weights))
        assert frontier.size >= 1
        best = frontier.best_by("latency")
        assert best is not None
        assert best.vector.latency_ms > 0

    def test_solve_and_select(self, cost_model, layer_weights):
        opt = ParetoPartitionOptimizer(cost_model, ["gpu-0", "gpu-1"])
        solution = opt.solve_and_select(
            len(layer_weights),
            weights={"latency": 0.7, "memory": 0.3},
        )
        assert solution.num_nodes >= 1
        assert solution.max_node_time_ms > 0

    def test_frontier_summary(self, cost_model, layer_weights):
        opt = ParetoPartitionOptimizer(cost_model, ["gpu-0", "gpu-1"])
        frontier = opt.solve(len(layer_weights))
        summary = frontier.summary()
        assert "Pareto frontier" in summary


# ── Learned Cost Model ───────────────────────────────────────────────────────


class TestGradientBoostedTrees:
    def test_fit_predict(self):
        X = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]
        y = [10.0, 20.0, 30.0, 40.0]
        model = GradientBoostedTrees(n_estimators=10, learning_rate=0.1)
        model.fit(X, y)
        assert model.is_trained()
        pred = model.predict([4.0, 5.0])
        assert pred > 0

    def test_predict_before_fit(self):
        model = GradientBoostedTrees()
        with pytest.raises(RuntimeError):
            model.predict([1.0, 2.0])

    def test_serialization(self):
        X = [[1.0], [2.0], [3.0], [4.0]]
        y = [10.0, 20.0, 30.0, 40.0]
        model = GradientBoostedTrees(n_estimators=5)
        model.fit(X, y)
        data = model.to_dict()
        restored = GradientBoostedTrees.from_dict(data)
        assert restored.is_trained()
        assert abs(model.predict([2.5]) - restored.predict([2.5])) < 0.01


class TestDecisionStump:
    def test_fit_predict(self):
        X = [[1.0], [2.0], [3.0], [4.0], [5.0]]
        y = [1.0, 2.0, 3.0, 4.0, 5.0]
        stump = DecisionStump(max_depth=2)
        stump.fit(X, y)
        pred = stump.predict_one([3.5])
        assert pred > 0

    def test_variance(self):
        assert DecisionStump._variance([1.0, 1.0, 1.0]) == 0.0
        assert DecisionStump._variance([]) == 0.0
        assert DecisionStump._variance([1.0, 3.0]) == 1.0


class TestLearnedCostModel:
    def test_fallback_to_base(self, cost_model, layer_weights, gpu_profiles):
        learned = LearnedCostModel(
            base_cost_model=cost_model,
            gpu_profiles=gpu_profiles,
            min_samples_to_train=50,
        )
        assert not learned.is_learned
        result = learned.evaluate("gpu-0", 0, len(layer_weights))
        assert result.total_time_ms > 0

    def test_record_and_metrics(self, cost_model, layer_weights, gpu_profiles):
        learned = LearnedCostModel(
            base_cost_model=cost_model,
            gpu_profiles=gpu_profiles,
            min_samples_to_train=5,
        )
        for i in range(10):
            learned.record(RuntimeObservation(
                node_id="gpu-0",
                num_layers=32,
                hidden_size=4096,
                intermediate_size=11008,
                batch_size=1,
                seq_len=4096,
                gpu_tflops=989.0,
                gpu_mem_bw_gbps=3350.0,
                gpu_mem_bytes=80 * 1024**3,
                is_nvlink=False,
                is_infiniband=False,
                comm_bandwidth_gbps=25.0,
                measured_latency_ms=50.0 + i,
            ))
        assert learned.num_observations == 10

    def test_insufficient_samples(self, cost_model, gpu_profiles):
        learned = LearnedCostModel(
            base_cost_model=cost_model,
            gpu_profiles=gpu_profiles,
            min_samples_to_train=100,
        )
        metrics = learned.train()
        assert metrics.num_samples == 0


class TestFeatureExtractor:
    def test_extract(self, gpu_profiles, layer_weights, topology):
        features = FeatureExtractor.extract(
            "gpu-0", 0, 10, 1, 4096,
            gpu_profiles, layer_weights, topology,
        )
        assert len(features) == 15
        assert all(isinstance(f, float) for f in features)

    def test_feature_names(self):
        names = FeatureExtractor.feature_names()
        assert len(names) == 15
        assert "gpu_tflops" in names


# ── Quantization-Aware Partitioning ──────────────────────────────────────────


class TestQuantAwarePartitionSolver:
    def test_no_nodes(self, cost_model):
        solver = QuantAwarePartitionSolver(cost_model, [])
        solution = solver.solve(32)
        assert solution.num_nodes == 0

    def test_single_node(self, cost_model, layer_weights):
        solver = QuantAwarePartitionSolver(
            cost_model, ["gpu-0"],
            device_types={"gpu-0": "cuda"},
        )
        solution = solver.solve(len(layer_weights))
        assert solution.num_nodes == 1
        assert solution.max_node_time_ms > 0
        assert len(solution.quant_methods_used()) >= 1

    def test_two_nodes(self, cost_model, layer_weights):
        solver = QuantAwarePartitionSolver(
            cost_model, ["gpu-0", "gpu-1"],
            device_types={"gpu-0": "cuda", "gpu-1": "cuda"},
        )
        solution = solver.solve(len(layer_weights))
        assert solution.num_nodes >= 1
        assert solution.max_node_time_ms > 0

    def test_quality_constraint(self, cost_model, layer_weights):
        solver = QuantAwarePartitionSolver(
            cost_model, ["gpu-0"],
            device_types={"gpu-0": "cuda"},
            max_quality_loss=0.0,
        )
        solution = solver.solve(len(layer_weights))
        assert solution.avg_quality_loss == 0.0

    def test_to_partition_solution(self, cost_model, layer_weights):
        solver = QuantAwarePartitionSolver(cost_model, ["gpu-0"])
        solution = solver.solve(len(layer_weights))
        ps = solution.to_partition_solution()
        assert ps.num_nodes >= 1

    def test_summary(self, cost_model, layer_weights):
        solver = QuantAwarePartitionSolver(cost_model, ["gpu-0"])
        solution = solver.solve(len(layer_weights))
        summary = solution.summary()
        assert "QuantAwareSolution" in summary

    def test_heterogeneous_nodes(self, cost_model, layer_weights):
        solver = QuantAwarePartitionSolver(
            cost_model, ["gpu-0", "gpu-2"],
            device_types={"gpu-0": "cuda", "gpu-2": "cuda"},
            allow_oom=True,
        )
        solution = solver.solve(len(layer_weights))
        assert solution.num_nodes >= 1


# ── Adaptive Re-Partitioning ─────────────────────────────────────────────────


class TestStragglerDetector:
    def test_no_straggler(self):
        detector = StragglerDetector(abs_threshold=1.5)
        detector.set_expected({"node-0": 100.0, "node-1": 100.0})

        for _ in range(5):
            report = detector.record(LatencySample(node_id="node-0", latency_ms=100.0))
        assert report is None

    def test_detects_straggler(self):
        detector = StragglerDetector(abs_threshold=1.5)
        detector.set_expected({"node-0": 100.0})

        # The detector compares the *sliding-window average* against the
        # expected latency, so the straggling latency must be sustained
        # (a single spike is diluted by prior healthy samples).
        for _ in range(5):
            detector.record(LatencySample(node_id="node-0", latency_ms=200.0))

        report = detector.record(LatencySample(node_id="node-0", latency_ms=200.0))
        assert report is not None
        assert report.node_id == "node-0"
        assert report.ratio > 1.5

    def test_get_stats(self):
        detector = StragglerDetector()
        for i in range(10):
            detector.record(LatencySample(node_id="node-0", latency_ms=100.0 + i))
        stats = detector.get_all_stats()
        assert "node-0" in stats
        assert stats["node-0"]["samples"] == 10


class TestAdaptiveRepartitioner:
    def test_no_repartition_needed(self, cost_model, layer_weights):
        from distllm.dist.partition.adaptive import AdaptiveConfig
        repartitioner = AdaptiveRepartitioner(
            cost_model=cost_model,
            node_ids=["gpu-0", "gpu-1"],
            config=AdaptiveConfig(straggler_threshold=1.5),
        )

        from distllm.dist.partition.optimizer import PartitionOptimizer
        opt = PartitionOptimizer(cost_model, ["gpu-0", "gpu-1"])
        solution = opt.solve(len(layer_weights))
        repartitioner.set_initial_partition(solution, len(layer_weights))

        result = repartitioner.check_and_repartition({
            "gpu-0": solution.points[0].estimated_time_ms,
            "gpu-1": solution.points[1].estimated_time_ms if len(solution.points) > 1 else 50.0,
        })
        assert result is None

    def test_force_repartition(self, cost_model, layer_weights):
        from distllm.dist.partition.adaptive import AdaptiveConfig
        repartitioner = AdaptiveRepartitioner(
            cost_model=cost_model,
            node_ids=["gpu-0", "gpu-1"],
            config=AdaptiveConfig(enabled=True),
        )

        from distllm.dist.partition.optimizer import PartitionOptimizer
        opt = PartitionOptimizer(cost_model, ["gpu-0", "gpu-1"])
        solution = opt.solve(len(layer_weights))
        repartitioner.set_initial_partition(solution, len(layer_weights))

        result = repartitioner.force_repartition()
        assert result is not None
        assert result.num_nodes >= 1
        assert len(repartitioner.repartition_history) == 1


class TestLatencySample:
    def test_defaults(self):
        s = LatencySample(node_id="n0", latency_ms=50.0)
        assert s.node_id == "n0"
        assert s.latency_ms == 50.0
        assert s.timestamp > 0


# ── Cloud Arbitrage ──────────────────────────────────────────────────────────


class TestCloudArbitrageEngine:
    def test_list_instances(self):
        engine = CloudArbitrageEngine()
        instances = engine.list_instances()
        assert len(instances) > 0
        assert all("provider" in i for i in instances)

    def test_list_instances_by_provider(self):
        engine = CloudArbitrageEngine()
        aws = engine.list_instances(provider=CloudProvider.AWS)
        assert all(i["provider"] == "aws" for i in aws)

    def test_list_instances_by_memory(self):
        engine = CloudArbitrageEngine()
        filtered = engine.list_instances(min_gpu_memory_gb=80)
        assert all(i["gpu_memory_gb"] >= 80 for i in filtered)


class TestCloudNode:
    def test_creation(self):
        instance = InstanceType(
            provider=CloudProvider.AWS,
            instance_name="test",
            gpu_name="H100",
            gpu_count=1,
            gpu_memory_gb=80,
            gpu_tflops=989.0,
            gpu_mem_bw_gbps=3350.0,
            vcpus=8,
            ram_gb=64,
            pricing={PricingTier.SPOT: 1.0},
        )
        node = CloudNode(
            node_id="test-0",
            instance=instance,
            pricing_tier=PricingTier.SPOT,
            hourly_cost=1.0,
        )
        assert node.node_id == "test-0"
        assert node.hourly_cost == 1.0


class TestArbitragePlan:
    def test_summary(self):
        plan = ArbitragePlan(
            nodes=[],
            partition_solution=MagicMock(max_node_time_ms=50.0, num_nodes=2),
            total_cost_per_hour=10.0,
            cost_per_million_tokens=0.05,
            estimated_throughput_tok_s=100.0,
            meets_throughput_target=True,
            meets_budget=True,
            preemption_risk=0.05,
            recommended_checkpoint_interval_s=300.0,
        )
        summary = plan.summary()
        assert "$10.00/hr" in summary

    def test_to_dict(self):
        plan = ArbitragePlan(
            nodes=[],
            partition_solution=MagicMock(max_node_time_ms=50.0, num_nodes=2),
            total_cost_per_hour=10.0,
            cost_per_million_tokens=0.05,
            estimated_throughput_tok_s=100.0,
            meets_throughput_target=True,
            meets_budget=True,
            preemption_risk=0.05,
            recommended_checkpoint_interval_s=300.0,
        )
        d = plan.to_dict()
        assert d["total_cost_per_hour"] == 10.0


# ── Benchmark Suite ──────────────────────────────────────────────────────────


class TestPartitionBenchmarkSuite:
    def test_list_scenarios(self):
        suite = PartitionBenchmarkSuite()
        scenarios = suite.list_scenarios()
        assert len(scenarios) >= 5
        assert all("name" in s for s in scenarios)

    def test_run_single_scenario(self):
        suite = PartitionBenchmarkSuite()
        results = suite.run_scenario("7B_single_node")
        assert len(results) >= 1
        assert results[0].scenario == "7B_single_node"
        assert results[0].max_latency_ms > 0

    def test_run_two_identical(self):
        suite = PartitionBenchmarkSuite()
        results = suite.run_scenario("7B_two_identical")
        dp = next(r for r in results if r.strategy == "dp_minimax")
        assert dp.max_latency_ms > 0
        assert dp.throughput_tok_s > 0

    def test_result_summary(self):
        suite = PartitionBenchmarkSuite()
        results = suite.run_all()
        assert results.passed() > 0
        summary = results.summary()
        assert "Benchmark Suite" in summary

    def test_result_serialization(self):
        result = BenchmarkResult(
            scenario="test", strategy="dp_minimax",
            max_latency_ms=50.0, throughput_tok_s=100.0,
            num_nodes=2, total_memory_gb=10.0, solve_time_ms=5.0,
        )
        suite_result = BenchmarkSuiteResult(results=[result])
        d = suite_result.to_dict()
        assert d["passed"] == 1
        assert len(d["results"]) == 1

    def test_regression_detection(self):
        current = BenchmarkSuiteResult(results=[
            BenchmarkResult(
                scenario="test", strategy="dp_minimax",
                max_latency_ms=50.0, throughput_tok_s=100.0,
                num_nodes=2, total_memory_gb=10.0, solve_time_ms=5.0,
            ),
        ])
        baseline = {
            "results": [
                {
                    "scenario": "test", "strategy": "dp_minimax",
                    "max_latency_ms": 48.0, "throughput_tok_s": 100.0,
                },
            ],
        }
        comparison = PartitionBenchmarkSuite.compare_runs(current, baseline, tolerance_pct=5.0)
        assert comparison["stable"] is True

    def test_regression_detection_finds_regression(self):
        current = BenchmarkSuiteResult(results=[
            BenchmarkResult(
                scenario="test", strategy="dp_minimax",
                max_latency_ms=100.0, throughput_tok_s=50.0,
                num_nodes=2, total_memory_gb=10.0, solve_time_ms=5.0,
            ),
        ])
        baseline = {
            "results": [
                {
                    "scenario": "test", "strategy": "dp_minimax",
                    "max_latency_ms": 50.0, "throughput_tok_s": 100.0,
                },
            ],
        }
        comparison = PartitionBenchmarkSuite.compare_runs(current, baseline, tolerance_pct=5.0)
        assert comparison["stable"] is False
        assert len(comparison["regressions"]) == 1


# ── Config ───────────────────────────────────────────────────────────────────


class TestNewConfigs:
    def test_pareto_config(self):
        from distllm.dist.partition.config import ParetoConfig
        config = ParetoConfig()
        assert config.enabled is False
        assert config.frontier_limit == 32
        assert "latency" in config.weights

    def test_learned_cost_config(self):
        from distllm.dist.partition.config import LearnedCostConfig
        config = LearnedCostConfig()
        assert config.enabled is False
        assert config.min_samples_to_train == 50

    def test_adaptive_config(self):
        from distllm.dist.partition.config import AdaptiveConfig
        config = AdaptiveConfig()
        assert config.enabled is False
        assert config.straggler_threshold == 1.5

    def test_cloud_arbitrage_config(self):
        from distllm.dist.partition.config import CloudArbitrageConfig
        config = CloudArbitrageConfig()
        assert config.enabled is False
        assert config.prefer_spot is True
        assert "aws" in config.allowed_providers

    def test_auto_partition_config_new_fields(self):
        from distllm.dist.partition.config import AutoPartitionConfig
        config = AutoPartitionConfig()
        assert config.pareto.enabled is False
        assert config.learned_cost.enabled is False
        assert config.adaptive.enabled is False
        assert config.cloud_arbitrage.enabled is False
        assert "pareto" in config.strategy or config.strategy == "auto"
