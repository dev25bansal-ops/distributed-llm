"""Tests for dist/partition/adaptive module -- real objects, zero mocks."""
from __future__ import annotations

import math
import time

import pytest


# =============================================================================
# Test data classes
# =============================================================================


class TestRepartitionTrigger:
    """RepartitionTrigger enum values and semantics."""

    def test_enum_values(self):
        from distllm.dist.partition.adaptive import RepartitionTrigger

        assert RepartitionTrigger.NONE.value == "none"
        assert RepartitionTrigger.STRAGGLER.value == "straggler"
        assert RepartitionTrigger.OOM.value == "oom"
        assert RepartitionTrigger.THROUGHPUT_DROP.value == "throughput_drop"
        assert RepartitionTrigger.MANUAL.value == "manual"

    def test_enum_is_str_enum(self):
        from distllm.dist.partition.adaptive import RepartitionTrigger

        assert issubclass(RepartitionTrigger, str)


class TestLatencySample:
    """LatencySample dataclass construction and defaults."""

    def test_defaults(self):
        from distllm.dist.partition.adaptive import LatencySample

        sample = LatencySample(node_id="gpu-0", latency_ms=10.0)
        assert sample.node_id == "gpu-0"
        assert sample.latency_ms == 10.0
        assert sample.batch_size == 1
        assert sample.seq_len == 4096
        assert sample.timestamp > 0

    def test_custom_values(self):
        from distllm.dist.partition.adaptive import LatencySample

        ts = 12345.0
        sample = LatencySample(
            node_id="gpu-1",
            latency_ms=25.5,
            batch_size=4,
            seq_len=8192,
            timestamp=ts,
        )
        assert sample.node_id == "gpu-1"
        assert sample.latency_ms == 25.5
        assert sample.batch_size == 4
        assert sample.seq_len == 8192
        assert sample.timestamp == ts

    def test_zero_latency(self):
        from distllm.dist.partition.adaptive import LatencySample

        sample = LatencySample(node_id="gpu-0", latency_ms=0.0)
        assert sample.latency_ms == 0.0

    def test_negative_latency(self):
        from distllm.dist.partition.adaptive import LatencySample

        sample = LatencySample(node_id="gpu-0", latency_ms=-1.0)
        assert sample.latency_ms == -1.0


class TestStragglerReport:
    """StragglerReport dataclass construction."""

    def test_full_construction(self):
        from distllm.dist.partition.adaptive import (
            RepartitionTrigger,
            StragglerReport,
        )

        report = StragglerReport(
            node_id="gpu-0",
            observed_latency_ms=100.0,
            expected_latency_ms=50.0,
            ratio=2.0,
            severity=0.5,
            trigger=RepartitionTrigger.STRAGGLER,
        )
        assert report.node_id == "gpu-0"
        assert report.observed_latency_ms == 100.0
        assert report.expected_latency_ms == 50.0
        assert report.ratio == 2.0
        assert report.severity == 0.5
        assert report.trigger == RepartitionTrigger.STRAGGLER
        assert report.timestamp > 0

    def test_zero_values(self):
        from distllm.dist.partition.adaptive import (
            RepartitionTrigger,
            StragglerReport,
        )

        report = StragglerReport(
            node_id="",
            observed_latency_ms=0.0,
            expected_latency_ms=0.0,
            ratio=0.0,
            severity=0.0,
            trigger=RepartitionTrigger.MANUAL,
        )
        assert report.ratio == 0.0
        assert report.severity == 0.0


class TestRepartitionEvent:
    """RepartitionEvent dataclass construction."""

    def test_defaults(self):
        from distllm.dist.partition.adaptive import (
            RepartitionEvent,
            RepartitionTrigger,
        )

        event = RepartitionEvent(
            trigger=RepartitionTrigger.MANUAL,
            old_solution=None,
            new_solution=None,
            straggler_report=None,
        )
        assert event.trigger == RepartitionTrigger.MANUAL
        assert event.old_solution is None
        assert event.new_solution is None
        assert event.straggler_report is None
        assert event.duration_ms == 0.0
        assert event.timestamp > 0

    def test_with_solution_stubs(self):
        from distllm.dist.partition.adaptive import (
            RepartitionEvent,
            RepartitionTrigger,
        )

        event = RepartitionEvent(
            trigger=RepartitionTrigger.OOM,
            old_solution=None,
            new_solution=None,
            straggler_report=None,
            duration_ms=150.5,
        )
        assert event.duration_ms == 150.5
        assert event.trigger == RepartitionTrigger.OOM


# =============================================================================
# Helper to build a real PartitionSolution
# =============================================================================


def _make_simple_partition_solution(
    num_nodes: int = 2,
    num_layers: int = 5,
) -> tuple:
    """Build real PartitionCostModel + PartitionSolution for tests.

    Returns (cost_model, solution, node_ids, num_layers).
    """
    from distllm.dist.partition.cost_model import PartitionCostModel
    from distllm.dist.partition.optimizer import PartitionOptimizer
    from distllm.dist.partition.profiles import GPUProfile, LayerWeights
    from distllm.dist.partition.topology import LinkProfile, TopologyGraph

    node_ids = [f"node-{i}" for i in range(num_nodes)]

    # Use A100-class specs so compute times are non-zero
    gpu_profile = GPUProfile(
        gpu_id=0,
        name="A100",
        total_memory_bytes=80 * 1024**3,
        compute_tflops=312.0,
        memory_bandwidth_gbps=2039.0,
    )
    profiles = {nid: gpu_profile for nid in node_ids}

    layers = [
        LayerWeights(
            layer_id=0,
            layer_type="embed",
            weight_memory_bytes=512 * 1024**2,
            flops_per_seq=1_000_000,
            activation_memory_bytes=4096 * 2,
        ),
    ]
    for i in range(1, num_layers):
        layers.append(
            LayerWeights(
                layer_id=i,
                layer_type="transformer",
                weight_memory_bytes=2 * 1024**3,
                flops_per_seq=2_000_000_000,
                kv_cache_bytes_per_token=2048,
                activation_memory_bytes=4096 * 2,
            ),
        )

    links = []
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            links.append(
                LinkProfile(
                    source=node_ids[i],
                    target=node_ids[j],
                    bandwidth_gbps=600.0,
                    latency_us=5.0,
                    is_nvlink=True,
                ),
            )

    topology = TopologyGraph(node_ids=node_ids, links=links)
    cost_model = PartitionCostModel(
        gpu_profiles=profiles,
        layer_weights=layers,
        topology=topology,
    )
    optimizer = PartitionOptimizer(
        cost_model=cost_model,
        node_ids=node_ids,
    )
    solution = optimizer.solve(num_layers)
    return cost_model, solution, node_ids, num_layers


# =============================================================================
# Test StragglerDetector
# =============================================================================


class TestStragglerDetector:
    """StragglerDetector sliding-window detection logic."""

    def test_init_defaults(self):
        from distllm.dist.partition.adaptive import StragglerDetector

        detector = StragglerDetector()
        assert detector is not None

    def test_init_custom(self):
        from distllm.dist.partition.adaptive import StragglerDetector

        detector = StragglerDetector(
            window_size=50,
            abs_threshold=2.0,
            rel_threshold=3.0,
            trend_window=10,
            trend_slope_threshold=1.0,
        )
        assert detector is not None

    def test_record_no_samples_returns_none(self):
        from distllm.dist.partition.adaptive import (
            LatencySample,
            StragglerDetector,
        )

        detector = StragglerDetector()
        # Single sample -- not enough (needs >= 3)
        result = detector.record(
            LatencySample(node_id="gpu-0", latency_ms=10.0),
        )
        assert result is None

    def test_record_fewer_than_three_returns_none(self):
        from distllm.dist.partition.adaptive import (
            LatencySample,
            StragglerDetector,
        )

        detector = StragglerDetector()
        detector.record(LatencySample(node_id="gpu-0", latency_ms=10.0))
        detector.record(LatencySample(node_id="gpu-0", latency_ms=12.0))
        # Still only 2 samples
        result = detector.record(
            LatencySample(node_id="gpu-0", latency_ms=11.0),
        )
        # With 3 samples but no expected set and abs_threshold not reached,
        # it may return None or a report depending on ratio
        # Default abs_threshold=1.5 -- avg ~11, expected falls back to 11,
        # ratio=1.0 < 1.5, then relative check with single node no median,
        # then trend check with <20 samples (trend_window=20).
        # Result should be None.
        assert result is None

    def test_record_triggers_on_abs_threshold(self):
        from distllm.dist.partition.adaptive import (
            LatencySample,
            StragglerDetector,
        )

        detector = StragglerDetector(abs_threshold=1.5, rel_threshold=10.0)
        # Set expected to 10ms so ratio is clear
        detector.set_expected({"gpu-0": 10.0})

        # Record enough samples to pass the >= 3 check
        for _ in range(5):
            detector.record(LatencySample(node_id="gpu-0", latency_ms=10.0))

        # Now record a slow sample -- should trigger
        result = detector.record(
            LatencySample(node_id="gpu-0", latency_ms=50.0),
        )
        assert result is not None
        assert result.node_id == "gpu-0"
        assert result.trigger.value == "straggler"
        assert result.ratio >= 1.5
        assert result.severity > 0

    def test_abs_threshold_exact_boundary(self):
        from distllm.dist.partition.adaptive import (
            LatencySample,
            StragglerDetector,
        )

        detector = StragglerDetector(abs_threshold=2.0, rel_threshold=10.0)
        detector.set_expected({"gpu-0": 20.0})

        # Record several samples all at the threshold multiple
        for _ in range(5):
            detector.record(LatencySample(node_id="gpu-0", latency_ms=40.0))

        # All samples at 40ms, expected 20ms => ratio = 2.0 exactly
        result = detector.record(
            LatencySample(node_id="gpu-0", latency_ms=40.0),
        )
        assert result is not None
        assert result.ratio >= 2.0

    def test_abs_threshold_below_boundary(self):
        from distllm.dist.partition.adaptive import (
            LatencySample,
            StragglerDetector,
        )

        detector = StragglerDetector(abs_threshold=2.0, rel_threshold=10.0)
        detector.set_expected({"gpu-0": 10.0})

        # Record all samples at 15ms, expected 10ms => ratio = 1.5 < 2.0
        for _ in range(5):
            detector.record(LatencySample(node_id="gpu-0", latency_ms=15.0))

        result = detector.record(
            LatencySample(node_id="gpu-0", latency_ms=15.0),
        )
        assert result is None

    def test_rel_threshold_triggers(self):
        from distllm.dist.partition.adaptive import (
            LatencySample,
            StragglerDetector,
        )

        # Set abs very high so only relative triggers.
        # Use 3 nodes: one slow, two fast -- median will be low.
        detector = StragglerDetector(
            abs_threshold=10.0,
            rel_threshold=1.2,
        )

        for _ in range(5):
            detector.record(LatencySample(node_id="gpu-0", latency_ms=100.0))
            detector.record(LatencySample(node_id="gpu-1", latency_ms=10.0))
            detector.record(LatencySample(node_id="gpu-2", latency_ms=10.0))

        # gpu-0 avg = 100, gpu-1 median = 10, gpu-2 median = 10
        # medians = [100, 10, 10], sorted = [10, 10, 100]
        # median_latency = 10, rel_ratio = 100 / 10 = 10 >= 1.2
        result = detector.record(
            LatencySample(node_id="gpu-0", latency_ms=100.0),
        )
        assert result is not None, "Relative threshold should trigger"
        assert result.node_id == "gpu-0"
        assert result.trigger.value == "straggler"

    def test_rel_threshold_single_node_no_median(self):
        from distllm.dist.partition.adaptive import (
            LatencySample,
            StragglerDetector,
        )

        detector = StragglerDetector(abs_threshold=10.0, rel_threshold=2.0)
        detector.set_expected({"gpu-0": 10.0})

        for _ in range(5):
            detector.record(LatencySample(node_id="gpu-0", latency_ms=30.0))

        # Single node, rel check needs median list, which needs other nodes.
        # With only one node, _check_node still attempts:
        #   all_nodes = [gpu-0]
        #   medians = [sorted lat of gpu-0]
        #   median_latency = that value = 30
        #   rel_ratio = 30/30 = 1.0 < 2.0 -> no rel trigger
        # But if abs_threshold is 1.5 and expected is 10, ratio=3 >= 1.5
        # So with abs=10.0, this won't trigger anything (unless trend kicks in)
        result = detector.record(
            LatencySample(node_id="gpu-0", latency_ms=30.0),
        )
        # No trigger expected with these settings
        assert result is None

    def test_trend_detection_triggers(self):
        from distllm.dist.partition.adaptive import (
            LatencySample,
            StragglerDetector,
        )

        # Small trend window for quick triggering
        detector = StragglerDetector(
            abs_threshold=10.0,
            rel_threshold=10.0,
            trend_window=5,
            trend_slope_threshold=2.0,
        )

        # Add 5 samples with increasing latency to create a positive slope
        for i in range(5):
            result = detector.record(
                LatencySample(node_id="gpu-0", latency_ms=10.0 + i * 5.0),
            )
        # The last record should complete the trend window and detect the slope
        # slope = ~5.0 which is > 2.0
        assert result is not None
        assert result.node_id == "gpu-0"
        assert result.trigger.value == "straggler"

    def test_trend_not_enough_samples(self):
        from distllm.dist.partition.adaptive import (
            LatencySample,
            StragglerDetector,
        )

        detector = StragglerDetector(
            abs_threshold=10.0,
            rel_threshold=10.0,
            trend_window=20,
            trend_slope_threshold=0.1,
        )

        for i in range(5):
            result = detector.record(
                LatencySample(node_id="gpu-0", latency_ms=10.0 + i * 5.0),
            )
        # With trend_window=20 but only 5 samples, trend check is skipped
        assert result is None

    def test_set_expected_zero(self):
        from distllm.dist.partition.adaptive import (
            LatencySample,
            StragglerDetector,
        )

        detector = StragglerDetector(abs_threshold=1.5)
        # Setting expected to 0 for a node
        detector.set_expected({"gpu-0": 0.0})

        for _ in range(5):
            detector.record(LatencySample(node_id="gpu-0", latency_ms=10.0))

        # expected <= 0 --> return None
        result = detector.record(
            LatencySample(node_id="gpu-0", latency_ms=50.0),
        )
        assert result is None

    def test_get_all_stats_empty(self):
        from distllm.dist.partition.adaptive import StragglerDetector

        detector = StragglerDetector()
        stats = detector.get_all_stats()
        assert stats == {}

    def test_get_all_stats_with_samples(self):
        from distllm.dist.partition.adaptive import (
            LatencySample,
            StragglerDetector,
        )

        detector = StragglerDetector()
        for _ in range(10):
            detector.record(LatencySample(node_id="gpu-0", latency_ms=10.0))
            detector.record(LatencySample(node_id="gpu-1", latency_ms=20.0))

        stats = detector.get_all_stats()
        assert "gpu-0" in stats
        assert "gpu-1" in stats
        assert stats["gpu-0"]["mean_ms"] == 10.0
        assert stats["gpu-0"]["median_ms"] == 10.0
        assert stats["gpu-0"]["min_ms"] == 10.0
        assert stats["gpu-0"]["max_ms"] == 10.0
        assert stats["gpu-0"]["samples"] == 10
        assert stats["gpu-1"]["mean_ms"] == 20.0
        assert stats["gpu-1"]["samples"] == 10

    def test_get_all_stats_single_sample_p99(self):
        from distllm.dist.partition.adaptive import (
            LatencySample,
            StragglerDetector,
        )

        detector = StragglerDetector()
        detector.record(LatencySample(node_id="gpu-0", latency_ms=10.0))

        stats = detector.get_all_stats()
        assert stats["gpu-0"]["p99_ms"] == 10.0

    def test_compute_slope_less_than_two(self):
        from distllm.dist.partition.adaptive import (
            LatencySample,
            StragglerDetector,
        )

        detector = StragglerDetector()
        samples = [LatencySample(node_id="gpu-0", latency_ms=10.0)]
        slope = detector._compute_slope(samples)
        assert slope == 0.0

        slope = detector._compute_slope([])
        assert slope == 0.0

    def test_compute_slope_positive(self):
        from distllm.dist.partition.adaptive import (
            LatencySample,
            StragglerDetector,
        )

        detector = StragglerDetector()
        samples = [
            LatencySample(node_id="gpu-0", latency_ms=10.0),
            LatencySample(node_id="gpu-0", latency_ms=20.0),
            LatencySample(node_id="gpu-0", latency_ms=30.0),
        ]
        slope = detector._compute_slope(samples)
        # Linear regression: (0-1)*10 + (1-1)*20 + (2-1)*30
        # x_mean=1, y_mean=20
        # num = (0-1)*(10-20) + (1-1)*(20-20) + (2-1)*(30-20) = 10 + 0 + 10 = 20
        # den = (0-1)^2 + (1-1)^2 + (2-1)^2 = 1 + 0 + 1 = 2
        # slope = 20 / 2 = 10
        assert slope == 10.0

    def test_compute_slope_flat(self):
        from distllm.dist.partition.adaptive import (
            LatencySample,
            StragglerDetector,
        )

        detector = StragglerDetector()
        samples = [
            LatencySample(node_id="gpu-0", latency_ms=10.0),
            LatencySample(node_id="gpu-0", latency_ms=10.0),
            LatencySample(node_id="gpu-0", latency_ms=10.0),
        ]
        slope = detector._compute_slope(samples)
        assert slope == 0.0

    def test_check_node_missing_node(self):
        from distllm.dist.partition.adaptive import (
            LatencySample,
            StragglerDetector,
        )

        detector = StragglerDetector()
        # _check_node with no samples for this node
        result = detector._check_node("nonexistent")
        assert result is None

    def test_straggler_report_severity_clamped(self):
        from distllm.dist.partition.adaptive import (
            LatencySample,
            StragglerDetector,
        )

        detector = StragglerDetector(abs_threshold=1.5)
        detector.set_expected({"gpu-0": 10.0})

        for _ in range(3):
            detector.record(LatencySample(node_id="gpu-0", latency_ms=10.0))

        # Very high ratio should cap severity at 1.0
        result = detector.record(
            LatencySample(node_id="gpu-0", latency_ms=1000.0),
        )
        assert result is not None
        assert result.severity == 1.0
        assert result.ratio > 10.0


# =============================================================================
# Test AdaptiveConfig
# =============================================================================


class TestAdaptiveConfig:
    """AdaptiveConfig dataclass defaults and overrides."""

    def test_defaults(self):
        from distllm.dist.partition.adaptive import AdaptiveConfig

        config = AdaptiveConfig()
        assert config.enabled is True
        assert config.straggler_threshold == 1.5
        assert config.min_repartition_interval_s == 30.0
        assert config.cooldown_after_repartition_s == 60.0
        assert config.max_repartitions_per_hour == 10
        assert config.require_quorum is True
        assert config.quorum_fraction == 0.5

    def test_custom_values(self):
        from distllm.dist.partition.adaptive import AdaptiveConfig

        config = AdaptiveConfig(
            enabled=False,
            straggler_threshold=3.0,
            min_repartition_interval_s=0.0,
            max_repartitions_per_hour=5,
            require_quorum=False,
            quorum_fraction=0.75,
        )
        assert config.enabled is False
        assert config.straggler_threshold == 3.0
        assert config.min_repartition_interval_s == 0.0
        assert config.max_repartitions_per_hour == 5
        assert config.require_quorum is False
        assert config.quorum_fraction == 0.75


# =============================================================================
# Test AdaptiveRepartitioner
# =============================================================================


class TestAdaptiveRepartitioner:
    """AdaptiveRepartitioner orchestration logic."""

    def test_init(self):
        from distllm.dist.partition.adaptive import AdaptiveRepartitioner

        cost_model, _, node_ids, _ = _make_simple_partition_solution()
        repartitioner = AdaptiveRepartitioner(
            cost_model=cost_model,
            node_ids=node_ids,
        )
        assert repartitioner is not None
        assert repartitioner.current_solution is None
        assert repartitioner.repartition_history == []
        assert repartitioner.straggler_stats == {}

    def test_init_with_config(self):
        from distllm.dist.partition.adaptive import (
            AdaptiveConfig,
            AdaptiveRepartitioner,
        )

        cost_model, _, node_ids, _ = _make_simple_partition_solution()
        config = AdaptiveConfig(enabled=False)
        repartitioner = AdaptiveRepartitioner(
            cost_model=cost_model,
            node_ids=node_ids,
            config=config,
        )
        assert repartitioner is not None

    def test_set_initial_partition(self):
        from distllm.dist.partition.adaptive import AdaptiveRepartitioner

        cost_model, solution, node_ids, num_layers = _make_simple_partition_solution()
        repartitioner = AdaptiveRepartitioner(
            cost_model=cost_model,
            node_ids=node_ids,
        )
        repartitioner.set_initial_partition(solution, num_layers)
        assert repartitioner.current_solution is solution
        assert repartitioner.current_solution.num_nodes > 0

    def test_record_latency(self):
        from distllm.dist.partition.adaptive import AdaptiveRepartitioner

        cost_model, solution, node_ids, num_layers = _make_simple_partition_solution()
        repartitioner = AdaptiveRepartitioner(
            cost_model=cost_model,
            node_ids=node_ids,
        )
        repartitioner.set_initial_partition(solution, num_layers)

        # Single record should not return a report (not enough samples)
        result = repartitioner.record_latency(node_ids[0], 10.0)
        assert result is None

    def test_disabled_config_returns_none(self):
        from distllm.dist.partition.adaptive import (
            AdaptiveConfig,
            AdaptiveRepartitioner,
        )

        cost_model, solution, node_ids, num_layers = _make_simple_partition_solution()
        config = AdaptiveConfig(enabled=False)
        repartitioner = AdaptiveRepartitioner(
            cost_model=cost_model,
            node_ids=node_ids,
            config=config,
        )
        repartitioner.set_initial_partition(solution, num_layers)

        result = repartitioner.check_and_repartition(
            {node_ids[0]: 1000.0},
        )
        assert result is None

    def test_no_initial_partition_returns_none(self):
        from distllm.dist.partition.adaptive import AdaptiveRepartitioner

        cost_model, _, node_ids, _ = _make_simple_partition_solution()
        repartitioner = AdaptiveRepartitioner(
            cost_model=cost_model,
            node_ids=node_ids,
        )

        result = repartitioner.check_and_repartition(
            {node_ids[0]: 1000.0},
        )
        assert result is None

    def test_check_and_repartition_detects_straggler(self):
        from distllm.dist.partition.adaptive import (
            AdaptiveConfig,
            AdaptiveRepartitioner,
        )

        cost_model, solution, node_ids, num_layers = _make_simple_partition_solution()
        config = AdaptiveConfig(
            min_repartition_interval_s=0.0,
            max_repartitions_per_hour=100,
            require_quorum=False,
            straggler_threshold=1.1,
        )
        repartitioner = AdaptiveRepartitioner(
            cost_model=cost_model,
            node_ids=node_ids,
            config=config,
        )
        repartitioner.set_initial_partition(solution, num_layers)

        # Use a very high observed latency to trigger detection
        straggler_node = node_ids[0]
        high_latency = 1e9
        result = repartitioner.check_and_repartition(
            {straggler_node: high_latency},
        )
        assert result is not None
        assert isinstance(result, object)
        assert result.num_nodes > 0

    def test_check_and_repartition_no_straggler_returns_none(self):
        from distllm.dist.partition.adaptive import (
            AdaptiveConfig,
            AdaptiveRepartitioner,
        )

        cost_model, solution, node_ids, num_layers = _make_simple_partition_solution()
        config = AdaptiveConfig(
            min_repartition_interval_s=0.0,
            require_quorum=False,
            straggler_threshold=10.0,
        )
        repartitioner = AdaptiveRepartitioner(
            cost_model=cost_model,
            node_ids=node_ids,
            config=config,
        )
        repartitioner.set_initial_partition(solution, num_layers)

        # Low latency should not trigger straggler detection
        result = repartitioner.check_and_repartition(
            {node_ids[0]: 0.001},
        )
        assert result is None

    def test_force_repartition_without_initial(self):
        from distllm.dist.partition.adaptive import AdaptiveRepartitioner

        cost_model, _, node_ids, _ = _make_simple_partition_solution()
        repartitioner = AdaptiveRepartitioner(
            cost_model=cost_model,
            node_ids=node_ids,
        )

        result = repartitioner.force_repartition()
        assert result is None

    def test_force_repartition_with_initial(self):
        from distllm.dist.partition.adaptive import (
            AdaptiveConfig,
            AdaptiveRepartitioner,
        )

        cost_model, solution, node_ids, num_layers = _make_simple_partition_solution()
        config = AdaptiveConfig(
            min_repartition_interval_s=0.0,
            max_repartitions_per_hour=100,
        )
        repartitioner = AdaptiveRepartitioner(
            cost_model=cost_model,
            node_ids=node_ids,
            config=config,
        )
        repartitioner.set_initial_partition(solution, num_layers)

        result = repartitioner.force_repartition(reason="test")
        assert result is not None
        assert result.num_nodes > 0

    def test_repartition_history(self):
        from distllm.dist.partition.adaptive import (
            AdaptiveConfig,
            AdaptiveRepartitioner,
        )

        cost_model, solution, node_ids, num_layers = _make_simple_partition_solution()
        config = AdaptiveConfig(
            min_repartition_interval_s=0.0,
            max_repartitions_per_hour=100,
            require_quorum=False,
            straggler_threshold=1.1,
        )
        repartitioner = AdaptiveRepartitioner(
            cost_model=cost_model,
            node_ids=node_ids,
            config=config,
        )
        repartitioner.set_initial_partition(solution, num_layers)

        repartitioner.check_and_repartition({node_ids[0]: 1e9})
        assert len(repartitioner.repartition_history) == 1
        assert repartitioner.repartition_history[0].trigger.value == "straggler"

        repartitioner.check_and_repartition({node_ids[0]: 1e9})
        assert len(repartitioner.repartition_history) == 2

    def test_straggler_stats_property(self):
        from distllm.dist.partition.adaptive import (
            AdaptiveConfig,
            AdaptiveRepartitioner,
        )

        cost_model, solution, node_ids, num_layers = _make_simple_partition_solution()
        config = AdaptiveConfig(min_repartition_interval_s=0.0)
        repartitioner = AdaptiveRepartitioner(
            cost_model=cost_model,
            node_ids=node_ids,
            config=config,
        )
        repartitioner.set_initial_partition(solution, num_layers)
        repartitioner.record_latency(node_ids[0], 10.0)
        repartitioner.record_latency(node_ids[0], 12.0)

        stats = repartitioner.straggler_stats
        assert node_ids[0] in stats
        assert "mean_ms" in stats[node_ids[0]]

    def test_quorum_blocks_straggler_detection(self):
        from distllm.dist.partition.adaptive import (
            AdaptiveConfig,
            AdaptiveRepartitioner,
        )

        # 3 nodes, require 50% quorum
        nid = ["node-a", "node-b", "node-c"]
        profiles = {}
        for i, name in enumerate(nid):
            from distllm.dist.partition.profiles import GPUProfile
            profiles[name] = GPUProfile(
                gpu_id=i, name="A100",
                total_memory_bytes=80 * 1024**3,
                compute_tflops=312.0,
                memory_bandwidth_gbps=2039.0,
            )
        from distllm.dist.partition.profiles import LayerWeights

        layers = [LayerWeights(layer_id=0, layer_type="embed", weight_memory_bytes=512*1024**2, flops_per_seq=1_000_000)]
        for i in range(1, 5):
            layers.append(LayerWeights(layer_id=i, layer_type="transformer", weight_memory_bytes=2*1024**3, flops_per_seq=2_000_000_000))
        from distllm.dist.partition.topology import TopologyGraph, LinkProfile

        links = [LinkProfile(source=nid[0], target=nid[1], bandwidth_gbps=600.0),
                 LinkProfile(source=nid[0], target=nid[2], bandwidth_gbps=600.0),
                 LinkProfile(source=nid[1], target=nid[2], bandwidth_gbps=600.0)]
        topo = TopologyGraph(node_ids=nid, links=links)
        from distllm.dist.partition.cost_model import PartitionCostModel

        cm = PartitionCostModel(gpu_profiles=profiles, layer_weights=layers, topology=topo)
        from distllm.dist.partition.optimizer import PartitionOptimizer

        opt = PartitionOptimizer(cost_model=cm, node_ids=nid)
        sol = opt.solve(5)

        config = AdaptiveConfig(
            min_repartition_interval_s=0.0,
            require_quorum=True,
            quorum_fraction=0.5,
            straggler_threshold=1.1,
        )
        repartitioner = AdaptiveRepartitioner(
            cost_model=cm,
            node_ids=nid,
            config=config,
        )
        repartitioner.set_initial_partition(sol, 5)

        # Quorum checks straggler_fraction = len(reports) / len(observed)
        # Pass ALL 3 nodes in observed, but only 1 is slow => 1/3 < 0.5
        result = repartitioner.check_and_repartition(
            {nid[0]: 1e9, nid[1]: 10.0, nid[2]: 10.0},
        )
        assert result is None

    def test_quorum_met_with_two_stragglers(self):
        from distllm.dist.partition.adaptive import (
            AdaptiveConfig,
            AdaptiveRepartitioner,
        )

        nid = ["node-a", "node-b", "node-c"]
        profiles = {}
        for i, name in enumerate(nid):
            from distllm.dist.partition.profiles import GPUProfile

            profiles[name] = GPUProfile(gpu_id=i, name="A100", total_memory_bytes=80*1024**3, compute_tflops=312.0, memory_bandwidth_gbps=2039.0)
        from distllm.dist.partition.profiles import LayerWeights

        layers = [LayerWeights(layer_id=0, layer_type="embed", weight_memory_bytes=512*1024**2, flops_per_seq=1_000_000)]
        for i in range(1, 5):
            layers.append(LayerWeights(layer_id=i, layer_type="transformer", weight_memory_bytes=2*1024**3, flops_per_seq=2_000_000_000))
        from distllm.dist.partition.topology import TopologyGraph, LinkProfile

        links = [LinkProfile(source=nid[0], target=nid[1], bandwidth_gbps=600.0),
                 LinkProfile(source=nid[0], target=nid[2], bandwidth_gbps=600.0),
                 LinkProfile(source=nid[1], target=nid[2], bandwidth_gbps=600.0)]
        topo = TopologyGraph(node_ids=nid, links=links)
        from distllm.dist.partition.cost_model import PartitionCostModel

        cm = PartitionCostModel(gpu_profiles=profiles, layer_weights=layers, topology=topo)
        from distllm.dist.partition.optimizer import PartitionOptimizer

        opt = PartitionOptimizer(cost_model=cm, node_ids=nid)
        sol = opt.solve(5)

        config = AdaptiveConfig(
            min_repartition_interval_s=0.0,
            require_quorum=True,
            quorum_fraction=0.5,
            straggler_threshold=1.1,
        )
        repartitioner = AdaptiveRepartitioner(
            cost_model=cm,
            node_ids=nid,
            config=config,
        )
        repartitioner.set_initial_partition(sol, 5)

        # 2 out of 3 nodes straggling -- quorum fraction 0.5 is met (0.66 > 0.5)
        result = repartitioner.check_and_repartition(
            {nid[0]: 1e9, nid[1]: 1e9},
        )
        assert result is not None

    def test_rate_limit_blocks_repartition(self):
        from distllm.dist.partition.adaptive import (
            AdaptiveConfig,
            AdaptiveRepartitioner,
        )

        cost_model, solution, node_ids, num_layers = _make_simple_partition_solution()
        config = AdaptiveConfig(
            min_repartition_interval_s=0.0,
            max_repartitions_per_hour=1,
            require_quorum=False,
            straggler_threshold=1.1,
        )
        repartitioner = AdaptiveRepartitioner(
            cost_model=cost_model,
            node_ids=node_ids,
            config=config,
        )
        repartitioner.set_initial_partition(solution, num_layers)

        # First repartition should succeed
        result = repartitioner.check_and_repartition({node_ids[0]: 1e9})
        assert result is not None

        # Second repartition within the same "hour" should be blocked
        result = repartitioner.check_and_repartition({node_ids[0]: 1e9})
        assert result is None

    def test_callback_invoked(self):
        from distllm.dist.partition.adaptive import (
            AdaptiveConfig,
            AdaptiveRepartitioner,
        )

        callbacks = []

        def on_repartition(event):
            callbacks.append(event)

        cost_model, solution, node_ids, num_layers = _make_simple_partition_solution()
        config = AdaptiveConfig(
            min_repartition_interval_s=0.0,
            require_quorum=False,
            straggler_threshold=1.1,
        )
        repartitioner = AdaptiveRepartitioner(
            cost_model=cost_model,
            node_ids=node_ids,
            config=config,
            on_repartition=on_repartition,
        )
        repartitioner.set_initial_partition(solution, num_layers)

        repartitioner.check_and_repartition({node_ids[0]: 1e9})
        assert len(callbacks) == 1
        assert callbacks[0].trigger.value == "straggler"

    def test_callback_exception_does_not_crash(self):
        from distllm.dist.partition.adaptive import (
            AdaptiveConfig,
            AdaptiveRepartitioner,
        )

        def failing_callback(event):
            raise RuntimeError("callback failure")

        cost_model, solution, node_ids, num_layers = _make_simple_partition_solution()
        config = AdaptiveConfig(
            min_repartition_interval_s=0.0,
            require_quorum=False,
            straggler_threshold=1.1,
        )
        repartitioner = AdaptiveRepartitioner(
            cost_model=cost_model,
            node_ids=node_ids,
            config=config,
            on_repartition=failing_callback,
        )
        repartitioner.set_initial_partition(solution, num_layers)

        # Should not raise
        result = repartitioner.check_and_repartition({node_ids[0]: 1e9})
        assert result is not None

    def test_empty_observed_latencies(self):
        from distllm.dist.partition.adaptive import (
            AdaptiveConfig,
            AdaptiveRepartitioner,
        )

        cost_model, solution, node_ids, num_layers = _make_simple_partition_solution()
        config = AdaptiveConfig(min_repartition_interval_s=0.0)
        repartitioner = AdaptiveRepartitioner(
            cost_model=cost_model,
            node_ids=node_ids,
            config=config,
        )
        repartitioner.set_initial_partition(solution, num_layers)

        result = repartitioner.check_and_repartition({})
        assert result is None

    def test_empty_node_ids(self):
        from distllm.dist.partition.adaptive import (
            AdaptiveConfig,
            AdaptiveRepartitioner,
        )

        cost_model, _, _, _ = _make_simple_partition_solution()
        repartitioner = AdaptiveRepartitioner(
            cost_model=cost_model,
            node_ids=[],
        )
        assert repartitioner is not None

    def test_multiple_nodes_mixed_latencies(self):
        from distllm.dist.partition.adaptive import (
            AdaptiveConfig,
            AdaptiveRepartitioner,
        )

        # Test with 3 nodes
        nid = ["node-x", "node-y", "node-z"]
        from distllm.dist.partition.profiles import GPUProfile, LayerWeights
        from distllm.dist.partition.topology import TopologyGraph, LinkProfile
        from distllm.dist.partition.cost_model import PartitionCostModel
        from distllm.dist.partition.optimizer import PartitionOptimizer

        profiles = {name: GPUProfile(gpu_id=i, name="A100", total_memory_bytes=80*1024**3, compute_tflops=312.0, memory_bandwidth_gbps=2039.0) for i, name in enumerate(nid)}
        layers = [LayerWeights(layer_id=0, layer_type="embed", weight_memory_bytes=512*1024**2, flops_per_seq=1_000_000)]
        for i in range(1, 7):
            layers.append(LayerWeights(layer_id=i, layer_type="transformer", weight_memory_bytes=2*1024**3, flops_per_seq=2_000_000_000))
        links = [LinkProfile(source=nid[i], target=nid[j], bandwidth_gbps=600.0) for i in range(3) for j in range(i+1, 3)]
        topo = TopologyGraph(node_ids=nid, links=links)
        cm = PartitionCostModel(gpu_profiles=profiles, layer_weights=layers, topology=topo)
        opt = PartitionOptimizer(cost_model=cm, node_ids=nid)
        sol = opt.solve(7)

        config = AdaptiveConfig(
            min_repartition_interval_s=0.0,
            require_quorum=False,
            straggler_threshold=1.5,
        )
        repartitioner = AdaptiveRepartitioner(
            cost_model=cm,
            node_ids=nid,
            config=config,
        )
        repartitioner.set_initial_partition(sol, 7)

        # One node slow, others normal
        result = repartitioner.check_and_repartition(
            {nid[0]: 1e9, nid[1]: 10.0, nid[2]: 10.0},
        )
        assert result is not None
        assert result.num_nodes > 0


# =============================================================================
# Test _OverriddenCostModel
# =============================================================================


class TestOverriddenCostModel:
    """_OverriddenCostModel wrapper with per-node cost multipliers."""

    def test_no_override_passthrough(self):
        from distllm.dist.partition.adaptive import _OverriddenCostModel
        from distllm.dist.partition.cost_model import PartitionCostModel

        cm, _, node_ids, _ = _make_simple_partition_solution()
        overridden = _OverriddenCostModel(base=cm, overrides={})

        cost = overridden.evaluate(node_ids[0], 0, 3)
        expected = cm.evaluate(node_ids[0], 0, 3)
        assert cost.total_time_ms == expected.total_time_ms
        assert cost.compute_time_ms == expected.compute_time_ms

    def test_override_multiplies_time(self):
        from distllm.dist.partition.adaptive import _OverriddenCostModel

        cm, _, node_ids, _ = _make_simple_partition_solution()
        overridden = _OverriddenCostModel(
            base=cm,
            overrides={node_ids[0]: 2.5},
        )

        cost = overridden.evaluate(node_ids[0], 0, 3)
        expected = cm.evaluate(node_ids[0], 0, 3)
        assert cost.total_time_ms == pytest.approx(expected.total_time_ms * 2.5)
        assert cost.compute_time_ms == pytest.approx(
            expected.compute_time_ms * 2.5,
        )

    def test_override_multiplier_one_no_change(self):
        from distllm.dist.partition.adaptive import _OverriddenCostModel

        cm, _, node_ids, _ = _make_simple_partition_solution()
        overridden = _OverriddenCostModel(
            base=cm,
            overrides={node_ids[0]: 1.0},
        )

        cost = overridden.evaluate(node_ids[0], 0, 3)
        expected = cm.evaluate(node_ids[0], 0, 3)
        assert cost.total_time_ms == expected.total_time_ms
        assert cost.compute_time_ms == expected.compute_time_ms

    def test_override_multiplier_below_one_not_applied(self):
        from distllm.dist.partition.adaptive import _OverriddenCostModel

        cm, _, node_ids, _ = _make_simple_partition_solution()
        overridden = _OverriddenCostModel(
            base=cm,
            overrides={node_ids[0]: 0.5},
        )

        cost = overridden.evaluate(node_ids[0], 0, 3)
        expected = cm.evaluate(node_ids[0], 0, 3)
        # Multiplier < 1.0 should NOT be applied (only > 1.0 triggers)
        assert cost.total_time_ms == expected.total_time_ms
        assert cost.compute_time_ms == expected.compute_time_ms

    def test_override_different_node_not_affected(self):
        from distllm.dist.partition.adaptive import _OverriddenCostModel

        cm, _, node_ids, _ = _make_simple_partition_solution()
        overridden = _OverriddenCostModel(
            base=cm,
            overrides={node_ids[0]: 5.0},
        )

        cost_normal = overridden.evaluate(node_ids[1], 0, 3)
        expected_normal = cm.evaluate(node_ids[1], 0, 3)
        # Non-overridden node should be unchanged
        assert cost_normal.total_time_ms == expected_normal.total_time_ms

        cost_slow = overridden.evaluate(node_ids[0], 0, 3)
        expected_slow = cm.evaluate(node_ids[0], 0, 3)
        assert cost_slow.total_time_ms == pytest.approx(
            expected_slow.total_time_ms * 5.0,
        )

    def test_evaluate_partition(self):
        from distllm.dist.partition.adaptive import _OverriddenCostModel

        cm, _, node_ids, _ = _make_simple_partition_solution()
        overridden = _OverriddenCostModel(
            base=cm,
            overrides={node_ids[0]: 3.0},
        )

        partition = [(node_ids[0], 0, 3), (node_ids[1], 3, 5)]
        costs = overridden.evaluate_partition(partition)
        assert len(costs) == 2
        # First cost should be multiplied
        expected0 = cm.evaluate(node_ids[0], 0, 3)
        assert costs[0].total_time_ms == pytest.approx(
            expected0.total_time_ms * 3.0,
        )
        # Second cost should be unchanged
        expected1 = cm.evaluate(node_ids[1], 3, 5)
        assert costs[1].total_time_ms == expected1.total_time_ms

    def test_combined_throughput_empty_partition(self):
        from distllm.dist.partition.adaptive import _OverriddenCostModel

        cm, _, node_ids, _ = _make_simple_partition_solution()
        overridden = _OverriddenCostModel(base=cm, overrides={})

        throughput = overridden.combined_throughput([])
        assert throughput == 0.0

    def test_combined_throughput_non_empty(self):
        from distllm.dist.partition.adaptive import _OverriddenCostModel

        cm, _, node_ids, _ = _make_simple_partition_solution()
        overridden = _OverriddenCostModel(base=cm, overrides={})

        partition = [(node_ids[0], 0, 3), (node_ids[1], 3, 5)]
        throughput = overridden.combined_throughput(partition)
        assert throughput > 0.0

    def test_wraps_base_attributes(self):
        from distllm.dist.partition.adaptive import _OverriddenCostModel

        cm, _, node_ids, _ = _make_simple_partition_solution()
        overridden = _OverriddenCostModel(base=cm, overrides={})
        # Should have _layer_weights and _topology from base
        assert overridden._layer_weights is cm._layer_weights
        assert overridden._topology is cm._topology
