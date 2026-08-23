"""Tests for PipelineStrategy and StrategySelector using only real objects (zero mocks)."""

from __future__ import annotations

from typing import Any

import pytest

from distllm.dist.pipeline.strategy import PipelineStrategy, StrategySelector


def _make_node(tflops: float = 312.0) -> Any:
    """Create a minimal node object with a gpu_compute_tflops attribute."""
    return type("Node", (), {"gpu_compute_tflops": tflops})()


class TestPipelineStrategy:
    """PipelineStrategy enum behavior."""

    def test_values(self) -> None:
        assert PipelineStrategy.SEQUENTIAL.value == "sequential"
        assert PipelineStrategy.OVERLAP.value == "overlap"
        assert PipelineStrategy.ASYNC_1F1B.value == "async_1f1b"
        assert PipelineStrategy.STAGED.value == "staged"
        assert PipelineStrategy.DISAGGREGATED.value == "disaggregated"
        assert PipelineStrategy.REDUNDANT.value == "redundant"

    def test_membership(self) -> None:
        for s in (
            PipelineStrategy.SEQUENTIAL,
            PipelineStrategy.OVERLAP,
            PipelineStrategy.ASYNC_1F1B,
            PipelineStrategy.STAGED,
            PipelineStrategy.DISAGGREGATED,
            PipelineStrategy.REDUNDANT,
        ):
            assert s in PipelineStrategy

    def test_all_values_unique(self) -> None:
        values = [s.value for s in PipelineStrategy]
        assert len(values) == len(set(values))

    def test_enum_comparison(self) -> None:
        assert PipelineStrategy.SEQUENTIAL is PipelineStrategy("sequential")
        assert PipelineStrategy.OVERLAP is PipelineStrategy("overlap")
        assert PipelineStrategy.ASYNC_1F1B is PipelineStrategy("async_1f1b")
        assert PipelineStrategy.STAGED is PipelineStrategy("staged")
        assert PipelineStrategy.DISAGGREGATED is PipelineStrategy("disaggregated")
        assert PipelineStrategy.REDUNDANT is PipelineStrategy("redundant")


class TestStrategySelectorInit:
    """StrategySelector construction."""

    def test_default_construction(self) -> None:
        selector = StrategySelector()
        assert selector._model_size == "7B"
        assert selector._hidden_dim == 4096
        assert selector._num_heads == 32
        assert selector._head_dim == 128
        assert selector._vocab_size == 32000
        # Internal state initialized
        assert list(selector._strategy_latency.keys()) == [s.value for s in PipelineStrategy]
        assert selector._cached_simulator is None
        assert selector._cached_node_signature == ""

    def test_custom_construction(self) -> None:
        selector = StrategySelector(
            model_size="70B",
            hidden_dim=8192,
            num_heads=64,
            head_dim=256,
            vocab_size=64000,
        )
        assert selector._model_size == "70B"
        assert selector._hidden_dim == 8192
        assert selector._num_heads == 64
        assert selector._head_dim == 256
        assert selector._vocab_size == 64000

    def test_strategy_latency_queues_initialized(self) -> None:
        selector = StrategySelector()
        for s in PipelineStrategy:
            assert s.value in selector._strategy_latency
            # deque with maxlen=32
            assert selector._strategy_latency[s.value].maxlen == 32


class TestStrategySelectorSelect:
    """Strategy selection logic (early-return paths and simulation path)."""

    # -- Early return: redundant / disaggregated flags ----

    def test_redundant_enabled_returns_redundant(self) -> None:
        selector = StrategySelector()
        result = selector.select_strategy(
            num_nodes=2,
            total_layers=32,
            batch_size=1,
            seq_len=2048,
            current_load=0,
            nodes={},
            redundant_enabled=True,
        )
        assert result is PipelineStrategy.REDUNDANT

    def test_redundant_overrides_single_node(self) -> None:
        selector = StrategySelector()
        result = selector.select_strategy(
            num_nodes=1,
            total_layers=32,
            batch_size=1,
            seq_len=2048,
            current_load=0,
            nodes={},
            redundant_enabled=True,
        )
        assert result is PipelineStrategy.REDUNDANT

    def test_disaggregated_enabled_returns_disaggregated(self) -> None:
        selector = StrategySelector()
        result = selector.select_strategy(
            num_nodes=2,
            total_layers=32,
            batch_size=1,
            seq_len=2048,
            current_load=0,
            nodes={},
            disaggregated_enabled=True,
        )
        assert result is PipelineStrategy.DISAGGREGATED

    def test_disaggregated_overrides_single_node(self) -> None:
        selector = StrategySelector()
        result = selector.select_strategy(
            num_nodes=1,
            total_layers=32,
            batch_size=1,
            seq_len=2048,
            current_load=0,
            nodes={},
            disaggregated_enabled=True,
        )
        assert result is PipelineStrategy.DISAGGREGATED

    def test_redundant_takes_priority_over_disaggregated(self) -> None:
        """redundant_enabled is checked first in select_strategy."""
        selector = StrategySelector()
        result = selector.select_strategy(
            num_nodes=2,
            total_layers=32,
            batch_size=1,
            seq_len=2048,
            current_load=0,
            nodes={},
            redundant_enabled=True,
            disaggregated_enabled=True,
        )
        assert result is PipelineStrategy.REDUNDANT

    # -- Early return: num_nodes <= 1 ----

    def test_single_node_returns_sequential(self) -> None:
        selector = StrategySelector()
        result = selector.select_strategy(
            num_nodes=1,
            total_layers=32,
            batch_size=1,
            seq_len=2048,
            current_load=0,
            nodes={"node0": _make_node()},
        )
        assert result is PipelineStrategy.SEQUENTIAL

    def test_zero_nodes_returns_sequential(self) -> None:
        selector = StrategySelector()
        result = selector.select_strategy(
            num_nodes=0,
            total_layers=32,
            batch_size=1,
            seq_len=2048,
            current_load=0,
            nodes={},
        )
        assert result is PipelineStrategy.SEQUENTIAL

    def test_negative_nodes_returns_sequential(self) -> None:
        selector = StrategySelector()
        result = selector.select_strategy(
            num_nodes=-1,
            total_layers=32,
            batch_size=1,
            seq_len=2048,
            current_load=0,
            nodes={},
        )
        assert result is PipelineStrategy.SEQUENTIAL

    # -- Simulation path: basic ----

    def test_multi_node_returns_valid_strategy(self) -> None:
        selector = StrategySelector(model_size="1B")
        nodes = {"node0": _make_node(312.0), "node1": _make_node(312.0)}
        result = selector.select_strategy(
            num_nodes=2,
            total_layers=32,
            batch_size=1,
            seq_len=128,
            current_load=0,
            nodes=nodes,
        )
        assert isinstance(result, PipelineStrategy)

    def test_multi_node_deterministic(self) -> None:
        selector = StrategySelector(model_size="1B")
        nodes = {"node0": _make_node(312.0), "node1": _make_node(312.0)}
        result1 = selector.select_strategy(
            num_nodes=2,
            total_layers=32,
            batch_size=1,
            seq_len=128,
            current_load=0,
            nodes=nodes,
        )
        result2 = selector.select_strategy(
            num_nodes=2,
            total_layers=32,
            batch_size=1,
            seq_len=128,
            current_load=0,
            nodes=nodes,
        )
        assert result1 == result2

    # -- Simulation path: flag toggles ----

    def test_overlap_disabled_returns_sequential_with_2_nodes(self) -> None:
        """With 2 nodes and overlap disabled, only SEQUENTIAL is a candidate."""
        selector = StrategySelector(model_size="1B")
        nodes = {"node0": _make_node(312.0), "node1": _make_node(312.0)}
        result = selector.select_strategy(
            num_nodes=2,
            total_layers=32,
            batch_size=1,
            seq_len=128,
            current_load=0,
            nodes=nodes,
            enable_overlap=False,
        )
        assert result is PipelineStrategy.SEQUENTIAL

    def test_all_candidate_flags_false_returns_sequential(self) -> None:
        selector = StrategySelector(model_size="1B")
        nodes = {"node0": _make_node(312.0), "node1": _make_node(312.0)}
        result = selector.select_strategy(
            num_nodes=2,
            total_layers=32,
            batch_size=1,
            seq_len=128,
            current_load=0,
            nodes=nodes,
            enable_overlap=False,
            stages_enabled=False,
            use_async_pipeline=False,
            redundant_enabled=False,
            disaggregated_enabled=False,
        )
        assert result is PipelineStrategy.SEQUENTIAL

    def test_async_pipeline_candidate_included(self) -> None:
        selector = StrategySelector(model_size="1B")
        nodes = {"node0": _make_node(312.0), "node1": _make_node(312.0)}
        result = selector.select_strategy(
            num_nodes=2,
            total_layers=32,
            batch_size=1,
            seq_len=128,
            current_load=0,
            nodes=nodes,
            use_async_pipeline=True,
        )
        assert isinstance(result, PipelineStrategy)

    def test_stages_enabled_with_4_nodes(self) -> None:
        selector = StrategySelector(model_size="1B")
        nodes = {f"node{i}": _make_node(312.0) for i in range(4)}
        result = selector.select_strategy(
            num_nodes=4,
            total_layers=32,
            batch_size=1,
            seq_len=128,
            current_load=0,
            nodes=nodes,
        )
        assert isinstance(result, PipelineStrategy)

    def test_stages_disabled_with_4_nodes_excludes_staged(self) -> None:
        selector = StrategySelector(model_size="1B")
        nodes = {f"node{i}": _make_node(312.0) for i in range(4)}
        result = selector.select_strategy(
            num_nodes=4,
            total_layers=32,
            batch_size=1,
            seq_len=128,
            current_load=0,
            nodes=nodes,
            stages_enabled=False,
        )
        assert isinstance(result, PipelineStrategy)
        assert result is not PipelineStrategy.STAGED

    # -- Simulation path: node variations ----

    def test_empty_nodes_dict(self) -> None:
        selector = StrategySelector(model_size="1B")
        result = selector.select_strategy(
            num_nodes=2,
            total_layers=32,
            batch_size=1,
            seq_len=128,
            current_load=0,
            nodes={},
        )
        assert isinstance(result, PipelineStrategy)

    def test_nodes_without_tflops_attribute(self) -> None:
        selector = StrategySelector(model_size="1B")
        bare_node = type("Node", (), {})()
        nodes = {"node0": bare_node, "node1": bare_node}
        result = selector.select_strategy(
            num_nodes=2,
            total_layers=32,
            batch_size=1,
            seq_len=128,
            current_load=0,
            nodes=nodes,
        )
        assert isinstance(result, PipelineStrategy)

    def test_mixed_tflops_nodes(self) -> None:
        selector = StrategySelector(model_size="1B")
        nodes = {"fast": _make_node(500.0), "slow": _make_node(200.0)}
        result = selector.select_strategy(
            num_nodes=2,
            total_layers=32,
            batch_size=1,
            seq_len=128,
            current_load=0,
            nodes=nodes,
        )
        assert isinstance(result, PipelineStrategy)

    # -- Simulation path: boundary values ----

    def test_zero_seq_len(self) -> None:
        selector = StrategySelector(model_size="1B")
        nodes = {"node0": _make_node(312.0), "node1": _make_node(312.0)}
        result = selector.select_strategy(
            num_nodes=2,
            total_layers=32,
            batch_size=1,
            seq_len=0,
            current_load=0,
            nodes=nodes,
        )
        assert isinstance(result, PipelineStrategy)

    def test_zero_total_layers(self) -> None:
        selector = StrategySelector(model_size="1B")
        nodes = {"node0": _make_node(312.0), "node1": _make_node(312.0)}
        result = selector.select_strategy(
            num_nodes=2,
            total_layers=0,
            batch_size=1,
            seq_len=128,
            current_load=0,
            nodes=nodes,
        )
        assert isinstance(result, PipelineStrategy)

    def test_large_num_nodes(self) -> None:
        selector = StrategySelector(model_size="1B")
        nodes = {f"node{i}": _make_node(312.0) for i in range(16)}
        result = selector.select_strategy(
            num_nodes=16,
            total_layers=64,
            batch_size=4,
            seq_len=512,
            current_load=5,
            nodes=nodes,
        )
        assert isinstance(result, PipelineStrategy)

    # -- Simulation path: load boundary ----

    def test_current_load_boundary_six(self) -> None:
        """Load=6 uses latency-based sorting; load=7 uses throughput-based."""
        selector = StrategySelector(model_size="1B")
        nodes = {"node0": _make_node(312.0), "node1": _make_node(312.0)}

        result_at_6 = selector.select_strategy(
            num_nodes=2,
            total_layers=32,
            batch_size=1,
            seq_len=128,
            current_load=6,
            nodes=nodes,
        )
        result_at_7 = selector.select_strategy(
            num_nodes=2,
            total_layers=32,
            batch_size=1,
            seq_len=128,
            current_load=7,
            nodes=nodes,
        )
        assert isinstance(result_at_6, PipelineStrategy)
        assert isinstance(result_at_7, PipelineStrategy)


class TestStrategySelectorLatency:
    """record_strategy_latency and its effect on selection."""

    def test_record_sequential(self) -> None:
        selector = StrategySelector()
        selector.record_strategy_latency("sequential", 100.0)
        assert len(selector._strategy_latency["sequential"]) == 1
        assert selector._strategy_latency["sequential"][0] == 100.0

    def test_record_multiple_values(self) -> None:
        selector = StrategySelector()
        for v in [10.0, 20.0, 30.0]:
            selector.record_strategy_latency("sequential", v)
        assert list(selector._strategy_latency["sequential"]) == [10.0, 20.0, 30.0]

    def test_record_unknown_strategy_does_not_raise(self) -> None:
        selector = StrategySelector()
        selector.record_strategy_latency("nonexistent", 100.0)

    def test_record_negative_latency(self) -> None:
        selector = StrategySelector()
        selector.record_strategy_latency("sequential", -1.0)
        assert selector._strategy_latency["sequential"][0] == -1.0

    def test_record_zero_latency(self) -> None:
        selector = StrategySelector()
        selector.record_strategy_latency("sequential", 0.0)
        assert selector._strategy_latency["sequential"][0] == 0.0

    def test_record_all_strategies(self) -> None:
        selector = StrategySelector()
        for s in PipelineStrategy:
            selector.record_strategy_latency(s.value, 50.0)
        for s in PipelineStrategy:
            assert len(selector._strategy_latency[s.value]) == 1

    def test_deque_maxlen_enforced(self) -> None:
        selector = StrategySelector()
        for i in range(64):
            selector.record_strategy_latency("sequential", float(i))
        # maxlen=32, only the last 32 values survive
        assert len(selector._strategy_latency["sequential"]) == 32
        assert selector._strategy_latency["sequential"][0] == 32.0

    def test_latency_affects_selection_after_enough_records(self) -> None:
        """After 3+ latency records, the blending path in select_strategy runs."""
        selector = StrategySelector(model_size="1B")
        nodes = {"node0": _make_node(312.0), "node1": _make_node(312.0)}

        # Record high latency for sequential
        for _ in range(5):
            selector.record_strategy_latency("sequential", 99999.0)

        result = selector.select_strategy(
            num_nodes=2,
            total_layers=32,
            batch_size=1,
            seq_len=128,
            current_load=0,
            nodes=nodes,
        )
        assert isinstance(result, PipelineStrategy)


class TestStrategySelectorCaching:
    """_build_simulator caching behavior via public select_strategy calls."""

    def test_cache_hit_same_nodes(self) -> None:
        selector = StrategySelector(model_size="1B")
        nodes = {"node0": _make_node(312.0), "node1": _make_node(312.0)}

        selector.select_strategy(
            num_nodes=2,
            total_layers=32,
            batch_size=1,
            seq_len=128,
            current_load=0,
            nodes=nodes,
        )
        cached_first = selector._cached_simulator

        selector.select_strategy(
            num_nodes=2,
            total_layers=32,
            batch_size=1,
            seq_len=128,
            current_load=5,
            nodes=nodes,
        )
        assert selector._cached_simulator is cached_first

    def test_cache_miss_different_node_keys(self) -> None:
        selector = StrategySelector(model_size="1B")

        selector.select_strategy(
            num_nodes=2,
            total_layers=32,
            batch_size=1,
            seq_len=128,
            current_load=0,
            nodes={"node_a": _make_node(312.0), "node_b": _make_node(312.0)},
        )
        cached_first = selector._cached_simulator

        selector.select_strategy(
            num_nodes=2,
            total_layers=32,
            batch_size=1,
            seq_len=128,
            current_load=0,
            nodes={"node_x": _make_node(312.0), "node_y": _make_node(312.0)},
        )
        assert selector._cached_simulator is not cached_first

    def test_cache_miss_different_tflops(self) -> None:
        selector = StrategySelector(model_size="1B")

        selector.select_strategy(
            num_nodes=2,
            total_layers=32,
            batch_size=1,
            seq_len=128,
            current_load=0,
            nodes={"node0": _make_node(312.0)},
        )
        cached_first = selector._cached_simulator

        selector.select_strategy(
            num_nodes=2,
            total_layers=32,
            batch_size=1,
            seq_len=128,
            current_load=0,
            nodes={"node0": _make_node(500.0)},
        )
        assert selector._cached_simulator is not cached_first

    def test_cache_hit_empty_to_empty(self) -> None:
        selector = StrategySelector(model_size="1B")

        selector.select_strategy(
            num_nodes=2,
            total_layers=32,
            batch_size=1,
            seq_len=128,
            current_load=0,
            nodes={},
        )
        cached_first = selector._cached_simulator

        selector.select_strategy(
            num_nodes=2,
            total_layers=32,
            batch_size=1,
            seq_len=128,
            current_load=0,
            nodes={},
        )
        assert selector._cached_simulator is cached_first

    def test_cache_miss_empty_to_nonempty(self) -> None:
        selector = StrategySelector(model_size="1B")

        selector.select_strategy(
            num_nodes=2,
            total_layers=32,
            batch_size=1,
            seq_len=128,
            current_load=0,
            nodes={},
        )
        cached_first = selector._cached_simulator

        selector.select_strategy(
            num_nodes=2,
            total_layers=32,
            batch_size=1,
            seq_len=128,
            current_load=0,
            nodes={"node0": _make_node(312.0)},
        )
        assert selector._cached_simulator is not cached_first

    def test_cached_simulator_uses_averaged_tflops(self) -> None:
        """Build simulator with uneven nodes and verify the average is used."""
        selector = StrategySelector(model_size="1B")
        nodes = {"fast": _make_node(400.0), "slow": _make_node(200.0)}
        selector.select_strategy(
            num_nodes=2,
            total_layers=32,
            batch_size=1,
            seq_len=128,
            current_load=0,
            nodes=nodes,
        )
        sim = selector._cached_simulator
        assert sim is not None
        # Average of 400 and 200 = 300
        assert sim.gpu_tflops == 300.0
