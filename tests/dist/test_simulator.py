"""Tests for PipelineSimulator using only real objects (zero mocks)."""

from __future__ import annotations

import math

import pytest

from distllm.dist.pipeline.simulator import PipelineSimulator


class TestPipelineSimulatorInit:
    """Construction and default values."""

    def test_default_construction(self) -> None:
        sim = PipelineSimulator()
        assert sim.model_size == "7B"
        assert sim.gpu_tflops == 312.0
        assert sim.gpu_bandwidth_gbps == 600.0
        assert sim.interconnect_gbps == 50.0
        assert sim.hidden_dim == 4096
        assert sim.num_heads == 32
        assert sim.head_dim == 128
        assert sim.vocab_size == 32000
        assert sim.activation_ratio == 3.5

    def test_custom_construction(self) -> None:
        sim = PipelineSimulator(
            model_size="70B",
            gpu_tflops=500.0,
            gpu_bandwidth_gbps=800.0,
            interconnect_gbps=100.0,
            hidden_dim=8192,
            num_heads=64,
            head_dim=256,
            vocab_size=64000,
            activation_ratio=4.0,
        )
        assert sim.model_size == "70B"
        assert sim.gpu_tflops == 500.0
        assert sim.gpu_bandwidth_gbps == 800.0
        assert sim.interconnect_gbps == 100.0
        assert sim.hidden_dim == 8192
        assert sim.num_heads == 64
        assert sim.head_dim == 256
        assert sim.vocab_size == 64000
        assert sim.activation_ratio == 4.0

    def test_unknown_model_size_falls_back(self) -> None:
        sim = PipelineSimulator(model_size="999B")
        # FLOPS_PER_LAYER.get("999B", 3.5e11) -> 3.5e11
        assert sim.model_size == "999B"

    def test_flops_per_layer_class_constant(self) -> None:
        assert "70B" in PipelineSimulator.FLOPS_PER_LAYER
        assert "13B" in PipelineSimulator.FLOPS_PER_LAYER
        assert "7B" in PipelineSimulator.FLOPS_PER_LAYER
        assert "3B" in PipelineSimulator.FLOPS_PER_LAYER
        assert "1B" in PipelineSimulator.FLOPS_PER_LAYER
        assert PipelineSimulator.FLOPS_PER_LAYER["7B"] == 3.5e11


class TestPipelineSimulatorComputeTime:
    """_compute_time_per_layer edge cases."""

    def test_minimum_seq_len(self) -> None:
        sim = PipelineSimulator()
        t = sim._compute_time_per_layer(seq_len=1, batch_size=1)
        assert t >= 0.01
        assert isinstance(t, float)

    def test_minimum_batch_size(self) -> None:
        sim = PipelineSimulator()
        t = sim._compute_time_per_layer(seq_len=2048, batch_size=1)
        assert t >= 0.01

    def test_large_batch_and_seq(self) -> None:
        sim = PipelineSimulator()
        t = sim._compute_time_per_layer(seq_len=8192, batch_size=64)
        assert t >= 0.01

    def test_zero_seq_len_still_positive(self) -> None:
        sim = PipelineSimulator()
        t = sim._compute_time_per_layer(seq_len=0, batch_size=1)
        # total_flops = flops * 1 * 0 = 0, so compute_ms = 0, then max(0, 0.01) = 0.01
        assert t == 0.01

    def test_zero_batch_still_positive(self) -> None:
        sim = PipelineSimulator()
        t = sim._compute_time_per_layer(seq_len=2048, batch_size=0)
        assert t == 0.01

    def test_unknown_model_size_uses_fallback(self) -> None:
        sim = PipelineSimulator(model_size="unknown")
        t = sim._compute_time_per_layer(seq_len=2048, batch_size=1)
        assert t >= 0.01


class TestPipelineSimulatorCommTime:
    """Communication time helpers."""

    def test_comm_time_hidden_zero_batch(self) -> None:
        sim = PipelineSimulator()
        t = sim._comm_time_hidden(batch_size=0)
        assert t == 0.0

    def test_comm_time_hidden_small_batch(self) -> None:
        sim = PipelineSimulator()
        t = sim._comm_time_hidden(batch_size=1)
        assert t > 0.0

    def test_comm_time_hidden_large_batch(self) -> None:
        sim = PipelineSimulator()
        t = sim._comm_time_hidden(batch_size=256)
        assert t > 0.0

    def test_comm_time_kv_zero_batch(self) -> None:
        sim = PipelineSimulator()
        t = sim._comm_time_kv(seq_len=2048, batch_size=0, num_layers_per_node=4)
        assert t == 0.0

    def test_comm_time_kv_minimum_layers(self) -> None:
        sim = PipelineSimulator()
        t = sim._comm_time_kv(seq_len=2048, batch_size=1, num_layers_per_node=0)
        assert t > 0.0  # max(0, 1) used

    def test_comm_time_kv_zero_seq(self) -> None:
        sim = PipelineSimulator()
        t = sim._comm_time_kv(seq_len=0, batch_size=1, num_layers_per_node=1)
        assert t == 0.0

    def test_comm_time_kv_typical(self) -> None:
        sim = PipelineSimulator()
        t = sim._comm_time_kv(seq_len=2048, batch_size=4, num_layers_per_node=4)
        assert t > 0.0


class TestPipelineSimulatorSimulate:
    """Public simulate() method."""

    def test_minimal_simulation(self) -> None:
        sim = PipelineSimulator()
        result = sim.simulate(num_nodes=1, num_layers=32)
        assert isinstance(result, dict)
        assert "config" in result
        assert "per_node_estimate_ms" in result
        assert "strategies" in result
        assert "recommendation" in result

    def test_two_node_simulation(self) -> None:
        sim = PipelineSimulator()
        result = sim.simulate(num_nodes=2, num_layers=32)
        assert result["config"]["num_nodes"] == 2
        assert result["config"]["num_layers"] == 32
        assert result["config"]["layers_per_node"] == 16.0

    def test_simulation_with_custom_batch_and_seq(self) -> None:
        sim = PipelineSimulator()
        result = sim.simulate(num_nodes=4, num_layers=32, batch_size=8, seq_len=4096)
        assert result["config"]["batch_size"] == 8
        assert result["config"]["seq_len"] == 4096

    def test_simulation_with_explicit_stages(self) -> None:
        sim = PipelineSimulator()
        result = sim.simulate(num_nodes=8, num_layers=32, num_stages=4)
        assert result["config"]["num_stages"] == 4
        strategies = result["strategies"]
        assert strategies["staged"]["stages"] == 4

    def test_simulation_with_micro_batches(self) -> None:
        sim = PipelineSimulator()
        result = sim.simulate(num_nodes=4, num_layers=32, num_micro_batches=8)
        assert result["config"]["num_micro_batches"] == 8

    def test_single_node_strategies_present(self) -> None:
        sim = PipelineSimulator()
        result = sim.simulate(num_nodes=1, num_layers=8)
        strategies = result["strategies"]
        assert "sequential" in strategies
        assert "overlap" in strategies
        assert "async_1f1b" in strategies
        assert "staged" in strategies

    def test_strategy_latencies_are_positive(self) -> None:
        sim = PipelineSimulator()
        result = sim.simulate(num_nodes=4, num_layers=32)
        for name, data in result["strategies"].items():
            if "latency_ms" in data:
                assert data["latency_ms"] > 0, f"{name} latency should be positive"

    def test_strategy_throughputs_are_non_negative(self) -> None:
        sim = PipelineSimulator()
        result = sim.simulate(num_nodes=4, num_layers=32)
        for name, data in result["strategies"].items():
            if "throughput_tok_s" in data:
                assert data["throughput_tok_s"] >= 0, f"{name} throughput should be >= 0"

    def test_per_node_estimate_has_all_keys(self) -> None:
        sim = PipelineSimulator()
        result = sim.simulate(num_nodes=4, num_layers=32)
        estimate = result["per_node_estimate_ms"]
        assert "compute" in estimate
        assert "comm_hidden" in estimate
        assert "comm_kv" in estimate
        assert "total" in estimate
        assert estimate["total"] > 0

    def test_config_contains_model_name(self) -> None:
        sim = PipelineSimulator(model_size="13B")
        result = sim.simulate(num_nodes=4, num_layers=32)
        assert result["config"]["model"] == "LLaMA-13B"

    def test_bottleneck_is_string(self) -> None:
        sim = PipelineSimulator()
        result = sim.simulate(num_nodes=4, num_layers=32)
        for name, data in result["strategies"].items():
            if "bottleneck" in data:
                assert data["bottleneck"] in ("compute", "communication")

    def test_recommendation_is_string(self) -> None:
        sim = PipelineSimulator()
        result = sim.simulate(num_nodes=4, num_layers=32)
        assert isinstance(result["recommendation"], str)
        assert len(result["recommendation"]) > 0

    def test_simulate_large_cluster(self) -> None:
        sim = PipelineSimulator()
        result = sim.simulate(num_nodes=64, num_layers=80)
        assert result["config"]["num_nodes"] == 64
        strategies = result["strategies"]
        assert strategies["staged"]["stages"] > 1

    def test_simulate_with_very_few_layers(self) -> None:
        sim = PipelineSimulator()
        result = sim.simulate(num_nodes=4, num_layers=2)
        assert result["config"]["layers_per_node"] == 0.5

    def test_async_bubble_ratio(self) -> None:
        sim = PipelineSimulator()
        result = sim.simulate(num_nodes=4, num_layers=32, num_micro_batches=4)
        async_data = result["strategies"]["async_1f1b"]
        assert "bubble_ratio" in async_data
        # bubble_ratio = (4-1) / (4+4-1) = 3/7 ≈ 0.429
        assert async_data["bubble_ratio"] == pytest.approx(3.0 / 7.0, abs=0.001)

    def test_staged_nodes_per_stage(self) -> None:
        sim = PipelineSimulator()
        result = sim.simulate(num_nodes=8, num_layers=32, num_stages=2)
        staged = result["strategies"]["staged"]
        assert staged["nodes_per_stage"] == 4

    def test_rounding_to_two_decimals(self) -> None:
        sim = PipelineSimulator()
        result = sim.simulate(num_nodes=4, num_layers=32)
        estimate = result["per_node_estimate_ms"]
        for key in ("compute", "comm_hidden", "comm_kv", "total"):
            val = estimate[key]
            # Check that the value has at most 2 decimal places (within floating error)
            assert round(val, 2) == val, f"{key}={val} should be rounded to 2 decimals"

    def test_different_model_size_changes_compute(self) -> None:
        sim_small = PipelineSimulator(model_size="1B")
        sim_large = PipelineSimulator(model_size="70B")
        res_small = sim_small.simulate(num_nodes=2, num_layers=32)
        res_large = sim_large.simulate(num_nodes=2, num_layers=32)
        # Larger model should have higher compute time
        assert res_large["per_node_estimate_ms"]["compute"] > res_small["per_node_estimate_ms"]["compute"]

    def test_higher_gpu_tflops_reduces_compute(self) -> None:
        sim_slow = PipelineSimulator(gpu_tflops=100.0)
        sim_fast = PipelineSimulator(gpu_tflops=1000.0)
        res_slow = sim_slow.simulate(num_nodes=2, num_layers=32)
        res_fast = sim_fast.simulate(num_nodes=2, num_layers=32)
        assert res_slow["per_node_estimate_ms"]["compute"] > res_fast["per_node_estimate_ms"]["compute"]

    def test_higher_interconnect_reduces_comm(self) -> None:
        sim_slow = PipelineSimulator(interconnect_gbps=10.0)
        sim_fast = PipelineSimulator(interconnect_gbps=200.0)
        res_slow = sim_slow.simulate(num_nodes=2, num_layers=32)
        res_fast = sim_fast.simulate(num_nodes=2, num_layers=32)
        assert res_slow["per_node_estimate_ms"]["comm_hidden"] > res_fast["per_node_estimate_ms"]["comm_hidden"]

    def test_higher_bandwidth_reduces_kv_comm(self) -> None:
        sim_slow = PipelineSimulator(gpu_bandwidth_gbps=100.0)
        sim_fast = PipelineSimulator(gpu_bandwidth_gbps=2000.0)
        res_slow = sim_slow.simulate(num_nodes=2, num_layers=32)
        res_fast = sim_fast.simulate(num_nodes=2, num_layers=32)
        assert res_slow["per_node_estimate_ms"]["comm_kv"] > res_fast["per_node_estimate_ms"]["comm_kv"]

    def test_sequential_latency_scales_with_nodes(self) -> None:
        sim = PipelineSimulator()
        res_2 = sim.simulate(num_nodes=2, num_layers=32)
        res_4 = sim.simulate(num_nodes=4, num_layers=32)
        # Sequential latency should approximately double when nodes double
        assert res_4["strategies"]["sequential"]["latency_ms"] > res_2["strategies"]["sequential"]["latency_ms"]

    def test_simulate_roundtrip_keys_exist(self) -> None:
        """Verify all documented keys are present in the result."""
        sim = PipelineSimulator()
        result = sim.simulate(num_nodes=4, num_layers=32)
        config = result["config"]
        assert "model" in config
        assert "num_nodes" in config
        assert "num_layers" in config
        assert "layers_per_node" in config
        assert "batch_size" in config
        assert "seq_len" in config
        assert "num_stages" in config
        assert "num_micro_batches" in config
        assert "gpu_tflops" in config
        assert "interconnect_gbps" in config

    def test_async_throughput_not_zero(self) -> None:
        sim = PipelineSimulator()
        result = sim.simulate(num_nodes=4, num_layers=32)
        assert result["strategies"]["async_1f1b"]["throughput_tok_s"] > 0


class TestPipelineSimulatorRecommendation:
    """_recommend_strategy edge cases."""

    def test_single_node_no_crash(self) -> None:
        sim = PipelineSimulator()
        # Exercise the private method indirectly via simulate
        result = sim.simulate(num_nodes=1, num_layers=8)
        assert isinstance(result["recommendation"], str)

    def test_recommendation_mentions_node_count(self) -> None:
        sim = PipelineSimulator()
        result = sim.simulate(num_nodes=2, num_layers=32)
        assert "2" in result["recommendation"] or "nodes" in result["recommendation"]

    def test_recommendation_for_many_nodes(self) -> None:
        sim = PipelineSimulator()
        result = sim.simulate(num_nodes=16, num_layers=80)
        assert isinstance(result["recommendation"], str)
        assert len(result["recommendation"]) > 20


class TestPipelineSimulatorEdgeCases:
    """Corner and boundary cases."""

    def test_one_node_one_layer(self) -> None:
        sim = PipelineSimulator()
        result = sim.simulate(num_nodes=1, num_layers=1)
        assert result["config"]["layers_per_node"] == 1.0
        assert result["config"]["num_nodes"] == 1

    def test_maximum_nodes(self) -> None:
        sim = PipelineSimulator()
        result = sim.simulate(num_nodes=256, num_layers=80)
        assert result["config"]["num_nodes"] == 256
        # num_stages should be based on log2(256) = 8
        assert result["config"]["num_stages"] == 8

    def test_zero_batch_size_raises_zero_division(self) -> None:
        """batch_size=0 triggers ZeroDivisionError in seq_latency_ms / batch_size."""
        sim = PipelineSimulator()
        with pytest.raises(ZeroDivisionError):
            sim.simulate(num_nodes=4, num_layers=32, batch_size=0)

    def test_zero_seq_len_does_not_crash(self) -> None:
        sim = PipelineSimulator()
        result = sim.simulate(num_nodes=4, num_layers=32, seq_len=0)
        assert result["config"]["seq_len"] == 0

    def test_very_large_activation_ratio(self) -> None:
        sim = PipelineSimulator(activation_ratio=100.0)
        result = sim.simulate(num_nodes=4, num_layers=32)
        assert isinstance(result, dict)

    def test_custom_hidden_dim_changes_comm(self) -> None:
        sim_small = PipelineSimulator(hidden_dim=512)
        sim_large = PipelineSimulator(hidden_dim=16384)
        res_small = sim_small.simulate(num_nodes=2, num_layers=32)
        res_large = sim_large.simulate(num_nodes=2, num_layers=32)
        assert res_small["per_node_estimate_ms"]["comm_hidden"] < res_large["per_node_estimate_ms"]["comm_hidden"]

    def test_custom_num_heads_changes_kv_comm(self) -> None:
        sim_few = PipelineSimulator(num_heads=8)
        sim_many = PipelineSimulator(num_heads=128)
        res_few = sim_few.simulate(num_nodes=2, num_layers=32)
        res_many = sim_many.simulate(num_nodes=2, num_layers=32)
        assert res_few["per_node_estimate_ms"]["comm_kv"] < res_many["per_node_estimate_ms"]["comm_kv"]

    def test_num_stages_large_node_count(self) -> None:
        sim = PipelineSimulator()
        # num_nodes <= 4: num_stages = max(1, num_nodes) = 4
        res_4 = sim.simulate(num_nodes=4, num_layers=32)
        assert res_4["config"]["num_stages"] == 4

        # 4 < num_nodes <= 16: num_stages = max(1, int(sqrt(num_nodes)))
        res_9 = sim.simulate(num_nodes=9, num_layers=32)
        assert res_9["config"]["num_stages"] == 3  # int(sqrt(9)) = 3

        # num_nodes > 16: num_stages = max(2, int(log2(num_nodes)))
        res_32 = sim.simulate(num_nodes=32, num_layers=80)
        assert res_32["config"]["num_stages"] == 5  # int(log2(32)) = 5

    def test_per_node_total_is_sum_of_parts(self) -> None:
        sim = PipelineSimulator()
        result = sim.simulate(num_nodes=4, num_layers=32)
        estimate = result["per_node_estimate_ms"]
        expected_total = estimate["compute"] + estimate["comm_hidden"] + estimate["comm_kv"]
        assert abs(estimate["total"] - expected_total) < 0.01
