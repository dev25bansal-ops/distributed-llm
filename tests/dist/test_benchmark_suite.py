"""Real tests for partition/benchmark_suite — PartitionBenchmarkSuite."""
from __future__ import annotations


class TestBenchmarkSuite:
    def test_suite_init(self):
        from distllm.dist.partition.benchmark_suite import PartitionBenchmarkSuite

        suite = PartitionBenchmarkSuite()
        assert suite is not None

    def test_benchmark_result(self):
        from distllm.dist.partition.benchmark_suite import BenchmarkResult

        result = BenchmarkResult(
            scenario="test", strategy="dp",
            max_latency_ms=100.0, throughput_tok_s=1000.0,
            num_nodes=4, total_memory_gb=320.0, solve_time_ms=500.0,
        )
        assert result.scenario == "test"

    def test_benchmark_scenario(self):
        from distllm.dist.partition.benchmark_suite import BenchmarkScenario

        nodes = [{"gpu_memory_gb": 80, "bandwidth_gbps": 2000, "tflops": 312}] * 4
        scenario = BenchmarkScenario(
            name="uniform-4xA100",
            description="4x A100 test",
            num_layers=32, hidden_size=4096,
            intermediate_size=11008, num_heads=32,
            head_dim=128, vocab_size=32000,
            batch_size=1, seq_len=4096,
            nodes=nodes,
        )
        assert scenario.name == "uniform-4xA100"
