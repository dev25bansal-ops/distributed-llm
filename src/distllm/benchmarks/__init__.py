"""DistLLM benchmark suite.

Provides standardized benchmarks for measuring throughput, latency,
and scaling across different model sizes, hardware configurations,
and parallelism strategies.
"""

from distllm.benchmarks.scaling import ScalingBenchmark

__all__ = ["ScalingBenchmark"]
