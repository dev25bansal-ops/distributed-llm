"""Partition quality benchmark suite.

Standardized benchmarks for evaluating and comparing partition
strategies.  Provides reproducible test scenarios, competitive
comparison, and regression detection.

Typical usage::

    suite = PartitionBenchmarkSuite()
    results = suite.run_all()
    print(results.summary())
    suite.save_results("benchmark_results.json")
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger

from distllm.dist.partition.cost_model import PartitionCostModel
from distllm.dist.partition.optimizer import PartitionOptimizer, PartitionSolution
from distllm.dist.partition.profiles import GPUProfile, GPUProfiler, LayerWeights
from distllm.dist.partition.topology import LinkProfile, TopologyGraph


@dataclass
class BenchmarkScenario:
    """A reproducible partition benchmark scenario."""
    name: str
    description: str
    num_layers: int
    hidden_size: int
    intermediate_size: int
    num_heads: int
    head_dim: int
    vocab_size: int
    batch_size: int
    seq_len: int
    nodes: list[dict[str, Any]]
    expected_min_improvement_pct: float = 0.0


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""
    scenario: str
    strategy: str
    max_latency_ms: float
    throughput_tok_s: float
    num_nodes: int
    total_memory_gb: float
    solve_time_ms: float
    improvement_over_equal_pct: float = 0.0
    meets_expectations: bool = True


@dataclass
class BenchmarkSuiteResult:
    """Aggregated results from all benchmark scenarios."""
    results: list[BenchmarkResult] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    total_time_s: float = 0.0

    def summary(self) -> str:
        lines = [
            f"Benchmark Suite: {len(self.results)} results",
            f"Total time: {self.total_time_s:.1f}s",
            "",
            f"{'Scenario':<35} {'Strategy':<15} {'Latency':>10} {'Throughput':>12} {'Improvement':>12} {'Pass':>6}",
            "-" * 95,
        ]
        for r in self.results:
            lines.append(
                f"{r.scenario:<35} {r.strategy:<15} {r.max_latency_ms:>9.1f}ms "
                f"{r.throughput_tok_s:>10.0f}t/s {r.improvement_over_equal_pct:>10.1f}% "
                f"{'  OK' if r.meets_expectations else 'FAIL':>6}"
            )
        return "\n".join(lines)

    def passed(self) -> int:
        return sum(1 for r in self.results if r.meets_expectations)

    def failed(self) -> int:
        return sum(1 for r in self.results if not r.meets_expectations)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "total_time_s": self.total_time_s,
            "passed": self.passed(),
            "failed": self.failed(),
            "results": [
                {
                    "scenario": r.scenario,
                    "strategy": r.strategy,
                    "max_latency_ms": r.max_latency_ms,
                    "throughput_tok_s": r.throughput_tok_s,
                    "num_nodes": r.num_nodes,
                    "total_memory_gb": r.total_memory_gb,
                    "solve_time_ms": r.solve_time_ms,
                    "improvement_over_equal_pct": r.improvement_over_equal_pct,
                    "meets_expectations": r.meets_expectations,
                }
                for r in self.results
            ],
        }


# Standard benchmark scenarios
STANDARD_SCENARIOS: list[BenchmarkScenario] = [
    BenchmarkScenario(
        name="7B_two_identical",
        description="7B model on 2 identical GPUs",
        num_layers=32,
        hidden_size=4096,
        intermediate_size=11008,
        num_heads=32,
        head_dim=128,
        vocab_size=32000,
        batch_size=1,
        seq_len=4096,
        nodes=[
            {"node_id": "gpu-0", "gpu_name": "A100", "tflops": 312.0, "mem_bw": 2039.0, "vram_gb": 80},
            {"node_id": "gpu-1", "gpu_name": "A100", "tflops": 312.0, "mem_bw": 2039.0, "vram_gb": 80},
        ],
        expected_min_improvement_pct=5.0,
    ),
    BenchmarkScenario(
        name="13B_two_identical",
        description="13B model on 2 identical GPUs",
        num_layers=40,
        hidden_size=5120,
        intermediate_size=13824,
        num_heads=40,
        head_dim=128,
        vocab_size=32000,
        batch_size=1,
        seq_len=4096,
        nodes=[
            {"node_id": "gpu-0", "gpu_name": "A100", "tflops": 312.0, "mem_bw": 2039.0, "vram_gb": 80},
            {"node_id": "gpu-1", "gpu_name": "A100", "tflops": 312.0, "mem_bw": 2039.0, "vram_gb": 80},
        ],
        expected_min_improvement_pct=8.0,
    ),
    BenchmarkScenario(
        name="70B_four_heterogeneous",
        description="70B model on 4 heterogeneous GPUs",
        num_layers=80,
        hidden_size=8192,
        intermediate_size=28672,
        num_heads=64,
        head_dim=128,
        vocab_size=32000,
        batch_size=1,
        seq_len=4096,
        nodes=[
            {"node_id": "gpu-0", "gpu_name": "H100", "tflops": 989.0, "mem_bw": 3350.0, "vram_gb": 80},
            {"node_id": "gpu-1", "gpu_name": "A100", "tflops": 312.0, "mem_bw": 2039.0, "vram_gb": 80},
            {"node_id": "gpu-2", "gpu_name": "A100", "tflops": 312.0, "mem_bw": 2039.0, "vram_gb": 40},
            {"node_id": "gpu-3", "gpu_name": "L4", "tflops": 121.0, "mem_bw": 300.0, "vram_gb": 24},
        ],
        expected_min_improvement_pct=15.0,
    ),
    BenchmarkScenario(
        name="7B_three_heterogeneous",
        description="7B model on 3 GPUs (fast, medium, slow)",
        num_layers=32,
        hidden_size=4096,
        intermediate_size=11008,
        num_heads=32,
        head_dim=128,
        vocab_size=32000,
        batch_size=1,
        seq_len=4096,
        nodes=[
            {"node_id": "gpu-0", "gpu_name": "H100", "tflops": 989.0, "mem_bw": 3350.0, "vram_gb": 80},
            {"node_id": "gpu-1", "gpu_name": "A100", "tflops": 312.0, "mem_bw": 2039.0, "vram_gb": 80},
            {"node_id": "gpu-2", "gpu_name": "T4", "tflops": 65.0, "mem_bw": 320.0, "vram_gb": 16},
        ],
        expected_min_improvement_pct=20.0,
    ),
    BenchmarkScenario(
        name="13B_batch8",
        description="13B model with batch_size=8",
        num_layers=40,
        hidden_size=5120,
        intermediate_size=13824,
        num_heads=40,
        head_dim=128,
        vocab_size=32000,
        batch_size=8,
        seq_len=2048,
        nodes=[
            {"node_id": "gpu-0", "gpu_name": "H100", "tflops": 989.0, "mem_bw": 3350.0, "vram_gb": 80},
            {"node_id": "gpu-1", "gpu_name": "H100", "tflops": 989.0, "mem_bw": 3350.0, "vram_gb": 80},
        ],
        expected_min_improvement_pct=5.0,
    ),
    BenchmarkScenario(
        name="7B_single_node",
        description="7B model on a single GPU (baseline)",
        num_layers=32,
        hidden_size=4096,
        intermediate_size=11008,
        num_heads=32,
        head_dim=128,
        vocab_size=32000,
        batch_size=1,
        seq_len=4096,
        nodes=[
            {"node_id": "gpu-0", "gpu_name": "A100", "tflops": 312.0, "mem_bw": 2039.0, "vram_gb": 80},
        ],
        expected_min_improvement_pct=0.0,
    ),
    BenchmarkScenario(
        name="7B_consumer_gpus",
        description="7B model on consumer RTX GPUs",
        num_layers=32,
        hidden_size=4096,
        intermediate_size=11008,
        num_heads=32,
        head_dim=128,
        vocab_size=32000,
        batch_size=1,
        seq_len=4096,
        nodes=[
            {"node_id": "gpu-0", "gpu_name": "RTX 4090", "tflops": 330.0, "mem_bw": 1008.0, "vram_gb": 24},
            {"node_id": "gpu-1", "gpu_name": "RTX 3090", "tflops": 142.0, "mem_bw": 936.0, "vram_gb": 24},
        ],
        expected_min_improvement_pct=10.0,
    ),
    BenchmarkScenario(
        name="7B_apple_silicon",
        description="7B model on Apple Silicon",
        num_layers=32,
        hidden_size=4096,
        intermediate_size=11008,
        num_heads=32,
        head_dim=128,
        vocab_size=32000,
        batch_size=1,
        seq_len=4096,
        nodes=[
            {"node_id": "gpu-0", "gpu_name": "Apple M4 Max", "tflops": 18.4, "mem_bw": 816.0, "vram_gb": 40},
        ],
        expected_min_improvement_pct=0.0,
    ),
]


class PartitionBenchmarkSuite:
    """Runs standardized partition benchmarks.

    Args:
        scenarios: Custom scenarios (uses STANDARD_SCENARIOS if None).
        include_strategies: Strategies to benchmark.
    """

    def __init__(
        self,
        scenarios: list[BenchmarkScenario] | None = None,
        include_strategies: list[str] | None = None,
    ):
        self._scenarios = scenarios or STANDARD_SCENARIOS
        self._strategies = include_strategies or ["dp_minimax", "equal_split", "proportional_split"]

    def run_all(self) -> BenchmarkSuiteResult:
        """Run all benchmark scenarios."""
        t0 = time.time()
        results = BenchmarkSuiteResult()

        for scenario in self._scenarios:
            logger.info(f"Running benchmark: {scenario.name}")
            scenario_results = self._run_scenario(scenario)
            results.results.extend(scenario_results)

        results.total_time_s = round(time.time() - t0, 2)
        logger.info(
            f"Benchmark suite complete: {results.passed()} passed, "
            f"{results.failed()} failed in {results.total_time_s:.1f}s"
        )
        return results

    def run_scenario(self, name: str) -> list[BenchmarkResult]:
        """Run a single named scenario."""
        scenario = next((s for s in self._scenarios if s.name == name), None)
        if scenario is None:
            raise ValueError(f"Unknown scenario: {name}")
        return self._run_scenario(scenario)

    def list_scenarios(self) -> list[dict[str, Any]]:
        """List available benchmark scenarios."""
        return [
            {
                "name": s.name,
                "description": s.description,
                "num_layers": s.num_layers,
                "nodes": len(s.nodes),
                "batch_size": s.batch_size,
            }
            for s in self._scenarios
        ]

    def _run_scenario(self, scenario: BenchmarkScenario) -> list[BenchmarkResult]:
        gpu_profiles = self._build_gpu_profiles(scenario)
        layer_weights = self._estimate_layers(scenario)
        topology = self._build_topology(scenario)

        cost_model = PartitionCostModel(gpu_profiles, layer_weights, topology)
        node_ids = [n["node_id"] for n in scenario.nodes]

        optimizer = PartitionOptimizer(
            cost_model=cost_model,
            node_ids=node_ids,
            batch_size=scenario.batch_size,
            seq_len=scenario.seq_len,
            allow_oom=False,
        )

        total_layers = len(layer_weights)
        results: list[BenchmarkResult] = []

        comparison = optimizer.compare_strategies(
            total_layers, scenario.batch_size, scenario.seq_len,
        )

        for strategy in self._strategies:
            if strategy == "dp_minimax":
                t0 = time.time()
                solution = optimizer.solve(total_layers)
                solve_ms = (time.time() - t0) * 1000

                improvement = self._parse_improvement(comparison.get("improvement_over_equal", "0%"))

                results.append(BenchmarkResult(
                    scenario=scenario.name,
                    strategy=strategy,
                    max_latency_ms=solution.max_node_time_ms,
                    throughput_tok_s=solution.estimated_throughput_tok_s,
                    num_nodes=solution.num_nodes,
                    total_memory_gb=sum(
                        cost_model.evaluate(p.node_id, p.start_layer, p.end_layer, scenario.batch_size, scenario.seq_len).memory_bytes
                        for p in solution.points
                    ) / (1024**3),
                    solve_time_ms=round(solve_ms, 2),
                    improvement_over_equal_pct=improvement,
                    meets_expectations=improvement >= scenario.expected_min_improvement_pct,
                ))
            elif strategy in comparison:
                data = comparison[strategy]
                results.append(BenchmarkResult(
                    scenario=scenario.name,
                    strategy=strategy,
                    max_latency_ms=data.get("max_latency_ms", 0),
                    throughput_tok_s=data.get("throughput", 0),
                    num_nodes=len(node_ids),
                    total_memory_gb=0,
                    solve_time_ms=0,
                    improvement_over_equal_pct=0,
                    meets_expectations=True,
                ))

        return results

    def _build_gpu_profiles(self, scenario: BenchmarkScenario) -> dict[str, GPUProfile]:
        profiles: dict[str, GPUProfile] = {}
        for i, node in enumerate(scenario.nodes):
            profiles[node["node_id"]] = GPUProfile(
                gpu_id=i,
                name=node["gpu_name"],
                total_memory_bytes=int(node["vram_gb"] * 1024**3),
                compute_tflops=node["tflops"],
                memory_bandwidth_gbps=node["mem_bw"],
            )
        return profiles

    def _estimate_layers(self, scenario: BenchmarkScenario) -> list[LayerWeights]:
        profiler = GPUProfiler()
        return profiler.estimate_layer_weights(
            hidden_size=scenario.hidden_size,
            intermediate_size=scenario.intermediate_size,
            num_layers=scenario.num_layers,
            num_heads=scenario.num_heads,
            head_dim=scenario.head_dim,
            vocab_size=scenario.vocab_size,
        )

    def _build_topology(self, scenario: BenchmarkScenario) -> TopologyGraph:
        node_ids = [n["node_id"] for n in scenario.nodes]
        links: list[LinkProfile] = []
        for i, n1 in enumerate(node_ids):
            for j, n2 in enumerate(node_ids):
                if i < j:
                    links.append(LinkProfile(
                        source=n1, target=n2,
                        bandwidth_gbps=25.0,
                        latency_us=500.0,
                    ))
        return TopologyGraph(
            node_ids=node_ids,
            gpu_counts={n["node_id"]: 1 for n in scenario.nodes},
            links=links,
        )

    @staticmethod
    def _parse_improvement(s: str) -> float:
        try:
            return float(s.replace("%", "").strip())
        except (ValueError, AttributeError):
            return 0.0

    @staticmethod
    def save_results(results: BenchmarkSuiteResult, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(results.to_dict(), f, indent=2)
        logger.info(f"Benchmark results saved to {path}")

    @staticmethod
    def load_results(path: str | Path) -> dict[str, Any]:
        with open(path) as f:
            return json.load(f)

    @staticmethod
    def compare_runs(
        current: BenchmarkSuiteResult,
        baseline: dict[str, Any],
        tolerance_pct: float = 5.0,
    ) -> dict[str, Any]:
        """Compare current results against a baseline for regression detection."""
        baseline_map: dict[str, float] = {}
        for r in baseline.get("results", []):
            key = f"{r['scenario']}:{r['strategy']}"
            baseline_map[key] = r["max_latency_ms"]

        regressions: list[dict[str, Any]] = []
        improvements: list[dict[str, Any]] = []

        for r in current.results:
            key = f"{r.scenario}:{r.strategy}"
            baseline_ms = baseline_map.get(key)
            if baseline_ms is None:
                continue

            diff_pct = ((r.max_latency_ms - baseline_ms) / max(baseline_ms, 0.001)) * 100

            if diff_pct > tolerance_pct:
                regressions.append({
                    "scenario": r.scenario,
                    "strategy": r.strategy,
                    "baseline_ms": baseline_ms,
                    "current_ms": r.max_latency_ms,
                    "regression_pct": round(diff_pct, 1),
                })
            elif diff_pct < -tolerance_pct:
                improvements.append({
                    "scenario": r.scenario,
                    "strategy": r.strategy,
                    "baseline_ms": baseline_ms,
                    "current_ms": r.max_latency_ms,
                    "improvement_pct": round(-diff_pct, 1),
                })

        return {
            "regressions": regressions,
            "improvements": improvements,
            "stable": len(regressions) == 0,
            "summary": (
                f"{'PASS' if not regressions else 'FAIL'}: "
                f"{len(regressions)} regressions, "
                f"{len(improvements)} improvements"
            ),
        }
