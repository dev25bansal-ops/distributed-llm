"""Pipeline strategy selection and AutoML."""

from __future__ import annotations

import collections
import threading
from enum import Enum
from typing import Any

from loguru import logger

from distllm.core.debug import is_debug_mode
from distllm.dist.pipeline.simulator import PipelineSimulator


class PipelineStrategy(Enum):
    """Pipeline execution strategies."""

    SEQUENTIAL = "sequential"
    OVERLAP = "overlap"
    ASYNC_1F1B = "async_1f1b"
    STAGED = "staged"
    DISAGGREGATED = "disaggregated"
    REDUNDANT = "redundant"


class StrategySelector:
    """Auto-selects the optimal pipeline strategy per-request.

    Uses PipelineSimulator for analytical cost modeling combined with
    real GPU profiling data and live latency measurements.
    """

    def __init__(
        self,
        model_size: str = "7B",
        hidden_dim: int = 4096,
        num_heads: int = 32,
        head_dim: int = 128,
        vocab_size: int = 32000,
    ):
        self._model_size = model_size
        self._hidden_dim = hidden_dim
        self._num_heads = num_heads
        self._head_dim = head_dim
        self._vocab_size = vocab_size
        self._strategy_latency: dict[str, collections.deque] = {
            s.value: collections.deque(maxlen=32) for s in PipelineStrategy
        }
        self._lock = threading.Lock()
        # Cached simulator — reused across requests with same node config
        self._cached_simulator: PipelineSimulator | None = None
        self._cached_node_signature: str = ""

    def _build_simulator(self, nodes: dict[str, Any]) -> PipelineSimulator:
        """Build a PipelineSimulator calibrated to current node capabilities.

        Reuses a cached simulator when the node configuration hasn't changed.
        """
        # Compute a signature from node capabilities
        sig_parts = []
        for nid in sorted(nodes.keys()):
            tf = getattr(nodes[nid], "gpu_compute_tflops", 0)
            sig_parts.append(f"{nid}:{tf}")
        sig = "|".join(sig_parts)

        if self._cached_simulator is not None and sig == self._cached_node_signature:
            return self._cached_simulator

        gpu_tflops = 312.0
        gpu_bandwidth = 600.0
        interconnect = 50.0

        tflops_vals = []
        for node in nodes.values():
            tf = getattr(node, "gpu_compute_tflops", 0)
            if tf and tf > 0:
                tflops_vals.append(tf)
        if tflops_vals:
            gpu_tflops = sum(tflops_vals) / len(tflops_vals)

        sim = PipelineSimulator(
            model_size=self._model_size,
            gpu_tflops=gpu_tflops,
            gpu_bandwidth_gbps=gpu_bandwidth,
            interconnect_gbps=interconnect,
            hidden_dim=self._hidden_dim,
            num_heads=self._num_heads,
            head_dim=self._head_dim,
            vocab_size=self._vocab_size,
        )

        self._cached_simulator = sim
        self._cached_node_signature = sig
        return sim

    def select_strategy(
        self,
        num_nodes: int,
        total_layers: int,
        batch_size: int,
        seq_len: int,
        current_load: int,
        nodes: dict[str, Any],
        enable_overlap: bool = True,
        stages_enabled: bool = True,
        use_async_pipeline: bool = False,
        redundant_enabled: bool = False,
        disaggregated_enabled: bool = False,
    ) -> PipelineStrategy:
        """Select the optimal strategy for a given request."""
        if redundant_enabled:
            return PipelineStrategy.REDUNDANT
        if disaggregated_enabled:
            return PipelineStrategy.DISAGGREGATED

        if num_nodes <= 1:
            return PipelineStrategy.SEQUENTIAL

        simulator = self._build_simulator(nodes)
        result = simulator.simulate(
            num_nodes=num_nodes,
            num_layers=total_layers,
            batch_size=batch_size,
            seq_len=seq_len,
        )

        candidates: list[tuple[PipelineStrategy, float, float]] = []
        strategies = result.get("strategies", {})

        seq_data = strategies.get("sequential", {})
        candidates.append((
            PipelineStrategy.SEQUENTIAL,
            seq_data.get("throughput_tok_s", 0),
            seq_data.get("latency_ms", 0),
        ))

        if enable_overlap and num_nodes > 1:
            ov_data = strategies.get("overlap", {})
            candidates.append((
                PipelineStrategy.OVERLAP,
                ov_data.get("throughput_tok_s", 0),
                ov_data.get("latency_ms", 0),
            ))

        if use_async_pipeline:
            async_data = strategies.get("async_1f1b", {})
            candidates.append((
                PipelineStrategy.ASYNC_1F1B,
                async_data.get("throughput_tok_s", 0),
                async_data.get("latency_ms", 0),
            ))

        if stages_enabled and num_nodes >= 4:
            staged_data = strategies.get("staged", {})
            candidates.append((
                PipelineStrategy.STAGED,
                staged_data.get("throughput_tok_s", 0),
                staged_data.get("latency_ms", 0),
            ))

        with self._lock:
            for strat, _, _ in candidates:
                history = self._strategy_latency.get(strat.value)
                if history and len(history) >= 3:
                    avg_real = sum(history) / len(history)
                    # M-05: Mutate in-place so blended latency persists
                    for i, c in enumerate(candidates):
                        if c[0] == strat and c[2] > 0:
                            alpha = 0.7
                            blended = alpha * c[2] + (1 - alpha) * avg_real
                            candidates[i] = (c[0], c[1], blended)

        if current_load > 6:
            candidates.sort(key=lambda x: (-x[1], x[2]))
        else:
            candidates.sort(key=lambda x: (x[2], -x[1]))

        best = candidates[0][0] if candidates else PipelineStrategy.SEQUENTIAL

        if is_debug_mode():
            logger.debug(
                f"[StrategySelector] nodes={num_nodes}, seq_len={seq_len}, "
                f"load={current_load} -> {best.value} "
                f"(candidates: {[(s.value, f'{tp:.0f}tok/s', f'{lat:.1f}ms') for s, tp, lat in candidates]})"
            )
        return best

    def record_strategy_latency(self, strategy: str, latency_ms: float) -> None:
        """Record observed latency for a strategy to improve future predictions."""
        with self._lock:
            if strategy in self._strategy_latency:
                self._strategy_latency[strategy].append(latency_ms)
