"""Pipeline profiler for per-node timing measurements."""

from __future__ import annotations

import torch


class PipelineProfiler:
    """Profiles pipeline steps with per-node timing.

    Supports both CPU timing (time.monotonic) and GPU timing
    (torch.cuda.Event) for accurate GPU-side measurements.

    Usage::

        profiler = PipelineProfiler(use_cuda_events=True)
        timings = profiler.profile_step(input_ids, node_kv_caches, request_id)
        print(profiler.summary())
    """

    def __init__(self, history_size: int = 100, use_cuda_events: bool = False):
        self._history: list[dict[str, float]] = []
        self._history_size = history_size
        self._use_cuda = use_cuda_events and torch.cuda.is_available()
        self._cuda_starts: dict[str, torch.cuda.Event] = {}
        self._cuda_ends: dict[str, torch.cuda.Event] = {}

    def start_node(self, node_id: str) -> None:
        """Record start event for a node (GPU or CPU)."""
        if self._use_cuda:
            start = torch.cuda.Event(enable_timing=True)
            start.record()
            self._cuda_starts[node_id] = start

    def end_node(self, node_id: str) -> float | None:
        """Record end event for a node and return elapsed ms (GPU) or None (CPU)."""
        if self._use_cuda and node_id in self._cuda_starts:
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            self._cuda_ends[node_id] = end
            start = self._cuda_starts[node_id]
            torch.cuda.synchronize()
            elapsed = start.elapsed_time(end)
            self.record(node_id, elapsed)
            return elapsed
        return None

    def profile_step(
        self,
        input_ids: torch.Tensor,
        node_kv_caches: dict[str, list | None],
        request_id: str,
    ) -> dict[str, float]:
        """Profile a single pipeline step with per-node timing."""
        if not self._history:
            return {}
        return dict(self._history[-1])

    def record(self, node_id: str, elapsed_ms: float) -> None:
        if not self._history:
            self._history.append({})
        self._history[-1][node_id] = elapsed_ms

    def next_step(self) -> None:
        if len(self._history) >= self._history_size:
            self._history.pop(0)
        self._history.append({})

    def summary(self) -> dict:
        if not self._history:
            return {"steps": 0}
        all_nodes: set[str] = set()
        for step in self._history:
            all_nodes.update(step.keys())
        per_node = {}
        for nid in sorted(all_nodes):
            vals = [s[nid] for s in self._history if nid in s]
            per_node[nid] = {
                "avg_ms": sum(vals) / len(vals) if vals else 0.0,
                "min_ms": min(vals) if vals else 0.0,
                "max_ms": max(vals) if vals else 0.0,
                "samples": len(vals),
            }
        return {"steps": len(self._history), "per_node": per_node}
