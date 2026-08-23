"""What-If engine for cluster capacity planning.

Answers questions about hypothetical cluster changes:
- Adding/removing nodes
- Upgrading node hardware
- Changing batch sizes

Each projection returns estimated performance delta and confidence.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

from distllm.dist.simulation.cluster_simulator import (
    ClusterSimulator,
    ModelConfig,
    NodeSpec,
    SimulatedPipelineResult,
    get_model_preset,
)


@dataclass
class ProjectedChange:
    """Projected performance impact of a cluster change."""

    description: str
    current_latency_ms: float = 0.0
    projected_latency_ms: float = 0.0
    current_throughput_tok_s: float = 0.0
    projected_throughput_tok_s: float = 0.0

    @property
    def latency_delta_pct(self) -> float:
        """Relative latency change (+ = slower, - = faster)."""
        if self.current_latency_ms <= 0:
            return 0.0
        return (
            (self.projected_latency_ms - self.current_latency_ms)
            / self.current_latency_ms
            * 100.0
        )

    @property
    def throughput_delta_pct(self) -> float:
        """Relative throughput change (+ = better, - = worse)."""
        if self.current_throughput_tok_s <= 0:
            return 0.0
        return (
            (self.projected_throughput_tok_s - self.current_throughput_tok_s)
            / self.current_throughput_tok_s
            * 100.0
        )

    @property
    def improvement_score(self) -> float:
        """Single score: positive is better, negative is worse.

        Combines latency improvement and throughput improvement
        into a normalized score in [-1, 1].
        """
        lat_score = -self.latency_delta_pct / 100.0  # - means good
        tp_score = self.throughput_delta_pct / 100.0  # + means good
        return (lat_score + tp_score) / 2.0

    @property
    def is_improvement(self) -> bool:
        return self.improvement_score > 0.05


class WhatIfEngine:
    """Answer 'what if' questions about cluster changes.

    Uses the ClusterSimulator to project performance changes from
    hypothetical modifications to the cluster hardware or workload.
    """

    def __init__(self) -> None:
        self._simulator = ClusterSimulator()
        self._baseline_nodes: dict[str, NodeSpec] = {}
        self._baseline_result: SimulatedPipelineResult | None = None
        self._model_config: ModelConfig | None = None
        self._batch_size: int = 1
        self._seq_len: int = 2048

    # ── Baseline setup ───────────────────────────────────────────────────

    def set_baseline(
        self,
        nodes: list[NodeSpec] | dict[str, NodeSpec],
        model: str | ModelConfig = "LLaMA-7B",
        batch_size: int = 1,
        seq_len: int = 2048,
    ) -> SimulatedPipelineResult:
        """Set the current cluster as the baseline for comparisons.

        Args:
            nodes: The current set of nodes in the cluster.
            model: Model preset name or ModelConfig.
            batch_size: Batch size for the baseline workload.
            seq_len: Sequence length for the baseline workload.

        Returns:
            Baseline SimulatedPipelineResult.
        """
        if isinstance(nodes, dict):
            self._baseline_nodes = dict(nodes)
        else:
            self._baseline_nodes = {n.node_id: n for n in nodes}

        self._model_config = (
            get_model_preset(model) if isinstance(model, str) else model
        )
        self._batch_size = batch_size
        self._seq_len = seq_len

        self._simulator.clear_nodes()
        for spec in self._baseline_nodes.values():
            self._simulator.add_node(spec)

        self._baseline_result = self._simulator.run_pipeline(
            model=self._model_config,
            nodes=list(self._baseline_nodes.values()),
            batch_size=batch_size,
            seq_len=seq_len,
        )
        return self._baseline_result

    def baseline(self) -> SimulatedPipelineResult | None:
        """Return the stored baseline result, if set."""
        return self._baseline_result

    # ── What-if scenarios ────────────────────────────────────────────────

    def what_if_add_node(
        self,
        node_spec: NodeSpec,
    ) -> ProjectedChange:
        """Project the impact of adding a new node to the cluster.

        Args:
            node_spec: Hardware spec of the node to add.

        Returns:
            ProjectedChange with current vs. projected metrics.
        """
        self._require_baseline()

        # Clone and add the new node
        new_nodes = dict(self._baseline_nodes)
        new_nodes[node_spec.node_id] = node_spec

        self._simulator.clear_nodes()
        for spec in new_nodes.values():
            self._simulator.add_node(spec)

        projected = self._simulator.run_pipeline(
            model=self._model_config,
            nodes=list(new_nodes.values()),
            batch_size=self._batch_size,
            seq_len=self._seq_len,
        )

        # Restore baseline state
        self._simulator.clear_nodes()
        for spec in self._baseline_nodes.values():
            self._simulator.add_node(spec)

        assert self._baseline_result is not None
        return ProjectedChange(
            description=f"Add node '{node_spec.node_id}' "
            f"({node_spec.gpu_name} x{node_spec.gpu_count})",
            current_latency_ms=self._baseline_result.latency_ms,
            projected_latency_ms=projected.latency_ms,
            current_throughput_tok_s=self._baseline_result.throughput_tok_s,
            projected_throughput_tok_s=projected.throughput_tok_s,
        )

    def what_if_remove_node(
        self,
        node_id: str,
    ) -> ProjectedChange:
        """Project the impact of removing a node from the cluster.

        Args:
            node_id: The node to remove.

        Returns:
            ProjectedChange with current vs. projected metrics.
        """
        self._require_baseline()

        if node_id not in self._baseline_nodes:
            return ProjectedChange(
                description=f"Remove node '{node_id}' (not found)",
                current_latency_ms=(
                    self._baseline_result.latency_ms
                    if self._baseline_result
                    else 0.0
                ),
                projected_latency_ms=(
                    self._baseline_result.latency_ms
                    if self._baseline_result
                    else 0.0
                ),
                current_throughput_tok_s=(
                    self._baseline_result.throughput_tok_s
                    if self._baseline_result
                    else 0.0
                ),
                projected_throughput_tok_s=(
                    self._baseline_result.throughput_tok_s
                    if self._baseline_result
                    else 0.0
                ),
            )

        new_nodes = {
            nid: spec
            for nid, spec in self._baseline_nodes.items()
            if nid != node_id
        }

        if not new_nodes:
            # Removing the last node - mark as catastrophic
            assert self._baseline_result is not None
            return ProjectedChange(
                description=f"Remove node '{node_id}' (last node - cluster empty)",
                current_latency_ms=self._baseline_result.latency_ms,
                projected_latency_ms=float("inf"),
                current_throughput_tok_s=self._baseline_result.throughput_tok_s,
                projected_throughput_tok_s=0.0,
            )

        self._simulator.clear_nodes()
        for spec in new_nodes.values():
            self._simulator.add_node(spec)

        projected = self._simulator.run_pipeline(
            model=self._model_config,
            nodes=list(new_nodes.values()),
            batch_size=self._batch_size,
            seq_len=self._seq_len,
        )

        # Restore baseline
        self._simulator.clear_nodes()
        for spec in self._baseline_nodes.values():
            self._simulator.add_node(spec)

        assert self._baseline_result is not None
        return ProjectedChange(
            description=f"Remove node '{node_id}'",
            current_latency_ms=self._baseline_result.latency_ms,
            projected_latency_ms=projected.latency_ms,
            current_throughput_tok_s=self._baseline_result.throughput_tok_s,
            projected_throughput_tok_s=projected.throughput_tok_s,
        )

    def what_if_upgrade(
        self,
        node_id: str,
        new_spec: NodeSpec,
    ) -> ProjectedChange:
        """Project the impact of upgrading a node's hardware.

        Args:
            node_id: The node to upgrade.
            new_spec: The new hardware spec for the node.

        Returns:
            ProjectedChange with current vs. projected metrics.
        """
        self._require_baseline()

        if node_id not in self._baseline_nodes:
            return ProjectedChange(
                description=f"Upgrade node '{node_id}' (not found - treating as add)",
                current_latency_ms=(
                    self._baseline_result.latency_ms
                    if self._baseline_result
                    else 0.0
                ),
                projected_latency_ms=(
                    self._baseline_result.latency_ms
                    if self._baseline_result
                    else 0.0
                ),
                current_throughput_tok_s=(
                    self._baseline_result.throughput_tok_s
                    if self._baseline_result
                    else 0.0
                ),
                projected_throughput_tok_s=(
                    self._baseline_result.throughput_tok_s
                    if self._baseline_result
                    else 0.0
                ),
            )

        old_spec = self._baseline_nodes[node_id]

        # Build new node set
        new_nodes = dict(self._baseline_nodes)
        new_nodes[node_id] = new_spec

        self._simulator.clear_nodes()
        for spec in new_nodes.values():
            self._simulator.add_node(spec)

        projected = self._simulator.run_pipeline(
            model=self._model_config,
            nodes=list(new_nodes.values()),
            batch_size=self._batch_size,
            seq_len=self._seq_len,
        )

        # Restore baseline
        self._simulator.clear_nodes()
        for spec in self._baseline_nodes.values():
            self._simulator.add_node(spec)

        assert self._baseline_result is not None
        return ProjectedChange(
            description=f"Upgrade node '{node_id}': "
            f"{old_spec.gpu_name} -> {new_spec.gpu_name}",
            current_latency_ms=self._baseline_result.latency_ms,
            projected_latency_ms=projected.latency_ms,
            current_throughput_tok_s=self._baseline_result.throughput_tok_s,
            projected_throughput_tok_s=projected.throughput_tok_s,
        )

    def what_if_change_batch_size(
        self,
        new_batch_size: int,
    ) -> ProjectedChange:
        """Project the impact of changing the batch size.

        Args:
            new_batch_size: The new batch size to evaluate.

        Returns:
            ProjectedChange with current vs. projected metrics.
        """
        self._require_baseline()

        projected = self._simulator.run_pipeline(
            model=self._model_config,
            nodes=list(self._baseline_nodes.values()),
            batch_size=new_batch_size,
            seq_len=self._seq_len,
        )

        assert self._baseline_result is not None
        return ProjectedChange(
            description=f"Change batch size: "
            f"{self._batch_size} -> {new_batch_size}",
            current_latency_ms=self._baseline_result.latency_ms,
            projected_latency_ms=projected.latency_ms,
            current_throughput_tok_s=self._baseline_result.throughput_tok_s,
            projected_throughput_tok_s=projected.throughput_tok_s,
        )

    def what_if_change_seq_len(
        self,
        new_seq_len: int,
    ) -> ProjectedChange:
        """Project the impact of changing the sequence length.

        Args:
            new_seq_len: The new sequence length to evaluate.

        Returns:
            ProjectedChange with current vs. projected metrics.
        """
        self._require_baseline()

        projected = self._simulator.run_pipeline(
            model=self._model_config,
            nodes=list(self._baseline_nodes.values()),
            batch_size=self._batch_size,
            seq_len=new_seq_len,
        )

        assert self._baseline_result is not None
        return ProjectedChange(
            description=f"Change sequence length: "
            f"{self._seq_len} -> {new_seq_len}",
            current_latency_ms=self._baseline_result.latency_ms,
            projected_latency_ms=projected.latency_ms,
            current_throughput_tok_s=self._baseline_result.throughput_tok_s,
            projected_throughput_tok_s=projected.throughput_tok_s,
        )

    # ── Batch projections ────────────────────────────────────────────────

    def batch_what_if(
        self,
        changes: list[dict[str, Any]],
    ) -> list[ProjectedChange]:
        """Run multiple what-if projections in sequence.

        Each entry in ``changes`` should be a dict with:
          - ``type``: one of "add_node", "remove_node", "upgrade",
                      "change_batch_size", "change_seq_len"
          - Additional params as required by the specific method.

        Args:
            changes: List of change descriptions.

        Returns:
            List of ProjectedChange results.
        """
        results: list[ProjectedChange] = []
        for change in changes:
            change_type = change.get("type", "")
            try:
                if change_type == "add_node":
                    spec = change.get("spec")
                    if isinstance(spec, dict):
                        spec = NodeSpec(**spec)
                    if spec is not None:
                        results.append(self.what_if_add_node(spec))
                elif change_type == "remove_node":
                    node_id = change.get("node_id", "")
                    results.append(self.what_if_remove_node(node_id))
                elif change_type == "upgrade":
                    node_id = change.get("node_id", "")
                    new_spec = change.get("new_spec")
                    if isinstance(new_spec, dict):
                        new_spec = NodeSpec(**new_spec)
                    if new_spec is not None:
                        results.append(
                            self.what_if_upgrade(node_id, new_spec)
                        )
                elif change_type == "change_batch_size":
                    new_size = change.get("new_batch_size", 1)
                    results.append(
                        self.what_if_change_batch_size(new_size)
                    )
                elif change_type == "change_seq_len":
                    new_len = change.get("new_seq_len", 2048)
                    results.append(
                        self.what_if_change_seq_len(new_len)
                    )
            except Exception as exc:
                results.append(
                    ProjectedChange(
                        description=(
                            f"Error in '{change_type}': {exc}"
                        ),
                    )
                )
        return results

    # ── Internal helpers ─────────────────────────────────────────────────

    def _require_baseline(self) -> None:
        """Ensure a baseline has been set before running what-if scenarios."""
        if self._baseline_result is None:
            raise RuntimeError(
                "No baseline set. Call set_baseline() first."
            )
        if self._model_config is None:
            raise RuntimeError(
                "No model config. Call set_baseline() first."
            )

    def summary(self) -> str:
        """Human-readable summary of the what-if engine state."""
        lines: list[str] = ["WhatIfEngine"]
        if self._baseline_result is not None:
            assert self._model_config is not None
            lines.append(
                f"  Baseline: {self._model_config.name}, "
                f"batch={self._batch_size}, seq_len={self._seq_len}"
            )
            lines.append(
                f"  Nodes: {len(self._baseline_nodes)}"
            )
            lines.append(
                f"  Latency: {self._baseline_result.latency_ms} ms"
            )
            lines.append(
                f"  Throughput: {self._baseline_result.throughput_tok_s} tok/s"
            )
        else:
            lines.append("  No baseline set")
        return "\n".join(lines)
