"""Heterogeneous device-aware scheduling."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from loguru import logger

from distllm.core.scheduler.budget import IterationBudget


class DeviceClass(str, Enum):
    """GPU device class for scheduling decisions."""
    DATA_CENTER = "data_center"
    WORKSTATION = "workstation"
    CONSUMER = "consumer"
    MOBILE = "mobile"
    CPU = "cpu"


@dataclass
class NodeCapabilityInfo:
    """Capability information for a single node."""
    node_id: str
    device_class: DeviceClass = DeviceClass.CONSUMER
    gpu_tflops: float = 0.0
    gpu_memory_gb: float = 0.0
    bandwidth_gbps: float = 0.0
    num_layers_assigned: int = 0
    current_load: float = 0.0
    is_spot: bool = False
    cost_per_hour: float = 0.0


class HeterogeneousBudgetComputer:
    """Computes iteration budgets adapted to heterogeneous GPU capabilities.

    Adjusts prefill/decode token budgets based on the slowest node
    in the pipeline to avoid straggler bottlenecks.
    """

    def __init__(self, nodes: dict[str, NodeCapabilityInfo] | None = None):
        self._nodes = nodes or {}

    def update_nodes(self, nodes: dict[str, NodeCapabilityInfo]) -> None:
        self._nodes = nodes

    def set_nodes(self, nodes: dict[str, NodeCapabilityInfo]) -> None:
        """Set the node capability map (batch_scheduler contract name)."""
        self.update_nodes(nodes)

    def stats(self) -> dict[str, Any]:
        """Return heterogeneous-scheduling statistics (batch_scheduler contract)."""
        return {
            "node_count": len(self._nodes),
            "slowest_tflops": min(
                (n.gpu_tflops for n in self._nodes.values() if n.gpu_tflops > 0),
                default=0.0,
            ),
        }

    def compute_budget(
        self,
        base_prefill_tokens: int,
        base_decode_tokens: int,
        base_batch_size: int,
        base_total_tokens: int,
    ) -> IterationBudget:
        """Adjust the budget based on node capabilities (budget_computer contract).

        Called once per scheduling iteration with the four scalar budget
        fields; returns a NEW ``IterationBudget`` scaled for cluster
        heterogeneity.  Inputs are never mutated.

        When the slowest node has less than half the TFLOPS of the
        fastest, prefill is throttled by the min/max ratio so straggler
        nodes are not starved of decode slots.
        """
        if self._nodes:
            tflops = [n.gpu_tflops for n in self._nodes.values() if n.gpu_tflops > 0]
            # Nodes may be registered without compute info (e.g. bandwidth
            # only) — an empty tflops list must pass the budget through
            # instead of raising ValueError from min()/max().
            if tflops:
                min_tflops = min(tflops)
                max_tflops = max(tflops)
                if max_tflops > 0 and min_tflops / max_tflops < 0.5:
                    # Significant heterogeneity — reduce prefill to avoid straggler
                    ratio = min_tflops / max_tflops
                    base_prefill_tokens = int(base_prefill_tokens * ratio)
                    logger.debug(
                        f"Heterogeneous budget: prefill scaled by {ratio:.2f} "
                        f"(slowest={min_tflops:.0f} TFLOPS)"
                    )

        return IterationBudget(
            max_prefill_tokens=base_prefill_tokens,
            max_decode_tokens=base_decode_tokens,
            max_batch_size=base_batch_size,
            max_total_tokens=base_total_tokens,
        )
