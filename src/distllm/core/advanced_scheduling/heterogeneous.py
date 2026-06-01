"""Heterogeneous device-aware scheduling."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from loguru import logger


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

    def compute_budget(self, base_budget: Any) -> Any:
        """Adjust budget based on node capabilities."""
        if not self._nodes:
            return base_budget

        # Find the slowest node
        min_tflops = min(n.gpu_tflops for n in self._nodes.values() if n.gpu_tflops > 0)
        max_tflops = max(n.gpu_tflops for n in self._nodes.values() if n.gpu_tflops > 0)

        if max_tflops == 0:
            return base_budget

        # Scale down prefill budget if there's a slow node
        ratio = min_tflops / max_tflops
        if ratio < 0.5:
            # Significant heterogeneity — reduce prefill to avoid straggler
            base_budget.max_prefill_tokens = int(base_budget.max_prefill_tokens * ratio)
            logger.debug(
                f"Heterogeneous budget: prefill scaled by {ratio:.2f} "
                f"(slowest={min_tflops:.0f} TFLOPS)"
            )

        return base_budget
