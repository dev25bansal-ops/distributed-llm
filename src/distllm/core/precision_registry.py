"""Precision registry for heterogeneous precision serving.

Tracks per-node precision assignments and capabilities for
mixed-hardware cluster serving (FP16/INT8/FP8/INT4).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from enum import Enum

from loguru import logger


class Precision(str, Enum):
    """Supported precision types for model serving."""
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    INT8 = "int8"
    INT4 = "int4"
    FP8 = "fp8"
    FP8_E4M3 = "fp8_e4m3"
    FP8_E5M2 = "fp8_e5m2"


@dataclass
class NodePrecision:
    """Precision capabilities and assignment for a node."""
    node_id: str
    precision: Precision = Precision.FP16
    gpu_type: str = ""
    vram_gb: float = 0.0
    capabilities: list[Precision] = field(default_factory=lambda: [Precision.FP16])
    compute_capability: str = ""  # e.g., "8.0" for Ampere, "9.0" for Hopper
    assigned_at: float = 0.0

    def supports(self, precision: Precision) -> bool:
        """Check if this node supports the given precision."""
        return precision in self.capabilities


class PrecisionRegistry:
    """Thread-safe registry for per-node precision assignments.

    The coordinator assigns precision to nodes based on their GPU specs
    at registration time. This registry tracks assignments and enables
    precision-aware routing.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._nodes: dict[str, NodePrecision] = {}

    def assign_precision(
        self,
        node_id: str,
        precision: Precision,
        gpu_type: str = "",
        vram_gb: float = 0.0,
        capabilities: list[Precision] | None = None,
        compute_capability: str = "",
    ) -> None:
        """Assign precision to a node.

        Args:
            node_id: Unique node identifier.
            precision: Assigned serving precision.
            gpu_type: GPU model name (e.g., "H100", "A100", "RTX 4090").
            vram_gb: Available VRAM in GB.
            capabilities: List of precisions this node can serve.
            compute_capability: CUDA compute capability string.
        """
        import time

        if capabilities is None:
            capabilities = self._infer_capabilities(precision, compute_capability)

        with self._lock:
            self._nodes[node_id] = NodePrecision(
                node_id=node_id,
                precision=precision,
                gpu_type=gpu_type,
                vram_gb=vram_gb,
                capabilities=capabilities,
                compute_capability=compute_capability,
                assigned_at=time.time(),
            )
        logger.info(
            f"Assigned precision {precision.value} to node {node_id} "
            f"({gpu_type}, {vram_gb}GB VRAM)"
        )

    def get_node_precision(self, node_id: str) -> NodePrecision | None:
        """Get precision assignment for a node."""
        with self._lock:
            return self._nodes.get(node_id)

    def get_nodes_with_precision(self, precision: Precision) -> list[str]:
        """Get all nodes that can serve at the given precision."""
        with self._lock:
            return [
                nid for nid, np in self._nodes.items()
                if np.supports(precision)
            ]

    def get_compatible_nodes(self, required_precision: Precision) -> list[str]:
        """Get nodes that can handle the required precision level."""
        with self._lock:
            return [
                nid for nid, np in self._nodes.items()
                if np.precision == required_precision or np.supports(required_precision)
            ]

    def select_node_for_precision(
        self,
        required_precision: Precision,
        prefer_lowest_load: bool = True,
    ) -> str | None:
        """Select a node that can serve the required precision.

        Args:
            required_precision: Minimum precision needed.
            prefer_lowest_load: If True, prefer nodes with more available capacity.

        Returns:
            Node ID of the best candidate, or None if no compatible node.
        """
        candidates = self.get_compatible_nodes(required_precision)
        if not candidates:
            return None

        # For now, return first candidate (load-aware selection
        # would integrate with the geo router's LoadReporter)
        return candidates[0]

    def list_all(self) -> dict[str, NodePrecision]:
        """Return all precision assignments."""
        with self._lock:
            return dict(self._nodes)

    def remove_node(self, node_id: str) -> None:
        """Remove a node from the registry."""
        with self._lock:
            self._nodes.pop(node_id, None)
        logger.info(f"Removed node {node_id} from precision registry")

    @staticmethod
    def _infer_capabilities(
        precision: Precision,
        compute_capability: str,
    ) -> list[Precision]:
        """Infer a node's precision capabilities from its primary precision and CC."""
        caps: list[Precision] = [precision]

        # All GPUs support FP32 and FP16
        if Precision.FP32 not in caps:
            caps.append(Precision.FP32)
        if Precision.FP16 not in caps:
            caps.append(Precision.FP16)

        # BF16 requires CC >= 8.0 (Ampere+)
        if compute_capability:
            cc_major = int(compute_capability.split(".")[0])
            if cc_major >= 8 and Precision.BF16 not in caps:
                caps.append(Precision.BF16)
            # FP8 requires CC >= 9.0 (Hopper+)
            if cc_major >= 9:
                caps.extend([Precision.FP8, Precision.FP8_E4M3, Precision.FP8_E5M2])

        # INT8/INT4 generally available on CC >= 7.0
        if compute_capability:
            cc_major = int(compute_capability.split(".")[0])
            if cc_major >= 7:
                if Precision.INT8 not in caps:
                    caps.append(Precision.INT8)
                if Precision.INT4 not in caps:
                    caps.append(Precision.INT4)

        return caps
