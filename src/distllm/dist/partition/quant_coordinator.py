"""Distributed quantization coordinator.

Collects GPU profiles from all nodes in a cluster, runs the Adaptive
Precision Optimizer (APO), and distributes quantization plans.
Handles fallback if a node reports quant method failure.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from loguru import logger


@dataclass
class NodeProfile:
    """Profile collected from a remote node."""
    node_id: str
    gpu_name: str = ""
    total_memory_bytes: int = 0
    compute_tflops: float = 0.0
    bandwidth_gbps: float = 0.0
    compute_capability: float = 0.0
    is_hopper_or_newer: bool = False
    status: str = "online"
    last_heartbeat: float = 0.0
    error: str = ""


@dataclass
class NodeQuantAssignment:
    """Quantization assignment sent to a node."""
    node_id: str
    quant_method: str
    activation_quant: str = "none"
    kv_cache_bits: str = "none"
    max_quality_loss: float = 0.05
    mixed_precision_plan: dict[str, Any] = field(default_factory=dict)


@dataclass
class CoordinatorState:
    """Full coordinator state."""
    nodes: dict[str, NodeProfile] = field(default_factory=dict)
    assignments: dict[str, NodeQuantAssignment] = field(default_factory=dict)
    plan_json: str = ""
    last_update: float = 0.0
    model_name: str = ""
    model_size_bytes: int = 0
    num_layers: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": {nid: {
                "gpu_name": n.gpu_name,
                "total_memory_bytes": n.total_memory_bytes,
                "compute_tflops": n.compute_tflops,
                "status": n.status,
            } for nid, n in self.nodes.items()},
            "assignments": {nid: {
                "quant_method": a.quant_method,
                "activation_quant": a.activation_quant,
                "kv_cache_bits": a.kv_cache_bits,
            } for nid, a in self.assignments.items()},
            "model_name": self.model_name,
            "model_size_bytes": self.model_size_bytes,
            "num_layers": self.num_layers,
            "last_update": self.last_update,
        }


class QuantizationCoordinator:
    """Distributed quantization coordinator.

    Workflow:
    1. Collect GPU profiles from all cluster nodes
    2. Run APO to generate optimal quantization plan
    3. Distribute assignments to each node
    4. Handle fallback if a node reports method failure
    5. Re-run APO when cluster topology changes
    """

    def __init__(
        self,
        model_name: str = "",
        model_size_bytes: int = 0,
        num_layers: int = 32,
        max_quality_loss: float = 0.05,
        prefer_speed: bool = False,
        require_calibration: bool = False,
    ):
        self._model_name = model_name
        self._model_size_bytes = model_size_bytes
        self._num_layers = num_layers
        self._max_quality_loss = max_quality_loss
        self._prefer_speed = prefer_speed
        self._require_calibration = require_calibration

        self._state = CoordinatorState(
            model_name=model_name,
            model_size_bytes=model_size_bytes,
            num_layers=num_layers,
        )
        self._fallback_count: dict[str, int] = {}

    def register_node(self, profile: NodeProfile) -> None:
        """Register or update a node's GPU profile."""
        profile.last_heartbeat = time.time()
        profile.status = "online"
        self._state.nodes[profile.node_id] = profile
        self._state.last_update = time.time()
        logger.info(
            f"Node registered: {profile.node_id} "
            f"({profile.gpu_name}, {profile.total_memory_bytes / (1024**3):.0f}GB)"
        )

    def unregister_node(self, node_id: str) -> None:
        """Mark a node as offline."""
        if node_id in self._state.nodes:
            self._state.nodes[node_id].status = "offline"
            logger.info(f"Node unregistered: {node_id}")

    def generate_plan(self) -> dict[str, Any]:
        """Run APO and generate quantization plan for all online nodes.

        Returns:
            Dict with plan details and per-node assignments.
        """
        from distllm.dist.partition.quantization_tuner import (
            QuantizationAutoTuner, NodeInfo, QuantizationPlan,
        )
        from distllm.dist.partition.quant_report import ReportGenerator

        online_nodes = {
            nid: n for nid, n in self._state.nodes.items()
            if n.status == "online"
        }

        if not online_nodes:
            return {"error": "No online nodes", "assignments": {}}

        # Build NodeInfo list
        node_infos = []
        for nid, profile in online_nodes.items():
            node_infos.append(NodeInfo(
                node_id=nid,
                device_type="cuda",
                total_memory_bytes=profile.total_memory_bytes,
                compute_capability=profile.compute_capability,
                gpu_name=profile.gpu_name,
                bandwidth_gbps=profile.bandwidth_gbps,
                is_hopper_or_newer=profile.is_hopper_or_newer,
            ))

        # Run APO
        tuner = QuantizationAutoTuner(
            max_quality_loss=self._max_quality_loss,
            prefer_speed=self._prefer_speed,
            require_calibration=self._require_calibration,
        )
        plan = tuner.recommend(
            node_infos, self._model_size_bytes, self._num_layers,
        )

        # Generate report
        reporter = ReportGenerator()
        report = reporter.generate(
            plan, node_infos, self._model_size_bytes, self._num_layers,
        )

        # Build assignments
        assignments = {}
        for rec in plan.recommendations:
            assignments[rec.node_id] = NodeQuantAssignment(
                node_id=rec.node_id,
                quant_method=rec.method.value,
                activation_quant=rec.activation_quant.value,
                kv_cache_bits=rec.kv_cache_bits.value,
                max_quality_loss=self._max_quality_loss,
                mixed_precision_plan=(
                    {"num_layers": rec.mixed_precision_plan.num_layers}
                    if rec.mixed_precision_plan else {}
                ),
            )

        self._state.assignments = assignments
        self._state.plan_json = plan.to_json()
        self._state.last_update = time.time()

        return {
            "plan": plan.to_dict(),
            "assignments": {
                nid: {
                    "quant_method": a.quant_method,
                    "activation_quant": a.activation_quant,
                    "kv_cache_bits": a.kv_cache_bits,
                }
                for nid, a in assignments.items()
            },
            "report": report.to_dict(),
        }

    def report_failure(self, node_id: str, method: str, error: str) -> dict[str, Any]:
        """Handle a node reporting that its quant method failed.

        Falls back to a less aggressive method and re-generates the plan.
        """
        self._fallback_count[node_id] = self._fallback_count.get(node_id, 0) + 1
        logger.warning(
            f"Node {node_id} reported {method} failure: {error} "
            f"(fallback #{self._fallback_count[node_id]})"
        )

        # If a node has failed too many times, mark it offline
        if self._fallback_count[node_id] >= 3:
            logger.error(f"Node {node_id} exceeded fallback limit, marking offline")
            self.unregister_node(node_id)
            return self.generate_plan()

        # Tighten quality loss to force less aggressive method
        old_loss = self._max_quality_loss
        self._max_quality_loss = max(0.01, self._max_quality_loss * 0.5)
        logger.info(
            f"Reducing max_quality_loss from {old_loss:.3f} to {self._max_quality_loss:.3f}"
        )

        result = self.generate_plan()

        # Restore original quality loss for future runs
        self._max_quality_loss = old_loss

        return result

    def get_assignment(self, node_id: str) -> NodeQuantAssignment | None:
        """Get the quantization assignment for a specific node."""
        return self._state.assignments.get(node_id)

    def get_state(self) -> CoordinatorState:
        """Get the full coordinator state."""
        return self._state

    def status(self) -> dict[str, Any]:
        """Get a summary status of the coordinator."""
        online = sum(1 for n in self._state.nodes.values() if n.status == "online")
        total = len(self._state.nodes)
        assigned = len(self._state.assignments)
        return {
            "model": self._model_name,
            "model_size_gb": round(self._model_size_bytes / (1024**3), 2),
            "num_layers": self._num_layers,
            "nodes_online": online,
            "nodes_total": total,
            "nodes_assigned": assigned,
            "max_quality_loss": self._max_quality_loss,
            "prefer_speed": self._prefer_speed,
            "last_update": self._state.last_update,
        }
