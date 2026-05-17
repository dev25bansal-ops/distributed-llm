"""Hybrid Parallelism Engine: auto-selects TP (within node), PP (across nodes), EP (for MoE).

Strategy selection is based on hardware topology:
- TP: multiple GPUs per node with NVLink → split heads/neurons across GPUs
- PP: multiple nodes via network → split layers across nodes
- EP: MoE models → replicate/spread experts across nodes

Integrates with tp_launcher (TP), pipeline_orchestrator (PP), and moe_orchestrator (EP).
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


class ParallelStrategy(str, Enum):
    TP = "tensor_parallel"
    PP = "pipeline_parallel"
    EP = "expert_parallel"
    TP_PP = "tp_pp"
    TP_EP = "tp_ep"
    PP_EP = "pp_ep"
    TP_PP_EP = "tp_pp_ep"


@dataclass
class TopologyInfo:
    """Hardware topology description for parallelism strategy selection."""
    num_nodes: int = 1
    gpus_per_node: int = 1
    has_nvlink: bool = False
    has_infiniband: bool = False
    total_gpus: int = 1
    interconnect_bandwidth_gbps: float = 12.5  # default PCIe 4.0 x16
    node_hostnames: List[str] = field(default_factory=list)
    gpu_memory_gb: List[float] = field(default_factory=list)


@dataclass
class ParallelPlan:
    """Selected parallelism configuration."""
    strategy: ParallelStrategy = ParallelStrategy.PP
    tp_world_size: int = 1
    pp_num_stages: int = 1
    ep_num_experts_per_node: int = 1
    ep_replication_factor: int = 1
    layers_per_stage: List[Tuple[int, int]] = field(default_factory=list)
    nodes_per_stage: List[List[str]] = field(default_factory=list)
    expert_assignment: Dict[str, List[int]] = field(default_factory=dict)
    explanation: str = ""


class HardwareProber:
    """Probes hardware topology for parallelism strategy selection."""

    @staticmethod
    def probe() -> TopologyInfo:
        """Detect available hardware topology."""
        info = TopologyInfo()
        if torch.cuda.is_available():
            info.gpus_per_node = torch.cuda.device_count()
            info.total_gpus = info.gpus_per_node
            for i in range(info.gpus_per_node):
                try:
                    props = torch.cuda.get_device_properties(i)
                    info.gpu_memory_gb.append(props.total_memory / (1024 ** 3))
                except Exception:
                    info.gpu_memory_gb.append(0.0)
            try:
                nvlink = torch.cuda.is_initialized()
                if nvlink:
                    info.has_nvlink = True
            except Exception:
                pass
            if info.gpus_per_node > 1:
                info.has_nvlink = True
        import os
        ib = os.environ.get("DISTLLM_INFINIBAND", "").lower()
        info.has_infiniband = ib in ("1", "true", "yes")
        ib_bw = os.environ.get("DISTLLM_IB_BANDWIDTH_GBPS", "")
        if ib_bw:
            try:
                info.interconnect_bandwidth_gbps = float(ib_bw)
            except ValueError:
                pass
        return info


class HybridParallelPlanner:
    """Selects and configures the optimal parallelism strategy."""

    def __init__(self, topology: Optional[TopologyInfo] = None):
        self.topology = topology or HardwareProber.probe()
        self._plan: Optional[ParallelPlan] = None

    def plan(
        self,
        total_layers: int,
        num_experts: int = 0,
        use_moe: bool = False,
        pp_overlap: bool = True,
        tp_enabled: bool = True,
        ep_enabled: bool = True,
    ) -> ParallelPlan:
        plan = ParallelPlan()
        gpus = self.topology.total_gpus
        nodes = self.topology.num_nodes
        gpu_per_node = self.topology.gpus_per_node

        has_experts = use_moe and num_experts > 0

        can_tp = tp_enabled and gpu_per_node > 1 and self.topology.has_nvlink and gpus > 1
        can_pp = nodes > 1 or (not can_tp and total_layers > 20)
        can_ep = has_experts and ep_enabled

        if can_tp and can_pp and can_ep:
            plan.strategy = ParallelStrategy.TP_PP_EP
        elif can_tp and can_pp:
            plan.strategy = ParallelStrategy.TP_PP
        elif can_tp and can_ep:
            plan.strategy = ParallelStrategy.TP_EP
        elif can_pp and can_ep:
            plan.strategy = ParallelStrategy.PP_EP
        elif can_tp:
            plan.strategy = ParallelStrategy.TP
        elif can_ep:
            plan.strategy = ParallelStrategy.EP
        else:
            plan.strategy = ParallelStrategy.PP

        if plan.strategy in (ParallelStrategy.TP, ParallelStrategy.TP_PP, ParallelStrategy.TP_PP_EP, ParallelStrategy.TP_EP):
            plan.tp_world_size = gpu_per_node
        else:
            plan.tp_world_size = 1

        if plan.strategy in (ParallelStrategy.PP, ParallelStrategy.TP_PP, ParallelStrategy.TP_PP_EP, ParallelStrategy.PP_EP):
            plan.pp_num_stages = max(1, nodes)
        else:
            plan.pp_num_stages = 1

        if plan.strategy in (ParallelStrategy.EP, ParallelStrategy.TP_EP, ParallelStrategy.TP_PP_EP, ParallelStrategy.PP_EP):
            cap_factor = 1.0
            if total_layers > 0:
                layers_per_node = max(1, total_layers // max(plan.pp_num_stages, 1))
                capacity_needed = num_experts * 2
                nodes_available = plan.pp_num_stages * gpu_per_node
                plan.ep_replication_factor = max(1, min(4, nodes_available // max(num_experts, 1)))
            plan.ep_num_experts_per_node = max(1, num_experts // max(plan.pp_num_stages, 1))

        plan.layers_per_stage = self._distribute_layers(total_layers, plan.pp_num_stages)
        plan.nodes_per_stage = self._assign_nodes(plan.pp_num_stages)

        if has_experts:
            plan.expert_assignment = self._assign_experts(num_experts, nodes)

        plan.explanation = self._build_explanation(plan)
        self._plan = plan
        return plan

    def _distribute_layers(self, total_layers: int, stages: int) -> List[Tuple[int, int]]:
        if stages <= 0 or total_layers <= 0:
            return [(0, max(0, total_layers - 1))]
        base = total_layers // stages
        rem = total_layers % stages
        result = []
        start = 0
        for i in range(stages):
            n = base + (1 if i < rem else 0)
            end = start + n - 1
            result.append((start, end))
            start = end + 1
        return result

    def _assign_nodes(self, stages: int) -> List[List[str]]:
        if stages <= 1:
            return [["node_0"]]
        return [[f"node_{i}"] for i in range(stages)]

    def _assign_experts(self, num_experts: int, nodes: int) -> Dict[str, List[int]]:
        if nodes <= 0:
            return {}
        from distllm.core.moe_orchestrator import replicate_experts_across_nodes
        return replicate_experts_across_nodes(num_experts, nodes, replication_factor=1)

    def _build_explanation(self, plan: ParallelPlan) -> str:
        parts = [f"Strategy: {plan.strategy.value}"]
        if plan.tp_world_size > 1:
            parts.append(f"TP={plan.tp_world_size} GPU(s)/node")
        if plan.pp_num_stages > 1:
            parts.append(f"PP={plan.pp_num_stages} stage(s)")
        if plan.ep_replication_factor > 1:
            parts.append(f"EP replication={plan.ep_replication_factor}x")
        parts.append(f"GPUs={self.topology.total_gpus}, Nodes={self.topology.num_nodes}")
        return " | ".join(parts)

    @property
    def current_plan(self) -> Optional[ParallelPlan]:
        return self._plan


class HybridParallelExecutor:
    """Executes a hybrid parallelism plan across TP, PP, and EP modalities.

    Integrates with:
    - tp_launcher.launch_tp_workers for TP within each node
    - pipeline_orchestrator.run_pipeline for PP across nodes
    - moe_orchestrator.all_to_all_dispatch for EP across nodes
    """

    def __init__(self, plan: ParallelPlan, coordinator: Any = None):
        self._plan = plan
        self._coordinator = coordinator
        self._tp_processes: List[Any] = []

    def launch_tp(self, model_name: str, dtype: str = "float16") -> None:
        if self._plan.tp_world_size <= 1:
            return
        from distllm.core.tp_launcher import launch_tp_workers
        self._tp_processes.append(launch_tp_workers(
            model_name=model_name,
            num_gpus=self._plan.tp_world_size,
            dtype=dtype,
        ))
        logger.info(f"Launched TP workers: world_size={self._plan.tp_world_size}")

    def configure_pp(self, pipeline: Any) -> None:
        if self._plan.pp_num_stages <= 1:
            return
        if hasattr(pipeline, 'enable_overlap'):
            pipeline.enable_overlap = True
        if hasattr(pipeline, 'group_nodes_into_stages'):
            pipeline.group_nodes_into_stages(self._plan.pp_num_stages)
        logger.info(f"Configured PP: {self._plan.pp_num_stages} stages")

    def configure_ep(self, moe_orchestrator: Any, node_ids: List[str]) -> None:
        if not self._plan.expert_assignment:
            return
        for node_id, expert_ids in self._plan.expert_assignment.items():
            logger.info(f"EP: node {node_id} assigned experts {expert_ids}")

    def shutdown(self) -> None:
        for proc in self._tp_processes:
            if hasattr(proc, 'terminate'):
                try:
                    proc.terminate()
                except Exception:
                    pass
        self._tp_processes.clear()
