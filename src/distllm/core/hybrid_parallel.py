"""Hybrid Parallelism Engine: auto-selects TP (within node), PP (across nodes), EP (for MoE).

Strategy selection is based on hardware topology:
- TP: multiple GPUs per node with NVLink → split heads/neurons across GPUs
- PP: multiple nodes via network → split layers across nodes
- EP: MoE models → replicate/spread experts across nodes

Integrates with tp_launcher (TP), pipeline_orchestrator (PP), and moe_orchestrator (EP).
"""

import time
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import torch
import torch.nn as nn
from loguru import logger


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
    node_hostnames: list[str] = field(default_factory=list)
    gpu_memory_gb: list[float] = field(default_factory=list)


@dataclass
class ParallelPlan:
    """Selected parallelism configuration."""
    strategy: ParallelStrategy = ParallelStrategy.PP
    tp_world_size: int = 1
    pp_num_stages: int = 1
    ep_num_experts_per_node: int = 1
    ep_replication_factor: int = 1
    layers_per_stage: list[tuple[int, int]] = field(default_factory=list)
    nodes_per_stage: list[list[str]] = field(default_factory=list)
    expert_assignment: dict[str, list[int]] = field(default_factory=dict)
    explanation: str = ""


class HardwareProber:
    """Probes hardware topology for parallelism strategy selection."""

    @staticmethod
    def probe() -> TopologyInfo:
        """Detect available hardware topology."""
        import os

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

            # NVLink detection via bandwidth measurement (not peer-accessibility).
            # PCIe P2P is 16-32 GB/s; NVLink is 50-250 GB/s.
            # A 50 GB/s threshold reliably distinguishes NVLink from PCIe.
            if info.gpus_per_node > 1:
                info.has_nvlink = HardwareProber._detect_nvlink(info.gpus_per_node)

        ib = os.environ.get("DISTLLM_INFINIBAND", "").lower()
        info.has_infiniband = ib in ("1", "true", "yes")
        ib_bw = os.environ.get("DISTLLM_IB_BANDWIDTH_GBPS", "")
        if ib_bw:
            try:
                info.interconnect_bandwidth_gbps = float(ib_bw)
            except ValueError:
                pass
        return info

    @staticmethod
    def _detect_nvlink(num_gpus: int, threshold_gbps: float = 50.0) -> bool:
        """Detect NVLink by measuring inter-GPU bandwidth.

        Returns True if any GPU pair achieves bandwidth above threshold.
        """
        try:
            size = 64 * 1024 * 1024  # 64 MB tensor
            iterations = 5
            for i in range(num_gpus):
                for j in range(i + 1, num_gpus):
                    src = torch.randn(size, dtype=torch.float32, device=f"cuda:{i}")
                    dst = torch.empty_like(src, device=f"cuda:{j}")
                    torch.cuda.synchronize(f"cuda:{i}")
                    torch.cuda.synchronize(f"cuda:{j}")

                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record(stream=torch.cuda.Stream(f"cuda:{i}"))
                    for _ in range(iterations):
                        dst.copy_(src, non_blocking=True)
                    end.record(stream=torch.cuda.Stream(f"cuda:{i}"))
                    end.synchronize()

                    elapsed_ms = start.elapsed_time(end)
                    if elapsed_ms <= 0:
                        continue
                    bandwidth_gbps = (size * 4 * iterations) / (elapsed_ms / 1000) / 1e9
                    if bandwidth_gbps > threshold_gbps:
                        return True
        except Exception:
            pass
        return False


@dataclass
class ProfileResult:
    """Results from the startup hardware profiling phase (~10s)."""
    compute_tokens_per_sec_per_gpu: float = 0.0
    intra_node_bw_gbps: float = 0.0
    inter_node_bw_gbps: float = 0.0
    free_memory_per_gpu: list[float] = field(default_factory=list)
    peak_memory_per_token_mb: float = 0.0
    profile_duration_seconds: float = 0.0


@dataclass
class TunedConfig:
    """Optimal parallelism configuration from the auto-tuner."""
    tp_degree: int = 1
    pp_stages: int = 1
    micro_batch_size: int = 1
    estimated_step_latency_ms: float = 0.0
    explanation: str = ""


class ParallelAutoTuner:
    """Lightweight (~10s) startup profiler + auto-tuner for parallelism strategy.

    Profiles compute throughput, intra-node/inter-node bandwidth, and VRAM per GPU
    at startup, then selects optimal (TP degree, PP stages, micro-batch size) using
    a cost model: ``step_latency = max(compute_time, comm_time)``.

    Typical usage::

        tuner = ParallelAutoTuner()
        config = tuner.tune(total_layers=32, hidden_size=4096)
        planner = HybridParallelPlanner()
        plan = planner.plan(total_layers=32, tuned_config=config)
    """

    _PROFILE_DURATION_S: float = 10.0

    def __init__(self, topology: TopologyInfo | None = None):
        self.topology = topology or HardwareProber.probe()
        self._profile: ProfileResult | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def profile(
        self,
        hidden_size: int = 4096,
        num_layers: int = 1,
        dtype: torch.dtype = torch.float16,
        batch_size: int = 1,
        seq_len: int = 128,
    ) -> ProfileResult:
        """Run hardware profile measuring compute, bandwidth, and memory.

        Args:
            hidden_size: Model hidden dimension for synthetic forward pass.
            num_layers: Number of transformer layers in synthetic model.
            dtype: Data type for profiling tensors.
            batch_size: Micro-batch size for profiling.
            seq_len: Sequence length for profiling.

        Returns:
            ProfileResult with measured throughput, bandwidths, and memory.
        """
        start_time = time.monotonic()
        result = ProfileResult()

        result.free_memory_per_gpu = self._probe_free_memory()

        comp = self._bench_compute(hidden_size, num_layers, dtype, batch_size, seq_len)
        result.compute_tokens_per_sec_per_gpu = comp["tokens_per_sec"]
        result.peak_memory_per_token_mb = comp["mem_per_token_mb"]

        bw = self._bench_bandwidth()
        result.intra_node_bw_gbps = bw["intra_node"]
        result.inter_node_bw_gbps = bw["inter_node"]

        result.profile_duration_seconds = time.monotonic() - start_time
        self._profile = result
        logger.info(
            f"Profile done in {result.profile_duration_seconds:.1f}s: "
            f"compute={result.compute_tokens_per_sec_per_gpu:.0f} tok/s/gpu, "
            f"intra_bw={result.intra_node_bw_gbps:.1f} Gbps, "
            f"inter_bw={result.inter_node_bw_gbps:.1f} Gbps, "
            f"free_mem={min(result.free_memory_per_gpu) if result.free_memory_per_gpu else 0:.1f} GiB"
        )
        return result

    def tune(
        self,
        total_layers: int,
        hidden_size: int = 4096,
        num_experts: int = 0,
        use_moe: bool = False,
        seq_len: int = 2048,
        max_micro_batch: int = 32,
    ) -> TunedConfig:
        """Select optimal (TP degree, PP stages, micro-batch size).

        Sweeps valid combinations and picks the one with lowest estimated
        step latency.  Runs :meth:`profile` automatically if not already done.

        Args:
            total_layers: Total transformer layers in the model.
            hidden_size: Model hidden dimension.
            num_experts: Number of MoE experts (0 = no MoE).
            use_moe: Whether the model uses MoE layers.
            seq_len: Target sequence length for estimation.
            max_micro_batch: Maximum micro-batch size to consider.

        Returns:
            TunedConfig with the optimal settings.
        """
        if self._profile is None:
            self.profile(hidden_size=hidden_size)

        p = self._profile
        topo = self.topology

        # Valid TP degrees (must divide gpus_per_node)
        tp_cands = [d for d in (1, 2, 4, 8) if d <= topo.gpus_per_node and topo.gpus_per_node % d == 0]
        if not topo.has_nvlink:
            tp_cands = [1]

        # Valid PP stages (must divide total GPUs into equal groups)
        pp_cands = [s for s in (1, 2, 4, 8, 16) if s <= topo.total_gpus]

        # Micro-batch candidates (powers of 2)
        mb_cands = [m for m in (1, 2, 4, 8, 16, max_micro_batch) if m <= max_micro_batch]

        best: tuple | None = None
        best_lat = float("inf")

        for tp in tp_cands:
            for pp in pp_cands:
                if tp * pp > topo.total_gpus:
                    continue
                if topo.total_gpus % max(tp * pp, 1) != 0:
                    continue
                for mb in mb_cands:
                    lat = self._estimate_latency(tp, pp, mb, total_layers, hidden_size, seq_len)
                    if lat < best_lat:
                        best_lat = lat
                        best = (tp, pp, mb)

        if best is None:
            best = (1, 1, 1)
            best_lat = 0.0

        result = TunedConfig(
            tp_degree=best[0],
            pp_stages=best[1],
            micro_batch_size=best[2],
            estimated_step_latency_ms=best_lat,
            explanation=self._explain(best[0], best[1], best[2], best_lat, total_layers, hidden_size),
        )
        logger.info(f"Auto-tune result: {result.explanation}")
        return result

    # ------------------------------------------------------------------
    # Internal profilers
    # ------------------------------------------------------------------

    def _probe_free_memory(self) -> list[float]:
        mem: list[float] = []
        for i in range(self.topology.gpus_per_node):
            try:
                free, _ = torch.cuda.mem_get_info(f"cuda:{i}")
                mem.append(free / (1024**3))
            except Exception:
                mem.append(0.0)
        return mem

    def _bench_compute(
        self, hidden_size: int, num_layers: int, dtype: torch.dtype,
        batch_size: int, seq_len: int,
    ) -> dict:
        """Measure compute throughput with a synthetic transformer block."""
        device = "cuda:0"
        n_layers = min(num_layers, 2)
        n_heads = max(1, hidden_size // 64)

        try:
            block = self._test_block(hidden_size, n_heads, dtype).to(device)
            inp = torch.randn(batch_size, seq_len, hidden_size, dtype=dtype, device=device)

            for _ in range(3):
                block(inp)
            torch.cuda.synchronize(device)

            iters = max(5, int(self._PROFILE_DURATION_S * 0.4))
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            for _ in range(iters):
                block(inp)
            end.record()
            torch.cuda.synchronize(device)

            elapsed_ms = start.elapsed_time(end) / iters
            tokens_per_ms = batch_size * seq_len * n_layers / elapsed_ms
            tokens_per_sec = tokens_per_ms * 1000

            # Estimate memory-per-token with a larger allocation
            mem_before = torch.cuda.memory_allocated(device)
            big = torch.randn(batch_size * 2, seq_len, hidden_size, dtype=dtype, device=device)
            block(big)
            torch.cuda.synchronize(device)
            mem_after = torch.cuda.memory_allocated(device)
            peak_mb = (mem_after - mem_before) / (1024**2)
            mem_per_token_mb = peak_mb / (batch_size * 2 * seq_len) * 1.5

            del block, inp, big

        except Exception:
            tokens_per_sec = 1000.0 / (hidden_size / 4096)
            mem_per_token_mb = hidden_size * 4 / 1024

        return {"tokens_per_sec": tokens_per_sec, "mem_per_token_mb": mem_per_token_mb}

    def _bench_bandwidth(self) -> dict:
        bw: dict[str, float] = {"intra_node": 0.0, "inter_node": 0.0}

        if self.topology.gpus_per_node > 1:
            try:
                bw["intra_node"] = self._measure_p2p_bw(0, 1, 128)
            except Exception:
                bw["intra_node"] = 50.0 if self.topology.has_nvlink else 12.5
        else:
            bw["intra_node"] = 12.5

        if self.topology.num_nodes > 1:
            try:
                bw["inter_node"] = self._measure_network_bw()
            except Exception:
                bw["inter_node"] = self.topology.interconnect_bandwidth_gbps
        else:
            bw["inter_node"] = self.topology.interconnect_bandwidth_gbps

        return bw

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _test_block(hidden_size: int, n_heads: int, dtype: torch.dtype) -> nn.Module:
        return nn.Sequential(OrderedDict([
            ("attn_norm", nn.LayerNorm(hidden_size, dtype=dtype)),
            ("attn_q", nn.Linear(hidden_size, hidden_size, dtype=dtype)),
            ("attn_o", nn.Linear(hidden_size, hidden_size, dtype=dtype)),
            ("mlp_norm", nn.LayerNorm(hidden_size, dtype=dtype)),
            ("mlp_gate", nn.Linear(hidden_size, hidden_size * 4, dtype=dtype)),
            ("mlp_down", nn.Linear(hidden_size * 4, hidden_size, dtype=dtype)),
        ]))

    @staticmethod
    def _measure_p2p_bw(src: int, dst: int, size_mb: int = 128) -> float:
        """Measure P2P bandwidth between two GPUs (GiB/s → Gbps)."""
        size = size_mb * 1024 * 1024 // 4
        src_t = torch.randn(size, dtype=torch.float32, device=f"cuda:{src}")
        dst_t = torch.empty_like(src_t, device=f"cuda:{dst}")

        torch.cuda.synchronize(f"cuda:{src}")
        torch.cuda.synchronize(f"cuda:{dst}")

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        iters = max(5, int(10.0 * 0.15))

        start.record(stream=torch.cuda.Stream(f"cuda:{src}"))
        for _ in range(iters):
            dst_t.copy_(src_t, non_blocking=True)
        end.record(stream=torch.cuda.Stream(f"cuda:{src}"))
        end.synchronize()

        elapsed_ms = start.elapsed_time(end)
        if elapsed_ms <= 0:
            return 0.0
        return (size * 4 * iters) / (elapsed_ms / 1000) / 1e9

    @staticmethod
    def _measure_network_bw() -> float:
        import os
        bw = os.environ.get("DISTLLM_IB_BANDWIDTH_GBPS", "")
        if bw:
            try:
                return float(bw)
            except ValueError:
                pass
        return 12.5

    # ------------------------------------------------------------------
    # Cost model
    # ------------------------------------------------------------------

    def _estimate_latency(
        self, tp: int, pp: int, mb: int,
        total_layers: int, hidden_size: int, seq_len: int,
    ) -> float:
        """Estimate step latency using cost model.

        ``latency = max(compute_time, comm_time)``
        """
        p = self._profile
        if p is None:
            return float("inf")

        # Layers assigned to this pipeline stage
        layers_per_stage = max(1.0, total_layers / pp)

        # ---- compute ----
        # Profile measures throughput for one GPU on one forward pass.
        # With TP, each GPU does 1/tp of the work per layer.
        # With PP, each stage has layers_per_stage layers.
        tokens_per_step = mb * seq_len
        base_ms = (tokens_per_step / max(p.compute_tokens_per_sec_per_gpu, 1e-6)) * 1000
        compute_ms = base_ms * layers_per_stage / tp

        # ---- TP all-reduce ----
        # Ring all-reduce: 2 * (tp-1)/tp * message_size / bandwidth
        tp_comm_ms = 0.0
        if tp > 1 and p.intra_node_bw_gbps > 0:
            msg_bytes = hidden_size * seq_len * 2  # fp16 activations per layer
            allreduce_bw = p.intra_node_bw_gbps * 1e9 / 8  # bytes/sec
            allreduce_time = 2 * msg_bytes * (tp - 1) / tp / allreduce_bw
            tp_comm_ms = allreduce_time * 1000 * layers_per_stage

        # ---- PP P2P ----
        # Each pipeline stage sends/receives activations = hidden_size * seq_len * dtype_bytes
        pp_comm_ms = 0.0
        if pp > 1 and p.inter_node_bw_gbps > 0:
            msg_bytes = hidden_size * seq_len * 2
            net_bw = p.inter_node_bw_gbps * 1e9 / 8
            p2p_time = msg_bytes / net_bw  # one transfer (fwd activation)
            pp_comm_ms = p2p_time * 1000 * 2  # fwd + bwd

        # ---- pipeline bubble ----
        # Micro-batches incur (pp-1) bubbles of single-micro-batch compute time
        bubble_ms = 0.0
        if pp > 1 and mb > 0:
            bubble_ms = (pp - 1) * (base_ms / tp) * layers_per_stage

        # ---- memory check ----
        if self._would_oom(tp, mb, hidden_size, seq_len, total_layers):
            return float("inf")

        return compute_ms + max(tp_comm_ms, pp_comm_ms) + bubble_ms

    def _would_oom(
        self, tp: int, mb: int, hidden_size: int, seq_len: int, total_layers: int,
    ) -> bool:
        """Check if this config would exceed VRAM."""
        p = self._profile
        if not p or not p.free_memory_per_gpu:
            return False

        # Rough per-GPU memory estimate in GiB
        model_params = total_layers * hidden_size * hidden_size * 4 * 12
        kv_cache = 2 * mb * seq_len * total_layers * hidden_size * 2
        activations = mb * seq_len * hidden_size * 2

        total_gb = (model_params + kv_cache + activations) / (1024**3) / tp

        avg_free = sum(p.free_memory_per_gpu) / max(len(p.free_memory_per_gpu), 1)
        return total_gb > avg_free * 0.9

    @staticmethod
    def _explain(tp: int, pp: int, mb: int, lat: float, total_layers: int, hidden_size: int) -> str:
        return (
            f"TP={tp}, PP={pp}, micro_batch={mb} | "
            f"est {lat:.1f}ms/step | {total_layers}Lx{hidden_size}D"
        )


class HybridParallelPlanner:
    """Selects and configures the optimal parallelism strategy."""

    def __init__(self, topology: TopologyInfo | None = None):
        self.topology = topology or HardwareProber.probe()
        self._plan: ParallelPlan | None = None

    def plan(
        self,
        total_layers: int,
        num_experts: int = 0,
        use_moe: bool = False,
        pp_overlap: bool = True,
        tp_enabled: bool = True,
        ep_enabled: bool = True,
        tuned_config: TunedConfig | None = None,
    ) -> ParallelPlan:
        plan = ParallelPlan()
        gpus = self.topology.total_gpus
        nodes = self.topology.num_nodes
        gpu_per_node = self.topology.gpus_per_node

        has_experts = use_moe and num_experts > 0

        # Override with auto-tuned config if provided
        if tuned_config is not None:
            plan.strategy = self._strategy_for(tuned_config, has_experts)
            plan.tp_world_size = tuned_config.tp_degree
            plan.pp_num_stages = tuned_config.pp_stages
            plan.explanation = tuned_config.explanation
            plan.layers_per_stage = self._distribute_layers(total_layers, plan.pp_num_stages)
            plan.nodes_per_stage = self._assign_nodes(plan.pp_num_stages)
            if has_experts:
                plan.expert_assignment = self._assign_experts(num_experts, nodes)
            self._plan = plan
            return plan

        # Heuristic fallback (existing logic)
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
            if total_layers > 0:
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

    @staticmethod
    def _strategy_for(config: TunedConfig, has_experts: bool) -> ParallelStrategy:
        if config.tp_degree > 1 and config.pp_stages > 1 and has_experts:
            return ParallelStrategy.TP_PP_EP
        if config.tp_degree > 1 and config.pp_stages > 1:
            return ParallelStrategy.TP_PP
        if config.tp_degree > 1 and has_experts:
            return ParallelStrategy.TP_EP
        if config.pp_stages > 1 and has_experts:
            return ParallelStrategy.PP_EP
        if config.tp_degree > 1:
            return ParallelStrategy.TP
        if has_experts:
            return ParallelStrategy.EP
        return ParallelStrategy.PP

    def _distribute_layers(self, total_layers: int, stages: int) -> list[tuple[int, int]]:
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

    def _assign_nodes(self, stages: int) -> list[list[str]]:
        if stages <= 1:
            return [["node_0"]]
        return [[f"node_{i}"] for i in range(stages)]

    def _assign_experts(self, num_experts: int, nodes: int) -> dict[str, list[int]]:
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
    def current_plan(self) -> ParallelPlan | None:
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
        self._tp_processes: list[Any] = []

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

    def configure_ep(self, moe_orchestrator: Any, node_ids: list[str]) -> None:
        if not self._plan.expert_assignment:
            return
        for node_id, expert_ids in self._plan.expert_assignment.items():
            logger.info(f"EP: node {node_id} assigned experts {expert_ids}")

    def execute(
        self,
        step_input: torch.Tensor,
        node_kv_caches: dict,
        request_id: str = "",
        draft_tokens: list | None = None,
    ) -> torch.Tensor:
        """Execute a step through the hybrid parallelism pipeline.

        Dispatches through TP → PP → EP based on the plan strategy.
        Falls back to pipeline.run_pipeline when no TP/EP is needed.
        """
        strategy = self._plan.strategy if self._plan else None

        # EP dispatch (all-to-all)
        if strategy in (ParallelStrategy.EP, ParallelStrategy.TP_EP, ParallelStrategy.PP_EP, ParallelStrategy.TP_PP_EP):
            moe = getattr(self._coordinator, '_moe_orchestrator', None)
            if moe and hasattr(moe, 'all_to_all_dispatch'):
                step_input = moe.all_to_all_dispatch(step_input)

        # TP forward via launched workers
        if strategy in (ParallelStrategy.TP, ParallelStrategy.TP_PP, ParallelStrategy.TP_EP, ParallelStrategy.TP_PP_EP):
            if self._tp_processes:
                from distllm.core.tp_launcher import tp_forward
                step_input = tp_forward(step_input, self._tp_processes)

        # PP forward via pipeline
        pipeline = getattr(self._coordinator, '_pipeline', None)
        if pipeline and strategy in (ParallelStrategy.PP, ParallelStrategy.TP_PP, ParallelStrategy.PP_EP, ParallelStrategy.TP_PP_EP):
            if getattr(pipeline, 'enable_overlap', False):
                logits = pipeline.run_pipeline_overlap(
                    step_input, node_kv_caches, request_id=request_id,
                    draft_tokens=draft_tokens,
                )
            else:
                logits = pipeline.run_pipeline(
                    step_input, node_kv_caches, request_id=request_id,
                    draft_tokens=draft_tokens,
                )
        elif pipeline:
            logits = pipeline.run_pipeline(
                step_input, node_kv_caches, request_id=request_id,
                draft_tokens=draft_tokens,
            )
        else:
            raise RuntimeError("No pipeline available for hybrid parallel execution")

        return logits

    def shutdown(self) -> None:
        for proc in self._tp_processes:
            context = getattr(proc, "process_context", None)
            if context is not None and hasattr(context, "processes"):
                for child in context.processes:
                    if hasattr(child, "terminate"):
                        try:
                            child.terminate()
                        except Exception:
                            pass
            elif hasattr(proc, 'terminate'):
                try:
                    proc.terminate()
                except Exception:
                    pass
        self._tp_processes.clear()
