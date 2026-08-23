"""Hybrid Parallelism Engine: auto-selects TP (within node), PP (across nodes), EP (for MoE).

Strategy selection is based on hardware topology:
- TP: multiple GPUs per node with NVLink → split heads/neurons across GPUs
- PP: multiple nodes via network → split layers across nodes
- EP: MoE models → replicate/spread experts across nodes

Integrates with tp_launcher (TP), pipeline_orchestrator (PP), and moe_orchestrator (EP).
"""


from __future__ import annotations
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import torch
import torch.nn as nn
import torch.distributed as dist
from loguru import logger

from distllm.dist.tp_launcher import launch_tp_workers, tp_forward


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
    num_nodes: int = 1
    gpus_per_node: int = 1
    has_nvlink: bool = False
    has_infiniband: bool = False
    total_gpus: int = 1
    interconnect_bandwidth_gbps: float = 12.5
    node_hostnames: list[str] = field(default_factory=list)
    gpu_memory_gb: list[float] = field(default_factory=list)
    gpu_free_memory_bytes: list[int] = field(default_factory=list)

    def min_free_memory_bytes(self) -> int:
        """Return the smallest free memory across all GPUs (best estimate of tightest GPU)."""

        if self.gpu_free_memory_bytes:
            return min(self.gpu_free_memory_bytes)
        if self.gpu_memory_gb:
            return int(min(self.gpu_memory_gb) * 0.85 * (1024 ** 3))
        return 0


@dataclass
class ParallelPlan:
    strategy: ParallelStrategy = ParallelStrategy.PP
    tp_world_size: int = 1
    pp_num_stages: int = 1
    ep_num_experts_per_node: int = 1
    ep_replication_factor: int = 1
    layers_per_stage: list[tuple[int, int]] = field(default_factory=list)
    nodes_per_stage: list[list[str]] = field(default_factory=list)
    expert_assignment: dict[str, list[int]] = field(default_factory=dict)
    tp_group_size: int = 1
    ep_group_size: int = 1
    tp_groups: list[list[int]] = field(default_factory=list)
    ep_groups: list[list[int]] = field(default_factory=list)
    explanation: str = ""


def estimate_layer_memory(
    hidden_size: int,
    intermediate_size: int,
    num_attention_heads: int,
    num_key_value_heads: int | None = None,
    dtype_bits: int = 16,
    vocab_size: int = 0,
) -> dict[str, int]:
    """Estimate per-layer memory requirements.


    Returns a dict with ``parameters`` (count), ``weight_bytes`` (total
    weight memory), ``activation_bytes`` (per-token activation memory),
    and ``total_per_layer_bytes``.

    The estimate covers:
    - Q, K, V, O projections
    - Gate, up, down MLP projections
    - Input and post-attention layer norms
    """

    kv_heads = num_key_value_heads or num_attention_heads
    head_dim = hidden_size // num_attention_heads

    q_proj = hidden_size * hidden_size
    k_proj = hidden_size * (kv_heads * head_dim)
    v_proj = hidden_size * (kv_heads * head_dim)
    o_proj = hidden_size * hidden_size
    gate_proj = hidden_size * intermediate_size
    up_proj = hidden_size * intermediate_size
    down_proj = intermediate_size * hidden_size
    input_norm = hidden_size * 2
    post_attn_norm = hidden_size * 2

    total_params = q_proj + k_proj + v_proj + o_proj + gate_proj + up_proj + down_proj + input_norm + post_attn_norm
    bytes_per_param = dtype_bits // 8
    weight_bytes = total_params * bytes_per_param

    act_per_token = (
        hidden_size                      # residual stream
        + (kv_heads * head_dim * 2)      # K, V cache per layer
        + intermediate_size              # MLP activation
    ) * bytes_per_param
    activation_bytes = act_per_token

    return {
        "parameters": total_params,
        "weight_bytes": weight_bytes,
        "activation_bytes": activation_bytes,
        "total_per_layer_bytes": weight_bytes + activation_bytes,
    }


def choose_tp_degree(
    layer_memory_bytes: int,
    per_gpu_free_bytes: int,
    max_tp: int = 8,
) -> tuple[int, str]:
    """Choose the minimum TP degree so that a single layer fits on one GPU.


    Args:
        layer_memory_bytes: Total memory needed for one transformer layer.
        per_gpu_free_bytes: Free memory available on the tightest GPU.
        max_tp: Maximum TP degree to consider.

    Returns:
        ``(tp_degree, reason)`` tuple.
    """

    for tp in (1, 2, 4, max_tp):
        per_gpu = layer_memory_bytes // tp
        if per_gpu < per_gpu_free_bytes * 0.85:
            return tp, f"Layer {layer_memory_bytes // (1024**2)}MB fits in {per_gpu_free_bytes // (1024**2)}MB GPU at TP={tp}"
    return max_tp, f"Forcing TP={max_tp} (layer too large for {per_gpu_free_bytes // (1024**2)}MB GPU)"


class HardwareProber:
    """Probes hardware topology for parallelism strategy selection."""


    @staticmethod
    def probe() -> TopologyInfo:
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
        try:
            size = 64 * 1024 * 1024
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
    compute_tokens_per_sec_per_gpu: float = 0.0
    intra_node_bw_gbps: float = 0.0
    inter_node_bw_gbps: float = 0.0
    free_memory_per_gpu: list[float] = field(default_factory=list)
    peak_memory_per_token_mb: float = 0.0
    profile_duration_seconds: float = 0.0


@dataclass
class TunedConfig:
    tp_degree: int = 1
    pp_stages: int = 1
    micro_batch_size: int = 1
    estimated_step_latency_ms: float = 0.0
    explanation: str = ""


class ParallelAutoTuner:
    """Lightweight (~10s) startup profiler + auto-tuner for parallelism strategy."""


    _PROFILE_DURATION_S: float = 10.0

    def __init__(self, topology: TopologyInfo | None = None):
        self.topology = topology or HardwareProber.probe()
        self._profile: ProfileResult | None = None

    def profile(
        self,
        hidden_size: int = 4096,
        num_layers: int = 1,
        dtype: torch.dtype = torch.float16,
        batch_size: int = 1,
        seq_len: int = 128,
    ) -> ProfileResult:
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
        if self._profile is None:
            self.profile(hidden_size=hidden_size)

        p = self._profile
        topo = self.topology

        tp_cands = [d for d in (1, 2, 4, 8) if d <= topo.gpus_per_node and topo.gpus_per_node % d == 0]
        if not topo.has_nvlink:
            tp_cands = [1]

        pp_cands = [s for s in (1, 2, 4, 8, 16) if s <= topo.total_gpus]

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

    def _estimate_latency(
        self, tp: int, pp: int, mb: int,
        total_layers: int, hidden_size: int, seq_len: int,
    ) -> float:
        p = self._profile
        if p is None:
            return float("inf")

        layers_per_stage = max(1.0, total_layers / pp)

        tokens_per_step = mb * seq_len
        base_ms = (tokens_per_step / max(p.compute_tokens_per_sec_per_gpu, 1e-6)) * 1000
        compute_ms = base_ms * layers_per_stage / tp

        tp_comm_ms = 0.0
        if tp > 1 and p.intra_node_bw_gbps > 0:
            msg_bytes = hidden_size * seq_len * 2
            allreduce_bw = p.intra_node_bw_gbps * 1e9 / 8
            allreduce_time = 2 * msg_bytes * (tp - 1) / tp / allreduce_bw
            tp_comm_ms = allreduce_time * 1000 * layers_per_stage

        pp_comm_ms = 0.0
        if pp > 1 and p.inter_node_bw_gbps > 0:
            msg_bytes = hidden_size * seq_len * 2
            net_bw = p.inter_node_bw_gbps * 1e9 / 8
            p2p_time = msg_bytes / net_bw
            pp_comm_ms = p2p_time * 1000 * 2

        bubble_ms = 0.0
        if pp > 1 and mb > 0:
            bubble_ms = (pp - 1) * (base_ms / tp) * layers_per_stage

        if self._would_oom(tp, mb, hidden_size, seq_len, total_layers):
            return float("inf")

        return compute_ms + max(tp_comm_ms, pp_comm_ms) + bubble_ms

    def _would_oom(
        self, tp: int, mb: int, hidden_size: int, seq_len: int, total_layers: int,
    ) -> bool:
        p = self._profile
        if not p or not p.free_memory_per_gpu:
            return False

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
        hidden_size: int = 0,
        intermediate_size: int = 0,
        num_attention_heads: int = 0,
        num_key_value_heads: int | None = None,
    ) -> ParallelPlan:
        plan = ParallelPlan()
        gpus = self.topology.total_gpus
        nodes = self.topology.num_nodes
        gpu_per_node = self.topology.gpus_per_node

        has_experts = use_moe and num_experts > 0

        # Per-layer memory check: if a single layer exceeds GPU capacity,
        # force TP regardless of NVLink availability.
        forced_tp = 1
        if tp_enabled and hidden_size > 0 and gpu_per_node > 1:
            free_bytes = self.topology.min_free_memory_bytes()
            if free_bytes > 0:
                layer_est = estimate_layer_memory(
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size or hidden_size * 4,
                    num_attention_heads=num_attention_heads or max(1, hidden_size // 64),
                    num_key_value_heads=num_key_value_heads,
                )
                layer_bytes = layer_est["total_per_layer_bytes"]
                forced_tp, _ = choose_tp_degree(layer_bytes, free_bytes, max_tp=gpu_per_node)

        if tuned_config is not None:
            plan.strategy = self._strategy_for(tuned_config, has_experts)
            plan.tp_world_size = max(tuned_config.tp_degree, forced_tp)
            plan.pp_num_stages = tuned_config.pp_stages
            plan.explanation = tuned_config.explanation
            plan.layers_per_stage = self._distribute_layers(total_layers, plan.pp_num_stages)
            plan.nodes_per_stage = self._assign_nodes(plan.pp_num_stages)
            if has_experts:
                plan.expert_assignment = self._assign_experts(num_experts, nodes)
            self._plan = plan
            return plan

        can_tp = (tp_enabled and gpu_per_node > 1
                  and (self.topology.has_nvlink or forced_tp > 1)
                  and gpus > 1)
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
            plan.tp_world_size = max(gpu_per_node, forced_tp)
        elif forced_tp > 1:
            plan.strategy = ParallelStrategy.TP_PP if nodes > 1 or total_layers > 1 else ParallelStrategy.TP
            plan.tp_world_size = forced_tp
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

        if forced_tp > 1:
            plan.explanation = f"TP-forced={forced_tp} (layer exceeds single-GPU memory) | " + self._build_explanation(plan)
        else:
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

    def _build_tp_ep_groups(
        self,
        num_gpus: int,
        tp_world_size: int,
        ep_world_size: int,
    ) -> tuple[list[list[int]], list[list[int]], int, int]:
        if tp_world_size <= 1 and ep_world_size <= 1:
            return [], [], 1, 1

        actual_tp = max(1, tp_world_size)
        actual_ep = max(1, ep_world_size)

        if actual_tp * actual_ep > num_gpus:
            factor = num_gpus / (actual_tp * actual_ep)
            actual_tp = max(1, int(actual_tp * factor ** 0.5))
            actual_ep = max(1, int(actual_ep * factor ** 0.5))

        actual_tp = max(1, min(actual_tp, num_gpus))
        actual_ep = max(1, num_gpus // actual_tp)

        tp_groups: list[list[int]] = []
        ep_groups: list[list[int]] = []
        gpu_idx = 0

        for _ in range(actual_ep):
            group_size = min(actual_tp, num_gpus - gpu_idx)
            if group_size <= 0:
                break
            group = list(range(gpu_idx, gpu_idx + group_size))
            tp_groups.append(group)
            gpu_idx += group_size

        for tp_i in range(actual_tp):
            group = []
            for ep_i in range(actual_ep):
                if tp_i < len(tp_groups[ep_i]):
                    group.append(tp_groups[ep_i][tp_i])
            if len(group) > 1:
                ep_groups.append(group)

        return tp_groups, ep_groups, actual_tp, actual_ep

    def build_plan_groups(self) -> None:
        if self._plan is None:
            return
        plan = self._plan
        gpus = self.topology.total_gpus

        ts = plan.strategy
        is_tp = ts in (
            ParallelStrategy.TP, ParallelStrategy.TP_PP,
            ParallelStrategy.TP_EP, ParallelStrategy.TP_PP_EP,
        )
        is_ep = ts in (
            ParallelStrategy.EP, ParallelStrategy.TP_EP,
            ParallelStrategy.PP_EP, ParallelStrategy.TP_PP_EP,
        )

        tp_size = plan.tp_world_size if is_tp else 1
        ep_size = len(plan.expert_assignment) if is_ep else 1

        if not is_tp and not is_ep:
            return

        tp_groups, ep_groups, actual_tp, actual_ep = self._build_tp_ep_groups(
            gpus, tp_size, ep_size,
        )

        plan.tp_groups = tp_groups
        plan.ep_groups = ep_groups
        plan.tp_group_size = actual_tp
        plan.ep_group_size = actual_ep

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


def _compute_chunk_sizes(
    hidden_size: int,
    num_chunks: int,
) -> list[int]:
    base = hidden_size // num_chunks
    rem = hidden_size % num_chunks
    sizes = []
    offset = 0
    for i in range(num_chunks):
        chunk = base + (1 if i < rem else 0)
        sizes.append((offset, offset + chunk))
        offset += chunk
    return sizes


class HybridParallelExecutor:
    """Executes a hybrid parallelism plan across TP, PP, and EP modalities."""


    def __init__(self, plan: ParallelPlan, coordinator: Any = None):
        self._plan = plan
        self._coordinator = coordinator
        self._tp_processes: list[Any] = []
        self._tp_group: dist.ProcessGroup | None = None
        self._ep_group: dist.ProcessGroup | None = None
        self._tp_ep_comm_handles: dict[str, Any] = {}

    def launch_tp(self, model_name: str, dtype: str = "float16") -> None:
        if self._plan.tp_world_size <= 1:
            return
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

    def _setup_tp_ep_comm_groups(
        self,
        rank: int,
        world_size: int,
    ) -> None:
        tp_groups = self._plan.tp_groups
        ep_groups = self._plan.ep_groups

        for group_ranks in tp_groups:
            if rank in group_ranks:
                pg = dist.new_group(ranks=group_ranks)
                if rank == group_ranks[0] or True:
                    self._tp_group = pg

        for group_ranks in ep_groups:
            if rank in group_ranks:
                pg = dist.new_group(ranks=group_ranks)
                if rank == group_ranks[0] or True:
                    self._ep_group = pg

    def _tp_ep_fused_forward(
        self,
        hidden_states: torch.Tensor,
        experts: list[Any],
        router_logits: torch.Tensor,
        top_k: int = 2,
    ) -> torch.Tensor:
        from distllm.core.moe_router import TopkRouter

        batch, seq, hidden = hidden_states.shape
        flat_hidden = hidden_states.reshape(-1, hidden)
        num_tokens = flat_hidden.shape[0]

        device = flat_hidden.device

        router = TopkRouter(
            num_experts=len(experts),
            top_k=top_k,
        )

        routing_weights, expert_indices = router(flat_hidden)

        dispatch_mask = torch.zeros(
            num_tokens, len(experts), dtype=torch.bool, device=device,
        )
        for k in range(top_k):
            dispatch_mask.scatter_(
                1, expert_indices[:, k:k+1], True,
            )

        tokens_per_expert = dispatch_mask.sum(dim=0).tolist()
        expert_offsets = [0]
        for c in tokens_per_expert:
            expert_offsets.append(expert_offsets[-1] + c)

        send_buf = torch.zeros(
            num_tokens, hidden, device=device,
        )
        for eid in range(len(experts)):
            mask = dispatch_mask[:, eid]
            if mask.any():
                send_buf[mask] = flat_hidden[mask]

        ep_group = self._ep_group
        if ep_group is not None and dist.get_world_size(ep_group) > 1:
            ep_size = dist.get_world_size(ep_group)
            max_tokens = max(tokens_per_expert) if tokens_per_expert else 0
            max_tokens = max(max_tokens, 1)

            send_counts = torch.tensor(
                tokens_per_expert * [hidden], device=device,
            )
            recv_counts = torch.zeros(ep_size, device=device, dtype=torch.long)
            local_tokens = torch.tensor(
                sum(tokens_per_expert), device=device, dtype=torch.long,
            )
            dist.all_gather_into_tensor(
                recv_counts, local_tokens, group=ep_group,
            )

            max_recv = int(recv_counts.max().item())
            recv_buf = torch.zeros(
                max_recv, hidden, device=device,
            )

            reqs = []
            for i in range(ep_size):
                if i == dist.get_rank(ep_group):
                    continue
                dst = send_buf[:local_tokens.item()].contiguous()
                src = torch.zeros(max_recv, hidden, device=device)
                req = dist.broadcast(
                    dst, src=i, group=ep_group, async_op=True,
                )
                reqs.append(req)

            for req in reqs:
                req.wait()

            processed = send_buf[:local_tokens.item()]
        else:
            processed = send_buf[:num_tokens]

        my_experts = self._plan.expert_assignment.get(
            f"node_{dist.get_rank() % max(dist.get_world_size(), 1)}",
            list(range(len(experts))),
        )
        output_buf = torch.zeros_like(processed)

        for k in range(top_k):
            e_ids = expert_indices[:, k]
            w = routing_weights[:, k]

            for eid in my_experts:
                if eid >= len(experts):
                    continue
                mask = (e_ids == eid)
                if not mask.any():
                    continue

                token_input = processed[mask]
                expert_out = experts[eid](token_input)
                output_buf[mask] += expert_out * w[mask].unsqueeze(-1)

        tp_group = self._tp_group
        if tp_group is not None and dist.get_world_size(tp_group) > 1:
            dist.all_reduce(output_buf, group=tp_group)

        if ep_group is not None and dist.get_world_size(ep_group) > 1:
            ep_size = dist.get_world_size(ep_group)
            gathered = [
                torch.zeros_like(output_buf) for _ in range(ep_size)
            ]
            dist.all_gather(gathered, output_buf, group=ep_group)
            output_buf = sum(gathered)

        output = output_buf[:num_tokens].reshape(batch, seq, hidden)
        return output

    def _tp_fwd(
        self,
        hidden: torch.Tensor,
    ) -> torch.Tensor:
        if self._tp_processes:
            return tp_forward(hidden, self._tp_processes)
        return hidden

    def _pp_fwd(
        self,
        hidden: torch.Tensor,
        node_kv_caches: dict,
        request_id: str = "",
        draft_tokens: list | None = None,
    ) -> torch.Tensor:
        pipeline = getattr(self._coordinator, '_pipeline', None)
        if pipeline is None:
            raise RuntimeError("No pipeline available for hybrid parallel execution")

        if getattr(pipeline, 'enable_overlap', False):
            return pipeline.run_pipeline_overlap(
                hidden, node_kv_caches,
                request_id=request_id, draft_tokens=draft_tokens,
            )
        return pipeline.run_pipeline(
            hidden, node_kv_caches,
            request_id=request_id, draft_tokens=draft_tokens,
        )

    def _tp_pp_fused_forward(
        self,
        step_input: torch.Tensor,
        node_kv_caches: dict,
        request_id: str = "",
        draft_tokens: list | None = None,
    ) -> torch.Tensor:
        tp_out = self._tp_fwd(step_input)
        return self._pp_fwd(tp_out, node_kv_caches, request_id, draft_tokens)

    def execute(
        self,
        step_input: torch.Tensor,
        node_kv_caches: dict,
        request_id: str = "",
        draft_tokens: list | None = None,
    ) -> torch.Tensor:
        strategy = self._plan.strategy if self._plan else None

        needs_ep = strategy in (
            ParallelStrategy.EP, ParallelStrategy.PP_EP,
            ParallelStrategy.TP_PP_EP,
        )
        if needs_ep:
            moe = getattr(self._coordinator, '_moe_orchestrator', None)
            if moe and hasattr(moe, 'all_to_all_dispatch'):
                step_input = moe.all_to_all_dispatch(step_input)

        if strategy == ParallelStrategy.TP_EP:
            moe = getattr(self._coordinator, '_moe_orchestrator', None)
            experts = getattr(moe, 'experts', []) if moe else []
            router_logits = getattr(moe, 'last_router_logits', None)

            if experts and router_logits is not None:
                step_input = self._tp_ep_fused_forward(
                    step_input, experts, router_logits,
                )
            else:
                step_input = self._tp_fwd(step_input)

            pipeline = getattr(self._coordinator, '_pipeline', None)
            if pipeline:
                return self._pp_fwd(
                    step_input, node_kv_caches,
                    request_id=request_id, draft_tokens=draft_tokens,
                )
            return step_input

        if strategy == ParallelStrategy.TP_PP:
            return self._tp_pp_fused_forward(
                step_input, node_kv_caches,
                request_id=request_id, draft_tokens=draft_tokens,
            )

        if strategy == ParallelStrategy.TP_PP_EP:
            return self._tp_pp_fused_forward(
                step_input, node_kv_caches,
                request_id=request_id, draft_tokens=draft_tokens,
            )

        if strategy == ParallelStrategy.TP:
            return self._tp_fwd(step_input)

        if strategy in (ParallelStrategy.PP, ParallelStrategy.PP_EP):
            return self._pp_fwd(
                step_input, node_kv_caches,
                request_id=request_id, draft_tokens=draft_tokens,
            )

        return self._pp_fwd(
            step_input, node_kv_caches,
            request_id=request_id, draft_tokens=draft_tokens,
        )

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
