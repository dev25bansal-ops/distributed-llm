"""Hardware topology probing, layer memory estimation, and TP degree selection.

Provides the foundational data structures and utilities for hybrid parallelism:

- :class:`TopologyInfo` — hardware topology dataclass
- :func:`estimate_layer_memory` — per-layer memory estimation
- :func:`choose_tp_degree` — TP degree selection
- :class:`HardwareProber` — GPU / network topology prober
- :class:`ProfileResult` — profiling result dataclass
- :class:`TunedConfig` — auto-tuned configuration dataclass
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import torch


# ---------------------------------------------------------------------------
# TopologyInfo  —  hardware topology dataclass
# ---------------------------------------------------------------------------


@dataclass
class TopologyInfo:
    """Describes the hardware topology available for parallelism."""

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


# ---------------------------------------------------------------------------
# estimate_layer_memory  —  per-layer memory arithmetic
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# choose_tp_degree  —  find the minimum TP degree so a layer fits on a GPU
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# HardwareProber  —  probes GPU / network topology
# ---------------------------------------------------------------------------


class HardwareProber:
    """Probes hardware topology for parallelism strategy selection."""


    @staticmethod
    def probe() -> TopologyInfo:
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


# ---------------------------------------------------------------------------
# ProfileResult  —  profiling result dataclass
# ---------------------------------------------------------------------------


@dataclass
class ProfileResult:
    """Results from a short profiling run."""

    compute_tokens_per_sec_per_gpu: float = 0.0
    intra_node_bw_gbps: float = 0.0
    inter_node_bw_gbps: float = 0.0
    free_memory_per_gpu: list[float] = field(default_factory=list)
    peak_memory_per_token_mb: float = 0.0
    profile_duration_seconds: float = 0.0


# ---------------------------------------------------------------------------
# TunedConfig  —  auto-tuned configuration dataclass
# ---------------------------------------------------------------------------


@dataclass
class TunedConfig:
    """Result from :class:`ParallelAutoTuner.tune`."""

    tp_degree: int = 1
    pp_stages: int = 1
    micro_batch_size: int = 1
    estimated_step_latency_ms: float = 0.0
    explanation: str = ""
