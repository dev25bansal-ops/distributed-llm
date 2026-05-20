from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class GPUProfile:
    """Profiled capabilities of a single GPU."""
    gpu_id: int
    name: str
    total_memory_bytes: int = 0
    free_memory_bytes: int = 0
    compute_tflops: float = 0.0
    memory_bandwidth_gbps: float = 0.0
    memory_bus_width_bits: int = 0
    mem_clock_khz: int = 0
    sm_count: int = 0
    max_threads_per_sm: int = 0
    peak_tflops_fp16: float = 0.0
    peak_tflops_fp32: float = 0.0
    measured_tflops_fp16: float = 0.0
    measured_tflops_fp32: float = 0.0
    measured_memory_bandwidth_gbps: float = 0.0


@dataclass
class LayerWeights:
    """Memory and compute requirements for a single transformer layer."""
    layer_id: int
    layer_type: str = "transformer"  # transformer, embed, lm_head, norm
    weight_memory_bytes: int = 0
    activation_memory_bytes: int = 0
    flops_per_token: int = 0
    flops_per_seq: int = 0
    kv_cache_bytes_per_token: int = 0

    @property
    def total_memory_bytes(self) -> int:
        return self.weight_memory_bytes + self.activation_memory_bytes


_KNOWN_GPU_SPECS: dict[str, tuple[float, float, float, int, int]] = {
    # (fp16_tflops, fp32_tflops, mem_bw_gbps, sm_count, mem_bus_bits)
    "A100": (312, 156, 2039, 108, 5120),
    "A100-SXM-80GB": (312, 156, 2039, 108, 5120),
    "A100-PCIE-40GB": (312, 156, 1555, 108, 5120),
    "A10": (125, 62, 600, 72, 3072),
    "A10G": (125, 62, 600, 72, 3072),
    "A30": (165, 82, 933, 56, 3072),
    "A40": (150, 75, 696, 84, 3840),
    "H100": (989, 494, 3350, 132, 5120),
    "H100-SXM-80GB": (989, 494, 3350, 132, 5120),
    "H100-PCIE-80GB": (989, 494, 2000, 114, 5120),
    "H200": (989, 494, 4800, 132, 5120),
    "V100": (125, 125, 900, 80, 4096),
    "V100-SXM-32GB": (125, 125, 900, 80, 4096),
    "V100-PCIE-32GB": (125, 125, 900, 80, 4096),
    "RTX 4090": (330, 165, 1008, 128, 3840),
    "RTX 4080": (180, 90, 716, 76, 2560),
    "RTX 3090": (142, 71, 936, 82, 3840),
    "RTX 3080": (90, 45, 760, 68, 3200),
    "RTX 4070": (89, 44, 504, 46, 1920),
    "RTX 4060": (50, 25, 272, 24, 1280),
    "L4": (121, 30, 300, 24, 1920),
    "L40S": (362, 91, 864, 84, 3840),
    "T4": (65, 32, 320, 40, 2560),
    "P100": (21, 21, 732, 56, 4096),
}

_LARGE_MATRIX_DIM = 4096
_LARGE_MATRIX_ELEMS = _LARGE_MATRIX_DIM * _LARGE_MATRIX_DIM
_BENCHMARK_MATRIX_BYTES = _LARGE_MATRIX_ELEMS * 2  # fp16


class GPUProfiler:
    """Profiles GPU compute capability, memory bandwidth, and capacity.

    Combines known specs lookup with runtime microbenchmarks for
    accurate TFLOPS and bandwidth measurements.

    Usage:
        profiler = GPUProfiler()
        profiles = profiler.profile_all_gpus()

        weights = profiler.estimate_layer_weights(
            hidden_size=4096, intermediate_size=11008,
            num_layers=32, num_heads=32, head_dim=128,
        )
    """

    # ------------------------------------------------------------------
    # GPU enumeration
    # ------------------------------------------------------------------

    def profile_all_gpus(self) -> list[GPUProfile]:
        """Profile all visible CUDA GPUs.

        Returns a list of GPUProfile, one per device.
        Falls back to known specs if benchmarking is unavailable.
        """
        profiles: list[GPUProfile] = []

        for gpu_id in range(self._device_count()):
            profile = self._profile_single_gpu(gpu_id)
            profiles.append(profile)

        if not profiles:
            profiles.append(GPUProfile(gpu_id=0, name="cpu_fallback"))

        return profiles

    def _profile_single_gpu(self, gpu_id: int) -> GPUProfile:
        name = self._get_device_name(gpu_id)
        mem_total, mem_free = self._get_memory_info(gpu_id)

        known = _KNOWN_GPU_SPECS.get(name, _KNOWN_GPU_SPECS.get(self._match_known_spec(name)))

        if known:
            fp16_tflops, fp32_tflops, mem_bw, sm_count, mem_bus = known
            profile = GPUProfile(
                gpu_id=gpu_id,
                name=name,
                total_memory_bytes=mem_total,
                free_memory_bytes=mem_free,
                compute_tflops=fp16_tflops,
                memory_bandwidth_gbps=mem_bw,
                sm_count=sm_count,
                memory_bus_width_bits=mem_bus,
                peak_tflops_fp16=fp16_tflops,
                peak_tflops_fp32=fp32_tflops,
            )
        else:
            profile = GPUProfile(
                gpu_id=gpu_id,
                name=name,
                total_memory_bytes=mem_total,
                free_memory_bytes=mem_free,
                compute_tflops=self._estimate_tflops_from_name(name),
                memory_bandwidth_gbps=self._estimate_bw_from_name(name),
            )

        # Run microbenchmarks if CUDA is available
        measured_fp16 = self._bench_matmul_fp16(gpu_id)
        measured_bw = self._bench_memory_bandwidth(gpu_id)

        if measured_fp16 > 0:
            profile.measured_tflops_fp16 = measured_fp16
            profile.compute_tflops = measured_fp16

        if measured_bw > 0:
            profile.measured_memory_bandwidth_gbps = measured_bw
            profile.memory_bandwidth_gbps = measured_bw

        return profile

    # ------------------------------------------------------------------
    # Layer weight estimation
    # ------------------------------------------------------------------

    def estimate_layer_weights(
        self,
        hidden_size: int = 4096,
        intermediate_size: int = 11008,
        num_layers: int = 32,
        num_heads: int = 32,
        head_dim: int = 128,
        vocab_size: int = 32000,
        max_seq_len: int = 4096,
        dtype_bytes: int = 2,
    ) -> list[LayerWeights]:
        """Estimate memory and compute for each layer in the model.

        Accounts for non-uniformity: embedding, first/last layers,
        and intermediate transformer layers.

        Returns a list of LayerWeights, one per logical "layer"
        including embeddings, transformer layers, and LM head.
        """
        kv_heads = max(1, num_heads)
        layers: list[LayerWeights] = []

        activation_per_token = hidden_size * dtype_bytes

        # Embedding layer (non-uniform)
        embed_mem = vocab_size * hidden_size * dtype_bytes
        embed_flops = 0
        layers.append(LayerWeights(
            layer_id=0,
            layer_type="embed",
            weight_memory_bytes=embed_mem,
            activation_memory_bytes=activation_per_token,
            flops_per_token=embed_flops,
            flops_per_seq=embed_flops,
            kv_cache_bytes_per_token=0,
        ))

        for i in range(num_layers):
            is_first = i == 0
            is_last = i == num_layers - 1

            weight_mem = self._single_layer_weight_bytes(
                hidden_size, intermediate_size, dtype_bytes
            )
            if is_first or is_last:
                weight_mem += 2 * hidden_size * dtype_bytes

            flops_per_token = self._single_layer_flops(
                hidden_size, intermediate_size, num_heads, head_dim,
            )
            kv_per_token = 2 * kv_heads * head_dim * dtype_bytes

            layers.append(LayerWeights(
                layer_id=i + 1,
                layer_type="transformer",
                weight_memory_bytes=weight_mem,
                activation_memory_bytes=activation_per_token,
                flops_per_token=flops_per_token,
                flops_per_seq=flops_per_token,
                kv_cache_bytes_per_token=kv_per_token,
            ))

        # LM head (non-uniform)
        lm_head_mem = vocab_size * hidden_size * dtype_bytes
        lm_head_flops = vocab_size * hidden_size * 2
        layers.append(LayerWeights(
            layer_id=num_layers + 1,
            layer_type="lm_head",
            weight_memory_bytes=lm_head_mem,
            activation_memory_bytes=activation_per_token,
            flops_per_token=lm_head_flops,
            flops_per_seq=lm_head_flops,
            kv_cache_bytes_per_token=0,
        ))

        return layers

    def _single_layer_weight_bytes(
        self, h: int, intermediate: int, dtype_bytes: int
    ) -> int:
        qkv = 3 * h * h
        o_proj = h * h
        gate = h * intermediate
        up = h * intermediate
        down = intermediate * h
        norms = 4 * h
        return (qkv + o_proj + gate + up + down + norms) * dtype_bytes

    def _single_layer_flops(
        self, h: int, intermediate: int, num_heads: int, head_dim: int
    ) -> int:
        qkv = 3 * 2 * h * h
        attn = 2 * 2 * num_heads * head_dim
        o_proj = 2 * h * h
        mlp_gate = 2 * h * intermediate
        mlp_up = 2 * h * intermediate
        mlp_down = 2 * intermediate * h
        return qkv + attn + o_proj + mlp_gate + mlp_up + mlp_down

    # ------------------------------------------------------------------
    # Microbenchmarks (CUDA)
    # ------------------------------------------------------------------

    def _bench_matmul_fp16(self, gpu_id: int, iterations: int = 20) -> float:
        try:
            import torch

            if not torch.cuda.is_available():
                return 0.0

            with torch.cuda.device(gpu_id):
                a = torch.randn(_LARGE_MATRIX_DIM, _LARGE_MATRIX_DIM, dtype=torch.float16, device="cuda")
                b = torch.randn(_LARGE_MATRIX_DIM, _LARGE_MATRIX_DIM, dtype=torch.float16, device="cuda")

                # Warmup
                for _ in range(5):
                    c = a @ b
                torch.cuda.synchronize()

                # Benchmark
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(iterations):
                    c = a @ b
                end.record()
                torch.cuda.synchronize()

                elapsed_ms = start.elapsed_time(end)
                elapsed_s = elapsed_ms / 1000.0

                flops_per_matmul = 2 * _LARGE_MATRIX_DIM ** 3
                total_flops = flops_per_matmul * iterations
                tflops = (total_flops / elapsed_s) / 1e12
                return round(tflops, 2)
        except Exception:
            return 0.0

    def _bench_memory_bandwidth(self, gpu_id: int, iterations: int = 20) -> float:
        try:
            import torch

            if not torch.cuda.is_available():
                return 0.0

            with torch.cuda.device(gpu_id):
                size = 256 * 1024 * 1024  # 256M elements
                a = torch.randn(size, dtype=torch.float16, device="cuda")
                b = torch.zeros(size, dtype=torch.float16, device="cuda")

                # Warmup
                for _ in range(3):
                    b.copy_(a)
                torch.cuda.synchronize()

                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(iterations):
                    b.copy_(a)
                end.record()
                torch.cuda.synchronize()

                elapsed_ms = start.elapsed_time(end)
                elapsed_s = elapsed_ms / 1000.0

                bytes_per_copy = size * 2  # fp16
                total_bytes = bytes_per_copy * iterations * 2  # read + write
                bw_gbps = (total_bytes / elapsed_s) / 1e9
                return round(bw_gbps, 2)
        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _device_count(self) -> int:
        try:
            import torch
            return torch.cuda.device_count() if torch.cuda.is_available() else 0
        except Exception:
            return 0

    def _get_device_name(self, gpu_id: int) -> str:
        try:
            import torch
            return torch.cuda.get_device_name(gpu_id)
        except Exception:
            return f"GPU-{gpu_id}"

    def _get_memory_info(self, gpu_id: int) -> tuple[int, int]:
        try:
            import torch
            total = torch.cuda.get_device_properties(gpu_id).total_memory
            free = total - torch.cuda.memory_allocated(gpu_id)
            return total, free
        except Exception:
            return 0, 0

    def _estimate_tflops_from_name(self, name: str) -> float:
        if "4090" in name or "H100" in name or "H200" in name:
            return 300.0
        if "4080" in name or "A100" in name or "A10" in name:
            return 150.0
        if "3090" in name or "A40" in name or "L40" in name:
            return 140.0
        if "3080" in name or "A30" in name or "V100" in name:
            return 80.0
        if "4070" in name or "L4" in name:
            return 80.0
        if "4060" in name or "T4" in name:
            return 50.0
        return 50.0

    def _estimate_bw_from_name(self, name: str) -> float:
        if "H200" in name:
            return 4800.0
        if "H100" in name or "H200" in name:
            return 3000.0
        if "A100" in name:
            return 2000.0
        if "4090" in name:
            return 1000.0
        if "A10" in name or "3090" in name or "L40" in name:
            return 900.0
        if "A30" in name or "A40" in name:
            return 900.0
        if "V100" in name:
            return 900.0
        if "3080" in name:
            return 760.0
        if "L4" in name or "4070" in name:
            return 500.0
        if "T4" in name or "4060" in name:
            return 300.0
        return 500.0

    def _match_known_spec(self, name: str) -> str | None:
        upper = name.upper()
        for known in _KNOWN_GPU_SPECS:
            if known.upper() in upper:
                return known
        return None

    def profile_to_dict(self, profile: GPUProfile) -> dict[str, Any]:
        return {
            "gpu_id": profile.gpu_id,
            "name": profile.name,
            "total_memory_gb": round(profile.total_memory_bytes / (1024**3), 2),
            "free_memory_gb": round(profile.free_memory_bytes / (1024**3), 2),
            "compute_tflops": profile.compute_tflops,
            "memory_bandwidth_gbps": profile.memory_bandwidth_gbps,
            "sm_count": profile.sm_count,
            "measured_tflops_fp16": profile.measured_tflops_fp16,
        }
