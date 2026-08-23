from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    from distllm.core.device_registry import detect_platform
except ImportError:
    detect_platform = None


@dataclass
class GPUProfile:
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
    layer_id: int
    layer_type: str = "transformer"
    weight_memory_bytes: int = 0
    activation_memory_bytes: int = 0
    flops_per_token: int = 0
    flops_per_seq: int = 0
    kv_cache_bytes_per_token: int = 0

    @property
    def total_memory_bytes(self) -> int:
        return self.weight_memory_bytes + self.activation_memory_bytes


_KNOWN_GPU_SPECS: dict[str, tuple[float, float, float, int, int, str]] = {
    # NVIDIA
    "A100": (312, 156, 2039, 108, 5120, "nvidia"),
    "A100-SXM-80GB": (312, 156, 2039, 108, 5120, "nvidia"),
    "A100-PCIE-40GB": (312, 156, 1555, 108, 5120, "nvidia"),
    "A10": (125, 62, 600, 72, 3072, "nvidia"),
    "A10G": (125, 62, 600, 72, 3072, "nvidia"),
    "A30": (165, 82, 933, 56, 3072, "nvidia"),
    "A40": (150, 75, 696, 84, 3840, "nvidia"),
    "H100": (989, 494, 3350, 132, 5120, "nvidia"),
    "H100-SXM-80GB": (989, 494, 3350, 132, 5120, "nvidia"),
    "H100-PCIE-80GB": (989, 494, 2000, 114, 5120, "nvidia"),
    "H200": (989, 494, 4800, 132, 5120, "nvidia"),
    "V100": (125, 125, 900, 80, 4096, "nvidia"),
    "V100-SXM-32GB": (125, 125, 900, 80, 4096, "nvidia"),
    "V100-PCIE-32GB": (125, 125, 900, 80, 4096, "nvidia"),
    "RTX 4090": (330, 165, 1008, 128, 3840, "nvidia"),
    "RTX 4080": (180, 90, 716, 76, 2560, "nvidia"),
    "RTX 3090": (142, 71, 936, 82, 3840, "nvidia"),
    "RTX 3080": (90, 45, 760, 68, 3200, "nvidia"),
    "RTX 4070": (89, 44, 504, 46, 1920, "nvidia"),
    "RTX 4060": (50, 25, 272, 24, 1280, "nvidia"),
    "L4": (121, 30, 300, 24, 1920, "nvidia"),
    "L40S": (362, 91, 864, 84, 3840, "nvidia"),
    "T4": (65, 32, 320, 40, 2560, "nvidia"),
    "P100": (21, 21, 732, 56, 4096, "nvidia"),
    # AMD
    "MI300X": (653, 327, 5300, 220, 8192, "amd"),
    "MI250X": (383, 192, 3277, 220, 8192, "amd"),
    "MI250": (383, 192, 3277, 220, 8192, "amd"),
    "MI210": (181, 91, 1638, 104, 4096, "amd"),
    "MI100": (185, 185, 1229, 120, 5120, "amd"),
    "MI50": (26, 26, 1024, 60, 4096, "amd"),
    "RX 7900 XTX": (122, 61, 960, 96, 6144, "amd"),
    "RX 7900 XT": (103, 52, 800, 84, 5120, "amd"),
    "RX 7900 GRE": (71, 36, 576, 64, 4096, "amd"),
    "RX 7800 XT": (74, 37, 624, 60, 4096, "amd"),
    "RX 7700 XT": (70, 35, 432, 54, 4096, "amd"),
    "RX 7600": (44, 22, 288, 32, 2048, "amd"),
    "RX 6900 XT": (48, 48, 512, 80, 4096, "amd"),
    "RX 6800 XT": (41, 41, 512, 72, 4096, "amd"),
    "RX 6700 XT": (26, 26, 384, 40, 3072, "amd"),
    # Intel
    "Arc A770": (39, 20, 560, 64, 4096, "intel"),
    "Arc A750": (33, 17, 512, 56, 4096, "intel"),
    "Arc A580": (23, 12, 384, 48, 3072, "intel"),
    "Arc A380": (11, 6, 186, 16, 1536, "intel"),
    "Max 1550": (52, 26, 2048, 128, 4096, "intel"),
    "Max 1350": (36, 18, 1366, 96, 3072, "intel"),
    "Max 1100": (18, 9, 1024, 56, 2048, "intel"),
    # Apple (estimated — metrics from ANE/GPU)
    "Apple M1": (2.6, 1.3, 68, 7, 1024, "apple"),
    "Apple M1 Pro": (5.3, 2.6, 136, 16, 1024, "apple"),
    "Apple M1 Max": (10.6, 5.3, 272, 32, 1024, "apple"),
    "Apple M1 Ultra": (21.2, 10.6, 544, 64, 1024, "apple"),
    "Apple M2": (3.6, 1.8, 102, 10, 1024, "apple"),
    "Apple M2 Pro": (6.8, 3.4, 204, 19, 1024, "apple"),
    "Apple M2 Max": (13.6, 6.8, 408, 38, 1024, "apple"),
    "Apple M2 Ultra": (27.2, 13.6, 816, 76, 1024, "apple"),
    "Apple M3": (4.1, 2.0, 153, 10, 1024, "apple"),
    "Apple M3 Pro": (7.4, 3.7, 306, 18, 1024, "apple"),
    "Apple M3 Max": (14.8, 7.4, 612, 40, 1024, "apple"),
    "Apple M3 Ultra": (29.6, 14.8, 1224, 80, 1024, "apple"),
    "Apple M4": (4.6, 2.3, 204, 10, 1024, "apple"),
    "Apple M4 Pro": (9.2, 4.6, 408, 20, 1024, "apple"),
    "Apple M4 Max": (18.4, 9.2, 816, 40, 1024, "apple"),
}

_LARGE_MATRIX_DIM = 4096
_LARGE_MATRIX_ELEMS = _LARGE_MATRIX_DIM * _LARGE_MATRIX_DIM
_BENCHMARK_MATRIX_BYTES = _LARGE_MATRIX_ELEMS * 2


class GPUProfiler:
    def profile_all_gpus(self) -> list[GPUProfile]:
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
            fp16_tflops, fp32_tflops, mem_bw, sm_count, mem_bus, _platform = known
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

        # 3.3: Run all benchmarks (gracefully degrade if unavailable)
        measured_fp16 = self._bench_matmul_fp16(gpu_id)
        measured_bw = self._bench_memory_bandwidth(gpu_id)
        strided_bw = self._bench_strided_bandwidth(gpu_id)
        p2p_bw = self._bench_p2p_bandwidth(gpu_id)
        cpu_tflops = self._bench_cpu_flops()

        if measured_fp16 > 0:
            profile.measured_tflops_fp16 = measured_fp16
            profile.compute_tflops = measured_fp16

        if measured_bw > 0:
            profile.measured_memory_bandwidth_gbps = measured_bw
            profile.memory_bandwidth_gbps = measured_bw

        # Store strided and P2P measurements for cost model
        profile._strided_bw_gbps = strided_bw  # type: ignore[attr-defined]
        profile._p2p_bw_gbps = p2p_bw  # type: ignore[attr-defined]
        profile._cpu_tflops = cpu_tflops  # type: ignore[attr-defined]

        return profile

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
        kv_heads = max(1, num_heads)
        layers: list[LayerWeights] = []

        activation_per_token = hidden_size * dtype_bytes

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

    def _bench_matmul_fp16(self, gpu_id: int, iterations: int = 20) -> float:
        try:
            import torch
            plat = detect_platform()

            if plat == "cuda" or plat == "rocm":
                device_str = "cuda"
                synchronize = torch.cuda.synchronize
                Event = torch.cuda.Event
            elif plat == "xpu":
                device_str = "xpu"
                synchronize = torch.xpu.synchronize
                Event = torch.xpu.Event
            elif plat == "mps":
                device_str = "mps"
                import time as _time

                with torch.device(device_str):
                    a = torch.randn(_LARGE_MATRIX_DIM, _LARGE_MATRIX_DIM, dtype=torch.float16, device=device_str)
                    b = torch.randn(_LARGE_MATRIX_DIM, _LARGE_MATRIX_DIM, dtype=torch.float16, device=device_str)
                    for _ in range(3):
                        c = a @ b
                    torch.mps.synchronize()
                    start = _time.time()
                    for _ in range(iterations):
                        c = a @ b
                    torch.mps.synchronize()
                    elapsed_s = _time.time() - start
                    flops = 2 * _LARGE_MATRIX_DIM ** 3
                    tflops = (flops * iterations / elapsed_s) / 1e12
                    return round(tflops, 2)
            else:
                return 0.0

            with torch.device(device_str):
                a = torch.randn(_LARGE_MATRIX_DIM, _LARGE_MATRIX_DIM, dtype=torch.float16, device=device_str)
                b = torch.randn(_LARGE_MATRIX_DIM, _LARGE_MATRIX_DIM, dtype=torch.float16, device=device_str)
                for _ in range(5):
                    c = a @ b
                synchronize()
                start = Event(enable_timing=True)
                end = Event(enable_timing=True)
                start.record()
                for _ in range(iterations):
                    c = a @ b
                end.record()
                synchronize()
                elapsed_ms = start.elapsed_time(end)
                elapsed_s = elapsed_ms / 1000.0
                flops = 2 * _LARGE_MATRIX_DIM ** 3
                tflops = (flops * iterations / elapsed_s) / 1e12
                return round(tflops, 2)
        except Exception:
            return 0.0

    def _bench_memory_bandwidth(self, gpu_id: int, iterations: int = 20) -> float:
        try:
            import torch
            plat = detect_platform()

            if plat == "cuda" or plat == "rocm":
                device_str = "cuda"
                synchronize = torch.cuda.synchronize
                Event = torch.cuda.Event
            elif plat == "xpu":
                device_str = "xpu"
                synchronize = torch.xpu.synchronize
                Event = torch.xpu.Event
            elif plat == "mps":
                device_str = "mps"
                import time as _time

                with torch.device(device_str):
                    size = 64 * 1024 * 1024
                    a = torch.randn(size, dtype=torch.float16, device=device_str)
                    b = torch.zeros(size, dtype=torch.float16, device=device_str)
                    for _ in range(3):
                        b.copy_(a)
                    torch.mps.synchronize()
                    start = _time.time()
                    for _ in range(iterations):
                        b.copy_(a)
                    torch.mps.synchronize()
                    elapsed_s = _time.time() - start
                    bytes_total = size * 2 * iterations * 2
                    bw_gbps = (bytes_total / elapsed_s) / 1e9
                    return round(bw_gbps, 2)
            else:
                return 0.0

            with torch.device(device_str):
                size = 256 * 1024 * 1024
                a = torch.randn(size, dtype=torch.float16, device=device_str)
                b = torch.zeros(size, dtype=torch.float16, device=device_str)
                for _ in range(3):
                    b.copy_(a)
                synchronize()
                start = Event(enable_timing=True)
                end = Event(enable_timing=True)
                start.record()
                for _ in range(iterations):
                    b.copy_(a)
                end.record()
                synchronize()
                elapsed_ms = start.elapsed_time(end)
                elapsed_s = elapsed_ms / 1000.0
                bytes_total = size * 2 * iterations * 2
                bw_gbps = (bytes_total / elapsed_s) / 1e9
                return round(bw_gbps, 2)
        except Exception:
            return 0.0

    def _device_count(self) -> int:
        try:
            import torch
            plat = detect_platform()
            if plat == "cuda" or plat == "rocm":
                return torch.cuda.device_count()
            if plat == "xpu":
                return torch.xpu.device_count()
            if plat == "mps":
                return 1
            return 0
        except Exception:
            return 0

    def _bench_strided_bandwidth(self, gpu_id: int, iterations: int = 10) -> float:
        """3.3: Benchmark strided memory access pattern.

        Real workloads use strided access (not contiguous copies),
        which can be 30-50% slower than peak bandwidth.
        """
        try:
            import torch
            plat = detect_platform()
            if plat not in ("cuda", "rocm", "xpu", "mps"):
                return 0.0

            device_str = "cuda" if plat in ("cuda", "rocm") else plat
            size = 64 * 1024 * 1024  # 128MB
            stride = 256  # 256-element stride

            with torch.device(device_str):
                a = torch.randn(size, dtype=torch.float16, device=device_str)
                b = torch.zeros(size // stride, dtype=torch.float16, device=device_str)

                for _ in range(3):
                    b.copy_(a[::stride])

                if device_str == "cuda":
                    torch.cuda.synchronize()
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    for _ in range(iterations):
                        b.copy_(a[::stride])
                    end.record()
                    torch.cuda.synchronize()
                    elapsed_s = start.elapsed_time(end) / 1000.0
                else:
                    import time as _time
                    if device_str == "mps":
                        torch.mps.synchronize()
                    start_t = _time.time()
                    for _ in range(iterations):
                        b.copy_(a[::stride])
                    if device_str == "mps":
                        torch.mps.synchronize()
                    elapsed_s = _time.time() - start_t

                bytes_total = (size // stride) * 2 * iterations * 2
                return round((bytes_total / elapsed_s) / 1e9, 2)
        except Exception:
            return 0.0

    def _bench_p2p_bandwidth(self, gpu_id: int, iterations: int = 10) -> float:
        """3.3: Benchmark peer-to-peer GPU bandwidth.

        Measures actual P2P bandwidth between GPU 0 and GPU 1
        (or single GPU if only one available).
        """
        try:
            import torch
            plat = detect_platform()
            if plat not in ("cuda", "rocm"):
                return 0.0

            if torch.cuda.device_count() < 2:
                return 0.0

            src = 0
            dst = 1
            size = 64 * 1024 * 1024  # 128MB

            with torch.device(f"cuda:{src}"):
                a = torch.randn(size, dtype=torch.float16, device=f"cuda:{src}")

            with torch.device(f"cuda:{dst}"):
                b = torch.zeros(size, dtype=torch.float16, device=f"cuda:{dst}")

            # Warmup
            for _ in range(3):
                b.copy_(a)
            torch.cuda.synchronize(device=dst)

            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iterations):
                b.copy_(a)
            end.record()
            torch.cuda.synchronize(device=dst)

            elapsed_s = start.elapsed_time(end) / 1000.0
            bytes_total = size * 2 * iterations * 2
            return round((bytes_total / elapsed_s) / 1e9, 2)
        except Exception:
            return 0.0

    def _bench_cpu_flops(self, iterations: int = 5) -> float:
        """3.3: Benchmark CPU compute capability.

        Uses numpy/OpenBLAS matrix multiply for CPU TFLOPS.
        Returns measured TFLOPS (fp32) for CPU fallback cost model.
        """
        try:
            import time as _time
            import numpy as np

            dim = 2048
            a = np.random.randn(dim, dim).astype(np.float32)
            b = np.random.randn(dim, dim).astype(np.float32)

            # Warmup
            _ = a @ b

            start = _time.time()
            for _ in range(iterations):
                _ = a @ b
            elapsed_s = _time.time() - start

            flops = 2 * dim ** 3 * iterations
            tflops = flops / elapsed_s / 1e12
            return round(tflops, 2)
        except Exception:
            return 0.0

    def _get_device_name(self, gpu_id: int) -> str:
        try:
            import torch
            plat = detect_platform()
            if plat == "cuda" or plat == "rocm":
                return torch.cuda.get_device_name(gpu_id)
            if plat == "xpu":
                return torch.xpu.get_device_name(gpu_id)
            if plat == "mps":
                import platform as _platform
                return f"Apple {_platform.machine()}"
            return f"GPU-{gpu_id}"
        except Exception:
            return f"GPU-{gpu_id}"

    def _get_memory_info(self, gpu_id: int) -> tuple[int, int]:
        try:
            import torch
            plat = detect_platform()
            if plat == "cuda" or plat == "rocm":
                total = torch.cuda.get_device_properties(gpu_id).total_memory
                free = total - torch.cuda.memory_allocated(gpu_id)
                return total, free
            if plat == "xpu":
                total = torch.xpu.get_device_properties(gpu_id).total_memory
                free = total - torch.xpu.memory_allocated(gpu_id)
                return total, free
            if plat == "mps":
                from distllm.core.device_registry import _get_mps_memory
                total = _get_mps_memory()
                used = getattr(torch.mps, "current_allocated_memory", lambda: 0)()
                return total, total - used
            return 0, 0
        except Exception:
            return 0, 0

    def _estimate_tflops_from_name(self, name: str) -> float:
        upper = name.upper()
        known_match = self._match_known_spec(upper)
        if known_match:
            return _KNOWN_GPU_SPECS[known_match][0]
        if "RX" in upper or "MI" in upper:
            return 50.0
        if "ARC" in upper or "MAX" in upper:
            return 20.0
        if "APPLE M" in upper or "M1" in upper or "M2" in upper or "M3" in upper or "M4" in upper:
            return 5.0
        if "4090" in upper or "H100" in upper or "H200" in upper:
            return 300.0
        if "4080" in upper or "A100" in upper or "A10" in upper:
            return 150.0
        if "3090" in upper or "A40" in upper or "L40" in upper:
            return 140.0
        if "3080" in upper or "A30" in upper:
            return 80.0
        if "V100" in upper:
            return 125.0
        if "4070" in upper or "L4" in upper:
            return 80.0
        if "4060" in upper or "T4" in upper:
            return 50.0
        return 50.0

    def _estimate_bw_from_name(self, name: str) -> float:
        upper = name.upper()
        known_match = self._match_known_spec(upper)
        if known_match:
            return _KNOWN_GPU_SPECS[known_match][2]
        if "RX 7900" in upper:
            return 960.0
        if "RX" in upper:
            return 500.0
        if "MI300" in upper:
            return 5300.0
        if "MI250" in upper or "MI200" in upper:
            return 3200.0
        if "MI100" in upper:
            return 1200.0
        if "ARC" in upper or "MAX" in upper:
            return 500.0
        if "M4" in upper:
            return 200.0
        if "M3" in upper:
            return 150.0
        if "M2" in upper or "M1" in upper:
            return 100.0
        if "H200" in upper:
            return 4800.0
        if "H100" in upper:
            return 3000.0
        if "A100" in upper:
            return 2000.0
        if "4090" in upper:
            return 1000.0
        if "A10" in upper or "3090" in upper or "L40" in upper:
            return 900.0
        if "A30" in upper:
            return 933.0
        if "A40" in upper:
            return 696.0
        if "V100" in upper:
            return 900.0
        if "3080" in upper:
            return 760.0
        if "L4" in upper or "4070" in upper:
            return 500.0
        if "T4" in upper or "4060" in upper:
            return 300.0
        return 500.0

    def _match_known_spec(self, name: str) -> str | None:
        upper = name.upper().replace("-", " ").replace("_", " ")
        for known in _KNOWN_GPU_SPECS:
            known_upper = known.upper().replace("-", " ").replace("_", " ")
            if known_upper in upper and len(known_upper) > 3:
                return known
        for known in _KNOWN_GPU_SPECS:
            short = known.upper().split()[-1] if " " in known else known.upper()
            if short in upper and len(short) > 2:
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
