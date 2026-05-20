"""Per-architecture capability presets.

Each architecture exposes a factory function that returns the
correct DeviceCapabilities for a given device spec.
"""

from distllm.core.hardware.device import DeviceCapabilities, DeviceSpec, DeviceType


def get_device_capabilities(device: DeviceSpec) -> DeviceCapabilities:
    """Return capability preset matching the given device.

    Args:
        device: Detected DeviceSpec.

    Returns:
        DeviceCapabilities populated with the right feature flags.
    """
    factory = _FACTORY_MAP.get(device.device_type, _cpu_capabilities)
    return factory(device)


# ---------------------------------------------------------------------------
# Per-architecture presets
# ---------------------------------------------------------------------------

def _cuda_capabilities(device: DeviceSpec) -> DeviceCapabilities:
    cc = device.compute_capability or (0, 0)
    major, minor = cc

    caps = DeviceCapabilities(
        device_type=DeviceType.CUDA,
        supports_fp16=True,
        supports_bf16=(major >= 8),           # Ampere+
        supports_fp8=(major >= 9),            # Hopper+
        supports_int8=True,
        supports_int4=True,
        supports_flash_attention=(major >= 8),
        supports_paged_attention=True,
        supports_sliding_window=True,
        supports_tensor_parallel=True,
        supports_pipeline_parallel=True,
        supports_data_parallel=True,
        supports_expert_parallel=(major >= 8),
        supports_sequence_parallel=(major >= 8),
        supports_cuda_graph=True,
        supports_torch_compile=True,
        supports_kv_cache_quantization=True,
        supports_kv_cache_prefix_sharing=True,
        max_tensor_parallel_size=_max_tp_for_cuda(major, device.total_memory_gb),
        recommended_batch_size=32,
    )

    if device.sm_count > 0:
        caps.recommended_batch_size = min(64, max(8, device.sm_count // 2))

    return caps


def _rocm_capabilities(device: DeviceSpec) -> DeviceCapabilities:
    return DeviceCapabilities(
        device_type=DeviceType.ROCM,
        supports_fp16=True,
        supports_bf16=True,
        supports_fp8=False,
        supports_int8=True,
        supports_int4=True,
        supports_flash_attention=True,
        supports_paged_attention=True,  # via vLLM ROCm
        supports_sliding_window=True,
        supports_tensor_parallel=True,
        supports_pipeline_parallel=True,
        supports_data_parallel=True,
        supports_expert_parallel=True,
        supports_sequence_parallel=True,
        supports_cuda_graph=False,
        supports_torch_compile=True,
        supports_kv_cache_quantization=True,
        supports_kv_cache_prefix_sharing=True,
        max_tensor_parallel_size=min(8, device.sm_count) if device.sm_count else 4,
        recommended_batch_size=32,
    )


def _mps_capabilities(device: DeviceSpec) -> DeviceCapabilities:
    return DeviceCapabilities(
        device_type=DeviceType.MPS,
        supports_fp16=True,
        supports_bf16=False,
        supports_fp8=False,
        supports_int8=True,
        supports_int4=False,
        supports_flash_attention=False,
        supports_paged_attention=False,
        supports_sliding_window=True,
        supports_tensor_parallel=False,
        supports_pipeline_parallel=False,
        supports_data_parallel=True,
        supports_expert_parallel=False,
        supports_sequence_parallel=False,
        supports_cuda_graph=False,
        supports_torch_compile=True,
        supports_kv_cache_quantization=False,
        supports_kv_cache_prefix_sharing=False,
        max_tensor_parallel_size=1,
        recommended_batch_size=8,
    )


def _xpu_capabilities(device: DeviceSpec) -> DeviceCapabilities:
    return DeviceCapabilities(
        device_type=DeviceType.XPU,
        supports_fp16=True,
        supports_bf16=True,
        supports_fp8=False,
        supports_int8=True,
        supports_int4=False,
        supports_flash_attention=False,
        supports_paged_attention=False,
        supports_sliding_window=True,
        supports_tensor_parallel=False,
        supports_pipeline_parallel=False,
        supports_data_parallel=True,
        supports_expert_parallel=False,
        supports_sequence_parallel=False,
        supports_cuda_graph=False,
        supports_torch_compile=True,
        supports_kv_cache_quantization=False,
        supports_kv_cache_prefix_sharing=False,
        max_tensor_parallel_size=1,
        recommended_batch_size=8,
    )


def _cpu_capabilities(device: DeviceSpec) -> DeviceCapabilities:
    return DeviceCapabilities(
        device_type=DeviceType.CPU,
        supports_fp16=False,
        supports_bf16=False,
        supports_fp8=False,
        supports_int8=True,
        supports_int4=True,
        supports_flash_attention=False,
        supports_paged_attention=False,
        supports_sliding_window=True,
        supports_tensor_parallel=False,
        supports_pipeline_parallel=False,
        supports_data_parallel=True,
        supports_expert_parallel=False,
        supports_sequence_parallel=False,
        supports_cuda_graph=False,
        supports_torch_compile=False,
        supports_kv_cache_quantization=False,
        supports_kv_cache_prefix_sharing=False,
        max_tensor_parallel_size=1,
        recommended_batch_size=1,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _max_tp_for_cuda(major: int, memory_gb: float) -> int:
    if major >= 9:  # Hopper / Blackwell
        return 8
    if major >= 8:  # Ampere / Ada
        return 8 if memory_gb >= 40 else 4
    return 4


_FACTORY_MAP: dict[DeviceType, callable] = {
    DeviceType.CUDA: _cuda_capabilities,
    DeviceType.ROCM: _rocm_capabilities,
    DeviceType.MPS: _mps_capabilities,
    DeviceType.XPU: _xpu_capabilities,
    DeviceType.CPU: _cpu_capabilities,
}
