"""Tests for the multi-architecture hardware abstraction layer."""

from __future__ import annotations

import pytest

from distllm.core.hardware import (
    BackendSelector,
    BackendVariant,
    DeviceCapabilities,
    DeviceSpec,
    DeviceType,
    HardwareDetector,
    HardwareRegistry,
    get_device_capabilities,
)


class TestDeviceType:
    def test_from_string_cuda(self) -> None:
        assert DeviceType.from_string("cuda") == DeviceType.CUDA
        assert DeviceType.from_string("nvidia") == DeviceType.CUDA
        assert DeviceType.from_string("gpu") == DeviceType.CUDA

    def test_from_string_rocm(self) -> None:
        assert DeviceType.from_string("rocm") == DeviceType.ROCM
        assert DeviceType.from_string("amd") == DeviceType.ROCM

    def test_from_string_mps(self) -> None:
        assert DeviceType.from_string("mps") == DeviceType.MPS
        assert DeviceType.from_string("metal") == DeviceType.MPS
        assert DeviceType.from_string("apple") == DeviceType.MPS

    def test_from_string_xpu(self) -> None:
        assert DeviceType.from_string("xpu") == DeviceType.XPU
        assert DeviceType.from_string("intel") == DeviceType.XPU
        assert DeviceType.from_string("oneapi") == DeviceType.XPU

    def test_from_string_cpu(self) -> None:
        assert DeviceType.from_string("cpu") == DeviceType.CPU

    def test_from_string_unknown(self) -> None:
        assert DeviceType.from_string("unknown") == DeviceType.UNKNOWN
        assert DeviceType.from_string("nonsense") == DeviceType.UNKNOWN


class TestDeviceSpec:
    def test_memory_gb(self) -> None:
        spec = DeviceSpec(
            device_type=DeviceType.CUDA,
            device_id=0,
            total_memory_bytes=8589934592,  # 8 GB
        )
        assert spec.total_memory_gb == 8.0

    def test_is_accelerator(self) -> None:
        for dt in (DeviceType.CUDA, DeviceType.ROCM, DeviceType.MPS, DeviceType.XPU):
            spec = DeviceSpec(device_type=dt, device_id=0)
            assert spec.is_accelerator
        spec = DeviceSpec(device_type=DeviceType.CPU, device_id=0)
        assert not spec.is_accelerator

    def test_is_type_helpers(self) -> None:
        spec = DeviceSpec(device_type=DeviceType.CUDA, device_id=0)
        assert spec.is_cuda
        assert not spec.is_rocm
        assert not spec.is_mps
        assert not spec.is_xpu
        assert not spec.is_cpu

    def test_summary(self) -> None:
        spec = DeviceSpec(
            device_type=DeviceType.CUDA,
            device_id=0,
            name="Test GPU",
            total_memory_bytes=8589934592,
            backend="pytorch",
        )
        assert "cuda:0" in spec.summary()
        assert "Test GPU" in spec.summary()
        assert "8.0GB" in spec.summary()

    def test_to_dict(self) -> None:
        spec = DeviceSpec(
            device_type=DeviceType.CUDA,
            device_id=0,
            name="Test GPU",
            total_memory_bytes=8589934592,
            compute_capability=(8, 0),
            backend="pytorch",
            sm_count=80,
            clock_rate_mhz=1500,
        )
        d = spec.to_dict()
        assert d["device_type"] == "cuda"
        assert d["total_memory_gb"] == 8.0
        assert d["compute_capability"] == "8.0"


class TestDeviceCapabilities:
    def test_preferred_dtype(self) -> None:
        caps = DeviceCapabilities(supports_fp8=True)
        assert caps.preferred_dtype == "fp8"
        caps = DeviceCapabilities(supports_bf16=True)
        assert caps.preferred_dtype == "bf16"
        caps = DeviceCapabilities(supports_fp16=True)
        assert caps.preferred_dtype == "fp16"
        caps = DeviceCapabilities()
        assert caps.preferred_dtype == "fp32"

    def test_supported_precisions(self) -> None:
        caps = DeviceCapabilities(
            supports_fp8=True,
            supports_bf16=True,
            supports_int8=True,
        )
        precisions = caps._supported_precisions()
        assert "fp8" in precisions
        assert "bf16" in precisions
        assert "fp32" in precisions  # always included
        assert "int4" not in precisions

    def test_to_dict(self) -> None:
        caps = DeviceCapabilities(
            device_type=DeviceType.CUDA,
            supports_tensor_parallel=True,
            supports_cuda_graph=True,
        )
        d = caps.to_dict()
        assert d["device_type"] == "cuda"
        assert d["tensor_parallel"] is True
        assert d["cuda_graph"] is True


@pytest.mark.parametrize(
    "device_type,expected_flags",
    [
        (
            DeviceType.CUDA,
            {
                "supports_fp16": True,
                "supports_tensor_parallel": True,
                "supports_pipeline_parallel": True,
                "supports_cuda_graph": True,
            },
        ),
        (
            DeviceType.ROCM,
            {
                "supports_fp16": True,
                "supports_bf16": True,
                "supports_tensor_parallel": True,
                "supports_cuda_graph": False,
            },
        ),
        (
            DeviceType.MPS,
            {
                "supports_fp16": True,
                "supports_bf16": False,
                "supports_tensor_parallel": False,
                "supports_flash_attention": False,
                "supports_cuda_graph": False,
            },
        ),
        (
            DeviceType.XPU,
            {
                "supports_fp16": True,
                "supports_bf16": True,
                "supports_tensor_parallel": False,
                "supports_paged_attention": False,
            },
        ),
        (
            DeviceType.CPU,
            {
                "supports_fp16": False,
                "supports_int8": True,
                "supports_int4": True,
                "supports_tensor_parallel": False,
                "supports_cuda_graph": False,
                "supports_torch_compile": False,
            },
        ),
    ],
)
def test_get_device_capabilities_presets(
    device_type: DeviceType, expected_flags: dict[str, bool]
) -> None:
    device = DeviceSpec(device_type=device_type, device_id=0, sm_count=40, total_memory_bytes=17179869184)
    caps = get_device_capabilities(device)
    for flag, expected in expected_flags.items():
        assert getattr(caps, flag) == expected, f"{flag}: expected {expected}, got {getattr(caps, flag)}"


class TestBackendSelector:
    def test_preferred_cuda(self) -> None:
        selector = BackendSelector()
        device = DeviceSpec(device_type=DeviceType.CUDA, device_id=0)
        # In CI without vLLM, may fall back to PyTorch
        variant = selector.preferred_backend(device)
        assert variant in (BackendVariant.VLLM, BackendVariant.PYTORCH, BackendVariant.LLAMACPP)

    def test_preferred_cpu(self) -> None:
        selector = BackendSelector()
        device = DeviceSpec(device_type=DeviceType.CPU, device_id=0)
        variant = selector.preferred_backend(device)
        assert variant in (BackendVariant.LLAMACPP, BackendVariant.PYTORCH)

    def test_preferred_mps(self) -> None:
        selector = BackendSelector()
        device = DeviceSpec(device_type=DeviceType.MPS, device_id=0)
        variant = selector.preferred_backend(device)
        # MPS doesn't support vLLM or llama.cpp typically
        assert variant == BackendVariant.PYTORCH

    def test_preferred_xpu(self) -> None:
        selector = BackendSelector()
        device = DeviceSpec(device_type=DeviceType.XPU, device_id=0)
        variant = selector.preferred_backend(device)
        assert variant == BackendVariant.PYTORCH

    def test_available_backends(self) -> None:
        selector = BackendSelector()
        device = DeviceSpec(device_type=DeviceType.CUDA, device_id=0)
        backends = selector.available_backends(device)
        assert isinstance(backends, list)
        for b in backends:
            assert isinstance(b, BackendVariant)

    def test_get_adapter_class(self) -> None:
        selector = BackendSelector()
        # PyTorch is always available
        cls = selector.get_adapter_class(BackendVariant.PYTORCH)
        assert cls is not None

    def test_import_cache_works(self) -> None:
        selector = BackendSelector()
        device = DeviceSpec(device_type=DeviceType.MPS, device_id=0)
        v1 = selector.preferred_backend(device)
        v2 = selector.preferred_backend(device)
        assert v1 == v2


class TestHardwareRegistry:
    def test_singleton(self) -> None:
        r1 = HardwareRegistry()
        r2 = HardwareRegistry()
        assert r1 is r2

    def test_detect_returns_list(self) -> None:
        registry = HardwareRegistry()
        devices = registry.detect()
        assert isinstance(devices, list)

    def test_primary_device(self) -> None:
        registry = HardwareRegistry()
        registry.detect()
        primary = registry.primary_device
        # At minimum a CPU device should be detected
        assert primary is not None
        assert isinstance(primary, DeviceSpec)

    def test_get_capabilities_primary(self) -> None:
        registry = HardwareRegistry()
        registry.detect()
        caps = registry.get_capabilities()
        assert caps is not None
        assert isinstance(caps, DeviceCapabilities)

    def test_get_capabilities_by_device(self) -> None:
        registry = HardwareRegistry()
        devices = registry.detect()
        for dev in devices:
            caps = registry.get_capabilities(dev)
            assert caps is not None
            assert caps.device_type == dev.device_type

    def test_select_device(self) -> None:
        registry = HardwareRegistry()
        registry.detect()
        devices = registry.devices
        if devices:
            target_id = devices[0].device_id
            registry.select_device(target_id)
            selected = registry.selected_device
            assert selected is not None
            assert selected.device_id == target_id

    def test_count_by_type(self) -> None:
        registry = HardwareRegistry()
        registry.detect()
        assert registry.count_by_type(DeviceType.CPU) >= 1

    def test_summary(self) -> None:
        registry = HardwareRegistry()
        registry.detect()
        summary = registry.summary()
        assert "total_devices" in summary
        assert "primary" in summary
        assert "by_type" in summary

    def test_reset(self) -> None:
        registry = HardwareRegistry()
        registry.detect()
        registry.reset()
        assert registry.devices == []
        assert registry.primary_device is None


class TestHardwareDetector:
    def test_detect_all(self) -> None:
        detector = HardwareDetector()
        devices = detector.detect_all()
        assert isinstance(devices, list)
        # At minimum CPU should be detected
        assert any(d.device_type == DeviceType.CPU for d in devices)

    def test_get_primary_device(self) -> None:
        detector = HardwareDetector()
        primary = detector.get_primary_device()
        assert isinstance(primary, DeviceSpec)
        assert primary.device_type in DeviceType
