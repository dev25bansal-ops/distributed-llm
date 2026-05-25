"""Tests: hardware detection (CUDA, CPU, ROCm), capabilities, device mgmt, backend selection.

Covers HardwareDetector mocks, DeviceCapabilities per-arch,
DeviceSpec construction, HardwareRegistry selection, BackendSelector.

Run: pytest tests/core/test_hardware_detection.py -v
"""

from unittest.mock import MagicMock, patch, PropertyMock

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
from distllm.core.hardware.capabilities import _max_tp_for_cuda


# ===========================================================================
# 1. Hardware Detection — CUDA
# ===========================================================================


class TestCudaDetection:
    """CUDA available → devices detected correctly with capabilities."""

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.device_count", return_value=2)
    @patch("torch.cuda.get_device_properties")
    def test_detect_cuda_returns_devices(self, mock_props, mock_count, mock_avail):
        def props_side_effect(i):
            mock = MagicMock()
            mock.name = "NVIDIA A100"
            mock.total_memory = 85899345920  # 80 GB
            mock.major = 8
            mock.minor = 0
            mock.multi_processor_count = 108
            mock.max_threads_per_multi_processor = 2048
            mock.clock_rate = 1410000  # kHz -> /1000 = 1410 MHz
            return mock
        mock_props.side_effect = props_side_effect

        detector = HardwareDetector()
        devices = detector.detect_cuda()
        assert len(devices) == 2
        for d in devices:
            assert d.device_type == DeviceType.CUDA
            assert d.is_accelerator
            assert d.total_memory_bytes == 85899345920
            assert d.total_memory_gb == 80.0
            assert d.compute_capability == (8, 0)
            assert d.sm_count == 108
            assert d.max_threads_per_sm == 2048
            assert d.clock_rate_mhz == 1410

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.device_count", return_value=1)
    @patch("torch.cuda.get_device_properties")
    def test_detect_cuda_hopper_fp8_capability(self, mock_props, mock_count, mock_avail):
        mock = MagicMock()
        mock.name = "NVIDIA H100"
        mock.total_memory = 83886080000  # ~80 GB
        mock.major = 9
        mock.minor = 0
        mock.multi_processor_count = 132
        mock.max_threads_per_multi_processor = 2048
        mock.clock_rate = 1980000000
        mock_props.return_value = mock

        detector = HardwareDetector()
        devices = detector.detect_cuda()
        assert len(devices) == 1
        d = devices[0]
        caps = get_device_capabilities(d)
        assert caps.supports_fp8 is True
        assert caps.supports_bf16 is True
        assert caps.max_tensor_parallel_size == 8

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.device_count", return_value=1)
    @patch("torch.cuda.get_device_properties")
    def test_detect_cuda_ampere(self, mock_props, mock_count, mock_avail):
        mock = MagicMock()
        mock.name = "NVIDIA A100"
        mock.total_memory = 17179869184  # 16 GB
        mock.major = 8
        mock.minor = 0
        mock.multi_processor_count = 80
        mock.max_threads_per_multi_processor = 2048
        mock.clock_rate = 1410000000
        mock_props.return_value = mock

        d = HardwareDetector().detect_cuda()[0]
        caps = get_device_capabilities(d)
        assert caps.supports_fp8 is False
        assert caps.supports_bf16 is True
        assert caps.supports_flash_attention is True
        assert caps.supports_expert_parallel is True
        assert caps.max_tensor_parallel_size == 4  # 16 GB < 40 GB

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.device_count", return_value=1)
    @patch("torch.cuda.get_device_properties")
    def test_detect_cuda_volta(self, mock_props, mock_count, mock_avail):
        mock = MagicMock()
        mock.name = "NVIDIA V100"
        mock.total_memory = 17179869184
        mock.major = 7
        mock.minor = 0
        mock.multi_processor_count = 80
        mock.max_threads_per_multi_processor = 2048
        mock.clock_rate = 1530000000
        mock_props.return_value = mock

        d = HardwareDetector().detect_cuda()[0]
        caps = get_device_capabilities(d)
        assert caps.supports_fp8 is False
        assert caps.supports_bf16 is False  # CC < 8
        assert caps.supports_fp16 is True
        assert caps.supports_flash_attention is False  # CC < 8
        assert caps.max_tensor_parallel_size == 4

    @patch("torch.cuda.is_available", return_value=False)
    def test_detect_cuda_not_available_returns_empty(self, mock_avail):
        devices = HardwareDetector().detect_cuda()
        assert devices == []

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.device_count", return_value=0)
    def test_detect_cuda_zero_devices(self, mock_count, mock_avail):
        devices = HardwareDetector().detect_cuda()
        assert devices == []


# ===========================================================================
# 2. Hardware Detection — CPU
# ===========================================================================


class TestCpuDetection:
    """No GPU available → CPU detected as fallback."""

    @patch("distllm.core.hardware.detect.HardwareDetector.detect_cuda", return_value=[])
    @patch("distllm.core.hardware.detect.HardwareDetector.detect_rocm", return_value=[])
    @patch("distllm.core.hardware.detect.HardwareDetector.detect_mps", return_value=[])
    @patch("distllm.core.hardware.detect.HardwareDetector.detect_xpu", return_value=[])
    @patch("psutil.virtual_memory")
    @patch("psutil.cpu_count", return_value=8)
    def test_detect_all_falls_back_to_cpu(self, mock_count, mock_mem, *mocks):
        mock_mem.return_value.total = 17179869184
        mock_mem.return_value.available = 8589934592

        detector = HardwareDetector()
        devices = detector.detect_all()
        assert len(devices) >= 1
        cpu_devices = [d for d in devices if d.device_type == DeviceType.CPU]
        assert len(cpu_devices) >= 1
        cpu = cpu_devices[0]
        assert cpu.is_accelerator is False
        assert cpu.backend == "llamacpp"

    @patch("distllm.core.hardware.detect.HardwareDetector.detect_cuda", return_value=[])
    @patch("distllm.core.hardware.detect.HardwareDetector.detect_rocm", return_value=[])
    @patch("distllm.core.hardware.detect.HardwareDetector.detect_mps", return_value=[])
    @patch("distllm.core.hardware.detect.HardwareDetector.detect_xpu", return_value=[])
    def test_get_primary_device_returns_cpu_when_no_gpu(self, *mocks):
        detector = HardwareDetector()
        primary = detector.get_primary_device()
        assert primary.device_type == DeviceType.CPU

    @patch("distllm.core.hardware.detect.HardwareDetector.detect_cuda", return_value=[])
    @patch("distllm.core.hardware.detect.HardwareDetector.detect_rocm", return_value=[])
    @patch("distllm.core.hardware.detect.HardwareDetector.detect_mps", return_value=[])
    @patch("distllm.core.hardware.detect.HardwareDetector.detect_xpu", return_value=[])
    def test_cpu_capabilities_no_acceleration(self, *mocks):
        detector = HardwareDetector()
        devices = detector.detect_all()
        cpu = next(d for d in devices if d.device_type == DeviceType.CPU)
        caps = get_device_capabilities(cpu)
        assert caps.supports_fp16 is False
        assert caps.supports_bf16 is False
        assert caps.supports_fp8 is False
        assert caps.supports_int8 is True
        assert caps.supports_int4 is True
        assert caps.supports_cuda_graph is False
        assert caps.supports_torch_compile is False
        assert caps.supports_tensor_parallel is False
        assert caps.supports_flash_attention is False
        assert caps.max_tensor_parallel_size == 1
        assert caps.recommended_batch_size == 1


# ===========================================================================
# 3. Hardware Detection — ROCm
# ===========================================================================


class TestRocmDetection:
    """AMD GPU → ROCm detected with correct capabilities."""

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.device_count", return_value=1)
    @patch("torch.cuda.get_device_properties")
    def test_detect_rocm_returns_device(self, mock_props, mock_count, mock_avail):
        mock = MagicMock()
        mock.name = "AMD MI250"
        mock.total_memory = 68719476736  # 64 GB
        mock.multi_processor_count = 220
        mock_props.return_value = mock

        with patch("torch.version.hip", "6.0.0", create=True):
            devices = HardwareDetector().detect_rocm()
        assert len(devices) == 1
        d = devices[0]
        assert d.device_type == DeviceType.ROCM
        assert d.is_accelerator
        assert "AMD" in d.name
        assert d.total_memory_gb == 64.0
        assert d.backend == "pytorch"

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.device_count", return_value=1)
    @patch("torch.cuda.get_device_properties")
    def test_rocm_capabilities(self, mock_props, mock_count, mock_avail):
        mock = MagicMock()
        mock.name = "AMD MI250"
        mock.total_memory = 68719476736
        mock.multi_processor_count = 220
        mock_props.return_value = mock

        with patch("torch.version.hip", "6.0.0", create=True):
            d = HardwareDetector().detect_rocm()[0]
        caps = get_device_capabilities(d)
        assert caps.device_type == DeviceType.ROCM
        assert caps.supports_fp16 is True
        assert caps.supports_bf16 is True
        assert caps.supports_fp8 is False
        assert caps.supports_int8 is True
        assert caps.supports_flash_attention is True
        assert caps.supports_cuda_graph is False
        assert caps.supports_tensor_parallel is True
        assert caps.max_tensor_parallel_size >= 1

    def test_detect_rocm_no_hip_returns_empty(self):
        devices = HardwareDetector().detect_rocm()
        assert devices == []

    @patch("torch.cuda.is_available", return_value=False)
    def test_detect_rocm_cuda_not_available_returns_empty(self, mock_avail):
        with patch("torch.version.hip", "6.0.0", create=True):
            devices = HardwareDetector().detect_rocm()
        assert devices == []


# ===========================================================================
# 4. Capability Detection
# ===========================================================================


class TestCapabilityDetection:
    """Compute capability, VRAM, SM count detection."""

    def test_max_tp_for_cuda_hopper(self):
        assert _max_tp_for_cuda(9, 80.0) == 8

    def test_max_tp_for_cuda_ampere_large_vram(self):
        assert _max_tp_for_cuda(8, 80.0) == 8
        assert _max_tp_for_cuda(8, 40.0) == 8

    def test_max_tp_for_cuda_ampere_small_vram(self):
        assert _max_tp_for_cuda(8, 16.0) == 4
        assert _max_tp_for_cuda(8, 39.0) == 4

    def test_max_tp_for_cuda_pre_ampere(self):
        assert _max_tp_for_cuda(7, 32.0) == 4
        assert _max_tp_for_cuda(6, 12.0) == 4

    def test_cuda_recommended_batch_from_sm_count(self):
        d = DeviceSpec(device_type=DeviceType.CUDA, device_id=0, sm_count=100,
                       compute_capability=(8, 0))
        caps = get_device_capabilities(d)
        assert caps.recommended_batch_size == 50  # 100 // 2

    def test_cuda_recommended_batch_minimum(self):
        d = DeviceSpec(device_type=DeviceType.CUDA, device_id=0, sm_count=4,
                       compute_capability=(8, 0))
        caps = get_device_capabilities(d)
        assert caps.recommended_batch_size == 8  # min(64, max(8, 2)) = 8

    def test_cuda_recommended_batch_capped(self):
        d = DeviceSpec(device_type=DeviceType.CUDA, device_id=0, sm_count=200,
                       compute_capability=(8, 0))
        caps = get_device_capabilities(d)
        assert caps.recommended_batch_size == 64  # min(64, 100) = 64

    def test_cuda_no_sm_count_fallback_batch(self):
        d = DeviceSpec(device_type=DeviceType.CUDA, device_id=0, sm_count=0,
                       compute_capability=(8, 0))
        caps = get_device_capabilities(d)
        assert caps.recommended_batch_size == 32  # fallback

    def test_rocm_sm_count_based_tp(self):
        d = DeviceSpec(device_type=DeviceType.ROCM, device_id=0, sm_count=110)
        caps = get_device_capabilities(d)
        assert caps.max_tensor_parallel_size == 8  # min(8, 110)

    def test_rocm_no_sm_count_fallback_tp(self):
        d = DeviceSpec(device_type=DeviceType.ROCM, device_id=0, sm_count=0)
        caps = get_device_capabilities(d)
        assert caps.max_tensor_parallel_size == 4

    def test_mps_capabilities(self):
        d = DeviceSpec(device_type=DeviceType.MPS, device_id=0)
        caps = get_device_capabilities(d)
        assert caps.supports_fp16 is True
        assert caps.supports_bf16 is False
        assert caps.supports_fp8 is False
        assert caps.supports_int4 is False
        assert caps.supports_flash_attention is False
        assert caps.supports_tensor_parallel is False
        assert caps.supports_cuda_graph is False
        assert caps.supports_kv_cache_quantization is False
        assert caps.recommended_batch_size == 8

    def test_xpu_capabilities(self):
        d = DeviceSpec(device_type=DeviceType.XPU, device_id=0)
        caps = get_device_capabilities(d)
        assert caps.supports_fp16 is True
        assert caps.supports_bf16 is True
        assert caps.supports_fp8 is False
        assert caps.supports_int8 is True
        assert caps.supports_int4 is False
        assert caps.supports_flash_attention is False
        assert caps.supports_tensor_parallel is False
        assert caps.recommended_batch_size == 8

    def test_cpu_capabilities(self):
        d = DeviceSpec(device_type=DeviceType.CPU, device_id=0)
        caps = get_device_capabilities(d)
        assert caps.supports_fp16 is False
        assert caps.supports_bf16 is False
        assert caps.supports_fp8 is False
        assert caps.supports_int8 is True
        assert caps.supports_int4 is True
        assert caps.supports_tensor_parallel is False
        assert caps.supports_cuda_graph is False
        assert caps.supports_torch_compile is False
        assert caps.recommended_batch_size == 1


# ===========================================================================
# 5. Device Management
# ===========================================================================


class TestDeviceManagement:
    """Device selection and allocation via HardwareRegistry."""

    def test_select_device_by_valid_id(self):
        registry = HardwareRegistry()
        registry.reset()
        devices = registry.detect()
        if len(devices) > 0:
            target = devices[0].device_id
            registry.select_device(target)
            assert registry.selected_device is not None
            assert registry.selected_device.device_id == target

    def test_select_device_invalid_id_does_not_crash(self):
        registry = HardwareRegistry()
        registry.reset()
        registry.detect()
        registry.select_device(9999)
        selected = registry.selected_device
        assert selected is None

    def test_get_device_ids(self):
        registry = HardwareRegistry()
        registry.reset()
        registry.detect()
        ids = registry.get_device_ids()
        assert isinstance(ids, list)
        assert all(isinstance(i, int) for i in ids)

    def test_count_by_type_cpu(self):
        registry = HardwareRegistry()
        registry.reset()
        registry.detect()
        assert registry.count_by_type(DeviceType.CPU) >= 1

    def test_count_by_type_cuda_when_not_available(self):
        registry = HardwareRegistry()
        registry.reset()
        assert registry.count_by_type(DeviceType.CUDA) >= 0

    def test_registry_summary_keys(self):
        registry = HardwareRegistry()
        registry.reset()
        registry.detect()
        s = registry.summary()
        assert "total_devices" in s
        assert "primary" in s
        assert "by_type" in s

    def test_registry_reset_clears_everything(self):
        registry = HardwareRegistry()
        registry.detect()
        registry.reset()
        assert registry.devices == []
        assert registry.primary_device is None
        assert registry.selected_device is None
        assert registry.get_capabilities() is None

    def test_get_capabilities_no_device_returns_none(self):
        registry = HardwareRegistry()
        registry.reset()
        assert registry.get_capabilities() is None


# ===========================================================================
# 6. Backend Selection
# ===========================================================================


class TestBackendSelection:
    """Hardware type determines optimal backend."""

    def test_preferred_cuda(self):
        selector = BackendSelector()
        device = DeviceSpec(device_type=DeviceType.CUDA, device_id=0)
        variant = selector.preferred_backend(device)
        assert variant in (BackendVariant.VLLM, BackendVariant.PYTORCH)

    def test_preferred_rocm(self):
        selector = BackendSelector()
        device = DeviceSpec(device_type=DeviceType.ROCM, device_id=0)
        variant = selector.preferred_backend(device)
        assert variant in (BackendVariant.VLLM, BackendVariant.PYTORCH)

    def test_preferred_mps(self):
        selector = BackendSelector()
        device = DeviceSpec(device_type=DeviceType.MPS, device_id=0)
        variant = selector.preferred_backend(device)
        assert variant == BackendVariant.PYTORCH

    def test_preferred_xpu(self):
        selector = BackendSelector()
        device = DeviceSpec(device_type=DeviceType.XPU, device_id=0)
        variant = selector.preferred_backend(device)
        assert variant == BackendVariant.PYTORCH

    def test_preferred_cpu(self):
        selector = BackendSelector()
        device = DeviceSpec(device_type=DeviceType.CPU, device_id=0)
        variant = selector.preferred_backend(device)
        assert variant in (BackendVariant.LLAMACPP, BackendVariant.PYTORCH)

    def test_available_backends_list(self):
        selector = BackendSelector()
        device = DeviceSpec(device_type=DeviceType.CUDA, device_id=0)
        backends = selector.available_backends(device)
        assert len(backends) >= 1
        for b in backends:
            assert isinstance(b, BackendVariant)

    def test_get_adapter_class_pytorch(self):
        selector = BackendSelector()
        cls = selector.get_adapter_class(BackendVariant.PYTORCH)
        assert cls is not None

    def test_adapter_class_is_callable(self):
        selector = BackendSelector()
        cls = selector.get_adapter_class(BackendVariant.PYTORCH)
        assert callable(cls)

    def test_cached_imports(self):
        selector = BackendSelector()
        d1 = selector.preferred_backend(DeviceSpec(device_type=DeviceType.MPS, device_id=0))
        d2 = selector.preferred_backend(DeviceSpec(device_type=DeviceType.MPS, device_id=0))
        assert d1 == d2


# ===========================================================================
# 7. HardwareDetector Edge Cases
# ===========================================================================


class TestHardwareDetectorEdgeCases:
    """Edge cases for HardwareDetector."""

    def test_detect_cuda_exception_handled(self):
        with patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.device_count", return_value=1), \
             patch("torch.cuda.get_device_properties", side_effect=RuntimeError("no GPU")):
            devices = HardwareDetector().detect_cuda()
            assert devices == []

    def test_detect_all_cuda_exception_falls_through(self):
        with patch.object(HardwareDetector, "detect_cuda",
                          side_effect=RuntimeError("CUDA error")), \
             patch.object(HardwareDetector, "detect_rocm", return_value=[]), \
             patch.object(HardwareDetector, "detect_mps", return_value=[]), \
             patch.object(HardwareDetector, "detect_xpu", return_value=[]):
            devices = HardwareDetector().detect_all()
            assert len(devices) >= 1
            assert any(d.device_type == DeviceType.CPU for d in devices)

    def test_get_primary_device_no_torch(self):
        with patch.dict("sys.modules", {"torch": None}):
            detector = HardwareDetector()
            devices = detector.detect_all()
            assert any(d.device_type == DeviceType.CPU for d in devices)

    def test_detect_xpu_not_available(self):
        devices = HardwareDetector().detect_xpu()
        assert devices == []
