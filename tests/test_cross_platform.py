"""Tests for cross-platform heterogeneous GPU support.

Uses direct file imports to avoid circular dependency in distllm/__init__.py.
"""

import importlib.util
import sys
import types
from pathlib import Path

import pytest

SRC_DIR = Path(__file__).resolve().parent.parent / "src"


def _make_fake_package(name: str, path: Path):
    """Create a fake package in sys.modules to avoid __init__.py loading."""
    mod = types.ModuleType(name)
    mod.__path__ = [str(path)]
    mod.__package__ = name
    sys.modules.setdefault(name, mod)
    return mod


# Inject fake packages to prevent real distllm/__init__.py from loading
_make_fake_package("distllm", SRC_DIR / "distllm")
_make_fake_package("distllm.core", SRC_DIR / "distllm/core")
_make_fake_package("distllm.dist", SRC_DIR / "distllm/dist")
_make_fake_package("distllm.dist.partition", SRC_DIR / "distllm/dist/partition")
_make_fake_package("distllm.backends", SRC_DIR / "distllm/backends")


def _load_module(rel_path: str):
    """Load a module directly from file, bypassing distllm/__init__.py.

    The fake package entries in sys.modules prevent Python from
    importing distllm/__init__.py when resolving ``from distllm.X import Y``.
    """
    filepath = SRC_DIR / rel_path
    if not filepath.exists():
        raise FileNotFoundError(f"{filepath} not found")

    # Build dotted name within the fake distllm package
    rel = filepath.relative_to(SRC_DIR)
    parts = list(rel.parent.parts) + [filepath.stem]
    # Skip "distllm" if it's the first part (it's already our fake package)
    if parts[0] == "distllm":
        dotted = ".".join(parts)
    else:
        dotted = "distllm." + ".".join(parts)

    if dotted in sys.modules:
        return sys.modules[dotted]

    spec = importlib.util.spec_from_file_location(dotted, filepath, submodule_search_locations=[])
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {filepath}")

    mod = importlib.util.module_from_spec(spec)
    sys.modules[dotted] = mod
    spec.loader.exec_module(mod)
    return mod


# Pre-load commonly used modules
_constants = _load_module("distllm/constants.py")
_device_registry = _load_module("distllm/core/device_registry.py")
_profiles = _load_module("distllm/dist/partition/profiles.py")


class TestDeviceRegistry:
    def test_detect_platform_returns_string(self):
        platform = _device_registry.detect_platform()
        assert platform in ("cuda", "rocm", "mps", "xpu", "cpu")

    def test_detect_all_devices_returns_list(self):
        devices = _device_registry.detect_all_devices()
        assert isinstance(devices, list)
        if devices:
            dev = devices[0]
            assert hasattr(dev, "device_type")
            assert hasattr(dev, "device_family")
            assert hasattr(dev, "device_id")

    def test_device_family_mapping(self):
        assert _constants.DEVICE_TO_FAMILY["cuda"] == _constants.DeviceFamily.NVIDIA
        assert _constants.DEVICE_TO_FAMILY["rocm"] == _constants.DeviceFamily.AMD
        assert _constants.DEVICE_TO_FAMILY["mps"] == _constants.DeviceFamily.APPLE
        assert _constants.DEVICE_TO_FAMILY["xpu"] == _constants.DeviceFamily.INTEL
        assert _constants.DEVICE_TO_FAMILY["cpu"] == _constants.DeviceFamily.CPU

    def test_device_family_unknown_default(self):
        assert _device_registry.get_device_family("nonexistent") == _constants.DeviceFamily.UNKNOWN

    def test_format_device_summary(self):
        devices = [
            _device_registry.DeviceInfo(device_type="cuda", device_family=_constants.DeviceFamily.NVIDIA, device_id=0, name="RTX 4090", total_memory_bytes=24 * 1024**3),
            _device_registry.DeviceInfo(device_type="rocm", device_family=_constants.DeviceFamily.AMD, device_id=0, name="RX 7900 XTX", total_memory_bytes=24 * 1024**3),
        ]
        summary = _device_registry.format_device_summary(devices)
        assert "CUDA" in summary
        assert "ROCM" in summary
        assert "24.0 GB" in summary


class TestHeterogeneousScheduler:
    _mod = None

    @classmethod
    def _get_mod(cls):
        if cls._mod is None:
            cls._mod = _load_module("distllm/core/heterogeneous_scheduler.py")
        return cls._mod

    def test_build_heterogeneous_cluster(self):
        mod = self._get_mod()
        configs = [
            {"node_id": "node0", "host": "10.0.0.1", "port": 50051, "device_type": "cuda"},
            {"node_id": "node1", "host": "10.0.0.2", "port": 50051, "device_type": "rocm"},
            {"node_id": "node2", "host": "10.0.0.3", "port": 50051, "device_type": "mps"},
        ]
        cluster = mod.build_heterogeneous_cluster(configs, total_layers=32)
        assert len(cluster.nodes) == 3
        assert cluster.is_heterogeneous
        assert cluster.total_layers == 32

    def test_assign_layers_proportional(self):
        mod = self._get_mod()
        configs = [
            {"node_id": "big", "host": "10.0.0.1", "port": 50051, "device_type": "cuda", "total_memory": 80 * 1024**3, "gpu_name": "A100"},
            {"node_id": "small", "host": "10.0.0.2", "port": 50051, "device_type": "cuda", "total_memory": 24 * 1024**3, "gpu_name": "RTX 4090"},
        ]
        cluster = mod.build_heterogeneous_cluster(configs, total_layers=40)
        cluster = mod.assign_layers_proportional(cluster)
        assert cluster.nodes[0].start_layer <= cluster.nodes[1].start_layer
        assert cluster.nodes[0].end_layer - cluster.nodes[0].start_layer > 0
        assert cluster.nodes[1].end_layer == cluster.total_layers - 1

    def test_order_nodes_by_throughput(self):
        mod = self._get_mod()
        fast = mod.HeterogeneousNode(
            node_id="fast", host="", port=0,
            device_info=_device_registry.DeviceInfo(device_type="cuda", device_family=_constants.DeviceFamily.NVIDIA, device_id=0, tflops_fp16=300.0),
            throughput_score=300.0,
        )
        slow = mod.HeterogeneousNode(
            node_id="slow", host="", port=0,
            device_info=_device_registry.DeviceInfo(device_type="cuda", device_family=_constants.DeviceFamily.NVIDIA, device_id=1, tflops_fp16=50.0),
            throughput_score=50.0,
        )
        cluster = mod.HeterogeneousCluster(nodes=[slow, fast])
        cluster = mod.order_nodes_by_throughput(cluster)
        assert cluster.nodes[0].node_id == "fast"
        assert cluster.nodes[1].node_id == "slow"

    def test_estimate_heterogeneous_throughput(self):
        mod = self._get_mod()
        cluster = mod.HeterogeneousCluster(nodes=[
            mod.HeterogeneousNode(
                node_id="n1", host="", port=0,
                device_info=_device_registry.DeviceInfo(device_type="cuda", device_family=_constants.DeviceFamily.NVIDIA, device_id=0),
                throughput_score=100.0,
            ),
            mod.HeterogeneousNode(
                node_id="n2", host="", port=0,
                device_info=_device_registry.DeviceInfo(device_type="cuda", device_family=_constants.DeviceFamily.NVIDIA, device_id=1),
                throughput_score=80.0,
            ),
        ])
        tput = mod.estimate_heterogeneous_throughput(cluster)
        assert tput > 0

    def test_schedule_heterogeneous_pipeline(self):
        mod = self._get_mod()
        configs = [
            {"node_id": "nvidia-node", "host": "10.0.0.1", "port": 50051, "device_type": "cuda", "total_memory": 80 * 1024**3, "gpu_name": "A100"},
            {"node_id": "amd-node", "host": "10.0.0.2", "port": 50051, "device_type": "rocm", "total_memory": 48 * 1024**3, "gpu_name": "MI250"},
            {"node_id": "apple-node", "host": "10.0.0.3", "port": 50051, "device_type": "mps", "total_memory": 64 * 1024**3, "gpu_name": "Apple M3 Max"},
        ]
        assignments = mod.schedule_heterogeneous_pipeline(configs, total_layers=40)
        assert len(assignments) == 3
        for a in assignments:
            assert "start_layer" in a
            assert "end_layer" in a
            assert "device_type" in a
            assert "device_family" in a
        assert assignments[0]["device_family"] == "nvidia"
        assert assignments[1]["device_family"] == "amd"
        assert assignments[2]["device_family"] == "apple"

    def test_get_device_compatibility_map(self):
        mod = self._get_mod()
        compat = mod.get_device_compatibility_map()
        assert "cuda" in compat
        assert "rocm" in compat
        assert "mps" in compat
        assert "cpu" in compat


class TestGPUProfilerCrossPlatform:
    def test_gpu_profile_dataclass(self):
        p = _profiles.GPUProfile(gpu_id=0, name="Test GPU", total_memory_bytes=8589934592)
        assert p.gpu_id == 0
        assert p.total_memory_bytes == 8589934592

    def test_layer_weights(self):
        lw = _profiles.LayerWeights(layer_id=0, weight_memory_bytes=1000, activation_memory_bytes=500)
        assert lw.total_memory_bytes == 1500

    def test_known_gpu_specs_include_all_platforms(self):
        platforms = set(v[5] for v in _profiles._KNOWN_GPU_SPECS.values())
        assert "nvidia" in platforms
        assert "amd" in platforms
        assert "intel" in platforms
        assert "apple" in platforms

    def test_known_gpu_specs_has_specific_gpus(self):
        assert "RX 7900 XTX" in _profiles._KNOWN_GPU_SPECS
        assert "Arc A770" in _profiles._KNOWN_GPU_SPECS
        assert "Apple M3 Max" in _profiles._KNOWN_GPU_SPECS
        assert "MI300X" in _profiles._KNOWN_GPU_SPECS
        assert "RTX 4090" in _profiles._KNOWN_GPU_SPECS
        assert "H100" in _profiles._KNOWN_GPU_SPECS

    def test_match_known_spec(self):
        profiler = _profiles.GPUProfiler()
        result = profiler._match_known_spec("NVIDIA RTX 4090")
        assert result == "RTX 4090"
        result = profiler._match_known_spec("AMD Radeon RX 7900 XTX")
        assert result == "RX 7900 XTX"
        result = profiler._match_known_spec("Intel Arc A770")
        assert result == "Arc A770"

    def test_estimate_from_known_spec(self):
        fp16, fp32, bw, sm, bus, plat = _profiles._KNOWN_GPU_SPECS["RX 7900 XTX"]
        assert fp16 == 122.0
        assert plat == "amd"

    def test_profile_all_gpus_mocked(self):
        profiler = _profiles.GPUProfiler()
        result = profiler.profile_all_gpus()
        assert isinstance(result, list)

    def test_device_count_returns_int(self):
        profiler = _profiles.GPUProfiler()
        count = profiler._device_count()
        assert isinstance(count, int)
        assert count >= 0


class TestConstants:
    def test_device_enum_has_all_platforms(self):
        assert _constants.Device.CUDA.value == "cuda"
        assert _constants.Device.ROCM.value == "rocm"
        assert _constants.Device.MPS.value == "mps"
        assert _constants.Device.XPU.value == "xpu"
        assert _constants.Device.VULKAN.value == "vulkan"

    def test_device_family_enum(self):
        assert _constants.DeviceFamily.NVIDIA.value == "nvidia"
        assert _constants.DeviceFamily.AMD.value == "amd"
        assert _constants.DeviceFamily.APPLE.value == "apple"
        assert _constants.DeviceFamily.INTEL.value == "intel"

    def test_platform_backend_priority(self):
        assert "nvidia" in _constants.PLATFORM_BACKEND_PRIORITY
        assert "amd" in _constants.PLATFORM_BACKEND_PRIORITY
        assert "apple" in _constants.PLATFORM_BACKEND_PRIORITY
        assert "intel" in _constants.PLATFORM_BACKEND_PRIORITY
        assert _constants.PLATFORM_BACKEND_PRIORITY["amd"]["llamacpp"] == 9
        assert _constants.PLATFORM_BACKEND_PRIORITY["apple"]["llamacpp"] == 9
        assert _constants.PLATFORM_BACKEND_PRIORITY["nvidia"]["vllm"] == 10
        assert _constants.PLATFORM_BACKEND_PRIORITY["intel"]["onnx"] == 9


class TestGPUSpecs:
    def test_nvidia_specs_present(self):
        for name, spec in _profiles._KNOWN_GPU_SPECS.items():
            if spec[5] == "nvidia":
                assert len(spec) == 6
                assert spec[0] > 0

    def test_amd_specs_present(self):
        for name, spec in _profiles._KNOWN_GPU_SPECS.items():
            if spec[5] == "amd":
                assert len(spec) == 6
                assert spec[0] > 0

    def test_intel_specs_present(self):
        for name, spec in _profiles._KNOWN_GPU_SPECS.items():
            if spec[5] == "intel":
                assert len(spec) == 6

    def test_apple_specs_present(self):
        for name, spec in _profiles._KNOWN_GPU_SPECS.items():
            if spec[5] == "apple":
                assert len(spec) == 6

    def test_estimate_bw_nvidia(self):
        profiler = _profiles.GPUProfiler()
        assert profiler._estimate_bw_from_name("NVIDIA H100") > 0

    def test_estimate_bw_amd(self):
        profiler = _profiles.GPUProfiler()
        assert profiler._estimate_bw_from_name("AMD RX 7900 XTX") > 0

    def test_estimate_bw_intel(self):
        profiler = _profiles.GPUProfiler()
        assert profiler._estimate_bw_from_name("Intel Arc A770") > 0

    def test_estimate_bw_apple(self):
        profiler = _profiles.GPUProfiler()
        assert profiler._estimate_bw_from_name("Apple M3 Max") > 0

    def test_estimate_tflops_amd(self):
        profiler = _profiles.GPUProfiler()
        assert profiler._estimate_tflops_from_name("AMD MI300X") > 0

    def test_estimate_bw_mi300(self):
        profiler = _profiles.GPUProfiler()
        bw = profiler._estimate_bw_from_name("AMD Instinct MI300X")
        assert bw == 5300.0

    def test_estimate_tflops_rtx4090(self):
        profiler = _profiles.GPUProfiler()
        assert profiler._estimate_tflops_from_name("NVIDIA RTX 4090") == 330.0


class TestBackendRegistryDetectDevice:
    """Tests that don't require importing distllm.backends.registry."""

    def test_device_registry_has_detect(self):
        assert hasattr(_device_registry, "detect_platform")
        assert hasattr(_device_registry, "detect_all_devices")


class TestWorkerGetDevice:
    """Tests for worker _get_device logic (without importing WorkerNode)."""

    def test_get_device_auto(self):
        platform = _device_registry.detect_platform()
        assert platform in ("cuda", "rocm", "mps", "xpu", "cpu")
