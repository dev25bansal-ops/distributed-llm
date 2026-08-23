"""Tests for GPUProfiler and GPUInfo (no GPU required, no mocks)."""

from __future__ import annotations

from tests._import_helper import bootstrap_fake_packages, load_module

bootstrap_fake_packages()

_mod = load_module("distllm/core/gpu_profiler.py")
GPUProfiler = _mod.GPUProfiler
GPUInfo = _mod.GPUInfo


class TestGPUInfo:
    """GPUInfo dataclass construction."""

    def test_default_construction(self) -> None:
        info = GPUInfo(gpu_id=0, name="TestGPU", total_memory=8589934592)
        assert info.gpu_id == 0
        assert info.name == "TestGPU"
        assert info.total_memory == 8589934592
        assert info.used_memory == 0
        assert info.free_memory == 0
        assert info.utilization == 0.0
        assert info.compute_tflops == 0.0
        assert info.memory_bandwidth_gbps == 0.0
        assert info.sm_count == 0

    def test_creation_with_all_fields(self) -> None:
        info = GPUInfo(
            gpu_id=1,
            name="NVIDIA A100",
            total_memory=85899345920,
            used_memory=42949672960,
            free_memory=42949672960,
            utilization=45.5,
            compute_tflops=19.5,
            memory_bandwidth_gbps=1555.0,
            sm_count=108,
        )
        assert info.gpu_id == 1
        assert info.name == "NVIDIA A100"
        assert info.total_memory == 85899345920
        assert info.used_memory == 42949672960
        assert info.free_memory == 42949672960
        assert info.utilization == 45.5
        assert info.compute_tflops == 19.5
        assert info.memory_bandwidth_gbps == 1555.0
        assert info.sm_count == 108

    def test_memory_consistency(self) -> None:
        total = 8589934592
        info = GPUInfo(gpu_id=0, name="GPU", total_memory=total, used_memory=2147483648, free_memory=6442450944)
        assert info.used_memory + info.free_memory <= info.total_memory


class TestGPUProfiler:
    """GPUProfiler -- works without CUDA (enumerate_gpus returns []).

    Note: These tests run on any system since they don't require a GPU.
    When CUDA is unavailable, enumerate_gpus returns an empty list.
    """

    def test_default_construction(self) -> None:
        profiler = GPUProfiler()
        assert isinstance(profiler, GPUProfiler)

    def test_enumerate_gpus_returns_list(self) -> None:
        profiler = GPUProfiler()
        gpus = profiler.enumerate_gpus()
        assert isinstance(gpus, list)
        # On a system without CUDA, this will be an empty list.
        # On a system with CUDA, it returns GPUInfo objects.
        if len(gpus) > 0:
            assert all(isinstance(g, GPUInfo) for g in gpus)
        else:
            import torch
            assert not torch.cuda.is_available() or torch.cuda.device_count() == 0
