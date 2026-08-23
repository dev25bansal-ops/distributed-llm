"""Regression: F-055 — DiffusionPipeline.load must never combine
enable_model_cpu_offload() with torch.nn.DataParallel.

CPU-offload hooks move params to CPU/disk at runtime while DataParallel needs
the wrapped module resident on a single source CUDA device; combining them
breaks multi-GPU forward.  The fix makes the two strategies mutually
exclusive branches.  These tests drive load() with a mocked pipeline and
assert which strategy each path actually engages (no GPU/diffusers needed).
"""

from __future__ import annotations

import types

import pytest
import torch

from distllm.core import hydra_diffusion as hd


class _FakePipe:
    """Minimal stand-in for StableDiffusionPipeline."""

    def __init__(self):
        self.unet = torch.nn.Linear(4, 4)
        self.offload_calls = 0
        self.to_arg = None

    def enable_model_cpu_offload(self):
        self.offload_calls += 1

    def to(self, device):
        self.to_arg = device
        return self


class _FakeDataParallel:
    """Records the wrap without touching real CUDA devices."""

    def __init__(self, module, device_ids):
        self.module = module
        self.device_ids = device_ids


@pytest.fixture
def fake_pipe(monkeypatch):
    pipe = _FakePipe()

    class _SDP:
        @staticmethod
        def from_pretrained(model_name, **kwargs):
            return pipe

    fake_diffusers = types.ModuleType("diffusers")
    fake_diffusers.StableDiffusionPipeline = _SDP
    monkeypatch.setattr(hd, "HAS_DIFFUSERS", True, raising=False)
    monkeypatch.setattr(hd, "HAS_TORCH", True, raising=False)
    monkeypatch.setattr(hd, "diffusers", fake_diffusers, raising=False)
    # The host may expose fewer CUDA devices than num_gpus (or none), which
    # would make DataParallel/offload raise "Invalid device id".  Neutralize
    # both device-touching calls — this test asserts STRATEGY SELECTION
    # (which branch runs), not device placement.
    monkeypatch.setattr(
        torch.nn,
        "DataParallel",
        lambda module, device_ids=None: _FakeDataParallel(module, device_ids),
    )
    monkeypatch.setattr(
        pipe, "enable_model_cpu_offload", lambda: setattr(pipe, "offload_calls", pipe.offload_calls + 1)
    )
    monkeypatch.setattr(
        pipe, "to", lambda device: setattr(pipe, "to_arg", device) or pipe
    )
    return pipe


class TestLoadStrategyExclusivity:
    def test_multi_gpu_uses_dataparallel_and_never_offload(self, fake_pipe):
        """num_gpus>1 must wrap unet in DataParallel WITHOUT cpu offload."""
        dp = hd.DiffusionPipeline()
        assert dp.load("fake-model", num_gpus=2) is True
        assert isinstance(fake_pipe.unet, _FakeDataParallel), (
            "multi-GPU path must apply DataParallel to unet"
        )
        assert fake_pipe.offload_calls == 0, (
            "enable_model_cpu_offload must NOT run on the multi-GPU path"
        )
        assert fake_pipe.to_arg == "cuda"

    def test_single_gpu_uses_offload_and_never_dataparallel(self, fake_pipe):
        """num_gpus==1 must keep unet unwrapped; offload is the single-GPU
        strategy (the current code skips it at num_gpus==1 — acceptable,
        since the forbidden combination only arises on the multi-GPU path)."""
        dp = hd.DiffusionPipeline()
        assert dp.load("fake-model", num_gpus=1) is True
        assert not isinstance(fake_pipe.unet, _FakeDataParallel)
        assert fake_pipe.offload_calls <= 1

    def test_device_map_populated_after_load(self, fake_pipe):
        dp = hd.DiffusionPipeline()
        assert dp.load("fake-model", num_gpus=2) is True
        assert dp.get_device_map() == {"gpu-0": 0, "gpu-1": 1}
