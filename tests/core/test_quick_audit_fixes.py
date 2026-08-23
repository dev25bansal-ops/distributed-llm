"""Regression tests for three quick Audit-01 fixes:
- F-020 GPUResourceManager.snapshot used/free MB were swapped
- F-056 NeuralBanditRouter crashed on instantiation (missing import torch)
- F-038 `distllm system doctor` never ran (doctor's argparse choked on the
  Typer subcommand tokens; rich.group.Group import was also stale)
"""

from __future__ import annotations

import pytest


class TestGPUSnapshotUsedFree:
    def test_snapshot_used_and_free_not_swapped(self, monkeypatch):
        import torch

        from distllm.core import device_registry, gpu_resource_manager as grm

        monkeypatch.setattr(device_registry, "detect_platform", lambda: "cuda")
        # memory_allocated/memory_reserved take bytes; 300 MB used.
        monkeypatch.setattr(torch.cuda, "memory_allocated", lambda dev: 300.0 * 1024**2)
        monkeypatch.setattr(torch.cuda, "memory_reserved", lambda dev: 400.0 * 1024**2)
        props = type("P", (), {"total_memory": 1024 * 1024 * 1024})()
        monkeypatch.setattr(torch.cuda, "get_device_properties", lambda dev: props)

        mgr = grm.GPUResourceManager()
        mgr._devices.add(0)
        mgr._total_per_device[0] = 1000.0  # MB

        snap = mgr.snapshot(0)
        assert snap is not None
        assert snap.used_mb == 300.0, "used_mb must equal allocated memory"
        assert snap.free_mb == pytest.approx(1000.0 - 300.0), "free_mb must be total - used"


class TestNeuralBanditRouterImport:
    def test_instantiation_does_not_nameerror(self):
        from unittest.mock import MagicMock

        from distllm.core.learning_router import NeuralBanditRouter

        router = NeuralBanditRouter(base_router=MagicMock(), models=["m1"])
        assert router is not None


class TestSystemDoctorCLI:
    def test_doctor_main_accepts_argv(self):
        """The doctor CLI must parse an explicit argv (used by `system doctor`)
        instead of only sys.argv which contains the Typer subcommand tokens."""
        from distllm.cli import doctor

        with pytest.raises(SystemExit) as excinfo:
            doctor.main(["--help"])
        assert excinfo.value.code == 0

    def test_doctor_module_imports_cleanly(self):
        """rich.group.Group is gone in rich 13 — the import must use rich.console."""
        import distllm.cli.doctor  # noqa: F401  (import-time smoke test)
