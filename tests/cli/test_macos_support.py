"""Regression tests: the CLI must not tell a MacBook user their machine is broken.

Reproduced before the fix, on a simulated darwin host with no nvidia-smi:
``distllm devices`` printed Memory n/a (both /proc/meminfo and the Windows
ctypes probe were the only backends), and ``distllm doctor`` emitted two
``error`` rows for CUDA and exited 1.  Separately, ``distllm devices`` showed
``cuda_version: "Field"`` because ``_gpu_info`` queried the nonexistent
nvidia-smi field ``cuda_version`` and parsed the diagnostic text.

Each test here fails against the pre-fix code and passes after it.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest


class TestMemoryProbePerPlatform:
    """``_memory_total_available`` must consult the right backend per OS."""

    def test_darwin_uses_psutil_not_proc(self):
        from distllm.cli import devices

        with (
            patch.object(devices, "_read_memory_psutil", return_value=(16, 8)) as ps,
            patch.object(devices, "_read_meminfo_linux") as lin,
        ):
            total, avail = devices._memory_total_available("Darwin")
            ps.assert_called_once()
            lin.assert_not_called()
        assert (total, avail) == (16, 8)

    def test_linux_uses_proc(self):
        from distllm.cli import devices

        with (
            patch.object(devices, "_read_meminfo_linux", return_value=(32, 16)) as lin,
            patch.object(devices, "_read_memory_psutil") as ps,
        ):
            devices._memory_total_available("Linux")
            lin.assert_called_once()
            ps.assert_not_called()

    def test_windows_uses_ctypes(self):
        from distllm.cli import devices

        with patch.object(devices, "_read_meminfo_windows", return_value=(64, 32)) as win:
            devices._memory_total_available("Windows")
            win.assert_called_once()

    def test_psutil_probe_returns_real_bytes(self):
        """psutil is a core dependency; the probe must return non-zero here."""
        from distllm.cli.devices import _read_memory_psutil

        total, avail = _read_memory_psutil()
        assert total > 0
        assert avail > 0
        assert avail <= total

    def test_psutil_probe_degrades_to_zero_on_import_error(self):
        from distllm.cli import devices

        with patch.dict("sys.modules", {"psutil": None}):
            assert devices._read_memory_psutil() == (0, 0)


class TestRunDevicesOnDarwin:
    """``run_devices(system=...)`` must populate memory on every platform."""

    def test_darwin_reports_memory(self):
        from distllm.cli import devices

        with patch.object(devices, "_read_memory_psutil", return_value=(17_179_869_184, 8_589_934_592)):
            result = devices.run_devices(system="Darwin")
        assert result["memory"]["total_bytes"] == 17_179_869_184
        assert result["memory"]["available_bytes"] == 8_589_934_592

    def test_darwin_does_not_read_linux_driver_file(self):
        """The NVIDIA kernel-module probe is Linux-only."""
        from distllm.cli import devices

        with patch.object(devices, "_driver_version_linux") as drv:
            with patch.object(devices, "_read_memory_psutil", return_value=(1, 1)):
                devices.run_devices(system="Darwin")
            drv.assert_not_called()


class TestNvidiaSmiCudaVersion:
    """The CUDA version must come from the header, never from a bad query."""

    def test_parses_modern_umd_header(self):
        from distllm.cli.devices import _nvidia_smi_cuda_version

        header = "| NVIDIA-SMI 621.01  KMD Version: 621.01  CUDA UMD Version: 13.5  |"
        with patch("distllm.cli.devices._safe_run", return_value=header):
            assert _nvidia_smi_cuda_version() == "13.5"

    def test_parses_legacy_header(self):
        from distllm.cli.devices import _nvidia_smi_cuda_version

        header = "| NVIDIA-SMI 550.54  Driver Version: 550.54  CUDA Version: 12.4 |"
        with patch("distllm.cli.devices._safe_run", return_value=header):
            assert _nvidia_smi_cuda_version() == "12.4"

    def test_never_returns_the_query_error_string(self):
        """The pre-fix bug: a bad field made nvidia-smi answer ``Field ...``."""
        from distllm.cli.devices import _nvidia_smi_cuda_version

        error = 'Field "cuda_version" is not a valid field to query.'
        with patch("distllm.cli.devices._safe_run", return_value=error):
            assert _nvidia_smi_cuda_version() == "unknown"

    def test_unknown_on_empty_output(self):
        from distllm.cli.devices import _nvidia_smi_cuda_version

        with patch("distllm.cli.devices._safe_run", return_value=""):
            assert _nvidia_smi_cuda_version() == "unknown"


class TestMpsGpuDetection:
    """A Mac's Metal GPU must appear in ``distllm devices`` output."""

    def _fake_torch(self, mps_available: bool):
        import types

        return types.SimpleNamespace(
            version=types.SimpleNamespace(cuda=None),
            cuda=types.SimpleNamespace(is_available=lambda: False, device_count=lambda: 0),
            backends=types.SimpleNamespace(
                mps=types.SimpleNamespace(is_available=lambda: mps_available),
            ),
        )

    def test_mps_device_is_reported(self):
        from distllm.cli import devices

        with (
            patch.dict("sys.modules", {"torch": self._fake_torch(True)}),
            patch("distllm.cli.devices.platform.machine", return_value="arm64"),
            patch("distllm.cli.devices._read_memory_psutil", return_value=(16 * 1024**3, 8 * 1024**3)),
        ):
            gpus = devices._gpu_info()

        assert len(gpus) == 1
        assert "MPS" in gpus[0]["name"]
        assert gpus[0]["cuda_version"].startswith("n/a")
        assert gpus[0]["vram_bytes"] == 16 * 1024**3

    def test_no_mps_device_yields_empty_list(self):
        from distllm.cli import devices

        with (
            patch.dict("sys.modules", {"torch": self._fake_torch(False)}),
            patch("distllm.cli.devices._safe_run", return_value=""),
        ):
            gpus = devices._gpu_info()

        assert gpus == []


class TestDoctorOnDarwin:
    """A healthy MacBook must not produce error rows from the CUDA checks.

    The doctor exits non-zero when any category reports an ``error``, so an
    error here means a correct Mac is told to fix a GPU that cannot exist.
    """

    def _torch_patch(self, has_mps):
        import types

        fake_torch = types.SimpleNamespace(
            __version__="2.11.0",
            cuda=types.SimpleNamespace(is_available=lambda: False, device_count=lambda: 0),
            backends=types.SimpleNamespace(
                mps=types.SimpleNamespace(is_available=lambda: has_mps),
            ),
        )
        return patch.dict("sys.modules", {"torch": fake_torch})

    def test_cuda_check_is_not_an_error_on_darwin(self):
        from distllm.cli import doctor

        with (
            self._torch_patch(has_mps=True),
            patch("distllm.cli.doctor.sys.platform", "darwin"),
            patch("distllm.cli.doctor.subprocess.run", side_effect=FileNotFoundError),
        ):
            results = doctor._check_cuda()

        cuda_rows = [r for r in results if r["check"] == "CUDA available"]
        assert cuda_rows, "the CUDA row must still be present"
        assert cuda_rows[0]["status"] == "ok"
        assert "macOS" in cuda_rows[0]["value"]

    def test_cuda_warns_but_does_not_error_without_metal(self):
        from distllm.cli import doctor

        with (
            self._torch_patch(has_mps=False),
            patch("distllm.cli.doctor.sys.platform", "darwin"),
            patch("distllm.cli.doctor.subprocess.run", side_effect=FileNotFoundError),
        ):
            results = doctor._check_cuda()

        cuda_row = next(r for r in results if r["check"] == "CUDA available")
        assert cuda_row["status"] == "warn"

    def test_nvidia_smi_row_is_skipped_on_darwin(self):
        from distllm.cli import doctor

        with (
            self._torch_patch(has_mps=True),
            patch("distllm.cli.doctor.sys.platform", "darwin"),
        ):
            results = doctor._check_cuda()

        smi_row = next(r for r in results if r["check"] == "nvidia-smi")
        assert smi_row["status"] == "skip"

    def test_model_compat_is_info_not_error_on_darwin(self):
        from distllm.cli import doctor

        with (
            self._torch_patch(has_mps=True),
            patch("distllm.cli.doctor.sys.platform", "darwin"),
        ):
            results = doctor._check_model_compatibility("llama-8b")

        mem_row = next(r for r in results if r["check"] == "GPU memory")
        assert mem_row["status"] == "info"
        assert "macOS" in mem_row["value"]

    def test_darwin_full_cuda_and_model_checks_are_error_free(self):
        """The exact combination that exited 1 before the fix."""
        from distllm.cli import doctor

        with (
            self._torch_patch(has_mps=True),
            patch("distllm.cli.doctor.sys.platform", "darwin"),
        ):
            combined = doctor._check_cuda() + doctor._check_model_compatibility("llama-8b")

        assert [r for r in combined if r["status"] == "error"] == []

    def test_non_darwin_missing_cuda_still_errors(self):
        """The macOS carve-out must not weaken the check elsewhere."""
        from distllm.cli import doctor

        with (
            self._torch_patch(has_mps=True),
            patch("distllm.cli.doctor.sys.platform", "linux"),
            patch("distllm.cli.doctor.subprocess.run", side_effect=FileNotFoundError),
        ):
            results = doctor._check_cuda()

        cuda_row = next(r for r in results if r["check"] == "CUDA available")
        assert cuda_row["status"] == "error"
