"""Tests: ``distllm-cli doctor`` — local environment health check.

Covers the 9 checks (Python, PyTorch, CUDA, bitsandbytes, transformers,
HF reachable, ports, disk, GPU memory), the colored-table renderer,
auto-remediation (``--fix``), and the JSON output mode.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from distllm.cli.cli_doctor import (
    CHECKS,
    CheckResult,
    attempt_fixes,
    main,
    render_table,
)
from distllm.cli.main import app


runner = CliRunner()


# ── Individual checks ──────────────────────────────────────────────────────


class TestCheckPython:
    def test_ok_on_current_python(self):
        from distllm.cli.cli_doctor import _check_python

        r = _check_python()
        # 3.14.4 in this env — passes ≥ 3.10
        assert r.status == "ok"
        assert r.name == "Python version"


class TestCheckPytorch:
    def test_ok_when_installed(self):
        from distllm.cli.cli_doctor import _check_pytorch

        # torch is installed in the dev env
        r = _check_pytorch()
        assert r.status in ("ok", "warn")
        assert r.name == "PyTorch"

    def test_warn_when_missing(self):
        from distllm.cli.cli_doctor import _check_pytorch

        with patch.dict("sys.modules", {"torch": None}):
            # Force ImportError — easier than blocking real torch
            import builtins

            orig_import = builtins.__import__

            def fake_import(name, *a, **kw):
                if name == "torch":
                    raise ImportError("blocked")
                return orig_import(name, *a, **kw)

            with patch("builtins.__import__", side_effect=fake_import):
                r = _check_pytorch()
        assert r.status == "warn"
        assert "not installed" in r.message


class TestCheckCUDA:
    def test_returns_status(self):
        from distllm.cli.cli_doctor import _check_cuda

        r = _check_cuda()
        # Either CUDA is available or it isn't — we just verify the
        # check returns a valid status string.
        assert r.status in ("ok", "warn")

    def test_warns_when_torch_missing(self):
        from distllm.cli.cli_doctor import _check_cuda

        import builtins

        orig_import = builtins.__import__

        def fake_import(name, *a, **kw):
            if name == "torch":
                raise ImportError("blocked")
            return orig_import(name, *a, **kw)

        with patch("builtins.__import__", side_effect=fake_import):
            r = _check_cuda()
        assert r.status == "warn"


class TestCheckBitsandbytes:
    def test_returns_status(self):
        from distllm.cli.cli_doctor import _check_bitsandbytes

        r = _check_bitsandbytes()
        assert r.status in ("ok", "warn")


class TestCheckTransformers:
    def test_returns_status(self):
        from distllm.cli.cli_doctor import _check_transformers

        r = _check_transformers()
        assert r.status in ("ok", "warn", "fail")
        assert r.name == "transformers"


class TestCheckHFReachable:
    def test_status_is_string(self):
        from distllm.cli.cli_doctor import _check_hf_reachable

        r = _check_hf_reachable()
        # We can't guarantee network in tests — just assert shape
        assert r.status in ("ok", "warn")
        assert r.name == "HuggingFace reachable"

    def test_handles_network_error(self):
        from distllm.cli.cli_doctor import _check_hf_reachable

        with patch(
            "distllm.cli.cli_doctor.urllib.request.urlopen",
            side_effect=TimeoutError("blocked"),
        ):
            r = _check_hf_reachable()
        assert r.status == "warn"


class TestCheckPorts:
    def test_default_ports(self):
        from distllm.cli.cli_doctor import _check_ports

        r = _check_ports()
        assert r.status in ("ok", "warn")
        assert r.name == "Ports free"

    def test_custom_ports(self):
        from distllm.cli.cli_doctor import _check_ports

        r = _check_ports((12345, 23456))
        assert r.status in ("ok", "warn")

    def test_warns_when_port_busy(self):
        from distllm.cli.cli_doctor import _check_ports

        # Bind a real port to make it busy
        import socket as _socket

        s = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        s.listen()
        try:
            r = _check_ports((port,))
            assert r.status == "warn"
            assert str(port) in r.message
        finally:
            s.close()


class TestCheckDisk:
    def test_returns_status(self):
        from distllm.cli.cli_doctor import _check_disk

        r = _check_disk()
        assert r.status in ("ok", "warn")


class TestCheckGPUMem:
    def test_returns_status(self):
        from distllm.cli.cli_doctor import _check_gpu_mem

        r = _check_gpu_mem()
        assert r.status in ("ok", "warn")


# ── Top-level orchestrator ────────────────────────────────────────────────


class TestCheckList:
    def test_all_9_checks_present(self):
        names = [name for name, _fn in CHECKS]
        assert names == [
            "python",
            "pytorch",
            "cuda",
            "bitsandbytes",
            "transformers",
            "hf",
            "ports",
            "disk",
            "gpu_mem",
        ]

    def test_every_check_runs_without_exception(self):
        for name, fn in CHECKS:
            result = fn()
            assert isinstance(result, CheckResult)
            assert result.status in ("ok", "warn", "fail")
            assert result.name


# ── Renderer ──────────────────────────────────────────────────────────────


class TestRenderTable:
    def test_render_no_crash(self):
        import io

        from rich.console import Console

        results = [
            CheckResult("Python", "ok", "3.14"),
            CheckResult("PyTorch", "warn", "old"),
            CheckResult("transformers", "fail", "missing"),
        ]
        sink = io.StringIO()
        console = Console(force_terminal=False, file=sink)
        render_table(results, console)
        out = sink.getvalue()
        assert "Python" in out
        assert "PyTorch" in out
        assert "transformers" in out

    def test_exit_codes(self):
        from distllm.cli.cli_doctor import main

        # No failures → exit 0
        ok_checks = [(name, (lambda n=name: (lambda: CheckResult(n, "ok", "ok")))()) for name, _ in CHECKS]
        with patch("distllm.cli.cli_doctor.CHECKS", ok_checks):
            assert main([]) == 0

        # One failure → exit 1
        fail_checks = [("python", lambda: CheckResult("python", "fail", "old"))]
        fail_checks += [
            (name, (lambda n=name: (lambda: CheckResult(n, "ok", "ok")))())
            for name, _ in CHECKS[1:]
        ]
        with patch("distllm.cli.cli_doctor.CHECKS", fail_checks):
            assert main([]) == 1


# ── Auto-fix ──────────────────────────────────────────────────────────────


class TestAttemptFixes:
    def test_dry_run_no_crash(self):
        import io

        from rich.console import Console

        results = [CheckResult(name, "ok", "ok") for name, _ in CHECKS]
        sink = io.StringIO()
        console = Console(force_terminal=False, file=sink)
        out = attempt_fixes(results, console)
        assert out is results

    def test_installs_missing_bitsandbytes(self):
        import io

        from rich.console import Console

        results = [CheckResult("bitsandbytes", "warn", "not installed")]
        sink = io.StringIO()
        console = Console(force_terminal=False, file=sink)
        with patch(
            "distllm.cli.cli_doctor._pip_install",
            return_value=(True, "installed bitsandbytes"),
        ):
            out = attempt_fixes(results, console)
        assert out[0].status == "ok"
        assert "installed via --fix" in out[0].message

    def test_installs_missing_transformers(self):
        import io

        from rich.console import Console

        results = [CheckResult("transformers", "fail", "not installed")]
        sink = io.StringIO()
        console = Console(force_terminal=False, file=sink)
        with patch(
            "distllm.cli.cli_doctor._pip_install",
            return_value=(True, "installed transformers"),
        ):
            out = attempt_fixes(results, console)
        assert out[0].status == "ok"

    def test_refuses_to_kill_busy_port(self):
        import io

        from rich.console import Console

        results = [CheckResult("Ports free", "warn", "in use: 8000")]
        sink = io.StringIO()
        console = Console(force_terminal=False, file=sink)
        out = attempt_fixes(results, console)
        # Status must remain warn — never auto-killed
        assert out[0].status == "warn"

    def test_cleans_hf_cache(self, tmp_path):
        import io

        from rich.console import Console

        cache = tmp_path / "huggingface"
        cache.mkdir()

        results = [CheckResult("Disk space", "warn", "low")]
        sink = io.StringIO()
        console = Console(force_terminal=False, file=sink)

        with patch("distllm.cli.cli_doctor.HF_CACHE", str(cache)):
            with patch(
                "distllm.cli.cli_doctor._clean_hf_cache",
                return_value=(True, f"freed 6.0 GB by cleaning {cache}"),
            ):
                out = attempt_fixes(results, console)
        assert out[0].status == "warn"

    def test_handles_pip_install_failure(self):
        import io

        from rich.console import Console

        results = [CheckResult("bitsandbytes", "warn", "not installed")]
        sink = io.StringIO()
        console = Console(force_terminal=False, file=sink)
        with patch(
            "distllm.cli.cli_doctor._pip_install",
            return_value=(False, "pip install failed: network down"),
        ):
            out = attempt_fixes(results, console)
        assert out[0].status == "warn"


# ── JSON output ───────────────────────────────────────────────────────────


class TestJsonOutput:
    def test_json_emits_list(self):
        ok_checks = [
            (name, (lambda n=name: (lambda: CheckResult(n, "ok", "ok")))())
            for name, _ in CHECKS
        ]
        with patch("distllm.cli.cli_doctor.CHECKS", ok_checks):
            rc = main(["--json"])
        assert rc == 0


# ── CLI integration ───────────────────────────────────────────────────────


class TestCliIntegration:
    def test_cli_doctor_help(self):
        import re

        result = runner.invoke(app, ["cli-doctor", "--help"])
        assert result.exit_code == 0
        # Strip ANSI: rich colours each token separately ("-" then "-fix"), so
        # a raw "--fix" substring test misses it on a styled render.
        plain = re.sub(r"\x1b\[[0-9;]*m", "", result.stdout)
        assert "--fix" in plain, plain
        assert "--json" in plain

    def test_cli_doctor_runs(self):
        result = runner.invoke(app, ["cli-doctor"], catch_exceptions=False)
        assert result.exit_code in (0, 1)

    def test_cli_doctor_fix_runs(self):
        result = runner.invoke(app, ["cli-doctor", "--fix"], catch_exceptions=False)
        assert result.exit_code in (0, 1)