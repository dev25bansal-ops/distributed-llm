"""Tests: CLI doctor command — diagnostics, checks."""

import os
import socket
from unittest.mock import MagicMock, patch

import pytest


class TestCheckCUDA:
    def test_cuda_check_returns_list(self):
        from distllm.cli.doctor import _check_cuda

        results = _check_cuda()
        assert isinstance(results, list)
        assert len(results) > 0
        # Should always have torch version check
        assert any("pytorch" in r["check"].lower() or "torch" in r["check"].lower() for r in results)

    def test_cuda_check_has_status_fields(self):
        from distllm.cli.doctor import _check_cuda

        results = _check_cuda()
        for r in results:
            assert "check" in r
            assert "status" in r
            assert r["status"] in ("ok", "warn", "error", "info")


class TestCheckPorts:
    def test_check_ports_returns_list(self):
        from distllm.cli.doctor import _check_ports

        results = _check_ports([50050])
        assert isinstance(results, list)
        assert len(results) == 1
        assert results[0]["check"] == "port 50050"
        assert results[0]["status"] in ("ok", "info")

    def test_check_ports_default_list(self):
        from distllm.cli.doctor import _check_ports

        results = _check_ports()
        assert len(results) >= 5  # 50050, 50051, 50052, 50060, 8000, 8500

    def test_check_ports_custom(self):
        from distllm.cli.doctor import _check_ports

        results = _check_ports([12345])
        assert len(results) == 1
        assert "12345" in results[0]["check"]


class TestCheckConfig:
    def test_check_config_no_files(self):
        from distllm.cli.doctor import _check_config

        with patch("os.path.exists", return_value=False):
            results = _check_config()
            assert isinstance(results, list)

    def test_check_config_valid_yaml(self, tmp_path):
        from distllm.cli.doctor import _check_config

        config = tmp_path / "config.yaml"
        config.write_text("model:\n  name: test\n")

        # Patch exists to return True for config paths, False otherwise
        original_exists = os.path.exists
        def mock_exists(path):
            if "config.yaml" in str(path):
                return True
            return original_exists(path)

        with patch("os.path.exists", side_effect=mock_exists):
            with patch("builtins.open", side_effect=lambda p, *a, **kw: open(config)):
                results = _check_config()
                assert isinstance(results, list)


class TestCheckDisk:
    def test_check_disk_returns_list(self):
        from distllm.cli.doctor import _check_disk

        results = _check_disk()
        assert isinstance(results, list)

    def test_check_disk_with_existing_dir(self):
        from distllm.cli.doctor import _check_disk

        with patch("os.path.exists", return_value=True):
            with patch("shutil.disk_usage", return_value=MagicMock(free=100e9)):
                results = _check_disk()
                assert any(r["status"] == "ok" for r in results)

    def test_check_disk_low_space(self):
        from distllm.cli.doctor import _check_disk

        with patch("os.path.exists", return_value=True):
            with patch("shutil.disk_usage", return_value=MagicMock(free=20e9)):
                results = _check_disk()
                assert any(r["status"] == "warn" for r in results)


class TestRenderResults:
    """The old `_print_results(category, results)` helper was replaced by the
    Doctor renderer, which takes an argparse Namespace and writes through
    rich. These assert the rendering contract, not the removed function."""

    def _doctor(self):
        import argparse

        from distllm.cli.doctor import Doctor

        args = argparse.Namespace(json=False, terse=False, category=None, verbose=False)
        return Doctor(args)

    def test_category_results_are_rendered(self, capsys):
        results = [
            {"check": "test", "status": "ok", "value": "42"},
            {"check": "warn_test", "status": "warn", "value": "low"},
            {"check": "err_test", "status": "error", "value": "bad"},
        ]
        doctor = self._doctor()
        doctor.console = __import__("rich.console", fromlist=["Console"]).Console()
        doctor._render_category("Test Category", results)
        captured = capsys.readouterr()
        assert "Test Category" in captured.out
        assert "test" in captured.out

    def test_status_labels_cover_every_status_the_checks_emit(self):
        """A status with no label would render as a KeyError at runtime."""
        from distllm.cli.doctor import Doctor

        emitted = {"ok", "pass", "warn", "error", "fail", "info", "skip"}
        assert emitted <= set(Doctor.STATUS_LABELS)
        assert emitted <= set(Doctor.STATUS_STYLES)
