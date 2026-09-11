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
        # Should always have the PyTorch version check
        assert any(r["check"] == "PyTorch version" for r in results)

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
        assert len(results) == 6  # 50050, 50051, 50052, 50060, 8000, 8500
        assert [r["check"] for r in results] == [
            "port 50050",
            "port 50051",
            "port 50052",
            "port 50060",
            "port 8000",
            "port 8500",  # dashboard port (DASHBOARD_PORT)
        ]

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
                # 3-band ladder: ok requires > 50 GB free
                assert any(r["status"] == "ok" for r in results)

    def test_check_disk_low_space(self):
        from distllm.cli.doctor import _check_disk

        with patch("os.path.exists", return_value=True):
            # 10 GB < free <= 50 GB → warn
            with patch("shutil.disk_usage", return_value=MagicMock(free=20e9)):
                results = _check_disk()
                assert any(r["status"] == "warn" for r in results)

            # <= 10 GB free → error (critical, cannot fit a model)
            with patch("shutil.disk_usage", return_value=MagicMock(free=5e9)):
                results = _check_disk()
                assert any(r["status"] == "error" for r in results)


class TestPrintResults:
    def test_print_results(self, capsys):
        import argparse

        from distllm.cli.doctor import Doctor

        args = argparse.Namespace(
            gpu=False, network=False, model=None, nodes=None,
            json=False, terse=False, verbose=False,
        )
        doctor = Doctor(args)

        results = [
            {"check": "test", "status": "ok", "value": "42"},
            {"check": "warn_test", "status": "warn", "value": "low"},
            {"check": "err_test", "status": "error", "value": "bad"},
        ]
        doctor._render_category("Test Category", results)
        captured = capsys.readouterr()
        assert "Test Category" in captured.out
        assert "test" in captured.out
