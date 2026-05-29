"""Tests: CLI benchmark command — run, JSON output, compare."""

import json
from unittest.mock import MagicMock, patch

import httpx
import pytest

from rich.console import Console


class TestRunBenchmarks:
    def test_run_benchmarks_success(self):
        from distllm.cli.benchmark import _run_benchmarks

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"text": "Hello world test output"}],
            "usage": {"completion_tokens": 5},
        }

        console = Console(quiet=True)

        with patch("distllm.cli.benchmark.httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.post.return_value = mock_resp
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client_cls.return_value = mock_client

            results = _run_benchmarks("model", "localhost", 8000, 3, 64, console)

            assert len(results) == 3
            for r in results:
                assert "prompt" in r
                assert "elapsed" in r
                assert "tokens" in r
                assert "tokens_per_sec" in r
                assert r["tokens"] == 5

    def test_run_benchmarks_connection_error(self):
        from distllm.cli.benchmark import _run_benchmarks

        console = Console(quiet=True)

        with patch("distllm.cli.benchmark.httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.post.side_effect = httpx.ConnectError("refused")
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client_cls.return_value = mock_client

            results = _run_benchmarks("model", "localhost", 8000, 3, 64, console)
            assert results == []

    def test_run_benchmarks_http_error(self):
        from distllm.cli.benchmark import _run_benchmarks

        mock_resp = MagicMock()
        mock_resp.status_code = 500

        console = Console(quiet=True)

        with patch("distllm.cli.benchmark.httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.post.return_value = mock_resp
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client_cls.return_value = mock_client

            results = _run_benchmarks("model", "localhost", 8000, 2, 64, console)
            assert results == []


class TestPrintResults:
    def test_print_results_empty(self):
        from distllm.cli.benchmark import _print_results

        console = Console(quiet=True)
        _print_results([], console)

    def test_print_results_valid(self):
        from distllm.cli.benchmark import _print_results

        console = Console(quiet=True)
        results = [
            {"prompt": "test", "elapsed": 1.0, "tokens": 10, "tokens_per_sec": 10.0},
            {"prompt": "test2", "elapsed": 2.0, "tokens": 20, "tokens_per_sec": 10.0},
        ]
        avg = _print_results(results, console)
        assert avg is not None
        assert len(avg) == 3


class TestBenchmarkJson:
    def test_run_benchmark_json_success(self):
        from distllm.cli.benchmark import run_benchmark_json

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"text": "output text here"}],
            "usage": {"completion_tokens": 3},
        }

        with patch("distllm.cli.benchmark.httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.post.return_value = mock_resp
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result_json = run_benchmark_json("model", "localhost", 8000, 2, 64, False)
            data = json.loads(result_json)
            assert "avg_latency_seconds" in data
            assert "avg_throughput_tps" in data
            assert data["num_runs"] == 2

    def test_run_benchmark_json_no_results(self):
        from distllm.cli.benchmark import run_benchmark_json

        with patch("distllm.cli.benchmark.httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.post.side_effect = httpx.ConnectError("refused")
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result_json = run_benchmark_json("model", "localhost", 8000, 2, 64, False)
            data = json.loads(result_json)
            assert "error" in data


class TestBenchmarkCompare:
    def test_compare_with_baseline(self, tmp_path):
        from distllm.cli.benchmark import run_benchmark_compare

        baseline = {
            "model": "model",
            "avg_latency_seconds": 1.0,
            "avg_throughput_tps": 10.0,
            "avg_tokens": 10.0,
        }
        baseline_path = tmp_path / "baseline.json"
        baseline_path.write_text(json.dumps(baseline))

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"text": "output"}],
            "usage": {"completion_tokens": 5},
        }

        console = Console(quiet=True)

        with patch("distllm.cli.benchmark.httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.post.return_value = mock_resp
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client_cls.return_value = mock_client

            run_benchmark_compare(
                "model", "localhost", 8000, 2, 64,
                str(baseline_path), False, console,
            )
