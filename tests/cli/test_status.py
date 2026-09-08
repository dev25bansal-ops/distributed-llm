"""Tests: ``distllm status`` one-line cluster health summary."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import httpx
import pytest
from typer.testing import CliRunner

from distllm.cli.main import app
from distllm.cli.status import (
    fetch_one_line_summary,
    parse_status_payload,
    render_status_line,
)


runner = CliRunner()


class TestFetchOneLineSummary:
    def test_returns_json_payload(self):
        with patch("distllm.cli.status.httpx.Client") as mock_client:
            mock_instance = MagicMock()
            mock_client.return_value.__enter__.return_value = mock_instance

            def make_resp(status_code=200, payload=None, text=""):
                r = MagicMock()
                r.status_code = status_code
                r.json.return_value = payload or {}
                r.text = text
                r.raise_for_status.return_value = None
                return r

            health = make_resp(payload={"status": "healthy"})
            mock_instance.get.side_effect = lambda url, **kw: (
                health if "/health" in url else make_resp(payload={"data": []})
            )

            payload = fetch_one_line_summary("localhost", 8000)
            info = parse_status_payload(payload)
            assert info["status"] == "ok"
            assert "OK" in info["line"]
            assert "coordinator=localhost:8000" in info["line"]

    def test_connection_refused_marks_unreachable(self):
        with patch("distllm.cli.status.httpx.Client") as mock_client:
            mock_instance = MagicMock()
            mock_client.return_value.__enter__.return_value = mock_instance
            mock_instance.get.side_effect = httpx.ConnectError("nope")
            payload = fetch_one_line_summary("localhost", 9999)
            info = parse_status_payload(payload)
            assert info["status"] == "unreachable"
            assert "DOWN" in info["line"]

    def test_degraded_when_health_not_ok(self):
        with patch("distllm.cli.status.httpx.Client") as mock_client:
            mock_instance = MagicMock()
            mock_client.return_value.__enter__.return_value = mock_instance

            def make_resp(payload=None, status_code=200):
                r = MagicMock()
                r.status_code = status_code
                r.json.return_value = payload or {}
                r.raise_for_status.return_value = None
                return r

            health = make_resp(payload={"status": "degraded"})
            mock_instance.get.side_effect = lambda url, **kw: (
                health if "/health" in url else make_resp(payload={})
            )

            payload = fetch_one_line_summary("localhost", 8000)
            info = parse_status_payload(payload)
            assert info["status"] == "degraded"

    def test_degraded_when_some_nodes_unhealthy(self):
        with patch("distllm.cli.status.httpx.Client") as mock_client:
            mock_instance = MagicMock()
            mock_client.return_value.__enter__.return_value = mock_instance

            def make_resp(payload=None, status_code=200):
                r = MagicMock()
                r.status_code = status_code
                r.json.return_value = payload or {}
                r.raise_for_status.return_value = None
                return r

            health = make_resp(payload={"status": "healthy"})
            nodes = make_resp(
                payload={"nodes": [{"healthy": True}, {"healthy": False}]}
            )
            models = make_resp(payload={"data": [{"id": "m1"}]})

            def side(url, **kw):
                if "/health" in url:
                    return health
                if "nodes" in url:
                    return nodes
                return models

            mock_instance.get.side_effect = side

            payload = fetch_one_line_summary("localhost", 8000)
            info = parse_status_payload(payload)
            assert info["status"] == "degraded"
            assert info["nodes"] == 2
            assert info["models"] == 1

    def test_load_computed_from_memory(self):
        with patch("distllm.cli.status.httpx.Client") as mock_client:
            mock_instance = MagicMock()
            mock_client.return_value.__enter__.return_value = mock_instance

            def make_resp(payload=None, status_code=200):
                r = MagicMock()
                r.status_code = status_code
                r.json.return_value = payload or {}
                r.raise_for_status.return_value = None
                return r

            health = make_resp(payload={"status": "healthy"})
            nodes = make_resp(
                payload={
                    "nodes": [
                        {"memory_used": 40, "memory_total": 100},
                        {"memory_used": 10, "memory_total": 100},
                    ]
                }
            )
            models = make_resp(payload={"data": []})

            def side(url, **kw):
                if "/health" in url:
                    return health
                if "nodes" in url:
                    return nodes
                return models

            mock_instance.get.side_effect = side

            payload = fetch_one_line_summary("localhost", 8000)
            info = parse_status_payload(payload)
            assert info["load"] == 0.25

    def test_handles_bare_list_nodes_payload(self):
        with patch("distllm.cli.status.httpx.Client") as mock_client:
            mock_instance = MagicMock()
            mock_client.return_value.__enter__.return_value = mock_instance

            def make_resp(payload=None, status_code=200):
                r = MagicMock()
                r.status_code = status_code
                r.json.return_value = payload or {}
                r.raise_for_status.return_value = None
                return r

            health = make_resp(payload={"status": "healthy"})
            nodes = make_resp(payload=[{"healthy": True}])
            models = make_resp(payload={"data": []})

            def side(url, **kw):
                if "/health" in url:
                    return health
                if "nodes" in url:
                    return nodes
                return models

            mock_instance.get.side_effect = side

            payload = fetch_one_line_summary("localhost", 8000)
            info = parse_status_payload(payload)
            assert info["nodes"] == 1


class TestParseStatusPayload:
    def test_handles_invalid_json(self):
        info = parse_status_payload("not-json")
        assert info["status"] == "unreachable"
        assert info["line"] == "not-json"

    def test_handles_none(self):
        info = parse_status_payload("")  # JSON-decode fails
        assert info["status"] == "unreachable"


class TestRenderStatusLine:
    def test_render_does_not_raise(self):
        from rich.console import Console
        console = Console()
        # Just verify no exceptions
        render_status_line("DOWN  test", "unreachable", console)
        render_status_line("OK    test", "ok", console)
        render_status_line("DEGR  test", "degraded", console)


class TestRegisterCommand:
    def test_register_attaches_status_to_app(self):
        from distllm.cli.status import register

        import typer

        fresh_app = typer.Typer()
        register(fresh_app)
        names = [c.callback.__name__ for c in fresh_app.registered_commands]
        assert "status_command" in names


class TestCliInvocation:
    def test_status_help(self):
        result = runner.invoke(app, ["status", "--help"])
        assert result.exit_code == 0
        assert "--host" in result.stdout

    def test_status_runs_and_exits_unreachable(self):
        with patch("distllm.cli.status.httpx.Client") as mock_client:
            mock_instance = MagicMock()
            mock_client.return_value.__enter__.return_value = mock_instance
            mock_instance.get.side_effect = httpx.ConnectError("nope")
            result = runner.invoke(app, ["status", "--host", "localhost", "--port", "9999"])
            assert result.exit_code == 1

    def test_status_runs_json_output(self):
        with patch("distllm.cli.status.httpx.Client") as mock_client:
            mock_instance = MagicMock()
            mock_client.return_value.__enter__.return_value = mock_instance

            def make_resp(payload=None, status_code=200):
                r = MagicMock()
                r.status_code = status_code
                r.json.return_value = payload or {}
                r.raise_for_status.return_value = None
                return r

            health = make_resp(payload={"status": "healthy"})
            mock_instance.get.side_effect = lambda url, **kw: (
                health if "/health" in url else make_resp(payload={})
            )

            result = runner.invoke(app, ["status", "--json"], catch_exceptions=False)
            assert result.exit_code == 0