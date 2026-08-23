"""Tests for `distllm cluster` subcommands — list-nodes and status.

Uses ``httpx.MockTransport`` (patched in as the transport of the module's
``httpx.Client``) so requests flow through real httpx machinery while the
network is fully mocked.  Invocations go through the Typer CLI entrypoint
(``distllm cluster list-nodes`` / ``distllm cluster status``) to verify
wiring end-to-end, and assert on rendered Rich table output.
"""

from __future__ import annotations

from typing import Callable
from unittest.mock import patch

import httpx
from typer.testing import CliRunner

from distllm.cli.main import app

runner = CliRunner()


# ── Helpers ────────────────────────────────────────────────────────────────


# Capture before any patching — patch() swaps httpx.Client globally, so the
# factory must construct clients via the original class.
_REAL_CLIENT = httpx.Client


def _patched_client(
    handler: Callable[[httpx.Request], httpx.Response],
):
    """Factory patching ``distllm.cli.cluster.httpx.Client`` with a client
    backed by an ``httpx.MockTransport`` running *handler*."""

    def factory(**kwargs):
        return _REAL_CLIENT(transport=httpx.MockTransport(handler), **kwargs)

    return factory


def _nodes_payload() -> dict:
    """Shape mirrors api.routes.admin.NodeListResponse."""
    return {
        "nodes": [
            {
                "node_id": "node-alpha",
                "host": "10.0.0.1",
                "port": 50051,
                "healthy": True,
                "draining": False,
                "state": "healthy",
                "start_layer": 0,
                "end_layer": 15,
                "gpu_name": "NVIDIA A100",
                "gpu_memory_free": 68 * 1024**3,
                "gpu_memory_total": 80 * 1024**3,
            },
            {
                "node_id": "node-beta",
                "host": "10.0.0.2",
                "port": 50052,
                "healthy": False,
                "draining": True,
                "state": "draining",
                "start_layer": 16,
                "end_layer": 31,
                "gpu_name": "",
                "gpu_memory_free": None,
                "gpu_memory_total": 0,
            },
        ],
        "total_nodes": 2,
        "healthy_count": 1,
        "draining_count": 1,
        "total_layers": 32,
    }


def _health_payload() -> dict:
    """Shape mirrors api.routes.health.health_check."""
    return {
        "status": "healthy",
        "model": "test-model",
        "nodes": 2,
        "node_health": {
            "node-alpha": {"healthy": True},
            "node-beta": {"healthy": False},
        },
    }


def _refused(request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError("connection refused", request=request)


# ── cluster list-nodes ─────────────────────────────────────────────────────


class TestClusterListNodes:
    def test_success_renders_node_table(self, monkeypatch):
        monkeypatch.delenv("DISTLLM_API_KEY", raising=False)
        monkeypatch.delenv("API_KEY", raising=False)
        with patch("distllm.cli.cluster.httpx.Client",
            _patched_client(lambda req: httpx.Response(200, json=_nodes_payload())),
        ):
            result = runner.invoke(app, ["cluster", "list-nodes"])

        assert result.exit_code == 0, result.output
        # Table renders node identity, endpoint, state, and layer ranges.
        assert "Cluster Nodes (2 total)" in result.output
        assert "node-alpha" in result.output
        assert "node-beta" in result.output
        assert "10.0.0.1:50051" in result.output
        assert "0-15" in result.output
        assert "16-31" in result.output
        assert "NVIDIA A100" in result.output
        # VRAM free formatted from bytes; None falls back gracefully.
        assert "68.0GB" in result.output
        # Summary line counts healthy/draining.
        assert "1 healthy" in result.output
        assert "1 draining" in result.output

    def test_hits_admin_v1_nodes_endpoint(self, monkeypatch):
        monkeypatch.delenv("DISTLLM_API_KEY", raising=False)
        monkeypatch.delenv("API_KEY", raising=False)
        seen: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return httpx.Response(200, json=_nodes_payload())

        with patch("distllm.cli.cluster.httpx.Client", _patched_client(handler)
        ):
            result = runner.invoke(app, ["cluster", "list-nodes"])

        assert result.exit_code == 0, result.output
        assert len(seen) == 1
        assert seen[0].url.path == "/admin/v1/nodes"

    def test_sends_bearer_token_from_env(self, monkeypatch):
        monkeypatch.setenv("DISTLLM_API_KEY", "secret-admin-key")
        monkeypatch.delenv("API_KEY", raising=False)
        seen: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return httpx.Response(200, json=_nodes_payload())

        with patch("distllm.cli.cluster.httpx.Client", _patched_client(handler)
        ):
            runner.invoke(app, ["cluster", "list-nodes"])

        assert seen[0].headers.get("authorization") == "Bearer secret-admin-key"

    def test_unauthorized_mentions_api_key(self, monkeypatch):
        monkeypatch.delenv("DISTLLM_API_KEY", raising=False)
        monkeypatch.delenv("API_KEY", raising=False)
        with patch("distllm.cli.cluster.httpx.Client",
            _patched_client(lambda req: httpx.Response(401, json={"detail": "unauthorized"})),
        ):
            result = runner.invoke(app, ["cluster", "list-nodes"])

        assert result.exit_code == 0, result.output
        assert "Unauthorized" in result.output
        assert "DISTLLM_API_KEY" in result.output

    def test_connection_refused_friendly_message(self, monkeypatch):
        monkeypatch.delenv("DISTLLM_API_KEY", raising=False)
        monkeypatch.delenv("API_KEY", raising=False)
        with patch("distllm.cli.cluster.httpx.Client", _patched_client(_refused)
        ):
            result = runner.invoke(
                app, ["cluster", "list-nodes", "--coordinator", "localhost", "--port", "9999"]
            )

        assert result.exit_code == 0, result.output
        assert "Could not connect to coordinator at localhost:9999" in result.output
        assert "distllm cluster start" in result.output

    def test_non_http_response_handled(self, monkeypatch):
        """Hitting the gRPC port must produce guidance, not a traceback."""
        monkeypatch.delenv("DISTLLM_API_KEY", raising=False)
        monkeypatch.delenv("API_KEY", raising=False)

        def garbage_protocol(request: httpx.Request) -> httpx.Response:
            raise httpx.RemoteProtocolError("illegal request line", request=request)

        with patch("distllm.cli.cluster.httpx.Client", _patched_client(garbage_protocol)
        ):
            result = runner.invoke(app, ["cluster", "list-nodes"])

        assert result.exit_code == 0, result.output
        assert "Could not reach coordinator" in result.output
        assert "RemoteProtocolError" in result.output
        assert "REST API port" in result.output

    def test_empty_cluster(self, monkeypatch):
        monkeypatch.delenv("DISTLLM_API_KEY", raising=False)
        monkeypatch.delenv("API_KEY", raising=False)
        payload = {"nodes": [], "total_nodes": 0, "healthy_count": 0,
                   "draining_count": 0, "total_layers": 32}
        with patch("distllm.cli.cluster.httpx.Client",
            _patched_client(lambda req: httpx.Response(200, json=payload)),
        ):
            result = runner.invoke(app, ["cluster", "list-nodes"])

        assert result.exit_code == 0, result.output
        assert "No nodes registered" in result.output


# ── cluster status ─────────────────────────────────────────────────────────


class TestClusterStatus:
    def test_success_renders_status_table(self, monkeypatch):
        monkeypatch.delenv("DISTLLM_API_KEY", raising=False)
        monkeypatch.delenv("API_KEY", raising=False)
        seen: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return httpx.Response(200, json=_health_payload())

        with patch("distllm.cli.cluster.httpx.Client", _patched_client(handler)
        ):
            result = runner.invoke(app, ["cluster", "status"])

        assert result.exit_code == 0, result.output
        assert seen[0].url.path == "/v1/health"
        assert "Cluster Status" in result.output
        assert "healthy" in result.output
        assert "test-model" in result.output
        # Per-node health table rows.
        assert "node-alpha" in result.output
        assert "node-beta" in result.output

    def test_connection_refused_friendly_message(self, monkeypatch):
        monkeypatch.delenv("DISTLLM_API_KEY", raising=False)
        monkeypatch.delenv("API_KEY", raising=False)
        with patch("distllm.cli.cluster.httpx.Client", _patched_client(_refused)
        ):
            result = runner.invoke(
                app, ["cluster", "status", "--host", "localhost", "--port", "9999"]
            )

        assert result.exit_code == 0, result.output
        assert "Could not connect to coordinator at localhost:9999" in result.output
        assert "distllm cluster start" in result.output

    def test_503_no_model_loaded(self, monkeypatch):
        monkeypatch.delenv("DISTLLM_API_KEY", raising=False)
        monkeypatch.delenv("API_KEY", raising=False)

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(503, json={"detail": "No model loaded"})

        with patch("distllm.cli.cluster.httpx.Client", _patched_client(handler)
        ):
            result = runner.invoke(app, ["cluster", "status"])

        assert result.exit_code == 0, result.output
        assert "no model is loaded" in result.output

    def test_http_error_shows_status_and_body(self, monkeypatch):
        monkeypatch.delenv("DISTLLM_API_KEY", raising=False)
        monkeypatch.delenv("API_KEY", raising=False)

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, text="boom")

        with patch("distllm.cli.cluster.httpx.Client", _patched_client(handler)
        ):
            result = runner.invoke(app, ["cluster", "status"])

        assert result.exit_code == 0, result.output
        assert "500" in result.output
