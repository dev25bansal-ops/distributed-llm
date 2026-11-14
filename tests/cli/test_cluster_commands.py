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
import pytest
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


# ── One key precedence across the whole cluster CLI ────────────────────────


class TestClusterKeyPrecedence:
    """Audit finding: one CLI, two env orders.

    ``cluster join`` registration read ``API_KEY`` first while
    ``_admin_headers`` / ``resolve_api_key`` read ``DISTLLM_API_KEY`` first,
    so with both vars set to different values (the v0.3->v0.4 migration the
    docs describe) join and list-nodes sent different keys to the same
    server. All three paths must resolve through one precedence:
    ``API_KEY`` > ``DISTLLM_API_KEY`` > key file.
    """

    @pytest.fixture(autouse=True)
    def _isolated_key_env(self, tmp_path, monkeypatch):
        for var in ("API_KEY", "DISTLLM_API_KEY", "DISTLLM_KEY_FILE"):
            monkeypatch.delenv(var, raising=False)
        self.key_file = tmp_path / "api_key"
        monkeypatch.setenv("DISTLLM_KEY_FILE", str(self.key_file))

    def test_admin_headers_match_shared_resolution_with_both_env_vars(self, monkeypatch):
        from distllm.cli.cluster import _admin_headers
        from distllm.cli.http_client import resolve_api_key

        monkeypatch.setenv("API_KEY", "sk-a")
        monkeypatch.setenv("DISTLLM_API_KEY", "sk-b")

        # And no divergence: the same request key everywhere.
        assert _admin_headers()["Authorization"] == f"Bearer {resolve_api_key()}"
        assert resolve_api_key() == "sk-a"

    def test_admin_headers_fall_back_to_the_key_file(self):
        from distllm.cli.cluster import _admin_headers

        self.key_file.write_text("sk-from-file\n", encoding="utf-8")

        assert _admin_headers() == {"Authorization": "Bearer sk-from-file"}

    def test_admin_headers_distllm_api_key_alone(self, monkeypatch):
        from distllm.cli.cluster import _admin_headers

        monkeypatch.setenv("DISTLLM_API_KEY", "sk-b")

        assert _admin_headers() == {"Authorization": "Bearer sk-b"}

    def test_join_registration_sends_the_same_key_as_admin_commands(self, monkeypatch):
        """The registration POST must carry exactly the key `_admin_headers` uses."""
        import socket as socket_mod
        from unittest.mock import MagicMock

        from distllm.cli import cluster as cluster_mod

        monkeypatch.setenv("API_KEY", "sk-a")
        monkeypatch.setenv("DISTLLM_API_KEY", "sk-b")

        captured: dict = {}

        class _FakeSocket:
            def __init__(self, *args, **kwargs):
                pass

            def settimeout(self, _):
                pass

            def connect_ex(self, addr):
                return 0  # worker port open — skip the 60s poll

            def close(self):
                pass

        class _FakeProc:
            pid = 4242
            stdout = None

            def __init__(self):
                import io

                self.stdout = io.StringIO("")

        monkeypatch.setattr(socket_mod, "socket", _FakeSocket)
        monkeypatch.setattr(cluster_mod.subprocess, "Popen", lambda *a, **k: _FakeProc())

        def _fake_post(url, json=None, headers=None, timeout=None):
            captured["url"] = str(url)
            captured["headers"] = headers or {}
            resp = MagicMock()
            resp.status_code = 200
            resp.text = ""
            return resp

        monkeypatch.setattr(cluster_mod.httpx, "post", _fake_post)

        cluster_mod._cluster_join(
            coordinator_host="10.9.9.9",
            coordinator_port=50050,
            node_id="w1",
            start_layer=0,
            end_layer=1,
            total_layers=2,
            listen_port=50051,
            device="cpu",
            model="test-model",
        )

        assert captured["url"].endswith("/admin/v1/nodes/register")
        assert captured["headers"].get("Authorization") == "Bearer sk-a", (
            "join registration must resolve the same key _admin_headers sends"
        )

    def test_join_registration_reads_the_key_file(self, monkeypatch):
        """Default file-only deployment: the registration POST authenticates."""
        import socket as socket_mod
        from unittest.mock import MagicMock

        from distllm.cli import cluster as cluster_mod

        self.key_file.write_text("sk-from-file\n", encoding="utf-8")

        captured: dict = {}

        class _FakeSocket:
            def __init__(self, *args, **kwargs):
                pass

            def settimeout(self, _):
                pass

            def connect_ex(self, addr):
                return 0

            def close(self):
                pass

        class _FakeProc:
            pid = 4242

            def __init__(self):
                import io

                self.stdout = io.StringIO("")

        monkeypatch.setattr(socket_mod, "socket", _FakeSocket)
        monkeypatch.setattr(cluster_mod.subprocess, "Popen", lambda *a, **k: _FakeProc())

        def _fake_post(url, json=None, headers=None, timeout=None):
            captured["headers"] = headers or {}
            resp = MagicMock()
            resp.status_code = 200
            resp.text = ""
            return resp

        monkeypatch.setattr(cluster_mod.httpx, "post", _fake_post)

        cluster_mod._cluster_join(
            coordinator_host="10.9.9.9",
            coordinator_port=50050,
            node_id="w2",
            start_layer=0,
            end_layer=1,
            total_layers=2,
            listen_port=50051,
            device="cpu",
            model="test-model",
        )

        assert captured["headers"].get("Authorization") == "Bearer sk-from-file"


# ── cluster leave / drain port defaults ────────────────────────────────────
#
# `cluster leave` and `cluster drain` call the REST admin routes
# (/admin/v1/nodes/...), which live on the API port (8000 by default), but
# leave defaulted to 50050 — the gRPC port. Against a default `cluster
# start` the command could never work, and against an idle port it ended in
# a raw WinError instead of the friendly admin-path message. The default
# must match the REST port the route actually lives on.


class TestClusterLeavePortDefaults:
    def test_leave_defaults_to_the_rest_port(self):
        """No --port on the CLI means port 8000, where /admin/v1 lives."""
        result = runner.invoke(app, ["cluster", "leave", "--help"])

        assert result.exit_code == 0, result.output
        assert "default: 8000" in result.output, (
            "cluster leave talks to the REST admin routes; its --port default "
            "must be the REST port"
        )
        assert "default: 50050" not in result.output

    def test_leave_zero_config_hits_admin_routes_on_8000(self, monkeypatch):
        """The zero-config invocation must reach /admin/v1/nodes/{id} on :8000."""
        monkeypatch.delenv("DISTLLM_API_KEY", raising=False)
        monkeypatch.delenv("API_KEY", raising=False)
        seen: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            if request.method == "DELETE":
                return httpx.Response(
                    200,
                    json={"status": "ok", "node_id": "w1", "message": "gone"},
                )
            return httpx.Response(
                200,
                json={"status": "ok", "node_id": "w1", "message": "draining"},
            )

        with patch("distllm.cli.cluster.httpx.Client", _patched_client(handler)):
            result = runner.invoke(app, ["cluster", "leave", "w1"])

        assert result.exit_code == 0, result.output
        assert seen, "leave must actually call the coordinator"
        assert all(r.url.port == 8000 for r in seen), (
            f"expected the REST port; got {[r.url for r in seen]}"
        )
        assert seen[0].url.path == "/admin/v1/nodes/w1/drain"
        assert seen[-1].url.path == "/admin/v1/nodes/w1"
        assert seen[-1].method == "DELETE"

    def test_leave_drains_then_deregisters(self, monkeypatch):
        """Leave must fully remove the node, not just flag it draining."""
        monkeypatch.delenv("DISTLLM_API_KEY", raising=False)
        monkeypatch.delenv("API_KEY", raising=False)
        seen: list[tuple[str, str]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append((request.method, request.url.path))
            return httpx.Response(
                200, json={"status": "ok", "node_id": "w1", "message": "ok"},
            )

        with patch("distllm.cli.cluster.httpx.Client", _patched_client(handler)):
            result = runner.invoke(app, ["cluster", "leave", "w1"])

        assert result.exit_code == 0, result.output
        assert seen == [
            ("POST", "/admin/v1/nodes/w1/drain"),
            ("DELETE", "/admin/v1/nodes/w1"),
        ]
        assert "left the cluster" in result.output

    def test_leave_reports_unregistered_node(self, monkeypatch):
        """404 on drain is the friendly 'not registered' path, not a traceback."""
        monkeypatch.delenv("DISTLLM_API_KEY", raising=False)
        monkeypatch.delenv("API_KEY", raising=False)

        with patch(
            "distllm.cli.cluster.httpx.Client",
            _patched_client(
                lambda req: httpx.Response(404, json={"detail": "Node not found"})
            ),
        ):
            result = runner.invoke(app, ["cluster", "leave", "ghost"])

        assert result.exit_code == 0, result.output
        assert "not registered" in result.output

    def test_drain_also_talks_to_rest_port(self, monkeypatch):
        monkeypatch.delenv("DISTLLM_API_KEY", raising=False)
        monkeypatch.delenv("API_KEY", raising=False)
        seen: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return httpx.Response(
                200, json={"status": "ok", "node_id": "w1", "message": "draining"},
            )

        with patch("distllm.cli.cluster.httpx.Client", _patched_client(handler)):
            result = runner.invoke(app, ["cluster", "drain", "w1"])

        assert result.exit_code == 0, result.output
        assert seen[0].url.port == 8000
        assert seen[0].url.path == "/admin/v1/nodes/w1/drain"
