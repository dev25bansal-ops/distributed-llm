"""Behavioural assertions for CLI paths previously covered by assertion-free tests.

Provenance: an AST scan found 45 test functions under ``tests/cli`` and
``tests/sdk`` with no assertion at all — they passed as long as the code did
not raise, so a silently no-op implementation also passed. This module
re-exercises 15 of the behaviour-guarding ones (chat loop, model commands,
status rendering, cluster status/list-nodes) and asserts on the real rendered
output, the HTTP calls made, and the payloads sent.

Every test here is designed to FAIL if the guarded behaviour is removed (e.g.
the function is replaced by ``pass`` or the relevant branch is dropped), and to
PASS on the current source. Functions under test are imported inside the test
body, mirroring the originals, so a monkeypatched module attribute takes effect.
"""

from __future__ import annotations

import io
from unittest.mock import MagicMock, patch

import httpx
from rich.console import Console


# ── Helpers ────────────────────────────────────────────────────────────────


def _capture_console() -> tuple[Console, io.StringIO]:
    """A real Rich console whose rendered text can be inspected."""
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=False, width=120)
    return console, buf


def _chat_client(post_result, sent: list | None = None) -> MagicMock:
    """Mock for ``distllm.cli.chat.client_for`` — usable in ``with`` blocks.

    ``post_result`` is either a response-like object (single ``post`` result)
    or a list (``side_effect`` sequence, e.g. one error then one success).

    ``run_chat`` keeps one live ``conversation`` list and passes it by reference,
    so the mock's recorded call args show the list as mutated *after* the reply
    was appended. When ``sent`` is given, a deep copy of each request body is
    snapshotted at call time so tests can assert what was actually sent.
    """
    client = MagicMock()
    client.__enter__.return_value = client
    client.__exit__.return_value = False

    results = post_result if isinstance(post_result, list) else None
    queue = list(results) if results is not None else None

    def _post(url, **kwargs):
        if sent is not None:
            import copy

            sent.append(copy.deepcopy(kwargs.get("json")))
        if queue is not None:
            nxt = queue.pop(0)
            if isinstance(nxt, Exception):
                raise nxt
            return nxt
        return post_result

    client.post.side_effect = _post
    return client


def _ok_response(
    content: str = "Hi!",
    completion_tokens: int | None = None,
    generation_time: float | None = None,
) -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    payload: dict = {"choices": [{"message": {"content": content}}]}
    if completion_tokens is not None:
        payload["usage"] = {"completion_tokens": completion_tokens}
    if generation_time is not None:
        payload["generation_time"] = generation_time
    resp.json.return_value = payload
    resp.raise_for_status.return_value = None
    return resp


def _run_chat_with(inputs, client: MagicMock, console: Console) -> None:
    from distllm.cli.chat import run_chat

    with patch("builtins.input", side_effect=lambda _prompt="": next(inputs)):
        with patch("distllm.cli.chat.client_for", return_value=client):
            run_chat("model", "localhost", 8000, 128, 0.7, console)


# ===========================================================================
# chat.py — run_chat
#   Covers: test_chat_quit_command, test_chat_clear_resets_conversation,
#           test_chat_empty_prompt_skipped, test_chat_http_error_handled,
#           test_chat_displays_usage_stats
# ===========================================================================


class TestRunChatQuality:
    def test_quit_command_prints_goodbye_and_sends_no_request(self):
        """'quit' must actually terminate with a farewell and no chat POST."""
        console, buf = _capture_console()
        client = _chat_client(_ok_response())

        _run_chat_with(iter(["quit"]), client, console)

        out = buf.getvalue()
        assert "DistLLM Interactive Chat" in out
        assert "Goodbye!" in out
        # The pre-flight liveness probe happened...
        client.get.assert_any_call("/health")
        # ...but no chat completion was ever requested.
        client.post.assert_not_called()

    def test_clear_resets_conversation_history(self):
        """After 'clear', the next POST carries only the new user message."""
        console, buf = _capture_console()
        sent: list[dict] = []
        client = _chat_client(_ok_response("Hi!"), sent=sent)

        _run_chat_with(
            iter(["Hello", "clear", "Second question", "quit"]), client, console
        )

        assert "Conversation cleared." in buf.getvalue()
        assert len(sent) == 2

        first = sent[0]["messages"]
        second = sent[1]["messages"]
        assert first == [{"role": "user", "content": "Hello"}]
        # If 'clear' were a no-op this would be
        # [user Hello, assistant Hi!, user Second question].
        assert second == [{"role": "user", "content": "Second question"}]

    def test_empty_and_whitespace_prompts_send_no_post(self):
        """Blank input is skipped — it must not produce a chat request."""
        console, buf = _capture_console()
        client = _chat_client(_ok_response())

        _run_chat_with(iter(["", "   ", "\t", "quit"]), client, console)

        client.post.assert_not_called()
        assert "Assistant:" not in buf.getvalue()

    def test_http_error_is_printed_and_loop_continues(self):
        """A failed POST is reported and the next message still works."""
        console, buf = _capture_console()
        error = httpx.HTTPStatusError(
            "500 Internal Server Error",
            request=MagicMock(),
            response=MagicMock(status_code=500),
        )
        client = _chat_client([error, _ok_response("Recovered")])

        _run_chat_with(iter(["First", "Second", "quit"]), client, console)

        out = buf.getvalue()
        assert "Error:" in out
        assert "500 Internal Server Error" in out
        # The loop must survive the error and serve the second message.
        assert client.post.call_count == 2
        assert "Recovered" in out

    def test_usage_stats_line_computed_from_response(self):
        """Usage + generation_time produce a real tokens/s line and payload."""
        console, buf = _capture_console()
        sent: list[dict] = []
        client = _chat_client(
            _ok_response("Hi there!", completion_tokens=3, generation_time=0.5),
            sent=sent,
        )

        _run_chat_with(iter(["Hello", "quit"]), client, console)

        out = buf.getvalue()
        assert "Hi there!" in out
        # 3 tokens / 0.5 s = 6.0 tokens/s — the exact rendered figure.
        assert "3 tokens in 0.5s (6.0 tokens/s)" in out

        assert len(sent) == 1
        payload = sent[0]
        assert payload["model"] == "model"
        assert payload["max_tokens"] == 128
        assert payload["temperature"] == 0.7
        assert payload["messages"] == [{"role": "user", "content": "Hello"}]


# ===========================================================================
# models.py — _list_models, _model_info
#   Covers: test_list_models_happy_path, test_list_models_empty,
#           test_model_info_found, test_model_info_not_found
# ===========================================================================


class TestModelCommandsQuality:
    @staticmethod
    def _client_returning(payload) -> MagicMock:
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = payload
        resp.raise_for_status.return_value = None

        client = MagicMock()
        client.__enter__.return_value = client
        client.__exit__.return_value = False
        client.get.return_value = resp
        return client

    def _invoke(self, func_name, args, payload):
        console, buf = _capture_console()
        client = self._client_returning(payload)
        import distllm.cli.models as models_mod

        fn = getattr(models_mod, func_name)
        with patch("distllm.cli.models.httpx.Client", return_value=client):
            with patch("distllm.cli.models.console", console):
                fn(*args)
        return client, buf.getvalue()

    def test_list_models_renders_table_with_ids(self):
        client, out = self._invoke(
            "_list_models",
            ("localhost", 8000),
            {
                "data": [
                    {"id": "model-1", "object": "model", "owned_by": "org-a"},
                    {"id": "model-2", "object": "model", "owned_by": "org-b"},
                ]
            },
        )
        assert "Models" in out
        assert "model-1" in out
        assert "model-2" in out
        assert "org-a" in out and "org-b" in out
        client.get.assert_called_once_with("/v1/models")

    def test_list_models_empty_reports_no_models(self):
        _client, out = self._invoke("_list_models", ("localhost", 8000), {"data": []})
        assert "No models available" in out
        # No table headers were rendered.
        assert "Owned By" not in out

    def test_model_info_found_renders_properties(self):
        client, out = self._invoke(
            "_model_info",
            ("localhost", 8000, "my-model"),
            {"data": [{"id": "my-model", "object": "model", "owned_by": "me"}]},
        )
        assert "Model: my-model" in out
        assert "owned_by" in out
        assert "me" in out
        assert "not found" not in out.lower()
        client.get.assert_called_once_with("/v1/models")

    def test_model_info_missing_reports_not_found(self):
        _client, out = self._invoke(
            "_model_info",
            ("localhost", 8000, "missing-model"),
            {"data": [{"id": "other-model", "object": "model", "owned_by": "org"}]},
        )
        assert "Model 'missing-model' not found" in out
        # The found-model property table must not be rendered.
        assert "Property" not in out
        assert "other-model" not in out


# ===========================================================================
# status.py — show_status, render_status_line
#   Covers: test_status_healthy, test_status_connect_error,
#           test_render_does_not_raise
# ===========================================================================


class TestStatusQuality:
    def test_show_status_healthy_renders_full_report(self):
        from distllm.cli.status import show_status

        health_resp = MagicMock()
        health_resp.json.return_value = {
            "status": "healthy",
            "model": "test-model",
            "nodes": 2,
            "node_health": {
                "node-0": {"healthy": True, "memory_used": 1024, "memory_total": 8192},
                "node-1": {"healthy": False, "memory_used": 0, "memory_total": 0},
            },
        }
        models_resp = MagicMock()
        models_resp.json.return_value = {"data": [{"id": "model-1"}, {"id": "model-2"}]}
        metrics_resp = MagicMock()
        metrics_resp.text = "# HELP cpu_usage\ncpu_usage 42\n"

        def get_side_effect(url, **kw):
            if "/health" in url:
                return health_resp
            if "/v1/models" in url:
                return models_resp
            if "/metrics" in url:
                return metrics_resp
            raise AssertionError(f"unexpected URL {url}")

        console, buf = _capture_console()
        with patch("distllm.cli.status.httpx.Client") as mc:
            inst = MagicMock()
            inst.get.side_effect = get_side_effect
            mc.return_value.__enter__ = MagicMock(return_value=inst)
            mc.return_value.__exit__ = MagicMock(return_value=False)

            show_status("localhost", 8000, console)

        out = buf.getvalue()
        assert "Cluster Status" in out
        assert "healthy" in out
        assert "test-model" in out
        # Per-node health is rendered differently for healthy/unhealthy...
        assert "✓ Healthy" in out
        assert "✗ Unhealthy" in out
        # ...with memory pressure computed for the node that reports it.
        assert "12.5%" in out
        # Models and metrics sections rendered their payloads.
        assert "model-1" in out
        assert "model-2" in out
        assert "cpu_usage 42" in out
        # Comment lines from the metrics text are filtered out.
        assert "# HELP" not in out

    def test_show_status_connect_error_reports_unreachable(self):
        from distllm.cli.status import show_status

        console, buf = _capture_console()
        with patch("distllm.cli.status.httpx.Client") as mc:
            inst = MagicMock()
            inst.get.side_effect = httpx.ConnectError("refused")
            mc.return_value.__enter__ = MagicMock(return_value=inst)
            mc.return_value.__exit__ = MagicMock(return_value=False)

            show_status("badhost", 9999, console)

        out = buf.getvalue()
        assert "Could not connect to http://badhost:9999" in out
        assert "Is the API server running?" in out

    def test_render_status_line_maps_status_to_style(self):
        """The colour is the behaviour: each status maps to its own style."""
        from distllm.cli.status import render_status_line

        console = MagicMock()
        render_status_line("DOWN  test", "unreachable", console)
        render_status_line("OK    test", "ok", console)
        render_status_line("DEGR  test", "degraded", console)
        render_status_line("????  test", "something-else", console)

        assert console.print.call_count == 4
        console.print.assert_any_call("DOWN  test", style="red")
        console.print.assert_any_call("OK    test", style="green")
        console.print.assert_any_call("DEGR  test", style="yellow")
        console.print.assert_any_call("????  test", style="white")


# ===========================================================================
# cluster.py — _cluster_status, _cluster_list_nodes
#   Covers: test_status_no_nodes, test_status_connection_error,
#           test_list_nodes_success
# ===========================================================================


def _cluster_client(get_result) -> MagicMock:
    client = MagicMock()
    client.__enter__.return_value = client
    client.__exit__.return_value = False
    if isinstance(get_result, Exception):
        client.get.side_effect = get_result
    else:
        client.get.return_value = get_result
    return client


class TestClusterQuality:
    def test_cluster_status_no_nodes_still_queries_health(self):
        """Empty roster must still produce the status table from /v1/health."""
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"nodes": [], "summary": {}}
        resp.raise_for_status.return_value = None
        client = _cluster_client(resp)

        console, buf = _capture_console()
        from distllm.cli import cluster

        with patch("distllm.cli.cluster.client_for", return_value=client):
            with patch("distllm.cli.cluster.console", console):
                cluster._cluster_status("localhost", 8000)

        out = buf.getvalue()
        client.get.assert_called_once_with("/v1/health")
        assert "Cluster Status" in out
        # Missing fields fall back to documented defaults, not blanks.
        assert "unknown" in out
        # No node-health table when the payload carries no node_health.
        assert "Node Health" not in out

    def test_cluster_status_connect_error_names_start_command(self):
        client = _cluster_client(httpx.ConnectError("refused"))

        console, buf = _capture_console()
        from distllm.cli import cluster

        with patch("distllm.cli.cluster.client_for", return_value=client):
            with patch("distllm.cli.cluster.console", console):
                cluster._cluster_status("localhost", 8000)

        out = buf.getvalue()
        assert "Could not connect to coordinator at localhost:8000" in out
        assert "distllm cluster start" in out

    def test_list_nodes_renders_node_rows_and_summary(self):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {
            "nodes": [
                {
                    "node_id": "node_0", "host": "10.0.0.1", "port": 50051,
                    "healthy": True, "state": "healthy",
                    "start_layer": 0, "end_layer": 15,
                    "gpu_name": "A100", "gpu_memory_free": 80 * 1024**3,
                },
                {
                    "node_id": "node_1", "host": "10.0.0.2", "port": 50052,
                    "healthy": True, "state": "healthy",
                    "start_layer": 16, "end_layer": 31,
                    "gpu_name": "A100", "gpu_memory_free": 40 * 1024**3,
                },
            ],
            "total_nodes": 2,
            "healthy_count": 2,
            "draining_count": 0,
        }
        resp.raise_for_status.return_value = None
        client = _cluster_client(resp)

        console, buf = _capture_console()
        from distllm.cli import cluster

        with patch("distllm.cli.cluster.client_for", return_value=client):
            with patch("distllm.cli.cluster.console", console):
                cluster._cluster_list_nodes("localhost", 8000)

        out = buf.getvalue()
        # Requested the live admin roster route, not the old nonexistent one.
        assert client.get.call_args.args[0] == "/admin/v1/nodes"
        # Both nodes, their endpoints, layer ranges and VRAM are rendered.
        assert "node_0" in out
        assert "node_1" in out
        assert "10.0.0.1:50051" in out
        assert "10.0.0.2:50052" in out
        assert "0-15" in out and "16-31" in out
        assert "80.0GB" in out and "40.0GB" in out
        # Summary line reflects the server's own counts.
        assert "2 nodes" in out
        assert "2 healthy" in out
        assert "0 draining" in out
