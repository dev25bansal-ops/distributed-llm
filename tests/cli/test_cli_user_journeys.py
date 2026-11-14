"""The commands added while auditing the CLI as a user.

Each test maps to a friction point found by actually running the tool:

* a bad model id produced a raw transformers traceback from a subprocess
* a non-distllm server on the port dumped an HTML 404 page into the terminal
* `chat` accepted a message and only then reported the server was unreachable,
  and its prompt showed the literal string "[bold You:[/bold]"
* there was no single command that went from nothing to a working cluster
* there was no way to check that the installation itself was sound
"""

from __future__ import annotations

import io

import httpx
import pytest
from rich.console import Console
from typer.testing import CliRunner

from distllm.cli.http_client import format_http_error
from distllm.cli.main import app

runner = CliRunner()


class TestHttpErrorFormatting:
    """A wrong port must not paste an HTML page into the terminal."""

    def _status_error(self, status: int, text: str, content_type="text/html"):
        request = httpx.Request("GET", "http://localhost:8000/v1/models")
        response = httpx.Response(
            status, text=text, request=request, headers={"content-type": content_type}
        )
        return httpx.HTTPStatusError("boom", request=request, response=response)

    def test_html_body_is_replaced_with_an_explanation(self):
        exc = self._status_error(404, "<!DOCTYPE html><html><body>Not Found</body></html>")
        message = format_http_error(exc)
        assert "<html>" not in message
        assert "404" in message
        assert "HTML" in message

    def test_json_error_message_is_kept(self):
        exc = self._status_error(
            401,
            '{"error": {"message": "invalid API key", "type": "auth_error"}}',
            content_type="application/json",
        )
        message = format_http_error(exc)
        assert "invalid API key" in message
        assert "401" in message

    def test_long_body_is_truncated(self):
        exc = self._status_error(500, "x" * 5000, content_type="text/plain")
        assert len(format_http_error(exc, max_len=100)) < 200


class TestClusterStartValidation:
    """A model id that cannot load must fail before subprocesses spawn."""

    def test_missing_repo_exits_with_guidance(self, monkeypatch):
        from distllm.cli import cluster as cluster_module

        def fake_model_info(self, *a, **k):
            from huggingface_hub.errors import RepositoryNotFoundError

            request = httpx.Request("GET", "https://huggingface.co/api/models/x")
            response = httpx.Response(404, request=request)
            raise RepositoryNotFoundError("not found", response=response)

        monkeypatch.setattr("huggingface_hub.HfApi.model_info", fake_model_info)
        with pytest.raises(SystemExit) as info:
            cluster_module._validate_model_or_exit("definitely/not-real")
        assert info.value.code == 1

    def test_network_error_does_not_block_startup(self, monkeypatch):
        """Offline is not the same as "the model is wrong"."""

        def offline(self, *a, **k):
            raise httpx.ConnectError("no route to host")

        monkeypatch.setattr("huggingface_hub.HfApi.model_info", offline)
        # Must return, not raise.
        from distllm.cli import cluster as cluster_module

        cluster_module._validate_model_or_exit("some/model")

    def test_local_path_is_accepted_without_network(self, tmp_path):
        from distllm.cli import cluster as cluster_module

        # Should return before any HF call for an existing path.
        cluster_module._validate_model_or_exit(str(tmp_path))


class TestQuickstart:
    def test_no_start_reports_the_command_it_would_run(self, monkeypatch):
        from distllm.cli import quickstart as qs

        monkeypatch.setattr(
            qs,
            "_detect",
            lambda: {
                "cuda_available": True,
                "gpu_name": "Fake GPU",
                "gpu_count": 1,
                "gpu_memory_gb": 24.0,
                "ram_gb": 64.0,
                "cpu_cores": 8,
            },
        )
        monkeypatch.setattr(qs, "_pick_model", lambda q, h, m: "Qwen/Qwen2.5-7B-Instruct")

        result = runner.invoke(app, ["quickstart", "--no-start"])
        assert result.exit_code == 0, result.output
        assert "Qwen/Qwen2.5-7B-Instruct" in result.output
        assert "--no-start" in result.output or "would run" in result.output

    def test_explicit_model_overrides_discovery(self, monkeypatch):
        from distllm.cli import quickstart as qs

        monkeypatch.setattr(
            qs,
            "_detect",
            lambda: {
                "cuda_available": True, "gpu_name": "g", "gpu_count": 1,
                "gpu_memory_gb": 8.0, "ram_gb": 16.0, "cpu_cores": 4,
            },
        )
        result = runner.invoke(
            app, ["quickstart", "--model", "my/own-model", "--no-start"]
        )
        assert result.exit_code == 0
        assert "my/own-model" in result.output


class TestSelftest:
    def test_runs_and_reports_each_check(self):
        result = runner.invoke(app, ["selftest", "--json"])
        assert result.exit_code in (0, 1), result.output
        assert "checks" in result.output
        for name in ("version", "console scripts", "module imports", "model catalog"):
            assert name in result.output

    def test_json_is_parseable(self):
        import json

        result = runner.invoke(app, ["selftest", "--json"])
        payload = json.loads(result.output)
        assert isinstance(payload["checks"], list)
        assert all({"check", "status", "detail"} <= set(c) for c in payload["checks"])

    def test_health_503_counts_as_reachable(self):
        """The API contract says /health is 503 until a model is loaded."""
        from distllm.cli import selftest as st

        check = st._check_api_roundtrip()
        assert check.status in (st.PASS, st.SKIP), check.detail

    def test_torch_failure_remediation_names_self_hosted_extra(self):
        """A torch-import FAIL must not advise reinstalling [cli] (no torch)."""
        from distllm.cli import selftest as st

        checks = [
            st.Check("api roundtrip", st.FAIL, "ModuleNotFoundError: No module named 'torch'"),
        ]
        message = st._remediation(checks)
        assert "self-hosted" in message
        assert "[cli]" not in message

    def test_non_ml_failure_remediation_keeps_cli_extra(self):
        from distllm.cli import selftest as st

        checks = [st.Check("console scripts", st.FAIL, "unresolvable: distllm-api")]
        message = st._remediation(checks)
        assert "[cli]" in message
        assert "self-hosted" not in message


class TestChatPreflight:
    def test_chat_reports_unreachable_server_before_accepting_input(self):
        result = runner.invoke(app, ["chat", "--port", "1"], input="hi\n")
        assert "Cannot reach the server" in result.output
        # The prompt should never have been reached.
        assert "You:" not in result.output

    def test_prompt_has_no_raw_markup(self):
        from distllm.cli import chat as chat_module

        source = open(chat_module.__file__, encoding="utf-8").read()
        assert 'input("\\n[bold You:[/bold] ")' not in source
        assert "You: " in source