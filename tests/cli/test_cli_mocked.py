"""Tests for CLI modules with mocked HTTP responses and retry logic."""
from __future__ import annotations
import pytest
from typer.testing import CliRunner
from distllm.cli.main import app

from tests.cli.stubs import StubClient, StubHttpxClient, StubResponse, StubFn

runner = CliRunner()


class TestModelCommands:
    def test_model_list(self, monkeypatch):
        import distllm.cli.models as _m
        monkeypatch.setattr(_m, "_list_models", StubFn(), raising=False)
        result = runner.invoke(app, ["model", "list"])
        assert result.exit_code == 0

    def test_model_list_connection_error(self, monkeypatch):
        import distllm.cli.models as _m
        monkeypatch.setattr(_m, "_list_models", StubFn(), raising=False)
        result = runner.invoke(app, ["model", "list"])
        assert result.exit_code == 0


class TestClusterCommands:
    def test_cluster_status(self, monkeypatch):
        client = StubHttpxClient()
        client.set_get(StubResponse(
            status_code=200,
            json_data={"nodes": [{"id": "node-1", "status": "healthy"}]},
        ))
        monkeypatch.setattr("distllm.cli.cluster._get_client", StubFn(return_value=client))
        result = runner.invoke(app, ["cluster", "status"])
        assert result.exit_code == 0

    def test_cluster_status_connection_error(self, monkeypatch):
        import httpx
        client = StubHttpxClient()
        client.set_get(side_effect=httpx.ConnectError("refused"))
        monkeypatch.setattr("distllm.cli.cluster._get_client", StubFn(return_value=client))
        result = runner.invoke(app, ["cluster", "status"])
        assert result.exit_code == 0


class _StubSession:
    """Stub session for DistLLMClient retry tests."""

    def __init__(self):
        self.request = _StubRequestMethod()

    def close(self):
        pass


class _StubRequestMethod:
    """Stub for session.request() in retry tests."""

    def __init__(self):
        self.side_effect = []

    def set_side_effect(self, responses: list) -> None:
        self.side_effect = list(responses)
        self._idx = 0

    def __call__(self, method, path, **kwargs):
        if self._idx < len(self.side_effect):
            result = self.side_effect[self._idx]
            self._idx += 1
            if isinstance(result, Exception):
                raise result
            return result
        return None


class _StubResponseRetry:
    """HTTP response stub for retry tests."""

    def __init__(self, status_code: int = 200, json_data: dict | None = None,
                 headers: dict | None = None):
        self.status_code = status_code
        self._json_data = json_data
        self.headers = headers or {}

    def json(self):
        return self._json_data


class TestClientRetry:
    def test_retry_on_5xx(self, monkeypatch):
        from distllm.cli.client import DistLLMClient, ClientConfig
        c = DistLLMClient(config=ClientConfig(
            base_url="http://localhost:8000", max_retries=2, retry_delay=0.01,
        ))
        session = _StubSession()
        session.request.set_side_effect([
            _StubResponseRetry(status_code=503, json_data={"error": "overloaded"}),
            _StubResponseRetry(status_code=503, json_data={"error": "overloaded"}),
            _StubResponseRetry(status_code=200, json_data={"status": "ok"}),
        ])
        monkeypatch.setattr(c, "_get_session", StubFn(return_value=session))
        result = c.get("/health")
        assert result == {"status": "ok"}

    def test_no_retry_on_4xx(self, monkeypatch):
        from distllm.cli.client import DistLLMClient, ClientConfig, DistLLMError
        c = DistLLMClient(config=ClientConfig(
            base_url="http://localhost:8000", max_retries=2, retry_delay=0.01,
        ))
        session = _StubSession()
        session.request.set_side_effect([
            _StubResponseRetry(status_code=401, json_data={"error": "unauthorized"}),
        ])
        monkeypatch.setattr(c, "_get_session", StubFn(return_value=session))
        with pytest.raises(DistLLMError):
            c.get("/v1/models")

    def test_rate_limit_retry(self, monkeypatch):
        from distllm.cli.client import DistLLMClient, ClientConfig
        c = DistLLMClient(config=ClientConfig(
            base_url="http://localhost:8000", max_retries=2, retry_delay=0.01,
        ))
        session = _StubSession()
        session.request.set_side_effect([
            _StubResponseRetry(status_code=429, headers={"Retry-After": "0"}),
            _StubResponseRetry(status_code=429, headers={"Retry-After": "0"}),
            _StubResponseRetry(status_code=200, json_data={"status": "ok"}),
        ])
        monkeypatch.setattr(c, "_get_session", StubFn(return_value=session))
        result = c.get("/health")
        assert result == {"status": "ok"}

    def test_auth_header(self):
        from distllm.cli.client import DistLLMClient
        client = DistLLMClient(base_url="http://localhost:8000", api_key="test-key-123")
        assert client._headers["Authorization"] == "Bearer test-key-123"

    def test_no_auth_when_no_key(self):
        from distllm.cli.client import DistLLMClient
        client = DistLLMClient(base_url="http://localhost:8000")
        assert "Authorization" not in client._headers


class TestHelpOutputs:
    def test_main_help(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        for cmd in ["chat", "completion", "dashboard", "tutorial"]:
            assert cmd in result.stdout

    def test_model_help(self):
        result = runner.invoke(app, ["model", "--help"])
        assert result.exit_code == 0
        assert "list" in result.stdout

    def test_cluster_help(self):
        result = runner.invoke(app, ["cluster", "--help"])
        assert result.exit_code == 0

    def test_config_help(self):
        result = runner.invoke(app, ["config", "--help"])
        assert result.exit_code == 0
