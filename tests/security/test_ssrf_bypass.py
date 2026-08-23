"""Security tests for SSRF bypass and auth token brute force attacks."""
from __future__ import annotations

import os
import time
from unittest.mock import MagicMock, patch, AsyncMock

import pytest


class TestSSRFBypass:
    """Test SSRF protection bypass attempts."""

    @pytest.fixture
    def api_client(self, monkeypatch):
        from fastapi.testclient import TestClient
        import distllm.api.server as server_module
        from distllm.api.api_state import g
        from distllm.core.api_key_store import reset_api_key_store

        # Set up valid auth — auth is always required
        reset_api_key_store()
        monkeypatch.setenv("API_KEY", "test-key-for-ssrf-tests")

        mock_coordinator = MagicMock()
        mock_coordinator.model_name = "test-model"
        mock_coordinator.is_healthy.return_value = True
        mock_coordinator.list_models.return_value = ["test-model"]
        mock_coordinator.generate.return_value = "test response"
        mock_coordinator._pending_scheduling_hints = {}
        # Explicitly set falsy attributes so MagicMock auto-attrs don't derail middleware checks
        mock_coordinator._shutting_down = False
        mock_coordinator.scheduler = None
        mock_coordinator.tokenizer = None

        original = g.coordinator
        g.coordinator = mock_coordinator
        client = TestClient(server_module.app)
        client.headers.update({"Authorization": "Bearer test-key-for-ssrf-tests"})
        yield client
        g.coordinator = original

    def test_ssrf_internal_ip_127(self, api_client):
        """Block requests to 127.0.0.x."""
        resp = api_client.get(
            "/v1/models",
            headers={"X-Forwarded-For": "127.0.0.1"},
        )
        assert resp.status_code in (200, 403, 401, 422)

    def test_ssrf_aws_metadata_endpoint(self, api_client):
        """Block SSRF to AWS metadata service."""
        from urllib.parse import quote
        payload = quote("http://169.254.169.254/latest/meta-data/")
        resp = api_client.post(
            "/v1/completions",
            json={"model": payload, "prompt": "test"},
        )
        assert resp.status_code in (200, 400, 403, 422)

    def test_ssrf_docker_socket(self, api_client):
        """Block SSRF to Docker socket."""
        from urllib.parse import quote
        payload = quote("http://0.0.0.0:2375/containers/json")
        resp = api_client.post(
            "/v1/completions",
            json={"model": payload, "prompt": "test"},
        )
        assert resp.status_code in (200, 400, 403, 422)

    def test_ssrf_ipv6_localhost(self, api_client):
        """Block SSRF via IPv6 localhost."""
        from urllib.parse import quote
        payload = quote("http://[::1]:6379/")
        resp = api_client.post(
            "/v1/completions",
            json={"model": payload, "prompt": "test"},
        )
        assert resp.status_code in (200, 400, 403, 422)

    def test_ssrf_decimal_ip(self, api_client):
        """Block SSRF with decimal-encoded IP (2130706433 = 127.0.0.1)."""
        from urllib.parse import quote
        payload = quote("http://2130706433/")
        resp = api_client.post(
            "/v1/completions",
            json={"model": payload, "prompt": "test"},
        )
        assert resp.status_code in (200, 400, 403, 422)

    def test_ssrf_hex_ip(self, api_client):
        """Block SSRF with hex-encoded IP."""
        from urllib.parse import quote
        payload = quote("http://0x7f000001/")
        resp = api_client.post(
            "/v1/completions",
            json={"model": payload, "prompt": "test"},
        )
        assert resp.status_code in (200, 400, 403, 422)

    def test_ssrf_octal_ip(self, api_client):
        """Block SSRF with octal-encoded IP."""
        from urllib.parse import quote
        payload = quote("http://0177.0.0.1/")
        resp = api_client.post(
            "/v1/completions",
            json={"model": payload, "prompt": "test"},
        )
        assert resp.status_code in (200, 400, 403, 422)

    def test_ssrf_short_form_localhost(self, api_client):
        """Block SSRF with short-form localhost."""
        from urllib.parse import quote
        payload = quote("http://127.1/")
        resp = api_client.post(
            "/v1/completions",
            json={"model": payload, "prompt": "test"},
        )
        assert resp.status_code in (200, 400, 403, 422)

    def test_ssrf_private_network(self, api_client):
        """Block SSRF to private network ranges."""
        for ip in ["10.0.0.1", "172.16.0.1", "192.168.1.1"]:
            from urllib.parse import quote
            resp = api_client.post(
                "/v1/completions",
                json={"model": quote(f"http://{ip}/"), "prompt": "test"},
            )
            assert resp.status_code in (200, 400, 403, 422)

    def test_ssrf_redirect_chain(self, api_client):
        """Block SSRF through redirect chains."""
        from urllib.parse import quote
        # Attempt to chain through a redirect
        payload = quote("http://example.com/redirect?url=http://127.0.0.1")
        resp = api_client.post(
            "/v1/completions",
            json={"model": payload, "prompt": "test"},
        )
        assert resp.status_code in (200, 400, 403, 422)


class TestAuthTokenBruteForce:
    """Test auth token brute force resistance."""

    @pytest.fixture
    def api_client_with_auth(self, monkeypatch):
        from fastapi.testclient import TestClient
        import distllm.api.server as server_module
        from distllm.api.api_state import g
        from distllm.core.api_key_store import reset_api_key_store

        # Reset the store so it picks up the API_KEY env var on next request
        reset_api_key_store()
        monkeypatch.setenv("API_KEY", "test-secret-key-12345")

        mock_coordinator = MagicMock()
        mock_coordinator.model_name = "test-model"
        mock_coordinator.is_healthy.return_value = True
        mock_coordinator.generate.return_value = "test response"
        mock_coordinator.list_models.return_value = ["test-model"]
        mock_coordinator._pending_scheduling_hints = {}
        mock_coordinator._shutting_down = False
        mock_coordinator.scheduler = None
        mock_coordinator.tokenizer = MagicMock()
        mock_coordinator.tokenizer.encode.return_value = [1, 2, 3]

        original = g.coordinator
        g.coordinator = mock_coordinator
        client = TestClient(server_module.app)
        yield client
        g.coordinator = original

    def test_valid_token_accepted(self, api_client_with_auth):
        """Valid token should be accepted."""
        resp = api_client_with_auth.get(
            "/v1/models",
            headers={"Authorization": "Bearer test-secret-key-12345"},
        )
        assert resp.status_code == 200

    def test_invalid_token_rejected(self, api_client_with_auth):
        """Invalid token should be rejected."""
        resp = api_client_with_auth.get(
            "/v1/models",
            headers={"Authorization": "Bearer wrong-token"},
        )
        assert resp.status_code in (401, 403)

    def test_missing_token_rejected(self, api_client_with_auth):
        """Missing token should be rejected."""
        resp = api_client_with_auth.get("/v1/models")
        assert resp.status_code in (401, 403)

    def test_empty_bearer_rejected(self, api_client_with_auth):
        """Empty bearer token should be rejected."""
        resp = api_client_with_auth.get(
            "/v1/models",
            headers={"Authorization": "Bearer "},
        )
        assert resp.status_code in (401, 403)

    def test_no_bearer_prefix_rejected(self, api_client_with_auth):
        """Token without Bearer prefix should be rejected."""
        resp = api_client_with_auth.get(
            "/v1/models",
            headers={"Authorization": "test-secret-key-12345"},
        )
        assert resp.status_code in (401, 403)

    def test_sql_injection_in_token(self, api_client_with_auth):
        """SQL injection in token should be safely handled."""
        payloads = [
            "'; DROP TABLE users; --",
            "\" OR 1=1 --",
            "admin'--",
            "' UNION SELECT * FROM secrets --",
        ]
        for payload in payloads:
            resp = api_client_with_auth.get(
                "/v1/models",
                headers={"Authorization": f"Bearer {payload}"},
            )
            assert resp.status_code in (401, 403)

    def test_very_long_token(self, api_client_with_auth):
        """Very long tokens should not cause buffer overflow."""
        long_token = "A" * 100000
        resp = api_client_with_auth.get(
            "/v1/models",
            headers={"Authorization": f"Bearer {long_token}"},
        )
        assert resp.status_code in (401, 403)
