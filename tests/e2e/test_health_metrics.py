"""E2E tests for node lifecycle: register, request, failover."""

import pytest


@pytest.mark.e2e
class TestNodeLifecycleE2E:
    def test_health_endpoint_returns_ok(self, e2e_api_client):
        """Health endpoint should return 200 when coordinator is running."""
        response = e2e_api_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data

    def test_metrics_endpoint_returns_ok(self, e2e_api_client):
        """Metrics endpoint should return 200 (may be empty without exporter)."""
        response = e2e_api_client.get("/metrics")
        assert response.status_code == 200

    def test_models_endpoint_lists_model(self, e2e_api_client):
        """Models endpoint should list available models."""
        response = e2e_api_client.get("/v1/models")
        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        assert isinstance(data["data"], list)

    def test_api_server_rejects_request_without_coordinator(self, api_client_no_coordinator):
        """API should return 503 when coordinator is not available."""
        response = api_client_no_coordinator.post(
            "/v1/chat/completions",
            json={
                "model": "distributed-llm",
                "messages": [{"role": "user", "content": "test"}],
                "max_tokens": 5,
            },
        )
        assert response.status_code == 503
