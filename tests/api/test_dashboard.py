"""Tests for the embedded dashboard in the API server.

The standalone dashboard was removed in v0.4.0.  The dashboard is now
embedded in the API server at ``/dashboard`` serving ``static_v2/index.html``.
"""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def server_client(monkeypatch):
    """TestClient for the main API server, authenticated with a known key."""
    from distllm.api.server import app
    from distllm.core.api_key_store import reset_api_key_store

    monkeypatch.setenv("API_KEY", "dash-test-key-0123456789abcdef")
    reset_api_key_store()
    c = TestClient(app)
    c.headers["Authorization"] = "Bearer dash-test-key-0123456789abcdef"
    yield c
    reset_api_key_store()


class TestDashboardPage:
    def test_dashboard_serves_html(self, server_client):
        """Dashboard at /dashboard should return HTML."""
        response = server_client.get("/dashboard")
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")
        assert "DistLLM" in response.text or "Dashboard" in response.text

    def test_dashboard_has_v2_elements(self, server_client):
        """V2 dashboard should have modern UI elements."""
        response = server_client.get("/dashboard")
        text = response.text
        assert "WebSocket" in text or "ws" in text, "WebSocket connection expected"
        assert "Real-Time Dashboard" in text, "Dashboard title expected"
        assert "node-list" in text, "Node health section expected"

    def test_dashboard_waterfall_endpoint(self, server_client):
        """Waterfall endpoint returns empty list when no coordinator."""
        response = server_client.get("/api/requests/waterfall?limit=10")
        assert response.status_code == 200
        assert response.json() == []

    def test_dashboard_cluster_nodes_endpoint(self, server_client):
        """Cluster nodes endpoint returns empty list when no coordinator."""
        response = server_client.get("/api/cluster/nodes")
        assert response.status_code == 200
        data = response.json()
        assert "nodes" in data
