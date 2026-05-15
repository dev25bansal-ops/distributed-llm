"""Tests for Feature 27: Web Dashboard."""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def dashboard_client():
    """TestClient for the dashboard app."""
    from distllm.dashboard.app import dashboard_app

    return TestClient(dashboard_app)


class TestDashboardPages:
    def test_dashboard_serves_html(self, dashboard_client):
        """Dashboard root should return HTML."""
        response = dashboard_client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")
        assert "DistLLM" in response.text

    def test_dashboard_has_v2_elements(self, dashboard_client):
        """V2 dashboard should have modern UI elements."""
        response = dashboard_client.get("/")
        text = response.text
        assert "WebSocket" in text or "ws" in text  # WebSocket connection
        assert "node-list" in text  # Node health section
        assert "scheduler-stats" in text  # Scheduler section


class TestDashboardAPI:
    def test_status_without_coordinator(self, dashboard_client):
        """Status endpoint returns error when no coordinator."""
        response = dashboard_client.get("/api/status")
        assert response.status_code == 200
        data = response.json()
        assert "error" in data or "No coordinator" in str(data)

    def test_nodes_without_coordinator(self, dashboard_client):
        """Nodes endpoint returns empty list when no coordinator."""
        response = dashboard_client.get("/api/nodes")
        assert response.status_code == 200
        assert response.json() == []

    def test_metrics_without_coordinator(self, dashboard_client):
        """Metrics endpoint returns empty dict when no coordinator."""
        response = dashboard_client.get("/api/metrics")
        assert response.status_code == 200
        assert response.json() == {}

    def test_update_config_without_coordinator(self, dashboard_client):
        """Config update returns 503 when no coordinator."""
        response = dashboard_client.post(
            "/api/config",
            json={"batch_size": 16},
        )
        assert response.status_code == 503


class TestDashboardWebSocket:
    def test_websocket_connect(self, dashboard_client):
        """WebSocket endpoint should accept connections."""
        with dashboard_client.websocket_connect("/ws") as websocket:
            # Connection should be established
            # Send a message to keep alive (will immediately disconnect)
            websocket.send_text("ping")

    def test_websocket_multiple_connections(self, dashboard_client):
        """Multiple WebSocket connections should work."""
        with dashboard_client.websocket_connect("/ws") as ws1:
            ws1.send_text("ping")
            with dashboard_client.websocket_connect("/ws") as ws2:
                ws2.send_text("ping")
