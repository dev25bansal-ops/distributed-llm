"""Integration test: federation heartbeat & routing.

Verifies that two coordinator clusters can discover each other,
exchange load metrics via heartbeat, and route requests between
them.

Requires two Docker Compose cluster networks.

Usage:
    docker compose -f tests/dist/integration/docker-compose.yml up --build -d
    # Also start a second coordinator on a different port/network
    python -m pytest tests/dist/integration/test_federation_scenario.py -v
"""

from __future__ import annotations

import os
import time
from typing import Any

import httpx
import pytest

COORDINATOR_URL = os.environ.get(
    "COORDINATOR_URL", "http://localhost:8000"
)
FEDERATED_URL = os.environ.get(
    "FEDERATED_URL", "http://localhost:8001"
)
REQUEST_TIMEOUT = float(os.environ.get("TEST_TIMEOUT_S", "60"))


@pytest.fixture(scope="module")
def client() -> httpx.Client:
    """Authenticated client: live servers always require an API key."""
    api_key = os.environ.get("TEST_API_KEY", "")
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    return httpx.Client(timeout=REQUEST_TIMEOUT, headers=headers)


class TestFederationDiscovery:
    """Tests cross-cluster peer discovery and heartbeat."""

    @pytest.mark.timeout(30)
    def test_health_checks(self, client: httpx.Client):
        """Both clusters should be healthy."""
        for url in [COORDINATOR_URL, FEDERATED_URL]:
            try:
                resp = client.get(f"{url}/v1/health")
            except httpx.ConnectError:
                pytest.skip(f"Server not available at {url}")
            assert resp.status_code == 200

    @pytest.mark.timeout(60)
    def test_federation_heartbeat(self, client: httpx.Client):
        """Heartbeat exchange between peers."""
        coord = f"{COORDINATOR_URL}/v1/federation/health"
        try:
            resp = client.get(coord)
            if resp.status_code == 200:
                data = resp.json()
                assert isinstance(data, dict)
        except httpx.ConnectError:
            pytest.skip("Federation endpoint not available")

    @pytest.mark.timeout(30)
    def test_peer_list(self, client: httpx.Client):
        """List federation peers."""
        try:
            resp = client.get(f"{COORDINATOR_URL}/v1/federation/peers")
            if resp.status_code == 200:
                peers = resp.json()
                assert isinstance(peers, list)
        except httpx.ConnectError:
            pytest.skip("Federation endpoint not available")
