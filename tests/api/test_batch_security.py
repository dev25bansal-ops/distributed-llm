"""Security regression tests for the background batch endpoints.

Covers two P0 findings from the audit:
1. SSRF — ``webhook_url`` was POSTed to arbitrary hosts (including cloud
   metadata / loopback), carrying batch results and the caller's
   ``webhook_token``.
2. IDOR — status / stream / cancel were keyed only on the global ``batch_id``
   with no ownership binding, so any authenticated key could read or cancel
   another tenant's batch.
"""

from __future__ import annotations

import json
import os
import secrets
import time
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from distllm.api.api_state import g
from distllm.api.server import app
from distllm.core.api_key_store import reset_api_key_store


@pytest.fixture(autouse=True)
def _two_keys(monkeypatch):
    """Register two distinct API keys so cross-tenant access is testable."""
    global KEY_A, KEY_B
    KEY_A = secrets.token_urlsafe(32)
    KEY_B = secrets.token_urlsafe(32)
    monkeypatch.setenv(
        "API_KEYS",
        json.dumps({
            "keys": [
                {"key": KEY_A, "role": "admin", "label": "a", "key_id": "key-a"},
                {"key": KEY_B, "role": "admin", "label": "b", "key_id": "key-b"},
            ]
        }),
    )
    monkeypatch.delenv("API_KEY", raising=False)
    reset_api_key_store()
    yield
    monkeypatch.delenv("API_KEYS", raising=False)


@pytest.fixture
def coordinator():
    coord = MagicMock()
    coord.model_name = "test-model"
    coord.nodes = {}
    coord._shutting_down = False
    coord.generate = MagicMock(
        side_effect=lambda *a, **k: (time.sleep(0.5) or "batch response")
    )
    return coord


def _client(key: str) -> TestClient:
    client = TestClient(app)
    client.headers["Authorization"] = f"Bearer {key}"
    return client


class _WithCoordinator:
    @pytest.fixture(autouse=True)
    def _set_coordinator(self, coordinator):
        original = g.coordinator
        g.coordinator = coordinator
        yield
        g.coordinator = original


class TestBatchWebhookSSRF(_WithCoordinator):
    def test_http_scheme_rejected(self):
        resp = _client(KEY_A).post(
            "/v1/batch/submit",
            json={"items": [{"prompt": "hi"}], "webhook_url": "http://example.com/hook"},
        )
        assert resp.status_code == 400

    def test_loopback_rejected(self):
        resp = _client(KEY_A).post(
            "/v1/batch/submit",
            json={"items": [{"prompt": "hi"}], "webhook_url": "https://127.0.0.1/hook"},
        )
        assert resp.status_code == 400

    def test_ipv6_loopback_rejected(self):
        resp = _client(KEY_A).post(
            "/v1/batch/submit",
            json={"items": [{"prompt": "hi"}], "webhook_url": "https://[::1]/hook"},
        )
        assert resp.status_code == 400

    def test_cloud_metadata_rejected(self):
        resp = _client(KEY_A).post(
            "/v1/batch/submit",
            json={
                "items": [{"prompt": "hi"}],
                "webhook_url": "https://169.254.169.254/latest/meta-data/",
            },
        )
        assert resp.status_code == 400

    def test_private_rfc1918_rejected(self):
        resp = _client(KEY_A).post(
            "/v1/batch/submit",
            json={"items": [{"prompt": "hi"}], "webhook_url": "https://10.0.0.5/hook"},
        )
        assert resp.status_code == 400

    def test_public_https_accepted(self, monkeypatch):
        # Pin DNS to a public address so the test needs no real network.
        monkeypatch.setattr(
            "socket.getaddrinfo",
            lambda *a, **k: [(2, 1, 6, "", ("93.184.216.34", 443))],
        )
        resp = _client(KEY_A).post(
            "/v1/batch/submit",
            json={"items": [{"prompt": "hi"}], "webhook_url": "https://example.com/hook"},
        )
        assert resp.status_code == 202


class TestBatchOwnership(_WithCoordinator):
    def test_owner_can_read_status(self):
        client_a = _client(KEY_A)
        created = client_a.post(
            "/v1/batch/submit", json={"items": [{"prompt": "hi"}]}
        )
        assert created.status_code == 202
        batch_id = created.json()["batch_id"]

        resp = client_a.get(f"/v1/batch/{batch_id}/status")
        assert resp.status_code == 200

    def test_other_tenant_cannot_read_status(self):
        client_a = _client(KEY_A)
        created = client_a.post(
            "/v1/batch/submit", json={"items": [{"prompt": "hi"}]}
        )
        batch_id = created.json()["batch_id"]

        resp = _client(KEY_B).get(f"/v1/batch/{batch_id}/status")
        assert resp.status_code == 404  # existence is not leaked

    def test_other_tenant_cannot_cancel(self):
        client_a = _client(KEY_A)
        created = client_a.post(
            "/v1/batch/submit", json={"items": [{"prompt": "hi"}]}
        )
        batch_id = created.json()["batch_id"]

        resp = _client(KEY_B).post(f"/v1/batch/{batch_id}/cancel")
        assert resp.status_code == 404

    def test_owner_can_cancel(self):
        client_a = _client(KEY_A)
        created = client_a.post(
            "/v1/batch/submit", json={"items": [{"prompt": "hi"}]}
        )
        batch_id = created.json()["batch_id"]

        resp = client_a.post(f"/v1/batch/{batch_id}/cancel")
        assert resp.status_code == 200
        assert resp.json()["status"] == "cancelled"

    def test_other_tenant_cannot_stream(self):
        client_a = _client(KEY_A)
        created = client_a.post(
            "/v1/batch/submit", json={"items": [{"prompt": "hi"}]}
        )
        batch_id = created.json()["batch_id"]

        resp = _client(KEY_B).get(f"/v1/batch/{batch_id}/stream")
        assert resp.status_code == 404
