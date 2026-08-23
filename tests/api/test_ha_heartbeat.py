"""Tests for the HA leader-election heartbeat route (POST /api/v1/ha/heartbeat).

The route is the transport endpoint for ``RayFaultTolerance`` outbound
heartbeats (C1 fix): peers POST their election term, the route refreshes the
sender's liveness via ``handle_heartbeat_request``, and the shared HA secret
is enforced when ``DISTLLM_HA_SECRET`` is configured.
"""

from __future__ import annotations

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
def _setup_auth(monkeypatch):
    test_api_key = secrets.token_hex(32)
    monkeypatch.setenv("API_KEY", test_api_key)
    monkeypatch.delenv("API_KEY_WAS_SET", raising=False)
    reset_api_key_store()
    g._startup_time = time.time()
    yield
    monkeypatch.delenv("API_KEY", raising=False)


def _ha_coordinator():
    election = MagicMock()
    election.handle_heartbeat_request.return_value = {
        "coordinator_id": "coordinator-b",
        "state": "follower",
        "term": 2,
        "leader_id": "coordinator-a",
    }
    coord = MagicMock()
    coord._shutting_down = False  # don't trip the shutdown gate
    coord._election._ha_election = election
    return coord, election


class TestHAHeartbeat:
    def test_heartbeat_when_no_coordinator(self):
        original = g.coordinator
        g.coordinator = None
        try:
            resp = TestClient(app).post(
                "/api/v1/ha/heartbeat",
                json={"coordinator_id": "a", "term": 1},
            )
            assert resp.status_code == 200
            assert resp.json()["status"] == "error"
        finally:
            g.coordinator = original

    def test_heartbeat_dispatches_to_election(self, monkeypatch):
        coord, election = _ha_coordinator()
        original = g.coordinator
        g.coordinator = coord
        monkeypatch.setenv("DISTLLM_HA_SECRET", "s3cret")
        try:
            resp = TestClient(app).post(
                "/api/v1/ha/heartbeat",
                json={"coordinator_id": "coordinator-a", "term": 2, "state": {"k": 1}},
                headers={"X-HA-Secret": "s3cret"},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "ok"
            assert data["peer"]["leader_id"] == "coordinator-a"
            election.handle_heartbeat_request.assert_called_once_with(
                "coordinator-a", 2, {"k": 1}
            )
        finally:
            g.coordinator = original
            monkeypatch.delenv("DISTLLM_HA_SECRET", raising=False)

    def test_heartbeat_fail_closed_when_secret_unset(self):
        """Without DISTLLM_HA_SECRET the route must REFUSE (fail closed), not
        accept unauthenticated leader-election input from any socket."""
        coord, _ = _ha_coordinator()
        original = g.coordinator
        g.coordinator = coord
        try:
            resp = TestClient(app).post(
                "/api/v1/ha/heartbeat",
                json={"coordinator_id": "attacker", "term": 999999, "state": {"evil": True}},
            )
            assert resp.status_code == 403
            assert "secret" in resp.json().get("detail", "").lower()
        finally:
            g.coordinator = original

    def test_heartbeat_snapshot_fail_closed_when_secret_unset(self):
        """The sibling /api/v1/ha/snapshot route must also fail closed."""
        coord = MagicMock()
        coord._shutting_down = False
        original = g.coordinator
        g.coordinator = coord
        try:
            resp = TestClient(app).post(
                "/api/v1/ha/snapshot",
                json={"nodes": {"evil-node": {"host": "x"}}, "metadata": {"m": 1}},
                headers={"Authorization": f"Bearer {os.environ['API_KEY']}"},
            )
            assert resp.status_code == 403
            assert "secret" in resp.json().get("detail", "").lower()
            # The snapshot must never be applied.
            coord.apply_state_snapshot.assert_not_called()
        finally:
            g.coordinator = original

    def test_heartbeat_snapshot_requires_correct_secret(self, monkeypatch):
        coord = MagicMock()
        coord._shutting_down = False
        original = g.coordinator
        g.coordinator = coord
        monkeypatch.setenv("DISTLLM_HA_SECRET", "s3cret")
        try:
            denied = TestClient(app).post(
                "/api/v1/ha/snapshot",
                json={"nodes": {"n": {}}},
                headers={
                    "Authorization": f"Bearer {os.environ['API_KEY']}",
                    "X-HA-Secret": "wrong",
                },
            )
            assert denied.status_code == 403
            coord.apply_state_snapshot.assert_not_called()

            allowed = TestClient(app).post(
                "/api/v1/ha/snapshot",
                json={"nodes": {"n": {}}, "metadata": {}},
                headers={
                    "Authorization": f"Bearer {os.environ['API_KEY']}",
                    "X-HA-Secret": "s3cret",
                },
            )
            assert allowed.status_code == 200
            coord.apply_state_snapshot.assert_called_once()
        finally:
            g.coordinator = original
            monkeypatch.delenv("DISTLLM_HA_SECRET", raising=False)

    def test_heartbeat_missing_coordinator_id_rejected(self, monkeypatch):
        coord, _ = _ha_coordinator()
        original = g.coordinator
        g.coordinator = coord
        monkeypatch.setenv("DISTLLM_HA_SECRET", "s3cret")
        try:
            resp = TestClient(app).post(
                "/api/v1/ha/heartbeat",
                json={"term": 1},
                headers={"X-HA-Secret": "s3cret"},
            )
            assert resp.status_code == 400
        finally:
            g.coordinator = original
            monkeypatch.delenv("DISTLLM_HA_SECRET", raising=False)

    def test_heartbeat_shared_secret_enforced(self, monkeypatch):
        coord, election = _ha_coordinator()
        original = g.coordinator
        g.coordinator = coord
        monkeypatch.setenv("DISTLLM_HA_SECRET", "s3cret")
        try:
            denied = TestClient(app).post(
                "/api/v1/ha/heartbeat",
                json={"coordinator_id": "a", "term": 1},
            )
            assert denied.status_code == 403

            allowed = TestClient(app).post(
                "/api/v1/ha/heartbeat",
                json={"coordinator_id": "a", "term": 1},
                headers={"X-HA-Secret": "s3cret"},
            )
            assert allowed.status_code == 200
            assert allowed.json()["status"] == "ok"
        finally:
            g.coordinator = original
            monkeypatch.delenv("DISTLLM_HA_SECRET", raising=False)
