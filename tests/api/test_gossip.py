"""Gossip API route tests: POST /api/v1/gossip/exchange, /api/v1/gossip/fetch."""

import secrets
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from distllm.api.api_state import g
from distllm.api.server import app
from distllm.core.api_key_store import reset_api_key_store

_TEST_API_KEY = secrets.token_hex(32)


@pytest.fixture(autouse=True)
def _setup_auth(monkeypatch):
    monkeypatch.delenv("API_KEY_WAS_SET", raising=False)
    monkeypatch.setenv("API_KEY", _TEST_API_KEY)
    reset_api_key_store()


@pytest.fixture
def client():
    c = TestClient(app)
    c.headers["Authorization"] = f"Bearer {_TEST_API_KEY}"
    return c


@pytest.fixture
def coord():
    c = MagicMock()
    c._shutting_down = False
    g.coordinator = c
    return c


class TestExchangeGossip:
    @pytest.fixture(autouse=True)
    def setup(self, coord):
        original = g.coordinator
        g.coordinator = coord
        yield
        g.coordinator = original

    def test_exchange_no_coordinator(self, client):
        original = g.coordinator
        g.coordinator = None
        resp = client.post("/api/v1/gossip/exchange", json={"node_id": "peer-1"})
        g.coordinator = original
        assert resp.status_code == 503

    def test_exchange_no_gossip_protocol(self, coord, client):
        coord._gossip_protocol = None
        resp = client.post("/api/v1/gossip/exchange", json={"node_id": "peer-1"})
        assert resp.status_code == 503

    def test_exchange_success(self, coord, client):
        proto = MagicMock()
        proto.advertise.return_value = {"node_id": "local", "entries": 5}
        proto.verify_message.return_value = True
        coord._gossip_protocol = proto

        resp = client.post("/api/v1/gossip/exchange", json={"node_id": "peer-1", "_hmac": "validhmac"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["node_id"] == "local"
        assert data["entries"] == 5
        proto.process_advertisement.assert_called_once()

    def test_exchange_invalid_hmac(self, coord, client):
        proto = MagicMock()
        proto.verify_message.return_value = False
        coord._gossip_protocol = proto

        resp = client.post(
            "/api/v1/gossip/exchange",
            json={"node_id": "peer-1", "_hmac": "badhmac"},
        )
        assert resp.status_code == 403

    def test_exchange_unsigned_peer_rejected(self, coord, client):
        """Unsigned messages are now rejected with 403 (HMAC required)."""
        proto = MagicMock()
        proto.advertise.return_value = {"node_id": "local"}
        coord._gossip_protocol = proto

        resp = client.post("/api/v1/gossip/exchange", json={"node_id": "peer-1"})
        assert resp.status_code == 403


class TestFetchGossip:
    @pytest.fixture(autouse=True)
    def setup(self, coord):
        original = g.coordinator
        g.coordinator = coord
        yield
        g.coordinator = original

    def test_fetch_no_coordinator(self, client):
        original = g.coordinator
        g.coordinator = None
        resp = client.post("/api/v1/gossip/fetch", json={"prefix_hashes": ["abc"]})
        g.coordinator = original
        assert resp.status_code == 503

    def test_fetch_no_gossip_protocol(self, coord, client):
        coord._gossip_protocol = None
        resp = client.post("/api/v1/gossip/fetch", json={"prefix_hashes": ["abc"]})
        assert resp.status_code == 503

    def test_fetch_success(self, coord, client):
        proto = MagicMock()
        proto.state.local_entries = {
            "abc": {"node_id": "peer-1", "seq": 1},
            "def": {"node_id": "peer-1", "seq": 2},
        }
        coord._gossip_protocol = proto

        resp = client.post(
            "/api/v1/gossip/fetch",
            json={"prefix_hashes": ["abc", "def", "xyz"]},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["entries_returned"] == 2
        assert "abc" in data["cache_entries"]
        assert "def" in data["cache_entries"]
        assert "xyz" not in data["cache_entries"]

    def test_fetch_empty_hashes(self, coord, client):
        proto = MagicMock()
        proto.state.local_entries = {"abc": {"node_id": "peer-1"}}
        coord._gossip_protocol = proto

        resp = client.post("/api/v1/gossip/fetch", json={"prefix_hashes": []})
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["entries_returned"] == 0

    def test_fetch_requester_id(self, coord, client):
        proto = MagicMock()
        proto.state.local_entries = {}
        coord._gossip_protocol = proto

        resp = client.post(
            "/api/v1/gossip/fetch",
            json={"requester_id": "node-x", "prefix_hashes": ["abc"]},
        )
        assert resp.status_code == 200
        assert resp.json()["entries_returned"] == 0

    def test_fetch_no_match(self, coord, client):
        proto = MagicMock()
        proto.state.local_entries = {"abc": {"node_id": "peer-1"}}
        coord._gossip_protocol = proto

        resp = client.post(
            "/api/v1/gossip/fetch",
            json={"prefix_hashes": ["nonexistent"]},
        )
        assert resp.status_code == 200
        assert resp.json()["entries_returned"] == 0
