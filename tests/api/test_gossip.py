"""Gossip API route tests: POST /api/v1/gossip/exchange, /api/v1/gossip/fetch."""

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from distllm.api.api_state import g
from distllm.api.server import app


@pytest.fixture(autouse=True)
def _disable_auth(monkeypatch):
    monkeypatch.setenv("DISABLE_AUTH", "1")
    monkeypatch.setenv("DISTLLM_DEV_MODE", "1")
    monkeypatch.delenv("API_KEY", raising=False)


@pytest.fixture
def client():
    return TestClient(app)


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
        coord._gossip_protocol = proto

        resp = client.post("/api/v1/gossip/exchange", json={"node_id": "peer-1"})
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

    def test_exchange_unsigned_peer_accepted(self, coord, client):
        proto = MagicMock()
        proto.advertise.return_value = {"node_id": "local"}
        coord._gossip_protocol = proto

        resp = client.post("/api/v1/gossip/exchange", json={"node_id": "peer-1"})
        assert resp.status_code == 200
        assert resp.json()["node_id"] == "local"


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
