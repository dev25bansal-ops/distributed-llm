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

    @staticmethod
    def _make_mock_proto(local_entries=None):
        """Mock protocol pre-configured for the fetch endpoint contract."""
        proto = MagicMock()
        proto.state.local_entries = local_entries or {}
        # Default: authorization passes (legacy mode semantics)
        proto.authorize_fetch_request.return_value = (True, "")
        proto.has_shared_hmac_key = False
        return proto

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
        proto = self._make_mock_proto({
            "abc": {"node_id": "peer-1", "seq": 1},
            "def": {"node_id": "peer-1", "seq": 2},
        })
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
        proto = self._make_mock_proto({"abc": {"node_id": "peer-1"}})
        coord._gossip_protocol = proto

        resp = client.post("/api/v1/gossip/fetch", json={"prefix_hashes": []})
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["entries_returned"] == 0

    def test_fetch_requester_id(self, coord, client):
        proto = self._make_mock_proto()
        coord._gossip_protocol = proto

        resp = client.post(
            "/api/v1/gossip/fetch",
            json={"requester_id": "node-x", "prefix_hashes": ["abc"]},
        )
        assert resp.status_code == 200
        assert resp.json()["entries_returned"] == 0

    def test_fetch_no_match(self, coord, client):
        proto = self._make_mock_proto({"abc": {"node_id": "peer-1"}})
        coord._gossip_protocol = proto

        resp = client.post(
            "/api/v1/gossip/fetch",
            json={"prefix_hashes": ["nonexistent"]},
        )
        assert resp.status_code == 200
        assert resp.json()["entries_returned"] == 0


class TestFetchGossipAuth:
    """HMAC enforcement on POST /api/v1/gossip/fetch (Wave 2 item 3).

    Uses a REAL GossipProtocol (not a mock) so signatures round-trip
    through the actual HMAC machinery.
    """

    SHARED_KEY = "wave2-gossip-fetch-shared-secret-0123456789abcdef"

    @pytest.fixture(autouse=True)
    def setup(self, coord, monkeypatch, tmp_path):
        # Isolate the persistent-key fallback from the developer's machine
        monkeypatch.setenv(
            "DISTLLM_GOSSIP_KEY_FILE", str(tmp_path / "gossip_hmac.key")
        )
        original = g.coordinator
        g.coordinator = coord
        yield
        g.coordinator = original

    @staticmethod
    def _fetch_url():
        return "/api/v1/gossip/fetch"

    def _signed_body(self, protocol, requester_id="peer-b", hashes=None):
        hashes = hashes if hashes is not None else ["abc"]
        wire = {"requester_id": requester_id, "prefix_hashes": hashes}
        return protocol.sign_fetch_request(wire)

    def test_signed_request_accepted_end_to_end(self, coord, client):
        """A correctly-signed fetch against a shared-key protocol returns 200
        with the requested entries and a signed response."""
        from distllm.dist.p2p.gossip import GossipProtocol

        proto = GossipProtocol(node_id="node-a", hmac_key=self.SHARED_KEY)
        proto.store_local("abc", "ref-abc")
        coord._gossip_protocol = proto

        resp = client.post(self._fetch_url(), json=self._signed_body(proto))
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["cache_entries"]["abc"] == "ref-abc"
        # Response is signed so the requester can authenticate it
        assert "_hmac" in data
        unsigned = {k: v for k, v in data.items() if k != "_hmac"}
        import hashlib
        import hmac as hmac_mod
        import json

        serialized = json.dumps(
            unsigned, separators=(",", ":"), sort_keys=True
        ).encode("utf-8")
        expected = hmac_mod.new(
            self.SHARED_KEY.encode(), msg=serialized, digestmod=hashlib.sha256
        ).hexdigest()
        assert hmac_mod.compare_digest(data["_hmac"], expected)

    def test_tampered_signature_rejected(self, coord, client):
        """A valid signature over modified content is rejected with 403."""
        from distllm.dist.p2p.gossip import GossipProtocol

        proto = GossipProtocol(node_id="node-a", hmac_key=self.SHARED_KEY)
        proto.store_local("abc", "ref-abc")
        coord._gossip_protocol = proto

        body = self._signed_body(proto, hashes=["abc"])
        body["prefix_hashes"] = ["secret-entry-not-mine"]  # tamper after signing
        resp = client.post(self._fetch_url(), json=body)
        assert resp.status_code == 403
        assert "invalid_signature" in resp.json()["error"]["message"]

    def test_unsigned_request_rejected_when_secret_configured(self, coord, client):
        """No _hmac field -> 403 (fail closed under a shared key)."""
        from distllm.dist.p2p.gossip import GossipProtocol

        proto = GossipProtocol(node_id="node-a", hmac_key=self.SHARED_KEY)
        proto.store_local("abc", "ref-abc")
        coord._gossip_protocol = proto

        resp = client.post(
            self._fetch_url(),
            json={"requester_id": "peer-b", "prefix_hashes": ["abc"]},
        )
        assert resp.status_code == 403
        assert "missing_signature" in resp.json()["error"]["message"]

    def test_wrong_key_signature_rejected(self, coord, client):
        """A signature produced under a different secret is rejected."""
        from distllm.dist.p2p.gossip import GossipProtocol

        proto = GossipProtocol(node_id="node-a", hmac_key=self.SHARED_KEY)
        coord._gossip_protocol = proto

        attacker = GossipProtocol(node_id="attacker", hmac_key="not-the-shared-key")
        body = self._signed_body(attacker)
        resp = client.post(self._fetch_url(), json=body)
        assert resp.status_code == 403

    def test_non_string_signature_rejected(self, coord, client):
        """A hostile non-string _hmac must 403, not crash with TypeError."""
        from distllm.dist.p2p.gossip import GossipProtocol

        proto = GossipProtocol(node_id="node-a", hmac_key=self.SHARED_KEY)
        coord._gossip_protocol = proto

        resp = client.post(
            self._fetch_url(),
            json={
                "requester_id": "peer-b",
                "prefix_hashes": ["abc"],
                "_hmac": {"evil": "dict"},
            },
        )
        # Pydantic coerces/rejects the dict against the str-typed field;
        # either way it must not reach compare_digest as a non-str.
        assert resp.status_code in (403, 422)

    def test_legacy_mode_serves_and_warns_once(self, coord, client, tmp_path):
        """Without a shared secret the fetch is served (backward compat) and
        a loud warning is logged exactly once (kademlia_dht convention)."""
        from loguru import logger

        from distllm.dist.p2p.gossip import GossipProtocol

        # Force the node-local persistent-key path (no shared key).
        proto = GossipProtocol(node_id="legacy-node", hmac_key="")
        assert proto.has_shared_hmac_key is False
        proto.store_local("abc", "ref-abc")
        coord._gossip_protocol = proto

        records = []

        def _sink(message):
            records.append(message.record)

        handler_id = logger.add(_sink, level="WARNING")
        try:
            resp = client.post(
                self._fetch_url(),
                json={"requester_id": "peer-b", "prefix_hashes": ["abc"]},
            )
        finally:
            logger.remove(handler_id)

        assert resp.status_code == 200
        assert resp.json()["cache_entries"]["abc"] == "ref-abc"
        warnings = [
            r for r in records if "KV fetch" in r["message"] or "lookup requests" in r["message"]
        ]
        assert len(warnings) >= 1

        # Second request must NOT re-log (one-time warning).
        records.clear()
        handler_id = logger.add(_sink, level="WARNING")
        try:
            resp2 = client.post(
                self._fetch_url(),
                json={"requester_id": "peer-b", "prefix_hashes": ["abc"]},
            )
        finally:
            logger.remove(handler_id)
        assert resp2.status_code == 200
        warnings2 = [
            r for r in records if "KV fetch" in r["message"] or "lookup requests" in r["message"]
        ]
        assert len(warnings2) == 0
