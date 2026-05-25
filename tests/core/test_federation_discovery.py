"""Tests for FederationPeerDiscovery: discover peer clusters via seed nodes."""

import json
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

import pytest

from distllm.core.federation_discovery import FederationPeerDiscovery, PeerInfo


class MockFederationHandler(BaseHTTPRequestHandler):
    """Simulates a federation seed coordinator."""

    peers_data: dict = {}
    registered: list[dict] = []

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/api/v1/federation/peers":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(self.peers_data).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length > 0 else b"{}"
        data = json.loads(body)
        MockFederationHandler.registered.append(data)

        path = urlparse(self.path).path
        if path == "/api/v1/federation/register":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"ok": True}).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, fmt, *args):
        pass


@pytest.fixture
def seed_server():
    MockFederationHandler.peers_data = {}
    MockFederationHandler.registered = []

    server = HTTPServer(("127.0.0.1", 0), MockFederationHandler)
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    yield port, f"http://127.0.0.1:{port}"
    server.shutdown()


class TestPeerInfo:
    def test_defaults(self):
        p = PeerInfo(cluster_id="c1", host="host1", port=8000)
        assert p.cluster_id == "c1"
        assert p.host == "host1"
        assert p.port == 8000
        assert p.is_edge is False
        assert p.region == ""
        assert p.metadata == {}

    def test_url_property(self):
        p = PeerInfo(cluster_id="c1", host="10.0.0.1", port=9000)
        assert p.url == "http://10.0.0.1:9000"

    def test_url_edge(self):
        p = PeerInfo(cluster_id="c2", host="edge1", port=8080, is_edge=True)
        assert p.url == "http://edge1:8080"

    def test_custom_metadata(self):
        p = PeerInfo(
            cluster_id="c3",
            host="h3",
            port=7000,
            region="us-east",
            metadata={"gpu": "A100", "pool": "prod"},
        )
        assert p.region == "us-east"
        assert p.metadata["gpu"] == "A100"

    def test_discovered_at_set_on_register(self):
        p = PeerInfo(cluster_id="c1", host="h", port=1)
        assert p.discovered_at == 0.0
        assert p.last_seen == 0.0


class TestFederationPeerDiscovery:
    def test_init(self):
        fd = FederationPeerDiscovery("my-cluster", "my-host", 5000)
        assert fd.own_cluster_id == "my-cluster"
        assert fd.own_host == "my-host"
        assert fd.own_port == 5000
        assert fd.get_peers() == []

    def test_add_seed_nodes(self):
        fd = FederationPeerDiscovery("c1", "h1", 1)
        fd.add_seed_nodes(["http://seed1:8000", "http://seed2:8000"])
        assert len(fd._seed_nodes) == 2

    def test_discover_peers_from_seed(self, seed_server):
        port, url = seed_server
        MockFederationHandler.peers_data = {
            "peers": [
                {"cluster_id": "cluster-b", "host": "10.0.0.2", "port": 8000},
                {"cluster_id": "cluster-c", "host": "10.0.0.3", "port": 8001},
            ]
        }

        fd = FederationPeerDiscovery("my-cluster", "my-host", 5000)
        peers = fd.discover_peers(seed_nodes=[url])

        assert len(peers) == 2
        ids = {p.cluster_id for p in peers}
        assert ids == {"cluster-b", "cluster-c"}

    def test_discover_excludes_self(self, seed_server):
        port, url = seed_server
        MockFederationHandler.peers_data = {
            "peers": [
                {"cluster_id": "my-cluster", "host": "my-host", "port": 5000},
                {"cluster_id": "cluster-b", "host": "10.0.0.2", "port": 8000},
            ]
        }

        fd = FederationPeerDiscovery("my-cluster", "my-host", 5000)
        peers = fd.discover_peers(seed_nodes=[url])
        ids = {p.cluster_id for p in peers}
        assert ids == {"cluster-b"}

    def test_discover_multiple_seeds(self, seed_server):
        port, url = seed_server
        MockFederationHandler.peers_data = {
            "peers": [
                {"cluster_id": "cluster-b", "host": "10.0.0.2", "port": 8000},
            ]
        }

        fd = FederationPeerDiscovery("my-cluster", "my-host", 5000)
        peers = fd.discover_peers(seed_nodes=[url, url])
        ids = {p.cluster_id for p in peers}
        assert ids == {"cluster-b"}

    def test_discover_from_failed_seed(self):
        fd = FederationPeerDiscovery("c1", "h1", 1)
        peers = fd.discover_peers(seed_nodes=["http://127.0.0.1:1"])
        assert peers == []

    def test_register_self_success(self, seed_server):
        port, url = seed_server
        fd = FederationPeerDiscovery("my-cluster", "my-host", 5000)
        result = fd.register_self(url)
        assert result is True
        assert len(MockFederationHandler.registered) == 1
        assert MockFederationHandler.registered[0]["cluster_id"] == "my-cluster"

    def test_register_self_failure(self):
        fd = FederationPeerDiscovery("c1", "h1", 1)
        result = fd.register_self("http://127.0.0.1:1")
        assert result is False

    def test_get_peer(self):
        fd = FederationPeerDiscovery("c1", "h1", 1)
        fd._peers["peer-a"] = PeerInfo(cluster_id="peer-a", host="h2", port=8000)
        p = fd.get_peer("peer-a")
        assert p is not None
        assert p.host == "h2"
        assert fd.get_peer("nonexistent") is None

    def test_discover_peers_updates_last_seen(self, seed_server):
        port, url = seed_server
        MockFederationHandler.peers_data = {
            "peers": [
                {"cluster_id": "cluster-b", "host": "10.0.0.2", "port": 8000},
            ]
        }

        fd = FederationPeerDiscovery("my-cluster", "my-host", 5000)
        peers = fd.discover_peers(seed_nodes=[url])
        assert peers[0].last_seen > 0
        assert peers[0].discovered_at > 0

    def test_discover_called_twice_updates_timestamp(self, seed_server):
        port, url = seed_server
        MockFederationHandler.peers_data = {
            "peers": [
                {"cluster_id": "cluster-b", "host": "10.0.0.2", "port": 8000},
            ]
        }

        fd = FederationPeerDiscovery("my-cluster", "my-host", 5000)
        peers1 = fd.discover_peers(seed_nodes=[url])
        ts1 = peers1[0].discovered_at

        peers2 = fd.discover_peers(seed_nodes=[url])
        assert len(peers2) == 1
        assert peers2[0].cluster_id == "cluster-b"
        # discovered_at should be close to first registration time
        assert abs(peers2[0].discovered_at - ts1) < 1.0

    def test_peer_url_strips_trailing_slash(self, seed_server):
        port, url = seed_server
        fd = FederationPeerDiscovery("c1", "h1", 1)
        result = fd.register_self(f"{url}/")
        assert result is True
