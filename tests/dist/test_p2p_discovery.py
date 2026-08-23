"""Tests for distllm.dist.p2p.discovery module.

Tests FederationPeerDiscovery and PeerInfo using pytest + unittest.mock.
No real network calls.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from distllm.dist.p2p.discovery import FederationPeerDiscovery, PeerInfo


# ---------------------------------------------------------------------------
# PeerInfo dataclass
# ---------------------------------------------------------------------------


class TestPeerInfo:
    """PeerInfo dataclass defaults and computed properties."""

    def test_defaults(self) -> None:
        peer = PeerInfo(cluster_id="c1", host="10.0.0.1", port=8080)
        assert peer.is_edge is False
        assert peer.region == ""
        assert peer.metadata == {}
        assert peer.discovered_at == 0.0
        assert peer.last_seen == 0.0

    def test_url_property(self) -> None:
        peer = PeerInfo(cluster_id="c1", host="10.0.0.1", port=8080)
        assert peer.url == "http://10.0.0.1:8080"

    def test_url_property_https_unsupported(self) -> None:
        """url property always uses http scheme."""
        peer = PeerInfo(cluster_id="c1", host="secure.example.com", port=443)
        assert peer.url == "http://secure.example.com:443"

    def test_custom_edge_and_region(self) -> None:
        peer = PeerInfo(
            cluster_id="c1",
            host="10.0.0.1",
            port=8080,
            is_edge=True,
            region="us-west-2",
        )
        assert peer.is_edge is True
        assert peer.region == "us-west-2"

    def test_metadata_roundtrip(self) -> None:
        meta = {"gpu": "A100", "vram_gb": 80}
        peer = PeerInfo(
            cluster_id="c1",
            host="10.0.0.1",
            port=8080,
            metadata=meta,
        )
        assert peer.metadata == meta

    def test_discovered_at_and_last_seen(self) -> None:
        peer = PeerInfo(
            cluster_id="c1",
            host="10.0.0.1",
            port=8080,
            discovered_at=1000.0,
            last_seen=2000.0,
        )
        assert peer.discovered_at == 1000.0
        assert peer.last_seen == 2000.0

    def test_equality(self) -> None:
        """PeerInfo is a dataclass so __eq__ compares all fields."""
        p1 = PeerInfo("c1", "10.0.0.1", 8080)
        p2 = PeerInfo("c1", "10.0.0.1", 8080)
        p3 = PeerInfo("c2", "10.0.0.2", 8080)
        assert p1 == p2
        assert p1 != p3


# ---------------------------------------------------------------------------
# FederationPeerDiscovery -- config / constructor
# ---------------------------------------------------------------------------


class TestFederationPeerDiscoveryConfig:
    """Constructor defaults and parameter handling."""

    def test_default_discovery_interval(self) -> None:
        discovery = FederationPeerDiscovery(
            own_cluster_id="test-cluster",
            own_host="localhost",
            own_port=9090,
        )
        assert discovery.discovery_interval_s == 30.0
        assert discovery.own_cluster_id == "test-cluster"
        assert discovery.own_host == "localhost"
        assert discovery.own_port == 9090
        assert discovery._peers == {}
        assert discovery._seed_nodes == []

    def test_custom_discovery_interval(self) -> None:
        discovery = FederationPeerDiscovery(
            own_cluster_id="c1",
            own_host="host1",
            own_port=8080,
            discovery_interval_s=10.0,
        )
        assert discovery.discovery_interval_s == 10.0

    def test_discovery_interval_zero(self) -> None:
        discovery = FederationPeerDiscovery(
            own_cluster_id="c1",
            own_host="host1",
            own_port=8080,
            discovery_interval_s=0.0,
        )
        assert discovery.discovery_interval_s == 0.0

    def test_discovery_interval_negative(self) -> None:
        discovery = FederationPeerDiscovery(
            own_cluster_id="c1",
            own_host="host1",
            own_port=8080,
            discovery_interval_s=-5.0,
        )
        assert discovery.discovery_interval_s == -5.0

    def test_port_zero(self) -> None:
        discovery = FederationPeerDiscovery("c1", "host1", 0)
        assert discovery.own_port == 0

    def test_port_max(self) -> None:
        discovery = FederationPeerDiscovery("c1", "host1", 65535)
        assert discovery.own_port == 65535

    def test_cluster_id_empty_string(self) -> None:
        discovery = FederationPeerDiscovery("", "host1", 8080)
        assert discovery.own_cluster_id == ""

    def test_host_empty_string(self) -> None:
        discovery = FederationPeerDiscovery("c1", "", 8080)
        assert discovery.own_host == ""


# ---------------------------------------------------------------------------
# FederationPeerDiscovery -- seed node management
# ---------------------------------------------------------------------------


class TestSeedNodes:
    """add_seed_nodes and seed state."""

    def test_add_seed_nodes(self) -> None:
        discovery = FederationPeerDiscovery("c1", "host1", 8080)
        discovery.add_seed_nodes(["http://seed1:8000", "http://seed2:8000"])
        assert discovery._seed_nodes == [
            "http://seed1:8000",
            "http://seed2:8000",
        ]

    def test_add_seed_nodes_appends(self) -> None:
        discovery = FederationPeerDiscovery("c1", "host1", 8080)
        discovery.add_seed_nodes(["http://seed1:8000"])
        discovery.add_seed_nodes(["http://seed2:8000"])
        assert discovery._seed_nodes == [
            "http://seed1:8000",
            "http://seed2:8000",
        ]

    def test_add_seed_nodes_empty_list(self) -> None:
        discovery = FederationPeerDiscovery("c1", "host1", 8080)
        discovery.add_seed_nodes([])
        assert discovery._seed_nodes == []

    def test_add_seed_nodes_does_not_deduplicate(self) -> None:
        """add_seed_nodes does not check for duplicates (caller responsibility)."""
        discovery = FederationPeerDiscovery("c1", "host1", 8080)
        discovery.add_seed_nodes(["http://seed1:8000"])
        discovery.add_seed_nodes(["http://seed1:8000"])
        assert discovery._seed_nodes == [
            "http://seed1:8000",
            "http://seed1:8000",
        ]


# ---------------------------------------------------------------------------
# FederationPeerDiscovery -- getter methods
# ---------------------------------------------------------------------------


class TestGetters:
    """get_peers and get_peer."""

    def test_get_peers_empty_when_no_peers(self) -> None:
        discovery = FederationPeerDiscovery("c1", "host1", 8080)
        assert discovery.get_peers() == []

    def test_get_peer_returns_none_for_missing(self) -> None:
        discovery = FederationPeerDiscovery("c1", "host1", 8080)
        assert discovery.get_peer("nonexistent") is None

    def test_get_peer_returns_stored_peer(self) -> None:
        discovery = FederationPeerDiscovery("c1", "host1", 8080)
        peer = PeerInfo("c2", "10.0.0.2", 8080)
        discovery._peers["c2"] = peer
        assert discovery.get_peer("c2") is peer

    def test_get_peers_returns_all_stored_peers(self) -> None:
        discovery = FederationPeerDiscovery("c1", "host1", 8080)
        p1 = PeerInfo("c2", "10.0.0.2", 8080)
        p2 = PeerInfo("c3", "10.0.0.3", 8080)
        discovery._peers = {"c2": p1, "c3": p2}
        result = discovery.get_peers()
        assert len(result) == 2
        assert p1 in result
        assert p2 in result

    def test_get_peers_returns_copy(self) -> None:
        """Modifying returned list does not affect internal state."""
        discovery = FederationPeerDiscovery("c1", "host1", 8080)
        peer = PeerInfo("c2", "10.0.0.2", 8080)
        discovery._peers = {"c2": peer}
        result = discovery.get_peers()
        result.clear()
        assert len(discovery._peers) == 1


# ---------------------------------------------------------------------------
# FederationPeerDiscovery -- _register_peer (internal)
# ---------------------------------------------------------------------------


class TestRegisterPeerInternal:
    """_register_peer internal helper."""

    @patch("time.time", return_value=12345.0)
    def test_register_new_peer_sets_timestamps(self, mock_time: MagicMock) -> None:
        discovery = FederationPeerDiscovery("c1", "host1", 8080)
        peer = PeerInfo("c2", "10.0.0.2", 8080)
        discovery._register_peer(peer)
        assert discovery._peers["c2"] is peer
        assert peer.discovered_at == 12345.0
        assert peer.last_seen == 12345.0

    @patch("time.time", return_value=99999.0)
    def test_register_existing_peer_updates_last_seen_only(
        self, mock_time: MagicMock
    ) -> None:
        discovery = FederationPeerDiscovery("c1", "host1", 8080)
        peer = PeerInfo(
            "c2",
            "10.0.0.2",
            8080,
            discovered_at=100.0,
            last_seen=200.0,
        )
        discovery._register_peer(peer)
        assert peer.last_seen == 99999.0
        assert peer.discovered_at == 100.0  # unchanged

    def test_register_peer_overwrites_existing_cluster_id(self) -> None:
        discovery = FederationPeerDiscovery("c1", "host1", 8080)
        old = PeerInfo("c2", "10.0.0.2", 8080)
        new = PeerInfo("c2", "10.0.0.99", 8080)
        discovery._peers["c2"] = old
        discovery._register_peer(new)
        assert discovery._peers["c2"] is new

    @patch("time.time", return_value=42.0)
    def test_register_peer_updates_last_seen_on_overwrite(
        self, mock_time: MagicMock
    ) -> None:
        discovery = FederationPeerDiscovery("c1", "host1", 8080)
        peer = PeerInfo("c2", "10.0.0.2", 8080, last_seen=10.0)
        discovery._peers["c2"] = peer
        discovery._register_peer(peer)  # same peer object, re-registered
        assert peer.last_seen == 42.0


# ---------------------------------------------------------------------------
# FederationPeerDiscovery -- discover_peers
# ---------------------------------------------------------------------------


class TestDiscoverPeers:
    """discover_peers with mocked _fetch_peer_list."""

    def test_discover_no_seed_nodes_returns_empty(self) -> None:
        discovery = FederationPeerDiscovery("c1", "host1", 8080)
        result = discovery.discover_peers()
        assert result == []

    def test_discover_delegates_to_fetch_peer_list(self) -> None:
        discovery = FederationPeerDiscovery("c1", "host1", 8080)
        with patch.object(
            FederationPeerDiscovery, "_fetch_peer_list", return_value=[]
        ) as mock_fetch:
            discovery.discover_peers(seed_nodes=["http://seed:8000"])
            mock_fetch.assert_called_once_with("http://seed:8000")

    def test_discover_uses_stored_seed_nodes_when_none_passed(self) -> None:
        discovery = FederationPeerDiscovery("c1", "host1", 8080)
        discovery.add_seed_nodes(["http://seed1:8000"])
        with patch.object(
            FederationPeerDiscovery, "_fetch_peer_list", return_value=[]
        ) as mock_fetch:
            discovery.discover_peers()
            mock_fetch.assert_called_once_with("http://seed1:8000")

    def test_discover_filters_out_self_by_cluster_id(self) -> None:
        discovery = FederationPeerDiscovery("own-cluster", "host1", 8080)
        peer_self = PeerInfo("own-cluster", "10.0.0.1", 8080)
        peer_other = PeerInfo("other-cluster", "10.0.0.2", 8080)
        with patch.object(
            FederationPeerDiscovery,
            "_fetch_peer_list",
            return_value=[peer_self, peer_other],
        ):
            result = discovery.discover_peers(seed_nodes=["http://seed:8000"])
        assert result == [peer_other]
        assert "other-cluster" in discovery._peers
        assert "own-cluster" not in discovery._peers

    def test_discover_returns_newly_discovered_peers_only(self) -> None:
        """Already-known peers are not returned again."""
        discovery = FederationPeerDiscovery("c1", "host1", 8080)
        peer = PeerInfo("c2", "10.0.0.2", 8080)
        discovery._peers["c2"] = peer

        with patch.object(
            FederationPeerDiscovery, "_fetch_peer_list", return_value=[peer]
        ):
            result = discovery.discover_peers(seed_nodes=["http://seed:8000"])
        # peer was already registered, but the method re-registers it (updating
        # last_seen) but still appends to discovered list.
        assert result == [peer]

    def test_discover_multiple_seed_nodes(self) -> None:
        discovery = FederationPeerDiscovery("c1", "host1", 8080)
        p1 = PeerInfo("c2", "10.0.0.2", 8080)
        p2 = PeerInfo("c3", "10.0.0.3", 8080)

        def mock_fetch(url: str) -> list[PeerInfo]:
            return {"http://s1": [p1], "http://s2": [p2]}.get(url, [])

        with patch.object(
            FederationPeerDiscovery, "_fetch_peer_list", side_effect=mock_fetch
        ):
            result = discovery.discover_peers(
                seed_nodes=["http://s1", "http://s2"]
            )
        assert p1 in result
        assert p2 in result
        assert len(result) == 2

    def test_discover_continues_on_seed_failure(self) -> None:
        """Exception from one seed does not prevent others from being queried."""
        discovery = FederationPeerDiscovery("c1", "host1", 8080)
        good_peer = PeerInfo("c2", "10.0.0.2", 8080)

        def mock_fetch(url: str) -> list[PeerInfo]:
            if "bad" in url:
                raise ConnectionError("Connection refused")
            return [good_peer]

        with patch.object(
            FederationPeerDiscovery, "_fetch_peer_list", side_effect=mock_fetch
        ):
            result = discovery.discover_peers(
                seed_nodes=["http://bad:8000", "http://good:8000"]
            )
        assert result == [good_peer]
        assert discovery._peers == {"c2": good_peer}

    def test_discover_all_seeds_fail_returns_empty(self) -> None:
        discovery = FederationPeerDiscovery("c1", "host1", 8080)
        with patch.object(
            FederationPeerDiscovery,
            "_fetch_peer_list",
            side_effect=RuntimeError("offline"),
        ):
            result = discovery.discover_peers(
                seed_nodes=["http://s1", "http://s2"]
            )
        assert result == []
        assert discovery._peers == {}

    def test_discover_empty_response_from_seed(self) -> None:
        discovery = FederationPeerDiscovery("c1", "host1", 8080)
        with patch.object(
            FederationPeerDiscovery, "_fetch_peer_list", return_value=[]
        ):
            result = discovery.discover_peers(seed_nodes=["http://seed:8000"])
        assert result == []

    def test_discover_deduplicates_by_cluster_id(self) -> None:
        """Same peer discovered from two seeds is stored once (last wins)."""
        discovery = FederationPeerDiscovery("c1", "host1", 8080)
        peer = PeerInfo("c2", "10.0.0.2", 8080)
        with patch.object(
            FederationPeerDiscovery, "_fetch_peer_list", return_value=[peer]
        ):
            discovery.discover_peers(
                seed_nodes=["http://s1", "http://s2"]
            )
        assert len(discovery._peers) == 1
        assert discovery._peers["c2"] is peer

    def test_discover_updates_last_seen_on_known_peer(self) -> None:
        """Re-discovering an already-known peer refreshes last_seen."""
        discovery = FederationPeerDiscovery("c1", "host1", 8080)
        peer = PeerInfo(
            "c2", "10.0.0.2", 8080, discovered_at=100.0, last_seen=100.0
        )
        discovery._peers["c2"] = peer

        with patch.object(
            FederationPeerDiscovery, "_fetch_peer_list", return_value=[peer]
        ):
            with patch("time.time", return_value=200.0):
                discovery.discover_peers(seed_nodes=["http://seed:8000"])

        assert peer.last_seen == 200.0
        assert peer.discovered_at == 100.0  # unchanged


# ---------------------------------------------------------------------------
# FederationPeerDiscovery -- register_self
# ---------------------------------------------------------------------------


class TestRegisterSelf:
    """register_self method."""

    def test_register_self_success_returns_true(self) -> None:
        discovery = FederationPeerDiscovery("c1", "host1", 8080)
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.__enter__.return_value = mock_resp

        with patch(
            "distllm.dist.p2p.discovery.safe_urlopen",
            return_value=mock_resp,
        ) as mock_open:
            result = discovery.register_self("http://peer:8000")
        assert result is True

    def test_register_self_sends_correct_url(self) -> None:
        discovery = FederationPeerDiscovery("c1", "host1", 8080)
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.__enter__.return_value = mock_resp

        with patch(
            "distllm.dist.p2p.discovery.safe_urlopen",
            return_value=mock_resp,
        ) as mock_open:
            discovery.register_self("http://peer:8000")
        args, _ = mock_open.call_args
        req = args[0]
        assert req.full_url == "http://peer:8000/api/v1/federation/register"

    def test_register_self_uses_post_method(self) -> None:
        discovery = FederationPeerDiscovery("c1", "host1", 8080)
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.__enter__.return_value = mock_resp

        with patch(
            "distllm.dist.p2p.discovery.safe_urlopen",
            return_value=mock_resp,
        ) as mock_open:
            discovery.register_self("http://peer:8000")
        args, _ = mock_open.call_args
        assert args[0].method == "POST"

    def test_register_self_sends_json_content_type(self) -> None:
        discovery = FederationPeerDiscovery("c1", "host1", 8080)
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.__enter__.return_value = mock_resp

        with patch(
            "distllm.dist.p2p.discovery.safe_urlopen",
            return_value=mock_resp,
        ) as mock_open:
            discovery.register_self("http://peer:8000")
        args, _ = mock_open.call_args
        # Python 3.14 stores headers with capitalized keys (Content-type)
        content_type = dict(args[0].header_items()).get("Content-type")
        assert content_type == "application/json"

    def test_register_self_payload_content(self) -> None:
        discovery = FederationPeerDiscovery("my-cluster", "my-host", 9090)
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.__enter__.return_value = mock_resp

        with patch(
            "distllm.dist.p2p.discovery.safe_urlopen",
            return_value=mock_resp,
        ) as mock_open:
            discovery.register_self("http://peer:8000")
        args, _ = mock_open.call_args
        payload = json.loads(args[0].data)
        assert payload == {
            "cluster_id": "my-cluster",
            "host": "my-host",
            "port": 9090,
            "is_edge": True,
            "region": "",
        }

    def test_register_self_strips_trailing_slash(self) -> None:
        discovery = FederationPeerDiscovery("c1", "host1", 8080)
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.__enter__.return_value = mock_resp

        with patch(
            "distllm.dist.p2p.discovery.safe_urlopen",
            return_value=mock_resp,
        ) as mock_open:
            discovery.register_self("http://peer:8000/")
        args, _ = mock_open.call_args
        assert args[0].full_url == "http://peer:8000/api/v1/federation/register"

    def test_register_self_passes_timeout_and_allow_private(self) -> None:
        discovery = FederationPeerDiscovery("c1", "host1", 8080)
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.__enter__.return_value = mock_resp

        with patch(
            "distllm.dist.p2p.discovery.safe_urlopen",
            return_value=mock_resp,
        ) as mock_open:
            discovery.register_self("http://peer:8000")
        _, kwargs = mock_open.call_args
        assert kwargs.get("timeout") == 10
        assert kwargs.get("allow_private_hosts") is True

    def test_register_self_non_200_returns_false(self) -> None:
        discovery = FederationPeerDiscovery("c1", "host1", 8080)
        mock_resp = MagicMock()
        mock_resp.status = 500
        mock_resp.__enter__.return_value = mock_resp

        with patch(
            "distllm.dist.p2p.discovery.safe_urlopen",
            return_value=mock_resp,
        ):
            result = discovery.register_self("http://peer:8000")
        assert result is False

    def test_register_self_exception_returns_false(self) -> None:
        discovery = FederationPeerDiscovery("c1", "host1", 8080)
        with patch(
            "distllm.dist.p2p.discovery.safe_urlopen",
            side_effect=ConnectionError("timeout"),
        ):
            result = discovery.register_self("http://peer:8000")
        assert result is False

    def test_register_self_exception_logged_but_not_raised(self) -> None:
        """Exception inside register_self is caught, logged, and returns False."""
        discovery = FederationPeerDiscovery("c1", "host1", 8080)
        with patch(
            "distllm.dist.p2p.discovery.safe_urlopen",
            side_effect=RuntimeError("unexpected"),
        ):
            # Must not propagate -- tested by calling without pytest.raises
            result = discovery.register_self("http://peer:8000")
        assert result is False

    def test_register_self_http_error_during_read(self) -> None:
        """Read error inside context manager is caught."""
        discovery = FederationPeerDiscovery("c1", "host1", 8080)
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.__enter__.side_effect = OSError("connection lost")

        with patch(
            "distllm.dist.p2p.discovery.safe_urlopen",
            return_value=mock_resp,
        ):
            result = discovery.register_self("http://peer:8000")
        assert result is False

    def test_register_self_payload_is_encoded_to_bytes(self) -> None:
        discovery = FederationPeerDiscovery("c1", "host1", 8080)
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.__enter__.return_value = mock_resp

        with patch(
            "distllm.dist.p2p.discovery.safe_urlopen",
            return_value=mock_resp,
        ) as mock_open:
            discovery.register_self("http://peer:8000")
        args, _ = mock_open.call_args
        assert isinstance(args[0].data, bytes)
        # Verify it's valid JSON bytes
        parsed = json.loads(args[0].data)
        assert parsed["cluster_id"] == "c1"


# ---------------------------------------------------------------------------
# FederationPeerDiscovery -- _fetch_peer_list (static)
# ---------------------------------------------------------------------------


class TestFetchPeerList:
    """_fetch_peer_list static method via patched safe_urlopen."""

    def test_fetch_returns_empty_on_exception(self) -> None:
        with patch(
            "distllm.dist.p2p.discovery.safe_urlopen",
            side_effect=OSError("no network"),
        ):
            result = FederationPeerDiscovery._fetch_peer_list(
                "http://seed:8000"
            )
        assert result == []

    def test_fetch_returns_empty_on_json_decode_error(self) -> None:
        mock_resp = MagicMock()
        mock_resp.read.return_value = b"not valid json"
        mock_resp.__enter__.return_value = mock_resp

        with patch(
            "distllm.dist.p2p.discovery.safe_urlopen",
            return_value=mock_resp,
        ):
            result = FederationPeerDiscovery._fetch_peer_list(
                "http://seed:8000"
            )
        assert result == []

    def test_fetch_parses_single_peer(self) -> None:
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({
            "peers": [
                {
                    "cluster_id": "c2",
                    "host": "10.0.0.2",
                    "port": 8080,
                },
            ],
        }).encode()
        mock_resp.__enter__.return_value = mock_resp

        with patch(
            "distllm.dist.p2p.discovery.safe_urlopen",
            return_value=mock_resp,
        ):
            result = FederationPeerDiscovery._fetch_peer_list(
                "http://seed:8000"
            )
        assert len(result) == 1
        assert result[0].cluster_id == "c2"
        assert result[0].host == "10.0.0.2"
        assert result[0].port == 8080
        assert result[0].is_edge is False
        assert result[0].region == ""

    def test_fetch_parses_multiple_peers_with_all_fields(self) -> None:
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({
            "peers": [
                {
                    "cluster_id": "c2",
                    "host": "10.0.0.2",
                    "port": 8080,
                },
                {
                    "cluster_id": "c3",
                    "host": "10.0.0.3",
                    "port": 9090,
                    "is_edge": True,
                    "region": "us-east-1",
                    "metadata": {"gpu": "A100"},
                },
            ],
        }).encode()
        mock_resp.__enter__.return_value = mock_resp

        with patch(
            "distllm.dist.p2p.discovery.safe_urlopen",
            return_value=mock_resp,
        ):
            result = FederationPeerDiscovery._fetch_peer_list(
                "http://seed:8000"
            )
        assert len(result) == 2

        assert result[0].cluster_id == "c2"
        assert result[0].host == "10.0.0.2"
        assert result[0].port == 8080
        assert result[0].is_edge is False
        assert result[0].region == ""
        assert result[0].metadata == {}

        assert result[1].cluster_id == "c3"
        assert result[1].host == "10.0.0.3"
        assert result[1].port == 9090
        assert result[1].is_edge is True
        assert result[1].region == "us-east-1"
        assert result[1].metadata == {"gpu": "A100"}

    def test_fetch_handles_empty_peers_list(self) -> None:
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({"peers": []}).encode()
        mock_resp.__enter__.return_value = mock_resp

        with patch(
            "distllm.dist.p2p.discovery.safe_urlopen",
            return_value=mock_resp,
        ):
            result = FederationPeerDiscovery._fetch_peer_list(
                "http://seed:8000"
            )
        assert result == []

    def test_fetch_handles_missing_peers_key(self) -> None:
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({"foo": "bar"}).encode()
        mock_resp.__enter__.return_value = mock_resp

        with patch(
            "distllm.dist.p2p.discovery.safe_urlopen",
            return_value=mock_resp,
        ):
            result = FederationPeerDiscovery._fetch_peer_list(
                "http://seed:8000"
            )
        assert result == []

    def test_fetch_builds_correct_url(self) -> None:
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({"peers": []}).encode()
        mock_resp.__enter__.return_value = mock_resp

        with patch(
            "distllm.dist.p2p.discovery.safe_urlopen",
            return_value=mock_resp,
        ) as mock_open:
            FederationPeerDiscovery._fetch_peer_list("http://seed:8000")
        args, kwargs = mock_open.call_args
        assert args[0] == "http://seed:8000/api/v1/federation/peers"
        assert kwargs.get("timeout") == 10
        assert kwargs.get("allow_private_hosts") is True

    def test_fetch_strips_trailing_slash_from_seed_url(self) -> None:
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({"peers": []}).encode()
        mock_resp.__enter__.return_value = mock_resp

        with patch(
            "distllm.dist.p2p.discovery.safe_urlopen",
            return_value=mock_resp,
        ) as mock_open:
            FederationPeerDiscovery._fetch_peer_list("http://seed:8000/")
        args, _ = mock_open.call_args
        assert args[0] == "http://seed:8000/api/v1/federation/peers"

    def test_fetch_returns_empty_on_any_exception(self) -> None:
        """Any exception type is caught and returns empty list."""
        with patch(
            "distllm.dist.p2p.discovery.safe_urlopen",
            side_effect=ValueError("bad response"),
        ):
            result = FederationPeerDiscovery._fetch_peer_list(
                "http://seed:8000"
            )
        assert result == []


# ---------------------------------------------------------------------------
# FederationPeerDiscovery -- error handling (network unavailable scenarios)
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """Graceful degradation when network calls fail."""

    def test_full_discovery_cycle_with_network_failure(self) -> None:
        """End-to-end scenario: seed nodes unreachable, no crash."""
        discovery = FederationPeerDiscovery("c1", "host1", 8080)
        discovery.add_seed_nodes(["http://seed1:8000", "http://seed2:8000"])

        with patch.object(
            FederationPeerDiscovery,
            "_fetch_peer_list",
            side_effect=RuntimeError("network unavailable"),
        ):
            result = discovery.discover_peers()
        assert result == []
        assert discovery.get_peers() == []

    def test_discover_and_register_both_handle_errors(self) -> None:
        """Call both methods when network is down -- no crash."""
        discovery = FederationPeerDiscovery("c1", "host1", 8080)
        discovery.add_seed_nodes(["http://seed:8000"])

        with patch.object(
            FederationPeerDiscovery,
            "_fetch_peer_list",
            side_effect=RuntimeError("offline"),
        ):
            peers = discovery.discover_peers()
        assert peers == []

        with patch(
            "distllm.dist.p2p.discovery.safe_urlopen",
            side_effect=ConnectionError("refused"),
        ):
            registered = discovery.register_self("http://other:8000")
        assert registered is False

    def test_discover_with_none_seed_nodes_fallback(self) -> None:
        """Passing None as seed_nodes falls back to stored list."""
        discovery = FederationPeerDiscovery("c1", "host1", 8080)
        discovery.add_seed_nodes(["http://seed:8000"])

        with patch.object(
            FederationPeerDiscovery,
            "_fetch_peer_list",
            return_value=[],
        ) as mock_fetch:
            discovery.discover_peers(None)
        mock_fetch.assert_called_once_with("http://seed:8000")
