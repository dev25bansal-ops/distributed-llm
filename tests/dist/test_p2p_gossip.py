"""Tests for distllm.dist.p2p.gossip — fully mocked, edge-case focused.

Covers GossipProtocol.__init__, sign_message/verify_message,
add_peer/remove_peer, store_local/lookup, advertise/process_advertisement,
has_changes_since, request_cache_from_peers, tombstone_entry,
cleanup_expired, VectorClock, and LWWRegister.

Uses pytest, unittest.mock, and fixtures throughout. No integration tests.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor as RealThreadPoolExecutor
from unittest.mock import MagicMock, call, patch

import pytest
from loguru import logger

from distllm.dist.p2p.gossip import (
    GossipProtocol,
    LWWRegister,
    VectorClock,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HMAC_KEY = "0123456789abcdef0123456789abcdef"
FIXED_TIME = 1_000_000.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_hmac(body: dict, key: str = HMAC_KEY) -> str:
    """Return the HMAC-SHA256 hex digest *body* would produce when signed."""
    serialized = json.dumps(body, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hmac.new(key.encode(), msg=serialized, digestmod=hashlib.sha256).hexdigest()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_time():
    """Fix time.time to a known value."""
    with patch("time.time", return_value=FIXED_TIME) as mt:
        yield mt


@pytest.fixture
def gp(mock_time):
    """GossipProtocol with fixed time and known HMAC key."""
    return GossipProtocol(node_id="test-node", hmac_key=HMAC_KEY)


@pytest.fixture
def gp_peer():
    """Another GossipProtocol for peer interactions (no mock_time needed)."""
    return GossipProtocol(node_id="peer-node", hmac_key=HMAC_KEY)


# =========================================================================
# 1. GossipProtocol.__init__ — HMAC key loading, fallback, error
# =========================================================================

class TestInit:
    """HMAC key loading, fallback paths, and error conditions."""

    def test_init_with_hmac_key(self):
        """Explicit hmac_key is used directly."""
        p = GossipProtocol(node_id="n1", hmac_key=HMAC_KEY)
        assert p.state.node_id == "n1"
        assert p._hmac_key == HMAC_KEY

    def test_init_from_env_var(self):
        """Key from DISTLLM_GOSSIP_HMAC_KEY env var is used."""
        with patch.dict(os.environ, {"DISTLLM_GOSSIP_HMAC_KEY": HMAC_KEY}, clear=True):
            p = GossipProtocol(node_id="n1")
            assert p._hmac_key == HMAC_KEY

    def test_init_env_var_overrides_persistent(self):
        """Env var takes precedence over a persistent key file."""
        with patch.dict(os.environ, {"DISTLLM_GOSSIP_HMAC_KEY": "from-env"}, clear=True):
            with patch("distllm.dist.p2p.gossip.os.path.exists", return_value=True):
                mock_file = MagicMock()
                mock_file.read.return_value = "from-file"
                mock_file.__enter__.return_value = mock_file
                with patch("builtins.open", return_value=mock_file):
                    p = GossipProtocol(node_id="n1")
                    # Env var wins — no load_or_create is called when key is set
                    assert p._hmac_key == "from-env"

    def test_init_raises_when_no_key_and_no_fallback(self):
        """ValueError when no key, no env, and no insecure override."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="GossipProtocol requires a per-deployment HMAC key"):
                GossipProtocol(node_id="n1")

    def test_init_fallback_with_insecure_env_var(self):
        """DISTLLM_ALLOW_INSECURE_GOSSIP_KEY=1 triggers persistent key fallback."""
        with patch.dict(os.environ, {"DISTLLM_ALLOW_INSECURE_GOSSIP_KEY": "1"}, clear=True):
            with patch("distllm.dist.p2p.gossip.secrets.token_urlsafe", return_value="persistent-key"):
                with patch("distllm.dist.p2p.gossip.os.path.exists", return_value=False):
                    with patch("distllm.dist.p2p.gossip.os.makedirs"):
                        with patch("builtins.open", new_callable=MagicMock):
                            p = GossipProtocol(node_id="n1")
                            assert p._hmac_key == "persistent-key"

    def test_init_fallback_dev_mode(self):
        """DISTLLM_DEV_MODE=1 triggers persistent key fallback."""
        with patch.dict(os.environ, {"DISTLLM_DEV_MODE": "1"}, clear=True):
            with patch("distllm.dist.p2p.gossip.secrets.token_urlsafe", return_value="dev-key"):
                with patch("distllm.dist.p2p.gossip.os.path.exists", return_value=False):
                    with patch("distllm.dist.p2p.gossip.os.makedirs"):
                        with patch("builtins.open", new_callable=MagicMock):
                            p = GossipProtocol(node_id="n1")
                            assert p._hmac_key == "dev-key"

    def test_init_fallback_during_pytest(self):
        """PYTEST_CURRENT_TEST being set triggers persistent key fallback."""
        with patch.dict(os.environ, {"PYTEST_CURRENT_TEST": "test_x.py::test_y"}, clear=True):
            with patch("distllm.dist.p2p.gossip.secrets.token_urlsafe", return_value="pytest-key"):
                with patch("distllm.dist.p2p.gossip.os.path.exists", return_value=False):
                    with patch("distllm.dist.p2p.gossip.os.makedirs"):
                        with patch("builtins.open", new_callable=MagicMock):
                            p = GossipProtocol(node_id="n1")
                            assert p._hmac_key == "pytest-key"

    def test_persistent_key_loads_existing(self):
        """Existing key file is loaded and returned."""
        with patch.dict(os.environ, {"DISTLLM_DEV_MODE": "1"}, clear=True):
            with patch("distllm.dist.p2p.gossip.os.path.exists", return_value=True):
                mock_file = MagicMock()
                mock_file.read.return_value = "stored-key\n"
                mock_file.__enter__.return_value = mock_file
                with patch("builtins.open", return_value=mock_file):
                    p = GossipProtocol(node_id="n1")
                    assert p._hmac_key == "stored-key"

    def test_persistent_key_empty_file_generates_new(self):
        """Empty key file triggers new key generation."""
        with patch.dict(os.environ, {"DISTLLM_DEV_MODE": "1"}, clear=True):
            with patch("distllm.dist.p2p.gossip.os.path.exists", return_value=True):
                read_file = MagicMock()
                read_file.read.return_value = ""
                read_file.__enter__.return_value = read_file
                # First open() for read, second open() for write
                write_file = MagicMock()
                with patch("builtins.open", side_effect=[read_file, write_file]):
                    with patch("distllm.dist.p2p.gossip.os.path.exists", return_value=True):
                        with patch("distllm.dist.p2p.gossip.secrets.token_urlsafe", return_value="generated-key"):
                            with patch("distllm.dist.p2p.gossip.os.makedirs"):
                                p = GossipProtocol(node_id="n1")
                                assert p._hmac_key == "generated-key"

    def test_persistent_key_io_error_on_read_falls_back(self):
        """OSError on read triggers new key generation."""
        with patch.dict(os.environ, {"DISTLLM_DEV_MODE": "1"}, clear=True):
            with patch("distllm.dist.p2p.gossip.os.path.exists", return_value=True):
                read_file = MagicMock()
                read_file.read.side_effect = OSError("read error")
                read_file.__enter__.return_value = read_file
                write_file = MagicMock()
                with patch("builtins.open", side_effect=[read_file, write_file]):
                    with patch("distllm.dist.p2p.gossip.secrets.token_urlsafe", return_value="fallback-key"):
                        with patch("distllm.dist.p2p.gossip.os.makedirs"):
                            p = GossipProtocol(node_id="n1")
                            assert p._hmac_key == "fallback-key"

    def test_persistent_key_io_error_on_write_does_not_crash(self):
        """OSError on write is logged but does not crash init."""
        with patch.dict(os.environ, {"DISTLLM_DEV_MODE": "1"}, clear=True):
            with patch("distllm.dist.p2p.gossip.os.path.exists", return_value=False):
                with patch("distllm.dist.p2p.gossip.secrets.token_urlsafe", return_value="gen-key"):
                    with patch("distllm.dist.p2p.gossip.os.makedirs", side_effect=OSError("mkdir error")):
                        p = GossipProtocol(node_id="n1")
                        assert p._hmac_key == "gen-key"

    def test_default_params(self):
        """Default max_peers=16 and cache_ttl=300.0."""
        p = GossipProtocol(node_id="n1", hmac_key=HMAC_KEY)
        assert p.max_peers == 16
        assert p.cache_ttl == 300.0

    def test_custom_params(self):
        """Custom max_peers and cache_ttl are respected."""
        p = GossipProtocol(node_id="n1", max_peers=4, cache_ttl=60.0, hmac_key=HMAC_KEY)
        assert p.max_peers == 4
        assert p.cache_ttl == 60.0


# =========================================================================
# 2. sign_message / verify_message
# =========================================================================

class TestHMAC:
    """HMAC signing roundtrip for gossip messages."""

    def test_sign_adds_hmac_field(self, gp):
        """sign_message injects the _hmac key."""
        msg = {"node_id": "test-node", "cache_prefixes": ["abc"]}
        signed = gp.sign_message(msg)
        assert "_hmac" in signed
        assert len(signed["_hmac"]) > 0

    def test_sign_does_not_mutate_input(self, gp):
        """The original dict is not modified."""
        original = {"a": 1}
        gp.sign_message(original)
        assert "_hmac" not in original

    def test_sign_and_verify_roundtrip(self, gp):
        """sign_message followed by verify_message succeeds."""
        signed = gp.sign_message({"data": "hello"})
        assert gp.verify_message(signed) is True

    def test_verify_no_hmac_field(self, gp):
        """Message without _hmac returns False."""
        assert gp.verify_message({"data": "hello"}) is False

    def test_verify_wrong_signature(self, gp):
        """Tampered _hmac value returns False."""
        signed = gp.sign_message({"data": "hello"})
        signed["_hmac"] = "a" * 64
        assert gp.verify_message(signed) is False

    def test_verify_tampered_body(self, gp):
        """Changed body content after signing returns False."""
        signed = gp.sign_message({"data": "original"})
        signed["data"] = "tampered"
        assert gp.verify_message(signed) is False

    def test_sign_with_body_subkey(self, gp):
        """When _body subkey is present, only _body is serialized for signing."""
        msg = {"_body": {"node_id": "n1", "payload": "secret"}, "extra": "ignored"}
        signed = gp.sign_message(msg)
        assert gp.verify_message(signed) is True

    def test_different_key_fails_verification(self, gp):
        """Message signed with one key cannot be verified with another."""
        other = GossipProtocol(node_id="n2", hmac_key="different-key-9876543210fedcba")
        signed = gp.sign_message({"data": "hello"})
        assert other.verify_message(signed) is False

    def test_sign_empty_message(self, gp):
        """Empty dict can be signed and verified."""
        signed = gp.sign_message({})
        assert gp.verify_message(signed) is True

    def test_sign_nested_structure(self, gp):
        """Nested dict is serialised deterministically."""
        nested = {"node_id": "n1", "metadata": {"ts": 100.0, "writer": "a"}}
        signed = gp.sign_message(nested)
        assert gp.verify_message(signed) is True

    def test_verify_empty_message_no_hmac(self, gp):
        """Empty dict without _hmac returns False."""
        assert gp.verify_message({}) is False

    def test_verify_empty_hmac_string(self, gp):
        """Empty _hmac string returns False (does not match expected digest)."""
        signed = gp.sign_message({"x": "y"})
        signed["_hmac"] = ""
        assert gp.verify_message(signed) is False


# =========================================================================
# 3. add_peer / remove_peer
# =========================================================================

class TestPeerManagement:
    """Peer set management with max_peers enforcement."""

    def test_add_peer_adds_to_known_peers(self, gp):
        gp.add_peer("peer-1")
        assert "peer-1" in gp.state.known_peers

    def test_add_peer_duplicate_is_idempotent(self, gp):
        gp.add_peer("peer-1")
        gp.add_peer("peer-1")
        assert len(gp.state.known_peers) == 1

    def test_add_peer_multiple(self, gp):
        gp.add_peer("p1")
        gp.add_peer("p2")
        gp.add_peer("p3")
        assert gp.state.known_peers == {"p1", "p2", "p3"}

    def test_add_peer_evicts_when_exceeds_max(self, gp):
        """When known_peers exceeds max_peers, random eviction occurs."""
        gp.max_peers = 3
        # Mock random.sample so the newest peer ("p4") survives
        with patch("distllm.dist.p2p.gossip.random.sample", return_value=["p1", "p2", "p4"]):
            gp.add_peer("p1")
            gp.add_peer("p2")
            gp.add_peer("p3")
            gp.add_peer("p4")  # len becomes 4 > max_peers=3 → eviction
        assert gp.state.known_peers == {"p1", "p2", "p4"}

    def test_remove_peer_removes_existing(self, gp):
        gp.add_peer("peer-1")
        gp.remove_peer("peer-1")
        assert "peer-1" not in gp.state.known_peers

    def test_remove_peer_nonexistent_no_error(self, gp):
        """Removing a peer not in the set does not raise."""
        gp.remove_peer("never-added")
        assert gp.state.known_peers == set()

    def test_remove_from_empty_set_no_error(self, gp):
        gp.remove_peer("anything")
        assert gp.state.known_peers == set()

    def test_get_peers(self, gp):
        gp.add_peer("p1")
        gp.add_peer("p2")
        assert set(gp.get_peers()) == {"p1", "p2"}

    def test_get_peers_empty(self, gp):
        assert gp.get_peers() == []

    def test_select_peer(self, gp):
        gp.add_peer("p1")
        gp.add_peer("p2")
        selected = gp.select_peer()
        assert selected in ("p1", "p2")

    def test_select_peer_empty(self, gp):
        assert gp.select_peer() is None

    def test_add_peer_thread_safety(self, gp):
        """Concurrent add_peer calls do not corrupt the set."""
        errors: list[Exception] = []

        def adder(pid: str) -> None:
            try:
                for _ in range(100):
                    gp.add_peer(pid)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=adder, args=(f"p{i}",)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        for i in range(4):
            assert f"p{i}" in gp.state.known_peers

    def test_remove_peer_thread_safety(self, gp):
        """Concurrent add/remove of the same peer does not raise."""
        gp.add_peer("target")
        errors: list[Exception] = []

        def toggle() -> None:
            try:
                for _ in range(50):
                    gp.add_peer("target")
                    gp.remove_peer("target")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=toggle) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors


# =========================================================================
# 4. store_local / lookup
# =========================================================================

class TestStoreAndLookup:
    """Cache entry storage and retrieval."""

    def test_store_local_sets_entry(self, gp, mock_time):
        gp.store_local("hash-abc", "ref-xyz")
        assert gp.state.local_entries["hash-abc"] == "ref-xyz"

    def test_store_local_increments_vector_clock(self, gp, mock_time):
        gp.store_local("h1", "r1")
        assert gp.state.vector_clock.clocks["test-node"] == 1

    def test_store_local_multiple_increments(self, gp, mock_time):
        gp.store_local("h1", "r1")
        gp.store_local("h2", "r2")
        gp.store_local("h3", "r3")
        assert gp.state.vector_clock.clocks["test-node"] == 3

    def test_store_local_creates_entry_metadata(self, gp, mock_time):
        gp.store_local("hash-abc", "ref-xyz")
        reg = gp.state.entry_metadata["hash-abc"]
        assert reg.value == "ref-xyz"
        assert reg.timestamp == FIXED_TIME
        assert reg.writer_id == "test-node"

    def test_store_local_overwrite(self, gp, mock_time):
        gp.store_local("hash-abc", "ref-old")
        with patch("time.time", return_value=FIXED_TIME + 100):
            gp.store_local("hash-abc", "ref-new")
        assert gp.state.local_entries["hash-abc"] == "ref-new"
        assert gp.state.entry_metadata["hash-abc"].value == "ref-new"
        assert gp.state.entry_metadata["hash-abc"].timestamp == FIXED_TIME + 100

    def test_store_local_overwrite_check_reg_timestamp(self, gp, mock_time):
        """store_local only updates metadata if time.time > reg.timestamp."""
        gp.store_local("h1", "v1")  # timestamp = FIXED_TIME
        # Second store with a "time travel" — time is before the last write
        with patch("time.time", return_value=FIXED_TIME - 1):
            gp.store_local("h1", "v2")
        # Metadata should NOT be updated because FIXED_TIME - 1 < FIXED_TIME
        assert gp.state.entry_metadata["h1"].value == "v1"

    def test_store_local_creates_cache_index(self, gp, mock_time):
        gp.store_local("hash-abc", "ref-xyz")
        entries = gp.state.cache_index["hash-abc"]
        assert len(entries) == 1
        assert entries[0][0] == "test-node"
        assert entries[0][1] == "ref-xyz"
        assert entries[0][2] == FIXED_TIME

    def test_store_local_replaces_self_in_cache_index(self, gp, mock_time):
        """Old self-entry is removed before appending new one."""
        gp.store_local("hash-abc", "ref-v1")
        with patch("time.time", return_value=FIXED_TIME + 100):
            gp.store_local("hash-abc", "ref-v2")
        self_entries = [
            (n, r, t) for n, r, t in gp.state.cache_index["hash-abc"]
            if n == "test-node"
        ]
        assert len(self_entries) == 1
        assert self_entries[0][1] == "ref-v2"

    def test_lookup_finds_own_entry(self, gp, mock_time):
        gp.store_local("hash-abc", "ref-xyz")
        assert gp.lookup("hash-abc") == "test-node"

    def test_lookup_missing_returns_none(self, gp):
        assert gp.lookup("nonexistent") is None

    def test_lookup_empty_cache_index(self, gp):
        assert gp.lookup("anything") is None

    def test_lookup_returns_most_recent(self, gp):
        """lookup sorts by timestamp descending and returns the newest node."""
        gp.state.cache_index["h1"] = [
            ("old-peer", "ref1", 100.0),
            ("new-peer", "ref2", 300.0),
            ("mid-peer", "ref3", 200.0),
        ]
        assert gp.lookup("h1") == "new-peer"

    def test_lookup_single_entry(self, gp):
        gp.state.cache_index["h1"] = [("only-peer", "ref", 100.0)]
        assert gp.lookup("h1") == "only-peer"

    def test_store_local_thread_safety(self, gp, mock_time):
        """Concurrent store_local calls complete without error."""
        errors: list[Exception] = []

        def store(h: str) -> None:
            try:
                for _ in range(50):
                    gp.store_local(h, f"ref-{h}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=store, args=(f"h{i}",)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors
        for i in range(5):
            assert f"h{i}" in gp.state.local_entries

    def test_lookup_on_empty_cache_index(self, gp):
        """A cache_index with the key but empty list returns None."""
        gp.state.cache_index["empty"] = []
        assert gp.lookup("empty") is None


# =========================================================================
# 5. advertise / process_advertisement
# =========================================================================

class TestAdvertise:
    """Delta and full advertisement construction."""

    def test_advertise_full_empty_state(self, gp, mock_time):
        """Full advertisement with no entries."""
        ad = gp.advertise(delta_only=False)
        assert ad["node_id"] == "test-node"
        assert ad["cache_prefixes"] == []
        assert ad["total_cache_entries"] == 0
        assert ad["timestamp"] == FIXED_TIME

    def test_advertise_full_includes_all_entries(self, gp, mock_time):
        gp.store_local("h1", "r1")
        gp.store_local("h2", "r2")
        ad = gp.advertise(delta_only=False)
        assert set(ad["cache_prefixes"]) == {"h1", "h2"}
        assert ad["total_cache_entries"] == 2

    def test_advertise_full_excludes_tombstoned(self, gp, mock_time):
        gp.store_local("h1", "r1")
        gp.tombstone_entry("h1")
        ad = gp.advertise(delta_only=False)
        assert "h1" not in ad["cache_prefixes"]

    def test_advertise_full_includes_metadata(self, gp, mock_time):
        gp.store_local("h1", "r1")
        ad = gp.advertise(delta_only=False)
        assert ad["entry_metadata"]["h1"]["value"] == "r1"
        assert ad["entry_metadata"]["h1"]["writer_id"] == "test-node"

    def test_advertise_full_includes_vector_clock(self, gp, mock_time):
        gp.store_local("h1", "r1")
        gp.store_local("h2", "r2")
        ad = gp.advertise(delta_only=False)
        assert ad["vector_clock"]["test-node"] == 2

    def test_advertise_full_includes_tombstones(self, gp, mock_time):
        gp.tombstone_entry("dead")
        gp.tombstone_entry("gone")
        ad = gp.advertise(delta_only=False)
        assert "dead" in ad["tombstones"]
        assert "gone" in ad["tombstones"]

    def test_advertise_delta_first_call_is_full(self, gp, mock_time):
        """First delta-only call returns full state (last_exchange_time == 0)."""
        gp.store_local("h1", "r1")
        ad = gp.advertise(delta_only=True)
        assert ad["is_delta"] is False
        assert "h1" in ad["cache_prefixes"]

    def test_advertise_delta_empty_when_no_changes(self, gp, mock_time):
        gp.store_local("h1", "r1")
        gp.advertise(delta_only=True)  # first → full, sets last_exchange_time
        ad = gp.advertise(delta_only=True)  # second → delta, nothing new
        assert len(ad["cache_prefixes"]) == 0

    def test_advertise_delta_includes_recent_changes(self, gp, mock_time):
        gp.store_local("h1", "r1")
        gp.advertise(delta_only=True)  # saves last_exchange_time = FIXED_TIME
        with patch("time.time", return_value=FIXED_TIME + 50):
            gp.store_local("h2", "r2")
        with patch("time.time", return_value=FIXED_TIME + 100):
            ad = gp.advertise(delta_only=True)
            assert "h2" in ad["cache_prefixes"]

    def test_advertise_delta_excludes_old_entries(self, gp, mock_time):
        gp.store_local("old1", "r1")
        with patch("time.time", return_value=FIXED_TIME + 100):
            gp.store_local("new1", "r2")
        with patch("time.time", return_value=FIXED_TIME + 200):
            gp.advertise(delta_only=True)  # full, sets last_exchange = FIXED_TIME + 200
        with patch("time.time", return_value=FIXED_TIME + 300):
            gp.store_local("newest", "r3")
        with patch("time.time", return_value=FIXED_TIME + 400):
            ad = gp.advertise(delta_only=True)
        assert "old1" not in ad["cache_prefixes"]
        assert "new1" not in ad["cache_prefixes"]
        assert "newest" in ad["cache_prefixes"]

    def test_advertise_delta_includes_recent_tombstones(self, gp, mock_time):
        gp.store_local("h1", "r1")
        gp.advertise(delta_only=True)
        with patch("time.time", return_value=FIXED_TIME + 50):
            gp.tombstone_entry("h1")
        with patch("time.time", return_value=FIXED_TIME + 100):
            ad = gp.advertise(delta_only=True)
            assert "h1" in ad["tombstones"]

    def test_advertise_delta_empty_state(self, gp, mock_time):
        """Delta with no stored entries at all."""
        gp.advertise(delta_only=True)  # first call, full but empty
        ad = gp.advertise(delta_only=True)  # second call, delta
        assert ad["cache_prefixes"] == []
        assert ad["total_cache_entries"] == 0
        assert ad["is_delta"] is False  # len(prefixes) == len(local_entries) == 0


class TestProcessAdvertisement:
    """Processing peer advertisements — HMAC and state merge."""

    def test_process_invalid_hmac_returns_empty(self, gp):
        ad = {"node_id": "peer-1", "cache_prefixes": ["abc"]}
        result = gp.process_advertisement(ad)
        assert result == []

    def test_process_valid_ad_adds_peer(self, gp, gp_peer, mock_time):
        gp_peer.store_local("h1", "r1")
        signed = gp_peer.sign_message(gp_peer.advertise(delta_only=False))
        missing = gp.process_advertisement(signed)
        assert "peer-node" in gp.state.known_peers
        assert "h1" in missing

    def test_process_ad_returns_missing_prefixes(self, gp, gp_peer, mock_time):
        gp.store_local("shared", "r1")
        gp_peer.store_local("shared", "r1")
        gp_peer.store_local("peer-only", "r2")
        signed = gp_peer.sign_message(gp_peer.advertise(delta_only=False))
        missing = gp.process_advertisement(signed)
        assert "peer-only" in missing
        assert "shared" not in missing

    def test_process_ad_empty_prefixes(self, gp, gp_peer, mock_time):
        """No cache entries in peer ad."""
        signed = gp_peer.sign_message(gp_peer.advertise(delta_only=False))
        missing = gp.process_advertisement(signed)
        assert missing == []

    def test_process_ad_skips_tombstoned_peer_prefixes(self, gp, gp_peer, mock_time):
        """Locally tombstoned entries are not returned as missing."""
        gp.tombstone_entry("dead")
        gp_peer.store_local("dead", "r1")
        signed = gp_peer.sign_message(gp_peer.advertise(delta_only=False))
        missing = gp.process_advertisement(signed)
        assert "dead" not in missing

    def test_process_ad_merges_vector_clock(self, gp, gp_peer, mock_time):
        gp_peer.store_local("h1", "r1")
        signed = gp_peer.sign_message(gp_peer.advertise(delta_only=False))
        gp.process_advertisement(signed)
        assert gp.state.vector_clock.clocks.get("peer-node", 0) >= 1

    def test_process_ad_merges_tombstones_from_peer(self, gp, gp_peer, mock_time):
        gp_peer.tombstone_entry("peer-dead")
        signed = gp_peer.sign_message(gp_peer.advertise(delta_only=False))
        gp.process_advertisement(signed)
        assert "peer-dead" in gp.state.tombstones

    def test_process_ad_tombstone_removes_local_entry(self, gp, gp_peer, mock_time):
        gp.store_local("to-remove", "r1")
        gp_peer.tombstone_entry("to-remove")
        signed = gp_peer.sign_message(gp_peer.advertise(delta_only=False))
        gp.process_advertisement(signed)
        assert "to-remove" not in gp.state.local_entries

    def test_process_ad_merges_entry_metadata(self, gp, gp_peer, mock_time):
        gp_peer.store_local("hash-abc", "peer-ref")
        signed = gp_peer.sign_message(gp_peer.advertise(delta_only=False))
        gp.process_advertisement(signed)
        assert gp.state.entry_metadata["hash-abc"].value == "peer-ref"

    def test_process_ad_metadata_lww_newer_wins(self, gp, gp_peer, mock_time):
        """When both have metadata for same hash, peer's newer timestamp wins."""
        gp.store_local("common", "local-ref")
        # Peer writes later
        with patch("time.time", return_value=FIXED_TIME + 100):
            gp_peer.store_local("common", "peer-ref")
        signed = gp_peer.sign_message(gp_peer.advertise(delta_only=False))
        gp.process_advertisement(signed)
        assert gp.state.entry_metadata["common"].value == "peer-ref"

    def test_process_ad_no_cache_index_duplicate(self, gp, gp_peer, mock_time):
        """Peer appears only once per prefix in cache_index."""
        gp_peer.store_local("h1", "r1")
        signed = gp_peer.sign_message(gp_peer.advertise(delta_only=False))
        gp.process_advertisement(signed)
        gp.process_advertisement(signed)  # second time
        peer_refs = [
            (n, r) for n, r, _ in gp.state.cache_index.get("h1", [])
            if n == "peer-node"
        ]
        assert len(peer_refs) == 1

    def test_process_ad_without_vector_clock(self, gp, gp_peer, mock_time):
        """Missing vector_clock does not raise."""
        gp_peer.store_local("h1", "r1")
        ad = gp_peer.advertise(delta_only=False)
        signed = gp_peer.sign_message(ad)
        del signed["vector_clock"]
        gp.process_advertisement(signed)  # no exception

    def test_process_ad_without_tombstones(self, gp, gp_peer, mock_time):
        """Missing tombstones does not raise."""
        gp_peer.store_local("h1", "r1")
        ad = gp_peer.advertise(delta_only=False)
        signed = gp_peer.sign_message(ad)
        del signed["tombstones"]
        gp.process_advertisement(signed)

    def test_process_ad_without_entry_metadata(self, gp, gp_peer, mock_time):
        """Missing entry_metadata does not raise."""
        gp_peer.store_local("h1", "r1")
        ad = gp_peer.advertise(delta_only=False)
        signed = gp_peer.sign_message(ad)
        del signed["entry_metadata"]
        gp.process_advertisement(signed)

    def test_process_ad_triggers_random_cleanup(self, gp, gp_peer, mock_time):
        """~1% of process_ad calls invoke _cleanup_cache_index."""
        gp_peer.store_local("h1", "r1")
        signed = gp_peer.sign_message(gp_peer.advertise(delta_only=False))
        with patch.object(gp, "_cleanup_cache_index") as cleanup_mock:
            with patch("distllm.dist.p2p.gossip.random.random", return_value=0.0):
                gp.process_advertisement(signed)
                cleanup_mock.assert_called_once()

    def test_process_ad_skips_cleanup_most_of_the_time(self, gp, gp_peer, mock_time):
        """With random >= 0.01, cleanup is not called."""
        gp_peer.store_local("h1", "r1")
        signed = gp_peer.sign_message(gp_peer.advertise(delta_only=False))
        with patch.object(gp, "_cleanup_cache_index") as cleanup_mock:
            with patch("distllm.dist.p2p.gossip.random.random", return_value=0.5):
                gp.process_advertisement(signed)
                cleanup_mock.assert_not_called()


# =========================================================================
# 6. has_changes_since
# =========================================================================

class TestHasChangesSince:
    """Pre-exchange change detection (Bloom-filter-like pre-check)."""

    def test_no_changes_empty_state(self, gp):
        assert gp.has_changes_since(0.0) is False

    def test_no_changes_when_since_is_after_all_entries(self, gp, mock_time):
        gp.store_local("h1", "r1")  # timestamp = FIXED_TIME
        assert gp.has_changes_since(FIXED_TIME + 1) is False

    def test_has_changes_when_entry_exists(self, gp, mock_time):
        gp.store_local("h1", "r1")
        assert gp.has_changes_since(FIXED_TIME - 100) is True

    def test_has_changes_due_to_tombstone(self, gp, mock_time):
        gp.tombstone_entry("h1")
        assert gp.has_changes_since(FIXED_TIME - 100) is True

    def test_no_changes_tombstone_before_cutoff(self, gp, mock_time):
        gp.tombstone_entry("h1")
        assert gp.has_changes_since(FIXED_TIME + 100) is False

    def test_has_changes_with_mixed_entries(self, gp, mock_time):
        gp.store_local("old", "r1")  # FIXED_TIME
        with patch("time.time", return_value=FIXED_TIME + 50):
            gp.store_local("new", "r2")
        assert gp.has_changes_since(FIXED_TIME + 25) is True
        assert gp.has_changes_since(FIXED_TIME + 100) is False

    def test_has_changes_many_entries(self, gp, mock_time):
        """Multiple entries, at least one matches."""
        for i in range(20):
            gp.state.entry_metadata[f"h{i}"] = LWWRegister(
                value=f"r{i}", timestamp=FIXED_TIME - 1000 + i * 100, writer_id="x"
            )
        gp.state._max_metadata_ts = max(
            m.timestamp for m in gp.state.entry_metadata.values()
        )
        # Timestamps range from FIXED_TIME-1000 to FIXED_TIME+900
        assert gp.has_changes_since(FIXED_TIME - 500) is True
        assert gp.has_changes_since(FIXED_TIME + 1000) is False


# =========================================================================
# 7. request_cache_from_peers — using a mock GossipClient
# =========================================================================

class TestRequestCacheFromPeers:
    """KV cache fetch from peers via mock GossipClient."""

    def test_no_client_returns_none(self, gp):
        """None client returns None immediately."""
        result = gp.request_cache_from_peers("hash-abc")
        assert result is None

    def test_no_peers_returns_none(self, gp):
        """Empty known_peers returns None."""
        client = MagicMock()
        result = gp.request_cache_from_peers("hash-abc", client=client)
        assert result is None
        client.fetch_kv_cache.assert_not_called()

    def test_single_peer_success(self, gp):
        """Returns result from the only peer."""
        gp.add_peer("peer-1")
        client = MagicMock()
        client.fetch_kv_cache.return_value = {"data": "cached_value"}

        result = gp.request_cache_from_peers("hash-abc", client=client)

        assert result == {"data": "cached_value"}
        client.fetch_kv_cache.assert_called_once()

    def test_multiple_peers_returns_first_success(self, gp):
        """First successful peer result is returned."""
        gp.add_peer("p1")
        gp.add_peer("p2")
        client = MagicMock()
        # Both return a result; which future finishes first is non-deterministic
        client.fetch_kv_cache.return_value = {"data": "found"}

        result = gp.request_cache_from_peers("hash-abc", client=client)

        assert result == {"data": "found"}

    def test_all_peers_return_none(self, gp):
        """When all peers return None, returns None."""
        gp.add_peer("p1")
        gp.add_peer("p2")
        client = MagicMock()
        client.fetch_kv_cache.return_value = None

        result = gp.request_cache_from_peers("hash-abc", client=client)

        assert result is None

    def test_peer_raises_exception_skips_it(self, gp):
        """Exception from one peer does not stop trying others."""
        gp.add_peer("p1")
        gp.add_peer("p2")
        client = MagicMock()
        # Both raise — all are skipped
        client.fetch_kv_cache.side_effect = RuntimeError("timeout")

        result = gp.request_cache_from_peers("hash-abc", client=client)

        assert result is None

    def test_mixed_peer_results(self, gp):
        """Exception on some peer, success on another."""
        gp.add_peer("p1")
        gp.add_peer("p2")
        gp.add_peer("p3")
        client = MagicMock()
        # Return value order not guaranteed with threads, but at least one is a valid result
        client.fetch_kv_cache.return_value = {"data": "ok"}

        result = gp.request_cache_from_peers("hash-abc", client=client)

        assert result == {"data": "ok"}

    @pytest.mark.timeout(30)
    def test_max_workers_capped(self, gp):
        """ThreadPoolExecutor max_workers is capped at min(len(peers), 32)."""
        gp.max_peers = 50  # prevent eviction when adding many peers
        for i in range(50):
            gp.add_peer(f"p{i:03d}")
        client = MagicMock()
        client.fetch_kv_cache.return_value = {"data": "v"}

        # Use wraps so as_completed gets real Futures (not MagicMock futures
        # that never complete and cause a hang).
        with patch("concurrent.futures.ThreadPoolExecutor", wraps=RealThreadPoolExecutor) as mock_exec:
            gp.request_cache_from_peers("hash-abc", client=client)

        _, kwargs = mock_exec.call_args
        assert kwargs["max_workers"] == 32

    def test_calls_fetch_kv_cache_with_correct_args(self, gp):
        """fetch_kv_cache is called with (peer_id, prefix_hash) for each peer."""
        gp.add_peer("peer-a")
        gp.add_peer("peer-b")
        client = MagicMock()
        # immediate return so we don't need to deal with threading
        client.fetch_kv_cache.return_value = {"data": "val"}

        gp.request_cache_from_peers("hash-xyz", client=client)

        calls = client.fetch_kv_cache.call_args_list
        assert len(calls) == 2
        call_args_set = {(c[0][0], c[0][1]) for c in calls}
        assert call_args_set == {("peer-a", "hash-xyz"), ("peer-b", "hash-xyz")}

    def test_empty_prefix_hash(self, gp):
        """Empty prefix hash is passed through."""
        gp.add_peer("peer-1")
        client = MagicMock()
        client.fetch_kv_cache.return_value = {}

        result = gp.request_cache_from_peers("", client=client)

        client.fetch_kv_cache.assert_called_once_with("peer-1", "")
        assert result == {}


# =========================================================================
# 8. tombstone_entry
# =========================================================================

class TestTombstoneEntry:
    """Entry removal via tombstone."""

    def test_tombstone_removes_local_entry(self, gp, mock_time):
        gp.store_local("h1", "r1")
        gp.tombstone_entry("h1")
        assert "h1" not in gp.state.local_entries

    def test_tombstone_adds_to_tombstones(self, gp, mock_time):
        gp.store_local("h1", "r1")
        gp.tombstone_entry("h1")
        assert gp.state.tombstones["h1"] == FIXED_TIME

    def test_tombstone_nonexistent_entry(self, gp, mock_time):
        """Tombstoning something never stored works (no-op local_entries removal)."""
        gp.tombstone_entry("never-stored")
        assert gp.state.tombstones["never-stored"] == FIXED_TIME
        assert "never-stored" not in gp.state.local_entries

    def test_tombstone_increments_vector_clock(self, gp, mock_time):
        gp.tombstone_entry("h1")
        assert gp.state.vector_clock.clocks["test-node"] == 1

    def test_tombstone_preserves_later_timestamp(self, gp, mock_time):
        """Second tombstone keeps the later timestamp."""
        gp.tombstone_entry("h1")
        ts1 = gp.state.tombstones["h1"]
        with patch("time.time", return_value=FIXED_TIME + 100):
            gp.tombstone_entry("h1")
        assert gp.state.tombstones["h1"] == FIXED_TIME + 100
        assert gp.state.tombstones["h1"] >= ts1

    def test_tombstone_does_not_roll_back(self, gp, mock_time):
        """If existing tombstone is newer, the older time is ignored."""
        gp.state.tombstones["h1"] = FIXED_TIME + 1000  # pre-set future tombstone
        gp.tombstone_entry("h1")  # now = FIXED_TIME, which is older
        assert gp.state.tombstones["h1"] == FIXED_TIME + 1000  # unchanged

    def test_tombstone_does_not_modify_cache_index(self, gp, mock_time):
        """tombstone_entry leaves cache_index intact."""
        gp.store_local("h1", "r1")
        gp.tombstone_entry("h1")
        assert "h1" in gp.state.cache_index
        assert "h1" in gp.state.entry_metadata

    def test_tombstone_does_not_remove_metadata(self, gp, mock_time):
        """entry_metadata is preserved (cleanup_expired handles it later)."""
        gp.store_local("h1", "r1")
        gp.tombstone_entry("h1")
        assert "h1" in gp.state.entry_metadata


# =========================================================================
# 9. cleanup_expired
# =========================================================================

class TestCleanupExpired:
    """TTL-based cleanup of stale cache entries and old tombstones."""

    def test_cleanup_empty(self, gp):
        assert gp.cleanup_expired() == 0

    def test_cleanup_fresh_entries_kept(self, gp, mock_time):
        gp.state.cache_index["h1"] = [("peer", "ref", FIXED_TIME)]
        assert gp.cleanup_expired() == 0
        assert "h1" in gp.state.cache_index

    def test_cleanup_stale_entry_removed(self, gp, mock_time):
        gp.state.cache_index["stale"] = [("peer", "ref", FIXED_TIME - gp.cache_ttl - 10)]
        assert gp.cleanup_expired() == 1
        assert "stale" not in gp.state.cache_index

    def test_cleanup_mixed_fresh_and_stale(self, gp, mock_time):
        gp.state.cache_index["fresh"] = [("peer", "ref", FIXED_TIME)]
        gp.state.cache_index["stale"] = [("peer", "ref", FIXED_TIME - gp.cache_ttl - 10)]
        assert gp.cleanup_expired() == 1
        assert "fresh" in gp.state.cache_index
        assert "stale" not in gp.state.cache_index

    def test_cleanup_empty_ref_list_stale_removed(self, gp, mock_time):
        """Empty ref list is removed (entry_metadata is NOT cleaned here)."""
        gp.state.cache_index["empty-stale"] = []
        assert gp.cleanup_expired() == 1
        assert "empty-stale" not in gp.state.cache_index

    def test_cleanup_empty_ref_list_all_deleted_unconditionally(self, gp, mock_time):
        """cleanup_expired unconditionally deletes any empty ref list,
        regardless of metadata freshness."""
        gp.state.cache_index["empty-fresh"] = []
        gp.state.entry_metadata["empty-fresh"] = LWWRegister(
            value="ref",
            timestamp=FIXED_TIME,
            writer_id="peer",
        )
        assert gp.cleanup_expired() == 1
        assert "empty-fresh" not in gp.state.cache_index

    def test_cleanup_empty_ref_list_no_metadata_removed(self, gp, mock_time):
        """Empty ref list with no metadata at all is removed."""
        gp.state.cache_index["no-meta"] = []
        assert gp.cleanup_expired() == 1
        assert "no-meta" not in gp.state.cache_index

    def test_cleanup_old_tombstones_removed(self, gp, mock_time):
        """Tombstones older than 2x cache_ttl are cleaned up."""
        old_ts = FIXED_TIME - gp.cache_ttl * 3
        gp.state.tombstones["old"] = old_ts
        gp.state.entry_metadata["old"] = LWWRegister(
            value="ref", timestamp=old_ts, writer_id="a"
        )
        gp.cleanup_expired()
        assert "old" not in gp.state.tombstones
        assert "old" not in gp.state.entry_metadata

    def test_cleanup_keeps_recent_tombstones(self, gp, mock_time):
        """Recent tombstones (within 2x cache_ttl) are preserved."""
        gp.state.tombstones["recent"] = FIXED_TIME - gp.cache_ttl  # within 2x TTL
        gp.cleanup_expired()
        assert "recent" in gp.state.tombstones

    def test_cleanup_counts_only_cache_index_removals(self, gp, mock_time):
        """Returned count reflects removed cache_index entries, not tombstones."""
        gp.state.cache_index["s1"] = [("p", "r", FIXED_TIME - gp.cache_ttl - 10)]
        gp.state.cache_index["s2"] = [("p", "r", FIXED_TIME - gp.cache_ttl - 20)]
        gp.state.tombstones["old-tomb"] = FIXED_TIME - gp.cache_ttl * 3  # removed but not counted
        count = gp.cleanup_expired()
        assert count == 2
        assert "old-tomb" not in gp.state.tombstones  # still cleaned up though

    def test_cleanup_no_stale_entries(self, gp, mock_time):
        """Entries with cutoff boundary are kept."""
        boundary = FIXED_TIME - gp.cache_ttl
        gp.state.cache_index["boundary"] = [("p", "r", boundary)]
        assert gp.cleanup_expired() == 0
        assert "boundary" in gp.state.cache_index


# =========================================================================
# Auxiliary — _cleanup_cache_index (private method)
# =========================================================================

class TestCleanupCacheIndex:
    """Private cache index pruning (filters unknown peers, stale empty)."""

    def test_cleanup_removes_stale_empty_prefixes(self, gp, mock_time):
        """Empty ref list with metadata past TTL is removed."""
        gp.state.cache_index["stale"] = []
        gp.state.entry_metadata["stale"] = LWWRegister(
            value="ref", timestamp=FIXED_TIME - gp.cache_ttl - 10, writer_id="p"
        )
        gp._cleanup_cache_index()
        assert "stale" not in gp.state.cache_index
        assert "stale" not in gp.state.entry_metadata

    def test_cleanup_keeps_fresh_empty_prefixes(self, gp, mock_time):
        """Empty ref list with recent metadata is kept."""
        gp.state.cache_index["fresh"] = []
        gp.state.entry_metadata["fresh"] = LWWRegister(
            value="ref", timestamp=FIXED_TIME, writer_id="p"
        )
        gp._cleanup_cache_index()
        assert "fresh" in gp.state.cache_index

    def test_cleanup_filters_unknown_peer_refs(self, gp, mock_time):
        """References from unknown peers are filtered out."""
        gp.state.known_peers = {"known-peer"}
        gp.state.cache_index["h1"] = [
            ("known-peer", "r1", FIXED_TIME),
            ("unknown-peer", "r2", FIXED_TIME),
        ]
        gp._cleanup_cache_index()
        refs = gp.state.cache_index["h1"]
        assert len(refs) == 1
        assert refs[0][0] == "known-peer"

    def test_cleanup_keeps_all_known_peer_refs(self, gp, mock_time):
        """All references from known peers are preserved."""
        gp.state.known_peers = {"p1", "p2"}
        gp.state.cache_index["h1"] = [
            ("p1", "r1", FIXED_TIME),
            ("p2", "r2", FIXED_TIME),
        ]
        gp._cleanup_cache_index()
        assert len(gp.state.cache_index["h1"]) == 2


# =========================================================================
# 10. VectorClock
# =========================================================================

class TestVectorClock:
    """Vector clock: increment, merge, happens_before, is_concurrent."""

    def test_default_empty(self):
        vc = VectorClock()
        assert vc.clocks == {}

    def test_increment_new_node(self):
        vc = VectorClock()
        vc.increment("a")
        assert vc.clocks == {"a": 1}

    def test_increment_existing_node(self):
        vc = VectorClock({"a": 5})
        vc.increment("a")
        assert vc.clocks == {"a": 6}

    def test_increment_multiple_nodes(self):
        vc = VectorClock()
        vc.increment("a")
        vc.increment("b")
        vc.increment("a")
        assert vc.clocks == {"a": 2, "b": 1}

    def test_merge_empty_self(self):
        vc = VectorClock({"a": 1})
        vc.merge(VectorClock())
        assert vc.clocks == {"a": 1}

    def test_merge_empty_other(self):
        vc = VectorClock()
        vc.merge(VectorClock({"a": 5}))
        assert vc.clocks == {"a": 5}

    def test_merge_takes_max_for_each_key(self):
        vc1 = VectorClock({"a": 3, "b": 1})
        vc2 = VectorClock({"a": 1, "b": 5})
        vc1.merge(vc2)
        assert vc1.clocks == {"a": 3, "b": 5}

    def test_merge_disjoint_keys(self):
        vc1 = VectorClock({"a": 1})
        vc2 = VectorClock({"b": 2})
        vc1.merge(vc2)
        assert vc1.clocks == {"a": 1, "b": 2}

    def test_happens_before_true(self):
        assert VectorClock({"a": 1}).happens_before(VectorClock({"a": 2})) is True
        assert VectorClock({"a": 1, "b": 2}).happens_before(VectorClock({"a": 2, "b": 3})) is True

    def test_happens_before_false_self_larger(self):
        assert VectorClock({"a": 5}).happens_before(VectorClock({"a": 2})) is False

    def test_happens_before_equal(self):
        vc = VectorClock({"a": 2})
        assert vc.happens_before(vc) is False

    def test_happens_before_empty_vs_nonempty(self):
        assert VectorClock().happens_before(VectorClock({"a": 1})) is True
        assert VectorClock({"a": 1}).happens_before(VectorClock()) is False

    def test_happens_before_both_empty(self):
        assert VectorClock().happens_before(VectorClock()) is False

    def test_happens_before_disjoint_keys(self):
        """Disjoint keys: neither happens before the other (each has a key > other's 0)."""
        a = VectorClock({"a": 1})
        b = VectorClock({"b": 1})
        assert a.happens_before(b) is False
        assert b.happens_before(a) is False

    def test_is_concurrent_true(self):
        assert VectorClock({"a": 1, "b": 2}).is_concurrent(VectorClock({"a": 2, "b": 1})) is True

    def test_is_concurrent_false_when_equal(self):
        assert VectorClock({"a": 1}).is_concurrent(VectorClock({"a": 1})) is False

    def test_is_concurrent_false_when_before(self):
        assert VectorClock({"a": 1}).is_concurrent(VectorClock({"a": 2})) is False

    def test_is_concurrent_empty_vs_nonempty(self):
        """Empty clock happens-before nonempty, so not concurrent."""
        assert VectorClock().is_concurrent(VectorClock({"a": 1})) is False

    def test_is_concurrent_disjoint_keys(self):
        """Disjoint keys where each has >= 1 for a key the other lacks are concurrent."""
        a = VectorClock({"a": 1})
        b = VectorClock({"b": 1})
        assert a.is_concurrent(b) is True
        assert b.is_concurrent(a) is True


# =========================================================================
# 11. LWWRegister
# =========================================================================

class TestLWWRegister:
    """Last-writer-wins register merge with timestamps."""

    def test_default_values(self):
        reg = LWWRegister()
        assert reg.value == ""
        assert reg.timestamp == 0.0
        assert reg.writer_id == ""

    def test_merge_newer_timestamp_wins(self):
        older = LWWRegister(value="old", timestamp=100.0, writer_id="a")
        newer = LWWRegister(value="new", timestamp=200.0, writer_id="b")
        older.merge(newer)
        assert older.value == "new"
        assert older.timestamp == 200.0
        assert older.writer_id == "b"

    def test_merge_older_timestamp_loses(self):
        older = LWWRegister(value="old", timestamp=100.0, writer_id="a")
        newer = LWWRegister(value="new", timestamp=200.0, writer_id="b")
        newer.merge(older)
        assert newer.value == "new"
        assert newer.timestamp == 200.0
        assert newer.writer_id == "b"

    def test_merge_same_timestamp_higher_writer_wins(self):
        low = LWWRegister(value="low", timestamp=100.0, writer_id="aaaa")
        high = LWWRegister(value="high", timestamp=100.0, writer_id="zzzz")
        low.merge(high)
        assert low.value == "high"
        assert low.writer_id == "zzzz"

    def test_merge_same_timestamp_lower_writer_loses(self):
        low = LWWRegister(value="low", timestamp=100.0, writer_id="aaaa")
        high = LWWRegister(value="high", timestamp=100.0, writer_id="zzzz")
        high.merge(low)
        assert high.value == "high"
        assert high.writer_id == "zzzz"

    def test_merge_self_unchanged_when_other_older(self):
        reg = LWWRegister(value="x", timestamp=50.0, writer_id="a")
        reg.merge(LWWRegister(value="y", timestamp=30.0, writer_id="b"))
        assert reg.value == "x"
        assert reg.writer_id == "a"
        assert reg.timestamp == 50.0

    def test_merge_identical_register_noop(self):
        a = LWWRegister(value="same", timestamp=100.0, writer_id="a")
        b = LWWRegister(value="same", timestamp=100.0, writer_id="a")
        a.merge(b)
        assert a.value == "same"
        assert a.writer_id == "a"

    def test_merge_into_default(self):
        default = LWWRegister()
        other = LWWRegister(value="val", timestamp=50.0, writer_id="x")
        default.merge(other)
        assert default.value == "val"
        assert default.writer_id == "x"
        assert default.timestamp == 50.0

    def test_merge_default_into_existing(self):
        reg = LWWRegister(value="val", timestamp=100.0, writer_id="x")
        reg.merge(LWWRegister())
        assert reg.value == "val"
        assert reg.timestamp == 100.0

    def test_merge_writer_tiebreaker_lexicographic(self):
        """Same timestamp, writer "zzz" beats "aaa"."""
        x = LWWRegister(value="lose", timestamp=100.0, writer_id="aaa")
        y = LWWRegister(value="win", timestamp=100.0, writer_id="zzz")
        x.merge(y)
        assert x.value == "win"
        assert x.writer_id == "zzz"


# =========================================================================
# Auxiliary — process_response / build_request
# =========================================================================

class TestProcessResponse:
    """Cache index updates from peer responses."""

    def test_process_response_counts_entries(self, gp):
        resp = {"success": True, "cache_entries": {"h1": "r1", "h2": "r2"}}
        assert gp.process_response(resp) == 2

    def test_process_response_not_success(self, gp):
        assert gp.process_response({"success": False}) == 0

    def test_process_response_empty_entries(self, gp):
        assert gp.process_response({"success": True, "cache_entries": {}}) == 0

    def test_build_request(self, gp):
        req = gp.build_request("target-peer", ["h1", "h2"])
        assert req["requester_id"] == "test-node"
        assert req["target_node_id"] == "target-peer"
        assert req["requested_prefixes"] == ["h1", "h2"]


# =========================================================================
# 14. Signed KV lookup/fetch (Wave 2 item 3)
#
# Advertisements were always HMAC-authenticated; the fetch path (lookup
# + response merge) previously accepted anything from anyone.  These
# tests pin the closed gap: lookups signed with the existing shared-key
# scheme are authorized, tampered/unsigned ones rejected, and legacy
# (no shared key) deployments keep working with a loud warning.
# =========================================================================


class TestSignedKVFetch:
    """HMAC authorization of the gossip KV fetch path."""

    KEY = "wave2-fetch-shared-secret-abcdef0123456789"

    @pytest.fixture
    def receiver(self):
        """Node holding cache entries, under the shared deployment key."""
        return GossipProtocol(node_id="receiver", hmac_key=self.KEY)

    @pytest.fixture
    def sender(self):
        """Legitimate peer under the same shared deployment key."""
        return GossipProtocol(node_id="sender", hmac_key=self.KEY)

    # ------------------------------------------------------------------
    # authorize_fetch_request — signed / tampered / unsigned
    # ------------------------------------------------------------------

    def test_signed_lookup_accepted(self, receiver, sender):
        """A correctly-signed fetch is authorized."""
        request = sender.sign_fetch_request(
            {"requester_id": "sender", "prefix_hashes": ["h1"]}
        )
        ok, reason = receiver.authorize_fetch_request(request)
        assert ok is True
        assert reason == ""

    def test_signed_lookup_survives_json_roundtrip(self, receiver, sender):
        """Authorization works after serialize/deserialize (wire fidelity)."""
        request = sender.sign_fetch_request(
            {"requester_id": "sender", "prefix_hashes": ["h1", "h2"]}
        )
        wire = json.loads(json.dumps(request))
        ok, _ = receiver.authorize_fetch_request(wire)
        assert ok is True

    def test_tampered_signature_rejected(self, receiver, sender):
        """Modifying content after signing invalidates the signature."""
        request = sender.sign_fetch_request(
            {"requester_id": "sender", "prefix_hashes": ["h1"]}
        )
        request["prefix_hashes"] = ["victim-only-entry"]
        ok, reason = receiver.authorize_fetch_request(request)
        assert ok is False
        assert reason == "invalid_signature"

    def test_unsigned_rejected_when_secret_configured(self, receiver):
        """Fail closed: no _hmac field -> rejected under a shared key."""
        ok, reason = receiver.authorize_fetch_request(
            {"requester_id": "anyone", "prefix_hashes": ["h1"]}
        )
        assert ok is False
        assert reason == "missing_signature"

    def test_missing_signature_value_rejected_when_secret_configured(self, receiver):
        ok, reason = receiver.authorize_fetch_request(
            {"requester_id": "anyone", "_hmac": None}
        )
        assert ok is False

    def test_non_string_signature_rejected(self, receiver):
        """Hostile non-str _hmac must be rejected, not raise TypeError
        out of hmac.compare_digest."""
        ok, reason = receiver.authorize_fetch_request(
            {"requester_id": "anyone", "_hmac": 12345}
        )
        assert ok is False
        assert reason in ("missing_signature", "invalid_signature")

    def test_none_request_rejected_when_secret_configured(self, receiver):
        ok, reason = receiver.authorize_fetch_request(None)
        assert ok is False
        assert reason == "missing_request"

    def test_wrong_key_signature_rejected(self, receiver):
        """Signature produced under a different secret fails verification."""
        attacker = GossipProtocol(node_id="attacker", hmac_key="different-key")
        request = attacker.sign_fetch_request(
            {"requester_id": "attacker", "prefix_hashes": ["h1"]}
        )
        ok, reason = receiver.authorize_fetch_request(request)
        assert ok is False
        assert reason == "invalid_signature"

    def test_verify_message_non_string_hmac_is_false_not_crash(self):
        """Regression guard: verify_message on hostile types returns False."""
        p = GossipProtocol(node_id="n1", hmac_key=self.KEY)
        assert p.verify_message({"data": "x", "_hmac": 42}) is False
        assert p.verify_message({"data": "x", "_hmac": b"bytes"}) is False
        assert p.verify_message({"data": "x", "_hmac": {"a": 1}}) is False

    # ------------------------------------------------------------------
    # Legacy mode — no shared key configured
    # ------------------------------------------------------------------

    def test_legacy_mode_accepts_unsigned_with_warning(self, monkeypatch, tmp_path):
        """No shared secret -> unsigned fetch served for backward compat,
        with a loud one-time warning (kademlia_dht.py convention)."""
        # Isolate the persistent-key fallback from the developer's machine.
        monkeypatch.setenv(
            "DISTLLM_GOSSIP_KEY_FILE", str(tmp_path / "gossip_hmac.key")
        )
        records: list = []
        handler_id = logger.add(lambda m: records.append(m.record), level="WARNING")
        try:
            legacy = GossipProtocol(node_id="legacy-node")
            assert legacy.has_shared_hmac_key is False
            request = {"requester_id": "p", "prefix_hashes": ["h1"]}  # unsigned
            ok, reason = legacy.authorize_fetch_request(request)
            assert ok is True
            assert reason == ""
            # Signed-by-nobody content still passes in legacy mode.
            ok2, _ = legacy.authorize_fetch_request(dict(request))
            assert ok2 is True
        finally:
            logger.remove(handler_id)

        warnings = [r for r in records if "cache lookup requests" in r["message"]]
        assert len(warnings) >= 1
        assert "NOT authenticated" in warnings[0]["message"]

    def test_legacy_mode_warns_only_once(self, monkeypatch, tmp_path):
        """The unauthenticated-fetch warning fires once per protocol."""
        monkeypatch.setenv(
            "DISTLLM_GOSSIP_KEY_FILE", str(tmp_path / "gossip_hmac.key")
        )
        records: list = []
        handler_id = logger.add(lambda m: records.append(m.record), level="WARNING")
        try:
            legacy = GossipProtocol(node_id="legacy-node-2")
            legacy.authorize_fetch_request({"prefix_hashes": []})
            legacy.authorize_fetch_request({"prefix_hashes": []})
            legacy.authorize_fetch_request({"prefix_hashes": []})
        finally:
            logger.remove(handler_id)

        warnings = [r for r in records if "cache lookup requests" in r["message"]]
        assert len(warnings) == 1

    def test_shared_mode_does_not_emit_legacy_warning(self, receiver):
        """Under a shared key, authorization paths reject instead of
        warning about unauthenticated mode."""
        request = {
            "requester_id": "s",
            "prefix_hashes": ["h1"],
            "_hmac": "f" * 64,
        }
        ok, reason = receiver.authorize_fetch_request(request)
        assert ok is False
        assert reason == "invalid_signature"
        assert not getattr(receiver, "_fetch_warning_logged", False)

    # ------------------------------------------------------------------
    # process_response — forged cache-index entries (area-analysis H7)
    # ------------------------------------------------------------------

    def test_process_response_accepts_validly_signed_entries(self, receiver, sender):
        payload = sender.sign_message({
            "success": True,
            "cache_entries": {"h1": "ref-1"},
        })
        count = receiver.process_response(payload)
        assert count == 1
        assert any(
            nid == "unknown" and ref == "ref-1"
            for nid, ref, _ in receiver.state.cache_index["h1"]
        )

    def test_process_response_rejects_forged_entries(self, receiver):
        """Tampered/unsigned-with-bad-sig response merges nothing."""
        forged = {
            "success": True,
            "cache_entries": {"evil": "forged-ref"},
            "_hmac": "a" * 64,
        }
        count = receiver.process_response(forged)
        assert count == 0
        assert "evil" not in receiver.state.cache_index

    def test_process_response_unsigned_still_merges_in_dev_mode(self, monkeypatch, tmp_path):
        """Legacy mode (no key): unsigned responses merge as before."""
        monkeypatch.setenv(
            "DISTLLM_GOSSIP_KEY_FILE", str(tmp_path / "gossip_hmac.key")
        )
        dev = GossipProtocol(node_id="dev-node")
        resp = {"success": True, "cache_entries": {"h1": "r1"}}
        assert dev.process_response(resp) == 1
        assert "h1" in dev.state.cache_index

    # ------------------------------------------------------------------
    # handle_bloom_exchange — response signing under shared key
    # ------------------------------------------------------------------

    def test_bloom_exchange_response_is_signed_under_shared_key(self, receiver, sender):
        msg = sender.sign_message({
            "node_id": "sender",
            "type": "bloom_exchange",
            "bloom_filter": sender.build_bloom_filter(),
        })
        resp = receiver.handle_bloom_exchange(msg)
        assert resp["success"] is True
        assert "_hmac" in resp
        assert receiver.verify_message(resp) is True


class TestTransportFetchResponseVerification:
    """GossipTransport verifies fetch responses before returning them."""

    KEY = "transport-fetch-key-0123456789abcdef"

    @staticmethod
    def _make_transport(hmac_key=None):
        """Transport with a resolver pointing at a dummy address (the
        session is mocked, so no real network I/O happens)."""
        from distllm.dist.p2p.transport import GossipTransport

        return GossipTransport(
            node_id="t-node",
            peer_resolver=lambda pid: ("127.0.0.1", 50052),
            hmac_key=hmac_key,
        )

    @staticmethod
    def _sign(payload: dict, key: str) -> dict:
        serialized = json.dumps(
            payload, separators=(",", ":"), sort_keys=True
        ).encode("utf-8")
        sig = hmac.new(key.encode(), msg=serialized, digestmod=hashlib.sha256)
        out = dict(payload)
        out["_hmac"] = sig.hexdigest()
        return out

    def test_unsigned_response_rejected_when_key_configured(self):
        transport = self._make_transport(hmac_key=self.KEY)
        # Patch session.post to return an unsigned 200 response.
        fake_resp = MagicMock()
        fake_resp.status_code = 200
        fake_resp.text = json.dumps({"success": True, "cache_entries": {"h1": "r"}})
        session = MagicMock()
        session.post.return_value = fake_resp
        transport._session = session

        result = transport.request_kv_cache("peer-1", ["h1"])
        assert result is None  # unsigned data must not reach callers

    def test_validly_signed_response_accepted(self):
        transport = self._make_transport(hmac_key=self.KEY)
        body = self._sign(
            {"success": True, "cache_entries": {"h1": "r"}}, self.KEY
        )
        fake_resp = MagicMock()
        fake_resp.status_code = 200
        fake_resp.text = json.dumps(body)
        session = MagicMock()
        session.post.return_value = fake_resp
        transport._session = session

        result = transport.request_kv_cache("peer-1", ["h1"])
        assert result is not None
        assert result["cache_entries"]["h1"] == "r"

    def test_tampered_response_rejected(self):
        transport = self._make_transport(hmac_key=self.KEY)
        body = self._sign(
            {"success": True, "cache_entries": {"h1": "benign"}}, self.KEY
        )
        body["cache_entries"]["h2"] = "injected"  # tamper post-signature
        fake_resp = MagicMock()
        fake_resp.status_code = 200
        fake_resp.text = json.dumps(body)
        session = MagicMock()
        session.post.return_value = fake_resp
        transport._session = session

        result = transport.request_kv_cache("peer-1", ["h1"])
        assert result is None

    def test_no_key_legacy_mode_accepts_unsigned(self):
        """Without a key, behavior is unchanged (legacy open mode)."""
        transport = self._make_transport(hmac_key=None)
        fake_resp = MagicMock()
        fake_resp.status_code = 200
        fake_resp.text = json.dumps({"success": True, "cache_entries": {"h1": "r"}})
        session = MagicMock()
        session.post.return_value = fake_resp
        transport._session = session

        result = transport.request_kv_cache("peer-1", ["h1"])
        assert result is not None
        assert result["success"] is True
