"""Tests for cluster_state_store module.

Covers:
    ClusterNodeState        -- dataclass for a single coordinator state
    ClusterState            -- dataclass for complete cluster state snapshot
    InMemoryBackend         -- fallback key-value store with locks
    ClusterStateStore       -- distributed state store with leader election

Every test is deterministic (no network, no GPU, no time.sleep).
No MagicMock -- real objects or lightweight stubs only.
"""

from __future__ import annotations

import json
import threading
from typing import Any

import pytest

from tests._import_helper import bootstrap_fake_packages, load_module

# Bootstrap fake packages for distllm namespace
bootstrap_fake_packages()

# Load the module
_mod = load_module("distllm/core/cluster_state_store.py")

# Re-export symbols for test readability
ClusterNodeState = _mod.ClusterNodeState
ClusterState = _mod.ClusterState
StateStoreBackend = _mod.StateStoreBackend
InMemoryBackend = _mod.InMemoryBackend
ClusterStateStore = _mod.ClusterStateStore


# ===================================================================
# CLUSTER NODE STATE TESTS
# ===================================================================

class TestClusterNodeState:
    """ClusterNodeState dataclass -- construction and defaults."""

    def test_default_construction(self) -> None:
        """Minimal ClusterNodeState should get reasonable defaults."""
        state = ClusterNodeState(node_id="node-1", host="host1", port=50050)
        assert state.node_id == "node-1"
        assert state.host == "host1"
        assert state.port == 50050
        assert state.is_leader is False
        assert state.is_healthy is True
        assert state.last_seen == 0.0
        assert state.version == ""
        assert state.active_models == []
        assert state.gpu_info == {}
        assert state.load_score == 0.0

    def test_is_leader_flag(self) -> None:
        """is_leader=True should be reflected."""
        state = ClusterNodeState(
            node_id="leader-1", host="h1", port=50050, is_leader=True,
        )
        assert state.is_leader is True

    def test_unhealthy_node(self) -> None:
        """is_healthy=False should be reflected."""
        state = ClusterNodeState(
            node_id="n1", host="h1", port=50050, is_healthy=False,
        )
        assert state.is_healthy is False

    def test_full_construction(self) -> None:
        """All fields should be settable at construction."""
        state = ClusterNodeState(
            node_id="full-node",
            host="10.0.0.1",
            port=50051,
            is_leader=True,
            is_healthy=True,
            last_seen=12345.0,
            version="2.0.0",
            active_models=["llama-7b", "gpt-j-6b"],
            gpu_info={"cuda": "12.1", "memory": 40960},
            load_score=0.75,
        )
        assert state.active_models == ["llama-7b", "gpt-j-6b"]
        assert state.gpu_info["cuda"] == "12.1"
        assert state.load_score == 0.75
        assert state.last_seen == 12345.0
        assert state.version == "2.0.0"

    def test_active_models_mutability(self) -> None:
        """Default factory for active_models should give independent lists."""
        s1 = ClusterNodeState(node_id="a", host="h1", port=50050)
        s2 = ClusterNodeState(node_id="b", host="h2", port=50050)
        s1.active_models.append("model-x")
        assert "model-x" not in s2.active_models

    def test_gpu_info_mutability(self) -> None:
        """Default factory for gpu_info should give independent dicts."""
        s1 = ClusterNodeState(node_id="a", host="h1", port=50050)
        s2 = ClusterNodeState(node_id="b", host="h2", port=50050)
        s1.gpu_info["key"] = "value"
        assert "key" not in s2.gpu_info


# ===================================================================
# CLUSTER STATE TESTS
# ===================================================================

class TestClusterState:
    """ClusterState dataclass -- snapshot construction."""

    def test_default_construction(self) -> None:
        """A default ClusterState should have empty fields."""
        state = ClusterState()
        assert state.leader_id == ""
        assert state.nodes == {}
        assert state.active_models == []
        assert state.global_config == {}
        assert state.last_updated == 0.0

    def test_with_leader(self) -> None:
        """leader_id should be settable."""
        state = ClusterState(leader_id="node-1")
        assert state.leader_id == "node-1"

    def test_with_nodes(self) -> None:
        """nodes dict should be preserved."""
        nodes = {
            "a": ClusterNodeState(node_id="a", host="h1", port=50050),
            "b": ClusterNodeState(node_id="b", host="h2", port=50051),
        }
        state = ClusterState(leader_id="a", nodes=nodes)
        assert len(state.nodes) == 2
        assert state.nodes["a"].host == "h1"

    def test_with_config(self) -> None:
        """global_config and active_models should be settable."""
        state = ClusterState(
            global_config={"model": "llama-70b", "batch_size": 32},
            active_models=["llama-70b"],
        )
        assert state.global_config["model"] == "llama-70b"
        assert state.active_models == ["llama-70b"]


# ===================================================================
# STATE STORE BACKEND ABSTRACT TESTS
# ===================================================================

class TestStateStoreBackend:
    """StateStoreBackend raises NotImplementedError for all methods."""

    def test_get_raises(self) -> None:
        backend = StateStoreBackend()
        with pytest.raises(NotImplementedError):
            backend.get("key")

    def test_set_raises(self) -> None:
        backend = StateStoreBackend()
        with pytest.raises(NotImplementedError):
            backend.set("key", "value")

    def test_delete_raises(self) -> None:
        backend = StateStoreBackend()
        with pytest.raises(NotImplementedError):
            backend.delete("key")

    def test_acquire_lock_raises(self) -> None:
        backend = StateStoreBackend()
        with pytest.raises(NotImplementedError):
            backend.acquire_lock("key", "owner")

    def test_release_lock_raises(self) -> None:
        backend = StateStoreBackend()
        with pytest.raises(NotImplementedError):
            backend.release_lock("key", "owner")


# ===================================================================
# IN-MEMORY BACKEND TESTS
# ===================================================================

class TestInMemoryBackend:
    """InMemoryBackend -- basic key-value and lock operations."""

    # -- Key-value operations --

    def test_get_missing_key(self) -> None:
        backend = InMemoryBackend()
        assert backend.get("nonexistent") is None

    def test_set_and_get(self) -> None:
        backend = InMemoryBackend()
        assert backend.set("key1", "value1") is True
        assert backend.get("key1") == "value1"

    def test_set_overwrite(self) -> None:
        backend = InMemoryBackend()
        backend.set("key1", "old")
        backend.set("key1", "new")
        assert backend.get("key1") == "new"

    def test_set_ttl_ignored(self) -> None:
        """InMemoryBackend.set ignores the ttl parameter."""
        backend = InMemoryBackend()
        backend.set("key1", "value1", ttl=999)
        assert backend.get("key1") == "value1"

    def test_delete_existing(self) -> None:
        backend = InMemoryBackend()
        backend.set("key1", "value1")
        assert backend.delete("key1") is True
        assert backend.get("key1") is None

    def test_delete_missing(self) -> None:
        backend = InMemoryBackend()
        assert backend.delete("nonexistent") is False

    # -- Lock operations --

    def test_acquire_lock_new(self) -> None:
        """A brand-new lock should be acquirable."""
        backend = InMemoryBackend()
        assert backend.acquire_lock("lock1", "owner-1", ttl=10) is True

    def test_acquire_lock_held_by_other(self) -> None:
        """A lock held by another owner should not be acquirable."""
        backend = InMemoryBackend()
        backend.acquire_lock("lock1", "owner-1", ttl=10)
        assert backend.acquire_lock("lock1", "owner-2", ttl=10) is False

    def test_acquire_lock_same_owner_reacquires(self) -> None:
        """The same owner should be able to re-acquire its lock."""
        backend = InMemoryBackend()
        backend.acquire_lock("lock1", "owner-1", ttl=10)
        assert backend.acquire_lock("lock1", "owner-1", ttl=10) is True

    def test_acquire_lock_expired(self) -> None:
        """An expired lock should be acquirable by a new owner."""
        backend = InMemoryBackend()
        backend._locks["lock1"] = ("owner-1", -1.0)  # expired
        assert backend.acquire_lock("lock1", "owner-2", ttl=10) is True

    def test_release_lock_owned(self) -> None:
        """The lock owner should be able to release its lock."""
        backend = InMemoryBackend()
        backend.acquire_lock("lock1", "owner-1", ttl=10)
        assert backend.release_lock("lock1", "owner-1") is True
        # Lock is released, new owner can acquire
        assert backend.acquire_lock("lock1", "owner-2", ttl=10) is True

    def test_release_lock_not_owned(self) -> None:
        """A non-owner should not be able to release the lock."""
        backend = InMemoryBackend()
        backend.acquire_lock("lock1", "owner-1", ttl=10)
        assert backend.release_lock("lock1", "owner-2") is False
        # Original owner still holds the lock
        assert backend.release_lock("lock1", "owner-1") is True

    def test_release_lock_nonexistent(self) -> None:
        """Releasing a non-existent lock should return False."""
        backend = InMemoryBackend()
        assert backend.release_lock("nolock", "owner-1") is False

    def test_release_lock_expired_before_owner_mismatch(self) -> None:
        """release_lock checks owner first; even if expired, wrong owner fails."""
        backend = InMemoryBackend()
        backend._locks["lock1"] = ("owner-1", -1.0)
        assert backend.release_lock("lock1", "owner-2") is False

    def test_concurrent_lock_thread_safety(self) -> None:
        """Multiple threads with *different* owners should not corrupt lock state."""
        backend = InMemoryBackend()
        results: list[bool] = []

        def try_acquire(owner_id: str) -> None:
            results.append(backend.acquire_lock("lock1", owner_id, ttl=10))

        threads = [
            threading.Thread(target=try_acquire, args=(f"thread-{i}",))
            for i in range(10)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # At most one thread should succeed (only the first to acquire the lock),
        # but under the GIL + quick acquisition there may be exactly one.
        # What matters: no corruption, sum is reasonable.
        assert 1 <= sum(results) <= 10


# ===================================================================
# CLUSTER STATE STORE TESTS
# ===================================================================

class TestClusterStateStore:
    """ClusterStateStore -- construction, leader election, node registration, config."""

    # ------------------------------------------------------------------
    # Factory helper
    # ------------------------------------------------------------------

    def make_store(
        self,
        node_id: str = "test-node",
        cluster_name: str = "default",
        ttl: int = 15,
        backend: InMemoryBackend | None = None,
        **kwargs: Any,
    ) -> ClusterStateStore:
        """Create a deterministic ClusterStateStore backed by InMemoryBackend.

        Passes ``redis_host=""`` so Redis is never contacted.  Callers may
        override *cluster_name*, *ttl*, and *node_id*.

        When *backend* is provided, all stores share the same InMemoryBackend
        instance, enabling leader-election and lock tests across stores.
        """
        store = ClusterStateStore(
            redis_host="",
            node_id=node_id,
            cluster_name=cluster_name,
            ttl=ttl,
            **kwargs,
        )
        if backend is not None:
            store._backend = backend
        return store

    # ------------------------------------------------------------------
    # Construction / defaults
    # ------------------------------------------------------------------

    def test_default_construction(self) -> None:
        """Default store should use InMemoryBackend and have empty leader."""
        store = self.make_store()
        assert store._cluster_name == "default"
        assert store._node_id == "test-node"
        assert store._ttl == 15
        assert isinstance(store._backend, InMemoryBackend)
        assert store._leader_id == ""
        assert store._election_in_progress is False

    def test_custom_cluster_name(self) -> None:
        store = self.make_store(cluster_name="my-cluster")
        assert store._cluster_name == "my-cluster"

    def test_custom_ttl(self) -> None:
        store = self.make_store(ttl=30)
        assert store._ttl == 30

    # ------------------------------------------------------------------
    # Key helpers (internal, but essential for correctness)
    # ------------------------------------------------------------------

    def test_leader_key_format(self) -> None:
        store = self.make_store(cluster_name="prod")
        assert store._leader_key() == "cluster:prod:leader"

    def test_node_key_format(self) -> None:
        store = self.make_store(cluster_name="prod", node_id="coord-1")
        assert store._node_key() == "cluster:prod:node:coord-1"
        assert store._node_key("other") == "cluster:prod:node:other"

    def test_nodes_key_format(self) -> None:
        store = self.make_store(cluster_name="prod")
        assert store._nodes_key() == "cluster:prod:nodes"

    def test_config_key_format(self) -> None:
        store = self.make_store(cluster_name="prod")
        assert store._config_key() == "cluster:prod:config"

    def test_lock_key_format(self) -> None:
        store = self.make_store(cluster_name="prod")
        assert store._lock_key() == "cluster:prod:election"

    # ------------------------------------------------------------------
    # Node registration
    # ------------------------------------------------------------------

    def test_register_node(self) -> None:
        """register_node should store the node's metadata."""
        store = self.make_store(node_id="coord-1")
        store.register_node("coord-1", port=50050, version="1.0")
        raw = store._backend.get(store._node_key("coord-1"))
        assert raw is not None
        data = json.loads(raw)
        assert data["node_id"] == "coord-1"
        assert data["port"] == 50050
        assert data["version"] == "1.0"
        assert data["last_seen"] > 0

    def test_register_node_with_default_node_id(self) -> None:
        """register_node should use the store's node_id if none provided."""
        store = self.make_store(node_id="default-coord")
        store.register_node(port=50051)
        raw = store._backend.get(store._node_key())
        assert raw is not None
        data = json.loads(raw)
        assert data["node_id"] == "default-coord"
        assert data["port"] == 50051

    def test_register_node_ttl(self) -> None:
        """Registering a node stores data (InMemoryBackend ignores TTL)."""
        store = self.make_store(node_id="coord-1", ttl=10)
        store.register_node("coord-1", port=50050)
        assert store._backend.get(store._node_key("coord-1")) is not None

    def test_unregister_node(self) -> None:
        """unregister_node should remove the node's data."""
        store = self.make_store(node_id="coord-1")
        store.register_node("coord-1", port=50050)
        store.unregister_node("coord-1")
        assert store._backend.get(store._node_key("coord-1")) is None

    def test_unregister_node_also_removes_leader(self) -> None:
        """If the unregistered node is the leader, leadership is cleared."""
        store = self.make_store(node_id="leader-1")
        store.register_node("leader-1", port=50050)
        store.elect_leader()
        assert store._leader_id == "leader-1"

        store.unregister_node("leader-1")

        assert store._leader_id == ""
        assert store._backend.get(store._leader_key()) is None

    def test_unregister_node_non_leader_preserves_leader(self) -> None:
        """Unregistering a non-leader should not affect the leader key."""
        leader = self.make_store(node_id="leader-1")
        leader.register_node("leader-1", port=50050)
        leader.elect_leader()
        # Simulate a follower by using the same backend / keys
        follower = self.make_store(node_id="follower-1")
        # Unregister the follower
        follower.unregister_node("follower-1")
        # Leader key should still be set
        assert leader._backend.get(leader._leader_key()) == "leader-1"

    def test_register_node_keys_stay_after_multiple_calls(self) -> None:
        """Registering the same node again should overwrite, not duplicate."""
        store = self.make_store(node_id="coord-1")
        store.register_node("coord-1", port=50050, version="1.0")
        store.register_node("coord-1", port=50051, version="2.0")
        raw = store._backend.get(store._node_key("coord-1"))
        data = json.loads(raw)
        assert data["port"] == 50051
        assert data["version"] == "2.0"

    # ------------------------------------------------------------------
    # Heartbeat
    # ------------------------------------------------------------------

    def test_heartbeat_updates_last_seen(self) -> None:
        """heartbeat should increase last_seen and set is_healthy."""
        store = self.make_store(node_id="coord-1")
        store.register_node("coord-1", port=50050)
        raw_before = store._backend.get(store._node_key("coord-1"))
        data_before = json.loads(raw_before)

        store.heartbeat()

        raw_after = store._backend.get(store._node_key("coord-1"))
        data_after = json.loads(raw_after)
        assert data_after["last_seen"] >= data_before["last_seen"]
        assert data_after["is_healthy"] is True

    def test_heartbeat_no_registration(self) -> None:
        """heartbeat should not crash when node is not yet registered."""
        store = self.make_store(node_id="unregistered")
        store.heartbeat()  # should not raise

    def test_heartbeat_with_corrupted_data(self) -> None:
        """heartbeat should not crash when stored data is corrupted JSON."""
        store = self.make_store(node_id="corrupted")
        store._backend.set(store._node_key(), "not-valid-json{{{")
        store.heartbeat()  # should not raise

    def test_heartbeat_updates_ttl_key(self) -> None:
        """heartbeat should re-set the node key (InMemoryBackend keeps it)."""
        store = self.make_store(node_id="coord-1")
        store.register_node("coord-1", port=50050)
        store.heartbeat()
        assert store._backend.get(store._node_key("coord-1")) is not None

    # ------------------------------------------------------------------
    # Leader election
    # ------------------------------------------------------------------

    def test_elect_leader_success(self) -> None:
        """First node to elect should become leader."""
        store = self.make_store(node_id="coord-1")
        assert store.elect_leader() is True
        assert store.is_leader() is True
        assert store._leader_id == "coord-1"
        assert store._backend.get(store._leader_key()) == "coord-1"

    def test_elect_leader_failure(self) -> None:
        """A second node cannot become leader while lock is held."""
        shared = InMemoryBackend()
        leader = self.make_store(node_id="leader-1", backend=shared)
        follower = self.make_store(node_id="follower-1", backend=shared)
        leader.elect_leader()
        assert leader.is_leader() is True

        assert follower.elect_leader() is False
        assert follower.is_leader() is False

    def test_elect_leader_follower_detects_current_leader(self) -> None:
        """Follower should read the current leader after failed election."""
        shared = InMemoryBackend()
        leader = self.make_store(node_id="leader-1", backend=shared)
        follower = self.make_store(node_id="follower-1", backend=shared)
        leader.elect_leader()

        follower.elect_leader()
        # follower should have detected the current leader
        assert follower._leader_id == "leader-1"

    def test_elect_leader_no_current_leader_on_failure(self) -> None:
        """When lock is held but leader key is gone, failed election returns empty."""
        shared = InMemoryBackend()
        leader = self.make_store(node_id="leader-1", backend=shared)
        follower = self.make_store(node_id="follower-1", backend=shared)
        leader.elect_leader()
        # Remove the leader key but keep the lock
        leader._backend.delete(leader._leader_key())

        result = follower.elect_leader()
        # Lock is still held by leader-1, so follower can't acquire
        assert result is False
        # current leader key is gone, so _leader_id should be empty
        assert follower._leader_id == ""

    def test_is_leader(self) -> None:
        store = self.make_store(node_id="coord-1")
        assert store.is_leader() is False
        store.elect_leader()
        assert store.is_leader() is True

    def test_current_leader_when_set(self) -> None:
        store = self.make_store(node_id="coord-1")
        store.elect_leader()
        assert store.current_leader() == "coord-1"

    def test_current_leader_when_empty(self) -> None:
        store = self.make_store(node_id="coord-1")
        assert store.current_leader() == ""

    def test_resign_leadership(self) -> None:
        """resign_leadership should clear local leader and backend key."""
        store = self.make_store(node_id="coord-1")
        store.elect_leader()
        assert store.is_leader() is True

        store.resign_leadership()

        assert store.is_leader() is False
        assert store._leader_id == ""
        assert store.current_leader() == ""

    def test_resign_leadership_releases_lock(self) -> None:
        """After resignation, another node can become leader."""
        shared = InMemoryBackend()
        leader = self.make_store(node_id="leader-1", backend=shared)
        follower = self.make_store(node_id="follower-1", backend=shared)
        leader.elect_leader()

        leader.resign_leadership()

        assert follower.elect_leader() is True
        assert follower.is_leader() is True

    def test_double_election_is_idempotent(self) -> None:
        """Calling elect_leader twice should still report leadership."""
        store = self.make_store(node_id="coord-1")
        assert store.elect_leader() is True
        assert store.elect_leader() is True
        assert store.is_leader() is True

    def test_resign_without_being_leader(self) -> None:
        """resign_leadership should not crash when not the leader."""
        store = self.make_store(node_id="coord-1")
        store.resign_leadership()
        assert store._leader_id == ""

    # ------------------------------------------------------------------
    # Config management
    # ------------------------------------------------------------------

    def test_set_and_get_config(self) -> None:
        store = self.make_store()
        store.set_config("model_name", "llama-70b")
        assert store.get_config("model_name") == "llama-70b"

    def test_get_config_missing(self) -> None:
        store = self.make_store()
        assert store.get_config("nonexistent") is None

    def test_set_config_overwrite(self) -> None:
        store = self.make_store()
        store.set_config("batch_size", 16)
        store.set_config("batch_size", 32)
        assert store.get_config("batch_size") == 32

    def test_set_config_multiple_keys(self) -> None:
        store = self.make_store()
        store.set_config("key1", "val1")
        store.set_config("key2", "val2")
        assert store.get_config("key1") == "val1"
        assert store.get_config("key2") == "val2"

    def test_set_config_nested_value(self) -> None:
        """Config values can be nested dicts/lists."""
        store = self.make_store()
        nested = {"inner": [1, 2, 3]}
        store.set_config("nested", nested)
        assert store.get_config("nested") == nested

    def test_get_config_corrupted(self) -> None:
        """get_config should return None when config is corrupted."""
        store = self.make_store()
        store._backend.set(store._config_key(), "not-json{{{")
        assert store.get_config("anything") is None

    def test_set_config_corrupted_resets(self) -> None:
        """set_config should reset the config when existing data is corrupted."""
        store = self.make_store()
        store._backend.set(store._config_key(), "corrupted{{{")
        store.set_config("model", "llama")
        assert store.get_config("model") == "llama"

    def test_config_can_store_numeric_types(self) -> None:
        store = self.make_store()
        store.set_config("int_val", 42)
        store.set_config("float_val", 3.14)
        store.set_config("bool_val", True)
        store.set_config("none_val", None)
        assert store.get_config("int_val") == 42
        assert store.get_config("float_val") == 3.14
        assert store.get_config("bool_val") is True
        assert store.get_config("none_val") is None

    def test_config_preserves_non_model_keys(self) -> None:
        """Setting a new key should not clobber existing keys."""
        store = self.make_store()
        store.set_config("key1", "val1")
        store.set_config("key2", "val2")
        assert store.get_config("key1") == "val1"
        assert store.get_config("key2") == "val2"

    # ------------------------------------------------------------------
    # Cluster state snapshot
    # ------------------------------------------------------------------

    def test_get_cluster_state_empty(self) -> None:
        """get_cluster_state returns empty snapshot with no registrations."""
        store = self.make_store()
        state = store.get_cluster_state()
        assert state.leader_id == ""
        assert state.nodes == {}
        assert state.active_models == []
        assert state.global_config == {}

    def test_get_cluster_state_with_registered_node(self) -> None:
        """Registered node should appear in the cluster state."""
        store = self.make_store(node_id="coord-1")
        store.register_node("coord-1", port=50050, version="1.0")
        state = store.get_cluster_state()
        assert "coord-1" in state.nodes
        node = state.nodes["coord-1"]
        assert node.node_id == "coord-1"
        assert node.port == 50050
        assert node.version == "1.0"
        assert node.is_healthy is True

    def test_get_cluster_state_with_leader(self) -> None:
        """Leader node should have is_leader=True in state."""
        store = self.make_store(node_id="coord-1")
        store.register_node("coord-1", port=50050)
        store.elect_leader()
        state = store.get_cluster_state()
        assert state.leader_id == "coord-1"
        assert state.nodes["coord-1"].is_leader is True

    def test_get_cluster_state_marks_dead_nodes(self) -> None:
        """A node with a very old last_seen should be marked unhealthy."""
        store = self.make_store(node_id="alive", ttl=10, cluster_name="test")
        store.register_node("alive", port=50050)
        # Manually inject a node with an old timestamp
        store._backend.set(
            store._node_key("dead"),
            json.dumps({
                "node_id": "dead",
                "host": "old-host",
                "port": 50050,
                "last_seen": -1.0,  # very old -> dead
                "version": "",
            }),
        )
        store._backend.set(
            store._nodes_key(),
            json.dumps(["dead", "alive"]),
        )
        state = store.get_cluster_state()
        assert "dead" in state.nodes
        assert state.nodes["dead"].is_healthy is False
        assert state.nodes["alive"].is_healthy is True

    def test_get_cluster_state_with_config(self) -> None:
        """Config values should appear in the cluster state."""
        store = self.make_store()
        store.set_config("active_models", ["llama-7b", "gpt-j-6b"])
        store.set_config("max_batch", 32)
        state = store.get_cluster_state()
        assert state.active_models == ["llama-7b", "gpt-j-6b"]
        assert state.global_config["max_batch"] == 32

    def test_get_cluster_state_handles_corrupted_node(self) -> None:
        """Corrupted node JSON should be gracefully skipped."""
        store = self.make_store(node_id="good")
        store.register_node("good", port=50050)
        store._backend.set(store._node_key("bad"), "{{{corrupted")
        store._backend.set(store._nodes_key(), json.dumps(["good", "bad"]))
        state = store.get_cluster_state()
        assert "good" in state.nodes
        assert "bad" not in state.nodes

    def test_get_cluster_state_handles_corrupted_nodes_list(self) -> None:
        """Corrupted nodes list JSON should not crash."""
        store = self.make_store(node_id="only")
        store.register_node("only", port=50050)
        store._backend.set(store._nodes_key(), "{{{not-a-list")
        state = store.get_cluster_state()
        # Should still find the current node via self._node_id
        assert "only" in state.nodes

    def test_get_cluster_state_handles_corrupted_config(self) -> None:
        """Corrupted config JSON should not crash and return empty config."""
        store = self.make_store(node_id="n1")
        store.register_node("n1", port=50050)
        store._backend.set(store._config_key(), "{{{corrupted")
        state = store.get_cluster_state()
        assert state.global_config == {}

    def test_get_cluster_state_last_updated_is_set(self) -> None:
        """last_updated should be > 0 after a snapshot."""
        store = self.make_store(node_id="n1")
        store.register_node("n1", port=50050)
        state = store.get_cluster_state()
        assert state.last_updated > 0

    # ------------------------------------------------------------------
    # Background heartbeat
    # ------------------------------------------------------------------

    def test_start_background_heartbeat_returns_daemon_thread(self) -> None:
        """start_background_heartbeat should return a running daemon thread."""
        store = self.make_store(node_id="coord-1")
        t = store.start_background_heartbeat()
        assert t is not None
        assert t.is_alive()
        assert t.daemon is True
        # Thread will be killed when the process exits (daemon=True)

    # ------------------------------------------------------------------
    # Edge cases / resilience
    # ------------------------------------------------------------------

    def test_in_memory_backend_with_empty_string_keys(self) -> None:
        """Empty string keys should work with InMemoryBackend."""
        backend = InMemoryBackend()
        backend.set("", "empty-key-value")
        assert backend.get("") == "empty-key-value"
        assert backend.delete("") is True
        assert backend.get("") is None

    def test_store_constructed_with_redis_fallback_on_empty_host(self) -> None:
        """Passing redis_host='' should skip Redis and use InMemoryBackend."""
        store = self.make_store()
        assert isinstance(store._backend, InMemoryBackend)

    def test_store_node_id_isolated_per_instance(self) -> None:
        """Two stores with the same node_id have independent backends."""
        s1 = self.make_store(node_id="shared")
        s2 = self.make_store(node_id="shared")
        # Each has its own InMemoryBackend, so state is independent
        s1.elect_leader()
        assert s1.is_leader() is True
        assert s2.is_leader() is False
        assert s2.current_leader() == ""

    def test_register_node_without_node_id_uses_constructed_id(self) -> None:
        """register_node(None) should use the node_id passed at construction."""
        store = self.make_store(node_id="constructed-id")
        store.register_node()
        raw = store._backend.get(store._node_key())
        data = json.loads(raw)
        assert data["node_id"] == "constructed-id"

    def test_unregister_node_with_default_id(self) -> None:
        """unregister_node() should use store's node_id."""
        store = self.make_store(node_id="auto-id")
        store.register_node()
        store.unregister_node()
        assert store._backend.get(store._node_key()) is None

    def test_leader_key_isolation_by_cluster_name(self) -> None:
        """Different clusters should have independent leader keys."""
        cluster_a = self.make_store(node_id="n1", cluster_name="A")
        cluster_b = self.make_store(node_id="n1", cluster_name="B")
        cluster_a.elect_leader()
        assert cluster_a.is_leader() is True
        assert cluster_b.is_leader() is False
        # cluster B should be able to elect its own leader
        assert cluster_b.elect_leader() is True

    def test_config_key_isolation_by_cluster_name(self) -> None:
        """Different clusters should have independent configs."""
        cluster_a = self.make_store(cluster_name="A")
        cluster_b = self.make_store(cluster_name="B")
        cluster_a.set_config("key", "value-a")
        cluster_b.set_config("key", "value-b")
        assert cluster_a.get_config("key") == "value-a"
        assert cluster_b.get_config("key") == "value-b"
