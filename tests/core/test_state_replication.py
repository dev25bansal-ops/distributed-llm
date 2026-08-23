"""Tests for state replication -- StateReplicationStore and TopologyStateStore.

Covers:
- StateReplicationStore: memory and file backends
- put/get/delete/list_keys CRUD operations
- Version conflict detection
- get_versioned with metadata
- TopologyStateStore: save/load topology and checkpoints
"""

from __future__ import annotations

import json
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from tests._import_helper import bootstrap_fake_packages, load_module

bootstrap_fake_packages()

_repl = load_module("distllm/core/state_replication.py")
StateReplicationStore = _repl.StateReplicationStore
ReplicatedState = _repl.ReplicatedState
TopologyStateStore = _repl.TopologyStateStore


@dataclass
class _FakeTopologyNode:
    """Deterministic stub for topology nodes used in TopologyStateStore tests."""
    node_id: str = ""
    host: str = "localhost"
    port: int = 50051
    start_layer: int = 0
    end_layer: int = 5
    healthy: bool = True


@dataclass
class _FakeCheckpoint:
    """Deterministic stub for checkpoints used in TopologyStateStore tests."""
    request_id: str = ""
    prompt_tokens: list[int] = field(default_factory=list)
    generated_tokens: list[int] = field(default_factory=list)
    node_id: str = ""


# ======================================================================
# ReplicatedState
# ======================================================================


class TestReplicatedState:
    def test_create(self):
        rs = ReplicatedState(key="k", value="v", version=1)
        assert rs.key == "k"
        assert rs.value == "v"
        assert rs.version == 1

    def test_default_version_zero(self):
        rs = ReplicatedState(key="k", value="v")
        assert rs.version == 0

    def test_default_updated_at_set(self):
        rs = ReplicatedState(key="k", value="v")
        assert rs.updated_at > 0

    def test_default_updated_by_empty(self):
        rs = ReplicatedState(key="k", value="v")
        assert rs.updated_by == ""


# ======================================================================
# StateReplicationStore (memory backend)
# ======================================================================


class TestStateReplicationStoreMemory:
    """Memory backend CRUD operations."""

    @pytest.fixture
    def store(self):
        return StateReplicationStore(backend="memory", node_id="test-node")

    def test_put_and_get(self, store):
        ver = store.put("my-key", {"data": 42})
        assert ver == 1
        assert store.get("my-key") == {"data": 42}

    def test_get_nonexistent(self, store):
        assert store.get("nope") is None

    def test_get_versioned(self, store):
        store.put("k", "v")
        rs = store.get_versioned("k")
        assert isinstance(rs, ReplicatedState)
        assert rs.value == "v"
        assert rs.version == 1
        assert rs.updated_by == "test-node"

    def test_get_versioned_nonexistent(self, store):
        assert store.get_versioned("nope") is None

    def test_put_version_increments(self, store):
        v1 = store.put("k", "a")
        v2 = store.put("k", "b")
        assert v1 == 1
        assert v2 == 2

    def test_put_version_conflict_raises(self, store):
        store.put("k", "first")
        with pytest.raises(ValueError, match="Version conflict"):
            store.put("k", "second", version=99)

    def test_put_version_zero_force_writes(self, store):
        store.put("k", "first")
        ver = store.put("k", "second", version=0)
        assert ver == 2  # force write, but version still increments
        assert store.get("k") == "second"

    def test_delete_existing(self, store):
        store.put("k", "v")
        assert store.delete("k") is True
        assert store.get("k") is None

    def test_delete_nonexistent(self, store):
        assert store.delete("nope") is False

    def test_list_keys(self, store):
        store.put("a", 1)
        store.put("b", 2)
        keys = store.list_keys()
        assert set(keys) == {"a", "b"}

    def test_list_keys_empty(self, store):
        assert store.list_keys() == []

    def test_put_overwrite_updates_value(self, store):
        store.put("k", "v1")
        store.put("k", "v2")
        assert store.get("k") == "v2"

    def test_node_id_recorded(self, store):
        store.put("k", "v")
        rs = store.get_versioned("k")
        assert rs.updated_by == "test-node"


class TestStateReplicationStoreMemoryVersioned:
    """Version conflict scenarios."""

    def test_put_with_correct_version_succeeds(self, store):
        store.put("k", "v1")
        store.put("k", "v2", version=1)  # correct version
        assert store.get("k") == "v2"

    def test_put_version_after_delete(self, store):
        store.put("k", "v1")
        store.delete("k")
        ver = store.put("k", "v2")
        assert ver == 1  # resets since old was deleted

    @pytest.fixture
    def store(self):
        return StateReplicationStore(backend="memory", node_id="t")


# ======================================================================
# StateReplicationStore (file backend)
# ======================================================================


class TestStateReplicationStoreFile:
    """File backend CRUD with atomic writes."""

    @pytest.fixture
    def tmp_path(self):
        with tempfile.TemporaryDirectory() as d:
            yield d

    @pytest.fixture
    def store(self, tmp_path):
        return StateReplicationStore(
            backend="file", path=str(tmp_path), node_id="file-node"
        )

    def test_put_and_get(self, store, tmp_path):
        store.put("key-a", [1, 2, 3])
        assert store.get("key-a") == [1, 2, 3]
        # File should exist on disk
        assert (Path(tmp_path) / "key-a.json").exists()

    def test_get_nonexistent(self, store):
        assert store.get("nope") is None

    def test_delete(self, store, tmp_path):
        store.put("k", "v")
        assert store.delete("k") is True
        assert not (Path(tmp_path) / "k.json").exists()

    def test_delete_nonexistent(self, store):
        assert store.delete("nope") is False

    def test_list_keys(self, store):
        store.put("x", 1)
        store.put("y", 2)
        assert set(store.list_keys()) == {"x", "y"}

    def test_versioning(self, store):
        v1 = store.put("k", "a")
        v2 = store.put("k", "b")
        assert v2 > v1

    def test_version_conflict(self, store):
        store.put("k", "a")
        with pytest.raises(ValueError, match="Version conflict"):
            store.put("k", "b", version=99)

    def test_get_versioned(self, store):
        store.put("k", {"n": 42})
        rs = store.get_versioned("k")
        assert rs.version > 0
        assert rs.updated_by == "file-node"

    def test_file_content_on_disk(self, store, tmp_path):
        store.put("cfg", {"mode": "test"})
        path = Path(tmp_path) / "cfg.json"
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["value"] == {"mode": "test"}
        assert "version" in data
        assert "updated_by" in data

    def test_directory_created(self, tmp_path):
        sub = Path(tmp_path) / "nested" / "dir"
        store = StateReplicationStore(backend="file", path=str(sub))
        assert sub.exists()


# ======================================================================
# TopologyStateStore
# ======================================================================


class TestTopologyStateStore:
    @pytest.fixture
    def store(self):
        inner = StateReplicationStore(backend="memory", node_id="topo-node")
        return TopologyStateStore(inner)

    def _make_node(self, nid, host="localhost", port=50051, start=0, end=5, healthy=True):
        return _FakeTopologyNode(
            node_id=nid, host=host, port=port,
            start_layer=start, end_layer=end, healthy=healthy,
        )

    def test_save_load_topology(self, store):
        nodes = {
            "node-0": self._make_node("node-0", port=50051, end=5),
            "node-1": self._make_node("node-1", port=50052, start=6, end=11),
        }
        ver = store.save_topology(nodes, ["node-0", "node-1"])
        assert ver > 0

        loaded = store.load_topology()
        assert loaded is not None
        assert "nodes" in loaded
        assert "node_order" in loaded
        assert loaded["node_order"] == ["node-0", "node-1"]
        assert "node-0" in loaded["nodes"]

    def test_load_topology_empty(self, store):
        assert store.load_topology() is None

    def test_save_checkpoints(self, store):
        ckpt = _FakeCheckpoint(
            request_id="req-1",
            prompt_tokens=[1, 2, 3],
            generated_tokens=[4, 5],
            node_id="node-0",
        )

        ver = store.save_checkpoints({"req-1": ckpt})
        assert ver > 0

        loaded = store.load_checkpoints()
        assert loaded is not None
        assert "checkpoints" in loaded
        assert "req-1" in loaded["checkpoints"]

    def test_load_checkpoints_empty(self, store):
        assert store.load_checkpoints() is None
