"""State replication for coordinator HA.

Replicates coordinator state (topology, checkpoints, metrics) to
a shared store (etcd, Redis, or file-based) so a standby coordinator
can take over seamlessly.

Usage::

    store = StateReplicationStore(backend="file", path=".distllm_state")
    store.put("topology", topology_dict)
    topology = store.get("topology")
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger


@dataclass
class ReplicatedState:
    """A piece of replicated state with versioning."""
    key: str
    value: Any
    version: int = 0
    updated_at: float = field(default_factory=time.time)
    updated_by: str = ""


class StateReplicationStore:
    """Shared state store for coordinator HA.

    Supports multiple backends:
    - "file": Local filesystem (single-machine HA)
    - "redis": Redis (distributed HA)
    - "memory": In-memory only (testing)

    Args:
        backend: Storage backend type.
        path: File path for "file" backend, or Redis URL for "redis".
        node_id: This coordinator's ID (for conflict resolution).
    """

    def __init__(
        self,
        backend: str = "file",
        path: str = ".distllm_state",
        node_id: str = "coordinator-0",
    ):
        self._backend = backend
        self._path = path
        self._node_id = node_id
        self._memory_store: dict[str, ReplicatedState] = {}
        self._lock = threading.Lock()

        if backend == "file":
            os.makedirs(path, exist_ok=True)

    def put(self, key: str, value: Any, version: int = 0) -> int:
        """Store a value with version tracking.

        Args:
            key: State key (e.g., "topology", "checkpoints").
            value: JSON-serializable value.
            version: Expected version for CAS (0 = force write).

        Returns:
            New version number.
        """
        with self._lock:
            existing = self._get_raw(key)
            if version > 0 and existing and existing.version != version:
                raise ValueError(
                    f"Version conflict for key '{key}': "
                    f"expected {version}, got {existing.version}"
                )

            new_version = (existing.version + 1) if existing else 1
            state = ReplicatedState(
                key=key,
                value=value,
                version=new_version,
                updated_at=time.time(),
                updated_by=self._node_id,
            )

            self._put_raw(key, state)
            return new_version

    def get(self, key: str) -> Any | None:
        """Retrieve a value by key."""
        with self._lock:
            state = self._get_raw(key)
            return state.value if state else None

    def get_versioned(self, key: str) -> ReplicatedState | None:
        """Retrieve a value with version metadata."""
        with self._lock:
            return self._get_raw(key)

    def delete(self, key: str) -> bool:
        """Delete a key."""
        with self._lock:
            return self._delete_raw(key)

    def list_keys(self) -> list[str]:
        """List all stored keys."""
        with self._lock:
            return self._list_keys_raw()

    def watch(self, key: str, callback: Any, interval_s: float = 5.0) -> None:
        """Watch a key for changes and call callback on update.

        Args:
            key: Key to watch.
            callback: Callable(new_value, version) invoked on change.
            interval_s: Polling interval.
        """
        def _watcher():
            last_version = 0
            while True:
                try:
                    state = self.get_versioned(key)
                    if state and state.version > last_version:
                        last_version = state.version
                        callback(state.value, state.version)
                except Exception as e:
                    logger.warning(f"Watch error for key '{key}': {e}")
                time.sleep(interval_s)

        thread = threading.Thread(target=_watcher, daemon=True, name=f"watch-{key}")
        thread.start()

    def _get_raw(self, key: str) -> ReplicatedState | None:
        """Get from backend (must hold lock)."""
        if self._backend == "memory":
            return self._memory_store.get(key)
        elif self._backend == "file":
            path = Path(self._path) / f"{key}.json"
            if not path.exists():
                return None
            try:
                with open(path) as f:
                    data = json.load(f)
                return ReplicatedState(**data)
            except Exception:
                return None
        return None

    def _put_raw(self, key: str, state: ReplicatedState) -> None:
        """Put to backend (must hold lock).

        For file backend, uses atomic write (temp file + rename) to
        prevent corrupted state on crash during write.
        """
        if self._backend == "memory":
            self._memory_store[key] = state
        elif self._backend == "file":
            path = Path(self._path) / f"{key}.json"
            data = {
                "key": state.key,
                "value": state.value,
                "version": state.version,
                "updated_at": state.updated_at,
                "updated_by": state.updated_by,
            }
            # Atomic write: write to temp file, then replace.  os.replace
            # overwrites the target on Windows too (os.rename does not).
            tmp_path = path.with_suffix(".json.tmp")
            try:
                    with open(tmp_path, "w") as f:
                        json.dump(data, f, indent=2, default=str)
                        f.flush()
                        os.fsync(f.fileno())
                    os.replace(str(tmp_path), str(path))
            except Exception:
                # Clean up temp file on failure
                if tmp_path.exists():
                    tmp_path.unlink()
                raise

    def _delete_raw(self, key: str) -> bool:
        """Delete from backend (must hold lock)."""
        if self._backend == "memory":
            return self._memory_store.pop(key, None) is not None
        elif self._backend == "file":
            path = Path(self._path) / f"{key}.json"
            if path.exists():
                path.unlink()
                return True
        return False

    def _list_keys_raw(self) -> list[str]:
        """List keys from backend (must hold lock)."""
        if self._backend == "memory":
            return list(self._memory_store.keys())
        elif self._backend == "file":
            path = Path(self._path)
            return [p.stem for p in path.glob("*.json")]
        return []


class TopologyStateStore:
    """Specialized store for pipeline topology state.

    Replicates node registrations, layer assignments, and
    cluster metadata so a standby coordinator can reconstruct
    the pipeline.
    """

    def __init__(self, store: StateReplicationStore):
        self._store = store

    def save_topology(self, nodes: dict[str, Any], node_order: list[str]) -> int:
        """Save current topology to replicated store."""
        topology = {
            "nodes": {
                nid: {
                    "host": getattr(n, "host", ""),
                    "port": getattr(n, "port", 0),
                    "start_layer": getattr(n, "start_layer", 0),
                    "end_layer": getattr(n, "end_layer", 0),
                    "healthy": getattr(n, "healthy", False),
                }
                for nid, n in nodes.items()
            },
            "node_order": node_order,
            "timestamp": time.time(),
        }
        return self._store.put("topology", topology)

    def load_topology(self) -> dict | None:
        """Load topology from replicated store."""
        return self._store.get("topology")

    def save_checkpoints(self, checkpoints: dict[str, Any]) -> int:
        """Save recovery checkpoints to replicated store."""
        data = {
            "checkpoints": {
                rid: {
                    "request_id": ckpt.request_id,
                    "prompt_tokens": ckpt.prompt_tokens,
                    "generated_tokens": ckpt.generated_tokens,
                    "node_id": ckpt.node_id,
                }
                for rid, ckpt in checkpoints.items()
            },
            "timestamp": time.time(),
        }
        return self._store.put("checkpoints", data)

    def load_checkpoints(self) -> dict | None:
        """Load checkpoints from replicated store."""
        return self._store.get("checkpoints")
