"""Cluster State Store — Redis-based shared state for coordinator HA.

Provides a distributed key-value store with leader election, state
synchronization between coordinators, and health-based failover.
Falls back to in-memory storage when Redis is unavailable.
"""

from __future__ import annotations

import json
import os
import socket
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

from loguru import logger


@dataclass
class ClusterNodeState:
    """State of a single coordinator in the cluster."""
    node_id: str
    host: str
    port: int
    is_leader: bool = False
    is_healthy: bool = True
    last_seen: float = 0.0
    version: str = ""
    active_models: list[str] = field(default_factory=list)
    gpu_info: dict[str, Any] = field(default_factory=dict)
    load_score: float = 0.0


@dataclass
class ClusterState:
    """Complete cluster state snapshot."""
    leader_id: str = ""
    nodes: dict[str, ClusterNodeState] = field(default_factory=dict)
    active_models: list[str] = field(default_factory=list)
    global_config: dict[str, Any] = field(default_factory=dict)
    last_updated: float = 0.0


class StateStoreBackend:
    """Abstract interface for state store backends."""

    def get(self, key: str) -> str | None:
        raise NotImplementedError

    def set(self, key: str, value: str, ttl: int = 0) -> bool:
        raise NotImplementedError

    def delete(self, key: str) -> bool:
        raise NotImplementedError

    def acquire_lock(self, key: str, owner: str, ttl: int = 10) -> bool:
        raise NotImplementedError

    def release_lock(self, key: str, owner: str) -> bool:
        raise NotImplementedError


class InMemoryBackend(StateStoreBackend):
    """Fallback in-memory state store (no Redis)."""

    def __init__(self) -> None:
        self._data: dict[str, str] = {}
        self._locks: dict[str, tuple[str, float]] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> str | None:
        return self._data.get(key)

    def set(self, key: str, value: str, ttl: int = 0) -> bool:
        self._data[key] = value
        return True

    def delete(self, key: str) -> bool:
        return self._data.pop(key, None) is not None

    def acquire_lock(self, key: str, owner: str, ttl: int = 10) -> bool:
        with self._lock:
            now = time.time()
            existing = self._locks.get(key)
            if existing is None or now > existing[1] or existing[0] == owner:
                self._locks[key] = (owner, now + ttl)
                return True
            return False

    def release_lock(self, key: str, owner: str) -> bool:
        with self._lock:
            existing = self._locks.get(key)
            if existing and existing[0] == owner:
                self._locks.pop(key, None)
                return True
            return False


class RedisBackend(StateStoreBackend):
    """Redis-backed state store."""

    def __init__(self, host: str = "localhost", port: int = 6379,
                 password: str | None = None, db: int = 0,
                 socket_timeout: float = 2.0) -> None:
        self._host = host
        self._port = port
        self._password = password
        self._db = db
        self._timeout = socket_timeout
        self._redis: Any = None
        self._connect()

    def _connect(self) -> None:
        try:
            import redis
            self._redis = redis.Redis(
                host=self._host, port=self._port, password=self._password,
                db=self._db, socket_timeout=self._timeout,
                decode_responses=True,
            )
            self._redis.ping()
            logger.info(f"Connected to Redis at {self._host}:{self._port}")
        except ImportError:
            logger.warning("redis-py not installed, falling back to in-memory")
            self._redis = None
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}, falling back to in-memory")
            self._redis = None

    def _ok(self) -> bool:
        if self._redis is None:
            return False
        try:
            self._redis.ping()
            return True
        except Exception as e:
            logger.debug(f"Redis ping failed: {e}")
            self._redis = None
            return False

    def get(self, key: str) -> str | None:
        if not self._ok():
            return None
        try:
            return self._redis.get(key)
        except Exception as e:
            logger.debug(f"Redis GET {key} failed: {e}")
            return None

    def set(self, key: str, value: str, ttl: int = 0) -> bool:
        if not self._ok():
            return False
        try:
            if ttl > 0:
                return bool(self._redis.setex(key, ttl, value))
            return bool(self._redis.set(key, value))
        except Exception as e:
            logger.debug(f"Redis SET {key} failed: {e}")
            return False

    def delete(self, key: str) -> bool:
        if not self._ok():
            return False
        try:
            return bool(self._redis.delete(key))
        except Exception as e:
            logger.debug(f"Redis DEL {key} failed: {e}")
            return False

    def acquire_lock(self, key: str, owner: str, ttl: int = 10) -> bool:
        if not self._ok():
            return False
        try:
            return bool(self._redis.set(f"lock:{key}", owner, nx=True, ex=ttl))
        except Exception as e:
            logger.debug(f"Redis lock acquire {key} failed: {e}")
            return False

    def release_lock(self, key: str, owner: str) -> bool:
        if not self._ok():
            return False
        try:
            current = self._redis.get(f"lock:{key}")
            if current == owner:
                self._redis.delete(f"lock:{key}")
                return True
            return False
        except Exception as e:
            logger.debug(f"Redis lock release {key} failed: {e}")
            return False


class ClusterStateStore:
    """Distributed state store with leader election.

    Usage:
        store = ClusterStateStore(redis_host="10.0.0.1")
        store.register_node("coord-1", 50050)
        store.heartbeat()
        if store.is_leader():
            store.set_config("model", "llama-70b")
    """

    LEADER_TTL: int = 15
    HEARTBEAT_INTERVAL: float = 5.0

    def __init__(
        self,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_password: str | None = None,
        cluster_name: str = "default",
        node_id: str | None = None,
        ttl: int = 15,
    ) -> None:
        self._cluster_name = cluster_name
        self._node_id = node_id or f"{socket.gethostname()}-{os.getpid()}"
        self._ttl = ttl
        self._backend: StateStoreBackend = InMemoryBackend()
        self._election_in_progress = False
        self._lock = threading.Lock()

        # Try Redis first
        if redis_host:
            rb = RedisBackend(redis_host, redis_port, redis_password)
            if rb._redis is not None:
                self._backend = rb

        self._leader_id = self._backend.get(self._leader_key()) or ""

    # ── Key helpers ─────────────────────────────────────────────────────

    def _leader_key(self) -> str:
        return f"cluster:{self._cluster_name}:leader"

    def _node_key(self, node_id: str | None = None) -> str:
        nid = node_id or self._node_id
        return f"cluster:{self._cluster_name}:node:{nid}"

    def _nodes_key(self) -> str:
        return f"cluster:{self._cluster_name}:nodes"

    def _config_key(self) -> str:
        return f"cluster:{self._cluster_name}:config"

    def _lock_key(self) -> str:
        return f"cluster:{self._cluster_name}:election"

    # ── Node registration / heartbeat ───────────────────────────────────

    def register_node(
        self, node_id: str | None = None, port: int = 50050,
        version: str = "",
    ) -> None:
        """Register this coordinator in the cluster state."""
        nid = node_id or self._node_id
        state = ClusterNodeState(
            node_id=nid,
            host=socket.gethostname(),
            port=port,
            last_seen=time.time(),
            version=version,
        )
        self._backend.set(
            self._node_key(nid),
            json.dumps({
                "node_id": nid,
                "host": state.host,
                "port": state.port,
                "last_seen": state.last_seen,
                "version": state.version,
            }, default=str),
            ttl=self._ttl * 2,
        )
        logger.info(f"Registered node {nid} in cluster {self._cluster_name}")

    def heartbeat(self) -> None:
        """Update heartbeat TTL for this node."""
        nid = self._node_id
        raw = self._backend.get(self._node_key(nid))
        if raw:
            try:
                data = json.loads(raw)
                data["last_seen"] = time.time()
                data["is_healthy"] = True
                self._backend.set(
                    self._node_key(nid), json.dumps(data, default=str),
                    ttl=self._ttl * 2,
                )
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning(f"Failed to update heartbeat for {nid}: {e}")

    def unregister_node(self, node_id: str | None = None) -> None:
        nid = node_id or self._node_id
        self._backend.delete(self._node_key(nid))
        if self._leader_id == nid:
            self._backend.delete(self._leader_key())
            self._leader_id = ""
        logger.info(f"Unregistered node {nid}")

    # ── Leader election ─────────────────────────────────────────────────

    def elect_leader(self) -> bool:
        """Attempt to become the cluster leader.

        Returns True if this node is now the leader.
        """
        if self._backend.acquire_lock(
            self._lock_key(), self._node_id, ttl=self._ttl,
        ):
            self._leader_id = self._node_id
            self._backend.set(
                self._leader_key(), self._node_id, ttl=self._ttl * 2,
            )
            logger.info(f"Node {self._node_id} elected as leader")
            return True

        current = self._backend.get(self._leader_key())
        if current:
            self._leader_id = current
        return False

    def is_leader(self) -> bool:
        return self._leader_id == self._node_id

    def current_leader(self) -> str:
        leader = self._backend.get(self._leader_key())
        return leader or ""

    def resign_leadership(self) -> None:
        self._backend.release_lock(self._lock_key(), self._node_id)
        self._backend.delete(self._leader_key())
        self._leader_id = ""

    # ── State management ────────────────────────────────────────────────

    def get_cluster_state(self) -> ClusterState:
        """Return a snapshot of the entire cluster state."""
        nodes: dict[str, ClusterNodeState] = {}
        now = time.time()

        # Get all node keys
        raw_nodes = self._backend.get(self._nodes_key())
        node_keys = []
        if raw_nodes:
            try:
                node_keys = json.loads(raw_nodes)
            except (json.JSONDecodeError, TypeError) as e:
                logger.debug(f"Failed to parse node keys: {e}")

        # Also scan known nodes from our stored key pattern
        # (In Redis we'd use SCAN, but for in-memory we list known)
        for node_id in list(node_keys) + [self._node_id]:
            raw = self._backend.get(self._node_key(node_id))
            if raw:
                try:
                    data = json.loads(raw)
                    is_alive = (now - data.get("last_seen", 0)) < self._ttl * 3
                    nodes[node_id] = ClusterNodeState(
                        node_id=data["node_id"],
                        host=data.get("host", ""),
                        port=data.get("port", 0),
                        is_leader=(node_id == self._leader_id),
                        is_healthy=is_alive,
                        last_seen=data.get("last_seen", 0),
                        version=data.get("version", ""),
                    )
                except (json.JSONDecodeError, KeyError, TypeError) as e:
                    logger.debug(f"Failed to parse node state for {node_id}: {e}")

        raw_config = self._backend.get(self._config_key())
        config = {}
        if raw_config:
            try:
                config = json.loads(raw_config)
            except (json.JSONDecodeError, TypeError) as e:
                logger.debug(f"Failed to parse cluster config: {e}")

        return ClusterState(
            leader_id=self._leader_id,
            nodes=nodes,
            active_models=config.get("active_models", []),
            global_config=config,
            last_updated=time.time(),
        )

    def set_config(self, key: str, value: Any) -> None:
        """Store a cluster-wide config value."""
        raw = self._backend.get(self._config_key()) or "{}"
        try:
            config = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            logger.debug("Cluster config corrupted, resetting")
            config = {}
        config[key] = value
        self._backend.set(self._config_key(), json.dumps(config, default=str))

    def get_config(self, key: str) -> Any:
        raw = self._backend.get(self._config_key())
        if not raw:
            return None
        try:
            config = json.loads(raw)
            return config.get(key)
        except (json.JSONDecodeError, TypeError) as e:
            logger.debug(f"Failed to parse cluster config: {e}")
            return None

    # ── Background heartbeat ────────────────────────────────────────────

    def start_background_heartbeat(self) -> threading.Thread:
        """Start a daemon thread that periodically sends heartbeats."""
        def _loop() -> None:
            while True:
                try:
                    self.heartbeat()
                    if not self._leader_id:
                        self.elect_leader()
                except Exception as e:
                    logger.debug(f"Heartbeat/election failed: {e}")
                time.sleep(self.HEARTBEAT_INTERVAL)

        t = threading.Thread(target=_loop, daemon=True)
        t.start()
        return t
