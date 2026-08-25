"""Kademlia DHT overlay for WAN peer discovery.

Implements the Kademlia Distributed Hash Table protocol for
decentralized peer discovery in the federation layer.

Protocol details:
- 160-bit node IDs (SHA-1 based, the standard Kademlia hash)
- K=20 nodes per k-bucket
- alpha=3 parallel lookup factor
- XOR-distance based routing
- Iterative node lookup with O(log N) hops
- UDP-based RPC: PING, STORE, FIND_NODE, FIND_VALUE
- Periodic bucket refresh for churn resilience
- Bootstrap via seed nodes or relay nodes
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import os
import random
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

# Kademlia protocol constants
K: int = 20  # Bucket size (max nodes per k-bucket)
ALPHA: int = 3  # Parallelism factor for iterative lookups
B: int = 160  # Number of bits in node IDs
BUCKET_REFRESH_INTERVAL: float = 3600.0  # Refresh stale buckets every hour
REPLICATE_INTERVAL: float = 3600.0  # Re-publish values every hour
EXPIRE_TIME: float = 86400.0  # Value expiration (24 hours)
MAX_DATAGRAM_SIZE: int = 1400  # Conservative UDP datagram limit (bytes)
RPC_TIMEOUT: float = 5.0  # Default RPC timeout (seconds)

# Time-bound window for STORE capability tokens (seconds).  A token issued now
# is only accepted for this window, limiting replay.
STORE_TOKEN_TTL: int = 300
# Allowable clock skew between peers when validating a STORE token.
STORE_TOKEN_SKEW: int = 300

# Environment variable carrying the DHT shared secret.  When set, it is used as
# the default ``shared_secret`` for KademliaDHT instances that do not pass one
# explicitly, enabling HMAC-authenticated STORE across the deployment.
DHT_SECRET_ENV_VAR: str = "DISTLLM_DHT_SECRET"


def _generate_node_id(seed: bytes | None = None) -> bytes:
    """Generate a 160-bit node ID, optionally from a seed."""
    if seed is not None:
        # Standard Kademlia uses SHA-1 for 160-bit IDs; collision
        # resistance is not a security requirement of the protocol.
        return hashlib.sha1(seed).digest()  # noqa: DUO124
    return hashlib.sha1(random.randbytes(20)).digest()  # noqa: DUO124


def _xor_distance(a: bytes, b: bytes) -> int:
    """Compute the XOR distance between two 160-bit node IDs."""
    return int.from_bytes(a, "big") ^ int.from_bytes(b, "big")


def _log_distance(a: bytes, b: bytes) -> int:
    """Compute the bucket index, i.e. log2 of the XOR distance."""
    d = _xor_distance(a, b)
    if d == 0:
        return 0
    return d.bit_length() - 1


def _node_id_to_hex(nid: bytes) -> str:
    return nid.hex()


def _hex_to_node_id(hex_str: str) -> bytes:
    return bytes.fromhex(hex_str)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class KademliaNode:
    """A remote node in the DHT overlay network."""

    node_id: bytes
    ip: str
    port: int  # UDP port for DHT RPC
    tcp_port: int = 0  # Optional TCP port for data transfers

    def __hash__(self) -> int:
        return hash(self.node_id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, KademliaNode):
            return NotImplemented
        return self.node_id == other.node_id

    def distance_to(self, other_id: bytes) -> int:
        """XOR distance from this node to ``other_id``."""
        return _xor_distance(self.node_id, other_id)

    @property
    def hex_id(self) -> str:
        return _node_id_to_hex(self.node_id)

    @property
    def address(self) -> tuple[str, int]:
        return (self.ip, self.port)

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.hex_id,
            "ip": self.ip,
            "port": self.port,
            "tcp_port": self.tcp_port,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "KademliaNode":
        return cls(
            node_id=_hex_to_node_id(data["node_id"]),
            ip=data["ip"],
            port=data["port"],
            tcp_port=data.get("tcp_port", 0),
        )


class KBucket:
    """A k-bucket storing up to K nodes with LRU replacement.

    The most recently seen node is at the tail (end); the least recently
    seen is at the head (front). When the bucket is full, new incoming
    nodes displace the head only after the head has been pinged and
    confirmed unresponsive.
    """

    __slots__ = ("range_start", "range_end", "_nodes", "_last_accessed")

    def __init__(self, range_start: int, range_end: int) -> None:
        self.range_start: int = range_start
        self.range_end: int = range_end
        self._nodes: OrderedDict[bytes, KademliaNode] = OrderedDict()
        self._last_accessed: float = time.time()

    # -- read-only properties ------------------------------------------------

    @property
    def nodes(self) -> list[KademliaNode]:
        return list(self._nodes.values())

    @property
    def node_ids(self) -> set[bytes]:
        return set(self._nodes.keys())

    @property
    def is_full(self) -> bool:
        return len(self._nodes) >= K

    @property
    def is_empty(self) -> bool:
        return len(self._nodes) == 0

    def has_node(self, node_id: bytes) -> bool:
        return node_id in self._nodes

    # -- mutations -----------------------------------------------------------

    def touch(self) -> None:
        """Mark the bucket as recently accessed."""
        self._last_accessed = time.time()

    def add_node(self, node: KademliaNode) -> bool:
        """Add or refresh a node.

        Returns True if the node was added or already present and
        refreshed. Returns False when the bucket is full and the
        caller should probe the head before evicting.
        """
        self._last_accessed = time.time()

        if node.node_id in self._nodes:
            # Move to end (most recently seen)
            self._nodes.move_to_end(node.node_id)
            return True

        if not self.is_full:
            self._nodes[node.node_id] = node
            return True

        # Bucket is full — caller must probe the head first
        return False

    def remove_node(self, node_id: bytes) -> None:
        self._nodes.pop(node_id, None)

    def head(self) -> KademliaNode | None:
        """Return the least recently seen node, or None."""
        if not self._nodes:
            return None
        return next(iter(self._nodes.values()))

    def split(self) -> tuple["KBucket", "KBucket"]:
        """Split this bucket into two equal-range halves.

        The caller is responsible for replacing the original bucket
        in the routing table with the two returned buckets.
        """
        mid = (self.range_start + self.range_end) // 2
        left = KBucket(self.range_start, mid)
        right = KBucket(mid + 1, self.range_end)

        for nid, node in self._nodes.items():
            target = left if int.from_bytes(nid, "big") <= mid else right
            target._nodes[nid] = node

        return left, right

    def in_range(self, node_id: bytes) -> bool:
        val = int.from_bytes(node_id, "big")
        return self.range_start <= val <= self.range_end

    def needs_refresh(self) -> bool:
        """True if the bucket has not been touched within the refresh interval."""
        return (time.time() - self._last_accessed) > BUCKET_REFRESH_INTERVAL


class RoutingTable:
    """Manages k-buckets over the full 160-bit address space.

    The address space is partitioned into contiguous k-buckets.
    Buckets that contain the local node's ID are split when they
    exceed K entries, ensuring the local node maintains detailed
    knowledge of nearby address ranges.
    """

    def __init__(self, local_node_id: bytes) -> None:
        self.local_node_id: bytes = local_node_id
        self._buckets: list[KBucket] = [KBucket(0, 2**B - 1)]

    # -- read-only properties ------------------------------------------------

    @property
    def buckets(self) -> list[KBucket]:
        return list(self._buckets)

    @property
    def num_buckets(self) -> int:
        return len(self._buckets)

    @property
    def num_nodes(self) -> int:
        return sum(len(b._nodes) for b in self._buckets)

    def all_nodes(self) -> list[KademliaNode]:
        nodes: list[KademliaNode] = []
        for bucket in self._buckets:
            nodes.extend(bucket.nodes)
        return nodes

    def find_bucket(self, node_id: bytes) -> KBucket | None:
        """Return the bucket whose range contains ``node_id``."""
        for bucket in self._buckets:
            if bucket.in_range(node_id):
                return bucket
        return None

    def insert_node(self, node: KademliaNode) -> bool:
        """Insert a node into the routing table.

        Returns True if the node was inserted. Returns False if the
        node was the local node, the bucket was full and could not
        be split, or the node ID was out of range.
        """
        if node.node_id == self.local_node_id:
            return False

        bucket = self.find_bucket(node.node_id)
        if bucket is None:
            return False

        if bucket.add_node(node):
            return True

        local_val = int.from_bytes(self.local_node_id, "big")
        can_split = (
            bucket.range_start <= local_val <= bucket.range_end
            and (bucket.range_end - bucket.range_start) > 1
        )

        if can_split:
            left, right = bucket.split()
            idx = self._buckets.index(bucket)
            self._buckets[idx : idx + 1] = [left, right]
            new_bucket = self.find_bucket(node.node_id)
            if new_bucket is not None and new_bucket.add_node(node):
                return True

        # Bucket is full and cannot be split — caller should probe
        return False

    def remove_node(self, node_id: bytes) -> None:
        bucket = self.find_bucket(node_id)
        if bucket is not None:
            bucket.remove_node(node_id)

    def find_nearest(self, target_id: bytes, count: int = K) -> list[KademliaNode]:
        """Return the ``count`` closest nodes to ``target_id``.

        Searches across all buckets and sorts by XOR distance.
        """
        all_nodes = self.all_nodes()
        all_nodes.sort(key=lambda n: _xor_distance(n.node_id, target_id))
        return all_nodes[:count]

    def find_nearest_bucket(self, target_id: bytes) -> KBucket:
        """Return the bucket whose range contains ``target_id``."""
        bucket = self.find_bucket(target_id)
        if bucket is not None:
            return bucket
        # Fallback: return the bucket whose range-start is closest
        target_val = int.from_bytes(target_id, "big")
        return min(
            self._buckets,
            key=lambda b: abs(b.range_start - target_val),
        )

    def buckets_needing_refresh(self) -> list[KBucket]:
        """Return all buckets that are past their refresh interval."""
        return [b for b in self._buckets if b.needs_refresh()]


# ---------------------------------------------------------------------------
# Asyncio UDP protocol layer
# ---------------------------------------------------------------------------


class _RPCProtocol(asyncio.DatagramProtocol):
    """Async UDP protocol for Kademlia RPC message exchange.

    Messages are JSON-encoded. Each request carries a random 31-bit
    transaction ID enabling response correlation. Incoming datagrams
    are dispatched to the appropriate handler on the DHT instance.
    """

    def __init__(self, dht: "KademliaDHT") -> None:
        self._dht = dht
        self._transport: asyncio.DatagramTransport | None = None
        self._pending: dict[int, asyncio.Future[tuple[dict, tuple[str, int]]]] = {}
        self._lock = asyncio.Lock()

    # -- transport callbacks ------------------------------------------------

    def connection_made(self, transport: asyncio.DatagramTransport) -> None:  # type: ignore[override]
        self._transport = transport

    def connection_lost(self, exc: Exception | None) -> None:
        if exc is not None:
            logger.error(f"Kademlia UDP transport lost: {exc}")
        # Cancel all pending futures so callers don't hang
        for fut in self._pending.values():
            if not fut.done():
                fut.set_exception(RuntimeError("Transport closed"))
        self._pending.clear()

    def error_received(self, exc: Exception) -> None:
        logger.warning(f"Kademlia UDP error: {exc}")

    def datagram_received(self, data: bytes, addr: tuple[str, int]) -> None:
        try:
            msg: dict[str, Any] = json.loads(data.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            logger.debug(f"Failed to decode Kademlia datagram from {addr}: {exc}")
            return

        tx_id = msg.get("tx_id")
        msg_type = msg.get("type")

        if msg_type == "RESPONSE" and tx_id is not None:
            future = self._pending.pop(tx_id, None)
            if future is not None and not future.done():
                future.set_result((msg, addr))
            return

        # It is a request — dispatch to the appropriate handler
        asyncio.ensure_future(self._dispatch_request(msg, addr))

    # -- request / response -------------------------------------------------

    async def _dispatch_request(self, msg: dict, addr: tuple[str, int]) -> None:
        """Handle an incoming RPC request."""
        rpc_type: str | None = msg.get("rpc")
        tx_id: int | None = msg.get("tx_id")

        if rpc_type is None or tx_id is None:
            return

        handler = {
            "PING": self._dht._handle_ping,
            "STORE": self._dht._handle_store,
            "FIND_NODE": self._dht._handle_find_node,
            "FIND_VALUE": self._dht._handle_find_value,
        }.get(rpc_type)

        if handler is None:
            return

        try:
            result = await handler(msg, addr)
        except Exception as exc:
            logger.warning(f"Error handling {rpc_type} from {addr}: {exc}")
            result = {"error": str(exc)}

        response = {
            "type": "RESPONSE",
            "tx_id": tx_id,
            "rpc": rpc_type,
            "result": result,
        }
        await self._send_to(response, addr)

    async def send_request(
        self,
        rpc_type: str,
        target: tuple[str, int],
        params: dict[str, Any] | None = None,
        timeout: float = RPC_TIMEOUT,
    ) -> dict[str, Any] | None:
        """Send an RPC request and wait for the correlated response."""
        tx_id = random.randint(0, 2**31 - 1)
        msg: dict[str, Any] = {
            "type": "REQUEST",
            "tx_id": tx_id,
            "rpc": rpc_type,
            "params": params or {},
            "sender": {
                "node_id": self._dht.local_node.hex_id,
                "ip": self._dht.local_node.ip,
                "port": self._dht.local_node.port,
            },
        }

        future: asyncio.Future[tuple[dict, tuple[str, int]]] = asyncio.Future()
        async with self._lock:
            self._pending[tx_id] = future

        try:
            await self._send_to(msg, target)
            response, _ = await asyncio.wait_for(future, timeout=timeout)
            # Return the handler result embedded in the response
            return response.get("result") or response
        except asyncio.TimeoutError:
            return None
        except Exception as exc:
            logger.debug(f"RPC {rpc_type} to {target} failed: {exc}")
            return None
        finally:
            async with self._lock:
                self._pending.pop(tx_id, None)

    # -- helpers ------------------------------------------------------------

    async def _send_to(self, msg: dict[str, Any], addr: tuple[str, int]) -> None:
        if self._transport is None:
            return
        data = json.dumps(msg, separators=(",", ":")).encode("utf-8")
        if len(data) > MAX_DATAGRAM_SIZE:
            logger.warning(
                "Kademlia datagram too large (%d bytes) for %s — dropping",
                len(data),
                addr,
            )
            return
        self._transport.sendto(data, addr)


# ---------------------------------------------------------------------------
# Main DHT implementation
# ---------------------------------------------------------------------------


class KademliaDHT:
    """Full Kademlia DHT implementation with async UDP transport.

    Features
    --------
    - PING / STORE / FIND_NODE / FIND_VALUE RPCs per the Kademlia spec
    - Iterative node lookup in O(log N) hops with alpha parallelism
    - Key-value storage with local expiry
    - Bootstrapping from seed nodes
    - Periodic bucket refresh for churn resilience

    Usage
    -----
        dht = KademliaDHT(host="0.0.0.0", port=0,
                          shared_secret="cluster-wide-secret")
        port = await dht.start()
        await dht.bootstrap(seed_nodes)
        nearest = await dht.find_node(target_id)
        value = await dht.find_value("some-key")
        await dht.stop()

    Without ``shared_secret`` (or the ``DISTLLM_DHT_SECRET`` env var) external
    STORE requests are rejected unless ``allow_unauthenticated=True`` is passed
    explicitly.
    """

    def __init__(
        self,
        node_id: bytes | None = None,
        host: str = "0.0.0.0",
        port: int = 0,
        k: int = K,
        alpha: int = ALPHA,
        shared_secret: str | None = None,
        allow_unauthenticated: bool | None = None,
    ) -> None:
        self.local_node = KademliaNode(
            node_id=node_id if node_id is not None else _generate_node_id(),
            ip=host if host != "0.0.0.0" else "127.0.0.1",
            port=port,
        )
        self.k: int = k
        self.alpha: int = alpha

        # Shared secret that authenticates STORE requests (HMAC capability
        # token).  Resolution order: explicit constructor arg > DISTLLM_DHT_SECRET
        # env var > empty (no authentication material).
        env_secret = os.environ.get(DHT_SECRET_ENV_VAR, "")
        if shared_secret is None:
            shared_secret = env_secret
        self._shared_secret = shared_secret

        # Fail-closed default for external STORE requests when no secret is
        # configured: without a secret nothing can be verified, so unauthenticated
        # writes are rejected.  Legacy deployments that genuinely want the old
        # open mode must opt in explicitly via ``allow_unauthenticated=True``.
        # An explicit ``False`` is equivalent to the default but documents intent.
        if allow_unauthenticated is None:
            allow_unauthenticated = False
        self._allow_unauthenticated = bool(allow_unauthenticated)

        if self._shared_secret:
            logger.info(
                "KademliaDHT: STORE requests are authenticated (shared-secret HMAC)"
            )
        elif self._allow_unauthenticated:
            logger.warning(
                "KademliaDHT: no shared secret configured and "
                "allow_unauthenticated=True — STORE requests are UNAUTHENTICATED "
                "and any reachable peer can poison the DHT. This legacy mode is "
                "for development only; set {} to enable store-authorisation.",
                DHT_SECRET_ENV_VAR,
            )
        else:
            logger.warning(
                "KademliaDHT: no shared secret configured — external STORE "
                "requests WITHOUT a valid token will be REJECTED (fail-closed). "
                "Set {} to enable authenticated stores, or pass "
                "allow_unauthenticated=True to restore legacy open behaviour.",
                DHT_SECRET_ENV_VAR,
            )

        self.routing_table = RoutingTable(self.local_node.node_id)

        # Local key-value store: key -> (value_bytes, expiry_timestamp)
        self._store: dict[str, tuple[bytes, float]] = {}

        # UDP protocol / transport (set by start())
        self._protocol: _RPCProtocol | None = None
        self._transport: asyncio.DatagramTransport | None = None

        # Lifecycle
        self._running: bool = False
        self._refresh_task: asyncio.Task[None] | None = None

    # -- lifecycle ----------------------------------------------------------

    async def start(self, bind_addr: str = "0.0.0.0", port: int = 0) -> int:
        """Start the DHT node and bind UDP.

        Returns the actual bound port number.
        """
        loop = asyncio.get_running_loop()
        protocol = _RPCProtocol(self)

        transport, _ = await loop.create_datagram_endpoint(
            lambda: protocol,  # type: ignore[arg-type]
            local_addr=(bind_addr, port),
        )

        self._protocol = protocol
        self._transport = transport
        actual_port: int = transport.get_extra_info("sockname")[1]

        self.local_node.port = actual_port
        if bind_addr != "0.0.0.0":
            self.local_node.ip = bind_addr

        self._running = True
        self._refresh_task = asyncio.create_task(self._refresh_loop())

        logger.info(
            "Kademlia DHT started on {}:{} (node_id={})",
            bind_addr,
            actual_port,
            self.local_node.hex_id,
        )
        return actual_port

    async def stop(self) -> None:
        """Shut down the DHT node cleanly."""
        self._running = False

        if self._refresh_task is not None:
            self._refresh_task.cancel()
            try:
                await self._refresh_task
            except asyncio.CancelledError:
                pass
            self._refresh_task = None

        if self._transport is not None:
            self._transport.close()
            self._transport = None
            self._protocol = None

        logger.info("Kademlia DHT stopped")

    @property
    def is_running(self) -> bool:
        return self._running

    # -- bootstrap ----------------------------------------------------------

    async def bootstrap(self, seed_nodes: list[KademliaNode]) -> int:
        """Bootstrap the routing table by contacting seed nodes.

        Each reachable seed is PING-ed, inserted into the routing
        table, then used as a starting point for an iterative
        FIND_NODE lookup targeting the local node ID.

        Returns the number of seed nodes successfully contacted.
        """
        contacted = 0

        for seed in seed_nodes:
            if seed.node_id == self.local_node.node_id:
                continue

            ok = await self._call_rpc("PING", seed.address)
            if ok is None:
                continue

            self.routing_table.insert_node(seed)
            contacted += 1

            # Use this seed to discover more peers via FIND_NODE on our ID
            await self.iterative_find_node(
                self.local_node.node_id,
                seed_addr=seed.address,
            )

        logger.info(
            "Bootstrapped: contacted {} seed node(s)",
            contacted,
        )
        return contacted

    # -- public RPC helpers -------------------------------------------------

    async def ping(self, node: KademliaNode) -> bool:
        """Ping a remote node and update the routing table on success."""
        result = await self._call_rpc("PING", node.address)
        if result is not None:
            self.routing_table.insert_node(node)
            return True
        return False

    async def store(self, key: str, value: bytes) -> bool:
        """Store a value locally and replicate to the k closest nodes.

        Returns True once the local store has accepted the entry;
        replication is best-effort.
        """
        self._store[key] = (value, time.time() + EXPIRE_TIME)

        key_id = hashlib.sha1(key.encode()).digest()  # noqa: DUO124
        nearest = self.routing_table.find_nearest(key_id, self.k)

        # Time-bound capability token so a peer only stores on behalf of this
        # node while the secret is known and the window is open.
        value_hex = value.hex()
        expires = int(time.time()) + STORE_TOKEN_TTL
        token = self._make_store_token(self.local_node.hex_id, key, value_hex, expires)

        tasks = []
        for node in nearest:
            if node.node_id == self.local_node.node_id:
                continue
            tasks.append(
                self._call_rpc(
                    "STORE",
                    node.address,
                    {
                        "key": key,
                        "value": value_hex,
                        "token": token,
                        "expires": expires,
                    },
                )
            )

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        return True

    async def find_value(self, key: str) -> bytes | list[KademliaNode]:
        """Look up a value stored under ``key``.

        Returns the value bytes if found, otherwise a list of the
        k closest nodes to the key's ID.
        """
        # Check local store first
        local = self._check_local_store(key)
        if local is not None:
            return local

        key_id = hashlib.sha1(key.encode()).digest()  # noqa: DUO124
        await self.iterative_find_node(key_id, lookup_value=True, value_key=key)

        # Re-check local store (iterative lookup may have populated it)
        local = self._check_local_store(key)
        if local is not None:
            return local

        # Fall back to returning the k closest nodes
        return self.routing_table.find_nearest(key_id, self.k)

    async def find_node(self, node_id: bytes) -> list[KademliaNode]:
        """Look up the k closest nodes to ``node_id`` in the network.

        This is the primary mechanism for peer discovery.
        """
        return await self.iterative_find_node(node_id)

    # -- iterative lookup ---------------------------------------------------

    async def iterative_find_node(
        self,
        target_id: bytes,
        seed_addr: tuple[str, int] | None = None,
        lookup_value: bool = False,
        value_key: str | None = None,
    ) -> list[KademliaNode]:
        """Iterative node lookup with alpha parallelism (core Kademlia algorithm).

        Steps:
        1. Seed the shortlist with the alpha closest known nodes.
        2. Send parallel FIND_NODE (or FIND_VALUE) to each unqueried node.
        3. Merge response nodes into the shortlist, keeping the k closest.
        4. Repeat until no closer nodes are found or all closest have been
           queried.
        5. Return the k closest nodes from the shortlist.

        When ``lookup_value`` is True, sends FIND_VALUE instead of FIND_NODE
        and stores any discovered value locally.
        """
        rpc: str = "FIND_VALUE" if lookup_value else "FIND_NODE"

        queried: set[bytes] = set()
        # shortlist entries: (xor_distance, KademliaNode)
        shortlist: list[tuple[int, KademliaNode]] = []

        # Seed from our routing table
        for node in self.routing_table.find_nearest(target_id, self.k):
            d = _xor_distance(node.node_id, target_id)
            shortlist.append((d, node))

        # If a specific seed address was given, create a temporary node
        # entry so we can query it.
        if seed_addr is not None:
            tmp_node = KademliaNode(
                node_id=b"\x00" * 20,
                ip=seed_addr[0],
                port=seed_addr[1],
            )
            shortlist.append((2**B, tmp_node))

        shortlist.sort(key=lambda x: x[0])
        shortlist = shortlist[: self.k]

        while True:
            # Pick up to alpha unqueried nodes from the closest end
            to_query: list[KademliaNode] = []
            for _, node in shortlist:
                if node.node_id not in queried and len(to_query) < self.alpha:
                    to_query.append(node)

            if not to_query:
                break

            for node in to_query:
                queried.add(node.node_id)

            # Fire parallel RPCs
            tasks = []
            for node in to_query:
                params: dict[str, Any] = {}
                if lookup_value and value_key is not None:
                    params["key"] = value_key
                else:
                    params["target"] = _node_id_to_hex(target_id)
                tasks.append(self._call_rpc(rpc, node.address, params))

            results = await asyncio.gather(*tasks, return_exceptions=True)

            found_new = False

            for result in results:
                if isinstance(result, Exception) or result is None:
                    continue

                # FIND_VALUE success — value found
                if isinstance(result, dict) and "value" in result:
                    if value_key is not None:
                        val_info = result["value"]
                        if isinstance(val_info, dict) and "value" in val_info:
                            self._store[value_key] = (
                                bytes.fromhex(val_info["value"]),
                                time.time() + EXPIRE_TIME,
                            )
                            # Found the value; we can continue with the
                            # rest of the iteration but mark it discovered
                        # Fall through: also harvest returned nodes
                    # May also have nodes alongside the value
                    nodes_data = result.get("nodes", [])

                elif isinstance(result, dict):
                    # FIND_NODE response or FIND_VALUE with only nodes
                    nodes_data = result.get("nodes", [])
                else:
                    continue

                # Merge returned nodes into the shortlist
                for nd in nodes_data:
                    if isinstance(nd, dict):
                        node = KademliaNode.from_dict(nd)
                    else:
                        continue

                    if node.node_id == self.local_node.node_id:
                        continue
                    if node.node_id in queried:
                        continue

                    self.routing_table.insert_node(node)

                    d = _xor_distance(node.node_id, target_id)
                    # Avoid duplicates
                    already = any(n.node_id == node.node_id for _, n in shortlist)
                    if not already:
                        shortlist.append((d, node))
                        found_new = True

            # Reshape: keep only the k closest
            shortlist.sort(key=lambda x: x[0])
            shortlist = shortlist[: self.k]

            # Termination condition: no new nodes discovered
            if not found_new:
                break

        return [node for _, node in shortlist[: self.k]]

    # -- RPC handlers -------------------------------------------------------

    async def _handle_ping(self, msg: dict[str, Any], addr: tuple[str, int]) -> dict[str, Any]:
        """Handle incoming PING: return our node ID and register the sender."""
        sender_data = msg.get("sender", {})
        sender_id_hex: str = sender_data.get("node_id", "")
        if sender_id_hex:
            sender = KademliaNode(
                node_id=_hex_to_node_id(sender_id_hex),
                # SECURITY: trust the packet source address (addr), not the
                # sender-declared ip/port — a spoofed datagram must not be able
                # to insert an arbitrary (node_id, victim_ip:port) entry and
                # eclipse the routing table (F-050).
                ip=addr[0],
                port=addr[1],
            )
            self.routing_table.insert_node(sender)
        return {"node_id": self.local_node.hex_id}

    def _make_store_token(self, node_id_hex: str, key: str, value_hex: str, expires: int) -> str:
        """Return a time-bound HMAC capability token for a STORE request.

        Binds the sender identity, key, value, and expiry so a token cannot be
        replayed with different data or by a different node.
        """
        if not self._shared_secret:
            return ""
        raw = f"{node_id_hex}|{key}|{value_hex}|{expires}"
        return hmac.new(self._shared_secret.encode(), raw.encode(), hashlib.sha256).hexdigest()

    def _verify_store_token(
        self, node_id_hex: str, key: str, value_hex: str, token: str, expires: Any
    ) -> bool:
        """Validate a STORE capability token (time-bound, constant-time).

        Fail-closed: without a shared secret there is no authentication
        material, so verification succeeds only in explicit legacy mode
        (``allow_unauthenticated=True``, warned at construction).
        """
        if not self._shared_secret:
            # No secret configured — nothing can be verified.
            return bool(self._allow_unauthenticated)
        try:
            expires_int = int(expires)
        except (TypeError, ValueError):
            return False
        now = int(time.time())
        if expires_int < now - STORE_TOKEN_SKEW or expires_int > now + STORE_TOKEN_SKEW:
            return False
        expected = self._make_store_token(node_id_hex, key, value_hex, expires_int)
        return hmac.compare_digest(expected, token or "")

    async def _handle_store(self, msg: dict[str, Any], addr: tuple[str, int]) -> dict[str, Any]:
        """Handle incoming STORE: persist a key-value pair.

        The request must carry a valid time-bound HMAC token (fail closed) so
        arbitrary peers cannot poison the DHT — both when a shared secret is
        configured and (by default) when it is not.  Legacy open behaviour is
        available only via ``allow_unauthenticated=True``.
        """
        params = msg.get("params", {})
        key: str = params.get("key", "")
        value_hex: str = params.get("value", "")

        # Register the sender in the routing table for future lookups
        sender_data = msg.get("sender", {})
        sender_id_hex: str = sender_data.get("node_id", "")
        if sender_id_hex:
            sender = KademliaNode(
                node_id=_hex_to_node_id(sender_id_hex),
                # SECURITY: trust the packet source address (addr), not the
                # sender-declared ip/port — a spoofed datagram must not be able
                # to insert an arbitrary (node_id, victim_ip:port) entry and
                # eclipse the routing table (F-050).
                ip=addr[0],
                port=addr[1],
            )
            self.routing_table.insert_node(sender)

        if not key or not value_hex:
            return {"stored": False, "error": "missing key or value"}

        # SECURITY: authenticate the STORE with the shared-secret capability.
        if not self._verify_store_token(
            sender_id_hex,
            key,
            value_hex,
            params.get("token", ""),
            params.get("expires"),
        ):
            logger.warning(
                f"KademliaDHT: rejected unauthenticated STORE from {addr} for "
                f"key={key[:24]!r} "
                f"(secret_configured={bool(self._shared_secret)}, "
                f"token_present={bool(params.get('token'))})"
            )
            return {"stored": False, "error": "invalid or expired store token"}

        self._store[key] = (
            bytes.fromhex(value_hex),
            time.time() + EXPIRE_TIME,
        )
        return {"stored": True}

    async def _handle_find_node(self, msg: dict[str, Any], addr: tuple[str, int]) -> dict[str, Any]:
        """Handle incoming FIND_NODE: return k closest nodes to target."""
        params = msg.get("params", {})
        target_hex: str = params.get("target", "")

        # Register sender
        sender_data = msg.get("sender", {})
        sender_id_hex: str = sender_data.get("node_id", "")
        if sender_id_hex:
            sender = KademliaNode(
                node_id=_hex_to_node_id(sender_id_hex),
                # SECURITY: trust the packet source address (addr), not the
                # sender-declared ip/port — a spoofed datagram must not be able
                # to insert an arbitrary (node_id, victim_ip:port) entry and
                # eclipse the routing table (F-050).
                ip=addr[0],
                port=addr[1],
            )
            self.routing_table.insert_node(sender)

        if not target_hex:
            return {"nodes": []}

        target_id = _hex_to_node_id(target_hex)
        nearest = self.routing_table.find_nearest(target_id, self.k)
        return {"nodes": [n.to_dict() for n in nearest]}

    async def _handle_find_value(self, msg: dict[str, Any], addr: tuple[str, int]) -> dict[str, Any]:
        """Handle incoming FIND_VALUE: return value or k closest nodes.

        If we have the value in our local store, return it directly.
        Otherwise, fall through to FIND_NODE behaviour (return nearest
        nodes).
        """
        params = msg.get("params", {})
        key: str = params.get("key", "")

        # Register sender
        sender_data = msg.get("sender", {})
        sender_id_hex: str = sender_data.get("node_id", "")
        if sender_id_hex:
            sender = KademliaNode(
                node_id=_hex_to_node_id(sender_id_hex),
                # SECURITY: trust the packet source address (addr), not the
                # sender-declared ip/port — a spoofed datagram must not be able
                # to insert an arbitrary (node_id, victim_ip:port) entry and
                # eclipse the routing table (F-050).
                ip=addr[0],
                port=addr[1],
            )
            self.routing_table.insert_node(sender)

        # Check local store
        local = self._check_local_store(key)
        if local is not None:
            return {
                "value": {"key": key, "value": local.hex()},
            }

        # Fall through: return nearest nodes
        key_id = hashlib.sha1(key.encode()).digest()  # noqa: DUO124
        nearest = self.routing_table.find_nearest(key_id, self.k)
        return {"nodes": [n.to_dict() for n in nearest]}

    # -- periodic refresh ---------------------------------------------------

    async def _refresh_loop(self) -> None:
        """Periodically refresh stale buckets to maintain routing health."""
        while self._running:
            try:
                await asyncio.sleep(BUCKET_REFRESH_INTERVAL)
                await self._refresh_buckets()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning("Kademlia bucket refresh failed: {}", exc)

    async def _refresh_buckets(self) -> None:
        """Refresh all stale buckets by performing a random-node lookup."""
        stale = self.routing_table.buckets_needing_refresh()
        for bucket in stale:
            # Pick a random ID within this bucket's range
            rand_val = random.randint(bucket.range_start, bucket.range_end)
            rand_id = rand_val.to_bytes(20, "big")
            try:
                await asyncio.wait_for(
                    self.iterative_find_node(rand_id),
                    timeout=RPC_TIMEOUT * 4,
                )
            except asyncio.TimeoutError:
                logger.debug("Bucket refresh lookup timed out")
            bucket.touch()

    # -- helpers ------------------------------------------------------------

    def _check_local_store(self, key: str) -> bytes | None:
        """Return the value for ``key`` from local storage, or None."""
        entry = self._store.get(key)
        if entry is None:
            return None
        value, expiry = entry
        if time.time() < expiry:
            return value
        # Expired
        del self._store[key]
        return None

    async def _call_rpc(
        self,
        rpc_type: str,
        addr: tuple[str, int],
        params: dict[str, Any] | None = None,
        timeout: float = RPC_TIMEOUT,
    ) -> dict[str, Any] | None:
        """Low-level RPC call via the UDP protocol layer."""
        if self._protocol is None:
            return None
        return await self._protocol.send_request(rpc_type, addr, params, timeout)
