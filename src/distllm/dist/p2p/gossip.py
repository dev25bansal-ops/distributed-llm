"""Anti-entropy gossip protocol for P2P KV cache discovery.

Nodes periodically exchange cache availability advertisements
to build a distributed index of where cache entries are located.

Upgraded with CRDT semantics for eventual consistency:
- G-Set (grow-only set) for cache entries: entries only added during merge
- LWW-Register (last-writer-wins) for entry metadata: highest timestamp wins
- Vector clocks for causal ordering across nodes
- Tombstones for tracked deletions
"""

from __future__ import annotations
import hashlib
import hmac
import math
import os
import random
import secrets
import struct
import threading
import time
from dataclasses import dataclass, field
from typing import List
from loguru import logger

from distllm.dist.p2p.transport import GossipTransport

def _serialize_for_hmac(data: dict) -> bytes:
    import json
    return json.dumps(data, separators=(",", ":"), sort_keys=True).encode("utf-8")

@dataclass
class VectorClock:
    clocks: dict[str, int] = field(default_factory=dict)

    def increment(self, node_id: str) -> None:
        self.clocks[node_id] = self.clocks.get(node_id, 0) + 1

    def merge(self, other: "VectorClock") -> None:
        for nid, ts in other.clocks.items():
            self.clocks[nid] = max(self.clocks.get(nid, 0), ts)

    def happens_before(self, other: "VectorClock") -> bool:
        has_less = False
        for nid in set(list(self.clocks.keys()) + list(other.clocks.keys())):
            self_val = self.clocks.get(nid, 0)
            other_val = other.clocks.get(nid, 0)
            if self_val > other_val:
                return False
            if self_val < other_val:
                has_less = True
        return has_less

    def is_concurrent(self, other: "VectorClock") -> bool:
        return not self.happens_before(other) and not other.happens_before(self) and self.clocks != other.clocks

@dataclass
class LWWRegister:
    value: str = ""
    timestamp: float = 0.0
    writer_id: str = ""

    def merge(self, other: "LWWRegister") -> None:
        if other.timestamp > self.timestamp:
            self.value = other.value
            self.timestamp = other.timestamp
            self.writer_id = other.writer_id
        elif other.timestamp == self.timestamp and other.writer_id > self.writer_id:
            self.value = other.value
            self.writer_id = other.writer_id

# ---------------------------------------------------------------------------
# Well-known DH parameters (RFC 3526 Group 14 — 2048-bit MODP)
# ---------------------------------------------------------------------------
_DH_PRIME = 0xFFFFFFFFFFFFFFFFC90FDAA22168C234C4C6628B80DC1CD129024E088A67CC74020BBEA63B139B22514A08798E3404DDEF9519B3CD3A431B302B0A6DF25F14374FE1356D6D51C245E485B576625E7EC6F44C42E9A637ED6B0BFF5CB6F406B7EDEE386BFB5A899FA5AE9F24117C4B1FE649286651ECE45B3DC2007CB8A163BF0598DA48361C55D39A69163FA8FD24CF5F83655D23DCA3AD961C62F356208552BB9ED529077096966D670C354E4ABC9804F1746C08CA18217C32905E462E36CE3BE39E772C180E86039B2783A2EC07A28FB5C55DF06F4C52C9DE2BCBF6955817183995497CEA956AE515D2261898FA051015728E5A8AACAA68FFFFFFFFFFFFFFFF  # noqa: E501
_DH_GENERATOR = 2


@dataclass
class PeerMetrics:
    """Per-peer performance tracking for weighted selection."""

    successful_exchanges: int = 0
    total_exchanges: int = 0
    last_rtt: float = 0.0
    avg_rtt: float = 0.0
    joined_at: float = 0.0
    last_success_time: float = 0.0

    @property
    def success_ratio(self) -> float:
        if self.total_exchanges == 0:
            return 0.0
        return self.successful_exchanges / self.total_exchanges

    @property
    def uptime(self) -> float:
        if self.joined_at == 0.0:
            return 0.0
        return time.time() - self.joined_at


class GossipBloomFilter:
    """Compact bloom filter for gossip sync pre-checks.

    Uses multiple hash functions derived from SHA256 with different
    salt prefixes.  The bit array is serialised as a byte string for
    wire transfer.
    """

    def __init__(self, capacity: int = 1000, error_rate: float = 0.01):
        self.capacity = max(1, capacity)
        self.error_rate = error_rate
        self._m = self._optimal_m(self.capacity, error_rate)
        self._k = self._optimal_k(self._m, self.capacity)
        self._bitset = bytearray(self._m)
        self._count = 0

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _optimal_m(n: int, p: float) -> int:
        m = -n * math.log(p) / (math.log(2) ** 2)
        return max(1, int(math.ceil(m)))

    @staticmethod
    def _optimal_k(m: int, n: int) -> int:
        k = (m / n) * math.log(2)
        return max(1, int(math.ceil(k)))

    def _hashes(self, item: str) -> list[int]:
        results: list[int] = []
        for i in range(self._k):
            h = hashlib.sha256(f"{i}:{item}".encode()).digest()
            val = int.from_bytes(h[:8], 'big') % self._m
            results.append(val)
        return results

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def add(self, item: str) -> None:
        for h in self._hashes(item):
            byte_idx = h // 8
            bit_offset = h % 8
            self._bitset[byte_idx] |= (1 << bit_offset)
        self._count += 1

    def contains(self, item: str) -> bool:
        for h in self._hashes(item):
            byte_idx = h // 8
            bit_offset = h % 8
            if not (self._bitset[byte_idx] & (1 << bit_offset)):
                return False
        return True

    def to_bytes(self) -> bytes:
        return bytes(self._bitset)

    def to_dict(self) -> dict:
        return {
            "m": self._m,
            "k": self._k,
            "capacity": self.capacity,
            "error_rate": self.error_rate,
            "bits": self.to_bytes().hex(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "GossipBloomFilter":
        m = data["m"]
        k = data["k"]
        bits_hex = data["bits"]
        bf = cls(capacity=data.get("capacity", m * 8), error_rate=data.get("error_rate", 0.01))
        bf._m = m
        bf._k = k
        bf._bitset = bytearray(bytes.fromhex(bits_hex))
        return bf

    @property
    def size_bytes(self) -> int:
        return self._m

    @property
    def count(self) -> int:
        return self._count


@dataclass
class GossipState:
    node_id: str = ""
    known_peers: set[str] = field(default_factory=set)
    cache_index: dict[str, list[tuple]] = field(default_factory=dict)
    last_exchange_time: float = 0.0
    local_entries: dict[str, str] = field(default_factory=dict)

    vector_clock: VectorClock = field(default_factory=VectorClock)
    entry_metadata: dict[str, LWWRegister] = field(default_factory=dict)
    tombstones: dict[str, float] = field(default_factory=dict)

    page_table_index: dict[str, list[str]] = field(default_factory=dict)
    peer_merkle_roots: dict[str, str] = field(default_factory=dict)
    local_block_hashes: list[str] = field(default_factory=list)

    # O(1) dirty tracking for has_changes_since
    _max_metadata_ts: float = 0.0
    _max_tombstone_ts: float = 0.0

class GossipProtocol:
    def __init__(self, node_id: str, max_peers: int = 16, cache_ttl: float = 300.0,
                 hmac_key: str | None = None,
                 bloom_filter_enabled: bool = True,
                 key_rotation_interval: float = 86400.0,
                 key_overlap_period: float = 3600.0):
        self.state = GossipState(node_id=node_id)
        self.max_peers = max_peers
        self.cache_ttl = cache_ttl
        self.bloom_filter_enabled = bloom_filter_enabled
        self._lock = threading.Lock()

        # --- peer metrics for weighted selection ---
        self._peer_metrics: dict[str, PeerMetrics] = {}

        # --- Diffie-Hellman key state ---
        self._dh_private_key: int = secrets.randbits(2048)
        self._dh_public_key: int = pow(_DH_GENERATOR, self._dh_private_key, _DH_PRIME)
        self._key_rotation_interval: float = key_rotation_interval
        self._key_overlap_period: float = key_overlap_period
        self._last_key_rotation: float = time.time()
        self._peer_hmac_keys: dict[str, str] = {}  # peer_id -> per-peer HMAC key
        self._pending_key_exchanges: dict[str, dict] = {}  # peer_id -> {'dh_pub': int, 'ts': float}
        configured_key = hmac_key or os.environ.get("DISTLLM_GOSSIP_HMAC_KEY")
        self._shared_hmac_key: bool = False
        if not configured_key:
            allow_insecure = (
                os.environ.get("DISTLLM_ALLOW_INSECURE_GOSSIP_KEY") == "1"
                or os.environ.get("DISTLLM_DEV_MODE") == "1"
                or os.environ.get("PYTEST_CURRENT_TEST") is not None
            )
            if not allow_insecure:
                raise ValueError(
                    "GossipProtocol requires a per-deployment HMAC key. "
                    "Set DISTLLM_GOSSIP_HMAC_KEY or pass hmac_key explicitly."
                )
            # SECURITY WARNING: In dev/test mode, each node generates its own
            # HMAC key. This key is NOT shared with peers, so gossip
            # advertisements between nodes are NOT authenticated. The
            # persistent key only survives restarts of the same node.
            configured_key = self._load_or_create_persistent_key()
            logger.warning(
                "Gossip HMAC: No shared key configured. "
                "Gossip advertisements between nodes are NOT authenticated. "
                "Set DISTLLM_GOSSIP_HMAC_KEY to the same value on all nodes "
                "for authenticated gossip. "
                f"Using node-local persistent key at: "
                f"{os.environ.get('DISTLLM_GOSSIP_KEY_FILE', '~/.distllm/gossip_hmac.key')}"
            )
        else:
            # A shared deployment HMAC key is set — never rotate it locally, or
            # peers would stop being able to verify our messages (F-027).
            self._shared_hmac_key = True
        self._hmac_key: str = configured_key

    def _load_or_create_persistent_key(self) -> str:
        """Load a persistent gossip HMAC key from disk, or create one.

        The key is stored in the DistLLM data directory so it survives
        restarts. NOTE: This key is node-local and NOT shared with peers.
        In dev/test mode, gossip between nodes is unauthenticated.
        For production, set DISTLLM_GOSSIP_HMAC_KEY to the same value
        on all nodes.
        """
        key_path = os.environ.get(
            "DISTLLM_GOSSIP_KEY_FILE",
            os.path.join(
                os.environ.get("DISTLLM_DATA_DIR", os.path.expanduser("~/.distllm")),
                "gossip_hmac.key",
            ),
        )
        # Try to load existing key
        if os.path.exists(key_path):
            try:
                with open(key_path, "r") as f:
                    key = f.read().strip()
                    if key:
                        return key
            except (OSError, IOError) as e:
                logger.warning(f"Could not read gossip key from {key_path}: {e}")

        # Generate new persistent key
        key = secrets.token_urlsafe(32)
        try:
            os.makedirs(os.path.dirname(key_path), exist_ok=True)
            with open(key_path, "w") as f:
                f.write(key + "\n")
            logger.info(f"Persistent gossip HMAC key created at {key_path}")
        except (OSError, IOError) as e:
            logger.warning(f"Could not persist gossip key to {key_path}: {e}")
        return key

    def sign_message(self, message: dict) -> dict:
        msg = dict(message)
        body = msg.get("_body", msg)
        serialized = _serialize_for_hmac(body)
        signature = hmac.new(
            self._hmac_key.encode(), msg=serialized, digestmod=hashlib.sha256
        ).hexdigest()
        msg["_hmac"] = signature
        return msg

    def verify_message(self, message: dict) -> bool:
        signature = message.get("_hmac")
        # Guard against non-string signatures: hmac.compare_digest raises
        # TypeError on mixed types, which would turn a malformed/malicious
        # message into an unhandled 500 instead of a clean rejection.
        if not isinstance(signature, str):
            return False
        unsigned = dict(message)
        unsigned.pop("_hmac", None)
        body = unsigned.get("_body", unsigned)
        serialized = _serialize_for_hmac(body)
        expected = hmac.new(
            self._hmac_key.encode(), msg=serialized, digestmod=hashlib.sha256
        ).hexdigest()
        return hmac.compare_digest(signature, expected)

    def add_peer(self, peer_id: str) -> None:
        with self._lock:
            was_new = peer_id not in self.state.known_peers
            self.state.known_peers.add(peer_id)
            if was_new and peer_id not in self._peer_metrics:
                self._peer_metrics[peer_id] = PeerMetrics(joined_at=time.time())
            if len(self.state.known_peers) > self.max_peers:
                excess = self.state.known_peers - set(random.sample(list(self.state.known_peers), self.max_peers))
                self.state.known_peers -= excess

    def remove_peer(self, peer_id: str) -> None:
        with self._lock:
            self.state.known_peers.discard(peer_id)

    def store_local(self, prefix_hash: str, entry_ref: str) -> None:
        with self._lock:
            self.state.vector_clock.increment(self.state.node_id)

            now = time.time()
            if prefix_hash not in self.state.entry_metadata:
                self.state.entry_metadata[prefix_hash] = LWWRegister(
                    value=entry_ref, timestamp=now, writer_id=self.state.node_id
                )
            else:
                reg = self.state.entry_metadata[prefix_hash]
                if now > reg.timestamp:
                    reg.value = entry_ref
                    reg.timestamp = now
                    reg.writer_id = self.state.node_id

            if prefix_hash not in self.state.cache_index:
                self.state.cache_index[prefix_hash] = []
            self.state.cache_index[prefix_hash] = [
                (nid, ref, ts)
                for nid, ref, ts in self.state.cache_index[prefix_hash]
                if nid != self.state.node_id
            ]
            self.state.cache_index[prefix_hash].append(
                (self.state.node_id, entry_ref, now)
            )

            self.state.local_entries[prefix_hash] = entry_ref

            if now > self.state._max_metadata_ts:
                self.state._max_metadata_ts = now

    def advertise(self, delta_only: bool = True) -> dict:
        """Build an advertisement dict for gossip exchange.

        Args:
            delta_only: If True, only include entries changed since the
                last exchange (delta propagation). If False, include all.

        Returns:
            Advertisement dict with node_id, prefixes, metadata, etc.
        """
        now = time.time()

        if delta_only and self.state.last_exchange_time > 0:
            # Only include entries modified since last exchange
            cutoff = self.state.last_exchange_time
            prefixes = []
            entry_metadata = {}
            for prefix_hash, meta in self.state.entry_metadata.items():
                if prefix_hash in self.state.tombstones:
                    continue
                if meta.timestamp > cutoff:
                    prefixes.append(prefix_hash)
                    entry_metadata[prefix_hash] = {
                        "value": meta.value,
                        "timestamp": meta.timestamp,
                        "writer_id": meta.writer_id,
                    }
            tombstones = {
                k: v for k, v in self.state.tombstones.items()
                if v > cutoff
            }
        else:
            # Full state
            prefixes = [
                k for k in self.state.local_entries.keys()
                if k not in self.state.tombstones
            ]
            entry_metadata = {
                k: {"value": v.value, "timestamp": v.timestamp, "writer_id": v.writer_id}
                for k, v in self.state.entry_metadata.items()
            }
            tombstones = dict(self.state.tombstones)

        self.state.last_exchange_time = now

        return {
            "node_id": self.state.node_id,
            "cache_prefixes": prefixes,
            "total_cache_entries": len(prefixes),
            "timestamp": now,
            "vector_clock": dict(self.state.vector_clock.clocks),
            "tombstones": tombstones,
            "entry_metadata": entry_metadata,
            "is_delta": delta_only and len(prefixes) < len(self.state.local_entries),
        }

    def has_changes_since(self, since_time: float) -> bool:
        """Quick check if any entries changed since a timestamp.

        Uses O(1) max-timestamp tracking instead of scanning all entries.
        """
        return max(
            self.state._max_metadata_ts,
            self.state._max_tombstone_ts,
        ) > since_time

    def process_advertisement(self, peer_ad: dict) -> list[str]:
        # Verify HMAC signature before processing any peer data
        if not self.verify_message(peer_ad):
            logger.warning(
                f"Gossip advertisement from {peer_ad.get('node_id', 'unknown')} "
                f"failed HMAC verification — ignoring"
            )
            return []

        peer_id = peer_ad["node_id"]
        peer_prefixes = set(peer_ad["cache_prefixes"])
        local_prefixes = set(self.state.local_entries.keys())

        if "vector_clock" in peer_ad:
            peer_vc = VectorClock(clocks=peer_ad["vector_clock"])
            self.state.vector_clock.merge(peer_vc)

        now = time.time()
        if "tombstones" in peer_ad:
            for prefix_hash, ts in peer_ad["tombstones"].items():
                existing_ts = self.state.tombstones.get(prefix_hash, 0.0)
                if ts > existing_ts:
                    self.state.tombstones[prefix_hash] = ts
                    self.state.local_entries.pop(prefix_hash, None)
                    if ts > self.state._max_tombstone_ts:
                        self.state._max_tombstone_ts = ts

        if "entry_metadata" in peer_ad:
            for prefix_hash, meta_dict in peer_ad["entry_metadata"].items():
                peer_reg = LWWRegister(
                    value=meta_dict["value"],
                    timestamp=meta_dict["timestamp"],
                    writer_id=meta_dict["writer_id"],
                )
                if prefix_hash in self.state.entry_metadata:
                    self.state.entry_metadata[prefix_hash].merge(peer_reg)
                else:
                    self.state.entry_metadata[prefix_hash] = peer_reg
                meta_ts = self.state.entry_metadata[prefix_hash].timestamp
                if meta_ts > self.state._max_metadata_ts:
                    self.state._max_metadata_ts = meta_ts

        for prefix_hash in peer_prefixes:
            if prefix_hash in self.state.tombstones:
                continue
            if prefix_hash not in self.state.cache_index:
                self.state.cache_index[prefix_hash] = []
            already_known = any(
                nid == peer_id for nid, _, _ in self.state.cache_index[prefix_hash]
            )
            if not already_known:
                self.state.cache_index[prefix_hash].append(
                    (peer_id, "", now)
                )

        self.add_peer(peer_id)

        # Periodically prune stale cache_index entries to prevent
        # unbounded memory growth.  Entries that no peer references and
        # whose metadata is older than the TTL are removed.
        if random.random() < 0.01:  # ~1% chance per call
            self._cleanup_cache_index()

        missing = peer_prefixes - local_prefixes
        missing -= set(self.state.tombstones.keys())
        return list(missing)

    def _cleanup_cache_index(self) -> None:
        """Remove stale entries from ``cache_index`` to bound memory growth."""
        now = time.time()
        stale_prefixes = []
        for prefix_hash, refs in list(self.state.cache_index.items()):
            # Entry is stale when no peer references it and metadata is past TTL
            if not refs:
                meta = self.state.entry_metadata.get(prefix_hash)
                if meta is None or (now - meta.timestamp) > self.cache_ttl:
                    stale_prefixes.append(prefix_hash)
            # Remove references from peers that are no longer known
            self.state.cache_index[prefix_hash] = [
                r for r in refs if r[0] in self.state.known_peers
            ]
        for h in stale_prefixes:
            self.state.cache_index.pop(h, None)
            self.state.entry_metadata.pop(h, None)

        # Recalculate max timestamps after removal
        self.state._max_metadata_ts = max(
            (m.timestamp for m in self.state.entry_metadata.values()),
            default=0.0,
        )
        self.state._max_tombstone_ts = max(
            self.state.tombstones.values(),
            default=0.0,
        )

    def build_request(self, target_node_id: str, missing_prefixes: list[str]) -> dict:
        return {
            "requester_id": self.state.node_id,
            "target_node_id": target_node_id,
            "requested_prefixes": missing_prefixes,
        }

    # ------------------------------------------------------------------
    # Signed KV lookup / fetch requests
    #
    # Advertisements have always been HMAC-authenticated, but cache
    # *lookups* (the fetch path that serves actual cache entry data) were
    # not: any peer could enumerate another node's local cache index.
    # The same shared-key scheme used for advertisements covers fetches:
    #   - senders sign via sign_message()/sign_fetch_request()
    #   - receivers verify via authorize_fetch_request() before serving
    #     any cache data (fail-closed when a shared key is configured)
    # ------------------------------------------------------------------

    @property
    def has_shared_hmac_key(self) -> bool:
        """True when a deployment-wide shared HMAC key is configured."""
        return bool(getattr(self, "_shared_hmac_key", False))

    def sign_fetch_request(self, request: dict) -> dict:
        """Sign an outgoing KV lookup/fetch request with the gossip HMAC key.

        Thin alias over :meth:`sign_message` kept for call-site clarity at
        fetch boundaries.  When no key is configured the request is returned
        unsigned (legacy mode; receivers log a loud warning instead of
        rejecting).
        """
        return self.sign_message(request)

    def authorize_fetch_request(self, request: dict | None) -> tuple[bool, str]:
        """Verify an incoming KV lookup/fetch request before serving data.

        Policy:
          - Shared key configured (production): fail closed.  Missing,
            malformed, or invalid ``_hmac`` signatures are rejected.
          - No shared key (legacy dev/test mode): requests are accepted for
            backward compatibility, but a loud one-time warning is logged
            (same convention as ``kademlia_dht.py``'s unauthenticated STORE
            mode).  Node-local keys differ per node, so cross-node
            signatures cannot be verified meaningfully in this mode.

        Returns:
            ``(authorized, reason)`` — reason is "" on success, otherwise
            a short machine-readable rejection cause for logging/HTTP.
        """
        if not self.has_shared_hmac_key:
            # Legacy unauthenticated mode (warned once, below).
            self._warn_legacy_fetch_mode()
            return True, ""
        if request is None:
            return False, "missing_request"
        if not isinstance(request.get("_hmac"), str):
            return False, "missing_signature"
        if not self.verify_message_any_key(request):
            return False, "invalid_signature"
        return True, ""

    def _warn_legacy_fetch_mode(self) -> None:
        """Log the unauthenticated-fetch warning once per process."""
        if getattr(self, "_fetch_warning_logged", False):
            return
        self._fetch_warning_logged = True
        logger.warning(
            "Gossip KV fetch: no shared HMAC key configured — incoming "
            "cache lookup requests are NOT authenticated and any reachable "
            "peer can read this node's cache index. "
            "Set DISTLLM_GOSSIP_HMAC_KEY to the same value on all nodes "
            "for authenticated lookups."
        )

    def process_response(self, response: dict | None) -> int:
        if response is None:
            return 0
        if not response.get("success", False):
            return 0

        # SECURITY (area-analysis H7): fetch responses carry peer-supplied
        # cache-index entries.  Verify the signature whenever one is present
        # and reject tampered payloads outright.  Enforcement of
        # signature *presence* happens at the network boundaries (the
        # /gossip/fetch route and GossipTransport.request_kv_cache), which
        # are the only paths untrusted data can reach this method through;
        # this internal layer stays callable with already-authenticated
        # local data.
        if "_hmac" in response and not self.verify_message_any_key(response):
            logger.warning(
                "Gossip fetch response failed HMAC verification — ignoring "
                f"{len(response.get('cache_entries', {}))} cache entries"
            )
            return 0

        cache_entries = response.get("cache_entries", {})
        count = 0
        now = time.time()

        for prefix_hash, entry_ref in cache_entries.items():
            if prefix_hash not in self.state.cache_index:
                self.state.cache_index[prefix_hash] = []
            updated = False
            for i, (nid, ref, ts) in enumerate(self.state.cache_index[prefix_hash]):
                if ref == "":
                    self.state.cache_index[prefix_hash][i] = (nid, entry_ref, ts)
                    updated = True
                    break
            if not updated:
                self.state.cache_index[prefix_hash].append(
                    ("unknown", entry_ref, now)
                )
            count += 1

        return count

    def select_peer(self) -> str | None:
        if not self.state.known_peers:
            return None
        return random.choice(list(self.state.known_peers))

    def get_peers(self) -> list[str]:
        return list(self.state.known_peers)

    def lookup(self, prefix_hash: str) -> str | None:
        entries = self.state.cache_index.get(prefix_hash)
        if entries:
            entries.sort(key=lambda x: x[2], reverse=True)
            return entries[0][0]
        return None

    def request_cache_from_peers(self, prefix_hash: str, client: "GossipClient | None" = None) -> dict | None:
        if client is None:
            return None
        from concurrent.futures import ThreadPoolExecutor, as_completed

        peers = list(self.state.known_peers)
        if not peers:
            return None

        with ThreadPoolExecutor(max_workers=min(len(peers), 32)) as executor:
            # H-01: Don't cancel futures as it doesn't stop running threads.
            # Instead, return on first successful result; remaining futures
            # run to completion harmlessly in the executor's thread pool.
            futures = {executor.submit(client.fetch_kv_cache, pid, prefix_hash): pid for pid in peers}
            for future in as_completed(futures):
                try:
                    result = future.result(timeout=10)
                    if result is not None:
                        return result
                except Exception:
                    continue
        return None

    def tombstone_entry(self, prefix_hash: str) -> None:
        now = time.time()
        self.state.vector_clock.increment(self.state.node_id)
        existing_ts = self.state.tombstones.get(prefix_hash, 0.0)
        if now > existing_ts:
            self.state.tombstones[prefix_hash] = now
        self.state.local_entries.pop(prefix_hash, None)
        if now > self.state._max_tombstone_ts:
            self.state._max_tombstone_ts = now

    def cleanup_expired(self) -> int:
        now = time.time()
        cutoff = now - self.cache_ttl
        removed = 0

        for prefix_hash in list(self.state.cache_index.keys()):
            self.state.cache_index[prefix_hash] = [
                (nid, ref, ts)
                for nid, ref, ts in self.state.cache_index[prefix_hash]
                if ts >= cutoff
            ]
            if not self.state.cache_index[prefix_hash]:
                del self.state.cache_index[prefix_hash]
                removed += 1

        tombstone_cutoff = now - (self.cache_ttl * 2)
        for prefix_hash in list(self.state.tombstones.keys()):
            if self.state.tombstones[prefix_hash] < tombstone_cutoff:
                del self.state.tombstones[prefix_hash]
                self.state.entry_metadata.pop(prefix_hash, None)

        return removed

    # ------------------------------------------------------------------
    # Weighted peer selection with performance metrics
    # ------------------------------------------------------------------

    def select_peer(self) -> str | None:
        """Weighted-random peer selection preferring high-success, low-RTT, long-lived peers."""
        peers = list(self.state.known_peers)
        if not peers:
            return None
        if len(peers) == 1:
            return peers[0]

        weights: list[float] = []
        now = time.time()
        for peer in peers:
            m = self._peer_metrics.get(peer, PeerMetrics(joined_at=now))

            # success ratio (0..10)
            success_w = m.success_ratio * 10.0

            # RTT: lower is better, 0-10 scale;  200ms RTT → 0
            if m.avg_rtt > 0:
                rtt_w = max(0.0, 10.0 - (m.avg_rtt / 20.0))
            else:
                rtt_w = 5.0  # neutral for untested peers

            # uptime (max 5 at 5+ hours)
            uptime_w = min(5.0, m.uptime / 3600.0)

            weights.append(max(0.1, success_w + rtt_w + uptime_w))

        return random.choices(peers, weights=weights, k=1)[0]

    def update_peer_metrics(self, peer_id: str, success: bool, rtt: float) -> None:
        """Record the outcome of a gossip exchange with *peer_id*."""
        m = self._peer_metrics.setdefault(peer_id, PeerMetrics(joined_at=time.time()))
        m.total_exchanges += 1
        if success:
            m.successful_exchanges += 1
            m.last_success_time = time.time()
        # Exponentially weighted moving average for RTT
        if m.avg_rtt == 0.0:
            m.avg_rtt = rtt
        else:
            m.avg_rtt = 0.7 * m.avg_rtt + 0.3 * rtt
        m.last_rtt = rtt

    def get_peer_metrics(self, peer_id: str) -> PeerMetrics | None:
        return self._peer_metrics.get(peer_id)

    def peer_stats(self) -> dict[str, dict]:
        """Return aggregated peer metrics for observability."""
        return {
            pid: {
                "success_ratio": round(m.success_ratio, 3),
                "avg_rtt_ms": round(m.avg_rtt, 1),
                "total_exchanges": m.total_exchanges,
                "uptime_s": round(m.uptime, 1),
            }
            for pid, m in self._peer_metrics.items()
        }

    # ------------------------------------------------------------------
    # Bloom filter helpers
    # ------------------------------------------------------------------

    def build_bloom_filter(self, capacity: int | None = None) -> dict:
        """Build a bloom filter from local (non-tombstoned) entries.

        Returns a serialised dict suitable for wire transfer.
        """
        items = [
            k for k in self.state.local_entries
            if k not in self.state.tombstones
        ]
        cap = capacity if capacity is not None else max(10, len(items) * 2)
        bf = GossipBloomFilter(capacity=cap)
        for item in items:
            bf.add(item)
        return bf.to_dict()

    def entries_missing_from_bloom(self, bloom_data: dict) -> list[str]:
        """Return the set of local entries *not* covered by *bloom_data*.

        These are entries the remote peer is missing and should fetch.
        """
        bf = GossipBloomFilter.from_dict(bloom_data)
        missing: list[str] = []
        for k in self.state.local_entries:
            if k in self.state.tombstones:
                continue
            if not bf.contains(k):
                missing.append(k)
        return missing

    def handle_bloom_exchange(self, msg: dict) -> dict:
        """Process an incoming bloom-filter-based exchange.

        Returns a response dict containing the cache entries the caller
        is missing (determined by checking the caller's bloom filter
        against *our* local entries).
        """
        if not self.verify_message(msg):
            return {"success": False, "error": "HMAC verification failed", "cache_entries": {}}

        # Merge CRDT metadata carried in the message
        if "vector_clock" in msg:
            peer_vc = VectorClock(clocks=msg["vector_clock"])
            with self._lock:
                self.state.vector_clock.merge(peer_vc)

        if "tombstones" in msg:
            with self._lock:
                for prefix_hash, ts in msg["tombstones"].items():
                    existing_ts = self.state.tombstones.get(prefix_hash, 0.0)
                    if ts > existing_ts:
                        self.state.tombstones[prefix_hash] = ts
                        self.state.local_entries.pop(prefix_hash, None)
                        if ts > self.state._max_tombstone_ts:
                            self.state._max_tombstone_ts = ts

        if "entry_metadata" in msg:
            with self._lock:
                for prefix_hash, meta_dict in msg["entry_metadata"].items():
                    peer_reg = LWWRegister(
                        value=meta_dict["value"],
                        timestamp=meta_dict["timestamp"],
                        writer_id=meta_dict["writer_id"],
                    )
                    if prefix_hash in self.state.entry_metadata:
                        self.state.entry_metadata[prefix_hash].merge(peer_reg)
                    else:
                        self.state.entry_metadata[prefix_hash] = peer_reg
                    meta_ts = self.state.entry_metadata[prefix_hash].timestamp
                    if meta_ts > self.state._max_metadata_ts:
                        self.state._max_metadata_ts = meta_ts

        # Determine which of our entries are missing on the caller
        bloom_data = msg.get("bloom_filter")
        push_entries: dict[str, str] = {}
        if bloom_data:
            bf = GossipBloomFilter.from_dict(bloom_data)
            with self._lock:
                for k, v in self.state.local_entries.items():
                    if k in self.state.tombstones:
                        continue
                    if not bf.contains(k):
                        push_entries[k] = v

        # Build a bloom filter of our entries so the caller can reciprocate
        our_bf = self.build_bloom_filter()
        self.add_peer(msg.get("node_id", ""))

        response = {
            "success": True,
            "cache_entries": push_entries,
            "bloom_filter": our_bf,
            "node_id": self.state.node_id,
        }
        # Sign only when a shared deployment key exists: node-local dev/test
        # keys differ per node, so attaching an unverifiable signature would
        # make every peer drop these entries at verification time.
        if self.has_shared_hmac_key:
            return self.sign_message(response)
        return response

    # ------------------------------------------------------------------
    # DH-based HMAC key distribution & rotation
    # ------------------------------------------------------------------

    def _derive_hmac_key(self, shared_secret: int, salt: str = "gossip-hmac-v1") -> str:
        """Derive a 32-byte hex HMAC key from a DH shared secret."""
        raw = f"{shared_secret}:{salt}".encode()
        return hashlib.sha256(raw).hexdigest()

    def build_key_exchange_request(self) -> dict:
        """Build a DH public-key offer for embedding in gossip messages."""
        return {
            "dh_public_key": self._dh_public_key,
            "dh_group": "rfc3526-2048",
            "key_exchange": True,
        }

    def process_key_exchange(self, peer_id: str, exchange_data: dict) -> dict | None:
        """Handle an incoming key exchange request.

        If the peer offers a DH public key, compute the shared secret,
        derive a per-peer HMAC key, and return our own public key so the
        peer can do the same.

        Returns a response dict, or ``None`` if no key exchange is needed.
        """
        peer_pub = exchange_data.get("dh_public_key")
        if peer_pub is None:
            return None

        try:
            shared = pow(int(peer_pub), self._dh_private_key, _DH_PRIME)
        except (ValueError, TypeError):
            logger.warning(f"DH key exchange with {peer_id}: invalid public key")
            return None

        peer_key = self._derive_hmac_key(shared)
        self._peer_hmac_keys[peer_id] = peer_key
        self._pending_key_exchanges.pop(peer_id, None)
        logger.info(f"DH key exchange completed with {peer_id}")

        # Return our public key so the peer can complete the exchange
        return {
            "dh_public_key": self._dh_public_key,
            "dh_group": "rfc3526-2048",
            "key_exchange_ack": True,
        }

    def complete_key_exchange(self, peer_id: str, response_data: dict) -> bool:
        """Complete a DH key exchange using the peer's response.

        Called after sending a key exchange request and receiving the
        peer's DH public key in the response.
        """
        peer_pub = response_data.get("dh_public_key")
        if peer_pub is None:
            return False

        try:
            shared = pow(int(peer_pub), self._dh_private_key, _DH_PRIME)
        except (ValueError, TypeError):
            logger.warning(f"DH key exchange completion with {peer_id}: invalid public key")
            return False

        peer_key = self._derive_hmac_key(shared)
        self._peer_hmac_keys[peer_id] = peer_key
        self._pending_key_exchanges.pop(peer_id, None)
        logger.info(f"DH key exchange completed with {peer_id}")
        return True

    def get_peer_hmac_key(self, peer_id: str) -> str | None:
        """Return the per-peer HMAC key if one has been established."""
        return self._peer_hmac_keys.get(peer_id)

    def check_key_rotation(self) -> list[str]:
        """Check whether keys need rotation and return list of stale peers.

        Keys older than *key_rotation_interval* are rotated by clearing the
        per-peer key; the next gossip exchange will re-establish it.
        During the overlap period both the old and new key are accepted.
        """
        now = time.time()
        since_last = now - self._last_key_rotation
        if since_last < self._key_rotation_interval:
            return []

        # A deployment-wide shared HMAC key must NOT be rotated to a random
        # node-local value — other nodes don't possess it, so gossip auth would
        # permanently break after the overlap expires (F-027). Only rotate the
        # node-local dev/test persistent key.
        if getattr(self, "_shared_hmac_key", False):
            self._last_key_rotation = now
            return []

        self._last_key_rotation = now
        stale_peers = list(self._peer_hmac_keys.keys())
        # Clear per-peer keys so the next exchange triggers re-key
        self._peer_hmac_keys.clear()
        logger.info(
            f"HMAC key rotation triggered ({since_last / 3600:.1f}h since last rotation); "
            f"{len(stale_peers)} peer keys invalidated"
        )

        old_key = self._hmac_key
        # Generate a new primary key for future exchanges
        self._hmac_key = secrets.token_hex(32)

        # During the overlap period, both keys are accepted.
        # We store the old key as a fallback.
        self._overlap_hmac_key: str | None = old_key
        # The overlap period timer is tracked so we can drop the old key later.
        self._overlap_start: float = now

        return stale_peers

    def verify_message_any_key(self, message: dict) -> bool:
        """Verify HMAC using the current key OR the overlap key.

        This allows messages signed with the previous key to be accepted
        during the rotation overlap period.
        """
        # Try the current key first
        if self.verify_message(message):
            return True

        # Check if we're in the overlap period
        overlap_key = getattr(self, "_overlap_hmac_key", None)
        overlap_start = getattr(self, "_overlap_start", 0.0)
        if overlap_key is not None and (time.time() - overlap_start) < self._key_overlap_period:
            signature = message.get("_hmac")
            if not isinstance(signature, str):
                return False
            unsigned = dict(message)
            unsigned.pop("_hmac", None)
            body = unsigned.get("_body", unsigned)
            serialized = _serialize_for_hmac(body)
            expected = hmac.new(
                overlap_key.encode(), msg=serialized, digestmod=hashlib.sha256
            ).hexdigest()
            return hmac.compare_digest(signature, expected)

        return False

    def _cleanup_overlap_key(self) -> None:
        """Drop the overlap key after the overlap period expires."""
        overlap_start = getattr(self, "_overlap_start", 0.0)
        if overlap_start > 0 and (time.time() - overlap_start) > self._key_overlap_period:
            self._overlap_hmac_key = None
            self._overlap_start = 0.0

    def process_advertisement_any_key(self, peer_ad: dict) -> list[str]:
        """Like process_advertisement but accepts both current and overlap keys."""
        # Try to clear overlap key if period expired
        self._cleanup_overlap_key()

        if self.verify_message_any_key(peer_ad):
            return self.process_advertisement(peer_ad)
        else:
            logger.warning(
                f"Gossip advertisement from {peer_ad.get('node_id', 'unknown')} "
                f"failed HMAC verification (tried current + overlap key) — ignoring"
            )
            return []

    def store_block_hash(self, block_hash: str, node_id: str | None = None) -> None:
        owner = node_id or self.state.node_id
        self.state.vector_clock.increment(self.state.node_id)

        if block_hash not in self.state.page_table_index:
            self.state.page_table_index[block_hash] = []
        if owner not in self.state.page_table_index[block_hash]:
            self.state.page_table_index[block_hash].append(owner)

        if owner == self.state.node_id and block_hash not in self.state.local_block_hashes:
            self.state.local_block_hashes.append(block_hash)

    def store_local_block_hashes(self, block_hashes: List[str]) -> None:
        self.state.local_block_hashes = list(block_hashes)
        self.state.vector_clock.increment(self.state.node_id)

        for bh in block_hashes:
            if bh not in self.state.page_table_index:
                self.state.page_table_index[bh] = []
            if self.state.node_id not in self.state.page_table_index[bh]:
                self.state.page_table_index[bh].append(self.state.node_id)

    def update_merkle_root(self, root_hash: str) -> None:
        self.state.peer_merkle_roots[self.state.node_id] = root_hash

    def build_page_advertisement(self) -> dict:
        state = self.state
        now = time.time()
        state.last_exchange_time = now

        return {
            "node_id": state.node_id,
            "merkle_root": state.peer_merkle_roots.get(state.node_id, ""),
            "block_count": len(state.local_block_hashes),
            "block_hashes_sample": state.local_block_hashes[:100],
            "total_blocks_advertised": len(state.local_block_hashes),
            "page_table_entries": len(state.page_table_index),
            "timestamp": now,
        }

    def process_page_advertisement(self, peer_ad: dict) -> list[str]:
        peer_id = peer_ad["node_id"]
        peer_merkle_root = peer_ad.get("merkle_root", "")

        if peer_merkle_root:
            self.state.peer_merkle_roots[peer_id] = peer_merkle_root

        peer_blocks = peer_ad.get("block_hashes_sample", [])
        local_hashes = set(self.state.local_block_hashes)

        missing: list[str] = []
        for bh in peer_blocks:
            if bh not in self.state.page_table_index:
                self.state.page_table_index[bh] = []
            if peer_id not in self.state.page_table_index[bh]:
                self.state.page_table_index[bh].append(peer_id)

            if bh not in local_hashes and bh not in missing:
                missing.append(bh)

        self.add_peer(peer_id)
        return missing

    def lookup_block(self, block_hash: str) -> list[str]:
        return list(self.state.page_table_index.get(block_hash, []))

    def remove_block_hash(self, block_hash: str) -> None:
        self.state.local_block_hashes = [h for h in self.state.local_block_hashes if h != block_hash]

class GossipClient:
    def __init__(
        self,
        node_id: str = "gossip-client",
        peer_resolver=None,
        transport=None,
        enable_network: bool = True,
        hmac_key: str | None = None,
    ):
        self._peer_resolver = peer_resolver
        self._transport = transport
        if self._transport is None and enable_network and peer_resolver is not None:
            self._transport = GossipTransport(
                node_id=node_id,
                peer_resolver=peer_resolver,
                hmac_key=hmac_key,
            )
        elif transport is not None and hmac_key is not None:
            # Do NOT mutate the injected transport's hmac_key — that would be
            # shared-state mutation across clients.  Log a warning if the
            # transport's key doesn't match so the caller can manage it.
            transport_key = getattr(transport, '_hmac_key', None)
            if transport_key != hmac_key:
                logger.warning(
                    "Injected transport has hmac_key=%s but caller provided "
                    "hmac_key=%s; the transport key is left unchanged.",
                    transport_key, hmac_key,
                )
        self._request_count = 0
        self._response_count = 0

    def exchange(self, peer_id: str, advertisement: dict) -> dict | None:
        self._request_count += 1

        if self._transport is not None:
            result = self._transport.exchange_advertisements(peer_id, advertisement)
            if result:
                self._response_count += 1
            return result

        logger.debug(f"Gossip exchange with {peer_id}: {advertisement.get('total_cache_entries', 0)} entries (stub)")
        return None

    def request_entries(
        self, peer_id: str, request: dict
    ) -> dict:
        self._request_count += 1

        if self._transport is not None:
            prefix_hashes = request.get("requested_prefixes", [])
            result = self._transport.request_kv_cache(peer_id, prefix_hashes)
            if result:
                self._response_count += 1
                return result

        missing = request.get("requested_prefixes", [])
        logger.debug(f"Requested {len(missing)} entries from {peer_id} (stub)")
        return {"success": False, "cache_entries": {}, "entries_returned": 0}

    def fetch_kv_cache(self, peer_id: str, prefix_hash: str) -> dict | None:
        self._request_count += 1

        if self._transport is not None:
            result = self._transport.request_kv_cache(peer_id, [prefix_hash])
            if result and result.get("cache_entries"):
                self._response_count += 1
                return result["cache_entries"].get(prefix_hash)

        return None

    @property
    def stats(self) -> dict:
        transfer_stats = {}
        if self._transport:
            transfer_stats = self._transport.transfer_stats

        return {
            "requests_sent": self._request_count,
            "responses_received": self._response_count,
            "transfer": transfer_stats,
        }

    def close(self) -> None:
        if self._transport is not None:
            if hasattr(self._transport, 'close'):
                self._transport.close()

class GossipReplicator:
    def __init__(
        self,
        protocol: GossipProtocol,
        client: GossipClient,
        interval_s: float = 30.0,
        fanout: int = 3,
        bloom_filter_enabled: bool = True,
        enable_key_rotation: bool = True,
    ):
        self._protocol = protocol
        self._client = client
        self._interval_s = interval_s
        self._fanout = max(1, fanout)
        self._bloom_filter_enabled = bloom_filter_enabled
        self._enable_key_rotation = enable_key_rotation
        self._running = False
        self._thread: threading.Thread | None = None
        self._rounds_completed = 0
        self._last_round_duration = 0.0
        self._last_peer_exchange: dict[str, float] = {}  # peer_id -> last exchange time
        self._stop_event = threading.Event()

        # Wire the protocol's HMAC key into the client's transport so that
        # outgoing gossip advertisements are signed and incoming ones verified.
        if hasattr(self._protocol, '_hmac_key') and self._protocol._hmac_key:
            hmac_key = self._protocol._hmac_key
            if hasattr(self._client, '_transport') and self._client._transport is not None:
                if not getattr(self._client._transport, '_hmac_key', None):
                    self._client._transport._hmac_key = hmac_key

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info(
            f"Gossip replicator started (interval={self._interval_s}s)"
        )

    def stop(self) -> None:
        self._running = False
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
        logger.info("Gossip replicator stopped")

    def _compute_fanout(self) -> int:
        """Adaptive fanout based on cluster size.

        - Small cluster (< 10 nodes):  fanout = max(2, cluster_size - 1)
        - Medium (10 - 100 nodes):     fanout = sqrt(cluster_size)
        - Large (> 100 nodes):          fanout = log2(cluster_size) * 2

        The result is clamped to ``[1, self._fanout]``.
        """
        cluster_size = len(self._protocol.state.known_peers)
        if cluster_size <= 1:
            return 1
        if cluster_size < 10:
            desired = max(2, cluster_size - 1)
        elif cluster_size <= 100:
            desired = max(2, int(math.sqrt(cluster_size)))
        else:
            desired = max(2, int(math.log2(cluster_size) * 2))
        return min(desired, self._fanout)

    def _sync_with_bloom_precheck(self, peer: str, result: dict, t0: float) -> bool:
        """Try a bloom-filter pre-check exchange with *peer*.

        Returns ``True`` if the bloom exchange was used (whether or not it
        produced entries).  Returns ``False`` to tell the caller it should
        fall through to the standard exchange.
        """
        if not self._bloom_filter_enabled:
            return False
        if not self._protocol.bloom_filter_enabled:
            return False

        bloom_data = self._protocol.build_bloom_filter()
        msg: dict = {
            "node_id": self._protocol.state.node_id,
            "type": "bloom_exchange",
            "bloom_filter": bloom_data,
            "entry_metadata": {
                k: {"value": v.value, "timestamp": v.timestamp, "writer_id": v.writer_id}
                for k, v in self._protocol.state.entry_metadata.items()
            },
            "tombstones": dict(self._protocol.state.tombstones),
            "vector_clock": dict(self._protocol.state.vector_clock.clocks),
        }
        msg = self._protocol.sign_message(msg)

        resp = self._client.exchange(peer, msg)
        if resp is None:
            return False  # fall through to standard exchange

        # Check if the peer understood the bloom exchange
        if resp.get("type") == "bloom_exchange" or "cache_entries" in resp:
            # Track exchange time
            now = time.time()
            self._last_peer_exchange[peer] = now
            success = resp.get("success", False)
            self._protocol.update_peer_metrics(peer, success, (now - t0) * 1000)

            # Process any CRDT metadata in the response
            if "entry_metadata" in resp or "tombstones" in resp or "vector_clock" in resp:
                # Reconstruct as a pseudo-advertisement for CRDT merge
                pseudo_ad = {
                    "node_id": resp.get("node_id", peer),
                    "cache_prefixes": [],
                    "entry_metadata": resp.get("entry_metadata", {}),
                    "tombstones": resp.get("tombstones", {}),
                    "vector_clock": resp.get("vector_clock", self._protocol.state.vector_clock.clocks),
                }
                self._protocol.process_advertisement(self._protocol.sign_message(pseudo_ad))

            # Process cache entries from the peer's bloom response
            if success:
                fetched = self._protocol.process_response(resp)
                result["entries_fetched"] += fetched

            result["peers_contacted"].append(peer)
            return True

        return False  # peer didn't understand bloom exchange

    def sync_once(self) -> dict:
        """Run one gossip sync round, contacting up to fanout peers.

        Uses adaptive fanout, delta propagation, a bloom-filter pre-check
        (when enabled), and key rotation checks.
        """
        import time

        t0 = time.time()
        result: dict = {
            "peers_contacted": [],
            "entries_missing": 0,
            "entries_fetched": 0,
            "expired_removed": 0,
            "duration_ms": 0.0,
            "skipped_no_changes": 0,
            "bloom_used": 0,
        }

        result["expired_removed"] = self._protocol.cleanup_expired()

        # Check key rotation once per round
        if self._enable_key_rotation:
            self._protocol._cleanup_overlap_key()
            self._protocol.check_key_rotation()

        # Use adaptive fanout based on current cluster size
        effective_fanout = self._compute_fanout()

        for _ in range(effective_fanout):
            peer = self._protocol.select_peer()
            if peer is None:
                break

            # Pre-check: skip if no changes since last exchange with this peer
            last_exchange = self._last_peer_exchange.get(peer, 0)
            if last_exchange > 0 and not self._protocol.has_changes_since(last_exchange):
                result["skipped_no_changes"] += 1
                continue

            # Try bloom-filter pre-check (fast path)
            if self._sync_with_bloom_precheck(peer, result, t0):
                result["bloom_used"] += 1
                continue

            # Standard exchange (fallback path)
            result["peers_contacted"].append(peer)
            ad = self._protocol.advertise(delta_only=True)

            peer_ad = self._client.exchange(peer, ad)
            if peer_ad is None:
                continue

            # Track last exchange time for this peer
            now = time.time()
            self._last_peer_exchange[peer] = now
            self._protocol.update_peer_metrics(peer, True, (now - t0) * 1000)

            # Accept messages signed with current or overlap key during rotation
            missing = self._protocol.process_advertisement_any_key(peer_ad)
            result["entries_missing"] += len(missing)

            if missing:
                req = self._protocol.build_request(peer, missing)
                resp = self._client.request_entries(peer, req)
                fetched = self._protocol.process_response(resp)
                result["entries_fetched"] += fetched

        self._rounds_completed += 1
        result["duration_ms"] = (time.time() - t0) * 1000
        return result

    def _run_loop(self) -> None:
        while self._running:
            try:
                self.sync_once()
            except Exception as exc:
                logger.warning(f"Gossip sync round failed: {exc}")

            if self._running:
                self._stop_event.wait(timeout=self._interval_s)
                self._stop_event.clear()

    @property
    def stats(self) -> dict:
        return {
            "running": self._running,
            "interval_s": self._interval_s,
            "rounds_completed": self._rounds_completed,
            "last_round_duration_ms": round(self._last_round_duration, 1),
        }
