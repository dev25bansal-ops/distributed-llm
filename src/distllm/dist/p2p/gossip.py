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
import os
import random
import secrets
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

class GossipProtocol:
    def __init__(self, node_id: str, max_peers: int = 16, cache_ttl: float = 300.0, hmac_key: str | None = None):
        self.state = GossipState(node_id=node_id)
        self.max_peers = max_peers
        self.cache_ttl = cache_ttl
        self._lock = threading.Lock()
        configured_key = hmac_key or os.environ.get("DISTLLM_GOSSIP_HMAC_KEY")
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
        if signature is None:
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
            self.state.known_peers.add(peer_id)
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

        Used as a Bloom-filter-like pre-check to skip unnecessary exchanges.
        """
        for meta in self.state.entry_metadata.values():
            if meta.timestamp > since_time:
                return True
        for ts in self.state.tombstones.values():
            if ts > since_time:
                return True
        return False

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

    def build_request(self, target_node_id: str, missing_prefixes: list[str]) -> dict:
        return {
            "requester_id": self.state.node_id,
            "target_node_id": target_node_id,
            "requested_prefixes": missing_prefixes,
        }

    def process_response(self, response: dict) -> int:
        if not response.get("success", False):
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
            # Ensure the injected transport has an HMAC key for signing
            if not getattr(transport, '_hmac_key', None):
                transport._hmac_key = hmac_key
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
    ):
        self._protocol = protocol
        self._client = client
        self._interval_s = interval_s
        self._fanout = max(1, fanout)
        self._running = False
        self._thread: threading.Thread | None = None
        self._rounds_completed = 0
        self._last_round_duration = 0.0
        self._last_peer_exchange: dict[str, float] = {}  # peer_id -> last exchange time

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
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
        logger.info("Gossip replicator stopped")

    def sync_once(self) -> dict:
        """Run one gossip sync round, contacting up to fanout peers.

        Uses delta propagation (only changed entries) and a pre-check
        to skip unnecessary exchanges when nothing has changed.
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
        }

        result["expired_removed"] = self._protocol.cleanup_expired()

        # Contact up to fanout peers per round for faster convergence
        for _ in range(self._fanout):
            peer = self._protocol.select_peer()
            if peer is None:
                break

            # Pre-check: skip if no changes since last exchange with this peer
            last_exchange = getattr(self, '_last_peer_exchange', {}).get(peer, 0)
            if last_exchange > 0 and not self._protocol.has_changes_since(last_exchange):
                result["skipped_no_changes"] += 1
                continue

            result["peers_contacted"].append(peer)
            ad = self._protocol.advertise(delta_only=True)

            peer_ad = self._client.exchange(peer, ad)
            if peer_ad is None:
                continue

            # Track last exchange time for this peer
            self._last_peer_exchange[peer] = time.time()

            missing = self._protocol.process_advertisement(peer_ad)
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
                import time
                deadline = time.time() + self._interval_s
                while self._running and time.time() < deadline:
                    time.sleep(0.1)

    @property
    def stats(self) -> dict:
        return {
            "running": self._running,
            "interval_s": self._interval_s,
            "rounds_completed": self._rounds_completed,
            "last_round_duration_ms": round(self._last_round_duration, 1),
        }
