"""Anti-entropy gossip protocol for P2P KV cache discovery.

Nodes periodically exchange cache availability advertisements
to build a distributed index of where cache entries are located.

Upgraded with CRDT semantics for eventual consistency:
- G-Set (grow-only set) for cache entries: entries only added during merge
- LWW-Register (last-writer-wins) for entry metadata: highest timestamp wins
- Vector clocks for causal ordering across nodes
- Tombstones for tracked deletions
"""

import hashlib
import hmac
import os
import random
import secrets
import time
from dataclasses import dataclass, field
from typing import List
from loguru import logger


def _serialize_for_hmac(data: dict) -> bytes:
    """Deterministically serialize a dict for HMAC signing."""
    import json
    return json.dumps(data, separators=(",", ":"), sort_keys=True).encode("utf-8")


@dataclass
class VectorClock:
    """Vector clock for causal ordering across nodes.

    Each node maintains a counter that increments on local writes.
    The vector clock is the map of node_id -> counter value.
    """
    clocks: dict[str, int] = field(default_factory=dict)

    def increment(self, node_id: str) -> None:
        """Increment the clock for a specific node."""
        self.clocks[node_id] = self.clocks.get(node_id, 0) + 1

    def merge(self, other: "VectorClock") -> None:
        """Merge with another vector clock (element-wise max)."""
        for nid, ts in other.clocks.items():
            self.clocks[nid] = max(self.clocks.get(nid, 0), ts)

    def happens_before(self, other: "VectorClock") -> bool:
        """Check if this clock causally happens-before another."""
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
        """Check if two clocks are concurrent (neither happens-before)."""
        return not self.happens_before(other) and not other.happens_before(self) and self.clocks != other.clocks


@dataclass
class LWWRegister:
    """Last-Writer-Wins Register for entry metadata.

    When conflicting values exist, the one with the highest timestamp wins.
    """
    value: str = ""
    timestamp: float = 0.0
    writer_id: str = ""

    def merge(self, other: "LWWRegister") -> None:
        """Merge with another register using LWW semantics."""
        if other.timestamp > self.timestamp:
            self.value = other.value
            self.timestamp = other.timestamp
            self.writer_id = other.writer_id
        elif other.timestamp == self.timestamp and other.writer_id > self.writer_id:
            # Tie-break by writer_id (lexicographic)
            self.value = other.value
            self.writer_id = other.writer_id


@dataclass
class GossipState:
    """Internal state for the gossip protocol with CRDT semantics."""
    node_id: str = ""
    known_peers: set[str] = field(default_factory=set)
    # prefix_hash -> list of (node_id, entry_ref, timestamp)
    cache_index: dict[str, list[tuple]] = field(default_factory=dict)
    last_exchange_time: float = 0.0
    # Local cache advertisements: prefix_hash -> entry_ref
    local_entries: dict[str, str] = field(default_factory=dict)

    # CRDT state
    vector_clock: VectorClock = field(default_factory=VectorClock)
    # LWW registers for entry metadata: prefix_hash -> LWWRegister
    entry_metadata: dict[str, LWWRegister] = field(default_factory=dict)
    # Tombstones for deletions: prefix_hash -> timestamp
    tombstones: dict[str, float] = field(default_factory=dict)

    # Distributed PagedAttention: block-level page table sharing
    # block_hash -> list of node_ids that have this block
    page_table_index: dict[str, list[str]] = field(default_factory=dict)
    # node_id -> Merkle root hash of their page table
    peer_merkle_roots: dict[str, str] = field(default_factory=dict)
    # Local block hashes (for advertisement building)
    local_block_hashes: list[str] = field(default_factory=list)


class GossipProtocol:
    """Anti-entropy gossip protocol for distributed cache discovery.

    Each node periodically:
    1. Builds an advertisement of its local cache entries
    2. Sends it to a random peer
    3. Receives the peer's advertisement
    4. Merges missing entries into its index
    5. Requests missing cache entries from the peer
    """

    def __init__(self, node_id: str, max_peers: int = 16, cache_ttl: float = 300.0, hmac_key: str | None = None):
        self.state = GossipState(node_id=node_id)
        self.max_peers = max_peers
        self.cache_ttl = cache_ttl
        # HMAC key for message authentication; auto-generated if not provided
        self._hmac_key: str = hmac_key or secrets.token_hex(32)

    def sign_message(self, message: dict) -> dict:
        """Sign a gossip message with HMAC-SHA256 for authenticity.

        Args:
            message: The message dict to sign.

        Returns:
            Copy of the message with an HMAC signature added.
        """
        msg = dict(message)
        body = msg.get("_body", msg)  # sign the actual content
        serialized = _serialize_for_hmac(body)
        signature = hmac.new(
            self._hmac_key.encode(), msg=serialized, digestmod=hashlib.sha256
        ).hexdigest()
        msg["_hmac"] = signature
        return msg

    def verify_message(self, message: dict) -> bool:
        """Verify the HMAC signature on a gossip message.

        Args:
            message: The received message dict with '_hmac' field.

        Returns:
            True if signature is valid, False otherwise.
        """
        signature = message.pop("_hmac", None)
        if signature is None:
            return False
        body = message.get("_body", message)
        serialized = _serialize_for_hmac(body)
        expected = hmac.new(
            self._hmac_key.encode(), msg=serialized, digestmod=hashlib.sha256
        ).hexdigest()
        return hmac.compare_digest(signature, expected)

    def add_peer(self, peer_id: str) -> None:
        """Add a known peer to the gossip network."""
        self.state.known_peers.add(peer_id)
        if len(self.state.known_peers) > self.max_peers:
            # Evict a random peer
            excess = self.state.known_peers - set(random.sample(list(self.state.known_peers), self.max_peers))
            self.state.known_peers -= excess

    def remove_peer(self, peer_id: str) -> None:
        """Remove a peer from the gossip network."""
        self.state.known_peers.discard(peer_id)

    def store_local(self, prefix_hash: str, entry_ref: str) -> None:
        """Record a local cache entry.

        Uses CRDT semantics: increments vector clock, creates LWW register,
        and adds to G-Set cache index.
        """
        # Increment vector clock for this write
        self.state.vector_clock.increment(self.state.node_id)

        # Create/update LWW register for metadata
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

        # G-Set: add to cache index (grow-only, no removal during merge)
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

    def advertise(self) -> dict:
        """Build a gossip advertisement with CRDT state.

        Returns:
            Dict with node_id, cache_prefixes, total_cache_entries, timestamp,
            vector_clock, tombstones, and entry_metadata.
        """
        now = time.time()
        self.state.last_exchange_time = now

        # Filter out expired and tombstoned entries
        cutoff = now - self.cache_ttl
        prefixes = []
        for prefix_hash, entry_ref in self.state.local_entries.items():
            # Skip tombstoned entries
            if prefix_hash in self.state.tombstones:
                continue
            prefixes.append(prefix_hash)

        return {
            "node_id": self.state.node_id,
            "cache_prefixes": prefixes,
            "total_cache_entries": len(prefixes),
            "timestamp": now,
            # CRDT state
            "vector_clock": dict(self.state.vector_clock.clocks),
            "tombstones": dict(self.state.tombstones),
            "entry_metadata": {
                k: {"value": v.value, "timestamp": v.timestamp, "writer_id": v.writer_id}
                for k, v in self.state.entry_metadata.items()
            },
        }

    def process_advertisement(self, peer_ad: dict) -> list[str]:
        """Process a peer's advertisement with CRDT merge semantics.

        Merges using:
        1. Vector clock merge (element-wise max) for causal ordering
        2. Tombstone merge: entries tombstoned by peer are marked locally
        3. LWW-Register merge: highest timestamp wins for metadata
        4. G-Set merge: peer's cache entries are added (grow-only)

        Args:
            peer_ad: Advertisement dict from a peer.

        Returns:
            List of prefix hashes that are missing locally.
        """
        peer_id = peer_ad["node_id"]
        peer_prefixes = set(peer_ad["cache_prefixes"])
        local_prefixes = set(self.state.local_entries.keys())

        # 1. Merge vector clocks (causal ordering)
        if "vector_clock" in peer_ad:
            peer_vc = VectorClock(clocks=peer_ad["vector_clock"])
            self.state.vector_clock.merge(peer_vc)

        # 2. Merge tombstones (tombstones are also G-Set: only grow)
        now = time.time()
        if "tombstones" in peer_ad:
            for prefix_hash, ts in peer_ad["tombstones"].items():
                # LWW for tombstones: keep the latest tombstone timestamp
                existing_ts = self.state.tombstones.get(prefix_hash, 0.0)
                if ts > existing_ts:
                    self.state.tombstones[prefix_hash] = ts
                    # Remove from local entries if tombstoned
                    self.state.local_entries.pop(prefix_hash, None)

        # 3. Merge LWW registers for entry metadata
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

        # 4. G-Set merge: add peer's entries to cache index (grow-only)
        for prefix_hash in peer_prefixes:
            # Skip tombstoned entries
            if prefix_hash in self.state.tombstones:
                continue
            if prefix_hash not in self.state.cache_index:
                self.state.cache_index[prefix_hash] = []
            # Check if this peer's entry already exists (avoid duplicates)
            already_known = any(
                nid == peer_id for nid, _, _ in self.state.cache_index[prefix_hash]
            )
            if not already_known:
                self.state.cache_index[prefix_hash].append(
                    (peer_id, "", now)  # ref filled in by GossipResponse
                )

        # Add peer to known peers
        self.add_peer(peer_id)

        # Return missing prefixes (not in local entries and not tombstoned)
        missing = peer_prefixes - local_prefixes
        missing -= set(self.state.tombstones.keys())
        return list(missing)

    def build_request(self, target_node_id: str, missing_prefixes: list[str]) -> dict:
        """Build a gossip request for missing entries.

        Args:
            target_node_id: The peer to request from.
            missing_prefixes: Prefix hashes to request.

        Returns:
            Dict with requester_id, target_node_id, requested_prefixes.
        """
        return {
            "requester_id": self.state.node_id,
            "target_node_id": target_node_id,
            "requested_prefixes": missing_prefixes,
        }

    def process_response(self, response: dict) -> int:
        """Process a gossip response with cache entries.

        Args:
            response: Dict with success, cache_entries (prefix_hash -> ref), entries_returned.

        Returns:
            Number of entries successfully merged.
        """
        if not response.get("success", False):
            return 0

        cache_entries = response.get("cache_entries", {})
        count = 0
        now = time.time()

        for prefix_hash, entry_ref in cache_entries.items():
            if prefix_hash not in self.state.cache_index:
                self.state.cache_index[prefix_hash] = []
            # Find and update the entry reference
            updated = False
            for i, (nid, ref, ts) in enumerate(self.state.cache_index[prefix_hash]):
                if ref == "":  # Placeholder from advertisement
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
        """Select a random peer for the next gossip round.

        Returns:
            Peer node ID, or None if no peers known.
        """
        if not self.state.known_peers:
            return None
        return random.choice(list(self.state.known_peers))

    def get_peers(self) -> list[str]:
        """Get all known peers.

        Returns:
            List of peer node IDs.
        """
        return list(self.state.known_peers)

    def lookup(self, prefix_hash: str) -> str | None:
        """Look up which node holds a cache entry.

        Args:
            prefix_hash: Hash of the token sequence.

        Returns:
            Node ID holding the entry, or None.
        """
        entries = self.state.cache_index.get(prefix_hash)
        if entries:
            # Return the most recent entry
            entries.sort(key=lambda x: x[2], reverse=True)
            return entries[0][0]
        return None

    def request_cache_from_peers(self, prefix_hash: str, client: "GossipClient" | None = None) -> dict | None:
        """Actively request a KV cache entry from all known peers.

        Broadcasts to all peers in order until one returns the entry.
        Used when local lookup and cache index miss — actively pulls
        from the network rather than waiting for periodic gossip sync.

        Args:
            prefix_hash: Hash of the prefix to fetch.
            client: GossipClient for network transport.

        Returns:
            KV cache entry data dict, or None if no peer has it.
        """
        if client is None:
            return None
        for peer_id in list(self.state.known_peers):
            try:
                result = client.fetch_kv_cache(peer_id, prefix_hash)
                if result is not None:
                    return result
            except Exception:
                continue
        return None

    def tombstone_entry(self, prefix_hash: str) -> None:
        """Mark a cache entry as deleted using a tombstone.

        Uses CRDT tombstone semantics: once tombstoned, the entry is
        removed from local state but the tombstone persists for gossip
        propagation to ensure eventual consistency.

        Args:
            prefix_hash: Hash of the prefix to delete.
        """
        now = time.time()
        # Increment vector clock for this deletion
        self.state.vector_clock.increment(self.state.node_id)
        # Create tombstone (LWW: only update if newer)
        existing_ts = self.state.tombstones.get(prefix_hash, 0.0)
        if now > existing_ts:
            self.state.tombstones[prefix_hash] = now
        # Remove from local entries
        self.state.local_entries.pop(prefix_hash, None)

    def cleanup_expired(self) -> int:
        """Remove expired entries and old tombstones from the cache index.

        Returns:
            Number of entries removed.
        """
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

        # Clean old tombstones (older than 2x TTL to ensure propagation)
        tombstone_cutoff = now - (self.cache_ttl * 2)
        for prefix_hash in list(self.state.tombstones.keys()):
            if self.state.tombstones[prefix_hash] < tombstone_cutoff:
                del self.state.tombstones[prefix_hash]
                # Also clean up associated metadata
                self.state.entry_metadata.pop(prefix_hash, None)

        return removed


    # ------------------------------------------------------------------
    # Distributed PagedAttention: page-table (block-level) sync
    # ------------------------------------------------------------------

    def store_block_hash(self, block_hash: str, node_id: str | None = None) -> None:
        """Record a block hash in the page-table index.

        Args:
            block_hash: SHA-256 hex hash of a KV cache block.
            node_id: Owning node (defaults to local node).
        """
        owner = node_id or self.state.node_id
        self.state.vector_clock.increment(self.state.node_id)

        if block_hash not in self.state.page_table_index:
            self.state.page_table_index[block_hash] = []
        if owner not in self.state.page_table_index[block_hash]:
            self.state.page_table_index[block_hash].append(owner)

        # Track as local block hash
        if owner == self.state.node_id and block_hash not in self.state.local_block_hashes:
            self.state.local_block_hashes.append(block_hash)

    def store_local_block_hashes(self, block_hashes: List[str]) -> None:
        """Batch-store local block hashes and rebuild the Merkle root."""
        self.state.local_block_hashes = list(block_hashes)
        self.state.vector_clock.increment(self.state.node_id)

        for bh in block_hashes:
            if bh not in self.state.page_table_index:
                self.state.page_table_index[bh] = []
            if self.state.node_id not in self.state.page_table_index[bh]:
                self.state.page_table_index[bh].append(self.state.node_id)

    def update_merkle_root(self, root_hash: str) -> None:
        """Update this node's Merkle root for gossip advertisements."""
        self.state.peer_merkle_roots[self.state.node_id] = root_hash

    def build_page_advertisement(self) -> dict:
        """Build a block-level page-table advertisement.

        Includes the Merkle root and a compact list of block hashes
        for incremental sync.

        Returns:
            Advertisement dict with merkle_root, block_count, block_hashes_sample.
        """
        state = self.state
        now = time.time()
        state.last_exchange_time = now

        return {
            "node_id": state.node_id,
            "merkle_root": state.peer_merkle_roots.get(state.node_id, ""),
            "block_count": len(state.local_block_hashes),
            "block_hashes_sample": state.local_block_hashes[:100],  # cap to limit payload
            "total_blocks_advertised": len(state.local_block_hashes),
            "page_table_entries": len(state.page_table_index),
            "timestamp": now,
        }

    def process_page_advertisement(self, peer_ad: dict) -> list[str]:
        """Process a peer's page-table advertisement.

        Merges block-level entries into the page-table index and returns
        block hashes that are *missing locally* (candidates for fetch).

        Args:
            peer_ad: Page-table advertisement dict from :meth:`build_page_advertisement`.

        Returns:
            List of block hashes that this node should fetch from the peer.
        """
        peer_id = peer_ad["node_id"]
        peer_merkle_root = peer_ad.get("merkle_root", "")

        # Update peer's Merkle root
        if peer_merkle_root:
            self.state.peer_merkle_roots[peer_id] = peer_merkle_root

        # Merge peer's block hashes into page-table index
        peer_blocks = peer_ad.get("block_hashes_sample", [])
        local_hashes = set(self.state.local_block_hashes)

        missing: list[str] = []
        for bh in peer_blocks:
            if bh not in self.state.page_table_index:
                self.state.page_table_index[bh] = []
            if peer_id not in self.state.page_table_index[bh]:
                self.state.page_table_index[bh].append(peer_id)

            # Check if we don't have this block locally
            if bh not in local_hashes and bh not in missing:
                missing.append(bh)

        self.add_peer(peer_id)
        return missing

    def lookup_block(self, block_hash: str) -> list[str]:
        """Find which nodes have a block with the given hash.

        Args:
            block_hash: SHA-256 hex hash of the block.

        Returns:
            List of node IDs that have this block (empty if unknown).
        """
        return list(self.state.page_table_index.get(block_hash, []))

    def remove_block_hash(self, block_hash: str) -> None:
        """Remove a block hash from the local page table (after eviction)."""
        self.state.local_block_hashes = [h for h in self.state.local_block_hashes if h != block_hash]
        # Keep page_table_index entry (other nodes may still have it)


class GossipClient:
    """Network transport layer for the gossip protocol.

    Handles the actual HTTP/gRPC communication between nodes for
    exchanging cache advertisements and fetching cache entries.

    Uses GossipTransport for bandwidth-aware transfers.
    """

    def __init__(
        self,
        node_id: str = "gossip-client",
        peer_resolver=None,
        transport=None,
        enable_network: bool = True,
    ):
        """Initialize the gossip client.

        Args:
            node_id: Local node identifier used by the network transport.
            peer_resolver: Callable that resolves a peer_id to (host, port).
                          If None, peer resolution is not supported.
            transport: Optional GossipTransport instance. If None, uses
                      an HTTP transport when a peer resolver is provided.
            enable_network: If False, keep stub/no-op behavior for tests.
        """
        self._peer_resolver = peer_resolver
        self._transport = transport
        if self._transport is None and enable_network and peer_resolver is not None:
            from distllm.core.gossip_transport import GossipTransport

            self._transport = GossipTransport(
                node_id=node_id,
                peer_resolver=peer_resolver,
            )
        self._request_count = 0
        self._response_count = 0

    def exchange(self, peer_id: str, advertisement: dict) -> dict | None:
        """Exchange advertisements with a peer.

        Sends our cache advertisement to the peer and receives
        the peer's advertisement in return.

        Args:
            peer_id: The peer node ID to exchange with.
            advertisement: Our cache advertisement dict.

        Returns:
            Peer's advertisement dict, or None if exchange failed.
        """
        self._request_count += 1

        # If transport is available, use it for real communication
        if self._transport is not None:
            result = self._transport.exchange_advertisements(peer_id, advertisement)
            if result:
                self._response_count += 1
            return result

        # Stub mode: simulate exchange for local testing
        logger.debug(f"Gossip exchange with {peer_id}: {advertisement.get('total_cache_entries', 0)} entries (stub)")
        return None

    def request_entries(
        self, peer_id: str, request: dict
    ) -> dict:
        """Request specific cache entries from a peer.

        Args:
            peer_id: The peer node ID to request from.
            request: Dict with requested_prefixes.

        Returns:
            Response dict with cache_entries (prefix_hash -> entry_ref).
        """
        self._request_count += 1

        if self._transport is not None:
            prefix_hashes = request.get("requested_prefixes", [])
            result = self._transport.request_kv_cache(peer_id, prefix_hashes)
            if result:
                self._response_count += 1
                return result

        # Stub mode
        missing = request.get("requested_prefixes", [])
        logger.debug(f"Requested {len(missing)} entries from {peer_id} (stub)")
        return {"success": False, "cache_entries": {}, "entries_returned": 0}

    def fetch_kv_cache(self, peer_id: str, prefix_hash: str) -> dict | None:
        """Fetch a single KV cache entry from a peer.

        Args:
            peer_id: Peer node ID.
            prefix_hash: Hash of the prefix to fetch.

        Returns:
            KV cache data dict, or None if fetch failed.
        """
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
        """Close the transport and release resources."""
        if self._transport is not None:
            if hasattr(self._transport, 'close'):
                self._transport.close()
