"""Anti-entropy gossip protocol for P2P KV cache discovery.

Nodes periodically exchange cache availability advertisements
to build a distributed index of where cache entries are located.
"""

import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set


@dataclass
class GossipState:
    """Internal state for the gossip protocol."""
    node_id: str = ""
    known_peers: Set[str] = field(default_factory=set)
    # prefix_hash -> list of (node_id, entry_ref, timestamp)
    cache_index: Dict[str, List[tuple]] = field(default_factory=dict)
    last_exchange_time: float = 0.0
    # Local cache advertisements: prefix_hash -> entry_ref
    local_entries: Dict[str, str] = field(default_factory=dict)


class GossipProtocol:
    """Anti-entropy gossip protocol for distributed cache discovery.

    Each node periodically:
    1. Builds an advertisement of its local cache entries
    2. Sends it to a random peer
    3. Receives the peer's advertisement
    4. Merges missing entries into its index
    5. Requests missing cache entries from the peer
    """

    def __init__(self, node_id: str, max_peers: int = 16, cache_ttl: float = 300.0):
        self.state = GossipState(node_id=node_id)
        self.max_peers = max_peers
        self.cache_ttl = cache_ttl

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
        """Record a local cache entry."""
        self.state.local_entries[prefix_hash] = entry_ref
        # Also add to cache_index with local node
        if prefix_hash not in self.state.cache_index:
            self.state.cache_index[prefix_hash] = []
        self.state.cache_index[prefix_hash] = [
            (nid, ref, ts)
            for nid, ref, ts in self.state.cache_index[prefix_hash]
            if nid != self.state.node_id
        ]
        self.state.cache_index[prefix_hash].append(
            (self.state.node_id, entry_ref, time.time())
        )

    def advertise(self) -> dict:
        """Build a gossip advertisement.

        Returns:
            Dict with node_id, cache_prefixes, total_cache_entries, timestamp.
        """
        now = time.time()
        self.state.last_exchange_time = now

        # Filter out expired entries
        cutoff = now - self.cache_ttl
        prefixes = []
        for prefix_hash, entries in self.state.local_entries.items():
            prefixes.append(prefix_hash)

        return {
            "node_id": self.state.node_id,
            "cache_prefixes": prefixes,
            "total_cache_entries": len(prefixes),
            "timestamp": now,
        }

    def process_advertisement(self, peer_ad: dict) -> List[str]:
        """Process a peer's advertisement.

        Updates the cache index with peer's entries and returns
        the list of prefixes that this node doesn't have.

        Args:
            peer_ad: Advertisement dict from a peer.

        Returns:
            List of prefix hashes that are missing locally.
        """
        peer_id = peer_ad["node_id"]
        peer_prefixes = set(peer_ad["cache_prefixes"])
        local_prefixes = set(self.state.local_entries.keys())

        # Update cache index with peer's entries
        now = time.time()
        for prefix_hash in peer_prefixes:
            if prefix_hash not in self.state.cache_index:
                self.state.cache_index[prefix_hash] = []
            # Update or add entry for this peer
            self.state.cache_index[prefix_hash] = [
                (nid, ref, ts)
                for nid, ref, ts in self.state.cache_index[prefix_hash]
                if nid != peer_id
            ]
            self.state.cache_index[prefix_hash].append(
                (peer_id, "", now)  # ref will be filled in by GossipResponse
            )

        # Add peer to known peers
        self.add_peer(peer_id)

        # Return missing prefixes
        missing = peer_prefixes - local_prefixes
        return list(missing)

    def build_request(self, target_node_id: str, missing_prefixes: List[str]) -> dict:
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

    def select_peer(self) -> Optional[str]:
        """Select a random peer for the next gossip round.

        Returns:
            Peer node ID, or None if no peers known.
        """
        if not self.state.known_peers:
            return None
        return random.choice(list(self.state.known_peers))

    def get_peers(self) -> List[str]:
        """Get all known peers.

        Returns:
            List of peer node IDs.
        """
        return list(self.state.known_peers)

    def lookup(self, prefix_hash: str) -> Optional[str]:
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

    def cleanup_expired(self) -> int:
        """Remove expired entries from the cache index.

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

        return removed
