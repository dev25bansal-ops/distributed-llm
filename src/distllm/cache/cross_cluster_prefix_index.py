"""Cross-cluster prefix KV cache index with gossip-based digest exchange.

Geo-distributed clusters share content-addressed prefix KV cache entries
through a lightweight digest dissemination protocol.  Each cluster
maintains a local index of :class:`CacheDigest` entries and periodically
exchanges compact digests with peer clusters to discover cache entries
available remotely.

Classes:

    * :class:`CacheDigest` — immutable metadata for one cached prefix.
    * :class:`CacheGossipProtocol` — message construction, merge, and
      conflict resolution.
    * :class:`CrossClusterPrefixIndex` — local index with peer discovery,
      lookup, announcement, and gossip integration.
"""

from __future__ import annotations

import struct
import threading
import time
from dataclasses import dataclass

from loguru import logger

from distllm.cache.crdt import (
    CRDTCacheMap,
    HybridLogicalClock,
    LWWRegister,
    Timestamp,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_WIRE_FORMAT_VERSION: int = 1
"""Internal wire-format version for ``CacheDigest.to_bytes``."""

_DEFAULT_GOSSIP_INTERVAL: float = 30.0
"""Default interval (seconds) between gossip sync rounds."""


# ---------------------------------------------------------------------------
# CacheDigest
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CacheDigest:
    """Immutable metadata describing one cached prefix KV block in a cluster.

    Each digest uniquely identifies a cache entry by the tuple
    ``(cluster_id, prefix_hash, model_id)``.

    Attributes:
        cluster_id: Identifies the cluster that owns this cache entry.
        prefix_hash: Content hash of the prefix (e.g. SHA-256 of the
            input tokens).
        model_id: Model identifier for which the KV cache was computed.
        ttl: Time-to-live in seconds from creation.  The entry is
            considered expired when ``time.time() - last_access > ttl``.
        last_access: Unix timestamp of the most recent access.
        reuse_count: Number of times this cache entry has been reused
            (used as conflict-resolution tiebreaker).
        kv_block_ref: Opaque reference to the actual KV block data
            (e.g. a URL, storage key, or block address).
    """

    cluster_id: str
    prefix_hash: str
    model_id: str
    ttl: float
    last_access: float
    reuse_count: int
    kv_block_ref: str = ""

    # ------------------------------------------------------------------
    # Wire format
    # ------------------------------------------------------------------

    def to_bytes(self) -> bytes:
        """Serialize this digest to a compact binary wire format.

        The wire format is::

            version (u8) | cluster_len (u32) | cluster (utf-8) |
            hash_len (u32) | prefix_hash (utf-8) |
            model_len (u32) | model_id (utf-8) |
            ttl (f64) | last_access (f64) | reuse_count (u64) |
            ref_len (u32) | kv_block_ref (utf-8)

        Returns:
            A ``bytes`` object suitable for network transmission.
        """
        cluster_bytes = self.cluster_id.encode("utf-8")
        hash_bytes = self.prefix_hash.encode("utf-8")
        model_bytes = self.model_id.encode("utf-8")
        ref_bytes = self.kv_block_ref.encode("utf-8")

        header = struct.pack(
            "!B III 2d Q I",
            _WIRE_FORMAT_VERSION,
            len(cluster_bytes),
            len(hash_bytes),
            len(model_bytes),
            self.ttl,
            self.last_access,
            self.reuse_count,
            len(ref_bytes),
        )
        return header + cluster_bytes + hash_bytes + model_bytes + ref_bytes

    @staticmethod
    def from_bytes(data: bytes) -> CacheDigest:
        """Deserialize a digest from its binary wire format.

        Args:
            data: Bytes produced by :meth:`to_bytes`.

        Returns:
            A new ``CacheDigest`` instance.

        Raises:
            ValueError: If the data is malformed or has an unsupported
                version.
        """
        min_header_size = struct.calcsize("!B III 2d Q I")
        if len(data) < min_header_size:
            raise ValueError(
                f"Wire data too short: got {len(data)} bytes, "
                f"need at least {min_header_size}"
            )

        (
            version,
            clen,
            hlen,
            mlen,
            ttl,
            last_access,
            reuse_count,
            rlen,
        ) = struct.unpack_from("!B III 2d Q I", data)

        if version != _WIRE_FORMAT_VERSION:
            raise ValueError(
                f"Unsupported wire format version {version}; "
                f"expected {_WIRE_FORMAT_VERSION}"
            )

        offset = min_header_size
        cluster_id = data[offset : offset + clen].decode("utf-8")
        offset += clen
        prefix_hash = data[offset : offset + hlen].decode("utf-8")
        offset += hlen
        model_id = data[offset : offset + mlen].decode("utf-8")
        offset += mlen
        kv_block_ref = data[offset : offset + rlen].decode("utf-8")

        return CacheDigest(
            cluster_id=cluster_id,
            prefix_hash=prefix_hash,
            model_id=model_id,
            ttl=ttl,
            last_access=last_access,
            reuse_count=reuse_count,
            kv_block_ref=kv_block_ref,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def is_expired(self) -> bool:
        """Check whether this digest's TTL has elapsed.

        Returns:
            ``True`` if ``time.time() - last_access > ttl``.
        """
        return (time.time() - self.last_access) > self.ttl


# ---------------------------------------------------------------------------
# CacheGossipProtocol
# ---------------------------------------------------------------------------

@dataclass
class CacheGossipProtocol:
    """Gossip-level logic for cache digest exchange.

    This class is stateless with respect to the gossip round — it
    constructs digest messages and merges remote digests into a local
    :class:`~distllm.cache.crdt.CRDTCacheMap`.  The actual convergent state
    lives in :class:`CrossClusterPrefixIndex`; this protocol only knows how to
    (de)serialise :class:`CacheDigest` entries and feed their HLC timestamps
    into the CRDT.

    Conflict resolution is **conflict-free** (CRDT), *not* LWW: two clusters
    that each took a different local write (or a concurrent add/remove of the
    same entry) converge to an *identical* state after exchanging merges — the
    result does not depend on message arrival order.  The previous
    higher-reuse-count / later-``last_access`` policy was a wall-clock LWW and
    diverged exactly when cluster wall clocks disagreed (the classic
    cross-cluster cache divergence bug).
    """

    def build_digest_message(
        self,
        entries: list[CacheDigest],
        *,
        compact: bool = True,
        hlc: dict[tuple[str, str], list] | None = None,
    ) -> dict:
        """Create a compact digest message for gossip exchange.

        Args:
            entries: The local cache digests to include.
            compact: When ``True`` (default), omits ``kv_block_ref``
                from the serialized entries to reduce message size.
                The full reference can be fetched on demand after a
                ``lookup`` hit.
            hlc: Optional mapping of ``(prefix_hash, model_id)`` to the
                HLC wire form ``[physical_ns, counter, node_id]`` of the
                authoritative write.  Carrying the HLC lets the peer merge
                *without trusting its own wall clock*, which is what makes
                the cross-cluster merge convergent (no LWW divergence).

        Returns:
            A JSON-serialisable dict with keys:

            - ``"type"``: ``"cache_digest"``
            - ``"entries"``: list of serialised digest dicts
            - ``"count"``: number of entries
            - ``"compact"``: whether refs were stripped
        """
        serialized: list[dict] = []
        for entry in entries:
            d: dict = {
                "c": entry.cluster_id,
                "h": entry.prefix_hash,
                "m": entry.model_id,
                "t": entry.ttl,
                "a": entry.last_access,
                "r": entry.reuse_count,
            }
            if not compact:
                d["b"] = entry.kv_block_ref
            ts = hlc.get((entry.prefix_hash, entry.model_id)) if hlc else None
            if ts is not None:
                d["ts"] = ts  # [physical_ns, counter, node_id]
            serialized.append(d)

        return {
            "type": "cache_digest",
            "entries": serialized,
            "count": len(serialized),
            "compact": compact,
        }

    def merge_digest(
        self,
        local_index: CRDTCacheMap[tuple[str, str], CacheDigest],
        remote_msg: dict,
    ) -> list[CacheDigest]:
        """Merge remote digest entries into a local CRDT-backed index.

        Each remote entry is folded into the local :class:`CRDTCacheMap`
        along with its authoritative HLC timestamp (if present on the wire).
        Because the merge is a CRDT union, the result is identical regardless
        of the order in which messages arrive, and concurrent writes from two
        clusters converge.

        Args:
            local_index: A :class:`CRDTCacheMap` mapping
                ``(prefix_hash, model_id)`` to ``CacheDigest``.  Updated
                in-place.
            remote_msg: A digest message dict produced by
                :meth:`build_digest_message`.

        Returns:
            List of digests that are present in the local index after the
            merge (empty when no remote entries were supplied).
        """
        compact = remote_msg.get("compact", True)

        for raw in remote_msg.get("entries", []):
            try:
                remote = self._deserialize_entry(raw, compact)
            except (KeyError, ValueError, TypeError) as exc:
                logger.warning("Skipping malformed digest entry: {}", exc)
                continue

            key = (remote.prefix_hash, remote.model_id)
            ts_wire = raw.get("ts")
            if ts_wire is not None:
                local_index.ingest(key, remote, Timestamp.from_wire(ts_wire))
            else:
                # No authoritative clock on the wire: apply a fresh local
                # stamp and merge — still CRDT-safe, just ordered by this node.
                local_index.put(key, remote)

        # Return the converged view (everything currently present).
        return [
            local_index.get(k)
            for k in local_index.keys()
        ]

    # ------------------------------------------------------------------
    # Backward-compatibility adapter (deprecated)
    # ------------------------------------------------------------------

    @staticmethod
    def resolve_conflict(
        entry_a: CacheDigest,
        entry_b: CacheDigest,
    ) -> CacheDigest:
        """Deprecated LWW conflict resolver — retained only for API compat.

        The cache layer no longer uses wall-clock LWW; conflicts are resolved
        by the CRDT merge in :meth:`merge_digest` using HLC timestamps.  This
        method is kept so external callers that referenced it do not break, but
        it should not be used for new code: it preserves the old
        higher-reuse / later-``last_access`` tie-break purely as a fallback.
        """
        if entry_a.reuse_count > entry_b.reuse_count:
            return entry_a
        if entry_b.reuse_count > entry_a.reuse_count:
            return entry_b
        if entry_a.last_access >= entry_b.last_access:
            return entry_a
        return entry_b

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _deserialize_entry(
        raw: dict,
        compact: bool,
    ) -> CacheDigest:
        """Reconstruct a ``CacheDigest`` from a message dict.

        Keys are single-character abbreviations: ``c`` (cluster_id),
        ``h`` (prefix_hash), ``m`` (model_id), ``t`` (ttl), ``a``
        (last_access), ``r`` (reuse_count), ``b`` (kv_block_ref).
        """
        return CacheDigest(
            cluster_id=raw["c"],
            prefix_hash=raw["h"],
            model_id=raw["m"],
            ttl=raw["t"],
            last_access=raw["a"],
            reuse_count=raw["r"],
            kv_block_ref=raw.get("b", "") if not compact else "",
        )


# ---------------------------------------------------------------------------
# CrossClusterPrefixIndex
# ---------------------------------------------------------------------------

class CrossClusterPrefixIndex:
    """Local index of KV prefix cache entries, shared across clusters via
    a gossip-based digest protocol.

    Typical usage::

        index = CrossClusterPrefixIndex(cluster_id="us-east-1")
        index.add_peer("eu-west-1")
        index.add_peer("ap-southeast-1")

        # Advertise a newly computed cache entry.
        index.announce(
            prefix_hash="abc123...",
            model_id="llama-70b",
            kv_block_ref="s3://bucket/kv/abc123...",
        )

        # Query the index (local + any digests received via gossip).
        digest = index.lookup(prefix_hash="abc123...", model_id="llama-70b")

    Attributes:
        cluster_id: This cluster's identifier (used in digest entries).
        gossip_interval: Nominal interval in seconds between gossip rounds
            (the caller drives the loop; this is a hint for scheduling).
    """

    def __init__(
        self,
        cluster_id: str,
        gossip_interval: float = _DEFAULT_GOSSIP_INTERVAL,
    ) -> None:
        """Initialise the prefix index.

        Args:
            cluster_id: Unique identifier for this cluster (e.g.
                ``"us-east-1"``).  Stamped on every digest this cluster
                announces.
            gossip_interval: Suggested interval in seconds between gossip
                sync rounds (default 30.0).  Passed to any
                :class:`CacheGossipProtocol` instance internally.
        """
        self.cluster_id: str = cluster_id
        self.gossip_interval: float = gossip_interval

        # Internal state — CRDT-backed key->value map.  Membership of keys is
        # an OR-Set and each value is an HLC-stamped LWW register, so concurrent
        # writes from two clusters converge after exchanging merges (no LWW
        # divergence on cross-cluster wall-clock skew).
        self._lock = threading.Lock()
        self._index: CRDTCacheMap[tuple[str, str], CacheDigest] = CRDTCacheMap(
            node_id=cluster_id
        )

        self._peers: dict[str, float] = {}
        """Maps peer cluster ID to last-contact timestamp."""

        self._gossip_protocol = CacheGossipProtocol()

        self._announce_count: int = 0
        self._lookup_count: int = 0
        self._gossip_messages_received: int = 0
        self._entries_merged: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def lookup(
        self,
        prefix_hash: str,
        model_id: str,
    ) -> CacheDigest | None:
        """Look up a cache entry by prefix hash and model.

        Queries the local CRDT index.  Entries that have expired are silently
        removed (and tombstoned) and not returned.

        Args:
            prefix_hash: Content hash of the prefix to look up.
            model_id: Model identifier.

        Returns:
            The matching ``CacheDigest`` if found and not expired, or
            ``None``.
        """
        self._lookup_count += 1
        key = (prefix_hash, model_id)

        with self._lock:
            entry = self._index.get(key)

        if entry is None:
            return None

        if entry.is_expired():
            with self._lock:
                self._index.remove(key)
            logger.debug("Removed expired digest: {}", key)
            return None

        # Bump last_access and reuse_count on hit (a local overwrite — CRDT
        # merge keeps the highest-stamped value, so two replicas hitting the
        # same key still converge).
        updated = CacheDigest(
            cluster_id=entry.cluster_id,
            prefix_hash=entry.prefix_hash,
            model_id=entry.model_id,
            ttl=entry.ttl,
            last_access=time.time(),
            reuse_count=entry.reuse_count + 1,
            kv_block_ref=entry.kv_block_ref,
        )
        with self._lock:
            # Preserve the authoritative HLC ordering: only overwrite locally
            # if our clock is ahead of the stored write (keeps convergence).
            existing_ts = self._index.timestamp_of(key)
            if existing_ts is None or self._index.membership.clock.update(
                existing_ts.physical_ns, existing_ts.counter, existing_ts.node_id
            ) > existing_ts:
                self._index.put(key, updated)

        return updated

    def announce(
        self,
        prefix_hash: str,
        model_id: str,
        kv_block_ref: str,
        *,
        ttl: float = 300.0,
    ) -> CacheDigest:
        """Register (or update) a local cache entry and make it available
        for gossip propagation to peer clusters.

        Args:
            prefix_hash: Content hash of the prefix.
            model_id: Model identifier.
            kv_block_ref: Opaque reference to the cached KV block data.
            ttl: Time-to-live in seconds (default 300).

        Returns:
            The newly created ``CacheDigest`` that was inserted into the
            local index.
        """
        now = time.time()
        digest = CacheDigest(
            cluster_id=self.cluster_id,
            prefix_hash=prefix_hash,
            model_id=model_id,
            ttl=ttl,
            last_access=now,
            reuse_count=1,
            kv_block_ref=kv_block_ref,
        )

        key = (prefix_hash, model_id)
        with self._lock:
            existing = self._index.get(key)
            if existing is not None:
                # Preserve accumulated reuse count and refresh.
                digest = CacheDigest(
                    cluster_id=self.cluster_id,
                    prefix_hash=prefix_hash,
                    model_id=model_id,
                    ttl=ttl,
                    last_access=now,
                    reuse_count=existing.reuse_count + 1,
                    kv_block_ref=kv_block_ref,
                )
            # CRDT put stamps the write with this cluster's HLC, so a peer that
            # merges it converges to the same value regardless of arrival order.
            self._index.put(key, digest)

        self._announce_count += 1
        logger.debug(
            "Announced cache entry: {} / {} (ref={})",
            prefix_hash[:12],
            model_id,
            kv_block_ref[:40],
        )
        return digest

    def get_peers(self) -> list[str]:
        """Return the list of peer cluster IDs known to this index.

        Returns:
            Copy of the current peer list (snapshot under the lock).
        """
        with self._lock:
            return list(self._peers.keys())

    @property
    def digest_size(self) -> int:
        """Return the number of entries currently in the local index.

        This includes both locally announced entries and those learned
        via gossip from peer clusters.  Expired entries are not counted
        (they are lazily removed during lookups).
        """
        with self._lock:
            return len(self._index.keys())

    def process_gossip_message(self, msg: dict) -> int:
        """Process an incoming digest message from a peer cluster.

        Delegates to :meth:`CacheGossipProtocol.merge_digest`, which folds the
        remote digests (with their HLC stamps) into the local CRDT.  Because the
        merge is conflict-free, the result converges to the same state on every
        replica no matter the arrival order.

        Args:
            msg: A digest message dict (as produced by
                :meth:`CacheGossipProtocol.build_digest_message`).

        Returns:
            Number of entries now present in the local index after the merge.
        """
        self._gossip_messages_received += 1
        if msg.get("type") != "cache_digest":
            logger.warning("Ignoring non-digest gossip message: {}", msg.get("type"))
            return 0

        with self._lock:
            changed = self._gossip_protocol.merge_digest(self._index, msg)

        self._entries_merged += len(changed)
        if changed:
            logger.debug(
                "Merged {} entries from gossip message",
                len(changed),
            )
        return len(changed)

    # ------------------------------------------------------------------
    # Peer management
    # ------------------------------------------------------------------

    def add_peer(self, cluster_id: str) -> None:
        """Register a peer cluster for gossip exchange.

        Args:
            cluster_id: The peer cluster's identifier.
        """
        with self._lock:
            if cluster_id not in self._peers:
                self._peers[cluster_id] = 0.0
                logger.info("Added peer cluster: {}", cluster_id)

    def remove_peer(self, cluster_id: str) -> None:
        """Unregister a peer cluster.

        Args:
            cluster_id: The peer cluster's identifier.
        """
        with self._lock:
            self._peers.pop(cluster_id, None)
            logger.info("Removed peer cluster: {}", cluster_id)

    # ------------------------------------------------------------------
    # Gossip message construction
    # ------------------------------------------------------------------

    def build_gossip_message(self, *, compact: bool = True) -> dict:
        """Build a digest of the local index for gossip exchange.

        Convenience wrapper around
        :meth:`CacheGossipProtocol.build_digest_message` that feeds the
        current local index entries, attaching each entry's authoritative HLC
        timestamp so a peer can merge without trusting its own wall clock.

        Args:
            compact: Forwarded to ``build_digest_message``.

        Returns:
            A JSON-serialisable dict.
        """
        with self._lock:
            entries = list(self._index.values())
            hlc = {
                k: self._index.timestamp_of(k).to_wire()
                for k in self._index.keys()
                if self._index.timestamp_of(k) is not None
            }
        return self._gossip_protocol.build_digest_message(
            entries, compact=compact, hlc=hlc
        )

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def evict_expired(self) -> int:
        """Remove all expired entries from the local index.

        This is called automatically during :meth:`lookup` for matching
        entries.  The caller may also invoke it periodically as a
        proactive cleanup.  Expired entries are *tombstoned* in the CRDT
        (not silently deleted) so the removal still propagates to peers
        without resurrecting the entry.

        Returns:
            Number of entries evicted.
        """
        now = time.time()
        evicted: list[tuple[str, str]] = []

        with self._lock:
            for key in list(self._index.keys()):
                entry = self._index.get(key)
                if entry is None:
                    continue
                if (now - entry.last_access) > entry.ttl:
                    evicted.append(key)
            for key in evicted:
                self._index.remove(key)

        if evicted:
            logger.debug("Evicted {} expired digest entries", len(evicted))
        return len(evicted)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def stats(self) -> dict:
        """Return operational statistics for monitoring / observability.

        Returns:
            Dict with keys: ``announce_count``, ``lookup_count``,
            ``gossip_messages_received``, ``entries_merged``,
            ``digest_size``, ``peer_count``, ``cluster_id``,
            ``gossip_interval``.
        """
        with self._lock:
            peer_count = len(self._peers)
        return {
            "announce_count": self._announce_count,
            "lookup_count": self._lookup_count,
            "gossip_messages_received": self._gossip_messages_received,
            "entries_merged": self._entries_merged,
            "digest_size": self.digest_size,
            "peer_count": peer_count,
            "cluster_id": self.cluster_id,
            "gossip_interval": self.gossip_interval,
        }
