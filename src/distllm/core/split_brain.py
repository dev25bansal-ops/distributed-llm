"""Split-brain detection for federated clusters.

Detects when a network partition causes two halves of a federation
to operate independently, and provides fencing mechanisms to prevent
divergent state.

Usage::

    detector = SplitBrainDetector(
        cluster_id="us-east",
        peer_cluster_ids=["us-west", "eu-central"],
        quorum_size=2,
    )
    detector.heartbeat("us-west", timestamp=time.time())
    if detector.is_partitioned():
        peers = detector.get_partitioned_peers()
        logger.warning(f"Split-brain detected: {peers} unreachable")
"""

from __future__ import annotations

import time
import threading
from dataclasses import dataclass, field
from typing import Any

from loguru import logger


@dataclass
class PeerState:
    """State of a peer cluster."""

    cluster_id: str
    last_heartbeat: float = 0.0
    consecutive_failures: int = 0
    is_alive: bool = True
    fence_token: int = 0  # Monotonically increasing fencing token


class SplitBrainDetector:
    """Detects network partitions in federated clusters.

    Monitors heartbeats from peer clusters and declares a partition
    when a quorum of peers become unreachable.

    Args:
        cluster_id: This cluster's ID.
        peer_cluster_ids: IDs of all peer clusters in the federation.
        quorum_size: Minimum number of alive peers to consider the
            federation healthy. If fewer peers are alive, a partition
            is declared.
        heartbeat_timeout_s: Seconds without heartbeat before a peer
            is considered unreachable.
        failure_threshold: Consecutive failures before marking a peer dead.
    """

    def __init__(
        self,
        cluster_id: str,
        peer_cluster_ids: list[str] | None = None,
        quorum_size: int = 2,
        heartbeat_timeout_s: float = 30.0,
        failure_threshold: int = 3,
    ):
        self._cluster_id = cluster_id
        self._peers: dict[str, PeerState] = {}
        self._quorum_size = quorum_size
        self._heartbeat_timeout_s = heartbeat_timeout_s
        self._failure_threshold = failure_threshold
        self._lock = threading.Lock()
        self._fence_token = 0
        self._partition_detected = False
        self._partition_peers: list[str] = []

        for pid in (peer_cluster_ids or []):
            # A configured peer with no heartbeat yet is not alive: it must
            # not vouch for quorum until its first heartbeat arrives, so a
            # silent peer fails closed instead of masking a partition.
            self._peers[pid] = PeerState(cluster_id=pid, is_alive=False)

    def heartbeat(self, cluster_id: str, timestamp: float | None = None) -> None:
        """Record a heartbeat from a peer cluster.

        Args:
            cluster_id: The peer cluster ID.
            timestamp: Heartbeat timestamp (uses time.time() if None).
        """
        with self._lock:
            if cluster_id not in self._peers:
                self._peers[cluster_id] = PeerState(cluster_id=cluster_id)
            peer = self._peers[cluster_id]
            peer.last_heartbeat = timestamp or time.time()
            peer.consecutive_failures = 0
            peer.is_alive = True

    def record_failure(self, cluster_id: str) -> None:
        """Record a failed heartbeat to a peer cluster."""
        with self._lock:
            if cluster_id not in self._peers:
                self._peers[cluster_id] = PeerState(cluster_id=cluster_id)
            peer = self._peers[cluster_id]
            peer.consecutive_failures += 1
            if peer.consecutive_failures >= self._failure_threshold:
                peer.is_alive = False

    def check_partition(self) -> bool:
        """Check if a split-brain partition has occurred.

        Returns:
            True if the federation is partitioned (quorum not met).
        """
        with self._lock:
            now = time.time()
            alive_count = 0
            partitioned = []

            for pid, peer in self._peers.items():
                # Check heartbeat timeout
                if peer.last_heartbeat > 0:
                    elapsed = now - peer.last_heartbeat
                    if elapsed > self._heartbeat_timeout_s:
                        peer.consecutive_failures += 1
                        if peer.consecutive_failures >= self._failure_threshold:
                            peer.is_alive = False

                if peer.is_alive:
                    alive_count += 1
                else:
                    partitioned.append(pid)

            # Need quorum_size peers alive (excluding self)
            self._partition_detected = alive_count < self._quorum_size
            self._partition_peers = partitioned
            return self._partition_detected

    def is_partitioned(self) -> bool:
        """Return True if a partition was detected on the last check."""
        with self._lock:
            return self._partition_detected

    def get_partitioned_peers(self) -> list[str]:
        """Return list of peer cluster IDs that are unreachable."""
        with self._lock:
            return list(self._partition_peers)

    def get_alive_peers(self) -> list[str]:
        """Return list of peer cluster IDs that are alive."""
        with self._lock:
            return [pid for pid, peer in self._peers.items() if peer.is_alive]

    def get_fence_token(self) -> int:
        """Get the current fencing token.

        Fencing tokens are monotonically increasing integers that can
        be used to prevent stale writers from modifying shared state.
        """
        with self._lock:
            return self._fence_token

    def increment_fence_token(self) -> int:
        """Increment and return the fencing token.

        Call this when a partition is detected to invalidate any
        requests from the old leader.
        """
        with self._lock:
            self._fence_token += 1
            return self._fence_token

    def should_accept_request(self, fence_token: int) -> bool:
        """Check if a request with the given fence token should be accepted.

        Returns True if the token matches the current fence token,
        indicating the request came from the current leader.

        Args:
            fence_token: The fence token from the request.
        """
        with self._lock:
            return fence_token >= self._fence_token

    def quorum_check(self) -> bool:
        """Check if this cluster has quorum in the federation.

        A cluster has quorum when the number of alive peers (including self)
        meets or exceeds the quorum size.  Without quorum, the cluster
        should reject writes to prevent divergent state.

        Returns:
            True if quorum is maintained.
        """
        with self._lock:
            alive_count = sum(1 for p in self._peers.values() if p.is_alive)
            # Include self in the count
            total_alive = alive_count + 1
            return total_alive >= self._quorum_size

    def fence_request(self, fence_token: int) -> tuple[bool, str]:
        """Combined quorum + fence-token check for request admission.

        Checks both that the federation has quorum AND that the request
        carries a valid fence token.  Returns a tuple of (accept, reason).

        Args:
            fence_token: The fence token from the request.

        Returns:
            (True, "ok") if the request should be accepted,
            (False, reason) if it should be rejected.
        """
        with self._lock:
            # Check quorum first
            alive_count = sum(1 for p in self._peers.values() if p.is_alive)
            total_alive = alive_count + 1
            if total_alive < self._quorum_size:
                return False, f"no_quorum ({total_alive}/{self._quorum_size} alive)"

            # Check fence token
            if fence_token < self._fence_token:
                return False, f"stale_fence_token ({fence_token} < {self._fence_token})"

            return True, "ok"

    def stats(self) -> dict:
        """Return partition detection statistics."""
        with self._lock:
            return {
                "cluster_id": self._cluster_id,
                "peers": len(self._peers),
                "alive": sum(1 for p in self._peers.values() if p.is_alive),
                "partitioned": self._partition_detected,
                "partitioned_peers": list(self._partition_peers),
                "fence_token": self._fence_token,
                "quorum_size": self._quorum_size,
            }
