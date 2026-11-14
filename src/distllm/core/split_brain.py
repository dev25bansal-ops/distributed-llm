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

import json
import os
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger


_FENCE_STATE_DIR_ENV = "DISTLLM_FENCE_STATE_DIR"


def _default_fence_state_path(cluster_id: str) -> Path:
    """Return the default durable fence-token path for a cluster."""
    base = os.environ.get(_FENCE_STATE_DIR_ENV)
    if base:
        root = Path(base)
    else:
        try:
            root = Path.home() / ".distllm" / "fence"
        except RuntimeError:
            # No resolvable home directory (unusual CI containers): fall back
            # to the temp dir rather than failing detector construction.
            root = Path(tempfile.gettempdir()) / "distllm-fence"
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in cluster_id) or "cluster"
    return root / f"{safe}.json"


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
        fence_state_path: Where the fencing token is persisted. Defaults to
            ``$DISTLLM_FENCE_STATE_DIR/<cluster>.json`` (or
            ``~/.distllm/fence/<cluster>.json``), so tokens survive process
            restarts. Pass the empty string to disable persistence.
    """

    def __init__(
        self,
        cluster_id: str,
        peer_cluster_ids: list[str] | None = None,
        quorum_size: int = 2,
        heartbeat_timeout_s: float = 30.0,
        failure_threshold: int = 3,
        fence_state_path: str | os.PathLike | None = None,
    ):
        self._cluster_id = cluster_id
        self._peers: dict[str, PeerState] = {}
        self._quorum_size = quorum_size
        self._heartbeat_timeout_s = heartbeat_timeout_s
        self._failure_threshold = failure_threshold
        self._lock = threading.Lock()
        self._partition_detected = False
        self._partition_peers: list[str] = []

        # Fence tokens must be durable and comparable across nodes. Two
        # defects followed from the old bare `self._fence_token = 0`:
        #   - a restart reset the counter to 0, so a token the previous
        #     incarnation had already superseded was admitted again
        #     (`should_accept_request` is `>=`), i.e. a fenced-out leader's
        #     stale token became valid on every rolling deploy;
        #   - nothing ever exchanged tokens between nodes, so a token
        #     issued at one node could not be ordered against another
        #     node's counter.
        # The token is now persisted (issue/observe), reloaded on start,
        # and mergeable across nodes via ``observe_fence_token`` /
        # ``heartbeat(..., fence_token=...)``.
        if fence_state_path is None:
            fence_state_path = _default_fence_state_path(cluster_id)
        self._fence_state_path: Path | None = Path(fence_state_path) if fence_state_path else None
        self._fence_token = self._load_fence_token()
        # Floor recorded at construction: tokens at or below it were issued
        # (or adopted) before this incarnation and cannot be attested any
        # more — the token may have been superseded while we were down. They
        # are only acceptable again once a strictly newer token proves the
        # fencing authority re-issued (see _token_acceptable_locked).
        self._restart_floor = self._fence_token

        for pid in (peer_cluster_ids or []):
            # A configured peer with no heartbeat yet is not alive: it must
            # not vouch for quorum until its first heartbeat arrives, so a
            # silent peer fails closed instead of masking a partition.
            self._peers[pid] = PeerState(cluster_id=pid, is_alive=False)

    # ── Fence-token durability ──

    def _load_fence_token(self) -> int:
        """Load the durable fence token, tolerating a missing/corrupt file."""
        if self._fence_state_path is None:
            return 0
        try:
            data = json.loads(self._fence_state_path.read_text())
            token = int(data.get("fence_token", 0))
            return token if token >= 0 else 0
        except FileNotFoundError:
            return 0
        except (OSError, ValueError, json.JSONDecodeError, TypeError) as e:
            logger.warning(f"Unreadable fence state {self._fence_state_path}: {e}")
            return 0

    def _token_acceptable_locked(self, fence_token: int) -> bool:
        """Fence-token admission rule, assuming ``self._lock`` is held.

        - A strictly newer token is always acceptable (it can only come from
          a later fencing decision).
        - An older token is always stale.
        - An equal token is acceptable only when *this incarnation* issued
          it (the counter has moved past the value loaded at start), or in
          the initial never-fenced state (token 0 everywhere). A token equal
          to the restart floor is unattested: while we were down the counter
          may have been superseded, and the old leader's copy is exactly the
          token a restart used to re-admit.
        """
        if fence_token > self._fence_token:
            return True
        if fence_token < self._fence_token:
            return False
        if self._fence_token == 0:
            return True
        return self._fence_token > self._restart_floor

    def _observe_fence_token_locked(self, fence_token: int) -> None:
        """Merge a token seen from a peer/heartbeat into our counter."""
        if fence_token > self._fence_token:
            self._fence_token = fence_token
            self._persist_fence_token_locked()

    def observe_fence_token(self, fence_token: int) -> int:
        """Merge a fence token received from another node; return ours.

        Fence tokens are only comparable across nodes if nodes exchange
        them. Call this (or ``heartbeat(..., fence_token=...)``) with any
        token received from a peer: our counter converges to the maximum,
        so a token issued at node X orders above every token issued before
        it at node Y, and the losing side's old token becomes stale
        everywhere.
        """
        with self._lock:
            self._observe_fence_token_locked(int(fence_token))
            return self._fence_token

    def _persist_fence_token_locked(self) -> None:
        """Persist the current token (best effort) while holding the lock."""
        if self._fence_state_path is None:
            return
        try:
            self._fence_state_path.parent.mkdir(parents=True, exist_ok=True)
            # tmp+rename so a crash mid-write can't leave a truncated file.
            tmp = self._fence_state_path.with_suffix(".tmp")
            tmp.write_text(json.dumps({
                "cluster_id": self._cluster_id,
                "fence_token": self._fence_token,
                "updated_at": time.time(),
            }))
            os.replace(tmp, self._fence_state_path)
        except OSError as e:
            logger.warning(f"Failed to persist fence state {self._fence_state_path}: {e}")

    def heartbeat(
        self,
        cluster_id: str,
        timestamp: float | None = None,
        fence_token: int | None = None,
    ) -> None:
        """Record a heartbeat from a peer cluster.

        Args:
            cluster_id: The peer cluster ID.
            timestamp: Heartbeat timestamp (uses time.time() if None).
            fence_token: Optional fence token carried by the peer. When it is
                newer than ours it is adopted (fence tokens must be
                comparable across nodes; a token issued at any node orders
                above every token issued before it anywhere).
        """
        with self._lock:
            if cluster_id not in self._peers:
                self._peers[cluster_id] = PeerState(cluster_id=cluster_id)
            peer = self._peers[cluster_id]
            peer.last_heartbeat = timestamp or time.time()
            peer.consecutive_failures = 0
            peer.is_alive = True
            if fence_token is not None:
                peer.fence_token = max(peer.fence_token, fence_token)
                self._observe_fence_token_locked(fence_token)

    def record_failure(self, cluster_id: str) -> None:
        """Record a failed heartbeat to a peer cluster."""
        with self._lock:
            if cluster_id not in self._peers:
                self._peers[cluster_id] = PeerState(cluster_id=cluster_id)
            peer = self._peers[cluster_id]
            peer.consecutive_failures += 1
            if peer.consecutive_failures >= self._failure_threshold:
                peer.is_alive = False

    def _has_quorum_locked(self, alive_peers: int) -> bool:
        """Single quorum helper — consistently counts self.

        All quorum decisions include this cluster itself (total_alive =
        alive_peers + 1).  This avoids the prior asymmetry where
        check_partition excluded self but quorum_check included it.
        """
        return (alive_peers + 1) >= self._quorum_size

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

            self._partition_detected = not self._has_quorum_locked(alive_count)
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
        requests from the old leader. The durable file is re-read first so
        an overlapping process (or a restart ancestor) that already bumped
        the counter is not overrun, and the new token is persisted so a
        restart cannot resurrect tokens this increment superseded.
        """
        with self._lock:
            self._fence_token = max(self._fence_token, self._load_fence_token()) + 1
            self._persist_fence_token_locked()
            return self._fence_token

    def should_accept_request(self, fence_token: int) -> bool:
        """Check if a request with the given fence token should be accepted.

        Returns True when the token is not stale: newer tokens are always
        acceptable, older ones never, and an equal token only if this
        incarnation issued it (an equal token reloaded after a restart is
        unattested — see ``_token_acceptable_locked``). Accepting the token
        also merges it into our counter, so the issuing node's token becomes
        the ordering basis cluster-wide.

        Args:
            fence_token: The fence token from the request.
        """
        with self._lock:
            if not self._token_acceptable_locked(fence_token):
                return False
            # Adopt the token: once accepted, it is a lower bound for every
            # later decision at this node, so an older token can never be
            # re-accepted here.
            self._observe_fence_token_locked(fence_token)
            return True

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
            return self._has_quorum_locked(alive_count)

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
            # Check quorum first (via shared helper)
            alive_count = sum(1 for p in self._peers.values() if p.is_alive)
            if not self._has_quorum_locked(alive_count):
                total_alive = alive_count + 1
                return False, f"no_quorum ({total_alive}/{self._quorum_size} alive)"

            # Check fence token (same admission rule as should_accept_request)
            if not self._token_acceptable_locked(fence_token):
                return False, f"stale_fence_token ({fence_token} < {self._fence_token})"
            self._observe_fence_token_locked(fence_token)

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
