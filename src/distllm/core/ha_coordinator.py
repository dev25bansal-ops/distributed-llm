"""High-availability coordinator with Raft-like leader election.

Provides fault-tolerant coordinator leadership using a heartbeat-based
election protocol with Raft-like term tracking and state replication.

Features:
- Term-based leader election (prevents stale leaders)
- Quorum-based split-brain prevention
- State replication to standby coordinators
- Automatic failover on leader death

Usage::

    election = RayFaultTolerance("coordinator-1")
    election.add_peer("coordinator-2", "10.0.0.2", 50051)
    election.start()

    if election.is_leader():
        # This coordinator is the leader
        handle_requests()
    else:
        # Forward to leader
        leader = election.get_leader()

    election.stop()
"""

from __future__ import annotations

import enum
import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable

from loguru import logger


def _default_heartbeat_transport(
    peer_id: str,
    host: str,
    port: int,
    payload: dict[str, Any],
    timeout: float = 2.0,
) -> dict[str, Any]:
    """POST an election heartbeat to a peer coordinator's HA endpoint.

    Returns the peer's response dict (its term / leader view). Raises on any
    network or non-2xx error so the caller treats the peer as unreachable.
    The peer coordinator must expose ``POST /api/v1/ha/heartbeat`` and route
    it to :meth:`RayFaultTolerance.handle_heartbeat_request`.
    """
    import httpx

    url = f"http://{host}:{port}/api/v1/ha/heartbeat"
    headers: dict[str, str] = {}
    secret = os.environ.get("DISTLLM_HA_SECRET")
    if secret:
        headers["X-HA-Secret"] = secret
    with httpx.Client(timeout=timeout) as client:
        resp = client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        return resp.json()


class CoordinatorState(enum.Enum):
    """State of a coordinator in the leader election protocol."""

    FOLLOWER = "follower"
    CANDIDATE = "candidate"
    LEADER = "leader"


class RayFaultTolerance:
    """Heartbeat-based leader election for coordinator HA.

    Each coordinator instance runs a background heartbeat loop
    that exchanges state with peers. The coordinator with the
    lowest ID (lexicographic) that is alive becomes the leader.

    Implements Raft-like features:
    - Monotonically increasing term numbers
    - State replication to standby coordinators
    - Quorum-based election (prevents split-brain)

    Args:
        coordinator_id: Unique identifier for this coordinator.
        heartbeat_interval_s: Seconds between heartbeat rounds.
        election_timeout_s: Seconds without heartbeat before declaring leader dead.
    """

    def __init__(
        self,
        coordinator_id: str,
        heartbeat_interval_s: float = 2.0,
        election_timeout_s: float = 10.0,
        heartbeat_transport: Callable[..., dict] | None = None,
    ) -> None:
        self._id = coordinator_id
        self._heartbeat_interval_s = heartbeat_interval_s
        self._election_timeout_s = election_timeout_s

        self._state = CoordinatorState.FOLLOWER
        self._leader_id: str | None = None
        self._peers: dict[str, dict[str, Any]] = {}  # id -> {host, port, last_seen}
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None
        self._initial_cluster_size: int = 1  # Updated when peers are added

        # Raft-like term tracking
        self._current_term: int = 0
        self._voted_for: str | None = None

        # State replication
        self._replicated_state: dict[str, Any] = {}
        self._on_state_change: Callable[[dict], None] | None = None

        # Heartbeat transport used to actively probe peer liveness. Defaults
        # to an HTTP POST to each peer's ``/api/v1/ha/heartbeat`` endpoint.
        # Without a working transport a peer is only considered alive while it
        # has recently sent us an inbound heartbeat.
        self._heartbeat_transport = heartbeat_transport or _default_heartbeat_transport

    @property
    def current_term(self) -> int:
        """Return the current election term."""
        with self._lock:
            return self._current_term

    def start(self) -> None:
        """Start the heartbeat election loop."""
        with self._lock:
            if self._running:
                return
            self._running = True
            self._thread = threading.Thread(
                target=self._election_loop,
                daemon=True,
                name=f"ha-{self._id[:8]}",
            )
            self._thread.start()
            logger.debug(f"HA election started for {self._id} (term={self._current_term})")

    def stop(self) -> None:
        """Stop the heartbeat election loop."""
        with self._lock:
            self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=self._heartbeat_interval_s * 2)
        with self._lock:
            self._state = CoordinatorState.FOLLOWER
            self._leader_id = None

    def add_peer(self, peer_id: str, host: str, port: int) -> None:
        """Register a peer coordinator."""
        with self._lock:
            self._peers[peer_id] = {
                "host": host,
                "port": port,
                "last_seen": time.monotonic(),
                "online": True,
            }
            # Track initial cluster size for quorum computation
            self._initial_cluster_size = max(self._initial_cluster_size, len(self._peers) + 1)

    def remove_peer(self, peer_id: str) -> None:
        """Remove a peer coordinator."""
        with self._lock:
            self._peers.pop(peer_id, None)

    def set_heartbeat_transport(self, transport: Callable[..., dict]) -> None:
        """Override the transport used to probe peer liveness.

        The transport is a callable ``(peer_id, host, port, payload) -> dict``
        that raises on failure and returns the peer's heartbeat response dict
        on success. Primarily used for tests and non-HTTP deployments.
        """
        self._heartbeat_transport = transport

    def get_state(self) -> CoordinatorState:
        """Return the current election state."""
        with self._lock:
            return self._state

    def is_leader(self) -> bool:
        """Return True if this coordinator is the leader."""
        with self._lock:
            return self._state == CoordinatorState.LEADER

    def get_leader(self) -> str | None:
        """Return the leader's coordinator ID, or None if unknown."""
        with self._lock:
            return self._leader_id

    def replicate_state(self, key: str, value: Any) -> None:
        """Replicate a state value to standby coordinators.

        Called by the leader to replicate state changes.
        Standbys receive state via handle_heartbeat_request.
        """
        with self._lock:
            self._replicated_state[key] = value

    def get_replicated_state(self) -> dict[str, Any]:
        """Return the current replicated state."""
        with self._lock:
            return dict(self._replicated_state)

    def on_state_change(self, callback: Callable[[dict], None]) -> None:
        """Register a callback for state replication events."""
        self._on_state_change = callback

    def handle_heartbeat_request(self, sender_id: str, term: int, state: dict | None = None) -> dict:
        """Handle an incoming heartbeat from a peer.

        Args:
            sender_id: The ID of the coordinator sending the heartbeat.
            term: The election term of the sender.
            state: Optional replicated state from the leader.

        Returns:
            A response dict with this coordinator's state.
        """
        with self._lock:
            # Update the sender's last-seen time and (re)admit it. A peer
            # marked offline after a transient timeout — or not yet registered
            # — must re-enter the quorum on its next inbound heartbeat,
            # otherwise a single network blip permanently removes it (B11).
            peer_info = self._peers.get(sender_id)
            if peer_info is None:
                self._peers[sender_id] = {
                    "host": "",
                    "port": 0,
                    "last_seen": time.monotonic(),
                    "online": True,
                }
            else:
                peer_info["last_seen"] = time.monotonic()
                peer_info["online"] = True

            # Raft: if sender has higher term, update our term and become follower
            if term > self._current_term:
                self._current_term = term
                self._voted_for = None
                if self._state == CoordinatorState.LEADER:
                    logger.info(f"{self._id}: Stepping down, {sender_id} has higher term {term}")
                self._state = CoordinatorState.FOLLOWER

            # If sender claims to be leader and has lower ID, follow
            if sender_id < self._id:
                if self._state == CoordinatorState.LEADER:
                    logger.info(
                        f"{self._id}: Stepping down, {sender_id} has lower ID"
                    )
                    self._state = CoordinatorState.FOLLOWER
                self._leader_id = sender_id

            # Apply replicated state from leader
            if state and self._state == CoordinatorState.FOLLOWER:
                self._replicated_state.update(state)
                if self._on_state_change:
                    try:
                        self._on_state_change(state)
                    except Exception as e:
                        logger.warning(f"State change callback error: {e}")

            return {
                "coordinator_id": self._id,
                "state": self._state.value,
                "term": self._current_term,
                "leader_id": self._leader_id,
            }

    def save_election_state(self, path: str = ".distllm_election.json") -> None:
        """Persist election state to survive restarts."""
        with self._lock:
            data = {
                "coordinator_id": self._id,
                "current_term": self._current_term,
                "voted_for": self._voted_for,
                "leader_id": self._leader_id,
                "initial_cluster_size": self._initial_cluster_size,
            }
        try:
            Path(path).write_text(json.dumps(data, indent=2))
        except OSError as e:
            logger.warning(f"Failed to save election state: {e}")

    def load_election_state(self, path: str = ".distllm_election.json") -> None:
        """Restore election state after restart."""
        try:
            data = json.loads(Path(path).read_text())
            with self._lock:
                self._current_term = data.get("current_term", 0)
                self._voted_for = data.get("voted_for")
                self._initial_cluster_size = data.get("initial_cluster_size", 1)
            logger.info(f"Loaded election state: term={self._current_term}")
        except (FileNotFoundError, json.JSONDecodeError):
            pass

    def stats(self) -> dict:
        """Return election statistics."""
        with self._lock:
            return {
                "coordinator_id": self._id,
                "state": self._state.value,
                "leader_id": self._leader_id,
                "current_term": self._current_term,
                "peers": len(self._peers),
                "peer_list": list(self._peers.keys()),
                "replicated_keys": len(self._replicated_state),
            }

    def _election_loop(self) -> None:
        """Background loop that runs heartbeats and elects leaders."""
        while self._running:
            try:
                time.sleep(self._heartbeat_interval_s)
                if not self._running:
                    break
                self._run_election_round()
            except Exception as e:
                logger.warning(f"HA election error: {e}")

    def _probe_peer(self, peer_id: str, info: dict[str, Any]) -> bool:
        """Send an outbound heartbeat to a peer and refresh its liveness.

        The peer is only treated as alive when the transport succeeds. On
        success its ``last_seen`` is refreshed (mirroring an inbound
        heartbeat) and a higher term reported by the peer is adopted so a
        low-term node cannot believe it is the leader.

        Returns:
            ``True`` if the peer responded, ``False`` otherwise.
        """
        if self._heartbeat_transport is None:
            return False
        try:
            with self._lock:
                payload = {
                    "coordinator_id": self._id,
                    "term": self._current_term,
                    "state": dict(self._replicated_state),
                }
            resp = self._heartbeat_transport(peer_id, info["host"], info["port"], payload)
            with self._lock:
                if peer_id not in self._peers:
                    return False
                self._peers[peer_id]["last_seen"] = time.monotonic()
                self._peers[peer_id]["online"] = True
                if isinstance(resp, dict):
                    peer_term = resp.get("term", 0)
                    if peer_term > self._current_term:
                        self._current_term = peer_term
                        self._voted_for = None
                        if self._state == CoordinatorState.LEADER:
                            logger.info(
                                f"{self._id}: Stepping down, {peer_id} reports term {peer_term}"
                            )
                        self._state = CoordinatorState.FOLLOWER
            return True
        except Exception as e:  # noqa: BLE001 - transport failures are expected
            logger.debug(f"{self._id}: heartbeat probe to {peer_id} failed: {e}")
            return False

    def _run_election_round(self) -> None:
        """Run one round of the election protocol.

        Implements a simple leader election:
        1. Actively probe peers for liveness (outbound heartbeats)
        2. Mark peers offline that haven't been heard from within timeout
           (they are kept in membership and re-admitted on any heartbeat)
        3. Among alive peers (including self), lowest ID becomes leader
        4. Fail closed: never self-elect without a confirmed majority
        """
        # Probe peers outside the lock — network I/O must not block election
        # bookkeeping for concurrent callers.
        with self._lock:
            peers_snapshot = {pid: dict(info) for pid, info in self._peers.items()}
        for pid, info in peers_snapshot.items():
            self._probe_peer(pid, info)

        with self._lock:
            now = time.monotonic()

            # Mark stale peers offline rather than evicting them (B11). A node
            # that missed one election timeout must remain recoverable via its
            # next inbound heartbeat or a successful outbound probe; deleting
            # it here would let a transient network blip permanently remove it
            # from the quorum.
            for pid, info in self._peers.items():
                if now - info["last_seen"] > self._election_timeout_s:
                    if info.get("online", True):
                        logger.info(f"{self._id}: Peer {pid} timed out, marking offline")
                    info["online"] = False

            # Count alive nodes (including self). Offline peers stay in
            # membership but are excluded from the liveness quorum.
            alive_ids = sorted(
                [self._id]
                + [
                    pid
                    for pid, info in self._peers.items()
                    if info.get("online", True)
                ]
            )
            total_alive = len(alive_ids)

            # Split-brain prevention: require a quorum of the *fixed* initial
            # cluster membership. A coordinator must never self-elect without
            # a confirmed majority — even if every peer has been evicted,
            # ``_initial_cluster_size`` (and therefore the quorum) does not
            # shrink, so a partitioned minority stays FOLLOWER.
            quorum = (self._initial_cluster_size // 2) + 1
            if total_alive < quorum:
                logger.warning(
                    f"{self._id}: No quorum ({total_alive} alive, need {quorum} "
                    f"of {self._initial_cluster_size} initial). "
                    f"Staying as FOLLOWER to prevent split-brain."
                )
                if self._state == CoordinatorState.LEADER:
                    logger.info(f"{self._id}: Stepping down — lost quorum")
                    self._state = CoordinatorState.FOLLOWER
                    self._leader_id = None
                return

            # Simple election: lowest ID among alive peers is leader
            new_leader = alive_ids[0] if alive_ids else self._id

            old_state = self._state
            if new_leader == self._id:
                if self._state != CoordinatorState.LEADER:
                    self._current_term += 1
                    self._voted_for = self._id
                    logger.info(f"{self._id}: Elected as leader (term={self._current_term})")
                self._state = CoordinatorState.LEADER
            else:
                if self._state == CoordinatorState.LEADER:
                    logger.info(f"{self._id}: Stepping down, {new_leader} is leader")
                self._state = CoordinatorState.FOLLOWER

            self._leader_id = new_leader

            # Log state transitions
            if old_state != self._state:
                logger.info(
                    f"{self._id}: State transition {old_state.value} → {self._state.value} "
                    f"(leader={new_leader}, term={self._current_term}, peers={len(self._peers)})"
                )
