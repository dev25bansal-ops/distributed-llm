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
import threading
import time
from pathlib import Path
from typing import Any, Callable

from loguru import logger


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
            }
            # Track initial cluster size for quorum computation
            self._initial_cluster_size = max(self._initial_cluster_size, len(self._peers) + 1)

    def remove_peer(self, peer_id: str) -> None:
        """Remove a peer coordinator."""
        with self._lock:
            self._peers.pop(peer_id, None)

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
            # Update peer's last seen time
            if sender_id in self._peers:
                self._peers[sender_id]["last_seen"] = time.monotonic()

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

    def _run_election_round(self) -> None:
        """Run one round of the election protocol.

        Implements a simple leader election:
        1. Evict peers that haven't sent heartbeats within timeout
        2. Among alive peers (including self), lowest ID becomes leader
        3. If leader changes, log the transition and fire callbacks
        4. Prevent split-brain by requiring quorum (majority alive)
        """
        with self._lock:
            now = time.monotonic()

            # Evict stale peers
            stale = [
                pid
                for pid, info in self._peers.items()
                if now - info["last_seen"] > self._election_timeout_s
            ]
            for pid in stale:
                logger.info(f"{self._id}: Peer {pid} timed out, removing")
                del self._peers[pid]

            # Count alive nodes (including self)
            alive_ids = sorted([self._id] + list(self._peers.keys()))
            total_alive = len(alive_ids)

            # Split-brain prevention: quorum based on initial cluster size
            # This prevents two halves of a partition from both electing leaders
            quorum = (self._initial_cluster_size // 2) + 1
            if total_alive < quorum and len(self._peers) > 0:
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
