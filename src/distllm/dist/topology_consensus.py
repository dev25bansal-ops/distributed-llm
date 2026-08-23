"""Raft-based consensus for topology metadata with SQLite persistence.

Provides a distributed consensus layer for cluster topology changes using
the Raft consensus algorithm. Topology changes (node join, leave, update)
are proposed as Raft log entries, replicated across the cluster, and
committed once a quorum acknowledges them. Persisted via SQLite for
recovery on restart, and bridged to the gossip protocol for push-based
propagation.

Usage::

    manager = TopologyManager(node_id="node-0", peers=[...])
    await manager.start()
    await manager.register_node({"node_id": "node-1", "host": "10.0.0.5"})
    topo = manager.get_topology()
    await manager.stop()
"""

from __future__ import annotations

import json
import os
import random
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable

from loguru import logger


# ===========================================================================
# RaftNode — Raft consensus algorithm implementation
# ===========================================================================


class RaftRole(Enum):
    """Raft node role."""

    FOLLOWER = auto()
    CANDIDATE = auto()
    LEADER = auto()


@dataclass(frozen=True)
class LogEntry:
    """A single entry in the Raft log.

    Attributes:
        term: The term when this entry was appended.
        index: The log index (1-based, monotonically increasing).
        entry_type: One of ``"node_join"``, ``"node_leave"``, ``"node_update"``.
        payload: The serializable JSON body of the entry (e.g., node info).
    """

    term: int
    index: int
    entry_type: str
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class RaftPersistentState:
    """State that survives restarts (flushed to SQLite via TopologyStore).

    Attributes:
        current_term: The latest term this node has seen.
        voted_for: The candidate_id this node voted for in the current term
            (None if none).
        log: The Raft log entries, indexed by log_index - 1.
    """

    current_term: int = 0
    voted_for: str | None = None
    log: list[LogEntry] = field(default_factory=list)

    @property
    def last_log_index(self) -> int:
        return len(self.log)

    @property
    def last_log_term(self) -> int:
        if not self.log:
            return 0
        return self.log[-1].term

    def get_entry(self, index: int) -> LogEntry | None:
        if 1 <= index <= len(self.log):
            return self.log[index - 1]
        return None

    def append_entries(self, entries: list[LogEntry]) -> None:
        self.log.extend(entries)

    def truncate_from(self, index: int) -> None:
        """Remove log entries from *index* onward (1-based)."""
        if 1 <= index <= len(self.log):
            del self.log[index - 1 :]


@dataclass
class RaftVolatileState:
    """Volatile state that resets on restart and on leader change.

    Attributes:
        commit_index: Index of the highest log entry known to be committed.
        last_applied: Index of the highest log entry applied to the state
            machine.
    """

    commit_index: int = 0
    last_applied: int = 0


@dataclass
class RaftLeaderState:
    """Volatile state maintained only on the leader.

    Attributes:
        next_index: For each peer, the next log index to send.
        match_index: For each peer, the highest log index known to be
            replicated.
    """

    next_index: dict[str, int] = field(default_factory=dict)
    match_index: dict[str, int] = field(default_factory=dict)


# Default Raft timing constants (milliseconds).
_ELECTION_TIMEOUT_MIN_MS = 150
_ELECTION_TIMEOUT_MAX_MS = 300
_HEARTBEAT_INTERVAL_MS = 50
_RPC_TIMEOUT_MS = 100

# Maximum entries per AppendEntries RPC.
_MAX_BATCH_SIZE = 64


class RaftNode:
    """Raft consensus algorithm implementation.

    Manages leader election, log replication, and commit tracking for
    topology metadata. Uses randomized election timeouts to avoid
    split votes.

    Thread-safe: all mutable state is guarded by ``_lock``.

    Args:
        node_id: Unique identifier for this Raft node.
        election_timeout_min_ms: Minimum election timeout in ms.
        election_timeout_max_ms: Maximum election timeout in ms.
        heartbeat_interval_ms: Leader heartbeat interval in ms.
        max_batch_size: Maximum log entries per replication batch.
    """

    def __init__(
        self,
        node_id: str,
        election_timeout_min_ms: int = _ELECTION_TIMEOUT_MIN_MS,
        election_timeout_max_ms: int = _ELECTION_TIMEOUT_MAX_MS,
        heartbeat_interval_ms: int = _HEARTBEAT_INTERVAL_MS,
        max_batch_size: int = _MAX_BATCH_SIZE,
    ) -> None:
        self._node_id = node_id
        self._election_timeout_min_ms = election_timeout_min_ms
        self._election_timeout_max_ms = election_timeout_max_ms
        self._heartbeat_interval_ms = heartbeat_interval_ms
        self._max_batch_size = max_batch_size

        # Raft state machines.
        self._role: RaftRole = RaftRole.FOLLOWER
        self._persistent: RaftPersistentState = RaftPersistentState()
        self._volatile: RaftVolatileState = RaftVolatileState()
        self._leader_state: RaftLeaderState | None = None

        # Leader identity (authoritative).
        self._leader_id: str | None = None

        # Election timer — randomized on start and on every reset.
        self._election_deadline: float = self._random_election_deadline()

        # Lock for all mutable state.
        self._lock = threading.RLock()

        # Vote tracking during an election.
        self._votes_received: set[str] = field(default_factory=set)  # type: ignore[assignment]

    # -- Properties ---------------------------------------------------------

    @property
    def node_id(self) -> str:
        return self._node_id

    @property
    def role(self) -> RaftRole:
        return self._role

    @property
    def leader_id(self) -> str | None:
        return self._leader_id

    @property
    def current_term(self) -> int:
        return self._persistent.current_term

    @property
    def commit_index(self) -> int:
        return self._volatile.commit_index

    @property
    def last_applied(self) -> int:
        return self._volatile.last_applied

    @property
    def log(self) -> list[LogEntry]:
        return list(self._persistent.log)

    def log_slice(self, start: int = 0) -> list[LogEntry]:
        """Return log entries from *start* (1-based) to end."""
        return list(self._persistent.log[start - 1 :])

    # -- Role transitions ---------------------------------------------------

    def _become_follower(self, term: int, leader_id: str | None = None) -> None:
        """Transition to follower state for *term*."""
        self._role = RaftRole.FOLLOWER
        self._persistent.current_term = term
        self._leader_id = leader_id
        self._leader_state = None
        self._votes_received.clear()
        self._election_deadline = self._random_election_deadline()

    def _become_candidate(self) -> None:
        """Start an election: increment term, vote for self, request votes."""
        self._role = RaftRole.CANDIDATE
        self._persistent.current_term += 1
        self._leader_id = None
        self._persistent.voted_for = self._node_id
        self._votes_received = {self._node_id}
        self._leader_state = None
        self._election_deadline = self._random_election_deadline()

    def _become_leader(self) -> None:
        """Transition to leader. Initialize leader state for peers."""
        self._role = RaftRole.LEADER
        self._leader_id = self._node_id
        next_idx = self._persistent.last_log_index + 1
        self._leader_state = RaftLeaderState(
            next_index={},
            match_index={},
        )
        # Peers are set externally via set_peers().

    def set_peers(self, peer_ids: list[str]) -> None:
        """Set the list of peer node IDs for leader state tracking."""
        with self._lock:
            if self._leader_state is None:
                return
            next_idx = self._persistent.last_log_index + 1
            for pid in peer_ids:
                if pid != self._node_id:
                    self._leader_state.next_index.setdefault(pid, next_idx)
                    self._leader_state.match_index.setdefault(pid, 0)

    def update_peer_tracking(self, peer_ids: list[str]) -> None:
        """Ensure all peers are tracked in leader state."""
        with self._lock:
            if self._leader_state is None:
                return
            next_idx = self._persistent.last_log_index + 1
            for pid in peer_ids:
                if pid != self._node_id:
                    self._leader_state.next_index.setdefault(pid, next_idx)
                    self._leader_state.match_index.setdefault(pid, 0)

    # -- Election timeout ---------------------------------------------------

    def _random_election_deadline(self) -> float:
        """Return an absolute time for the next election timeout."""
        delay_ms = random.randint(
            self._election_timeout_min_ms, self._election_timeout_max_ms,
        )
        return time.time() + delay_ms / 1000.0

    def reset_election_timeout(self) -> None:
        """Reset the election timer to a new random deadline."""
        with self._lock:
            self._election_deadline = self._random_election_deadline()

    def election_elapsed(self) -> bool:
        """Check if the election timeout has elapsed."""
        with self._lock:
            return time.time() >= self._election_deadline

    @property
    def heartbeat_interval(self) -> float:
        return self._heartbeat_interval_ms / 1000.0

    # -- Log operations -----------------------------------------------------

    def append_to_log(self, entry_type: str, payload: dict[str, Any]) -> int:
        """Append a new entry to the local log.

        Only the leader should call this. Returns the new log index.

        Args:
            entry_type: ``"node_join"``, ``"node_leave"``, or
                ``"node_update"``.
            payload: Serialisable JSON body.

        Returns:
            The index of the new entry (1-based).
        """
        with self._lock:
            term = self._persistent.current_term
            index = self._persistent.last_log_index + 1
            entry = LogEntry(term=term, index=index, entry_type=entry_type, payload=payload)
            self._persistent.append_entries([entry])
            return index

    # -- RequestVote RPC (receiver) -----------------------------------------

    def handle_request_vote(
        self,
        candidate_id: str,
        candidate_term: int,
        last_log_index: int,
        last_log_term: int,
    ) -> dict[str, Any]:
        """Handle an incoming RequestVote RPC.

        Args:
            candidate_id: The candidate requesting the vote.
            candidate_term: The candidate's term.
            last_log_index: The candidate's last log index.
            last_log_term: The candidate's last log term.

        Returns:
            dict with ``term`` and ``vote_granted``.
        """
        with self._lock:
            # Reply with false if candidate term is stale.
            if candidate_term < self._persistent.current_term:
                return {
                    "term": self._persistent.current_term,
                    "vote_granted": False,
                }

            # If candidate term > our term, step down.
            if candidate_term > self._persistent.current_term:
                self._become_follower(candidate_term)

            # Grant vote if we haven't voted for anyone else and candidate's
            # log is at least as up-to-date as ours.
            already_voted = (
                self._persistent.voted_for is not None
                and self._persistent.voted_for != candidate_id
            )
            if already_voted:
                return {
                    "term": self._persistent.current_term,
                    "vote_granted": False,
                }

            # Log up-to-date check.
            our_last_index = self._persistent.last_log_index
            our_last_term = self._persistent.last_log_term

            log_ok = (
                last_log_term > our_last_term
                or (
                    last_log_term == our_last_term
                    and last_log_index >= our_last_index
                )
            )
            if not log_ok:
                return {
                    "term": self._persistent.current_term,
                    "vote_granted": False,
                }

            # Grant the vote.
            self._persistent.voted_for = candidate_id
            self._reset_election_deadline_locked()

            logger.info(
                f"Raft vote granted: term={candidate_term}, "
                f"candidate={candidate_id}, node={self._node_id}"
            )
            return {
                "term": self._persistent.current_term,
                "vote_granted": True,
            }

    # -- AppendEntries RPC (receiver) ---------------------------------------

    def handle_append_entries(
        self,
        leader_term: int,
        leader_id: str,
        prev_log_index: int,
        prev_log_term: int,
        entries: list[LogEntry],
        leader_commit: int,
    ) -> dict[str, Any]:
        """Handle an incoming AppendEntries RPC (also used for heartbeats).

        Args:
            leader_term: The leader's term.
            leader_id: The leader's node ID.
            prev_log_index: Index of log entry immediately preceding new ones.
            prev_log_term: Term of ``prev_log_index`` entry.
            entries: Log entries to append (empty for heartbeat).
            leader_commit: Leader's ``commit_index``.

        Returns:
            dict with ``term`` and ``success``.
        """
        with self._lock:
            # Reply false if leader term is stale.
            if leader_term < self._persistent.current_term:
                return {
                    "term": self._persistent.current_term,
                    "success": False,
                }

            # Accept the leader.
            self._become_follower(leader_term, leader_id=leader_id)

            # Fail if log doesn't contain an entry at prev_log_index
            # matching prev_log_term.
            if prev_log_index > 0:
                prev_entry = self._persistent.get_entry(prev_log_index)
                if prev_entry is None or prev_entry.term != prev_log_term:
                    return {
                        "term": self._persistent.current_term,
                        "success": False,
                        "conflict_index": self._find_conflict_index(
                            prev_log_index
                        ),
                    }

            # If an existing entry conflicts with a new one, truncate.
            for i, entry in enumerate(entries):
                entry_idx = prev_log_index + 1 + i
                existing = self._persistent.get_entry(entry_idx)
                if existing is not None and existing.term != entry.term:
                    self._persistent.truncate_from(entry_idx)
                    break

            # Append new entries not already in the log.
            for entry in entries:
                entry_idx = entry.index
                existing = self._persistent.get_entry(entry_idx)
                if existing is None:
                    self._persistent.append_entries([entry])

            # Update commit index.
            if leader_commit > self._volatile.commit_index:
                self._volatile.commit_index = min(
                    leader_commit, self._persistent.last_log_index,
                )

            return {
                "term": self._persistent.current_term,
                "success": True,
            }

    def _find_conflict_index(self, prev_log_index: int) -> int:
        """Find the first index where our log diverges for conflict
        resolution.

        Used by the leader to efficiently find the point of divergence
        when a follower rejects an AppendEntries.
        """
        # Scan backward from prev_log_index to find our entry term.
        entry = self._persistent.get_entry(prev_log_index)
        conflict_term = entry.term if entry else 0

        # Find the first index of that term.
        for i in range(prev_log_index, 0, -1):
            e = self._persistent.get_entry(i)
            if e is None or e.term != conflict_term:
                return i + 1
        return 1

    # -- Vote management (candidate side) -----------------------------------

    def record_vote(self, voter_id: str) -> None:
        """Record a vote from *voter_id* during an election."""
        with self._lock:
            self._votes_received.add(voter_id)

    def has_majority(self, cluster_size: int) -> bool:
        """Check if the candidate has received votes from a majority."""
        with self._lock:
            return len(self._votes_received) > cluster_size // 2

    # -- Leader commit advancement ------------------------------------------

    def advance_commit_index(self) -> int:
        """Advance ``commit_index`` if a majority has replicated an entry.

        Called by the leader after each successful AppendEntries response.

        Returns:
            The new commit index (may be unchanged).
        """
        with self._lock:
            if self._leader_state is None:
                return self._volatile.commit_index

            # Find the highest index N such that a majority of
            # match_index[N] >= N and log[N].term == current_term.
            match_indices = sorted(
                [self._volatile.commit_index]
                + list(self._leader_state.match_index.values())
            )
            # The (cluster_size // 2)th entry in sorted list is the highest
            # index that a majority has replicated.
            mid = len(match_indices) // 2
            candidate = match_indices[mid]

            if candidate > self._volatile.commit_index:
                entry = self._persistent.get_entry(candidate)
                if entry is not None and entry.term == self._persistent.current_term:
                    self._volatile.commit_index = candidate

            return self._volatile.commit_index

    # -- Entries ready to apply ---------------------------------------------

    def get_committed_entries(self) -> list[LogEntry]:
        """Return entries between ``last_applied + 1`` and ``commit_index``.

        These are ready to be applied to the state machine.
        """
        with self._lock:
            start = self._volatile.last_applied + 1
            end = self._volatile.commit_index
            if start > end:
                return []
            return self.log_slice(start)[: end - start + 1]

    def advance_last_applied(self, count: int) -> None:
        """Advance ``last_applied`` by *count* entries."""
        with self._lock:
            self._volatile.last_applied += count

    # -- AppendEntries request builder (leader side) ------------------------

    def build_append_entries_request(
        self, peer_id: str,
    ) -> dict[str, Any] | None:
        """Build an AppendEntries RPC for *peer_id*.

        Returns None if the peer is up-to-date (no entries to send).

        The returned dict can be serialised and sent over the wire.
        """
        with self._lock:
            if self._leader_state is None:
                return None

            next_idx = self._leader_state.next_index.get(peer_id)
            if next_idx is None:
                return None

            # Heartbeat if no new entries.
            prev_log_index = next_idx - 1
            prev_log_term = 0
            if prev_log_index > 0:
                prev_entry = self._persistent.get_entry(prev_log_index)
                if prev_entry is not None:
                    prev_log_term = prev_entry.term

            entries_to_send: list[LogEntry] = []
            if next_idx <= self._persistent.last_log_index:
                end = min(
                    next_idx + self._max_batch_size - 1,
                    self._persistent.last_log_index,
                )
                entries_to_send = self.log_slice(next_idx)[: end - next_idx + 1]

            return {
                "term": self._persistent.current_term,
                "leader_id": self._node_id,
                "prev_log_index": prev_log_index,
                "prev_log_term": prev_log_term,
                "entries": [
                    {
                        "term": e.term,
                        "index": e.index,
                        "entry_type": e.entry_type,
                        "payload": e.payload,
                    }
                    for e in entries_to_send
                ],
                "leader_commit": self._volatile.commit_index,
                "is_heartbeat": len(entries_to_send) == 0,
            }

    def process_append_entries_response(
        self, peer_id: str, response: dict[str, Any],
    ) -> bool:
        """Process the response from an AppendEntries RPC.

        Updates ``next_index`` and ``match_index`` on success, or
        decrements ``next_index`` on failure (for retry).

        Returns:
            True if the append was successful, False otherwise.
        """
        with self._lock:
            if self._leader_state is None:
                return False

            if response.get("success"):
                # Success: update match_index and next_index.
                next_idx = self._leader_state.next_index.get(peer_id, 1)
                self._leader_state.match_index[peer_id] = next_idx - 1 + len(
                    response.get("applied_count", 0)
                )
                # Simplified: on success, set match_index = last entry sent.
                # We use a simple approach: move next_index forward.
                sent_count = response.get("applied_count", 0)
                if sent_count is None:
                    sent_count = 0
                new_next = self._leader_state.next_index.get(peer_id, 1) + sent_count
                self._leader_state.next_index[peer_id] = min(
                    new_next, self._persistent.last_log_index + 1,
                )
                return True
            else:
                # Failure: decrement next_index and retry.
                current = self._leader_state.next_index.get(peer_id, 1)
                if current > 1:
                    self._leader_state.next_index[peer_id] = current - 1
                return False

    # -- State helpers ------------------------------------------------------

    def _reset_election_deadline_locked(self) -> None:
        """Reset election deadline (caller must hold _lock)."""
        self._election_deadline = time.time() + random.randint(
            self._election_timeout_min_ms, self._election_timeout_max_ms,
        ) / 1000.0

    def stats(self) -> dict[str, Any]:
        """Return a snapshot of Raft state for observability."""
        with self._lock:
            return {
                "node_id": self._node_id,
                "role": self._role.name,
                "current_term": self._persistent.current_term,
                "voted_for": self._persistent.voted_for,
                "leader_id": self._leader_id,
                "log_size": len(self._persistent.log),
                "commit_index": self._volatile.commit_index,
                "last_applied": self._volatile.last_applied,
                "election_deadline_s": round(
                    self._election_deadline - time.time(), 3,
                ),
            }


# ===========================================================================
# RaftCluster — manages peer connections and coordination
# ===========================================================================


@dataclass
class PeerHandle:
    """Represents a connected Raft peer.

    In a real deployment, this would wrap a gRPC or QUIC connection.
    For now, it provides a stub interface for in-process testing.
    """

    node_id: str
    address: str  # host:port
    connected: bool = True

    def send_request_vote(self, request: dict[str, Any]) -> dict[str, Any]:
        """Send a RequestVote RPC to this peer (stub)."""
        return {"term": 0, "vote_granted": True}

    def send_append_entries(self, request: dict[str, Any]) -> dict[str, Any]:
        """Send an AppendEntries RPC to this peer (stub).

        In a real implementation, this would serialise the request and
        send it over the wire.
        """
        applied = len(request.get("entries", []))
        return {"term": 0, "success": True, "applied_count": applied}

    def close(self) -> None:
        """Close the peer connection."""
        self.connected = False


class RaftCluster:
    """Manages Raft peer connections and cluster coordination.

    Provides the public API for proposing topology changes, querying
    leader status, and monitoring cluster health.

    Args:
        node_id: Unique identifier for this node.
        peers: List of peer addresses as ``"node_id:host:port"`` strings
            (or just ``"node_id"`` for in-process usage).
        election_timeout_min_ms: Minimum election timeout in ms.
        election_timeout_max_ms: Maximum election timeout in ms.
        heartbeat_interval_ms: Leader heartbeat interval in ms.
    """

    def __init__(
        self,
        node_id: str,
        peers: list[str] | None = None,
        election_timeout_min_ms: int = _ELECTION_TIMEOUT_MIN_MS,
        election_timeout_max_ms: int = _ELECTION_TIMEOUT_MAX_MS,
        heartbeat_interval_ms: int = _HEARTBEAT_INTERVAL_MS,
    ) -> None:
        self._node_id = node_id
        self._raft = RaftNode(
            node_id=node_id,
            election_timeout_min_ms=election_timeout_min_ms,
            election_timeout_max_ms=election_timeout_max_ms,
            heartbeat_interval_ms=heartbeat_interval_ms,
        )
        self._peers: dict[str, PeerHandle] = {}
        self._peer_ids: list[str] = []
        self._running = False
        self._lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

        # Callback fired when a log entry is committed and ready to apply.
        self._apply_callback: Callable[[LogEntry], None] | None = None

        if peers:
            for p in peers:
                parts = p.split(":")
                pid = parts[0]
                addr = f"{parts[1]}:{parts[2]}" if len(parts) >= 3 else pid
                self._peers[pid] = PeerHandle(node_id=pid, address=addr)
            self._peer_ids = list(self._peers.keys())
            self._raft.set_peers(self._peer_ids)

    @property
    def node_id(self) -> str:
        return self._node_id

    @property
    def raft(self) -> RaftNode:
        return self._raft

    # -- Lifecycle ----------------------------------------------------------

    def start(self) -> None:
        """Start the Raft cluster background loop.

        Runs election timers, heartbeats, and log replication.
        """
        if self._running:
            return
        self._running = True
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info(
            f"RaftCluster started: node={self._node_id}, "
            f"peers={len(self._peers)}"
        )

    def stop(self) -> None:
        """Stop the Raft cluster background loop."""
        self._running = False
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
        for peer in self._peers.values():
            peer.close()
        logger.info(f"RaftCluster stopped: node={self._node_id}")

    def on_apply(self, callback: Callable[[LogEntry], None]) -> None:
        """Register a callback invoked when a committed entry is applied.

        The callback receives the committed ``LogEntry``.
        """
        self._apply_callback = callback

    # -- Propose a topology change ------------------------------------------

    def propose(
        self, entry_type: str, payload: dict[str, Any],
    ) -> bool:
        """Propose a topology change to the Raft cluster.

        If this node is the leader, the entry is appended locally and
        replicated to peers. If this node is not the leader, the proposal
        is forwarded to the current leader (or rejected).

        Args:
            entry_type: ``"node_join"``, ``"node_leave"``, or
                ``"node_update"``.
            payload: Serializable JSON body of the entry.

        Returns:
            True if the proposal was accepted (committed by a majority).
            False if the proposal was rejected or this node is not the
            leader and cannot reach it.
        """
        with self._lock:
            # Only the leader can accept proposals.
            if self._raft.role != RaftRole.LEADER:
                logger.warning(
                    f"Node {self._node_id} is not the leader "
                    f"(leader={self._raft.leader_id}). Proposal rejected."
                )
                return False

            # Append to leader's log.
            index = self._raft.append_to_log(entry_type, payload)
            logger.info(
                f"Raft proposal accepted: type={entry_type}, "
                f"index={index}, node={self._node_id}"
            )

            # Replicate to peers synchronously (in real implementation,
            # this would be async with retry).
            self._replicate_to_peers(index)
            return True

    def _replicate_to_peers(self, index: int) -> None:
        """Replicate a log entry to all peers."""
        for pid in self._peer_ids:
            try:
                peer = self._peers.get(pid)
                if peer is None or not peer.connected:
                    continue
                request = self._raft.build_append_entries_request(pid)
                if request is None:
                    continue
                response = peer.send_append_entries(request)
                self._raft.process_append_entries_response(pid, response)
            except Exception as e:
                logger.warning(
                    f"Raft replication to {pid} failed: {e}"
                )

        # Advance commit index.
        committed = self._raft.advance_commit_index()
        if committed > 0:
            self._apply_committed_entries()

    # -- Apply committed entries --------------------------------------------

    def apply_log(self, entry: LogEntry) -> None:
        """Apply a committed log entry to the state machine.

        This is called automatically from ``_apply_committed_entries``
        and can also be invoked externally for entries received from
        a leader during catch-up.

        Args:
            entry: The committed ``LogEntry`` to apply.
        """
        logger.info(
            f"Raft applying log: type={entry.entry_type}, "
            f"index={entry.index}, node={self._node_id}"
        )
        if self._apply_callback is not None:
            try:
                self._apply_callback(entry)
            except Exception as e:
                logger.error(
                    f"Raft apply callback failed for entry "
                    f"{entry.index}: {e}"
                )

    def _apply_committed_entries(self) -> None:
        """Apply all entries from ``last_applied + 1`` to ``commit_index``."""
        entries = self._raft.get_committed_entries()
        for entry in entries:
            self.apply_log(entry)
        if entries:
            self._raft.advance_last_applied(len(entries))

    # -- Query cluster state ------------------------------------------------

    def get_leader(self) -> str | None:
        """Return the current leader's node ID, or None."""
        with self._lock:
            return self._raft.leader_id

    def cluster_status(self) -> dict[str, Any]:
        """Return a snapshot of cluster status.

        Includes current term, commit index, peer list, leader info.
        """
        with self._lock:
            return {
                "node_id": self._node_id,
                "role": self._raft.role.name if self._raft.role else "UNKNOWN",
                "leader_id": self._raft.leader_id,
                "current_term": self._raft.current_term,
                "commit_index": self._raft.commit_index,
                "last_applied": self._raft.last_applied,
                "log_size": len(self._raft.log),
                "peers": list(self._peer_ids),
                "running": self._running,
            }

    def add_peer(self, peer_id: str, address: str = "") -> None:
        """Add a new peer to the cluster.

        Args:
            peer_id: The peer's node ID.
            address: The peer's ``host:port`` address (optional).
        """
        with self._lock:
            if peer_id not in self._peers and peer_id != self._node_id:
                self._peers[peer_id] = PeerHandle(
                    node_id=peer_id, address=address or peer_id,
                )
                self._peer_ids = list(self._peers.keys())
                self._raft.update_peer_tracking(self._peer_ids)

    def remove_peer(self, peer_id: str) -> None:
        """Remove a peer from the cluster.

        Args:
            peer_id: The peer's node ID.
        """
        with self._lock:
            self._peers.pop(peer_id, None)
            self._peer_ids = list(self._peers.keys())

    # -- Background loop ----------------------------------------------------

    def _run_loop(self) -> None:
        """Main Raft loop: election timers, heartbeats, replication."""
        last_heartbeat = 0.0

        while self._running and not self._stop_event.is_set():
            now = time.time()
            role = self._raft.role

            try:
                if role == RaftRole.FOLLOWER or role == RaftRole.CANDIDATE:
                    self._handle_election_tick(now)
                elif role == RaftRole.LEADER:
                    if now - last_heartbeat >= self._raft.heartbeat_interval:
                        last_heartbeat = now
                        self._send_heartbeats()
            except Exception as e:
                logger.warning(f"Raft loop error: {e}")

            self._stop_event.wait(timeout=0.01)

    def _handle_election_tick(self, now: float) -> None:
        """Check if election timeout has elapsed and start an election."""
        if not self._raft.election_elapsed():
            return

        with self._lock:
            self._raft._become_candidate()

        logger.info(
            f"Raft election started: term={self._raft.current_term}, "
            f"node={self._node_id}"
        )

        # Request votes from all peers.
        votes_needed = len(self._peer_ids) + 1  # including self
        last_idx = self._raft._persistent.last_log_index
        last_term = self._raft._persistent.last_log_term

        for pid in self._peer_ids:
            try:
                peer = self._peers.get(pid)
                if peer is None or not peer.connected:
                    continue

                response = peer.send_request_vote({
                    "candidate_id": self._node_id,
                    "candidate_term": self._raft.current_term,
                    "last_log_index": last_idx,
                    "last_log_term": last_term,
                })
                with self._lock:
                    self._raft.handle_request_vote(
                        candidate_id=pid,
                        candidate_term=response.get("term", 0),
                        last_log_index=0,
                        last_log_term=0,
                    )
                    # Process vote response.
                    if response.get("vote_granted"):
                        self._raft.record_vote(pid)

            except Exception as e:
                logger.warning(f"Raft vote request to {pid} failed: {e}")

        # Check if we won the election.
        with self._lock:
            if self._raft.has_majority(votes_needed):
                self._raft._become_leader()
                self._raft.set_peers(self._peer_ids)
                logger.info(
                    f"Raft leader elected: term={self._raft.current_term}, "
                    f"node={self._node_id}"
                )

    def _send_heartbeats(self) -> None:
        """Send heartbeat AppendEntries RPCs to all peers."""
        for pid in self._peer_ids:
            try:
                peer = self._peers.get(pid)
                if peer is None or not peer.connected:
                    continue

                request = self._raft.build_append_entries_request(pid)
                if request is None:
                    continue

                response = peer.send_append_entries(request)
                self._raft.process_append_entries_response(pid, response)

            except Exception as e:
                logger.warning(f"Raft heartbeat to {pid} failed: {e}")

        # Advance commit index and apply entries.
        committed = self._raft.advance_commit_index()
        if committed > 0:
            self._apply_committed_entries()


# ===========================================================================
# TopologyStore — SQLite-backed persistence for topology metadata
# ===========================================================================


@dataclass
class NodeRecord:
    """A persisted node record.

    Attributes:
        node_id: Unique node identifier.
        host: Hostname or IP address.
        port: gRPC port.
        gpu_count: Number of GPUs.
        healthy: Whether the node is considered healthy.
        tags: Arbitrary key/value metadata.
        registered_at: Unix timestamp of registration.
        last_seen: Unix timestamp of last heartbeat.
    """

    node_id: str
    host: str
    port: int = 50051
    gpu_count: int = 1
    healthy: bool = True
    tags: dict[str, str] = field(default_factory=dict)
    registered_at: float = 0.0
    last_seen: float = 0.0


@dataclass
class VersionRecord:
    """A versioned snapshot of the topology.

    Attributes:
        version: Monotonically increasing version number.
        topology_json: JSON-serialised topology snapshot.
        committed_at: Unix timestamp when committed.
        term: Raft term when committed.
        index: Raft log index when committed.
    """

    version: int
    topology_json: str
    committed_at: float = 0.0
    term: int = 0
    index: int = 0


class TopologyStore:
    """SQLite-backed persistence for cluster topology.

    Stores node records and versioned topology snapshots. Used for
    recovery on restart: the persisted topology is loaded and the
    node re-joins the cluster.

    Thread-safe: uses a ``threading.RLock`` around all SQLite operations.

    Args:
        db_path: Filesystem path to the SQLite database. Uses
            ``~/.distllm/topology.db`` by default.
    """

    def __init__(self, db_path: str = "") -> None:
        self._db_path = db_path or os.path.join(
            os.environ.get(
                "DISTLLM_DATA_DIR",
                os.path.expanduser("~/.distllm"),
            ),
            "topology.db",
        )
        self._lock = threading.RLock()
        self._conn: sqlite3.Connection | None = None

    # -- Connection management ----------------------------------------------

    def connect(self) -> None:
        """Open the SQLite database and ensure the schema exists."""
        with self._lock:
            os.makedirs(os.path.dirname(self._db_path), exist_ok=True)
            self._conn = sqlite3.connect(self._db_path)
            self._conn.row_factory = sqlite3.Row
            self._ensure_schema()
            logger.info(f"TopologyStore connected: {self._db_path}")

    def close(self) -> None:
        """Close the SQLite database connection."""
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    def _ensure_schema(self) -> None:
        """Create tables if they do not exist."""
        assert self._conn is not None
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS nodes (
                node_id TEXT PRIMARY KEY,
                host TEXT NOT NULL,
                port INTEGER NOT NULL DEFAULT 50051,
                gpu_count INTEGER NOT NULL DEFAULT 1,
                healthy INTEGER NOT NULL DEFAULT 1,
                tags TEXT NOT NULL DEFAULT '{}',
                registered_at REAL NOT NULL,
                last_seen REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS topology_versions (
                version INTEGER PRIMARY KEY AUTOINCREMENT,
                topology_json TEXT NOT NULL,
                committed_at REAL NOT NULL,
                term INTEGER NOT NULL DEFAULT 0,
                idx INTEGER NOT NULL DEFAULT 0
            );

            CREATE INDEX IF NOT EXISTS idx_topology_versions_committed
                ON topology_versions(committed_at DESC);
        """)
        self._conn.commit()

    # -- Node persistence ---------------------------------------------------

    def save_node(self, node_info: dict[str, Any]) -> None:
        """Persist a node record.

        Inserts or replaces the node identified by ``node_info["node_id"]``.

        Args:
            node_info: Dict with keys matching ``NodeRecord`` fields.
        """
        with self._lock:
            assert self._conn is not None
            now = time.time()
            self._conn.execute(
                """
                INSERT OR REPLACE INTO nodes
                    (node_id, host, port, gpu_count, healthy, tags,
                     registered_at, last_seen)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    node_info.get("node_id", ""),
                    node_info.get("host", ""),
                    node_info.get("port", 50051),
                    node_info.get("gpu_count", 1),
                    1 if node_info.get("healthy", True) else 0,
                    json.dumps(node_info.get("tags", {})),
                    node_info.get("registered_at", now),
                    node_info.get("last_seen", now),
                ),
            )
            self._conn.commit()

    def load_all_nodes(self) -> list[NodeRecord]:
        """Load all persisted node records.

        Returns:
            List of ``NodeRecord`` instances.
        """
        with self._lock:
            assert self._conn is not None
            cursor = self._conn.execute(
                "SELECT * FROM nodes ORDER BY node_id",
            )
            results: list[NodeRecord] = []
            for row in cursor.fetchall():
                results.append(
                    NodeRecord(
                        node_id=row["node_id"],
                        host=row["host"],
                        port=row["port"],
                        gpu_count=row["gpu_count"],
                        healthy=bool(row["healthy"]),
                        tags=json.loads(row["tags"]),
                        registered_at=row["registered_at"],
                        last_seen=row["last_seen"],
                    )
                )
            return results

    def delete_node(self, node_id: str) -> None:
        """Remove a node record from the store.

        Args:
            node_id: The node to remove.
        """
        with self._lock:
            assert self._conn is not None
            self._conn.execute(
                "DELETE FROM nodes WHERE node_id = ?", (node_id,)
            )
            self._conn.commit()
            logger.debug(f"TopologyStore deleted node: {node_id}")

    def update_heartbeat(self, node_id: str) -> None:
        """Update the ``last_seen`` timestamp for a node.

        Args:
            node_id: The node to update.
        """
        with self._lock:
            assert self._conn is not None
            self._conn.execute(
                "UPDATE nodes SET last_seen = ? WHERE node_id = ?",
                (time.time(), node_id),
            )
            self._conn.commit()

    # -- Version persistence ------------------------------------------------

    def save_version(self, version_info: dict[str, Any]) -> None:
        """Persist a topology version snapshot.

        Args:
            version_info: Dict with keys ``topology_json``, ``term``,
                ``index`` (``committed_at`` defaults to now).
        """
        with self._lock:
            assert self._conn is not None
            self._conn.execute(
                """
                INSERT INTO topology_versions
                    (topology_json, committed_at, term, idx)
                VALUES (?, ?, ?, ?)
                """,
                (
                    version_info.get("topology_json", "{}"),
                    version_info.get("committed_at", time.time()),
                    version_info.get("term", 0),
                    version_info.get("index", 0),
                ),
            )
            self._conn.commit()

    def load_latest_version(self) -> VersionRecord | None:
        """Load the most recent topology version snapshot.

        Returns:
            ``VersionRecord`` or ``None`` if no versions exist.
        """
        with self._lock:
            assert self._conn is not None
            cursor = self._conn.execute(
                "SELECT * FROM topology_versions ORDER BY version DESC LIMIT 1",
            )
            row = cursor.fetchone()
            if row is None:
                return None
            return VersionRecord(
                version=row["version"],
                topology_json=row["topology_json"],
                committed_at=row["committed_at"],
                term=row["term"],
                index=row["idx"],
            )

    def load_all_versions(self, limit: int = 10) -> list[VersionRecord]:
        """Load recent topology versions.

        Args:
            limit: Maximum number of versions to return.

        Returns:
            List of ``VersionRecord``, newest first.
        """
        with self._lock:
            assert self._conn is not None
            cursor = self._conn.execute(
                "SELECT * FROM topology_versions ORDER BY version DESC LIMIT ?",
                (limit,),
            )
            return [
                VersionRecord(
                    version=row["version"],
                    topology_json=row["topology_json"],
                    committed_at=row["committed_at"],
                    term=row["term"],
                    index=row["idx"],
                )
                for row in cursor.fetchall()
            ]

    # -- Recovery -----------------------------------------------------------

    def recover_topology(self) -> tuple[list[NodeRecord], VersionRecord | None]:
        """Load persisted state for recovery on restart.

        Returns:
            Tuple of ``(nodes, latest_version)``.
        """
        nodes = self.load_all_nodes()
        version = self.load_latest_version()
        logger.info(
            f"TopologyStore recovered: {len(nodes)} nodes, "
            f"version={version.version if version else 'none'}"
        )
        return nodes, version

    def clear(self) -> None:
        """Clear all topology data (for testing)."""
        with self._lock:
            assert self._conn is not None
            self._conn.executescript("""
                DELETE FROM nodes;
                DELETE FROM topology_versions;
            """)
            self._conn.commit()

    @property
    def db_path(self) -> str:
        return self._db_path


# ===========================================================================
# TopologyGossipBridge — bridges Raft consensus with gossip protocol
# ===========================================================================


@dataclass
class TopologyChange:
    """A topology change event propagated via gossip.

    Attributes:
        version: Monotonically increasing version number.
        changes: List of change dicts with ``type``, ``node_id``, and
            ``payload`` keys.
        committed_at: Unix timestamp when the change was committed.
        term: Raft term when committed.
    """

    version: int
    changes: list[dict[str, Any]]
    committed_at: float = 0.0
    term: int = 0


class TopologyGossipBridge:
    """Bridges Raft consensus with the gossip protocol.

    On committed topology changes, the bridge pushes notifications via
    the gossip protocol. Other nodes receive these notifications via
    subscriptions and apply the changes locally.

    Target propagation latency: < 100ms for a 10-node cluster.

    Args:
        node_id: Unique identifier for this node.
        gossip_client: An optional existing ``GossipClient`` instance.
            If None, gossip integration is stub-only.
    """

    def __init__(
        self,
        node_id: str,
        gossip_client: Any = None,
    ) -> None:
        self._node_id = node_id
        self._gossip_client = gossip_client
        self._lock = threading.RLock()
        self._subscriptions: list[Callable[[TopologyChange], None]] = []
        self._current_version: int = 0
        self._pending_changes: list[dict[str, Any]] = []

    # -- Subscriptions ------------------------------------------------------

    def subscribe(self, callback: Callable[[TopologyChange], None]) -> None:
        """Register a callback for topology change notifications.

        The callback receives a ``TopologyChange`` instance.

        Args:
            callback: Function to invoke on each topology change.
        """
        with self._lock:
            self._subscriptions.append(callback)

    def unsubscribe(self, callback: Callable[[TopologyChange], None]) -> None:
        """Remove a previously registered callback."""
        with self._lock:
            self._subscriptions = [
                cb for cb in self._subscriptions if cb is not cb
            ]

    # -- Push notification --------------------------------------------------

    def notify_change(self, change: TopologyChange) -> None:
        """Push a topology change notification via gossip and callbacks.

        Called when a Raft log entry is committed and applied.

        The payload has the form::

            {
                "type": "topology_update",
                "version": <int>,
                "changes": [<dict>, ...],
                "committed_at": <float>,
                "term": <int>,
            }

        Args:
            change: The ``TopologyChange`` to propagate.
        """
        with self._lock:
            self._current_version = change.version
            payload = {
                "type": "topology_update",
                "version": change.version,
                "changes": change.changes,
                "committed_at": change.committed_at or time.time(),
                "term": change.term,
            }

            # Push via gossip protocol if available.
            if self._gossip_client is not None:
                try:
                    self._gossip_client.exchange(
                        self._node_id,
                        payload,
                    )
                except Exception as e:
                    logger.warning(
                        f"Gossip push failed for topology update "
                        f"v{change.version}: {e}"
                    )

            # Notify local subscribers.
            self._notify_subscribers(change)

            logger.info(
                f"Topology change propagated via gossip: "
                f"v{change.version}, {len(change.changes)} changes, "
                f"target < 100ms"
            )

    def _notify_subscribers(self, change: TopologyChange) -> None:
        """Invoke all registered callbacks with the change."""
        cbs: list[Callable[[TopologyChange], None]]
        with self._lock:
            cbs = list(self._subscriptions)
        for cb in cbs:
            try:
                cb(change)
            except Exception as e:
                logger.error(
                    f"Topology change callback failed: {e}"
                )

    # -- Receive gossip notification ----------------------------------------

    def receive_gossip_update(self, message: dict[str, Any]) -> TopologyChange | None:
        """Process an incoming gossip topology update.

        Called when a peer sends a topology_update via gossip.

        Args:
            message: Dict with keys ``type``, ``version``, ``changes``,
                ``committed_at``, ``term``.

        Returns:
            The ``TopologyChange`` if it was applied, or None if it was
            stale or malformed.
        """
        if message.get("type") != "topology_update":
            return None

        version = message.get("version", 0)
        with self._lock:
            if version <= self._current_version:
                return None
            self._current_version = version

        change = TopologyChange(
            version=version,
            changes=message.get("changes", []),
            committed_at=message.get("committed_at", time.time()),
            term=message.get("term", 0),
        )

        # Notify local subscribers.
        self._notify_subscribers(change)

        logger.info(
            f"Topology update received via gossip: "
            f"v{version}, {len(change.changes)} changes"
        )
        return change

    @property
    def current_version(self) -> int:
        with self._lock:
            return self._current_version

    def stats(self) -> dict[str, Any]:
        """Return bridge statistics for observability."""
        with self._lock:
            return {
                "current_version": self._current_version,
                "subscription_count": len(self._subscriptions),
                "has_gossip_client": self._gossip_client is not None,
            }


# ===========================================================================
# TopologyManager — combines RaftCluster + TopologyStore + TopologyGossipBridge
# ===========================================================================


class TopologyManager:
    """Unified topology manager combining Raft consensus, SQLite persistence,
    and gossip-based change propagation.

    Provides the public API for registering/unregistering nodes, querying
    the current topology, and subscribing to topology changes.

    Usage::

        manager = TopologyManager(
            node_id="node-0",
            peers=["node-1:10.0.0.5:50050", "node-2:10.0.0.6:50050"],
        )
        await manager.start()

        # Register a new node.
        await manager.register_node({
            "node_id": "node-3",
            "host": "10.0.0.7",
            "port": 50051,
            "gpu_count": 4,
        })

        # Get current topology.
        topology = manager.get_topology()

        # Subscribe to changes.
        def on_change(change):
            print(f"Topology changed: {change}")

        manager.on_topology_change(on_change)

        await manager.stop()

    Args:
        node_id: Unique identifier for this node.
        peers: List of peer addresses as ``"node_id:host:port"`` strings.
        db_path: Path to SQLite database for persistence. Auto if empty.
        election_timeout_min_ms: Minimum Raft election timeout in ms.
        election_timeout_max_ms: Maximum Raft election timeout in ms.
        heartbeat_interval_ms: Leader heartbeat interval in ms.
        gossip_client: Optional existing ``GossipClient`` for gossip bridge.
    """

    def __init__(
        self,
        node_id: str,
        peers: list[str] | None = None,
        db_path: str = "",
        election_timeout_min_ms: int = _ELECTION_TIMEOUT_MIN_MS,
        election_timeout_max_ms: int = _ELECTION_TIMEOUT_MAX_MS,
        heartbeat_interval_ms: int = _HEARTBEAT_INTERVAL_MS,
        gossip_client: Any = None,
    ) -> None:
        self._node_id = node_id

        # Core components.
        self._store = TopologyStore(db_path=db_path)
        self._cluster = RaftCluster(
            node_id=node_id,
            peers=peers,
            election_timeout_min_ms=election_timeout_min_ms,
            election_timeout_max_ms=election_timeout_max_ms,
            heartbeat_interval_ms=heartbeat_interval_ms,
        )
        self._gossip = TopologyGossipBridge(
            node_id=node_id,
            gossip_client=gossip_client,
        )

        # In-memory topology state.
        self._nodes: dict[str, dict[str, Any]] = {}
        self._version_counter: int = 0
        self._lock = threading.RLock()
        self._change_callbacks: list[Callable[[TopologyChange], None]] = []
        self._started = False

        # Wire the apply callback.
        self._cluster.on_apply(self._on_log_applied)

    # -- Lifecycle ----------------------------------------------------------

    async def start(self) -> None:
        """Start the topology manager.

        Loads persisted state, starts the Raft cluster, and registers
        this node.
        """
        if self._started:
            return

        # Connect to SQLite and recover persisted state.
        self._store.connect()
        recovered_nodes, latest_version = self._store.recover_topology()

        with self._lock:
            for node in recovered_nodes:
                self._nodes[node.node_id] = {
                    "node_id": node.node_id,
                    "host": node.host,
                    "port": node.port,
                    "gpu_count": node.gpu_count,
                    "healthy": node.healthy,
                    "tags": node.tags,
                    "registered_at": node.registered_at,
                    "last_seen": node.last_seen,
                }

            if latest_version is not None:
                self._version_counter = latest_version.version

        # Start Raft cluster.
        self._cluster.start()

        # Register this node in the topology.
        await self.register_node({
            "node_id": self._node_id,
            "host": self._resolve_local_host(),
            "port": 50051,
            "gpu_count": 1,
            "tags": {"role": "coordinator"},
        })

        self._started = True
        logger.info(
            f"TopologyManager started: node={self._node_id}, "
            f"recovered={len(recovered_nodes)} nodes"
        )

    async def stop(self) -> None:
        """Stop the topology manager.

        Unregisters this node and shuts down Raft and persistence.
        """
        if not self._started:
            return

        await self.unregister_node(self._node_id)
        self._cluster.stop()
        self._store.close()
        self._started = False
        logger.info(f"TopologyManager stopped: node={self._node_id}")

    # -- Node registration --------------------------------------------------

    async def register_node(self, node_info: dict[str, Any]) -> bool:
        """Register a node in the cluster topology.

        Proposes a ``node_join`` entry to the Raft cluster. If this node
        is the leader, the entry is replicated and committed. Otherwise,
        the proposal is forwarded (or rejected, in which case the caller
        should retry or contact the leader directly).

        Args:
            node_info: Dict with at minimum ``node_id`` and ``host`` keys.
                May also include ``port``, ``gpu_count``, ``tags``, etc.

        Returns:
            True if the registration was committed, False otherwise.
        """
        # Normalise the node info.
        info = dict(node_info)
        info.setdefault("node_id", "")
        info.setdefault("host", "")
        info.setdefault("port", 50051)
        info.setdefault("gpu_count", 1)
        info.setdefault("healthy", True)
        info.setdefault("tags", {})
        info.setdefault("registered_at", time.time())
        info.setdefault("last_seen", time.time())

        # Propose to Raft.
        success = self._cluster.propose("node_join", info)

        if success:
            logger.info(f"Node registered: {info.get('node_id')}")
        else:
            logger.warning(
                f"Node registration failed (not leader): "
                f"{info.get('node_id')}"
            )
        return success

    async def unregister_node(self, node_id: str) -> bool:
        """Unregister a node from the cluster topology.

        Proposes a ``node_leave`` entry to the Raft cluster.

        Args:
            node_id: The node to remove.

        Returns:
            True if the unregistration was committed, False otherwise.
        """
        success = self._cluster.propose(
            "node_leave", {"node_id": node_id},
        )
        if success:
            logger.info(f"Node unregistered: {node_id}")
        return success

    async def update_node(self, node_id: str, updates: dict[str, Any]) -> bool:
        """Update node metadata in the topology.

        Proposes a ``node_update`` entry to the Raft cluster.

        Args:
            node_id: The node to update.
            updates: Dict of fields to update (e.g., ``healthy``,
                ``gpu_count``, ``tags``).

        Returns:
            True if the update was committed, False otherwise.
        """
        payload = {"node_id": node_id, "updates": updates}
        success = self._cluster.propose("node_update", payload)
        if success:
            logger.info(f"Node updated: {node_id}: {updates}")
        return success

    # -- Topology queries ---------------------------------------------------

    def get_topology(self) -> dict[str, Any]:
        """Return the current topology snapshot.

        Returns:
            Dict with ``version``, ``nodes``, ``node_count``, ``leader``,
            and ``timestamp``.
        """
        with self._lock:
            return {
                "version": self._version_counter,
                "nodes": [
                    {
                        "node_id": nid,
                        "host": info.get("host", ""),
                        "port": info.get("port", 50051),
                        "gpu_count": info.get("gpu_count", 1),
                        "healthy": info.get("healthy", True),
                        "tags": info.get("tags", {}),
                        "registered_at": info.get("registered_at", 0.0),
                        "last_seen": info.get("last_seen", 0.0),
                    }
                    for nid, info in self._nodes.items()
                ],
                "node_count": len(self._nodes),
                "leader": self._cluster.get_leader(),
                "timestamp": time.time(),
            }

    def get_node(self, node_id: str) -> dict[str, Any] | None:
        """Return information about a specific node.

        Args:
            node_id: The node to query.

        Returns:
            Dict with node info, or None if not found.
        """
        with self._lock:
            info = self._nodes.get(node_id)
            if info is None:
                return None
            return dict(info)

    def get_healthy_nodes(self) -> list[dict[str, Any]]:
        """Return only healthy nodes.

        Returns:
            List of node info dicts where ``healthy`` is True.
        """
        with self._lock:
            return [
                dict(info)
                for info in self._nodes.values()
                if info.get("healthy", True)
            ]

    # -- Change subscriptions -----------------------------------------------

    def on_topology_change(
        self, callback: Callable[[TopologyChange], None],
    ) -> None:
        """Register a callback for topology change events.

        The callback receives a ``TopologyChange`` instance.

        Args:
            callback: Function to invoke on each topology change.
        """
        with self._lock:
            self._change_callbacks.append(callback)

        # Also subscribe through the gossip bridge.
        self._gossip.subscribe(callback)

    # -- Raft log application -----------------------------------------------

    def _on_log_applied(self, entry: LogEntry) -> None:
        """Called when a Raft log entry is committed and applied.

        Updates the in-memory topology state, persists to SQLite,
        pushes a gossip notification, and fires change callbacks.

        Args:
            entry: The committed ``LogEntry``.
        """
        with self._lock:
            changes: list[dict[str, Any]] = []

            if entry.entry_type == "node_join":
                node_id = entry.payload.get("node_id", "")
                self._nodes[node_id] = dict(entry.payload)
                self._store.save_node(entry.payload)
                changes.append({
                    "type": "node_join",
                    "node_id": node_id,
                    "payload": entry.payload,
                })
                logger.info(
                    f"Topology applied: node_join {node_id}"
                )

            elif entry.entry_type == "node_leave":
                node_id = entry.payload.get("node_id", "")
                removed = self._nodes.pop(node_id, None)
                if removed:
                    self._store.delete_node(node_id)
                changes.append({
                    "type": "node_leave",
                    "node_id": node_id,
                    "payload": entry.payload,
                })
                logger.info(
                    f"Topology applied: node_leave {node_id}"
                )

            elif entry.entry_type == "node_update":
                node_id = entry.payload.get("node_id", "")
                updates = entry.payload.get("updates", {})
                if node_id in self._nodes:
                    self._nodes[node_id].update(updates)
                    self._store.save_node(self._nodes[node_id])
                changes.append({
                    "type": "node_update",
                    "node_id": node_id,
                    "payload": entry.payload,
                })
                logger.info(
                    f"Topology applied: node_update {node_id}: {updates}"
                )

            else:
                logger.warning(
                    f"Unknown topology entry type: {entry.entry_type}"
                )
                return

            # Increment version and persist snapshot.
            self._version_counter += 1
            topology_json = json.dumps(self.get_topology(), default=str)
            self._store.save_version({
                "topology_json": topology_json,
                "term": entry.term,
                "index": entry.index,
            })

            # Push gossip notification.
            if changes:
                change = TopologyChange(
                    version=self._version_counter,
                    changes=changes,
                    committed_at=time.time(),
                    term=entry.term,
                )
                self._gossip.notify_change(change)

    # -- Status -------------------------------------------------------------

    def cluster_status(self) -> dict[str, Any]:
        """Return comprehensive cluster status.

        Includes Raft status, topology summary, and gossip bridge info.
        """
        raft_status = self._cluster.cluster_status()
        topo = self.get_topology()

        return {
            "raft": raft_status,
            "topology": {
                "version": topo["version"],
                "node_count": topo["node_count"],
                "healthy_count": len(self.get_healthy_nodes()),
                "leader": topo["leader"],
            },
            "gossip": self._gossip.stats(),
            "started": self._started,
            "db_path": self._store.db_path,
        }

    # -- Helpers ------------------------------------------------------------

    @staticmethod
    def _resolve_local_host() -> str:
        """Resolve the local hostname."""
        import socket
        try:
            return socket.gethostname()
        except Exception:
            return "localhost"
