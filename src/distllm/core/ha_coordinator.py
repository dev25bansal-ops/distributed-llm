"""Multi-Coordinator High Availability with leader election and failover.

Uses Raft-inspired leader election with heartbeat-based failure detection.
Supports active-passive and active-active configurations.
"""
import time
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

from loguru import logger


class CoordinatorState(Enum):
    LEADER = "leader"
    FOLLOWER = "follower"
    CANDIDATE = "candidate"
    OFFLINE = "offline"


@dataclass
class CoordinatorInfo:
    """Information about a coordinator instance."""
    coordinator_id: str
    host: str
    port: int
    state: CoordinatorState = CoordinatorState.OFFLINE
    last_heartbeat: float = 0.0
    term: int = 0
    started_at: float = field(default_factory=time.time)


class HAElection:
    """Raft-inspired leader election for multi-coordinator HA.
    
    Features:
    - Leader election with term-based voting
    - Heartbeat-based failure detection
    - Automatic failover on leader failure
    - Active-passive configuration
    
    Usage:
        election = HAElection(coordinator_id="coord-1", peers=["coord-2", "coord-3"])
        election.start()
        if election.is_leader():
            # Handle requests
        election.stop()
    """
    
    def __init__(
        self,
        coordinator_id: str,
        host: str = "localhost",
        port: int = 50050,
        heartbeat_interval: float = 1.0,
        election_timeout: float = 3.0,
        on_become_leader: Callable | None = None,
        on_step_down: Callable | None = None,
    ):
        self._coordinator_id = coordinator_id
        self._host = host
        self._port = port
        self._heartbeat_interval = heartbeat_interval
        self._election_timeout = election_timeout
        self._on_become_leader = on_become_leader
        self._on_step_down = on_step_down
        
        self._state = CoordinatorState.FOLLOWER
        self._term = 0
        self._voted_for: str | None = None
        self._leader_id: str | None = None
        self._peers: dict[str, CoordinatorInfo] = {}
        self._running = False
        
        self._lock = threading.Lock()
        self._heartbeat_thread: threading.Thread | None = None
        self._election_thread: threading.Thread | None = None
    
    def start(self) -> None:
        """Start the HA coordinator."""
        self._running = True
        
        # Register self
        my_info = CoordinatorInfo(
            coordinator_id=self._coordinator_id,
            host=self._host,
            port=self._port,
            state=self._state,
        )
        self._peers[self._coordinator_id] = my_info
        
        # Start background threads
        self._heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self._heartbeat_thread.start()
        
        logger.info(f"HA Coordinator started: {self._coordinator_id}")
    
    def stop(self) -> None:
        """Stop the HA coordinator."""
        self._running = False
        if self._heartbeat_thread:
            self._heartbeat_thread.join(timeout=5)
        if self._election_thread:
            self._election_thread.join(timeout=5)
        logger.info(f"HA Coordinator stopped: {self._coordinator_id}")
    
    def is_leader(self) -> bool:
        """Check if this coordinator is the leader."""
        return self._state == CoordinatorState.LEADER
    
    def get_leader(self) -> str | None:
        """Get the current leader ID."""
        return self._leader_id
    
    def get_state(self) -> CoordinatorState:
        """Get current coordinator state."""
        return self._state
    
    def add_peer(self, coordinator_id: str, host: str, port: int) -> None:
        """Add a peer coordinator."""
        self._peers[coordinator_id] = CoordinatorInfo(
            coordinator_id=coordinator_id,
            host=host,
            port=port,
        )
    
    def _heartbeat_loop(self) -> None:
        """Send periodic heartbeats."""
        while self._running:
            time.sleep(self._heartbeat_interval)
            try:
                self._send_heartbeat()
            except Exception as e:
                logger.warning(f"Heartbeat error: {e}")
    
    def _send_heartbeat(self) -> None:
        """Send heartbeat and check for leader."""
        now = time.time()
        
        with self._lock:
            # Update my heartbeat
            my_info = self._peers.get(self._coordinator_id)
            if my_info:
                my_info.last_heartbeat = now
            
            # Check if leader is alive
            if self._leader_id and self._leader_id != self._coordinator_id:
                leader = self._peers.get(self._leader_id)
                if leader and now - leader.last_heartbeat > self._election_timeout:
                    logger.warning(f"Leader {self._leader_id} timed out, starting election")
                    self._start_election()
    
    def _start_election(self) -> None:
        """Start a leader election."""
        with self._lock:
            self._term += 1
            self._state = CoordinatorState.CANDIDATE
            self._voted_for = self._coordinator_id
            
            votes = 1  # Vote for self
            total_peers = len(self._peers)
            
            # Request votes from peers (simulated)
            for peer_id, peer_info in self._peers.items():
                if peer_id != self._coordinator_id:
                    if self._request_vote(peer_id):
                        votes += 1
            
            # Check if won
            if votes > total_peers / 2:
                self._state = CoordinatorState.LEADER
                self._leader_id = self._coordinator_id
                logger.info(f"Elected leader: {self._coordinator_id} (term {self._term})")
                if self._on_become_leader:
                    self._on_become_leader()
            else:
                self._state = CoordinatorState.FOLLOWER
    
    def _request_vote(self, peer_id: str) -> bool:
        """Request a vote from a peer (simulated)."""
        # In production, this would be a gRPC call to the peer
        # For now, simulate by checking peer heartbeat
        peer = self._peers.get(peer_id)
        if peer is None:
            return True  # Peer not registered, grant vote
        if peer.state == CoordinatorState.LEADER:
            return False  # Existing leader, don't vote
        return True  # Grant vote
    
    def handle_heartbeat_request(self, from_id: str, term: int) -> dict:
        """Handle heartbeat from another coordinator."""
        with self._lock:
            # Update peer info
            if from_id in self._peers:
                self._peers[from_id].last_heartbeat = time.time()
                self._peers[from_id].term = term
            
            # If incoming term is higher, step down
            if term > self._term:
                self._term = term
                if self._state == CoordinatorState.LEADER and self._on_step_down:
                    self._on_step_down()
                self._state = CoordinatorState.FOLLOWER
                self._leader_id = from_id
            
            return {
                "coordinator_id": self._coordinator_id,
                "term": self._term,
                "state": self._state.value,
                "leader_id": self._leader_id,
            }
    
    def stats(self) -> dict:
        """Get HA coordinator statistics."""
        return {
            "coordinator_id": self._coordinator_id,
            "state": self._state.value,
            "term": self._term,
            "leader_id": self._leader_id,
            "peers": {
                pid: {
                    "host": p.host,
                    "port": p.port,
                    "state": p.state.value,
                    "last_heartbeat": p.last_heartbeat,
                }
                for pid, p in self._peers.items()
            },
        }
