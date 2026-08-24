"""HA election and state replication extracted from Coordinator.

Provides the ``CoordinatorElection`` class that encapsulates leader
election, HA status, state snapshots, and peer replication logic.
The coordinator instantiates this class and delegates HA methods to it.
"""

from __future__ import annotations

import os
import threading
import time
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from distllm.core.coordinator import Coordinator


def _ha_auth_headers() -> dict[str, str]:
    """Build auth headers matching the HA receiver's fail-closed check.

    The receiver (``POST /api/v1/ha/snapshot`` in api/server.py) rejects any
    request whose ``X-HA-Secret`` header does not match ``DISTLLM_HA_SECRET``.
    Senders must attach the same secret or every push is silently 403'd.
    """
    secret = os.environ.get("DISTLLM_HA_SECRET", "")
    return {"X-HA-Secret": secret} if secret else {}


class CoordinatorElection:
    """HA leader election and state replication for the Coordinator.

    Takes the coordinator instance as a parameter and accesses its
    attributes and methods for state snapshot, replication, and failover.

    The coordinator should create this once in ``__init__`` and delegate
    all HA-related public and private methods to this object.
    """

    def __init__(self, coordinator: Coordinator) -> None:
        self._coordinator = coordinator

        # High-availability election (optional)
        self._ha_election: Any = None
        self._is_standby = False

        self._replication_thread: threading.Thread | None = None
        self._replication_peers: list[str] = []

    # ── Leader Election ──

    def enable_ha(
        self,
        coordinator_id: str | None = None,
        peer_coordinators: list[tuple[str, str, int]] | None = None,
        heartbeat_interval_s: float = 2.0,
        election_timeout_s: float = 10.0,
    ) -> None:
        """Enable high-availability mode with leader election.

        When enabled, this coordinator participates in leader election
        with peer coordinators. Only the leader accepts requests; standbys
        replicate state and wait for failover.

        Args:
            coordinator_id: Unique ID for this coordinator. Defaults to
                hostname:port.
            peer_coordinators: List of (id, host, port) tuples for peers.
            heartbeat_interval_s: Seconds between heartbeats.
            election_timeout_s: Seconds without heartbeat before election.
        """
        from distllm.core.ha_coordinator import RayFaultTolerance

        cid = coordinator_id or f"{self._coordinator.model_name}:{self._coordinator.port}"
        self._ha_election = RayFaultTolerance(
            coordinator_id=cid,
            heartbeat_interval_s=heartbeat_interval_s,
            election_timeout_s=election_timeout_s,
        )

        if peer_coordinators:
            for peer_id, peer_host, peer_port in peer_coordinators:
                self._ha_election.add_peer(peer_id, peer_host, peer_port)

        self._ha_election.start()

        # B6: standby replication was a permanent no-op because _is_standby
        # was initialized False and never updated, so apply_state_snapshot()
        # always returned early and standbys never warmed their state. Mark
        # this node as a standby when HA is enabled and it is not the
        # elected leader, and wire the election's state-change callback so a
        # leader-pushed snapshot is applied on the standby.
        self._is_standby = not self.is_leader
        self._ha_election.on_state_change(self._on_leader_state)

        logger.info(f"HA enabled for coordinator {cid}")

    @property
    def is_leader(self) -> bool:
        """Return True if this coordinator is the elected leader."""
        if self._ha_election is None:
            return True  # No HA = always leader
        return self._ha_election.is_leader()

    @property
    def ha_status(self) -> dict:
        """Return HA election status."""
        if self._ha_election is None:
            return {"enabled": False}
        return self._ha_election.stats()

    # ── State Snapshots ──

    def state_snapshot(self) -> dict[str, Any]:
        """Create a snapshot of coordinator state for replication.

        Standby coordinators can use this to maintain a warm copy
        of the leader's state for fast failover.

        Returns:
            Dict with node registrations, model info, and config.
        """
        coord = self._coordinator
        return {
            "model_name": coord.model_name,
            "total_layers": coord.total_layers,
            "nodes": {
                nid: {
                    "host": getattr(n, "host", ""),
                    "port": getattr(n, "port", 0),
                    "start_layer": getattr(n, "start_layer", 0),
                    "end_layer": getattr(n, "end_layer", 0),
                    # PipelineNode's canonical health attribute is
                    # ``is_healthy`` (schedulers filter on it); ``.healthy``
                    # was never defined on the dataclass so every node used
                    # to snapshot as unhealthy.
                    "healthy": bool(
                        getattr(n, "is_healthy", getattr(n, "healthy", True))
                    ),
                }
                for nid, n in coord.nodes.items()
            },
            "node_order": list(coord.node_order),
            "timestamp": time.time(),
        }

    def apply_state_snapshot(self, snapshot: dict[str, Any]) -> None:
        """Apply a state snapshot from the leader (for standby coordinators).

        Re-registers nodes and updates internal state to match the leader.
        """
        # B6: re-sync standby status so a promoted leader immediately stops
        # applying snapshots while a standby keeps warming its state.
        self._sync_standby_status()
        if not self._is_standby:
            logger.warning("apply_state_snapshot called on non-standby coordinator")
            return

        nodes = snapshot.get("nodes", {})
        for nid, info in nodes.items():
            if nid not in self._coordinator.nodes:
                try:
                    self._coordinator.manual_register(
                        node_id=nid,
                        host=info["host"],
                        port=info["port"],
                        start_layer=info["start_layer"],
                        end_layer=info["end_layer"],
                    )
                except Exception as e:
                    logger.warning(f"Failed to apply snapshot node {nid}: {e}")

        logger.info(f"Applied state snapshot: {len(nodes)} nodes")

    def _sync_standby_status(self) -> None:
        """Re-derive standby status from the current leadership state.

        A coordinator is a standby when HA is enabled and it is not the
        elected leader. Promoting to leader resets the flag so the node
        stops applying leader snapshots. When HA is disabled the flag is
        left untouched so direct callers (and tests) that manage it
        explicitly keep working.
        """
        if self._ha_election is not None:
            self._is_standby = not self.is_leader

    def _on_leader_state(self, state: dict[str, Any]) -> None:
        """Apply a state snapshot pushed by the leader via the election.

        Invoked from the HA heartbeat path when the leader's replicated
        state arrives. Re-syncs standby status first so a promoted node
        immediately stops acting as a standby, then applies the snapshot.
        """
        self._sync_standby_status()
        if state:
            self.apply_state_snapshot(state)

    # ── HA State Replication ──

    def _start_state_replication(self) -> None:
        """Start continuous state replication to HA peer coordinators.

        Runs a background thread that pushes state snapshots to
        all configured peers every 1 second for sub-second failover.
        """
        if not self._replication_peers:
            return
        self._replication_thread = threading.Thread(
            target=self._replication_loop,
            daemon=True,
            name="state-replication",
        )
        self._replication_thread.start()
        logger.info(f"State replication started for {len(self._replication_peers)} peers")

    def _replication_loop(self) -> None:
        """Continuously push state snapshots to HA peers.

        Uses a 1-second interval, but only sends a full snapshot every
        10th call (10s) when the cluster is stable.  Intermediate ticks
        send a lightweight heartbeat to detect peer liveness.  This
        reduces the O(n * peers) serialization + network overhead for
        large clusters by ~90% during steady state.
        """
        import httpx
        tick = 0
        with httpx.Client(timeout=2.0) as client:
            while self._coordinator._running.is_set():
                tick += 1
                try:
                    # Full snapshot every ~10s, lightweight ping otherwise
                    if tick % 10 == 0:
                        snapshot = self.state_snapshot()
                    else:
                        snapshot = {
                            "heartbeat": True,
                            "node_count": len(self._coordinator.nodes),
                            "healthy": self._coordinator._health_mgr.is_healthy(),
                        }
                    for peer_url in self._replication_peers:
                        try:
                            resp = client.post(
                                f"{peer_url.rstrip('/')}/api/v1/ha/snapshot",
                                json=snapshot,
                                headers=_ha_auth_headers(),
                            )
                            if resp.status_code != 200:
                                logger.warning(
                                    f"Replication to {peer_url} returned {resp.status_code}"
                                )
                        except Exception as e:
                            logger.warning(f"Replication to {peer_url} failed: {e}")
                except Exception as e:
                    logger.warning(f"State replication error: {e}")
                time.sleep(1.0)

    def set_replication_peers(self, peer_urls: list[str]) -> None:
        """Set HA peer coordinator URLs for state replication.

        Args:
            peer_urls: List of peer API base URLs (e.g. ["http://10.0.0.2:8000"]).
        """
        self._replication_peers = peer_urls
        if self._coordinator._running.is_set():
            self._start_state_replication()
