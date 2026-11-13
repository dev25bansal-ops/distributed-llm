"""Regression tests for the HA fail-open and constructor-ordering defects.

Three reproduced defects (audit fleet 2026-11-13):

1. RayFaultTolerance.add_peer seeded every peer ``online=True,
   last_seen=now``, so a freshly started coordinator counted the full
   membership alive without a single heartbeat — a node restarting inside a
   partition saw a phantom majority, self-elected, and accepted writes for
   a full election window while the real leader was still serving.
2. The ``info.get("online", True)`` defaults made any peer dict missing the
   key count as alive (fail-open).
3. Coordinator.__init__ called set_replication_peers() before
   self._running existed, so ha_enabled + ha_replication_peers crashed the
   constructor with AttributeError — the whole HA path was unreachable.
"""

from __future__ import annotations

from typing import Any

from distllm.core.ha_coordinator import CoordinatorState, RayFaultTolerance


def _dead_transport(*args: Any, **kwargs: Any) -> dict[str, Any]:
    raise ConnectionError("test: unreachable")


class TestAddPeerFailClosed:
    def test_new_peer_starts_unconfirmed(self) -> None:
        n = RayFaultTolerance("coordinator-a")
        n.add_peer("coordinator-b", "127.0.0.1", 5002)
        info = n._peers["coordinator-b"]
        assert info["online"] is False
        assert info["last_seen"] == 0.0

    def test_never_contacted_peer_does_not_grant_quorum(self) -> None:
        """The split-brain scenario: restart inside a partition.

        A 3-node cluster {a,b,c}. Node a restarts and cannot reach anyone.
        Before the fix, a counted b and c alive for a full election window
        and self-elected within one round. It must not.
        """
        a = RayFaultTolerance("coordinator-a", election_timeout_s=10.0)
        a.add_peer("coordinator-b", "127.0.0.1", 5002)
        a.add_peer("coordinator-c", "127.0.0.1", 5003)
        a.set_heartbeat_transport(_dead_transport)

        a._run_election_round()

        assert not a.is_leader(), "self-elected on never-contacted peers"
        assert a.get_state() == CoordinatorState.FOLLOWER
        assert not a.has_quorum()

    def test_reachable_peer_is_confirmed_in_first_round(self) -> None:
        """Fail-closed must not break the healthy path: one probe round."""
        a = RayFaultTolerance("coordinator-a", election_timeout_s=10.0)
        b = RayFaultTolerance("coordinator-b", election_timeout_s=10.0)
        a.add_peer("coordinator-b", "127.0.0.1", 5002)
        b.add_peer("coordinator-a", "127.0.0.1", 5001)

        def linked(peer_id: str, host: str, port: int, payload: dict) -> dict:
            target = b if peer_id == "coordinator-b" else a
            return target.handle_heartbeat_request(
                payload.get("coordinator_id", ""), payload.get("term", 0), payload.get("state"),
            )

        a.set_heartbeat_transport(linked)
        b.set_heartbeat_transport(linked)

        for _ in range(10):
            a._run_election_round()
            b._run_election_round()

        assert a._peers["coordinator-b"]["online"] is True
        leaders = [n for n in (a, b) if n.is_leader()]
        assert len(leaders) == 1

    def test_missing_online_key_counts_as_dead(self) -> None:
        """The dict-default is fail-closed now, not fail-open."""
        n = RayFaultTolerance("coordinator-a")
        n.add_peer("coordinator-b", "127.0.0.1", 5002)
        del n._peers["coordinator-b"]["online"]  # simulate a legacy/partial dict
        n.set_heartbeat_transport(_dead_transport)
        assert not n.has_quorum()


class TestCoordinatorHaConstructorOrdering:
    def test_ha_enabled_with_replication_peers_constructs(self) -> None:
        """Used to raise AttributeError: 'Coordinator' object has no
        attribute '_running' (created after the HA block read it)."""
        from distllm.core.coordinator import Coordinator

        c = Coordinator(
            model_name="m",
            ha_enabled=True,
            ha_replication_peers=["http://10.0.0.2:8000"],
        )
        assert c._running is not None
        assert c._election._replication_peers == ["http://10.0.0.2:8000"]


class TestStaleLeaderCannotDeposeByIDAlone:
    """handle_heartbeat_request used to step a leader down on ANY heartbeat
    from a lower-ID peer, with no term comparison — so a stale old leader
    (partitioned, rejoining at its old term) unseated the valid one."""

    def test_lower_id_stale_term_does_not_depose_leader(self) -> None:
        n = RayFaultTolerance("coordinator-b")
        n._state = CoordinatorState.LEADER
        n._leader_id = "coordinator-b"
        n._current_term = 5

        n.handle_heartbeat_request("coordinator-a", term=3)  # lower ID, stale term

        assert n.get_state() == CoordinatorState.LEADER, "stale lower-term peer deposed the leader"
        assert n.get_leader() == "coordinator-b"

    def test_lower_id_equal_term_is_followed(self) -> None:
        n = RayFaultTolerance("coordinator-b")
        n._state = CoordinatorState.LEADER
        n._leader_id = "coordinator-b"
        n._current_term = 5

        n.handle_heartbeat_request("coordinator-a", term=5)

        assert n.get_state() == CoordinatorState.FOLLOWER
        assert n.get_leader() == "coordinator-a"

    def test_lower_id_higher_term_is_followed(self) -> None:
        n = RayFaultTolerance("coordinator-b")
        n._state = CoordinatorState.LEADER
        n._current_term = 3

        n.handle_heartbeat_request("coordinator-a", term=7)

        assert n.get_state() == CoordinatorState.FOLLOWER
        assert n._current_term == 7


class TestWriteFenceRunsWithoutDetector:
    """replicate_state's fence only ran when a split_brain_detector was
    installed, and nothing in production ever installed one — so the
    documented write fence never executed. It must consult the built-in
    quorum view on every leader."""

    def test_solo_cluster_still_writes(self) -> None:
        """Default single-node deployments (size 1, quorum 1) must not be
        fenced out by the built-in view."""
        n = RayFaultTolerance("coordinator-a")
        n._state = CoordinatorState.LEADER
        n.replicate_state("k", "v")
        assert n.get_replicated_state() == {"k": "v"}

    def test_leader_without_quorum_refuses_write_and_steps_down(self) -> None:
        n = RayFaultTolerance("coordinator-a")
        n._initial_cluster_size = 3
        n.add_peer("coordinator-b", "127.0.0.1", 5002)
        n.add_peer("coordinator-c", "127.0.0.1", 5003)
        n.set_heartbeat_transport(_dead_transport)
        n._run_election_round()  # sets states; force leader for the fence test
        n._state = CoordinatorState.LEADER
        n._leader_id = "coordinator-a"

        n.replicate_state("k", "v")

        assert n.get_replicated_state() == {}, "write accepted without quorum"
        assert n.get_state() == CoordinatorState.FOLLOWER

    def test_leader_with_quorum_writes(self) -> None:
        n = RayFaultTolerance("coordinator-a")
        n._initial_cluster_size = 3
        n.add_peer("coordinator-b", "127.0.0.1", 5002)
        n.add_peer("coordinator-c", "127.0.0.1", 5003)
        n._peers["coordinator-b"]["online"] = True
        n._peers["coordinator-b"]["last_seen"] = 1e18  # confirmed alive
        n._state = CoordinatorState.LEADER
        n._leader_id = "coordinator-a"

        n.replicate_state("k", "v")

        assert n.get_replicated_state() == {"k": "v"}
