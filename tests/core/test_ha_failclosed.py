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


class TestPartitionedRestartJoinsAsNonLeader:
    """The audit's dual-leader variant: 3-node cluster {a,b,c}, a restarts
    inside a partition. Before the fix a counted b and c as online without
    one heartbeat, co-founded a second quorum, and accepted writes while b
    was the elected leader. Exactly one leader, exactly one writer."""

    def _three_node_partition(self, run_order: tuple[str, ...]):
        a = RayFaultTolerance("coordinator-a", election_timeout_s=10.0)
        b = RayFaultTolerance("coordinator-b", election_timeout_s=10.0)
        c = RayFaultTolerance("coordinator-c", election_timeout_s=10.0)
        by_id = {"a": a, "b": b, "c": c}
        nodes = {"coordinator-a": a, "coordinator-b": b, "coordinator-c": c}
        for n in (a, b, c):
            for pid in nodes:
                if pid != n._id:
                    n.add_peer(pid, "127.0.0.1", 5000)

        # a is partitioned from everyone; its probes all fail.
        a.set_heartbeat_transport(_dead_transport)

        def linked(peer_id: str, host: str, port: int, payload: dict) -> dict:
            if peer_id == "coordinator-a":
                raise ConnectionError("a is partitioned")
            return nodes[peer_id].handle_heartbeat_request(
                payload.get("coordinator_id", ""), payload.get("term", 0), payload.get("state"),
            )

        b.set_heartbeat_transport(linked)
        c.set_heartbeat_transport(linked)
        # From b/c, the partitioned a has never been heard from.
        for n in (b, c):
            n._peers["coordinator-a"]["last_seen"] = 0.0
            n._peers["coordinator-a"]["online"] = False

        for nid in run_order:
            by_id[nid]._run_election_round()
        return a, b, c

    def test_partitioned_restart_never_co_leads(self) -> None:
        for order in (("a", "b", "c"), ("b", "c", "a")):
            a, b, c = self._three_node_partition(order)
            leaders = [n._id for n in (a, b, c) if n.is_leader()]
            writers = [n._id for n in (a, b, c) if n.can_accept_writes()]
            assert leaders == ["coordinator-b"], f"order={order}: leaders={leaders}"
            assert writers == ["coordinator-b"], f"order={order}: writers={writers}"

    def test_healthy_majority_elects_new_leader(self) -> None:
        """The remaining two nodes (b, c) still form a majority: one of
        them must lead, so the fix does not break failover."""
        a, b, c = self._three_node_partition(("c", "b", "a"))
        assert b.is_leader()
        assert not a.is_leader()
        assert not c.is_leader()


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


class TestNoQuorumNodeCannotStoreState:
    """The write fence must cover every node, not only leaders.

    The audit's check for the 'dead write fence' finding ran
    replicate_state on a node that had *lost* its self-election (after the
    fail-closed election fix it is a FOLLOWER with no quorum) and observed
    the value stored anyway. A node that cannot confirm a majority must not
    diverge its state history regardless of its role.
    """

    def test_no_quorum_follower_refuses_state_store(self) -> None:
        n = RayFaultTolerance("coordinator-a")
        n.add_peer("coordinator-b", "127.0.0.1", 5002)
        n.set_heartbeat_transport(_dead_transport)
        n._run_election_round()
        assert n.get_state() == CoordinatorState.FOLLOWER
        assert not n.has_quorum()

        n.replicate_state("topology", {"v": 2})

        assert n.get_replicated_state() == {}, "no-quorum node stored replicated state"

    def test_no_quorum_follower_with_initial_size_3_refuses(self) -> None:
        n = RayFaultTolerance("coordinator-a")
        n._initial_cluster_size = 3
        n.add_peer("coordinator-b", "127.0.0.1", 5002)
        n.add_peer("coordinator-c", "127.0.0.1", 5003)
        n.set_heartbeat_transport(_dead_transport)
        n._run_election_round()

        n.replicate_state("topology", {"v": 2})

        assert n.get_replicated_state() == {}


class TestStaleSenderCannotRegressStandbyState:
    """handle_heartbeat_request used to apply ANY sender's replicated state
    whenever the local node was a FOLLOWER — no leader check, no term check
    (plain dict.update). A demoted ex-leader, or a lagging follower whose
    probe still carried its old copy, regressed the standby's newer state.
    State must only be applied from the node this heartbeat establishes as
    the leader, and never from a sender behind the current term."""

    def test_stale_lower_term_sender_cannot_regress_state(self) -> None:
        """The audit's exact scenario: leader c pushes v10; the stale
        ex-leader a (term 1) probes with its old v7; v10 must survive."""
        b = RayFaultTolerance("coordinator-b")
        b.handle_heartbeat_request("coordinator-c", 9, {"topology": "v10"})
        assert b.get_replicated_state() == {"topology": "v10"}

        b.handle_heartbeat_request("coordinator-a", 1, {"topology": "v7"})

        assert b.get_replicated_state() == {"topology": "v10"}, (
            "a stale lower-term sender regressed the standby's state"
        )
        assert b.get_leader() == "coordinator-c", "stale sender stole the leader belief"

    def test_higher_term_sender_is_recognized_and_applied(self) -> None:
        b = RayFaultTolerance("coordinator-b")
        b.handle_heartbeat_request("coordinator-c", 5, {"topology": "v10"})
        assert b.get_leader() == "coordinator-c"
        assert b.get_replicated_state() == {"topology": "v10"}

    def test_equal_term_higher_id_non_leader_state_rejected(self) -> None:
        """A same-term sender that never won an election (higher ID than
        us at equal term establishes nothing) cannot push state."""
        b = RayFaultTolerance("coordinator-b")
        b.handle_heartbeat_request("coordinator-z", 0, {"evil": True})
        assert b.get_replicated_state() == {}
        assert b.get_leader() is None

    def test_leader_keeps_its_own_state_against_lower_peer(self) -> None:
        """A recognised leader (higher term than the sender) must retain
        both leadership and state while a stale peer probes at an old term."""
        n = RayFaultTolerance("coordinator-b")
        n._state = CoordinatorState.LEADER
        n._leader_id = "coordinator-b"
        n._current_term = 7

        n.handle_heartbeat_request("coordinator-a", 3, {"topology": "v7"})

        assert n.get_state() == CoordinatorState.LEADER
        assert n.get_replicated_state() == {}


class TestProbeDoesNotPropagateStaleState:
    """_probe_peer used to attach the local _replicated_state to every
    outbound probe, so follower-to-follower probes carried stale copies of
    cluster state around the ring. Only a leader has authoritative state
    to offer."""

    def _capturing_transport(self, sent: list, response: dict):
        def transport(peer_id, host, port, payload, timeout=2.0):
            sent.append(payload)
            return response

        return transport

    def test_follower_probe_carries_no_state(self) -> None:
        sent: list = []
        n = RayFaultTolerance("coordinator-a")
        n.add_peer("coordinator-b", "127.0.0.1", 5002)
        n._state = CoordinatorState.FOLLOWER
        n._replicated_state["stale"] = "v7"
        n.set_heartbeat_transport(
            self._capturing_transport(
                sent,
                {"coordinator_id": "coordinator-b", "term": 0, "state": "follower", "leader_id": None},
            )
        )

        n._probe_peer("coordinator-b", dict(n._peers["coordinator-b"]))

        assert sent and sent[0]["state"] == {}, "follower probe leaked stale state"

    def test_leader_probe_carries_state(self) -> None:
        sent: list = []
        n = RayFaultTolerance("coordinator-a")
        n.add_peer("coordinator-b", "127.0.0.1", 5002)
        n._state = CoordinatorState.LEADER
        n._leader_id = "coordinator-a"
        n._replicated_state["topology"] = "v10"
        n.set_heartbeat_transport(
            self._capturing_transport(
                sent,
                {"coordinator_id": "coordinator-b", "term": 0, "state": "follower", "leader_id": "coordinator-a"},
            )
        )

        n._probe_peer("coordinator-b", dict(n._peers["coordinator-b"]))

        assert sent and sent[0]["state"] == {"topology": "v10"}

    def test_probe_response_teaches_follower_the_leader(self) -> None:
        """A follower must learn a *higher-ID* leader from the probe
        response (the lower-ID rule alone can never establish it), so
        leader-pushed state is honoured afterwards."""
        sent: list = []
        n = RayFaultTolerance("coordinator-a")  # a < b, b is a higher-ID leader
        n.add_peer("coordinator-b", "127.0.0.1", 5002)
        n.set_heartbeat_transport(
            self._capturing_transport(
                sent,
                {"coordinator_id": "coordinator-b", "term": 3, "state": "leader", "leader_id": "coordinator-b"},
            )
        )

        n._probe_peer("coordinator-b", dict(n._peers["coordinator-b"]))

        assert n.get_leader() == "coordinator-b"
        # ...and the leader's state is now accepted.
        n.handle_heartbeat_request("coordinator-b", 3, {"k": "v"})
        assert n.get_replicated_state() == {"k": "v"}
