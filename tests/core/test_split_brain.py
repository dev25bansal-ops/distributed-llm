"""Tests for split-brain detection -- SplitBrainDetector.

Covers:
- Construction with peer cluster IDs
- Heartbeat recording and failure counting
- Partition detection (check_partition / is_partitioned)
- Alive/partitioned peer queries
- Fencing tokens (increment, should_accept_request, durability/restart)
- Combined quorum + fence (fence_request)
- quorum_check
- stats()
"""

from __future__ import annotations

import time

import pytest

from tests._import_helper import bootstrap_fake_packages, load_module

bootstrap_fake_packages()

_sb = load_module("distllm/core/split_brain.py")
SplitBrainDetector = _sb.SplitBrainDetector
PeerState = _sb.PeerState


@pytest.fixture(autouse=True)
def _isolated_fence_state(tmp_path, monkeypatch):
    """Keep the durable fence-token files inside the test.

    SplitBrainDetector persists its fencing token per cluster id (by design:
    a restart must not reset the counter). Without isolation, token state
    would leak between tests that reuse cluster ids like ``c1`` and between
    separate pytest runs.
    """
    monkeypatch.setenv("DISTLLM_FENCE_STATE_DIR", str(tmp_path))
    yield


# ======================================================================
# PeerState
# ======================================================================


class TestPeerState:
    def test_defaults(self):
        ps = PeerState(cluster_id="us-east")
        assert ps.cluster_id == "us-east"
        assert ps.last_heartbeat == 0.0
        assert ps.consecutive_failures == 0
        assert ps.is_alive is True
        assert ps.fence_token == 0

    def test_custom_values(self):
        ps = PeerState(cluster_id="eu", last_heartbeat=100.0, consecutive_failures=2, is_alive=False)
        assert ps.cluster_id == "eu"
        assert ps.last_heartbeat == 100.0
        assert ps.consecutive_failures == 2
        assert ps.is_alive is False


# ======================================================================
# SplitBrainDetector construction
# ======================================================================


class TestSplitBrainDetectorInit:
    def test_defaults(self):
        d = SplitBrainDetector(cluster_id="us-east")
        assert d._cluster_id == "us-east"
        assert d._peers == {}
        assert d._quorum_size == 2
        assert d._heartbeat_timeout_s == 30.0
        assert d._failure_threshold == 3

    def test_with_peers(self):
        d = SplitBrainDetector(
            cluster_id="us-east",
            peer_cluster_ids=["us-west", "eu-central"],
            quorum_size=2,
        )
        assert "us-west" in d._peers
        assert "eu-central" in d._peers
        # A configured peer is not alive until its first heartbeat.
        assert d._peers["us-west"].is_alive is False
        assert d._peers["us-west"].last_heartbeat == 0.0


# ======================================================================
# Heartbeat and failure recording
# ======================================================================


class TestHeartbeat:
    def test_heartbeat_records_timestamp(self):
        d = SplitBrainDetector(cluster_id="c1", peer_cluster_ids=["p1"])
        ts = 1000.0
        d.heartbeat("p1", timestamp=ts)
        assert d._peers["p1"].last_heartbeat == ts

    def test_heartbeat_unknown_peer_adds_it(self):
        d = SplitBrainDetector(cluster_id="c1")
        d.heartbeat("unknown-peer", timestamp=100.0)
        assert "unknown-peer" in d._peers

    def test_heartbeat_resets_consecutive_failures(self):
        d = SplitBrainDetector(cluster_id="c1", peer_cluster_ids=["p1"])
        d.record_failure("p1")
        d.record_failure("p1")
        assert d._peers["p1"].consecutive_failures == 2
        d.heartbeat("p1", timestamp=200.0)
        assert d._peers["p1"].consecutive_failures == 0

    def test_record_failure_increments(self):
        d = SplitBrainDetector(cluster_id="c1", peer_cluster_ids=["p1"])
        d.record_failure("p1")
        assert d._peers["p1"].consecutive_failures == 1

    def test_record_failure_marks_dead(self):
        d = SplitBrainDetector(
            cluster_id="c1", peer_cluster_ids=["p1"], failure_threshold=2
        )
        d.record_failure("p1")
        d.record_failure("p1")
        assert d._peers["p1"].is_alive is False

    def test_record_failure_unknown_peer_adds_it(self):
        d = SplitBrainDetector(cluster_id="c1")
        d.record_failure("new-peer")
        assert "new-peer" in d._peers


# ======================================================================
# Partition detection
# ======================================================================


class TestPartitionDetection:
    def test_no_partition_all_alive(self):
        d = SplitBrainDetector(
            cluster_id="c1",
            peer_cluster_ids=["p1", "p2"],
            quorum_size=2,
        )
        d.heartbeat("p1", timestamp=time.time())
        d.heartbeat("p2", timestamp=time.time())
        assert d.check_partition() is False
        assert d.is_partitioned() is False

    def test_partition_when_quorum_not_met(self):
        d = SplitBrainDetector(
            cluster_id="c1",
            peer_cluster_ids=["p1", "p2"],
            quorum_size=2,
        )
        # Both peers are dead (no heartbeats at all -> never alive)
        # With 0 alive peers + self = 1 total, quorum_size=2 not met
        assert d.check_partition() is True
        assert d.is_partitioned() is True

    def test_partition_from_heartbeat_timeout(self):
        """A timed-out heartbeat marks the peer dead; with the quorum size
        no longer reachable without it, a partition is declared.

        Quorum counts include this cluster itself (alive_peers + 1), so
        quorum_size must be 2 here: self alone cannot hold quorum, and
        the timed-out peer must not vouch for it.
        """
        d = SplitBrainDetector(
            cluster_id="c1",
            peer_cluster_ids=["p1"],
            quorum_size=2,
            heartbeat_timeout_s=0.01,
            failure_threshold=1,
        )
        # Set old heartbeat
        d.heartbeat("p1", timestamp=time.time() - 60.0)
        # Timeout must mark the peer dead and the quorum lost
        assert d.check_partition() is True
        assert d.get_alive_peers() == []
        assert d.get_partitioned_peers() == ["p1"]

    def test_no_partition_when_self_holds_quorum(self):
        """With quorum_size=1 the cluster itself holds quorum, so a dead
        peer alone must NOT declare a partition (self-counting quorum)."""
        d = SplitBrainDetector(
            cluster_id="c1",
            peer_cluster_ids=["p1"],
            quorum_size=1,
            heartbeat_timeout_s=0.01,
            failure_threshold=1,
        )
        d.heartbeat("p1", timestamp=time.time() - 60.0)
        assert d.check_partition() is False
        # The peer is still marked dead even though quorum is held
        assert d.get_alive_peers() == []
        assert d.get_partitioned_peers() == ["p1"]

    def test_no_partition_with_enough_alive(self):
        d = SplitBrainDetector(
            cluster_id="c1",
            peer_cluster_ids=["p1", "p2", "p3"],
            quorum_size=3,
        )
        d.heartbeat("p1", timestamp=time.time())
        d.heartbeat("p2", timestamp=time.time())
        d.heartbeat("p3", timestamp=time.time())
        assert d.check_partition() is False


# ======================================================================
# Queries: alive / partitioned peers
# ======================================================================


class TestPeerQueries:
    def test_get_alive_peers(self):
        d = SplitBrainDetector(cluster_id="c1", peer_cluster_ids=["p1", "p2"])
        d.heartbeat("p1", timestamp=time.time())
        d.heartbeat("p2", timestamp=time.time())
        d.record_failure("p2")
        d.record_failure("p2")
        d.record_failure("p2")  # dead
        alive = d.get_alive_peers()
        assert "p1" in alive
        assert "p2" not in alive

    def test_get_partitioned_peers_empty_initially(self):
        d = SplitBrainDetector(cluster_id="c1", peer_cluster_ids=["p1"])
        assert d.get_partitioned_peers() == []

    def test_get_partitioned_peers_after_check(self):
        d = SplitBrainDetector(
            cluster_id="c1",
            peer_cluster_ids=["p1"],
            quorum_size=2,
            heartbeat_timeout_s=0.01,
            failure_threshold=1,
        )
        d.heartbeat("p1", timestamp=time.time() - 60.0)
        d.check_partition()
        parts = d.get_partitioned_peers()
        assert "p1" in parts


# ======================================================================
# Fencing tokens
# ======================================================================


class TestFencingTokens:
    def test_initial_token_zero(self):
        d = SplitBrainDetector(cluster_id="c1")
        assert d.get_fence_token() == 0

    def test_increment_token(self):
        d = SplitBrainDetector(cluster_id="c1")
        assert d.increment_fence_token() == 1
        assert d.get_fence_token() == 1

    def test_should_accept_request_equal_token(self):
        d = SplitBrainDetector(cluster_id="c1")
        d.increment_fence_token()
        assert d.should_accept_request(1) is True

    def test_should_accept_request_higher_token(self):
        d = SplitBrainDetector(cluster_id="c1")
        d.increment_fence_token()
        assert d.should_accept_request(2) is True

    def test_should_accept_request_stale_token(self):
        d = SplitBrainDetector(cluster_id="c1")
        d.increment_fence_token()
        assert d.should_accept_request(0) is False

    def test_should_accept_request_initial(self):
        d = SplitBrainDetector(cluster_id="c1")
        assert d.should_accept_request(0) is True  # 0 >= 0


# ======================================================================
# Fence-token durability and cross-node comparability
# ======================================================================


class TestFenceTokenDurability:
    """Regression: the token counter used to be a bare in-process int.

    A restart reset it to 0, so a fenced-out leader's old token (which the
    old `>=` check admitted at the reset value) became valid again on every
    rolling deploy; and no node ever exchanged its token, so a token issued
    at one node could not be ordered against another node's counter.
    """

    def test_stale_token_not_accepted_after_restart(self):
        """The audit's exact check: a token superseded before the restart
        must not be re-admitted by the restarted detector."""
        d = SplitBrainDetector("a", ["b"], quorum_size=1)
        d.increment_fence_token()
        stale = d.get_fence_token()

        d2 = SplitBrainDetector("a", ["b"], quorum_size=1)  # same cluster=state file

        assert d2.should_accept_request(stale) is False, (
            "a restart re-admitted a fenced-out leader's stale token"
        )

    def test_token_survives_restart(self):
        d = SplitBrainDetector("c9", quorum_size=1)
        d.increment_fence_token()
        d.increment_fence_token()
        assert d.get_fence_token() == 2

        d2 = SplitBrainDetector("c9", quorum_size=1)

        assert d2.get_fence_token() == 2

    def test_new_token_after_restart_is_accepted(self):
        """A restart must not brick admission: once the authority issues a
        new token, that token is valid at the restarted node."""
        d = SplitBrainDetector("c9b", quorum_size=1)
        d.increment_fence_token()

        d2 = SplitBrainDetector("c9b", quorum_size=1)
        fresh = d2.increment_fence_token()

        assert d2.should_accept_request(fresh) is True
        assert d2.should_accept_request(fresh - 1) is False

    def test_observe_merges_token_across_nodes(self):
        """Token at node X must order above every token issued earlier at
        node Y once they exchange it."""
        y = SplitBrainDetector("y", quorum_size=1)
        y.increment_fence_token()  # y's token is 1

        # Node x, a different cluster state, observed token 3 from a peer.
        x = SplitBrainDetector("x", quorum_size=1, fence_state_path="")
        x.observe_fence_token(3)

        assert x.get_fence_token() == 3
        assert y.should_accept_request(3) is True  # 3 > 1: newer wins
        # y's old token 1 is now stale at x.
        assert x.should_accept_request(1) is False

    def test_heartbeat_piggybacks_token(self):
        d = SplitBrainDetector("hb", ["p1"], quorum_size=1)
        d.heartbeat("p1", fence_token=5)
        assert d.get_fence_token() == 5
        assert d._peers["p1"].fence_token == 5

    def test_persistence_can_be_disabled(self):
        """fence_state_path="" keeps the old pure in-process behaviour for
        callers that manage durability themselves."""
        d = SplitBrainDetector("nofile", quorum_size=1, fence_state_path="")
        d.increment_fence_token()
        assert d.get_fence_token() == 1
        d2 = SplitBrainDetector("nofile", quorum_size=1, fence_state_path="")
        assert d2.get_fence_token() == 0

    def test_corrupt_state_file_tolerated(self, tmp_path):
        path = tmp_path / "corrupt.json"
        path.write_text("{not json")
        d = SplitBrainDetector("corrupt", quorum_size=1, fence_state_path=path)
        assert d.get_fence_token() == 0
        assert d.increment_fence_token() == 1


# ======================================================================
# quorum_check
# ======================================================================


class TestQuorumCheck:
    def test_quorum_with_self_only(self):
        d = SplitBrainDetector(cluster_id="c1", quorum_size=1)
        assert d.quorum_check() is True  # self counts

    def test_quorum_fails(self):
        d = SplitBrainDetector(
            cluster_id="c1",
            peer_cluster_ids=["p1", "p2"],
            quorum_size=3,
        )
        # 0 alive peers + self = 1 < 3
        assert d.quorum_check() is False

    def test_quorum_ok_with_alive_peers(self):
        d = SplitBrainDetector(
            cluster_id="c1",
            peer_cluster_ids=["p1", "p2"],
            quorum_size=2,
        )
        d.heartbeat("p1", timestamp=time.time())
        d.heartbeat("p2", timestamp=time.time())
        assert d.quorum_check() is True  # 2 alive + self = 3 >= 2


# ======================================================================
# fence_request (combined quorum + fence)
# ======================================================================


class TestFenceRequest:
    def test_accept_good_token_with_quorum(self):
        d = SplitBrainDetector(
            cluster_id="c1",
            peer_cluster_ids=["p1", "p2"],
            quorum_size=2,
            heartbeat_timeout_s=60.0,
        )
        d.heartbeat("p1", time.time())
        d.heartbeat("p2", time.time())
        ok, reason = d.fence_request(0)
        assert ok is True
        assert reason == "ok"

    def test_reject_no_quorum(self):
        d = SplitBrainDetector(
            cluster_id="c1",
            peer_cluster_ids=["p1"],
            quorum_size=2,  # self + 1 alive peer needed
        )
        # p1 is dead (no heartbeat)
        ok, reason = d.fence_request(0)
        assert ok is False
        assert "no_quorum" in reason

    def test_reject_stale_token(self):
        d = SplitBrainDetector(
            cluster_id="c1",
            peer_cluster_ids=["p1"],
            quorum_size=1,
        )
        d.heartbeat("p1", time.time())
        d.increment_fence_token()
        ok, reason = d.fence_request(0)
        assert ok is False
        assert "stale_fence_token" in reason


# ======================================================================
# stats()
# ======================================================================


class TestStats:
    def test_stats_structure(self):
        d = SplitBrainDetector(
            cluster_id="c1",
            peer_cluster_ids=["p1", "p2"],
            quorum_size=2,
        )
        s = d.stats()
        assert s["cluster_id"] == "c1"
        assert s["peers"] == 2
        assert "alive" in s
        assert "partitioned" in s
        assert "fence_token" in s
        assert s["quorum_size"] == 2
