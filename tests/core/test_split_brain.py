"""Tests for split-brain detection -- SplitBrainDetector.

Covers:
- Construction with peer cluster IDs
- Heartbeat recording and failure counting
- Partition detection (check_partition / is_partitioned)
- Alive/partitioned peer queries
- Fencing tokens (increment, should_accept_request)
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
        d = SplitBrainDetector(
            cluster_id="c1",
            peer_cluster_ids=["p1"],
            quorum_size=1,
            heartbeat_timeout_s=0.01,
            failure_threshold=1,
        )
        # Set old heartbeat
        d.heartbeat("p1", timestamp=time.time() - 60.0)
        # Should detect timeout and mark dead
        assert d.check_partition() is True

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
