"""Tests for split-brain detection in federated clusters.

Verifies that the SplitBrainDetector correctly identifies network
partitions and manages fencing tokens.
"""

import time

import pytest

from distllm.core.split_brain import SplitBrainDetector, PeerState


class TestSplitBrainDetection:
    """Test split-brain detection scenarios."""

    def test_no_partition_when_all_peers_alive(self):
        """No partition when all peers send heartbeats."""
        detector = SplitBrainDetector(
            cluster_id="cluster-a",
            peer_cluster_ids=["cluster-b", "cluster-c"],
            quorum_size=2,
        )

        detector.heartbeat("cluster-b")
        detector.heartbeat("cluster-c")

        assert not detector.is_partitioned()
        assert len(detector.get_alive_peers()) == 2

    def test_partition_detected_when_quorum_lost(self):
        """Partition detected when fewer than quorum peers are alive."""
        detector = SplitBrainDetector(
            cluster_id="cluster-a",
            peer_cluster_ids=["cluster-b", "cluster-c"],
            quorum_size=2,
            failure_threshold=2,
        )

        # Only one peer alive — below quorum
        detector.heartbeat("cluster-b")
        for _ in range(3):
            detector.record_failure("cluster-c")

        is_partitioned = detector.check_partition()
        assert is_partitioned
        assert "cluster-c" in detector.get_partitioned_peers()

    def test_heartbeat_timeout_triggers_partition(self):
        """Peers that stop sending heartbeats are detected."""
        detector = SplitBrainDetector(
            cluster_id="cluster-a",
            peer_cluster_ids=["cluster-b"],
            quorum_size=1,
            heartbeat_timeout_s=0.1,
            failure_threshold=1,
        )

        detector.heartbeat("cluster-b")
        time.sleep(0.15)

        is_partitioned = detector.check_partition()
        assert is_partitioned

    def test_fence_token_increment_on_partition(self):
        """Fence token increments when partition is detected."""
        detector = SplitBrainDetector(
            cluster_id="cluster-a",
            peer_cluster_ids=["cluster-b"],
            quorum_size=1,
            failure_threshold=1,
        )

        initial_token = detector.get_fence_token()

        # Trigger partition
        for _ in range(2):
            detector.record_failure("cluster-b")
        detector.check_partition()
        detector.increment_fence_token()

        assert detector.get_fence_token() > initial_token

    def test_fence_token_rejects_stale_requests(self):
        """Requests with old fence tokens are rejected."""
        detector = SplitBrainDetector(
            cluster_id="cluster-a",
            peer_cluster_ids=["cluster-b"],
            quorum_size=1,
        )

        # Increment fence token
        detector.increment_fence_token()
        detector.increment_fence_token()
        current = detector.get_fence_token()

        # Old token rejected
        assert not detector.should_accept_request(current - 1)
        # Current token accepted
        assert detector.should_accept_request(current)
        # Newer token accepted
        assert detector.should_accept_request(current + 1)

    def test_recovery_after_partition_heals(self):
        """Detector recovers when peers start heartbeating again."""
        detector = SplitBrainDetector(
            cluster_id="cluster-a",
            peer_cluster_ids=["cluster-b"],
            quorum_size=1,
            failure_threshold=2,
        )

        # Create partition
        for _ in range(3):
            detector.record_failure("cluster-b")
        detector.check_partition()
        assert detector.is_partitioned()

        # Heal
        detector.heartbeat("cluster-b")
        detector.check_partition()
        assert not detector.is_partitioned()
        assert "cluster-b" in detector.get_alive_peers()

    def test_stats_reporting(self):
        """Stats include all relevant information."""
        detector = SplitBrainDetector(
            cluster_id="cluster-a",
            peer_cluster_ids=["cluster-b", "cluster-c"],
            quorum_size=2,
            failure_threshold=1,
        )

        # Fail cluster-c so only cluster-b is alive
        detector.record_failure("cluster-c")
        detector.record_failure("cluster-c")
        detector.check_partition()

        detector.heartbeat("cluster-b")
        stats = detector.stats()

        assert stats["cluster_id"] == "cluster-a"
        assert stats["peers"] == 2
        assert stats["alive"] >= 1
        assert "fence_token" in stats

    def test_multiple_partitions_detected(self):
        """Multiple successive partitions are detected."""
        detector = SplitBrainDetector(
            cluster_id="cluster-a",
            peer_cluster_ids=["cluster-b", "cluster-c", "cluster-d"],
            quorum_size=2,
            failure_threshold=2,
        )

        # First partition: lose cluster-c
        for _ in range(3):
            detector.record_failure("cluster-c")
        detector.check_partition()

        # Second partition: lose cluster-d
        for _ in range(3):
            detector.record_failure("cluster-d")
        detector.check_partition()

        assert len(detector.get_partitioned_peers()) == 2
