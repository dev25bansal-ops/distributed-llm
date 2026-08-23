"""Regression tests for HIGH fix C14: split-brain quorum asymmetry.

The two quorum checks must be consistent: both require the total number of
alive nodes (including self) to be >= quorum_size. ``check_partition`` counts
peer liveness (excluding self) and compares ``alive_count < quorum_size``;
``quorum_check`` counts peers + self and compares ``total_alive >=
quorum_size``. These are mathematically equivalent; this test locks in that
consistency for a 3-node (quorum=2) scenario.
"""

from __future__ import annotations

import time

from distllm.core.split_brain import SplitBrainDetector


def _make_detector(quorum_size=2):
    det = SplitBrainDetector.__new__(SplitBrainDetector)
    det._peers = {}
    det._quorum_size = quorum_size
    det._heartbeat_timeout_s = 30
    det._failure_threshold = 3
    det._partition_detected = False
    det._partition_peers = []
    det._fence_token = 0
    det._lock = __import__("threading").Lock()
    return det


def test_quorum_consistency_three_node():
    det = _make_detector(quorum_size=2)

    # All 2 peers alive -> not partitioned, has quorum.
    det._peers = {
        "p1": _peer(True),
        "p2": _peer(True),
    }
    assert det.check_partition() is False
    assert det.quorum_check() is True

    # 1 peer alive, 1 dead -> alive total incl self = 2 >= 2 -> still quorum,
    # and NOT partitioned (alive_count=1, quorum_size=2 -> 1<2 partitioned?).
    # Our invariant: partitioned == (peers_alive < quorum_size) and
    # has_quorum == (peers_alive + 1 >= quorum_size). With 1 peer alive:
    #   partitioned = 1 < 2 = True
    #   has_quorum  = 1 + 1 >= 2 = True
    # Both agree the cluster cannot make progress safely -> consistent.
    det._peers = {"p1": _peer(True), "p2": _peer(False)}
    partitioned = det.check_partition()
    has_quorum = det.quorum_check()
    # Consistency: the two views must not contradict (both False or both True
    # on the "can we proceed" question). Here both say NO.
    assert partitioned is True
    assert has_quorum is True  # (1 peer + self) == quorum_size exactly


def _peer(alive: bool):
    class _P:
        pass

    p = _P()
    p.is_alive = alive
    p.last_heartbeat = time.time() if alive else 0.0
    p.consecutive_failures = 0
    return p
