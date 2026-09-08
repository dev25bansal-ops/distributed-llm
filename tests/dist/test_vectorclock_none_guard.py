"""Regression tests for O-03: VectorClock must not crash on malformed
(None/negative/non-int) timestamps from a buggy or adversarial peer.

Before the fix, ``VectorClock.merge`` called ``max(self.clocks.get(nid, 0),
ts)`` with ``ts=None`` → ``TypeError`` on the gossip anti-entropy path, and
``happens_before`` compared ``None > int`` from a tainted clock the same way.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from distllm.dist.p2p.gossip import VectorClock  # noqa: E402


class TestVectorClockMergeMalformed:
    def test_merge_none_timestamp_skipped(self) -> None:
        vc = VectorClock(clocks={"healthy": 5})
        malformed = VectorClock(clocks={"healthy": None, "bad": None})
        vc.merge(malformed)  # must not raise
        assert vc.clocks == {"healthy": 5}

    def test_merge_positive_timestamp_kept(self) -> None:
        vc = VectorClock(clocks={"a": 1})
        vc.merge(VectorClock(clocks={"a": 9, "b": 3}))
        assert vc.clocks == {"a": 9, "b": 3}

    def test_merge_negative_timestamp_skipped(self) -> None:
        vc = VectorClock(clocks={"a": 4})
        vc.merge(VectorClock(clocks={"a": -1, "b": -7}))
        assert vc.clocks == {"a": 4}

    def test_merge_non_int_timestamp_skipped(self) -> None:
        vc = VectorClock(clocks={"a": 2})
        vc.merge(VectorClock(clocks={"a": "7", "b": 3.14}))
        assert vc.clocks == {"a": 2}


class TestVectorClockHappensBeforeMalformed:
    def test_happens_before_none_left(self) -> None:
        # None coerces to epoch 0: {a: 0} < {a: 5} → happens_before True.
        # The O-03 contract is "no TypeError", not a particular ordering.
        self_val = None
        assert VectorClock(clocks={"a": self_val}).happens_before(
            VectorClock(clocks={"a": 5})
        )

    def test_happens_before_none_right(self) -> None:
        other_val = None
        assert not VectorClock(clocks={"a": 5}).happens_before(
            VectorClock(clocks={"a": other_val})
        )

    def test_happens_before_mixed_malformed(self) -> None:
        # Multiple malformed keys on both sides; must not raise and must
        # fall back to a safe ordering.
        vc = VectorClock(clocks={"a": None, "b": -3, "c": 1})
        other = VectorClock(clocks={"a": 0, "b": 5, "c": None})
        # c: self 1 > 0 → not happens_before
        assert not vc.happens_before(other)


class TestVectorClockHealthy:
    def test_increment_and_merge_roundtrip(self) -> None:
        vc = VectorClock()
        vc.increment("n")
        vc.increment("n")
        assert vc.clocks == {"n": 2}
        vc.merge(VectorClock(clocks={"n": 5, "m": 1}))
        assert vc.clocks == {"n": 5, "m": 1}

    def test_is_concurrent_healthy(self) -> None:
        vc = VectorClock(clocks={"a": 1})
        other = VectorClock(clocks={"b": 1})
        assert vc.is_concurrent(other)