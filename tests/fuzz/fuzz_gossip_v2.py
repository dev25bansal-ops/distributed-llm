"""Seed-heavy fuzz target for ``src/distllm/dist/p2p/gossip.py`` — v2.

This is the *second* gossip fuzzer.  Compared with
``fuzz_gossip_protocol.py`` (which exercises the broad public surface
with budget-50 examples), this file focuses on three specific areas that
are crash-prone and historically under-tested:

1. **Vector clock ordering** — adversarial clocks with mixed types
   (str/int/float/None), cycles, decrements, very large values, NaN/
   infinity, conflicting merges.
2. **Bloom filter edge cases** — capacity-1 / capacity-huge filters,
   filters with more set bits than capacity, all-zeros / all-ones,
   mismatched m/k ratios, negative k, double-add identical items,
   serialization round-trips for tiny filters.
3. **Large message handling** — vector clocks / tombstones / cache
   prefixes / entry_metadata of 10k–100k entries, deeply nested
   structures, payloads near the typical UDP MTU (8 KiB).

The hypothesis example budget is bumped for the targeted seeds so each
specific crash gets enough examples to find it.

Run with::

    python -m pytest tests/fuzz/fuzz_gossip_v2.py -q
"""

from __future__ import annotations

import math
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(HERE, "..", "..", "src"))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import (  # noqa: E402
    given,
    settings,
    strategies as st,
    HealthCheck,
)

# Activate dev/insecure mode so the protocol can be instantiated without
# a shared key.  The fuzzer never performs a real network exchange.
os.environ.setdefault("DISTLLM_DEV_MODE", "1")

from distllm.dist.p2p.gossip import (  # noqa: E402
    GossipProtocol,
    GossipBloomFilter,
    LWWRegister,
    VectorClock,
)


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

# Seed-heavy: bump the budget for the targeted scenarios so any single
# regression has the budget to be discovered.
_FUZZ_SETTINGS = dict(
    max_examples=int(os.environ.get("HYPOTHESIS_EXAMPLES", "120")),
    deadline=None,
    suppress_health_check=[
        HealthCheck.function_scoped_fixture,
        HealthCheck.too_slow,
        HealthCheck.filter_too_much,
    ],
)

GRACEFUL_EXC = (
    TypeError,
    ValueError,
    AttributeError,
    KeyError,
    ArithmeticError,
    IndexError,
    ZeroDivisionError,
    OverflowError,
    RecursionError,
)


def _make_protocol(hmac_key: str = "fuzzer-v2-key") -> GossipProtocol:
    return GossipProtocol(node_id="fuzzer-v2", hmac_key=hmac_key)


# ---------------------------------------------------------------------------
# Strategies for vector clocks
# ---------------------------------------------------------------------------

# Mixed-type values: strings, ints, floats (incl. NaN/inf), None, booleans.
_VC_VALUES = st.one_of(
    st.integers(min_value=-(2**31), max_value=2**31 - 1),
    st.floats(allow_nan=True, allow_infinity=True),
    st.text(max_size=8),
    st.booleans(),
    st.none(),
)


@st.composite
def vector_clock_payload(draw, *, min_size: int = 0, max_size: int = 50) -> dict:
    """Generate an adversarial vector clock mapping."""
    n = draw(st.integers(min_value=min_size, max_value=max_size))
    out: dict = {}
    for _ in range(n):
        # Mix of string / int keys.
        k = draw(st.one_of(
            st.text(min_size=1, max_size=16, alphabet="abcdefghijk"),
            st.integers(min_value=0, max_value=10**6).map(str),
        ))
        v = draw(_VC_VALUES)
        out[k] = v
    return out


# ---------------------------------------------------------------------------
# 1) Vector clock ordering fuzzing
# ---------------------------------------------------------------------------


class TestVectorClockOrdering:
    """VectorClock.happens_before / is_concurrent / merge must be total.

    Properties verified:

    * happens_before is a strict partial order (irreflexive).
    * concurrent is symmetric.
    * merge() takes the per-key max (not the sum) — even when the
      incoming clock is older.
    """

    def test_happens_before_is_irreflexive(self) -> None:
        """A clock never happens-before itself."""
        vc = VectorClock({"a": 1, "b": 2})
        assert vc.happens_before(vc) is False

    def test_increment_advances_clock(self) -> None:
        vc = VectorClock({"a": 1})
        vc.increment("a")
        assert vc.clocks["a"] == 2
        vc.increment("b")
        assert vc.clocks["b"] == 1

    def test_merge_takes_per_key_max(self) -> None:
        a = VectorClock({"x": 5, "y": 1})
        b = VectorClock({"x": 2, "y": 7, "z": 3})
        a.merge(b)
        assert a.clocks == {"x": 5, "y": 7, "z": 3}

    def test_concurrent_symmetric(self) -> None:
        a = VectorClock({"x": 1, "y": 2})
        b = VectorClock({"x": 2, "y": 1})
        assert a.is_concurrent(b) is True
        assert b.is_concurrent(a) is True

    def test_happens_before_is_transitive(self) -> None:
        a = VectorClock({"x": 1})
        b = VectorClock({"x": 2})
        c = VectorClock({"x": 3})
        assert a.happens_before(b) is True
        assert b.happens_before(c) is True
        assert a.happens_before(c) is True

    @given(payload=vector_clock_payload())
    @settings(**_FUZZ_SETTINGS)
    def test_vector_clock_construction_never_crashes(self, payload: dict) -> None:
        """Random payloads must round-trip through the VectorClock ctor."""
        try:
            vc = VectorClock(clocks=payload)
        except GRACEFUL_EXC:
            return
        except Exception as exc:  # pragma: no cover
            pytest.fail(f"VectorClock() raised {type(exc).__name__}: {exc}")

        # happens_before must accept any other VectorClock without raising.
        try:
            other = VectorClock(clocks=dict(payload))
            vc.happens_before(other)
            other.happens_before(vc)
            vc.is_concurrent(other)
            vc.merge(other)
        except GRACEFUL_EXC:
            return
        except Exception as exc:  # pragma: no cover
            pytest.fail(f"VectorClock op raised {type(exc).__name__}: {exc}")

    @given(payload=vector_clock_payload())
    @settings(**_FUZZ_SETTINGS)
    def test_merge_commutative_under_stress(self, payload: dict) -> None:
        """merge(a, b) and merge(b, a) yield the same final clock.

        Note: ``VectorClock.merge`` uses ``max(self.get(nid, 0), ts)``,
        so values that don't compare to ``int`` (e.g. ``None``, strings)
        raise ``TypeError`` — that's a graceful failure mode (the
        comparator rejects the input) and we accept it.
        """
        a = VectorClock(clocks=dict(payload))
        b = VectorClock(clocks=dict(payload))
        # Perturb b slightly.
        b.clocks["zz_perturb"] = 99
        try:
            # a.merge(b):
            a1 = VectorClock(clocks=dict(a.clocks))
            a1.merge(b)
            # b.merge(a):
            b1 = VectorClock(clocks=dict(b.clocks))
            b1.merge(a)
        except GRACEFUL_EXC:
            return
        assert a1.clocks == b1.clocks


# ---------------------------------------------------------------------------
# 2) Bloom filter edge cases
# ---------------------------------------------------------------------------


class TestBloomFilterEdges:
    """Bloom filter under adversarial inputs."""

    def test_capacity_1_filter(self) -> None:
        """capacity=1 yields a tiny filter that still accepts/denies."""
        bf = GossipBloomFilter(capacity=1, error_rate=0.01)
        bf.add("alpha")
        assert bf.contains("alpha") is True
        # The filter has very few bits — a different item may still
        # collide, but that is the documented property of bloom filters
        # with a 1% error rate.  The test only asserts no crash.

    def test_huge_capacity_filter(self) -> None:
        bf = GossipBloomFilter(capacity=100_000, error_rate=0.01)
        for i in range(100):
            bf.add(f"key-{i}")
        for i in range(100):
            assert bf.contains(f"key-{i}") is True

    def test_double_add_same_item_is_idempotent(self) -> None:
        bf = GossipBloomFilter(capacity=16)
        bf.add("dup")
        bits_first = bytes(bf._bitset)
        bf.add("dup")
        bf.add("dup")
        assert bytes(bf._bitset) == bits_first

    def test_round_trip_tiny_filter(self) -> None:
        """A filter that contains just one entry survives a dict round-trip."""
        bf = GossipBloomFilter(capacity=4)
        bf.add("only")
        as_dict = bf.to_dict()
        restored = GossipBloomFilter.from_dict(as_dict)
        assert restored.contains("only") is True

    @given(
        capacity=st.integers(min_value=1, max_value=4096),
        items=st.lists(st.text(min_size=1, max_size=32), min_size=0, max_size=64),
    )
    @settings(**_FUZZ_SETTINGS)
    def test_arbitrary_capacity_and_items(
        self, capacity: int, items: list
    ) -> None:
        bf = GossipBloomFilter(capacity=capacity)
        for it in items:
            try:
                bf.add(it)
            except GRACEFUL_EXC:
                return
        try:
            for it in set(items):
                assert bf.contains(it) is True
        except GRACEFUL_EXC:
            return
        except Exception as exc:  # pragma: no cover
            pytest.fail(f"contains() raised {type(exc).__name__}: {exc}")

    @given(
        capacity=st.integers(min_value=0, max_value=10**5),
        error_rate=st.floats(min_value=1e-6, max_value=0.99, allow_nan=False),
    )
    @settings(**_FUZZ_SETTINGS)
    def test_arbitrary_error_rate(
        self, capacity: int, error_rate: float
    ) -> None:
        try:
            bf = GossipBloomFilter(capacity=capacity, error_rate=error_rate)
        except GRACEFUL_EXC:
            return
        # Sanity: any filter accepts and rejects without crashing.
        try:
            bf.add("x")
            bf.contains("x")
            bf.contains("y")
        except GRACEFUL_EXC:
            return
        except Exception as exc:  # pragma: no cover
            pytest.fail(f"add/contains raised {type(exc).__name__}: {exc}")

    @given(
        m=st.integers(min_value=-100, max_value=4096),
        k=st.integers(min_value=-10, max_value=64),
        bits_hex=st.text(
            alphabet="0123456789abcdef", min_size=0, max_size=256
        ),
    )
    @settings(**_FUZZ_SETTINGS)
    def test_from_dict_adversarial(
        self, m: int, k: int, bits_hex: str
    ) -> None:
        """A from_dict() with adversarial (m, k, bits) must not crash."""
        try:
            obj = GossipBloomFilter.from_dict({
                "m": m,
                "k": k,
                "bits": bits_hex,
            })
        except GRACEFUL_EXC:
            return
        except Exception as exc:  # pragma: no cover
            pytest.fail(f"from_dict raised {type(exc).__name__}: {exc}")

        try:
            obj.contains("anything")
        except GRACEFUL_EXC:
            return
        except Exception as exc:  # pragma: no cover
            pytest.fail(f"contains() raised {type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# 3) Large message handling
# ---------------------------------------------------------------------------


class TestLargeMessageHandling:
    """Process adversarial messages near MTU-scale sizes."""

    @given(
        n_entries=st.integers(min_value=10_000, max_value=50_000),
    )
    @settings(max_examples=10, deadline=None)
    def test_vector_clock_with_50k_entries(self, n_entries: int) -> None:
        """A vector clock of 50k entries must not OOM or deadlock."""
        proto = _make_protocol()
        # Build a valid, well-formed advertisement so the only variable
        # is the vector_clock size.
        peer = GossipProtocol(node_id="big-peer", hmac_key=proto._hmac_key)
        peer.store_local("p1", "r1")
        ad = peer.sign_message(peer.advertise(delta_only=False))
        ad["vector_clock"] = {f"node-{i}": i for i in range(n_entries)}
        try:
            result = proto.process_advertisement(ad)
        except GRACEFUL_EXC:
            return
        except Exception as exc:  # pragma: no cover
            pytest.fail(
                f"process_advertisement raised {type(exc).__name__}: {exc}"
            )
        assert isinstance(result, list)
        # Lock must still be acquirable.
        assert proto._lock.acquire(timeout=5.0)
        proto._lock.release()

    @given(
        n_tombstones=st.integers(min_value=10_000, max_value=50_000),
    )
    @settings(max_examples=10, deadline=None)
    def test_tombstones_with_50k_entries(self, n_tombstones: int) -> None:
        proto = _make_protocol()
        peer = GossipProtocol(node_id="tomb-peer", hmac_key=proto._hmac_key)
        peer.store_local("p1", "r1")
        ad = peer.sign_message(peer.advertise(delta_only=False))
        ad["tombstones"] = {f"prefix-{i}": float(i) for i in range(n_tombstones)}
        try:
            result = proto.process_advertisement(ad)
        except GRACEFUL_EXC:
            return
        except Exception as exc:  # pragma: no cover
            pytest.fail(
                f"process_advertisement raised {type(exc).__name__}: {exc}"
            )
        assert isinstance(result, list)

    @given(
        n_prefixes=st.integers(min_value=10_000, max_value=50_000),
    )
    @settings(max_examples=10, deadline=None)
    def test_cache_prefixes_with_50k_entries(self, n_prefixes: int) -> None:
        proto = _make_protocol()
        peer = GossipProtocol(node_id="pref-peer", hmac_key=proto._hmac_key)
        peer.store_local("p1", "r1")
        ad = peer.sign_message(peer.advertise(delta_only=False))
        ad["cache_prefixes"] = [f"prefix-{i}" for i in range(n_prefixes)]
        try:
            result = proto.process_advertisement(ad)
        except GRACEFUL_EXC:
            return
        except Exception as exc:  # pragma: no cover
            pytest.fail(
                f"process_advertisement raised {type(exc).__name__}: {exc}"
            )
        assert isinstance(result, list)

    @given(
        n_meta=st.integers(min_value=5_000, max_value=20_000),
    )
    @settings(max_examples=10, deadline=None)
    def test_entry_metadata_with_20k_entries(self, n_meta: int) -> None:
        proto = _make_protocol()
        peer = GossipProtocol(node_id="meta-peer", hmac_key=proto._hmac_key)
        peer.store_local("p1", "r1")
        ad = peer.sign_message(peer.advertise(delta_only=False))
        ad["entry_metadata"] = {
            f"prefix-{i}": {
                "value": f"v{i}",
                "timestamp": float(i),
                "writer_id": f"w{i}",
            }
            for i in range(n_meta)
        }
        try:
            result = proto.process_advertisement(ad)
        except GRACEFUL_EXC:
            return
        except Exception as exc:  # pragma: no cover
            pytest.fail(
                f"process_advertisement raised {type(exc).__name__}: {exc}"
            )
        assert isinstance(result, list)

    def test_deeply_nested_metadata_value(self) -> None:
        """A 256-deep nested list inside entry_metadata.value must not crash."""
        proto = _make_protocol()
        peer = GossipProtocol(node_id="deep-peer", hmac_key=proto._hmac_key)
        peer.store_local("p1", "r1")
        ad = peer.sign_message(peer.advertise(delta_only=False))
        nested: list = []
        cur = nested
        for _ in range(256):
            nxt: list = []
            cur.append(nxt)
            cur = nxt
        ad["entry_metadata"] = {
            "k": {"value": nested, "timestamp": 1.0, "writer_id": "deep"},
        }
        try:
            result = proto.process_advertisement(ad)
        except GRACEFUL_EXC:
            return
        assert isinstance(result, list)

    def test_mtu_scale_payload(self) -> None:
        """A payload of ~8 KiB (typical UDP MTU) must process cleanly."""
        proto = _make_protocol()
        peer = GossipProtocol(node_id="mtu-peer", hmac_key=proto._hmac_key)
        peer.store_local("p1", "r1")
        ad = peer.sign_message(peer.advertise(delta_only=False))
        # Build ~8 KiB of metadata.
        chunk = []
        size = 0
        i = 0
        while size < 8192:
            entry = {
                "value": f"v{i}",
                "timestamp": float(i),
                "writer_id": f"w{i}",
            }
            chunk.append((f"prefix-{i}", entry))
            size += len(f"prefix-{i}") + len(entry["value"]) + 50
            i += 1
        ad["entry_metadata"] = dict(chunk)
        try:
            result = proto.process_advertisement(ad)
        except GRACEFUL_EXC:
            return
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# 4) LWW register edge cases
# ---------------------------------------------------------------------------


class TestLWWRegister:
    """LWWRegister.merge semantics: latest timestamp wins; ties broken by writer_id."""

    def test_strictly_newer_timestamp_wins(self) -> None:
        a = LWWRegister(value="old", timestamp=1.0, writer_id="w1")
        b = LWWRegister(value="new", timestamp=2.0, writer_id="w2")
        a.merge(b)
        assert a.value == "new"
        assert a.timestamp == 2.0

    def test_equal_timestamp_breaks_by_writer_id(self) -> None:
        a = LWWRegister(value="a", timestamp=5.0, writer_id="alpha")
        b = LWWRegister(value="b", timestamp=5.0, writer_id="beta")
        a.merge(b)
        # Tie-break: writer_id > self.writer_id wins. "beta" > "alpha"
        # alphabetically, so b should win.
        assert a.value == "b"

    def test_strictly_older_timestamp_loses(self) -> None:
        a = LWWRegister(value="newer", timestamp=10.0, writer_id="x")
        b = LWWRegister(value="older", timestamp=1.0, writer_id="y")
        a.merge(b)
        assert a.value == "newer"

    @given(
        ts_a=st.floats(allow_nan=True, allow_infinity=True),
        ts_b=st.floats(allow_nan=True, allow_infinity=True),
        writer_a=st.text(min_size=1, max_size=8),
        writer_b=st.text(min_size=1, max_size=8),
    )
    @settings(**_FUZZ_SETTINGS)
    def test_merge_never_crashes(
        self,
        ts_a: float,
        ts_b: float,
        writer_a: str,
        writer_b: str,
    ) -> None:
        a = LWWRegister(value="a", timestamp=ts_a, writer_id=writer_a)
        b = LWWRegister(value="b", timestamp=ts_b, writer_id=writer_b)
        try:
            a.merge(b)
        except GRACEFUL_EXC:
            return
        except Exception as exc:  # pragma: no cover
            pytest.fail(f"LWWRegister.merge raised {type(exc).__name__}: {exc}")
        # Result must be one of the two values.
        assert a.value in ("a", "b")


if __name__ == "__main__":
    pytest.main([__file__, "-q"])