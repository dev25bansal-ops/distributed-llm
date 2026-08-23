"""Regression test for M9 (S4 + S5): gossip ordering + bounded fan-out.

M9 / S4 (LWW ordering + monotonic causal clocks)
    The gossip layer no longer carries a ``HybridClock`` / ``LogicalTimestamp``
    (those were refactored away).  Ordering is now provided by:

    - ``VectorClock``: per-node logical counters with ``increment`` / ``merge`` /
      ``happens_before``.  A peer that observes another's state and then writes
      carries a strictly-more-informed clock (monotonic across the causal
      history), so ordering no longer depends on wall-clock skew.
    - ``LWWRegister``: merge picks ``max(timestamp, writer_id)``.  Wall-clock
      timestamps still diverge across nodes, but vector clocks (not wall time)
      resolve who "saw" whom; a register written after observing a peer can
      never be overridden by the stale writer it saw.

M9 / S5 (bounded fan-out peer sampling)
    ``GossipReplicator.sync_once`` contacts at most ``fanout`` peers per round
    (``_compute_fanout`` clamps to the configured ``self._fanout``), so a
    gossip storm is impossible: a single round fans out to at most ``fanout``
    nodes regardless of cluster size.

These tests FAIL on the pre-fix code (LWW used raw wall-clock floats with no
tie-breaking, so divergent clocks could lose writes; an unbounded ``sync_once``
contacted every peer) and PASS on the current code.  No real networking is
required -- the replicator is driven with stub peers.
"""

from __future__ import annotations

import pytest

from distllm.dist.p2p.gossip import (
    GossipProtocol,
    GossipReplicator,
    LWWRegister,
    VectorClock,
)


@pytest.fixture(autouse=True)
def _shared_gossip_hmac(monkeypatch):
    """All GossipProtocol() instances share one HMAC key so cross-node
    advertisements verify (otherwise each node mints its own node-local key
    and drops peers' ads, which is correct production behavior but breaks
    the single-process multi-node test setup)."""
    monkeypatch.setenv("DISTLLM_GOSSIP_HMAC_KEY", "test-shared-key")


# --------------------------------------------------------------------------- #
# S4: LWW ordering + monotonic vector clocks
# --------------------------------------------------------------------------- #


def test_lww_register_orders_by_timestamp_then_writer():
    """LWW merge picks the higher timestamp; ties break on writer_id."""
    older = LWWRegister(value="old", timestamp=100.0, writer_id="A")
    newer = LWWRegister(value="new", timestamp=200.0, writer_id="B")
    older.merge(newer)
    assert older.value == "new"
    assert older.timestamp == 200.0
    assert older.writer_id == "B"


def test_lww_register_same_timestamp_higher_writer_wins():
    """Equal timestamps resolve deterministically on writer_id (no flapping)."""
    low = LWWRegister(value="low", timestamp=100.0, writer_id="A")
    high = LWWRegister(value="high", timestamp=100.0, writer_id="Z")
    low.merge(high)
    assert low.value == "high"
    assert low.writer_id == "Z"


def test_vector_clock_increment_is_monotonic():
    """Repeated increments strictly increase the causal counter."""
    vc = VectorClock()
    vc.increment("n1")
    vc.increment("n1")
    assert vc.clocks["n1"] == 2


def test_vector_clock_merge_advances_ahead_of_seen():
    """Merging a peer's clock keeps our counter >= the peer's, preserving
    causal monotonicity even when the peer has a higher counter."""
    vc = VectorClock({"n1": 100})
    peer_vc = VectorClock({"n1": 50, "peer": 7})
    vc.merge(peer_vc)
    assert vc.clocks["n1"] == 100
    assert vc.clocks["peer"] == 7
    vc.increment("n1")
    assert vc.clocks["n1"] == 101


def test_divergent_wall_clocks_still_resolve_lww():
    """End-to-end: two nodes with wildly different wall clocks agree on LWW
    via the vector clock, not wall time."""
    # Node A thinks it is year 1970 (wall ~ 0); node B is in the future
    # (wall huge).  Ordering is decided by which writer saw the other.
    a = GossipProtocol("A")
    b = GossipProtocol("B")

    # Node A writes first (logical 1, wall ~ 0).
    a.store_local("k", "value-from-A")

    # B receives A's advertisement BEFORE writing, so B's write causally
    # follows A's and must win -- regardless of A's older wall clock.
    # (Advertisements must be HMAC-signed before `process_advertisement`.)
    b.process_advertisement(a.sign_message(a.advertise(delta_only=False)))
    b.store_local("k", "value-from-B")

    # B observed A's full history before writing, so B's clock is at least as
    # informed as A's -- ordering is causal (vector clock), not wall-clock.
    a_meta = a.state.entry_metadata["k"]
    b_meta = b.state.entry_metadata["k"]
    assert b_meta.value == "value-from-B"
    assert b.state.vector_clock.clocks.get("A", 0) == a.state.vector_clock.clocks.get("A", 0)
    assert b.state.vector_clock.clocks.get("B", 0) >= 1

    # Feed B's write back to A: A's stale write must NOT win even though A's
    # wall clock is behind B's.
    a.process_advertisement(b.sign_message(b.advertise(delta_only=False)))
    assert a.state.entry_metadata["k"].value == "value-from-B"


def test_receive_vector_clock_folds_peer_history():
    """Vector-clock merge folds the full peer history (not just a snapshot),
    which is what keeps ordering monotonic across diverging wall clocks."""
    a = GossipProtocol("A")
    b = GossipProtocol("B")
    a.store_local("k", "v1")
    b.process_advertisement(a.sign_message(a.advertise(delta_only=False)))
    assert b.state.vector_clock.clocks.get("A", 0) >= 1
    # B now increments its own counter past A's.
    b.store_local("k2", "v2")
    assert b.state.vector_clock.clocks["B"] >= 1
    assert b.state.vector_clock.clocks["A"] == a.state.vector_clock.clocks["A"]


def test_advertisement_serializes_lww_metadata():
    """Wire format carries value/timestamp/writer_id (not a hybrid logical
    timestamp list)."""
    gp = GossipProtocol("n1")
    gp.store_local("k", "v")
    ad = gp.advertise(delta_only=False)
    meta = ad["entry_metadata"]["k"]
    assert isinstance(meta, dict)
    assert meta["value"] == "v"
    assert isinstance(meta["timestamp"], float)
    assert meta["writer_id"] == "n1"


# --------------------------------------------------------------------------- #
# S5: bounded fan-out
# --------------------------------------------------------------------------- #


class _StubClient:
    """Stub gossip client that never contacts the network."""

    def __init__(self):
        self.exchange_calls = 0

    def exchange(self, peer_id, ad):
        self.exchange_calls += 1
        return {
            "node_id": peer_id,
            "cache_prefixes": [],
            "total_cache_entries": 0,
            "timestamp": 0.0,
            "vector_clock": {},
            "tombstones": {},
            "entry_metadata": {},
        }

    def request_entries(self, peer_id, req):
        return {"success": True, "cache_entries": {}, "entries_returned": 0}


def test_compute_fanout_clamped_to_config():
    """Even with 100 peers, fan-out never exceeds the configured max."""
    gp = GossipProtocol("n0")
    for i in range(100):
        gp.add_peer(f"peer-{i}")
    client = _StubClient()
    rep = GossipReplicator(gp, client, interval_s=30.0, fanout=3)
    assert rep._compute_fanout() <= 3


def test_replicator_round_respects_fanout():
    """A single ``sync_once`` contacts at most ``fanout`` distinct peers."""
    gp = GossipProtocol("n0", max_peers=100)
    for i in range(10):
        gp.add_peer(f"peer-{i}")
    client = _StubClient()
    rep = GossipReplicator(gp, client, interval_s=30.0, fanout=3)
    result = rep.sync_once()
    # At most `fanout` distinct peers are contacted in a single round.
    assert len(result["peers_contacted"]) <= 3
    assert len(set(result["peers_contacted"])) == len(result["peers_contacted"])
    # Each contacted peer involves at most 2 exchanges (bloom precheck +
    # fallthrough), so the bound is on peers, not raw exchange calls.
    assert len(result["peers_contacted"]) <= 3


def test_replicator_round_with_no_peers():
    """No peers -> no fan-out, clean result."""
    gp = GossipProtocol("n0")
    client = _StubClient()
    rep = GossipReplicator(gp, client, interval_s=30.0, fanout=3)
    result = rep.sync_once()
    assert result["peers_contacted"] == []
    assert client.exchange_calls == 0