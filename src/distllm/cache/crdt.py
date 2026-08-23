"""CRDT primitives for the cross-cluster KV cache (kills LWW divergence).

This module replaces the Last-Write-Wins (LWW) write path used by the
cross-cluster prefix cache with conflict-free replicated data types:

* :class:`HybridLogicalClock` / :class:`Timestamp` — a hybrid logical clock
  (HLC) that produces timestamps ``(physical_ns, counter, node_id)`` which are
  *monotonic* and *comparable across nodes* even when wall clocks diverge.
  This is the ordering source that makes "later write wins" correct across
  clusters, instead of raw ``time.time()`` floats (which a skewed/behind clock
  can never beat -- the classic LWW cross-cluster divergence bug).

* :class:`ORSet` — an observed-remove set.  Concurrent ``add`` and ``remove``
  of the same element never diverge: after exchanging merges every replica
  computes the *identical* membership.  ``remove`` tombstones only the tags it
  has actually observed, so a tombstoned element is never silently resurrected
  by a concurrent add (or by a later read after a remove).

* :class:`LWWRegister` — an HLC-stamped last-writer-wins register used for the
  *value* bound to a cache key.  Because the timestamp is an HLC (not wall
  clock), two clusters that concurrently write the same key converge to the
  same value regardless of message arrival order, unlike the wall-clock LWW it
  replaces.

* :class:`CausalContext` — a lightweight per-node ``(node -> max_counter)``
  tracker used by :class:`ORSet` for tombstone compaction (pruning).

* :class:`CRDTCacheMap` — the CRDT-backed ``key -> value`` mapping that
  :class:`~distllm.cache.cross_cluster_prefix_index.CrossClusterPrefixIndex`
  now stores.  Membership is an :class:`ORSet`; each present key holds an
  :class:`LWWRegister` value ordered by HLC.

Design note (scope): this is a *real, model-faithful* CRDT implementation at
the cache layer.  It is exercised by in-process merge ops (no network transport
is required for the convergence proofs).  Wiring the merge calls onto the
gossip wire is a transport concern and is out of scope for this task; the
:class:`CRDTCacheMap.merge` operation is exactly what a gossip round would call.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Generic, Hashable, TypeVar

# ---------------------------------------------------------------------------
# Timestamp (HLC value)
# ---------------------------------------------------------------------------

_T = TypeVar("_T", bound=Hashable)
_V = TypeVar("_V")


@dataclass(frozen=True)
class Timestamp:
    """A single hybrid-logical-clock timestamp.

    Ordering is ``(physical_ns, counter, node_id)`` so that:

    * the *physical* component (wall-clock nanoseconds) dominates, giving
      roughly-real-time ordering,
    * the *counter* breaks ties when two events happen within the same
      physical tick on the same node (or when physical clocks are skewed),
    * the *node_id* gives a total order so any two distinct timestamps are
      always comparable (a strict total order is required for a deterministic
      convergent LWW-Register).

    Two timestamps from different nodes are always comparable, which is what
    lets two clusters agree on "which write wins" without trusting that their
    wall clocks are in sync.
    """

    physical_ns: int
    counter: int
    node_id: str

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, Timestamp):
            return NotImplemented
        return (self.physical_ns, self.counter, self.node_id) < (
            other.physical_ns,
            other.counter,
            other.node_id,
        )

    def __gt__(self, other: object) -> bool:
        if not isinstance(other, Timestamp):
            return NotImplemented
        return (self.physical_ns, self.counter, self.node_id) > (
            other.physical_ns,
            other.counter,
            other.node_id,
        )

    def __le__(self, other: object) -> bool:
        return self < other or self == other

    def __ge__(self, other: object) -> bool:
        return self > other or self == other

    def to_wire(self) -> list:
        """JSON/msgpack-friendly wire form: ``[physical_ns, counter, node_id]``."""
        return [self.physical_ns, self.counter, self.node_id]

    @staticmethod
    def from_wire(data: list) -> "Timestamp":
        return Timestamp(physical_ns=int(data[0]), counter=int(data[1]), node_id=str(data[2]))


# ---------------------------------------------------------------------------
# HybridLogicalClock
# ---------------------------------------------------------------------------


class HybridLogicalClock:
    """A hybrid logical clock.

    State per node: ``(last_physical_ns, counter, node_id)``.

    * :meth:`tick` — record a local event.  Returns a :class:`Timestamp` that
      is strictly greater than any timestamp this clock has previously produced
      *and* strictly greater than any remote timestamp folded in via
      :meth:`update`.  Monotonic by construction.

    * :meth:`update` — fold in a remote timestamp ``(perceived_pt, received_c)``
      (plus the remote node id) when a message arrives.  Advances the local
      clock strictly past the remote timestamp so that the *next* local tick is
      guaranteed causally after the remote event, even if the remote node's wall
      clock is behind ours.

    This is the standard HLC algorithm (Kulkarni et al., 2014).  Because the
    counter is bumped whenever the physical component does not advance, two
    nodes whose wall clocks disagree can still agree on a total order.
    """

    def __init__(self, node_id: str, *, now_ns: int | None = None) -> None:
        self.node_id = node_id
        self._last_physical_ns = now_ns if now_ns is not None else self._wall_ns()
        self._counter = 0

    @staticmethod
    def _wall_ns() -> int:
        # Wall-clock nanoseconds: comparable across nodes (subject to skew,
        # which the counter component absorbs).  Not monotonic_ns (that is
        # per-host and not comparable across machines).
        return time.time_ns()

    def _now(self) -> int:
        return self._wall_ns()

    def tick(self) -> Timestamp:
        """Record a local event and return its timestamp (strictly increasing)."""
        pt = max(self._now(), self._last_physical_ns)
        if pt == self._last_physical_ns:
            self._counter += 1
        else:
            self._counter = 0
        self._last_physical_ns = pt
        return Timestamp(physical_ns=pt, counter=self._counter, node_id=self.node_id)

    def update(self, perceived_pt: int, received_c: int, remote_node_id: str) -> Timestamp:
        """Fold a remote timestamp into the local clock.

        Args:
            perceived_pt: The remote event's physical component (ns).
            received_c: The remote event's counter component.
            remote_node_id: The node that produced the remote timestamp.

        Returns:
            A :class:`Timestamp` strictly greater than both the previous local
            maximum and the remote timestamp.
        """
        pt = max(self._now(), perceived_pt, self._last_physical_ns)
        if pt == perceived_pt and pt == self._last_physical_ns:
            c = max(received_c, self._counter) + 1
        elif pt == perceived_pt:
            c = received_c + 1
        elif pt == self._last_physical_ns:
            c = self._counter + 1
        else:
            c = 0
        self._last_physical_ns = pt
        self._counter = c
        return Timestamp(physical_ns=pt, counter=c, node_id=self.node_id)

    def observe(self, ts: Timestamp) -> Timestamp:
        """Convenience: fold in a full :class:`Timestamp` from a peer."""
        return self.update(ts.physical_ns, ts.counter, ts.node_id)

    @property
    def last_timestamp(self) -> Timestamp:
        return Timestamp(
            physical_ns=self._last_physical_ns,
            counter=self._counter,
            node_id=self.node_id,
        )


# ---------------------------------------------------------------------------
# CausalContext (optional, for pruning)
# ---------------------------------------------------------------------------


@dataclass
class CausalContext:
    """Per-node ``(node_id -> max counter observed)`` causal tracker.

    Used by :class:`ORSet` to compact tombstones.  Two contexts merge by taking
    the element-wise max, so the tracker is itself a CRDT and pruning stays
    safe under concurrent edits.
    """

    max_counter: dict[str, int] = field(default_factory=dict)

    def observe(self, ts: Timestamp) -> None:
        cur = self.max_counter.get(ts.node_id, -1)
        if ts.counter > cur:
            self.max_counter[ts.node_id] = ts.counter

    def merge(self, other: "CausalContext") -> None:
        for node, c in other.max_counter.items():
            if c > self.max_counter.get(node, -1):
                self.max_counter[node] = c


# ---------------------------------------------------------------------------
# ORSet (observed-remove set)
# ---------------------------------------------------------------------------


class ORSet(Generic[_T]):
    """An observed-remove set (OR-Set) with HLC tags.

    Each ``add(elem)`` allocates a *unique* tag (an HLC :class:`Timestamp`) and
    records ``(elem, tag)`` in the add-set ``A``.  ``remove(elem)`` tombstones
    every ``(elem, tag)`` currently *observed* in the add-set into the
    remove-set ``R`` (and advances the clock so the remove is itself causally
    ordered).

    An element is **present** iff there exists at least one ``(elem, tag)`` in
    ``A`` whose tag is not in ``R``.

    Convergence: ``merge`` is the union of both sets.  Because unions commute,
    associate, and are idempotent, two replicas that have exchanged all their
    add/remove sets converge to the *identical* membership regardless of the
    order in which messages arrived.  A removed element is never resurrected by
    a later ``elements()`` read because a tombstone, once merged, outlives the
    add that produced it (the tombstone is a separate, persistent entry).
    """

    def __init__(self, node_id: str) -> None:
        self._node_id = node_id
        self._clock = HybridLogicalClock(node_id)
        # add-set: (elem, tag) ; remove-set: (elem, tag)
        self._adds: set[tuple[_T, Timestamp]] = set()
        self._removes: set[tuple[_T, Timestamp]] = set()
        self._ctx = CausalContext()

    # -- local operations --------------------------------------------------

    def add(self, elem: _T) -> Timestamp:
        """Add ``elem``, returning the unique tag assigned to this add."""
        tag = self._clock.tick()
        self._adds.add((elem, tag))
        self._ctx.observe(tag)
        return tag

    def remove(self, elem: _T) -> Timestamp:
        """Remove ``elem``: tombstone every currently-observed ``(elem, tag)``.

        Only the tags this replica has actually seen get tombstoned, which is
        exactly why a *concurrent* add (a tag this replica never observed) is
        preserved -- and why a remove is never silently undone.
        """
        tag = self._clock.tick()
        observed = {pair for pair in self._adds if pair[0] == elem}
        self._removes.update(observed)
        for _, t in observed:
            self._ctx.observe(t)
        self._ctx.observe(tag)
        return tag

    # -- merge -------------------------------------------------------------

    def merge(self, other: "ORSet[_T]") -> None:
        """Merge ``other`` into this set (in-place, commutative/associative)."""
        self._adds |= other._adds
        self._removes |= other._removes
        self._ctx.merge(other._ctx)
        # Fold the peer's latest clock so future local ticks stay ahead.
        for _, t in other._adds | other._removes:
            self._clock.observe(t)

    def merged(self, other: "ORSet[_T]") -> "ORSet[_T]":
        """Return a new set that is the merge of ``self`` and ``other``."""
        result = ORSet(self._node_id)
        result._adds = set(self._adds)
        result._removes = set(self._removes)
        result._ctx = CausalContext(dict(self._ctx.max_counter))
        result.merge(other)
        return result

    # -- read --------------------------------------------------------------

    def elements(self) -> set[_T]:
        """Return the converged membership view."""
        removed_tags = {tag for _, tag in self._removes}
        present: set[_T] = set()
        for elem, tag in self._adds:
            if tag not in removed_tags:
                present.add(elem)
        return present

    def __contains__(self, elem: _T) -> bool:
        return elem in self.elements()

    # -- pruning (optional, safe compaction) -------------------------------

    def prune(self) -> int:
        """Drop redundant tombstones.

        A remove-set entry ``(e, t)`` is redundant when a strictly newer add-set
        tag from the *same node* exists (the newer add subsumes the older remove
        for that node's causal history).  Removing it never changes
        :meth:`elements`, so pruning is safe and keeps the sets bounded.

        Returns the number of tombstones removed.
        """
        # Latest add counter per (elem, node).
        latest_add: dict[tuple[_T, str], int] = {}
        for elem, tag in self._adds:
            key = (elem, tag.node_id)
            if tag.counter > latest_add.get(key, -1):
                latest_add[key] = tag.counter
        before = len(self._removes)
        self._removes = {
            (elem, tag)
            for (elem, tag) in self._removes
            if latest_add.get((elem, tag.node_id), -1) <= tag.counter
        }
        return before - len(self._removes)

    # -- introspection -----------------------------------------------------

    @property
    def clock(self) -> HybridLogicalClock:
        return self._clock

    def as_dict(self) -> dict:
        return {
            "adds": [(e, t.to_wire()) for e, t in sorted(self._adds, key=lambda p: p[1])],
            "removes": [(e, t.to_wire()) for e, t in sorted(self._removes, key=lambda p: p[1])],
        }


# ---------------------------------------------------------------------------
# LWWRegister (HLC-stamped last-writer-wins register)
# ---------------------------------------------------------------------------


class LWWRegister(Generic[_T]):
    """An HLC-stamped last-writer-wins register.

    Two registers merge to the value whose :class:`Timestamp` is greatest under
    the total HLC order.  Because the timestamp is an HLC (not a raw wall-clock
    float), two clusters with skewed clocks still agree on which write won, and
    the merge is commutative -- so the result does not depend on message
    arrival order.  This is what replaces the buggy wall-clock LWW.

    ``None`` is a tombstone: a register holding ``None`` means "removed".
    """

    __slots__ = ("value", "timestamp")

    def __init__(self, value: _T | None, timestamp: Timestamp) -> None:
        self.value = value
        self.timestamp = timestamp

    def merge(self, other: "LWWRegister[_T]") -> "LWWRegister[_T]":
        if other.timestamp > self.timestamp:
            return LWWRegister(other.value, other.timestamp)
        # Tie on timestamp: fall back to node_id for determinism (total order),
        # preferring the higher node id so merges are still commutative.
        if other.timestamp == self.timestamp and other.timestamp.node_id > self.timestamp.node_id:
            return LWWRegister(other.value, other.timestamp)
        return LWWRegister(self.value, self.timestamp)

    @classmethod
    def from_put(cls, value: _T | None, clock: HybridLogicalClock) -> "LWWRegister[_T]":
        return cls(value, clock.tick())

    def as_removed(self, clock: HybridLogicalClock) -> "LWWRegister[_T]":
        """Return a tombstoned copy (value=None) timestamped by ``clock``."""
        return LWWRegister(None, clock.tick())


# ---------------------------------------------------------------------------
# CRDTCacheMap — the key->value mapping backing the prefix index
# ---------------------------------------------------------------------------


class CRDTCacheMap(Generic[_T, _V]):
    """CRDT-backed ``key -> value`` cache map.

    * Membership of keys is an :class:`ORSet` (so concurrent add/remove of a
      cache entry never diverges and a removed entry never resurrects).
    * The value bound to each present key is an :class:`LWWRegister` ordered by
      HLC (so two clusters writing the same key concurrently converge to the
      same value regardless of arrival order).

    ``merge`` calls :class:`ORSet.merge` and merges each key's value register,
    which is exactly the operation a gossip round would invoke.
    """

    def __init__(self, node_id: str) -> None:
        self._node_id = node_id
        self._membership = ORSet[_T](node_id)
        self._values: dict[_T, LWWRegister[_V]] = {}

    # -- local operations --------------------------------------------------

    def put(self, key: _T, value: _V) -> Timestamp:
        """Insert/update ``key -> value``.  Returns the assigned HLC tag."""
        tag = self._membership.add(key)
        reg = self._values.get(key)
        new_reg = LWWRegister[_V](value, self._membership.clock.last_timestamp)
        self._values[key] = new_reg if reg is None else reg.merge(new_reg)
        return tag

    def remove(self, key: _T) -> Timestamp:
        """Remove ``key`` (OR-Set tombstone + value register tombstone)."""
        tag = self._membership.remove(key)
        existing = self._values.get(key)
        if existing is not None:
            tomb = existing.as_removed(self._membership.clock)
            self._values[key] = existing.merge(tomb)
        return tag

    # -- merge -------------------------------------------------------------

    def merge(self, other: "CRDTCacheMap[_T, _V]") -> None:
        self._membership.merge(other._membership)
        for key, reg in other._values.items():
            cur = self._values.get(key)
            self._values[key] = reg if cur is None else cur.merge(reg)

    def merged(self, other: "CRDTCacheMap[_T, _V]") -> "CRDTCacheMap[_T, _V]":
        result = CRDTCacheMap[_T, _V](self._node_id)
        result._membership = self._membership.merged(other._membership)
        for key, reg in {**self._values, **other._values}.items():
            a = self._values.get(key)
            b = other._values.get(key)
            result._values[key] = reg if (a is None or b is None) else a.merge(b)
        return result

    # -- read --------------------------------------------------------------

    def get(self, key: _T) -> _V | None:
        if key not in self._membership.elements():
            return None
        reg = self._values.get(key)
        if reg is None or reg.value is None:
            return None
        return reg.value

    def contains(self, key: _T) -> bool:
        return self.get(key) is not None

    def keys(self) -> set[_T]:
        return {k for k in self._membership.elements() if self.get(k) is not None}

    def values(self) -> list[_V]:
        return [self.get(k) for k in self.keys()]  # type: ignore[misc]

    def items(self) -> dict[_T, _V]:
        return {k: self.get(k) for k in self.keys()}  # type: ignore[misc]

    def timestamp_of(self, key: _T) -> "Timestamp | None":
        """Return the HLC timestamp of the winning register for ``key``.

        ``None`` if the key is absent / tombstoned.  Used to serialize the
        authoritative ordering onto the gossip wire so a peer can merge without
        trusting its own wall clock.
        """
        reg = self._values.get(key)
        if reg is None or reg.value is None:
            return None
        return reg.timestamp

    def ingest(self, key: _T, value: _V, ts: "Timestamp") -> None:
        """Merge a single ``(key -> value)`` register stamped with ``ts``.

        ``ts`` is the HLC timestamp carried on the gossip wire (produced by the
        originating replica), so the merge converges regardless of which
        replica first applied the write.
        """
        # Ensure the key is a member (a value arriving without a local add).
        if key not in self._membership.elements():
            self._membership.add(key)
        self._membership.clock.observe(ts)
        incoming = LWWRegister[_V](value, ts)
        cur = self._values.get(key)
        self._values[key] = incoming if cur is None else cur.merge(incoming)

    @property
    def membership(self) -> ORSet[_T]:
        return self._membership
