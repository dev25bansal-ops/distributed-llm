"""Load-aware router that tracks active connections per node.

Selects the candidate with the fewest outstanding connections,
providing simple least-connections load balancing.
"""

from __future__ import annotations

import sys


class LoadAwareRouter:
    """Tracks active connection counts per node and selects the least-loaded
    candidate.

    Connections are recorded via ``record_connection(node_id, delta)``
    where *delta* is typically ``+1`` for a new connection and ``-1``
    for a completed one.

    Parameters
    ----------
    initial_load :
        Optional mapping of node_id -> initial connection count.
    """

    def __init__(
        self, initial_load: dict[str, int] | None = None
    ) -> None:
        self._connections: dict[str, int] = {}
        if initial_load:
            for nid, count in initial_load.items():
                if count < 0:
                    raise ValueError(
                        f"Initial load for {nid!r} must be "
                        f"non-negative, got {count}"
                    )
                self._connections[nid] = count

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select_node(self, candidates: list[str]) -> str:
        """Return the candidate with the fewest active connections.

        Candidates with no recorded load are treated as having zero
        connections.  If multiple candidates tie, the first encountered
        in *candidates* order is returned.

        Raises ``ValueError`` when *candidates* is empty.
        """
        if not candidates:
            raise ValueError("candidates list must not be empty")

        best = candidates[0]
        best_load = self._connections.get(best, 0)

        for node_id in candidates[1:]:
            load = self._connections.get(node_id, 0)
            if load < best_load:
                best = node_id
                best_load = load

        return best

    def record_connection(self, node_id: str, delta: int) -> None:
        """Adjust the active connection count for *node_id* by *delta*.

        Typical usage::

            router.record_connection("node-a", +1)  # connect
            ...
            router.record_connection("node-a", -1)  # disconnect

        The internal counter is clamped at zero (never negative).
        """
        current = self._connections.get(node_id, 0)
        updated = max(0, current + delta)
        self._connections[node_id] = updated

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_load(self, node_id: str) -> int:
        """Return the active connection count for *node_id* (0 if unknown)."""
        return self._connections.get(node_id, 0)

    @property
    def connections(self) -> dict[str, int]:
        """Return a copy of the node -> active-connection map."""
        return dict(self._connections)

    @property
    def total_connections(self) -> int:
        """Return the sum of active connections across all tracked nodes."""
        return sum(self._connections.values())

    @property
    def size(self) -> int:
        """Return the number of tracked nodes."""
        return len(self._connections)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(nodes={self.size}, "
            f"active={self.total_connections})"
        )


if sys.version_info >= (3, 9):
    __all__ = ("LoadAwareRouter",)
else:
    __all__ = ["LoadAwareRouter"]
