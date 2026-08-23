"""Federated incentive primitive — credit + reputation ledger.

This module provides :class:`CreditLedger`, a minimal but *real* economic
primitive for the federated training layer.  It exists to fix the "Petals
flaw": without an incentive layer, contributors have no reason to stay
online.  ``CreditLedger`` attaches credit and reputation to every federated
round so that honest, productive nodes accumulate balance and a rising
reputation score.

Design goals (scoped):
- **Stdlib only** (``sqlite3`` is in the standard library) so the ledger can
  be imported and unit-tested without the heavy ML stack (torch, etc.).
- **Deterministic & testable**: credit is a pure function of the supplied
  ``weight_metric``; reputation moves monotonically with success/failure.
- **In-memory by default**, with an *optional* sqlite store for durability.
- **No new network deps**: the ledger is a local primitive wired into the
  existing round flow; it does not introduce a settlement chain.

Wiring points (see ``federated_finetuner.py`` and ``dist/federated_merge.py``):
- When a node completes a round / submits a valid adapter, call
  ``record_contribution`` and ``apply_reputation(node_id, success=True)``.
- When a round fails or a node abandons, call
  ``apply_reputation(node_id, success=False)``.
"""

from __future__ import annotations

import sqlite3
import threading
from collections import defaultdict
from typing import Any

# Starting reputation for a node that has not yet been scored.  A neutral
# baseline of 1.0 means the first success raises it and the first failure
# lowers it, satisfying "get_reputation monotonic-ish (success raises,
# failure lowers)".
_BASELINE_REPUTATION = 1.0
_REP_SUCCESS_DELTA = 0.1
_REP_FAILURE_DELTA = 0.1
_REP_MIN = 0.0
_REP_MAX = 5.0

# Multiplier converting a contribution's ``weight_metric`` into credits.
_DEFAULT_CREDIT_PER_UNIT = 1.0


class CreditLedger:
    """In-memory (optionally sqlite-backed) credit + reputation ledger.

    A ``CreditLedger`` tracks two per-node quantities:

    * **balance** (credits) — accumulated economic credit, incremented by
      :meth:`record_contribution`.
    * **reputation** (float) — updated by :meth:`apply_reputation`, rising on
      success and falling on failure.

    Args:
        db_path: Optional path to a sqlite database.  When provided, state is
            persisted and reloaded on construction.  When ``None`` the ledger
            is purely in-memory.
        credit_per_unit: Multiplier converting a ``weight_metric`` into the
            number of credits awarded for a contribution.
    """

    def __init__(
        self,
        db_path: str | None = None,
        credit_per_unit: float = _DEFAULT_CREDIT_PER_UNIT,
    ):
        self._db_path = db_path
        self._credit_per_unit = float(credit_per_unit)
        self._lock = threading.RLock()

        # In-memory mirrors of persisted state (the source of truth when
        # no db_path is configured, and a hot cache when it is).
        self._balances: dict[str, float] = defaultdict(float)
        self._reputation: dict[str, float] = {}
        self._total_contributed: dict[str, float] = defaultdict(float)
        self._contribution_count: dict[str, int] = defaultdict(int)

        if self._db_path is not None:
            self._init_db()
            self._load()

    # ── Contribution / credit ──────────────────────────────────────────

    def record_contribution(
        self,
        round_id: str,
        node_id: str,
        weight_metric: float,
    ) -> float:
        """Record a contribution and award credit to ``node_id``.

        Args:
            round_id: Identifier of the federated round this contribution
                belongs to (used for the audit log).
            node_id: The contributing node.
            weight_metric: A positive scalar measuring the size/value of the
                contribution (e.g. dataset size or local steps).  Credit
                awarded equals ``weight_metric * credit_per_unit``.

        Returns:
            The number of credits awarded (>= 0).
        """
        credits = max(0.0, float(weight_metric)) * self._credit_per_unit
        with self._lock:
            self._balances[node_id] += credits
            self._total_contributed[node_id] += credits
            self._contribution_count[node_id] += 1
            if node_id not in self._reputation:
                self._reputation[node_id] = _BASELINE_REPUTATION
            self._persist_contribution(round_id, node_id, weight_metric, credits)
        return credits

    def get_balance(self, node_id: str) -> float:
        """Return the current credit balance for ``node_id`` (0.0 if unknown)."""
        with self._lock:
            return self._balances.get(node_id, 0.0)

    def get_total_contributed(self, node_id: str) -> float:
        """Return the lifetime credit awarded to ``node_id``."""
        with self._lock:
            return self._total_contributed.get(node_id, 0.0)

    def get_contribution_count(self, node_id: str) -> int:
        with self._lock:
            return self._contribution_count.get(node_id, 0)

    # ── Reputation ─────────────────────────────────────────────────────

    def apply_reputation(self, node_id: str, success: bool) -> float:
        """Update ``node_id``'s reputation based on an outcome.

        A *success* raises the reputation score; a *failure* lowers it.
        Movement is monotonic-ish: starting from the baseline, successes only
        ever increase the stored value and failures only ever decrease it
        (bounded to ``[_REP_MIN, _REP_MAX]``).

        Args:
            node_id: The node whose reputation is updated.
            success: ``True`` for a successful round/submit, ``False`` for a
                failure or abandonment.

        Returns:
            The node's new reputation score.
        """
        with self._lock:
            cur = self._reputation.get(node_id, _BASELINE_REPUTATION)
            if success:
                new = min(_REP_MAX, cur + _REP_SUCCESS_DELTA)
            else:
                new = max(_REP_MIN, cur - _REP_FAILURE_DELTA)
            self._reputation[node_id] = new
            self._persist_reputation(node_id, new)
            return new

    def get_reputation(self, node_id: str) -> float:
        """Return ``node_id``'s reputation (baseline if never scored)."""
        with self._lock:
            return self._reputation.get(node_id, _BASELINE_REPUTATION)

    # ── Snapshot / inspection ──────────────────────────────────────────

    def snapshot(self) -> dict[str, dict[str, Any]]:
        """Return a mapping ``node_id -> {balance, reputation, contributions}``."""
        with self._lock:
            nodes = set(self._balances) | set(self._reputation)
            return {
                node_id: {
                    "balance": self._balances.get(node_id, 0.0),
                    "reputation": self._reputation.get(node_id, _BASELINE_REPUTATION),
                    "contributions": self._contribution_count.get(node_id, 0),
                }
                for node_id in sorted(nodes)
            }

    # ── Persistence (optional) ─────────────────────────────────────────

    def _init_db(self) -> None:
        assert self._db_path is not None
        conn = sqlite3.connect(self._db_path)
        try:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS balances ("
                "  node_id TEXT PRIMARY KEY,"
                "  balance REAL NOT NULL,"
                "  reputation REAL NOT NULL,"
                "  contributions INTEGER NOT NULL"
                ")"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS contributions ("
                "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
                "  round_id TEXT NOT NULL,"
                "  node_id TEXT NOT NULL,"
                "  weight_metric REAL NOT NULL,"
                "  credits REAL NOT NULL"
                ")"
            )
            conn.commit()
        finally:
            conn.close()

    def _load(self) -> None:
        assert self._db_path is not None
        conn = sqlite3.connect(self._db_path)
        try:
            for node_id, balance, reputation, contributions in conn.execute(
                "SELECT node_id, balance, reputation, contributions FROM balances"
            ):
                self._balances[node_id] = balance
                self._reputation[node_id] = reputation
                self._contribution_count[node_id] = contributions
        finally:
            conn.close()

    def _persist_contribution(
        self, round_id: str, node_id: str, weight_metric: float, credits: float
    ) -> None:
        if self._db_path is None:
            return
        conn = sqlite3.connect(self._db_path)
        try:
            conn.execute(
                "INSERT INTO contributions (round_id, node_id, weight_metric, credits)"
                " VALUES (?, ?, ?, ?)",
                (round_id, node_id, float(weight_metric), credits),
            )
            conn.execute(
                "INSERT INTO balances (node_id, balance, reputation, contributions)"
                " VALUES (?, ?, ?, 1) ON CONFLICT(node_id) DO UPDATE SET"
                "  balance = balance + ?,"
                "  contributions = contributions + 1",
                (
                    node_id,
                    self._balances[node_id],
                    self._reputation.get(node_id, _BASELINE_REPUTATION),
                    credits,
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def _persist_reputation(self, node_id: str, reputation: float) -> None:
        if self._db_path is None:
            return
        conn = sqlite3.connect(self._db_path)
        try:
            conn.execute(
                "INSERT INTO balances (node_id, balance, reputation, contributions)"
                " VALUES (?, ?, ?, 0) ON CONFLICT(node_id) DO UPDATE SET"
                "  reputation = ?",
                (
                    node_id,
                    self._balances.get(node_id, 0.0),
                    reputation,
                    reputation,
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"<CreditLedger nodes={len(self.snapshot())} db={self._db_path!r}>"
