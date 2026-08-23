"""Privacy-Preserving Inference with Certified Differential Privacy.

Provides an epsilon-differential-privacy accountant that tracks per-user
privacy budgets across inference requests, plus a configuration container
for DP mechanisms applied at various injection points in the model.

Typical usage::

    accountant = PrivacyAccountant()
    budget = accountant.get_budget("user_abc")
    if budget.can_serve(request_epsilon=2.0):
        accountant.record_spend("user_abc", epsilon=2.0, model="llama-70b", prompt_len=512)
        # ... serve request with DP mechanism ...
    else:
        # budget exhausted — deny or fall back to non-DP path
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, ClassVar

# ---------------------------------------------------------------------------
# EpsilonBudget
# ---------------------------------------------------------------------------


@dataclass
class EpsilonBudget:
    """Tracks a single user's differential privacy (epsilon) budget.

    Attributes:
        user_id: Unique identifier for the user this budget belongs to.
        total_epsilon: Total epsilon budget allocated to this user.
        spent_epsilon: Epsilon already consumed by prior requests.
    """

    user_id: str
    total_epsilon: float = 10.0
    spent_epsilon: float = 0.0

    @property
    def remaining(self) -> float:
        """Return the remaining epsilon budget for this user.

        Returns:
            ``total_epsilon - spent_epsilon``, clamped to zero.
        """
        return max(0.0, self.total_epsilon - self.spent_epsilon)

    def can_serve(self, request_epsilon: float) -> bool:
        """Check whether *request_epsilon* fits within the remaining budget.

        Args:
            request_epsilon: Epsilon cost of the candidate request.

        Returns:
            ``True`` if ``remaining >= request_epsilon``, ``False`` otherwise.
        """
        return self.remaining >= request_epsilon

    def spend(self, request_epsilon: float) -> None:
        """Record an epsilon spend for this user, raising if budget exceeded.

        Args:
            request_epsilon: Epsilon cost to deduct from the budget.

        Raises:
            ValueError: If *request_epsilon* is negative.
            RuntimeError: If *request_epsilon* exceeds the remaining budget.
        """
        if request_epsilon < 0:
            raise ValueError(
                f"request_epsilon must be non-negative, got {request_epsilon}"
            )
        if not self.can_serve(request_epsilon):
            raise RuntimeError(
                f"Budget exhausted for user {self.user_id!r}: "
                f"remaining={self.remaining:.4f}, requested={request_epsilon:.4f}"
            )
        self.spent_epsilon += request_epsilon


# ---------------------------------------------------------------------------
# PrivacyAccountant
# ---------------------------------------------------------------------------


class PrivacyAccountant:
    """Persistent epsilon-differential-privacy accountant.

    Tracks per-user epsilon budgets via a simple JSON-backed store.
    Thread-safe for concurrent access from multiple request handlers.

    Args:
        storage_path: Directory path for the persistent budget database.
                      Created on first use if it does not exist.

    .. rubric: Atomicity

    Writes update the on-disk file under a ``threading.Lock``.  For
    multi-process deployments, replace the JSON file with a dedicated
    database backend (Redis, SQLite, …).
    """

    _DB_FILENAME: ClassVar[str] = "privacy_budgets.json"

    def __init__(self, storage_path: str = ".privacy_db") -> None:
        self._storage_path = Path(storage_path)
        self._db_path = self._storage_path / self._DB_FILENAME
        self._lock = threading.RLock()
        self._budgets: dict[str, EpsilonBudget] = {}
        self._load()

    # ---- public API -------------------------------------------------------

    def get_budget(self, user_id: str) -> EpsilonBudget:
        """Return (or create) the budget tracker for *user_id*.

        Args:
            user_id: The user whose budget to retrieve.

        Returns:
            An :class:`EpsilonBudget` instance for the user.  If no budget
            exists yet, a new one with default parameters is created.
        """
        with self._lock:
            if user_id not in self._budgets:
                self._budgets[user_id] = EpsilonBudget(user_id=user_id)
            return self._budgets[user_id]

    def check_request(self, user_id: str, request_epsilon: float) -> bool:
        """Check whether *user_id* has enough budget for *request_epsilon*.

        This is a read-only check; it does **not** deduct anything.

        Args:
            user_id: The user submitting the request.
            request_epsilon: The epsilon cost of the request.

        Returns:
            ``True`` if the user's remaining budget is sufficient.
        """
        return self.get_budget(user_id).can_serve(request_epsilon)

    def record_spend(
        self,
        user_id: str,
        epsilon: float,
        model: str,
        prompt_len: int,
    ) -> None:
        """Deduct *epsilon* from *user_id*'s budget and persist.

        The spend is recorded together with metadata (model, prompt length,
        timestamp) for auditability.

        Args:
            user_id: The user whose budget to charge.
            epsilon: Amount of epsilon to deduct.
            model: Model identifier (e.g. ``"llama-70b"``).
            prompt_len: Number of tokens in the prompt.

        Raises:
            ValueError: If *epsilon* is negative.
            RuntimeError: If the user's remaining budget is insufficient.
        """
        budget = self.get_budget(user_id)
        with self._lock:
            budget.spend(epsilon)
            self._persist(user_id, epsilon, model, prompt_len)

    def get_remaining_budget(self, user_id: str) -> float:
        """Return the remaining epsilon budget for *user_id*.

        Args:
            user_id: The user to query.

        Returns:
            Remaining epsilon as a ``float`` (clamped to zero).
        """
        return self.get_budget(user_id).remaining

    def summary(self) -> dict[str, Any]:
        """Return aggregate statistics across all tracked users.

        Returns:
            A dictionary with keys::

                {
                    "total_users":     int,
                    "total_spent":     float,
                    "avg_spent":       float,
                    "max_spent":       float,
                    "total_budget":    float,
                    "total_remaining": float,
                }
        """
        with self._lock:
            budgets = list(self._budgets.values())

        if not budgets:
            return {
                "total_users": 0,
                "total_spent": 0.0,
                "avg_spent": 0.0,
                "max_spent": 0.0,
                "total_budget": 0.0,
                "total_remaining": 0.0,
            }

        total_spent = sum(b.spent_epsilon for b in budgets)
        total_budget = sum(b.total_epsilon for b in budgets)
        max_spent = max(b.spent_epsilon for b in budgets)

        return {
            "total_users": len(budgets),
            "total_spent": round(total_spent, 6),
            "avg_spent": round(total_spent / len(budgets), 6),
            "max_spent": round(max_spent, 6),
            "total_budget": round(total_budget, 6),
            "total_remaining": round(total_budget - total_spent, 6),
        }

    # ---- persistence ------------------------------------------------------

    def _load(self) -> None:
        """Load persisted budgets from the JSON database file."""
        if not self._db_path.exists():
            return
        try:
            raw = self._db_path.read_text(encoding="utf-8")
            data = json.loads(raw)
            if isinstance(data, dict):
                for user_id, entry in data.items():
                    if isinstance(entry, dict):
                        self._budgets[user_id] = EpsilonBudget(
                            user_id=user_id,
                            total_epsilon=entry.get("total_epsilon", 10.0),
                            spent_epsilon=entry.get("spent_epsilon", 0.0),
                        )
        except (json.JSONDecodeError, OSError, KeyError) as exc:
            # Corrupted or unreadable file — start fresh.

            warnings.warn(
                f"Failed to load privacy budgets from {self._db_path}: {exc}. "
                f"Starting with an empty state."
            )

    def _persist(self, user_id: str, epsilon: float, model: str, prompt_len: int) -> None:
        """Write an audit entry and flush the full budget state to disk.

        Args:
            user_id: The user that was charged.
            epsilon: Epsilon amount that was spent.
            model: Model identifier for the request.
            prompt_len: Prompt token count for the request.
        """
        self._storage_path.mkdir(parents=True, exist_ok=True)

        # Build serialisable state.
        state: dict[str, dict[str, float]] = {}
        for uid, budget in self._budgets.items():
            state[uid] = {
                "total_epsilon": budget.total_epsilon,
                "spent_epsilon": budget.spent_epsilon,
            }

        payload: dict[str, Any] = {
            "state": state,
            "audit_log": [
                {
                    "user_id": user_id,
                    "epsilon": epsilon,
                    "model": model,
                    "prompt_len": prompt_len,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            ],
        }

        self._db_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


# ---------------------------------------------------------------------------
# DPInferenceConfig
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DPInferenceConfig:
    """Configuration for differentially-private inference.

    Specifies which DP mechanism to use, the privacy parameters, where in
    the model pipeline to inject noise, and gradient clipping norms.

    Attributes:
        mechanism: DP mechanism name — ``"gaussian"`` or ``"laplace"``.
        epsilon: Target epsilon value for (ε, δ)-DP (default 8.0).
        delta: Target delta value for (ε, δ)-DP (default 1e-5).
        sensitivity: L2 / L1 sensitivity of the query (default 1.0).
        injection_point: Where to apply the DP mechanism in the model::

            - ``"embeddings"`` — noise is added to the token embeddings.
            - ``"logits"`` — noise is added to the output logits (default).
            - ``"output"`` — noise is applied to the final decoded tokens.

        clip_norm: Maximum gradient / activation norm for clipping before
                   adding noise (default 1.0).
    """

    mechanism: str = "gaussian"
    epsilon: float = 8.0
    delta: float = 1e-5
    sensitivity: float = 1.0
    injection_point: str = "logits"
    clip_norm: float = 1.0

    def __post_init__(self) -> None:
        """Validate configuration parameters on construction."""
        if self.mechanism not in ("gaussian", "laplace"):
            raise ValueError(
                f"mechanism must be 'gaussian' or 'laplace', got {self.mechanism!r}"
            )
        if self.epsilon <= 0:
            raise ValueError(f"epsilon must be > 0, got {self.epsilon}")
        if self.delta <= 0 or self.delta >= 1:
            raise ValueError(f"delta must be in (0, 1), got {self.delta}")
        if self.sensitivity <= 0:
            raise ValueError(f"sensitivity must be > 0, got {self.sensitivity}")
        if self.injection_point not in ("embeddings", "logits", "output"):
            raise ValueError(
                f"injection_point must be one of 'embeddings', 'logits', 'output', "
                f"got {self.injection_point!r}"
            )
        if self.clip_norm <= 0:
            raise ValueError(f"clip_norm must be > 0, got {self.clip_norm}")
