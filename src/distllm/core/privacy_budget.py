"""Live per-tenant differential-privacy budget meter.

Builds on the existing Gaussian mechanism + advanced composition in
:mod:`distllm.core.differential_privacy` (H2 noise, M4 ε-composition) and
adds the missing operational piece: a **live, per-tenant privacy budget
meter**.  For regulated verticals this is what turns "we add DP noise" into a
provable, auditable claim — "tenant X has spent 2.3 of its 5.0 ε budget across
1,204 queries; 2.7 ε remain."

The meter:
* tracks the number of DP queries per tenant,
* recomputes the *composed* ε via advanced composition
  (``ε · sqrt(2k·ln(1.25/δ))``) on every query,
* exposes a live ``remaining`` budget and a hard ``exhausted`` gate so the
  serving layer can refuse further queries once the tenant's ε is spent
  (fail-closed, never silently over-spend).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from distllm.core.differential_privacy import DifferentialPrivacy, DifferentialPrivacyConfig


@dataclass
class TenantPrivacyBudget:
    """Live privacy accounting for a single tenant.

    Args:
        tenant_id: Tenant identifier.
        epsilon_limit: Total ε the tenant is allowed to spend.
        delta: δ used for composition (shared with the DP config).
        config: The DP config (per-query ε, δ, noise scale).
    """

    tenant_id: str
    epsilon_limit: float = 5.0
    delta: float = 1e-5
    config: DifferentialPrivacyConfig = field(default_factory=DifferentialPrivacyConfig)
    _queries: int = 0
    # Reentrant lock: record_query() holds the lock while calling
    # is_exhausted()/spent_epsilon()/snapshot(), each of which re-acquires it.
    _lock: threading.RLock = field(default_factory=threading.RLock)

    @property
    def queries(self) -> int:
        return self._queries

    def record_query(self) -> dict[str, Any]:
        """Record one DP query and return the live budget snapshot.

        Raises:
            RuntimeError: if the tenant's ε budget is already exhausted
                (fail-closed — callers must check :meth:`remaining` first or
                handle the exception to refuse the request).
        """
        with self._lock:
            if self.is_exhausted():
                raise RuntimeError(
                    f"Privacy budget exhausted for tenant {self.tenant_id}: "
                    f"spent {self.spent_epsilon():.3f} / {self.epsilon_limit:.3f} ε"
                )
            self._queries += 1
            return self.snapshot()

    def snapshot(self) -> dict[str, Any]:
        """Return a live budget snapshot (composed ε via advanced composition)."""
        with self._lock:
            k = self._queries
            spent = self._composed_epsilon(k)
            return {
                "tenant_id": self.tenant_id,
                "epsilon_limit": self.epsilon_limit,
                "delta": self.delta,
                "num_queries": k,
                "spent_epsilon": round(spent, 4),
                "total_epsilon": round(spent, 4),
                "remaining_epsilon": round(max(0.0, self.epsilon_limit - spent), 4),
                "exhausted": spent >= self.epsilon_limit,
                "noise_multiplier": round(self.config.sigma, 6),
            }

    def spent_epsilon(self) -> float:
        with self._lock:
            return self._composed_epsilon(self._queries)

    def remaining(self) -> float:
        return max(0.0, self.epsilon_limit - self.spent_epsilon())

    def is_exhausted(self) -> bool:
        return self.spent_epsilon() >= self.epsilon_limit

    def _composed_epsilon(self, k: int) -> float:
        """Advanced-composition total ε for k queries (matches M4 math)."""
        import math

        if k <= 0:
            return 0.0
        return self.config.epsilon * math.sqrt(2 * k * math.log(1.25 / self.delta))


class PrivacyBudgetMeter:
    """Registry of per-tenant privacy budgets with a live meter.

    Args:
        default_epsilon_limit: ε ceiling applied to tenants not explicitly
            registered.
        default_delta: δ for composition.
        default_config: DP config template (per-query ε, noise scale).
        dp: Optional :class:`DifferentialPrivacy` instance; when supplied,
            the meter can also apply noise (H2) and account for it together.
    """

    def __init__(
        self,
        default_epsilon_limit: float = 5.0,
        default_delta: float = 1e-5,
        default_config: DifferentialPrivacyConfig | None = None,
        dp: DifferentialPrivacy | None = None,
    ) -> None:
        self._default_epsilon = default_epsilon_limit
        self._default_delta = default_delta
        self._default_config = default_config or DifferentialPrivacyConfig()
        self._dp = dp
        self._budgets: dict[str, TenantPrivacyBudget] = {}
        self._lock = threading.RLock()

    def register_tenant(
        self,
        tenant_id: str,
        epsilon_limit: float | None = None,
        delta: float | None = None,
        config: DifferentialPrivacyConfig | None = None,
    ) -> TenantPrivacyBudget:
        with self._lock:
            if tenant_id in self._budgets:
                return self._budgets[tenant_id]
            budget = TenantPrivacyBudget(
                tenant_id=tenant_id,
                epsilon_limit=epsilon_limit if epsilon_limit is not None else self._default_epsilon,
                delta=delta if delta is not None else self._default_delta,
                config=config or self._default_config,
            )
            self._budgets[tenant_id] = budget
            return budget

    def get(self, tenant_id: str) -> TenantPrivacyBudget | None:
        with self._lock:
            return self._budgets.get(tenant_id)

    def meter(self, tenant_id: str) -> dict[str, Any]:
        """Live budget snapshot for a tenant (registers on first use)."""
        with self._lock:
            budget = self._budgets.get(tenant_id)
            if budget is None:
                budget = self.register_tenant(tenant_id)
        return budget.snapshot()

    def record_query(self, tenant_id: str) -> dict[str, Any]:
        """Record a DP query for a tenant and return the live snapshot.

        Fails closed (raises) when the tenant's budget is exhausted.
        """
        with self._lock:
            budget = self._budgets.get(tenant_id)
            if budget is None:
                budget = self.register_tenant(tenant_id)
        return budget.record_query()

    def all_snapshots(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            return {tid: b.snapshot() for tid, b in self._budgets.items()}
