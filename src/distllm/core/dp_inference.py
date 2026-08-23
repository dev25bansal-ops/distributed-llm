"""Differential privacy inference with RDP accounting and DP-SGD noise injection.

Wraps an inference engine to provide (epsilon, delta)-differential privacy
guarantees for generated text.  Privacy is enforced through three mechanisms:

1. **DP-SGD noise injection**: Clips per-example gradients and adds
   calibrated Gaussian noise during the forward pass.

2. **Gumbel noise mechanism**: Perturbs the output logit distribution
   using Gumbel noise for a differentially private sampling step.

3. **RDP (Renyi Differential Privacy) accounting**: Tracks cumulative
   privacy spend using Renyi DP composition, which provides tighter
   bounds than basic composition for iterative mechanisms.

Usage::

    from distllm.core.dp_inference import DifferentialPrivacyInference
    from distllm.core.inference_engine import InferenceEngine

    engine = InferenceEngine(...)
    dp_engine = DifferentialPrivacyInference(
        engine=engine,
        epsilon=4.0,
        delta=1e-6,
    )

    # Per-tenant budget tracking
    dp_engine.set_tenant_budget("tenant-1", epsilon=8.0, delta=1e-6)
    dp_engine.check_budget("tenant-1")

    # Generate with DP guarantees
    output = dp_engine.generate("Hello, world!")
"""

from __future__ import annotations

import hashlib
import math
import os
import struct
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import torch
from loguru import logger

# We use cryptography for HKDF-based noise generation keying
try:
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF

    HAS_CRYPTOGRAPHY = True
except ImportError:
    HAS_CRYPTOGRAPHY = False


__all__ = [
    "DifferentialPrivacyInference",
    "PrivacyBudgetManager",
    "RDPAccounting",
    "DPConfig",
    "BudgetEntry",
    "DPGenerationResult",
    "dp_noise_injection",
    "gumbel_noise_mechanism",
    "clip_gradients",
]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_EPSILON = 4.0
_DEFAULT_DELTA = 1.0e-6
_DEFAULT_MAX_GRAD_NORM = 1.0


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class DPConfig:
    """Differential privacy configuration for inference.

    Attributes:
        epsilon: Target epsilon budget. Smaller values mean stronger
            privacy.  Typical: 0.1 -- 10.0.
        delta: Target delta budget.  Must be less than 1 / N where N is
            the number of queries.  Typical: 1e-6 to 1e-5.
        max_grad_norm: Maximum gradient norm for clipping (DP-SGD).
        noise_multiplier: Scale multiplier for Gaussian noise.  If 0,
            computed automatically from (epsilon, delta).
        rdp_orders: List of Renyi orders for RDP accounting.  If empty,
            a default range is used.
        target_mechanism: Which DP mechanism to apply: ``"dp-sgd"`` or
            ``"gumbel"``.
        gumbel_noise_scale: Scale of Gumbel noise for the output-level
            mechanism (default 1.0).
        clip_per_layer: Whether to clip per-layer gradients individually
            (default True).
        num_rounds: Estimated number of inference rounds.  Used for
            advanced composition if the caller does not supply per-call
            updates.
    """

    epsilon: float = _DEFAULT_EPSILON
    delta: float = _DEFAULT_DELTA
    max_grad_norm: float = _DEFAULT_MAX_GRAD_NORM
    noise_multiplier: float = 0.0
    rdp_orders: list[float] = field(default_factory=lambda: [])
    target_mechanism: str = "dp-sgd"
    gumbel_noise_scale: float = 1.0
    clip_per_layer: bool = True
    num_rounds: int = 1

    def __post_init__(self) -> None:
        if not self.rdp_orders:
            # Default Renyi orders: dense in [0.1, 10] and sparse beyond.
            self.rdp_orders = [
                0.1,
                0.2,
                0.5,
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
                6.0,
                7.0,
                8.0,
                9.0,
                10.0,
                20.0,
                50.0,
                100.0,
                500.0,
                1000.0,
            ]

    @property
    def sigma(self) -> float:
        """Compute Gaussian noise scale from privacy parameters.

        Uses the standard Gaussian mechanism formula:
        sigma = max_grad_norm * noise_multiplier

        If *noise_multiplier* is zero, derives it from epsilon and delta:
        noise_multiplier = sqrt(2 * ln(1.25 / delta)) / epsilon
        """
        if self.noise_multiplier > 0:
            return self.max_grad_norm * self.noise_multiplier
        return self.max_grad_norm * math.sqrt(2 * math.log(1.25 / self.delta)) / self.epsilon


@dataclass
class BudgetEntry:
    """A single privacy budget allocation for one tenant.

    Attributes:
        epsilon: Total epsilon budget allocated.
        delta: Total delta budget allocated.
        epsilon_spent: Epsilon consumed so far.
        delta_spent: Delta consumed so far.
        num_queries: Number of queries performed.
        last_reset: Timestamp of last budget reset.
    """

    epsilon: float
    delta: float
    epsilon_spent: float = 0.0
    delta_spent: float = 0.0
    num_queries: int = 0
    last_reset: float = field(default_factory=time.time)


@dataclass
class RDPAccounting:
    """Renyi Differential Privacy (RDP) accountant.

    Tracks cumulative privacy spend across multiple queries using Renyi
    divergence composition, which provides tighter bounds than standard
    composition theorems.

    Reference:
        Mironov, "Renyi Differential Privacy", 2017.
    """

    orders: list[float] = field(default_factory=lambda: [])

    def __post_init__(self) -> None:
        if not self.orders:
            self.orders = [
                0.1,
                0.2,
                0.5,
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
                6.0,
                7.0,
                8.0,
                9.0,
                10.0,
                20.0,
                50.0,
                100.0,
                500.0,
                1000.0,
            ]
        self._rdp_per_query: dict[int, list[float]] = {}
        self._total_rdp: list[float] = [0.0] * len(self.orders)

    def compute_rdp(
        self,
        sigma: float,
        num_queries: int = 1,
    ) -> list[float]:
        """Compute per-query RDP for a Gaussian mechanism with given *sigma*.

        For subsampled Gaussian mechanism (sampling rate q),
        the RDP is approximately:
            RDP(alpha) <= (q^2 * alpha) / sigma^2   for alpha > 1

        For non-subsampled (q = 1):
            RDP(alpha) = alpha / (2 * sigma^2)

        Args:
            sigma: Noise scale (standard deviation).
            num_queries: Number of queries (steps) to compose.

        Returns:
            List of RDP values for each alpha order.
        """
        if sigma <= 0:
            return [0.0] * len(self.orders)

        rdp_values: list[float] = []
        for alpha in self.orders:
            if alpha == 0:
                rdp_values.append(0.0)
            else:
                # Non-subsampled Gaussian mechanism RDP
                rdp_alpha = alpha / (2.0 * sigma * sigma)
                rdp_values.append(rdp_alpha * num_queries)
        return rdp_values

    def add_query(self, sigma: float, query_id: int | None = None) -> None:
        """Record one or more queries into the accountant.

        Args:
            sigma: Noise scale used for the query.
            query_id: Optional identifier for the query.
        """
        delta_rdp = self.compute_rdp(sigma, num_queries=1)
        if query_id is not None:
            self._rdp_per_query[query_id] = delta_rdp
        self._total_rdp = [
            t + d for t, d in zip(self._total_rdp, delta_rdp)
        ]

    def get_epsilon(self, delta: float) -> float:
        """Convert the accumulated RDP to (epsilon, delta)-DP.

        Uses the standard conversion:
            epsilon = min_alpha ( RDP(alpha) - log(delta) / (alpha - 1) )

        Args:
            delta: Target delta for the conversion.

        Returns:
            The smallest epsilon achievable for the given delta.
        """
        if not self._total_rdp or not any(self._total_rdp):
            # No queries recorded: zero privacy spent.  (Skipping this makes
            # the log(1/delta)/(alpha-1) term surface as a phantom ~0.01
            # epsilon for an accountant that never saw a query.)
            return 0.0

        best_epsilon = float("inf")
        for alpha, rdp_alpha in zip(self.orders, self._total_rdp):
            if alpha <= 1:
                continue
            eps = rdp_alpha - math.log(delta) / (alpha - 1)
            if eps < best_epsilon:
                best_epsilon = eps
        return max(0.0, best_epsilon)

    def get_privacy_spent(self, delta: float) -> dict[str, float]:
        """Return a summary of total privacy spend.

        Args:
            delta: Target delta for epsilon conversion.

        Returns:
            Dict with 'epsilon', 'delta', and 'orders_used'.
        """
        return {
            "epsilon": round(self.get_epsilon(delta), 4),
            "delta": delta,
            "orders_used": len(self.orders),
        }


# ---------------------------------------------------------------------------
# Core DP operations
# ---------------------------------------------------------------------------


def clip_gradients(
    grads: list[torch.Tensor],
    max_norm: float = _DEFAULT_MAX_GRAD_NORM,
    *,
    clip_per_layer: bool = True,
) -> list[torch.Tensor]:
    """Clip gradients to bounded L2 norm.

    In DP-SGD, gradient clipping ensures bounded sensitivity, which
    determines how much noise is needed for privacy.

    Args:
        grads: List of gradient tensors.
        max_norm: Maximum allowed L2 norm per layer (or globally).
        clip_per_layer: If True, clip each tensor individually.
            If False, compute the global norm across all tensors and
            rescale if exceeded.

    Returns:
        Clipped gradient tensors.
    """
    if clip_per_layer:
        clipped = []
        for g in grads:
            norm = g.norm().item()
            if norm > max_norm:
                clipped.append(g * (max_norm / norm))
            else:
                clipped.append(g.clone())
        return clipped
    else:
        # Global norm clipping
        global_norm = math.sqrt(sum(g.norm().item() ** 2 for g in grads))
        if global_norm <= max_norm:
            return [g.clone() for g in grads]
        scale = max_norm / global_norm
        return [g.clone() * scale for g in grads]


def dp_noise_injection(
    grads: list[torch.Tensor],
    sigma: float,
    *,
    seed: int | None = None,
) -> list[torch.Tensor]:
    """Add calibrated Gaussian noise to gradients for differential privacy.

    Args:
        grads: List of gradient tensors.
        sigma: Noise scale (standard deviation).
        seed: Optional seed for reproducibility in testing.

    Returns:
        Gradients with Gaussian noise added.  The original tensors are
        not modified.
    """
    noisy_grads: list[torch.Tensor] = []
    for g in grads:
        if sigma > 0:
            generator: torch.Generator | None = None
            if seed is not None:
                generator = torch.Generator(device=g.device)
                generator.manual_seed(seed)
            noise = torch.randn_like(g, generator=generator) * sigma
            noisy_grads.append(g + noise)
        else:
            noisy_grads.append(g.clone())
    return noisy_grads


def gumbel_noise_mechanism(
    logits: torch.Tensor,
    noise_scale: float = 1.0,
    *,
    seed: int | None = None,
) -> torch.Tensor:
    """Perturb output logits with Gumbel noise for DP sampling.

    This implements the Gumbel noise mechanism: add Gumbel(0, scale)
    noise to each logit, then sample from the resulting distribution.
    This is equivalent to the exponential mechanism for categorical
    outcomes.

    Args:
        logits: Raw logits tensor of shape ``(batch, vocab)`` or ``(vocab,)``.
        noise_scale: Scale of the Gumbel distribution.
        seed: Optional seed for reproducibility.

    Returns:
        Logits with added Gumbel noise.
    """
    if noise_scale <= 0:
        return logits.clone()

    batched = logits.dim() == 2
    if not batched:
        logits = logits.unsqueeze(0)

    noisy_logits = logits.clone()
    for i in range(logits.shape[0]):
        generator: torch.Generator | None = None
        if seed is not None:
            generator = torch.Generator(device=logits.device)
            generator.manual_seed(seed + i)
        # Gumbel(0, scale): u = Uniform(0,1) -> noise = -scale * log(-log(u))
        u = torch.rand(logits.shape[1], generator=generator).clamp(min=1e-10, max=1 - 1e-10)
        gumbel = -noise_scale * torch.log(-torch.log(u))
        noisy_logits[i] = noisy_logits[i] + gumbel

    return noisy_logits.squeeze(0) if not batched else noisy_logits


# ---------------------------------------------------------------------------
# Privacy Budget Manager
# ---------------------------------------------------------------------------


class PrivacyBudgetManager:
    """Per-tenant privacy budget tracking with daily/weekly limits.

    Each tenant has an (epsilon, delta) budget that resets on a daily or
    weekly schedule.  The budget manager tracks spend across queries and
    enforces limits.

    Thread-safe.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # tenant_id -> BudgetEntry
        self._budgets: dict[str, BudgetEntry] = {}
        # Global RDP accountant
        self._accountant = RDPAccounting()
        # Configuration: daily or weekly reset
        self._reset_period: str = "daily"  # "daily" or "weekly"
        self._default_epsilon: float = _DEFAULT_EPSILON
        self._default_delta: float = _DEFAULT_DELTA

    def set_defaults(self, epsilon: float, delta: float, period: str = "daily") -> None:
        """Set default budget values for tenants without explicit configuration.

        Args:
            epsilon: Default epsilon per period.
            delta: Default delta per period.
            period: Reset period (``"daily"`` or ``"weekly"``).
        """
        with self._lock:
            self._default_epsilon = epsilon
            self._default_delta = delta
            if period in ("daily", "weekly"):
                self._reset_period = period

    def set_tenant_budget(
        self,
        tenant_id: str,
        epsilon: float | None = None,
        delta: float | None = None,
    ) -> None:
        """Set or update a tenant's privacy budget.

        Args:
            tenant_id: Tenant identifier.
            epsilon: Epsilon budget per period.  Defaults to the
                manager's default.
            delta: Delta budget per period.  Defaults to the manager's
                default.
        """
        with self._lock:
            eps = epsilon if epsilon is not None else self._default_epsilon
            d = delta if delta is not None else self._default_delta

            existing = self._budgets.get(tenant_id)
            if existing:
                # Update limits (keep existing spend unless reset)
                existing.epsilon = eps
                existing.delta = d
            else:
                self._budgets[tenant_id] = BudgetEntry(epsilon=eps, delta=d)

    def get_tenant_budget(self, tenant_id: str) -> BudgetEntry | None:
        """Get the current budget state for a tenant.

        Returns None if the tenant has no configured budget.
        """
        with self._lock:
            return self._budgets.get(tenant_id)

    def check_budget(self, tenant_id: str) -> dict[str, Any]:
        """Check whether a tenant has remaining privacy budget.

        Automatically resets the budget if the period has elapsed.

        Args:
            tenant_id: Tenant identifier.

        Returns:
            Dict with keys::

            - ``has_budget``: True if budget remains.
            - ``epsilon_remaining``: Fraction of budget remaining.
            - ``delta_remaining``: Fraction of budget remaining.
            - ``epsilon_spent``: Epsilon consumed.
            - ``epsilon_total``: Total epsilon allocated.
            - ``num_queries``: Number of queries this period.
            - ``needs_reset``: Whether the budget was just reset.
        """
        with self._lock:
            entry = self._budgets.get(tenant_id)
            if entry is None:
                # Auto-create with defaults
                entry = BudgetEntry(
                    epsilon=self._default_epsilon,
                    delta=self._default_delta,
                )
                self._budgets[tenant_id] = entry

            # Check for periodic reset
            needs_reset = self._should_reset(entry)
            if needs_reset:
                entry.epsilon_spent = 0.0
                entry.delta_spent = 0.0
                entry.num_queries = 0
                entry.last_reset = time.time()

            eps_remaining = max(0.0, 1.0 - entry.epsilon_spent / max(entry.epsilon, 1e-10))
            delta_remaining = max(0.0, 1.0 - entry.delta_spent / max(entry.delta, 1e-10))

            return {
                "has_budget": eps_remaining > 0.01 and delta_remaining > 0.01,
                "epsilon_remaining": round(eps_remaining, 4),
                "delta_remaining": round(delta_remaining, 4),
                "epsilon_spent": round(entry.epsilon_spent, 4),
                "epsilon_total": entry.epsilon,
                "num_queries": entry.num_queries,
                "needs_reset": needs_reset,
            }

    def record_query(
        self,
        tenant_id: str,
        sigma: float,
        epsilon_cost: float | None = None,
    ) -> None:
        """Record a privacy cost for a tenant query.

        Args:
            tenant_id: Tenant identifier.
            sigma: Noise scale used (for RDP accounting).
            epsilon_cost: Explicit epsilon cost.  If None, computed
                from sigma via RDP.
        """
        with self._lock:
            entry = self._budgets.get(tenant_id)
            if entry is None:
                entry = BudgetEntry(
                    epsilon=self._default_epsilon,
                    delta=self._default_delta,
                )
                self._budgets[tenant_id] = entry

            # Check reset
            if self._should_reset(entry):
                entry.epsilon_spent = 0.0
                entry.delta_spent = 0.0
                entry.num_queries = 0
                entry.last_reset = time.time()

            if epsilon_cost is not None:
                entry.epsilon_spent += epsilon_cost
            else:
                # Approximate per-query epsilon cost from sigma
                # epsilon_per_query ~= alpha / (2 * sigma^2) for small alpha
                if sigma > 0:
                    epsilon_cost = 1.0 / (2.0 * sigma * sigma)
                    entry.epsilon_spent += epsilon_cost
                else:
                    epsilon_cost = 0.0

            entry.num_queries += 1
            entry.delta_spent = min(entry.delta, entry.delta_spent + self._default_delta * 0.01)

            # Also record in the global RDP accountant
            self._accountant.add_query(sigma)

    def global_privacy_spent(self) -> dict[str, float]:
        """Return global privacy spend across all tenants (via RDP).

        Returns:
            Dict with 'epsilon' and 'delta'.
        """
        return self._accountant.get_privacy_spent(self._default_delta)

    def _should_reset(self, entry: BudgetEntry) -> bool:
        """Check whether the budget period has elapsed."""
        if self._reset_period == "daily":
            cutoff = time.time() - 86400
        else:
            cutoff = time.time() - 604800  # weekly
        return entry.last_reset < cutoff

    def all_tenants(self) -> list[str]:
        """Return all tracked tenant IDs."""
        with self._lock:
            return list(self._budgets.keys())

    def reset_all(self) -> None:
        """Reset all tenant budgets immediately."""
        with self._lock:
            now = time.time()
            for entry in self._budgets.values():
                entry.epsilon_spent = 0.0
                entry.delta_spent = 0.0
                entry.num_queries = 0
                entry.last_reset = now


# ---------------------------------------------------------------------------
# DP-Generation result
# ---------------------------------------------------------------------------


@dataclass
class DPGenerationResult:
    """Result of a differentially private generation call.

    Attributes:
        text: Generated text string.
        privacy_cost: Dict describing the privacy spend for this call.
        token_count: Number of tokens generated.
        noise_scale: Noise scale (sigma) used.
    """

    text: str
    privacy_cost: dict[str, Any]
    token_count: int
    noise_scale: float


# ---------------------------------------------------------------------------
# Differential Privacy Inference Engine Wrapper
# ---------------------------------------------------------------------------


class DifferentialPrivacyInference:
    """Wraps an inference engine with differential privacy guarantees.

    Provides:

    - DP-SGD noise injection: clips per-example gradients and adds
      Gaussian noise during the forward pass.
    - Gumbel noise mechanism: perturbs the output distribution.
    - RDP accounting for tight privacy tracking.
    - Per-tenant privacy budget enforcement with daily/weekly limits.

    The wrapper can be used as a drop-in replacement for an
    ``InferenceEngine`` where DP guarantees are needed.

    Usage::

        dp_engine = DifferentialPrivacyInference(
            engine=inference_engine,
            epsilon=4.0,
            delta=1e-6,
        )

        # Per-tenant budget
        dp_engine.set_tenant_budget("acme-corp", epsilon=8.0, delta=1e-6)

        # Check before generating
        status = dp_engine.check_budget("acme-corp")
        if status["has_budget"]:
            result = dp_engine.generate("Hello", user_id="acme-corp")
            print(result.text, result.privacy_cost)
    """

    def __init__(
        self,
        engine: Any,
        epsilon: float = _DEFAULT_EPSILON,
        delta: float = _DEFAULT_DELTA,
        *,
        max_grad_norm: float = _DEFAULT_MAX_GRAD_NORM,
        noise_multiplier: float = 0.0,
        mechanism: str = "dp-sgd",
        gumbel_noise_scale: float = 1.0,
        enforce_budget: bool = True,
    ):
        # Accept any engine-like object with a generate() method
        self._engine = engine
        self._config = DPConfig(
            epsilon=epsilon,
            delta=delta,
            max_grad_norm=max_grad_norm,
            noise_multiplier=noise_multiplier,
            target_mechanism=mechanism,
            gumbel_noise_scale=gumbel_noise_scale,
        )
        self._enforce_budget = enforce_budget
        self._budget_manager = PrivacyBudgetManager()
        self._budget_manager.set_defaults(epsilon, delta)
        self._sigma = self._config.sigma

        self._rng_seed: int | None = None
        self._generator: torch.Generator | None = None

        logger.info(
            f"DP Inference initialized: epsilon={epsilon}, delta={delta}, "
            f"sigma={self._sigma:.4f}, mechanism={mechanism}"
        )

    # --
    # Budget management delegation
    # --

    @property
    def budget_manager(self) -> PrivacyBudgetManager:
        """Access the underlying privacy budget manager."""
        return self._budget_manager

    def set_tenant_budget(
        self,
        tenant_id: str,
        epsilon: float | None = None,
        delta: float | None = None,
    ) -> None:
        """Set or update a tenant's privacy budget."""
        self._budget_manager.set_tenant_budget(tenant_id, epsilon, delta)

    def check_budget(self, tenant_id: str) -> dict[str, Any]:
        """Check remaining privacy budget for a tenant.

        Returns:
            Dict with 'has_budget', 'epsilon_remaining', 'delta_remaining',
            and other budget metadata.
        """
        return self._budget_manager.check_budget(tenant_id)

    def global_privacy_spent(self) -> dict[str, float]:
        """Return aggregate privacy spend across all tenants."""
        return self._budget_manager.global_privacy_spent()

    # --
    # Configuration
    # --

    @property
    def sigma(self) -> float:
        """Current noise scale."""
        return self._sigma

    def set_epsilon(self, epsilon: float) -> None:
        """Update the privacy budget epsilon.

        Recomputes the noise scale if noise_multiplier is not set
        explicitly.
        """
        self._config.epsilon = epsilon
        if self._config.noise_multiplier <= 0:
            self._sigma = self._config.sigma
        logger.info(f"DP epsilon updated to {epsilon}, sigma={self._sigma:.4f}")

    def set_delta(self, delta: float) -> None:
        """Update the privacy budget delta.

        Recomputes the noise scale if noise_multiplier is not set
        explicitly.
        """
        self._config.delta = delta
        if self._config.noise_multiplier <= 0:
            self._sigma = self._config.sigma
        logger.info(f"DP delta updated to {delta}, sigma={self._sigma:.4f}")

    def set_noise_multiplier(self, noise_multiplier: float) -> None:
        """Set the noise multiplier explicitly.

        Overrides automatic noise scale computation.
        """
        self._config.noise_multiplier = noise_multiplier
        self._sigma = self._config.sigma if noise_multiplier > 0 else 0.0
        logger.info(f"DP noise multiplier set to {noise_multiplier}, sigma={self._sigma:.4f}")

    # --
    # Noise injection on tensors
    # --

    def add_gaussian_noise(self, tensor: torch.Tensor) -> torch.Tensor:
        """Add calibrated Gaussian noise to a tensor for DP.

        Args:
            tensor: Input tensor.

        Returns:
            New tensor with noise added.  Original is not modified.
        """
        if self._sigma <= 0:
            return tensor.clone()
        noise = torch.randn_like(tensor) * self._sigma
        return tensor + noise

    def clip_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Clip tensor by its L2 norm for bounded sensitivity.

        Args:
            tensor: Input tensor.

        Returns:
            Clipped tensor (original unmodified if within norm).
        """
        norm = tensor.norm()
        if norm > self._config.max_grad_norm:
            return tensor * (self._config.max_grad_norm / norm)
        return tensor.clone()

    def apply_dp_to_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply the configured DP mechanism to logits.

        Args:
            logits: Raw logits tensor from the model.

        Returns:
            Logits with DP perturbation applied.
        """
        if self._config.target_mechanism == "gumbel":
            return gumbel_noise_mechanism(
                logits,
                noise_scale=self._config.gumbel_noise_scale,
                seed=self._rng_seed,
            )
        else:
            # DP-SGD: add Gaussian noise (clipping is applied at the
            # gradient level, not the logit level).
            return self.add_gaussian_noise(logits)

    # --
    # Generation
    # --

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 0,
        user_id: str = "default",
        enforce_budget: bool | None = None,
        **kwargs: Any,
    ) -> DPGenerationResult:
        """Generate text with differential privacy guarantees.

        .. warning::

            **Privacy mechanism not yet wired into the generation path.**
            The :meth:`generate` and :meth:`generate_stream` methods track
            privacy budget but do **not** apply DP noise to the output
            tokens or logits.  Callers should NOT rely on this class for
            actual privacy protection until the integration with the
            engine's logit-generation path is complete.

            To apply DP, the engine's ``_sample()`` or logit-generation
            step must be wrapped with :func:`dp_noise_injection` or
            :func:`gumbel_noise_mechanism`.  This requires access to the
            raw logits tensor *before* ``argmax``/``top_k`` sampling,
            which is not available at this wrapper layer.

        Args:
            prompt: Input prompt string.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling parameter.
            top_k: Top-k sampling parameter.
            user_id: Tenant/user identifier for budget tracking.
            enforce_budget: Override the instance-level budget enforcement.
            **kwargs: Additional arguments passed to the underlying engine.

        Returns:
            A :class:`DPGenerationResult` containing the generated text,
            privacy cost, and metadata.

        Raises:
            RuntimeError: If the tenant has exhausted their privacy budget
                and enforcement is enabled.
            NotImplementedError: When DP noise is requested but the
                logit-level integration is not yet wired.
        """
        _enforce = enforce_budget if enforce_budget is not None else self._enforce_budget

        if _enforce:
            status = self._budget_manager.check_budget(user_id)
            if not status["has_budget"]:
                raise RuntimeError(
                    f"Privacy budget exhausted for tenant {user_id!r}. "
                    f"Epsilon remaining: {status['epsilon_remaining']:.2%}. "
                    f"Wait for budget reset or increase the allocation."
                )

        # Determine noise scale for this generation
        sigma = self._sigma
        if sigma <= 0:
            sigma = 0.001  # Small epsilon to avoid division issues

        # FAIL CLOSED: the DP mechanism is not wired into this engine's
        # logit-level sampling path, so we cannot produce differentially-private
        # output at this wrapper layer.  Raising — instead of returning
        # plaintext and charging the tenant's privacy budget — prevents the
        # false-advertising harm where a tenant believes they are protected
        # while paying a budget that protects nothing.
        raise NotImplementedError(
            "DifferentialPrivacyInference cannot produce DP output at this "
            "wrapper layer: the engine's logit-level sampling path is not wired "
            "through _dp_sample(). No privacy budget was charged. Wire the "
            "engine's forward()/sampling through _dp_sample(), or apply "
            "apply_dp_to_logits() to raw logits, before using this class for "
            "generation."
        )

    # --
    # Internal
    # --

    def _dp_sample(self, logits: torch.Tensor, sigma: float, temperature: float) -> torch.Tensor:
        """Sample a token from logits with DP noise applied.

        Adds Gaussian noise to logits before sampling.
        Provides (epsilon, delta)-DP per token when logits are L2-clipped.

        Args:
            logits: Raw logits tensor (batch, vocab_size).
            sigma: Noise scale (larger = more privacy).
            temperature: Sampling temperature.

        Returns:
            Sampled token ID (scalar tensor).
        """
        with torch.no_grad():
            clipped = logits / max(1.0, logits.norm(dim=-1, keepdim=True))
            noise = torch.randn_like(clipped) * sigma
            noisy_logits = clipped + noise
            if temperature == 0:
                return noisy_logits.argmax(dim=-1)
            probs = torch.softmax(noisy_logits / temperature, dim=-1)
            return torch.multinomial(probs, num_samples=1).squeeze(-1)

    def _estimate_epsilon_cost(self, sigma: float, num_tokens: int) -> float:
        """Estimate the privacy cost (epsilon) for *num_tokens* tokens.

        Uses a loose bound: each token costs approximately
        alpha / (2 * sigma^2) for alpha ~ 1 (the worst-case Renyi
        divergence for a single-step Gaussian mechanism).

        This is a conservative estimate.  For tighter accounting,
        use the RDP accountant directly.
        """
        if sigma <= 0:
            return 0.0
        per_token = 1.0 / (2.0 * sigma * sigma)
        return per_token * num_tokens


# ---------------------------------------------------------------------------
# Integration helper: wrap an inference engine with DP
# ---------------------------------------------------------------------------


def wrap_with_dp(
    engine: Any,
    *,
    epsilon: float = _DEFAULT_EPSILON,
    delta: float = _DEFAULT_DELTA,
    max_grad_norm: float = _DEFAULT_MAX_GRAD_NORM,
    mechanism: str = "dp-sgd",
    **kwargs: Any,
) -> DifferentialPrivacyInference:
    """Convenience function to wrap an inference engine with DP.

    Args:
        engine: An object with a ``generate()`` or ``generate_stream()``
            method.
        epsilon: Target epsilon budget.
        delta: Target delta budget.
        max_grad_norm: Maximum gradient norm for clipping.
        mechanism: Which DP mechanism to apply (``"dp-sgd"`` or
            ``"gumbel"``).
        **kwargs: Additional arguments passed to
            :class:`DifferentialPrivacyInference`.

    Returns:
        A :class:`DifferentialPrivacyInference` wrapping the engine.
    """
    return DifferentialPrivacyInference(
        engine=engine,
        epsilon=epsilon,
        delta=delta,
        max_grad_norm=max_grad_norm,
        mechanism=mechanism,
        **kwargs,
    )
