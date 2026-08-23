"""Privacy configuration dataclasses for DP inference.

Extracted from :mod:`distllm.core.dp_inference`.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_EPSILON = 4.0
_DEFAULT_DELTA = 1.0e-6
_DEFAULT_MAX_GRAD_NORM = 1.0


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


__all__ = [
    "DPConfig",
    "BudgetEntry",
    "DPGenerationResult",
]
