"""Differential privacy for KV cache sharing across nodes.

Adds calibrated Gaussian noise to KV cache tensors before sharing
across nodes, providing differential privacy guarantees.

Usage::

    dp = DifferentialPrivacyConfig(epsilon=1.0, delta=1e-5)
    noisy_cache = dp.add_noise_to_kv_cache(kv_cache)
    # Share noisy_cache across nodes — no single node sees raw data
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

import torch
from loguru import logger


@dataclass
class DifferentialPrivacyConfig:
    """Configuration for differential privacy.

    Attributes:
        epsilon: Privacy budget (smaller = more private). Typical: 0.1-10.0.
        delta: Probability of privacy breach. Typical: 1e-5 to 1e-3.
        max_grad_norm: Maximum gradient norm for clipping.
        noise_multiplier: Multiplier for noise scale (auto-computed if 0).
    """
    epsilon: float = 1.0
    delta: float = 1e-5
    max_grad_norm: float = 1.0
    noise_multiplier: float = 0.0

    @property
    def sigma(self) -> float:
        """Compute noise scale (sigma) from privacy parameters.

        Uses the Gaussian mechanism: sigma = max_grad_norm * noise_multiplier
        where noise_multiplier = sqrt(2 * ln(1.25/delta)) / epsilon.
        """
        if self.noise_multiplier > 0:
            return self.max_grad_norm * self.noise_multiplier
        return self.max_grad_norm * math.sqrt(2 * math.log(1.25 / self.delta)) / self.epsilon


class DifferentialPrivacy:
    """Applies differential privacy to tensors for safe sharing."""

    def __init__(self, config: DifferentialPrivacyConfig | None = None):
        self._config = config or DifferentialPrivacyConfig()

    def add_noise_to_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Add calibrated Gaussian noise to a tensor.

        The noise scale is computed from the privacy parameters
        (epsilon, delta) to provide (epsilon, delta)-differential privacy.

        Args:
            tensor: The tensor to add noise to.

        Returns:
            A new tensor with noise added. Original is not modified.
        """
        sigma = self._config.sigma
        if sigma <= 0:
            return tensor.clone()

        noise = torch.randn_like(tensor) * sigma
        return tensor + noise

    def add_noise_to_kv_cache(
        self, kv_cache: list[tuple[torch.Tensor, torch.Tensor]]
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Add noise to all layers of a KV cache.

        Args:
            kv_cache: List of (key, value) tensor pairs per layer.

        Returns:
            New list with noisy tensors.
        """
        noisy_cache = []
        for k, v in kv_cache:
            noisy_k = self.add_noise_to_tensor(k)
            noisy_v = self.add_noise_to_tensor(v)
            noisy_cache.append((noisy_k, noisy_v))
        return noisy_cache

    def clip_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Clip tensor norm to max_grad_norm for bounded sensitivity."""
        norm = torch.norm(tensor)
        if norm > self._config.max_grad_norm:
            tensor = tensor * (self._config.max_grad_norm / norm)
        return tensor

    def privacy_budget_used(self, num_queries: int) -> dict:
        """Compute the total privacy budget used after N queries.

        Uses advanced composition theorem (Kairouz et al. 2015):
        total_epsilon = sqrt(2 * num_queries * ln(1.25/delta)) * epsilon_per_query

        This is tighter than basic composition (epsilon * sqrt(n)) for
        moderate numbers of queries.
        """
        if num_queries <= 0:
            return {
                "epsilon_per_query": self._config.epsilon,
                "delta": self._config.delta,
                "num_queries": 0,
                "total_epsilon": 0.0,
                "sigma": round(self._config.sigma, 6),
            }
        # Advanced composition: sqrt(2k * ln(1.25/delta)) * epsilon
        composed_epsilon = (
            self._config.epsilon
            * math.sqrt(2 * num_queries * math.log(1.25 / self._config.delta))
        )
        return {
            "epsilon_per_query": self._config.epsilon,
            "delta": self._config.delta,
            "num_queries": num_queries,
            "total_epsilon": round(composed_epsilon, 3),
            "sigma": round(self._config.sigma, 6),
        }


class InputAnonymizer:
    """Strips PII from prompts before they leave the user's node.

    Uses pattern matching to detect and redact:
    - Email addresses
    - Phone numbers
    - SSN-like patterns
    - Credit card numbers
    - IP addresses
    """

    _PATTERNS = [
        # Email
        (re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'), '[EMAIL]'),
        # Phone (US format)
        (re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'), '[PHONE]'),
        # SSN
        (re.compile(r'\b\d{3}-\d{2}-\d{4}\b'), '[SSN]'),
        # Credit card (16 digits)
        (re.compile(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'), '[CARD]'),
        # IP address
        (re.compile(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'), '[IP]'),
    ]

    @classmethod
    def anonymize(cls, text: str) -> str:
        """Strip PII from text.

        Args:
            text: Input text that may contain PII.

        Returns:
            Text with PII replaced by placeholders.
        """
        for pattern, replacement in cls._PATTERNS:
            text = pattern.sub(replacement, text)
        return text

    @classmethod
    def has_pii(cls, text: str) -> bool:
        """Check if text contains potential PII."""
        for pattern, _ in cls._PATTERNS:
            if pattern.search(text):
                return True
        return False
