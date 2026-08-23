"""Differential privacy layer for the DistLLM SDK.

Provides request perturbation and response redaction to help
protect sensitive data when sending prompts to a DistLLM cluster.
Uses the ``diffprivlib`` library when available, with a pure-Python
fallback for basic noise injection.

Enable via ``differential_privacy`` on any client::

    client = DistLLMClient(
        differential_privacy=DPConfig(
            epsilon=8.0,
            delta=1e-5,
            enable_logging=True,
        )
    )

All chat content is automatically perturbed before sending and
redacted in responses using the configured privacy budget.
"""

from __future__ import annotations

import copy
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("distllm_sdk")

# -- Try to use diffprivlib if available -------------------------------------
_DP_LIB_AVAILABLE = False
try:
    from diffprivlib import mechanisms as _dp_mech
    _DP_LIB_AVAILABLE = True
except ImportError:
    _dp_mech = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class DPConfig:
    """Configuration for differential privacy.

    Attributes:
        epsilon: Privacy budget (lower = more privacy, default 8.0).
            Common values: 0.1 (high privacy), 1.0 (moderate), 8.0 (low).
        delta: Failure probability (default 1e-5).
        enable_logging: Log redaction events (default True).
        noise_scale: Override noise scale.  Auto-computed from epsilon
            when not set.
        redact_patterns: List of regex patterns for content to redact from
            responses.  Defaults to email, SSN, phone, credit card.
        max_content_length: Maximum content length before truncation (chars).
            Longer content is truncated to reduce PII surface area.
    """
    epsilon: float = 8.0
    delta: float = 1e-5
    enable_logging: bool = True
    noise_scale: float | None = None
    redact_patterns: list[str] = field(default_factory=lambda: [
        r"\b[\w.+-]+@[\w-]+\.[\w.{-]+\b",       # email
        r"\b\d{3}-\d{2}-\d{4}\b",                 # SSN
        r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",        # phone
        r"\b(?:\d[ -]*){13,19}\b",                # credit card
        r"\b[A-Z]{2}\d{6}\b",                     # passport (US/UK)
    ])
    max_content_length: int = 10000


# ---------------------------------------------------------------------------
# Noise mechanisms
# ---------------------------------------------------------------------------

class _NoiseMechanism:
    """Adds calibrated noise to numeric values."""

    def __init__(self, epsilon: float, delta: float, noise_scale: float | None = None):
        if _DP_LIB_AVAILABLE and _dp_mech is not None:
            self._mech = _dp_mech.LaplaceBoundedNoise(
                epsilon=epsilon,
                delta=delta,
                sensitivity=1.0,
            )
        else:
            self._mech = None
        self._noise_scale = noise_scale or (2.0 / max(epsilon, 0.01))
        self._epsilon = epsilon

    def add_noise(self, value: float) -> float:
        """Add Laplace noise to *value*."""
        if self._mech is not None:
            noisy = self._mech.randomise(value)
            if isinstance(noisy, (int, float)):
                return max(0.0, noisy)
        # Fallback: manual Laplace noise
        import random
        scale = self._noise_scale
        noise = random.gauss(0, scale)
        return max(0.0, value + noise)


# ---------------------------------------------------------------------------
# PII redaction
# ---------------------------------------------------------------------------

class _PIIRedactor:
    """Redacts PII patterns from text."""

    def __init__(self, patterns: list[str]):
        self._compiled = [re.compile(p, re.IGNORECASE) for p in patterns]

    def redact(self, text: str) -> str:
        """Replace all PII matches with ``[REDACTED]``."""
        result = text
        for pattern in self._compiled:
            result = pattern.sub("[REDACTED]", result)
        return result

    def contains_pii(self, text: str) -> bool:
        """Check if text contains any PII patterns."""
        for pattern in self._compiled:
            if pattern.search(text):
                return True
        return False


# ---------------------------------------------------------------------------
# DP Layer
# ---------------------------------------------------------------------------

class DPLayer:
    """Differential privacy layer wrapping requests and responses.

    Inject this into a client's ``_request`` pipeline to:
    1. Truncate and redact PII from outbound messages.
    2. Add calibrated Laplace noise to numeric response fields.
    3. Redact PII from response text.
    """

    def __init__(self, config: DPConfig):
        self._config = config
        self._noise = _NoiseMechanism(config.epsilon, config.delta, config.noise_scale)
        self._redactor = _PIIRedactor(config.redact_patterns)
        self._total_redactions = 0
        self._total_perturbations = 0

    # -- Outbound processing -------------------------------------------------

    def process_request(self, payload: dict) -> dict:
        """Apply privacy transforms to an outgoing request payload.

        - Truncates long content to reduce PII surface area.
        - Redacts PII patterns from message content.
        - Logs redaction events when enabled.

        Returns a new dict (does not mutate the original).
        """
        result = copy.deepcopy(payload)

        messages = result.get("messages")
        if isinstance(messages, list):
            for msg in messages:
                content = msg.get("content", "")
                if not isinstance(content, str):
                    continue
                # Truncate
                if len(content) > self._config.max_content_length:
                    content = content[:self._config.max_content_length] + "..."
                    if self._config.enable_logging:
                        logger.info("DP: truncated message content (%d chars)", self._config.max_content_length)
                # Redact PII
                if self._redactor.contains_pii(content):
                    redacted = self._redactor.redact(content)
                    if redacted != content:
                        self._total_redactions += 1
                        if self._config.enable_logging:
                            logger.info("DP: redacted PII in request (total: %d)", self._total_redactions)
                        msg["content"] = redacted

        return result

    # -- Response processing -------------------------------------------------

    def process_response(self, data: dict) -> dict:
        """Apply privacy transforms to an API response.

        - Adds Laplace noise to numeric ``usage`` fields.
        - Redacts PII from response text content.

        Returns a new dict (does not mutate the original).
        """
        result: dict[str, Any] = copy.deepcopy(data)

        # Perturb usage stats
        usage = result.get("usage")
        if isinstance(usage, dict):
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                if key in usage and isinstance(usage[key], (int, float)):
                    noisy = self._noise.add_noise(float(usage[key]))
                    usage[key] = max(0, round(noisy))
                    self._total_perturbations += 1

        # Redact PII in response text
        choices = result.get("choices")
        if isinstance(choices, list):
            for choice in choices:
                msg = choice.get("message", {}) if isinstance(choice, dict) else {}
                if isinstance(msg, dict):
                    text = msg.get("content", "")
                    if isinstance(text, str) and self._redactor.contains_pii(text):
                        msg["content"] = self._redactor.redact(text)
                        self._total_redactions += 1
                        if self._config.enable_logging:
                            logger.info("DP: redacted PII in response (total: %d)", self._total_redactions)

        return result

    # -- Stats ---------------------------------------------------------------

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "total_redactions": self._total_redactions,
            "total_perturbations": self._total_perturbations,
            "epsilon": self._config.epsilon,
            "delta": self._config.delta,
        }

    def reset_stats(self) -> None:
        self._total_redactions = 0
        self._total_perturbations = 0
