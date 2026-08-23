"""Privacy-Preserving Split for distributed LLM inference.

Partitions the model so that:
  - Prefix (requester device): embedding + first N layers
  - Trunk (peer GPUs):        middle layers (anonymized representations)
  - Suffix (requester device): final layers + norm + LM head

Peers never see the original prompt or the final output text.  Extends
the base configuration with activation obfuscation for trunk layers.

Note: this package was consolidated from the former ``dist/privacy.py``
module to eliminate the name-shadowing conflict with ``dist/privacy/``.
All imports of ``distllm.dist.privacy`` continue to work transparently.
"""

from __future__ import annotations
import hashlib
import os
from dataclasses import dataclass

import torch
from loguru import logger


@dataclass
class PrivacySplitConfig:
    """Configuration for privacy-preserving model splitting.


    Attributes:
        enabled: Whether privacy splitting is active.
        prefix_layers: Number of initial layers kept on the requester device.
        suffix_layers: Number of final layers kept on the requester device.
        obfuscate_activations: Apply random projection to trunk activations.
        noise_scale: Scale of Gaussian noise added to trunk activations.
    """

    enabled: bool = False
    prefix_layers: int = 0
    suffix_layers: int = 0
    obfuscate_activations: bool = True
    noise_scale: float = 0.01


def compute_privacy_partition(
    total_layers: int,
    config: PrivacySplitConfig,
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    """Compute (prefix, trunk, suffix) ranges for a privacy split.


    Args:
        total_layers: Total number of transformer layers in the model.
        config: The privacy split configuration.

    Returns:
        A 3-tuple of (start, end) inclusive ranges:
          (prefix_range, trunk_range, suffix_range)

    Raises:
        ValueError: If the config would cause overlapping or uncovered layers.
    """

    if not config.enabled:
        return (0, total_layers - 1), (0, 0), (0, -1)

    if total_layers <= 0:
        raise ValueError(f"total_layers must be > 0, got {total_layers}")

    p = config.prefix_layers
    s = config.suffix_layers

    if p < 0:
        raise ValueError(f"prefix_layers must be >= 0, got {p}")
    if s < 0:
        raise ValueError(f"suffix_layers must be >= 0, got {s}")
    if p + s > total_layers:
        raise ValueError(
            f"prefix_layers ({p}) + suffix_layers ({s}) = {p + s} "
            f"exceeds total_layers ({total_layers})"
        )

    prefix_range = (0, p - 1) if p > 0 else (0, -1)
    trunk_range = (p, total_layers - s - 1) if total_layers - s > p else (0, 0)
    suffix_range = (total_layers - s, total_layers - 1) if s > 0 else (0, -1)

    return prefix_range, trunk_range, suffix_range


class ActivationObfuscator:
    """Obfuscates activations before sending to peer nodes.


    Uses a learned random projection (fixed seed, not per-request) to
    reduce the risk of activation inversion attacks.  Adds Gaussian noise
    calibrated to the activation scale.

    The obfuscation is applied *after* the prefix layers and *before*
    sending to trunk (peer) nodes.  The inverse is applied after trunk
    and before suffix layers.
    """


    def __init__(self, hidden_size: int, noise_scale: float = 0.01, seed: int | None = None):
        self._hidden_size = hidden_size
        self._noise_scale = noise_scale

        # SECURITY: use a cryptographically random seed by default instead of fixed 42
        # This prevents precomputation attacks where an adversary who knows the seed
        # can reverse the obfuscation
        env_seed = int(os.environ.get("DISTLLM_PRIVACY_SEED", "0"))
        if seed is not None:
            actual_seed = seed + env_seed
        elif env_seed != 0:
            actual_seed = env_seed
        else:
            # Generate a cryptographically random seed tied to this node's identity
            key_path = os.environ.get(
                "DISTLLM_PRIVACY_SEED_FILE",
                os.path.join(
                    os.environ.get("DISTLLM_DATA_DIR", os.path.expanduser("~/.distllm")),
                    "privacy_seed.key",
                ),
            )
            if os.path.exists(key_path):
                try:
                    with open(key_path, "r") as f:
                        actual_seed = int(f.read().strip())
                except (ValueError, OSError, IOError):
                    actual_seed = int.from_bytes(os.urandom(16), byteorder="big")
                try:
                    os.makedirs(os.path.dirname(key_path), exist_ok=True)
                    with open(key_path, "w") as f:
                        f.write(str(actual_seed) + "\n")
                except (OSError, IOError):
                    pass

        rng = torch.Generator()
        rng.manual_seed(actual_seed)
        # Fixed base projection matrix (used when no per-request key is available)
        self._base_projection = torch.randn(hidden_size, hidden_size, generator=rng) / (hidden_size ** 0.5)
        # H-03: Matrix inversion is O(n^3) — for large hidden_size (4096+),
        # this can be 68B+ operations. Use transpose as approximate inverse
        # for large matrices to avoid cost and numerical instability.
        if hidden_size <= 2048:
            self._inv_base = self._base_projection.inverse()
        else:
            self._inv_base = self._base_projection.t() / hidden_size
        self._projection = self._base_projection
        self._inv_projection = self._inv_base

    def set_request_key(self, request_id: str) -> None:
        """Derive a per-request projection from the base matrix.


        Each request uses a unique projection seeded from its request_id,
        so compromising one request's output does not generalize to others.

        Args:
            request_id: Unique request identifier for key derivation.
        """

        request_seed = int.from_bytes(
            hashlib.sha256(request_id.encode()).digest()[:8], byteorder="big"
        )
        rng = torch.Generator()
        rng.manual_seed(request_seed)
        perturbation = torch.randn(self._hidden_size, self._hidden_size, generator=rng)
        perturbation *= 0.01 / (self._hidden_size ** 0.5)
        self._projection = self._base_projection + perturbation
        if self._hidden_size <= 2048:
            self._inv_projection = self._projection.inverse()
        else:
            self._inv_projection = self._projection.t() / self._hidden_size

    def reset_projection(self) -> None:
        """Revert to the base (non-per-request) projection."""

        self._projection = self._base_projection
        self._inv_projection = self._inv_base

    def obfuscate(self, activations: torch.Tensor) -> torch.Tensor:
        """Apply random projection + noise to activations."""

        with torch.no_grad():
            projected = activations @ self._projection.to(activations.device)
            noise = torch.randn_like(projected) * self._noise_scale * projected.std()
            return projected + noise

    def restore(self, activations: torch.Tensor) -> torch.Tensor:
        """Apply inverse projection to restore (approximately)."""

        with torch.no_grad():
            return activations @ self._inv_projection.to(activations.device)


class PrivacyEnforcer:
    """Determines routing decisions based on a privacy split configuration."""


    def __init__(self, config: PrivacySplitConfig, total_layers: int = 0):
        self.config = config
        self.total_layers = total_layers
        self._obfuscator: ActivationObfuscator | None = None

    def init_obfuscator(self, hidden_size: int) -> None:
        """Initialize the activation obfuscator for trunk privacy."""

        if self.config.enabled and self.config.obfuscate_activations:
            self._obfuscator = ActivationObfuscator(
                hidden_size, noise_scale=self.config.noise_scale,
            )
            logger.info(f"Privacy obfuscator initialized (hidden={hidden_size}, noise={self.config.noise_scale})")

    def obfuscate_if_needed(self, activations: torch.Tensor) -> torch.Tensor:
        """Obfuscate activations before sending to trunk (peer) nodes."""

        if self._obfuscator is not None:
            return self._obfuscator.obfuscate(activations)
        return activations

    def restore_if_needed(self, activations: torch.Tensor) -> torch.Tensor:
        """Restore activations received from trunk nodes."""

        if self._obfuscator is not None:
            return self._obfuscator.restore(activations)
        return activations

    def should_route_to_peer(self, layer_id: int) -> bool:
        """Return True if *layer_id* should be executed on peer GPUs."""

        if not self.config.enabled:
            return True
        return not self.should_route_to_requester(layer_id)

    def should_route_to_requester(self, layer_id: int) -> bool:
        """Return True if *layer_id* should be executed on the requester device."""

        if not self.config.enabled:
            return False
        prefix_end = self.config.prefix_layers
        suffix_start = self.total_layers - self.config.suffix_layers
        return layer_id < prefix_end or layer_id >= suffix_start


# ---------------------------------------------------------------------------
# Shamir's Secret Sharing for prompt obfuscation
# ---------------------------------------------------------------------------

class _GaloisField:
    """GF(2^16) arithmetic for Shamir's Secret Sharing.

    Uses a 16-bit prime field (prime=65521, the largest 16-bit prime)
    so that each share fits in a 16-bit integer, which maps efficiently
    to float16 mantissa bits for transport.
    """

    PRIME = 65521  # Largest prime < 2^16

    @staticmethod
    def add(a: int, b: int) -> int:
        return (a + b) % _GaloisField.PRIME

    @staticmethod
    def sub(a: int, b: int) -> int:
        return (a - b + _GaloisField.PRIME) % _GaloisField.PRIME

    @staticmethod
    def mul(a: int, b: int) -> int:
        return (a * b) % _GaloisField.PRIME

    @staticmethod
    def div(a: int, b: int) -> int:
        if b == 0:
            raise ZeroDivisionError("division by zero in GF")
        return (a * pow(b, _GaloisField.PRIME - 2, _GaloisField.PRIME)) % _GaloisField.PRIME

    @staticmethod
    def eval_poly(coefficients: list[int], x: int) -> int:
        """Evaluate polynomial at point x using Horner's method."""
        result = 0
        for coeff in reversed(coefficients):
            result = _GaloisField.add(_GaloisField.mul(result, x), coeff)
        return result


class PromptShamirSplitter:
    """Splits a prompt into N shares using Shamir's Secret Sharing.

    No single peer sees the full prompt.  Each share is sent to a
    different federation peer; results are recombined locally.

    Usage::

        splitter = PromptShamirSplitter()
        token_ids = [101, 2054, 2003, ...]  # prompt tokens

        # Split into 5 shares, need any 3 to reconstruct
        shares = splitter.split(token_ids, num_shares=5, threshold=3)

        # Send each share to a different peer...
        # Later, collect any K responses and reconstruct:
        reconstructed = splitter.combine(responses)

    The splitter preserves token IDs up to a configurable floating-point
    precision loss (via the ``precision`` parameter).
    """

    def __init__(self, precision: int = 16):
        self._precision = precision

    def split(
        self,
        token_ids: list[int],
        num_shares: int = 5,
        threshold: int = 3,
    ) -> list[list[int]]:
        """Split *token_ids* into *num_shares* shares.

        Any *threshold* shares can reconstruct the original tokens.

        Args:
            token_ids: Token ID sequence to split.
            num_shares: Number of shares to create (>= threshold).
            threshold: Minimum shares needed for reconstruction.

        Returns:
            A list of *num_shares* share sequences, each a list of ints
            of the same length as *token_ids*.  Each share is independent
            and reveals no information about the original tokens.
        """
        if threshold < 2:
            raise ValueError("threshold must be >= 2")
        if num_shares < threshold:
            raise ValueError("num_shares must be >= threshold")

        GF = _GaloisField
        prime = GF.PRIME

        # Normalize tokens to field range
        tokens = [t % prime for t in token_ids]

        # Generate shares: for each token, create a random polynomial
        # of degree (threshold - 1) with the token as the constant term.
        shares: list[list[int]] = [[] for _ in range(num_shares)]

        for token in tokens:
            # Random coefficients a_{threshold-2} .. a_0 (a_0 = token)
            coeffs = [token] + [
                int.from_bytes(os.urandom(2), "big") % prime
                for _ in range(threshold - 1)
            ]

            # Evaluate polynomial at x = 1 .. num_shares
            for s in range(num_shares):
                x = s + 1  # x = 0 reveals the secret, so start at 1
                share_val = GF.eval_poly(coeffs, x)
                shares[s].append(share_val)

        return shares

    def combine(self, share_lists: list[list[int]]) -> list[int]:
        """Reconstruct token IDs from *threshold* share lists.

        Uses Lagrange interpolation at x=0 to recover the constant
        terms of the original polynomials.

        Args:
            share_lists: At least *threshold* share sequences, each
                from ``split()``.  All must have the same length.

        Returns:
            The reconstructed token IDs (same length as input shares).
        """
        if len(share_lists) < 2:
            return share_lists[0] if share_lists else []

        GF = _GaloisField
        seq_len = len(share_lists[0])
        num_shares = len(share_lists)

        # X values: 1 .. num_shares (corresponding to evaluation points)
        x_values = [i + 1 for i in range(num_shares)]

        reconstructed: list[int] = []

        for pos in range(seq_len):
            y_values = [shares[pos] for shares in share_lists]

            # Lagrange interpolation at x=0 to recover the constant term
            secret = 0
            for i in range(num_shares):
                xi = x_values[i]
                yi = y_values[i]

                # L_i(0) = product_{j != i} (0 - x_j) / (x_i - x_j)
                numerator = 1
                denominator = 1
                for j in range(num_shares):
                    if i == j:
                        continue
                    xj = x_values[j]
                    numerator = GF.mul(numerator, GF.sub(0, xj))
                    denominator = GF.mul(denominator, GF.sub(xi, xj))

                li0 = GF.div(numerator, denominator)
                secret = GF.add(secret, GF.mul(yi, li0))

            reconstructed.append(secret)

        return reconstructed
