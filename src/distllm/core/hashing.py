"""Stable, cross-process hashing helpers.

The builtin ``hash()`` is salted per Python process (PYTHONHASHSEED), so it
must NEVER be used for distributed identity, A/B bucketing, seeded-eval
determinism, or any value that must match across nodes/restarts. Use the
helpers here instead.
"""

from __future__ import annotations

import hashlib


def stable_hash(*parts: str, digest: int = 8) -> int:
    """Return a stable, non-salted 32-bit-ish unsigned int for *parts*.

    ``digest`` is the number of hex chars taken from the SHA-256 (max 64).
    Suitable for MinHash signatures, A/B bucketing, and seed derivation.
    """
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8"))
    # Interpret the first 8 hex chars as an unsigned int (mod 2**32).
    return int(h.hexdigest()[: max(1, min(64, digest))], 16)


def stable_seed(*parts: str) -> int:
    """Return a stable int usable as a ``random.seed`` (full 32-bit range)."""
    return stable_hash(*parts, digest=8) & 0xFFFFFFFF
