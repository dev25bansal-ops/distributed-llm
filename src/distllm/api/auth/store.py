"""Token revocation blocklist."""

from __future__ import annotations

import time

from loguru import logger


class TokenRevocationStore:
    """In-memory token revocation blocklist.

    Maps SHA-256 token hashes to revocation timestamps.
    Entries expire after ``revocation_ttl_s`` to bound memory growth.
    """

    def __init__(self, revocation_ttl_s: float = 3600.0):
        self._revoked: dict[str, float] = {}
        self._ttl = revocation_ttl_s
        self._last_cleanup: float = time.time()

    def revoke(self, token_hash: str) -> None:
        """Revoke a token by its SHA-256 hash."""
        self._revoked[token_hash] = time.time()
        logger.info(f"Token revoked (hash prefix: {token_hash[:16]}...)")

    def is_revoked(self, token_hash: str) -> bool:
        """Return True if the token hash is in the blocklist."""
        self._cleanup()
        return token_hash in self._revoked

    def _cleanup(self) -> None:
        """Remove expired revocation entries."""
        now = time.time()
        if now - self._last_cleanup < 300:  # every 5 min
            return
        cutoff = now - self._ttl
        self._revoked = {h: ts for h, ts in self._revoked.items() if ts > cutoff}
        self._last_cleanup = now
