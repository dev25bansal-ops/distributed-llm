"""Regression tests for HIGH fix C11: select() dropped NIM/Triton.

``BackendRegistry.select()`` filtered candidates via ``_check_health``, which
ran an *instance* ``health_check`` requiring a live server connection. NIM and
Triton backends report ``health_check() == False`` until connected, so they
were silently dropped even though they were import-available. ``_check_health``
now prefers a static ``is_available()`` check.
"""

from __future__ import annotations

import pytest

from distllm.backends.registry import _check_health


class _ImportAvailableButNotConnected:
    """Simulates NIM/Triton: import-available but not yet connected."""

    @classmethod
    def is_available(cls) -> bool:
        return True

    def health_check(self) -> bool:
        # Requires a live connection -> False until connected.
        return False


class _TrulyUnavailable:
    @classmethod
    def is_available(cls) -> bool:
        return False


def test_check_health_prefers_is_available():
    # Import-available (even if not connected) must be considered healthy.
    assert _check_health(_ImportAvailableButNotConnected) is True
    # Genuinely unavailable must be unhealthy.
    assert _check_health(_TrulyUnavailable) is False
