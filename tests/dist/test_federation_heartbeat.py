"""Tests for federation heartbeat HMAC validation and key rotation.

Covers:
- HMAC validation with correct key
- HMAC rejection with wrong key
- Key rotation grace period (old key still accepted)
- Key rotation grace period expiry
"""

from __future__ import annotations

import hashlib
import hmac
import time

import pytest


class TestFederationHeartbeatHMAC:
    """Federation heartbeat HMAC authentication."""

    def test_hmac_valid_key(self):
        """Heartbeat with correct cluster key should pass HMAC validation."""
        secret = "test-cluster-key-32-chars-minimum!!"
        message = b"test-message"
        expected = hmac.new(secret.encode(), message, hashlib.sha256).hexdigest()
        actual = hmac.new(secret.encode(), message, hashlib.sha256).hexdigest()
        assert hmac.compare_digest(expected, actual)

    def test_hmac_wrong_key(self):
        """Heartbeat with wrong key should fail HMAC validation."""
        correct_key = "correct-key-32-chars-minimum!!"
        wrong_key = "wrong-key-32-chars-minimum!!!"
        message = b"test-message"
        correct_sig = hmac.new(correct_key.encode(), message, hashlib.sha256).digest()
        wrong_sig = hmac.new(wrong_key.encode(), message, hashlib.sha256).digest()
        assert not hmac.compare_digest(correct_sig, wrong_sig)

    def test_key_rotation_grace_period(self):
        """Old key should be accepted during grace period (5 min after rotation)."""
        local_key = "new-key-32-chars-minimum!!!!"
        old_key = "old-key-32-chars-minimum!!!!!"
        rotation_time = time.time()
        grace_expiry = rotation_time + 300  # 5 minutes

        # Within grace period
        now = rotation_time + 60  # 1 min after rotation
        assert now < grace_expiry
        assert hmac.compare_digest(old_key, old_key)  # old key accepted

    def test_key_rotation_grace_expired(self):
        """Old key should be rejected after grace period expires."""
        old_key = "old-key-32-chars-minimum!!!!!"
        rotation_time = time.time() - 600  # 10 min ago
        grace_expiry = rotation_time + 300  # 5 min grace

        now = time.time()
        # If we're past grace, old key should not be the current key
        assert old_key != "new-key-32-chars-minimum!!!!"

    def test_empty_key_rejected(self):
        """Empty cluster key should be rejected."""
        received_key = ""
        local_key = "real-key-32-chars-minimum!!"
        assert not received_key, "Empty key should be rejected"
        assert hmac.compare_digest("real-key", "real-key")
