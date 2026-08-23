"""Tests for certificate rotation: expiry detection and renewal.

Covers:
- Certificate parsing and expiry detection
- needs_renewal() returns True near expiry
- cryptography vs fallback paths
"""

from __future__ import annotations

import os
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest


@pytest.fixture
def cert_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


class TestCertRotation:
    """Certificate rotation and expiry detection."""

    def test_cert_info_expiry_detection(self, cert_dir):
        """Certificate info should correctly report missing-cert state."""
        from distllm.core.cert_rotation import CertificateRotator

        rotator = CertificateRotator(
            cert_path=os.path.join(cert_dir, "cert.pem"),
            key_path=os.path.join(cert_dir, "key.pem"),
        )
        # Before any cert exists, check_certificate reports invalid
        info = rotator.check_certificate()
        assert info.days_remaining == 0
        assert not info.is_valid  # No cert file exists

    def test_needs_renewal_no_cert(self, cert_dir):
        """needs_renewal should return True when no cert exists."""
        from distllm.core.cert_rotation import CertificateRotator

        rotator = CertificateRotator(
            cert_path=os.path.join(cert_dir, "cert.pem"),
            key_path=os.path.join(cert_dir, "key.pem"),
            renew_before_days=10,
        )
        assert rotator.needs_renewal(), "Should need renewal when no cert exists"

    def test_datetime_utcnow_replaced(self, cert_dir):
        """Verify the fix for timezone-aware datetime comparison (C2 fix).

        ``needs_renewal`` compares ``not_valid_after_utc`` (timezone-aware)
        against ``datetime.now(timezone.utc)`` — a naive utcnow() here raised
        TypeError on modern cryptography, silently disabling auto-renewal.
        """
        from distllm.core.cert_rotation import CertificateRotator

        rotator = CertificateRotator(
            cert_path=os.path.join(cert_dir, "cert.pem"),
            key_path=os.path.join(cert_dir, "key.pem"),
            renew_before_days=60,
        )
        assert rotator.generate_self_signed(hostname="localhost", days=30)
        # 30-day cert vs 60-day renewal window -> needs renewal now.
        # Must not raise TypeError from naive/aware datetime mixing.
        assert rotator.needs_renewal() is True

        long_lived = CertificateRotator(
            cert_path=os.path.join(cert_dir, "cert2.pem"),
            key_path=os.path.join(cert_dir, "key2.pem"),
            renew_before_days=30,
        )
        assert long_lived.generate_self_signed(hostname="localhost", days=365)
        assert long_lived.needs_renewal() is False
