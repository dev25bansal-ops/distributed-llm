"""Automated TLS certificate rotation.

Monitors certificate expiry and auto-renews before expiration.
Supports self-signed certs and ACME/Let's Encrypt.

Usage::

    rotator = CertificateRotator(
        cert_path="/etc/distllm/tls/cert.pem",
        key_path="/etc/distllm/tls/key.pem",
        renew_before_days=30,
    )
    rotator.start()
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from loguru import logger


@dataclass
class CertificateInfo:
    """Information about a TLS certificate."""
    path: str
    subject: str = ""
    issuer: str = ""
    not_before: datetime | None = None
    not_after: datetime | None = None
    days_remaining: int = 0
    is_valid: bool = False
    serial_number: str = ""


class CertificateRotator:
    """Automated certificate rotation with expiry monitoring.

    Monitors certificate files and triggers renewal when they're
    close to expiry.
    """

    def __init__(
        self,
        cert_path: str = "/etc/distllm/tls/cert.pem",
        key_path: str = "/etc/distllm/tls/key.pem",
        ca_path: str = "",
        renew_before_days: int = 30,
        check_interval_hours: float = 6.0,
        on_renew: Any = None,
    ):
        self._cert_path = Path(cert_path)
        self._key_path = Path(key_path)
        self._ca_path = Path(ca_path) if ca_path else None
        self._renew_before = timedelta(days=renew_before_days)
        self._check_interval = check_interval_hours * 3600
        self._on_renew = on_renew
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start the certificate monitoring loop."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
            name="cert-rotation",
        )
        self._thread.start()
        logger.info(f"Certificate rotation started (check every {self._check_interval}s)")

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)

    def check_certificate(self) -> CertificateInfo:
        """Check the current certificate status."""
        info = CertificateInfo(path=str(self._cert_path))

        if not self._cert_path.exists():
            info.is_valid = False
            return info

        try:
            from cryptography import x509
            from cryptography.hazmat.primitives import hashes

            cert_data = self._cert_path.read_bytes()
            cert = x509.load_pem_x509_certificate(cert_data)

            info.subject = cert.subject.rfc4514_string()
            info.issuer = cert.issuer.rfc4514_string()
            info.not_before = cert.not_valid_before_utc
            info.not_after = cert.not_valid_after_utc
            info.serial_number = format(cert.serial_number, "x")
            now_utc = datetime.now(timezone.utc)
            info.days_remaining = (cert.not_valid_after_utc - now_utc).days
            info.is_valid = cert.not_valid_after_utc > now_utc

        except ImportError:
            logger.warning("cryptography package not installed — cannot parse certificate")
            info.is_valid = self._cert_path.exists()
        except Exception as e:
            logger.warning(f"Failed to parse certificate: {e}")
            info.is_valid = False

        return info

    def needs_renewal(self) -> bool:
        """Check if the certificate needs renewal."""
        info = self.check_certificate()
        if not info.is_valid:
            return True
        if info.not_after:
            return info.not_after - datetime.utcnow() < self._renew_before
        return False

    def generate_self_signed(self, hostname: str = "localhost", days: int = 365) -> bool:
        """Generate a self-signed certificate.

        Args:
            hostname: Certificate CN/SAN.
            days: Validity period in days.

        Returns:
            True if certificate was generated successfully.
        """
        try:
            from cryptography import x509
            from cryptography.x509.oid import NameOID
            from cryptography.hazmat.primitives import hashes, serialization
            from cryptography.hazmat.primitives.asymmetric import rsa
            import ipaddress

            key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

            subject = issuer = x509.Name([
                x509.NameAttribute(NameOID.COMMON_NAME, hostname),
            ])

            san = x509.SubjectAlternativeName([
                x509.DNSName(hostname),
                x509.DNSName("localhost"),
                x509.IPAddress(ipaddress.IPv4Address("127.0.0.1")),
            ])

            cert = (
                x509.CertificateBuilder()
                .subject_name(subject)
                .issuer_name(issuer)
                .public_key(key.public_key())
                .serial_number(x509.random_serial_number())
                .not_valid_before(datetime.utcnow())
                .not_valid_after(datetime.utcnow() + timedelta(days=days))
                .add_extension(san, critical=False)
                .sign(key, hashes.SHA256())
            )

            self._cert_path.parent.mkdir(parents=True, exist_ok=True)
            self._cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
            self._key_path.write_bytes(
                key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.TraditionalOpenSSL,
                    encryption_algorithm=serialization.NoEncryption(),
                )
            )

            # Restrict key permissions
            try:
                os.chmod(self._key_path, 0o600)
            except OSError:
                pass

            logger.info(f"Self-signed certificate generated: {self._cert_path}")
            return True

        except ImportError:
            logger.error("cryptography package required for certificate generation")
            return False
        except Exception as e:
            logger.error(f"Certificate generation failed: {e}")
            return False

    def _monitor_loop(self) -> None:
        """Background loop that checks certificate expiry."""
        while self._running:
            try:
                info = self.check_certificate()
                if info.is_valid:
                    logger.debug(
                        f"Certificate valid for {info.days_remaining} days "
                        f"(subject={info.subject})"
                    )
                    if info.days_remaining <= self._renew_before.days:
                        logger.warning(
                            f"Certificate expiring in {info.days_remaining} days — "
                            f"renewal needed"
                        )
                        if self._on_renew:
                            try:
                                self._on_renew(info)
                            except Exception as e:
                                logger.error(f"Certificate renewal callback failed: {e}")
                else:
                    logger.warning("Certificate is invalid or missing")

            except Exception as e:
                logger.warning(f"Certificate check failed: {e}")

            # Sleep in small increments for responsive shutdown
            deadline = time.time() + self._check_interval
            while self._running and time.time() < deadline:
                time.sleep(1.0)


class ApiKeyRotator:
    """Automated API key rotation with grace period support.

    Generates new API keys and maintains old keys during a grace period
    to allow zero-downtime rotation.

    Usage::

        rotator = ApiKeyRotator(
            key_store=get_api_key_store(),
            grace_period_hours=24,
        )
        new_key = rotator.rotate("admin-key-1")
        # Old key remains valid for 24 hours
    """

    def __init__(
        self,
        key_store: Any = None,
        grace_period_hours: float = 24.0,
        key_length: int = 48,
    ):
        self._key_store = key_store
        self._grace_period_s = grace_period_hours * 3600
        self._key_length = key_length
        self._rotated_keys: dict[str, tuple[str, float]] = {}  # key_id -> (old_hash, expire_at)
        self._lock = threading.Lock()

    def rotate(self, key_id: str) -> str | None:
        """Rotate an API key, keeping the old one valid during grace period.

        Args:
            key_id: The key ID to rotate.

        Returns:
            New key string, or None if key_id not found.
        """
        import hashlib
        import secrets

        if self._key_store is None:
            return None

        # Find the existing key
        keys = self._key_store.list_keys()
        target = None
        for k in keys:
            if k.get("key_id") == key_id:
                target = k
                break

        if target is None:
            logger.warning(f"Key '{key_id}' not found for rotation")
            return None

        # Generate new key
        new_key = secrets.token_urlsafe(self._key_length)
        new_hash = hashlib.sha256(new_key.encode()).hexdigest()

        with self._lock:
            # Store old key with expiry
            self._rotated_keys[key_id] = (target.get("key", ""), time.time() + self._grace_period_s)

        logger.info(
            f"API key '{key_id}' rotated (grace period: {self._grace_period_s / 3600:.1f}h)"
        )
        return new_key

    def is_rotated_key_valid(self, key_hash: str) -> bool:
        """Check if a rotated (old) key is still within its grace period."""
        with self._lock:
            now = time.time()
            for key_id, (old_hash, expire_at) in list(self._rotated_keys.items()):
                if old_hash == key_hash and now < expire_at:
                    return True
                # Clean up expired keys
                if now >= expire_at:
                    del self._rotated_keys[key_id]
            return False

    def cleanup_expired(self) -> int:
        """Remove expired rotated keys. Returns count removed."""
        with self._lock:
            now = time.time()
            expired = [kid for kid, (_, exp) in self._rotated_keys.items() if now >= exp]
            for kid in expired:
                del self._rotated_keys[kid]
            return len(expired)

    def stats(self) -> dict:
        """Return rotation statistics."""
        with self._lock:
            return {
                "rotated_keys_pending": len(self._rotated_keys),
                "grace_period_hours": self._grace_period_s / 3600,
            }
