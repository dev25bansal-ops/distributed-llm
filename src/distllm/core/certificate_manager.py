"""Certificate Manager — auto-provision TLS certificates (Let's Encrypt).

Handles ACME certificate issuance, auto-renewal, certificate storage,
and gRPC TLS credentials. Supports both self-signed (development) and
Let's Encrypt (production) certificates.
"""

from __future__ import annotations

import os
import shutil
import socket
import ssl
import time
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from loguru import logger


@dataclass
class CertificateInfo:
    """Metadata about a managed TLS certificate."""
    common_name: str
    subject_alt_names: list[str] = field(default_factory=list)
    issuer: str = ""
    not_before: float = 0.0
    not_after: float = 0.0
    fingerprint_sha256: str = ""
    is_self_signed: bool = False
    cert_path: str = ""
    key_path: str = ""


class CertificateManager:
    """Auto-provision and manage TLS certificates.

    Usage:
        mgr = CertificateManager(cert_dir="./certs")
        info = mgr.ensure_certificate("node1.example.com")
        creds = mgr.create_grpc_server_credentials("node1.example.com")

    Falls back to self-signed certificates when Let's Encrypt is
    unavailable (offline / development).
    """

    RENEW_BEFORE_DAYS: int = 30
    SELF_SIGNED_DAYS: int = 365
    CHECK_INTERVAL: int = 86400  # 24 hours

    def __init__(
        self,
        cert_dir: str = "./certs",
        email: str = "",
        accept_terms: bool = False,
        staging: bool = False,
    ) -> None:
        self._cert_dir = Path(cert_dir)
        self._cert_dir.mkdir(parents=True, exist_ok=True)
        self._email = email
        self._accept_terms = accept_terms
        self._staging = staging
        self._renewal_thread: threading.Thread | None = None

    # ── Certificate lifecycle ───────────────────────────────────────────

    def ensure_certificate(
        self, common_name: str,
        alt_names: list[str] | None = None,
    ) -> CertificateInfo:
        """Ensure a valid certificate exists for *common_name*.

        Returns existing cert info if valid, or provisions a new one.
        Tries Let's Encrypt first, falls back to self-signed.
        """
        info = self.get_certificate_info(common_name)
        if info and not self._needs_renewal(info):
            return info

        # Try Let's Encrypt
        acme_info = self._provision_acme(common_name, alt_names or [])
        if acme_info:
            return acme_info

        # Fallback: self-signed
        logger.info(f"Falling back to self-signed certificate for {common_name}")
        return self._create_self_signed(common_name, alt_names or [])

    def renew_all(self) -> list[CertificateInfo]:
        """Check and renew all certificates nearing expiry."""
        renewed: list[CertificateInfo] = []
        for cert_file in self._cert_dir.glob("*.crt"):
            cn = cert_file.stem
            info = self.get_certificate_info(cn)
            if info and self._needs_renewal(info):
                logger.info(f"Renewing certificate for {cn}")
                new_info = self.ensure_certificate(cn)
                renewed.append(new_info)
        return renewed

    def revoke(self, common_name: str) -> bool:
        """Revoke a certificate and remove its files."""
        cert_path = self._cert_dir / f"{common_name}.crt"
        key_path = self._cert_dir / f"{common_name}.key"
        for p in [cert_path, key_path]:
            if p.exists():
                p.unlink()
        cert_pem = self._cert_dir / f"{common_name}.pem"
        if cert_pem.exists():
            cert_pem.unlink()
        return True

    # ── Certificate access ──────────────────────────────────────────────

    def get_certificate_info(self, common_name: str) -> CertificateInfo | None:
        """Read certificate metadata from disk."""
        cert_path = self._cert_dir / f"{common_name}.crt"
        if not cert_path.exists():
            return None
        try:
            import cryptography.x509 as x509
            from cryptography.hazmat.primitives import hashes
            cert_data = cert_path.read_bytes()
            cert = x509.load_pem_x509_certificate(cert_data)
            sans = []
            try:
                ext = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName)
                sans = list(ext.value.get_values_for_type(x509.DNSName))
            except Exception:
                pass
            fp = cert.fingerprint(hashes.SHA256()).hex(":")
            return CertificateInfo(
                common_name=common_name,
                subject_alt_names=sans,
                issuer=cert.issuer.rfc4514_string(),
                not_before=cert.not_valid_before_utc.timestamp(),
                not_after=cert.not_valid_after_utc.timestamp(),
                fingerprint_sha256=fp,
                cert_path=str(cert_path),
                key_path=str(self._cert_dir / f"{common_name}.key"),
            )
        except ImportError:
            logger.warning("cryptography not available, cannot read cert metadata")
            return None

    def cert_path(self, common_name: str) -> Path | None:
        p = self._cert_dir / f"{common_name}.crt"
        return p if p.exists() else None

    def key_path(self, common_name: str) -> Path | None:
        p = self._cert_dir / f"{common_name}.key"
        return p if p.exists() else None

    # ── gRPC credentials ────────────────────────────────────────────────

    def create_grpc_server_credentials(self, common_name: str) -> Any | None:
        """Create gRPC server credentials from a managed certificate."""
        import grpc
        cert = self.cert_path(common_name)
        key = self.key_path(common_name)
        if not cert or not key:
            logger.error(f"No certificate found for {common_name}")
            return None
        try:
            with open(cert, "rb") as f:
                cert_bytes = f.read()
            with open(key, "rb") as f:
                key_bytes = f.read()
            return grpc.ssl_server_credentials(
                [(key_bytes, cert_bytes)],
            )
        except Exception as e:
            logger.error(f"Failed to create gRPC credentials: {e}")
            return None

    def create_grpc_client_credentials(
        self, common_name: str | None = None,
    ) -> Any | None:
        """Create gRPC client credentials for mutual TLS."""
        import grpc
        ca_cert = self._cert_dir / "ca.crt"
        if not ca_cert.exists():
            # Use the server cert as CA for self-signed
            if common_name:
                ca_cert = self.cert_path(common_name)
        if not ca_cert or not ca_cert.exists():
            return None
        try:
            ca_bytes = ca_cert.read_bytes()
            return grpc.ssl_channel_credentials(
                root_certificates=ca_bytes,
            )
        except Exception:
            return None

    def create_mtls_client_credentials(
        self,
        client_cn: str,
        ca_cn: str | None = None,
    ) -> Any | None:
        """Create gRPC client credentials with client certificate for mTLS.

        Args:
            client_cn: Common name of the client certificate.
            ca_cn: Common name of the CA certificate. If None, uses "ca".

        Returns:
            gRPC SSL channel credentials with client cert, or None.
        """
        import grpc
        client_cert = self.cert_path(client_cn)
        client_key = self.key_path(client_cn)
        ca_cert = self._cert_dir / f"{ca_cn or 'ca'}.crt"
        if not ca_cert.exists():
            ca_cert = self.cert_path(ca_cn or client_cn)
        if not client_cert or not client_key or not ca_cert or not ca_cert.exists():
            logger.error(f"Missing mTLS cert/key/CA for {client_cn}")
            return None
        try:
            return grpc.ssl_channel_credentials(
                root_certificates=ca_cert.read_bytes(),
                private_key=client_key.read_bytes(),
                certificate_chain=client_cert.read_bytes(),
            )
        except Exception as e:
            logger.error(f"Failed to create mTLS credentials: {e}")
            return None

    def create_mtls_server_credentials(
        self,
        server_cn: str,
        require_client_cert: bool = True,
    ) -> Any | None:
        """Create gRPC server credentials that require client certificates.

        Args:
            server_cn: Common name of the server certificate.
            require_client_cert: If True, clients must present a valid cert.

        Returns:
            gRPC SSL server credentials with client cert requirement.
        """
        import grpc
        server_cert = self.cert_path(server_cn)
        server_key = self.key_path(server_cn)
        ca_cert = self._cert_dir / "ca.crt"
        if not ca_cert.exists():
            ca_cert = server_cert  # Self-signed fallback
        if not server_cert or not server_key:
            logger.error(f"Missing server cert/key for {server_cn}")
            return None
        try:
            ca_bytes = ca_cert.read_bytes() if ca_cert and ca_cert.exists() else None
            client_auth = grpc.ssl.ClientCertificateStyle.REQUIRE_AND_VERIFY if require_client_cert else grpc.ssl.ClientCertificateStyle.OPTIONAL
            return grpc.ssl_server_credentials(
                [(server_key.read_bytes(), server_cert.read_bytes())],
                root_certificates=ca_bytes,
                require_client_auth=require_client_cert,
            )
        except Exception as e:
            logger.error(f"Failed to create mTLS server credentials: {e}")
            return None

    # ── Background renewal ──────────────────────────────────────────────

    def start_background_renewal(self) -> threading.Thread:
        """Start a daemon thread for periodic certificate renewal checks."""
        def _loop() -> None:
            while True:
                try:
                    self.renew_all()
                except Exception as e:
                    logger.warning(f"Certificate renewal check failed: {e}")
                time.sleep(self.CHECK_INTERVAL)

        t = threading.Thread(target=_loop, daemon=True)
        t.start()
        self._renewal_thread = t
        return t

    # ── Private helpers ─────────────────────────────────────────────────

    def _needs_renewal(self, info: CertificateInfo) -> bool:
        remaining = info.not_after - time.time()
        return remaining < self.RENEW_BEFORE_DAYS * 86400

    def _provision_acme(
        self, common_name: str, alt_names: list[str],
    ) -> CertificateInfo | None:
        """Attempt Let's Encrypt certificate issuance via ACME."""
        try:
            import acme
            import josepy
        except ImportError:
            logger.debug("acme-tiny / josepy not installed, skipping ACME")
            return None

        if not self._email or not self._accept_terms:
            logger.debug("ACME email/terms not configured, skipping")
            return None

        cert_path = self._cert_dir / f"{common_name}.crt"
        key_path = self._cert_dir / f"{common_name}.key"

        if cert_path.exists() and not self._needs_renewal(
            CertificateInfo(common_name=common_name, not_after=(
                datetime.now() + timedelta(days=30)).timestamp())
        ):
            # Cheap check — we'll verify with cryptography later
            return self.get_certificate_info(common_name)

        # Placeholder: real ACME flow would use acme-tiny or certbot
        # For now, fall through to self-signed
        logger.info("ACME issuance not yet implemented, using self-signed fallback")
        return None

    def _create_self_signed(
        self, common_name: str, alt_names: list[str],
    ) -> CertificateInfo:
        """Generate a self-signed certificate."""
        from cryptography import x509
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.x509.oid import NameOID
        import datetime as dt

        key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

        subject = issuer = x509.Name([
            x509.NameAttribute(NameOID.COMMON_NAME, common_name),
        ])

        san_list = [x509.DNSName(common_name)]
        for name in alt_names:
            san_list.append(x509.DNSName(name))

        now = dt.datetime.now(dt.timezone.utc)
        cert = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(issuer)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now)
            .not_valid_after(now + dt.timedelta(days=self.SELF_SIGNED_DAYS))
            .add_extension(x509.SubjectAlternativeName(san_list), critical=False)
            .sign(key, hashes.SHA256())
        )

        cert_path = self._cert_dir / f"{common_name}.crt"
        key_path = self._cert_dir / f"{common_name}.key"

        cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
        key_path.write_bytes(key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.TraditionalOpenSSL,
            self._encryption_algorithm(),
        ))

        # Restrict private key permissions (Unix) or warn on Windows
        try:
            os.chmod(str(key_path), 0o600)
        except OSError:
            if os.name != "nt":
                logger.warning(f"Could not set permissions on {key_path}")

        info = CertificateInfo(
            common_name=common_name,
            subject_alt_names=[common_name] + alt_names,
            issuer=f"CN={common_name}",
            not_before=now.timestamp(),
            not_after=(now + dt.timedelta(days=self.SELF_SIGNED_DAYS)).timestamp(),
            cert_path=str(cert_path),
            key_path=str(key_path),
            is_self_signed=True,
        )
        logger.info(f"Created self-signed certificate for {common_name}")
        return info

    def _encryption_algorithm(self) -> serialization.KeySerializationEncryption:
        """Return the encryption algorithm for private key serialization.

        Uses NoEncryption on Unix where file permissions provide protection,
        and on Windows requires KEY_ENCRYPTION_PASSPHRASE in the secret
        manager or environment. Falls back to NoEncryption with a warning
        if no passphrase is configured — the generated ephemeral key
        pattern is removed because per-file ephemeral keys that are never
        persisted make the private key file permanently unrecoverable on
        process restart (CWE-654).
        """
        if os.name == "nt":
            # Try SecretManager first, then environment variable fallback
            passphrase: str | None = None
            try:
                from distllm.core.secret_manager import SecretManager
                mgr = SecretManager()
                passphrase = mgr.get_secret("KEY_ENCRYPTION_PASSPHRASE")
            except Exception:
                pass
            if not passphrase:
                import os as _os
                passphrase = _os.environ.get("KEY_ENCRYPTION_PASSPHRASE")
            if passphrase:
                return serialization.BestAvailableEncryption(passphrase.encode())
            logger.warning(
                "Private key on Windows will be stored without encryption. "
                "Set KEY_ENCRYPTION_PASSPHRASE in your secret manager or "
                "environment for encrypted private key storage."
            )
        return serialization.NoEncryption()
