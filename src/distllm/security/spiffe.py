"""Zero-trust federation: per-peer SPIFFE/SVID verification (SOFTWARE SCAFFOLD).

SCAFFOLD MARKER: SCAFFOLD — this module is a SOFTWARE SCAFFOLD.  There is
NO real SPIRE agent / workload attestation on this machine.  SVIDs here are
ordinary X.509 certificates carrying a SPIFFE URI in the subjectAltName, signed
by a *locally generated, file-persisted DEV certificate authority* (DEV ONLY).

The point of the scaffold is to prove the trust *contract* and the integration
point:

  * ``issue_svid``   — a peer gets a UNIQUE per-peer certificate (not a shared
                        static secret).  Per-peer == zero-trust.
  * ``verify_svid``  — the receiving side validates (a) the cert chains to the
                        dev CA and (b) the SPIFFE URI belongs to the expected
                        trust domain.  This is exactly the verification a real
                        mTLS server-side handler would run against a presented
                        client cert.
  * ``extract_peer_id`` — the peer id is recovered from the SPIFFE URI so the
                        coordinator can attribute the connection to a specific
                        workload.

HONEST CAVEAT: the dev CA private key lives on disk (``.distllm/spiffe_dev_ca
*.pem``).  In production this is replaced by the SPIRE server / upstream CA,
and ``verify_svid`` is called by the mTLS listener itself (the connection's
client certificate is passed straight in).  The call sites do not change.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import NamedTuple

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.x509.oid import ExtensionOID, NameOID

# ── Honest scaffold markers (grepped by the regression tests) ────────────────
SCAFFOLD_MARKER = "SCAFFOLD"
DEV_ONLY_MARKER = "DEV ONLY"

# Per-task contract: shared static secret is the anti-pattern.  The legacy
# ``X-Cluster-Key`` path is gated behind this flag and MUST default to False.
LEGACY_CLUSTER_KEY_ENV = "DISTLLM_LEGACY_CLUSTER_KEY_ENABLED"
LEGACY_CLUSTER_KEY_DEFAULT = False

DEFAULT_TRUST_DOMAIN = "spiffe://distllm.cluster"

# Where the DEV CA is persisted (clearly dev-only artifacts).
_DEV_CA_DIR = Path(os.environ.get("DISTLLM_DEV_CA_DIR", ".distllm"))
_DEV_CA_CERT = _DEV_CA_DIR / "spiffe_dev_ca_cert.pem"
_DEV_CA_KEY = _DEV_CA_DIR / "spiffe_dev_ca_key.pem"

_lock = threading.Lock()


@dataclass
class PeerIdentity:
    """Identifies a federated peer workload by its SPIFFE identity.

    ``trust_domain`` is the SPIFFE trust domain (e.g. ``spiffe://distllm.cluster``);
    ``peer_id`` is the workload id (e.g. ``coordinator-a`` / ``worker-3``).
    The full SPIFFE ID is ``spiffe://<trust_domain-without-scheme>/peer/<peer_id>``.
    """

    trust_domain: str = DEFAULT_TRUST_DOMAIN
    peer_id: str = ""

    @property
    def spiffe_id(self) -> str:
        """Return the canonical SPIFFE ID for this peer (``spiffe://.../peer/...``)."""
        td = self.trust_domain
        if td.startswith("spiffe://"):
            td = td[len("spiffe://"):]
        td = td.rstrip("/")
        return f"spiffe://{td}/peer/{self.peer_id}"

    @classmethod
    def from_spiffe_id(cls, spiffe_id: str) -> "PeerIdentity":
        """Parse a SPIFFE ID back into a PeerIdentity (inverse of ``.spiffe_id``)."""
        if not spiffe_id.startswith("spiffe://"):
            raise ValueError(f"not a spiffe id: {spiffe_id!r}")
        rest = spiffe_id[len("spiffe://"):]
        if "/peer/" not in rest:
            raise ValueError(f"spiffe id has no /peer/ segment: {spiffe_id!r}")
        td, peer_id = rest.rsplit("/peer/", 1)
        return cls(trust_domain=f"spiffe://{td}", peer_id=peer_id)


class SVID(NamedTuple):
    """A issued SPIFFE Verifiable Identity Document (software scaffold).

    ``cert_pem`` / ``key_pem`` are PEM-encoded x509 cert + private key.
    ``spiffe_id`` is the canonical SPIFFE URI placed in the cert SAN.
    """

    cert_pem: str
    key_pem: str
    spiffe_id: str


# ── Dev CA (self-signed, file-persisted, DEV ONLY) ──────────────────────────

def _ensure_dev_ca() -> tuple[bytes, bytes]:
    """Return (ca_cert_pem, ca_key_pem), generating + persisting the DEV CA if absent.

    The CA is self-signed and stored under ``.distllm/``.  Clearly DEV ONLY —
    in production this is the SPIRE/server upstream CA, never a local file.
    """
    with _lock:
        if _DEV_CA_CERT.exists() and _DEV_CA_KEY.exists():
            return _DEV_CA_CERT.read_bytes(), _DEV_CA_KEY.read_bytes()
        _DEV_CA_DIR.mkdir(parents=True, exist_ok=True)

        key = rsa.generate_private_key(public_exponent=65537, key_size=3072)
        subject = x509.Name([
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "DistLLM-DEV"),
            x509.NameAttribute(NameOID.COMMON_NAME, "DistLLM Dev SPIFFE CA"),
        ])
        now = datetime.now(timezone.utc)
        cert = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(subject)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - timedelta(minutes=5))
            .not_valid_after(now + timedelta(days=3650))
            .add_extension(
                x509.BasicConstraints(ca=True, path_length=0), critical=True
            )
            .add_extension(
                x509.KeyUsage(
                    digital_signature=True, content_commitment=False,
                    key_encipherment=False, data_encipherment=False,
                    key_agreement=False, key_cert_sign=True, crl_sign=True,
                    encipher_only=False, decipher_only=False,
                ),
                critical=True,
            )
            .sign(key, hashes.SHA256())
        )
        cert_pem = cert.public_bytes(serialization.Encoding.PEM)
        key_pem = key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.TraditionalOpenSSL,
            serialization.NoEncryption(),
        )
        # 0600 — these are DEV ONLY secrets; restrict on disk.
        _DEV_CA_CERT.write_bytes(cert_pem)
        _DEV_CA_CERT.chmod(0o600)
        _DEV_CA_KEY.write_bytes(key_pem)
        _DEV_CA_KEY.chmod(0o600)
        return cert_pem, key_pem


def _load_dev_ca() -> tuple[x509.Certificate, rsa.RSAPrivateKey]:
    cert_pem, key_pem = _ensure_dev_ca()
    cert = x509.load_pem_x509_certificate(cert_pem)
    key = serialization.load_pem_private_key(key_pem, password=None)
    return cert, key  # type: ignore[return-value]


# ── Public API ────────────────────────────────────────────────────────────────

def issue_svid(identity: PeerIdentity, ca_key: bytes | None = None) -> SVID:
    """Issue a per-peer SVID: an x509 cert with a SPIFFE URI SAN, signed by the dev CA.

    Args:
        identity: the peer's SPIFFE identity (trust domain + peer id).
        ca_key:   optional PEM dev-CA private key.  When ``None`` the persisted
                  DEV CA is loaded/generated.

    Returns:
        SVID(cert_pem, key_pem, spiffe_id).  Each call produces a UNIQUE
        keypair + serial — i.e. no shared static secret across peers.
    """
    if not identity.peer_id:
        raise ValueError("PeerIdentity.peer_id must be non-empty")
    if ca_key is None:
        _, ca_key = _ensure_dev_ca()
    ca_cert_pem, _ = _ensure_dev_ca()
    ca_cert = x509.load_pem_x509_certificate(ca_cert_pem)
    ca_priv = serialization.load_pem_private_key(ca_key, password=None)

    leaf_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    spiffe_id = identity.spiffe_id
    now = datetime.now(timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(x509.Name([
            x509.NameAttribute(NameOID.COMMON_NAME, spiffe_id),
        ]))
        .issuer_name(ca_cert.subject)
        .public_key(leaf_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=1))
        .not_valid_after(now + timedelta(days=365))
        .add_extension(
            # The SPIFFE URI in the subjectAltName is the identity contract.
            x509.SubjectAlternativeName([x509.UniformResourceIdentifier(spiffe_id)]),
            critical=False,
        )
        .add_extension(
            x509.KeyUsage(
                digital_signature=True, content_commitment=False,
                key_encipherment=True, data_encipherment=False,
                key_agreement=False, key_cert_sign=False, crl_sign=False,
                encipher_only=False, decipher_only=False,
            ),
            critical=True,
        )
        .sign(ca_priv, hashes.SHA256())
    )
    cert_pem = cert.public_bytes(serialization.Encoding.PEM)
    key_pem = leaf_key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.TraditionalOpenSSL,
        serialization.NoEncryption(),
    )
    return SVID(cert_pem=cert_pem.decode("ascii"),
                key_pem=key_pem.decode("ascii"),
                spiffe_id=spiffe_id)


def _extract_spiffe_id(cert: x509.Certificate) -> str | None:
    try:
        san = cert.extensions.get_extension_for_oid(
            ExtensionOID.SUBJECT_ALTERNATIVE_NAME
        ).value
    except x509.ExtensionNotFound:
        return None
    for entry in san:
        if isinstance(entry, x509.UniformResourceIdentifier):
            val = entry.value
            if val.startswith("spiffe://"):
                return val
    return None


def extract_peer_id(svid_pem: str) -> str | None:
    """Recover the peer id from an SVID PEM (None if not a valid SPIFFE cert)."""
    try:
        cert = x509.load_pem_x509_certificate(svid_pem.encode("ascii"))
    except Exception:
        return None
    spiffe_id = _extract_spiffe_id(cert)
    if spiffe_id is None:
        return None
    if "/peer/" not in spiffe_id:
        return None
    return spiffe_id.rsplit("/peer/", 1)[1]


def verify_svid(svid_pem: str, trust_domain: str | None = None,
                ca_cert_pem: bytes | None = None) -> bool:
    """Verify an SVID against the dev CA and the expected trust domain.

    Contract (the verify a real mTLS server would run on a presented client
    cert):

      1. The cert parses and chains to the dev CA (signature + CA flag).
      2. A SPIFFE URI SAN is present.
      3. The SPIFFE URI's trust domain matches ``trust_domain`` (when given).
      4. The cert is currently time-valid (not expired / not yet valid).

    Args:
        svid_pem:     PEM-encoded peer certificate.
        trust_domain: expected trust domain (e.g. ``spiffe://distllm.cluster``).
                      If ``None``, any trust domain signed by the dev CA is
                      accepted (used when the caller only cares about provenance).
        ca_cert_pem:  optional PEM dev-CA cert; when ``None`` the persisted CA
                      is loaded.

    Returns:
        True iff all checks pass.
    """
    try:
        cert = x509.load_pem_x509_certificate(svid_pem.encode("ascii"))
    except Exception:
        return False

    if ca_cert_pem is None:
        ca_cert_pem, _ = _ensure_dev_ca()
    try:
        ca_cert = x509.load_pem_x509_certificate(ca_cert_pem)
    except Exception:
        return False

    # (1) chain to CA: verify the leaf signature with the CA public key.
    try:
        ca_cert.public_key().verify(  # type: ignore[attr-defined]
            cert.signature,
            cert.tbs_certificate_bytes,
            padding.PKCS1v15(),
            cert.signature_hash_algorithm,  # type: ignore[arg-type]
        )
    except Exception:
        return False

    # CA must actually be a CA (BasicConstraints).
    try:
        bc = ca_cert.extensions.get_extension_for_oid(
            ExtensionOID.BASIC_CONSTRAINTS
        ).value
        if not bc.ca:  # type: ignore[attr-defined]
            return False
    except x509.ExtensionNotFound:
        return False

    # (2) SPIFFE URI SAN present.
    spiffe_id = _extract_spiffe_id(cert)
    if spiffe_id is None:
        return False

    # (3) trust domain match.
    if trust_domain is not None:
        expected_td = trust_domain
        if expected_td.startswith("spiffe://"):
            expected_td = expected_td[len("spiffe://"):]
        expected_td = expected_td.rstrip("/")
        # The cert's trust domain is everything before /peer/.
        if "/peer/" not in spiffe_id:
            return False
        cert_td = spiffe_id[len("spiffe://"):].rsplit("/peer/", 1)[0]
        if cert_td.rstrip("/") != expected_td:
            return False

    # (4) time validity.
    now = datetime.now(timezone.utc)
    if now < cert.not_valid_before_utc or now > cert.not_valid_after_utc:
        return False

    return True


def legacy_cluster_key_enabled() -> bool:
    """Return whether the legacy static ``X-Cluster-Key`` path is enabled.

    DEFAULT FALSE — zero-trust SVID is the default federation auth.  Flip via
    ``DISTLLM_LEGACY_CLUSTER_KEY_ENABLED=1`` only for migration.
    """
    return os.environ.get(LEGACY_CLUSTER_KEY_ENV, "").lower() in ("1", "true", "yes")
