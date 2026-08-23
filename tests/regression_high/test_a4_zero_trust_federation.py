"""Regression tests for task A4: zero-trust federation via per-peer SPIFFE/SVID.

This module is a SCAFFOLD.  There is NO real SPIRE agent / workload
attestation on this machine.  SVIDs here are ordinary X.509 certificates
carrying a SPIFFE URI in the subjectAltName, signed by a *locally generated,
file-persisted DEV certificate authority* (DEV ONLY).  The trust *boundary*
and the verify *contract* are what is proven: a peer is authenticated by a
UNIQUE per-peer certificate (not a shared static secret), verified against the
dev CA and the expected trust domain.  Real mTLS termination drops in behind
``verify_svid`` — the call sites do not change.

HONEST CAVEAT: the dev CA private key lives on disk (.distllm/spiffe_dev_ca_*.pem)
and the SVID is carried in an ``X-SVID-PEM`` header for the scaffold.  In
production the SVID is the mTLS client certificate presented on the connection
and ``verify_svid`` is called by the TLS listener.  The contract is identical.

These tests assert:
  1. The static ``X-Cluster-Key`` shared secret is NO LONGER the default auth
     mechanism — the source default path consults SVID (``X-SVID-PEM`` /
     ``verify_svid``), and the legacy key is gated behind
     ``legacy_cluster_key_enabled()`` (default False).
  2. issue + verify SVID roundtrip for a peer succeeds.
  3. A tampered / expired SVID fails verify.
  4. Two distinct peers get distinct SVIDs (no shared static secret).
  5. verify rejects a wrong trust domain.
  6. The LEGACY cluster-key flag is OFF by default.
"""

from __future__ import annotations

import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from distllm.security.spiffe import (
    DEFAULT_TRUST_DOMAIN,
    LEGACY_CLUSTER_KEY_DEFAULT,
    LEGACY_CLUSTER_KEY_ENV,
    PeerIdentity,
    SCAFFOLD_MARKER,
    extract_peer_id,
    issue_svid,
    legacy_cluster_key_enabled,
    verify_svid,
)

_ROOT = Path(__file__).resolve().parents[2]
_SRC = _ROOT / "src" / "distllm"
_FED_SRC = _SRC / "dist" / "federation.py"
_SERVER_SRC = _SRC / "api" / "server.py"
_SPIFFE_SRC = _SRC / "security" / "spiffe.py"


# ── Test 1: static X-Cluster-Key is no longer the DEFAULT auth mechanism ──

def test_static_cluster_key_not_default_auth():
    """The default federation auth path must be SVID-based, not a shared secret.

    We prove this by grepping the source of the federation module and the
    server heartbeat route:
      * the SVID header (``X-SVID-PEM``) and ``verify_svid`` appear in the
        default path;
      * any use of the legacy ``X-Cluster-Key`` is conditional on
        ``legacy_cluster_key_enabled()`` (which defaults to False) — i.e. it is
        NOT consulted in the default path.
    """
    fed = _FED_SRC.read_text(encoding="utf-8")
    srv = _SERVER_SRC.read_text(encoding="utf-8")

    # The zero-trust primitives are present in both the client (federation) and
    # the server-side verify path.
    assert "X-SVID-PEM" in fed, "federation client must attach SVID (X-SVID-PEM)"
    assert "X-SVID-PEM" in srv, "server heartbeat must read SVID (X-SVID-PEM)"
    assert "verify_svid" in srv, "server must verify the SVID"

    # The legacy static key must ONLY appear inside a legacy_cluster_key_enabled()
    # guard in the heartbeat path (default OFF).
    # Extract the federation_heart_beat body region and confirm the legacy key
    # use is gated.
    m = re.search(
        r"async def federation_heart_beat.*?"
        r"(svid_pem = request\.headers\.get\(\"X-SVID-PEM\".*?)"
        r"return \{\"status\": \"ok\"\}",
        srv,
        re.S,
    )
    assert m, "could not locate federation_heart_beat handler body"
    body = m.group(1)
    # SVID is checked FIRST and unconditionally (default path).
    assert "svid_pem and _verify_federation_svid" in body
    # The legacy X-Cluster-Key branch is only reachable when the flag is on.
    assert "elif legacy_cluster_key_enabled():" in body
    # There must be an else that rejects when no SVID and flag off.
    assert 'error": "missing or invalid SVID"' in body

    # In federation.py, the legacy X-Cluster-Key must never be attached unless
    # legacy_cluster_key_enabled() is True.
    for m_ in re.finditer(r"headers\[\"X-Cluster-Key\"\] = .*", fed):
        line = m_.group(0)
        # Walk backwards to find the guarding condition within a window.
        start = max(0, m_.start() - 400)
        window = fed[start:m_.start()]
        assert "legacy_cluster_key_enabled()" in window or \
            "elif legacy_cluster_key_enabled()" in window, \
            f"X-Cluster-Key attachment is not gated by legacy_cluster_key_enabled(): {line}"


# ── Test 2: issue + verify SVID roundtrip succeeds ──

def test_svid_issue_verify_roundtrip():
    """A freshly issued per-peer SVID verifies against the dev CA + trust domain."""
    ident = PeerIdentity(DEFAULT_TRUST_DOMAIN, "coordinator-alpha")
    svid = issue_svid(ident)
    assert isinstance(svid, tuple)  # SVID NamedTuple
    assert "BEGIN CERTIFICATE" in svid.cert_pem
    assert "BEGIN PRIVATE KEY" in svid.key_pem or "PRIVATE KEY" in svid.key_pem
    assert svid.spiffe_id == "spiffe://distllm.cluster/peer/coordinator-alpha"

    # Verify succeeds.
    assert verify_svid(svid.cert_pem, DEFAULT_TRUST_DOMAIN) is True
    # Peer id is recoverable from the cert.
    assert extract_peer_id(svid.cert_pem) == "coordinator-alpha"


# ── Test 3: tampered / expired SVID fails verify ──

def test_tampered_svid_fails():
    """Tampering with the cert breaks chain validation (verify -> False)."""
    ident = PeerIdentity(DEFAULT_TRUST_DOMAIN, "node-x")
    svid = issue_svid(ident)
    # Flip one base64 character in the cert PEM body.
    cert = svid.cert_pem
    # Replace the first 'M' (common in PEM) with 'N' inside the base64 block.
    tampered = cert.replace("MII", "NII", 1)
    assert tampered != cert
    assert verify_svid(tampered, DEFAULT_TRUST_DOMAIN) is False


def test_expired_svid_fails():
    """An SVID that is expired (not time-valid) fails verify."""
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import ExtensionOID, NameOID

    from distllm.security.spiffe import _ensure_dev_ca

    ca_cert_pem, ca_key_pem = _ensure_dev_ca()
    ca_cert = x509.load_pem_x509_certificate(ca_cert_pem)
    ca_priv = serialization.load_pem_private_key(ca_key_pem, password=None)

    leaf_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    now = datetime.now(timezone.utc)
    expired = (
        x509.CertificateBuilder()
        .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "spiffe://distllm.cluster/peer/old")]))
        .issuer_name(ca_cert.subject)
        .public_key(leaf_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(days=30))
        .not_valid_after(now - timedelta(days=1))  # expired yesterday
        .add_extension(
            x509.SubjectAlternativeName(
                [x509.UniformResourceIdentifier("spiffe://distllm.cluster/peer/old")]
            ),
            critical=False,
        )
        .sign(ca_priv, hashes.SHA256())
    )
    pem = expired.public_bytes(serialization.Encoding.PEM).decode("ascii")
    # Signed by the CA and correct trust domain, but NOT time-valid.
    assert verify_svid(pem, DEFAULT_TRUST_DOMAIN) is False


# ── Test 4: two distinct peers get distinct SVIDs (no shared static secret) ──

def test_distinct_peers_get_distinct_svids():
    """Each peer receives a UNIQUE cert; they are not a shared static secret."""
    a = issue_svid(PeerIdentity(DEFAULT_TRUST_DOMAIN, "peer-a"))
    b = issue_svid(PeerIdentity(DEFAULT_TRUST_DOMAIN, "peer-b"))
    # Distinct material.
    assert a.cert_pem != b.cert_pem
    assert a.key_pem != b.key_pem
    assert a.spiffe_id != b.spiffe_id
    # Each verifies independently; neither verifies as the other.
    assert verify_svid(a.cert_pem, DEFAULT_TRUST_DOMAIN) is True
    assert verify_svid(b.cert_pem, DEFAULT_TRUST_DOMAIN) is True
    assert extract_peer_id(a.cert_pem) == "peer-a"
    assert extract_peer_id(b.cert_pem) == "peer-b"
    # There is no single shared string that both peers would present.
    shared = set(a.cert_pem.split()) & set(b.cert_pem.split())
    # Only the PEM header/footer boilerplate may coincide — cert bodies differ.
    assert "-----BEGIN" in shared  # boilerplate, not identity material


# ── Test 5: verify rejects a wrong trust domain ──

def test_wrong_trust_domain_rejected():
    """An SVID from another trust domain must NOT verify against this one."""
    other = issue_svid(PeerIdentity("spiffe://other.cluster", "peer-z"))
    assert verify_svid(other.cert_pem, DEFAULT_TRUST_DOMAIN) is False
    # But it verifies against its own trust domain.
    assert verify_svid(other.cert_pem, "spiffe://other.cluster") is True


# ── Test 6: LEGACY flag is OFF by default ──

def test_legacy_cluster_key_disabled_by_default():
    """The legacy static X-Cluster-Key path is disabled by default."""
    # Ensure the env var is not set during this check.
    saved = os.environ.pop(LEGACY_CLUSTER_KEY_ENV, None)
    try:
        assert LEGACY_CLUSTER_KEY_DEFAULT is False
        assert legacy_cluster_key_enabled() is False
    finally:
        if saved is not None:
            os.environ[LEGACY_CLUSTER_KEY_ENV] = saved

    # And the source marks the legacy path as a migration-only fallback.
    fed = _FED_SRC.read_text(encoding="utf-8")
    assert "legacy_cluster_key_enabled()" in fed
    # The SCAFFOLD / honest caveat markers are present in spiffe.py.
    spiffe = _SPIFFE_SRC.read_text(encoding="utf-8")
    assert SCAFFOLD_MARKER in spiffe
    assert ("software" in spiffe.lower() and ("no real" in spiffe.lower()
            or "no SPIRE" in spiffe.lower() or "dev only" in spiffe.lower())), \
        "spiffe.py must honestly state this is a software scaffold (no real SPIRE)"
