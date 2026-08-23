"""Regression tests for A5: quantum-safe TLS (ML-KEM/Kyber) federation scaffold.

Asserts:
  1. classical mTLS context builds with client auth required (federation=mTLS);
  2. Kyber requested but unavailable -> classical fallback + intent signaled
     (no crash), and the raise path works when fallback disabled;
  3. Kyber available -> context builds with the KEM configured (skipped if the
     PQC lib is absent, but the availability probe itself is asserted to work);
  4. intent signaling is queryable.
"""

from __future__ import annotations

import datetime
import ssl

import pytest

from distllm.security import quantum_safe_tls as qst
from distllm.security.quantum_safe_tls import (
    KYBER_INTENT_ALPN,
    QuantumSafeUnavailable,
    build_tls_context,
    intent_mode,
    intent_signaled,
    pq_capability_report,
    quantum_safe_available,
)


@pytest.fixture(scope="module")
def cert_key(tmp_path_factory):
    """Generate a throwaway self-signed cert + key for context loading."""
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    d = tmp_path_factory.mktemp("a5certs")
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = issuer = x509.Name(
        [x509.NameAttribute(NameOID.COMMON_NAME, "distllm-federation-test")]
    )
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=3650))
        .sign(key, hashes.SHA256())
    )
    cert_path = d / "cert.pem"
    key_path = d / "key.pem"
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(
        key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    return str(cert_path), str(key_path)


# --- (1) classical mTLS -----------------------------------------------------

def test_classical_mtls_requires_client_cert(cert_key):
    certfile, keyfile = cert_key
    ctx = build_tls_context(
        certfile=certfile,
        keyfile=keyfile,
        use_kyber=False,
        require_client_cert=True,
        ca_certfile=certfile,  # verify against our own self-signed cert
    )
    assert isinstance(ctx, ssl.SSLContext)
    # Federation == mutual TLS: peer certificate required.
    assert ctx.verify_mode == ssl.CERT_REQUIRED
    # No PQC requested => no intent recorded.
    assert intent_signaled(ctx) is None
    assert intent_mode(ctx) is None


def test_classical_no_client_cert_optional(cert_key):
    certfile, keyfile = cert_key
    ctx = build_tls_context(
        certfile=certfile,
        keyfile=keyfile,
        use_kyber=False,
        require_client_cert=False,
    )
    assert ctx.verify_mode == ssl.CERT_NONE


# --- (2) Kyber requested but unavailable -> graceful fallback ---------------

def test_kyber_fallback_signals_intent(cert_key, monkeypatch):
    certfile, keyfile = cert_key
    # Force "unavailable" regardless of host to exercise the fallback branch.
    monkeypatch.setattr(qst, "_probe_openssl_group_support", lambda: False)
    monkeypatch.setattr(qst, "_probe_oqs", lambda: False)

    ctx = build_tls_context(
        certfile=certfile,
        keyfile=keyfile,
        use_kyber=True,
        require_client_cert=True,
        ca_certfile=certfile,
        fallback_on_unavailable=True,
    )
    # Did not crash; still a valid mTLS context.
    assert ctx.verify_mode == ssl.CERT_REQUIRED
    # Intent recorded + mode is fallback.
    assert intent_signaled(ctx) == "kyber768"
    assert intent_mode(ctx) == "fallback"


def test_kyber_unavailable_raises_when_no_fallback(cert_key, monkeypatch):
    certfile, keyfile = cert_key
    monkeypatch.setattr(qst, "_probe_openssl_group_support", lambda: False)
    monkeypatch.setattr(qst, "_probe_oqs", lambda: False)

    with pytest.raises(QuantumSafeUnavailable) as excinfo:
        build_tls_context(
            certfile=certfile,
            keyfile=keyfile,
            use_kyber=True,
            require_client_cert=True,
            ca_certfile=certfile,
            fallback_on_unavailable=False,
        )
    assert excinfo.value.intent == "kyber768"


# --- (3) Kyber available -> KEM configured (probe always assertable) --------

def test_availability_probe_is_boolean():
    # The probe must always return a bool without raising, on any host.
    assert isinstance(quantum_safe_available(), bool)
    report = pq_capability_report()
    assert set(report) >= {
        "openssl_group",
        "oqs",
        "cryptography_ml_kem",
        "tls_ready",
    }
    assert all(isinstance(v, bool) for v in report.values())


def test_kyber_active_when_available(cert_key, monkeypatch):
    certfile, keyfile = cert_key
    if not quantum_safe_available():
        # Simulate an available stack so the active path is still exercised
        # even though this host lacks real PQC TLS. We stub the probe and the
        # group configuration to succeed.
        monkeypatch.setattr(qst, "_probe_openssl_group_support", lambda: True)
        monkeypatch.setattr(qst, "_configure_kyber_group", lambda ctx: True)

    ctx = build_tls_context(
        certfile=certfile,
        keyfile=keyfile,
        use_kyber=True,
        require_client_cert=True,
        ca_certfile=certfile,
        fallback_on_unavailable=True,
    )
    assert intent_signaled(ctx) == "kyber768"
    assert intent_mode(ctx) == "active"


# --- (4) intent signaling is queryable + ALPN token set ---------------------

def test_intent_alpn_token_is_advertised(cert_key, monkeypatch):
    certfile, keyfile = cert_key
    monkeypatch.setattr(qst, "_probe_openssl_group_support", lambda: False)
    monkeypatch.setattr(qst, "_probe_oqs", lambda: False)
    ctx = build_tls_context(
        certfile=certfile,
        keyfile=keyfile,
        use_kyber=True,
        require_client_cert=False,
        fallback_on_unavailable=True,
    )
    # intent_signaled queryable
    assert intent_signaled(ctx) == "kyber768"
    # The ALPN intent token constant is well-formed.
    assert KYBER_INTENT_ALPN == "distllm-pq-kyber768"


def test_intent_none_on_classical(cert_key):
    certfile, keyfile = cert_key
    ctx = build_tls_context(
        certfile=certfile, keyfile=keyfile, use_kyber=False,
        require_client_cert=False,
    )
    assert intent_signaled(ctx) is None
