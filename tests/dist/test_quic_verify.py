"""Regression tests: QUIC client must verify peer certificates when a CA is set.

P0 finding: ``dist/p2p/quic_transport.py`` set ``verify_mode = ssl.CERT_NONE`` on
every outgoing QUIC connection, so an on-path attacker impersonating a node with
a self-signed cert was fully MITM-able.  The fix verifies the peer against a
configured CA (``DISTLLM_QUIC_CA`` / ``ca_file``) with ``CERT_REQUIRED``.

These tests require aioquic (the real transport); they skip when it is absent.
"""

from __future__ import annotations

import os
import ssl

import pytest

aioquic = pytest.importorskip("aioquic")

from cryptography import x509  # noqa: E402
from cryptography.hazmat.primitives import hashes, serialization  # noqa: E402
from cryptography.hazmat.primitives.asymmetric import rsa  # noqa: E402
from cryptography.x509.oid import NameOID  # noqa: E402

from distllm.dist.p2p.quic_transport import QuicTransport  # noqa: E402


@pytest.fixture
def ca_file(tmp_path):
    """A minimal self-signed CA PEM used as the trust anchor."""
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "distllm-test-ca")])
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(__import__("datetime").datetime.now(__import__("datetime").timezone.utc))
        .not_valid_after(
            __import__("datetime").datetime.now(__import__("datetime").timezone.utc)
            + __import__("datetime").timedelta(days=1)
        )
        .add_extension(
            x509.BasicConstraints(ca=True, path_length=None), critical=True
        )
        .sign(key, hashes.SHA256())
    )
    path = tmp_path / "ca.pem"
    path.write_bytes(
        cert.public_bytes(serialization.Encoding.PEM)
    )
    return str(path)


class TestQuicPeerVerification:
    def test_client_verifies_peer_when_ca_configured(self, ca_file):
        transport = QuicTransport(ca_file=ca_file)
        config = transport._build_config(is_client=True)
        assert config.verify_mode == ssl.CERT_REQUIRED

    def test_client_skips_verification_without_ca(self):
        transport = QuicTransport()
        config = transport._build_config(is_client=True)
        assert config.verify_mode == ssl.CERT_NONE

    def test_ca_from_environment(self, ca_file, monkeypatch):
        monkeypatch.setenv("DISTLLM_QUIC_CA", ca_file)
        transport = QuicTransport()
        assert transport._ca_file == ca_file
        assert transport._build_config(is_client=True).verify_mode == ssl.CERT_REQUIRED

    def test_server_config_not_affected_by_ca(self, ca_file):
        # Server side still serves its own cert chain; client verification is
        # a client-side concern.
        transport = QuicTransport(ca_file=ca_file)
        assert transport._build_config(is_client=False).verify_mode == ssl.CERT_NONE
