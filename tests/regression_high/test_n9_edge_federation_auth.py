"""N9 — Edge federation auth hardening: mTLS + device-attestation.

Proves the edge-node policy (Dist N3) enforces, on top of the base A4 SVID /
A5 mTLS auth, an ADDITIONAL gate for browser/mobile/IoT peers:

  (a) an edge peer WITHOUT a device-attestation token is DENIED;
  (b) an edge peer WITH a valid attestation token AND an mTLS context that
      requires a client cert is ALLOWED;
  (c) a NON-edge peer is unaffected by the edge policy;
  (d) ``federation`` + ``edge_attestation`` import cleanly and the additive
      API surface exists (no breakage, N6 tracing untouched).

The policy reuses A4's ``verify_svid`` and A5's ``build_tls_context``.  It is
model-free and software-only (HMAC / SVID attestation scaffold); real TPM /
WebAuthn / AttestationDoc plugs in at the marked ``PLUGIN:`` points.  Tests
are network-free — the heartbeat HTTP POST is monkeypatched.

HONEST CAVEAT: the attestation token and mTLS context here are a SOFTWARE
SCAFFOLD (no real TPM / WebAuthn). The trust *contract* and policy decision
are what is proven; the hardware roots of trust are clearly marked.
"""

from __future__ import annotations

import os
import ssl

import pytest

from distllm.security.edge_attestation import (
    SCAFFOLD_MARKER,
    EDGE_MARKER,
    DeviceKind,
    DeviceProfile,
    EdgeAttestationPolicy,
    EdgeDecision,
)
from distllm.security.quantum_safe_tls import build_tls_context
from distllm.security.spiffe import issue_svid, PeerIdentity, DEFAULT_TRUST_DOMAIN


# ── helpers ────────────────────────────────────────────────────────────────

def _real_mtls_context():
    """Build a REAL mTLS client context (require_client_cert) via A5.

    We mint a throwaway dev-CA-signed client cert so A5 can load a cert chain,
    then assert the resulting context requires a client cert (CERT_REQUIRED).
    """
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography import x509
    from cryptography.x509.oid import NameOID
    from cryptography.hazmat.primitives import hashes, serialization
    from datetime import datetime, timedelta, timezone
    import tempfile

    from distllm.security.spiffe import _ensure_dev_ca

    ca_cert_pem, ca_key_pem = _ensure_dev_ca()
    ca_cert = x509.load_pem_x509_certificate(ca_cert_pem)
    ca_priv = serialization.load_pem_private_key(ca_key_pem, password=None)

    leaf_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    now = datetime.now(timezone.utc)
    client_cert = (
        x509.CertificateBuilder()
        .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "edge-client")]))
        .issuer_name(ca_cert.subject)
        .public_key(leaf_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=1))
        .not_valid_after(now + timedelta(days=365))
        .add_extension(
            x509.SubjectAlternativeName(
                [x509.UniformResourceIdentifier("spiffe://local/peer/edge-client")]
            ),
            critical=False,
        )
        .sign(ca_priv, hashes.SHA256())
    )
    cert_pem = client_cert.public_bytes(serialization.Encoding.PEM)
    key_pem = leaf_key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.TraditionalOpenSSL,
        serialization.NoEncryption(),
    )
    with tempfile.NamedTemporaryFile("w", suffix=".pem", delete=False) as cf, \
            tempfile.NamedTemporaryFile("w", suffix=".pem", delete=False) as kf:
        cf.write(cert_pem.decode("latin1"))
        kf.write(key_pem.decode("latin1"))
        certfile, keyfile = cf.name, kf.name

    ctx = build_tls_context(
        certfile=certfile,
        keyfile=keyfile,
        use_kyber=False,
        require_client_cert=True,
        ca_certfile=certfile,
        server_side=False,
    )
    # Sanity: the context genuinely requires a client cert (true mTLS).
    assert ctx.verify_mode == ssl.CERT_REQUIRED
    return ctx


def _make_edge_peer(edge_flag: bool, device_id="dev-1", kind="iot", secret="s3cr3t"):
    """Build a PeerInfo-like object with is_edge + edge metadata."""
    from dataclasses import dataclass

    @dataclass
    class _P:
        cluster_id: str = "edge-cluster"
        host: str = "10.0.0.9"
        port: int = 50050
        is_edge: bool = False
        region: str = ""
        metadata: dict = None

    meta = {
        "edge_device_id": device_id,
        "edge_device_kind": kind,
        "edge_secret": secret,
    }
    return _P(is_edge=edge_flag, metadata=meta)


# ── (d) import / API-surface sanity ────────────────────────────────────────

def test_imports_and_api_surface():
    """federation + edge_attestation import cleanly; the N9 API exists."""
    import distllm.dist.federation as fed  # must import without error
    from distllm.dist.federation import FederationCoordinator
    from distllm.security import edge_attestation

    # N6 tracing API must remain intact (we coexist with N6).
    assert hasattr(FederationCoordinator, "_trace_metadata")
    assert hasattr(FederationCoordinator, "extract_incoming_trace_context")

    # N9 additive API surface.
    assert hasattr(FederationCoordinator, "_edge_attestation_headers")
    assert hasattr(FederationCoordinator, "_build_edge_mtls_context")
    assert hasattr(fed, "EdgeAttestationPolicy") or hasattr(
        edge_attestation, "EdgeAttestationPolicy"
    )
    # Honest scaffold markers present in the new module.
    assert SCAFFOLD_MARKER in edge_attestation.__doc__
    # The EDGE marker constant exists (used for honest greppable marking).
    assert hasattr(edge_attestation, "EDGE_MARKER")
    assert "edge" in (edge_attestation.__doc__ or "").lower()


def test_policy_construction_defaults():
    """The policy constructs and exposes the right surface."""
    p = EdgeAttestationPolicy()
    assert p.require_mtls_for_edge() is True
    assert isinstance(p.authorize_edge_peer(_make_edge_peer(False), None),
                      EdgeDecision)


# ── (a) edge peer WITHOUT attestation token is DENIED ──────────────────────

def test_edge_peer_without_attestation_denied():
    """An edge peer with no device-attestation profile must be Denied."""
    p = EdgeAttestationPolicy()
    edge_ctx = _real_mtls_context()
    peer = _make_edge_peer(True)

    decision = p.authorize_edge_peer(
        peer, attestation=None, mtls_context=edge_ctx, require_mtls=True,
        attestation_secret="s3cr3t",
    )
    assert decision.allow is False
    assert "attestation" in decision.reason.lower()


def test_edge_peer_without_mtls_denied():
    """An edge peer without an mTLS client-cert context must be Denied."""
    p = EdgeAttestationPolicy()
    peer = _make_edge_peer(True)
    profile = DeviceProfile(device_id="dev-1", device_kind=DeviceKind.IOT)
    p.mint_attestation_token(profile, "s3cr3t")
    # No mTLS context supplied while require_mtls=True -> Deny.
    decision = p.authorize_edge_peer(
        peer, attestation=profile, mtls_context=None, require_mtls=True,
        attestation_secret="s3cr3t",
    )
    assert decision.allow is False
    assert "mtls" in decision.reason.lower()


def test_edge_peer_invalid_token_denied():
    """An edge peer with a bad/invalid attestation token must be Denied."""
    p = EdgeAttestationPolicy()
    edge_ctx = _real_mtls_context()
    peer = _make_edge_peer(True)
    # Token minted with a DIFFERENT secret than the one used to verify.
    profile = DeviceProfile(device_id="dev-1", device_kind=DeviceKind.IOT)
    p.mint_attestation_token(profile, "right-secret")
    decision = p.authorize_edge_peer(
        peer, attestation=profile, mtls_context=edge_ctx, require_mtls=True,
        attestation_secret="wrong-secret",
    )
    assert decision.allow is False
    assert "token" in decision.reason.lower()


# ── (b) edge peer WITH valid token + mTLS-required context is ALLOWED ──────

def test_edge_peer_valid_token_and_mtls_allowed():
    """Edge peer (HMAC token) + mTLS-required context -> Allowed."""
    p = EdgeAttestationPolicy()
    edge_ctx = _real_mtls_context()
    peer = _make_edge_peer(True, device_id="dev-1", secret="s3cr3t")
    profile = DeviceProfile(device_id="dev-1", device_kind=DeviceKind.IOT)
    p.mint_attestation_token(profile, "s3cr3t")
    decision = p.authorize_edge_peer(
        peer, attestation=profile, mtls_context=edge_ctx, require_mtls=True,
        attestation_secret="s3cr3t",
    )
    assert decision.allow is True
    assert decision.policy == "edge-attestation:token"


def test_edge_peer_valid_svid_attestation_allowed():
    """Edge peer presenting a valid SPIFFE SVID as attestation -> Allowed."""
    p = EdgeAttestationPolicy()
    edge_ctx = _real_mtls_context()
    peer = _make_edge_peer(True)
    svid = issue_svid(PeerIdentity(DEFAULT_TRUST_DOMAIN, "edge-device-1"))
    profile = DeviceProfile(
        device_id="edge-device-1", device_kind=DeviceKind.MOBILE,
        svid_pem=svid.cert_pem,
    )
    decision = p.authorize_edge_peer(
        peer, attestation=profile, mtls_context=edge_ctx, require_mtls=True,
    )
    assert decision.allow is True
    assert decision.policy == "edge-attestation:svid"


# ── (c) NON-edge peer is unaffected ────────────────────────────────────────

def test_non_edge_peer_not_subject_to_edge_policy():
    """A non-edge peer bypasses the edge attestation policy entirely."""
    p = EdgeAttestationPolicy()
    cluster_peer = _make_edge_peer(False)  # non-edge
    # No attestation, no mTLS context supplied — yet the decision is Allow
    # because edge policy simply does not apply to cluster peers.
    decision = p.authorize_edge_peer(
        cluster_peer, attestation=None, mtls_context=None, require_mtls=True,
    )
    assert decision.allow is True
    assert decision.policy == "edge-attestation:skip"


def test_non_edge_peer_headers_empty():
    """The federation edge-header helper emits nothing for non-edge peers."""
    from distllm.dist.federation import FederationConfig, FederationCoordinator

    cfg = FederationConfig(enabled=False, cluster_id="local")
    coord = FederationCoordinator(
        config=cfg, local_cluster_id="local", local_host="127.0.0.1",
        local_port=50050, coordinator_ref=None,
    )
    non_edge = _make_edge_peer(False)
    assert coord._edge_attestation_headers(non_edge) == {}


# ── federation-level integration (coexists with N6 tracing) ────────────────

def _make_coordinator_with_edge(edge_is_attested: bool):
    """FederationCoordinator with one edge peer (attested or not)."""
    from distllm.dist.federation import FederationConfig, FederationCoordinator
    from distllm.dist.p2p.discovery import PeerInfo

    cfg = FederationConfig(enabled=False, cluster_id="cluster-local")
    coord = FederationCoordinator(
        config=cfg, local_cluster_id="cluster-local", local_host="127.0.0.1",
        local_port=50050, coordinator_ref=None,
    )
    peer = PeerInfo(
        cluster_id="edge-remote",
        host="10.0.0.9",
        port=50050,
        is_edge=True,
        metadata={
            "edge_device_id": "dev-1",
            "edge_device_kind": "iot",
            "edge_secret": "s3cr3t",
        },
    )
    coord._peers = {"edge-remote": peer}
    coord._svid = None
    coord._get_local_load = lambda: {
        "active_requests": 0, "pending_requests": 0, "gpu_utilization": 0.0,
    }
    if edge_is_attested:
        # Attach a valid HMAC attestation profile so authorize_edge_peer passes.
        profile = DeviceProfile(device_id="dev-1", device_kind=DeviceKind.IOT)
        coord._edge_policy.mint_attestation_token(profile, "s3cr3t")
        peer._edge_attestation = profile  # type: ignore[attr-defined]
    return coord


def test_heartbeat_denies_unattested_edge_peer(monkeypatch):
    """An unattested edge peer is skipped (Denied) by the heartbeat loop."""
    coord = _make_coordinator_with_edge(edge_is_attested=False)
    posted = {"count": 0}

    def _fake_post(url, json=None, headers=None, timeout=None):
        posted["count"] += 1

        class _Resp:
            status_code = 200

            @staticmethod
            def json():
                return {"active_requests": 0, "pending_requests": 0}

        return _Resp()

    monkeypatch.setattr(coord._http_client, "post", _fake_post)
    coord._exchange_heartbeats()
    # Edge peer was DENIED before any POST (no request should be sent).
    assert posted["count"] == 0


def test_heartbeat_allows_attested_edge_peer(monkeypatch):
    """An attested edge peer is allowed and its heartbeat is sent w/ attestation headers."""
    coord = _make_coordinator_with_edge(edge_is_attested=True)
    captured = {}

    def _fake_post(url, json=None, headers=None, timeout=None):
        captured["headers"] = headers or {}

        class _Resp:
            status_code = 200

            @staticmethod
            def json():
                return {"active_requests": 0, "pending_requests": 0}

        return _Resp()

    monkeypatch.setattr(coord._http_client, "post", _fake_post)
    # Match the N6 tracing environment: a real SDK tracer provider must be set
    # so inject_trace_context actually emits a traceparent.
    from opentelemetry import trace as _otel_trace
    from opentelemetry.sdk.trace import TracerProvider
    _otel_trace.set_tracer_provider(TracerProvider())
    _tracer = _otel_trace.get_tracer("test_n9")
    with _tracer.start_as_current_span("federation-op"):
        coord._exchange_heartbeats()

    # The (patched) POST was reached -> edge peer authorized.
    assert "headers" in captured
    # N6 tracing coexistence: traceparent header still present.
    assert "traceparent" in captured["headers"]
    # N9: edge attestation headers present alongside base auth.
    assert captured["headers"].get("X-Edge-Attestation")
    assert captured["headers"].get("X-Edge-Device-Id") == "dev-1"
    assert captured["headers"].get("X-Edge-Device-Kind") == "iot"


def test_heartbeat_non_edge_peer_unaffected(monkeypatch):
    """A non-edge peer is heartbeated normally (edge policy skipped)."""
    from distllm.dist.federation import FederationConfig, FederationCoordinator
    from distllm.dist.p2p.discovery import PeerInfo

    cfg = FederationConfig(enabled=False, cluster_id="cluster-local")
    coord = FederationCoordinator(
        config=cfg, local_cluster_id="cluster-local", local_host="127.0.0.1",
        local_port=50050, coordinator_ref=None,
    )
    coord._peers = {
        "cluster-remote": PeerInfo(
            cluster_id="cluster-remote", host="10.0.0.2", port=50050,
            is_edge=False,
        )
    }
    coord._svid = None
    coord._get_local_load = lambda: {
        "active_requests": 0, "pending_requests": 0, "gpu_utilization": 0.0,
    }
    posted = {}

    def _fake_post(url, json=None, headers=None, timeout=None):
        posted["headers"] = headers or {}

        class _Resp:
            status_code = 200

            @staticmethod
            def json():
                return {"active_requests": 0, "pending_requests": 0}

        return _Resp()

    monkeypatch.setattr(coord._http_client, "post", _fake_post)
    # Match the N6 tracing environment: a real SDK tracer provider must be set
    # so inject_trace_context actually emits a traceparent.
    from opentelemetry import trace as _otel_trace
    from opentelemetry.sdk.trace import TracerProvider
    _otel_trace.set_tracer_provider(TracerProvider())
    _tracer = _otel_trace.get_tracer("test_n9")
    with _tracer.start_as_current_span("federation-op"):
        coord._exchange_heartbeats()
    # Non-edge peer heartbeated normally; no edge attestation headers.
    assert "headers" in posted
    assert "X-Edge-Attestation" not in posted["headers"]
    assert "traceparent" in posted["headers"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", "-p", "no:cacheprovider"]))
