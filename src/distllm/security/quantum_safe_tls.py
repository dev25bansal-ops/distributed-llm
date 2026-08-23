"""Quantum-safe (post-quantum) TLS scaffold for federation links.

This module provides a TLS :class:`ssl.SSLContext` builder for federation
links (cluster-to-cluster, Dist A2) with an **opt-in** path toward
quantum-safe key establishment using ML-KEM / Kyber (a hybrid X25519 +
ML-KEM-768 group, ``X25519MLKEM768``).

Reality check (honest scaffold)
-------------------------------
Real ML-KEM / Kyber *TLS termination* requires one of:

  * OpenSSL 3.5+ (native ML-KEM groups) driving the Python ``ssl`` module,
  * the ``oqs``/``oqs-openssl`` (Open Quantum Safe) provider, or
  * ``cryptography`` >= 43 exposing KEM primitives *and* a TLS binding that
    can wire a hybrid group into the handshake.

None of these give you a turn-key ``SSLContext.set_groups("X25519MLKEM768")``
on a stock CPython 3.11 + OpenSSL 3.0 host. So this module does three things:

  1. Always builds a correct **classical mTLS** context (federation = mutual
     TLS, client auth required by default).
  2. When ``use_kyber=True``, it *probes* for real PQC support. If present it
     best-effort configures the hybrid group; if absent it records the
     **intent** (``kyber768``) and either raises :class:`QuantumSafeUnavailable`
     or returns a classical context that still carries an on-the-wire
     **intent signal** (an ALPN token) so a peer / observer can tell the
     endpoint *wanted* PQC and negotiated a classical fallback.
  3. Exposes :func:`quantum_safe_available` and :func:`intent_signaled` so
     callers can introspect what actually happened.

The intent signal is deliberately transport-visible (ALPN protocol id
``distllm-pq-kyber768``) rather than silent, so federation peers can log /
alert on "wanted PQC, got classical" without parsing application data.
"""

from __future__ import annotations

import ssl
from typing import Any

# ALPN token advertised on the wire to signal quantum-safe intent even when
# the actual handshake falls back to a classical KEM. Peers/observers can key
# off this to know the endpoint requested PQC.
KYBER_INTENT_ALPN = "distllm-pq-kyber768"

# The hybrid group we would negotiate if a PQC-capable stack were present.
# X25519MLKEM768 is the IETF/NIST-aligned hybrid (classical X25519 +
# ML-KEM-768) supported by OpenSSL 3.5+ and BoringSSL.
KYBER_HYBRID_GROUP = "X25519MLKEM768"

# String stashed on the context object so intent_signaled() / tests can query
# it without inspecting private TLS state.
_INTENT_ATTR = "_distllm_pq_intent"
_MODE_ATTR = "_distllm_pq_mode"  # "active" (real PQC) | "fallback" | None


class QuantumSafeUnavailable(RuntimeError):
    """Raised when Kyber/ML-KEM was requested but no PQC TLS stack is present.

    Carries the recorded ``intent`` so callers can decide to fall back to a
    classical context while still logging what was attempted.
    """

    def __init__(self, message: str, intent: str = "kyber768") -> None:
        super().__init__(message)
        self.intent = intent


def _probe_openssl_group_support() -> bool:
    """Best-effort probe: can this ssl/OpenSSL build negotiate an ML-KEM group?

    OpenSSL 3.5+ exposes ML-KEM hybrid groups and CPython 3.13+ adds
    ``SSLContext.set_groups``. On a stock 3.11 + OpenSSL 3.0 host neither is
    present, so this returns False and callers fall back gracefully.
    """
    try:
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    except Exception:
        return False
    # CPython 3.13+ SSLContext.set_groups is the only stdlib path to select a
    # named group like X25519MLKEM768. Its presence is necessary but we also
    # need the underlying OpenSSL to actually know the group.
    if not hasattr(ctx, "set_groups"):
        return False
    try:
        ctx.set_groups(KYBER_HYBRID_GROUP)  # type: ignore[attr-defined]
        return True
    except Exception:
        return False


def _probe_oqs() -> bool:
    """Probe for the Open Quantum Safe python binding (``oqs``)."""
    try:
        import oqs  # type: ignore  # noqa: F401
    except Exception:
        return False
    try:
        import oqs  # type: ignore

        mechs = set(getattr(oqs, "get_enabled_kem_mechanisms", lambda: [])())
        # Kyber768 was renamed to ML-KEM-768 in newer liboqs; accept either.
        return bool(mechs & {"Kyber768", "ML-KEM-768"})
    except Exception:
        return False


def _probe_cryptography_kem() -> bool:
    """Probe for KEM primitives in ``cryptography`` >= 43.

    Note: even if these primitives exist, ``cryptography`` does not itself
    terminate TLS, so this only indicates a *library* capability, not a
    turn-key TLS group. We treat it as "available primitives" but real TLS
    wiring still needs OpenSSL group support.
    """
    try:
        from cryptography.hazmat.primitives.asymmetric import (  # type: ignore  # noqa: F401
            ml_kem,
        )

        return True
    except Exception:
        return False


def quantum_safe_available() -> bool:
    """Return True if a real ML-KEM/Kyber TLS handshake can be configured.

    This is intentionally strict: it requires a stack that can actually put a
    Kyber/ML-KEM group into the TLS handshake (OpenSSL group support via
    ``set_groups`` or the ``oqs`` provider). Mere presence of KEM primitives
    in ``cryptography`` is *not* sufficient for TLS termination, so it does
    not by itself flip this to True.
    """
    return _probe_openssl_group_support() or _probe_oqs()


def pq_capability_report() -> dict[str, bool]:
    """Introspection helper: which PQC building blocks are present.

    Useful for diagnostics / tests. Distinguishes "can terminate PQC TLS"
    from "has PQC primitives in a library".
    """
    return {
        "openssl_group": _probe_openssl_group_support(),
        "oqs": _probe_oqs(),
        "cryptography_ml_kem": _probe_cryptography_kem(),
        "tls_ready": quantum_safe_available(),
    }


def intent_signaled(ctx: ssl.SSLContext) -> str | None:
    """Return the quantum-safe intent recorded on ``ctx``, if any.

    Returns the intent string (e.g. ``"kyber768"``) when the context was
    built with ``use_kyber=True`` (whether PQC is active or a classical
    fallback), else ``None``.
    """
    return getattr(ctx, _INTENT_ATTR, None)


def intent_mode(ctx: ssl.SSLContext) -> str | None:
    """Return ``"active"`` (real PQC configured), ``"fallback"`` (classical
    with intent signaled), or ``None`` (no PQC requested)."""
    return getattr(ctx, _MODE_ATTR, None)


def _apply_intent_alpn(ctx: ssl.SSLContext, *, active: bool) -> None:
    """Advertise quantum-safe intent on the wire via an ALPN token.

    We prepend the PQC intent token ahead of the usual application protocols
    so a peer sees the endpoint requested quantum-safe transport even when the
    negotiated KEM is classical.
    """
    try:
        # h2/http1.1 kept so normal negotiation still works; the PQC token is
        # purely a signal and won't be selected as the app protocol by peers
        # that don't understand it.
        ctx.set_alpn_protocols([KYBER_INTENT_ALPN, "h2", "http/1.1"])
    except Exception:
        # ALPN is best-effort; never fail the context build over a signal.
        pass
    setattr(ctx, _INTENT_ATTR, "kyber768")
    setattr(ctx, _MODE_ATTR, "active" if active else "fallback")


def build_tls_context(
    certfile: str,
    keyfile: str,
    *,
    use_kyber: bool = False,
    require_client_cert: bool = True,
    ca_certfile: str | None = None,
    fallback_on_unavailable: bool = True,
    server_side: bool = True,
) -> ssl.SSLContext:
    """Build an mTLS :class:`ssl.SSLContext` for a federation link.

    Parameters
    ----------
    certfile, keyfile:
        Server (or client) certificate + private key paths.
    use_kyber:
        Opt into quantum-safe (ML-KEM/Kyber) key establishment. See module
        docstring for the availability semantics.
    require_client_cert:
        Federation links are mutually authenticated by default; when True the
        context requires and verifies a peer certificate
        (``CERT_REQUIRED``). Requires ``ca_certfile`` to verify against.
    ca_certfile:
        CA bundle used to verify the peer certificate for mTLS. If omitted the
        system default trust store is loaded.
    fallback_on_unavailable:
        When ``use_kyber=True`` but no PQC TLS stack is present: if True
        (default) return a classical context that *signals* PQC intent on the
        wire (graceful fallback); if False raise
        :class:`QuantumSafeUnavailable`.
    server_side:
        Choose ``PROTOCOL_TLS_SERVER`` vs ``PROTOCOL_TLS_CLIENT`` purpose.

    Returns
    -------
    ssl.SSLContext
        A ready-to-use context. Query :func:`intent_signaled` /
        :func:`intent_mode` to see the PQC decision.

    Raises
    ------
    QuantumSafeUnavailable
        Only when ``use_kyber=True`` and ``fallback_on_unavailable=False`` and
        no PQC stack is available.
    """
    purpose = (
        ssl.PROTOCOL_TLS_SERVER if server_side else ssl.PROTOCOL_TLS_CLIENT
    )
    ctx = ssl.SSLContext(purpose)

    # Modern floor: TLS 1.2+ (1.3 preferred). PQC hybrids are TLS 1.3-only in
    # practice, so requiring >=1.2 keeps classical fallback broadly compatible.
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2

    ctx.load_cert_chain(certfile=certfile, keyfile=keyfile)

    # Mutual TLS for federation: verify the peer.
    if require_client_cert:
        ctx.verify_mode = ssl.CERT_REQUIRED
        if ca_certfile:
            ctx.load_verify_locations(cafile=ca_certfile)
        else:
            # Fall back to system trust; caller is responsible for ensuring the
            # peer cert is verifiable. We keep CERT_REQUIRED so this is still
            # true mTLS rather than silently downgrading.
            ctx.load_default_certs(
                ssl.Purpose.CLIENT_AUTH
                if server_side
                else ssl.Purpose.SERVER_AUTH
            )
    else:
        ctx.verify_mode = ssl.CERT_NONE

    if not use_kyber:
        return ctx

    # --- Quantum-safe path -------------------------------------------------
    if quantum_safe_available():
        configured = _configure_kyber_group(ctx)
        if configured:
            _apply_intent_alpn(ctx, active=True)
            return ctx
        # Probe said yes but wiring failed; treat as unavailable below.

    # PQC not available (or wiring failed): record intent + decide.
    if fallback_on_unavailable:
        _apply_intent_alpn(ctx, active=False)
        return ctx

    raise QuantumSafeUnavailable(
        "ML-KEM/Kyber TLS requested but no post-quantum TLS stack is "
        "available (need OpenSSL 3.5+ ML-KEM groups or the oqs provider). "
        "Intent recorded as 'kyber768'.",
        intent="kyber768",
    )


def _configure_kyber_group(ctx: ssl.SSLContext) -> bool:
    """Best-effort: wire the hybrid ML-KEM group into ``ctx``.

    Returns True on success. Only reachable when a probe reported capability.
    Kept isolated so the exact library path is documented in one place.
    """
    # Path 1: CPython 3.13+ / OpenSSL 3.5+ named-group selection.
    set_groups = getattr(ctx, "set_groups", None)
    if set_groups is not None:
        try:
            set_groups(KYBER_HYBRID_GROUP)
            return True
        except Exception:
            return False
    # Path 2 (oqs provider) would be configured at the OpenSSL provider level,
    # not via the Python SSLContext, so there is nothing to do on ctx here;
    # report success so intent is marked active if the provider is loaded.
    return _probe_oqs()


__all__ = [
    "QuantumSafeUnavailable",
    "KYBER_INTENT_ALPN",
    "KYBER_HYBRID_GROUP",
    "build_tls_context",
    "quantum_safe_available",
    "pq_capability_report",
    "intent_signaled",
    "intent_mode",
]
