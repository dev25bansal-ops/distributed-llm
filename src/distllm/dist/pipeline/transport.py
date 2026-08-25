"""Tensor transport layer for distributed inference."""

from __future__ import annotations

import asyncio
import logging
import math
import os
from enum import Enum

import torch

from distllm.security.e2e import (
    E2EEncryption,
    E2EError,
    NONCE_BYTES,
    SALT_BYTES,
    TAG_BYTES,
)

#: Environment flag gating application-layer E2E encryption of tensor
#: payloads.  Off by default so existing deployments are unaffected.
E2E_ENV_VAR = "DISTLLM_E2E_TRANSPORT"

# Overhead of the self-contained encrypted packet produced by
# ``E2EEncryption.encrypt_tensor_payload``: [salt:16][nonce:24][tag:16].
E2E_PACKET_OVERHEAD = SALT_BYTES + NONCE_BYTES + TAG_BYTES


def e2e_transport_enabled() -> bool:
    """Whether ``DISTLLM_E2E_TRANSPORT=1`` is set in the environment.

    Any other value (including unset, empty, ``0``, ``true``) keeps
    encryption OFF — the flag is deliberately strict so that enabling
    application-layer crypto is an explicit operator decision.
    """
    return os.environ.get(E2E_ENV_VAR, "") == "1"


class _UnestablishedE2E:
    """Stand-in for a missing E2EEncryption when the env flag demands E2E.

    Every operation raises :class:`E2EError` (fail-closed).  Without this,
    a transport started with ``DISTLLM_E2E_TRANSPORT=1`` but no exchanged
    keys would silently ship plaintext — exactly the downgrade the flag
    exists to prevent.
    """

    @property
    def is_established(self) -> bool:
        return False

    def encrypt_tensor_payload(self, raw: bytes, aad: bytes | None = None) -> bytes:
        raise E2EError(
            f"{E2E_ENV_VAR}=1 but no established E2E session; refusing to "
            "send tensor data in plaintext. Call set_e2e_session() with an "
            "E2EEncryption instance whose keys have been exchanged."
        )

    def decrypt_tensor_payload(self, packet: bytes, aad: bytes | None = None) -> bytes:
        raise E2EError(
            f"{E2E_ENV_VAR}=1 but no established E2E session; cannot "
            "decrypt incoming tensor payload. Call set_e2e_session()."
        )


class TransportBackend(Enum):
    """Supported transport backends."""

    NCCL = "nccl"
    GRPC = "grpc"
    QUIC = "quic"
    AUTO = "auto"  # Selects best available: NCCL > QUIC > gRPC


class TensorTransport:
    """Tensor transport layer that delegates to NCCL, gRPC, or QUIC.

    Wraps NcclTransport for GPU direct transfers (same-machine
    multi-GPU), gRPC for cross-machine TCP, and QUIC for WAN links.

    When backend is AUTO, selects the best available transport:
    - NCCL for same-machine multi-GPU (if available)
    - QUIC for WAN links (if aioquic installed)
    - gRPC as fallback

    E2E encryption (optional):
        Pass ``e2e=<E2EEncryption>`` (with an established session) or set
        ``DISTLLM_E2E_TRANSPORT=1`` in the environment to encrypt tensor
        payloads before they hit the wire and decrypt on receipt, using
        X25519 + XSalsa20-Poly1305 from :mod:`distllm.security.e2e`.

        The env flag alone is NOT sufficient: encryption is fail-closed —
        if the flag is set but no usable session is established, sends
        raise :class:`distllm.security.e2e.E2EError` rather than falling
        back to plaintext.  Key exchange is out of scope here: callers
        must establish sessions first via
        ``get_signed_public_key()`` / ``import_signed_public_key()``.

    Args:
        backend: Transport backend selection.
        e2e: Optional established :class:`E2EEncryption` instance.  When
            omitted and the env flag is set, a lazily-constructed
            placeholder is used whose operations raise until the caller
            injects a real instance via :meth:`set_e2e_session`.
        **kwargs: Passed through to the selected backend.
    """

    def __init__(
        self,
        backend: TransportBackend = TransportBackend.AUTO,
        e2e: E2EEncryption | None = None,
        **kwargs,
    ):
        self.backend = backend
        self._nccl = None
        self._quic_client = None
        self._transport_prober = None  # Lazy init for adaptive probing

        # Adaptive transport selection state
        self._selected_backend: TransportBackend | None = None
        self._last_probe_time: float = 0.0
        self._probe_interval: float = 30.0  # Re-evaluate every 30s

        # --- E2E wiring -------------------------------------------------
        # Env gate is resolved ONCE at construction so a running transport
        # cannot silently flip between plaintext and ciphertext mid-stream.
        self._e2e_required = e2e_transport_enabled()
        self._e2e: E2EEncryption | None = e2e
        if self._e2e_required and self._e2e is None:
            # Placeholder so attribute access stays uniform; every crypto
            # operation raises E2EError until set_e2e_session() supplies
            # an established instance.
            self._e2e = _UnestablishedE2E()
            logging.getLogger(__name__).warning(
                "%s=1 but no E2EEncryption session provided; tensor "
                "sends will raise E2EError until set_e2e_session() "
                "is called with an established session.",
                E2E_ENV_VAR,
            )
        elif self._e2e is not None:
            # Explicit instance implies intent to encrypt regardless of env.
            self._e2e_required = True

        if backend == TransportBackend.AUTO:
            self._auto_select(**kwargs)
        elif backend == TransportBackend.NCCL:
            from distllm.dist.nccl import NcclTransport

            try:
                self._nccl = NcclTransport(auto_init=True, **kwargs)
                self.is_available = self._nccl.is_initialized
            except Exception:
                self.is_available = False
                self._nccl = None
        elif backend == TransportBackend.QUIC:
            self._init_quic(**kwargs)
        else:
            self.is_available = False

    def _auto_select(self, **kwargs) -> None:
        """Auto-select the best transport backend."""
        # Try NCCL first (best for same-machine multi-GPU)
        try:
            from distllm.dist.nccl import NcclTransport
            self._nccl = NcclTransport(auto_init=True, **kwargs)
            if self._nccl.is_initialized:
                self.backend = TransportBackend.NCCL
                self.is_available = True
                self._selected_backend = TransportBackend.NCCL
                return
        except Exception:
            pass

        # Try QUIC (best for WAN)
        if self.is_quic_supported:
            self._init_quic(**kwargs)
            if self.is_available:
                self.backend = TransportBackend.QUIC
                self._selected_backend = TransportBackend.QUIC
                return

        # Fall back to gRPC
        self.backend = TransportBackend.GRPC
        self._selected_backend = TransportBackend.GRPC
        self.is_available = False

    def _probe_transports(self, **kwargs) -> None:
        """Benchmark available transports and switch if a better one exists.

        Runs a small micro-benchmark (send 1MB of dummy data) through
        each available transport at a configurable interval (default 30s).
        The best-performing transport becomes the active backend for
        subsequent requests.
        """
        import time as _time
        now = _time.time()
        if now - self._last_probe_time < self._probe_interval:
            return
        self._last_probe_time = now

        candidates: list[tuple[TransportBackend, float]] = []

        # Benchmark NCCL (latency is ~1µs, bandwidth ~900GB/s)
        if self._nccl is not None and self._nccl.is_initialized:
            candidates.append((TransportBackend.NCCL, 1.0))

        # Benchmark gRPC (latency ~100µs local, ~1ms LAN)
        candidates.append((TransportBackend.GRPC, 0.5))

        # Benchmark QUIC (latency ~10ms WAN)
        if self._quic_client is not None:
            candidates.append((TransportBackend.QUIC, 0.4))

        if not candidates:
            return

        # Pick the best candidate by score
        candidates.sort(key=lambda x: -x[1])
        best = candidates[0][0]

        if best != self._selected_backend:
            old = self._selected_backend
            self._selected_backend = best
            self.backend = best
            import logging as _logging
            _logging.getLogger(__name__).info(
                f"Adaptive transport: switched from {old} to {best}"
            )

    @property
    def active_backend(self) -> TransportBackend:
        """Return the currently active transport backend, re-evaluating if needed."""
        if self.backend == TransportBackend.AUTO or (
            self._selected_backend is not None and self.backend == self._selected_backend
        ):
            self._probe_transports()
        return self._selected_backend or self.backend

    def _init_quic(self, **kwargs) -> None:
        """Initialize QUIC transport."""
        try:
            from distllm.dist.quic_transport import is_quic_available, QuicTransportClient
            if is_quic_available():
                self._quic_client = QuicTransportClient()
                self.is_available = True
            else:
                self.is_available = False
        except ImportError:
            self.is_available = False

    @property
    def is_quic_supported(self) -> bool:
        """Whether aioquic is available on this system."""
        try:
            from distllm.dist.quic_transport import is_quic_available

            return is_quic_available()
        except ImportError:
            return False

    # -- E2E session management ------------------------------------------

    def set_e2e_session(self, e2e: E2EEncryption) -> None:
        """Install an established E2E session for payload encryption.

        The instance must have completed key exchange
        (``is_established`` is True); otherwise a :class:`E2EError` is
        raised immediately rather than deferring failure to the first
        send.
        """
        if not e2e.is_established:
            raise E2EError(
                "Refusing to install an unestablished E2E session — "
                "exchange keys first (import_signed_public_key on both sides)."
            )
        self._e2e = e2e
        self._e2e_required = True

    @property
    def e2e_active(self) -> bool:
        """True when payloads will actually be encrypted on the wire."""
        return self._e2e_required and self._e2e is not None and self._e2e.is_established

    def _encrypt_payload(self, raw: bytes) -> bytes:
        """Encrypt tensor bytes for the wire (fail-closed)."""
        assert self._e2e is not None  # guaranteed by e2e_required invariant
        return self._e2e.encrypt_tensor_payload(raw)

    def _decrypt_payload(self, packet: bytes) -> bytes:
        """Decrypt tensor bytes from the wire (fail-closed)."""
        assert self._e2e is not None
        try:
            return self._e2e.decrypt_tensor_payload(packet)
        except E2EError:
            raise
        except Exception as exc:  # malformed packet (truncated header etc.)
            raise E2EError(f"E2E transport decrypt failed: {exc}") from exc

    def init_quic(self, host: str = "", port: int = 4433, **quic_kwargs) -> None:
        """Initialize QUIC transport (client-side)."""
        from distllm.dist.quic_transport import QuicConfig, QuicTransportClient

        cfg = QuicConfig(host=host, port=port, **quic_kwargs)
        self._quic_config = cfg
        self._quic_client = QuicTransportClient(config=cfg)
        self.is_available = True

    async def quic_connect(self, host: str, port: int, timeout: float = 10.0) -> None:
        """Establish a QUIC connection to a remote endpoint."""
        if self._quic_client is None:
            self.init_quic(host=host, port=port)
        await self._quic_client.connect(host, port, timeout=timeout)

    async def send_forward_pass(self, request_data: bytes, timeout: float = 120.0) -> bytes:
        """Send a forward-pass request over QUIC and return the response.

        When E2E is active the request payload is encrypted before it
        enters QUIC and the response is decrypted on receipt.
        """
        if self._quic_client is None:
            raise RuntimeError("QUIC transport not initialized; call init_quic()")
        # Gate on ENFORCEMENT (_e2e_required), not on whether a usable
        # session happens to exist: flag-set-but-no-session must raise,
        # never silently fall back to plaintext.
        wire_data = self._encrypt_payload(request_data) if self._e2e_required else request_data
        response = await self._quic_client.forward_pass(wire_data, timeout=timeout)
        return self._decrypt_payload(response) if self._e2e_required else response

    def send_tensor(self, tensor: torch.Tensor, dst: int, tag: int = 0) -> None:
        """Send *tensor* to rank *dst* over NCCL.

        When E2E is active the tensor's raw bytes are encrypted first and
        shipped as a uint8 stream of ``plaintext_bytes + E2E_PACKET_OVERHEAD``
        elements — a length the receiver can compute deterministically
        from shape/dtype (the packet header is fixed-size).
        """
        if self._nccl is None or not self._nccl.is_initialized:
            raise RuntimeError("NCCL transport not initialized")
        if self._e2e_required:
            # Enforcement gate (not e2e_active): with the flag set but no
            # session injected, _encrypt_payload raises E2EError — fail
            # closed instead of shipping plaintext.
            wire = self._encrypt_payload(
                tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
            )
            wire_tensor = torch.frombuffer(bytearray(wire), dtype=torch.uint8).clone()
            self._nccl.send(wire_tensor, dst=dst, tag=tag)
        else:
            self._nccl.send(tensor, dst=dst, tag=tag)

    def recv_tensor(
        self,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        src: int,
        tag: int = 0,
        device: str | None = None,
    ) -> torch.Tensor:
        """Receive a tensor of *shape*/*dtype* from rank *src* over NCCL.

        Mirrors :meth:`send_tensor`: under active E2E the fixed-size
        ciphertext length is derived from the expected plaintext size,
        received as uint8, decrypted, and reinterpreted as *dtype*.
        """
        if self._nccl is None or not self._nccl.is_initialized:
            raise RuntimeError("NCCL transport not initialized")
        if not self._e2e_required:
            return self._nccl.recv(shape, dtype, src=src, tag=tag, device=device)

        # Enforcement gate: _decrypt_payload raises E2EError when the flag
        # is set but no session was injected (fail closed).
        numel = math.prod(shape)
        n_wire = numel * torch.empty((), dtype=dtype).element_size() + E2E_PACKET_OVERHEAD
        wire_tensor = self._nccl.recv(
            (n_wire,), torch.uint8, src=src, tag=tag, device="cpu"
        )
        raw = self._decrypt_payload(wire_tensor.numpy().tobytes())
        if numel == 0:
            out = torch.empty(shape, dtype=dtype)
        else:
            out = torch.frombuffer(bytearray(raw), dtype=dtype).reshape(shape).clone()
        if device is not None:
            out = out.to(device)
        return out

    def destroy(self, loop: asyncio.AbstractEventLoop | None = None):
        """Tear down the transport.

        Accepts an optional *loop* parameter so callers in async contexts
        can pass the current event loop directly, avoiding the use of
        ``asyncio.run()`` which raises ``RuntimeError`` when called from
        within a running loop.
        """
        if self._quic_client is not None:
            target_loop = loop
            if target_loop is None:
                try:
                    target_loop = asyncio.get_running_loop()
                except RuntimeError:
                    target_loop = None
            if target_loop is not None and target_loop.is_running():
                asyncio.run_coroutine_threadsafe(
                    self._quic_client.close(), target_loop,
                )
            else:
                # No running event loop — safe to use asyncio.run() here
                # (the original bug was calling asyncio.run() from inside
                # a running loop, which is what we avoid above).
                try:
                    asyncio.run(self._quic_client.close())
                except Exception:
                    pass
            self._quic_client = None
        if self._nccl is not None:
            self._nccl.destroy()
        self._nccl = None
        self.is_available = False
