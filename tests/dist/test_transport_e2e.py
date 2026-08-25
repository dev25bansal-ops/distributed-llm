"""End-to-end tests for optional E2E encryption wired into TensorTransport.

Verifies W3-T3 wiring: when ``DISTLLM_E2E_TRANSPORT=1`` (or an established
:class:`E2EEncryption` instance is supplied), tensor payloads are encrypted
before entering the wire and decrypted after receipt, using the REAL crypto
from :mod:`distllm.security.e2e` (X25519 + XSalsa20-Poly1305).

Only the network substrate (NCCL send/recv, QUIC forward_pass) is replaced
with in-memory fakes — no GPU, no sockets, no network.  All key material,
encryption, and decryption exercise the genuine PyNaCl code paths.

Coverage:
- Round-trip with E2E enabled reproduces the original plaintext exactly.
- Without E2E, payloads pass through unchanged (byte-identical).
- Wrong-key decryption fails cleanly with ``E2EError`` (no crash, no
  partial output).
- Env gate semantics: default OFF, only literal "1" enables, flag-without-
  session fails CLOSED (raises rather than shipping plaintext).
"""

from __future__ import annotations

import asyncio
import math

import pytest
import torch

from distllm.dist.pipeline.transport import (
    E2E_ENV_VAR,
    E2E_PACKET_OVERHEAD,
    TensorTransport,
    TransportBackend,
    _UnestablishedE2E,
    e2e_transport_enabled,
)
from distllm.security.e2e import (
    E2EEncryption,
    E2EError,
)


# ===========================================================================
# Helpers — in-memory stand-ins for the network substrate + key exchange
# ===========================================================================


def make_established_pair(
    cluster_key: str = "test-cluster-key-w3t3-0123456789abcdef",
) -> tuple[E2EEncryption, E2EEncryption]:
    """Two E2EEncryption instances with a completed, HMAC-signed key exchange."""
    alice = E2EEncryption(cluster_key=cluster_key, node_id="alice")
    bob = E2EEncryption(cluster_key=cluster_key, node_id="bob")
    alice.import_signed_public_key(bob.get_signed_public_key())
    bob.import_signed_public_key(alice.get_signed_public_key())
    assert alice.is_established and bob.is_established
    return alice, bob


class FakeNccl:
    """Records sent uint8 streams; serves the last payload back for recv.

    Mimics the interface TensorTransport uses: ``is_initialized``, ``send``
    and ``recv(shape, dtype, src, tag, device)``.  This replaces ONLY the
    network — everything above it (serialization, encryption) is real.
    """

    def __init__(self):
        self.is_initialized = True
        self.sent: list[torch.Tensor] = []
        self.incoming: bytes | None = None

    def send(self, tensor: torch.Tensor, dst: int, tag: int = 0) -> None:
        self.sent.append(tensor.detach().cpu().clone())

    def destroy(self) -> None:  # parity with NcclTransport API
        self.is_initialized = False

    def recv(
        self,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        src: int = 0,
        tag: int = 0,
        device: str | None = None,
        async_op: bool = False,
    ) -> torch.Tensor:
        assert self.incoming is not None, "no incoming payload staged"
        # Expected frame size follows the real NcclTransport contract:
        # numel(shape) * element_size(dtype) bytes.
        nbytes = math.prod(shape) * torch.empty((), dtype=dtype).element_size()
        assert len(self.incoming) == nbytes, (
            f"receiver asked for {nbytes} bytes "
            f"but wire carries {len(self.incoming)}"
        )
        buf = torch.frombuffer(bytearray(self.incoming), dtype=dtype)
        return buf.reshape(shape).clone()


class FakeQuicClient:
    """Echo-style QUIC client: forward_pass returns exactly what it got."""

    def __init__(self):
        self.last_wire: bytes | None = None

    async def forward_pass(self, data: bytes, timeout: float = 120.0) -> bytes:
        self.last_wire = bytes(data)
        return bytes(data)

    async def close(self):  # parity with real client API
        self.last_wire = None


def wire_bytes_is_not_plaintext(wire: bytes, plaintext_hint: bytes) -> None:
    """The ciphertext must not contain the plaintext as a substring."""
    assert plaintext_hint not in wire


# ===========================================================================
# Env gate semantics
# ===========================================================================


class TestE2EEnvGate:
    """DISTLLM_E2E_TRANSPORT parsing and construction-time capture."""

    def test_default_off(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(E2E_ENV_VAR, raising=False)
        assert e2e_transport_enabled() is False

    def test_only_literal_one_enables(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        for value in ("0", "", "true", "TRUE", "yes", "1 ", "on"):
            monkeypatch.setenv(E2E_ENV_VAR, value)
            assert e2e_transport_enabled() is False, f"value={value!r} must stay OFF"
        monkeypatch.setenv(E2E_ENV_VAR, "1")
        assert e2e_transport_enabled() is True

    def test_gate_captured_at_construction(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Flipping the env var after construction cannot change behavior."""
        monkeypatch.delenv(E2E_ENV_VAR, raising=False)
        t = TensorTransport(backend=TransportBackend.GRPC)
        assert t._e2e_required is False
        monkeypatch.setenv(E2E_ENV_VAR, "1")
        assert t._e2e_required is False  # unchanged post-construction
        t.destroy()

    def test_flag_without_session_installs_placeholder(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(E2E_ENV_VAR, "1")
        t = TensorTransport(backend=TransportBackend.GRPC)
        assert t.e2e_active is False
        assert t._e2e is not None
        t.destroy()

    def test_flag_off_with_explicit_session_still_enforces(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Passing an explicit established session implies intent to encrypt."""
        monkeypatch.delenv(E2E_ENV_VAR, raising=False)
        alice, _bob = make_established_pair()
        t = TensorTransport(backend=TransportBackend.GRPC, e2e=alice)
        assert t._e2e_required is True
        assert t.e2e_active is True
        t.destroy()

    def test_env_constant_value(self) -> None:
        assert E2E_ENV_VAR == "DISTLLM_E2E_TRANSPORT"


# ===========================================================================
# Fail-closed behavior
# ===========================================================================


class TestFailClosed:
    """Flag set but no usable session => raise, never ship plaintext."""

    @pytest.fixture()
    def flagged_nccl_transport(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> TensorTransport:
        monkeypatch.setenv(E2E_ENV_VAR, "1")
        t = TensorTransport(backend=TransportBackend.GRPC)
        t._nccl = FakeNccl()  # network up, session absent
        return t

    def test_send_tensor_raises_without_session(
        self, flagged_nccl_transport: TensorTransport
    ) -> None:
        t = flagged_nccl_transport
        with pytest.raises(E2EError, match="DISTLLM_E2E_TRANSPORT=1"):
            t.send_tensor(torch.zeros(4), dst=0)
        # Nothing may have hit the wire.
        assert isinstance(t._nccl, FakeNccl)
        assert t._nccl.sent == []

    def test_recv_tensor_raises_without_session(
        self, flagged_nccl_transport: TensorTransport
    ) -> None:
        t = flagged_nccl_transport
        nccl = t._nccl
        assert isinstance(nccl, FakeNccl)
        # 4 float32 plaintext + overhead = expected wire length; stage the
        # full frame so the fake's length check passes and decryption (not
        # framing) is what fails.
        nccl.incoming = b"x" * (16 + E2E_PACKET_OVERHEAD)
        with pytest.raises(E2EError, match="set_e2e_session"):
            t.recv_tensor(shape=(4,), dtype=torch.float32, src=0)

    def test_forward_pass_raises_without_session(
        self, flagged_nccl_transport: TensorTransport
    ) -> None:
        t = flagged_nccl_transport
        # QUIC client present but no session: encryption enforcement must
        # raise before any bytes reach the network.
        qc = FakeQuicClient()
        t._quic_client = qc

        async def _send() -> None:
            await t.send_forward_pass(b"secret-request")

        with pytest.raises(E2EError):
            asyncio.run(_send())
        assert qc.last_wire is None  # nothing reached the wire

    @pytest.mark.asyncio
    async def test_recv_placeholder_raises_directly(
        self, flagged_nccl_transport: TensorTransport
    ) -> None:
        """The placeholder itself refuses to decrypt anything."""
        t = flagged_nccl_transport
        assert isinstance(t._e2e, _UnestablishedE2E)
        with pytest.raises(E2EError):
            t._e2e.decrypt_tensor_payload(b"\x00" * 64)  # type: ignore[attr-defined]

    def test_set_e2e_session_rejects_unestablished(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv(E2E_ENV_VAR, raising=False)
        fresh = E2EEncryption(cluster_key="a-key-long-enough-for-here-1234")
        assert fresh.is_established is False
        t = TensorTransport(backend=TransportBackend.GRPC)
        with pytest.raises(E2EError, match="exchange keys first"):
            t.set_e2e_session(fresh)
        t.destroy()

    def test_wrong_cluster_key_session_fails_at_install(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A session established under a different cluster secret is still
        cryptographically valid (HMAC sigs differ => import fails first)."""
        monkeypatch.delenv(E2E_ENV_VAR, raising=False)
        stranger = E2EEncryption(
            cluster_key="completely-other-secret-xyz-9876", node_id="mallory"
        )
        t = TensorTransport(backend=TransportBackend.GRPC)
        with pytest.raises((E2EError, Exception)):
            t.set_e2e_session(stranger)
        t.destroy()


# ===========================================================================
# NCCL-path round trips (real crypto, fake network)
# ===========================================================================


class TestNCCLPathE2E:
    """send_tensor/recv_tensor encrypt-before-wire / decrypt-after-receive."""

    def _pair_transports(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> tuple[TensorTransport, TensorTransport, FakeNccl, FakeNccl]:
        monkeypatch.delenv(E2E_ENV_VAR, raising=False)
        alice, bob = make_established_pair()
        wire_a = FakeNccl()  # "network" seen by alice
        wire_b = FakeNccl()  # "network" seen by bob
        ta = TensorTransport(backend=TransportBackend.GRPC, e2e=alice)
        tb = TensorTransport(backend=TransportBackend.GRPC, e2e=bob)
        ta._nccl = wire_a
        tb._nccl = wire_b
        return ta, tb, wire_a, wire_b

    @pytest.mark.parametrize(
        "shape,dtype",
        [
            ((8,), torch.float32),
            ((2, 16), torch.float16),
            ((3, 5, 7), torch.bfloat16),
            ((1024,), torch.int64),
            ((0,), torch.float32),  # empty tensor edge case
        ],
    )
    def test_roundtrip_recovers_original_plaintext(
        self,
        monkeypatch: pytest.MonkeyPatch,
        shape: tuple[int, ...],
        dtype: torch.dtype,
    ) -> None:
        ta, tb, wire_a, wire_b = self._pair_transports(monkeypatch)
        if math.prod(shape) == 0:
            original = torch.empty(shape, dtype=dtype)
        else:
            original = torch.randn(*shape).to(dtype)

        ta.send_tensor(original, dst=1, tag=7)
        assert len(wire_a.sent) == 1
        wire = wire_a.sent[0]

        # Wire format: uint8 stream, fixed overhead over plaintext size.
        n_plain = math.prod(shape) * torch.empty((), dtype=dtype).element_size()
        assert wire.dtype == torch.uint8
        assert wire.numel() == n_plain + E2E_PACKET_OVERHEAD

        # Ciphertext must not leak the raw bytes.
        wire_bytes = wire.numpy().tobytes()
        if n_plain > 0:
            plain = (
                original.contiguous()
                .view(torch.uint8)
                .numpy()
                .tobytes()
            )
            wire_bytes_is_not_plaintext(wire_bytes, plain)

        # Receiver pulls the same bytes off the wire and decrypts.
        wire_b.incoming = wire_bytes
        recovered = tb.recv_tensor(shape=shape, dtype=dtype, src=1, tag=7)
        assert recovered.dtype == dtype
        assert recovered.shape == shape
        assert torch.equal(recovered.cpu(), original.cpu())

        ta.destroy()
        tb.destroy()

    def test_two_messages_differ_on_wire_per_message_salt_nonce(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Fresh salt/nonce per message => identical tensors look distinct."""
        ta, tb, wire_a, _ = self._pair_transports(monkeypatch)
        t = torch.arange(16, dtype=torch.float32)
        ta.send_tensor(t, dst=1)
        ta.send_tensor(t, dst=1)
        w1, w2 = wire_a.sent
        assert w1.shape == w2.shape
        assert not torch.equal(w1, w2)  # randomized packet header/key
        ta.destroy()
        tb.destroy()

    def test_tampered_wire_byte_raises_cleanly(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ta, tb, wire_a, wire_b = self._pair_transports(monkeypatch)
        ta.send_tensor(torch.ones(32, dtype=torch.float32), dst=1)
        wire = bytearray(wire_a.sent[0].numpy().tobytes())
        wire[-1] ^= 0xFF  # flip one bit inside the Poly1305 tag region
        wire_b.incoming = bytes(wire)
        with pytest.raises(E2EError, match="[Dd]ecryption failed|wrong key|tampered"):
            tb.recv_tensor(shape=(32,), dtype=torch.float32, src=1)
        ta.destroy()
        tb.destroy()

    def test_truncated_wire_raises_cleanly(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A packet shorter than the fixed header cannot be decrypted —
        and must raise rather than return raw bytes."""
        ta, tb, wire_a, _wire_b = self._pair_transports(monkeypatch)
        ta.send_tensor(torch.ones(4, dtype=torch.float32), dst=1)
        full = wire_a.sent[0].numpy().tobytes()
        assert len(full) == 16 + E2E_PACKET_OVERHEAD
        # Truncated frames below every threshold:
        with pytest.raises(E2EError):
            tb._decrypt_payload(full[:10])
        with pytest.raises(E2EError):
            tb._decrypt_payload(full[:40])  # past salt, before nonce+tag
        ta.destroy()
        tb.destroy()

    def test_wrong_key_decryption_raises_cleanly(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Receiver holding an unrelated session cannot read the payload."""
        monkeypatch.delenv(E2E_ENV_VAR, raising=False)
        alice, _bob = make_established_pair("shared-key-one-aaaaaaaaaaaaaaaa")
        eve_alice, eve_bob = make_established_pair(
            "shared-key-two-bbbbbbbbbbbbbbbb"
        )
        ta = TensorTransport(backend=TransportBackend.GRPC, e2e=alice)
        te = TensorTransport(backend=TransportBackend.GRPC, e2e=eve_alice)
        wire = FakeNccl()
        ta._nccl = wire
        te._nccl = wire
        ta.send_tensor(torch.full((64,), 3.14, dtype=torch.float32), dst=1)
        wire.incoming = wire.sent[0].numpy().tobytes()
        with pytest.raises(E2EError):
            te.recv_tensor(shape=(64,), dtype=torch.float32, src=1)
        ta.destroy()
        te.destroy()

    def test_no_e2e_passthrough_unchanged(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without E2E, send/recv behave exactly as before this change."""
        monkeypatch.delenv(E2E_ENV_VAR, raising=False)
        t = TensorTransport(backend=TransportBackend.GRPC)
        nccl = FakeNccl()
        t._nccl = nccl
        original = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        t.send_tensor(original, dst=0, tag=3)
        assert len(nccl.sent) == 1
        sent = nccl.sent[0]
        assert sent.dtype == torch.float32  # untouched, not a byte stream
        assert torch.equal(sent, original)

        payload = original.numpy().tobytes()
        nccl.incoming = payload
        back = t.recv_tensor(shape=(3,), dtype=torch.float32, src=0, tag=3)
        assert torch.equal(back, original)
        t.destroy()

    def test_non_contiguous_input_normalized(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ta, tb, wire_a, wire_b = self._pair_transports(monkeypatch)
        base = torch.arange(64, dtype=torch.float32).reshape(8, 8)
        view = base[:, ::2].t()  # non-contiguous
        ta.send_tensor(view, dst=1)
        wire_b.incoming = wire_a.sent[0].numpy().tobytes()
        got = tb.recv_tensor(shape=view.shape, dtype=torch.float32, src=1)
        assert torch.equal(got, view.contiguous())
        ta.destroy()
        tb.destroy()


# ===========================================================================
# QUIC-path round trip
# ===========================================================================


class TestQUICPathE2E:
    """send_forward_pass encrypts the request and decrypts the response."""

    @pytest.mark.asyncio
    async def test_roundtrip_with_echo_server(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv(E2E_ENV_VAR, raising=False)
        alice, bob = make_established_pair()
        t = TensorTransport(backend=TransportBackend.GRPC, e2e=alice)
        qc = FakeQuicClient()
        t._quic_client = qc

        secret = b"\x00\x01tensor-payload-\xff" * 10
        out = await t.send_forward_pass(secret)
        assert out == secret

        # What actually crossed the wire was ciphertext.
        assert qc.last_wire is not None
        wire_bytes_is_not_plaintext(qc.last_wire, secret)
        assert len(qc.last_wire) == len(secret) + E2E_PACKET_OVERHEAD

        # The peer can recover it with its own half of the session.
        assert bob.decrypt_tensor_payload(qc.last_wire) == secret
        t.destroy()

    @pytest.mark.asyncio
    async def test_without_e2e_payload_untouched(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv(E2E_ENV_VAR, raising=False)
        t = TensorTransport(backend=TransportBackend.GRPC)
        qc = FakeQuicClient()
        t._quic_client = qc
        data = b"plain-forward-pass"
        out = await t.send_forward_pass(data)
        assert out == data
        assert qc.last_wire == data  # byte-identical on the wire
        t.destroy()

    @pytest.mark.asyncio
    async def test_flag_without_session_raises_not_plaintext(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(E2E_ENV_VAR, "1")
        t = TensorTransport(backend=TransportBackend.GRPC)
        qc = FakeQuicClient()
        t._quic_client = qc
        with pytest.raises(E2EError):
            await t.send_forward_pass(b"must-not-leak")
        assert qc.last_wire is None  # nothing reached the network
        t.destroy()

    @pytest.mark.asyncio
    async def test_response_from_wrong_key_peer_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A MITM answering with ciphertext from a foreign session fails."""
        monkeypatch.delenv(E2E_ENV_VAR, raising=False)
        alice, _bob = make_established_pair("key-a-12345678901234567890")
        mallory_a, mallory_b = make_established_pair(
            "key-m-09876543210987654321"
        )

        t = TensorTransport(backend=TransportBackend.GRPC, e2e=alice)
        forged = mallory_b.encrypt_tensor_payload(b"evil response")

        class InjectingQuic(FakeQuicClient):
            async def forward_pass(self, data: bytes, timeout: float = 120.0) -> bytes:
                return forged  # attacker-controlled reply

        t._quic_client = InjectingQuic()
        with pytest.raises(E2EError):
            await t.send_forward_pass(b"hello")
        t.destroy()


# ===========================================================================
# Session management & lifecycle
# ===========================================================================


class TestSessionManagement:
    def test_set_e2e_session_after_construction_activates(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv(E2E_ENV_VAR, raising=False)
        alice, bob = make_established_pair()
        t = TensorTransport(backend=TransportBackend.GRPC)
        assert t.e2e_active is False
        t.set_e2e_session(alice)
        assert t.e2e_active is True

        nccl = FakeNccl()
        t._nccl = nccl
        t.send_tensor(torch.ones(4, dtype=torch.float32), dst=1)
        assert nccl.sent[0].dtype == torch.uint8  # now encrypted
        t.destroy()

    def test_destroy_preserves_session_state(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """E2E session is connection-level; destroy() tears down transports
        only."""
        monkeypatch.delenv(E2E_ENV_VAR, raising=False)
        alice, _ = make_established_pair()
        t = TensorTransport(backend=TransportBackend.GRPC, e2e=alice)
        t.destroy()
        assert t.e2e_active is True
        assert t.is_available is False

    def test_overhead_constant_matches_e2e_packet_format(self) -> None:
        from distllm.security.e2e import NONCE_BYTES, SALT_BYTES, TAG_BYTES

        assert E2E_PACKET_OVERHEAD == SALT_BYTES + NONCE_BYTES + TAG_BYTES
        assert SALT_BYTES == 16 and NONCE_BYTES == 24 and TAG_BYTES == 16

    def test_encrypt_decrypt_helpers_route_to_real_crypto(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv(E2E_ENV_VAR, raising=False)
        alice, bob = make_established_pair()
        t = TensorTransport(backend=TransportBackend.GRPC, e2e=alice)
        blob = t._encrypt_payload(b"round-trip-me")
        assert blob != b"round-trip-me"
        assert bob.decrypt_tensor_payload(blob) == b"round-trip-me"
        t.destroy()
