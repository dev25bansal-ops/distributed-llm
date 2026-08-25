"""Regression: F-e2e-aad — SessionKeys.encrypt must AUTHENTICATE the aad parameter.

Pre-fix, ``encrypt(plaintext, aad=...)`` accepted an ``aad`` argument and
silently dropped it: identical ciphertext semantics regardless of AAD, so
callers believing they had associated-data authentication (node IDs,
request IDs, sequence numbers) had none.  An attacker able to replay a
ciphertext into a different context (different node/request) could not be
detected.

Fix approach — KDF mixing (chosen over prepend-and-authenticate):

    box_key = KDF(shared_key, salt, SHA256(b"distllm-e2e-aad-v1" || aad))

Wrong AAD derives a different box key, so the XSalsa20-Poly1305 tag fails
and decryption raises :class:`E2EError`.  Wire format is unchanged (AAD is
authenticated but never transmitted — the standard AEAD contract; peers
supply the context out-of-band).  The no-AAD derivation path is kept
bit-for-bit identical to the pre-fix construction, so legacy ciphertexts
still decrypt when ``aad=None``.

Defined semantics:
  * ``aad=None`` and ``aad=b""`` are EQUIVALENT — both mean "no associated
    data", matching AES-GCM / ChaCha20-Poly1305 conventions.
  * Any non-empty aad is bound; decrypting with a different aad (including
    ``None``) MUST fail.
"""

from __future__ import annotations

import hashlib

import pytest

from distllm.security.e2e import KEY_BYTES, SALT_BYTES, E2EError, SessionKeys

try:
    import nacl.bindings as _nacl_bindings

    HAS_NACL = True
except ImportError:  # pragma: no cover
    HAS_NACL = False

pytestmark = pytest.mark.skipif(not HAS_NACL, reason="PyNaCl not installed")


def _pair():
    shared = b"0123456789abcdef0123456789abcdef"  # exactly 32 bytes
    a = SessionKeys(shared, session_id="s1")
    b = SessionKeys(shared, session_id="s1")
    return a, b


class TestAadBinding:
    def test_encrypt_with_aad_decrypts_with_same_aad(self):
        a, b = _pair()
        ct, salt = a.encrypt(b"tensor-bytes", aad=b"req-42:src=nodeA:dst=nodeB")
        assert b.decrypt(ct, salt, aad=b"req-42:src=nodeA:dst=nodeB") == b"tensor-bytes"

    def test_wrong_aad_fails_to_decrypt(self):
        # THE regression: pre-fix this returned the plaintext (aad ignored).
        a, b = _pair()
        ct, salt = a.encrypt(b"secret-tensor", aad=b"context-one")
        with pytest.raises(E2EError):
            b.decrypt(ct, salt, aad=b"context-two")

    def test_omitting_aad_after_aad_encryption_fails(self):
        # Authentication cannot be stripped by simply not supplying the aad.
        a, b = _pair()
        ct, salt = a.encrypt(b"secret-tensor", aad=b"context-one")
        with pytest.raises(E2EError):
            b.decrypt(ct, salt)

    def test_ciphertexts_for_different_aads_diverge(self):
        a, _ = _pair()
        ct1, salt1 = a.encrypt(b"same-plaintext", aad=b"aad-A")
        ct2, salt2 = a.encrypt(b"same-plaintext", aad=b"aad-B")
        # Fresh salt+nonce already makes these differ; the meaningful
        # assertion is cross-decryption failure, covered above. Here we pin
        # that neither ciphertext decrypts under the OTHER aad.
        b_side = SessionKeys(a._shared_key, session_id="mirror")
        with pytest.raises(E2EError):
            b_side.decrypt(ct1, salt1, aad=b"aad-B")
        with pytest.raises(E2EError):
            b_side.decrypt(ct2, salt2, aad=b"aad-A")

    def test_no_aad_and_empty_aad_are_equivalent(self):
        a, b = _pair()
        ct, salt = a.encrypt(b"payload", aad=None)
        # Empty bytes means the same thing as None: no associated data.
        assert b.decrypt(ct, salt, aad=b"") == b"payload"

        ct2, salt2 = a.encrypt(b"payload", aad=b"")
        assert b.decrypt(ct2, salt2, aad=None) == b"payload"

    def test_str_aad_is_coerced_not_dropped(self):
        # A caller passing a str must get authentication, not silent ignore.
        a, b = _pair()
        ct, salt = a.encrypt(b"payload", aad="session-7")  # type: ignore[arg-type]
        assert b.decrypt(ct, salt, aad="session-7") == b"payload"  # type: ignore[arg-type]
        with pytest.raises(E2EError):
            b.decrypt(ct, salt, aad="session-8")

    def test_interleaved_aad_and_no_aad_traffic(self):
        # Mixed traffic must stay aligned (per-message salt already guarantees
        # this; verify AAD mixing did not break the resync property).
        a, b = _pair()
        msgs = [
            (f"m{i}".encode(), None if i % 3 == 0 else f"ctx-{i % 2}".encode())
            for i in range(30)
        ]
        for pt, aad in msgs:
            ct, salt = a.encrypt(pt, aad=aad)
            assert b.decrypt(ct, salt, aad=aad) == pt


class TestBackwardCompat:
    def test_legacy_ciphertext_without_aad_still_decrypts(self):
        # Old ciphertexts were produced with implicit no-AAD; they must keep
        # decrypting when aad=None is supplied.
        a, b = _pair()
        ct, salt = a.encrypt(b"legacy-frame")
        assert b.decrypt(ct, salt) == b"legacy-frame"
        assert b.decrypt(ct, salt, aad=None) == b"legacy-frame"
        assert b.decrypt(ct, salt, aad=b"") == b"legacy-frame"

    def test_no_aad_kdf_output_bit_for_bit_identical_to_pre_fix(self):
        # Pins the no-AAD derivation to the ORIGINAL construction
        # (Blake2b keyed PRF over exactly the salt) so a refactor can never
        # silently invalidate existing ciphertexts.
        shared = b"0123456789abcdef0123456789abcdef"
        sk = SessionKeys(shared, session_id="golden")
        salt = bytes(range(SALT_BYTES))
        expected = _nacl_bindings.crypto_generichash_blake2b_salt_personal(
            data=salt,
            key=shared,
            salt=b"\x00" * SALT_BYTES,
            person=b"distllm-kdf".ljust(16, b"\x00"),
            digest_size=KEY_BYTES,
        )
        assert sk.derive_box_key(salt=salt) == expected
        # With an aad bound, derivation MUST differ from the no-aad key.
        assert sk.derive_box_key(salt=salt, aad=b"x") != expected


class TestE2EEncryptionPassthrough:
    @staticmethod
    def _established_pair():
        from distllm.security.e2e import E2EEncryption

        node_a = E2EEncryption(cluster_key="cluster-secret-key-16")
        node_b = E2EEncryption(cluster_key="cluster-secret-key-16")
        pk_a, pk_b = node_a.get_public_key(), node_b.get_public_key()
        node_a.import_peer_key(pk_b)
        node_b.import_peer_key(pk_a)
        return node_a, node_b

    def test_controller_level_aad_roundtrip_and_reject(self):
        a, b = self._established_pair()
        ct, salt = a.encrypt(b"federated-tensor", aad=b"hop1>hop2:seq=17")

        assert b.decrypt(ct, salt, aad=b"hop1>hop2:seq=17") == b"federated-tensor"
        with pytest.raises(E2EError):
            b.decrypt(ct, salt, aad=b"hop2>hop1:seq=17")
        with pytest.raises(E2EError):
            b.decrypt(ct, salt)

    def test_tensor_payload_packet_unaffected_by_aad_support(self):
        # encrypt_tensor_payload/decrypt_tensor_payload keep their packet
        # layout ([salt:16][nonce+ct+tag]) whether or not aad is used.
        a, b = self._established_pair()
        pkt = a.encrypt_tensor_payload(b"raw-tensor", aad=b"sess-9")
        assert b.decrypt_tensor_payload(pkt, aad=b"sess-9") == b"raw-tensor"

        pkt_plain = a.encrypt_tensor_payload(b"raw-tensor")
        assert b.decrypt_tensor_payload(pkt_plain) == b"raw-tensor"
