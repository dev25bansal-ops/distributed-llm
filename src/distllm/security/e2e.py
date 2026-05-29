"""NaCl/libsodium end-to-end encryption for tensor transport.

Provides application-level encryption of tensor bytes between nodes
using X25519 key exchange + XSalsa20-Poly1305 authenticated encryption
(NaCl SecretBox). Designed for multi-hop federation: each hop uses its
own session key, so intermediate nodes cannot read plaintext tensors.

Flow:
  1. Key exchange: Both nodes generate ephemeral X25519 keypairs.
     Public keys are exchanged via the existing signaling channel
     (gRPC metadata or HTTP headers), authenticated with the shared
     ``cluster_key`` via HMAC-SHA256.
  2. Session key: Each side derives the shared symmetric key via
     ``crypto_box_beforenm`` (ECDH) and wraps it in a ``SecretBox``
     for fast symmetric encryption.
  3. Encrypt/Decrypt: Every tensor payload is encrypted with the
     session's SecretBox before entering the transport layer.

Usage:
    from distllm.security.e2e import E2EEncryption

    # On both nodes, after establishing a shared cluster_key:
    e2e = E2EEncryption(cluster_key="shared-secret")

    # Node A: export public key and send to Node B
    pub_a = e2e.get_public_key()

    # Node B: import Node A's public key, derive session
    e2e.import_peer_key(pub_a)
    pub_b = e2e.get_public_key()  # send back to Node A

    # Node A: import Node B's public key
    e2e.import_peer_key(pub_b)

    # Now both sides can encrypt/decrypt
    ciphertext, salt = e2e.encrypt(b"raw tensor bytes")
    plaintext = e2e.decrypt(ciphertext, salt)

    # Or use the convenience wrapper for TensorProto raw_data:
    from distllm.security.e2e import encrypt_tensor_payload
    encrypted = encrypt_tensor_payload(raw_bytes, session)
"""

from __future__ import annotations

import hashlib
import hmac
import os
import struct
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

HAS_NACL = False
try:
    import nacl.bindings as nacl_bindings
    import nacl.exceptions as nacl_exceptions
    from nacl.bindings import (
        crypto_box_beforenm,
        crypto_box_keypair,
        crypto_secretbox,
        crypto_secretbox_open,
    )

    HAS_NACL = True
except ImportError:
    nacl_bindings = None
    nacl_exceptions = None


SALT_BYTES = 16
NONCE_BYTES = 24
KEY_BYTES = 32
TAG_BYTES = 16


class E2EError(Exception):
    pass


class SessionKeys:
    """Holds key material for a single encrypted session between two nodes.

    Attributes:
        session_id: Short identifier for this session.
        shared_key: 32-byte symmetric key derived from ECDH.
        salt: Optional salt for key derivation (improves kDF).
    """

    def __init__(self, shared_key: bytes, session_id: str = "", salt: bytes | None = None):
        if len(shared_key) != KEY_BYTES:
            raise ValueError(f"shared_key must be {KEY_BYTES} bytes, got {len(shared_key)}")
        self.session_id = session_id or hashlib.sha256(shared_key).hexdigest()[:8]
        self._shared_key = shared_key
        self._salt = salt or os.urandom(SALT_BYTES)
        self._seq: int = 0

    def derive_box_key(self, salt: bytes | None = None) -> bytes:
        """Derive the per-session symmetric key using a proper KDF.

        Uses libsodium's ``crypto_generichash`` (Blake2b) in keyed PRF
        mode when PyNaCl is available — a proper replacement for the
        original ``SHA256(shared_key || salt)`` construction.  Falls
        back to HKDF-SHA256 per :rfc:`5869` using only stdlib
        ``hashlib`` + ``hmac``.

        Args:
            salt: Domain-separation salt.  Defaults to the session salt.
        """
        salt = salt if salt is not None else self._salt
        if HAS_NACL:
            # Blake2b keyed PRF — the recommended KDF in libsodium.
            return nacl.bindings.crypto_generichash(
                KEY_BYTES,
                salt,
                key=self._shared_key,
            )
        # Fallback: HKDF-SHA256 (RFC 5869) when PyNaCl is unavailable.
        prk = hmac.new(salt, self._shared_key, hashlib.sha256).digest()
        t = b""
        okm = b""
        hash_len = hashlib.sha256().digest_size
        n = (KEY_BYTES + hash_len - 1) // hash_len
        for i in range(1, n + 1):
            t = hmac.new(prk, t + b"distllm-e2e-key" + bytes([i]), hashlib.sha256).digest()
            okm += t
        return okm[:KEY_BYTES]

    def encrypt(self, plaintext: bytes, aad: bytes | None = None) -> tuple[bytes, bytes]:
        """Encrypt *plaintext* with ``crypto_secretbox``.

        Returns (ciphertext_with_nonce, salt). The nonce is prepended to
        the ciphertext so the recipient doesn't need to manage it separately.
        """
        if not HAS_NACL:
            raise E2EError("PyNaCl not installed; cannot encrypt")

        key = self.derive_box_key()
        nonce = os.urandom(NONCE_BYTES)
        ct_and_tag = crypto_secretbox(plaintext, nonce, key)
        return nonce + ct_and_tag, self._salt

    def decrypt(self, ciphertext: bytes, salt: bytes | None = None) -> bytes:
        """Decrypt *ciphertext* previously encrypted by the paired node.

        Expects the first ``NONCE_BYTES`` (24) bytes to be the nonce,
        followed by the ciphertext + MAC tag.

        Args:
            ciphertext: nonce + ciphertext + MAC tag.
            salt: Salt used during encryption. Defaults to session salt.

        Returns:
            Decrypted plaintext bytes.
        """
        if not HAS_NACL:
            raise E2EError("PyNaCl not installed; cannot decrypt")

        key = self.derive_box_key(salt=salt)

        if len(ciphertext) < NONCE_BYTES + TAG_BYTES:
            raise E2EError("Ciphertext too short")

        nonce = ciphertext[:NONCE_BYTES]
        ct = ciphertext[NONCE_BYTES:]

        try:
            plaintext = crypto_secretbox_open(ct, nonce, key)
        except nacl_exceptions.CryptoError as e:
            raise E2EError(f"Decryption failed (wrong key or tampered data): {e}") from e

        return plaintext


@dataclass
class E2EEncryption:
    """Main E2E encryption controller.

    Manages key exchange and per-session encryption of tensor payloads
    between two nodes.

    Usage:
        e2e = E2EEncryption(cluster_key="shared-secret")

        # Side A:
        pub_a = e2e.get_public_key()       # export
        e2e.import_peer_key(pub_b)         # import peer's key

        # Side B:
        pub_b = e2e.get_public_key()
        e2e.import_peer_key(pub_a)

        # Both sides can now encrypt/decrypt
        ct, salt = e2e.encrypt(b"data")
        pt = e2e.decrypt(ct, salt)
    """

    cluster_key: str = ""
    node_id: str = ""
    _private_key: bytes | None = None
    _public_key: bytes | None = None
    _peer_public_key: bytes | None = None
    _session: SessionKeys | None = None

    def __post_init__(self) -> None:
        if HAS_NACL:
            self._generate_keypair()
            self._session = None

    def _generate_keypair(self) -> None:
        """Generate an ephemeral X25519 keypair for this session."""
        if not HAS_NACL:
            raise E2EError("PyNaCl not installed")
        pk, sk = crypto_box_keypair()
        self._public_key = pk
        self._private_key = sk

    def get_public_key(self) -> bytes:
        """Return this node's public key for transmission to the peer."""
        if self._public_key is None:
            raise E2EError("No keypair generated")
        return self._public_key

    def get_signed_public_key(self) -> bytes:
        """Return public key signed with HMAC-SHA256 using cluster_key.

        The signature is prepended to allow the peer to verify authenticity.
        """
        pk = self.get_public_key()
        signature = hmac.new(
            self.cluster_key.encode(),
            pk,
            hashlib.sha256,
        ).digest()
        return signature + pk

    def import_signed_public_key(self, signed_data: bytes) -> None:
        """Import and verify a peer's signed public key.

        Raises:
            E2EError: If signature verification fails.
        """
        if len(signed_data) < 32:
            raise E2EError("Signed public key too short")
        signature = signed_data[:32]
        peer_pk = signed_data[32:]
        expected_sig = hmac.new(
            self.cluster_key.encode(),
            peer_pk,
            hashlib.sha256,
        ).digest()
        if not hmac.compare_digest(signature, expected_sig):
            raise E2EError("Peer public key signature verification failed")
        self.import_peer_key(peer_pk)

    def import_peer_key(self, peer_public_key: bytes) -> None:
        """Import the peer's public key and derive the shared session key.

        Once both sides have called this with each other's public key,
        the session is established and ``encrypt()``/``decrypt()`` work.
        """
        if not HAS_NACL:
            raise E2EError("PyNaCl not installed")
        if self._private_key is None:
            raise E2EError("No local keypair; generate one first")

        self._peer_public_key = peer_public_key
        shared_key = crypto_box_beforenm(peer_public_key, self._private_key)
        session_id_bytes = hashlib.sha256(shared_key).digest()[:4]
        self._session = SessionKeys(
            shared_key=shared_key,
            session_id=session_id_bytes.hex(),
        )
        logger.debug(
            f"E2E session established: {self._session.session_id}"
        )

    @property
    def is_established(self) -> bool:
        return self._session is not None

    @property
    def session_id(self) -> str:
        if self._session is None:
            return ""
        return self._session.session_id

    def encrypt(self, plaintext: bytes) -> tuple[bytes, bytes]:
        """Encrypt *plaintext* using the established session key.

        Returns:
            Tuple of (ciphertext, salt). Both must be sent to the peer.
        """
        if self._session is None:
            raise E2EError("Session not established; exchange keys first")
        return self._session.encrypt(plaintext)

    def decrypt(self, ciphertext: bytes, salt: bytes | None = None) -> bytes:
        """Decrypt *ciphertext* using the established session key."""
        if self._session is None:
            raise E2EError("Session not established; exchange keys first")
        return self._session.decrypt(ciphertext, salt=salt)

    def encrypt_tensor_payload(self, raw_tensor_bytes: bytes) -> bytes:
        """Encrypt raw tensor bytes and return a self-contained packet.

        The packet includes the salt, nonce, ciphertext, and MAC tag so the
        recipient only needs the session to decrypt.

        Packet format:
          [salt:16B][nonce+ciphertext+tag:...]
        """
        if not HAS_NACL:
            return raw_tensor_bytes

        ct_with_nonce, salt = self.encrypt(raw_tensor_bytes)
        return struct.pack("!16s", salt) + ct_with_nonce

    def decrypt_tensor_payload(self, encrypted_packet: bytes) -> bytes:
        """Decrypt a self-contained packet produced by ``encrypt_tensor_payload``.

        Returns the original plaintext tensor bytes.
        """
        if not HAS_NACL or len(encrypted_packet) < SALT_BYTES + NONCE_BYTES + TAG_BYTES:
            return encrypted_packet

        salt = encrypted_packet[:SALT_BYTES]
        ct_with_nonce = encrypted_packet[SALT_BYTES:]
        return self.decrypt(ct_with_nonce, salt=salt)

    def reset(self) -> None:
        """Generate a fresh keypair and discard the session."""
        if HAS_NACL:
            self._generate_keypair()
        self._peer_public_key = None
        self._session = None


def encrypt_tensor_payload(
    raw_bytes: bytes, e2e: E2EEncryption | None
) -> bytes:
    """Convenience wrapper: encrypt raw tensor bytes if E2E is active.

    If ``e2e`` is ``None`` or the session is not established, returns
    *raw_bytes* unmodified.
    """
    if e2e is None or not e2e.is_established:
        return raw_bytes
    return e2e.encrypt_tensor_payload(raw_bytes)


def decrypt_tensor_payload(
    encrypted_bytes: bytes, e2e: E2EEncryption | None
) -> bytes:
    """Convenience wrapper: decrypt tensor bytes if E2E is active.

    If ``e2e`` is ``None`` or the session is not established, returns
    *encrypted_bytes* unmodified.
    """
    if e2e is None or not e2e.is_established:
        return encrypted_bytes
    return e2e.decrypt_tensor_payload(encrypted_bytes)
