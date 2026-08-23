"""Regression: F-037 E2E ratchet must not diverge under asymmetric traffic.

The SessionKeys ratchet advances the local key on each local encrypt/decrypt.
With the salt-return fix, each encrypt returns the salt that ACTUALLY produced
the key, so the decrypt side derives its box key from the transmitted salt —
regardless of how many messages each side has sent/received.  This test drives
strongly asymmetric traffic (many tensors one way, few the other) and asserts
every message decrypts with no mid-stream failures.
"""

from __future__ import annotations

import pytest

from distllm.security.e2e import SessionKeys


def _pair():
    shared = b"0123456789abcdef0123456789abcdef"  # exactly 32 bytes
    a = SessionKeys(shared, session_id="s1")
    b = SessionKeys(shared, session_id="s1")
    return a, b


class TestAsymmetricRatchet:
    def test_asymmetric_traffic_never_diverges(self):
        a, b = _pair()

        # Node A sends MANY messages; node B sends FEW (asymmetric, like
        # token-streaming: many small tensors one way, few responses the other).
        sent = []
        for i in range(35):  # > RATCHET_INTERVAL (10) in one direction
            ct, salt = a.encrypt(f"payload-{i}".encode())
            sent.append((ct, salt))
            # B decodes each as it arrives.
            plain = b.decrypt(ct, salt)
            assert plain == f"payload-{i}".encode(), f"direction A->B failed at {i}"

        # B sends only 2 messages back.
        for i in range(2):
            ct, salt = b.encrypt(f"reply-{i}".encode())
            plain = a.decrypt(ct, salt)
            assert plain == f"reply-{i}".encode(), f"direction B->A failed at {i}"

        # After the asymmetry, both directions must STILL work (keys re-aligned
        # by the per-message salt).
        ct, salt = a.encrypt(b"final")
        assert b.decrypt(ct, salt) == b"final"

    def test_each_encrypt_returns_the_key_producing_salt(self):
        a, b = _pair()
        ct1, salt1 = a.encrypt(b"x")
        # The salt returned must be the pre-ratchet salt, so the peer using it
        # decrypts successfully even right after a ratchet boundary.
        for i in range(9):
            a.encrypt(b"warm")
        ct2, salt2 = a.encrypt(b"boundary")  # 11th encrypt -> ratchet fires
        assert b.decrypt(ct2, salt2) == b"boundary"
        assert b.decrypt(ct1, salt1) == b"x"  # earlier message still decrypts