"""Tests for KademliaDHT, focused on the P0 STORE-authorisation fix.

The DHT STORE RPC was previously unauthenticated: any peer that could reach the
UDP port could write arbitrary key->value bytes, poisoning peer-discovery and
KV-cache routing.  With a ``shared_secret`` configured, STORE now requires a
time-bound HMAC capability token (fail closed).
"""

from __future__ import annotations

import asyncio
import time

import pytest

from distllm.dist.p2p.kademlia_dht import (
    STORE_TOKEN_SKEW,
    STORE_TOKEN_TTL,
    KademliaDHT,
)

SECRET = "dht-test-shared-secret"
SENDER = "ab" * 32  # 64 hex chars = 32-byte node id


def _handle_store(dht, key="key", value_hex="00ff", token="", expires=None, sender=SENDER):
    return dht._handle_store(
        {
            "params": {"key": key, "value": value_hex, "token": token, "expires": expires},
            "sender": {"node_id": sender, "ip": "1.2.3.4", "port": 9999},
        },
        ("1.2.3.4", 9999),
    )


class TestStoreTokenGate:
    """With a shared secret, STORE must be authenticated."""

    def test_store_rejected_without_token(self):
        dht = KademliaDHT(shared_secret=SECRET)
        resp = asyncio.run(_handle_store(dht))
        assert resp["stored"] is False
        assert "key" not in dht._store

    def test_store_rejected_with_wrong_token(self):
        dht = KademliaDHT(shared_secret=SECRET)
        expires = int(time.time()) + 60
        resp = asyncio.run(_handle_store(dht, token="deadbeef", expires=expires))
        assert resp["stored"] is False
        assert "key" not in dht._store

    def test_store_rejected_with_expired_token(self):
        dht = KademliaDHT(shared_secret=SECRET)
        expires = int(time.time()) - STORE_TOKEN_SKEW - 10
        token = dht._make_store_token(SENDER, "key", "00ff", expires)
        resp = asyncio.run(_handle_store(dht, token=token, expires=expires))
        assert resp["stored"] is False

    def test_store_rejected_when_value_tampered(self):
        dht = KademliaDHT(shared_secret=SECRET)
        expires = int(time.time()) + 60
        token = dht._make_store_token(SENDER, "key", "00ff", expires)
        # Token was over value 00ff but the message carries 00fe.
        resp = asyncio.run(_handle_store(dht, value_hex="00fe", token=token, expires=expires))
        assert resp["stored"] is False

    def test_store_rejected_when_sender_spoofed(self):
        dht = KademliaDHT(shared_secret=SECRET)
        expires = int(time.time()) + 60
        token = dht._make_store_token(SENDER, "key", "00ff", expires)
        # Attacker impersonates a DIFFERENT sender id.
        resp = asyncio.run(_handle_store(dht, token=token, expires=expires, sender="cd" * 32))
        assert resp["stored"] is False

    def test_store_accepted_with_valid_token(self):
        dht = KademliaDHT(shared_secret=SECRET)
        expires = int(time.time()) + 60
        token = dht._make_store_token(SENDER, "key", "00ff", expires)
        resp = asyncio.run(_handle_store(dht, token=token, expires=expires))
        assert resp["stored"] is True
        assert dht._store["key"][0] == b"\x00\xff"

    def test_store_accepted_without_secret_legacy(self):
        # Backward compatibility: no secret -> unauthenticated mode preserved.
        dht = KademliaDHT()
        resp = asyncio.run(_handle_store(dht))
        assert resp["stored"] is True

    def test_token_verify_is_time_bound(self):
        dht = KademliaDHT(shared_secret=SECRET)
        expires = int(time.time()) - STORE_TOKEN_SKEW - 60
        token = dht._make_store_token(SENDER, "key", "00ff", expires)
        assert not dht._verify_store_token(SENDER, "key", "00ff", token, expires)

    def test_token_is_bound_to_key_and_value(self):
        dht = KademliaDHT(shared_secret=SECRET)
        expires = int(time.time()) + 60
        token = dht._make_store_token(SENDER, "key", "00ff", expires)
        # Same token must not authorise a different key.
        assert not dht._verify_store_token(SENDER, "other", "00ff", token, expires)
        assert not dht._verify_store_token(SENDER, "key", "00ee", token, expires)


class TestDHTRoundTrip:
    """Happy path still works: store + find_value on a single node."""

    def test_store_and_find_value_roundtrip(self):
        async def run():
            dht = KademliaDHT(shared_secret=SECRET)
            port = await dht.start(bind_addr="127.0.0.1", port=0)
            assert port > 0
            ok = await dht.store("hello", b"world")
            assert ok is True
            value = await dht.find_value("hello")
            assert value == b"world"
            await dht.stop()
            return True

        assert asyncio.run(run()) is True
