"""Tests for KademliaDHT STORE-authorisation.

The DHT STORE RPC was previously unauthenticated by default: any peer that
could reach the UDP port could write arbitrary key->value bytes, poisoning
peer-discovery and KV-cache routing.  Behaviour now:

- With a ``shared_secret`` (constructor arg or ``DISTLLM_DHT_SECRET`` env var):
  STORE requires a time-bound HMAC capability token (fail closed).
- Without any secret (default): external STORE requests are REJECTED unless
  ``allow_unauthenticated=True`` is passed explicitly (legacy opt-in).
"""

from __future__ import annotations

import asyncio
import time

import loguru

from distllm.dist.p2p.kademlia_dht import (
    DHT_SECRET_ENV_VAR,
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


class _LogSink:
    """Capture loguru records (caplog cannot see loguru)."""

    def __init__(self) -> None:
        self.records: list = []
        self._handler_id = loguru.logger.add(
            lambda m: self.records.append(m.record), level="WARNING"
        )

    @property
    def messages(self) -> list[str]:
        # loguru records are dicts.
        return [r["message"] for r in self.records]

    def close(self) -> None:
        loguru.logger.remove(self._handler_id)


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

    def test_ttl_constant_used_for_client_tokens(self):
        # Client-side store() issues tokens with the configured TTL window.
        assert STORE_TOKEN_TTL > 0


class TestFailClosedDefault:
    """Without a secret, external STORE is rejected by default (fail closed)."""

    def test_unsigned_store_rejected_by_default(self):
        dht = KademliaDHT()
        resp = asyncio.run(_handle_store(dht))
        assert resp["stored"] is False
        assert "key" not in dht._store

    def test_unsigned_store_rejected_with_explicit_false(self):
        dht = KademliaDHT(allow_unauthenticated=False)
        resp = asyncio.run(_handle_store(dht))
        assert resp["stored"] is False
        assert "key" not in dht._store

    def test_bogus_token_also_rejected(self):
        # A garbage token must not smuggle a write through either.
        dht = KademliaDHT()
        resp = asyncio.run(_handle_store(dht, token="cafebabe", expires=int(time.time()) + 60))
        assert resp["stored"] is False

    def test_verify_fails_closed(self):
        dht = KademliaDHT()
        assert dht._verify_store_token(SENDER, "key", "00ff", "", None) is False

    def test_make_token_empty_without_secret(self):
        dht = KademliaDHT()
        assert dht._make_store_token(SENDER, "key", "00ff", 12345) == ""

    def test_default_warns_about_fail_closed_mode(self):
        sink = _LogSink()
        try:
            KademliaDHT()
            assert any("REJECTED" in m and "fail-closed" in m for m in sink.messages)
            assert any(DHT_SECRET_ENV_VAR in m for m in sink.messages)
        finally:
            sink.close()


class TestLegacyOptIn:
    """allow_unauthenticated=True restores the old open behaviour, loudly."""

    def test_legacy_accepts_unsigned_store(self):
        dht = KademliaDHT(allow_unauthenticated=True)
        resp = asyncio.run(_handle_store(dht))
        assert resp["stored"] is True
        assert dht._store["key"][0] == b"\x00\xff"

    def test_verify_passes_in_legacy_mode(self):
        dht = KademliaDHT(allow_unauthenticated=True)
        assert dht._verify_store_token(SENDER, "key", "00ff", "", None) is True

    def test_legacy_mode_warns_loudly_at_init(self):
        sink = _LogSink()
        try:
            KademliaDHT(allow_unauthenticated=True)
            assert any(
                "UNAUTHENTICATED" in m and "allow_unauthenticated=True" in m
                for m in sink.messages
            )
        finally:
            sink.close()

    def test_secret_overrides_legacy_flag(self):
        # A real secret takes precedence: tokens are enforced even if the
        # legacy flag was passed (misconfiguration cannot silently reopen auth).
        dht = KademliaDHT(shared_secret=SECRET, allow_unauthenticated=True)
        resp = asyncio.run(_handle_store(dht))
        assert resp["stored"] is False


class TestSecretFromEnv:
    """DISTLLM_DHT_SECRET provides the default shared secret."""

    def test_env_secret_enables_auth(self, monkeypatch):
        monkeypatch.setenv(DHT_SECRET_ENV_VAR, SECRET)
        dht = KademliaDHT()
        assert dht._shared_secret == SECRET
        # Unsigned STORE rejected; signed accepted.
        assert asyncio.run(_handle_store(dht))["stored"] is False
        expires = int(time.time()) + 60
        token = dht._make_store_token(SENDER, "key", "00ff", expires)
        assert asyncio.run(_handle_store(dht, token=token, expires=expires))["stored"] is True

    def test_explicit_arg_wins_over_env(self, monkeypatch):
        monkeypatch.setenv(DHT_SECRET_ENV_VAR, "env-secret-value")
        dht = KademliaDHT(shared_secret="arg-secret")
        assert dht._shared_secret == "arg-secret"

    def test_explicit_empty_disables_env_secret(self, monkeypatch):
        # Passing shared_secret="" opts out of the env-provided secret
        # (e.g. an offline single-node deployment).
        monkeypatch.setenv(DHT_SECRET_ENV_VAR, SECRET)
        dht = KademliaDHT(shared_secret="")
        assert dht._shared_secret == ""
        # ...and the node therefore fails closed for external stores.
        assert asyncio.run(_handle_store(dht))["stored"] is False


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

    def test_roundtrip_with_env_secret(self, monkeypatch):
        async def run():
            monkeypatch.setenv(DHT_SECRET_ENV_VAR, SECRET)
            dht = KademliaDHT(host="127.0.0.1", port=0)
            port = await dht.start(bind_addr="127.0.0.1", port=0)
            assert port > 0
            ok = await dht.store("hello", b"world")
            assert ok is True
            value = await dht.find_value("hello")
            assert value == b"world"
            await dht.stop()
            return True

        assert asyncio.run(run()) is True
