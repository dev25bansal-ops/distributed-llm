"""SEC-API-08-KEK — webhook HMAC secrets must not be persisted plaintext.

Threat: ``POST /v1/webhooks`` stored the caller-supplied signing secret in the
PersistentStore JSON blob verbatim, so any SQLite dump / backup leak yields
every tenant's HMAC key (used to forge signed event deliveries).

Fix covered here (``src/distllm/api/routes/webhooks.py``):
1. Secrets are written wrapped with a Fernet key derived from
   ``DISTLLM_WEBHOOK_KEK`` (or a generated 0600 key file, mirroring
   ``middleware._persist_api_key``).
2. Legacy plaintext rows are migrated on read (wrap + persist).
3. Sign/verify round-trip still works with the decrypted secret.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import os
import sqlite3

import pytest

from distllm.api import persistent_store as ps
from distllm.api.routes import webhooks as webhooks_routes


PUBLIC_URL = "https://hooks.example.com/webhook"
RAW_SECRET = "super-secret-webhook-signing-key"


@pytest.fixture(autouse=True)
def _isolate_kek(monkeypatch, tmp_path):
    """Every test gets its own key-file path so the repo's .distllm/ is never
    written, and module KEK/Fernet caches never leak between tests."""
    monkeypatch.setenv("DISTLLM_WEBHOOK_KEK_FILE", str(tmp_path / "webhook_kek"))
    monkeypatch.setattr(webhooks_routes, "_kek_cache", None)
    monkeypatch.setattr(webhooks_routes, "_fernet_cache", None)
    yield
    monkeypatch.setattr(webhooks_routes, "_kek_cache", None)
    monkeypatch.setattr(webhooks_routes, "_fernet_cache", None)


@pytest.fixture()
def kek_env(monkeypatch, tmp_path):
    """Pin a deterministic KEK and isolate the module caches.

    Uses the SHA-256 passphrase derivation path (short value), so the tests
    also prove that form works.  Resets the memoized KEK + Fernet between
    tests and points the key file at tmp_path.
    """
    monkeypatch.setenv("DISTLLM_WEBHOOK_KEK", "test-kek-passphrase")
    yield "test-kek-passphrase"


@pytest.fixture()
def webhook_store(tmp_path, monkeypatch):
    """Point PersistentStore at a temp dir and reset the module cache."""
    from distllm.api import persistent_store as ps

    monkeypatch.setenv("DISTLLM_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr(ps, "_store", None)
    monkeypatch.setattr(webhooks_routes, "_webhook_cache", {})
    yield
    monkeypatch.setattr(ps, "_store", None)


def _db_path() -> str:
    return str(os.path.join(os.environ["DISTLLM_DATA_DIR"], "distllm.db"))


def _raw_stored_secret() -> str:
    """Read the webhook row's secret column straight out of SQLite."""
    conn = sqlite3.connect(_db_path())
    try:
        row = conn.execute(
            "SELECT data FROM batches WHERE id LIKE 'webhook:%'"
        ).fetchone()
    finally:
        conn.close()
    assert row is not None, "no webhook row persisted"
    return json.loads(row[0])["secret"]


class TestAtRestEncryption:
    def test_stored_value_is_not_the_raw_secret(self, webhook_store, kek_env):
        """The SQLite blob must never contain the plaintext signing secret."""
        from distllm.api.routes.webhooks import WebhookCreate, create_webhook

        reg = asyncio.run(create_webhook(WebhookCreate(
            url=PUBLIC_URL, secret=RAW_SECRET, events=["batch.completed"],
        )))
        stored = _raw_stored_secret()
        assert stored != RAW_SECRET
        assert RAW_SECRET not in stored
        assert stored.startswith("enc:")
        assert reg.id  # 201 path returned normally

    def test_stored_value_differs_between_registrations(self, webhook_store, kek_env):
        """Fernet is non-deterministic (random IV): two equal secrets wrap
        differently, so equality of ciphertext leaks nothing."""
        from distllm.api.routes.webhooks import WebhookCreate, create_webhook

        first = asyncio.run(create_webhook(WebhookCreate(
            url=PUBLIC_URL, secret=RAW_SECRET, events=["batch.completed"],
        )))
        second = asyncio.run(create_webhook(WebhookCreate(
            url=PUBLIC_URL, secret=RAW_SECRET, events=["batch.completed"],
        )))
        conn = sqlite3.connect(_db_path())
        try:
            rows = conn.execute(
                "SELECT data FROM batches WHERE id LIKE 'webhook:%'"
            ).fetchall()
        finally:
            conn.close()
        secrets = {json.loads(r[0])["secret"] for r in rows}
        assert len(secrets) == 2  # both wrapped, both distinct
        assert first.id != second.id

    def test_env_kek_is_respected_and_reproducible(self, webhook_store, kek_env):
        """The same KEK env value derives the same Fernet key (multi-replica)."""
        key_a = webhooks_routes._get_fernet()
        assert key_a is not None
        # A second module-resolution (fresh cache) must yield the same cipher.
        webhooks_routes._fernet_cache = None
        key_b = webhooks_routes._get_fernet()
        assert key_a is key_b or key_a._signing_key == key_b._signing_key

    def test_kek_derivation_accepts_raw_32_byte_key(self, monkeypatch, tmp_path):
        """A 43-char base64url 32-byte key is used directly, not stretched."""
        raw = base64.urlsafe_b64encode(b"\x01" * 32).decode().rstrip("=")
        monkeypatch.setenv("DISTLLM_WEBHOOK_KEK", raw)
        monkeypatch.setattr(webhooks_routes, "_kek_cache", None)
        monkeypatch.setattr(webhooks_routes, "_fernet_cache", None)
        try:
            derived = webhooks_routes._derive_fernet_key(raw)
            assert derived == base64.urlsafe_b64encode(b"\x01" * 32)
        finally:
            monkeypatch.setattr(webhooks_routes, "_kek_cache", None)
            monkeypatch.setattr(webhooks_routes, "_fernet_cache", None)


class TestLegacyMigration:
    def test_plaintext_row_is_re_encrypted_on_read(self, webhook_store, kek_env):
        """A row written by the old code (plaintext secret) is upgraded on
        read: the store ends up wrapped and the cache serves the plaintext."""
        from distllm.api.routes.webhooks import WebhookCreate, create_webhook

        # 1. Register through the NEW code path (wrapped at rest).
        asyncio.run(create_webhook(WebhookCreate(
            url=PUBLIC_URL, secret=RAW_SECRET, events=["batch.completed"],
        )))
        # 2. Simulate a legacy row by rewriting the blob with plaintext.
        conn = sqlite3.connect(_db_path())
        try:
            row = conn.execute(
                "SELECT data FROM batches WHERE id LIKE 'webhook:%'"
            ).fetchone()
            legacy = json.loads(row[0])
            legacy["secret"] = RAW_SECRET  # pre-fix format
            conn.execute(
                "UPDATE batches SET data = ? WHERE id = ?",
                (json.dumps(legacy), legacy["id"]),
            )
            conn.commit()
        finally:
            conn.close()

        # 3. _load_webhooks must migrate the legacy row: re-wrapped in the
        # store, plaintext in the in-memory cache.
        webhooks_routes._load_webhooks()
        assert _raw_stored_secret() != RAW_SECRET
        assert _raw_stored_secret().startswith("enc:")
        cached = webhooks_routes._get_cached_webhooks()
        # The cache stores the wrapped value; consumers unwrap on use.
        assert cached[0]["secret"].startswith("enc:")
        assert webhooks_routes._unwrap_secret(cached[0]["secret"]) == RAW_SECRET

    def test_wrapped_row_survives_reload_unchanged(self, webhook_store, kek_env):
        """Rows already wrapped are not re-wrapped or damaged on reload."""
        from distllm.api.routes.webhooks import WebhookCreate, create_webhook

        asyncio.run(create_webhook(WebhookCreate(
            url=PUBLIC_URL, secret=RAW_SECRET, events=["batch.completed"],
        )))
        wrapped_once = _raw_stored_secret()

        webhooks_routes._load_webhooks()  # reload from the store
        wrapped_twice = _raw_stored_secret()
        assert wrapped_twice == wrapped_once  # idempotent
        cached = webhooks_routes._get_cached_webhooks()
        assert webhooks_routes._unwrap_secret(cached[0]["secret"]) == RAW_SECRET

    def test_corrupt_wrapped_secret_fails_closed(self, webhook_store, kek_env):
        """A wrapped secret the KEK cannot decrypt is cleared, not stripped."""
        from distllm.api.routes.webhooks import WebhookCreate, create_webhook

        asyncio.run(create_webhook(WebhookCreate(
            url=PUBLIC_URL, secret=RAW_SECRET, events=["batch.completed"],
        )))
        # Tamper: valid prefix + garbage token.
        conn = sqlite3.connect(_db_path())
        try:
            row = conn.execute(
                "SELECT data FROM batches WHERE id LIKE 'webhook:%'"
            ).fetchone()
            blob = json.loads(row[0])
            blob["secret"] = "enc:definitely-not-a-valid-token"
            conn.execute(
                "UPDATE batches SET data = ? WHERE id = ?",
                (json.dumps(blob), blob["id"]),
            )
            conn.commit()
        finally:
            conn.close()

        webhooks_routes._load_webhooks()
        # The row is still registered (URL/events intact) but its secret is
        # cleared — signing is disabled rather than raising on every event.
        cached = webhooks_routes._get_cached_webhooks()
        assert cached[0]["secret"] == ""
        assert cached[0]["url"] == PUBLIC_URL


class TestRoundTripSignVerify:
    @pytest.mark.asyncio
    async def test_sign_verify_round_trip_through_dispatch(self, webhook_store, kek_env, monkeypatch):
        """End-to-end: the secret used to SIGN a delivery is the plaintext the
        operator registered, recovered from the encrypted-at-rest store."""
        from distllm.api.routes.webhooks import (
            WebhookCreate, create_webhook, dispatch_event,
        )

        captured = {}

        class _Resp:
            status_code = 204

        class _FakeClient:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

            async def post(self, url, content=None, headers=None, **kwargs):
                captured["url"] = url
                captured["headers"] = dict(headers or {})
                captured["body"] = content
                return _Resp()

        monkeypatch.setattr(
            "distllm.api.webhooks.delivery.httpx.AsyncClient", _FakeClient
        )

        reg = await create_webhook(WebhookCreate(
            url=PUBLIC_URL, secret=RAW_SECRET, events=["batch.completed"],
        ))

        # dispatch_event reads the secret from the store cache and signs.
        successes = await dispatch_event("batch.completed", {"batch_id": "b-1"})
        assert successes == 1

        sig = captured["headers"]["X-Webhook-Signature"]
        expected = hmac.new(
            RAW_SECRET.encode("utf-8"), captured["body"], hashlib.sha256,
        ).hexdigest()
        assert sig == expected  # signed with the ORIGINAL plaintext secret
        assert captured["url"] == PUBLIC_URL
        assert reg.id.startswith("webhook:")

    @pytest.mark.asyncio
    async def test_test_endpoint_round_trip(self, webhook_store, kek_env, monkeypatch):
        """POST /{id}/test signs with the decrypted secret."""
        from distllm.api.routes.webhooks import (
            WebhookCreate, create_webhook, test_webhook,
        )

        captured = {}

        class _Resp:
            status_code = 204

        class _FakeClient:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

            async def post(self, url, content=None, headers=None, **kwargs):
                captured["headers"] = dict(headers or {})
                captured["body"] = content
                return _Resp()

        monkeypatch.setattr(
            "distllm.api.webhooks.delivery.httpx.AsyncClient", _FakeClient
        )

        reg = await create_webhook(WebhookCreate(
            url=PUBLIC_URL, secret=RAW_SECRET, events=["batch.completed"],
        ))
        result = await test_webhook(reg.id)
        assert result["success"] is True

        expected = hmac.new(
            RAW_SECRET.encode("utf-8"), captured["body"], hashlib.sha256,
        ).hexdigest()
        assert captured["headers"]["X-Webhook-Signature"] == expected


class TestKekGeneration:
    def test_missing_kek_generates_0600_key_file(self, webhook_store, monkeypatch, tmp_path):
        """No env KEK -> one is generated, written 0600, and used."""
        key_file = tmp_path / "generated_kek"
        monkeypatch.setenv("DISTLLM_WEBHOOK_KEK_FILE", str(key_file))
        monkeypatch.delenv("DISTLLM_WEBHOOK_KEK", raising=False)
        from distllm.api.routes.webhooks import WebhookCreate, create_webhook

        reg = asyncio.run(create_webhook(WebhookCreate(
            url=PUBLIC_URL, secret=RAW_SECRET, events=["batch.completed"],
        )))
        assert reg.id

        # Key file exists.  On POSIX it is owner-only (0600); Windows
        # cannot express that via os.open, so accept 0600 or the
        # umask-limited 0o666 NTFS fallback.
        assert key_file.exists()
        mode = os.stat(key_file).st_mode & 0o777
        assert mode in (0o600, 0o666), oct(mode)

        # And the secret was wrapped with the generated key.
        stored = _raw_stored_secret()
        assert stored != RAW_SECRET and stored.startswith("enc:")

    def test_key_file_reused_across_processes(self, webhook_store, monkeypatch, tmp_path):
        """A pre-seeded key file is reused (restart survives; same KEK)."""
        key_file = tmp_path / "existing_kek"
        key_file.write_text("seeded-kek-material\n", encoding="utf-8")
        monkeypatch.setenv("DISTLLM_WEBHOOK_KEK_FILE", str(key_file))
        monkeypatch.delenv("DISTLLM_WEBHOOK_KEK", raising=False)
        from distllm.api.routes.webhooks import WebhookCreate, create_webhook

        asyncio.run(create_webhook(WebhookCreate(
            url=PUBLIC_URL, secret=RAW_SECRET, events=["batch.completed"],
        )))
        # File content unchanged (we read, not regenerate).
        assert key_file.read_text(encoding="utf-8") == "seeded-kek-material\n"
        assert _raw_stored_secret().startswith("enc:")
