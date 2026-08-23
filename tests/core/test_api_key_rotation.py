"""Regression tests: ApiKeyRotator actually retires the old key.

P0 finding: ``ApiKeyRotator.rotate()`` registered a replacement key but never
retired the OLD key — it stayed in ``ApiKeyStore._keys`` (and therefore
authenticated) forever, because ``cleanup_expired()`` was never scheduled and
the auth boundary had no expiry concept. A rotated/compromised key stayed valid
indefinitely, defeating rotation.

Fix: the store now carries a per-key ``expires_at`` and ``authenticate()``
rejects a retired key once its grace passes (the authoritative boundary);
``rotate()`` marks the old key as expiring; a background cleanup loop (started
lazily on rotate) purges expired entries.
"""

from __future__ import annotations

import threading
import time

from distllm.core.api_key_store import ApiKeyStore
from distllm.core.cert_rotation import ApiKeyRotator


def _store_with_key(key: str, key_id: str = "k1", role: str = "admin") -> ApiKeyStore:
    store = ApiKeyStore()
    store.add_key(key, role=role, key_id=key_id, label="rotation-test")
    return store


class TestStoreRetirement:
    def test_retire_key_marks_expiry_and_auth_rejects_after_deadline(self):
        store = _store_with_key("old-token")
        # Within grace -> still authenticates.
        store.retire_key("k1", time.time() + 3600)
        assert store.authenticate("old-token") is not None
        # Deadline passed -> rejected at the auth boundary.
        store.retire_key("k1", time.time() - 1)
        assert store.authenticate("old-token") is None

    def test_remove_expired_purges_retired_keys(self):
        store = _store_with_key("old-token")
        old_hash = store.get_key_hash("k1")
        # Retire only the OLD key (by hash), then add the replacement.
        store.retire_key_hash(old_hash, time.time() - 1)
        store.add_key("new-token", key_id="k1", role="admin")
        removed = store.remove_expired()
        assert removed == 1
        assert store.authenticate("old-token") is None
        assert store.authenticate("new-token") is not None

    def test_normal_keys_unaffected_by_expiry_check(self):
        store = _store_with_key("stable-token")
        store.add_key("other-token", key_id="k2", role="read-only")
        assert store.authenticate("stable-token") is not None
        assert store.authenticate("other-token") is not None


class TestApiKeyRotator:
    def test_rotate_returns_new_key_and_old_stays_valid_during_grace(self):
        store = _store_with_key("old-token")
        rotator = ApiKeyRotator(store, grace_period_hours=24.0)
        new_key = rotator.rotate("k1")

        assert new_key and new_key != "old-token"
        # Both old (grace) and new authenticate after rotation.
        assert store.authenticate("old-token") is not None
        assert store.authenticate(new_key) is not None
        assert store.authenticate(new_key)[0] == "k1"

    def test_old_key_rejected_after_grace_elapses(self):
        store = _store_with_key("old-token")
        # ~0.15s grace: tiny so the deadline passes during the test.
        rotator = ApiKeyRotator(store, grace_period_hours=0.00002)
        new_key = rotator.rotate("k1")
        assert new_key

        time.sleep(0.2)
        # The auth boundary now rejects the retired key.
        assert store.authenticate("old-token") is None
        assert store.authenticate(new_key) is not None

    def test_cleanup_expired_removes_old_key_from_store(self):
        store = _store_with_key("old-token")
        old_hash = store.get_key_hash("k1")
        rotator = ApiKeyRotator(store, grace_period_hours=0.00002)
        new_key = rotator.rotate("k1")

        # Simulate the grace deadline passing for the OLD key deterministically,
        # then clean up. The replacement must be unaffected.
        store.retire_key_hash(old_hash, time.time() - 1)
        count = rotator.cleanup_expired()
        assert count >= 1
        assert store.authenticate("old-token") is None
        assert store.authenticate(new_key) is not None

    def test_cleanup_thread_started_on_rotate(self):
        store = _store_with_key("old-token")
        rotator = ApiKeyRotator(store, cleanup_interval_seconds=5.0)
        rotator.rotate("k1")
        assert rotator._cleanup_thread is not None
        assert rotator._cleanup_thread.is_alive()
        rotator.stop()

    def test_rotate_unknown_key_returns_none(self):
        store = _store_with_key("old-token")
        rotator = ApiKeyRotator(store)
        assert rotator.rotate("missing-key") is None

    def test_concurrent_double_rotation_retires_all_intermediate_keys(self):
        store = _store_with_key("orig-token")
        rotator = ApiKeyRotator(store, grace_period_hours=0.00002)

        results: list[str | None] = []
        results_lock = threading.Lock()

        def _do_rotate() -> None:
            new_key = rotator.rotate("k1")
            with results_lock:
                results.append(new_key)

        threads = [threading.Thread(target=_do_rotate) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        new_keys = [k for k in results if k]
        assert len(new_keys) >= 2, "expected concurrent rotations to produce multiple keys"

        # After the grace period elapses, the original key AND every
        # intermediate key must be retired — exactly ONE key may remain valid
        # (the final active key).  If concurrent double-rotation left an
        # intermediate key unretired, two keys would authenticate.
        time.sleep(0.2)
        assert store.authenticate("orig-token") is None
        alive = [k for k in new_keys if store.authenticate(k) is not None]
        assert len(alive) == 1, f"expected exactly one live key after rotation, got {len(alive)}"
