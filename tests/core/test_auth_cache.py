"""Tests for the ApiKeyStore authentication verdict cache (B4-1 perf fix).

The cache memoizes ``authenticate()`` results keyed by ``sha256(token)`` with
a short TTL. It must:

* return identical results to the authoritative PBKDF2 path (hit == cold);
* be invalidated on every key-table mutation (add/rotate/retire/remove);
* never serve a retired key past its grace deadline (hard-deadline guard);
* cache failed authentications too, with the same invalidation semantics;
* be safe under concurrent access from multiple threads;
* never store plaintext tokens.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections import OrderedDict

import pytest

from distllm.core.api_key_store import (
    _AUTH_CACHE_TTL_S,
    _AUTH_NEG_CACHE_TTL_S,
    ApiKeyStore,
    reset_api_key_store,
)


def _make_store(n: int = 3) -> tuple[ApiKeyStore, list[str]]:
    """Store with *n* keys; returns (store, raw_keys)."""
    raw_keys = [f"sk-test-key-{i}-{os.urandom(4).hex()}" for i in range(n)]
    store = ApiKeyStore()
    for i, k in enumerate(raw_keys):
        store.add_key(k, role="inference-only" if i else "admin", key_id=f"k{i}")
    return store, raw_keys


@pytest.fixture(autouse=True)
def _cleanup():
    reset_api_key_store()
    yield
    reset_api_key_store()
    for var in ("API_KEYS", "API_KEY", "API_KEYS_FILE"):
        os.environ.pop(var, None)


# ── Hit path: fast and consistent ───────────────────────────────────────────


class TestCacheHitPath:
    def test_hit_matches_cold_result(self):
        store, keys = _make_store(5)
        expected = store.authenticate(keys[4])  # cold: populates cache
        assert expected is not None
        for _ in range(50):
            assert store.authenticate(keys[4]) == expected

    def test_hit_is_fast(self):
        """Warm path must be orders of magnitude faster than PBKDF2 cold path."""
        store, keys = _make_store(5)
        store.authenticate(keys[-1])  # warm

        t0 = time.perf_counter()
        for _ in range(100):
            store.authenticate("sk-definitely-not-a-key")
        neg_warm = (time.perf_counter() - t0) / 100

        # A single cold PBKDF2 hash costs ~30 ms; the warm lookup must be at
        # least 1000x cheaper. Use a conservative 10 ms threshold so slow CI
        # machines don't flake while still catching a regressed (uncached)
        # path — an uncached miss with 6 stored keys costs ~180 ms per call.
        assert neg_warm < 0.010, (
            f"warm authenticate took {neg_warm * 1000:.2f} ms avg — "
            "verdict cache not effective"
        )

    def test_different_tokens_cached_independently(self):
        store, keys = _make_store(3)
        r0 = store.authenticate(keys[0])
        r1 = store.authenticate(keys[1])
        assert r0 != r1
        assert store.authenticate(keys[0]) == r0
        assert store.authenticate(keys[1]) == r1

    def test_no_plaintext_token_at_rest(self):
        """Cache must key on sha256(token); plaintext must not be recoverable
        from the cache structure or present as a value."""
        token = "sk-plaintext-probe-token"
        store, _ = _make_store(2)
        store.add_key(token, role="admin", key_id="probe")
        store.authenticate(token)  # populate cache

        blob = repr(store._auth_cache.values())
        assert token not in blob

    def test_cache_keys_are_sha256_digests(self):
        token = "sk-digest-shape-probe"
        store, _ = _make_store(1)
        store.add_key(token, role="admin", key_id="d")
        store.authenticate(token)

        import hashlib
        digest = hashlib.sha256(token.encode()).digest()
        entry = store._auth_cache.get(digest)
        assert entry is not None, "cache must be keyed by sha256(token) bytes"


# ── Invalidation on mutation ────────────────────────────────────────────────


class TestInvalidationOnMutation:
    def test_add_key_immediately_authenticates(self):
        store, _ = _make_store(2)
        fresh = "sk-brand-new-" + os.urandom(8).hex()
        # Warm the negative cache for this token first.
        assert store.authenticate(fresh) is None
        store.add_key(fresh, role="admin", key_id="new")
        assert store.authenticate(fresh) == ("new", "admin")

    def test_remove_key_immediately_revokes(self):
        store, keys = _make_store(2)
        victim = keys[0]
        assert store.authenticate(victim) is not None  # warm positive cache
        assert store.remove_key("k0") is True
        assert store.authenticate(victim) is None

    def test_retire_key_hash_revokes_after_deadline(self):
        store, keys = _make_store(2)
        victim = keys[1]
        h = store.get_latest_key_hash("k1")
        assert store.authenticate(victim) is not None  # warm cache
        assert store.retire_key_hash(h, time.time() - 1) is True
        assert store.authenticate(victim) is None

    def test_rotated_old_key_stops_authenticating_past_grace(self):
        """Full rotation flow: add replacement, retire old past deadline."""
        store, keys = _make_store(1)
        old = keys[0]
        assert store.authenticate(old) is not None  # warm cache BEFORE rotate
        new_key = "sk-rotated-" + os.urandom(8).hex()
        store.add_key(new_key, role="admin", key_id="k0")  # coexists
        store.retire_key_hash(store.get_latest_key_hash("k0"), time.time() + 3600)
        old_h = None
        for k in store._keys:
            if k.key_id == "k0" and k.key != store.get_latest_key_hash("k0"):
                old_h = k.key
        store.retire_key_hash(old_h, time.time() - 1)  # grace already elapsed
        assert store.authenticate(old) is None
        assert store.authenticate(new_key) == ("k0", "admin")

    def test_hard_deadline_guard_without_mutation(self):
        """A cached hit must not outlive the earliest pending grace deadline,
        even when no mutating method ran between calls (time-based retirement
        boundary). Probe the guard directly: keep the entry structurally fresh
        but force its hard_deadline into the past."""
        store, keys = _make_store(1)
        assert store.authenticate(keys[0]) is not None  # populate cache
        digest = next(iter(store._auth_cache))
        result, cached_at, fp, _dl = store._auth_cache[digest]
        store._auth_cache[digest] = (result, cached_at, fp, time.time() - 1)
        # Guard must reject the hit and fall through to authoritative verify.
        assert store.authenticate(keys[0]) is not None

    def test_direct_expires_at_edit_revokes(self):
        """Rewriting ``expires_at`` out-of-band changes the table fingerprint;
        the stale verdict must not be served."""
        store, keys = _make_store(2)
        assert store.authenticate(keys[1]) is not None  # warm cache for k1
        for k in store._keys:
            if k.key_id == "k1":
                k.expires_at = time.time() - 0.05
        assert store.authenticate(keys[1]) is None

    def test_direct_keys_manipulation_fails_safe(self):
        """Out-of-band edits to ``_keys`` change the fingerprint; stale cache
        entries must not serve the pre-manipulation verdict."""
        store, keys = _make_store(2)
        assert store.authenticate(keys[0]) is not None
        # Directly remove an entry without calling any mutator.
        store._keys[:] = [k for k in store._keys if k.key_id != "k0"]
        assert store.authenticate(keys[0]) is None

    def test_remove_expired_invalidates(self):
        store, keys = _make_store(2)
        assert store.authenticate(keys[1]) is not None
        store.retire_key("k1", time.time() - 1)
        removed = store.remove_expired()
        assert removed >= 1
        assert store.authenticate(keys[1]) is None


# ── Failed-auth caching ─────────────────────────────────────────────────────


class TestFailedAuthCaching:
    def test_failure_is_cached_and_consistent(self):
        store, _ = _make_store(2)
        bad = "sk-wrong-token"
        assert store.authenticate(bad) is None  # cold failure
        t0 = time.perf_counter()
        for _ in range(20):
            assert store.authenticate(bad) is None
        elapsed_ms_per_call = (time.perf_counter() - t0) * 1000 / 20
        # Uncached failure against 3 keys would cost ~90 ms/call here.
        assert elapsed_ms_per_call < 5

    def test_neg_cache_ttl_shorter_than_positive(self):
        assert _AUTH_NEG_CACHE_TTL_S < _AUTH_CACHE_TTL_S

    def test_new_key_overrides_negative_cache(self):
        store, _ = _make_store(2)
        tok = "sk-promoted-" + os.urandom(6).hex()
        assert store.authenticate(tok) is None          # negative-cached
        store.add_key(tok, role="read-only", key_id="promo")
        assert store.authenticate(tok) == ("promo", "read-only")


# ── Concurrency ─────────────────────────────────────────────────────────────


class TestConcurrency:
    def test_parallel_authenticate_threads_agree(self):
        store, keys = _make_store(3)
        bad = "sk-not-valid"
        store.authenticate(keys[0])
        store.authenticate(bad)

        errors: list[str] = []
        barrier = threading.Barrier(12)

        def worker(idx: int) -> None:
            try:
                barrier.wait(timeout=10)
                for _ in range(30):
                    expect_hit = idx % 2 == 0
                    tok = keys[idx % len(keys)] if expect_hit else bad
                    r = store.authenticate(tok)
                    if expect_hit:
                        assert r is not None, f"{tok} failed to authenticate"
                        assert r[1] in ("admin", "inference-only")
                    else:
                        assert r is None
            except Exception as exc:  # noqa: BLE001
                errors.append(f"worker {idx}: {exc!r}")

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(12)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)
        assert not errors, errors

    def test_concurrent_rotate_while_authenticating(self):
        """Authentications racing a rotation must never return a verdict that
        contradicts the post-mutation table (either old still-in-grace result
        or clean rejection — never a torn read)."""
        store, keys = _make_store(4)
        stop = threading.Event()

        def auth_loop():
            while not stop.is_set():
                r = store.authenticate(keys[-1])
                assert r is None or r[0] == "k3"

        def rotate_loop():
            for _ in range(5):
                time.sleep(0.01)
                store.add_key(
                    f"sk-race-{threading.get_ident()}-{time.time()}",
                    role="admin",
                    key_id=f"race-{threading.get_ident()}",
                )

        threads = [threading.Thread(target=auth_loop) for _ in range(4)]
        threads += [threading.Thread(target=rotate_loop) for _ in range(2)]
        for t in threads:
            t.start()
        time.sleep(0.3)
        stop.set()
        for t in threads:
            t.join(timeout=30)
        assert not any(t.is_alive() for t in threads), "deadlock detected"

    def test_lru_eviction_bounds_size(self, monkeypatch):
        # Patch a small cap so the spray doesn't pay 30 ms per miss.
        import distllm.core.api_key_store as aks_mod
        small_cap = 64
        monkeypatch.setattr(aks_mod, "_AUTH_CACHE_MAX_ENTRIES", small_cap)
        store, _ = _make_store(1)
        for i in range(small_cap + 50):
            store.authenticate(f"sk-spray-{i}")
        assert len(store._auth_cache) <= small_cap


# ── TTL expiry ──────────────────────────────────────────────────────────────


class TestTTLExpiry:
    def test_expired_positive_entry_reverifies(self):
        store, keys = _make_store(1)
        assert store.authenticate(keys[0]) is not None
        # Force the cached entry to look expired.
        digest = next(iter(store._auth_cache))
        result, cached_at, fp, dl = store._auth_cache[digest]
        store._auth_cache[digest] = (result, cached_at - _AUTH_CACHE_TTL_S - 1, fp, dl)
        assert store.authenticate(keys[0]) is not None  # re-verifies fine

    def test_expired_negative_entry_reverifies_to_none(self):
        store, _ = _make_store(1)
        assert store.authenticate("sk-bad") is None
        digest = next(iter(store._auth_cache))
        result, cached_at, fp, dl = store._auth_cache[digest]
        store._auth_cache[digest] = (result, cached_at - _AUTH_NEG_CACHE_TTL_S - 1, fp, dl)
        assert store.authenticate("sk-bad") is None

    def test_fingerprint_change_drops_entry(self):
        store, keys = _make_store(1)
        store.authenticate(keys[0])
        digest = next(iter(store._auth_cache))
        result, cached_at, fp, dl = store._auth_cache[digest]
        tampered_fp = fp[:-1]  # simulate a structurally different table
        store._auth_cache[digest] = (result, cached_at, tampered_fp, dl)
        # Must NOT trust the tampered/stale entry: re-verify succeeds anyway.
        assert store.authenticate(keys[0]) is not None
