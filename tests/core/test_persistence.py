"""Tests for SQLiteBackend persistence layer.

Covers: initialize, CRUD for prompts, reviews, wallets,
purchases, listings, jobs, earnings, transactions.
"""

from __future__ import annotations

import os
import tempfile
import time

import pytest

from tests._import_helper import bootstrap_fake_packages, load_module

bootstrap_fake_packages()

_mod = load_module("distllm/core/persistence.py")
SQLiteBackend = _mod.SQLiteBackend
StorageBackend = _mod.StorageBackend


@pytest.fixture
def db_path():
    p = os.path.join(tempfile.gettempdir(), f"test_persistence_{time.time_ns()}.db")
    yield p
    for _ in range(3):
        try:
            if os.path.exists(p):
                os.unlink(p)
            break
        except PermissionError:
            time.sleep(0.1)


@pytest.fixture
def backend(db_path):
    b = SQLiteBackend(db_path)
    b.initialize()
    yield b
    b.close()


class TestStorageBackendInterface:
    def test_is_abstract(self):
        """StorageBackend cannot be instantiated directly."""
        with pytest.raises(TypeError):
            StorageBackend()


class TestSQLiteBackendInit:
    def test_initialize_creates_tables(self, backend, db_path):
        """initialize() returns schema version and creates tables."""
        b = SQLiteBackend(db_path)
        version = b.initialize()
        assert version == 1
        s = b.stats()
        assert s["prompts"] == 0
        assert s["reviews"] == 0
        assert s["wallets"] == 0
        b.close()

    def test_double_initialize_is_idempotent(self, backend):
        """Calling initialize() twice does not raise."""
        backend.initialize()  # second call


class TestPromptCRUD:
    def test_save_and_load(self, backend):
        prompt = {
            "prompt_id": "p1",
            "author_id": "author-1",
            "name": "Test Prompt",
            "description": "A test prompt",
            "category": "general",
            "system_prompt": "You are a helpful assistant.",
            "created_at": time.time(),
            "updated_at": time.time(),
        }
        backend.save_prompt(prompt)
        loaded = backend.load_prompt("p1")
        assert loaded is not None
        assert loaded["prompt_id"] == "p1"
        assert loaded["name"] == "Test Prompt"

    def test_load_missing_returns_none(self, backend):
        assert backend.load_prompt("nonexistent") is None

    def test_load_all_prompts(self, backend):
        t = time.time()
        for i in range(3):
            backend.save_prompt({
                "prompt_id": f"p{i}",
                "author_id": "a1",
                "name": f"Prompt {i}",
                "description": "",
                "category": "general",
                "system_prompt": "Hello",
                "created_at": t + i,
                "updated_at": t + i,
            })
        all_p = backend.load_all_prompts()
        assert len(all_p) == 3

    def test_delete_prompt(self, backend):
        prompt = {
            "prompt_id": "p-del",
            "author_id": "a1",
            "name": "Delete Me",
            "description": "",
            "category": "general",
            "system_prompt": "Bye",
            "created_at": time.time(),
            "updated_at": time.time(),
        }
        backend.save_prompt(prompt)
        assert backend.delete_prompt("p-del") is True
        assert backend.load_prompt("p-del") is None
        assert backend.delete_prompt("p-del") is False


class TestWalletOperations:
    def test_save_and_load_wallet(self, backend):
        wallet = {
            "user_id": "user-1",
            "balance_tokens": 1000,
            "total_earned": 500,
            "total_spent": 200,
            "created_at": time.time(),
        }
        backend.save_wallet(wallet)
        loaded = backend.load_wallet("user-1")
        assert loaded is not None
        assert loaded["balance_tokens"] == 1000

    def test_load_missing_wallet(self, backend):
        assert backend.load_wallet("nobody") is None


class TestReviewOperations:
    def test_save_and_load_reviews(self, backend):
        backend.save_prompt({
            "prompt_id": "p-rev",
            "author_id": "a1",
            "name": "R",
            "description": "",
            "category": "general",
            "system_prompt": "Hi",
            "created_at": time.time(),
            "updated_at": time.time(),
        })
        backend.save_review({
            "review_id": "r1",
            "prompt_id": "p-rev",
            "user_id": "u1",
            "rating": 5,
            "comment": "Great!",
            "created_at": time.time(),
        })
        reviews = backend.load_reviews("p-rev")
        assert len(reviews) == 1
        assert reviews[0]["rating"] == 5


class TestPurchaseOperations:
    def test_save_and_load_purchases(self, backend):
        backend.save_purchase("user-1", "prompt-1")
        backend.save_purchase("user-1", "prompt-2")
        purchases = backend.load_user_purchases("user-1")
        assert purchases == {"prompt-1", "prompt-2"}

    def test_load_user_library(self, backend):
        backend.save_purchase("user-1", "p-a")
        backend.save_purchase("user-1", "p-b")
        lib = backend.load_user_library("user-1")
        assert lib == ["p-a", "p-b"]


class TestListingOperations:
    def test_save_and_load_listing(self, backend):
        t = time.time()
        listing = {
            "listing_id": "l1",
            "provider_id": "prov-1",
            "created_at": t,
            "last_updated": t,
        }
        backend.save_listing(listing)
        loaded = backend.load_listing("l1")
        assert loaded is not None
        assert loaded["listing_id"] == "l1"

    def test_load_all_listings(self, backend):
        t = time.time()
        for i in range(2):
            backend.save_listing({
                "listing_id": f"l{i}",
                "provider_id": f"prov-{i}",
                "created_at": t + i,
                "last_updated": t + i,
            })
        assert len(backend.load_all_listings()) == 2

    def test_delete_listing(self, backend):
        t = time.time()
        backend.save_listing({
            "listing_id": "l-del",
            "provider_id": "prov-1",
            "created_at": t,
            "last_updated": t,
        })
        assert backend.delete_listing("l-del") is True
        assert backend.delete_listing("l-del") is False


class TestJobOperations:
    def test_save_and_load_job(self, backend):
        job = {
            "job_id": "j1",
            "requester_id": "req-1",
            "created_at": time.time(),
        }
        backend.save_job(job)
        loaded = backend.load_job("j1")
        assert loaded is not None
        assert loaded["job_id"] == "j1"

    def test_load_all_jobs(self, backend):
        t = time.time()
        for i in range(3):
            backend.save_job({
                "job_id": f"j{i}",
                "requester_id": "req-1",
                "created_at": t + i,
            })
        assert len(backend.load_all_jobs()) == 3


class TestEarningsOperations:
    def test_save_and_load_earnings(self, backend):
        earnings = {
            "provider_id": "prov-1",
            "total_earnings": 100.0,
            "total_gpu_hours": 42.0,
            "total_tokens_served": 50000,
            "total_jobs": 10,
        }
        backend.save_provider_earnings(earnings)
        loaded = backend.load_provider_earnings("prov-1")
        assert loaded is not None
        assert loaded["total_earnings"] == 100.0

    def test_load_missing_earnings(self, backend):
        assert backend.load_provider_earnings("nobody") is None


class TestTransactionOperations:
    def test_save_and_load_transactions(self, backend):
        t = time.time()
        backend.save_transaction({
            "transaction_id": "tx1",
            "kind": "purchase",
            "user_id": "u1",
            "amount_tokens": 100,
            "created_at": t,
        })
        txns = backend.load_transactions(user_id="u1")
        assert len(txns) == 1
        assert txns[0]["kind"] == "purchase"

    def test_transactions_filter_by_kind(self, backend):
        t = time.time()
        kinds = ("purchase", "refund", "purchase")
        for i, kind in enumerate(kinds):
            backend.save_transaction({
                "transaction_id": f"tx-{kind}-{t}-{i}",
                "kind": kind,
                "user_id": "u1",
                "amount_tokens": 50,
                "created_at": t,
            })
        purchases = backend.load_transactions(kind="purchase")
        assert len(purchases) == 2
        refunds = backend.load_transactions(kind="refund")
        assert len(refunds) == 1


class TestVacuumAndStats:
    def test_vacuum_does_not_raise(self, backend):
        backend.vacuum()

    def test_stats_counts(self, backend):
        t = time.time()
        backend.save_prompt({
            "prompt_id": "p1",
            "author_id": "a1",
            "name": "Test",
            "description": "",
            "category": "general",
            "system_prompt": "Hi",
            "created_at": t,
            "updated_at": t,
        })
        s = backend.stats()
        assert s["prompts"] == 1
