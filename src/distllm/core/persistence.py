"""Persistence layer for PromptExchange and Marketplace.

Provides a StorageBackend abstract interface and a SQLiteBackend
implementation that both the PromptExchange and Marketplace classes
can use for durable storage. Falls back to in-memory operation when
no backend is configured.

Usage::

    from distllm.core.persistence import SQLiteBackend

    backend = SQLiteBackend("exchange.db")
    backend.initialize()

    # PromptExchange integration
    exchange = PromptExchange(backend=backend)

    # Marketplace integration
    marketplace = Marketplace(backend=backend)
"""

from __future__ import annotations

import abc
import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from loguru import logger


# ---------------------------------------------------------------------------
# Schema version & migrations
# ---------------------------------------------------------------------------

_SCHEMA_VERSION = 1

_SCHEMA_MIGRATIONS: dict[int, Any] = {}


def _register_migration(version: int):  # type: ignore[type-arg]
    """Decorator to register a migration for a target schema version.

    Increment ``_SCHEMA_VERSION`` and decorate a function that receives
    ``(conn, from_version, to_version)`` and issues DDL/DML to upgrade.
    """

    def wrapper(func):  # type: ignore[no-untyped-def]
        _SCHEMA_MIGRATIONS[version] = func
        return func

    return wrapper


# ---------------------------------------------------------------------------
# SQL: table creation (idempotent)
# ---------------------------------------------------------------------------

_CREATE_TABLES = """
CREATE TABLE IF NOT EXISTS prompts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    prompt_id TEXT UNIQUE NOT NULL,
    author_id TEXT NOT NULL,
    name TEXT NOT NULL,
    description TEXT NOT NULL,
    category TEXT NOT NULL,
    system_prompt TEXT NOT NULL,
    tags TEXT NOT NULL DEFAULT '[]',
    license TEXT NOT NULL DEFAULT 'free',
    price_tokens INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'published',
    version INTEGER NOT NULL DEFAULT 1,
    parent_id TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    examples TEXT NOT NULL DEFAULT '[]',
    metadata TEXT NOT NULL DEFAULT '{}',
    -- metrics (denormalised for query performance)
    total_uses INTEGER NOT NULL DEFAULT 0,
    avg_throughput_tok_s REAL NOT NULL DEFAULT 0.0,
    avg_latency_ms REAL NOT NULL DEFAULT 0.0,
    avg_quality_score REAL NOT NULL DEFAULT 0.0,
    avg_cost_usd REAL NOT NULL DEFAULT 0.0,
    total_tokens_generated INTEGER NOT NULL DEFAULT 0,
    unique_users INTEGER NOT NULL DEFAULT 0,
    avg_rating REAL NOT NULL DEFAULT 0.0,
    rating_count INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS prompt_reviews (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    review_id TEXT UNIQUE NOT NULL,
    prompt_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    rating INTEGER NOT NULL,
    comment TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL,
    helpful_count INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY (prompt_id) REFERENCES prompts(prompt_id)
);

CREATE TABLE IF NOT EXISTS wallets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT UNIQUE NOT NULL,
    balance_tokens INTEGER NOT NULL DEFAULT 0,
    total_earned INTEGER NOT NULL DEFAULT 0,
    total_spent INTEGER NOT NULL DEFAULT 0,
    total_purchased INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS user_purchases (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    prompt_id TEXT NOT NULL,
    purchased_at REAL NOT NULL,
    UNIQUE(user_id, prompt_id)
);

CREATE TABLE IF NOT EXISTS marketplace_listings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    listing_id TEXT UNIQUE NOT NULL,
    provider_id TEXT NOT NULL,
    provider_name TEXT NOT NULL DEFAULT '',
    gpu_name TEXT NOT NULL DEFAULT '',
    gpu_memory_bytes INTEGER NOT NULL DEFAULT 0,
    gpu_count INTEGER NOT NULL DEFAULT 1,
    cpu_cores INTEGER NOT NULL DEFAULT 0,
    ram_bytes INTEGER NOT NULL DEFAULT 0,
    price_per_hour REAL NOT NULL DEFAULT 0.0,
    price_per_million_tokens REAL NOT NULL DEFAULT 0.0,
    currency TEXT NOT NULL DEFAULT 'USD',
    status TEXT NOT NULL DEFAULT 'active',
    available_from REAL NOT NULL DEFAULT 0.0,
    available_until REAL NOT NULL DEFAULT 0.0,
    max_concurrent_jobs INTEGER NOT NULL DEFAULT 1,
    current_jobs INTEGER NOT NULL DEFAULT 0,
    supported_models TEXT NOT NULL DEFAULT '[]',
    supported_dtypes TEXT NOT NULL DEFAULT '["float16"]',
    max_batch_size INTEGER NOT NULL DEFAULT 8,
    supports_streaming INTEGER NOT NULL DEFAULT 1,
    supports_quantization INTEGER NOT NULL DEFAULT 0,
    supports_lora INTEGER NOT NULL DEFAULT 0,
    region TEXT NOT NULL DEFAULT '',
    bandwidth_mbps REAL NOT NULL DEFAULT 0.0,
    latency_ms REAL NOT NULL DEFAULT 0.0,
    carbon_intensity REAL NOT NULL DEFAULT 0.0,
    renewable_pct REAL NOT NULL DEFAULT 0.0,
    reputation_score REAL NOT NULL DEFAULT 0.5,
    total_jobs_completed INTEGER NOT NULL DEFAULT 0,
    uptime_pct REAL NOT NULL DEFAULT 100.0,
    source TEXT NOT NULL DEFAULT 'peer',
    created_at REAL NOT NULL,
    last_updated REAL NOT NULL,
    tags TEXT NOT NULL DEFAULT '[]'
);

CREATE TABLE IF NOT EXISTS marketplace_jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT UNIQUE NOT NULL,
    requester_id TEXT NOT NULL,
    model_name TEXT NOT NULL DEFAULT '',
    min_gpu_memory_bytes INTEGER NOT NULL DEFAULT 0,
    min_gpu_count INTEGER NOT NULL DEFAULT 1,
    min_cpu_cores INTEGER NOT NULL DEFAULT 0,
    min_ram_bytes INTEGER NOT NULL DEFAULT 0,
    required_dtype TEXT NOT NULL DEFAULT 'float16',
    requires_streaming INTEGER NOT NULL DEFAULT 1,
    requires_quantization INTEGER NOT NULL DEFAULT 0,
    requires_lora INTEGER NOT NULL DEFAULT 0,
    max_price_per_hour REAL NOT NULL DEFAULT 0.0,
    max_price_per_million_tokens REAL NOT NULL DEFAULT 0.0,
    max_budget_total REAL NOT NULL DEFAULT 0.0,
    max_latency_ms REAL NOT NULL DEFAULT 5000.0,
    min_uptime_pct REAL NOT NULL DEFAULT 99.0,
    preferred_regions TEXT NOT NULL DEFAULT '[]',
    min_reputation REAL NOT NULL DEFAULT 0.3,
    status TEXT NOT NULL DEFAULT 'open',
    matched_listing_id TEXT NOT NULL DEFAULT '',
    matched_provider_id TEXT NOT NULL DEFAULT '',
    tokens_generated INTEGER NOT NULL DEFAULT 0,
    cost_accumulated REAL NOT NULL DEFAULT 0.0,
    started_at REAL NOT NULL DEFAULT 0.0,
    completed_at REAL NOT NULL DEFAULT 0.0,
    created_at REAL NOT NULL,
    priority INTEGER NOT NULL DEFAULT 2,
    tags TEXT NOT NULL DEFAULT '[]'
);

CREATE TABLE IF NOT EXISTS provider_earnings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    provider_id TEXT UNIQUE NOT NULL,
    total_earnings REAL NOT NULL DEFAULT 0.0,
    total_gpu_hours REAL NOT NULL DEFAULT 0.0,
    total_tokens_served INTEGER NOT NULL DEFAULT 0,
    total_jobs INTEGER NOT NULL DEFAULT 0,
    current_month_earnings REAL NOT NULL DEFAULT 0.0,
    pending_payout REAL NOT NULL DEFAULT 0.0,
    last_payout_at REAL NOT NULL DEFAULT 0.0
);

CREATE TABLE IF NOT EXISTS transactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    transaction_id TEXT UNIQUE NOT NULL,
    kind TEXT NOT NULL,
    user_id TEXT NOT NULL,
    prompt_id TEXT,
    amount_tokens INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL,
    metadata TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS exchange_schema_version (
    version INTEGER PRIMARY KEY
);

CREATE INDEX IF NOT EXISTS idx_prompts_status ON prompts(status);
CREATE INDEX IF NOT EXISTS idx_prompts_category ON prompts(category);
CREATE INDEX IF NOT EXISTS idx_prompts_author ON prompts(author_id);
CREATE INDEX IF NOT EXISTS idx_reviews_prompt ON prompt_reviews(prompt_id);
CREATE INDEX IF NOT EXISTS idx_purchases_user ON user_purchases(user_id);
CREATE INDEX IF NOT EXISTS idx_listings_status ON marketplace_listings(status);
CREATE INDEX IF NOT EXISTS idx_listings_provider ON marketplace_listings(provider_id);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON marketplace_jobs(status);
CREATE INDEX IF NOT EXISTS idx_jobs_requester ON marketplace_jobs(requester_id);
CREATE INDEX IF NOT EXISTS idx_transactions_user ON transactions(user_id);
"""


# ---------------------------------------------------------------------------
# Abstract backend interface
# ---------------------------------------------------------------------------


class StorageBackend(abc.ABC):
    """Abstract storage interface for the exchange and marketplace.

    Implementations must be safe to share across threads.
    """

    # -- lifecycle -----------------------------------------------------------

    @abc.abstractmethod
    def initialize(self) -> int:
        """Create tables if absent and run migrations.  Returns schema version."""

    @abc.abstractmethod
    def close(self) -> None:
        """Release resources."""

    # -- prompts -------------------------------------------------------------

    @abc.abstractmethod
    def save_prompt(self, prompt: dict[str, Any]) -> None:
        """Insert or replace a prompt row (full dict matching PublishedPrompt)."""

    @abc.abstractmethod
    def load_prompt(self, prompt_id: str) -> dict[str, Any] | None:
        """Return a prompt dict or None."""

    @abc.abstractmethod
    def load_all_prompts(self) -> list[dict[str, Any]]:
        """Return every prompt row."""

    @abc.abstractmethod
    def delete_prompt(self, prompt_id: str) -> bool:
        """Delete a prompt.  Returns True if a row was removed."""

    # -- reviews -------------------------------------------------------------

    @abc.abstractmethod
    def save_review(self, review: dict[str, Any]) -> None:
        """Insert or replace a review row."""

    @abc.abstractmethod
    def load_reviews(self, prompt_id: str) -> list[dict[str, Any]]:
        """Return all reviews for a prompt."""

    # -- wallets -------------------------------------------------------------

    @abc.abstractmethod
    def save_wallet(self, wallet: dict[str, Any]) -> None:
        """Insert or replace a wallet row."""

    @abc.abstractmethod
    def load_wallet(self, user_id: str) -> dict[str, Any] | None:
        """Return a wallet dict or None."""

    # -- purchases -----------------------------------------------------------

    @abc.abstractmethod
    def save_purchase(self, user_id: str, prompt_id: str) -> None:
        """Record a user->prompt purchase (idempotent)."""

    @abc.abstractmethod
    def load_user_purchases(self, user_id: str) -> set[str]:
        """Return set of prompt_ids the user has purchased."""

    @abc.abstractmethod
    def load_user_library(self, user_id: str) -> list[str]:
        """Return ordered list of prompt_ids in user's library."""

    # -- marketplace listings ------------------------------------------------

    @abc.abstractmethod
    def save_listing(self, listing: dict[str, Any]) -> None:
        """Insert or replace a GPU listing row."""

    @abc.abstractmethod
    def load_listing(self, listing_id: str) -> dict[str, Any] | None:
        """Return a listing dict or None."""

    @abc.abstractmethod
    def load_all_listings(self) -> list[dict[str, Any]]:
        """Return every listing row."""

    @abc.abstractmethod
    def delete_listing(self, listing_id: str) -> bool:
        """Delete a listing.  Returns True if a row was removed."""

    # -- marketplace jobs ----------------------------------------------------

    @abc.abstractmethod
    def save_job(self, job: dict[str, Any]) -> None:
        """Insert or replace a marketplace job row."""

    @abc.abstractmethod
    def load_job(self, job_id: str) -> dict[str, Any] | None:
        """Return a job dict or None."""

    @abc.abstractmethod
    def load_all_jobs(self) -> list[dict[str, Any]]:
        """Return every job row."""

    # -- provider earnings ---------------------------------------------------

    @abc.abstractmethod
    def save_provider_earnings(self, earnings: dict[str, Any]) -> None:
        """Insert or replace provider earnings."""

    @abc.abstractmethod
    def load_provider_earnings(self, provider_id: str) -> dict[str, Any] | None:
        """Return provider earnings dict or None."""

    # -- transactions --------------------------------------------------------

    @abc.abstractmethod
    def save_transaction(self, txn: dict[str, Any]) -> None:
        """Insert a transaction record."""

    @abc.abstractmethod
    def load_transactions(
        self,
        user_id: str = "",
        kind: str = "",
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Return transactions, optionally filtered by user and kind."""


# ---------------------------------------------------------------------------
# SQLite implementation
# ---------------------------------------------------------------------------


class SQLiteBackend(StorageBackend):
    """SQLite-backed persistence for the prompt exchange and marketplace.

    Thread-safe via WAL mode and a ``threading.Lock`` for writes.
    Each thread gets its own connection (thread-local).

    Usage::

        backend = SQLiteBackend("exchange.db")
        backend.initialize()
    """

    def __init__(self, db_path: str = "exchange.db", wal_mode: bool = True):
        self._db_path = db_path
        self._wal_mode = wal_mode
        self._local = threading.local()
        self._lock = threading.Lock()

    # -- connection helpers --------------------------------------------------

    def _get_conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(
                self._db_path,
                timeout=30,
                check_same_thread=False,
            )
            self._local.conn.row_factory = sqlite3.Row
            if self._wal_mode:
                self._local.conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn.execute("PRAGMA busy_timeout=5000")
            self._local.conn.execute("PRAGMA foreign_keys=ON")
        return self._local.conn

    @contextmanager
    def _transaction(self):  # type: ignore[no-untyped-def]
        conn = self._get_conn()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    # -- lifecycle -----------------------------------------------------------

    def initialize(self) -> int:
        """Create tables and run migrations.

        Returns the current schema version.
        """
        with self._transaction() as conn:
            conn.executescript(_CREATE_TABLES)
            row = conn.execute(
                "SELECT version FROM exchange_schema_version"
            ).fetchone()
            if row is None:
                conn.execute(
                    "INSERT INTO exchange_schema_version (version) VALUES (?)",
                    (_SCHEMA_VERSION,),
                )
                current = _SCHEMA_VERSION
            else:
                current = row["version"]
                if current < _SCHEMA_VERSION:
                    self._migrate(conn, current, _SCHEMA_VERSION)
                    conn.execute(
                        "UPDATE exchange_schema_version SET version=?",
                        (_SCHEMA_VERSION,),
                    )
                elif current > _SCHEMA_VERSION:
                    raise RuntimeError(
                        f"Database schema v{current} is newer than code "
                        f"supports (v{_SCHEMA_VERSION})."
                    )
        logger.info(
            f"Exchange persistence initialized (schema v{_SCHEMA_VERSION}) "
            f"at {self._db_path}"
        )
        return _SCHEMA_VERSION

    def _migrate(self, conn: sqlite3.Connection, from_ver: int, to_ver: int) -> None:
        logger.info(f"Migrating exchange schema v{from_ver} -> v{to_ver}")
        for ver in range(from_ver + 1, to_ver + 1):
            migration = _SCHEMA_MIGRATIONS.get(ver)
            if migration:
                logger.info(f"Applying migration v{ver}")
                migration(conn, from_ver, ver)
            else:
                logger.warning(f"No migration registered for v{ver}, skipping")

    def close(self) -> None:
        if hasattr(self._local, "conn") and self._local.conn is not None:
            self._local.conn.close()
            self._local.conn = None

    # -- prompts -------------------------------------------------------------

    def save_prompt(self, prompt: dict[str, Any]) -> None:
        m = prompt.get("metrics", {})
        with self._lock, self._transaction() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO prompts (
                    prompt_id, author_id, name, description, category,
                    system_prompt, tags, license, price_tokens, status,
                    version, parent_id, created_at, updated_at, examples,
                    metadata, total_uses, avg_throughput_tok_s, avg_latency_ms,
                    avg_quality_score, avg_cost_usd, total_tokens_generated,
                    unique_users, avg_rating, rating_count
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    prompt["prompt_id"],
                    prompt["author_id"],
                    prompt["name"],
                    prompt["description"],
                    prompt["category"],
                    prompt["system_prompt"],
                    json.dumps(prompt.get("tags", [])),
                    prompt.get("license", "free"),
                    prompt.get("price_tokens", 0),
                    prompt.get("status", "published"),
                    prompt.get("version", 1),
                    prompt.get("parent_id", ""),
                    prompt["created_at"],
                    prompt["updated_at"],
                    json.dumps(prompt.get("examples", [])),
                    json.dumps(prompt.get("metadata", {})),
                    m.get("total_uses", 0),
                    m.get("avg_throughput_tok_s", 0.0),
                    m.get("avg_latency_ms", 0.0),
                    m.get("avg_quality_score", 0.0),
                    m.get("avg_cost_usd", 0.0),
                    m.get("total_tokens_generated", 0),
                    m.get("unique_users", 0),
                    m.get("avg_rating", 0.0),
                    m.get("rating_count", 0),
                ),
            )

    def load_prompt(self, prompt_id: str) -> dict[str, Any] | None:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM prompts WHERE prompt_id=?", (prompt_id,)
        ).fetchone()
        return self._row_to_prompt(row) if row else None

    def load_all_prompts(self) -> list[dict[str, Any]]:
        conn = self._get_conn()
        rows = conn.execute("SELECT * FROM prompts").fetchall()
        return [self._row_to_prompt(r) for r in rows]

    def delete_prompt(self, prompt_id: str) -> bool:
        with self._lock, self._transaction() as conn:
            cur = conn.execute(
                "DELETE FROM prompts WHERE prompt_id=?", (prompt_id,)
            )
            return cur.rowcount > 0

    @staticmethod
    def _row_to_prompt(row: sqlite3.Row) -> dict[str, Any]:
        d = dict(row)
        return {
            "prompt_id": d["prompt_id"],
            "author_id": d["author_id"],
            "name": d["name"],
            "description": d["description"],
            "category": d["category"],
            "system_prompt": d["system_prompt"],
            "tags": json.loads(d["tags"]),
            "license": d["license"],
            "price_tokens": d["price_tokens"],
            "status": d["status"],
            "version": d["version"],
            "parent_id": d["parent_id"],
            "created_at": d["created_at"],
            "updated_at": d["updated_at"],
            "examples": json.loads(d["examples"]),
            "metadata": json.loads(d["metadata"]),
            "metrics": {
                "total_uses": d["total_uses"],
                "avg_throughput_tok_s": d["avg_throughput_tok_s"],
                "avg_latency_ms": d["avg_latency_ms"],
                "avg_quality_score": d["avg_quality_score"],
                "avg_cost_usd": d["avg_cost_usd"],
                "total_tokens_generated": d["total_tokens_generated"],
                "unique_users": d["unique_users"],
                "avg_rating": d["avg_rating"],
                "rating_count": d["rating_count"],
            },
        }

    # -- reviews -------------------------------------------------------------

    def save_review(self, review: dict[str, Any]) -> None:
        with self._lock, self._transaction() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO prompt_reviews
                   (review_id, prompt_id, user_id, rating, comment,
                    created_at, helpful_count)
                   VALUES (?,?,?,?,?,?,?)""",
                (
                    review["review_id"],
                    review["prompt_id"],
                    review["user_id"],
                    review["rating"],
                    review.get("comment", ""),
                    review["created_at"],
                    review.get("helpful_count", 0),
                ),
            )

    def load_reviews(self, prompt_id: str) -> list[dict[str, Any]]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM prompt_reviews WHERE prompt_id=? "
            "ORDER BY helpful_count DESC",
            (prompt_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    # -- wallets -------------------------------------------------------------

    def save_wallet(self, wallet: dict[str, Any]) -> None:
        with self._lock, self._transaction() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO wallets
                   (user_id, balance_tokens, total_earned, total_spent,
                    total_purchased, created_at)
                   VALUES (?,?,?,?,?,?)""",
                (
                    wallet["user_id"],
                    wallet.get("balance_tokens", 0),
                    wallet.get("total_earned", 0),
                    wallet.get("total_spent", 0),
                    wallet.get("total_purchased", 0),
                    wallet["created_at"],
                ),
            )

    def load_wallet(self, user_id: str) -> dict[str, Any] | None:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM wallets WHERE user_id=?", (user_id,)
        ).fetchone()
        return dict(row) if row else None

    # -- purchases -----------------------------------------------------------

    def save_purchase(self, user_id: str, prompt_id: str) -> None:
        with self._lock, self._transaction() as conn:
            conn.execute(
                """INSERT OR IGNORE INTO user_purchases
                   (user_id, prompt_id, purchased_at) VALUES (?,?,?)""",
                (user_id, prompt_id, time.time()),
            )

    def load_user_purchases(self, user_id: str) -> set[str]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT prompt_id FROM user_purchases WHERE user_id=?",
            (user_id,),
        ).fetchall()
        return {r["prompt_id"] for r in rows}

    def load_user_library(self, user_id: str) -> list[str]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT prompt_id FROM user_purchases WHERE user_id=? "
            "ORDER BY purchased_at",
            (user_id,),
        ).fetchall()
        return [r["prompt_id"] for r in rows]

    # -- marketplace listings ------------------------------------------------

    def save_listing(self, listing: dict[str, Any]) -> None:
        with self._lock, self._transaction() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO marketplace_listings (
                    listing_id, provider_id, provider_name, gpu_name,
                    gpu_memory_bytes, gpu_count, cpu_cores, ram_bytes,
                    price_per_hour, price_per_million_tokens, currency,
                    status, available_from, available_until,
                    max_concurrent_jobs, current_jobs, supported_models,
                    supported_dtypes, max_batch_size, supports_streaming,
                    supports_quantization, supports_lora, region,
                    bandwidth_mbps, latency_ms, carbon_intensity,
                    renewable_pct, reputation_score, total_jobs_completed,
                    uptime_pct, source, created_at, last_updated, tags
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    listing["listing_id"],
                    listing["provider_id"],
                    listing.get("provider_name", ""),
                    listing.get("gpu_name", ""),
                    listing.get("gpu_memory_bytes", 0),
                    listing.get("gpu_count", 1),
                    listing.get("cpu_cores", 0),
                    listing.get("ram_bytes", 0),
                    listing.get("price_per_hour", 0.0),
                    listing.get("price_per_million_tokens", 0.0),
                    listing.get("currency", "USD"),
                    listing.get("status", "active"),
                    listing.get("available_from", 0.0),
                    listing.get("available_until", 0.0),
                    listing.get("max_concurrent_jobs", 1),
                    listing.get("current_jobs", 0),
                    json.dumps(listing.get("supported_models", [])),
                    json.dumps(listing.get("supported_dtypes", ["float16"])),
                    listing.get("max_batch_size", 8),
                    int(listing.get("supports_streaming", True)),
                    int(listing.get("supports_quantization", False)),
                    int(listing.get("supports_lora", False)),
                    listing.get("region", ""),
                    listing.get("bandwidth_mbps", 0.0),
                    listing.get("latency_ms", 0.0),
                    listing.get("carbon_intensity", 0.0),
                    listing.get("renewable_pct", 0.0),
                    listing.get("reputation_score", 0.5),
                    listing.get("total_jobs_completed", 0),
                    listing.get("uptime_pct", 100.0),
                    listing.get("source", "peer"),
                    listing["created_at"],
                    listing["last_updated"],
                    json.dumps(listing.get("tags", [])),
                ),
            )

    def load_listing(self, listing_id: str) -> dict[str, Any] | None:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM marketplace_listings WHERE listing_id=?",
            (listing_id,),
        ).fetchone()
        return self._row_to_listing(row) if row else None

    def load_all_listings(self) -> list[dict[str, Any]]:
        conn = self._get_conn()
        rows = conn.execute("SELECT * FROM marketplace_listings").fetchall()
        return [self._row_to_listing(r) for r in rows]

    def delete_listing(self, listing_id: str) -> bool:
        with self._lock, self._transaction() as conn:
            cur = conn.execute(
                "DELETE FROM marketplace_listings WHERE listing_id=?",
                (listing_id,),
            )
            return cur.rowcount > 0

    @staticmethod
    def _row_to_listing(row: sqlite3.Row) -> dict[str, Any]:
        d = dict(row)
        return {
            "listing_id": d["listing_id"],
            "provider_id": d["provider_id"],
            "provider_name": d["provider_name"],
            "gpu_name": d["gpu_name"],
            "gpu_memory_bytes": d["gpu_memory_bytes"],
            "gpu_count": d["gpu_count"],
            "cpu_cores": d["cpu_cores"],
            "ram_bytes": d["ram_bytes"],
            "price_per_hour": d["price_per_hour"],
            "price_per_million_tokens": d["price_per_million_tokens"],
            "currency": d["currency"],
            "status": d["status"],
            "available_from": d["available_from"],
            "available_until": d["available_until"],
            "max_concurrent_jobs": d["max_concurrent_jobs"],
            "current_jobs": d["current_jobs"],
            "supported_models": json.loads(d["supported_models"]),
            "supported_dtypes": json.loads(d["supported_dtypes"]),
            "max_batch_size": d["max_batch_size"],
            "supports_streaming": bool(d["supports_streaming"]),
            "supports_quantization": bool(d["supports_quantization"]),
            "supports_lora": bool(d["supports_lora"]),
            "region": d["region"],
            "bandwidth_mbps": d["bandwidth_mbps"],
            "latency_ms": d["latency_ms"],
            "carbon_intensity": d["carbon_intensity"],
            "renewable_pct": d["renewable_pct"],
            "reputation_score": d["reputation_score"],
            "total_jobs_completed": d["total_jobs_completed"],
            "uptime_pct": d["uptime_pct"],
            "source": d["source"],
            "created_at": d["created_at"],
            "last_updated": d["last_updated"],
            "tags": json.loads(d["tags"]),
        }

    # -- marketplace jobs ----------------------------------------------------

    def save_job(self, job: dict[str, Any]) -> None:
        with self._lock, self._transaction() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO marketplace_jobs (
                    job_id, requester_id, model_name,
                    min_gpu_memory_bytes, min_gpu_count, min_cpu_cores,
                    min_ram_bytes, required_dtype, requires_streaming,
                    requires_quantization, requires_lora,
                    max_price_per_hour, max_price_per_million_tokens,
                    max_budget_total, max_latency_ms, min_uptime_pct,
                    preferred_regions, min_reputation, status,
                    matched_listing_id, matched_provider_id,
                    tokens_generated, cost_accumulated, started_at,
                    completed_at, created_at, priority, tags
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    job["job_id"],
                    job["requester_id"],
                    job.get("model_name", ""),
                    job.get("min_gpu_memory_bytes", 0),
                    job.get("min_gpu_count", 1),
                    job.get("min_cpu_cores", 0),
                    job.get("min_ram_bytes", 0),
                    job.get("required_dtype", "float16"),
                    int(job.get("requires_streaming", True)),
                    int(job.get("requires_quantization", False)),
                    int(job.get("requires_lora", False)),
                    job.get("max_price_per_hour", 0.0),
                    job.get("max_price_per_million_tokens", 0.0),
                    job.get("max_budget_total", 0.0),
                    job.get("max_latency_ms", 5000.0),
                    job.get("min_uptime_pct", 99.0),
                    json.dumps(job.get("preferred_regions", [])),
                    job.get("min_reputation", 0.3),
                    job.get("status", "open"),
                    job.get("matched_listing_id", ""),
                    job.get("matched_provider_id", ""),
                    job.get("tokens_generated", 0),
                    job.get("cost_accumulated", 0.0),
                    job.get("started_at", 0.0),
                    job.get("completed_at", 0.0),
                    job["created_at"],
                    job.get("priority", 2),
                    json.dumps(job.get("tags", [])),
                ),
            )

    def load_job(self, job_id: str) -> dict[str, Any] | None:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM marketplace_jobs WHERE job_id=?", (job_id,)
        ).fetchone()
        return self._row_to_job(row) if row else None

    def load_all_jobs(self) -> list[dict[str, Any]]:
        conn = self._get_conn()
        rows = conn.execute("SELECT * FROM marketplace_jobs").fetchall()
        return [self._row_to_job(r) for r in rows]

    @staticmethod
    def _row_to_job(row: sqlite3.Row) -> dict[str, Any]:
        d = dict(row)
        return {
            "job_id": d["job_id"],
            "requester_id": d["requester_id"],
            "model_name": d["model_name"],
            "min_gpu_memory_bytes": d["min_gpu_memory_bytes"],
            "min_gpu_count": d["min_gpu_count"],
            "min_cpu_cores": d["min_cpu_cores"],
            "min_ram_bytes": d["min_ram_bytes"],
            "required_dtype": d["required_dtype"],
            "requires_streaming": bool(d["requires_streaming"]),
            "requires_quantization": bool(d["requires_quantization"]),
            "requires_lora": bool(d["requires_lora"]),
            "max_price_per_hour": d["max_price_per_hour"],
            "max_price_per_million_tokens": d["max_price_per_million_tokens"],
            "max_budget_total": d["max_budget_total"],
            "max_latency_ms": d["max_latency_ms"],
            "min_uptime_pct": d["min_uptime_pct"],
            "preferred_regions": json.loads(d["preferred_regions"]),
            "min_reputation": d["min_reputation"],
            "status": d["status"],
            "matched_listing_id": d["matched_listing_id"],
            "matched_provider_id": d["matched_provider_id"],
            "tokens_generated": d["tokens_generated"],
            "cost_accumulated": d["cost_accumulated"],
            "started_at": d["started_at"],
            "completed_at": d["completed_at"],
            "created_at": d["created_at"],
            "priority": d["priority"],
            "tags": json.loads(d["tags"]),
        }

    # -- provider earnings ---------------------------------------------------

    def save_provider_earnings(self, earnings: dict[str, Any]) -> None:
        with self._lock, self._transaction() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO provider_earnings (
                    provider_id, total_earnings, total_gpu_hours,
                    total_tokens_served, total_jobs,
                    current_month_earnings, pending_payout, last_payout_at
                ) VALUES (?,?,?,?,?,?,?,?)""",
                (
                    earnings["provider_id"],
                    earnings.get("total_earnings", 0.0),
                    earnings.get("total_gpu_hours", 0.0),
                    earnings.get("total_tokens_served", 0),
                    earnings.get("total_jobs", 0),
                    earnings.get("current_month_earnings", 0.0),
                    earnings.get("pending_payout", 0.0),
                    earnings.get("last_payout_at", 0.0),
                ),
            )

    def load_provider_earnings(self, provider_id: str) -> dict[str, Any] | None:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM provider_earnings WHERE provider_id=?",
            (provider_id,),
        ).fetchone()
        return dict(row) if row else None

    # -- transactions --------------------------------------------------------

    def save_transaction(self, txn: dict[str, Any]) -> None:
        with self._lock, self._transaction() as conn:
            conn.execute(
                """INSERT INTO transactions
                   (transaction_id, kind, user_id, prompt_id,
                    amount_tokens, created_at, metadata)
                   VALUES (?,?,?,?,?,?,?)""",
                (
                    txn["transaction_id"],
                    txn["kind"],
                    txn["user_id"],
                    txn.get("prompt_id"),
                    txn.get("amount_tokens", 0),
                    txn["created_at"],
                    json.dumps(txn.get("metadata", {})),
                ),
            )

    def load_transactions(
        self,
        user_id: str = "",
        kind: str = "",
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        conn = self._get_conn()
        query = "SELECT * FROM transactions WHERE 1=1"
        params: list[Any] = []
        if user_id:
            query += " AND user_id=?"
            params.append(user_id)
        if kind:
            query += " AND kind=?"
            params.append(kind)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        rows = conn.execute(query, params).fetchall()
        result: list[dict[str, Any]] = []
        for r in rows:
            d = dict(r)
            d["metadata"] = json.loads(d["metadata"])
            result.append(d)
        return result

    # -- utility -------------------------------------------------------------

    def vacuum(self) -> None:
        with self._transaction() as conn:
            conn.execute("VACUUM")
        logger.info("Exchange database vacuumed")

    def stats(self) -> dict[str, Any]:
        conn = self._get_conn()
        db_size = Path(self._db_path).stat().st_size if Path(self._db_path).exists() else 0
        return {
            "prompts": conn.execute("SELECT COUNT(*) FROM prompts").fetchone()[0],
            "reviews": conn.execute("SELECT COUNT(*) FROM prompt_reviews").fetchone()[0],
            "wallets": conn.execute("SELECT COUNT(*) FROM wallets").fetchone()[0],
            "listings": conn.execute("SELECT COUNT(*) FROM marketplace_listings").fetchone()[0],
            "jobs": conn.execute("SELECT COUNT(*) FROM marketplace_jobs").fetchone()[0],
            "transactions": conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0],
            "db_size_mb": round(db_size / (1024 * 1024), 1),
        }
