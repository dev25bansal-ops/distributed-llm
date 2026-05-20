"""Persistent storage layer using SQLite for jobs, batch results, audit logs, and sessions.

Replaces in-memory storage for production durability across restarts.
Supports connection pooling, migrations, and thread-safe operations.
"""

import sqlite3
import threading
import time
import json
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger


# Schema version for migrations
_SCHEMA_VERSION = 1

_CREATE_TABLES = """
CREATE TABLE IF NOT EXISTS jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT UNIQUE NOT NULL,
    type TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    request_id TEXT,
    model TEXT,
    prompt TEXT,
    result TEXT,
    error TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    completed_at REAL,
    ttl REAL,
    metadata TEXT
);

CREATE TABLE IF NOT EXISTS batch_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_id TEXT UNIQUE NOT NULL,
    request_ids TEXT NOT NULL,
    results TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    created_at REAL NOT NULL,
    completed_at REAL,
    metadata TEXT
);

CREATE TABLE IF NOT EXISTS audit_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL NOT NULL,
    event_type TEXT NOT NULL,
    actor TEXT,
    request_id TEXT,
    details TEXT,
    ip_address TEXT
);

CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT UNIQUE NOT NULL,
    user_id TEXT,
    tenant TEXT NOT NULL DEFAULT 'default',
    created_at REAL NOT NULL,
    last_active REAL NOT NULL,
    metadata TEXT,
    expires_at REAL
);

CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY
);

CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_jobs_created ON jobs(created_at);
CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_logs(timestamp);
CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);
"""


@dataclass
class JobRecord:
    """A job record in the persistent store."""
    job_id: str
    type: str
    status: str = "pending"
    request_id: str | None = None
    model: str | None = None
    prompt: str | None = None
    result: str | None = None
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    completed_at: float | None = None
    ttl: float | None = None
    metadata: dict | None = None


@dataclass
class AuditRecord:
    """An audit log entry."""
    event_type: str
    actor: str | None = None
    request_id: str | None = None
    details: dict | None = None
    ip_address: str | None = None
    timestamp: float = field(default_factory=time.time)


class PersistentStore:
    """SQLite-backed persistent storage for jobs, batch results, audit, and sessions.

    Thread-safe with WAL mode for concurrent reads. Uses connection
    pooling via a thread-local pattern.

    Usage:
        store = PersistentStore("distllm.db")
        store.initialize()
        store.create_job(JobRecord(...))
        jobs = store.list_jobs(status="pending")
    """

    def __init__(self, db_path: str = "distllm.db", wal_mode: bool = True):
        self._db_path = db_path
        self._wal_mode = wal_mode
        self._local = threading.local()
        self._lock = threading.Lock()

    def _get_conn(self) -> sqlite3.Connection:
        """Get thread-local connection."""
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
    def _transaction(self):
        """Context manager for transactional operations."""
        conn = self._get_conn()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def initialize(self) -> int:
        """Initialize the database schema.

        Returns:
            Schema version number.
        """
        with self._transaction() as conn:
            conn.executescript(_CREATE_TABLES)
            # Check if version exists
            row = conn.execute("SELECT version FROM schema_version").fetchone()
            if row is None:
                conn.execute("INSERT INTO schema_version (version) VALUES (?)", (_SCHEMA_VERSION,))
            else:
                current = row["version"]
                if current < _SCHEMA_VERSION:
                    self._migrate(conn, current, _SCHEMA_VERSION)
        logger.info(f"Persistent store initialized (schema v{_SCHEMA_VERSION})")
        return _SCHEMA_VERSION

    def _migrate(self, conn, from_version: int, to_version: int) -> None:
        """Run migrations from current to target schema version."""
        logger.info(f"Migrating schema from v{from_version} to v{to_version}")
        # For now, v1 is the only version — add migration logic here for future versions

    # -- Jobs --

    def create_job(self, job: JobRecord) -> str:
        """Create a new job record."""
        with self._transaction() as conn:
            conn.execute(
                """INSERT INTO jobs (job_id, type, status, request_id, model, prompt,
                   created_at, updated_at, ttl, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    job.job_id, job.type, job.status, job.request_id,
                    job.model, job.prompt, job.created_at, job.updated_at,
                    job.ttl, json.dumps(job.metadata) if job.metadata else None,
                ),
            )
        return job.job_id

    def get_job(self, job_id: str) -> JobRecord | None:
        """Get a job by ID."""
        conn = self._get_conn()
        row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        if row is None:
            return None
        return self._row_to_job(row)

    def update_job_status(self, job_id: str, status: str, result: str | None = None, error: str | None = None) -> None:
        """Update a job's status and optionally result/error."""
        now = time.time()
        with self._transaction() as conn:
            if result is not None or error is not None:
                completed_at = now if status in ("completed", "failed") else None
                conn.execute(
                    """UPDATE jobs SET status=?, result=?, error=?, updated_at=?, completed_at=?
                       WHERE job_id=?""",
                    (status, result, error, now, completed_at, job_id),
                )
            else:
                conn.execute(
                    "UPDATE jobs SET status=?, updated_at=? WHERE job_id=?",
                    (status, now, job_id),
                )

    def list_jobs(self, status: str | None = None, limit: int = 100, offset: int = 0) -> list[JobRecord]:
        """List jobs with optional status filter."""
        conn = self._get_conn()
        if status:
            rows = conn.execute(
                "SELECT * FROM jobs WHERE status=? ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (status, limit, offset),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
        return [self._row_to_job(r) for r in rows]

    def delete_expired_jobs(self) -> int:
        """Delete jobs past their TTL."""
        now = time.time()
        with self._transaction() as conn:
            cursor = conn.execute(
                "DELETE FROM jobs WHERE ttl IS NOT NULL AND (created_at + ttl) < ?",
                (now,),
            )
            deleted = cursor.rowcount
        if deleted:
            logger.info(f"Deleted {deleted} expired jobs")
        return deleted

    # -- Audit Logs --

    def add_audit(self, record: AuditRecord) -> int:
        """Add an audit log entry."""
        with self._transaction() as conn:
            cursor = conn.execute(
                """INSERT INTO audit_logs (timestamp, event_type, actor, request_id, details, ip_address)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    record.timestamp, record.event_type, record.actor,
                    record.request_id, json.dumps(record.details) if record.details else None,
                    record.ip_address,
                ),
            )
            return cursor.lastrowid

    def list_audit_logs(
        self, event_type: str | None = None, start: float | None = None,
        end: float | None = None, limit: int = 100,
    ) -> list[dict]:
        """List audit log entries."""
        conn = self._get_conn()
        query = "SELECT * FROM audit_logs WHERE 1=1"
        params: list[Any] = []
        if event_type:
            query += " AND event_type=?"
            params.append(event_type)
        if start:
            query += " AND timestamp>=?"
            params.append(start)
        if end:
            query += " AND timestamp<=?"
            params.append(end)
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    # -- Sessions --

    def create_session(self, session_id: str, user_id: str | None = None,
                       tenant: str = "default", metadata: dict | None = None,
                       ttl_seconds: float | None = 3600) -> str:
        """Create a new session."""
        now = time.time()
        with self._transaction() as conn:
            conn.execute(
                """INSERT INTO sessions (session_id, user_id, tenant, created_at, last_active, metadata, expires_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    session_id, user_id, tenant, now, now,
                    json.dumps(metadata) if metadata else None,
                    now + ttl_seconds if ttl_seconds else None,
                ),
            )
        return session_id

    def get_session(self, session_id: str) -> dict | None:
        """Get a session by ID."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM sessions WHERE session_id=?", (session_id,)
        ).fetchone()
        if row is None:
            return None
        d = dict(row)
        if d.get("metadata"):
            d["metadata"] = json.loads(d["metadata"])
        return d

    def update_session_activity(self, session_id: str) -> bool:
        """Update session last_active timestamp."""
        now = time.time()
        with self._transaction() as conn:
            cursor = conn.execute(
                "UPDATE sessions SET last_active=? WHERE session_id=? AND (expires_at IS NULL OR expires_at > ?)",
                (now, session_id, now),
            )
            return cursor.rowcount > 0

    def delete_expired_sessions(self) -> int:
        """Delete expired sessions."""
        now = time.time()
        with self._transaction() as conn:
            cursor = conn.execute(
                "DELETE FROM sessions WHERE expires_at IS NOT NULL AND expires_at < ?",
                (now,),
            )
            return cursor.rowcount

    # -- Batch Results --

    def store_batch_result(self, batch_id: str, request_ids: list[str],
                           status: str = "pending", metadata: dict | None = None) -> None:
        """Store a batch result record."""
        now = time.time()
        with self._transaction() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO batch_results (batch_id, request_ids, results, status, created_at, metadata)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (batch_id, json.dumps(request_ids), None, status, now,
                 json.dumps(metadata) if metadata else None),
            )

    def update_batch_result(self, batch_id: str, results: list[dict], status: str = "completed") -> None:
        """Update a batch result with completion data."""
        now = time.time()
        with self._transaction() as conn:
            conn.execute(
                """UPDATE batch_results SET results=?, status=?, completed_at=? WHERE batch_id=?""",
                (json.dumps(results), status, now, batch_id),
            )

    def get_batch_result(self, batch_id: str) -> dict | None:
        """Get a batch result by ID."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM batch_results WHERE batch_id=?", (batch_id,)
        ).fetchone()
        if row is None:
            return None
        d = dict(row)
        for key in ("request_ids", "results", "metadata"):
            if d.get(key):
                d[key] = json.loads(d[key])
        return d

    # -- Utility --

    def _row_to_job(self, row: sqlite3.Row) -> JobRecord:
        """Convert a database row to JobRecord."""
        d = dict(row)
        metadata = None
        if d.get("metadata"):
            metadata = json.loads(d["metadata"])
        return JobRecord(
            job_id=d["job_id"],
            type=d["type"],
            status=d["status"],
            request_id=d.get("request_id"),
            model=d.get("model"),
            prompt=d.get("prompt"),
            result=d.get("result"),
            error=d.get("error"),
            created_at=d["created_at"],
            updated_at=d["updated_at"],
            completed_at=d.get("completed_at"),
            ttl=d.get("ttl"),
            metadata=metadata,
        )

    def close(self) -> None:
        """Close the thread-local connection."""
        if hasattr(self._local, "conn") and self._local.conn is not None:
            self._local.conn.close()
            self._local.conn = None

    def vacuum(self) -> None:
        """Reclaim unused space in the database."""
        with self._transaction() as conn:
            conn.execute("VACUUM")
        logger.info("Database vacuumed")

    def stats(self) -> dict:
        """Get database statistics."""
        conn = self._get_conn()
        return {
            "jobs_total": conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0],
            "jobs_pending": conn.execute("SELECT COUNT(*) FROM jobs WHERE status='pending'").fetchone()[0],
            "jobs_completed": conn.execute("SELECT COUNT(*) FROM jobs WHERE status='completed'").fetchone()[0],
            "audit_entries": conn.execute("SELECT COUNT(*) FROM audit_logs").fetchone()[0],
            "sessions_active": conn.execute(
                "SELECT COUNT(*) FROM sessions WHERE expires_at IS NULL OR expires_at > ?",
                (time.time(),),
            ).fetchone()[0],
            "db_size_mb": round(Path(self._db_path).stat().st_size / (1024 * 1024), 1),
        }
