"""SQLite-based persistent storage for batch, files, and fine-tuning APIs.

Replaces in-memory dicts with SQLite for data durability across restarts.
"""

import json
import os
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any


class PersistentStore:
    """Thread-safe SQLite storage for API data."""

    def __init__(self, db_path: str | Path = ":memory:"):
        self.db_path = str(db_path)
        self._lock = threading.Lock()
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    @contextmanager
    def _transaction(self):
        with self._lock:
            conn = self._get_conn()
            try:
                yield conn
                conn.commit()
            finally:
                conn.close()

    def _init_db(self):
        with self._transaction() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS batches (
                    id TEXT PRIMARY KEY,
                    data TEXT NOT NULL,
                    created_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS files (
                    id TEXT PRIMARY KEY,
                    data TEXT NOT NULL,
                    created_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS fine_tuning_jobs (
                    id TEXT PRIMARY KEY,
                    data TEXT NOT NULL,
                    created_at REAL NOT NULL
                );
            """)

    # --- Batch operations ---

    def save_batch(self, batch_id: str, data: dict[str, Any]) -> None:
        now = data.get("created_at", time.time())
        with self._transaction() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO batches (id, data, created_at) VALUES (?, ?, ?)",
                (batch_id, json.dumps(data), now),
            )

    def get_batch(self, batch_id: str) -> dict | None:
        with self._transaction() as conn:
            row = conn.execute("SELECT data FROM batches WHERE id = ?", (batch_id,)).fetchone()
            if row is None:
                return None
            return json.loads(row["data"])

    def list_batches(self, limit: int = 20, after: str | None = None) -> list[dict]:
        with self._transaction() as conn:
            query = "SELECT data FROM batches ORDER BY created_at DESC LIMIT ?"
            params: list = [limit]
            if after:
                query = "SELECT data FROM batches WHERE id < ? ORDER BY created_at DESC LIMIT ?"
                params = [after, limit]
            rows = conn.execute(query, params).fetchall()
            return [json.loads(r["data"]) for r in rows]

    def delete_batch(self, batch_id: str) -> bool:
        with self._transaction() as conn:
            cursor = conn.execute("DELETE FROM batches WHERE id = ?", (batch_id,))
            return cursor.rowcount > 0

    def update_batch(self, batch_id: str, updates: dict[str, Any]) -> dict | None:
        """Atomically update fields on a batch and return the updated data."""
        with self._transaction() as conn:
            row = conn.execute("SELECT data FROM batches WHERE id = ?", (batch_id,)).fetchone()
            if row is None:
                return None
            data = json.loads(row["data"])
            data.update(updates)
            conn.execute(
                "UPDATE batches SET data = ? WHERE id = ?",
                (json.dumps(data), batch_id),
            )
            return data

    # --- File operations ---

    def save_file(self, file_id: str, data: dict[str, Any]) -> None:
        now = data.get("created_at", time.time())
        with self._transaction() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO files (id, data, created_at) VALUES (?, ?, ?)",
                (file_id, json.dumps(data), now),
            )

    def get_file(self, file_id: str) -> dict | None:
        with self._transaction() as conn:
            row = conn.execute("SELECT data FROM files WHERE id = ?", (file_id,)).fetchone()
            return json.loads(row["data"]) if row else None

    def list_files(self, purpose: str | None = None) -> list[dict]:
        with self._transaction() as conn:
            if purpose:
                rows = conn.execute(
                    "SELECT data FROM files ORDER BY created_at DESC"
                ).fetchall()
                files = [json.loads(r["data"]) for r in rows]
                return [f for f in files if f.get("purpose") == purpose]
            else:
                rows = conn.execute(
                    "SELECT data FROM files ORDER BY created_at DESC"
                ).fetchall()
                return [json.loads(r["data"]) for r in rows]

    def delete_file(self, file_id: str) -> bool:
        with self._transaction() as conn:
            cursor = conn.execute("DELETE FROM files WHERE id = ?", (file_id,))
            return cursor.rowcount > 0

    # --- Fine-tuning operations ---

    def save_fine_tuning_job(self, job_id: str, data: dict[str, Any]) -> None:
        now = data.get("created_at", time.time())
        with self._transaction() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO fine_tuning_jobs (id, data, created_at) VALUES (?, ?, ?)",
                (job_id, json.dumps(data), now),
            )

    def get_fine_tuning_job(self, job_id: str) -> dict | None:
        with self._transaction() as conn:
            row = conn.execute(
                "SELECT data FROM fine_tuning_jobs WHERE id = ?", (job_id,)
            ).fetchone()
            return json.loads(row["data"]) if row else None

    def list_fine_tuning_jobs(self, limit: int = 20) -> list[dict]:
        with self._transaction() as conn:
            rows = conn.execute(
                "SELECT data FROM fine_tuning_jobs ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [json.loads(r["data"]) for r in rows]

    def update_fine_tuning_job(self, job_id: str, updates: dict[str, Any]) -> dict | None:
        """Atomically update fields on a fine-tuning job and return updated data."""
        with self._transaction() as conn:
            row = conn.execute("SELECT data FROM fine_tuning_jobs WHERE id = ?", (job_id,)).fetchone()
            if row is None:
                return None
            data = json.loads(row["data"])
            data.update(updates)
            conn.execute(
                "UPDATE fine_tuning_jobs SET data = ? WHERE id = ?",
                (json.dumps(data), job_id),
            )
            return data


# Global store instance - use env var for persistent path
_store: PersistentStore | None = None
_lock = threading.Lock()


def get_data_dir() -> Path:
    """Return the durable API data directory, creating it if needed."""
    db_dir = Path(os.getenv("DISTLLM_DATA_DIR", ".distllm")).expanduser()
    db_dir.mkdir(parents=True, exist_ok=True)
    return db_dir


def get_store() -> PersistentStore:
    """Get or create the global PersistentStore."""
    global _store
    if _store is None:
        with _lock:
            if _store is None:
                db_path = get_data_dir() / "distllm.db"
                _store = PersistentStore(db_path)
    return _store
