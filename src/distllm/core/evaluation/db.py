"""SQLite persistence for evaluation reports and results.

Extracted from :mod:`distllm.core.evaluation_harness`.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

from loguru import logger

from distllm.core.evaluation.constants import _DEFAULT_DB_PATH
from distllm.core.evaluation.models import EvalReport


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS eval_reports (
    report_id TEXT PRIMARY KEY,
    model_id TEXT NOT NULL,
    dataset TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    config TEXT NOT NULL DEFAULT '{}',
    metrics TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL,
    duration_s REAL NOT NULL DEFAULT 0.0
);

CREATE TABLE IF NOT EXISTS eval_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    report_id TEXT NOT NULL,
    question TEXT NOT NULL,
    answer TEXT,
    prediction TEXT NOT NULL,
    category TEXT NOT NULL DEFAULT 'general',
    score REAL NOT NULL DEFAULT 0.0,
    latency_ms REAL NOT NULL DEFAULT 0.0,
    prompt_tokens INTEGER NOT NULL DEFAULT 0,
    generated_tokens INTEGER NOT NULL DEFAULT 0,
    error TEXT,
    metadata TEXT NOT NULL DEFAULT '{}',
    FOREIGN KEY (report_id) REFERENCES eval_reports(report_id)
);

CREATE INDEX IF NOT EXISTS idx_eval_reports_model ON eval_reports(model_id);
CREATE INDEX IF NOT EXISTS idx_eval_results_report ON eval_results(report_id);
"""


class EvalDB:
    """SQLite persistence for evaluation reports and results.

    Follows the pattern from :mod:`distllm.core.persistence`.
    """

    def __init__(self, db_path: str | Path = "") -> None:
        self._db_path = Path(str(db_path) if db_path else _DEFAULT_DB_PATH)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn: sqlite3.Connection | None = None

    def initialize(self) -> None:
        """Create tables and ensure schema is current."""
        with self._lock:
            conn = self._get_conn()
            conn.executescript(_SCHEMA_SQL)
            conn.commit()
            logger.debug("Eval database initialized at {}", self._db_path)

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def save_report(self, report: EvalReport) -> None:
        """Persist an evaluation report and its results."""
        with self._lock:
            conn = self._get_conn()
            conn.execute(
                """INSERT OR REPLACE INTO eval_reports
                   (report_id, model_id, dataset, status, config, metrics, created_at, duration_s)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    report.report_id,
                    report.model_id,
                    report.dataset,
                    report.status.value,
                    json.dumps(report.config),
                    json.dumps(report.metrics),
                    report.created_at,
                    report.duration_s,
                ),
            )
            for result in report.results:
                conn.execute(
                    """INSERT INTO eval_results
                       (report_id, question, answer, prediction, category, score,
                        latency_ms, prompt_tokens, generated_tokens, error, metadata)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        report.report_id,
                        result.sample.question,
                        result.sample.answer,
                        result.prediction,
                        result.sample.category,
                        result.score,
                        result.latency_ms,
                        result.prompt_tokens,
                        result.generated_tokens,
                        result.error,
                        json.dumps(result.sample.metadata),
                    ),
                )
            conn.commit()

    def get_report(self, report_id: str) -> dict[str, Any] | None:
        """Retrieve a report header by ID."""
        with self._lock:
            conn = self._get_conn()
            row = conn.execute(
                "SELECT * FROM eval_reports WHERE report_id = ?", (report_id,)
            ).fetchone()
            if row is None:
                return None
            return dict(row)

    def get_report_results(self, report_id: str) -> list[dict[str, Any]]:
        """Retrieve all results for a given report."""
        with self._lock:
            conn = self._get_conn()
            rows = conn.execute(
                "SELECT * FROM eval_results WHERE report_id = ? ORDER BY id",
                (report_id,),
            ).fetchall()
            return [dict(r) for r in rows]

    def list_reports(
        self,
        model_id: str | None = None,
        dataset: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """List evaluation reports with optional filtering."""
        with self._lock:
            conn = self._get_conn()
            where = []
            params: list[Any] = []
            if model_id:
                where.append("model_id = ?")
                params.append(model_id)
            if dataset:
                where.append("dataset = ?")
                params.append(dataset)
            clause = f"WHERE {' AND '.join(where)}" if where else ""
            rows = conn.execute(
                f"SELECT * FROM eval_reports {clause} ORDER BY created_at DESC LIMIT ? OFFSET ?",
                [*params, limit, offset],
            ).fetchall()
            return [dict(r) for r in rows]

    def delete_report(self, report_id: str) -> bool:
        """Delete a report and its results."""
        with self._lock:
            conn = self._get_conn()
            conn.execute("DELETE FROM eval_results WHERE report_id = ?", (report_id,))
            cur = conn.execute("DELETE FROM eval_reports WHERE report_id = ?", (report_id,))
            conn.commit()
            return cur.rowcount > 0

    def close(self) -> None:
        """Close the database connection."""
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None


__all__ = [
    "EvalDB",
]
