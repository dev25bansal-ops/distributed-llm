"""Persistent model registry with lineage tracking and SQLite backing.

Extends the in-memory model registry with persistent storage for
model versions, lineage/provenance tracking, and artifact metadata.
Survives coordinator restarts.
"""

import hashlib
import json
import time
import threading
from dataclasses import dataclass, field
from typing import Any

from loguru import logger


@dataclass
class ModelLineage:
    """Lineage/provenance information for a model version."""
    source: str = ""  # "huggingface", "local", "compressed", "fine_tuned"
    source_url: str = ""
    commit_hash: str = ""
    training_dataset: str = ""
    base_model: str = ""  # Parent model if derived
    compression_method: str = ""  # "awq", "gptq", "int4", etc.
    compression_ratio: float = 0.0
    created_by: str = ""
    created_at: float = field(default_factory=time.time)
    signature: str = ""  # Model artifact hash for verification

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "source_url": self.source_url,
            "commit_hash": self.commit_hash,
            "training_dataset": self.training_dataset,
            "base_model": self.base_model,
            "compression_method": self.compression_method,
            "compression_ratio": self.compression_ratio,
            "created_by": self.created_by,
            "created_at": self.created_at,
            "signature": self.signature,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ModelLineage":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class ModelVersion:
    """A single version of a model with lineage."""
    version_id: str
    name: str  # Model name (e.g., "llama-3-8b")
    path: str
    total_layers: int
    precision: str = "float16"
    vram_mb: float = 0.0
    throughput_tok_s: float = 0.0
    latency_p50_ms: float = 0.0
    latency_p99_ms: float = 0.0
    registered_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    use_count: int = 0
    status: str = "active"  # active, deprecated, archived
    lineage: ModelLineage | None = None
    metadata: dict = field(default_factory=dict)

    def compute_signature(self) -> str:
        """Compute a signature hash for the model version."""
        h = hashlib.sha256(
            f"{self.name}:{self.version_id}:{self.path}:{self.precision}".encode()
        ).hexdigest()[:16]
        return h


class PersistentModelRegistry:
    """SQLite-backed model registry with lineage tracking.

    Thread-safe with persistent storage. Supports version management,
    A/B testing config, lineage tracking, and model search.

    Usage:
        registry = PersistentModelRegistry("models.db")
        registry.initialize()
        registry.register_version(ModelVersion(...))
        versions = registry.list_versions("llama-3-8b")
    """

    _CREATE_TABLES = """
    CREATE TABLE IF NOT EXISTS model_versions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        version_id TEXT UNIQUE NOT NULL,
        name TEXT NOT NULL,
        path TEXT NOT NULL,
        total_layers INTEGER NOT NULL,
        precision TEXT DEFAULT 'float16',
        vram_mb REAL DEFAULT 0,
        throughput_tok_s REAL DEFAULT 0,
        latency_p50_ms REAL DEFAULT 0,
        latency_p99_ms REAL DEFAULT 0,
        registered_at REAL NOT NULL,
        last_used REAL NOT NULL,
        use_count INTEGER DEFAULT 0,
        status TEXT DEFAULT 'active',
        lineage TEXT,
        metadata TEXT
    );

    CREATE TABLE IF NOT EXISTS model_configs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT UNIQUE NOT NULL,
        default_version TEXT,
        ab_test_enabled INTEGER DEFAULT 0,
        ab_traffic_split TEXT,
        created_at REAL NOT NULL
    );

    CREATE TABLE IF NOT EXISTS model_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        version_id TEXT NOT NULL,
        event_type TEXT NOT NULL,
        timestamp REAL NOT NULL,
        details TEXT
    );

    CREATE INDEX IF NOT EXISTS idx_mv_name ON model_versions(name);
    CREATE INDEX IF NOT EXISTS idx_mv_status ON model_versions(status);
    CREATE INDEX IF NOT EXISTS idx_events_version ON model_events(version_id);
    """

    def __init__(self, db_path: str = "models.db"):
        self._db_path = db_path
        self._local = threading.local()
        self._lock = threading.Lock()

    def _get_conn(self) -> Any:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            import sqlite3
            self._local.conn = sqlite3.connect(self._db_path, timeout=30)
            self._local.conn.row_factory = sqlite3.Row
            self._local.conn.execute("PRAGMA journal_mode=WAL")
        return self._local.conn

    def initialize(self) -> None:
        """Initialize the database schema."""
        conn = self._get_conn()
        conn.executescript(self._CREATE_TABLES)
        conn.commit()
        logger.info("Persistent model registry initialized")

    def register_version(self, version: ModelVersion) -> str:
        """Register a new model version."""
        if version.lineage is None:
            version.lineage = ModelLineage()
        version.lineage.signature = version.compute_signature()

        conn = self._get_conn()
        conn.execute(
            """INSERT INTO model_versions
               (version_id, name, path, total_layers, precision, vram_mb,
                throughput_tok_s, latency_p50_ms, latency_p99_ms,
                registered_at, last_used, use_count, status, lineage, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                version.version_id, version.name, version.path,
                version.total_layers, version.precision, version.vram_mb,
                version.throughput_tok_s, version.latency_p50_ms, version.latency_p99_ms,
                version.registered_at, version.last_used, version.use_count,
                version.status,
                json.dumps(version.lineage.to_dict()),
                json.dumps(version.metadata),
            ),
        )
        conn.commit()

        # Ensure model config exists
        self._ensure_model_config(version.name)

        # Log event
        self._log_event(version.version_id, "registered", {"path": version.path})

        logger.info(f"Registered model version {version.version_id} ({version.name})")
        return version.version_id

    def get_version(self, name: str, version_id: str) -> ModelVersion | None:
        """Get a specific model version."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM model_versions WHERE name=? AND version_id=?",
            (name, version_id),
        ).fetchone()
        if row is None:
            return None
        return self._row_to_version(row)

    def get_default_version(self, name: str) -> ModelVersion | None:
        """Get the default version for a model name."""
        conn = self._get_conn()
        config = conn.execute(
            "SELECT default_version FROM model_configs WHERE name=?", (name,)
        ).fetchone()
        if config is None or config["default_version"] is None:
            # Return most recent active version
            row = conn.execute(
                """SELECT * FROM model_versions WHERE name=? AND status='active'
                   ORDER BY registered_at DESC LIMIT 1""",
                (name,),
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT * FROM model_versions WHERE name=? AND version_id=?",
                (name, config["default_version"]),
            ).fetchone()

        if row is None:
            return None
        return self._row_to_version(row)

    def list_versions(self, name: str, status: str | None = None) -> list[ModelVersion]:
        """List all versions of a model."""
        conn = self._get_conn()
        if status:
            rows = conn.execute(
                "SELECT * FROM model_versions WHERE name=? AND status=? ORDER BY registered_at DESC",
                (name, status),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM model_versions WHERE name=? ORDER BY registered_at DESC",
                (name,),
            ).fetchall()
        return [self._row_to_version(r) for r in rows]

    def list_models(self) -> list[str]:
        """List all registered model names."""
        conn = self._get_conn()
        rows = conn.execute("SELECT DISTINCT name FROM model_versions ORDER BY name").fetchall()
        return [r["name"] for r in rows]

    def set_default_version(self, name: str, version_id: str) -> bool:
        """Set the default version for a model."""
        conn = self._get_conn()
        # Verify version exists
        row = conn.execute(
            "SELECT version_id FROM model_versions WHERE name=? AND version_id=?",
            (name, version_id),
        ).fetchone()
        if row is None:
            return False

        conn.execute(
            "UPDATE model_configs SET default_version=? WHERE name=?",
            (version_id, name),
        )
        conn.commit()
        self._log_event(version_id, "set_default", {"model": name})
        return True

    def deprecate_version(self, name: str, version_id: str) -> bool:
        """Mark a model version as deprecated."""
        conn = self._get_conn()
        cursor = conn.execute(
            "UPDATE model_versions SET status='deprecated' WHERE name=? AND version_id=?",
            (name, version_id),
        )
        conn.commit()
        if cursor.rowcount > 0:
            self._log_event(version_id, "deprecated", {"model": name})
            return True
        return False

    def record_usage(self, version_id: str, throughput: float = 0, latency_p50: float = 0, latency_p99: float = 0) -> None:
        """Record model usage metrics."""
        conn = self._get_conn()
        conn.execute(
            """UPDATE model_versions
               SET throughput_tok_s=?, latency_p50_ms=?, latency_p99_ms=?,
                   use_count=use_count+1, last_used=?
               WHERE version_id=?""",
            (throughput, latency_p50, latency_p99, time.time(), version_id),
        )
        conn.commit()

    def get_lineage(self, version_id: str) -> ModelLineage | None:
        """Get lineage information for a model version."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT lineage FROM model_versions WHERE version_id=?", (version_id,)
        ).fetchone()
        if row is None or row["lineage"] is None:
            return None
        return ModelLineage.from_dict(json.loads(row["lineage"]))

    def search_models(
        self,
        precision: str | None = None,
        max_vram_mb: float | None = None,
        status: str | None = None,
    ) -> list[ModelVersion]:
        """Search model versions by criteria."""
        conn = self._get_conn()
        query = "SELECT * FROM model_versions WHERE 1=1"
        params: list = []
        if precision:
            query += " AND precision=?"
            params.append(precision)
        if max_vram_mb is not None:
            query += " AND vram_mb<=?"
            params.append(max_vram_mb)
        if status:
            query += " AND status=?"
            params.append(status)
        query += " ORDER BY registered_at DESC"

        rows = conn.execute(query, params).fetchall()
        return [self._row_to_version(r) for r in rows]

    def _ensure_model_config(self, name: str) -> None:
        """Create model config if it doesn't exist."""
        conn = self._get_conn()
        conn.execute(
            "INSERT OR IGNORE INTO model_configs (name, created_at) VALUES (?, ?)",
            (name, time.time()),
        )
        conn.commit()

    def _log_event(self, version_id: str, event_type: str, details: dict | None = None) -> None:
        """Log a model event."""
        conn = self._get_conn()
        conn.execute(
            "INSERT INTO model_events (version_id, event_type, timestamp, details) VALUES (?, ?, ?, ?)",
            (version_id, event_type, time.time(), json.dumps(details) if details else None),
        )
        conn.commit()

    def get_events(self, version_id: str, limit: int = 50) -> list[dict]:
        """Get events for a model version."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM model_events WHERE version_id=? ORDER BY timestamp DESC LIMIT ?",
            (version_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def _row_to_version(self, row) -> ModelVersion:
        """Convert database row to ModelVersion."""
        d = dict(row)
        lineage = None
        if d.get("lineage"):
            lineage = ModelLineage.from_dict(json.loads(d["lineage"]))
        metadata = {}
        if d.get("metadata"):
            metadata = json.loads(d["metadata"])
        return ModelVersion(
            version_id=d["version_id"],
            name=d["name"],
            path=d["path"],
            total_layers=d["total_layers"],
            precision=d.get("precision", "float16"),
            vram_mb=d.get("vram_mb", 0),
            throughput_tok_s=d.get("throughput_tok_s", 0),
            latency_p50_ms=d.get("latency_p50_ms", 0),
            latency_p99_ms=d.get("latency_p99_ms", 0),
            registered_at=d["registered_at"],
            last_used=d["last_used"],
            use_count=d.get("use_count", 0),
            status=d.get("status", "active"),
            lineage=lineage,
            metadata=metadata,
        )

    def stats(self) -> dict:
        """Get registry statistics."""
        conn = self._get_conn()
        return {
            "total_models": conn.execute("SELECT COUNT(DISTINCT name) FROM model_versions").fetchone()[0],
            "total_versions": conn.execute("SELECT COUNT(*) FROM model_versions").fetchone()[0],
            "active_versions": conn.execute("SELECT COUNT(*) FROM model_versions WHERE status='active'").fetchone()[0],
            "deprecated_versions": conn.execute("SELECT COUNT(*) FROM model_versions WHERE status='deprecated'").fetchone()[0],
            "total_events": conn.execute("SELECT COUNT(*) FROM model_events").fetchone()[0],
        }
