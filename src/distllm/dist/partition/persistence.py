"""Partition persistence and versioning system.

SQLite-based store for tracking partition quality over time,
comparing between model versions, and rolling back to last-known-good
partitions.

Typical usage::

    store = PartitionStore("partitions.db")

    # Save a partition run
    run_id = store.save_run(
        model_name="meta-llama/Llama-3-70B",
        solution=solution,
        config={"hidden_size": 8192, "num_layers": 80},
        gpu_profiles=[...],
    )

    # Record actual runtime metrics
    store.record_metric(run_id, "actual_latency_ms", 45.2)
    store.record_metric(run_id, "actual_throughput_tok_s", 120.0)

    # Retrieve history
    runs = store.get_runs(model_name="meta-llama/Llama-3-70B")
    best = store.get_best_run(model_name="meta-llama/Llama-3-70B")

    # A/B comparison
    comparison = store.compare_runs(run_id_a, run_id_b)
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger


@dataclass
class PartitionRun:
    """A recorded partition run."""
    run_id: int
    model_name: str
    created_at: float
    config: dict[str, Any]
    solution: dict[str, Any]
    gpu_profiles: list[dict[str, Any]]
    metrics: dict[str, float] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    is_good: bool = True


@dataclass
class RunComparison:
    """Comparison between two partition runs."""
    run_a: PartitionRun
    run_b: PartitionRun
    latency_diff_ms: float
    latency_diff_pct: float
    throughput_diff_tok_s: float
    throughput_diff_pct: float
    memory_diff_gb: float
    winner: str
    summary: str


class PartitionStore:
    """SQLite-backed partition history store.

    Args:
        db_path: Path to SQLite database file.
    """

    def __init__(self, db_path: str | Path = "~/.distllm/partitions.db"):
        self._db_path = Path(db_path).expanduser().resolve()
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS runs (
                run_id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT NOT NULL,
                created_at REAL NOT NULL,
                config_json TEXT NOT NULL,
                solution_json TEXT NOT NULL,
                gpu_profiles_json TEXT NOT NULL,
                tags_json TEXT DEFAULT '[]',
                is_good INTEGER DEFAULT 1
            );

            CREATE TABLE IF NOT EXISTS metrics (
                metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                recorded_at REAL NOT NULL,
                FOREIGN KEY (run_id) REFERENCES runs(run_id)
            );

            CREATE INDEX IF NOT EXISTS idx_runs_model ON runs(model_name);
            CREATE INDEX IF NOT EXISTS idx_runs_created ON runs(created_at);
            CREATE INDEX IF NOT EXISTS idx_metrics_run ON metrics(run_id);
            CREATE INDEX IF NOT EXISTS idx_metrics_name ON metrics(metric_name);
        """)
        self._conn.commit()

    def save_run(
        self,
        model_name: str,
        solution: Any,
        config: dict[str, Any],
        gpu_profiles: list[dict[str, Any]] | None = None,
        tags: list[str] | None = None,
    ) -> int:
        """Save a partition run to the store.

        Args:
            model_name: Model identifier.
            solution: PartitionSolution or dict.
            config: Partition configuration.
            gpu_profiles: GPU profile dicts.
            tags: Optional tags for filtering.

        Returns:
            The run_id of the saved run.
        """
        solution_dict = self._solution_to_dict(solution)
        profiles = gpu_profiles or []
        tags_list = tags or []

        cursor = self._conn.execute(
            """INSERT INTO runs (model_name, created_at, config_json, solution_json,
               gpu_profiles_json, tags_json) VALUES (?, ?, ?, ?, ?, ?)""",
            (
                model_name,
                time.time(),
                json.dumps(config),
                json.dumps(solution_dict),
                json.dumps(profiles),
                json.dumps(tags_list),
            ),
        )
        self._conn.commit()
        run_id = cursor.lastrowid
        logger.debug(f"Saved partition run {run_id} for {model_name}")
        return run_id  # type: ignore[return-value]

    def record_metric(
        self, run_id: int, metric_name: str, metric_value: float,
    ) -> None:
        """Record an actual runtime metric for a partition run.

        Args:
            run_id: The partition run ID.
            metric_name: Name of the metric (e.g., "actual_latency_ms").
            metric_value: Measured value.
        """
        self._conn.execute(
            "INSERT INTO metrics (run_id, metric_name, metric_value, recorded_at) VALUES (?, ?, ?, ?)",
            (run_id, metric_name, metric_value, time.time()),
        )
        self._conn.commit()

    def get_run(self, run_id: int) -> PartitionRun | None:
        """Retrieve a single partition run by ID."""
        row = self._conn.execute(
            "SELECT * FROM runs WHERE run_id = ?", (run_id,),
        ).fetchone()
        if row is None:
            return None
        return self._row_to_run(row)

    def get_runs(
        self,
        model_name: str | None = None,
        limit: int = 50,
        tags: list[str] | None = None,
    ) -> list[PartitionRun]:
        """Retrieve partition runs with optional filtering."""
        query = "SELECT * FROM runs WHERE 1=1"
        params: list[Any] = []

        if model_name:
            query += " AND model_name = ?"
            params.append(model_name)

        if tags:
            for tag in tags:
                query += " AND tags_json LIKE ?"
                params.append(f'%"{tag}"%')

        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        rows = self._conn.execute(query, params).fetchall()
        return [self._row_to_run(row) for row in rows]

    def get_best_run(
        self, model_name: str, metric: str = "actual_latency_ms",
    ) -> PartitionRun | None:
        """Get the best-performing run for a model based on a metric.

        Args:
            model_name: Model identifier.
            metric: Metric name to optimize (lower is better for latency).

        Returns:
            Best PartitionRun, or None if no metrics recorded.
        """
        row = self._conn.execute("""
            SELECT r.*, m.metric_value
            FROM runs r
            JOIN metrics m ON r.run_id = m.run_id
            WHERE r.model_name = ? AND m.metric_name = ?
            ORDER BY m.metric_value ASC
            LIMIT 1
        """, (model_name, metric)).fetchone()

        if row is None:
            return None
        return self._row_to_run(row)

    def mark_run_quality(self, run_id: int, is_good: bool) -> None:
        """Mark a run as good or bad (for rollback filtering)."""
        self._conn.execute(
            "UPDATE runs SET is_good = ? WHERE run_id = ?",
            (1 if is_good else 0, run_id),
        )
        self._conn.commit()

    def get_last_known_good(
        self, model_name: str,
    ) -> PartitionRun | None:
        """Get the most recent good run for a model (for rollback)."""
        row = self._conn.execute(
            "SELECT * FROM runs WHERE model_name = ? AND is_good = 1 ORDER BY created_at DESC LIMIT 1",
            (model_name,),
        ).fetchone()
        if row is None:
            return None
        return self._row_to_run(row)

    def compare_runs(self, run_id_a: int, run_id_b: int) -> RunComparison | None:
        """Compare two partition runs side by side."""
        run_a = self.get_run(run_id_a)
        run_b = self.get_run(run_id_b)
        if run_a is None or run_b is None:
            return None

        metrics_a = self._get_metrics_dict(run_id_a)
        metrics_b = self._get_metrics_dict(run_id_b)

        lat_a = metrics_a.get("actual_latency_ms", run_a.solution.get("max_node_time_ms", 0))
        lat_b = metrics_b.get("actual_latency_ms", run_b.solution.get("max_node_time_ms", 0))
        tp_a = metrics_a.get("actual_throughput_tok_s", run_a.solution.get("estimated_throughput_tok_s", 0))
        tp_b = metrics_b.get("actual_throughput_tok_s", run_b.solution.get("estimated_throughput_tok_s", 0))
        mem_a = run_a.solution.get("total_memory_gb", 0)
        mem_b = run_b.solution.get("total_memory_gb", 0)

        lat_diff = lat_b - lat_a
        lat_pct = (lat_diff / max(lat_a, 0.001)) * 100
        tp_diff = tp_b - tp_a
        tp_pct = (tp_diff / max(tp_a, 0.001)) * 100
        mem_diff = mem_b - mem_a

        if lat_diff < 0 and tp_diff > 0:
            winner = "B"
        elif lat_diff > 0 and tp_diff < 0:
            winner = "A"
        elif abs(lat_pct) < 2 and abs(tp_pct) < 2:
            winner = "tie"
        elif abs(lat_pct) < abs(tp_pct):
            winner = "B" if tp_diff > 0 else "A"
        else:
            winner = "A" if lat_diff < 0 else "B"

        return RunComparison(
            run_a=run_a,
            run_b=run_b,
            latency_diff_ms=round(lat_diff, 2),
            latency_diff_pct=round(lat_pct, 1),
            throughput_diff_tok_s=round(tp_diff, 0),
            throughput_diff_pct=round(tp_pct, 1),
            memory_diff_gb=round(mem_diff, 2),
            winner=winner,
            summary=(
                f"Run {run_id_a} vs {run_id_b}: "
                f"latency {lat_a:.1f}ms → {lat_b:.1f}ms ({lat_pct:+.1f}%), "
                f"throughput {tp_a:.0f} → {tp_b:.0f} tok/s ({tp_pct:+.1f}%), "
                f"winner: {winner}"
            ),
        )

    def get_accuracy_report(self, model_name: str) -> dict[str, Any]:
        """Report predictive accuracy: predicted vs actual metrics.

        Returns dict with MAE, MAPE, and per-run accuracy.
        """
        runs = self.get_runs(model_name=model_name, limit=200)
        entries: list[dict[str, Any]] = []
        errors: list[float] = []

        for run in runs:
            metrics = self._get_metrics_dict(run.run_id)
            predicted_lat = run.solution.get("max_node_time_ms", 0)
            actual_lat = metrics.get("actual_latency_ms")
            if actual_lat is None or predicted_lat <= 0:
                continue

            error = abs(actual_lat - predicted_lat)
            pct_error = (error / max(actual_lat, 0.001)) * 100
            errors.append(error)
            entries.append({
                "run_id": run.run_id,
                "predicted_ms": predicted_lat,
                "actual_ms": actual_lat,
                "error_ms": round(error, 2),
                "error_pct": round(pct_error, 1),
            })

        if not errors:
            return {"num_samples": 0, "mae_ms": 0, "mape_pct": 0, "entries": []}

        mae = sum(errors) / len(errors)
        pct_errors = [abs(e["error_pct"]) for e in entries]
        mape = sum(pct_errors) / len(pct_errors)

        return {
            "num_samples": len(entries),
            "mae_ms": round(mae, 2),
            "mape_pct": round(mape, 1),
            "max_error_ms": round(max(errors), 2),
            "entries": entries[-20:],
        }

    def delete_run(self, run_id: int) -> None:
        """Delete a partition run and its metrics."""
        self._conn.execute("DELETE FROM metrics WHERE run_id = ?", (run_id,))
        self._conn.execute("DELETE FROM runs WHERE run_id = ?", (run_id,))
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def _row_to_run(self, row: sqlite3.Row) -> PartitionRun:
        metrics = self._get_metrics_dict(row["run_id"])
        return PartitionRun(
            run_id=row["run_id"],
            model_name=row["model_name"],
            created_at=row["created_at"],
            config=json.loads(row["config_json"]),
            solution=json.loads(row["solution_json"]),
            gpu_profiles=json.loads(row["gpu_profiles_json"]),
            metrics=metrics,
            tags=json.loads(row["tags_json"]),
            is_good=bool(row["is_good"]),
        )

    def _get_metrics_dict(self, run_id: int) -> dict[str, float]:
        rows = self._conn.execute(
            "SELECT metric_name, metric_value FROM metrics WHERE run_id = ?",
            (run_id,),
        ).fetchall()
        return {row["metric_name"]: row["metric_value"] for row in rows}

    def _solution_to_dict(self, solution: Any) -> dict[str, Any]:
        if isinstance(solution, dict):
            return solution
        if hasattr(solution, "__dict__"):
            d = {}
            for k, v in solution.__dict__.items():
                if k.startswith("_"):
                    continue
                if hasattr(v, "__dict__") and not isinstance(v, (int, float, str, bool, list, dict)):
                    d[k] = v.__dict__
                elif isinstance(v, list):
                    d[k] = [
                        item.__dict__ if hasattr(item, "__dict__") and not isinstance(item, (int, float, str, bool))
                        else item
                        for item in v
                    ]
                else:
                    d[k] = v
            return d
        return {"raw": str(solution)}
