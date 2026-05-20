"""Plugin telemetry for the DistLLM plugin marketplace.

Tracks per-plugin usage, hook execution counts, average duration,
error rates, and provides optional SQLite persistence.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger


@dataclass
class PluginStats:
    """Aggregated telemetry for a single plugin."""
    plugin_name: str
    total_executions: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    total_duration_ms: float = 0.0
    peak_duration_ms: float = 0.0
    min_duration_ms: float = float("inf")
    avg_duration_ms: float = 0.0
    error_rate: float = 0.0
    hook_counts: dict[str, int] = field(default_factory=dict)
    last_execution: float = 0.0

    @property
    def success_rate(self) -> float:
        if self.total_executions == 0:
            return 1.0
        return self.successful_executions / self.total_executions


@dataclass
class TelemetryRecord:
    """A single telemetry record entry."""
    plugin_name: str
    hook_name: str
    duration_ms: float
    success: bool
    timestamp: float
    error: str = ""


class PluginTelemetry:
    """Collects and queries per-plugin telemetry data.

    Thread-safe with optional SQLite persistence for long-term storage.
    """

    def __init__(self, persist_path: str | Path | None = None) -> None:
        self._lock = threading.Lock()
        self._records: list[TelemetryRecord] = []
        self._plugin_stats: dict[str, PluginStats] = {}
        self._persist_path = Path(persist_path) if persist_path else None
        self._max_records = 100_000  # Cap in-memory records

    def record_usage(
        self,
        plugin_name: str,
        hook_name: str,
        duration_ms: float,
        success: bool,
        error: str = "",
    ) -> None:
        """Record a hook execution event.

        Args:
            plugin_name: Name of the plugin.
            hook_name: Name of the hook point (e.g., "on_request").
            duration_ms: Execution duration in milliseconds.
            success: Whether the hook completed successfully.
            error: Error message if failed.
        """
        now = time.time()
        record = TelemetryRecord(
            plugin_name=plugin_name,
            hook_name=hook_name,
            duration_ms=duration_ms,
            success=success,
            timestamp=now,
            error=error,
        )

        with self._lock:
            # Trim old records if over limit
            if len(self._records) >= self._max_records:
                self._records = self._records[-(self._max_records // 2):]
            self._records.append(record)

            # Update aggregated stats
            stats = self._plugin_stats.setdefault(
                plugin_name,
                PluginStats(plugin_name=plugin_name),
            )
            stats.total_executions += 1
            if success:
                stats.successful_executions += 1
            else:
                stats.failed_executions += 1
            stats.total_duration_ms += duration_ms
            stats.peak_duration_ms = max(stats.peak_duration_ms, duration_ms)
            stats.min_duration_ms = min(stats.min_duration_ms, duration_ms)
            stats.avg_duration_ms = stats.total_duration_ms / stats.total_executions
            stats.error_rate = stats.failed_executions / stats.total_executions
            stats.hook_counts[hook_name] = stats.hook_counts.get(hook_name, 0) + 1
            stats.last_execution = now

            # Persist to SQLite if configured
            if self._persist_path:
                self._persist_record(record)

    def get_plugin_stats(self, plugin_name: str) -> PluginStats:
        """Return aggregated stats for a plugin."""
        with self._lock:
            return self._plugin_stats.get(
                plugin_name,
                PluginStats(plugin_name=plugin_name),
            )

    def get_all_stats(self) -> dict[str, PluginStats]:
        """Return stats for all tracked plugins."""
        with self._lock:
            return dict(self._plugin_stats)

    def get_error_rates(self) -> dict[str, float]:
        """Return error rate per plugin name."""
        with self._lock:
            return {
                name: stats.error_rate
                for name, stats in self._plugin_stats.items()
            }

    def get_recent_errors(self, limit: int = 50) -> list[TelemetryRecord]:
        """Return recent failed hook executions."""
        with self._lock:
            return [r for r in self._records if not r.success][-limit:]

    def get_records(
        self,
        plugin_name: str | None = None,
        hook_name: str | None = None,
        since: float | None = None,
        limit: int = 1000,
    ) -> list[TelemetryRecord]:
        """Query telemetry records with filters."""
        with self._lock:
            records = self._records
            if plugin_name:
                records = [r for r in records if r.plugin_name == plugin_name]
            if hook_name:
                records = [r for r in records if r.hook_name == hook_name]
            if since:
                records = [r for r in records if r.timestamp >= since]
            return records[-limit:]

    def reset(self, plugin_name: str | None = None) -> None:
        """Reset telemetry data, optionally for a single plugin."""
        with self._lock:
            if plugin_name:
                self._plugin_stats.pop(plugin_name, None)
                self._records = [r for r in self._records if r.plugin_name != plugin_name]
            else:
                self._plugin_stats.clear()
                self._records.clear()

    def export_json(self) -> str:
        """Export all stats as JSON."""
        with self._lock:
            data = {
                name: {
                    "total_executions": s.total_executions,
                    "successful_executions": s.successful_executions,
                    "failed_executions": s.failed_executions,
                    "avg_duration_ms": round(s.avg_duration_ms, 2),
                    "peak_duration_ms": round(s.peak_duration_ms, 2),
                    "error_rate": round(s.error_rate, 4),
                    "hook_counts": s.hook_counts,
                }
                for name, s in self._plugin_stats.items()
            }
            return json.dumps(data, indent=2)

    def _persist_record(self, record: TelemetryRecord) -> None:
        """Persist a record to SQLite."""
        try:
            import sqlite3
            conn = sqlite3.connect(str(self._persist_path))
            conn.execute("""
                CREATE TABLE IF NOT EXISTS plugin_telemetry (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    plugin_name TEXT NOT NULL,
                    hook_name TEXT NOT NULL,
                    duration_ms REAL NOT NULL,
                    success INTEGER NOT NULL,
                    timestamp REAL NOT NULL,
                    error TEXT
                )
            """)
            conn.execute(
                "INSERT INTO plugin_telemetry (plugin_name, hook_name, duration_ms, success, timestamp, error) VALUES (?, ?, ?, ?, ?, ?)",
                (record.plugin_name, record.hook_name, record.duration_ms, int(record.success), record.timestamp, record.error),
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.debug(f"Failed to persist telemetry record: {e}")
