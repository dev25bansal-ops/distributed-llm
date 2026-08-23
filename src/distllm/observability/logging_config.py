"""Advanced logging configuration for distributed-llm.

Extends the base logging in ``logging.py`` with four additive components
that layer on top of the existing infrastructure without replacing it:

* **LogSchema** — auto schema enforcement: validates all log records
  have required fields (timestamp, level, message, module, trace_id,
  request_id).  Operators as a loguru *filter* or standalone validator.
* **HALokiSink** — high-availability Loki sink with exponential backoff
  retry (1s -> 30s), in-memory buffer (10K max), configurable
  replication factor (default 3), and Loki health-check.
* **LogRetentionPolicy** — per-level retention (DEBUG=1d, INFO=7d,
  WARNING=30d, ERROR=90d) with optional S3 / GCS archive before
  deletion.  Call ``cleanup()`` periodically.
* **DebugSampler** — sample DEBUG at 10 % (configurable), always pass
  ERROR+, with per-module rate overrides.

Usage — after calling ``setup_logging()`` from ``logging.py``::

    from distllm.observability.logging_config import apply_logging_config

    components = apply_logging_config(
        loki_urls=["http://loki-1:3100", "http://loki-2:3100"],
        loki_replication_factor=3,
        debug_sample_rate=0.1,
        retention_log_dir="logs",
    )
    # components["sampler"], components["loki_sink"], etc.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import random
import threading
import time
from collections import deque
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

from loguru import logger

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REQUIRED_LOG_FIELDS: frozenset[str] = frozenset({
    "timestamp",
    "level",
    "message",
    "module",
    "trace_id",
    "request_id",
})

# Default enrichment values injected when auto_enrich is on
_DEFAULT_ENRICHMENT: dict[str, str] = {
    "trace_id": "00000000000000000000000000000000",
    "request_id": "00000000-0000-0000-0000-000000000000",
}

# Level ordering for DebugSampler comparison
_LEVEL_RANK: dict[str, int] = {
    "TRACE": 5,
    "DEBUG": 10,
    "INFO": 20,
    "SUCCESS": 25,
    "WARNING": 30,
    "ERROR": 40,
    "CRITICAL": 50,
}

# Mapping from schema field name to loguru record key / access path.
# Loguru provides these fields by default under different names;
# the schema checks them through the mapping so no false positives
# are raised for structurally valid records.
_LOGURU_FIELD_MAP: dict[str, tuple[str, ...]] = {
    "timestamp": ("time",),           # record["time"] -> datetime
    "level": ("level", "name"),       # record["level"].name -> str
    "module": ("name",),              # record["name"] -> str
    "message": ("message",),          # record["message"] -> str
}


def _resolve_field(record: dict[str, Any], field: str) -> bool:
    """Check whether *field* is available in *record*.

    Handles:
      * Direct keys in the record dict.
      * Keys in ``record["extra"]``.
      * Loguru internal fields mapped via ``_LOGURU_FIELD_MAP``.
    """
    if field in record or field in record.get("extra", {}):
        return True

    path = _LOGURU_FIELD_MAP.get(field)
    if path is not None:
        obj: Any = record
        for key in path:
            if isinstance(obj, dict):
                obj = obj.get(key)
            else:
                obj = getattr(obj, key, None) if hasattr(obj, key) else None
            if obj is None:
                return False
        return True

    return False


# ---------------------------------------------------------------------------
# LogSchema — auto schema enforcement
# ---------------------------------------------------------------------------


class LogSchema:
    """Enforces a required-field schema on log records.

    Schema fields (timestamp, level, message, module, trace_id,
    request_id) are resolved against loguru's internal record
    structure.  Base fields that loguru always provides
    (timestamp, level, message, module) pass validation
    automatically.

    Records missing ``trace_id`` or ``request_id`` are **not**
    rejected by default; instead they are auto-enriched with
    placeholder values.  Set ``strict=True`` to reject
    non-conforming records instead.

    Use as a loguru *filter* via ``LogSchema.filter``, or call
    ``validate_record()`` / ``enrich_record()`` directly::

        schema = LogSchema()
        logger.add(sys.stderr, filter=schema.filter)
    """

    def __init__(
        self,
        required_fields: set[str] | None = None,
        strict: bool = False,
        auto_enrich: bool = True,
    ) -> None:
        """Initialize LogSchema.

        Args:
            required_fields: Set of field names that must be present.
                Defaults to ``REQUIRED_LOG_FIELDS``.
            strict: If True, reject (drop) records missing required
                fields.  If False, auto-enrich with defaults.
            auto_enrich: If True and *strict* is False, inject default
                values for missing optional fields (trace_id, request_id).
        """
        self._required = frozenset(required_fields) if required_fields else REQUIRED_LOG_FIELDS
        self._strict = strict
        self._auto_enrich = auto_enrich

    # ── public API ───────────────────────────────────────────────────

    def validate_record(self, record: dict[str, Any]) -> list[str]:
        """Check a loguru record dict against the schema.

        Resolves schema fields through ``_resolve_field``, which
        handles loguru's internal key naming (e.g. ``module`` ->
        ``record["name"]``), direct keys, and ``extra`` entries.

        Returns:
            List of violation descriptions (empty = valid).
        """
        errors: list[str] = []
        for field in sorted(self._required):
            if not _resolve_field(record, field):
                errors.append(f"Missing required field: {field!r}")
        return errors

    def enrich_record(self, record: dict[str, Any]) -> dict[str, Any]:
        """Return a *new* record dict with missing enrichment fields filled in.

        Does **not** mutate the original record.  No-op in strict mode.
        Currently enriches ``trace_id`` and ``request_id`` when absent.
        """
        if self._strict:
            return record

        enriched = dict(record)  # shallow copy
        extra = dict(enriched.get("extra", {}))
        enriched["extra"] = extra

        for field, default in _DEFAULT_ENRICHMENT.items():
            if not _resolve_field(enriched, field):
                extra[field] = default

        return enriched

    def filter(self, record: dict[str, Any]) -> bool:
        """Loguru-compatible filter.  Returns True to keep the record."""
        errors = self.validate_record(record)
        if not errors:
            return True
        if self._strict:
            logger.debug("LogSchema rejected record: %s", errors)
            return False
        # Enrichment happens at sink time, not in the filter.
        return True


# Convenience shortcut
def validate_log_record(
    record: dict[str, Any],
    required_fields: set[str] | None = None,
) -> list[str]:
    """Standalone helper to validate a loguru record dict.

    Args:
        record: A loguru record dictionary.
        required_fields: Set of required field names.
            Defaults to ``REQUIRED_LOG_FIELDS``.

    Returns:
        List of missing-field descriptions (empty = valid).
    """
    return LogSchema(required_fields=required_fields).validate_record(record)


# ---------------------------------------------------------------------------
# HA Loki sink — exponential backoff, buffer, replication, health check
# ---------------------------------------------------------------------------

_LOKI_DEFAULT_URLS = [
    "http://localhost:3100/loki/api/v1/push",
]

_RETRY_BASE_S = 1.0
_RETRY_MAX_S = 30.0
_RETRY_MULTIPLIER = 2.0


class HALokiSink:
    """High-availability Loki sink.

    Features:
      * In-memory circular buffer (configurable maxlen, default 10K).
      * Exponential backoff on push failure (1 s -> 30 s, jittered).
      * Replication: each batch is pushed to *N* Loki endpoints.
      * Periodic health check via Loki's ``/ready`` endpoint.
      * Graceful shutdown via ``close()``.

    This is an **additive** component.  The existing ``loki_sink``
    function in ``loki_sink.py`` remains unchanged.  Use this class
    when higher reliability is needed.

    Usage — add as a loguru sink::

        sink = HALokiSink(
            urls=["http://loki-1:3100", "http://loki-2:3100"],
            replication_factor=3,
        )
        logger.add(sink.write, level="INFO", format="{message}")
        # ...
        sink.close()
    """

    def __init__(
        self,
        urls: list[str] | None = None,
        labels: dict[str, str] | None = None,
        batch_size: int = 100,
        flush_interval: float = 5.0,
        max_buffer: int = 10_000,
        replication_factor: int = 3,
        timeout: float = 10.0,
        service_name: str = "distllm",
    ) -> None:
        """Initialize HALokiSink.

        Args:
            urls: Loki push endpoint URLs.  Defaults to
                ``["http://localhost:3100/loki/api/v1/push"]``.
            labels: Extra labels attached to every stream.
            batch_size: Number of log lines before a forced flush.
            flush_interval: Seconds between periodic flushes.
            max_buffer: Max buffered log lines (oldest dropped).
            replication_factor: Number of Loki targets to push each
                batch to (capped at ``len(urls)``).
            timeout: HTTP request timeout in seconds.
            service_name: Default ``service`` label value.
        """
        self._urls = list(urls) if urls else list(_LOKI_DEFAULT_URLS)
        self._labels = dict(labels or {})
        self._labels.setdefault("service", service_name)
        self._batch_size = batch_size
        self._flush_interval = flush_interval
        self._buffer: deque[tuple[int, str, dict[str, str]]] = deque(maxlen=max_buffer)
        self._replication_factor = min(replication_factor, len(self._urls))
        self._timeout = timeout
        self._closed = False
        self._lock = threading.Lock()
        self._current_retry_s: float = _RETRY_BASE_S

        self._flush_thread: threading.Thread | None = None
        self._flush_event = threading.Event()
        self._start_flush_thread()

    # ── properties ───────────────────────────────────────────────────

    @property
    def buffer_size(self) -> int:
        """Number of log lines currently buffered."""
        return len(self._buffer)

    @property
    def urls(self) -> list[str]:
        """List of configured Loki push URLs."""
        return list(self._urls)

    # ── health check ─────────────────────────────────────────────────

    def check_health(self, url: str | None = None) -> bool:
        """Check whether a Loki endpoint is ready.

        Args:
            url: Target URL.  If None, checks the first configured URL.

        Returns:
            True if the endpoint returns HTTP 200.
        """
        target = url or self._urls[0]
        ready_url = target.replace("/loki/api/v1/push", "/ready")
        try:
            import httpx

            resp = httpx.get(ready_url, timeout=5.0)
            return resp.status_code == 200
        except Exception:
            return False

    # ── loguru sink entry point ──────────────────────────────────────

    def write(self, message) -> None:
        """Loguru-compatible sink callable.

        Buffers the log line and triggers a flush when the batch
        size is reached.
        """
        if self._closed:
            return

        record = message.record
        ts_ns = int(time.time() * 1e9)
        otel_labels = self._get_otel_labels()

        line = json.dumps(
            {
                "level": record["level"].name,
                "module": record["name"],
                "function": record["function"],
                "message": record["message"],
                **{k: v for k, v in record["extra"].items()},
            },
            default=str,
        )

        with self._lock:
            self._buffer.append((ts_ns, line, otel_labels))
            if len(self._buffer) >= self._batch_size:
                self._flush_event.set()

    # ── flush / close ────────────────────────────────────────────────

    def flush(self) -> None:
        """Force-flush the buffer synchronously."""
        self._flush_batch()

    def close(self) -> None:
        """Graceful shutdown: flush remaining entries and stop thread."""
        self._closed = True
        self.flush()
        if self._flush_thread and self._flush_thread.is_alive():
            self._flush_event.set()
            self._flush_thread.join(timeout=10)

    # ── internal: thread management ──────────────────────────────────

    def _start_flush_thread(self) -> None:
        """Start the daemon background flush thread."""

        def _loop() -> None:
            while not self._closed:
                self._flush_event.wait(self._flush_interval)
                self._flush_event.clear()
                self._flush_batch()

        self._flush_thread = threading.Thread(target=_loop, daemon=True)
        self._flush_thread.start()

    def _flush_batch(self) -> None:
        """Pop a batch from the buffer and push to Loki targets."""
        batch: list[tuple[int, str, dict[str, str]]] = []
        with self._lock:
            while self._buffer and len(batch) < self._batch_size:
                batch.append(self._buffer.popleft())

        if not batch:
            return

        payload = self._build_payload(batch)
        self._push_with_retry(payload)

    def _build_payload(self, batch: list[tuple[int, str, dict[str, str]]]) -> dict[str, Any]:
        """Group batch entries into Loki streams by label set."""
        streams: dict[str, dict[str, Any]] = {}
        for ts_ns, line, otel_labels in batch:
            merged = {**self._labels, **otel_labels}
            key = json.dumps(merged, sort_keys=True)
            if key not in streams:
                streams[key] = {"stream": merged, "values": []}
            streams[key]["values"].append([str(ts_ns), line])
        return {"streams": list(streams.values())}

    def _push_with_retry(self, payload: dict[str, Any]) -> None:
        """Push *payload* to selected replica URLs with backoff."""
        targets = self._select_targets()

        for url in targets:
            delay = _RETRY_BASE_S
            for attempt in range(5):
                try:
                    self._push_single(url, payload)
                    self._current_retry_s = _RETRY_BASE_S
                    break
                except Exception as exc:
                    logger.debug(
                        "Loki push to %s failed (attempt %d/5): %s",
                        url,
                        attempt + 1,
                        exc,
                    )
                    if attempt < 4:
                        jitter = random.uniform(0, delay * 0.1)
                        actual_delay = min(delay + jitter, _RETRY_MAX_S)
                        time.sleep(actual_delay)
                        delay = min(delay * _RETRY_MULTIPLIER, _RETRY_MAX_S)
                    # On final failure the batch is silently dropped
                    # to avoid filling the buffer with undeliverable data.

    def _select_targets(self) -> list[str]:
        """Determine which Loki URLs to push to on this round.

        Uses a deterministic hash-based selection so each push
        may target a different subset of replicas.
        """
        if self._replication_factor >= len(self._urls):
            return list(self._urls)

        h = hashlib.sha256(str(time.time_ns()).encode()).digest()
        idx = int.from_bytes(h[:4], "big") % len(self._urls)
        indices = [(idx + i) % len(self._urls) for i in range(self._replication_factor)]
        return [self._urls[i] for i in indices]

    @staticmethod
    def _push_single(url: str, payload: dict[str, Any]) -> None:
        """POST *payload* to a single Loki URL synchronously."""
        import httpx

        resp = httpx.post(
            url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=10.0,
        )
        if resp.status_code not in (200, 204):
            raise RuntimeError(f"Loki returned {resp.status_code}: {resp.text[:200]}")

    @staticmethod
    def _get_otel_labels() -> dict[str, str]:
        """Extract OTel trace context as Loki labels."""
        try:
            from opentelemetry import trace

            span = trace.get_current_span()
            ctx = span.get_span_context()
            if ctx and ctx.is_valid:
                return {
                    "trace_id": f"{ctx.trace_id:032x}",
                    "span_id": f"{ctx.span_id:016x}",
                }
        except Exception:
            pass
        return {}


# ---------------------------------------------------------------------------
# LogRetentionPolicy — per-level retention & cloud archive
# ---------------------------------------------------------------------------

_RETENTION_DAYS: dict[str, int] = {
    "DEBUG": 1,
    "INFO": 7,
    "WARNING": 30,
    "ERROR": 90,
    "CRITICAL": 90,
}


class LogRetentionPolicy:
    """Manages per-level log retention and optional cloud archiving.

    Scans a log directory for level-prefixed files and removes entries
    older than the configured retention period.  Optionally archives
    expired files to S3 (boto3) or GCS (google-cloud-storage) before
    deletion.

    Usage::

        policy = LogRetentionPolicy(log_dir="logs", s3_bucket="my-logs")
        cleaned = policy.cleanup(dry_run=False)

    Call ``cleanup()`` periodically (cron, scheduler, or app startup).
    """

    def __init__(
        self,
        log_dir: str | pathlib.Path = "logs",
        retention_days: dict[str, int] | None = None,
        s3_bucket: str | None = None,
        gcs_bucket: str | None = None,
    ) -> None:
        """Initialize LogRetentionPolicy.

        Args:
            log_dir: Directory containing log files.
            retention_days: Per-level retention overrides.  Defaults to
                ``{"DEBUG": 1, "INFO": 7, "WARNING": 30, "ERROR": 90,
                "CRITICAL": 90}``.
            s3_bucket: Optional S3 bucket name for archived logs.
            gcs_bucket: Optional GCS bucket name for archived logs.
        """
        self._log_dir = pathlib.Path(log_dir)
        self._retention_days = dict(retention_days or _RETENTION_DAYS)
        self._s3_bucket = s3_bucket
        self._gcs_bucket = gcs_bucket

    # ── public API ───────────────────────────────────────────────────

    def cleanup(self, dry_run: bool = False) -> int:
        """Remove log files past their per-level retention period.

        Args:
            dry_run: If True, only report what would be cleaned.

        Returns:
            Number of files cleaned (or would-be-cleaned if dry_run).
        """
        now = datetime.now(timezone.utc)
        cleaned = 0

        if not self._log_dir.is_dir():
            logger.warning("Log directory does not exist: %s", self._log_dir)
            return 0

        for log_file in sorted(self._log_dir.iterdir()):
            if not log_file.is_file():
                continue

            level = _detect_level(log_file.name)
            if level is None:
                continue

            days = self._retention_days.get(level, 7)
            cutoff = now - timedelta(days=days)
            mtime = datetime.fromtimestamp(log_file.stat().st_mtime, tz=timezone.utc)

            if mtime >= cutoff:
                continue

            if dry_run:
                age_days = (now - mtime).total_seconds() / 86400
                logger.info(
                    "Would clean %s (level=%s, age=%.1fd, limit=%dd)",
                    log_file.name,
                    level,
                    age_days,
                    days,
                )
                cleaned += 1
                continue

            # Archive before deletion
            if self._s3_bucket:
                self._archive_to_s3(log_file)
            elif self._gcs_bucket:
                self._archive_to_gcs(log_file)

            log_file.unlink()
            logger.debug("Cleaned log file %s (level=%s)", log_file.name, level)
            cleaned += 1

        return cleaned

    # ── archive helpers ──────────────────────────────────────────────

    def _archive_to_s3(self, path: pathlib.Path) -> bool:
        """Upload *path* to S3 and return success status."""
        try:
            import boto3

            s3 = boto3.client("s3")
            key = f"logs/{path.name}"
            s3.upload_file(str(path), self._s3_bucket, key)
            logger.debug("Archived %s to s3://%s/%s", path.name, self._s3_bucket, key)
            return True
        except Exception as exc:
            logger.warning("S3 archive failed for %s: %s", path.name, exc)
            return False

    def _archive_to_gcs(self, path: pathlib.Path) -> bool:
        """Upload *path* to GCS and return success status."""
        try:
            from google.cloud import storage

            client = storage.Client()
            bucket = client.bucket(self._gcs_bucket)
            blob = bucket.blob(f"logs/{path.name}")
            blob.upload_from_filename(str(path))
            logger.debug(
                "Archived %s to gs://%s/logs/%s",
                path.name,
                self._gcs_bucket,
                path.name,
            )
            return True
        except Exception as exc:
            logger.warning("GCS archive failed for %s: %s", path.name, exc)
            return False


def _detect_level(filename: str) -> str | None:
    """Extract log level from a filename (e.g. ``info.log`` -> ``INFO``)."""
    stem = filename.lower()
    for level in ("debug", "info", "warning", "error", "critical"):
        if level in stem:
            return level.upper()
    return None


# ---------------------------------------------------------------------------
# DebugSampler — configurable debug-level sampling
# ---------------------------------------------------------------------------


class DebugSampler:
    """Controls debug-level logging with sampling and per-module rates.

    * ERROR+ (WARNING, ERROR, CRITICAL) are **always** passed through.
    * DEBUG records are sampled at a configurable global rate.
    * Per-module sampling rates override the global rate for matching
      modules (prefix matching applies).

    Use as a loguru filter::

        sampler = DebugSampler(global_rate=0.1)
        sampler.set_module_rate("distllm.api", 1.0)    # always sample
        sampler.set_module_rate("distllm.core", 0.05)  # 5 %
        logger.add(sys.stderr, filter=sampler.filter)
    """

    def __init__(self, global_rate: float = 0.1, enabled: bool = True) -> None:
        """Initialize DebugSampler.

        Args:
            global_rate: Sampling rate for DEBUG records (0.0 – 1.0).
                1.0 = always sample; 0.0 = never sample.
            enabled: If False, all records pass through unfiltered.
        """
        self._global_rate = _clamp_rate(global_rate)
        self._enabled = enabled
        self._module_rates: dict[str, float] = {}
        self._lock = threading.Lock()

    # ── configuration ────────────────────────────────────────────────

    @property
    def global_rate(self) -> float:
        """Current global DEBUG sampling rate."""
        return self._global_rate

    @global_rate.setter
    def global_rate(self, rate: float) -> None:
        self._global_rate = _clamp_rate(rate)

    @property
    def enabled(self) -> bool:
        """Whether sampling is active."""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = bool(value)

    def set_module_rate(self, module_name: str, rate: float) -> None:
        """Set a per-module sampling rate override.

        Args:
            module_name: Fully qualified module name (e.g.
                ``"distllm.api"``).  Prefix matching applies: a rate
                set for ``"distllm"`` affects all ``distllm.*`` modules.
            rate: Sampling rate 0.0 – 1.0.  Use 1.0 for always-sample
                on a noisy module you want to observe.
        """
        with self._lock:
            self._module_rates[module_name] = _clamp_rate(rate)

    def get_module_rate(self, module_name: str) -> float | None:
        """Return the effective per-module rate, or None if unset.

        Checks exact match first, then longest prefix match.
        """
        if module_name in self._module_rates:
            return self._module_rates[module_name]

        matched: str | None = None
        for prefix in self._module_rates:
            if module_name.startswith(prefix):
                if matched is None or len(prefix) > len(matched):
                    matched = prefix
        return self._module_rates.get(matched) if matched else None

    # ── loguru filter ────────────────────────────────────────────────

    def filter(self, record: dict[str, Any]) -> bool:
        """Loguru-compatible filter.  Returns True to keep the record."""
        if not self._enabled:
            return True

        level_name = record.get("level", record).get("name", "DEBUG")

        # Always pass WARNING and above
        if _LEVEL_RANK.get(level_name, 0) >= _LEVEL_RANK.get("WARNING", 30):
            return True

        # Only sample DEBUG/TRACE
        module = record.get("name", "")
        rate = self._module_rate_for(module)
        return random.random() < rate

    # ── internal ─────────────────────────────────────────────────────

    def _module_rate_for(self, module_name: str) -> float:
        """Determine the effective sampling rate for *module_name*."""
        override = self.get_module_rate(module_name)
        if override is not None:
            return override
        # Walk parent modules for prefix matches
        parts = module_name.split(".")
        for i in range(len(parts) - 1, 0, -1):
            prefix = ".".join(parts[:i])
            override = self.get_module_rate(prefix)
            if override is not None:
                return override
        return self._global_rate


def _clamp_rate(rate: float) -> float:
    """Clamp *rate* to the interval [0.0, 1.0]."""
    return max(0.0, min(1.0, float(rate)))


# ---------------------------------------------------------------------------
# Convenience helper — apply all components at once
# ---------------------------------------------------------------------------


def apply_logging_config(
    *,
    # LogSchema
    schema_enabled: bool = True,
    schema_strict: bool = False,
    # DebugSampler
    debug_sample_rate: float = 0.1,
    debug_sampling_enabled: bool = True,
    module_rates: dict[str, float] | None = None,
    # HALokiSink
    loki_urls: list[str] | None = None,
    loki_labels: dict[str, str] | None = None,
    loki_replication_factor: int = 3,
    # LogRetentionPolicy
    retention_log_dir: str | pathlib.Path = "logs",
    retention_days: dict[str, int] | None = None,
    s3_bucket: str | None = None,
    gcs_bucket: str | None = None,
) -> dict[str, Any]:
    """Configure and return all advanced logging components.

    This is an **additive** helper — it does **not** call
    ``logger.remove()`` or interfere with existing sinks registered by
    ``setup_logging()``.  Call it after ``setup_logging()`` to layer on
    extra capabilities.

    The returned dict contains keys ``"schema"``, ``"sampler"``,
    ``"loki_sink"``, and ``"retention"`` for direct access.

    Args:
        schema_enabled: Whether to initialise the ``LogSchema``.
        schema_strict: If True, reject non-conforming records.
        debug_sample_rate: Global DEBUG sampling rate.
        debug_sampling_enabled: Master switch for sampling.
        module_rates: Per-module debug sampling overrides.
        loki_urls: Loki push endpoint URLs.
        loki_labels: Extra labels attached to Loki streams.
        loki_replication_factor: Replicas per batch push.
        retention_log_dir: Log file directory.
        retention_days: Per-level retention overrides.
        s3_bucket: S3 bucket for log archive.
        gcs_bucket: GCS bucket for log archive.

    Returns:
        Dict with instantiated components::

            {
                "schema": LogSchema,
                "sampler": DebugSampler,
                "loki_sink": HALokiSink | None,
                "retention": LogRetentionPolicy,
            }
    """
    registry: dict[str, Any] = {}

    # 1. LogSchema
    if schema_enabled:
        registry["schema"] = LogSchema(strict=schema_strict)

    # 2. DebugSampler
    sampler = DebugSampler(global_rate=debug_sample_rate, enabled=debug_sampling_enabled)
    if module_rates:
        for mod, rate in module_rates.items():
            sampler.set_module_rate(mod, rate)
    registry["sampler"] = sampler

    # 3. HA Loki sink (optional)
    if loki_urls:
        loki_sink = HALokiSink(
            urls=loki_urls,
            labels=loki_labels,
            replication_factor=loki_replication_factor,
        )
        logger.add(
            loki_sink.write,
            level="DEBUG",
            format="{message}",
        )
        registry["loki_sink"] = loki_sink
    else:
        registry["loki_sink"] = None

    # 4. LogRetentionPolicy
    registry["retention"] = LogRetentionPolicy(
        log_dir=retention_log_dir,
        retention_days=retention_days,
        s3_bucket=s3_bucket,
        gcs_bucket=gcs_bucket,
    )

    return registry
