"""Usage Metering — track API usage for billing, quotas.

Records token usage per request, tracks per-tenant/team quotas, generates
billing records, and integrates with rate limiters for quota enforcement.

Supports SQLite and JSONL storage backends.
"""

from __future__ import annotations

import csv
import io
import json
import os
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from loguru import logger


# ── Configuration ───────────────────────────────────────────────────────────

DEFAULT_DB_PATH = os.environ.get("DISTLLM_USAGE_DB", "")


# ── Constants / Enums ───────────────────────────────────────────────────────


class UsageRecordStatus(Enum):
    PENDING = "pending"
    BILLED = "billed"
    VOID = "void"
    ERROR = "error"


# ── Data classes ────────────────────────────────────────────────────────────


@dataclass
class UsageRecord:
    """A single usage record for a model inference request."""
    record_id: str
    tenant_id: str
    model_name: str
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    request_duration_ms: float = 0.0
    timestamp: float = 0.0
    cost: float = 0.0
    status: UsageRecordStatus = UsageRecordStatus.PENDING
    labels: dict[str, str] = field(default_factory=dict)
    endpoint: str = ""
    key_id: str = ""
    gpu_time_seconds: float = 0.0
    gpu_type: str = ""
    cost_usd: float = 0.0
    tokens_per_second: float = 0.0
    ttft_ms: float = 0.0


@dataclass
class QuotaLimit:
    """A usage quota for a tenant or team."""
    tenant_id: str
    max_tokens_per_day: int = 0
    max_requests_per_minute: int = 0
    max_tokens_per_request: int = 0
    max_concurrent_requests: int = 0
    cost_budget_per_month: float = 0.0
    overage_allowed: bool = False
    overage_multiplier: float = 2.0


@dataclass
class TenantUsage:
    """Aggregated usage for a tenant."""
    tenant_id: str
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_requests: int = 0
    total_cost: float = 0.0
    current_billing_period_start: float = 0.0
    current_billing_period_end: float = 0.0
    daily_tokens: dict[str, int] = field(default_factory=dict)


# ── SQLite schema ───────────────────────────────────────────────────────────

SQLITE_SCHEMA = """
CREATE TABLE IF NOT EXISTS usage_records (
    record_id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    key_id TEXT DEFAULT '',
    model_name TEXT DEFAULT '',
    input_tokens INTEGER DEFAULT 0,
    output_tokens INTEGER DEFAULT 0,
    total_tokens INTEGER DEFAULT 0,
    request_duration_ms REAL DEFAULT 0.0,
    timestamp REAL DEFAULT 0.0,
    cost REAL DEFAULT 0.0,
    endpoint TEXT DEFAULT '',
    status TEXT DEFAULT 'pending',
    labels TEXT DEFAULT '{}',
    gpu_time_seconds REAL DEFAULT 0.0,
    gpu_type TEXT DEFAULT '',
    cost_usd REAL DEFAULT 0.0,
    tokens_per_second REAL DEFAULT 0.0,
    ttft_ms REAL DEFAULT 0.0
);

CREATE TABLE IF NOT EXISTS quotas (
    tenant_id TEXT PRIMARY KEY,
    max_tokens_per_day INTEGER DEFAULT 0,
    max_requests_per_minute INTEGER DEFAULT 0,
    max_tokens_per_request INTEGER DEFAULT 0,
    max_concurrent_requests INTEGER DEFAULT 0,
    cost_budget_per_month REAL DEFAULT 0.0,
    overage_allowed INTEGER DEFAULT 0,
    overage_multiplier REAL DEFAULT 2.0
);

CREATE TABLE IF NOT EXISTS tenant_usage (
    tenant_id TEXT PRIMARY KEY,
    total_input_tokens INTEGER DEFAULT 0,
    total_output_tokens INTEGER DEFAULT 0,
    total_requests INTEGER DEFAULT 0,
    total_cost REAL DEFAULT 0.0,
    billing_period_start REAL DEFAULT 0.0,
    billing_period_end REAL DEFAULT 0.0,
    daily_tokens TEXT DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_records_tenant ON usage_records(tenant_id);
CREATE INDEX IF NOT EXISTS idx_records_timestamp ON usage_records(timestamp);
CREATE INDEX IF NOT EXISTS idx_records_key ON usage_records(key_id);
"""


# ── Usage Meter ─────────────────────────────────────────────────────────────


class UsageMeter:
    """Track model inference usage for billing and quota enforcement.

    Usage:
        meter = UsageMeter(db_path="/path/to/usage.db")
        meter.set_quota("tenant-1", QuotaLimit(max_tokens_per_day=1_000_000))
        record = meter.record_request(
            tenant_id="tenant-1",
            model_name="llama-70b",
            input_tokens=512,
            output_tokens=128,
        )
        allowed, reason = meter.check_quota("tenant-1")
    """

    PRICE_PER_1K_INPUT_TOKENS: float = 0.01
    PRICE_PER_1K_OUTPUT_TOKENS: float = 0.03

    def __init__(
        self,
        storage_path: str = "",
        input_price: float = 0.01,
        output_price: float = 0.03,
        use_sqlite: bool = True,
    ) -> None:
        self._input_price = input_price
        self._output_price = output_price
        self._lock = threading.RLock()
        self._in_memory_records: list[UsageRecord] = []
        self._quotas: dict[str, QuotaLimit] = {}
        self._tenants: dict[str, TenantUsage] = {}
        self._rate_counters: dict[str, list[float]] = {}
        self._concurrent: dict[str, int] = {}
        self._record_counter = 0
        self._use_sqlite = use_sqlite
        self._db_path: str = ""

        if use_sqlite:
            db_path = storage_path or DEFAULT_DB_PATH or ".usage.db"
            self._db_path = db_path
            # C11: Enable WAL mode for concurrent read/write safety
            self._conn = sqlite3.connect(db_path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA busy_timeout=5000")
            self._conn.executescript(SQLITE_SCHEMA)
            self._conn.commit()
            self._load_from_sqlite()
            logger.info(f"UsageMeter initialized with SQLite (WAL): {db_path}")
        else:
            self._storage_path = Path(storage_path) if storage_path else Path(".usage_records.jsonl")
            self._load_records()
            logger.info(f"UsageMeter initialized with JSONL: {self._storage_path}")

    # ── Recording ───────────────────────────────────────────────────────

    def record_request(
        self,
        tenant_id: str,
        model_name: str,
        input_tokens: int,
        output_tokens: int,
        duration_ms: float = 0.0,
        endpoint: str = "",
        labels: dict[str, str] | None = None,
        key_id: str = "",
        gpu_time_seconds: float = 0.0,
        gpu_type: str = "",
        cost_usd: float = 0.0,
        tokens_per_second: float = 0.0,
        ttft_ms: float = 0.0,
    ) -> UsageRecord:
        """Record a completed inference request.

        Args:
            tenant_id: Tenant identifier.
            model_name: Model used.
            input_tokens: Input token count.
            output_tokens: Output token count.
            duration_ms: Total request duration in ms.
            endpoint: API endpoint path.
            labels: Additional metadata labels.
            key_id: API key identifier.
            gpu_time_seconds: GPU seconds consumed by this request.
            gpu_type: GPU hardware type (e.g., "A100-80GB").
            cost_usd: Actual USD cost from cost tracker.
            tokens_per_second: Generation throughput.
            ttft_ms: Time to first token in ms.
        """
        with self._lock:
            self._record_counter += 1
            record_id = f"usage-{int(time.time())}-{self._record_counter}"

            total_tokens = input_tokens + output_tokens

            # Use actual cost if provided, otherwise estimate from token prices
            if cost_usd > 0:
                cost = cost_usd
            else:
                cost = (
                    (input_tokens / 1000) * self._input_price
                    + (output_tokens / 1000) * self._output_price
                )

            record = UsageRecord(
                record_id=record_id,
                tenant_id=tenant_id,
                model_name=model_name,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                request_duration_ms=duration_ms,
                timestamp=time.time(),
                cost=round(cost, 6),
                endpoint=endpoint,
                labels=labels or {},
                key_id=key_id,
                gpu_time_seconds=gpu_time_seconds,
                gpu_type=gpu_type,
                cost_usd=round(cost_usd, 8) if cost_usd > 0 else round(cost, 8),
                tokens_per_second=tokens_per_second,
                ttft_ms=ttft_ms,
            )
            self._in_memory_records.append(record)

            tenant = self._tenants.setdefault(tenant_id, TenantUsage(
                tenant_id=tenant_id,
                current_billing_period_start=self._billing_period_start(),
                current_billing_period_end=self._billing_period_end(),
            ))
            tenant.total_input_tokens += input_tokens
            tenant.total_output_tokens += output_tokens
            tenant.total_requests += 1
            tenant.total_cost = round(tenant.total_cost + cost, 6)

            day_key = datetime.now().strftime("%Y-%m-%d")
            tenant.daily_tokens[day_key] = (
                tenant.daily_tokens.get(day_key, 0) + total_tokens
            )

            if self._use_sqlite:
                self._sqlite_insert_record(record)
                self._sqlite_upsert_tenant(tenant)
            else:
                self._append_record(record)

            return record

    # ── Quota management ────────────────────────────────────────────────

    def set_quota(self, tenant_id: str, quota: QuotaLimit) -> None:
        with self._lock:
            self._quotas[tenant_id] = quota
            if self._use_sqlite:
                self._sqlite_upsert_quota(quota)

    def get_quota(self, tenant_id: str) -> QuotaLimit | None:
        return self._quotas.get(tenant_id)

    def remove_quota(self, tenant_id: str) -> bool:
        with self._lock:
            removed = self._quotas.pop(tenant_id, None) is not None
            if removed and self._use_sqlite:
                self._conn.execute("DELETE FROM quotas WHERE tenant_id = ?", (tenant_id,))
                self._conn.commit()
            return removed

    def check_quota(
        self, tenant_id: str, requested_tokens: int | None = None
    ) -> tuple[bool, str]:
        """Check if *tenant_id* has remaining quota for a request.

        When *requested_tokens* is provided and the tenant's quota sets a
        ``max_tokens_per_request`` limit, an oversize request is blocked
        before any other quota dimension is consulted.

        Returns ``(allowed, reason)``.
        """
        quota = self._quotas.get(tenant_id)
        if quota is None:
            return True, "no quota set"

        if (
            requested_tokens is not None
            and quota.max_tokens_per_request > 0
            and requested_tokens > quota.max_tokens_per_request
        ):
            return False, f"per-request token limit {quota.max_tokens_per_request} exceeded"

        tenant = self._tenants.get(tenant_id)
        if tenant is None:
            return True, "no usage yet"

        day_key = datetime.now().strftime("%Y-%m-%d")
        today_tokens = tenant.daily_tokens.get(day_key, 0)
        if quota.max_tokens_per_day > 0 and today_tokens >= quota.max_tokens_per_day:
            if quota.overage_allowed:
                return True, "overage allowed"
            return False, f"daily token limit {quota.max_tokens_per_day} exceeded"

        if quota.cost_budget_per_month > 0:
            if tenant.total_cost >= quota.cost_budget_per_month:
                if quota.overage_allowed:
                    return True, "overage allowed"
                return False, f"monthly budget {quota.cost_budget_per_month} exceeded"

        if quota.max_requests_per_minute > 0:
            now = time.time()
            with self._lock:
                counter = self._rate_counters.setdefault(tenant_id, [])
                counter[:] = [t for t in counter if now - t < 60.0]
                if len(counter) >= quota.max_requests_per_minute:
                    return False, f"rate limit {quota.max_requests_per_minute}/min exceeded"
                counter.append(now)

                # C12: Periodically clean up stale rate counters
                if len(self._rate_counters) > 100:
                    stale = [k for k, v in self._rate_counters.items() if not v or now - v[-1] > 120]
                    for k in stale:
                        del self._rate_counters[k]

        if quota.max_concurrent_requests > 0:
            current = self._concurrent.get(tenant_id, 0)
            if current >= quota.max_concurrent_requests:
                return False, f"concurrent limit {quota.max_concurrent_requests} exceeded"

        return True, "ok"

    def increment_concurrent(self, tenant_id: str) -> None:
        with self._lock:
            self._concurrent[tenant_id] = self._concurrent.get(tenant_id, 0) + 1

    def decrement_concurrent(self, tenant_id: str) -> None:
        with self._lock:
            current = self._concurrent.get(tenant_id, 0)
            self._concurrent[tenant_id] = max(0, current - 1)

    def get_concurrent(self, tenant_id: str) -> int:
        return self._concurrent.get(tenant_id, 0)

    def enforce_quota(
        self, tenant_id: str, raise_on_block: bool = True,
        requested_tokens: int | None = None,
    ) -> tuple[bool, str]:
        """Check quota and increment concurrent counter if allowed.

        Returns ``(allowed, reason)``.
        """
        allowed, reason = self.check_quota(tenant_id, requested_tokens=requested_tokens)
        if allowed:
            self.increment_concurrent(tenant_id)
        return allowed, reason

    def release_quota(self, tenant_id: str) -> None:
        """Decrement the concurrent counter after a request completes."""
        self.decrement_concurrent(tenant_id)

    # ── Queries ─────────────────────────────────────────────────────────

    def tenant_usage(self, tenant_id: str) -> TenantUsage | None:
        return self._tenants.get(tenant_id)

    def all_tenants(self) -> list[TenantUsage]:
        return list(self._tenants.values())

    def total_usage(self) -> dict[str, int | float]:
        total_input = sum(r.input_tokens for r in self._in_memory_records)
        total_output = sum(r.output_tokens for r in self._in_memory_records)
        return {
            "total_requests": len(self._in_memory_records),
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_tokens": total_input + total_output,
            "total_cost": round(sum(r.cost for r in self._in_memory_records), 4),
            "active_tenants": len(self._tenants),
        }

    def records(
        self, tenant_id: str | None = None,
        key_id: str | None = None,
        model_name: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[UsageRecord]:
        if self._use_sqlite:
            return self._sqlite_query_records(tenant_id, key_id, model_name, limit, offset)

        result = list(self._in_memory_records)
        if tenant_id:
            result = [r for r in result if r.tenant_id == tenant_id]
        if key_id:
            result = [r for r in result if r.key_id == key_id]
        if model_name:
            result = [r for r in result if r.model_name == model_name]
        return result[-limit:]

    # ── Billing ─────────────────────────────────────────────────────────

    def generate_invoice(
        self, tenant_id: str,
        period_start: float | None = None,
        period_end: float | None = None,
    ) -> dict[str, Any]:
        """Generate a billing invoice for a tenant's usage period."""
        start = period_start or self._billing_period_start()
        end = period_end or self._billing_period_end()

        records = [
            r for r in self._in_memory_records
            if r.tenant_id == tenant_id
            and start <= r.timestamp <= end
        ]

        total_input = sum(r.input_tokens for r in records)
        total_output = sum(r.output_tokens for r in records)
        total_cost = sum(r.cost for r in records)
        total_requests = len(records)

        quota = self._quotas.get(tenant_id)
        overage_cost = 0.0
        if quota and quota.overage_allowed:
            base_budget = quota.cost_budget_per_month
            if base_budget > 0 and total_cost > base_budget:
                overage = total_cost - base_budget
                overage_cost = overage * (quota.overage_multiplier - 1)

        return {
            "tenant_id": tenant_id,
            "period_start": start,
            "period_end": end,
            "total_requests": total_requests,
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_cost": round(total_cost, 4),
            "overage_cost": round(overage_cost, 4),
            "grand_total": round(total_cost + overage_cost, 4),
            "records_generated": len(records),
        }

    def generate_report(
        self, tenant_id: str | None = None,
        period_start: float | None = None,
        period_end: float | None = None,
    ) -> dict[str, Any]:
        """Generate a detailed usage report with per-model breakdown."""
        start = period_start or self._billing_period_start()
        end = period_end or self._billing_period_end()

        records = [
            r for r in self._in_memory_records
            if (tenant_id is None or r.tenant_id == tenant_id)
            and start <= r.timestamp <= end
        ]

        models: dict[str, dict[str, int | float]] = {}
        for r in records:
            m = models.setdefault(r.model_name, {
                "requests": 0, "input_tokens": 0, "output_tokens": 0,
                "total_tokens": 0, "cost": 0.0,
            })
            m["requests"] += 1
            m["input_tokens"] += r.input_tokens
            m["output_tokens"] += r.output_tokens
            m["total_tokens"] += r.total_tokens
            m["cost"] = round(float(m["cost"]) + r.cost, 4)

        return {
            "tenant_id": tenant_id or "*all*",
            "period_start": start,
            "period_end": end,
            "total_requests": len(records),
            "total_input_tokens": sum(r.input_tokens for r in records),
            "total_output_tokens": sum(r.output_tokens for r in records),
            "total_tokens": sum(r.total_tokens for r in records),
            "total_cost": round(sum(r.cost for r in records), 4),
            "models": models,
        }

    def export_csv(
        self,
        filepath: str | None = None,
        tenant_id: str | None = None,
        period_start: float | None = None,
        period_end: float | None = None,
    ) -> str:
        """Export usage records as CSV.

        If *filepath* is provided, writes to file and returns the path.
        Otherwise returns the CSV string.
        """
        start = period_start or self._billing_period_start()
        end = period_end or self._billing_period_end()

        records = [
            r for r in self._in_memory_records
            if (tenant_id is None or r.tenant_id == tenant_id)
            and start <= r.timestamp <= end
        ]

        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow([
            "record_id", "tenant_id", "key_id", "model_name",
            "input_tokens", "output_tokens", "total_tokens",
            "cost", "duration_ms", "endpoint", "timestamp",
        ])
        for r in records:
            writer.writerow([
                r.record_id, r.tenant_id, r.key_id, r.model_name,
                r.input_tokens, r.output_tokens, r.total_tokens,
                r.cost, r.request_duration_ms, r.endpoint, r.timestamp,
            ])

        csv_str = output.getvalue()
        output.close()

        if filepath:
            Path(filepath).write_text(csv_str)
            return filepath
        return csv_str

    # ── Bulk quota import ───────────────────────────────────────────────

    def import_quotas(self, quotas: list[dict[str, Any]]) -> int:
        """Import quotas from a list of dicts. Returns count imported."""
        count = 0
        for entry in quotas:
            q = QuotaLimit(
                tenant_id=entry["tenant_id"],
                max_tokens_per_day=entry.get("max_tokens_per_day", 0),
                max_requests_per_minute=entry.get("max_requests_per_minute", 0),
                max_tokens_per_request=entry.get("max_tokens_per_request", 0),
                max_concurrent_requests=entry.get("max_concurrent_requests", 0),
                cost_budget_per_month=entry.get("cost_budget_per_month", 0.0),
                overage_allowed=entry.get("overage_allowed", False),
                overage_multiplier=entry.get("overage_multiplier", 2.0),
            )
            self.set_quota(entry["tenant_id"], q)
            count += 1
        return count

    # ── Private helpers ─────────────────────────────────────────────────

    def _billing_period_start(self) -> float:
        now = datetime.now()
        return datetime(now.year, now.month, 1).timestamp()

    def _billing_period_end(self) -> float:
        now = datetime.now()
        if now.month == 12:
            return datetime(now.year + 1, 1, 1).timestamp()
        return datetime(now.year, now.month + 1, 1).timestamp()

    # ── JSONL persistence ───────────────────────────────────────────────

    def _append_record(self, record: UsageRecord) -> None:
        try:
            with open(self._storage_path, "a") as f:
                f.write(json.dumps({
                    "record_id": record.record_id,
                    "tenant_id": record.tenant_id,
                    "key_id": record.key_id,
                    "model_name": record.model_name,
                    "input_tokens": record.input_tokens,
                    "output_tokens": record.output_tokens,
                    "total_tokens": record.total_tokens,
                    "request_duration_ms": record.request_duration_ms,
                    "timestamp": record.timestamp,
                    "cost": record.cost,
                    "endpoint": record.endpoint,
                    "labels": record.labels,
                }, default=str) + "\n")
        except Exception as e:
            logger.warning(f"Failed to persist usage record: {e}")

    def _load_records(self) -> None:
        if not self._storage_path.exists():
            return
        try:
            with open(self._storage_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    self._in_memory_records.append(UsageRecord(
                        record_id=data["record_id"],
                        tenant_id=data["tenant_id"],
                        key_id=data.get("key_id", ""),
                        model_name=data.get("model_name", ""),
                        input_tokens=data.get("input_tokens", 0),
                        output_tokens=data.get("output_tokens", 0),
                        total_tokens=data.get("total_tokens", 0),
                        request_duration_ms=data.get("request_duration_ms", 0.0),
                        timestamp=data.get("timestamp", 0.0),
                        cost=data.get("cost", 0.0),
                        endpoint=data.get("endpoint", ""),
                        labels=data.get("labels", {}),
                    ))
        except Exception as e:
            logger.warning(f"Failed to load usage records: {e}")

    # ── SQLite persistence ──────────────────────────────────────────────

    def _sqlite_insert_record(self, record: UsageRecord) -> None:
        try:
            self._conn.execute(
                """INSERT INTO usage_records
                   (record_id, tenant_id, key_id, model_name,
                    input_tokens, output_tokens, total_tokens,
                    request_duration_ms, timestamp, cost, endpoint, status, labels,
                    gpu_time_seconds, gpu_type, cost_usd, tokens_per_second, ttft_ms)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    record.record_id, record.tenant_id, record.key_id,
                    record.model_name, record.input_tokens, record.output_tokens,
                    record.total_tokens, record.request_duration_ms,
                    record.timestamp, record.cost, record.endpoint,
                    record.status.value, json.dumps(record.labels),
                    record.gpu_time_seconds, record.gpu_type,
                    record.cost_usd, record.tokens_per_second, record.ttft_ms,
                ),
            )
            self._conn.commit()
        except Exception as e:
            logger.warning(f"Failed to insert usage record: {e}")

    def _sqlite_upsert_tenant(self, tenant: TenantUsage) -> None:
        try:
            self._conn.execute(
                """INSERT INTO tenant_usage
                   (tenant_id, total_input_tokens, total_output_tokens,
                    total_requests, total_cost,
                    billing_period_start, billing_period_end, daily_tokens)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(tenant_id) DO UPDATE SET
                   total_input_tokens = excluded.total_input_tokens,
                   total_output_tokens = excluded.total_output_tokens,
                   total_requests = excluded.total_requests,
                   total_cost = excluded.total_cost,
                   daily_tokens = excluded.daily_tokens""",
                (
                    tenant.tenant_id,
                    tenant.total_input_tokens, tenant.total_output_tokens,
                    tenant.total_requests, tenant.total_cost,
                    tenant.current_billing_period_start,
                    tenant.current_billing_period_end,
                    json.dumps(tenant.daily_tokens),
                ),
            )
            self._conn.commit()
        except Exception as e:
            logger.warning(f"Failed to upsert tenant usage: {e}")

    def _sqlite_upsert_quota(self, quota: QuotaLimit) -> None:
        try:
            self._conn.execute(
                """INSERT INTO quotas
                   (tenant_id, max_tokens_per_day, max_requests_per_minute,
                    max_tokens_per_request, max_concurrent_requests,
                    cost_budget_per_month, overage_allowed, overage_multiplier)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(tenant_id) DO UPDATE SET
                   max_tokens_per_day = excluded.max_tokens_per_day,
                   max_requests_per_minute = excluded.max_requests_per_minute,
                   max_tokens_per_request = excluded.max_tokens_per_request,
                   max_concurrent_requests = excluded.max_concurrent_requests,
                   cost_budget_per_month = excluded.cost_budget_per_month,
                   overage_allowed = excluded.overage_allowed,
                   overage_multiplier = excluded.overage_multiplier""",
                (
                    quota.tenant_id,
                    quota.max_tokens_per_day, quota.max_requests_per_minute,
                    quota.max_tokens_per_request, quota.max_concurrent_requests,
                    quota.cost_budget_per_month,
                    int(quota.overage_allowed), quota.overage_multiplier,
                ),
            )
            self._conn.commit()
        except Exception as e:
            logger.warning(f"Failed to upsert quota: {e}")

    def _sqlite_query_records(
        self, tenant_id: str | None = None,
        key_id: str | None = None,
        model_name: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[UsageRecord]:
        parts = ["SELECT * FROM usage_records"]
        params: list[Any] = []
        conditions: list[str] = []
        if tenant_id:
            conditions.append("tenant_id = ?")
            params.append(tenant_id)
        if key_id:
            conditions.append("key_id = ?")
            params.append(key_id)
        if model_name:
            conditions.append("model_name = ?")
            params.append(model_name)
        if conditions:
            parts.append("WHERE " + " AND ".join(conditions))
        parts.append("ORDER BY timestamp DESC LIMIT ? OFFSET ?")
        params.extend([limit, offset])

        try:
            rows = self._conn.execute(" ".join(parts), params).fetchall()
            return [
                UsageRecord(
                    record_id=row["record_id"],
                    tenant_id=row["tenant_id"],
                    key_id=row["key_id"] or "",
                    model_name=row["model_name"] or "",
                    input_tokens=row["input_tokens"] or 0,
                    output_tokens=row["output_tokens"] or 0,
                    total_tokens=row["total_tokens"] or 0,
                    request_duration_ms=row["request_duration_ms"] or 0.0,
                    timestamp=row["timestamp"] or 0.0,
                    cost=row["cost"] or 0.0,
                    endpoint=row["endpoint"] or "",
                    status=UsageRecordStatus(row["status"]) if row["status"] else UsageRecordStatus.PENDING,
                    labels=json.loads(row["labels"]) if row["labels"] else {},
                )
                for row in rows
            ]
        except Exception as e:
            logger.warning(f"Failed to query usage records: {e}")
            return []

    def _load_from_sqlite(self) -> None:
        """Load quotas and tenant usage from SQLite into memory."""
        try:
            rows = self._conn.execute("SELECT * FROM quotas").fetchall()
            for row in rows:
                self._quotas[row["tenant_id"]] = QuotaLimit(
                    tenant_id=row["tenant_id"],
                    max_tokens_per_day=row["max_tokens_per_day"] or 0,
                    max_requests_per_minute=row["max_requests_per_minute"] or 0,
                    max_tokens_per_request=row["max_tokens_per_request"] or 0,
                    max_concurrent_requests=row["max_concurrent_requests"] or 0,
                    cost_budget_per_month=row["cost_budget_per_month"] or 0.0,
                    overage_allowed=bool(row["overage_allowed"]),
                    overage_multiplier=row["overage_multiplier"] or 2.0,
                )
        except Exception as e:
            logger.warning(f"Failed to load quotas from SQLite: {e}")

        try:
            rows = self._conn.execute("SELECT * FROM tenant_usage").fetchall()
            for row in rows:
                self._tenants[row["tenant_id"]] = TenantUsage(
                    tenant_id=row["tenant_id"],
                    total_input_tokens=row["total_input_tokens"] or 0,
                    total_output_tokens=row["total_output_tokens"] or 0,
                    total_requests=row["total_requests"] or 0,
                    total_cost=row["total_cost"] or 0.0,
                    current_billing_period_start=row["billing_period_start"] or 0.0,
                    current_billing_period_end=row["billing_period_end"] or 0.0,
                    daily_tokens=json.loads(row["daily_tokens"]) if row["daily_tokens"] else {},
                )
        except Exception as e:
            logger.warning(f"Failed to load tenant usage from SQLite: {e}")

        # Load persisted usage records back into memory.  Without this, a
        # restart cleared every historical record from invoices / reports /
        # CSV exports (they only iterated _in_memory_records), causing
        # chronic underbilling.
        try:
            rows = self._conn.execute(
                "SELECT * FROM usage_records ORDER BY timestamp ASC"
            ).fetchall()
            for row in rows:
                self._in_memory_records.append(UsageRecord(
                    record_id=row["record_id"],
                    tenant_id=row["tenant_id"],
                    key_id=row["key_id"] or "",
                    model_name=row["model_name"] or "",
                    input_tokens=row["input_tokens"] or 0,
                    output_tokens=row["output_tokens"] or 0,
                    total_tokens=row["total_tokens"] or 0,
                    request_duration_ms=row["request_duration_ms"] or 0.0,
                    timestamp=row["timestamp"] or 0.0,
                    cost=row["cost"] or 0.0,
                    endpoint=row["endpoint"] or "",
                    status=UsageRecordStatus(row["status"]) if row["status"] else UsageRecordStatus.PENDING,
                    labels=json.loads(row["labels"]) if row["labels"] else {},
                    gpu_time_seconds=row["gpu_time_seconds"] or 0.0,
                    gpu_type=row["gpu_type"] or "",
                    cost_usd=row["cost_usd"] or 0.0,
                    tokens_per_second=row["tokens_per_second"] or 0.0,
                    ttft_ms=row["ttft_ms"] or 0.0,
                ))
        except Exception as e:
            logger.warning(f"Failed to load usage records from SQLite: {e}")

    def get_gpu_usage_summary(self, tenant_id: str = "") -> dict[str, Any]:
        """Get GPU-time usage summary.

        Args:
            tenant_id: If provided, filter by tenant. Otherwise aggregate all.

        Returns:
            Dict with GPU-time metrics.
        """
        with self._lock:
            records = self._in_memory_records
            if tenant_id:
                records = [r for r in records if r.tenant_id == tenant_id]

            if not records:
                return {
                    "total_gpu_seconds": 0.0,
                    "total_cost_usd": 0.0,
                    "avg_tokens_per_second": 0.0,
                    "avg_ttft_ms": 0.0,
                    "total_requests": 0,
                    "total_tokens": 0,
                }

            return {
                "total_gpu_seconds": round(sum(r.gpu_time_seconds for r in records), 2),
                "total_cost_usd": round(sum(r.cost_usd for r in records), 6),
                "avg_tokens_per_second": round(
                    sum(r.tokens_per_second for r in records if r.tokens_per_second > 0)
                    / max(sum(1 for r in records if r.tokens_per_second > 0), 1), 1
                ),
                "avg_ttft_ms": round(
                    sum(r.ttft_ms for r in records if r.ttft_ms > 0)
                    / max(sum(1 for r in records if r.ttft_ms > 0), 1), 1
                ),
                "total_requests": len(records),
                "total_tokens": sum(r.total_tokens for r in records),
                "gpu_types": list({r.gpu_type for r in records if r.gpu_type}),
            }

    def close(self) -> None:
        """Close the SQLite connection."""
        if self._use_sqlite and hasattr(self, "_conn"):
            self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self) -> None:
        self.close()


# ── Factory ─────────────────────────────────────────────────────────────────


def create_usage_meter(
    storage_path: str = "",
    input_price: float = 0.01,
    output_price: float = 0.03,
    use_sqlite: bool = True,
) -> UsageMeter:
    """Convenience factory for UsageMeter."""
    return UsageMeter(
        storage_path=storage_path,
        input_price=input_price,
        output_price=output_price,
        use_sqlite=use_sqlite,
    )
