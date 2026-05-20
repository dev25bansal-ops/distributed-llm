"""Request Auditor: compliance logging, PII detection, and audit trails.

Logs every request with full metadata for compliance auditing.
Supports PII pattern detection (emails, SSNs, phone numbers, API keys)
and structured audit log export.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# Common PII patterns
PII_PATTERNS: dict[str, re.Pattern] = {
    "email": re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "phone": re.compile(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b"),
    "credit_card": re.compile(r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b"),
    "ip_address": re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"),
    "api_key": re.compile(r"(?i)(sk-[a-zA-Z0-9]{20,}|api[-_]?key['\"]?\s*[:=]\s*['\"][a-zA-Z0-9_\-]{16,}['\"])"),
    "aws_key": re.compile(r"(?i)AKIA[0-9A-Z]{16}"),
}


@dataclass
class AuditEntry:
    """A single audit log entry for a request."""
    request_id: str
    timestamp: str
    user: str
    model: str
    prompt_hash: str
    prompt_length_chars: int
    response_hash: str | None = None
    response_length_chars: int = 0
    duration_ms: float = 0.0
    pii_found: list[str] = field(default_factory=list)
    status: str = "success"
    error: str | None = None
    ip_address: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class PIIInspector:
    """Detects PII in text using regex patterns."""

    def __init__(self, patterns: dict[str, re.Pattern] | None = None):
        self._patterns = patterns or dict(PII_PATTERNS)

    def inspect(self, text: str) -> list[str]:
        """Return list of PII types found in text."""
        found: list[str] = []
        for name, pattern in self._patterns.items():
            if pattern.search(text):
                found.append(name)
        return found

    def redact(self, text: str, replacement: str = "[REDACTED]") -> str:
        """Replace all PII matches with a placeholder."""
        result = text
        for pattern in self._patterns.values():
            result = pattern.sub(replacement, result)
        return result


class RequestAuditor:
    """Logs all requests with metadata for compliance.

    Maintains an in-memory ring buffer and optionally writes to
    a JSON-lines audit file on disk.

    Usage:
        auditor = RequestAuditor(log_dir="./audit_logs")
        auditor.record(request_id="abc", prompt="hello", user="user-1", model="gpt-4")
        auditor.export() -> list of AuditEntry dicts
    """

    def __init__(
        self,
        max_entries: int = 10000,
        log_dir: str | None = None,
        enable_pii_detection: bool = True,
    ):
        self._max = max_entries
        self._log_dir = Path(log_dir) if log_dir else None
        self._enable_pii = enable_pii_detection
        self._entries: list[AuditEntry] = []
        self._index: dict[str, AuditEntry] = {}
        self._lock = threading.Lock()
        self._pii = PIIInspector() if enable_pii_detection else None

        if self._log_dir:
            self._log_dir.mkdir(parents=True, exist_ok=True)

    def record(
        self,
        request_id: str,
        prompt: str,
        user: str = "default",
        model: str = "",
        response: str | None = None,
        duration_ms: float = 0.0,
        status: str = "success",
        error: str | None = None,
        ip_address: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record a request in the audit log."""
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:16]
        response_hash = (
            hashlib.sha256(response.encode()).hexdigest()[:16]
            if response else None
        )

        pii_found: list[str] = []
        if self._pii:
            pii_found = self._pii.inspect(prompt)
            if response:
                pii_found.extend(self._pii.inspect(response))
            pii_found = list(set(pii_found))

        entry = AuditEntry(
            request_id=request_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            user=user,
            model=model,
            prompt_hash=prompt_hash,
            prompt_length_chars=len(prompt),
            response_hash=response_hash,
            response_length_chars=len(response) if response else 0,
            duration_ms=duration_ms,
            pii_found=pii_found,
            status=status,
            error=error,
            ip_address=ip_address,
            metadata=metadata or {},
        )

        with self._lock:
            self._entries.append(entry)
            self._index[request_id] = entry
            if len(self._entries) > self._max:
                removed = self._entries.pop(0)
                self._index.pop(removed.request_id, None)

        # Write to audit log file if configured
        if self._log_dir:
            self._write_log(entry)

    def _write_log(self, entry: AuditEntry) -> None:
        """Append entry as JSON line to audit log file."""
        log_file = self._log_dir / f"audit-{datetime.now(timezone.utc).strftime('%Y%m%d')}.jsonl"
        try:
            with open(log_file, "a") as f:
                f.write(json.dumps(asdict(entry)) + "\n")
        except OSError as e:
            pass  # Fail silently for audit logging

    def get(self, request_id: str) -> AuditEntry | None:
        with self._lock:
            return self._index.get(request_id)

    def search(
        self,
        user: str | None = None,
        pii_type: str | None = None,
        status: str | None = None,
        limit: int = 100,
    ) -> list[AuditEntry]:
        """Search audit entries by filters."""
        with self._lock:
            results = list(self._entries)
        if user:
            results = [e for e in results if e.user == user]
        if pii_type:
            results = [e for e in results if pii_type in e.pii_found]
        if status:
            results = [e for e in results if e.status == status]
        return results[-limit:]

    def export(self, limit: int = 1000) -> list[dict]:
        with self._lock:
            return [asdict(e) for e in self._entries[-limit:]]

    def size(self) -> int:
        with self._lock:
            return len(self._entries)

    def stats(self) -> dict[str, Any]:
        with self._lock:
            total = len(self._entries)
            pii_count = sum(1 for e in self._entries if e.pii_found)
            error_count = sum(1 for e in self._entries if e.status == "error")
            return {
                "total_entries": total,
                "pii_detected": pii_count,
                "errors": error_count,
                "pii_rate": pii_count / max(total, 1),
            }
