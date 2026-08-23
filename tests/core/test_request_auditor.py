"""Tests for RequestAuditor, PIIInspector, and AuditEntry.

Covers:
- PIIInspector: inspect (email, SSN, phone, credit card, IP, API key, AWS key)
- PIIInspector: redact replaces matches
- PIIInspector: no false positives on clean text
- AuditEntry dataclass defaults and full construction
- RequestAuditor: record, get, search by user/pii/status
- RequestAuditor: export, size, stats
- RequestAuditor: ring buffer eviction at max_entries
- RequestAuditor: thread safety
- RequestAuditor: log_dir creates directory and writes JSONL
"""

from __future__ import annotations

import json
import os
import threading
import tempfile
from pathlib import Path
from typing import Any

import pytest

from tests._import_helper import bootstrap_fake_packages, load_module

bootstrap_fake_packages()

_mod = load_module("distllm/core/request_auditor.py")
RequestAuditor = _mod.RequestAuditor
PIIInspector = _mod.PIIInspector
AuditEntry = _mod.AuditEntry
PII_PATTERNS = _mod.PII_PATTERNS


# ---------------------------------------------------------------------------
# PIIInspector
# ---------------------------------------------------------------------------


class TestPIIInspector:
    """PII detection and redaction."""

    def test_inspect_clean_text(self) -> None:
        inspector = PIIInspector()
        result = inspector.inspect("Hello world, this is clean text.")
        assert result == []

    def test_inspect_email(self) -> None:
        inspector = PIIInspector()
        result = inspector.inspect("Contact me at user@example.com")
        assert "email" in result

    def test_inspect_ssn(self) -> None:
        inspector = PIIInspector()
        result = inspector.inspect("SSN: 123-45-6789")
        assert "ssn" in result

    def test_inspect_phone(self) -> None:
        inspector = PIIInspector()
        result = inspector.inspect("Call 555-123-4567")
        assert "phone" in result

    def test_inspect_credit_card(self) -> None:
        inspector = PIIInspector()
        result = inspector.inspect("Card: 4111-1111-1111-1111")
        assert "credit_card" in result

    def test_inspect_ip_address(self) -> None:
        inspector = PIIInspector()
        result = inspector.inspect("Server: 192.168.1.1")
        assert "ip_address" in result

    def test_inspect_api_key(self) -> None:
        inspector = PIIInspector()
        result = inspector.inspect("Key: sk-abcdefghijklmnopqrstuvwxyz")
        assert "api_key" in result

    def test_inspect_aws_key(self) -> None:
        inspector = PIIInspector()
        result = inspector.inspect("AWS: AKIA1234567890123456")
        assert "aws_key" in result

    def test_inspect_multiple_pii_types(self) -> None:
        inspector = PIIInspector()
        result = inspector.inspect("Email: a@b.com, IP: 1.2.3.4")
        assert "email" in result
        assert "ip_address" in result

    def test_inspect_empty_string(self) -> None:
        inspector = PIIInspector()
        assert inspector.inspect("") == []

    def test_redact_email(self) -> None:
        inspector = PIIInspector()
        result = inspector.redact("Email: user@example.com")
        assert "user@example.com" not in result
        assert "[REDACTED]" in result

    def test_redact_all_types(self) -> None:
        inspector = PIIInspector()
        text = "Email: a@b.com, SSN: 123-45-6789"
        result = inspector.redact(text)
        assert "[REDACTED]" in result
        assert "a@b.com" not in result
        assert "123-45-6789" not in result

    def test_redact_custom_replacement(self) -> None:
        inspector = PIIInspector()
        result = inspector.redact("Email: a@b.com", replacement="***")
        assert "***" in result
        assert "a@b.com" not in result

    def test_redact_clean_text_unchanged(self) -> None:
        inspector = PIIInspector()
        text = "Just some regular text without PII."
        assert inspector.redact(text) == text

    def test_custom_patterns(self) -> None:
        patterns = {"custom_id": __import__("re").compile(r"ID-\d{5}")}
        inspector = PIIInspector(patterns=patterns)
        assert inspector.inspect("ID-12345") == ["custom_id"]
        assert inspector.inspect("no match") == []


# ---------------------------------------------------------------------------
# AuditEntry dataclass
# ---------------------------------------------------------------------------


class TestAuditEntry:
    """AuditEntry dataclass defaults and construction."""

    def test_defaults(self) -> None:
        entry = AuditEntry(
            request_id="r1",
            timestamp="2025-01-01T00:00:00",
            user="user-1",
            model="gpt-4",
            prompt_hash="abc123",
            prompt_length_chars=5,
        )
        assert entry.response_hash is None
        assert entry.response_length_chars == 0
        assert entry.duration_ms == 0.0
        assert entry.pii_found == []
        assert entry.status == "success"
        assert entry.error is None
        assert entry.ip_address is None
        assert entry.metadata == {}

    def test_full_construction(self) -> None:
        entry = AuditEntry(
            request_id="r1",
            timestamp="2025-01-01T00:00:00",
            user="user-1",
            model="gpt-4",
            prompt_hash="abc",
            prompt_length_chars=5,
            response_hash="def",
            response_length_chars=12,
            duration_ms=45.0,
            pii_found=["email"],
            status="error",
            error="timeout",
            ip_address="10.0.0.1",
            metadata={"key": "val"},
        )
        assert entry.response_hash == "def"
        assert entry.duration_ms == 45.0
        assert entry.pii_found == ["email"]
        assert entry.status == "error"
        assert entry.error == "timeout"
        assert entry.ip_address == "10.0.0.1"


# ---------------------------------------------------------------------------
# RequestAuditor construction
# ---------------------------------------------------------------------------


class TestRequestAuditorConstruction:
    """Construction and initial state."""

    def test_default_construction(self) -> None:
        auditor = RequestAuditor()
        assert auditor._max == 10000
        assert auditor._log_dir is None
        assert auditor._enable_pii is True
        assert auditor._pii is not None
        assert auditor.size() == 0

    def test_pii_disabled(self) -> None:
        auditor = RequestAuditor(enable_pii_detection=False)
        assert auditor._pii is None

    def test_custom_max_entries(self) -> None:
        auditor = RequestAuditor(max_entries=50)
        assert auditor._max == 50


# ---------------------------------------------------------------------------
# RequestAuditor record / get
# ---------------------------------------------------------------------------


class TestRequestAuditorRecordAndGet:
    """Record entries and look them up."""

    def test_record_basic(self) -> None:
        auditor = RequestAuditor()
        auditor.record(request_id="r1", prompt="hello", user="u1", model="gpt-4")
        assert auditor.size() == 1

    def test_get_entry(self) -> None:
        auditor = RequestAuditor()
        auditor.record(request_id="r1", prompt="hello", model="gpt-4")
        entry = auditor.get("r1")
        assert entry is not None
        assert entry.request_id == "r1"
        assert entry.prompt_hash is not None
        assert entry.model == "gpt-4"

    def test_get_nonexistent(self) -> None:
        auditor = RequestAuditor()
        assert auditor.get("nonexistent") is None

    def test_record_with_response(self) -> None:
        auditor = RequestAuditor()
        auditor.record(request_id="r1", prompt="hello", response="world", duration_ms=10.0, model="gpt-4")
        entry = auditor.get("r1")
        assert entry is not None
        assert entry.response_hash is not None
        assert entry.duration_ms == 10.0

    def test_record_with_error(self) -> None:
        auditor = RequestAuditor()
        auditor.record(request_id="r1", prompt="hello", status="error", error="timeout", model="gpt-4")
        entry = auditor.get("r1")
        assert entry is not None
        assert entry.status == "error"
        assert entry.error == "timeout"

    def test_record_pii_detection_on_prompt(self) -> None:
        auditor = RequestAuditor()
        auditor.record(request_id="r1", prompt="Email: user@example.com", model="gpt-4")
        entry = auditor.get("r1")
        assert entry is not None
        assert "email" in entry.pii_found

    def test_record_pii_detection_on_response(self) -> None:
        auditor = RequestAuditor()
        auditor.record(request_id="r1", prompt="hello", response="IP: 10.0.0.1", model="gpt-4")
        entry = auditor.get("r1")
        assert entry is not None
        assert "ip_address" in entry.pii_found


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------


class TestRequestAuditorSearch:
    """Search by user, pii_type, status."""

    def test_search_by_user(self) -> None:
        auditor = RequestAuditor()
        auditor.record(request_id="r1", prompt="a", user="alice", model="gpt-4")
        auditor.record(request_id="r2", prompt="b", user="bob", model="gpt-4")
        results = auditor.search(user="alice")
        assert len(results) == 1
        assert results[0].request_id == "r1"

    def test_search_by_status(self) -> None:
        auditor = RequestAuditor()
        auditor.record(request_id="r1", prompt="a", status="success", model="gpt-4")
        auditor.record(request_id="r2", prompt="b", status="error", model="gpt-4")
        results = auditor.search(status="error")
        assert len(results) == 1
        assert results[0].request_id == "r2"

    def test_search_by_pii(self) -> None:
        auditor = RequestAuditor()
        auditor.record(request_id="r1", prompt="Email: a@b.com", model="gpt-4")
        auditor.record(request_id="r2", prompt="clean text", model="gpt-4")
        results = auditor.search(pii_type="email")
        assert len(results) == 1
        assert results[0].request_id == "r1"

    def test_search_no_match(self) -> None:
        auditor = RequestAuditor()
        auditor.record(request_id="r1", prompt="hello", model="gpt-4")
        assert auditor.search(user="nobody") == []

    def test_search_limit(self) -> None:
        auditor = RequestAuditor()
        for i in range(20):
            auditor.record(request_id=f"r{i}", prompt=str(i), model="gpt-4")
        results = auditor.search(limit=5)
        assert len(results) == 5


# ---------------------------------------------------------------------------
# export / size / stats
# ---------------------------------------------------------------------------


class TestRequestAuditorExport:
    """Export and stats methods."""

    def test_export(self) -> None:
        auditor = RequestAuditor()
        auditor.record(request_id="r1", prompt="hello", model="gpt-4")
        exported = auditor.export()
        assert len(exported) == 1
        assert exported[0]["request_id"] == "r1"
        assert "prompt_hash" in exported[0]
        assert "timestamp" in exported[0]

    def test_export_limit(self) -> None:
        auditor = RequestAuditor()
        for i in range(10):
            auditor.record(request_id=f"r{i}", prompt=str(i), model="gpt-4")
        assert len(auditor.export(limit=3)) == 3

    def test_size(self) -> None:
        auditor = RequestAuditor()
        assert auditor.size() == 0
        auditor.record(request_id="r1", prompt="a", model="gpt-4")
        assert auditor.size() == 1

    def test_stats_empty(self) -> None:
        auditor = RequestAuditor()
        stats = auditor.stats()
        assert stats["total_entries"] == 0
        assert stats["pii_detected"] == 0
        assert stats["errors"] == 0
        assert stats["pii_rate"] == 0.0

    def test_stats_with_data(self) -> None:
        auditor = RequestAuditor()
        auditor.record(request_id="r1", prompt="Email: a@b.com", model="gpt-4")
        auditor.record(request_id="r2", prompt="clean", status="error", model="gpt-4")
        auditor.record(request_id="r3", prompt="hello", model="gpt-4")
        stats = auditor.stats()
        assert stats["total_entries"] == 3
        assert stats["pii_detected"] == 1
        assert stats["errors"] == 1
        assert stats["pii_rate"] == pytest.approx(1 / 3)


# ---------------------------------------------------------------------------
# Ring buffer eviction
# ---------------------------------------------------------------------------


class TestRequestAuditorEviction:
    """Ring buffer eviction at max_entries."""

    def test_evicts_oldest_when_full(self) -> None:
        auditor = RequestAuditor(max_entries=3)
        auditor.record(request_id="r1", prompt="a", model="gpt-4")
        auditor.record(request_id="r2", prompt="b", model="gpt-4")
        auditor.record(request_id="r3", prompt="c", model="gpt-4")
        auditor.record(request_id="r4", prompt="d", model="gpt-4")
        assert auditor.size() == 3
        assert auditor.get("r1") is None  # evicted
        assert auditor.get("r2") is not None
        assert auditor.get("r4") is not None

    def test_index_removed_on_eviction(self) -> None:
        auditor = RequestAuditor(max_entries=2)
        auditor.record(request_id="r1", prompt="a", model="gpt-4")
        auditor.record(request_id="r2", prompt="b", model="gpt-4")
        auditor.record(request_id="r3", prompt="c", model="gpt-4")
        assert "r1" not in auditor._index


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------


class TestRequestAuditorThreadSafety:
    """Thread safety under concurrent access."""

    def test_concurrent_record(self) -> None:
        auditor = RequestAuditor(max_entries=500)
        errors: list[Exception] = []

        def record_range(start: int, count: int) -> None:
            try:
                for i in range(count):
                    auditor.record(request_id=f"r{start + i}", prompt=str(start + i), model="gpt-4")
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=record_range, args=(0, 50)),
            threading.Thread(target=record_range, args=(50, 50)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert auditor.size() == 100

    def test_concurrent_get_and_record(self) -> None:
        auditor = RequestAuditor()
        auditor.record(request_id="r1", prompt="hello", model="gpt-4")

        def query() -> None:
            for _ in range(50):
                auditor.get("r1")
                auditor.stats()

        threads = [threading.Thread(target=query) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert auditor.size() == 1


# ---------------------------------------------------------------------------
# File-based logging
# ---------------------------------------------------------------------------


class TestRequestAuditorFileLogging:
    """Audit log file written when log_dir is set."""

    def test_log_dir_creates_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = os.path.join(tmp, "audit_logs")
            auditor = RequestAuditor(log_dir=log_dir)
            assert os.path.isdir(log_dir)

    def test_writes_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            auditor = RequestAuditor(log_dir=tmp)
            auditor.record(request_id="r1", prompt="hello", model="gpt-4")
            # Find the log file
            log_files = list(Path(tmp).glob("audit-*.jsonl"))
            assert len(log_files) == 1
            content = log_files[0].read_text()
            assert "r1" in content
            assert "prompt_hash" in content
            # Validate JSON
            record = json.loads(content.strip())
            assert record["request_id"] == "r1"

    def test_write_failure_does_not_raise(self) -> None:
        auditor = RequestAuditor(log_dir="/nonexistent/path/should/fail")
        # Should not raise; errors are swallowed silently
        auditor.record(request_id="r1", prompt="hello", model="gpt-4")
        assert auditor.size() == 1
