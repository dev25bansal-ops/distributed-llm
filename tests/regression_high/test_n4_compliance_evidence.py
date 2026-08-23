"""Regression tests for N4: automated SOC 2 / ISO 27001 evidence collector.

These tests exercise ``distllm.core.compliance_evidence`` using *injected,
deterministic* configuration/state objects (no live server, no network). They
assert:

  1. Each collector returns structured evidence with a pass/fail/warn status.
  2. An insecure CORS config (``'*'`` + allow_credentials) is flagged FAIL.
  3. Overdue key/cert rotation is flagged FAIL.
  4. ``collect_all`` maps to SOC 2 + ISO 27001 control IDs and yields a
     JSON-serializable report.

No certification is claimed. The collector automates evidence-gathering only.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from distllm.core.compliance_evidence import (
    CONTROL_MAP,
    collect_all,
    collect_cors_evidence,
    collect_key_rotation_evidence,
    collect_quota_evidence,
    collect_tls_evidence,
    write_report,
)

NOW = datetime(2026, 7, 14, 12, 0, 0, tzinfo=timezone.utc)


def _cfg(**kwargs):
    """Build a trivial attribute-bearing config object from kwargs."""
    from types import SimpleNamespace

    return SimpleNamespace(**kwargs)


# --------------------------------------------------------------------------- #
# 1. Each collector returns structured evidence with a valid status
# --------------------------------------------------------------------------- #
def test_each_collector_returns_structured_evidence():
    cors = collect_cors_evidence(
        _cfg(cors_origins="https://a.example,https://b.example", allow_credentials=False),
        now=NOW,
    )
    tls = collect_tls_evidence(
        _cfg(enabled=True, min_tls_version=1.3, require_client_cert=True, ca_cert_file="/ca"),
        cert_expiry=NOW + timedelta(days=90),
        now=NOW,
    )
    quota = collect_quota_evidence(_cfg(enabled=True, per_tenant=True, default_rpm=60), now=NOW)
    rot = collect_key_rotation_evidence(
        _cfg(last_rotation_ts=NOW - timedelta(days=10), rotation_interval_days=90), now=NOW
    )

    for name, ev in (("CORS", cors), ("TLS", tls), ("QUOTA", quota), ("KEY_ROTATION", rot)):
        assert ev["control"] == name
        assert ev["status"] in ("pass", "fail", "warn"), ev
        assert isinstance(ev["evidence"], dict) and ev["evidence"]
        assert ev["collected_at"] == NOW.isoformat()
        # Mapped control IDs must be present (SOC2 + ISO)
        assert ev["control_ids"], f"{name} missing control_ids"
        assert any(c.startswith("ISO27001:") for c in ev["control_ids"]), name
        assert any(not c.startswith("ISO27001:") for c in ev["control_ids"]), name


# --------------------------------------------------------------------------- #
# 2. Insecure CORS ('*' + credentials) is flagged FAIL
# --------------------------------------------------------------------------- #
def test_insecure_cors_wildcard_with_credentials_flagged_fail():
    # wildcard + credentials
    ev = collect_cors_evidence(
        _cfg(cors_origins="https://a.example,*", allow_credentials=True), now=NOW
    )
    assert ev["status"] == "fail"
    assert ev["evidence"]["wildcard"] is True
    assert "INSECURE" in ev["notes"]

    # wildcard alone (no creds) is also fail (allow-all)
    ev2 = collect_cors_evidence(_cfg(cors_origins="*", allow_credentials=False), now=NOW)
    assert ev2["status"] == "fail"

    # explicit allowlist => pass
    ev3 = collect_cors_evidence(
        _cfg(cors_origins="https://a.example", allow_credentials=True), now=NOW
    )
    assert ev3["status"] == "pass"


# --------------------------------------------------------------------------- #
# 3. Overdue key rotation is flagged FAIL
# --------------------------------------------------------------------------- #
def test_overdue_key_rotation_flagged():
    ev = collect_key_rotation_evidence(
        _cfg(last_rotation_ts=NOW - timedelta(days=120), rotation_interval_days=90),
        now=NOW,
    )
    assert ev["status"] == "fail"
    assert "OVERDUE" in ev["notes"]
    assert ev["evidence"]["age_days"] == 120
    assert ev["evidence"]["rotation_interval_days"] == 90

    # not yet overdue => pass; soon => warn; missing ts => fail
    assert (
        collect_key_rotation_evidence(
            _cfg(last_rotation_ts=NOW - timedelta(days=10), rotation_interval_days=90), now=NOW
        )["status"]
        == "pass"
    )
    assert (
        collect_key_rotation_evidence(
            _cfg(last_rotation_ts=NOW - timedelta(days=80), rotation_interval_days=90), now=NOW
        )["status"]
        == "warn"
    )
    assert (
        collect_key_rotation_evidence(_cfg(rotation_interval_days=90), now=NOW)["status"]
        == "fail"
    )


# --------------------------------------------------------------------------- #
# 4. collect_all maps to control IDs and is serializable
# --------------------------------------------------------------------------- #
def test_collect_all_maps_controls_and_serializes():
    bundle = {
        "cors": _cfg(cors_origins="https://app.example", allow_credentials=False),
        "tls": _cfg(enabled=True, min_tls_version=1.2),
        "cert_expiry": NOW + timedelta(days=60),
        "quota": _cfg(enabled=True, per_tenant=True, default_rpm=120),
        "key_rotation": _cfg(
            last_rotation_ts=NOW - timedelta(days=20), rotation_interval_days=30
        ),
        "now": NOW,
    }
    report = collect_all(bundle=bundle, now=NOW)

    # Every domain present
    assert set(report["controls"]) == {"CORS", "TLS", "QUOTA", "KEY_ROTATION"}
    # control_id mapping reused from soc2_control_mapping.md via CONTROL_MAP
    assert CONTROL_MAP["CORS"]["soc2"] == ["CC6.6"]
    assert "CC6.1" in CONTROL_MAP["TLS"]["soc2"]
    for entry in report["controls"].values():
        assert any(c.startswith("ISO27001:") for c in entry["control_ids"])

    # Summary present and consistent
    assert report["summary"]["total"] == 4
    counted = (
        report["summary"]["pass"]
        + report["summary"]["fail"]
        + report["summary"]["warn"]
    )
    assert counted == 4
    assert "disclaimer" in report and "NOT" not in report["disclaimer"].upper().split()[0]

    # JSON serializable round-trip
    dumped = json.dumps(report)
    assert isinstance(dumped, str)
    reloaded = json.loads(dumped)
    assert reloaded["summary"]["total"] == 4


def test_collect_all_detects_fail_paths():
    """Show collect_all aggregates failures (CORS wildcard + disabled quota)."""
    bundle = {
        "cors": _cfg(cors_origins="*", allow_credentials=True),  # FAIL
        "tls": _cfg(enabled=True, min_tls_version=1.3),  # pass
        "quota": _cfg(enabled=False),  # FAIL
        "key_rotation": _cfg(
            last_rotation_ts=NOW - timedelta(days=200), rotation_interval_days=90  # FAIL
        ),
        "now": NOW,
    }
    report = collect_all(bundle=bundle, now=NOW)
    assert report["summary"]["fail"] == 3
    assert report["controls"]["CORS"]["status"] == "fail"
    assert report["controls"]["QUOTA"]["status"] == "fail"
    assert report["controls"]["KEY_ROTATION"]["status"] == "fail"


def test_write_report_emits_json_and_markdown(tmp_path):
    """Optional report writer produces JSON + Markdown files."""
    bundle = {
        "cors": _cfg(cors_origins="https://app.example"),
        "tls": _cfg(enabled=False),
        "quota": _cfg(enabled=True, per_tenant=False),
        "key_rotation": _cfg(last_rotation_ts=NOW - timedelta(days=1), rotation_interval_days=30),
        "now": NOW,
    }
    report = collect_all(bundle=bundle, now=NOW)
    written = write_report(report, output_dir=str(tmp_path), basename="N4_ComplianceReport")
    assert len(written) == 2
    json_text = (tmp_path / "N4_ComplianceReport.json").read_text(encoding="utf-8")
    md_text = (tmp_path / "N4_ComplianceReport.md").read_text(encoding="utf-8")
    # reloadable JSON
    json.loads(json_text)
    # markdown disclaims certification
    assert "NOT" in md_text and "certif" in md_text.lower()
    # CORS pass appears under its section
    assert "CORS" in md_text
