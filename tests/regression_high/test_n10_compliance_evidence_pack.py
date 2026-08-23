"""Regression tests for N10: compliance & certification evidence pack.

Exercises ``distllm.compliance.evidence_pack`` which aggregates N4's SOC 2 /
ISO 27001 evidence with doc-derived GDPR / Export-Control mappings and a
HIPAA control family (flagged UNVERIFIED, since no HIPAA source doc exists).

Model-free, no network. Asserts:
  (a) build_evidence_pack() returns a pack with soc2 / iso / hipaa keys
  (b) GDPR + Export controls are present and cite their source docs
  (c) HIPAA controls are flagged with a caveat when the source doc is absent
  (d) JSON + Markdown emit succeed and round-trip / parse
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from distllm.compliance.evidence_pack import (
    ControlRef,
    EvidencePack,
    build_evidence_pack,
    emit_json,
    emit_markdown,
    parse_export_controls,
    parse_gdpr_controls,
    parse_hipaa_controls,
    write_evidence_pack,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS_DIR = REPO_ROOT / "docs"
COMPLIANCE_DIR = REPO_ROOT / "compliance"

NOW = datetime(2026, 7, 14, 12, 0, 0, tzinfo=timezone.utc)


# --------------------------------------------------------------------------- #
# (a) pack builds with soc2 / iso / hipaa keys
# --------------------------------------------------------------------------- #
def test_build_evidence_pack_has_frameworks():
    pack = build_evidence_pack(
        docs_dir=DOCS_DIR,
        compliance_dir=COMPLIANCE_DIR,
        repo_root=REPO_ROOT,
        now=NOW,
    )
    assert isinstance(pack, EvidencePack)
    # every framework family key is present (as an attribute)
    for attr in (
        "soc2_controls",
        "iso27001_controls",
        "hipaa_controls",
        "gdpr_controls",
        "export_controls",
        "audit_log_samples",
        "pentest_summary",
        "caveats",
    ):
        assert hasattr(pack, attr), f"missing {attr}"
    # N4-derived families are non-empty (it always produces 4 domains)
    assert pack.soc2_controls, "SOC 2 controls should be derived from N4"
    assert pack.iso27001_controls, "ISO 27001 controls should be derived from N4"
    # HIPAA always produced (as unverified scaffold) when no source doc
    assert pack.hipaa_controls, "HIPAA controls should be present (unverified)"
    assert pack.generated_at == NOW.isoformat()
    # dict round-trip
    d = pack.to_dict()
    assert set(d) >= {
        "soc2_controls",
        "iso27001_controls",
        "hipaa_controls",
        "gdpr_controls",
        "export_controls",
        "audit_log_samples",
        "pentest_summary",
        "caveats",
        "generated_at",
    }


# --------------------------------------------------------------------------- #
# (b) GDPR + Export controls present and cite source docs
# --------------------------------------------------------------------------- #
def test_gdpr_controls_present_and_cited():
    gdpr = parse_gdpr_controls(DOCS_DIR)
    assert gdpr, "GDPR controls should be parsed from docs/GDPR.md"
    for c in gdpr:
        assert c.source_doc == "docs/GDPR.md"
        assert c.source_section.startswith("L")
        assert c.framework == "GDPR"
    # spot-check that well-known principles appear
    ids = {c.control_id for c in gdpr}
    assert any("Art. 5(1)(c)" in cid for cid in ids)  # data minimization
    assert any("Art. 32" in cid for cid in ids)  # integrity & confidentiality


def test_export_controls_present_and_cited():
    exp = parse_export_controls(DOCS_DIR)
    assert exp, "Export controls should be parsed from docs/EXPORT_CONTROLS.md"
    for c in exp:
        assert c.source_doc == "docs/EXPORT_CONTROLS.md"
        assert c.source_section.startswith("L")
        assert c.framework == "EXPORT"
    # US EAR + EU AI Act + Wassenaar should be present
    titles = " ".join(c.title.lower() for c in exp)
    assert "export administration" in titles
    assert "ai act" in titles
    assert "wassenaar" in titles


def test_pack_includes_gdpr_and_export_families():
    pack = build_evidence_pack(
        docs_dir=DOCS_DIR, compliance_dir=COMPLIANCE_DIR,
        repo_root=REPO_ROOT, now=NOW,
    )
    assert pack.gdpr_controls, "pack should carry GDPR family"
    assert pack.export_controls, "pack should carry Export family"
    assert pack.gdpr_controls[0].source_doc == "docs/GDPR.md"
    assert pack.export_controls[0].source_doc == "docs/EXPORT_CONTROLS.md"


# --------------------------------------------------------------------------- #
# (c) HIPAA controls flagged with caveat when source doc absent
# --------------------------------------------------------------------------- #
def test_hipaa_controls_unverified_no_source_doc():
    controls, support = parse_hipaa_controls(
        security_doc_paths=[REPO_ROOT / "SECURITY.md",
                            DOCS_DIR / "SECURITY_HARDENING.md"]
    )
    assert controls, "HIPAA provisions should be mapped"
    for c in controls:
        assert c.status == "unverified", c
        assert "UNVERIFIED" in c.notes, c
        assert "HIPAA.md" in c.source_doc or "none" in c.source_doc.lower(), c
    # spot-check a classic HIPAA Security Rule provision is present
    ids = {c.control_id for c in controls}
    assert any("164.312" in cid for cid in ids)


def test_pack_hipaa_caveat_present():
    pack = build_evidence_pack(
        docs_dir=DOCS_DIR, compliance_dir=COMPLIANCE_DIR,
        repo_root=REPO_ROOT, now=NOW,
    )
    # at least one caveat must mention HIPAA + absence of source doc
    hipaa_caveat = [c for c in pack.caveats if "HIPAA" in c and "HIPAA.md" in c]
    assert hipaa_caveat, "a HIPAA caveat (no source doc) must be present"
    assert any("UNVERIFIED" in c for c in hipaa_caveat)
    # every HIPAA control in the pack is unverified
    assert all(c.status == "unverified" for c in pack.hipaa_controls)


# --------------------------------------------------------------------------- #
# (d) JSON + Markdown emit succeed and parse
# --------------------------------------------------------------------------- #
def test_emit_json_and_markdown(tmp_path):
    pack = build_evidence_pack(
        docs_dir=DOCS_DIR, compliance_dir=COMPLIANCE_DIR,
        repo_root=REPO_ROOT, now=NOW,
    )
    written = write_evidence_pack(pack, output_dir=str(tmp_path), basename="N10_pack")
    assert len(written) == 2

    json_path = tmp_path / "N10_pack.json"
    md_path = tmp_path / "N10_pack.md"
    assert json_path.exists() and md_path.exists()

    # JSON parses and round-trips into a dict with the families
    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data["soc2_controls"] and data["iso27001_controls"]
    assert data["hipaa_controls"] and data["gdpr_controls"] and data["export_controls"]
    assert data["caveats"]

    # Markdown parses (contains key section headers + disclaims certification)
    md = md_path.read_text(encoding="utf-8")
    assert "NOT A CERTIFICATION" in md
    assert "## HIPAA Controls" in md
    assert "UNVERIFIED" in md
    assert "## GDPR Controls" in md
    assert "## Export Controls" in md
    assert "## Audit-log samples" in md


def test_emit_helpers_individually(tmp_path):
    pack = build_evidence_pack(
        docs_dir=DOCS_DIR, compliance_dir=COMPLIANCE_DIR,
        repo_root=REPO_ROOT, now=NOW,
    )
    jp = emit_json(pack, tmp_path / "sub" / "e.json")
    mp = emit_markdown(pack, tmp_path / "sub" / "e.md")
    assert Path(jp).exists() and Path(mp).exists()
    json.loads(Path(jp).read_text(encoding="utf-8"))  # parseable


def test_audit_log_samples_and_pentest_present():
    pack = build_evidence_pack(
        docs_dir=DOCS_DIR, compliance_dir=COMPLIANCE_DIR,
        repo_root=REPO_ROOT, now=NOW,
    )
    assert pack.audit_log_samples, "audit-log samples should be present"
    assert isinstance(pack.pentest_summary, dict)
    # honest: no actual pen-test report claimed
    assert pack.pentest_summary.get("actual_pentest_report") is False
    assert "disclosure_sla" in pack.pentest_summary
