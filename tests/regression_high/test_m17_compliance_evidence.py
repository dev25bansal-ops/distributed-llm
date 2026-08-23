"""Regression tests for M17: compliance evidence pack is a real, non-empty scaffold.

These tests assert that the ``compliance/`` evidence pack exists and that its SOC 2
control-mapping table references control IDs that *genuinely appear* in
``docs/SECURITY_HARDENING.md``. This prevents the scaffold from being an empty / fabricated
mapping: the overlap between the mapping table and the real source document must be non-trivial.

No certification is claimed by these tests. They only verify documentation scaffolding exists
and is wired to real control identifiers.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
COMPLIANCE_DIR = REPO_ROOT / "compliance"
DOCS_DIR = REPO_ROOT / "docs"
SEC_HARDENING = DOCS_DIR / "SECURITY_HARDENING.md"

# Real control identifiers that appear verbatim in docs/SECURITY_HARDENING.md.
# (Verified by grep during authoring; the test re-derives them from the file itself so it
#  cannot silently drift.)
KNOWN_REAL_CONTROL_IDS = [
    "TLS",
    "RBAC",
    "SSRF",
    "CORS",
    "Rate Limiting",
    "Secret Management",
    "audit logging",
    "encryption at rest",
    "Monitoring",
    "Incident Response",
    "Authentication",
    "Firewall",
    "Container Security",
    "Input Validation",
    "DISTLLM_AUDIT_LOGGING",
    "DISTLLM_DATA_RESIDENCY",
    "DISTLLM_DATA_RETENTION_DAYS",
    "DISTLLM_ENABLE_DOCS",
    "DISTLLM_CLUSTER_KEY",
    "api_key_store",
    "secret_manager",
]


def _read(path: Path) -> str:
    assert path.exists(), f"expected file missing: {path}"
    return path.read_text(encoding="utf-8")


def test_compliance_pack_files_exist():
    """All four evidence-pack files must be present."""
    required = [
        COMPLIANCE_DIR / "README.md",
        COMPLIANCE_DIR / "soc2_control_mapping.md",
        COMPLIANCE_DIR / "gdpr_evidence.md",
        COMPLIANCE_DIR / "export_controls_evidence.md",
    ]
    for f in required:
        assert f.exists(), f"compliance evidence file missing: {f}"
        assert f.stat().st_size > 0, f"compliance evidence file is empty: {f}"


def test_soc2_mapping_references_real_control_ids():
    """The SOC 2 mapping must reference control IDs that really exist in SECURITY_HARDENING.md.

    This is the core anti-fabrication check: we extract the control tokens the mapping claims,
    and assert a meaningful overlap with tokens actually present in the source document.
    """
    sec_text = _read(SEC_HARDENING)
    sec_lower = sec_text.lower()

    mapping_text = _read(COMPLIANCE_DIR / "soc2_control_mapping.md")

    # Confirm each KNOWN id is genuinely in the source doc (sanity for the test itself).
    missing_in_source = [cid for cid in KNOWN_REAL_CONTROL_IDS if cid.lower() not in sec_lower]
    assert not missing_in_source, (
        f"test fixture drift: these 'real' ids are not in {SEC_HARDENING.name}: "
        f"{missing_in_source}"
    )

    # How many real control ids does the mapping actually reference?
    referenced = [
        cid for cid in KNOWN_REAL_CONTROL_IDS if cid.lower() in mapping_text.lower()
    ]
    # Require a non-trivial overlap so the mapping cannot be an empty scaffold.
    assert len(referenced) >= 12, (
        f"SOC 2 mapping references too few real control ids from SECURITY_HARDENING.md "
        f"({len(referenced)} found, need >= 12): {referenced}"
    )


def test_soc2_mapping_has_status_column():
    """The mapping table must classify controls (implemented/scaffold/needed), not fake cert."""
    mapping_text = _read(COMPLIANCE_DIR / "soc2_control_mapping.md")
    for token in ("implemented", "scaffold", "needed"):
        assert token in mapping_text.lower(), (
            f"SOC 2 mapping must classify controls using status '{token}'"
        )
    # Explicitly NOT a certification.
    assert "not" in mapping_text.lower() and "certif" in mapping_text.lower(), (
        "SOC 2 mapping must explicitly state it is NOT a certification"
    )


def test_evidence_files_disclaim_certification():
    """Each evidence file must disclaim certification (honest scaffold)."""
    for name in ("README.md", "soc2_control_mapping.md", "gdpr_evidence.md",
                 "export_controls_evidence.md"):
        text = _read(COMPLIANCE_DIR / name)
        low = text.lower()
        assert "scaffold" in low, f"{name} must state it is a scaffold"
        assert "not" in low and "certif" in low, (
            f"{name} must explicitly state it is NOT a certification"
        )


def test_security_hardening_points_at_compliance_pack():
    """SECURITY_HARDENING.md SOC 2 section must point at compliance/ and disclaim cert."""
    sec_text = _read(SEC_HARDENING)
    low = sec_text.lower()
    assert "compliance/" in low, (
        "SECURITY_HARDENING.md must link to the compliance/ evidence pack"
    )
    # The disclaimer added near L290 must be present.
    assert "scaffold, not a certification" in low or (
        "scaffold" in low and "certif" in low
    ), "SECURITY_HARDENING.md SOC 2 note must state it is a scaffold, not a certification"
