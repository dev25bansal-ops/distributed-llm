"""Compliance evidence pack builder (N10 — certification scaffolding).

This module aggregates:

  * N4's point-in-time SOC 2 / ISO 27001 evidence (``distllm.core.compliance_evidence``)
  * doc-derived control mappings parsed from ``docs/GDPR.md`` and
    ``docs/EXPORT_CONTROLS.md``
  * a HIPAA control family mapped from the general security posture
    (``SECURITY.md`` + ``docs/SECURITY_HARDENING.md``), **flagged as UNVERIFIED
    because no HIPAA source document exists** — DistLLM is not a covered entity
    and holds no Business Associate Agreement.

into a single structured :class:`EvidencePack` that can be emitted as JSON and
Markdown for an auditor / certification exercise.

------------------------------------------------------------------------------
⚠️  HONEST DISCLAIMER — NOT A CERTIFICATION
------------------------------------------------------------------------------
This package automates *evidence-gathering and mapping only*. It is not an
auditor and does not certify the system against SOC 2, ISO 27001, GDPR, export
controls, or HIPAA. Where a control cannot be backed by an existing source
document it is either omitted or explicitly flagged ``unverified``. A ``pass``
from N4 means the introspected configuration matched the control's intent at a
point in time; an auditor must still independently review, sample, and attest.
------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# --------------------------------------------------------------------------- #
# Paths (repo-relative; overridable for testing)
# --------------------------------------------------------------------------- #
_THIS_FILE = Path(__file__).resolve()
# src/distllm/compliance/evidence_pack.py -> parents[3] == repo root
_DEFAULT_REPO_ROOT = _THIS_FILE.parents[3]
_DEFAULT_DOCS_DIR = _DEFAULT_REPO_ROOT / "docs"
_DEFAULT_COMPLIANCE_DIR = _DEFAULT_REPO_ROOT / "compliance"


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #
@dataclass
class ControlRef:
    """A single control reference in an evidence pack."""

    control_id: str
    framework: str  # SOC2 | ISO27001 | GDPR | EXPORT | HIPAA
    title: str
    source_doc: str  # e.g. "docs/GDPR.md"
    source_section: str  # e.g. "L39-43" or "### Data Minimization"
    status: str  # implemented | scaffold | documented | needed | unverified
    notes: str = ""


@dataclass
class EvidencePack:
    """Aggregated, auditor-facing compliance evidence pack."""

    generated_at: str
    soc2_controls: list[ControlRef] = field(default_factory=list)
    iso27001_controls: list[ControlRef] = field(default_factory=list)
    hipaa_controls: list[ControlRef] = field(default_factory=list)
    gdpr_controls: list[ControlRef] = field(default_factory=list)
    export_controls: list[ControlRef] = field(default_factory=list)
    audit_log_samples: list[str] = field(default_factory=list)
    pentest_summary: dict = field(default_factory=dict)
    caveats: list[str] = field(default_factory=list)
    source_report: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def to_markdown(self) -> str:
        return _render_markdown(self)


# --------------------------------------------------------------------------- #
# Doc parsing helpers (simple + robust regex over headings / keywords)
# --------------------------------------------------------------------------- #
_HEADING_RE = re.compile(r"^(#{2,4})\s+(.*?)\s*$", re.MULTILINE)


def _headings(text: str) -> list[tuple[int, str, str]]:
    """Return (line_no, level, title) for every Markdown heading in ``text``."""
    out: list[tuple[int, str, str]] = []
    for m in _HEADING_RE.finditer(text):
        line_no = text.count("\n", 0, m.start()) + 1
        out.append((line_no, m.group(1), m.group(2).strip()))
    return out


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _section_cite(line_no: int) -> str:
    return f"L{line_no}"


# --- GDPR (docs/GDPR.md) --------------------------------------------------- #
# Honest, standard GDPR article references for the *principles* the doc states.
_GDPR_ARTICLE_HINTS: list[tuple[str, str]] = [
    ("purpose limitation", "Art. 5(1)(b)"),
    ("data minimization", "Art. 5(1)(c)"),
    ("storage limitation", "Art. 5(1)(e)"),
    ("integrity and confidentiality", "Art. 32"),
    ("right of access", "Art. 15"),
    ("right to erasure", "Art. 17"),
    ("right to data portability", "Art. 20"),
    ("lawful basis", "Art. 6"),
    ("data subject rights", "Art. 12-22"),
    ("federated", "Art. 28 / Art. 32"),
    ("audit", "Art. 5(2) / Art. 30"),
    ("encryption", "Art. 32"),
    ("retention", "Art. 5(1)(e)"),
]


def _map_gdpr_article(title: str) -> Optional[str]:
    low = title.lower()
    for needle, article in _GDPR_ARTICLE_HINTS:
        if needle in low:
            return article
    return None


def parse_gdpr_controls(docs_dir: Path = _DEFAULT_DOCS_DIR) -> list[ControlRef]:
    """Parse ``docs/GDPR.md`` headings into GDPR control references.

    Controls are *derived* from the document (no fabricated control text): each
    heading is mapped to its standard GDPR article by keyword. If the doc is
    missing, an empty list is returned and a caveat is added by the caller.
    """
    path = docs_dir / "GDPR.md"
    text = _read_text(path)
    out: list[ControlRef] = []
    if not text:
        return out

    for line_no, _level, title in _headings(text):
        article = _map_gdpr_article(title)
        if article is None:
            continue
        cid = f"GDPR {article}"
        # The doc describes intended behaviour; implementation status is a
        # draft (see compliance/gdpr_evidence.md). We mark it 'documented'.
        out.append(
            ControlRef(
                control_id=cid,
                framework="GDPR",
                title=title,
                source_doc="docs/GDPR.md",
                source_section=_section_cite(line_no),
                status="documented",
                notes=(
                    "Principle documented in docs/GDPR.md. Implementation status "
                    "is a draft (compliance/gdpr_evidence.md); not certified."
                ),
            )
        )
    # De-duplicate by control_id, keeping the earliest section reference.
    seen: dict[str, ControlRef] = {}
    for c in out:
        if c.control_id not in seen:
            seen[c.control_id] = c
    return list(seen.values())


# --- Export Controls (docs/EXPORT_CONTROLS.md) ----------------------------- #
_EXPORT_REGIME_HINTS: list[tuple[str, str]] = [
    ("wassenaar", "Wassenaar Arrangement"),
    ("nuclear suppliers", "Nuclear Suppliers Group"),
    ("ai act", "EU AI Act (EU) 2024/1689"),
    ("export administration regulations", "US EAR (BIS)"),
    ("entity list", "US EAR — Entity List screening"),
    ("end-use", "End-use / end-user restrictions"),
    ("dual-use", "EU Dual-Use Reg (EU) 2021/821"),
    ("know your customer", "KYC / sanctions screening"),
    ("end-use monitoring", "End-use monitoring"),
    ("technical data", "Technical-data / deemed-export control"),
    ("self-hosted", "Local export-control law (self-hosted)"),
    ("model-specific", "Model-specific license gating"),
]


def _map_export_regime(title: str) -> Optional[str]:
    low = title.lower()
    for needle, regime in _EXPORT_REGIME_HINTS:
        if needle in low:
            return regime
    return None


def parse_export_controls(docs_dir: Path = _DEFAULT_DOCS_DIR) -> list[ControlRef]:
    """Parse ``docs/EXPORT_CONTROLS.md`` into export-control references.

    Scans the document (headings *and* bullet lines) for regime keywords so that
    both the ``## Applicable Regulations`` bullets (US EAR, EU AI Act, Wassenaar…)
    and the ``## User Responsibilities`` headings (KYC, end-use…) are captured.
    Controls are derived from the document — no fabricated control text.
    """
    path = docs_dir / "EXPORT_CONTROLS.md"
    text = _read_text(path)
    out: list[ControlRef] = []
    if not text:
        return out

    # Map each line (heading or bullet) to a regime if its text contains a hint.
    for idx, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip().lstrip("#").strip().lstrip("*-").strip()
        if not line:
            continue
        regime = _map_export_regime(line)
        if regime is None:
            continue
        out.append(
            ControlRef(
                control_id=f"EXPORT: {regime}",
                framework="EXPORT",
                title=line,
                source_doc="docs/EXPORT_CONTROLS.md",
                source_section=_section_cite(idx),
                status="documented",
                notes=(
                    "Obligation documented in docs/EXPORT_CONTROLS.md. No export "
                    "licence/ECCN determination exists (compliance/"
                    "export_controls_evidence.md); not certified."
                ),
            )
        )
    seen: dict[str, ControlRef] = {}
    for c in out:
        if c.control_id not in seen:
            seen[c.control_id] = c
    return list(seen.values())


# --- HIPAA (NO source doc — flag everything unverified) -------------------- #
# HIPAA Security Rule provisions we *would* map. None are backed by a HIPAA
# source document, so every entry is status='unverified' and the pack carries a
# caveat. We still surface them (derived from the general security posture) so
# an auditor can see the intended mapping — but never as an attestation.
_HIPAA_PROVISIONS: list[tuple[str, str, str, str]] = [
    # (provision_id, title, posture_keyword(s) searched in SECURITY.md, note)
    ("164.308(a)(1)(i)", "Risk Analysis", "vulnerab",
     "Risk analysis / management program."),
    ("164.308(a)(1)(ii)(D)", "Information System Activity Review",
     "audit|monitor|log", "Activity review via audit logging."),
    ("164.310(a)(1)", "Facility Access Controls", "physical",
     "Logical/physical access controls."),
    ("164.312(a)(1)", "Access Control", "authentic|authoriz|access",
     "Authentication/authorization of access."),
    ("164.312(b)", "Audit Controls", "audit|log",
     "Audit controls over activity in information systems."),
    ("164.312(c)(1)", "Integrity", "integrity|hash|sha",
     "Mechanism to authenticate ePHI (e.g. hashing)."),
    ("164.312(d)", "Person or Entity Authentication", "authentic",
     "Verify identity of persons/entities."),
    ("164.312(e)(1)", "Transmission Security", "tls|encrypt|transit",
     "Guard against unauthorized disclosure of ePHI in transit."),
    ("164.312(e)(2)(ii)", "Encryption of ePHI in Transit", "tls|encrypt",
     "Encryption of ePHI in transit."),
]


def parse_hipaa_controls(
    security_doc_paths: Optional[list[Path]] = None,
) -> tuple[list[ControlRef], list[str]]:
    """Map HIPAA Security Rule provisions from the general security posture.

    Returns ``(controls, support_notes)``. Every control is status='unverified'
    because there is **no HIPAA source document** (HIPAA.md does not exist) and
    DistLLM is not a covered entity. ``support_notes`` records, per provision,
    whether a *tangential* signal was found in SECURITY.md (still not an
    attestation).
    """
    if security_doc_paths is None:
        security_doc_paths = [
            _DEFAULT_REPO_ROOT / "SECURITY.md",
            _DEFAULT_REPO_ROOT / "docs" / "SECURITY_HARDENING.md",
        ]
    combined = "\n".join(_read_text(p) for p in security_doc_paths).lower()
    out: list[ControlRef] = []
    notes: list[str] = []
    for prov_id, title, kw, note in _HIPAA_PROVISIONS:
        hit = any(re.search(k, combined) for k in kw.split("|")) if kw else False
        support = (
            f"Tangential signal found in security posture ({note})."
            if hit
            else f"No supporting signal in security docs ({note})."
        )
        out.append(
            ControlRef(
                control_id=f"HIPAA {prov_id}",
                framework="HIPAA",
                title=title,
                source_doc="(none — HIPAA.md absent)",
                source_section="derived from SECURITY.md posture",
                status="unverified",
                notes=(
                    "UNVERIFIED / scaffold: no HIPAA source document exists and "
                    "DistLLM is not a covered entity / holds no BAA. " + support
                ),
            )
        )
        notes.append(f"{prov_id} ({title}): {'signal' if hit else 'no signal'}")
    return out, notes


# --------------------------------------------------------------------------- #
# Audit-log samples + pen-test summary (real excerpts, not invented)
# --------------------------------------------------------------------------- #
def _grep_excerpts(
    paths: list[Path], keywords: list[str], max_per_file: int = 3, max_total: int = 6
) -> list[str]:
    """Return verbatim, non-empty lines containing any keyword (case-insensitive)."""
    out: list[str] = []
    kw = [k.lower() for k in keywords]
    for p in paths:
        text = _read_text(p)
        if not text:
            continue
        count = 0
        for line in text.splitlines():
            s = line.strip()
            if not s:
                continue
            if any(k in s.lower() for k in kw):
                out.append(f"{p.name}: {s}")
                count += 1
                if count >= max_per_file:
                    break
            if len(out) >= max_total:
                break
        if len(out) >= max_total:
            break
    return out


def collect_audit_log_samples(
    compliance_dir: Path = _DEFAULT_COMPLIANCE_DIR,
    docs_dir: Path = _DEFAULT_DOCS_DIR,
    repo_root: Path = _DEFAULT_REPO_ROOT,
) -> list[str]:
    """Pull representative, *verbatim* audit/security excerpts from real docs.

    These are real lines from the existing compliance artifacts and SECURITY.md,
    not invented sample log entries.
    """
    candidates = [
        compliance_dir / "gdpr_evidence.md",
        compliance_dir / "export_controls_evidence.md",
        compliance_dir / "soc2_control_mapping.md",
        docs_dir / "GDPR.md",
        repo_root / "SECURITY.md",
    ]
    excerpts = _grep_excerpts(
        candidates,
        keywords=[
            "audit log",
            "audit logs",
            "timestamp",
            "client ip",
            "api key id",
            "tls",
            "authentication",
            "all api access is logged",
        ],
        max_per_file=2,
        max_total=8,
    )
    if not excerpts:
        excerpts = [
            "NOTE: no verbatim audit-log excerpt found in source docs; "
            "audit logging is documented as configurable (DISTLLM_AUDIT_LOGGING)."
        ]
    return excerpts


def collect_pentest_summary(
    security_doc: Path = _DEFAULT_REPO_ROOT / "SECURITY.md",
    soc2_map: Path = _DEFAULT_COMPLIANCE_DIR / "soc2_control_mapping.md",
) -> dict:
    """Build a *honest* pen-test / vuln-disclosure summary from real docs.

    There is no completed penetration-test report in the repo. This summary is
    derived from the vulnerability-disclosure program in SECURITY.md (real SLAs)
    and flags the absence of an actual pen-test artifact.
    """
    text = _read_text(security_doc)
    summary: dict[str, Any] = {
        "source": "SECURITY.md (vulnerability disclosure policy)",
        "actual_pentest_report": False,
        "note": (
            "No completed penetration-test report is present in the repository. "
            "This summary reflects the vulnerability-disclosure program, which is "
            "a control input but NOT a substitute for an independent pen-test."
        ),
        "disclosure_sla": {},
        "in_scope": [],
        "out_of_scope": [],
    }
    if "48 hours" in text:
        summary["disclosure_sla"]["acknowledge"] = "48 hours"
    if "5 business days" in text:
        summary["disclosure_sla"]["triage"] = "5 business days"
    for label, needle in (
        ("Critical", "Critical: 7 days"),
        ("High", "High: 14 days"),
        ("Medium", "Medium: 30 days"),
        ("Low", "Low: 60 days"),
    ):
        if needle in text:
            summary["disclosure_sla"]["fix_" + label.lower()] = needle.split(": ")[1]
    # In-scope bullets (verbatim-ish from SECURITY.md)
    for line in text.splitlines():
        s = line.strip().lstrip("* ").lstrip("- ").strip()
        low = s.lower()
        if "authentication/authorization" in low:
            summary["in_scope"].append("Authentication/authorization bypasses")
        elif "remote code execution" in low:
            summary["in_scope"].append("Remote code execution")
        elif "cryptographic weaknesses" in low:
            summary["in_scope"].append("Cryptographic weaknesses")
        elif "denial of service" in low and "volumetric" not in low:
            summary["in_scope"].append("Denial of service (resource exhaustion)")
        elif "social engineering" in low:
            summary["out_of_scope"].append("Social engineering")
        elif "physical attacks" in low:
            summary["out_of_scope"].append("Physical attacks")
    return summary


# --------------------------------------------------------------------------- #
# N4 integration: split a ComplianceReport into SOC2 / ISO27001 control refs
# --------------------------------------------------------------------------- #
def _report_to_control_refs(report: dict) -> tuple[list[ControlRef], list[ControlRef]]:
    soc2: list[ControlRef] = []
    iso: list[ControlRef] = []
    for domain, entry in report.get("controls", {}).items():
        status = entry.get("status", "unknown")
        for cid in entry.get("control_ids", []):
            if cid.startswith("ISO27001:"):
                iso.append(
                    ControlRef(
                        control_id=cid,
                        framework="ISO27001",
                        title=f"{domain} ({entry.get('notes','')[:60]})",
                        source_doc="distllm.core.compliance_evidence (N4)",
                        source_section=domain,
                        status=status,
                        notes=entry.get("notes", ""),
                    )
                )
            else:
                soc2.append(
                    ControlRef(
                        control_id=cid,
                        framework="SOC2",
                        title=f"{domain} ({entry.get('notes','')[:60]})",
                        source_doc="distllm.core.compliance_evidence (N4)",
                        source_section=domain,
                        status=status,
                        notes=entry.get("notes", ""),
                    )
                )
    return soc2, iso


# --------------------------------------------------------------------------- #
# Top-level builder
# --------------------------------------------------------------------------- #
def build_evidence_pack(
    report: Optional[dict] = None,
    docs_dir: Path = _DEFAULT_DOCS_DIR,
    compliance_dir: Path = _DEFAULT_COMPLIANCE_DIR,
    repo_root: Path = _DEFAULT_REPO_ROOT,
    now: Any = None,
    security_doc_paths: Optional[list[Path]] = None,
) -> EvidencePack:
    """Assemble the full EvidencePack.

    Reuses N4's ``collect_all`` (never modified) for SOC 2 / ISO 27001 evidence,
    parses ``docs/GDPR.md`` and ``docs/EXPORT_CONTROLS.md`` for their control
    families, and maps HIPAA from the security posture (flagged unverified).

    If ``report`` is None, N4's ``collect_all`` is invoked with ``now``.
    """
    if now is None:
        now = datetime.now(timezone.utc)
    generated_at = now.isoformat() if isinstance(now, datetime) else str(now)

    # 1) N4 SOC2 / ISO27001 (never modify core/compliance_evidence.py)
    if report is None:
        from distllm.core.compliance_evidence import collect_all

        report = collect_all(bundle={"now": now})
    soc2_controls, iso27001_controls = _report_to_control_refs(report)

    # 2) GDPR + Export Controls (doc-derived)
    gdpr_controls = parse_gdpr_controls(docs_dir)
    export_controls = parse_export_controls(docs_dir)

    # 3) HIPAA (no source doc -> unverified)
    hipaa_controls, hipaa_support = parse_hipaa_controls(security_doc_paths)

    # 4) Audit-log samples + pen-test summary (real excerpts)
    audit_log_samples = collect_audit_log_samples(compliance_dir, docs_dir, repo_root)
    pentest_summary = collect_pentest_summary()

    # 5) Caveats (honest gaps)
    caveats: list[str] = []
    if not gdpr_controls:
        caveats.append("GDPR: docs/GDPR.md could not be read; no GDPR controls mapped.")
    else:
        caveats.append(
            "GDPR: controls are DOCUMENTED in docs/GDPR.md only; DistLLM is not "
            "GDPR-certified and has produced no Art. 30 record / DPIA "
            "(see compliance/gdpr_evidence.md)."
        )
    if not export_controls:
        caveats.append(
            "Export Controls: docs/EXPORT_CONTROLS.md could not be read; no export "
            "controls mapped."
        )
    else:
        caveats.append(
            "Export Controls: obligations are DOCUMENTED only; no ECCN "
            "classification / KYC program exists (see compliance/"
            "export_controls_evidence.md)."
        )
    caveats.append(
        "HIPAA: NO HIPAA source document (HIPAA.md) exists. All HIPAA controls are "
        "UNVERIFIED scaffolds mapped from the general security posture. DistLLM is "
        "NOT a covered entity and holds NO Business Associate Agreement. This is not "
        "a HIPAA attestation. Support signals: " + "; ".join(hipaa_support) + "."
    )
    caveats.append(
        "SOC 2 / ISO 27001: N4 evidence is point-in-time configuration introspection "
        "only (NOT a Type II operating-evidence period). An auditor must independently "
        "review, sample, and attest."
    )
    caveats.append(
        "Pen-test: no completed penetration-test report is present; the pen-test "
        "summary reflects the SECURITY.md vulnerability-disclosure program only."
    )

    return EvidencePack(
        generated_at=generated_at,
        soc2_controls=soc2_controls,
        iso27001_controls=iso27001_controls,
        hipaa_controls=hipaa_controls,
        gdpr_controls=gdpr_controls,
        export_controls=export_controls,
        audit_log_samples=audit_log_samples,
        pentest_summary=pentest_summary,
        caveats=caveats,
        source_report=report,
    )


# --------------------------------------------------------------------------- #
# Emitters
# --------------------------------------------------------------------------- #
def emit_json(pack: EvidencePack, path: str | Path) -> str:
    """Write the pack as JSON. Returns the written path as a string."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(pack.to_json(), encoding="utf-8")
    return str(p)


def emit_markdown(pack: EvidencePack, path: str | Path) -> str:
    """Write the pack as Markdown. Returns the written path as a string."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(pack.to_markdown(), encoding="utf-8")
    return str(p)


def write_evidence_pack(
    pack: EvidencePack,
    output_dir: str | Path = "compliance",
    basename: str = "N10_EvidencePack",
) -> list[str]:
    """Write the pack as both JSON and Markdown. Returns the written paths."""
    out = Path(output_dir)
    written = [
        emit_json(pack, out / f"{basename}.json"),
        emit_markdown(pack, out / f"{basename}.md"),
    ]
    return written


# --------------------------------------------------------------------------- #
# Markdown rendering
# --------------------------------------------------------------------------- #
def _render_markdown(pack: EvidencePack) -> str:
    lines: list[str] = [
        "# Compliance & Certification Evidence Pack (N10)",
        "",
        f"- **Generated at:** {pack.generated_at}",
        "- **Frameworks:** SOC 2 (CC/A/C/PI), ISO 27001 Annex A, GDPR, Export "
        "Controls, HIPAA (scaffold)",
        "",
        "> ⚠️ **NOT A CERTIFICATION.** This document automates evidence-gathering "
        "and control mapping only. An independent auditor must review, sample, and "
        "attest. HIPAA controls are UNVERIFIED scaffolds (no HIPAA source doc).",
        "",
        "## Summary of control families",
        "",
        f"- SOC 2 controls (N4): **{len(pack.soc2_controls)}**",
        f"- ISO 27001 controls (N4): **{len(pack.iso27001_controls)}**",
        f"- GDPR controls (docs/GDPR.md): **{len(pack.gdpr_controls)}**",
        f"- Export controls (docs/EXPORT_CONTROLS.md): **{len(pack.export_controls)}**",
        f"- HIPAA controls (UNVERIFIED scaffold): **{len(pack.hipaa_controls)}**",
        "",
    ]

    def _section(title: str, controls: list[ControlRef]):
        lines.append(f"## {title}")
        lines.append("")
        if not controls:
            lines.append("_No controls mapped._")
            lines.append("")
            return
        lines.append("| Control ID | Status | Source | Section | Notes |")
        lines.append("|-----------|--------|--------|---------|-------|")
        for c in controls:
            note = c.notes.replace("\n", " ").replace("|", "\\|")
            lines.append(
                f"| {c.control_id} | {c.status} | {c.source_doc} | "
                f"{c.source_section} | {note} |"
            )
        lines.append("")

    _section("SOC 2 Controls (from N4 evidence)", pack.soc2_controls)
    _section("ISO 27001 Annex A Controls (from N4 evidence)", pack.iso27001_controls)
    _section("GDPR Controls (docs/GDPR.md)", pack.gdpr_controls)
    _section("Export Controls (docs/EXPORT_CONTROLS.md)", pack.export_controls)
    _section("HIPAA Controls (UNVERIFIED — no source doc)", pack.hipaa_controls)

    lines.append("## Audit-log samples (verbatim excerpts)")
    lines.append("")
    for s in pack.audit_log_samples:
        lines.append(f"- `{s}`")
    lines.append("")

    lines.append("## Pen-test / vulnerability-disclosure summary")
    lines.append("")
    ps = pack.pentest_summary
    lines.append(f"- Source: {ps.get('source')}")
    lines.append(f"- Actual pen-test report present: {ps.get('actual_pentest_report')}")
    lines.append(f"- Note: {ps.get('note')}")
    if ps.get("disclosure_sla"):
        lines.append("- Disclosure SLA:")
        for k, v in ps["disclosure_sla"].items():
            lines.append(f"  - {k}: {v}")
    if ps.get("in_scope"):
        lines.append("- In scope: " + "; ".join(ps["in_scope"]))
    if ps.get("out_of_scope"):
        lines.append("- Out of scope: " + "; ".join(ps["out_of_scope"]))
    lines.append("")

    lines.append("## Caveats")
    lines.append("")
    for cav in pack.caveats:
        lines.append(f"- ⚠️ {cav}")
    lines.append("")
    return "\n".join(lines)


__all__ = [
    "ControlRef",
    "EvidencePack",
    "build_evidence_pack",
    "parse_gdpr_controls",
    "parse_export_controls",
    "parse_hipaa_controls",
    "collect_audit_log_samples",
    "collect_pentest_summary",
    "emit_json",
    "emit_markdown",
    "write_evidence_pack",
]
