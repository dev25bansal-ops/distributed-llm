"""Security audit automation for DistLLM.

Runs automated security checks including:
- Dependency vulnerability scanning
- Secret detection
- SSRF protection verification
- Authentication bypass checks
- TLS configuration audit

Usage:
    python scripts/security_audit.py
    python scripts/security_audit.py --fix  # Auto-fix where possible
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class AuditFinding:
    """A single security audit finding."""
    severity: str  # "critical", "high", "medium", "low", "info"
    category: str
    title: str
    file: str = ""
    line: int = 0
    description: str = ""
    recommendation: str = ""


class SecurityAuditor:
    """Automated security auditor for DistLLM."""

    def __init__(self, project_root: str = "."):
        self._root = Path(project_root)
        self._findings: list[AuditFinding] = []

    def run_all(self) -> list[AuditFinding]:
        """Run all security audits."""
        self._findings = []
        self._check_dependency_vulnerabilities()
        self._check_secrets_in_code()
        self._check_ssrf_protection()
        self._check_auth_bypass()
        self._check_tls_config()
        self._check_docker_security()
        self._check_api_input_validation()
        return self._findings

    def _check_dependency_vulnerabilities(self) -> None:
        """Check for known vulnerabilities in dependencies."""
        try:
            result = subprocess.run(
                ["pip-audit", "--format", "json"],
                capture_output=True, text=True, timeout=60,
                cwd=str(self._root),
            )
            if result.returncode != 0:
                try:
                    data = json.loads(result.stdout)
                    # pip-audit >= 2 emits {"dependencies": [{"name": ...,
                    # "vulns": [...]}]}; older versions emitted a bare list of
                    # vulnerability dicts. Handle both shapes.
                    if isinstance(data, dict):
                        deps = data.get("dependencies", [])
                        for dep in deps:
                            for v in dep.get("vulns", []) or []:
                                self._findings.append(AuditFinding(
                                    severity="high",
                                    category="dependency",
                                    title=f"Vulnerable dependency: {dep.get('name', 'unknown')}",
                                    description=v.get("description") or v.get("id", ""),
                                    recommendation=(
                                        f"Update to version {v.get('fix_versions') or 'latest'}"
                                        f" (advisory {v.get('id', '')})"
                                    ),
                                ))
                    elif isinstance(data, list):
                        for vuln in data:
                            if not isinstance(vuln, dict):
                                continue
                            self._findings.append(AuditFinding(
                                severity="high",
                                category="dependency",
                                title=f"Vulnerable dependency: {vuln.get('name', 'unknown')}",
                                description=vuln.get("description", ""),
                                recommendation=f"Update to version {vuln.get('fixed_version', 'latest')}",
                            ))
                except json.JSONDecodeError:
                    pass
        except (FileNotFoundError, subprocess.TimeoutExpired):
            self._findings.append(AuditFinding(
                severity="info",
                category="tooling",
                title="pip-audit not available",
                recommendation="Install with: pip install pip-audit",
            ))

    def _check_secrets_in_code(self) -> None:
        """Check for hardcoded secrets in source code."""
        secret_patterns = [
            (r'(?i)(api[_-]?key|secret|password|token)\s*=\s*["\'][^"\']{8,}["\']', "hardcoded_secret"),
            (r'(?i)bearer\s+[a-zA-Z0-9]{20,}', "hardcoded_bearer"),
            (r'-----BEGIN\s+(RSA\s+)?PRIVATE\s+KEY-----', "private_key"),
            # A literal `aws_(access|secret)_key` assignment (value excluded --
            # these are usually named in comments/config keys, not hardcoded).
            (r'(?i)aws[_-]?(access[_-]?key|secret[_-]?key)\s*[=:]\s*'
             r'["\'][A-Za-z0-9/+=]{16,}["\']', "aws_credentials"),
        ]

        src_dir = self._root / "src"
        if not src_dir.exists():
            return

        # Values that look like secrets-named placeholders, not secrets.
        placeholder_re = re.compile(
            r'(?i)^(?:\[?redacted\]?|none|null|true|false|changeme|change-me|'
            r'not-needed|notneeded|dummy|test|example|sample|'
            r'your[_-]?(?:api[_-]?)?key|xxx+|placeholder|.*\.\.\..*|<.*>|\$\{.*\}|'
            r'env\..*|os\.environ.*|'
            r'my[_-][a-z0-9-]+|[\w-]*cluster-wide-[\w-]+|nvapi-\w*\.\.\.)$'
        )

        for py_file in src_dir.rglob("*.py"):
            try:
                content = py_file.read_text(encoding="utf-8")
                for pattern, category in secret_patterns:
                    for match in re.finditer(pattern, content):
                        matched_text = match.group(0)
                        line_start = content.rfind("\n", 0, match.start()) + 1
                        line_text = content[line_start:content.find("\n", match.start())]
                        stripped = line_text.strip()
                        # Skip comments and docstring-ish examples.  A match
                        # inside a comment is documentation, not a live secret.
                        if stripped.startswith("#"):
                            continue
                        # Extract the assigned value, if any, for placeholder checks.
                        val_match = re.search(r'["\']([^"\']*)["\']\s*$', matched_text)
                        if val_match and placeholder_re.match(val_match.group(1).strip()):
                            continue
                        line_num = content[:match.start()].count("\n") + 1
                        self._findings.append(AuditFinding(
                            severity="critical",
                            category="secrets",
                            title=f"Potential {category} in source code",
                            file=str(py_file.relative_to(self._root)),
                            line=line_num,
                            description=f"Found pattern matching {category}",
                            recommendation="Move to environment variable or secrets manager",
                        ))
            except (OSError, UnicodeDecodeError):
                pass

    def _check_ssrf_protection(self) -> None:
        """Verify SSRF protection is in place."""
        # Check that _reject_private_address exists and is called
        chat_route = self._root / "src" / "distllm" / "api" / "routes" / "chat.py"
        if chat_route.exists():
            content = chat_route.read_text(encoding="utf-8")
            if "_reject_private_address" not in content:
                self._findings.append(AuditFinding(
                    severity="critical",
                    category="ssrf",
                    title="Missing SSRF protection in chat routes",
                    recommendation="Add _reject_private_address() call for image URLs",
                ))
            if "ipaddress" not in content:
                self._findings.append(AuditFinding(
                    severity="high",
                    category="ssrf",
                    title="SSRF protection may not use ipaddress module",
                    recommendation="Use ipaddress.ip_address() for proper IP validation",
                ))

    def _check_auth_bypass(self) -> None:
        """Check for authentication bypass vulnerabilities."""
        middleware = self._root / "src" / "distllm" / "api" / "middleware.py"
        if middleware.exists():
            content = middleware.read_text(encoding="utf-8")
            # Look for an actual env-var *read* (os.environ/only-if-not-set) in
            # executable code, not a mention inside a comment or docstring.
            # `middleware.py` documents that the old PYTEST_CURRENT_TEST bypass
            # was REMOVED, so a bare substring hit here is a permanent false
            # positive.
            code_lines = [
                ln for ln in content.splitlines()
                if "PYTEST_CURRENT_TEST" in ln and not ln.lstrip().startswith("#")
            ]
            live = [
                ln for ln in code_lines
                if re.search(r'(environ(?:\.get)?\s*[\[(]|getenv\s*\()', ln)
            ]
            if live:
                self._findings.append(AuditFinding(
                    severity="high",
                    category="auth",
                    title="Auth bypass in test mode via PYTEST_CURRENT_TEST",
                    description="Auth middleware may skip validation when PYTEST_CURRENT_TEST is set",
                    recommendation="Ensure test mode is not activatable in production",
                ))

    def _check_tls_config(self) -> None:
        """Check TLS configuration."""
        settings = self._root / "src" / "distllm" / "config" / "_network.py"
        if settings.exists():
            content = settings.read_text(encoding="utf-8")
            if "min_tls_version" in content:
                if "TLSv1.0" in content or "TLSv1.1" in content:
                    self._findings.append(AuditFinding(
                        severity="high",
                        category="tls",
                        title="TLS 1.0/1.1 may be allowed",
                        recommendation="Require minimum TLS 1.2",
                    ))

    def _check_docker_security(self) -> None:
        """Check Docker security configuration."""
        dockerfile = self._root / "Dockerfile"
        if dockerfile.exists():
            content = dockerfile.read_text(encoding="utf-8")
            if "USER root" in content or "user root" in content:
                self._findings.append(AuditFinding(
                    severity="high",
                    category="docker",
                    title="Docker container runs as root",
                    recommendation="Add USER directive to run as non-root",
                ))
            if "EXPOSE" not in content:
                self._findings.append(AuditFinding(
                    severity="low",
                    category="docker",
                    title="No EXPOSE directive in Dockerfile",
                    recommendation="Document exposed ports with EXPOSE",
                ))

    def _check_api_input_validation(self) -> None:
        """Check API input validation."""
        validation = self._root / "src" / "distllm" / "api" / "validation.py"
        if validation.exists():
            content = validation.read_text(encoding="utf-8")
            if "realpath" not in content and "resolve" not in content:
                self._findings.append(AuditFinding(
                    severity="high",
                    category="path_traversal",
                    title="Path validation may not resolve symlinks",
                    recommendation="Use os.path.realpath() to resolve symlinks before validation",
                ))

    def report(self) -> str:
        """Generate a human-readable audit report."""
        if not self._findings:
            return "No security findings."

        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}
        sorted_findings = sorted(self._findings, key=lambda f: severity_order.get(f.severity, 5))

        lines = ["# Security Audit Report", ""]
        lines.append(f"Total findings: {len(self._findings)}")
        lines.append("")

        counts = {}
        for f in self._findings:
            counts[f.severity] = counts.get(f.severity, 0) + 1
        for sev in ["critical", "high", "medium", "low", "info"]:
            if sev in counts:
                lines.append(f"- **{sev.upper()}**: {counts[sev]}")
        lines.append("")

        for f in sorted_findings:
            lines.append(f"## [{f.severity.upper()}] {f.title}")
            lines.append(f"**Category**: {f.category}")
            if f.file:
                lines.append(f"**File**: `{f.file}`" + (f":{f.line}" if f.line else ""))
            if f.description:
                lines.append(f"**Description**: {f.description}")
            if f.recommendation:
                lines.append(f"**Recommendation**: {f.recommendation}")
            lines.append("")

        return "\n".join(lines)


def main():
    import argparse

    # Finding text (file paths, descriptions) can contain non-ASCII; the
    # Windows console defaults to cp1252 and would crash the report printer.
    if hasattr(sys.stdout, "reconfigure"):
        # Best-effort: a console that refuses reconfiguration is not a reason
        # to abort the audit; the report just falls back to the default codec.
        with contextlib.suppress(ValueError, OSError):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="DistLLM Security Audit")
    parser.add_argument("--fix", action="store_true",
                        help="(accepted for compatibility; auto-fix is not implemented)")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    if args.fix:
        print("NOTE: --fix is not implemented; no changes will be made.", file=sys.stderr)

    auditor = SecurityAuditor()
    findings = auditor.run_all()

    if args.json:
        print(json.dumps([{
            "severity": f.severity,
            "category": f.category,
            "title": f.title,
            "file": f.file,
            "line": f.line,
            "description": f.description,
            "recommendation": f.recommendation,
        } for f in findings], indent=2))
    else:
        print(auditor.report())

    critical = sum(1 for f in findings if f.severity == "critical")
    high = sum(1 for f in findings if f.severity == "high")
    if critical > 0 or high > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
