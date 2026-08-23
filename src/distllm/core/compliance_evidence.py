"""Automated SOC 2 / ISO 27001 evidence collector (N4 — API Cat5).

This module *introspects live configuration and runtime state* and emits
structured, machine-readable **evidence** for a set of SOC 2 Trust Services
Criteria (CC / A / C / PI) and ISO/IEC 27001:2022 Annex A controls. It covers:

  * CORS policy        (``collect_cors_evidence``)        -> CC6.6
  * TLS posture        (``collect_tls_evidence``)         -> CC6.1/C1.1/C1.2
  * Per-tenant quota    (``collect_quota_evidence``)       -> A1.1/PI1.1
  * Key/cert rotation  (``collect_key_rotation_evidence``) -> CC6.1/CC6.2/C1.1

``collect_all`` aggregates every collector into a ``ComplianceReport`` (a plain,
JSON-serializable ``dict``) keyed by control domain, each entry carrying the
mapped SOC 2 CC and ISO 27001 Annex A control IDs, a ``pass``/``fail``/``warn``
status, the raw evidence, and a collection timestamp.

------------------------------------------------------------------------------
⚠️  HONEST DISCLAIMER — NOT A CERTIFICATION
------------------------------------------------------------------------------
This module **automates evidence-gathering only**. It is *not* an auditor and
does not certify the system against SOC 2 or ISO 27001. It reads whatever
configuration/state it is pointed at and reports what it finds. A ``pass`` here
means "the introspected configuration matches the control's intent at this point
in time" — it is a point-in-time snapshot, not continuous operating evidence
(the "Type II" requirement), and an auditor must still review, sample, and
independently attest. Treat the output as input to an audit, never as the audit
itself.
------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

# --- Control mapping (reuses compliance/soc2_control_mapping.md) -------------
# Each evidence domain maps to SOC 2 TSC control IDs and ISO 27001:2022 Annex A
# control IDs. These are the same controls referenced by the M17 scaffold.
CONTROL_MAP: dict[str, dict[str, list[str]]] = {
    "CORS": {
        "soc2": ["CC6.6"],
        "iso27001": ["A.8.26", "A.5.18"],  # app security / access rights
    },
    "TLS": {
        "soc2": ["CC6.1", "C1.1", "C1.2"],
        "iso27001": ["A.8.24", "A.8.20"],  # use of cryptography / network security
    },
    "QUOTA": {
        "soc2": ["A1.1", "PI1.1"],
        "iso27001": ["A.8.6", "A.5.29"],  # capacity management / continuity testing
    },
    "KEY_ROTATION": {
        "soc2": ["CC6.1", "CC6.2", "C1.1"],
        "iso27001": ["A.8.24", "A.5.18"],  # cryptography / access rights
    },
}


def _control_ids(domain: str) -> list[str]:
    """Flatten the per-domain control map into a single list of IDs."""
    m = CONTROL_MAP.get(domain, {})
    ids: list[str] = list(m.get("soc2", []))
    ids += [f"ISO27001:{c}" for c in m.get("iso27001", [])]
    return ids


# --- small helpers for reading injected/real config objects -----------------
def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Read ``key`` from a dict, mapping, or attribute-bearing object."""
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    try:
        return getattr(obj, key, default)
    except Exception:
        return default


def _now_dt(now: Any) -> datetime:
    """Normalize ``now`` into a timezone-aware UTC datetime."""
    if now is None:
        return datetime.now(timezone.utc)
    if isinstance(now, datetime):
        return now if now.tzinfo else now.replace(tzinfo=timezone.utc)
    if isinstance(now, str):
        return datetime.fromisoformat(now)
    raise TypeError(f"unsupported 'now' type: {type(now)!r}")


def _now_iso(value: Any) -> Optional[str]:
    """Serialize a datetime/str/None to an ISO-8601 string (or None)."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _parse_dt(value: Any) -> Optional[datetime]:
    """Best-effort parse of a datetime / ISO string into a UTC datetime."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str):
        try:
            dt = datetime.fromisoformat(value)
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except ValueError:
            return None
    return None


def _valid_origin_scheme(origin: str) -> bool:
    return origin.startswith(
        ("http://", "https://", "chrome-extension://", "moz-extension://")
    )


# --- individual collectors --------------------------------------------------
def collect_cors_evidence(cors_config: Any, now: Any = None) -> dict:
    """Introspect the CORS allowlist and flag insecure configurations.

    Reads ``cors_origins`` (comma-separated string) and ``allow_credentials``
    from the injected/real config (e.g. ``CoordinatorSettings`` /
    ``config/_network.py``). A wildcard ``'*'`` origin is always flagged FAIL,
    and ``'*'`` *with* credentials is flagged FAIL with an explicit
    credential-leak reason.
    """
    origins_raw = _get(cors_config, "cors_origins", "") or ""
    allow_creds = bool(_get(cors_config, "allow_credentials", False))
    origins = [o.strip() for o in str(origins_raw).split(",") if o.strip()]
    wildcard = "*" in origins

    if not origins:
        status = "warn"
        reason = (
            "No CORS allowlist configured (deny-by-default; no cross-origin "
            "access). This is fail-closed but means CORS is unconfigured."
        )
    elif wildcard and allow_creds:
        status = "fail"
        reason = (
            "INSECURE: wildcard origin '*' combined with allow_credentials=True "
            "would leak credentials to ANY website. Forbidden by "
            "config/_network.py (CORSError)."
        )
    elif wildcard:
        status = "fail"
        reason = (
            "INSECURE: wildcard origin '*' allows any website to call the API "
            "(CORS allow-all). Rejected by config/_network.py policy."
        )
    elif any(not _valid_origin_scheme(o) for o in origins):
        status = "fail"
        reason = "One or more CORS origins are not valid URL schemes."
    else:
        status = "pass"
        reason = (
            f"{len(origins)} explicit CORS origin(s) configured; "
            f"allow_credentials={allow_creds}."
        )

    return {
        "control": "CORS",
        "control_ids": _control_ids("CORS"),
        "status": status,
        "evidence": {
            "cors_origins": origins,
            "allow_credentials": allow_creds,
            "wildcard": wildcard,
        },
        "collected_at": _now_iso(_now_dt(now)),
        "notes": reason,
    }


def collect_tls_evidence(
    tls_config: Any,
    cert_expiry: Any = None,
    now: Any = None,
) -> dict:
    """Introspect TLS posture: enablement, min version, cipher policy, mTLS,
    and certificate expiry (from ``cert_rotation`` state).
    """
    enabled = bool(_get(tls_config, "enabled", False))
    min_tls = _get(tls_config, "min_tls_version", None)
    cipher = _get(tls_config, "cipher_policy", None)
    require_client_cert = bool(_get(tls_config, "require_client_cert", False))
    ca_cert = _get(tls_config, "ca_cert_file", None)
    mtls = bool(require_client_cert and ca_cert)

    expiry_dt = _parse_dt(cert_expiry)
    days_remaining: Optional[int] = None
    if expiry_dt is not None:
        days_remaining = (expiry_dt - _now_dt(now)).days

    status = "pass"
    reasons: list[str] = []

    if not enabled:
        status = "fail"
        reasons.append(
            "TLS is DISABLED on the transport layer — no encryption in transit."
        )
    else:
        if min_tls is None:
            status = "warn"
            reasons.append(
                "min_tls_version not enforced at the application layer "
                "(relies on system/proxy TLS policy — per config/_network.py note)."
            )
        else:
            try:
                v = float(min_tls)
            except (TypeError, ValueError):
                v = None
            if v is None:
                status = "warn"
                reasons.append(f"Unparsable min_tls_version={min_tls!r}.")
            elif v < 1.2:
                status = "fail"
                reasons.append(f"TLS version {min_tls} is below the 1.2 minimum.")
            elif v < 1.3:
                status = "warn"
                reasons.append("TLS 1.2 in use; TLS 1.3 preferred.")
        if days_remaining is None:
            status = "warn"
            reasons.append(
                "Certificate expiry could not be determined from cert_rotation state."
            )
        elif days_remaining < 0:
            status = "fail"
            reasons.append("Certificate is EXPIRED.")
        elif days_remaining <= 30:
            status = "warn" if status == "pass" else status
            reasons.append(
                f"Certificate expires in {days_remaining} days (<=30d renew window)."
            )
    if mtls:
        reasons.append("Mutual TLS (require_client_cert + CA) enabled.")

    return {
        "control": "TLS",
        "control_ids": _control_ids("TLS"),
        "status": status,
        "evidence": {
            "enabled": enabled,
            "min_tls_version": min_tls,
            "cipher_policy": cipher,
            "require_client_cert": require_client_cert,
            "mtls": mtls,
            "cert_expiry": _now_iso(expiry_dt),
            "days_remaining": days_remaining,
        },
        "collected_at": _now_iso(_now_dt(now)),
        "notes": " ".join(reasons) or "TLS configuration acceptable.",
    }


def collect_quota_evidence(quota_config: Any, now: Any = None) -> dict:
    """Introspect whether per-tenant quota / rate-limiting is enabled."""
    enabled = bool(_get(quota_config, "enabled", False))
    per_tenant = _get(quota_config, "per_tenant", None)
    default_rpm = _get(quota_config, "default_rpm", None)

    if not enabled:
        status = "fail"
        reason = (
            "Per-tenant quota/rate-limiting is DISABLED — no protection against "
            "abuse/DoS and no per-tenant fairness."
        )
    elif per_tenant is False:
        status = "warn"
        reason = (
            "Rate limiting enabled but NOT per-tenant; only global limits apply."
        )
    else:
        status = "pass"
        reason = "Per-tenant quota/rate-limiting enabled."

    return {
        "control": "QUOTA",
        "control_ids": _control_ids("QUOTA"),
        "status": status,
        "evidence": {
            "enabled": enabled,
            "per_tenant": per_tenant,
            "default_rpm": default_rpm,
        },
        "collected_at": _now_iso(_now_dt(now)),
        "notes": reason,
    }


def collect_key_rotation_evidence(rotation_state: Any, now: Any = None) -> dict:
    """Introspect key/cert rotation recency.

    Reads ``last_rotation_ts`` (datetime/ISO) and ``rotation_interval_days``
    from the injected state. Flags overdue rotation as FAIL.
    """
    now_dt = _now_dt(now)
    last_ts = _parse_dt(_get(rotation_state, "last_rotation_ts", None))
    interval_days = _get(rotation_state, "rotation_interval_days", None)
    age_days: Optional[int] = None

    if last_ts is None:
        status = "fail"
        reason = (
            "No key/cert rotation timestamp available — rotation history unknown."
        )
    elif interval_days is None:
        status = "warn"
        age_days = (now_dt - last_ts).days
        reason = (
            f"Last rotation {age_days}d ago, but rotation interval is not "
            "configured; cannot assess overdue."
        )
    else:
        age_days = (now_dt - last_ts).days
        if age_days > interval_days:
            status = "fail"
            reason = (
                f"Key rotation OVERDUE: last rotation {age_days}d ago exceeds "
                f"interval {interval_days}d."
            )
        elif age_days > 0.8 * interval_days:
            status = "warn"
            reason = (
                f"Key rotation due soon: {age_days}d since last, interval "
                f"{interval_days}d."
            )
        else:
            status = "pass"
            reason = (
                f"Key rotation current: {age_days}d since last, interval "
                f"{interval_days}d."
            )

    return {
        "control": "KEY_ROTATION",
        "control_ids": _control_ids("KEY_ROTATION"),
        "status": status,
        "evidence": {
            "last_rotation_ts": _now_iso(last_ts),
            "rotation_interval_days": interval_days,
            "age_days": age_days,
        },
        "collected_at": _now_iso(now_dt),
        "notes": reason,
    }


# --- aggregate --------------------------------------------------------------
def bundle_from_settings() -> dict:
    """Best-effort build of an evidence bundle from the *real* config.

    Pulls CoordinatorSettings / TLSSettings / RateLimitSettings defaults and any
    available ``CertificateRotator`` state. Returns a dict suitable for
    ``collect_all``. Wrapped in try/except so a missing dependency never breaks
    import — callers that need determinism should inject the bundle explicitly.
    """
    bundle: dict[str, Any] = {}
    try:
        from distllm.config.settings import (
            CoordinatorSettings,
            RateLimitSettings,
            TLSSettings,
        )

        bundle["cors"] = CoordinatorSettings()
        bundle["tls"] = TLSSettings()
        bundle["quota"] = RateLimitSettings()
    except Exception:
        pass
    try:
        from distllm.core.cert_rotation import CertificateRotator

        rotator = CertificateRotator()
        info = rotator.check_certificate()
        if info.not_after is not None:
            bundle["cert_expiry"] = info.not_after
    except Exception:
        pass
    return bundle


def collect_all(bundle: Optional[dict] = None, now: Any = None) -> dict:
    """Aggregate all collectors into a serializable ``ComplianceReport``.

    ``bundle`` may contain keys: ``cors``, ``tls``, ``cert_expiry``, ``quota``,
    ``key_rotation``, ``now``. If ``bundle`` is None, an attempt is made to build
    it from the live configuration (see ``bundle_from_settings``).
    """
    if bundle is None:
        bundle = bundle_from_settings()
    now = bundle.get("now", now)

    cors = collect_cors_evidence(bundle.get("cors"), now=now)
    tls = collect_tls_evidence(
        bundle.get("tls"), cert_expiry=bundle.get("cert_expiry"), now=now
    )
    quota = collect_quota_evidence(bundle.get("quota"), now=now)
    rotation = collect_key_rotation_evidence(bundle.get("key_rotation"), now=now)

    controls = {
        "CORS": cors,
        "TLS": tls,
        "QUOTA": quota,
        "KEY_ROTATION": rotation,
    }

    summary_counts = {"pass": 0, "fail": 0, "warn": 0}
    all_control_ids: list[str] = []
    for entry in controls.values():
        summary_counts[entry["status"]] = summary_counts.get(entry["status"], 0) + 1
        for cid in entry["control_ids"]:
            if cid not in all_control_ids:
                all_control_ids.append(cid)

    report = {
        "generated_at": _now_iso(_now_dt(now)),
        "framework": "SOC2-CC / ISO27001-AnnexA",
        "disclaimer": (
            "Automated point-in-time evidence collection only. NOT a SOC 2 or "
            "ISO 27001 certification. An auditor must independently review, "
            "sample, and attest."
        ),
        "controls": controls,
        "summary": {
            **summary_counts,
            "total": len(controls),
            "control_ids": all_control_ids,
        },
    }
    return report


# --- reporting --------------------------------------------------------------
def write_report(
    report: dict,
    output_dir: str = "compliance",
    basename: str = "ComplianceReport",
) -> list[str]:
    """Write the report as JSON and a human-readable Markdown file.

    Returns the list of written file paths. Safe to call from tooling; failures
    are raised so callers can decide how to handle them.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    json_path = out / f"{basename}.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    written.append(str(json_path))

    md_path = out / f"{basename}.md"
    lines = [
        "# Automated Compliance Evidence Report",
        "",
        f"- **Generated at:** {report.get('generated_at')}",
        f"- **Framework:** {report.get('framework')}",
        "",
        "> ⚠️ " + report.get("disclaimer", ""),
        "",
        "## Summary",
        "",
        f"- pass: {report['summary']['pass']}  "
        f"fail: {report['summary']['fail']}  "
        f"warn: {report['summary']['warn']}  "
        f"total: {report['summary']['total']}",
        f"- Mapped control IDs: {', '.join(report['summary']['control_ids'])}",
        "",
        "## Controls",
        "",
    ]
    for domain, entry in report["controls"].items():
        lines.append(f"### {domain} — {entry['status'].upper()}")
        lines.append("")
        lines.append(f"- Control IDs: {', '.join(entry['control_ids'])}")
        lines.append(f"- Collected at: {entry['collected_at']}")
        lines.append(f"- Notes: {entry['notes']}")
        lines.append(f"- Evidence: `{json.dumps(entry['evidence'])}`")
        lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    written.append(str(md_path))
    return written


__all__ = [
    "CONTROL_MAP",
    "collect_cors_evidence",
    "collect_tls_evidence",
    "collect_quota_evidence",
    "collect_key_rotation_evidence",
    "bundle_from_settings",
    "collect_all",
    "write_report",
]
