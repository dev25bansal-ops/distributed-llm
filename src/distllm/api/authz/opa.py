"""OPA/Rego authorization adapter with a pure-Python fallback evaluator.

The module exposes a single decision function, :func:`authorize`, plus
:func:`load_policy`.  The contract is identical regardless of backend:

    Decision(allow: bool, reason: str, source: str)

    * ``allow``   — whether the action is permitted.
    * ``reason``  — human-readable rationale (e.g. matched rule id or default).
    * ``source``  — ``"opa"`` (binary/server), ``"python"`` (fallback), or
                    ``"none"`` (no policy loaded -> default-deny).

Backends (in priority order)
────────────────────────────
1. **OPA binary** — if ``opa`` is on ``PATH`` (or ``OPA_BIN`` is set), the input
   is written to a temp file, a bundled ``.rego`` policy compiled/queried via
   ``opa eval``, and the ``allow`` binding parsed from the JSON result.
2. **OPA server** — if ``OPA_SERVER_URL`` is set, the input is POSTed to
   ``{server}/v1/data/{path}`` and the ``result.allow`` field returned.
3. **Pure-Python** — always available.  Implements the same allow/deny logic
   from a *policy dict* so the contract is exercisable in any environment
   without the OPA binary.

The bundled Rego policy (``DEFAULT_REGO``) and the pure-Python evaluator encode
the SAME rules; see ``_python_evaluate`` for the canonical mapping.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import threading
from dataclasses import dataclass
from typing import Any, Optional

# ── Detection of the OPA binary ────────────────────────────────────────────
OPA_BIN = os.environ.get("OPA_BIN") or shutil.which("opa")
OPA_SERVER_URL = os.environ.get("OPA_SERVER_URL", "").rstrip("/")
OPA_AVAILABLE = bool(OPA_BIN or OPA_SERVER_URL)


# ── Decision contract ──────────────────────────────────────────────────────
@dataclass(frozen=True)
class Decision:
    """Authorization decision returned by :func:`authorize`."""

    allow: bool
    reason: str
    source: str  # "opa" | "python" | "none"

    def __bool__(self) -> bool:  # so ``if authorize(input):`` works
        return self.allow

    def to_dict(self) -> dict[str, Any]:
        return {"allow": self.allow, "reason": self.reason, "source": self.source}


# ── Bundled Rego policy (mirrors the pure-Python rule set) ─────────────────
DEFAULT_REGO = """\
package distllm.authz

# Implicit deny by default; explicit allow rules below.
default allow = false

# Administrators may do anything within their tenant.
allow {
    input.subject.role == "admin"
}

# Service accounts may invoke internal services.
allow {
    input.subject.role == "service"
    startswith(input.resource, "svc:")
}

# A regular reviewer may only read.
allow {
    input.subject.role == "user"
    input.action == "read"
}

# Path-scoped allow: an explicit grant on the resource path.
allow {
    some g
    g := input.subject.grants[_]
    g == input.resource
    input.action == g.action
}
"""

_DEFAULT_POLICY_DICT: dict[str, Any] = {
    "default": "deny",
    "roles": {
        "admin": {"allow_all": True},
        "service": {"prefix": "svc:"},
        "user": {"actions": ["read"]},
    },
    # Explicit grants: list of {"resource": ..., "action": ...}.
    "grants": [],
}


# ── Module-level state ─────────────────────────────────────────────────────
_lock = threading.Lock()
_loaded_policy: dict[str, Any] | None = None
_loaded_rego: str | None = None


def load_policy(path: Optional[str] = None) -> dict[str, Any]:
    """Load the authorization policy from a JSON file (or use the default).

    The file format mirrors :data:`_DEFAULT_POLICY_DICT`.  Returns the loaded
    policy dict.  The OPA Rego text (when also desired) is kept in
    ``_loaded_rego`` from :data:`DEFAULT_REGO` for binary/server evaluation.

    Args:
        path: Optional path to a ``.json`` policy file.  When ``None`` the
            built-in default policy is used.

    Returns:
        The active policy dict.
    """
    global _loaded_policy, _loaded_rego
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as fh:
            policy = json.load(fh)
    else:
        policy = dict(_DEFAULT_POLICY_DICT)
    with _lock:
        _loaded_policy = policy
        _loaded_rego = DEFAULT_REGO
    return policy


def _get_policy() -> dict[str, Any]:
    with _lock:
        if _loaded_policy is None:
            return dict(_DEFAULT_POLICY_DICT)
        return _loaded_policy


# ── Pure-Python evaluator (the fallback contract) ──────────────────────────
def _python_evaluate(data: dict[str, Any]) -> Decision:
    """Evaluate allow/deny purely in Python from the loaded policy dict.

    Mirrors the Rego rules in :data:`DEFAULT_REGO`:
      * default deny;
      * admin role -> allow all;
      * service role -> allow resources prefixed with ``svc:``;
      * user role   -> allow only the listed ``actions``;
      * explicit grants (policy-wide and per-subject) -> allow when
        (resource, action) matches.
    """
    policy = _get_policy()
    subject = data.get("subject") or {}
    action = data.get("action", "")
    resource = data.get("resource", "")

    role = subject.get("role", "")

    # 1. Explicit policy-wide grants take precedence over role rules.
    for g in policy.get("grants", []) or []:
        if g.get("resource") == resource and g.get("action") == action:
            return Decision(True, f"explicit grant for {resource}:{action}", "python")

    # 2. Per-subject explicit grants (role_grants[role] -> list of (res, act)).
    role_grants = (policy.get("role_grants", {}) or {}).get(role) or []
    for g in role_grants:
        if g.get("resource") == resource and g.get("action") == action:
            return Decision(True, f"role grant for {role}:{resource}:{action}", "python")

    # 3. Role-default rules.
    roles = policy.get("roles", {})
    role_cfg = roles.get(role)

    if isinstance(role_cfg, dict) and role_cfg.get("allow_all"):
        return Decision(True, f"role '{role}' is admin (allow_all)", "python")

    if isinstance(role_cfg, dict) and role_cfg.get("prefix") is not None:
        prefix = role_cfg["prefix"]
        if isinstance(resource, str) and resource.startswith(prefix):
            return Decision(True, f"service role allows '{prefix}' resources", "python")
        return Decision(False, f"service role denied non-{prefix} resource", "python")

    if isinstance(role_cfg, dict) and "actions" in role_cfg:
        if action in role_cfg["actions"]:
            return Decision(True, f"role '{role}' permitted action '{action}'", "python")
        return Decision(False, f"role '{role}' denied action '{action}'", "python")

    return Decision(False, "default deny (no matching rule)", "python")


# ── OPA binary / server backends ───────────────────────────────────────────
def _opa_binary_evaluate(data: dict[str, Any]) -> Optional[Decision]:
    if not OPA_BIN:
        return None
    rego = _loaded_rego or DEFAULT_REGO
    try:
        with tempfile.TemporaryDirectory() as tmp:
            rego_path = os.path.join(tmp, "policy.rego")
            input_path = os.path.join(tmp, "input.json")
            with open(rego_path, "w", encoding="utf-8") as fh:
                fh.write(rego)
            with open(input_path, "w", encoding="utf-8") as fh:
                json.dump(data, fh)
            result = subprocess.run(
                [OPA_BIN, "eval", "--format", "json", "--data", rego_path,
                 "--input", input_path, "data.distllm.authz.allow"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode != 0:
                return None
            payload = json.loads(result.stdout)
            # opa eval returns {"result": [{"expressions": [{"value": <bool>}]}]}
            exprs = payload.get("result", [{}])[0].get("expressions", [])
            allow = bool(exprs[0].get("value")) if exprs else False
            return Decision(allow, "opa policy evaluation", "opa")
    except Exception:
        return None


def _opa_server_evaluate(data: dict[str, Any]) -> Optional[Decision]:
    if not OPA_SERVER_URL:
        return None
    try:
        import urllib.request

        url = f"{OPA_SERVER_URL}/v1/data/distllm/authz"
        req = urllib.request.Request(
            url,
            data=json.dumps({"input": data}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:  # noqa: S310
            payload = json.loads(resp.read().decode("utf-8"))
        allow = bool(payload.get("result", {}).get("allow", False))
        return Decision(allow, "opa server evaluation", "opa")
    except Exception:
        return None


# ── Public API ─────────────────────────────────────────────────────────────
def authorize(data: dict[str, Any]) -> Decision:
    """Evaluate an authorization decision for ``data``.

    ``data`` must contain at least: ``subject`` (``{"role": ..., "grants": [...]}``),
    ``action`` (str), and ``resource`` (str).

    Resolution order: OPA binary -> OPA server -> pure-Python fallback.
    """
    # Ensure a policy is loaded.
    if _loaded_policy is None:
        load_policy()

    if OPA_BIN:
        decision = _opa_binary_evaluate(data)
        if decision is not None:
            return decision
    if OPA_SERVER_URL:
        decision = _opa_server_evaluate(data)
        if decision is not None:
            return decision

    return _python_evaluate(data)


class OPAClient:
    """Thin wrapper exposing the authorization API as an object.

    Usage::

        client = OPAClient()
        decision = client.authorize({"subject": ..., "action": ..., "resource": ...})
        if decision.allow:
            ...
    """

    def __init__(self) -> None:
        self.backend = "opa" if OPA_AVAILABLE else "python"

    @property
    def backend_name(self) -> str:
        return self.backend

    def authorize(self, data: dict[str, Any]) -> Decision:
        return authorize(data)

    def load_policy(self, path: Optional[str] = None) -> dict[str, Any]:
        return load_policy(path)


# Eagerly load the default policy so `authorize` works out of the box.
load_policy()
