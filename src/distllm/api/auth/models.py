"""Shared SSO data models — SSOUserInfo, role mapping."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SSOUserInfo:
    """User info returned after successful SSO authentication.

    Maps provider-specific claims to a standard format.
    """
    sub: str                          # Unique user ID
    email: str = ""
    name: str = ""
    roles: list[str] = field(default_factory=list)
    groups: list[str] = field(default_factory=list)
    provider: str = ""
    raw_attributes: dict[str, Any] = field(default_factory=dict)

    @property
    def to_api_key_role(self) -> str:
        """Map SSO roles/groups to DistLLM API key roles.

        Heuristic: if user has 'admin' group → 'admin'.
        If 'auditor' → 'auditor'. Otherwise → 'inference-only'.
        """
        all_roles = self.roles + self.groups
        if "admin" in all_roles or "Administrator" in all_roles:
            return "admin"
        if "auditor" in all_roles or "Auditor" in all_roles:
            return "auditor"
        if "read-only" in all_roles:
            return "read-only"
        return "inference-only"
