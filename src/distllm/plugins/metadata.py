"""Plugin metadata schema for the DistLLM plugin marketplace.

Defines structured metadata for plugins including author, license,
dependencies, compatibility range, categories, and configuration schema.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from packaging.version import Version, InvalidVersion


@dataclass
class PluginMetadata:
    """Complete metadata for a DistLLM plugin.

    Attributes:
        name: Unique plugin identifier (PEP 503 normalized).
        version: Semantic version string.
        description: Short human-readable description.
        author: Author or organization name.
        author_email: Contact email for the author.
        license: SPDX license identifier (e.g., "MIT", "Apache-2.0").
        dependencies: List of pip-installable dependency strings.
        min_host_version: Minimum compatible DistLLM version (inclusive).
        max_host_version: Maximum compatible DistLLM version (exclusive).
        categories: Taxonomy tags (e.g., "observability", "security", "routing").
        entry_point: Fully qualified module path to the plugin class.
        settings_schema: JSON Schema dict for plugin configuration validation.
        homepage: Project URL.
        repository: Source code URL.
        documentation: Documentation URL.
    """
    name: str
    version: str = "0.1.0"
    description: str = ""
    author: str = ""
    author_email: str = ""
    license: str = "MIT"
    dependencies: list[str] = field(default_factory=list)
    min_host_version: str = "0.1.0"
    max_host_version: str | None = None
    categories: list[str] = field(default_factory=list)
    entry_point: str = ""
    settings_schema: dict[str, Any] = field(default_factory=dict)
    homepage: str = ""
    repository: str = ""
    documentation: str = ""

    _NAME_RE = re.compile(r'^[a-z0-9]([a-z0-9._-]*[a-z0-9])?$')
    _SPDX_RE = re.compile(r'^[A-Za-z0-9.\-]+$')

    def validate(self) -> list[str]:
        """Return a list of validation errors (empty if valid)."""
        errors: list[str] = []

        if not self.name:
            errors.append("name is required")
        elif not self._NAME_RE.match(self.name):
            errors.append(f"name '{self.name}' is not a valid plugin identifier")

        try:
            Version(self.version)
        except InvalidVersion:
            errors.append(f"version '{self.version}' is not a valid semantic version")

        if not self.entry_point:
            errors.append("entry_point is required")
        elif "." not in self.entry_point:
            errors.append(f"entry_point '{self.entry_point}' must be a fully qualified path (module.Class)")

        if self.license and not self._SPDX_RE.match(self.license):
            errors.append(f"license '{self.license}' is not a valid SPDX identifier")

        if self.min_host_version:
            try:
                Version(self.min_host_version)
            except InvalidVersion:
                errors.append(f"min_host_version '{self.min_host_version}' is invalid")

        if self.max_host_version:
            try:
                Version(self.max_host_version)
            except InvalidVersion:
                errors.append(f"max_host_version '{self.max_host_version}' is invalid")

        for dep in self.dependencies:
            if not dep.strip():
                errors.append("dependencies contains empty string")

        return errors

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dictionary (for JSON/plugin.json)."""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "author": self.author,
            "author_email": self.author_email,
            "license": self.license,
            "dependencies": self.dependencies,
            "min_host_version": self.min_host_version,
            "max_host_version": self.max_host_version,
            "categories": self.categories,
            "entry_point": self.entry_point,
            "settings_schema": self.settings_schema,
            "homepage": self.homepage,
            "repository": self.repository,
            "documentation": self.documentation,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PluginMetadata":
        """Deserialize from a dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class PluginManifest:
    """Loads plugin metadata from plugin.json or pyproject.toml."""

    @staticmethod
    def from_plugin_json(path: str | Path) -> PluginMetadata:
        """Load metadata from a plugin.json file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"plugin.json not found at {path}")

        with open(path) as f:
            data = json.load(f)

        return PluginMetadata.from_dict(data)

    @staticmethod
    def from_pyproject(path: str | Path) -> PluginMetadata:
        """Load metadata from a pyproject.toml file.

        Extracts from [project] table and [tool.distllm-plugin] table.
        """
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib  # type: ignore[import-not-found]

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"pyproject.toml not found at {path}")

        with open(path, "rb") as f:
            pyproject = tomllib.load(f)

        project = pyproject.get("project", {})
        distllm = pyproject.get("tool", {}).get("distllm-plugin", {})

        return PluginMetadata(
            name=project.get("name", ""),
            version=project.get("version", "0.1.0"),
            description=project.get("description", ""),
            author=project.get("authors", [{}])[0].get("name", "") if project.get("authors") else "",
            license=project.get("license", {}).get("text", "") if isinstance(project.get("license"), dict) else str(project.get("license", "")),
            dependencies=project.get("dependencies", []),
            entry_point=distllm.get("entry_point", ""),
            min_host_version=distllm.get("min_host_version", "0.1.0"),
            max_host_version=distllm.get("max_host_version"),
            categories=distllm.get("categories", []),
            settings_schema=distllm.get("settings_schema", {}),
            homepage=project.get("urls", {}).get("Homepage", ""),
            repository=project.get("urls", {}).get("Repository", ""),
            documentation=project.get("urls", {}).get("Documentation", ""),
        )


VALID_CATEGORIES = {
    "observability", "security", "routing", "compression", "caching",
    "monitoring", "authentication", "authorization", "logging", "tracing",
    "metrics", "profiling", "testing", "debugging", "tooling", "integration",
}


def validate_metadata(meta: PluginMetadata) -> list[str]:
    """Validate plugin metadata and return a list of errors.

    Extends PluginMetadata.validate() with additional marketplace rules.
    """
    errors = meta.validate()

    # Check for unknown categories
    for cat in meta.categories:
        if cat.lower() not in VALID_CATEGORIES:
            errors.append(f"Unknown category '{cat}'. Valid: {', '.join(sorted(VALID_CATEGORIES))}")

    # Check entry_point format
    if meta.entry_point:
        parts = meta.entry_point.split(".")
        if len(parts) < 2:
            errors.append(f"entry_point '{meta.entry_point}' must have at least module.class format")

    return errors
