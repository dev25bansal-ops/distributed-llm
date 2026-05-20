"""Plugin installer for the DistLLM plugin marketplace.

Handles pip-based installation, uninstallation, version pinning,
dependency resolution, and plugin registry queries.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger

from distllm.plugins.metadata import PluginMetadata


@dataclass
class PluginInstallResult:
    """Result of a plugin installation."""
    success: bool
    metadata: PluginMetadata | None = None
    errors: list[str] = field(default_factory=list)
    dependencies_installed: list[str] = field(default_factory=list)


class PluginInstaller:
    """Installs and uninstalls DistLLM plugins via pip.

    Supports:
    - Installing from PyPI by name
    - Version pinning
    - Dependency resolution
    - Local file/wheel installation
    - Uninstallation
    """

    def __init__(
        self,
        plugin_registry_url: str = "https://pypi.org/simple/",
        pip_args: list[str] | None = None,
    ) -> None:
        self.registry_url = plugin_registry_url
        self.pip_args = pip_args or []
        self._installed: dict[str, PluginMetadata] = {}

    def install(
        self,
        plugin_name: str,
        version: str | None = None,
        extras: list[str] | None = None,
    ) -> PluginInstallResult:
        """Install a plugin by name.

        Args:
            plugin_name: Package name on PyPI.
            version: Specific version to install (latest if None).
            extras: Optional extras to install (e.g., ["gpu", "cuda"]).

        Returns:
            PluginInstallResult with metadata and any errors.
        """
        # Build pip spec
        if version:
            pip_spec = f"{plugin_name}=={version}"
        else:
            pip_spec = plugin_name

        if extras:
            extras_str = ",".join(extras)
            pip_spec = f"{plugin_name}[{extras_str}]" if version else f"{plugin_name}[{extras_str}]"

        # Install via pip
        success, errors = self._pip_install(pip_spec)
        if not success:
            return PluginInstallResult(success=False, errors=errors)

        # Load metadata from installed package
        try:
            metadata = self._load_installed_metadata(plugin_name)
        except Exception as e:
            logger.warning(f"Could not load metadata for {plugin_name}: {e}")
            metadata = PluginMetadata(name=plugin_name, version=version or "unknown")

        self._installed[plugin_name] = metadata

        return PluginInstallResult(
            success=True,
            metadata=metadata,
            dependencies_installed=metadata.dependencies if metadata else [],
        )

    def install_from_file(self, file_path: str) -> PluginInstallResult:
        """Install a plugin from a local file or wheel.

        Args:
            file_path: Path to .whl, .tar.gz, or directory with pyproject.toml.

        Returns:
            PluginInstallResult.
        """
        path = Path(file_path)
        if not path.exists():
            return PluginInstallResult(success=False, errors=[f"File not found: {file_path}"])

        success, errors = self._pip_install(str(path))
        if not success:
            return PluginInstallResult(success=False, errors=errors)

        # Try to extract name from filename
        name = path.stem.split("-")[0] if "-" in path.stem else path.stem

        try:
            metadata = self._load_installed_metadata(name)
        except Exception:
            metadata = PluginMetadata(name=name, version="unknown")

        self._installed[name] = metadata
        return PluginInstallResult(success=True, metadata=metadata)

    def uninstall(self, plugin_name: str) -> bool:
        """Uninstall a plugin by name.

        Args:
            plugin_name: Package name to uninstall.

        Returns:
            True if successfully uninstalled.
        """
        success, errors = self._pip_uninstall(plugin_name)
        if success:
            self._installed.pop(plugin_name, None)
            logger.info(f"Plugin '{plugin_name}' uninstalled")
        else:
            logger.error(f"Failed to uninstall '{plugin_name}': {errors}")
        return success

    def list_installed(self) -> list[PluginMetadata]:
        """Return metadata for all installed DistLLM plugins."""
        # Refresh from entry points
        return list(self._installed.values())

    def resolve_dependencies(self, metadata: PluginMetadata) -> list[str]:
        """Resolve and return the list of dependencies that need installation.

        Args:
            metadata: Plugin metadata with dependencies list.

        Returns:
            List of dependency strings that are not yet installed.
        """
        import importlib.metadata

        missing = []
        for dep in metadata.dependencies:
            pkg_name = dep.split(">=")[0].split("<")[0].split("==")[0].split("[")[0].strip()
            if not pkg_name:
                continue
            try:
                importlib.metadata.distribution(pkg_name)
            except importlib.metadata.PackageNotFoundError:
                missing.append(dep)

        return missing

    def _pip_install(self, package_spec: str) -> tuple[bool, list[str]]:
        """Run pip install for a package spec."""
        cmd = [
            sys.executable, "-m", "pip", "install",
            "--quiet",
            "--index-url", self.registry_url,
            *self.pip_args,
            package_spec,
        ]

        logger.info(f"Installing plugin: {cmd[-1]}")
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
            )
            if result.returncode == 0:
                logger.info(f"Successfully installed {package_spec}")
                return True, []
            else:
                errors = result.stderr.strip().split("\n")
                return False, errors
        except subprocess.TimeoutExpired:
            return False, [f"pip install timed out for {package_spec}"]
        except Exception as e:
            return False, [str(e)]

    def _pip_uninstall(self, package_name: str) -> tuple[bool, list[str]]:
        """Run pip uninstall for a package."""
        cmd = [
            sys.executable, "-m", "pip", "uninstall",
            "--yes", "--quiet",
            package_name,
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,
            )
            if result.returncode == 0:
                return True, []
            else:
                return False, result.stderr.strip().split("\n")
        except Exception as e:
            return False, [str(e)]

    def _load_installed_metadata(self, package_name: str) -> PluginMetadata:
        """Load metadata from an installed package via importlib.metadata."""
        import importlib.metadata

        dist = importlib.metadata.distribution(package_name)
        meta = dist.metadata

        return PluginMetadata(
            name=meta.get("Name", package_name),
            version=meta.get("Version", "unknown"),
            description=meta.get("Summary", ""),
            author=meta.get("Author", ""),
            author_email=meta.get("Author-email", ""),
            license=meta.get("License", ""),
            homepage=meta.get("Home-page", ""),
        )
