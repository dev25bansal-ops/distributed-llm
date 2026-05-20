"""Plugin compatibility checking for the DistLLM plugin marketplace.

Checks host version compatibility, Python version, required packages,
and GPU availability before plugin installation and loading.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

from packaging.version import Version, InvalidVersion
from packaging.specifiers import SpecifierSet, InvalidSpecifier
from loguru import logger


@dataclass
class CompatibilityResult:
    """Result of a compatibility check."""
    compatible: bool
    warnings: list[str]
    errors: list[str]

    @property
    def can_install(self) -> bool:
        return self.compatible and not self.errors

    @property
    def can_load(self) -> bool:
        return self.compatible


class CompatibilityChecker:
    """Checks plugin compatibility against the host environment.

    Validates:
    - DistLLM host version range (semver)
    - Python version compatibility
    - Required package dependencies
    - GPU availability for GPU-requiring plugins
    """

    def __init__(self, host_version: str = "0.1.0") -> None:
        self.host_version = host_version

    def check_compatibility(
        self,
        *,
        min_host_version: str | None = None,
        max_host_version: str | None = None,
        python_requires: str | None = None,
        requires_gpu: bool = False,
        dependencies: list[str] | None = None,
    ) -> CompatibilityResult:
        """Run all compatibility checks.

        Args:
            min_host_version: Minimum DistLLM version (inclusive).
            max_host_version: Maximum DistLLM version (exclusive).
            python_requires: Python version specifier (e.g., ">=3.10,<3.13").
            requires_gpu: Whether the plugin needs a GPU.
            dependencies: List of pip dependency strings.

        Returns:
            CompatibilityResult with errors/warnings.
        """
        errors: list[str] = []
        warnings: list[str] = []

        # Host version check
        host_errors = self._check_host_version(min_host_version, max_host_version)
        errors.extend(host_errors)

        # Python version check
        py_errors = self._check_python_version(python_requires)
        errors.extend(py_errors)

        # GPU check
        if requires_gpu:
            gpu_warnings = self._check_gpu_availability()
            warnings.extend(gpu_warnings)

        # Dependency check
        dep_errors = self._check_dependencies(dependencies or [])
        errors.extend(dep_errors)

        compatible = len(errors) == 0

        return CompatibilityResult(
            compatible=compatible,
            warnings=warnings,
            errors=errors,
        )

    def _check_host_version(
        self,
        min_version: str | None,
        max_version: str | None,
    ) -> list[str]:
        """Check if current host version is within the required range."""
        errors: list[str] = []

        try:
            host = Version(self.host_version)
        except InvalidVersion:
            errors.append(f"Invalid host version: {self.host_version}")
            return errors

        if min_version:
            try:
                if host < Version(min_version):
                    errors.append(
                        f"Host version {self.host_version} is below minimum required {min_version}"
                    )
            except InvalidVersion:
                errors.append(f"Invalid min_host_version: {min_version}")

        if max_version:
            try:
                if host >= Version(max_version):
                    errors.append(
                        f"Host version {self.host_version} is at or above maximum {max_version}"
                    )
            except InvalidVersion:
                errors.append(f"Invalid max_host_version: {max_version}")

        return errors

    def _check_python_version(self, requires: str | None) -> list[str]:
        """Check if current Python version meets requirements."""
        if not requires:
            return []

        current = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

        try:
            spec = SpecifierSet(requires)
            if not spec.contains(current):
                return [
                    f"Python {current} does not satisfy requirement '{requires}' "
                    f"(need {sys.version_info.major}.{sys.version_info.minor})"
                ]
        except InvalidSpecifier:
            return [f"Invalid python_requires specifier: {requires}"]

        return []

    def _check_gpu_availability(self) -> list[str]:
        """Check if a GPU is available."""
        warnings: list[str] = []
        try:
            import torch
            if not torch.cuda.is_available():
                warnings.append(
                    "Plugin requires GPU but no CUDA device is available. "
                    "The plugin may fail at runtime."
                )
        except ImportError:
            warnings.append(
                "Plugin requires GPU but PyTorch is not installed. "
                "Cannot verify GPU availability."
            )
        return warnings

    def _check_dependencies(self, dependencies: list[str]) -> list[str]:
        """Check if required dependencies are installed."""
        errors: list[str] = []
        import importlib.metadata

        for dep in dependencies:
            # Parse package name (ignore version specifiers for now)
            pkg_name = dep.split(">=")[0].split("<")[0].split("==")[0].split("[")[0].strip()
            if not pkg_name:
                continue

            try:
                importlib.metadata.distribution(pkg_name)
            except importlib.metadata.PackageNotFoundError:
                errors.append(f"Missing dependency: {pkg_name}")

        return errors
