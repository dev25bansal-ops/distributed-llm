"""DistLLM plugin marketplace infrastructure.

Provides metadata, installation, sandboxing, compatibility checking,
telemetry, and configuration validation for the plugin ecosystem.
"""

from distllm.plugins.metadata import PluginMetadata, PluginManifest, validate_metadata
from distllm.plugins.config_schema import PluginConfigValidator
from distllm.plugins.compatibility import CompatibilityChecker, CompatibilityResult
from distllm.plugins.installer import PluginInstaller, PluginInstallResult
from distllm.plugins.sandbox import PluginSandbox, SandboxContext, SandboxStats
from distllm.plugins.telemetry import PluginTelemetry, PluginStats, TelemetryRecord

__all__ = [
    "PluginMetadata",
    "PluginManifest",
    "validate_metadata",
    "PluginConfigValidator",
    "CompatibilityChecker",
    "CompatibilityResult",
    "PluginInstaller",
    "PluginInstallResult",
    "PluginSandbox",
    "SandboxContext",
    "SandboxStats",
    "PluginTelemetry",
    "PluginStats",
    "TelemetryRecord",
]
