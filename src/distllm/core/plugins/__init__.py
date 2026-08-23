"""Plugin sandboxing package.

Re-exports the signed-manifest + capability-scoped sandbox implementation
from :mod:`distllm.core.plugin_sandbox` so it is also importable from the
canonical ``distllm.core.plugins.sandbox`` path.
"""

from distllm.core.plugin_sandbox import (
    PluginCapability,
    PluginManifest,
    SandboxPolicy,
    generate_key_pair,
    public_key_from_pem,
    run_sandboxed,
    verify_manifest,
)

__all__ = [
    "PluginCapability",
    "PluginManifest",
    "SandboxPolicy",
    "generate_key_pair",
    "public_key_from_pem",
    "run_sandboxed",
    "verify_manifest",
]
