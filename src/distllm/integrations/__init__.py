"""DistLLM Integration namespace.

Third-party integration packages install into ``distllm.integrations.*``
so they are importable as ``distllm.integrations.<name>``.

This module discovers additional integration sub-packages by scanning
``sys.path`` for directories matching ``distllm/integrations/<name>/``.
This allows editable-installed integration packages (e.g. from the
``integrations/`` directory tree) to be importable under the
``distllm.integrations`` namespace.
"""

from __future__ import annotations

import os
import sys

# Re-export first-party integration components.
from distllm.integrations.mlflow_tracking import MLflowIntegration as MLflowIntegration
from distllm.integrations.ci.gitlab import GitLabCIIntegration as GitLabCIIntegration
from distllm.integrations.ci.jenkins import JenkinsIntegration as JenkinsIntegration


def _discover_integration_paths() -> list[str]:
    """Return a list of additional paths containing ``distllm/integrations/``.

    Scans ``sys.path`` for directories that have a
    ``distllm/integrations/`` subdirectory.  If that subdirectory is not
    already part of the current ``__path__`` it is added.
    """
    discovered: list[str] = []
    for entry in sys.path:
        if not isinstance(entry, str) or not os.path.isdir(entry):
            continue
        candidate = os.path.join(entry, "distllm", "integrations")
        if os.path.isdir(candidate) and candidate not in discovered:
            discovered.append(candidate)
    return discovered


def _discover_pkg_dirs() -> list[str]:
    """Discover integration directories from installed package metadata.

    Uses ``importlib.metadata`` to find packages that provide a
    ``distllm.integrations`` entry point group.
    """
    from importlib import metadata

    discovered: list[str] = []
    try:
        for ep in metadata.entry_points(group="distllm.integrations"):
            # The entry point value is a path to the integration module
            if ep.value and os.path.isdir(ep.value):
                if ep.value not in discovered:
                    discovered.append(ep.value)
    except TypeError:
        # Some Python versions need different entry_points() call
        pass
    except Exception:
        pass
    return discovered


# Extend the package path with discovered integration directories
_current = list(__path__)  # snapshot to avoid infinite recursion
for p in _discover_integration_paths():
    if p not in _current:
        __path__.append(p)  # type: ignore[arg-type]

for p in _discover_pkg_dirs():
    if p not in __path__:
        __path__.append(p)  # type: ignore[arg-type]
