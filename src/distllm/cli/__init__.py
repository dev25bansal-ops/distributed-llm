"""DistLLM CLI module — organized into logical command groups.

Groups:
  cluster   — cluster management (status, scale, start, join, deploy)
  model     — model lifecycle (list, load, compress, adapters)
  benchmark — benchmarking and profiling (run, compare, profile, verify)
  config    — configuration (setup, validate, webhook, quota, backup)
  security  — TLS certificates (cert)
  system    — daemons, diagnostics, logs, notifications (run, api, doctor, logs)

``app``/``main`` are exposed lazily via PEP 562 ``__getattr__`` so that
``python -m distllm.cli.main`` does not import this module's target twice
(avoids the runpy RuntimeWarning about re-importing ``distllm.cli.main``).
"""

from typing import Any

__all__ = [
    "app",
    "main",
]


def __getattr__(name: str) -> Any:
    if name in ("app", "main"):
        from distllm.cli import main as _main

        return getattr(_main, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
