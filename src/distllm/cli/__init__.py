"""DistLLM CLI module — organized into logical command groups.

Groups:
  cluster   — cluster management (status, scale, start, join, deploy)
  model     — model lifecycle (list, load, compress, adapters)
  benchmark — benchmarking and profiling (run, compare, profile, verify)
  config    — configuration (setup, validate, webhook, quota, backup)
  security  — TLS certificates (cert)
  system    — daemons, diagnostics, logs, notifications (run, api, doctor, logs)
"""

from distllm.cli.main import app, main

__all__ = [
    "app",
    "main",
]
