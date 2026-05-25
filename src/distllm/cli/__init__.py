"""DistLLM CLI module.

Commands: run, setup, status, models, cluster, adapters, logs,
compress, benchmark, chat, deploy, profile, dashboard.
"""

from distllm.cli.main import app, main

__all__ = [
    "app",
    "main",
]
