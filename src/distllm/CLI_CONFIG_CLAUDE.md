# CLAUDE.md — CLI & Config

## Your Scope
You have ownership of `src/distllm/cli/` and `src/distllm/config/` — all CLI commands, configuration management, and the unified HTTP client.

## Do NOT Touch
- `src/distllm/core/` — core engine
- `src/distllm/dist/` — distributed layer
- `src/distllm/api/` — API server

## Key Files

| File | Purpose |
|------|---------|
| `cli/main.py` | Typer app entry point (1675 lines — all commands inline) |
| `cli/chat.py` | Interactive chat CLI |
| `cli/cluster.py` | Cluster start/join/status |
| `cli/run.py` | Run inference |
| `cli/deploy.py` | Deploy models |
| `cli/setup.py` | Setup wizard |
| `cli/doctor.py` | Diagnostics |
| `cli/benchmark.py` | Benchmarks |
| `cli/tune.py` | Auto-tuning |
| `cli/backup.py` | Backup/restore (function-only, no typer app) |
| `cli/cert.py` | Certificate management (function-only) |
| `cli/notify.py` | Notifications (function-only) |
| `cli/quota.py` | Quota management (function-only) |
| `cli/verify.py` | Accuracy verification (function-only) |
| `cli/webhook.py` | Webhook management (function-only) |
| `cli/client.py` | Unified HTTP client (NEW) |
| `config/settings.py` | DistLLMSettings — all config fields |
| `config/loader.py` | YAML config loader |
| `config/resolver.py` | CLI config resolver |
| `config/profiles.py` | Config profiles |
| `config/_*.py` | Domain-specific config sections |

## Current State
- 6 orphan Typer apps removed from sub-modules
- Unified HTTP client `DistLLMClient` created
- Coverage config in pyproject.toml
- All CLI commands delegate to imported functions

## Commands
- `python -m pytest tests/cli/ -v` — run CLI tests
- `python -m pytest tests/config/ -v` — run config tests
