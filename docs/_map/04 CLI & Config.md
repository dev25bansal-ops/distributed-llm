---
tags:
  - cli
  - config
---
# CLI & Config

**Location:** `src/distllm/cli/` + `src/distllm/config/` — **665 KB + 288 KB, ~45 files**

**Commands:** `python -m pytest tests/cli/ -v` | `python -m pytest tests/config/ -v`

## CLI Commands
`distllm` entry point with sub-groups: `cluster`, `model`, `benchmark`, `config`, `security`, `system`, `router`, `tune`, `draft`, `daas`

## Key Config Files
| File | Purpose |
|------|---------|
| `config/settings.py` | `DistLLMSettings` — all config fields (40+) |
| `config/loader.py` | YAML config loader |
| `config/resolver.py` | CLI config resolver |
| `config/profiles.py` | Config profiles |
| `cli/client.py` | Unified HTTP client (NEW) |

## Dependencies → [[docs/_map/03 API Server]], [[docs/_map/01 Core Engine]]

## Recent Work
- ✅ Orphan Typer apps removed from 6 CLI modules
- ✅ Unified HTTP client (`DistLLMClient`) with retry + rate-limit handling
- ✅ Coverage config with `fail_under = 80` in pyproject.toml
