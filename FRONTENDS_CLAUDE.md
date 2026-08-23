# CLAUDE.md — Frontends (Tauri, Website, Dashboard, UI, Extensions)

## Your Scope
You have ownership of `tauri/`, `website/`, `extensions/vscode/`, and `src/distllm/dashboard/` + `src/distllm/ui/` — all user-facing frontend code.

## Do NOT Touch
- `src/distllm/core/` — core engine
- `src/distllm/dist/` — distributed layer
- `src/distllm/api/` — API server
- `tests/`

## Key Directories

| Directory | Stack | Purpose |
|-----------|-------|---------|
| `tauri/` | Svelte 5 + Tauri 2 + Rust | Desktop app with dashboard, chat, cluster map, logs |
| `website/` | HTML/CSS/JS | Marketing site with 4 locales (EN, JA, KO, ZH) |
| `extensions/vscode/` | TypeScript | VS Code extension — cluster health, selection inference |
| `src/distllm/dashboard/` | HTML/CSS/JS | API dashboard static files |
| `src/distllm/ui/` | Jinja2 + Python | Web UI templates |

## Setup Guides

| Guide | Location |
|-------|----------|
| Tauri desktop app | `tauri/SETUP.md` |
| VSCode extension | `extensions/vscode/README.md` |

## Commands
- `cd tauri && npm run tauri dev` — Tauri dev mode
- `cd tauri && npm run tauri build` — Tauri build
- `cd extensions/vscode && npm run compile` — VS Code extension build
- `cd website && python -m http.server 8080` — Website preview
