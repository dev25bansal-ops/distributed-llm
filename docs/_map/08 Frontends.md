---
tags:
  - frontend
  - tauri
  - vscode
  - website
---
# Frontends

**Location:** `tauri/` + `website/` + `extensions/vscode/` + `src/distllm/dashboard/` + `src/distllm/ui/` — **~3 MB**

## Desktop App — Tauri + Svelte
| Aspect | Detail |
|--------|--------|
| Stack | Svelte 5 + Tauri 2 + Rust |
| Features | Dashboard, Chat, Model Mgmt, Cluster Map, Logs |
| Setup | `tauri/SETUP.md` |
| Run | `cd tauri && npm run tauri dev` |

## Marketing Website
HTML/CSS/JS with 4 locales (EN, JA, KO, ZH)

## VS Code Extension
TypeScript extension with status bar (model, nodes, throughput) + "Send to DistLLM"
`cd extensions/vscode && npm run compile`
