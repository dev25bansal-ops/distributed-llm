---
tags:
  - frontend
  - tauri
  - vscode
  - website
  - apps
aliases:
  - Frontends
---
# Frontends — `tauri/` + `website/` + `extensions/` + `dashboard|ui`

> Everything a human interacts with outside the engine. These talk to DistLLM **only over its public HTTP endpoints** (`/health`, `/v1/*`, `/dashboard`); they share nothing with core/inference.

## Desktop app — `tauri/` (Svelte 5 + Tauri 2 + Rust)
- Runtime dashboard, chat, model management, cluster map, logs.
- Run: `cd tauri && npm run tauri dev`.
- (JS/Rust source + config; `node_modules/`/`build/` excluded from the vault; 92 files incl. `package-lock.json`/`Cargo.lock`.)

## VS Code extension — `extensions/vscode/` (TS)
- VS Code extension (TS)
- Status-bar items (model, node health, tok/s color-coded) polling `/health` + `/api/metrics/collector`; right-click "send selection" → `/v1/completions`; **DistLLM Models** tree; config validation + snippets (8, Python+JS/TS).
- Key files: `src/extension.ts` (356), `src/configValidation.ts`, `src/modelsApi.ts`, `src/modelsView.ts`, `snippets/distllm.json`.

<!-- recounted 2026-11-06 by hygiene pass -->

## Marketing/product website — `website/`
- Static multi-page site (Vite MPA build + docker/nginx): 23 HTML pages (home, api, benchmarks, enterprise, playground, i18n) + JS SDK-widget modules in `js/` (cluster-viz, benchmark-explorer, deploy-configurator, model-playground, i18n…), 4 locales (en/ja/ko/zh, partially translated), CSS design system, PWA service worker, SEO.
- Tests: Playwright (`accessibility`, `integration`, `security`, `visual-regression`) + Vitest units (`calculator`, `deploy-wizard`, `gpu-checker`, `model-explorer`, `model-rec`).

## Demo apps — deleted in 2026-11-06 hygiene pass

`apps/` (chat/rag/multi_agent) was removed as dead scaffolding in the hygiene pass. `examples/` still carries drop-in integration scripts (LangChain, LlamaIndex, CrewAI, Haystack, Agency Swarm, OpenAI-Agents SDK, routed multi-provider, direct `distllm.sdk`) and 4 notebooks (batch, cost, RAG, streaming).

## In-repo web front-ends — `src/distllm/dashboard`
- `dashboard/` — `ws_handler.py` (WS+SSE real-time metrics; v1 + `static_v2/` with WebRTC + leaderboard). Full detail in [[11 Platform Services]]; volume/captures duplicated as v1+v2.
- `src/distllm/ui/` was **removed 2026-11-08**: it was the standalone Jinja web UI that the CLI already documents as removed in v0.4.0 ("the dashboard is now embedded in the API server at `/dashboard`"). Nothing under `src/` imported it; only its own test did.

## Notes
- Website i18n partly translated; several widget modules wired to a single page.
- VS Code `package.json` `main` → `out/extension.js` (`src/*.ts` must compile first).
- Dashboard v1 (`static/`) and v2 (`static_v2/`) **duplicate** — not merged; `dashboard/ui/` empty dir. (v1 serves `/dashboard`, v2 serves leaderboard/models — both live, wired at `server.py:965,1179,1193`.)

## Tests
`tests/e2e/test_tauri_e2e.py`, `test_vscode_extension.py`; `tests/dashboard/test_ws_handler.py`; website `tests/*.spec.js`/`*.test.js`. (`tests/ui/test_ui.py` was removed with the package, 2026-11-08.)