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
# Frontends & Apps — `tauri/` + `website/` + `extensions/` + `apps/` + `dashboard|ui`

> Everything a human interacts with outside the engine. These talk to DistLLM **only over its public HTTP endpoints** (`/health`, `/v1/*`, `/dashboard`); they share nothing with core/inference.

## Desktop app — `tauri/` (Svelte 5 + Tauri 2 + Rust)
- Runtime dashboard, chat, model management, cluster map, logs.
- Run: `cd tauri && npm run tauri dev`.
- (JS/Rust source + config; `node_modules/`/`build/` excluded from the vault.)

## VS Code extension — `extensions/vscode/` (TS)
- Status-bar items (model, node health, tok/s color-coded) polling `/health` + `/api/metrics/collector`; right-click "send selection" → `/v1/completions`; **DistLLM Models** tree; config validation + snippets (8, Python+JS/TS).
- Key files: `src/extension.ts` (356), `src/configValidation.ts`, `src/modelsApi.ts`, `src/modelsView.ts`, `snippets/distllm.json`.

## Marketing/product website — `website/`
- Static multi-page site (Vite MPA build + docker/nginx): ~25 HTML pages (home, api, benchmarks, enterprise, playground, i18n) + ~46 JS SDK-widget modules (cluster-viz, benchmark-explorer, deploy-configurator, model-playground, i18n…), 4 locales (en/ja/ko/zh, partially translated), CSS design system, PWA service worker, SEO.
- Tests: Playwright (`accessibility`, `integration`, `security`, `visual-regression`) + Vitest units (`calculator`, `deploy-wizard`, `gpu-checker`, `model-explorer`, `model-rec`).

## Demo apps — `apps/` (3 apps)
| App | Stack | What it does |
|-----|-------|--------------|
| `chat/` | Flask + vanilla JS | streaming chat UI via SSE, model selector |
| `rag/` | Flask + numpy (+optional LlamaIndex) | `NumpyRAG` retriever → answer panel |
| `multi_agent/` | asyncio openai | Researcher→Writer→Reviewer chain |

Plus `examples/` — drop-in integration scripts (LangChain, LlamaIndex, CrewAI, Haystack, Agency Swarm, OpenAI-Agents SDK, routed multi-provider, direct `distllm.sdk`) and 4 notebooks (batch, cost, RAG, streaming).

## In-repo web front-ends — `src/distllm/dashboard` + `src/distllm/ui`
- `dashboard/` — `ws_handler.py` (WS+SSE real-time metrics; v1 + `static_v2/` with WebRTC + leaderboard + verification report). Full detail in [[11 Platform Services]]; volume/captures duplicated as v1+v2.
- `ui/` — FastAPI Jinja pages over coordinator (`ui/app.py` routes; templates + static).

## Notes
- Website i18n partly translated; several widget modules wired to a single page.
- VS Code `package.json` `main` → `out/extension.js` (`src/*.ts` must compile first).
- Dashboard v1 (`static/`) and v2 (`static_v2/`) **duplicate** — not merged; `dashboard/ui/` empty dir.

## Tests
`tests/e2e/test_tauri_e2e.py`, `test_vscode_extension.py`; `tests/dashboard/test_ws_handler.py`; `tests/ui/test_ui.py`; website `tests/*.spec.js`/`*.test.js`.