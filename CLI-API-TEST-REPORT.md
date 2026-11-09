# DistLLM — full CLI + API test on this laptop

Date: 2026-11-09
Worktree: `D:\distributed-llm\.claude\worktrees\bubbly-pondering-whisper`
Server: `http://localhost:8132`, model `roneneldan/TinyStories-1M` (CPU, 8 GiB RTX 5060 Laptop)
Python 3.14.4, torch 2.15.0+cpu

Every command and every endpoint was exercised against a live server, not just
unit-tested. All pytest runs used `PYTHONPATH=src` because `distllm` is installed
non-editable in site-packages (0.6.2.1) and the stale copy would otherwise shadow
the worktree.

---

## Headline

| Area | Coverage | Result |
|---|---|---|
| API test suite | `tests/api/` — all modules | **1306 passed, 0 failed** (10 skipped, 14 xfailed, 26 xpassed) |
| CLI + SDK test suites | `tests/cli/` + `tests/sdk/` | **510 passed, 0 failed** |
| API routes, live | all **161** OpenAPI operations | 86 OK, 26 mutations OK, 34 gated by config, 13 validating, 8 **broken** |
| CLI commands, live | all **96** commands | 51 OK, 26 **ERROR**, 6 hardcoded/fake, 2 interactive-only, 3 long-running |

**The unit tests all pass. A quarter of the CLI is broken against a live server, and eight API routes return 500 or are unreachable.** The gap is the point: the suites mock the HTTP layer, so they never catch client/server path drift.

---

## Critical: client and server disagree on paths

The CLI calls routes the server does not serve. Every one of these exits **0** and prints `Error: HTTP 404`, so a script or user sees success.

| CLI command | CLI calls | Server actually serves |
|---|---|---|
| `model load` | `/v1/models/load` | `/api/models/load` |
| `model unload` | `/v1/models/unload` | `/api/models/unload` |
| `model adapters list/load/set/unload` | `/v1/adapters*` | *nothing — no adapter routes exist* |
| `cluster drain` | `/v1/cluster/drain` | *nothing* |
| `cluster rebalance` | `/v1/cluster/rebalance` | *nothing* |
| `cluster scale` | `/v1/cluster/scale` | *nothing* |
| `system logs stream` | `/v1/logs` | `/admin/v1/logs` |
| `benchmark eval` | `/api/v1/eval/run` | `/v1/eval/run` |

## Critical: commands that can never authenticate

These build a bare `httpx` client with no `Authorization` header, so they 401 on every run against a key-protected server:

`cluster leave`, `cluster deploy`, `system defrag run`, `system defrag status`, `system defrag stats`, `system slo-report`, `draft fleet-status`, `draft migration-status`, `model federate status`, `tune batch`

`http_client.auth_headers()` already exists and five other modules use it. These ten do not.

Worse, three of them turn the 401 into fake success:
- `status` prints `models=0 nodes=0` when a model *is* loaded.
- `system slo-report` renders the 401 error JSON as if it were the report.
- `tune batch` computes from zero samples and emits a bogus recommendation with literal `Infinity` in the JSON.

## Critical: API routes that are broken or unreachable

All reproduced live by me, not just reported by an agent:

| Route | Status | Root cause (verified in source) |
|---|---|---|
| `GET /api/cost/history` | 500 | `cost_tracker.py:426` slices a `deque` — `self._history[-limit:]` raises `TypeError`; `_history` is a `deque` (line 134) |
| `GET /api/pipeline/health` | 500 | handler calls `get_pipeline_metrics()`; `PipelineOrchestrator` only has `stats()` and `health_check()` |
| `GET /v1/scheduler/config` | 500 | reads `scheduler._max_preempted`; the value lives at `scheduler._preemption_mgr._max_preempted` |
| `GET /ready` | 500 | `readiness_check` calls `.get("healthy")` on `node_health` values that are strings/ints once any node is registered |
| `GET /v1/leaderboard/{summary,top,community,verified}` | 404 | `/{entry_id}` is registered at line 199, before `/summary` at 244 — FastAPI matches in order, so the dynamic route always wins |
| `POST /v1/models/{model_id}/warmup` | unreachable | `{model_id}` cannot contain `/`; every real HF id (`org/name`) 404s. Also 200s a *nonexistent* model id |
| `GET /api/models/registry/{model_id}` | unreachable | same slash-in-path-parameter limitation |

## Hardcoded / fake behaviour

| Command | What it claims | What it does |
|---|---|---|
| `benchmark live` | streams "live" TTFT/tok-s metrics | fabricates them with `random.gauss`, never contacts the coordinator |
| `config webhook test` | "Test sent to `<url>`" | delivers nothing — `test.ping` isn't in the event enum, `dispatch()` only enqueues, and the worker never starts. Verified: a local receiver got 0 requests |
| `config webhook list/unregister` | manages webhook targets | per-process in-memory store; a registration is invisible to the next invocation |
| `model compress --target int8` | compresses the model | "Compression pipeline not available… Saving model as-is" — flag is inert, output is an unmodified copy |
| `router` | (subcommand group) | help text says subcommands "are not implemented"; always exits 2 |

## Commands that silently die

- `system coordinator` — logs "Coordinator started on port 50999", then **exits 0 by itself in ~16 s** without ever opening a socket. `system api` by contrast stays up.
- `system observe` — prints "✓ Metrics endpoint: http://localhost:9191/metrics" and returns; the server runs on a daemon thread and dies with the process. The port is dead after exit.

## API key rotation breaks the running server

`.distllm/api_key` changed **three times** during testing. Any process that boots without a pinned `API_KEYS` (`cluster start`) generates a new admin key and overwrites the file — while the already-running server keeps the old key in memory. The result is a silent 401 on every command until restart. This cost two agents real time during this session; the key that worked was the one captured before the first `cluster start`.

## Systemic: the circuit breaker counts gated features as failures

A fast sweep of deliberately-503 default-disabled routes (exchange, marketplace, federated, plugins) trips `CircuitBreakerMiddleware` and 503s *unrelated* routes for ~30 s. It counts feature-gated 503s alongside real failures.

## Test-quality findings

- **52 tests assert nothing.** They are "does not raise" smoke tests that would pass against a silently no-op implementation. 34 are in `test_cli_modules.py`, 11 in `test_chat.py`. `test_chat_clear_resets_conversation` doesn't check that the conversation was reset.
- **26 xpassed streaming tests** are marked xfail against a "known source bug" that no longer reproduces — stale markers that would fail under `xfail_strict`.
- **3 SDK tests** call async `close()` without awaiting it, so `close()` is never actually exercised.

## Verified working (the good news)

- **Chat**: `/v1/chat/completions` blocking **and** streaming (real SSE, `data: [DONE]`), 53.7 tok/s from the CLI, ~12.5 tok/s from the benchmark.
- **Embeddings**: accepts both `input:"hello"` and `input:["a","b"]` with correct cardinality — the fix from the previous session holds.
- **Batch**: submit → status → stream → cancel all work end-to-end.
- **Files**: full multipart CRUD with correct 404s.
- **Fine-tuning**: a job actually ran to `succeeded` with a `fine_tuned_model` id.
- **Auth**: `/v1/auth/{token,refresh,revoke}` validate correctly; OAuth2 CSRF state is enforced.
- **Health**: `/health`, `/ready`, `/live`, `/metrics` all live; `/ready` only breaks once a node registers.
- **CLI that really works**: `chat`, `cluster start` (real coordinator + API), `cluster autopsy` (real zip), config backup create/restore/delete, quota set/import/export/invoice, `security cert create/renew/revoke` (files really appear and disappear), `daas serve` + `daas benchmark`, `tune cache`, `tune quantize`, `system notify send`, `selftest`, `model search` (live HF).
- **All 1306 API and 510 CLI/SDK tests pass.**

---

## Recommended fix order

1. **The path drift** — 8 commands, mechanical fix, highest user-visible impact.
2. **The 10 missing-auth commands** — one-line change each to use `http_client.auth_headers()`.
3. **The 4 API 500s** — all four are one-line attribute/type bugs with exact locations above.
4. **Leaderboard route shadowing** — move the `/{entry_id}` decorator below the static routes.
5. **Key rotation** — either persist the key the server loaded, or reload it, or refuse to overwrite while a server is running.
6. **Fake-success paths** — `benchmark live`, `config webhook test`, `status`, `slo-report`, `tune batch` should fail loudly rather than print fabricated data.
7. **Test quality** — make the 52 no-assert tests assert their output.

## Artifacts

Six detailed reports with every command, status code, and response body:
`.tmp-testreports/{api-suite,cli-suite,api-live-core,api-live-mgmt,cli-live-read,cli-live-write}.md`
