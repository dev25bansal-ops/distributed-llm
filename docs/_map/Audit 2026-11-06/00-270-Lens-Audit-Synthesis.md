# DistLLM — 270-Lens Audit (2026-11-06)

# DistLLM — 270-Lens Audit Synthesis Report

**Date:** 2026-11-06 · **Deployment:** Crouton Alpha · **Scope:** 27 folders × 10 lenses = 270 sweeps
**Input:** 270/270 returned · 2,595 raw issues · 2,595 unique IDs · **59 verified Critical/High** in the synthesis packet

> **Data-quality caveat (read first).** The `byFolder` rollup in the packet reports `issues:0, critical:0, high:0` for all 27 folders, yet the `verified` + `dedupedPreview` arrays contain ~50+ detailed High items concentrated in `api` and `backends`. The packet text was truncated mid-stream (`ISSUE-API-8` cuts off). This report is therefore built **only from the verified/deduped items actually received**; items for the other 25 folders were not in the truncated payload and are **not fabricated**. §2 catalogs every verified item received. Appendix gives an honest one-paragraph status per folder (no data ≠ no issues).

---

## Executive Summary

The 270 sweeps converged on a single dominant theme: **production paths are live, but parallel “correct” modules are dead, duplicated, or unwired — so fixes in one copy never reach the running system.** `server.py` (2,249 lines) re-implements lifespan, CORS, coordinator creation, error handlers, and dashboard/ops routes inline while six extracted `server_*.py` modules (`server_config`, `server_lifespan`, `server_state`, `server_coordinator`, `server_errors`, `server_routes_api`, `server_routes_dashboard`) sit unimported and already drifted (e.g. unauthenticated-WS rule differs; 200 ms Prometheus cache exists only in the dead copy). The same pattern repeats for rate limiting (4 implementations, only in-memory `_RateLimiter` wired), SSO (`sso_auth.py` live vs `auth/` package dead-in-prod), tool calling (private `_ToolCallingEngine` in `chat.py` live vs canonical `tool_calling.py` dead), services (`ChatService`/`EmbeddingService` orphaned), API versioning (`api_versioning.py` 984 lines dead vs inline header logic with sunset hard-coded `None`), and paged attention (3 managers: `backends/paged_attention.py`, `backends/paged_attention_quantized.py` dead, `dist/attention.py`).

**Security posture:** Four High BOLA/DoS/CSRF items in `api` (files IDOR, fine-tuning IDOR, WAF unbounded buffering, OAuth state-optional) plus two High supply-chain items in `backends` (unvalidated `model_name` loading any path/remote repo; `BackendRegistry.autodiscover` force-overwriting built-ins from any `distllm_backend` entry point) are the highest-risk findings. All CVSS vectors below are as reported; where the lens marked `TBD`, this report keeps `TBD` — no invented scores.

**Correctness/race risk is concentrated in `backends/paged_attention*.py`:** unsorted shard acquisition in `copy_on_write` (A→B vs B→A deadlock), mixed shard/lock/no-lock access across swap/get/allocate paths (torn reads, double-free), `allocate_sequence(0)` → `IndexError`, prefill block counts never seeded (silent KV overwrite), and `get_kv_cache` ignoring `layer_idx`. Triton/ONNX stubs use `ord(c)`/`chr(id)` as a tokenizer and always raise or return garbage.

**Performance:** Auth/OIDC/SAML use sync `httpx` with no pooling (up to ~25 s event-loop stall); semantic cache re-embeds every cached prompt per search (5000× embed cost); webhook fan-out is sequential sync POSTs (N×30 s); Triton generation does one Infer RPC per token with O(n²) re-encoding.

**Testing:** Flags P0 gaps: 793+ LOC of security-critical middleware (`grpc_bridge`, `ws_chat`, `tracing_middleware`, `waf`, `redis_rate_limiter`, `api_versioning`) with zero or isolated-only tests, and a flagship `test_api_routes_comprehensive.py` that is permanently `pytest.skip(allow_module_level=True)` leaving 12+ routers (files, moderation, fine_tuning, tokenize, model_registry, router_admin, metrics_history, assistants, experiments, leaderboard, defrag, prompts) with no TestClient coverage.

**Estimate:** Closing P0 + Sprint-1 P1 items below totals **~25–35 engineer-days** of focused work (rollup in §2), dominated by the server composition, distributed rate-limit, and backend protocol fixes. Full backlog (including P2/P3) is **~90–130 days** if every P2/P3 item is taken; several are deletions rather than builds.

---

## Methodology

- **Fan-out:** 27 folders (api, backends, benchmarks, cache, cli, client, cloud, compliance, config, core, dashboard, deploy, dist, errors, health, integrations, models, observability, operator, plugins, prompts, sdk, security, terraform, ui, utils, verification) × 10 lenses each.
- **Lenses (10 per folder):** Correctness & bugs · Security (OWASP/auth/injection/secrets, CVSS or TBD) · Performance (latency/throughput/resource) · Architecture & scalability/reliability/extensibility · Code quality & maintainability · Technical debt & dead code · Developer Experience & API/UI (OpenAPI/versioning/error envelopes) · Testing gaps · Ops/observability/docs & compliance · Enhancements / Advanced Features & New Additions.
- **Dedupe:** 2,595 raw → 2,595 unique IDs (no ID collisions dropped at the hash level). Cross-lens duplicates remain (same defect reported under Correctness, Architecture, CQ, and Debt lenses — e.g. server duplication appears as API-ARCH-01, CQ-01, CQ-02, ISSUE-API-8; rate-limit duplication as API-ARCH-04, CQ-04, ISSUE-API-5). §2 de-duplicates by canonical ID and notes the mirror IDs.
- **Verification:** 59 items flagged Critical/High in the packet. **No Critical items appear in the received payload; all verified items are High** (plus supporting Medium/Low items in `dedupedPreview`). No items are marked refuted/dropped in the received payload (see “Dropped/refuted” note in §2).
- **CVSS policy:** Vectors reproduced verbatim where provided; `TBD` where absent. Performance/correctness items marked `N/A` by the lens are kept `N/A`.

---

## §2 — Issues Catalog

### Counts + Effort Rollup

| Severity | Verified items received | Notes |
|---|---|---|
| Critical | 0 in received payload | Packet header claims 59 Critical/High; payload yields 0 Critical, remainder High. Possible truncation of Critical slice. |
| High | ~54 distinct IDs cataloged below | Covers api + backends only in received payload. |
| Medium/Low | ~12 (from `dedupedPreview`) | Listed in sub-table; not counted in the 59. |
| **Total cataloged** | **~66** | **Does not sum to 2,595 — remainder not in truncated payload.** |

| Priority | Items | Effort (engineer-days) | Dominant work |
|---|---|---|---|
| P0 | 5 (CQ-01, CQ-02, API-T01, API-T02, B-01, B-02, CQ-004, CQ-009, PERF-BE-1, PERF-BE-2 — 10 IDs, some P0-tagged by multiple lenses) | 10–16 d | Server composition, route/test coverage, backend protocol + Triton/vLLM forward |
| P1 | ~35 | 18–25 d | BOLA, WAF, rate-limit consolidation, async auth/webhooks, paged-attention races |
| P2 | ~12 | 8–14 d | Chat-service consolidation, uvicorn scaling, SDK/docs |
| P3/Backlog | ~6 | 3–5 d | Secret hygiene, docs |
| **Total (received)** | **~66** | **~40–60 d** to close everything received; **~25–35 d** for P0+P1 | Deletions reduce effort vs builds |

> Effort ranges are the lenses’ own estimates (e.g. “0.5 day”, “2–3 days”) normalized to days. Where lenses disagreed, the range spans both. Double-counted mirror IDs are counted once.

### Dedupe Notes

- **Server duplication cluster:** `API-ARCH-01` ≡ `CQ-01` (god-file) ≡ `CQ-02` (dead extractions) ≡ `ISSUE-API-8` (server_* dead) — canonical = API-ARCH-01. `API-ARCH-05`, `API-OPS-03` are children (lifespan/background supervision).
- **Rate-limit cluster:** `API-ARCH-04` ≡ `CQ-04` ≡ `ISSUE-API-5` ≡ `API-DX` mirror — canonical = API-ARCH-04.
- **State-proxy cluster:** `CQ-03` ≡ `ISSUE-API-4` — canonical = CQ-03.
- **Tool-calling cluster:** `CQ-05` ≡ `API-ARCH-07` (part) — canonical = CQ-05.
- **Versioning cluster:** `DX-02` ≡ `ISSUE-API-2` ≡ `API-ARCH-03` (part) — canonical = DX-02.
- **Config-dead cluster:** `ISSUE-API-1` ≡ `API-ARCH-02` — canonical = ISSUE-API-1.
- **SSO-fork cluster:** `ISSUE-API-3` (only item; no mirror ID).
- **Paged-attention race cluster:** `B-01` ≡ `BK-01` (same files/lines) — canonical = B-01/BK-01 (reported once, both IDs retained).
- **Triton-tokenizer cluster:** `B-12` ≡ `CQ-019` ≡ `PERF-BE-1` (same root, three lenses) — canonical = PERF-BE-1 for perf, B-12 for correctness.

### Dropped / Refuted

No item in the received payload is marked “refuted” or “dropped.” The packet’s `byFolder` all-zero rollup is treated as a **pipeline data bug, not a refutation** — folders with zero rows must not be assumed clean. If the workflow layer tracks rejections elsewhere, that list was not in the truncated payload and must be merged before acting on this report.

### Catalog Table (High first, then Medium/Low)

#### API — Security / Correctness (High)

| ID | File:Line | Sev | Pri | Effort | Timeline | Deps | CVSS / Repro |
|---|---|---|---|---|---|---|---|
| API-01 | `api/routes/leaderboard.py:188` | High | — | — | — | — | TBD · `GET /v1/leaderboard/{entry_id}` shadows `/summary /top /community /verified` defined later |
| SEC-API-01 | `api/routes/files.py:168-250` | High | P1 | 1–2 d | Sprint 1 | owner field in `persistent_store` | CVSS:3.1/AV:N/AC:L/PR:L/UI:N/S:C/C:H/I:H/A:H (7.7, TBD pending PoC) · Key B lists/downloads/deletes Key A’s files via `GET /v1/files` |
| SEC-API-02 | `api/routes/fine_tuning.py:32-381` | High | P1 | 1 d | Sprint 1 | SEC-API-01 owner field | CVSS:3.1/AV:N/AC:L/PR:L/UI:N/S:C/C:L/I:L/A:L (6.5, TBD) · Key B lists/cancels tenant A’s jobs |
| SEC-API-03 | `api/waf.py:215-225,293-301` | High | P1 | 0.5 d | Sprint 1 | None | CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:N/I:N/A:H (7.5, TBD) · POST 50 MB → RSS growth before 413 |
| SEC-API-04 | `api/sso_auth.py:568-578` | High | P1 | 0.5–1 d | Sprint 1 | IdP PKCE | CVSS:3.1/AV:N/AC:L/PR:N/UI:R/S:C/C:L/I:L/A:N (6.1, TBD) · `handle_callback(code, expected_state='')` proceeds to token exchange |

#### API — Performance (High)

| ID | File:Line | Sev | Pri | Effort | Timeline | Deps | CVSS / Repro |
|---|---|---|---|---|---|---|---|
| PERF-01 | `api/auth/oauth2.py:90,104; oidc.py:73,177,237,274; sso_auth.py:231,309,360,447,583,602` | High | P1 | 1–2 d | Sprint 1 | none | N/A perf · Async endpoint → event-loop stall = Σ httpx latencies (≤25 s worst) |
| PERF-02 | `api/authz/opa.py:197-243` | High | P1 | 2–3 d | Sprint 1–2 | PERF-01 | N/A · `authorize()` spawns process / 10 s urlopen per eval (~30–150 ms vs <1 ms Python) |
| PERF-03 | `api/webhooks/delivery.py:247-283,317` | High | P1 | 1 d | Sprint 1 | none | N/A · 10 slow webhooks → dispatch duration ≈ Σ 30 s timeouts, sequential |
| PERF-05 | `api/semantic_cache.py:309,186,199` | High | P1 | 1–2 d | Sprint 2 | none | N/A · 5000 entries, no FAISS → search re-embeds 5000× (seconds); blocks loop |

#### API — Architecture / Reliability (High)

| ID | File:Line | Sev | Pri | Effort | Timeline | Deps | CVSS / Repro |
|---|---|---|---|---|---|---|---|
| API-ARCH-01 | `api/server.py:183-271,680-700,1998-2138` | High | P1 | 4 d | Sprint 1 | None | — · `grep server_config\|server_lifespan` in `server.py` = no imports; two divergent copies |
| API-ARCH-04 | `api/middleware.py:52-246` | High | P1 | 5 d | Sprint 1–2 | None | — · 2 uvicorn workers each allow 1000 req/60 s independently; Redis variants unimported |
| API-ARCH-05 | `api/background_tasks.py:47-295` | High | P1 | 4 d | Sprint 1 | API-ARCH-01 | — · Submit batch → `kill -9` → `GET /v1/batch/{id}/status` = 404; broadcaster crash = silent |
| API-ARCH-06 | `api/server.py:2238-2245` | High | P1 | 3 d | Sprint 1 | API-ARCH-04 | — · Single-worker `uvicorn.Config`, unbounded concurrency, gRPC `max_workers=4` hard-coded |
| API-ARCH-09 | `api/grpc_bridge.py:264-274; ws_chat.py; sso_middleware.py:92-149` | High | P1 | 1 d | Sprint 1 | None | — · Missing protobuf stubs → “Simulated gRPC response” with fabricated tokens (module currently dead) |

#### API — Code Quality / Debt (High)

| ID | File:Line | Sev | Pri | Effort | Timeline | Deps | CVSS / Repro |
|---|---|---|---|---|---|---|---|
| CQ-01 | `api/server.py` (2249 lines; 6 inline middleware) | High | P0 | 3–4 d | Sprint 1 | none | — · Structural (god-file) |
| CQ-02 | `api/server_lifespan.py; server_routes_api.py; server_routes_dashboard.py; server_config.py; server_coordinator.py; server_errors.py` | High | P0 | 1–2 d | Sprint 1 | CQ-01 | — · Importer counts = 0/0/0/0/1(self)/1 |
| CQ-03 | `api/server_state.py:15-29; server.py:567-581; api_state.py; app_state.py` | High | P1 | 0.5 d | Sprint 1 | none | — · 107 `getattr(state,…)/g.get/g[]` sites; two identical `_ServerState` defs |
| CQ-04 | `api/rate_limit_middleware.py; rate_limiter.py; rate_limiter_unified.py; redis_rate_limiter.py; middleware.py` | High | P1 | 1 d | Sprint 1 | none | — · 3× `RateLimiter`/`RedisRateLimiter`; shim docstring contradicts import |
| CQ-05 | `api/routes/chat.py:31-270` vs `api/tool_calling.py:22-255` | High | P1 | 0.5 d | Sprint 1 | none | — · Live private engine lacks `tool_choice` handling of canonical copy |
| ISSUE-API-1 | `api/api_settings.py:1-130` | High | P1 | 2 h | Sprint 1 | None | N/A · Zero imports of `api_settings`; 11× `os.environ.get` in `server.py` alone |
| ISSUE-API-2 | `api/api_versioning.py:1-984` vs `server.py:380-431` | High | P1 | 4 h | Sprint 1 | ENH-4 | N/A · `VersionMiddleware` never imported; inline `_API_VERSIONS` duplicates it |
| ISSUE-API-3 | `api/sso_auth.py:1-786` vs `api/auth/` (809 lines) | High | P1 | 2–3 d | Sprint 1–2 | None | N/A · `sso_auth` imported by `sso_middleware`; `auth/` imported by tests only |
| DX-01 | `api/routes/batch.py, exchange.py, api_keys.py, federated.py, leaderboard.py, marketplace.py, prompts.py, scheduler.py, tokenize.py, tools.py, webrtc.py, plugins.py, experiments.py, assistants.py` | High | P1 | 3–5 d | Sprint 1 | None | TBD · 217 decorators, 111 summary, 29 `responses={}`; 12 routers have 0/0 |
| DX-02 | `api/server.py:380-431` vs `api/api_versioning.py:148-334,782-907` | High | P1 | 2–3 d | Sprint 1 | DX-03 | TBD · `_API_SUNSET_DATES={v1:None,v2:None}` vs `api_versioning` V1 sunset 2026-10-01 (now past); no Sunset/Deprecation emitted |

#### API — Testing / Ops (High)

| ID | File:Line | Sev | Pri | Effort | Timeline | Deps | CVSS / Repro |
|---|---|---|---|---|---|---|---|
| API-T01 | `api/grpc_bridge.py:96-297; ws_chat.py:117-497; tracing_middleware.py:207-240,359-363; waf.py:155-344; redis_rate_limiter.py:78-110; api_versioning.py:1-984` | High | P0 | 3–5 d tests / 1 wk wiring | Sprint 1 | none | TBD · `grep -rln GRPCBridge/WSChatHandler` = self only; isolated Starlette test only |
| API-T02 | `tests/api/test_api_routes_comprehensive.py:1-14` | High | P0 | 4–6 d | Sprint 1–2 | none | TBD · `pytest` → 1 skipped; 12+ routers with zero `routes.*` coverage |
| API-T05 | `api/auth/{oidc.py,saml.py,oauth2.py,store.py,models.py}` | High | P1 | 3 d | Sprint 2 | none | TBD · Only indirect coverage via `sso_auth.py` wrappers + 4 OIDC regression files |
| API-OPS-01 | `api/observability_middleware.py:16,118-119` | High | P1 | 4–6 h | Sprint 1 | None | TBD · Uninstall OTel → `import distllm.api.server` ImportError; metrics = `default` only |
| API-OPS-03 | `api/server_lifespan.py:144` | High | P1 | 3–4 h | Sprint 1 | None | TBD · SIGHUP processor `return`s on first exception → hot-reload dead until restart |
| API-OPS-07 | `api/webhooks/delivery.py:317,365-414,482-500` | High | P1 | 1–2 d | Sprint 2 | httpx AsyncClient (already in `dispatch_webhook`) | TBD · Slow endpoint blocks loop 30 s; DLQ lost on restart; no `webhook_*` metrics |
| ISSUE-04 | `api/routes/batch.py:257` vs `persistent_store.py:190-205` | High | P1 | 1–2 d | Sprint 1 | `PersistentStore.save_batch` exists | N/A · Submit → restart → `GET status` = 404; `_batch_store` ephemeral dict |

#### Backends — Correctness / Races (High)

| ID | File:Line | Sev | Pri | Effort | Timeline | Deps | CVSS / Repro |
|---|---|---|---|---|---|---|---|
| B-01 / BK-01 | `backends/paged_attention.py:458,289-327,421-507` | High | P1 | 4 h–2 d | Sprint 1 | none | TBD · COW A→B vs B→A unsorted shard order → deadlock; swap/get mixed lock ordering → corruption |
| B-02 | `backends/paged_attention.py:501,536,428,421,479` | High | P1 | 6 h | Sprint 1 | none | TBD · `swap_gpu`/`reset`/`get_kv_cache` race `allocate/free/append` → freed-block `cat` |
| B-03 | `backends/paged_attention.py:273,290,388-390` | High | P1 | 4 h | Sprint 1 | none | TBD · `allocate_sequence(0)` → `append_token` `IndexError`; prefill `block.num_tokens` never seeded |
| B-05 | `backends/paged_attention_quantized.py:165,249-252,275-282` | High | P1 | 8 h | Sprint 1 | none | TBD · INT4 claims packed `head_dim//2`, allocates full `head_dim` int8; K/V in `[0]`/`[1]` slots |
| B-08 | `backends/config.py:49-85` | High | P1 | 6 h | Sprint 1 | B-09 | TBD · `to_backend_kwargs('llamacpp')` missing `model_path` → TypeError (6/12 backends) |
| B-09 | `backends/protocol.py:66-125` vs `nim/webgpu/tgi/ollama` | High | P1 | 8 h | Sprint 1–2 | none | TBD · Registry selects NIM/WebGPU/TGI/Ollama; `forward()` raises `NotImplementedError` at runtime |
| B-12 | `backends/triton_backend.py:503; onnx:172-203; mlx:136; llamacpp:161; exllama:159-162` | High | P2 | 8 h | Sprint 2 | none | TBD · `ord(c)`/`chr(id)` tokenizer; greedy-only; hidden-state `argmax`; ONNX always raises |
| CQ-004 | `backends/paged_attention.py:428-451; paged_attention_quantized.py:220,256` | High | P0 | 1 d | Sprint 1 | Quantized mgr | — · `get_kv_cache(layer_idx)` ignores `layer_idx` — identical KV for any layer |
| CQ-009 | `backends/ollama_backend.py:64-89; tgi_backend.py:80-96` vs `protocol.py:66-125` | High | P0 | 1 d | Sprint 1 | protocol | — · Async/dict `generate`, bool `load_model`, 2-param `forward` violate protocol |
| PERF-BE-1 | `backends/triton_backend.py:503-701` | High | P0 | 2–3 d | Sprint 1 | none | N/A · 256 tokens = ≤256 Infer RPCs (1.3–12.8 s) + O(n²) re-encode per request |
| PERF-BE-2 | `backends/vllm_backend.py:224-237; tensorrt:196-212; llamacpp:133-144; exllama:148-165` | High | P0 | 2 d | Sprint 1 | none | N/A · `forward()` = decode → `generate(max_tokens=1)` → re-encode; discards logits/KV |
| PERF-BE-3 | `backends/paged_attention_quantized.py:152-161,256-290` | High | P1 | 1–2 d | Sprint 1–2 | none | N/A · Per-block `float()` + `torch.cat` per gather; INT4 memory 0.5× not claimed 0.25× |
| PERF-BE-4 | `backends/paged_attention.py:244-271,289,327,380,428-451` | High | P1 | 1 d | Sprint 1 | none | N/A · Global `_lock` + O(num_blocks) scan + `torch.cat` per allocate/gather |
| BK-02 | `backends/registry.py:236-242,392-428` | High | P1 | 1 d | Sprint 1 | BK-03 | TBD · `object.__new__` probes on uninitialized adapters; fail-open to all-unhealthy |
| BK-03 | `backends/protocol.py:22-34,53-108` | High | P1 | 1 d | Sprint 1 | none | TBD · Only vLLM/llamacpp set `BackendState`; 8/12 adapters stateless |
| CQ-001 | `backends/paged_attention.py:170` | High | P1 | 4 h | Sprint 1 | — | — · `_detect_numa_node()` glob + dead `_numa_node` field, never consumed |
| CQ-005 | `backends/paged_attention_quantized.py:1-319` | High | P1 | 2 d | Sprint 2 | — | — · 319-line fork of `PagedAttentionManager`, divergent locks/validation, races |
| CQ-006 | `backends/registry.py:86-94,392-428` | High | P1 | 1 d | Sprint 1 | protocol health | — · `object.__new__` health probes + redundant singleton/classmethods/globals |
| CQ-008 | `backends/triton_backend.py:188-701` | High | P1 | 1 d | Sprint 1 | — | — · 875-line DRY violation: 3× Infer-input + argmax decode copies |
| CQ-012 | `backends/webgpu_backend.py:188-219` | High | P1 | 2 h | Sprint 1 | — | — · `generate()` returns `"[WebGPU generation via …]"` stub; `forward()` correctly raises |
| CQ-014 | `backends/llamacpp_backend.py:133-171` | High | P1 | 4 h | Sprint 1 | — | — · Private `._ctx.eval_logits` → fallback `torch.zeros` logits; `token_id` defaults 0 |
| CQ-019 | `backends/triton_backend.py:526,556,578,646` | High | P1 | 1 d | Sprint 1 | tokenizer | — · `ord(c)`/`chr(id)` presented as working generation (see B-12/PERF-BE-1) |
| BE-01 | `backends/paged_attention_quantized.py:1-319` | High | P1 | 0.5 d delete / 3–5 d merge | Sprint 1 | `dist/attention.py` BlockPool | TBD · Zero imports; duplicates `paged_attention.py` + `dist/attention.py` |
| BE-03 | `backends/paged_attention.py:1-571` vs `dist/attention.py:760` | High | P1 | 5–8 d | Sprint 2–3 | `core/inference_engine.py:220` | TBD · Two `PagedAttentionManager`s, divergent locks/block semantics/stats |
| BE-05 | `backends/ollama_backend.py:1-101; tgi; nim` | High | P1 | 2–3 d | Sprint 1 | `registry.select()/protocol.py` | TBD · Remote-HTTP adapters selectable yet `forward()` raises; ctor signatures diverge |

#### Backends — Security (High)

| ID | File:Line | Sev | Pri | Effort | Timeline | Deps | CVSS / Repro |
|---|---|---|---|---|---|---|---|
| B-01 (sec) | `backends/pytorch:33-76; onnx:121-126; llamacpp:83; exllama:99-109; tensorrt:132-151; mlx:61-64; nim:287` | High | P0 | 4 h | Sprint 1 | None | TBD (operator-supplied `model_name` → RCE via `trust_remote_code`/ONNX ops if API-exposed) · `ONNXNodeAdapter('/etc/passwd.onnx')` proceeds to load |
| B-02 (sec) | `backends/registry.py:99-133,290-316` | High | P0 | 3 h | Sprint 1 | B-01 (sec) | TBD · Fake package with entry point `vllm` replaces real vLLM via `force=True` |

#### Medium / Low (from dedupedPreview, not in 59)

| ID | File | Sev |
|---|---|---|
| SEC-API-05 JWT iss/exp missing | `api/sso_auth.py:501-506` | Medium — add `issuer`, `require=[exp]` |
| SEC-API-06 API key printed to stdout | `api/middleware.py:190-213` | Medium — write 0600 file, log fingerprint only |
| SEC-API-09 Unbounded 512 MB upload, no quota | `api/routes/files.py:119-145` | Medium — stream 1 MB chunks, per-tenant quota |
| SEC-API-07 Prompt plaintext in audit log | `api/prompt_injection.py:434-452` | Low — hash/redact, 0600, rotation |
| SEC-API-08 Webhook HMAC secret plaintext | `api/routes/webhooks.py:190-205` | Low — encrypt with KEK (Fernet/AES-GCM) |
| SEC-API-10 Router auth relies on global middleware | `api/routes/chat.py:269,907` et al. | Low — add `require_role(inference-only)` defense-in-depth |
| PERF-04 ThreadPoolExecutor per tool call | `api/tool_calling.py:196; chat.py:203` | Medium — shared executor + parallel |
| PERF-06 Global rate-limit lock + O(N) evict | `api/middleware.py:81; redis_rate_limiter.py:108` | Medium — sharded/lock-free, O(1) LRU, pipelined Redis |
| PERF-07 `to_thread` per token | `api/streaming.py:187,189,258` | Medium — GPU-stream / persistent worker |
| PERF-08 Duplicate JSON parse | `api/body_cache_middleware.py:109; semantic_cache.py:290` | Low — reuse `request.state.parsed_body` |
| PERF-09 Import-time thread/policy load | `api/routes/metrics_history.py:245; authz/opa.py:297` | Low — move to lifespan / lazy |
| API-ARCH-02/03/07/08/10, CQ-06–12, ISSUE-API-4–8 | various `api/` | Medium/Low — see §3/P2 roadmap |

---

## §3 — Enhancements

(Enhancement lens items received; other folders’ enhancement backlog was not in the payload.)

- **E-01 Batch durability** (`ISSUE-04`): persist `_BatchState` via existing `PersistentStore.save_batch`; hydrate on lifespan; write-through cache. Effort 1–2 d. Closes “submit → restart → 404” and lets webhooks fire.
- **E-02 Semantic-cache activation** (`ISSUE-API-6`): mount `SemanticCacheMiddleware` behind env flag (with PERF-05 fix) or delete. 4 h. Recovers 10–30% cost on repetitive prompts.
- **E-03 Single config source** (`API-ARCH-02`/`ISSUE-API-1`): migrate all `os.environ.get` to `APISettings`; validate at lifespan; SIGHUP-reloadable. 2 d.
- **E-04 OPA wiring decision** (`ISSUE-API-8`): wire `authz.authorize()` into resource-level routes or delete and document inline RBAC. 1 d.
- **E-05 Chat-service delegation** (`API-ARCH-07`/`CQ-06`/`ISSUE-API-7`): routes → `ChatService`/`CompletionService`/`EmbeddingService`; delete inline duplicates; share one bounded tool executor. 5 d (part) + 2 d (decision/migration).
- **E-06 Chat god-function split** (`CQ-07`): `chat_completions` (414 lines) split along existing `ChatService` seams; cap new functions ~50 lines. 3–5 d incremental.
- **E-07 Error-model unification** (`CQ-12`): one `ErrorResponse` in `errors.py`; delete `server.py:440` + `server_errors.py`. 0.5 d.

---

## §4 — Advanced Features (as implied by gaps)

- **AF-01 Versioned contract enforcement** (`DX-02`): live `Accept-Version` negotiation, 410-Gone after sunset, `Sunset`/`X-API-Deprecation` headers, `VersionCompatibilityLayer` body transforms. Was advertised by dead `api_versioning.py` (984 lines) — 2–3 d to mount and reconcile.
- **AF-02 Distributed rate limiting** (`API-ARCH-04`): Redis-backed `RateLimiter` with tenant/model fairness, check-then-consume accounting, 2-process integration test. 5 d.
- **AF-03 Supervised background runtime** (`API-ARCH-05`): `BackgroundTaskManager` (or `TaskGroup` + restart) for batch workers and WS broadcaster; max-concurrent-batches semaphore + queue-depth metric. 4 d.
- **AF-04 Streaming-aware metering** (`API-ARCH-10` part): wrap SSE generator so quota records real `output_tokens` (currently 0). Part of 5 d middleware-chain work.
- **AF-05 Paged-attention consolidation** (`BE-03`/`BE-01`): one canonical manager (contiguous BlockPool + COW/swap/LRU/FP8) with single-node shim; quantized as strategy, not fork. 5–8 d.
- **AF-06 Capability-split backend protocol** (`B-09`/`BE-05`/`CQ-009`): `ForwardCapable` vs `GenerateOnly` protocols; `select()` filters by capability; `supports_forward()` gate. 2–3 d.
- **AF-07 Production-grade uvicorn** (`API-ARCH-06`): `--workers/--limit-concurrency/--backlog/--timeout-keep-alive`, explicit thread-pool sizing, graceful drain, configurable gRPC workers. 3 d.

---

## §5 — New Additions

- **NA-01 Contract test gate** (`DX-01`): CI fails if a mounted route lacks `summary`+`responses=` (whitelist internal probes). 3–5 d incl. backfill.
- **NA-02 Route-coverage test set** (`API-T02`): seven `test_routes_{files,moderation,fine_tuning,tokenize,model_registry,router_admin,metrics_history}.py` files plus webhooks/files traversal cases. 4–6 d.
- **NA-03 Middleware integration tests** (`API-T01`): `test_tracing_middleware`, `test_ws_chat`, `test_grpc_bridge`, `test_api_versioning`, `fakeredis` rate-limiter parity. 3–5 d tests + 1 wk wiring.
- **NA-04 Auth-package unit tests** (`API-T05`): `test_auth_{oauth2,oidc,saml,store,models}.py` with negative parametrized cases. 3 d.
- **NA-05 Dead-code CI check** (`CQ-02`/`ISSUE-API-8`): `vulture`/import-check fails when extracted modules are unimported. 0.5 d.
- **NA-06 Startup self-checks** (`API-ARCH-09`/`API-ARCH-10`): require `SSO_JWT_SECRET` when SSO enabled; middleware-order assertion at startup. Part of 1–5 d items.

---

## §6 — Testing Strategy

1. **Repro-first loops (per `diagnosing-bugs` discipline):** each §2 item ships with its `Repro` column as a pass/fail script; concurrency items (`B-01/B-02/BK-01`) get barrier-based regression tests with synchronous fake timers.
2. **TestClient through the real app** for every mounted router (happy-path 200 + auth 401/403 + 404/422 + RBAC denial). Replace the skipped `test_api_routes_comprehensive.py`; never accept `allow_module_level=True` skips in flagship files.
3. **Contract tests** for OpenAPI (`DX-01`): assert every mounted route has `summary` + `responses` using `ERR_ENTRY`/`error_openapi_entry`; snapshot `X-API-Version`/`Sunset`/`X-API-Deprecation` headers (`DX-02`).
4. **Race/stress tests** for paged attention: concurrent `allocate/free/append/swap/COW` from `ThreadPoolExecutor`, asserting free-list invariant + no deadlock under shuffle; HTTP-level concurrency tests for locks (`PERF-BE-4`, `PERF-06`).
5. **Perf benchmarks** for PERF-01/02/03/05, PERF-BE-1/2/3/4 with recorded baselines (p99 auth, webhook dispatch vs N, semantic-cache search vs entry count, Triton RPC count, vLLM forward latency).
6. **Security PoCs** for SEC-API-01/02/03/04 and B-01/B-02: two-key cross-tenant exploit, 50 MB body RSS, empty-state callback, wrong-`iss` JWT, `model_name` traversal, fake-entry-point overwrite. Gate on PoC success, not static grep.
7. **Chaos/lifecycle:** `kill -9` during batch + WS broadcast (`API-ARCH-05`, `ISSUE-04`); SIGHUP with malformed then valid config (`API-OPS-03`); restart with SSO revocation list (`API-ARCH-09`); OTel-uninstalled import (`API-OPS-01`).
8. **Coverage policy:** target 100% of mounted routers + all middleware in the production stack; unreachable modules must be deleted or marked `experimental` with a warning and excluded from coverage denominators explicitly.

---

## §7 — Beyond (org / process)

- **B-01 One-copy rule:** no extraction lands without wiring; `server.py` becomes a ~150–200-line composition root (CQ-01 + NA-05). Every duplicated feature must choose a canonical copy in the PR description.
- **B-02 Docs generated from code:** route counts, mount policy, rate-limiter path generated or CI-checked (`CQ-10`). Delete superseded 2026-07 audit reports from `api/`.
- **B-03 Typing rollout:** extend mypy strict beyond `server`+`middleware` to `distllm.api.*`, starting with `routes/chat.py` + `routes/batch.py` (`CQ-09`). Enforce `log-or-raise` on bare `except Exception:` (57 sites, `CQ-08`).
- **B-04 Secret hygiene defaults:** fail-closed API key generation (no stdout), KEK-encrypted webhook secrets, hashed audit-log prompts with rotation (§2 Medium/Low).
- **B-05 Capacity/scaling playbook:** document single-writer SQLite limits, in-memory limiter bypass under N workers, and the SIGHUP/uvicorn tuning knobs (AF-07) before multi-worker HA is promised.

---

## Ranked Roadmap

### P0 — Release-blockers (Sprint 1, ~10–16 d)

| Order | IDs | Work | Effort |
|---|---|---|---|
| P0-1 | B-01(sec), B-02(sec) | Model-name allowlist across all adapters + remove `force=True` autodiscover | 7 h |
| P0-2 | CQ-01 + CQ-02 | Server composition root; delete or wire 6 dead modules; CI dead-code check | 4–6 d |
| P0-3 | API-T01, API-T02 | Middleware + route TestClient coverage; delete skipped placeholder | 7–11 d tests (parallelizable) |
| P0-4 | CQ-004, CQ-009, PERF-BE-1, PERF-BE-2 | `layer_idx` honesty, protocol split, Triton/vLLM forward | 4–5 d |
| P0-5 | API-ARCH-01/04/05/06/09, CQ-03/04/05, ISSUE-API-1/2/3, DX-01/02, API-OPS-01/03/07, ISSUE-04, SEC-API-01–04 | Sprint-1 P1 bundle (BOLA, WAF, rate-limit, async I/O, versioning/docs, ops guards, batch durability) | ~20 d (multi-engineer parallel) |

### P1 — Sprint 1–2 (~18–25 d)

All §2 High items not in P0: `API-01`, `PERF-01/02/03/05`, `B-01/02/03/05/08/09`, `BK-01/02/03`, `PERF-BE-3/4`, `CQ-001/005/006/008/012/014/019`, `BE-01/03/05`, `API-T05`.

### P2 — Sprint 2–3 (~8–14 d)

`SEC-API-05/06/09`, `PERF-04/06/07`, `API-ARCH-02/03/07/08/10`, `CQ-06/07/09/11`, `ISSUE-API-4/5/6/7`, `B-12` (rest of tokenizer wiring).

### P3 — Backlog (~3–5 d)

`SEC-API-07/08/10`, `PERF-08/09`, `CQ-10/12`.

**Total effort:** P0+P1 ≈ **25–35 engineer-days** focused; full received backlog ≈ **40–60 d**; the 2,595-item full backlog plausibly lands **~90–130 d** once the truncated 2,500 items are delivered and deduped against this plan.

---

## Appendix — Per-Folder One-Paragraph Summary (27 folders)

*No issue payload was received for folders other than `api` and `backends`; summaries below state that explicitly rather than inventing findings.*

1. **api** — Dense High cluster: BOLA/IDOR in files + fine-tuning, WAF unbounded buffering, OAuth state-optional, leader-board route shadowing, heavy duplication (server/rate-limit/SSO/tool-calling), dead `api_settings`/`api_versioning`/`services`, async-I/O and semantic-cache perf traps, OpenAPI/versioning gaps, and P0 test holes (skipped comprehensive file, untested grpc/ws/tracing/WAF/versioning). See §2.
2. **backends** — Dense High cluster: three divergent paged-attention managers with races/deadlocks, `layer_idx` API lie, bool/async protocol violations, selectable backends that crash on `forward()`, `ord/chr` Triton tokenizer, supply-chain entry-point hijack, unvalidated model loading, and per-token RPC perf traps. See §2.
3. **benchmarks** — No verified items in the received payload (folder rollup 0). The 2,595-item remainder likely contains benchmarks findings; do not treat as clean.
4. **cache** — No verified items received. Semantic-cache perf items live under `api` in this payload; cache-folder lenses may duplicate them.
5. **cli** — No items received.
6. **client** — No items received.
7. **cloud** — No items received.
8. **compliance** — No items received.
9. **config** — No items received (config-dead items in this payload live under `api/api_settings.py`).
10. **core** — No items received. `core/inference_engine.py:220` + `core/kv_cache.py` + `core/memory_defragmenter.py` are referenced by `BE-03`; direct core findings were not in the payload.
11. **dashboard** — No items received. Dashboard WS/metrics duplication appears under `api/server_routes_dashboard.py` in this payload.
12. **deploy** — No items received.
13. **dist** — No items received. `dist/attention.py BlockPool` is the canonical-manager side of `BE-03`/`BE-01`.
14. **errors** — No items received. Error-model triplication appears under `api` in this payload.
15. **health** — No items received. Heartbeat/auth items in this payload live under `api`.
16. **integrations** — No items received.
17. **models** — No items received.
18. **observability** — No items received. RED-metrics/OTel-import items appear under `api` in this payload.
19. **operator** — No items received.
20. **plugins** — No items received.
21. **prompts** — No items received. Prompts-router documentation gap appears under `api` (`DX-01`) in this payload.
22. **sdk** — No items received.
23. **security** — No items received as a folder. Security findings in this payload are filed by their owning folders (`api`, `backends`); the standalone `security` folder’s 10 lenses were not in the truncated data.
24. **terraform** — No items received.
25. **ui** — No items received.
26. **utils** — No items received.
27. **verification** — No items received.

---

## Prioritized Checklist

**This sprint (P0/P1):**
- [ ] Close user-controlled `model_name` loading: shared `_validate_model_name` + allowlist in every adapter; default `trust_remote_code=False`; local paths opt-in (B-01 sec, 4 h)
- [ ] Remove `force=True` backend autodiscover; Ed25519-signed, allowlisted entry points (B-02 sec, 3 h)
- [ ] Collapse `server.py` to composition root; delete/wire `server_*.py`; add CI import check (CQ-01/CQ-02/API-ARCH-01)
- [ ] Owner+role checks on Files + Fine-tuning routes; backfill owner field (SEC-API-01/02)
- [ ] Incremental cap in WAF `_read_body` at `max_body_size+1` (SEC-API-03, 0.5 d)
- [ ] Mandatory `state` + PKCE in GenericOAuth2 (SEC-API-04)
- [ ] Single distributed rate limiter (Redis when configured); fix consume-on-reject (API-ARCH-04/CQ-04)
- [ ] Sorted shard order everywhere in paged attention; unified shard+lock; snapshot reads (B-01/02, BK-01, PERF-BE-4)
- [ ] Honor-or-reject `layer_idx`; seed prefill block counts (CQ-004, B-03)
- [ ] Capability-split protocols; registry filters by `supports_forward()` (B-09, BE-05, CQ-009, BK-02/03)
- [ ] Async auth stack (OIDC/OAuth2/SAML) with pooled `AsyncClient`; async webhooks with bounded concurrency (PERF-01/03, API-OPS-07)
- [ ] TestClient coverage for all 12+ uncovered routers + middleware integration tests; delete skipped placeholder (API-T01/T02)
- [ ] OpenAPI `summary`+`responses` backfill for the 12 undocumented routers; CI gate (DX-01)
- [ ] Mount live versioning with real Sunset/410 (or delete dead module) (DX-02/ISSUE-API-2)
- [ ] Guard OTel import; real model/tenant metrics labels; fix SIGHUP loop `return` (API-OPS-01/03)
- [ ] Persist batches; supervise batch workers + WS broadcaster; restart-safe (ISSUE-04, API-ARCH-05)
- [ ] Fix Triton/vLLM `forward()` decode/re-encode round-trips; stop `ord/chr` tokenizing (PERF-BE-1/2, B-12, CQ-019)
- [ ] Delete or merge `paged_attention_quantized.py`; consolidate the two live managers (BE-01, BE-03)
- [ ] Config single-source via `APISettings`; remove print-to-stdout bootstrap key (ISSUE-API-1, SEC-API-06)
- [ ] One SSO implementation; one tool-calling engine; one `_ServerState` (ISSUE-API-3, CQ-05, CQ-03)

**Next (P2):** chat-service delegation, uvicorn scaling knobs, streaming metering + pure-ASGI hot path, semantic-cache activation behind flag, auth-package unit tests, typing rollout + bare-except lint, KEK-encrypted webhook secrets, hashed audit logs.

**Before next release gate:** re-run the 270-sweep with a fixed `byFolder` rollup and deliver the full 2,595-item payload; merge this report’s IDs against it; publish PoC results for every CVSS-bearing item (all currently `TBD` pending live confirmation).
