---
tags:
  - api
  - fastapi
  - rest
  - sso
aliases:
  - API Server
---
# API Server — `src/distllm/api/`

**89 .py files · ~22K LOC.**

> The outward-facing HTTP surface — a **FastAPI app** presenting an OpenAI-compatible API (`/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, assistants, files, fine-tuning, moderations, batch, tools, tokenize) fronting the distributed coordinator/backends, plus enterprise auth (SAML/OIDC/OAuth2 SSO, API-key RBAC, JWT, OPA/Rego) and ops (health/dashboard/metrics, admin, WebSocket chat).
>
> **Tests:** `python -m pytest tests/api/ -v` (~46 files).

## Root (`api/`)

| file | LOC | purpose |
|------|-----|---------|
| `__init__.py` | 9 | re-exports `app`, `create_coordinator`, `main`, `AuthMiddleware` |
| `server.py` | 1,690 | FastAPI app assembly, middleware stack, WS dashboard, SSE metrics, `main()` CLI |
| `server_config.py` | 116 | CORS-origin resolution + settings load |
| `server_coordinator.py` | 108 | `create_coordinator()` + gRPC node wrapper |
| `server_errors.py` | 63 | error models + exception handlers |
| `server_lifespan.py` | 152 | startup/shutdown lifespan async-gen |
| `server_routes_api.py` | 400 | ad-hoc `/api/*` op endpoints |
| `server_routes_dashboard.py` | 207 | dashboard/UI WS + page endpoints |
| `server_state.py` | 18 | `state` proxy over shared `AppState` |
| `api_state.py` | 34 | cross-module `g` state proxy (coordinator/store) |
| `app_state.py` | 22 | single `AppState` source of truth |
| `api_settings.py` | 91 | consolidated Pydantic `BaseSettings` |
| `api_versioning.py` | 828 | Accept-Version negotiation, deprecation scheduling, compat map |
| `errors.py` | 219 | standardized OpenAI-compatible error responses |
| `validation.py` | 45 | input validation utilities |
| `auth_deps.py` | 35 | `require_role(*roles)` RBAC dependency |
| `ip_utils.py` | 43 | `get_client_ip` (anti rate-limit bypass) |
| `grpc_bridge.py` | 254 | OpenAI-style → DistLLM gRPC bridge |
| `token_estimator.py` | 69 | `estimate_tokens` shared by cost/quota |
| `streaming.py` | 392 | SSE streaming helper `_stream_response` |
| `tool_calling.py` | 210 | execute OpenAI tool calls (JSON block + XML) |
| `ws_chat.py` | 403 | WebSocket chat streaming with backpressure |
| `persistent_store.py` | 203 | SQLite for batch/files/fine-tuning |
| `semantic_cache.py` | 278 | embedding-similarity semantic cache middleware |

### Middleware stack (registration order in `server.py`)
CORS → SecurityHeaders → CSRF → Timeout → **Auth** → RequestID → RequestRateLimit → Dedup → PromptInjection → (ContentModeration) → DocsAuth → RequestSizeLimit → Backpressure → CircuitBreaker → (PluginHook) → (CostTracking) → BodyCache → Tracing.
_Raw-ASGI `WAF` (`waf.py`) runs apart from this stack. SSO middleware (`sso_middleware.setup_sso`) is mounted at the outer edge (registered after tracing) — it tries SSO JWT first and falls through to API-key auth._

| middleware file | LOC | purpose |
|-----------------|-----|---------|
| `middleware.py` | 571 | `AuthMiddleware`, `RequestIDMiddleware`, `RequestRateLimitMiddleware`, `ContentModerationMiddleware`, `_RateLimiter` |
| `waf.py` | 294 | raw-ASGI WAF: inspect/rewrite body |
| `prompt_injection.py` | 349 | two-layer detection: BERT ~2ms + optional LLM-as-judge |
| `csrf_middleware.py` | 159 | same-origin Origin/Referer validation |
| `circuit_breaker_middleware.py` | 142 | server-side circuit breaker |
| `cost_middleware.py` | 152 | `X-DistLLM-Cost/Tokens/Savings` headers + accounting |
| `quota_middleware.py` | 153 | per-tenant token quotas + billing |
| `dedup.py` | 134 | content-fingerprint dedup + TTL cache |
| `body_cache_middleware.py` | 92 | cache request body for downstream reuse |
| `observability_middleware.py` | 147 | OTel spans + RED metrics + cost/GPU + anomaly samples |
| `tracing_middleware.py` | 486 | W3C traceparent parsing, root span + sub-spans |
| `rate_limiter.py` | 193 | token-bucket throttling |
| `rate_limiter_unified.py` | 767 | strategy-pattern unified limiter, hierarchical endpoints |
| `redis_rate_limiter.py` | 141 | distributed Redis rate limiter |
| `rate_limit_middleware.py` | 10 | re-export shim of `RequestRateLimitMiddleware` |
| `background_tasks.py` | 251 | health monitor, auto-restart, ordered shutdown |

### SSO & AuthZ
| file | LOC | purpose |
|------|-----|---------|
| `sso_auth.py` | 557 | `SSOAuthHandler`/`get_sso_handler` — SAML/OIDC/OAuth2 routing, token revoke/validate |
| `sso_middleware.py` | 626 | SSO middleware + `/v1/auth/{token,refresh,revoke}` (mounted in `server.py` via `setup_sso`; buffered provider handle) |
| `auth/__init__.py` | 147 | unified `SSOAuthHandler` facade, lazy provider dispatch |
| `auth/models.py` | 30 | `SSOUserInfo` + role mapping |
| `auth/oauth2.py` | 107 | Generic OAuth2 token exchange (GitHub/GitLab) |
| `auth/oidc.py` | 269 | OIDC discovery, token validate, state/nonce |
| `auth/saml.py` | 93 | SAML 2.0 IdP (needs pysaml2) |
| `auth/store.py` | 29 | token revocation blocklist |
| `authz/__init__.py` | 24 | `authorize`, `load_policy`, OPA_AVAILABLE |
| `authz/opa.py` | 235 | OPA/Rego adapter + pure-Python fallback evaluator |

## `routes/` (route catalog)

| file | LOC | purpose |
|------|-----|---------|
| `__init__.py` | 41 | imports 19+ routers (`chat_v2` incl.) |
| `chat.py` | 770 | `/v1/chat/completions` + v2; structured output, tools, multimodal |
| `completion.py` | 119 | `/v1/completions` |
| `embeddings.py` | 326 | `/v1/embeddings`, `/v1/rerank` |
| `batch.py` | 895 | `/v1/batch` bulk sync + async (SQLite) |
| `admin.py` | 481 | `/admin/v1/*` node drain/offline, runtime config, logs |
| `assistants.py` | 358 | OpenAI-compatible Assistants (threads/runs/messages/vector stores) |
| `api_keys.py` | 90 | `/v1/api-keys` create/list/revoke |
| `debug.py` | 154 | `/v1/debug/recent`, `/v1/debug/replay` |
| `defrag.py` | 52 | GPU memory defrag (admin) |
| `eval.py` | 200 | `/api/v1/eval/*` benchmarks |
| `exchange.py` | 293 | prompt-exchange marketplace + token gating |
| `experiments.py` | 236 | A/B experiments |
| `federated.py` | 119 | federated LoRA training/merge |
| `fine_tuning.py` | 307 | OpenAI fine-tuning jobs |
| `files.py` | 193 | OpenAI Files API |
| `gossip.py` | 60 | P2P gossip KV discovery |
| `health.py` | 309 | health/readiness/liveness/metrics/warmup |
| `leaderboard.py` | 300 | benchmark leaderboard |
| `marketplace.py` | 269 | GPU listing/job/matching |
| `metrics_history.py` | 343 | metrics + health trends + alert thresholds + topology |
| `model_registry.py` | 216 | loaded models, versions, cache (RBAC load/unload) |
| `moderation.py` | 347 | `/v1/moderations` |
| `plugins.py` | 128 | plugin registry manage |
| `prompts.py` | 273 | prompt template library + sharing |
| `router_admin.py` | 148 | `/v1/router/*` rules + dry-run |
| `scheduler.py` | 131 | live scheduler tuning + metrics |
| `tokenize.py` | 58 | `/v1/tokenize` |
| `tools.py` | 74 | `/v1/tools` discovery/invoke for framework adapters |
| `webhooks.py` | 261 | `/v1/webhooks` CRUD + trigger |
| `webrtc.py` | 148 | WebRTC signaling (experimental) |

## `services/` (routers' business logic)

| file | LOC | purpose |
|------|-----|---------|
| `chat_service.py` | 579 | `ChatService` — chat business logic |
| `completion_service.py` | 188 | `CompletionService` |
| `embedding_service.py` | 366 | `EmbeddingService` (embeddings/rerank) |
| `eval_service.py` | 165 | `EvalService` |

## `webhooks/`

| file | LOC | purpose |
|------|-----|---------|
| `__init__.py` | 15 | re-exports event/registration/delivery types |
| `delivery.py` | 377 | HMAC signing, retry/backoff (1s→16s ×5), DLQ |

## Notes / dead code

- **`server_middleware.py` was deleted (2026-08-08)** — it was dead code (394 LOC, never imported); its `SecurityHeaders/CSRF/Timeout/DocsAuth/SizeLimit/Backpressure/CircuitBreaker/PluginHook/CostTracking/BodyCache` definitions were already duplicated inline in `server.py`, which is the live stack. `api/CLAUDE.md` previously claimed it was "(DELETED)" while the file still existed; that is now actually true.
- **`sso_middleware.py` (626) is wired (2026-08-08)** — `setup_sso(app)` is called in `server.py`; it was previously unmounted. Fix also removed a latent crash: Starlette passes a wrapped ASGI app (not the FastAPI app) to `BaseHTTPMiddleware.__init__`, so the instance is now stashed lazily in `dispatch` instead of `__init__`.
- **Multiple rate-limiters coexist** — `rate_limiter(_unified).py`, `redis_rate_limiter.py`, legacy `_RateLimiter` — consolidation pending.
- `api/auth` duplicates `sso_auth.py` surface (façade pattern).

## Tests

`tests/api/` (~46 files) — auth, chat, batch, middleware chain, passive. Plus `tests/security/`, `tests/fuzz/fuzz_*`, `tests/e2e/` for streaming/chat.