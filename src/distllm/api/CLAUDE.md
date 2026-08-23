# CLAUDE.md — API Server

## Your Scope
You have ownership of `src/distllm/api/` — FastAPI server, all 17 route files, middleware, auth, SSO, streaming, rate limiting.

## Do NOT Touch
- `src/distllm/core/` — core engine (handled by another instance)
- `src/distllm/dist/` — distributed layer (handled by another instance)
- Any other module outside `api/`

## Key Files

| File | Purpose |
|------|---------|
| `server.py` | FastAPI app, middleware stack, all `/api/*` routes |
| `routes/chat.py` | `/v1/chat/completions` — structured output, tool calling, multimodal |
| `routes/completion.py` | `/v1/completions` |
| `routes/embeddings.py` | `/v1/embeddings` |
| `routes/health.py` | Health, readiness, warmup (with asyncio.to_thread fix) |
| `routes/admin.py` | Admin endpoints |
| `routes/model_registry.py` | Model load/unload with RBAC |
| `routes/defrag.py` | Defrag with admin-only access |
| `routes/exchange.py` | Prompt exchange with IDOR fix |
| `routes/federated.py` | Federation routes |
| `middleware.py` | AuthMiddleware (API key + rate limiting) |
| `auth_deps.py` | `require_role()` dependency factory |
| `sso_auth.py` | OAuth2/OIDC/SAML with CSRF state + nonce |
| `sso_middleware.py` | SSO middleware + `/v1/auth/{token,refresh,revoke}` (mounted in `server.py` via `setup_sso`) |
| `streaming.py` | SSE streaming with tool call support |
| `rate_limiter.py` | Token bucket rate limiter |
| `quota_middleware.py` | Per-tenant quota enforcement |
| `errors.py` | Error response formatting |
| `rate_limit_middleware.py` | Back-compat re-export of `RequestRateLimitMiddleware` (from `middleware.py`) |

## Current State
- All CRITICAL/HIGH security fixes applied
- JWT auth wired end-to-end via `_auth_header`
- OAuth2 state + OIDC nonce CSRF protection
- SSO middleware mounted (`setup_sso`) — `/v1/auth/{token,refresh,revoke}`
- Role-based access on ALL routes
- Cluster key rotation with grace period
- HA state snapshot endpoint

## Commands
- `python -m pytest tests/api/ -v` — run API tests
- `python -m pytest tests/api/test_auth_middleware.py -v` — auth tests
- `python -m pytest tests/api/test_chat.py -v` — chat completion tests
- `python -m pytest tests/api/test_health.py -v` — health endpoint tests
