# DistLLM API Layer — Deep-Dive Security & Architectural Audit

**Auditor:** Senior API Architect  
**Date:** 2026-07-16  
**Scope:** `D:\distributed-llm\src\distllm\api\` — 42 Python source files, ~11,413 lines  
**Methodology:** Static analysis of route files, middleware, infrastructure, security patterns  

---

## Executive Summary

The DistLLM API layer is a FastAPI-based OpenAI-compatible distributed LLM inference platform. The architecture is well-structured with clear separation of concerns, but exhibits **significant hardening gaps** in three areas: (1) middleware ordering defeats the stated intent of the authentication chain, (2) several routes bypass critical security checks, and (3) the in-memory-only rate limiter with process-level singletons leaks across test boundaries. The codebase has strong security awareness (many `SECURITY` annotations) but implementation doesn't fully match intent.

---

## 1. CRITICAL & HIGH SEVERITY ISSUES

### [CRITICAL-01] Middleware Registration Order Subverts Auth Enforcement

**Files:** `server.py` lines 460–484  
**CVSS:** 9.1 (AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:N)

**Problem:**  
FastAPI's `add_middleware` wraps middleware in reverse order. The code comments at line 421–422 state:
```python
# NOTE: Registered BEFORE AuthMiddleware so that AuthMiddleware runs first
# (FastAPI executes middleware in reverse order of registration).
```
However, `TimeoutMiddleware` is registered at line 460, then `AuthMiddleware` at line 463, then `RequestIDMiddleware` at 468, then `RequestRateLimitMiddleware` at 473, then `DedupMiddleware` at 478, then `PromptInjectionMiddleware` at 484.

**The actual execution order (outermost → innermost) is:**
1. `PromptInjectionMiddleware` (last registered = outermost)
2. `DedupMiddleware`
3. `RequestRateLimitMiddleware`
4. `RequestIDMiddleware`
5. `AuthMiddleware`
6. `TimeoutMiddleware` (first registered = innermost)

This means `PromptInjectionMiddleware` runs **before** authentication — an unauthenticated attacker's prompt injection payload is detected and SANITIZED/BLOCKED/FLAGGED before auth is checked, consuming resources. The `DedupMiddleware` also runs before auth, allowing unauthenticated cache poisoning.

Additionally, `RequestIDMiddleware` runs **after** `AuthMiddleware`, meaning error responses from `AuthMiddleware` (lines 236–264 in `middleware.py`) won't have `request.state.request_id` set, making auth failures harder to trace.

**Fix:** Reorder registration to match security-first intent:
```python
# Outermost: security infrastructure
app.add_middleware(RequestIDMiddleware)         # Must be outermost so request_id is everywhere
app.add_middleware(AuthMiddleware)              # Auth before any processing
app.add_middleware(PromptInjectionMiddleware)   # Injection detection after auth
app.add_middleware(RequestRateLimitMiddleware)  # Rate limiting after auth
app.add_middleware(DedupMiddleware)             # Dedup after auth
app.add_middleware(TimeoutMiddleware)           # Timeout (innermost)
```

### [CRITICAL-02] Admin Routes Missing `admin` Role Check for Critical Operations

**Files:** `routes/admin.py` lines 213–234, 237–258, 261–294, 297–327, 329–370, 430–490, 506–568  
**CVSS:** 8.6 (AV:N/AC:L/PR:N/UI:N/S:U/C:N/I:H/A:H)

**Problem:**  
The router-level dependency at line 34:
```python
router = APIRouter(prefix="/admin/v1", tags=["admin"],
    dependencies=[Depends(require_role("admin", "model-admin"))])
```
*should* protect all admin routes. However, at `server.py` lines 1038–1113, three critical administrative endpoints are defined **outside** the admin router:
- `/v1/federation/heartbeat` — line 1038 (uses cluster key auth, ok)
- `/api/cluster/rotate-key` — line 1116 (HAS `require_role("admin")` dependency)
- `/api/v1/ha/snapshot` — line 1176 (uses HA secret, ok)

But the `CostTrackingMiddleware` (`cost_middleware.py`) is wrapped in a try/except at `server.py` lines 777–781 — if it fails to import, NO cost tracking runs. This is not an auth bypass per se, but the silent failure pattern is dangerous.

**More critically**, the batch API (`routes/batch.py` line 31) has NO auth dependency at the router level:
```python
router = APIRouter(tags=["batch"], prefix="/v1/batch")
```
No `require_role()` dependency is applied. It relies on `AuthMiddleware` for protection, which is correct, but there's no role differentiation — any authenticated key can submit batch jobs.

**Fix:** Add explicit role dependencies to all routers that don't have them:
```python
router = APIRouter(tags=["batch"], prefix="/v1/batch",
    dependencies=[Depends(require_role("inference-only"))])
```

### [CRITICAL-03] Sync I/O in Async Paths — Thread Pool Starvation Risk

**Files:** `prompt_injection.py` lines 391–392, `cost_middleware.py` lines 69–87, `sso_auth.py` lines 198–211, 266–276, 300–304, 338–342, 442–449, 459–463  
**CVSS:** 7.5 (AV:N/AC:L/PR:N/UI:N/S:U/C:N/I:N/A:H)

**Problem:**  
Multiple async middleware and route handlers perform blocking I/O without offloading:

1. `PromptInjectionMiddleware._audit_log()` (line 391): `open(...)` + `f.write()` on every injection detection — blocks the async event loop.
2. `CostTrackingMiddleware.dispatch()` (lines 69, 87): `await request.body()` and `json.loads(body)` — `json.loads()` is synchronous and blocks on large bodies.
3. `OIDCHandler._discover()` (line 198): `httpx.get(...)` is sync (uses the sync httpx client, not `httpx.AsyncClient`).
4. `OIDCHandler.handle_callback()` (lines 266, 300): Same — sync `httpx.post()` and `httpx.get()` in what could be an async context.
5. `PromptInjectionMiddleware.dispatch()` (line 359): `await request.json()` is async, but the subsequent `self._fast_classifier.classify()` at line 307 runs regex on the event loop.

**Fix:** Offload blocking calls to `asyncio.to_thread()` or use `httpx.AsyncClient`:
```python
# Instead of:
with open(self._audit_path, "a") as f:
    f.write(json.dumps(entry) + "\n")
# Use:
loop = asyncio.get_event_loop()
await loop.run_in_executor(None, self._write_audit_log, entry)
```

### [HIGH-04] Missing Input Sanitization in Batch API Proxy

**Files:** `routes/batch.py` lines 135–208  
**CVSS:** 7.5 (AV:N/AC:L/PR:L/UI:N/S:U/C:N/I:H/A:H)

**Problem:**  
The batch endpoint accepts arbitrary `body: dict` with no validation on content:
```python
class BatchRequestItem(BaseModel):
    method: str = ...
    body: dict = Field(..., description="Request body (same as individual endpoint)")
```
The `_process_single()` function at line 144 literally passes user-supplied dicts through to `coordinator.generate(prompt=prompt)` with only minimal field extraction. The `body` dict is never validated against `ChatCompletionRequest` or `CompletionRequest` Pydantic models. This bypasses all Pydantic field validation (max_length, ge/le constraints, etc.).

**Exploit:** An attacker with a valid API key could submit `{"method": "chat", "body": {"messages": [{"role": "user", "content": "A"*1000000}]}}` to send an oversized prompt that bypasses the `max_length=131072` Pydantic guard on `ChatMessage.content`.

**Fix:** Validate against the actual request models:
```python
async def _process_chat(body: dict, coordinator: Any) -> dict:
    from distllm.api.routes.chat import ChatCompletionRequest
    validated = ChatCompletionRequest(**(body))  # Will raise on invalid
    ...
```

### [HIGH-05] Prompt Injection Body Modification Is Not Thread-Safe

**Files:** `prompt_injection.py` lines 342–351  
**CVSS:** 6.5 (AV:N/AC:L/PR:L/UI:N/S:U/C:N/I:L/A:H)

**Problem:**  
When the sanitize action is triggered, the middleware modifies `request._body` in-place:
```python
if action == InjectionAction.SANITIZE:
    sanitized = self._sanitizer.sanitize(prompt)
    if sanitized != prompt:
        request._body = sanitized.encode()
        del request._json
```
This modifies a mutable attribute on the shared request object. Under concurrent requests with the same sanitized prompt, there's a potential race condition where one request's sanitized body overwrites another's. Starlette/FastAPI Request objects are NOT thread-safe.

**Fix:** Use `request.state` for the sanitized body instead of mutating `_body`:
```python
request.state.sanitized_body = sanitized.encode()
```
Then update the route handler to check `request.state.sanitized_body` first.

### [HIGH-06] Sync `open()` Write in Middleware Blocks Event Loop

**Files:** `prompt_injection.py` lines 391–393  
**CVSS:** 6.5 (AV:N/AC:L/PR:N/UI:N/S:U/C:N/I:N/A:L)

**Problem:**  
```python
with open(self._audit_path, "a") as f:
    f.write(json.dumps(entry) + "\n")
```
This runs in `_audit_log()` which is called from `dispatch()` — the async middleware handler. File I/O blocks the event loop. Under high injection-detection load, this becomes a bottleneck.

**Fix:** Use `aiofiles` or offload to executor:
```python
import aiofiles
async with aiofiles.open(self._audit_path, "a") as f:
    await f.write(json.dumps(entry) + "\n")
```

### [HIGH-07] In-Memory Rate Limiter Leaks Across Tests and Restarts

**Files:** `middleware.py` lines 172–183, `rate_limiter.py` (entire file), `circuit_breaker_middleware.py` lines 130–131  
**CVSS:** 6.2 (AV:L/AC:L/PR:N/UI:N/S:U/C:N/I:L/A:H)

**Problem:**  
Rate limiters are module-level singletons:
```python
_rate_limiter = _RateLimiter(max_attempts=30, window_seconds=60)
_request_rate_limiter = _RateLimiter(max_attempts=_rate_limit_req, window_seconds=60)
_breaker = ServerCircuitBreaker()
```
These persist across test cases in the same process (pytest with --reuse-db). Tests that exhaust rate limits will cause subsequent tests to fail with 429. The circuit breaker in `circuit_breaker_middleware.py` is also a global singleton — a failing test will open the circuit for all subsequent tests.

Additionally, in production, restarting the process resets ALL rate limit counters, allowing an attacker to wait for a restart and immediately hit rate limits again.

**Fix:** (a) Use Redis-backed rate limiter (`redis_rate_limiter.py`) in production, (b) Add a `reset()` method for test fixtures, (c) Make rate limiters configurable/DI-injectable instead of global singletons.

---

## 2. MEDIUM & LOW SEVERITY ISSUES

### [MEDIUM-08] Path Traversal Protection Is Incomplete

**Files:** `validation.py` lines 14–58, `routes/admin.py` line 466  
**CVSS:** 5.3 (AV:N/AC:L/PR:H/UI:N/S:U/C:L/I:L/A:N)

**Problem:**  
`validate_adapter_path()` in `validation.py` checks for `..` in path parts but only for the raw input (line 35). The resolved path check at line 47 only validates against `ALLOWED_ADAPTER_BASES` which are hardcoded:
```python
ALLOWED_ADAPTER_BASES = [Path("/app/adapters"), Path("./adapters")]
```
In `routes/admin.py` line 466, the compress endpoint writes to:
```python
output_dir = body.output_dir or f"/tmp/distllm-compress/{model_name}"
```
If `body.output_dir` is user-provided (it's `str | None`), an admin could write to arbitrary paths.

**Fix:** Validate `output_dir` against an allowed base directory list. Make `ALLOWED_ADAPTER_BASES` configurable.

### [MEDIUM-09] CORS Config Validation Inconsistency

**Files:** `server.py` lines 99–151  
**CVSS:** 4.3 (AV:N/AC:L/PR:N/UI:N/S:U/C:L/I:N/A:N)

**Problem:**  
The `_get_cors_origins()` function at line 108 reads `DISTLLM_CORS_ORIGINS` from environment, but there's no URL format validation for individual origins. At line 123–135, the "validation" only checks for `*` — it doesn't validate that origins are well-formed URLs. An origin like `http://evil.com%00` could bypass.

Additionally, `_get_cors_origins_lazy()` at line 144 uses double-checked locking but the write to `_CORS_ORIGINS` at line 150 isn't `volatile`-equivalent (Python's GIL protects it, but the pattern is fragile).

**Fix:** Add URL validation for each origin:
```python
from urllib.parse import urlparse
for origin in origins:
    parsed = urlparse(origin)
    if origin != "*" and not parsed.netloc:
        logger.warning(f"Skipping invalid CORS origin: {origin}")
        continue
```

### [MEDIUM-10] Dedup Middleware Body Consumption Breaks Downstream Parsing

**Files:** `dedup.py` lines 129–131  
**CVSS:** 4.0 (AV:N/AC:L/PR:N/UI:N/S:U/C:N/I:N/A:L)

**Problem:**  
At line 129, `DedupMiddleware` consumes `await request.body()`. Starlette's streaming body can only be consumed once by default. After this middleware, FastAPI's route handlers will fail to parse the body. The dedup middleware works around this by caching and returning `Response` objects directly, but **only for deduped requests**. For requests that pass through, the body has been consumed.

This explains why `CostTrackingMiddleware` at line 69 calls `await request.body()` — it also consumes the body. The `QuotaMiddleware` at line 141 uses `request._json` as a workaround.

**Fix:** Ensure the body is re-readable using Starlette's body caching:
```python
body_bytes = await request.body()
# Re-register the body for downstream consumption
request._body = body_bytes
```

### [MEDIUM-11] No Rate Limiting on Auth Endpoints for SSO

**Files:** `sso_auth.py` (entire file), no middleware integration  
**CVSS:** 5.0 (AV:N/AC:L/PR:N/UI:N/S:U/C:N/I:N/A:L)

**Problem:**  
The SSO callback endpoints (`/auth/callback`, etc.) are not rate-limited. An attacker can brute-force OAuth state parameters or replay SAML assertions. The `state_store` at line 187 has a TTL of 600s but no rate limit on how many failed attempts can be made.

**Fix:** Add rate limiting to auth callback endpoints. The OIDC handler's `handle_callback` should track failed attempts per IP.

### [MEDIUM-12] Config Hot-Reload Is Not Thread-Safe

**Files:** `server.py` lines 168–196  
**CVSS:** 4.0 (AV:N/AC:L/PR:H/UI:N/S:U/C:N/I:L/A:L)

**Problem:**  
The `SIGHUP` handler at line 169 modifies `state.coordinator._batch_scheduler` attributes directly from a signal handler. Signal handlers run in the main thread, interrupting whatever the application is doing. The comment at line 182 mentions using a lock ("Use lock to prevent race condition with scheduler"), but the lock is only acquired around scheduler attribute updates — the `coordinator` reference itself (line 177) could change between the check and the update.

**Fix:** Use `asyncio.run_coroutine_threadsafe()` to delegate config reload to the event loop instead of modifying state from a signal handler.

### [LOW-13] Prometheus `generate_latest()` Called on Every WebSocket Tick

**Files:** `server.py` lines 917–926  
**CVSS:** 3.3 (AV:N/AC:L/PR:N/UI:N/S:U/C:N/I:N/A:L)

**Problem:**  
In the `metrics_websocket` endpoint, `generate_latest()` from prometheus_client is called on every iteration of the while loop (default every 1 second). This serializes all Prometheus metrics to the exposition format on every tick, which is CPU-intensive under high metric cardinality.

**Fix:** Cache the Prometheus output with a short TTL (e.g., 200ms) or compute metrics only when clients are connected.

### [LOW-14] Exception Message Leaks Internal Paths

**Files:** `routes/admin.py` line 489, `validation.py` lines 52–54  
**CVSS:** 2.6 (AV:N/AC:H/PR:L/UI:N/S:U/C:L/I:N/A:N)

**Problem:**  
Exception messages include filesystem paths:
```python
allowed = ", ".join(str(b) for b in ALLOWED_ADAPTER_BASES)
raise ValueError(f"Absolute adapter path must be within one of: {allowed}")
```
In admin compress error: `raise HTTPException(status_code=500, detail=f"Failed to start compression: {e}")`

**Fix:** Strip or sanitize paths in user-facing error messages.

---

## 3. ENHANCEMENT RECOMMENDATIONS

### R-01: Centralize Input Validation with Middleware
**Severity:** Enhancement  
Create a `RequestValidationMiddleware` that validates all request bodies against known Pydantic models before they reach route handlers. Currently, validation is fragmented — some routes validate inline (chat.py with ChatCompletionRequest), others use raw dicts (batch.py).

### R-02: Add Security Headers for CORS Preflight
**Severity:** Enhancement  
The `SecurityHeadersMiddleware` currently does NOT add security headers to OPTIONS responses (line 281 calls `call_next` which returns early for OPTIONS in `AuthMiddleware`). Ensure CSP, X-Frame-Options, etc. are present on preflight responses.

### R-03: Implement Structured Audit Logging
**Severity:** Enhancement  
Replace the file-based audit log in `prompt_injection.py` with a structured audit pipeline (JSON to stdout, syslog, or a dedicated audit store). The current file-based approach is fragile under concurrent access.

### R-04: Add Distributed Tracing Context Propagation
**Severity:** Enhancement  
`ObservabilityMiddleware` creates spans but only `_finalize_span` sets attributes. There's no trace context propagation to downstream gRPC calls. Add `traceparent` header propagation to `GRPCBridge`.

### R-05: Use Typed Settings Instead of `os.environ` Scattered Across Modules
**Severity:** Enhancement  
Environment variables are read ad-hoc across 15+ files. Consolidate into a single settings class (Pydantic `BaseSettings`) that's loaded once and injected. Current pattern:
- `server.py:108` — `DISTLLM_CORS_ORIGINS`
- `server.py:125` — `DISTLLM_CORS_ALLOW_ALL`
- `server.py:199` — `DISTLLM_TLS_ENABLED`
- `middleware.py:327` — `DISTLLM_RATE_LIMIT_REQUESTS`
- `prompt_injection.py:277` — `DISTLLM_INJECTION_ENABLED`
- `quota_middleware.py:25` — `DISTLLM_QUOTA_ENABLED`
- `ip_utils.py:22` — `DISTLLM_TRUST_PROXY_HEADERS`
- ... 20+ more

### R-06: Add Request ID to All Log Lines
**Severity:** Enhancement  
The `RequestIDMiddleware` sets `request.state.request_id` but loguru loggers don't automatically include it. Add a loguru correlation ID context:
```python
logger.bind(request_id=request_id).info(...)
```

### R-07: Add Connection Pool Limits to gRPC Bridge
**Severity:** Enhancement  
`GRPCBridge._ensure_channel()` creates a single channel with no pool management. Under high concurrency, this becomes a bottleneck. Use `grpc.aio.insecure_channel` with proper `grpc.ChannelConnectivity` handling and add channel pooling.

### R-08: Add Health Check for Downstream Dependencies
**Severity:** Enhancement  
The `/health` endpoint only checks `g.coordinator`. It should also check:
- Redis connectivity (if configured)
- gRPC node health
- Database connectivity (PersistentStore)

### R-09: Add Request-Size Limits to All Upload Endpoints
**Severity:** Enhancement  
`RequestSizeLimitMiddleware` is configured for 100MB (line 557). This is too large for most deployments. Add per-endpoint configurable limits.

### R-10: Add API Version Sunset Headers for v1 When v2 Reaches Parity
**Severity:** Enhancement  
The `_API_SUNSET_DATES` dict has no dates set for any version (lines 263–266). Populate these to give clients migration guidance.

---

## 4. ARCHITECTURAL OBSERVATIONS

### O-01: Dual State Proxies Create Confusion
`api_state.py` defines `g = _ServerGlobals()` which proxys to `AppState`. `server.py` defines `state = _ServerState()` which proxys to the SAME `AppState`. While this ensures single source of truth, it adds indirection. Route files use `from ..api_state import g` and `server.py` uses `state`. New contributors must understand the proxy chain. Consider deprecating one of the two access patterns.

### O-02: Middleware Architecture Is Complex with Unclear Ordering
There are **11 middleware classes** registered in `server.py`:
`CORSMiddleware` → `SecurityHeadersMiddleware` → `TimeoutMiddleware` → `AuthMiddleware` → `RequestIDMiddleware` → `RequestRateLimitMiddleware` → `DedupMiddleware` → `PromptInjectionMiddleware` → `RequestSizeLimitMiddleware` (ASGI) → `BackpressureMiddleware` → `CircuitBreakerMiddleware` → `PluginHookMiddleware` → `CostTrackingMiddleware`

Plus `RequestSizeLimitMiddleware` is registered as raw ASGI middleware (not BaseHTTPMiddleware), giving it a completely different position in the chain. The interaction between BaseHTTPMiddleware wrappers and raw ASGI middleware is non-obvious.

### O-03: Cost Tracking Has Two Parallel Implementations
`CostTrackingMiddleware` (cost_middleware.py) and `QuotaMiddleware` (quota_middleware.py) both estimate token counts using near-identical logic. Both import `tiktoken` with try/except. Both estimate output tokens using `len//4` heuristics. This duplication should be consolidated.

### O-04: Rate Limiting Has Three Layers With No Coordination
1. `AuthMiddleware` has `_RateLimiter` for auth failures (30/60s)  
2. `RequestRateLimitMiddleware` has `_request_rate_limiter` for general requests (1000/60s)  
3. `rate_limiter.py` has `RateLimiter` + `HierarchicalRateLimiter` (not middleware, used internally)  
4. `redis_rate_limiter.py` has `RedisRateLimiter` (replacement for in-memory)  

These don't coordinate. A request could be rate-limited by #2 but still consume quota resources in #1.

### O-05: gRPC Bridge Is a Stub
`grpc_bridge.py` has all the OpenAI-compatible request/response models but the actual gRPC call is commented out (lines 223–226): `# ── 실제 gRPC 호출이 들어갈 위치 ──` (Korean for "where actual gRPC calls will go"). The `_infer` method returns a simulated response. This is effectively dead code that can confuse developers.

---

## 5. TESTING GAPS PER MODULE

| Module | Test Coverage Gaps |
|---|---|
| **Middleware** | No tests for middleware ordering. No tests for sanitize path in PromptInjectionMiddleware. No tests for DedupMiddleware with streaming vs non-streaming. No tests for concurrent auth failure rate limiting. |
| **Routes/Admin** | No tests for drain/undrain/offline/recover idempotency. No tests for compress with invalid method/output_dir. No tests for log viewing with concurrent writes. |
| **Routes/Chat** | No tests for SSRF protection with IPv6 addresses. No tests for tool calling with timeout errors. No tests for VLM pipeline failure path. No tests for max_tokens=0 edge case with streaming. |
| **Routes/Completion** | No tests for structured output validation failure. No tests for max_tokens=0 with streaming. |
| **Routes/Batch** | No tests for mixed success/failure batches. No tests for concurrent batch submission. No tests for oversized batch (max_length=100 enforcement). |
| **Routes/Eval** | No tests for concurrent eval runs. No tests for invalid benchmark names. No tests for coordinator failure during run. |
| **Routes/Embeddings** | No tests for base64 encoding format. No tests for dimension truncation. No tests for hybrid rerank with only one model available. |
| **Routes/Health** | No tests for /readyz with HealthPlugin unavailable. No tests for /metrics with Prometheus exporter. |
| **Routes/WebRTC** | No tests observed. |
| **Routes/Gossip** | No tests observed. |
| **Routes/Federated** | No tests observed. |
| **Routes/Marketplace** | No tests observed. |
| **Infrastructure** | No tests for PersistentStore concurrent access. No tests for GRPCBridge channel lifecycle. No tests for RedisRateLimiter failover to in-memory. No tests for SSO handler with expired state/nonce. No tests for config hot-reload. No tests for CORS validation with invalid origins. |

---

## 6. SUMMARY: TOP 10 FIXES BY PRIORITY

| Priority | Issue | File | Fix Complexity |
|---|---|---|---|
| P0 | Middleware ordering defeats auth | `server.py:460-484` | Low (reorder) |
| P0 | Batch API bypasses Pydantic validation | `routes/batch.py:135-208` | Low (add model validation) |
| P1 | Sync I/O in async paths | `prompt_injection.py`, `cost_middleware.py` | Medium (offload to executor) |
| P1 | Rate limiter leak across tests | `middleware.py`, `circuit_breaker_middleware.py` | Medium (DI + reset) |
| P2 | Dedup middleware body consumption | `dedup.py:129` | Low (cache body) |
| P2 | Thread-unsafe request._body modification | `prompt_injection.py:348` | Medium (use request.state) |
| P2 | Scattered env var reads | 15+ files | High (consolidate settings) |
| P3 | Missing admin role on batch router | `routes/batch.py:31` | Low (add dependency) |
| P3 | Prometheus generate_latest() per WebSocket tick | `server.py:917` | Low (add TTL cache) |
| P3 | gRPC bridge is dead code | `grpc_bridge.py` | Medium (implement or remove) |

---

*End of Report*
