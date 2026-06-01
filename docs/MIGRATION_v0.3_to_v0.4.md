# Migration Guide: v0.3 → v0.4

This guide covers all breaking changes, new requirements, and step-by-step instructions for upgrading from DistLLM v0.3.0 to v0.4.0.

---

## Table of Contents

- [Breaking Changes](#breaking-changes)
- [New Required Dependencies](#new-required-dependencies)
- [Configuration Changes](#configuration-changes)
- [API Changes](#api-changes)
- [Docker / Kubernetes Changes](#docker--kubernetes-changes)
- [Upgrade Checklist](#upgrade-checklist)

---

## Breaking Changes

### 1. Structured Error Responses

All HTTP errors now return a standardized OpenAI-compatible format. Clients parsing raw error bodies must update.

**Before (v0.3):**
```json
// 401 response
{"detail": "Unauthorized"}

// 429 response
{"detail": "Rate limit exceeded"}

// 500 response
{"detail": "Internal server error"}
```

**After (v0.4):**
```json
// All error responses follow this structure
{
  "error": {
    "message": "Unauthorized: missing API key",
    "type": "auth_error",
    "param": null,
    "code": "invalid_api_key"
  },
  "request_id": "abc-123-def"
}
```

**Action required:** Update any error-handling code that reads `response["detail"]` to read `response["error"]["message"]` instead. The `code` field is new and can be used for programmatic error handling.

---

### 2. Docker Non-Root Execution

The container now runs as `distllm:distllm` instead of root. File permission errors will occur if you mount volumes without proper ownership.

**Before (v0.3):**
```dockerfile
# Ran as root — no permission issues with mounted volumes
CMD ["distllm-node", ...]
```

**After (v0.4):**
```dockerfile
# Runs as distllm:distllm (UID/GID auto-assigned)
USER distllm
ENTRYPOINT ["/app/docker-entrypoint.sh"]
```

**Action required:** Update volume mounts to grant access to the `distllm` user:

```bash
# Option 1: Set ownership on host
chown -R 1000:1000 /path/to/mounted/data

# Option 2: Use --user flag (override at runtime)
docker run --user root distributed-llm:latest

# Option 3: Kubernetes securityContext
securityContext:
  runAsUser: 1000
  fsGroup: 1000
```

---

### 3. Plugin System Module Prefix Allowlist

The plugin system now restricts which modules can be loaded. Arbitrary code execution via plugins is blocked by default.

**Before (v0.3):**
```python
# Any module path was accepted
plugin_system.load("my_malicious_plugin")
```

**After (v0.4):**
```python
# Only allowlisted prefixes are accepted
# Default allowlist: ["distllm.plugins.", "plugins."]
plugin_system.load("distllm.plugins.my_plugin")  # OK
plugin_system.load("plugins.custom_plugin")       # OK
plugin_system.load("os.system")                   # BLOCKED
```

**Action required:** Ensure custom plugins are under an allowlisted module prefix, or configure the allowlist in `config.yaml`:

```yaml
plugins:
  module_prefixes:
    - "distllm.plugins."
    - "plugins."
    - "my_company.distllm_plugins."
```

---

### 4. Circuit Breaker Thread Safety Fix

The circuit breaker now uses `threading.Lock` instead of `asyncio.Lock`. This fixes thread-safety issues but changes the concurrency model.

**Before (v0.3):**
```python
# Used asyncio.Lock — not thread-safe with synchronous gRPC calls
class CircuitBreaker:
    def __init__(self):
        self._lock = asyncio.Lock()
```

**After (v0.4):**
```python
# Uses threading.Lock — safe for mixed sync/async usage
class CircuitBreaker:
    def __init__(self):
        self._lock = threading.Lock()
```

**Action required:** If you subclassed or monkey-patched the circuit breaker, update your lock usage. No configuration changes needed.

---

### 5. Rate Limiter Bounded Memory

The rate limiter now evicts LRU entries when the tracked IP count exceeds `max_clients` (default: 10,000). Previously, memory grew unbounded.

**Before (v0.3):**
```python
# Unbounded memory — all IPs tracked forever
rate_limiter = RateLimiter()
```

**After (v0.4):**
```python
# Bounded memory — LRU eviction at max_clients
rate_limiter = RateLimiter(max_clients=10000)
```

**Action required:** If you relied on the rate limiter tracking all IPs indefinitely, increase `max_clients` or implement external persistence.

---

### 6. Embeddings Fallback Removed

The embeddings endpoint no longer returns random vectors when the model fails to produce embeddings. It now returns a proper error.

**Before (v0.3):**
```python
# Silent fallback to random vectors
response = client.embeddings.create(input="hello")
# Could return random noise without any error
```

**After (v0.4):**
```python
# Raises a proper error
try:
    response = client.embeddings.create(input="hello")
except APIError as e:
    print(e.code)  # "model_load_error" or "node_error"
```

**Action required:** Handle embedding errors explicitly in your code.

---

### 7. gRPC Infer/StreamInfer Routing Fix

The gRPC `Infer` and `StreamInfer` methods now correctly route to actual inference logic instead of pass-through health checks.

**Before (v0.3):**
```protobuf
// Both methods returned health check responses
rpc Infer(InferRequest) returns (InferResponse);
rpc StreamInfer(InferRequest) returns (stream InferResponse);
```

**After (v0.4):**
```protobuf
// Infer: single-shot inference
rpc Infer(InferRequest) returns (InferResponse);
// StreamInfer: streaming token generation
rpc StreamInfer(InferRequest) returns (stream InferResponse);
```

**Action required:** If you used the gRPC API directly and relied on `Infer` as a health check, switch to the `HealthCheck` RPC or `/health` HTTP endpoint.

---

## New Required Dependencies

v0.4 introduces no new mandatory dependencies in the core package. However, the following optional dependencies are now recommended for production:

| Dependency | Version | Purpose | Install |
|-----------|---------|---------|---------|
| `pynvml` | `>=12.0` | Real GPU stats in health checks | `pip install nvidia-ml-py>=12.0` |
| `cryptography` | `>=41.0` | TLS certificate generation | `pip install cryptography>=41.0` |
| `pydantic-settings` | `>=2.0.0` | YAML config loading with env overrides | `pip install pydantic-settings>=2.0.0` |

**Install all production dependencies:**
```bash
pip install "distributed-llm[self-hosted,observability,security]"
```

---

## Configuration Changes

### YAML Key Changes

| v0.3 Key | v0.4 Key | Notes |
|----------|----------|-------|
| `cors.origins` | `cors.allow_origins` | Renamed for clarity |
| `cors.allow_credentials` | `cors.allow_credentials` | Now defaults to `false` (was `true`) |
| `security.headers` | `security.headers` | Now includes `Referrer-Policy` by default |
| *(new)* | `plugins.module_prefixes` | Plugin allowlist (see breaking change #3) |
| *(new)* | `api.request_timeout` | Per-endpoint timeout configuration |
| *(new)* | `api.backpressure_threshold` | Max pending requests before 503 |

### New Configuration Options

```yaml
# v0.4 additions
api:
  # Per-endpoint request timeouts
  request_timeout:
    chat_completions: 300    # 5 minutes (seconds)
    embeddings: 60           # 1 minute
    completions: 300         # 5 minutes
  # Backpressure: reject when this many requests are pending
  backpressure_threshold: 1000

plugins:
  # Module prefix allowlist for plugin loading
  module_prefixes:
    - "distllm.plugins."
    - "plugins."

# Kubernetes probe configuration
probes:
  readiness:
    path: "/ready"
    initial_delay: 10
    period: 10
  liveness:
    path: "/live"
    initial_delay: 30
    period: 30
```

### Environment Variable Changes

| v0.3 Env Var | v0.4 Env Var | Notes |
|-------------|-------------|-------|
| `CORS_ORIGINS` | `DISTLLM__CORS__ALLOW_ORIGINS` | Now uses pydantic-settings prefix |
| `API_KEY` | `DISTLLM_API_KEY` | Consistent prefix (still accepts `API_KEY` with warning) |
| *(new)* | `DISTLLM_RATE_LIMIT_REQUESTS` | Request rate limit (default: 1000/60s) |
| *(new)* | `DISTLLM_TRUST_PROXY_HEADERS` | Set to `1` to read `X-Forwarded-For` |
| *(new)* | `DISTLLM_CONFIG` | Path to config.yaml (alternative to `--config`) |

---

## API Changes

### New Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/ready` | GET | Kubernetes readiness probe (returns 200 when node is healthy) |
| `/live` | GET | Kubernetes liveness probe (returns 200 with uptime) |
| `/metrics` | GET | Enhanced Prometheus metrics (now includes service status even when not initialized) |

### Removed / Changed Endpoints

| Endpoint | Change | Migration |
|----------|--------|-----------|
| `/v1/adapters` | No change | Still available |
| `/v1/update-params` | No change | Still available |
| `/health` | Enhanced | Now returns real GPU stats via `pynvml` (install `nvidia-ml-py`) |
| `/metrics` | Enhanced | Now returns 503-compatible metrics (service status, coordinator load) |

### Response Format Changes

**Error responses** (see breaking change #1): All errors now use the OpenAI-compatible format.

**Health check response** (enhanced):
```json
// v0.4 includes real GPU stats
{
  "status": "healthy",
  "uptime": 3600,
  "nodes": [
    {
      "id": "node_0",
      "status": "ready",
      "gpu": {
        "name": "NVIDIA RTX 4090",
        "utilization": 45.2,
        "memory_used": 12.5,
        "memory_total": 24.0,
        "temperature": 62
      }
    }
  ]
}
```

---

## Docker / Kubernetes Changes

### Docker Image

**Breaking:** The image now uses a non-root user. See [breaking change #2](#2-docker-non-root-execution).

**New labels:**
```bash
docker inspect distributed-llm:latest | jq '.[0].Config.Labels'
# {
#   "org.opencontainers.image.source": "https://github.com/distributed-llm/distributed-llm",
#   "org.opencontainers.image.version": "0.4.0",
#   "org.opencontainers.image.title": "Distributed LLM",
#   "org.opencontainers.image.licenses": "Apache-2.0"
# }
```

### Docker Compose

No breaking changes to `docker-compose.yml`. The non-root user is handled internally.

### Kubernetes Probes

Update your deployments to use the new probe endpoints:

```yaml
# Before (v0.3)
livenessProbe:
  httpGet:
    path: /health
    port: 8000
readinessProbe:
  httpGet:
    path: /health
    port: 8000

# After (v0.4) — recommended
livenessProbe:
  httpGet:
    path: /live
    port: 8000
  initialDelaySeconds: 30
  periodSeconds: 30
readinessProbe:
  httpGet:
    path: /ready
    port: 8000
  initialDelaySeconds: 10
  periodSeconds: 10
```

### TLS Certificates

v0.4 expands certificate SANs to include wildcards, IPv6, and internal DNS names. Regenerate certificates if you use TLS:

```bash
distllm cert generate \
  --hosts "*.distllm.local,localhost,127.0.0.1,::1" \
  --output-dir certs/
```

---

## Upgrade Checklist

Follow these steps in order:

### Pre-Upgrade

- [ ] **Back up your config.yaml** — the file itself is compatible, but review new options
- [ ] **Back up any persistent data** (KV cache, API keys store)
- [ ] **Review error handling** in client code — update `response["detail"]` → `response["error"]["message"]`
- [ ] **Check plugin module paths** — ensure they match the new allowlist prefixes

### Upgrade

- [ ] **Update the package:**
  ```bash
  pip install --upgrade distributed-llm==0.4.0
  ```
- [ ] **Update optional dependencies:**
  ```bash
  pip install "nvidia-ml-py>=12.0" "cryptography>=41.0" "pydantic-settings>=2.0.0"
  ```
- [ ] **Regenerate requirements.lock** (if vendoring):
  ```bash
  pip-compile --output-file=requirements.lock pyproject.toml
  ```

### Configuration

- [ ] **Update YAML config** (optional new keys):
  ```yaml
  # Add these if you want to customize timeouts
  api:
    request_timeout:
      chat_completions: 300
      embeddings: 60
    backpressure_threshold: 1000
  ```
- [ ] **Update environment variables** (if using env-based config):
  ```bash
  # Rename if applicable
  export DISTLLM__CORS__ALLOW_ORIGINS='["https://app.example.com"]'
  export DISTLLM_RATE_LIMIT_REQUESTS=1000
  ```

### Docker / Kubernetes

- [ ] **Rebuild Docker image:**
  ```bash
  docker build -t distributed-llm:0.4.0 .
  ```
- [ ] **Fix volume permissions** (if mounting host directories):
  ```bash
  chown -R 1000:1000 /path/to/mounted/data
  ```
- [ ] **Update Kubernetes probes** to use `/ready` and `/live`
- [ ] **Regenerate TLS certificates** (if using TLS)

### Post-Upgrade Verification

- [ ] **Health check returns real GPU stats:**
  ```bash
  curl http://localhost:8000/health | jq '.nodes[0].gpu'
  ```
- [ ] **Readiness probe works:**
  ```bash
  curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/ready
  # Expected: 200
  ```
- [ ] **Error responses use new format:**
  ```bash
  curl http://localhost:8000/v1/chat/completions
  # Should return: {"error": {"message": "...", "type": "...", ...}}
  ```
- [ ] **Run the test suite:**
  ```bash
  pytest tests/ -v --timeout=60
  ```
- [ ] **Check Prometheus metrics:**
  ```bash
  curl http://localhost:8000/metrics | head -20
  ```

---

## Rollback

If you need to revert to v0.3:

```bash
pip install distributed-llm==0.3.0
# Restore backed-up config.yaml
# Restart services
```

The data formats are compatible — no migration rollback is needed for persistent state.

---

## Getting Help

- [GitHub Issues](https://github.com/distributed-llm/distributed-llm/issues)
- [Troubleshooting Guide](TROUBLESHOOTING.md)
- [Architecture Overview](ARCHITECTURE.md)
