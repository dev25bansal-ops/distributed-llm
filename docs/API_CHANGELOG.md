# API Changelog

All breaking and notable changes to the DistLLM REST API are documented here.
The API follows [OpenAI compatibility](https://platform.openai.com/docs/api-reference)
where possible, with DistLLM-specific extensions documented below.

## Version Policy

- **v1** (`/v1/...`): Stable, production-ready. Breaking changes require major version bump.
- **v2** (`/v2/...`): Next-gen endpoints. May have breaking changes between minor releases.
- **Internal** (`/admin/...`, `/api/...`): No stability guarantee.

---

## v0.5.0 (Unreleased)

### New Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/embeddings/rerank` | POST | Cross-encoder reranking |
| `/v1/embeddings/rerank/hybrid` | POST | Hybrid embedding + reranking with RRF |
| `/v1/batch` | POST | Batch processing (up to 100 requests) |
| `/v1/scheduler/stats` | GET | Scheduler statistics |
| `/v1/scheduler/config` | GET/PATCH | Scheduler tuning |
| `/v1/defrag/status` | GET | GPU memory defragmentation status |
| `/v1/defrag/run` | POST | Trigger defragmentation |
| `/v2/chat/completions` | POST | v2 chat with system_fingerprint |

### Breaking Changes

- **`rate_limiter` module**: New `distllm.api.rate_limiter` module with `TokenBucket` and `RateLimiter` classes. The old in-memory limiter in `middleware.py` is unchanged.
- **`persistent_store` schema**: Added `schema_version` table. Existing databases auto-migrate on first access.
- **Error codes**: All domain errors now carry a `code` attribute (e.g., `MODEL_LOAD_ERROR`, `NODE_UNREACHABLE`).

### Deprecations

- `distllm.core.predictive_cache` — import from `distllm.dist.predictive_cache`
- `distllm.core.prefix_cache` — import from `distllm.dist.prefix_cache`
- `distllm.core.pipeline_orchestrator` — import from `distllm.dist.pipeline`
- `distllm.core.latency_tracker` — import from `distllm.dist.latency`
- `distllm.core.node_recovery` — import from `distllm.dist.recovery`
- `distllm.core.rebalancer` — import from `distllm.dist.rebalancer`
- `distllm.core.straggler_detector` — import from `distllm.dist.straggler`
- `distllm.core.vllm_backend` — import from `distllm.backends.vllm_backend`
- `distllm.core.llamacpp_backend` — import from `distllm.backends.llamacpp_backend`

---

## v0.4.0 (2026-05-16)

### New Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | Full OpenAI-compatible chat with tools, structured output |
| `/v1/completions` | POST | Text completion with streaming |
| `/v1/embeddings` | POST | Text embeddings |
| `/v1/models` | GET | List available models |
| `/health` | GET | Health check with node status |
| `/ready` | GET | Kubernetes readiness probe |
| `/live` | GET | Kubernetes liveness probe |
| `/metrics` | GET | Prometheus metrics |
| `/admin/v1/nodes` | GET | List cluster nodes |
| `/admin/v1/nodes/{id}/drain` | POST | Drain a node |
| `/admin/v1/config` | PATCH | Update runtime config |
| `/admin/v1/logs` | GET | View recent logs |
| `/v1/debug/recent` | GET | Recent requests (debug mode) |

### Request Format

**Chat Completion** (`POST /v1/chat/completions`):
```json
{
  "model": "distributed-llm",
  "messages": [{"role": "user", "content": "Hello"}],
  "max_tokens": 256,
  "temperature": 0.7,
  "top_p": 0.9,
  "stream": false,
  "tools": [{"type": "function", "function": {"name": "get_weather", "parameters": {...}}}],
  "response_format": {"type": "json_object"},
  "adapter": "my-lora-adapter",
  "priority": 2,
  "max_latency_ms": 5000,
  "scheduling": {"preemptible": true, "estimated_output_tokens": 100}
}
```

**Structured Output** (`response_format`):
```json
{"type": "json_object"}
{"type": "json_schema", "schema": {"type": "object", "properties": {...}}}
```

### Response Format

**Chat Completion Response**:
```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1715000000,
  "model": "distributed-llm",
  "choices": [{"index": 0, "message": {"role": "assistant", "content": "..."}, "finish_reason": "stop"}],
  "usage": {"prompt_tokens": 10, "completion_tokens": 50, "total_tokens": 60},
  "generation_time": 1.234
}
```

### Response Headers

| Header | Description |
|--------|-------------|
| `X-Request-ID` | Unique request identifier |
| `X-DistLLM-Cost` | Estimated cost in USD |
| `X-DistLLM-Tokens` | Token count |
| `X-DistLLM-Savings` | Savings vs cloud API |
| `X-RateLimit-Limit` | Rate limit (if enabled) |
| `X-RateLimit-Remaining` | Remaining requests |

### Authentication

```bash
# Bearer token
curl -H "Authorization: Bearer sk-your-key" http://localhost:8000/v1/chat/completions

# Environment variable
export API_KEY=sk-your-key
```

### Error Codes

| Status | Code | Description |
|--------|------|-------------|
| 401 | `auth_error` | Missing or invalid API key |
| 402 | `quota_exceeded` | Usage quota exceeded |
| 404 | `not_found` | Resource not found |
| 429 | `rate_limit` | Rate limit exceeded |
| 503 | `service_unavailable` | Model not loaded or shutting down |
| 504 | `timeout` | Request timed out |

---

## v0.3.0 (2026-05-14)

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | Basic chat completion |
| `/v1/completions` | POST | Text completion |
| `/v1/models` | GET | List models |
| `/health` | GET | Health check |
| `/v1/adapters` | GET/POST | LoRA adapter management |
| `/v1/update-params` | POST | Update generation params mid-stream |

### Notes

- CORS middleware added
- Security headers middleware added
- Authentication middleware with API key
- Rate limiting with token bucket algorithm
