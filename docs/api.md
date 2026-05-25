# API Reference

DistLLM provides an OpenAI-compatible REST API. All existing OpenAI SDKs and clients work with DistLLM by changing the base URL.

## Base URL

```
http://localhost:8000/v1
```

## Authentication

All requests require an `Authorization: Bearer <api-key>` header.

```bash
export DISTLLM_API_KEY=sk-your-master-key
```

## Endpoints

### Chat Completions

```
POST /v1/chat/completions
```

OpenAI-compatible chat endpoint with additional routing options.

**Request body:**
```json
{
  "model": "gpt-4o-mini",
  "messages": [{"role": "user", "content": "Hello!"}],
  "max_tokens": 256,
  "temperature": 0.7,
  "stream": false
}
```

**Routing options (DistLLM-specific):**
- `provider` — routing strategy: `"cheapest"`, `"fastest"`, or leave unset for default
- `provider_fallbacks` — list of fallback providers: `["together", "fireworks"]`

### Completions

```
POST /v1/completions
```

Standard text completion. Same routing options as chat.

### Models

```
GET /v1/models
```

List all available models across all configured providers.

### Embeddings

```
POST /v1/embeddings
```

Generate embeddings from configured embedding providers.

### Health

```
GET /health
GET /v1/health/readiness
GET /v1/health/liveness
```

### Metrics

```
GET /metrics
```

Prometheus-format metrics.

### Cost Tracking

```
GET /v1/costs
```

Returns per-tenant and per-model cost breakdown.

## SDK Usage

### Python (OpenAI SDK)

```python
import openai

client = openai.Client(
    base_url="http://localhost:8000/v1",
    api_key="sk-your-master-key",
)

# Route to cheapest provider
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello!"}],
    extra_body={"provider": "cheapest"},
)
```

### TypeScript (OpenAI SDK)

```typescript
import OpenAI from 'openai';

const client = new OpenAI({
  baseURL: 'http://localhost:8000/v1',
  apiKey: 'sk-your-master-key',
});

const response = await client.chat.completions.create({
  model: 'gpt-4o-mini',
  messages: [{ role: 'user', content: 'Hello!' }],
});
```

## Error Codes

| Status | Code | Description |
|--------|------|-------------|
| 401 | auth_error | Invalid or missing API key |
| 429 | auth_rate_limit | Too many failed auth attempts |
| 429 | rate_limit | Rate limit exceeded |
| 504 | timeout_error | Request exceeded timeout |
| 503 | provider_error | Upstream provider unavailable |
