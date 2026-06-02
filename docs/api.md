# API Reference

DistLLM provides an OpenAI-compatible REST API. All existing OpenAI SDKs and clients work with DistLLM by changing the base URL.

## Base URL

```
http://localhost:8000/v1
```

## Authentication

All requests require an `Authorization: Bearer <api-key>` header.

### Development (No Auth)

For local development, disable authentication:

```bash
distllm system api --model <model> --local --no-auth
```

No API key needed — anyone on localhost can connect.

### With API Key

```bash
export API_KEY="your-secret-key"
distllm system api --model <model> --local
```

The API key is shown on startup. Use it in requests:

```bash
curl -H "Authorization: Bearer your-secret-key" http://localhost:8000/v1/models
```

## Endpoints

### Chat Completions

```
POST /v1/chat/completions
```

OpenAI-compatible chat endpoint for distributed inference.

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

### Completions

```
POST /v1/completions
```

Standard text completion. Supports the same distributed pipeline execution as chat.

### Models

```
GET /v1/models
```

List all available models loaded in the distributed cluster.

### Embeddings

```
POST /v1/embeddings
```

Generate embeddings using the distributed pipeline's embedding model.

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

## SDK Usage

### Python (OpenAI SDK)

```python
import openai

client = openai.Client(
    base_url="http://localhost:8000/v1",
    api_key="sk-your-master-key",
)

response = client.chat.completions.create(
    model="meta-llama/Llama-3.2-7B",
    messages=[{"role": "user", "content": "Hello!"}],
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
| 503 | node_unavailable | Worker node unavailable or unreachable |
