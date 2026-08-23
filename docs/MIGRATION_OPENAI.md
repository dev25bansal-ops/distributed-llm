# Migrating from the OpenAI API to DistLLM

DistLLM exposes a **fully OpenAI-compatible REST API** (see
[docs/adr/0004-openai-compatible-api.md](adr/0004-openai-compatible-api.md)). You can point any
OpenAI SDK or client at a DistLLM cluster by changing one value: the `base_url`. No code changes
to your request shapes are required — the JSON schemas for requests and responses, as well as SSE
streaming, match OpenAI.

- DistLLM base URL: `http://localhost:8000/v1`
- `api_key` is **optional**. With `--no-auth` you can pass any non-empty string (e.g. `"not-needed"`).
- Supported endpoints: `POST /v1/chat/completions`, `POST /v1/completions`,
  `POST /v1/embeddings`, `GET /v1/models`.

## Prerequisites

You need a running DistLLM API server. The fastest local start:

```bash
# Python package
pip install distllm
distllm system api --model meta-llama/Llama-3.2-1B --local --no-auth

# Or via Docker Compose (recommended)
docker compose up -d
docker compose exec coordinator distllm model load meta-llama/Llama-3.2-1B
```

The server listens on port `8000` (override with `coordinator.api_port`). See
[docs/QUICKSTART.md](QUICKSTART.md) for full instructions, including API-key mode
(`export API_KEY=...`) and multi-node setups.

## Before / After

### Python (openai SDK)

```python
# BEFORE — OpenAI
import openai
client = openai.OpenAI(api_key="sk-...")

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello!"}],
)
```

```python
# AFTER — DistLLM (only base_url + api_key change)
import openai
client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed",   # or your DistLLM API key
)

response = client.chat.completions.create(
    model="meta-llama/Llama-3.2-1B",   # DistLLM model id, not "gpt-4o-mini"
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)
```

You can also use the native DistLLM SDK (see `examples/sdk_example.py`):

```python
from distllm_sdk import DistLLMClient
client = DistLLMClient(base_url="http://localhost:8000/v1", api_key="not-needed")
resp = client.chat.completions.create(
    model="meta-llama/Llama-3.2-1B",
    messages=[{"role": "user", "content": "Hello!"}],
)
```

### JavaScript (openai SDK)

```javascript
// BEFORE — OpenAI
import OpenAI from "openai";
const client = new OpenAI({ apiKey: "sk-..." });

const response = await client.chat.completions.create({
  model: "gpt-4o-mini",
  messages: [{ role: "user", content: "Hello!" }],
});
```

```javascript
// AFTER — DistLLM
import OpenAI from "openai";
const client = new OpenAI({
  baseURL: "http://localhost:8000/v1",
  apiKey: "not-needed",
});

const response = await client.chat.completions.create({
  model: "meta-llama/Llama-3.2-1B",
  messages: [{ role: "user", content: "Hello!" }],
});
```

### curl

```bash
# BEFORE — OpenAI
curl https://api.openai.com/v1/chat/completions \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-4o-mini","messages":[{"role":"user","content":"Hello!"}]}'
```

```bash
# AFTER — DistLLM
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer not-needed" \
  -H "Content-Type: application/json" \
  -d '{"model":"meta-llama/Llama-3.2-1B","messages":[{"role":"user","content":"Hello!"}]}'
```

## Endpoint Parity

| OpenAI endpoint | DistLLM endpoint | Notes |
|-----------------|------------------|-------|
| `POST /v1/chat/completions` | `POST /v1/chat/completions` | Identical schema; SSE streaming supported |
| `POST /v1/completions` | `POST /v1/completions` | Text completion |
| `POST /v1/embeddings` | `POST /v1/embeddings` | Embedding generation |
| `GET /v1/models` | `GET /v1/models` | Lists models loaded in the cluster |
| `GET /v1/health/*` | `GET /v1/health/readiness`, `GET /v1/health/liveness` | Health probes |

DistLLM also exposes additional non-OpenAI endpoints (marketplace, federation, defrag, batch,
gossip, admin) documented in [docs/api.md](api.md) — these are extensions and not part of the
OpenAI surface.

## Parameter / Config Mapping

DistLLM accepts the standard OpenAI generation parameters unchanged:

| OpenAI parameter | DistLLM support | Notes |
|------------------|-----------------|-------|
| `model` | ✅ | Use a DistLLM model id (HuggingFace id or local path), not an OpenAI model name |
| `messages` | ✅ | Same chat format |
| `prompt` | ✅ | For `/v1/completions` |
| `temperature` | ✅ | Default `0.7` |
| `top_p` | ✅ | Default `0.9` |
| `top_k` | ✅ | DistLLM extension (default `0`) |
| `max_tokens` | ✅ | Maps to generation `max_new_tokens` (default `256`) |
| `stream` | ✅ | SSE, OpenAI format |
| `n`, `stop`, `logprobs`, `frequency_penalty`, `presence_penalty` | ⚠️ | Best-effort; verify per endpoint |
| `tools` / `function_call` | ⚠️ | Tool-calling support varies; see [docs/api.md](api.md) |

Server-side defaults can be set in `config.yaml` under `generation` (see
[docs/CONFIG_REFERENCE.md](CONFIG_REFERENCE.md)).

## Caveats

- **Model names**: DistLLM serves models you load explicitly (e.g. `meta-llama/Llama-3.2-1B`).
  OpenAI model names like `gpt-4o-mini` will not resolve. Load the model first with
  `distllm model load <model>` (or via `multi_model.models` in config). See
  [docs/MODEL_COMPATIBILITY.md](MODEL_COMPATIBILITY.md).
- **Limits**: Rate limits are enforced (`rate_limit` config; default 60 RPM, 30 RPM for
  chat/completions). DistLLM is a self-hosted cluster, so token/context limits depend on your GPU
  memory, not a fixed OpenAI plan.
- **Streaming**: Supported via SSE exactly like OpenAI (`stream: true`). The non-streaming path is
  also byte-compatible.
- **Auth**: No-auth mode (`--no-auth`) is for local dev only. In production set `API_KEY` and send
  `Authorization: Bearer <key>`; a missing/invalid key returns `401 auth_error`.
- **Extensions**: Custom DistLLM features live in `X-DistLLM-*` headers, not in the OpenAI request
  body, to preserve compatibility.

## Verify

Confirm the server is up and lists your loaded model:

```bash
curl http://localhost:8000/v1/models
# -> {"object":"list","data":[{"id":"meta-llama/Llama-3.2-1B", ...}]}
```

Then run a real completion:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer not-needed" \
  -H "Content-Type: application/json" \
  -d '{"model":"meta-llama/Llama-3.2-1B","messages":[{"role":"user","content":"ping"}]}'
```

---

## SUMMARY

DistLLM is a drop-in OpenAI-compatible backend: swap `base_url` to `http://localhost:8000/v1`,
make `api_key` optional (`--no-auth`) or your DistLLM key, and replace OpenAI model names with
DistLLM model ids (HuggingFace ids you have loaded). All four core OpenAI endpoints
(`/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, `/v1/models`) share the same request/
response schemas and SSE streaming. Main caveats: model names must be pre-loaded, limits are
GPU-bound, and some OpenAI-only params are best-effort. Verify with `GET /v1/models`.
