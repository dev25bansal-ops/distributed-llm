# Migrating from Ollama to DistLLM

Ollama ships **two** API surfaces:

1. Its **native API** on `http://localhost:11434` — `/api/generate`, `/api/chat`,
   `/api/embeddings`, `/api/tags`, `/api/version`.
2. An **OpenAI-compatible mode** on `http://localhost:11434/v1` (since Ollama 0.1.42+).

DistLLM also speaks the OpenAI shape, so the cleanest migration is to **point your OpenAI-mode
clients at DistLLM's `/v1`**. If you have tools that only understand Ollama's native API, DistLLM
ships an **Ollama-compatible proxy** (`distllm-ollama-proxy`) that translates the native Ollama
endpoints to DistLLM — run it on port `11434` and your existing Ollama clients keep working.

- DistLLM OpenAI-compatible base URL: `http://localhost:8000/v1`
- Ollama proxy base URL (native API): `http://localhost:11434` → forwards to DistLLM
- `api_key` optional (`--no-auth` for local dev)

## Prerequisites

A running DistLLM API server with a model loaded:

```bash
pip install distllm
distllm system api --model meta-llama/Llama-3-8B-Instruct --local --no-auth
# or
docker compose up -d
docker compose exec coordinator distllm model load meta-llama/Llama-3-8B-Instruct
```

See [docs/QUICKSTART.md](QUICKSTART.md). If you want to keep Ollama-native clients working, also
start the proxy (it defaults to Ollama's port `11434`):

```bash
distllm-ollama-proxy --distllm-url http://localhost:8000 --port 11434
```

The proxy is implemented in
[integrations/ollama_compat/src/distllm_ollama/server.py](https://github.com/distributed-llm/distributed-llm/blob/main/integrations/ollama_compat/src/distllm_ollama/server.py)
and translates `/api/generate`, `/api/chat`, `/api/embeddings`, and `/api/tags` to DistLLM's
`/v1/completions`, `/v1/chat/completions`, `/v1/embeddings`, and `/v1/models`.

## Before / After (OpenAI-compatible mode)

If your code already uses Ollama's `/v1` OpenAI mode, you only change the host/port.

### Python

```python
# BEFORE — Ollama OpenAI mode
import openai
client = openai.OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

response = client.chat.completions.create(
    model="llama3",
    messages=[{"role": "user", "content": "Hello!"}],
)
```

```python
# AFTER — DistLLM
import openai
client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed",
)

response = client.chat.completions.create(
    model="meta-llama/Llama-3-8B-Instruct",   # DistLLM model id
    messages=[{"role": "user", "content": "Hello!"}],
)
```

### JavaScript

```javascript
// BEFORE — Ollama OpenAI mode
import OpenAI from "openai";
const client = new OpenAI({ baseURL: "http://localhost:11434/v1", apiKey: "ollama" });
```

```javascript
// AFTER — DistLLM
import OpenAI from "openai";
const client = new OpenAI({ baseURL: "http://localhost:8000/v1", apiKey: "not-needed" });
```

### curl

```bash
# BEFORE — Ollama OpenAI mode
curl http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"llama3","messages":[{"role":"user","content":"Hello!"}]}'

# AFTER — DistLLM
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer not-needed" \
  -H "Content-Type: application/json" \
  -d '{"model":"meta-llama/Llama-3-8B-Instruct","messages":[{"role":"user","content":"Hello!"}]}'
```

## Before / After (native Ollama API via proxy)

Keep your Ollama-native code untouched and run `distllm-ollama-proxy` on `:11434`:

```python
# UNCHANGED — still talks to http://localhost:11434 (now the DistLLM proxy)
import ollama
response = ollama.chat(model="meta-llama/Llama-3-8B-Instruct",
                      messages=[{"role": "user", "content": "Hello!"}])
```

```bash
# Ollama CLI still works if the proxy is on :11434
ollama run meta-llama/Llama-3-8B-Instruct "Hello!"
```

> With the proxy, the `model` field is passed straight through to DistLLM, so use DistLLM's model
> id (the model must be loaded in the cluster). `num_predict` maps to `max_tokens`, and
> `options.temperature` / `options.top_p` map to the DistLLM generation params.

## Model Name Mapping

| Ollama | DistLLM |
|--------|---------|
| `ollama run llama3` (tag) | `model: meta-llama/Llama-3-8B-Instruct` |
| `ollama run mistral` (tag) | `model: mistralai/Mistral-7B-v0.3` |
| `ollama run phi3` (tag) | `model: microsoft/Phi-3-mini-4k-instruct` |
| `ollama run <local.gguf>` | `model: <local path>` or `llamacpp.model_path` |
| `ollama pull <name>` | `distllm model load <hf-id>` (downloads from HuggingFace) |

Ollama short tags are convenience aliases; DistLLM uses HuggingFace model ids (or local paths).
Load the model into the cluster before requesting it. See
[docs/MODEL_COMPATIBILITY.md](MODEL_COMPATIBILITY.md).

## Equivalent Endpoints

| Ollama endpoint | DistLLM endpoint (native→DistLLM) | Notes |
|-----------------|------------------------------------|-------|
| `POST /api/chat` | `POST /v1/chat/completions` (via proxy) | Native Ollama chat → OpenAI chat shape |
| `POST /api/generate` | `POST /v1/completions` (via proxy) | `prompt` maps to `prompt` |
| `POST /api/embeddings` | `POST /v1/embeddings` (via proxy) | `prompt` → `input: [prompt]` |
| `GET /api/tags` | `GET /v1/models` (via proxy) | Lists loaded models |
| `GET /api/version` | `GET /health` (via proxy) | Health/version |
| `http://localhost:11434/v1/*` | `http://localhost:8000/v1/*` | Direct OpenAI-mode swap |

## Side-by-Side Config Table

| Concern | Ollama | DistLLM |
|---------|--------|---------|
| Serve address | `http://localhost:11434` | `http://localhost:8000` (REST `coordinator.api_port`) |
| OpenAI mode | `http://localhost:11434/v1` | `http://localhost:8000/v1` |
| Model source | `ollama pull <tag>` / Modelfile | `distllm model load <hf-id>` or `model.name` in `config.yaml` |
| Modelfile `FROM` | `FROM llama3` | `model.name: meta-llama/Llama-3-8B-Instruct` |
| Modelfile `PARAMETER num_ctx` | context window | `vllm.max_model_len` / `llamacpp.n_ctx` |
| Modelfile `PARAMETER temperature` | sampling | `generation.temperature` (default `0.7`) |
| Modelfile `PARAMETER num_gpu` | GPU layers | `tensor_parallel.num_gpus` / `llamacpp.n_gpu_layers` |
| GGUF models | native | `llamacpp.model_path` + `llamacpp.enabled: true` |
| Quantization | automatic (Q4_K_M etc.) | `quantization.method` (`bnb_4bit`, `gptq`, `awq`, `fp8`) |
| Auth | none | `API_KEY` env / `--no-auth` |
| Multi-node | no (single host) | yes — `nodes:` list, pipeline/tensor parallelism |
| Runtime config | `~/.ollama/...` + env | `config.yaml` or `DISTLLM__SECTION__FIELD` env vars |

See [docs/CONFIG_REFERENCE.md](CONFIG_REFERENCE.md) for the complete DistLLM config schema.

## Caveats

- **Model ids differ**: Ollama tags (`llama3`) are not DistLLM model ids. Use HuggingFace ids and
  load them first. Unloaded models return an error.
- **GGUF handling**: Ollama's native GGUF is served via DistLLM's `llamacpp` backend
  (`llamacpp.enabled: true`, `llamacpp.model_path`). The v1 OpenAI endpoints serve whatever model is
  loaded, regardless of format.
- **`num_predict` vs `max_tokens`**: When using the proxy, `num_predict` maps to `max_tokens`
  (DistLLM default `256`). Set it explicitly to avoid short outputs.
- **Streaming**: Both Ollama native (NDJSON) and DistLLM (`/v1` SSE) support streaming; the proxy
  converts between them.
- **Auth**: Ollama has no auth; DistLLM expects `Authorization: Bearer <key>` in production
  (`--no-auth` for local dev).
- **Throughput**: DistLLM adds distributed/consumer-hardware scaling that Ollama (single host) does
  not; latency characteristics will differ.

## Verify

Check the DistLLM `/v1/models` endpoint (directly, or through the proxy's `/api/tags`):

```bash
# Direct DistLLM
curl http://localhost:8000/v1/models
# -> {"object":"list","data":[{"id":"meta-llama/Llama-3-8B-Instruct", ...}]}

# Through the Ollama proxy
curl http://localhost:11434/api/tags

# Real completion
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer not-needed" \
  -H "Content-Type: application/json" \
  -d '{"model":"meta-llama/Llama-3-8B-Instruct","messages":[{"role":"user","content":"ping"}]}'
```

---

## SUMMARY

Migrate Ollama → DistLLM by switching your OpenAI-mode clients from `http://localhost:11434/v1` to
`http://localhost:8000/v1` and replacing Ollama tags (`llama3`) with DistLLM model ids
(`meta-llama/Llama-3-8B-Instruct`) that you load via `distllm model load`. To keep Ollama-native
clients/tools working unchanged, run `distllm-ollama-proxy` on `:11434` (it translates
`/api/chat`, `/api/generate`, `/api/embeddings`, `/api/tags` to DistLLM's endpoints). Key config
differences: Ollama Modelfile `FROM`/`PARAMETER` map to DistLLM `model.name`/`generation`;
Ollama's single-host GGUF becomes DistLLM's `llamacpp` backend. Verify with `GET /v1/models`.
