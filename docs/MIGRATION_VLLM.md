# Migrating from a vLLM OpenAI-Compatible Server to DistLLM

Both vLLM and DistLLM speak the OpenAI API shape, so the migration is mostly a **base URL swap**.
DistLLM's value-add is that it runs across **distributed and consumer hardware** (multiple GPUs,
multiple machines, even CPU fallback) using pipeline/tensor parallelism, whereas vLLM is tuned for
high-throughput single-node GPU serving. The client code you already wrote for vLLM works unchanged
against DistLLM — you just point it at a different host.

- DistLLM base URL: `http://localhost:8000/v1` (same `/v1` path vLLM uses)
- `api_key` optional (use `--no-auth` for local dev)
- Same four OpenAI endpoints: `POST /v1/chat/completions`, `POST /v1/completions`,
  `POST /v1/embeddings`, `GET /v1/models`

## Prerequisites

A running DistLLM API server. With the Python package:

```bash
pip install distllm
distllm system api --model meta-llama/Llama-3.2-1B --local --no-auth
```

Or Docker Compose (recommended):

```bash
docker compose up -d
docker compose exec coordinator distllm model load meta-llama/Llama-3.2-1B
```

See [docs/QUICKSTART.md](QUICKSTART.md) and [docs/adr/0004-openai-compatible-api.md](adr/0004-openai-compatible-api.md).

## Before / After

### Python (openai SDK)

```python
# BEFORE — vLLM (default port 8000)
import openai
client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY",   # vLLM often runs with no real key
)

response = client.chat.completions.create(
    model="meta-llama/Llama-3.2-1B",
    messages=[{"role": "user", "content": "Hello!"}],
)
```

```python
# AFTER — DistLLM (same base_url path, different server)
import openai
client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed",   # or your DistLLM API key
)

response = client.chat.completions.create(
    model="meta-llama/Llama-3.2-1B",
    messages=[{"role": "user", "content": "Hello!"}],
)
```

> If your vLLM used a non-default port (e.g. `8001`), update the port to DistLLM's
> `coordinator.api_port` (default `8000`).

### JavaScript (openai SDK)

```javascript
// BEFORE — vLLM
import OpenAI from "openai";
const client = new OpenAI({ baseURL: "http://localhost:8000/v1", apiKey: "EMPTY" });
```

```javascript
// AFTER — DistLLM
import OpenAI from "openai";
const client = new OpenAI({ baseURL: "http://localhost:8000/v1", apiKey: "not-needed" });
```

### curl

```bash
# BEFORE — vLLM
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"meta-llama/Llama-3.2-1B","messages":[{"role":"user","content":"Hello!"}]}'

# AFTER — DistLLM
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer not-needed" \
  -H "Content-Type: application/json" \
  -d '{"model":"meta-llama/Llama-3.2-1B","messages":[{"role":"user","content":"Hello!"}]}'
```

## Key Differences

| Area | vLLM | DistLLM |
|------|------|---------|
| Hardware target | Single-node GPU, high throughput (PagedAttention) | Distributed / consumer hardware; multi-GPU and multi-node pipeline + tensor parallelism; CPU fallback |
| Model loading | Loaded at server start via CLI args | Loaded explicitly with `distllm model load <model>` (or `multi_model.models` in config). Server can start before a model is resident. |
| Model ids | HuggingFace ids / served names | HuggingFace ids or local paths (must be loaded before use) |
| Speculative decoding | First-class, heavily optimized | Not a headline feature — speedups come from pipeline/tensor parallelism across nodes rather than vLLM-style speculative decoding |
| Concurrency model | Continuous batching within one process | Distributed batching (`batching`, `chunked_prefill`) across the cluster |
| Limits | Bounded by one host's GPUs | Bounded by aggregate cluster memory; add worker nodes to scale |
| Extra endpoints | OpenAI surface + vLLM-specific | OpenAI surface + DistLLM extensions (marketplace, federation, defrag, batch, gossip, admin) — see [docs/api.md](api.md) |

## Config Mapping Table

vLLM is typically configured via CLI flags at launch; DistLLM uses `config.yaml`
([docs/CONFIG_REFERENCE.md](CONFIG_REFERENCE.md)) or env vars (`DISTLLM__SECTION__FIELD`).

| vLLM launch flag | DistLLM equivalent | Notes |
|------------------|--------------------|-------|
| `--model <id>` | `model.name: <id>` | DistLLM also requires the model be loaded at runtime |
| `--tensor-parallel-size N` | `tensor_parallel.num_gpus: N` (or `vllm.tensor_parallel_size`) | DistLLM can also span nodes via pipeline parallelism |
| `--gpu-memory-utilization 0.9` | `vllm.gpu_memory_utilization: 0.9` | Same meaning when vLLM backend is enabled |
| `--max-model-len N` | `vllm.max_model_len: N` | Context length cap |
| `--dtype auto` | `model.dtype` / `vllm.dtype` | `float16`, `bfloat16`, `float32`, `auto` |
| `--max-num-seqs N` | `vllm.max_num_seqs: N` | Concurrent sequences |
| `--quantization awq` | `quantization.method: awq` (or `gptq`, `bnb_4bit`, `fp8`) | DistLLM supports BnB/GPTQ/AWQ/FP8 |
| `--enforce-eager` | `vllm.enforce_eager: true` | Disable CUDA graphs |
| `--api-key <key>` | `API_KEY` env var (or `--no-auth`) | DistLLM auth is bearer-token based |
| `--port 8000` | `coordinator.api_port: 8000` | REST API port |
| `--host 0.0.0.0` | `coordinator.host: 0.0.0.0` | Bind address |
| `--served-model-name x` | `multi_model.models: {x: <hf-id>}` | Alias a model name; set `default_model` |

> If you were relying on vLLM's vLLM backend specifically, DistLLM can also run the vLLM backend
> (`vllm.enabled: true`) while adding distributed scheduling on top — see
> [docs/CONFIG_REFERENCE.md](CONFIG_REFERENCE.md) §40.

## Caveats

- **Model must be loaded**: Unlike vLLM (which loads at startup), DistLLM serves only models that
  have been loaded into the cluster. Run `distllm model load <model>` or define `multi_model.models`.
  An unloaded model returns a `model_not_found`-style error.
- **Speculative decoding**: DistLLM's primary acceleration is distributed parallelism, not vLLM's
  speculative decoding. Do not assume identical per-token latency; benchmark on your hardware.
- **Throughput profile**: Single-stream latency may differ; DistLLM wins on models that don't fit
  on one GPU by sharding across nodes.
- **Auth**: vLLM often runs keyless; DistLLM in production expects `Authorization: Bearer <key>`
  (or `--no-auth` for local dev).
- **Limits**: Rate limits (`rate_limit` config) and context windows are GPU/cluster-bound, not
  fixed plan tiers.

## Verify

```bash
curl http://localhost:8000/v1/models
# -> {"object":"list","data":[{"id":"meta-llama/Llama-3.2-1B", ...}]}

curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer not-needed" \
  -H "Content-Type: application/json" \
  -d '{"model":"meta-llama/Llama-3.2-1B","messages":[{"role":"user","content":"ping"}]}'
```

---

## SUMMARY

Migrating from vLLM to DistLLM is a base-URL swap (both use `http://localhost:8000/v1` with the
OpenAI schema, optional api_key). The real differences are operational: DistLLM targets distributed
and consumer hardware via pipeline/tensor parallelism, requires models to be explicitly loaded
(`distllm model load`), and does not emphasize vLLM-style speculative decoding. Map vLLM CLI flags
to DistLLM's `config.yaml` (`model`, `tensor_parallel`/`vllm.*`, `quantization`, `coordinator`).
Verify with `GET /v1/models`.
