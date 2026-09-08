# DistLLM Example Notebooks

Four runnable Jupyter notebooks that demonstrate the DistLLM OpenAI-compatible API
using the official `openai` Python client (with a transparent fallback to
`distllm_sdk.compat.openai_compat` when `openai` is not installed).

All notebooks target the OpenAI-compatible endpoint:

```
http://localhost:8000/v1
api_key: not-needed   (or set DISTLLM_API_KEY if your deployment enforces auth)
```

The endpoint, model, and API key are read from environment variables
(`DISTLLM_BASE_URL`, `DISTLLM_MODEL`, `DISTLLM_API_KEY`) and fall back to sensible
defaults, so you can override them without editing cells.

## Notebooks

### 1. `rag.ipynb` — Retrieval-Augmented Generation
A minimal, self-contained RAG pipeline:
1. Defines a few sample documents.
2. Embeds them via the DistLLM **embeddings** endpoint.
3. Builds an in-notebook vector store (cosine similarity with `numpy` + `scikit-learn`,
   or a pure-numpy fallback if sklearn is absent).
4. Retrieves the top-k chunks for a question.
5. Generates an answer with a **chat completion** grounded in the retrieved context.

Optional: a markdown cell describes the LlamaIndex variant.

### 2. `streaming.ipynb` — Streaming Chat Completions
Streams tokens **token-by-token** from `chat.completions.create(stream=True)` using the
`openai` client, and a second example that streams with `stream_options={"include_usage": True}`
to surface a final usage summary.

### 3. `batch_processing.ipynb` — Batch Processing & Throughput
Sends a batch of prompts through chat completions and measures **throughput in
tokens/second** from `response.usage.completion_tokens` divided by wall-clock time.
Includes a sequential run and a concurrent `ThreadPoolExecutor` run to saturate the server.
Heavy deps (`concurrent.futures`) are guarded with try/except.

### 4. `cost_analysis.ipynb` — Token Usage & Cost Analysis
Collects per-request token usage and cost from `response.usage` (including the optional
`cost_usd` field), aggregates totals via the DistLLM SDK client `stats().estimate_cost()`,
and plots a bar chart (token counts + estimated cost per scenario) with `matplotlib`.

## Prerequisites

Install Python dependencies:

```bash
pip install jupyter openai distllm_sdk numpy scikit-learn matplotlib
```

- `openai` — preferred client for the streaming / batch / cost notebooks.
- `distllm_sdk` — provides `distllm_sdk.compat.openai_compat` (fallback) and the
  `DistLLMClientSync` stats aggregator used in `cost_analysis.ipynb`.
- `numpy` / `scikit-learn` — vector store for `rag.ipynb` (sklearn optional).
- `matplotlib` — charts for `cost_analysis.ipynb`.

## Start the DistLLM API server

Before running any notebook, launch the API so `http://localhost:8000/v1` is reachable:

```bash
# From the repo root
pip install -e ./sdk          # if distllm_sdk is not yet installed

# Start the server (example: a local model)
distllm-api --model distributed-llm --host 0.0.0.0 --port 8000
```

Verify it is up:

```bash
curl http://localhost:8000/v1/models
```

## Run the notebooks

```bash
cd examples/notebooks
jupyter notebook        # then open the .ipynb you want and Run All
# or, headless:
jupyter nbconvert --to notebook --execute --inplace rag.ipynb
```

Each notebook is written so that if the server is unreachable, the relevant code cell
prints a clear, non-crashing message (and a few fall back to dummy data so the rest of
the notebook still executes). Optional dependencies are guarded with try/except and
print a helpful install hint instead of raising.

## Configuration reference

| Env var             | Default                       | Purpose                          |
|---------------------|-------------------------------|----------------------------------|
| `DISTLLM_BASE_URL`  | `http://localhost:8000/v1`    | OpenAI-compatible API base URL   |
| `DISTLLM_API_KEY`   | `not-needed`                  | Bearer token (if auth is enabled)|
| `DISTLLM_MODEL`     | `distributed-llm`             | Model id passed to each request  |
