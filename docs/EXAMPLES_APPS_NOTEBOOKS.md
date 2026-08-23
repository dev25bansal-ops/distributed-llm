# Examples, Apps & Notebooks

This page summarizes the ready-to-use examples shipped with DistLLM:

- The **VS Code extension** (editor integration)
- Three **example apps** (chat, RAG, multi-agent) under `apps/`
- Four **Jupyter notebooks** under `examples/notebooks/`

All of them connect to the **OpenAI-compatible DistLLM API** (default `http://localhost:8000/v1`).

---

## 1. VS Code Extension (`extensions/vscode/`)

Real-time cluster health monitoring and inline LLM inference from the editor.

**Features**
- **Cluster health status bar** — model name, node health (healthy/total), and throughput (tokens/sec) with color coding. Click the model item to open the dashboard.
- **Send Selection to DistLLM** — right-click selected editor text → run inference on the cluster; results appear in the output panel.
- **Configuration validation** — on activation, `distllm.*` settings are validated and a single non-blocking warning lists any invalid values:
  - `distllm.apiUrl` must be a valid `http(s)` URL
  - `distllm.refreshInterval` ∈ `[2, 300]` seconds
  - `distllm.maxTokens` > 0
  - `distllm.temperature` ∈ `[0, 2]`
- **Code snippets** — 8 snippets for Python / JS / TS covering chat completion, streaming chat, embeddings, and RAG bootstrap. Trigger by typing `distllm-chat-py`, `distllm-stream-py`, `distllm-embed-py`, `distllm-rag-py`, `distllm-chat-js`, `distllm-stream-js`, `distllm-embed-js`, `distllm-rag-js`.
- **Model browser** — a **DistLLM** activity-bar container with a **Models** tree populated from `${distllm.apiUrl}/v1/models`. Click a model to set `distllm.model`; right-click → **Copy Model ID**; refresh via the view-title button or the command.

**Commands**
| Command | Title |
|---------|-------|
| `distllm.sendSelection` | Send to DistLLM |
| `distllm.openDashboard` | Open DistLLM Dashboard |
| `distllm.refreshStatus` | Refresh DistLLM Status |
| `distllm.refreshModels` | Refresh DistLLM Models |
| `distllm.setModel` | Set as Default Model |
| `distllm.copyModelId` | Copy Model ID |

**Key settings**
| Setting | Default | Description |
|---------|---------|-------------|
| `distllm.apiUrl` | `http://localhost:8000` | DistLLM API server URL |
| `distllm.model` | Auto-detect | Model name override |
| `distllm.refreshInterval` | 10 | Status bar refresh interval (s) |
| `distllm.maxTokens` | 256 | Default max generation tokens |
| `distllm.temperature` | 0.7 | Default sampling temperature |

**How to run:** install the extension in VS Code (1.85+), point `distllm.apiUrl` at a running DistLLM API server, reload the window, and use the status bar / context menu / Models tree.

---

## 2. Example Apps (`apps/`)

All three apps read configuration from environment variables (defaults: `DISTLLM_BASE_URL=http://localhost:8000/v1`, `DISTLLM_API_KEY=sk-noauth`, `DISTLLM_MODEL=distributed-llm`). They connect to the OpenAI-compatible API at `DISTLLM_BASE_URL`.

### 2.1 `apps/chat/` — Streaming Chat UI
A small Flask server serving a build-free vanilla-JS `index.html` that proxies **streaming chat completions** via the `openai` Python client. The UI has chat bubbles, live token streaming, and a model-selector dropdown populated from `GET /v1/models`.

**Run**
```bash
cd apps/chat
pip install -r requirements.txt
python app.py          # open http://localhost:5000
```
Files: `app.py`, `index.html`, `requirements.txt`. Uses `DISTLLM_CHAT_PORT` (default 5000).

### 2.2 `apps/rag/` — Retrieval-Augmented Generation
A RAG web app over a bundled `sample.txt`. Retrieval uses DistLLM embeddings + a **numpy cosine retriever**, with automatic fallback to a LlamaIndex `VectorStoreIndex` when `llama-index` is installed. The active retriever is shown as a UI badge (`/api/backend`).

**Run**
```bash
cd apps/rag
pip install -r requirements.txt
python app.py          # open http://localhost:5001
```
Files: `app.py`, `index.html`, `sample.txt`, `requirements.txt`. Uses `DISTLLM_RAG_PORT` (default 5001), `DISTLLM_EMBED_MODEL`, `DISTLLM_RAG_TOP_K` (default 3).

### 2.3 `apps/multi_agent/` — Multi-Agent Pipeline
Three DistLLM-powered agents collaborate via plain **`asyncio` + the `openai` client**: `Researcher → Writer → Reviewer`. `openai-agents`/`crewai` are optional. The full conversation is printed to the terminal.

**Run**
```bash
cd apps/multi_agent
pip install -r requirements.txt
python run.py "Explain vector databases in three paragraphs"
# or use the built-in default task:
python run.py
```
Files: `agents.py`, `run.py`, `requirements.txt`, `README.md`.

> All three apps degrade gracefully when the backend is unavailable: chat falls back to a default model list, RAG boots and reports errors at query time, and multi-agent prints a clear backend error.

---

## 3. Example Notebooks (`examples/notebooks/`)

Four runnable Jupyter notebooks using the `openai` Python client (with a fallback to `distllm_sdk.compat.openai_compat` when `openai` is not installed). All target the OpenAI-compatible endpoint `http://localhost:8000/v1` (override via `DISTLLM_BASE_URL`, `DISTLLM_MODEL`, `DISTLLM_API_KEY`).

| Notebook | What it demonstrates | API endpoint used |
|----------|----------------------|-------------------|
| `rag.ipynb` | Self-contained RAG: embed sample docs, build an in-notebook numpy/sklearn vector store, retrieve top-k, generate a grounded answer | `/v1/embeddings`, `/v1/chat/completions` |
| `streaming.ipynb` | Token-by-token streaming chat, plus a run with `stream_options={"include_usage": True}` | `/v1/chat/completions` (streaming) |
| `batch_processing.ipynb` | Batch prompts through chat completions; measures throughput (tokens/sec) sequentially and via `ThreadPoolExecutor` | `/v1/chat/completions` |
| `cost_analysis.ipynb` | Collects per-request token usage/cost from `response.usage` (incl. optional `cost_usd`), aggregates via `stats().estimate_cost()`, plots a bar chart with `matplotlib` | `/v1/chat/completions` |

**Prerequisites**
```bash
pip install jupyter openai distllm_sdk numpy scikit-learn matplotlib
```

**Start the server**
```bash
pip install -e ./sdk                       # if distllm_sdk is not installed
distllm-api --model distributed-llm --host 0.0.0.0 --port 8000
curl http://localhost:8000/v1/models       # verify
```

**Run**
```bash
cd examples/notebooks
jupyter notebook        # open a .ipynb and Run All
# or headless:
jupyter nbconvert --to notebook --execute --inplace rag.ipynb
```

Each notebook prints a clear, non-crashing message if the server is unreachable (and some fall back to dummy data), so the rest of the notebook still executes.

---

## References
- VS Code extension README: [`extensions/vscode/README.md`](../extensions/vscode/README.md)
- Apps README: [`apps/README.md`](../apps/README.md)
- Notebooks README: [`examples/notebooks/README.md`](../examples/notebooks/README.md)
- Framework integration examples: [`examples/README.md`](../examples/README.md)
