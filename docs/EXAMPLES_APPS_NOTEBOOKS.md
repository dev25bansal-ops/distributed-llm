# Examples, Apps & Notebooks

This page summarizes the ready-to-use examples shipped with DistLLM:

- The **VS Code extension** (editor integration)
- Seven **Jupyter notebooks** under `examples/notebooks/`

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

`apps/` was removed in the 2026-11-06 hygiene pass; live examples are `examples/` + `examples/notebooks/`.

---

## 3. Example Notebooks (`examples/notebooks/`)

Seven runnable Jupyter notebooks using the `openai` Python client (with a fallback to `distllm_sdk.compat.openai_compat` when `openai` is not installed). All target the OpenAI-compatible endpoint `http://localhost:8000/v1` (override via `DISTLLM_BASE_URL`, `DISTLLM_MODEL`, `DISTLLM_API_KEY`). The four API-demo notebooks below are documented in `examples/notebooks/README.md`; the other three are cloud-worker notebooks (Kaggle, Lightning.ai, Colab) that join a laptop cluster as a DistLLM worker.

| Notebook | What it demonstrates | API endpoint used |
|----------|----------------------|-------------------|
| `rag.ipynb` | Self-contained RAG: embed sample docs, build an in-notebook numpy/sklearn vector store, retrieve top-k, generate a grounded answer | `/v1/embeddings`, `/v1/chat/completions` |
| `streaming.ipynb` | Token-by-token streaming chat, plus a run with `stream_options={"include_usage": True}` | `/v1/chat/completions` (streaming) |
| `batch_processing.ipynb` | Batch prompts through chat completions; measures throughput (tokens/sec) sequentially and via `ThreadPoolExecutor` | `/v1/chat/completions` |
| `cost_analysis.ipynb` | Collects per-request token usage/cost from `response.usage` (incl. optional `cost_usd`), aggregates via `stats().estimate_cost()`, plots a bar chart with `matplotlib` | `/v1/chat/completions` |
| `distllm_kaggle_worker.ipynb` | Run a real DistLLM worker on free Kaggle GPUs (T4x2) joining a laptop cluster | — |
| `distllm_lightning_worker.ipynb` | Run a real DistLLM worker on Lightning.ai Studio GPUs joining a laptop cluster | — |
| `tinylm_colab_worker.ipynb` | Run a real DistLLM worker on free Colab T4 joining a laptop cluster | — |

**Prerequisites**
```bash
pip install jupyter openai distllm_sdk numpy scikit-learn matplotlib
```

**How to run:** `distllm-api --model <model> --host 0.0.0.0 --port 8000` (entry point `distllm-api = "distllm.api.server:main"`, `pyproject.toml` [project.scripts]; `--model/--host/--port` defined in `src/distllm/api/server.py` `main()`). Then verify with `curl http://localhost:8000/v1/models`.

> ⚠️ UNVERIFIED (hygiene pass 2026-11-06): `distllm-api`'s server-side default model name is config-driven, not the literal `distributed-llm` (no such default found in `src/distllm/config/`); `--model distributed-llm` is kept from the original text.

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
- Notebooks README: [`examples/notebooks/README.md`](../examples/notebooks/README.md)
- Framework integration examples: [`examples/README.md`](../examples/README.md)
