# DistLLM Integration Examples

This directory contains examples showing how to integrate DistLLM with popular AI frameworks.

## Prerequisites

Start the DistLLM API server before running any example:

```bash
# Local mode (single machine)
distllm-api --model roneneldan/TinyStories-1M --local

# Or with a custom model
distllm-api --model your-model-name --local --port 8000
```

The API server will be available at `http://localhost:8000`.

## Installation

All examples require the `distllm` package. Install the extras that match the examples you want to run:

```bash
# SDK example only
pip install distributed-llm[sdk]

# LangChain / LlamaIndex / CrewAI / Haystack / Agency Swarm examples
pip install distributed-llm[examples]

# OpenAI Agents SDK example
pip install distributed-llm[openai-agents]

# Everything (all framework extras + dev tools)
pip install distributed-llm[all-dev]
```

## Examples

### 1. SDK Example (Recommended)

Direct Python SDK usage with async/sync clients.

```bash
pip install distributed-llm[sdk]
python examples/sdk_example.py
```

### 2. LangChain Integration

Use DistLLM as an LLM backend for LangChain applications.

```bash
pip install distributed-llm[examples]
python examples/langchain_example.py
```

### 3. LlamaIndex Integration

Use DistLLM with LlamaIndex for RAG applications.

```bash
pip install distributed-llm[examples]
python examples/llamaindex_example.py
```

### 4. CrewAI Integration

Use DistLLM with CrewAI for multi-agent workflows.

```bash
pip install distributed-llm[examples]
python examples/crewai_example.py
```

### 5. Haystack Integration

Build RAG pipelines with Haystack (deepset) on top of DistLLM's OpenAI-compatible endpoint — document store, BM25 retriever, prompt builder, and `OpenAIGenerator` pointed at the coordinator.

```bash
pip install distributed-llm[examples]
python examples/haystack_example.py
```

Requires `haystack-ai>=2.0.0` (included in the `examples` extra). See [`integrations/haystack`](../integrations/haystack) for the dedicated `distllm-haystack` components.

### 6. OpenAI Agents SDK Integration

Demonstrates agents, tool calling, handoffs, and streaming with the `openai-agents` SDK, using `DistLLMAgentModel` / `DistLLMModelProvider` from `distllm.integrations.openai_agents`.

```bash
pip install distributed-llm[openai-agents]
python examples/openai_agents_example.py
```

Requires `openai-agents>=0.1.0`. See [`integrations/openai-agents`](../integrations/openai-agents) for details.

### 7. Agency Swarm Integration

Orchestrate multi-agent workflows with Agency Swarm, backed by the DistLLM OpenAI-compatible API.

```bash
pip install distributed-llm[examples]
python examples/agency_swarm_example.py
```

Requires `agency-swarm>=0.2.0` and `openai>=1.0.0` (both included in the `examples` extra).

### 8. Federated Training Demo (no server needed)

Two in-process `FederatedFineTuner` nodes exchange differentially-private LoRA gradients and print the cumulative RDP epsilon spend per round. Runs fully offline — no API server, no model download, only PyTorch.

**Prerequisites:** Install `torch`. This example does not require the API server or the `examples` extra.

```bash
pip install torch
python examples/federated_training_demo.py
```

See the [Federated Training docs](/docs/federated-training) for details.

### 9. Experimental: Multi-Provider Routing (`routed_inference.py`)

**Status: EXPERIMENTAL / PLANNED API — illustrative only.** This example sketches how multi-provider routing is intended to work from the DistLLM control plane. It sends `provider` and `provider_fallbacks` fields in the request `extra_body`:

```python
client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[...],
    extra_body={"provider": "cheapest"},
)
```

These parameters are **not implemented** in the current coordinator/API server; running the script against a live server will not perform routing as shown. The file is kept as a design illustration and will be wired up when provider routing lands.

```bash
export OPENAI_API_KEY=sk-...
export TOGETHER_API_KEY=...
python examples/routed_inference.py   # illustrative; routing not yet wired
```

Related existing utilities live in [`integrations/_common`](../integrations/_common) (`model_router.py`, `cost_tracker.py`).

## Configuration

All examples connect to the OpenAI-compatible API at `http://localhost:8000/v1`.

To use a different API URL or add authentication:

```python
# LangChain
llm = ChatOpenAI(
    model="distributed-llm",
    openai_api_base="http://your-server:8000/v1",
    openai_api_key="your-api-key",  # Optional
)

# LlamaIndex
Settings.llm = OpenAI(
    model="distributed-llm",
    api_base="http://your-server:8000/v1",
    api_key="your-api-key",
)

# SDK
client = DistLLMClient(
    base_url="http://your-server:8000",
    api_key="your-api-key",
)
```

## API Documentation

Full API documentation with interactive examples is available at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
