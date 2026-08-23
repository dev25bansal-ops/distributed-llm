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

## Examples

### 1. SDK Example (Recommended)

Direct Python SDK usage with async/sync clients.

```bash
pip install distllm[sdk]
python examples/sdk_example.py
```

### 2. LangChain Integration

Use DistLLM as an LLM backend for LangChain applications.

```bash
pip install distllm[examples]
python examples/langchain_example.py
```

### 3. LlamaIndex Integration

Use DistLLM with LlamaIndex for RAG applications.

```bash
pip install distllm[examples]
python examples/llamaindex_example.py
```

### 4. CrewAI Integration

Use DistLLM with CrewAI for multi-agent workflows.

```bash
pip install distllm[examples]
python examples/crewai_example.py
```

### 5. Federated Training Demo (no server needed)

Two in-process `FederatedFineTuner` nodes exchange differentially-private
LoRA gradients and print the cumulative RDP epsilon spend per round.
Runs fully offline — no API server, no model download, only PyTorch.

```bash
pip install torch
python examples/federated_training_demo.py
```

See the [Federated Training docs](/docs/federated-training) for details.

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
