# Portkey Observability Integration

[Portkey](https://portkey.ai) is an observability and gateway proxy for LLM
applications.  Since DistLLM exposes an OpenAI-compatible API, Portkey can
monitor all DistLLM traffic with zero code changes.

## Quick Start

```bash
# 1. Set your DistLLM endpoint as the OpenAI base URL in Portkey
export OPENAI_BASE_URL="http://localhost:8000/v1"
export OPENAI_API_KEY="your-distllm-api-key"  # optional

# 2. Run the Portkey gateway (via their SDK or Docker)
pip install portkey-ai
```

```python
from portkey import Portkey

# Point Portkey at DistLLM
client = Portkey(
    base_url="http://localhost:8000/v1",
    api_key="your-distllm-api-key",
)

# All calls are automatically logged with full observability
response = client.chat.completions.create(
    model="llama-3-70b",
    messages=[{"role": "user", "content": "Hello!"}],
)
```

## Using with the DistLLM SDK

```python
from distllm_sdk import DistLLMClient
from distllm_sdk.portkey_integration import PortkeyMonitor

monitor = PortkeyMonitor(api_key="pk-...")
client = DistLLMClient(base_url="http://localhost:8000", api_key="...")
monitor.wrap(client)

# All calls are now traced through Portkey
response = await client.chat_completions(...)
```

## Benefits

- **Request/response logging** — full trace of every LLM call
- **Latency monitoring** — P50/P95/P99 per model and endpoint
- **Cost tracking** — per-request and aggregated cost breakdowns
- **Error alerts** — real-time notification on failures
- **User analytics** — per-tenant usage patterns
