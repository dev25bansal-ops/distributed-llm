# Quickstart: Go from Zero to Routed Inference in 5 Minutes

This guide shows how to use DistLLM as a multi-provider inference control plane.

## 1. Installation

```bash
git clone https://github.com/distributed-llm/distributed-llm.git
cd distributed-llm
pip install -e .
```

## 2. Set Up API Keys

Create a `.env` file:

```bash
DISTLLM_API_KEY=sk-your-master-key
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
TOGETHER_API_KEY=...
FIREWORKS_API_KEY=...
GROQ_API_KEY=...
```

## 3. Start the Control Plane

```bash
distllm-api
```

The server starts on `http://localhost:8000` with auto-routing enabled.

## 4. Send Your First Routed Request

```python
import openai

client = openai.Client(
    base_url="http://localhost:8000/v1",
    api_key="sk-your-master-key",
)

# Route to the cheapest provider
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello!"}],
    extra_body={"provider": "cheapest"},
)
print(response.choices[0].message.content)
```

## 5. Route by Latency

```python
# Route to the fastest provider
response = client.chat.completions.create(
    model="claude-3-haiku",
    messages=[{"role": "user", "content": "Explain quantum computing"}],
    extra_body={"provider": "fastest"},
)
```

## 6. Set Up Fallbacks

```python
# Auto-failover if primary fails
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello!"}],
    extra_body={
        "provider_fallbacks": ["together", "fireworks"],
    },
)
```

## 7. Check Costs

```bash
curl http://localhost:8000/v1/costs -H "Authorization: Bearer $API_KEY"
```

## Next Steps

- [Provider Setup Guide](providers.md) — Configure 10+ providers
- [Self-Hosted Deployment](self-hosted.md) — Run models on your own GPUs
- [API Reference](api.md) — Full OpenAI-compatible API
