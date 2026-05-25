# Provider Setup Guide

DistLLM supports routing requests to any provider with an OpenAI-compatible API.

## Supported Providers

| Provider | Environment Variable | Base URL | Notes |
|----------|---------------------|----------|-------|
| OpenAI | `OPENAI_API_KEY` | `https://api.openai.com/v1` | Default |
| Anthropic | `ANTHROPIC_API_KEY` | `https://api.anthropic.com/v1` | Via proxy |
| Together AI | `TOGETHER_API_KEY` | `https://api.together.xyz/v1` | Best model catalog |
| Fireworks AI | `FIREWORKS_API_KEY` | `https://api.fireworks.ai/inference/v1` | Fast inference |
| Groq | `GROQ_API_KEY` | `https://api.groq.com/openai/v1` | Lowest latency |
| DeepInfra | `DEEPINFRA_API_KEY` | `https://api.deepinfra.com/v1/openai` | Cheapest |
| vLLM (self-hosted) | `VLLM_BASE_URL` | Custom | For self-hosted |

## Configuration

Create `config.yaml`:

```yaml
routing:
  strategy: "cost"  # cost, latency, or manual
  fallback_enabled: true
  max_retries: 3

providers:
  openai:
    enabled: true
    models:
      - "gpt-4o"
      - "gpt-4o-mini"
  together:
    enabled: true
    models:
      - "meta-llama/Llama-3.3-70B-Instruct"
  fireworks:
    enabled: true
    models:
      - "accounts/fireworks/models/llama-v3p3-70b-instruct"

cost_optimization:
  enabled: true
  check_interval_seconds: 300
  max_budget_per_month: 1000
```

## Custom Provider

Add any OpenAI-compatible API as a custom provider:

```yaml
providers:
  custom:
    base_url: "https://your-proxy.com/v1"
    api_key_env: "CUSTOM_API_KEY"
    models:
      - "your-custom-model"
```
