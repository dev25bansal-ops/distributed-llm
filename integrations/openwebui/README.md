# Open WebUI + DistLLM Integration

Connect [Open WebUI](https://openwebui.com) to DistLLM as an OpenAI-compatible provider.

## Prerequisites

- Running DistLLM coordinator with API server (default: `http://localhost:8000`)
- Open WebUI installed (Docker, local, or cloud)

## Configuration

### Step 1: Open Open WebUI Admin Panel

Navigate to **Admin Panel → Settings → Connections**.

### Step 2: Add OpenAI-Compatible Provider

| Field | Value |
|-------|-------|
| **Provider Name** | DistLLM |
| **API Base URL** | `http://host.docker.internal:8000/v1` (Docker) or `http://localhost:8000/v1` (native) |
| **API Key** | Any value (leave blank or use any string) |
| **Model** | `distributed-llm` |

### Step 3: Select Model

In any conversation, select the **DistLLM** model from the model dropdown.

## Docker Deployment

```yaml
services:
  distllm:
    image: distributed-llm:latest
    ports:
      - "8000:8000"
    networks:
      - openwebui-net

  open-webui:
    image: ghcr.io/open-webui/open-webui:main
    ports:
      - "3000:8080"
    environment:
      - OPENAI_API_BASE_URL=http://distllm:8000/v1
      - OPENAI_API_KEY=unused
    networks:
      - openwebui-net

networks:
  openwebui-net:
```

## Troubleshooting

- **Connection refused**: Ensure DistLLM is running and accessible from Open WebUI
- **No models visible**: Verify DistLLM API responds at `/v1/models`
- **Docker network**: Use `http://host.docker.internal:8000/v1` for host DistLLM from Docker Open WebUI
- **CORS errors**: DistLLM API must include CORS headers (enabled by default)
