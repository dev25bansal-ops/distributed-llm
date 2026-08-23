# CLAUDE.md — Integrations & SDK

## Your Scope
You have ownership of `integrations/` and `sdk/` — all framework integrations and multi-language SDKs.

## Do NOT Touch
- `src/distllm/` — core source code
- `tests/` — handled by another instance

## Key Directories

| Directory | Status | Purpose |
|-----------|--------|---------|
| `integrations/langchain/` | Exists | LangChain integration |
| `integrations/llamaindex/` | Exists | LlamaIndex integration |
| `integrations/crewai/` | Exists | CrewAI integration |
| `integrations/litellm/` | Exists | LiteLLM custom provider |
| `integrations/haystack/` | Exists | Haystack generator + embedder |
| `integrations/autogen/` | Exists | AutoGen config helper |
| `integrations/semantic_kernel/` | Exists | Semantic Kernel connector |
| `integrations/one-api/` | **NEW** | one-api provider module |
| `integrations/ollama_compat/` | Exists | Ollama compatibility |
| `integrations/openwebui/` | Exists | OpenWebUI integration |
| `integrations/fastapi_middleware/` | Exists | FastAPI middleware |
| `integrations/grpc_client/` | Exists | gRPC client library |
| `integrations/kubernetes/` | Exists | Helm chart, Dockerfile |
| `integrations/terraform/` | Exists | Terraform provider |
| `integrations/pulumi/` | Exists | Pulumi provider |
| `integrations/ansible/` | Exists | Ansible playbooks |
| `integrations/grafana/` | Exists | Grafana dashboards |
| `integrations/docker/` | Exists | Docker compose + monitoring |
| `sdk/` | Exists | Python, Go, JS, Rust SDKs + OpenAPI |

## Commands
- `python -m pytest integrations/langchain/tests/ -v` — LangChain tests
- `python -m pytest integrations/litellm/tests/ -v` — LiteLLM tests
- `cd sdk/go && go test ./...` — Go SDK tests
- `cd sdk/js && npm test` — JS SDK tests
