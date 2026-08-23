---
tags:
  - integrations
  - sdk
  - prompts
  - plugins
aliases:
  - Integrations
  - SDK
---
# Integrations & SDK — `integrations/` + `sdk/` + `prompts/` + `plugins/` + in-tree `integrations`/`sdk`

> The **compatibility & client surface**: framework adapters so external LLM orchestration libraries (LangChain, LlamaIndex, CrewAI, AutoGen, Haystack, …) drive DistLLM through their own native API; multi-language client SDKs (python/go/js/rust); a curated **prompt library + exchange**; the **hook-based plugin framework**. Everything talks to the engine over its OpenAI-compatible HTTP/gRPC surface (`distllm.sdk` → [[03 API Server]]).

## Framework adapters (`integrations/` — 28 dirs, 174 files)

| Adapter | Purpose |
|---------|---------|
| `langchain/` | `DistLLMChat`/`DistLLM`/`Embeddings`/`ToolProvider` (BaseChatModel) + tests |
| `llamaindex/` | `DistLLM` LLM, `DistLLMEmbeddings`, `ToolProvider` + tests |
| `crewai/` | `DistLLMCrewLLM`/`Embedder`/`KnowledgeSource`/`ToolProvider` + tests |
| `agno/` | `DistLLM` (OpenAIChat wrap), `Embedder`, `ToolProvider` |
| `autogen/` | `DistLLMConfig` llm_config builder |
| `autogpt/` | `DistLLMAutoGPTConfig` + launcher |
| `haystack/` | `DistLLMGenerator`/`TextEmbedder` (OpenAI subclasses) |
| `semantic_kernel/` | `DistLLMChatCompletion`/`EmbeddingService` |
| `litellm/` | `get_distllm_custom_llm` — LiteLLM custom backend |
| `one-api/` | register DistLLM as one-api provider |
| `openai-agents/` | `DistLLMAgentModel`/`ModelProvider`/`Runner` |
| `ollama_compat/` | FastAPI translating Ollama `/api/*` → DistLLM |
| `fastapi_middleware/` | `DistLLMMiddleware`/`create_distllm_router` proxy |
| `grpc_client/` | `DistLLMGrpcClient` async gRPC |
| `dify/` | single drop-in model-provider plugin |
| `_common/` | `BaseToolProvider`, `CostTracker`, `DistLLMModelRouter` (shared core) |
| **IaC/infra** | `terraform/` (Go provider: model/deployment/node/federation resources + tests), `pulumi/` (TS provider+resources), `aws-cdk/` (EC2 GPU fleet), `ansible/` (playbook+roles), `kubernetes/` (Helm chart, kustomize overlays, **kopf operator** w/ CRD + tests), `docker/` (compose), `grafana/` |
| Observability/CI | `wandb/` package (`tracker.py` 654), `mlflow_tracking`, `datadog_monitoring` (1046), `gitlab_ci`, `spark_connector` (764), `vector_db_pack`, `content_moderation` (1200), `gpu_spot_orchestrator` (1536) |
| **README-only stubs** | `genkit/`, `openwebui/`, `portkey/`, `spring-ai/` |

## Multi-language SDK (`sdk/` — 58 files)

| Root | Purpose |
|------|---------|
| `src/distllm_sdk/` (python) | `DistLLMClient`/`DistLLMClientSync`, typed responses, errors, circuit breaker, retry; client→vectorstore/cache, grpc_client, mcp_server, monitoring, admin, ab_testing, eval_harness, benchmark, tokenizer, privacy, synthetic_data, tracing, validator, cli; **framework adapters** (dspy, instructor, vercel_ai, openai_agents, portkey); `compat/openai_compat` |
| `go/` | `client.go` (351) + generated endpoints/types |
| `rust/` | `src/lib.rs` (296) + generated types |
| `js/` | `src/index.ts` (313) + generated |
| `openapi/` | `distllm.yaml` spec (446) + `generate.py` emits per-language stubs |
| `tests/` | `test_sdk.py`, `test_client_extended.py`, `test_grpc_client.py` |
| in-tree `src/distllm/sdk` | the **self-contained Python SDK** (~1.8K LOC) — `client.py` (1108), streaming (WS/client), transport, circuit_breaker, types, observability |

## In-tree `integrations/` (src/distllm/integrations — 12 files)

| file | LOC | purpose |
|------|-----|---------|
| `__init__.py` | 57 | namespace pkg; `_discover_*` extends `__path__` from entry points |
| `mlflow_tracking.py` | 308 | `MLflowIntegration` auto/thread-safe logging |
| `spark_connector.py` | 764 | `DistLLMSparkTransformer` PySpark batch inference |
| `ci/gitlab.py` + `ci/jenkins.py` | 292/226 | CI eval pipelines, artifact pull, MR comments |
| `ci/_common.py` | 73 | shared retries + frozen dataclasses |
| `_common/` | 3 | `BaseToolProvider`, `CostTracker`, `DistLLMModelRouter` |
| `wandb/__init__.py` | 56 | shim → external `distllm-wandb` package |

## `prompts/` — prompt library & template engine (17 files, ~3K LOC)

| file | LOC | purpose |
|------|-----|---------|
| `library.py` | 690 | **live** curated library: `SYSTEM_PROMPTS` (~54 prompts, 10 categories), lookups |
| `engine.py` | 96 | `TemplateEngine` (custom → built-in → auto-detect → tokenizer → fallback) |
| `templates.py` | 110 | chatml/llama2/llama3/mistral/zephyr/alpaca + `auto_detect_template` |
| `exchange.py` | 619 | `PromptExchange` token-gated community marketplace (publish/browse/fork/reviews) |
| `prompt_def.py`, `management.py`, `code/analysis/creative/…` (11 modules) | — | **dead parallel** registry — duplicates `library.py`, unreferenced |

## `plugins/` — hook plugin framework (9 files)

| file | LOC | purpose |
|------|-----|---------|
| `registry.py` | 108 | `PluginRegistry` — entry-point discovery, register, install |
| `builtin.py` | 299 | `RateLimitPlugin`, `AuditLogPlugin`, `MetricsPlugin`, `ContentModerationPlugin` |
| `auth_plugin.py` | 393 | `AuthPlugin` — JWT validation + RBAC + per-role rate limit |
| `cache_plugin.py` | 451 | LRU + Redis response cache |
| `health_plugin.py` | 333 | `/healthz` `/readyz` + watchdog + circuit breaker |
| `airflow.py`/`kubeflow.py`/`mlflow_plugin.py` | 91/69/60 | job-submit+poll integrations |
| `__init__.py` | 11 | docstring |

## Notes / dead code
- **Two prompt libraries:** `library.py` is the live one; `prompt_def.py` + all category modules are a dead parallel registration path.
- **Three OpenAI-compat clients** diverge: `sdk/src.../compat/openai_compat.py`, `sdk/compat/openai_compat.py` (older), `distllm_sdk/compat`.
- **Duplicate type ecosystems** across `sdk` (types / types_dataclass / generated / compat) — 4 families.
- **Version skew** in `sdk/`: pyproject 1.0.0, rust 0.4.0, built wheel 0.5.0.
- **README-only adapter stubs:** `genkit`, `openwebui`, `portkey`, `spring-ai`.
- `_common` fallback imports (`distllm.integrations._common` vs `_common`).
- `mlflow_plugin.py` does **not** subclass `PluginBase`.

## Tests
Per-adapter suites: `integrations/{langchain,llamaindex,crewai}/tests`, `kubernetes/operator/tests`, Go provider `provider_test.go`; `sdk/tests/*`; `tests/integrations/` (CI, MLflow, Spark, OpenAI-Agents); `src/distllm/integrations` → `tests/integrations/`. Prompts/plugins → `tests/prompts/`, `tests/plugins/`, `tests/api/test_exchange.py`.