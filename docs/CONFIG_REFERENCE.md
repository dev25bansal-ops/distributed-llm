# DistLLM Configuration Reference

> Complete reference for all configuration options in DistLLM.
>
> **Config source:** `src/distllm/config/settings.py` + domain modules in `src/distllm/config/_*.py`
>
> **Precedence:** CLI overrides > Environment variables > YAML file > Defaults

---

## Table of Contents

- [Environment Variable Pattern](#environment-variable-pattern)
- [Loading Configuration](#loading-configuration)
- [Model Settings](#1-model-settings)
- [Quantization Settings](#2-quantization-settings)
- [Speculative Decoding Settings](#3-speculative-decoding-settings)
- [LoRA Settings](#4-lora-settings)
- [SLoRA Settings](#5-slora-settings)
- [MoE Settings](#6-moe-settings)
- [Multi-Model Settings](#7-multi-model-settings)
- [Compression Settings](#8-compression-settings)
- [Adaptive Compression Settings](#9-adaptive-compression-settings)
- [Model Hub Settings](#10-model-hub-settings)
- [Embedding Settings](#11-embedding-settings)
- [Prompt Template Settings](#12-prompt-template-settings)
- [Coordinator Settings](#13-coordinator-settings)
- [Network Settings](#14-network-settings)
- [TLS Settings](#15-tls-settings)
- [Rate Limit Settings](#16-rate-limit-settings)
- [Wide Area Settings](#17-wide-area-settings)
- [Chat Router Settings](#18-chat-router-settings)
- [Prefix Cache Settings](#19-prefix-cache-settings)
- [Cache Persistence Settings](#20-cache-persistence-settings)
- [Gossip Settings](#21-gossip-settings)
- [Predictive Cache Settings](#22-predictive-cache-settings)
- [Unified Cache Settings](#23-unified-cache-settings)
- [Defragmentation Settings](#24-defragmentation-settings)
- [Node Settings](#25-node-settings)
- [Tensor Parallel Settings](#26-tensor-parallel-settings)
- [Hybrid Parallel Settings](#27-hybrid-parallel-settings)
- [Zero-Copy Settings](#28-zero-copy-settings)
- [Partitioning Settings](#29-partitioning-settings)
- [Rebalancer Settings](#30-rebalancer-settings)
- [Batching Settings](#31-batching-settings)
- [Chunked Prefill Settings](#32-chunked-prefill-settings)
- [Priority Settings](#33-priority-settings)
- [Disaggregation Settings](#34-disaggregation-settings)
- [CUDA Graph Settings](#35-cuda-graph-settings)
- [Compile Settings](#36-compile-settings)
- [Adaptive Precision Settings](#37-adaptive-precision-settings)
- [Self-Optimizing Settings](#38-self-optimizing-settings)
- [Hardware Settings](#39-hardware-settings)
- [vLLM Backend Settings](#40-vllm-backend-settings)
- [llama.cpp Backend Settings](#41-llamacpp-backend-settings)
- [Generation Settings](#42-generation-settings)
- [Monitoring Settings](#43-monitoring-settings)
- [Alerting Settings](#44-alerting-settings)
- [Chaos Settings](#45-chaos-settings)
- [Canary Settings](#46-canary-settings)
- [Version Settings](#47-version-settings)
- [Cost Settings](#48-cost-settings)
- [Tenant Settings](#49-tenant-settings)
- [RAG Settings](#50-rag-settings)
- [Agent Settings](#51-agent-settings)
- [Plugin Settings](#52-plugin-settings)
- [Top-Level Flat Fields](#top-level-flat-fields)
- [Complete YAML Example](#complete-yaml-example)

---

## Environment Variable Pattern

All settings are accessible via environment variables using the pattern:

```
DISTLLM__<SECTION>__<FIELD>
```

The delimiter between nesting levels is `__` (double underscore).

| Scope | Pattern | Example |
|-------|---------|---------|
| Top-level | `DISTLLM__FIELD` | `DISTLLM__NODES` |
| Nested section | `DISTLLM__SECTION__FIELD` | `DISTLLM__MODEL__NAME` |
| Deeply nested | `DISTLLM__SECTION__SUBSECTION__FIELD` | `DISTLLM__CACHE__PREFIX_MAX_ENTRIES` |

Environment variables take precedence over YAML config values.

A `.env` file in the working directory is also loaded automatically.

---

## Loading Configuration

```python
from distllm.config.settings import DistLLMSettings

# From defaults + env vars only
settings = DistLLMSettings()

# From YAML file
settings = DistLLMSettings.from_yaml("config.yaml")

# From YAML with CLI overrides
settings = DistLLMSettings.from_yaml("config.yaml", cli_overrides={"model": {"name": "llama3"}})

# From profile-based YAML (dev / staging / production)
settings = DistLLMSettings.from_profile("config.yaml", profile="production")

# Startup validation (prints errors and exits on failure)
settings = DistLLMSettings.validate_startup("config.yaml")
```

---

## 1. Model Settings

**Section key:** `model`
**Class:** `ModelSettings`
**Source:** `_model.py`

Model identification and loading configuration.

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `name` | `str` | `""` | Model name or path. Must be explicitly set — HuggingFace model ID (e.g. `meta-llama/Llama-2-7b`) or local path. **Required.** |
| `dtype` | `str` | `"float16"` | Data type for model weights. Allowed: `float16`, `float32`, `bfloat16`. |
| `trust_remote_code` | `bool` | `False` | Whether to trust remote code when loading models from HuggingFace. |

**Validators:**
- `name`: Must not be empty or whitespace-only. Raises `ValueError` with instructions to set `DISTLLM__MODEL__NAME`.
- `dtype`: Must be one of `float16`, `float32`, `bfloat16`.

```yaml
model:
  name: meta-llama/Llama-2-7b
  dtype: bfloat16
  trust_remote_code: false
```

---

## 2. Quantization Settings

**Section key:** `quantization`
**Class:** `QuantizationSettings`
**Source:** `_model.py`

Quantization configuration for model loading. Supports BitsAndBytes, GPTQ, AWQ, and FP8.

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `method` | `str` | `"none"` | Quantization method. Allowed: `none`, `bnb_4bit`, `bnb_8bit`, `gptq`, `awq`, `fp8`. |
| `bnb_4bit_compute_dtype` | `str` | `"float16"` | Compute dtype for BnB 4-bit quantization. |
| `bnb_4bit_quant_type` | `str` | `"nf4"` | Quantization type for BnB 4-bit (e.g. `nf4`, `fp4`). |
| `bnb_4bit_use_double_quant` | `bool` | `True` | Enable double quantization for BnB 4-bit. |
| `llm_int8_threshold` | `float` | `6.0` | Threshold for LLM.int8() outlier detection. |
| `gptq_bits` | `int` | `4` | GPTQ quantization bits. Allowed: `4`, `8`. |
| `gptq_group_size` | `int` | `128` | GPTQ group size. |
| `gptq_desc_act` | `bool` | `False` | GPTQ descending activation order. |
| `gptq_use_marlin` | `bool` | `True` | Use Marlin kernel for Hopper GPUs with GPTQ. |
| `awq_bits` | `int` | `4` | AWQ quantization bits. Allowed: `4`, `8`. |
| `awq_group_size` | `int` | `128` | AWQ group size. |
| `fp8_scheme` | `str` | `"e4m3"` | FP8 format scheme. Allowed: `e4m3`, `e5m2`. |
| `fp8_dynamic` | `bool` | `True` | Enable dynamic FP8 scaling. |
| `kv_cache_quant` | `bool` | `False` | Enable KV cache quantization. |
| `kv_cache_bits` | `int` | `8` | KV cache quantization bits. Allowed: `4`, `8`. |

**Validators:**
- `method`: Must be one of `none`, `bnb_4bit`, `bnb_8bit`, `gptq`, `awq`, `fp8`.
- `gptq_bits`, `awq_bits`: Must be 4 or 8.
- `kv_cache_bits`: Must be 4 or 8.

```yaml
quantization:
  method: bnb_4bit
  bnb_4bit_compute_dtype: bfloat16
  bnb_4bit_quant_type: nf4
  bnb_4bit_use_double_quant: true
  kv_cache_quant: true
  kv_cache_bits: 8
```

---

## 3. Speculative Decoding Settings

**Section key:** `speculative`
**Class:** `SpeculativeSettings`
**Source:** `_model.py`

Speculative decoding configuration for faster inference using draft models.

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `draft_model` | `str` | `""` | Draft model name or path for speculative decoding. |
| `num_assistant_tokens` | `int` | `5` | Number of tokens to speculate per step. Must be >= 1. |
| `min_acceptance_rate` | `float` | `0.3` | Minimum acceptance rate threshold. |
| `warmup_steps` | `int` | `10` | Warmup steps before enabling speculation. |
| `method` | `str` | `"draft_model"` | Speculative method. Allowed: `draft_model`, `medusa`, `eagle`, `ngram`, `auto`. |
| `medusa_num_heads` | `int` | `4` | Number of Medusa prediction heads. |
| `medusa_num_tokens_per_head` | `int` | `3` | Tokens per Medusa head. |
| `eagle_checkpoint` | `str` | `""` | Path to EAGLE checkpoint. |
| `eagle_variant` | `str` | `"eagle"` | EAGLE model variant name. |
| `eagle_hidden_size` | `int` | `4096` | EAGLE hidden layer size. |
| `eagle_vocab_size` | `int` | `32000` | EAGLE vocabulary size. |
| `eagle_num_layers` | `int` | `2` | EAGLE number of layers. |
| `ngram_min_match` | `int` | `4` | Minimum n-gram match length for ngram method. |

**Validators:**
- `num_assistant_tokens`: Must be >= 1.
- `method`: Must be one of `draft_model`, `medusa`, `eagle`, `ngram`, `auto`.

```yaml
speculative:
  method: draft_model
  draft_model: meta-llama/Llama-2-7b
  num_assistant_tokens: 5
  min_acceptance_rate: 0.3
```

---

## 4. LoRA Settings

**Section key:** `lora`
**Class:** `LoRASettings`
**Source:** `_model.py`

LoRA multi-adapter configuration.

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `enabled` | `bool` | `False` | Enable LoRA adapter serving. |
| `adapters` | `dict[str, str]` | `{}` | Map of adapter name to adapter path. |

```yaml
lora:
  enabled: true
  adapters:
    code: /path/to/code-adapter
    chat: /path/to/chat-adapter
```

---

## 5. SLoRA Settings

**Section key:** `slora`
**Class:** `SloRaSettings`
**Source:** `_model.py`

SLoRA multi-adapter serving for concurrent adapter usage.

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `enabled` | `bool` | `False` | Enable SLoRA serving. |
| `max_adapters` | `int` | `64` | Maximum number of concurrent adapters. |

```yaml
slora:
  enabled: true
  max_adapters: 32
```

---

## 6. MoE Settings

**Section key:** `moe`
**Class:** `MoESettings`
**Source:** `_model.py`

Mixture of Experts configuration.

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `enabled` | `bool` | `False` | Enable MoE routing. |
| `num_experts` | `int` | `8` | Total number of experts. |
| `num_experts_per_tok` | `int` | `2` | Number of experts activated per token. |

```yaml
moe:
  enabled: true
  num_experts: 8
  num_experts_per_tok: 2
```

---

## 7. Multi-Model Settings

**Section key:** `multi_model`
**Class:** `MultiModelSettings`
**Source:** `_model.py`

Multi-model serving configuration. The `max_models` limit is a safety cap — actual capacity depends on available GPU memory.

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `models` | `dict[str, str]` | `{}` | Map of model name to model path. |
| `default_model` | `str` | `""` | Default model to serve. |
| `max_models` | `int` | `4` | Maximum number of models to load concurrently. Must be >= 1. |

```yaml
multi_model:
  models:
    llama3: meta-llama/Llama-3-8B
    codellama: codellama/CodeLlama-7b
  default_model: llama3
  max_models: 4
```

---

## 8. Compression Settings

**Section key:** `compression`
**Class:** `CompressionSettings`
**Source:** `_model.py`

Model compression pipeline configuration.

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `enabled` | `bool` | `False` | Enable model compression. |
| `method` | `str` | `"none"` | Compression method. |
| `target_bits` | `int` | `8` | Target bit width. Allowed: `4`, `8`. |
| `pruning_ratio` | `float` | `0.0` | Pruning ratio (0.0–1.0). |
| `distillation_teacher` | `str \| None` | `None` | Teacher model path for distillation. |
| `calibration_samples` | `int` | `128` | Number of calibration samples. |
| `pruning_targets` | `list[str]` | `["q_proj", "v_proj"]` | Target layers for pruning. |

**Validators:**
- `target_bits`: Must be 4 or 8.
- `pruning_ratio`: Must be 0.0–1.0.

```yaml
compression:
  enabled: true
  method: int8
  target_bits: 8
  pruning_ratio: 0.3
  pruning_targets: [q_proj, v_proj]
```

---

## 9. Adaptive Compression Settings

**Section key:** `adaptive_compression`
**Class:** `AdaptiveCompressionSettings`
**Source:** `_model.py`

Adaptive compression during idle periods. When cluster utilization falls below `idle_threshold_pct` for at least `idle_duration_s` seconds, a background compression job is triggered.

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `enabled` | `bool` | `False` | Enable adaptive compression. |
| `idle_threshold_pct` | `float` | `30.0` | Idle utilization threshold percentage. |
| `idle_duration_s` | `int` | `60` | Seconds of idle time before triggering. |
| `check_interval_s` | `int` | `15` | Seconds between idle checks. |
| `compression_method` | `str` | `"int4"` | Compression method to apply. |
| `calibration_samples` | `int` | `128` | Number of calibration samples. |
| `output_dir` | `str` | `"/tmp/distllm-compress"` | Output directory for compressed models. |

```yaml
adaptive_compression:
  enabled: true
  idle_threshold_pct: 20
  idle_duration_s: 120
  compression_method: int4
```

---

## 10. Model Hub Settings

**Section key:** `model_hub`
**Class:** `ModelHubSettings`
**Source:** `_model.py`

HuggingFace model hub integration configuration.

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `enabled` | `bool` | `True` | Enable HuggingFace hub integration. |
| `cache_dir` | `str \| None` | `None` | Cache directory for downloaded models. |
| `max_cache_size_gb` | `float` | `50.0` | Maximum cache size in GB. Must be positive. |
| `offline_mode` | `bool` | `False` | Run in offline mode (no downloads). |
| `hf_token` | `SecretStr \| None` | `None` | HuggingFace token. **Must be set via env var**, not YAML. |
| `download_timeout_s` | `int` | `300` | Download timeout in seconds. Must be >= 1. |

**Validators:**
- `hf_token`: Rejects token set directly in config file. Must use `DISTLLM__MODEL_HUB__HF_TOKEN` or `HF_TOKEN` env var.
- `max_cache_size_gb`: Must be positive.
- `download_timeout_s`: Must be >= 1.

```yaml
model_hub:
  enabled: true
  max_cache_size_gb: 100
  offline_mode: false
  # hf_token: MUST be set via DISTLLM__MODEL_HUB__HF_TOKEN env var
```

---

## 11. Embedding Settings

**Section key:** `embedding`
**Class:** `EmbeddingSettings`
**Source:** `_model.py`

Embedding and reranking model configuration.

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `embedding_model` | `str` | `""` | Dedicated embedding model (e.g. sentence-transformers). |
| `rerank_model` | `str` | `""` | Cross-encoder reranking model. |
| `normalize` | `bool` | `True` | L2-normalize embeddings. |
| `max_length` | `int` | `512` | Maximum sequence length. |
| `batch_size` | `int` | `32` | Embedding batch size. |

```yaml
embedding:
  embedding_model: sentence-transformers/all-MiniLM-L6-v2
  normalize: true
  max_length: 512
```

---

## 12. Prompt Template Settings

**Section key:** `prompt_template`
**Class:** `PromptTemplateSettings`
**Source:** `_model.py`

Prompt template engine configuration.

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `template` | `str` | `"auto"` | Template name or `auto` for auto-detection. Must not be empty. |
| `custom_template_path` | `str \| None` | `None` | Path to a custom template file. |

```yaml
prompt_template:
  template: auto
  custom_template_path: /path/to/template.jinja2
```

---

## 13. Coordinator Settings

**Section key:** `coordinator`
**Class:** `CoordinatorSettings`
**Source:** `_network.py`

Coordinator server configuration — the central control plane for distributed inference.

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `host` | `str` | `"localhost"` | Coordinator hostname. |
| `port` | `int` | `50050` | gRPC port. Must be 1–65535. |
| `api_port` | `int` | `8000` | REST API port. Must be 1–65535. |
| `cors_origins` | `str` | `"http://localhost:3000,http://localhost:8080"` | Comma-separated CORS origins. |

**Validators:**
- `port`, `api_port`: Must be 1–65535.
- `cors_origins`: Must not be empty. Each origin must start with `http://`, `https://`, `chrome-extension://`, or `moz-extension://`.

```yaml
coordinator:
  host: 0.0.0.0
  port: 50050
  api_port: 8000
  cors_origins: "http://localhost:3000,https://app.example.com"
```

---

## 14. Network Settings

**Section key:** `network`
**Class:** `NetworkSettings`
**Source:** `_network.py`

Network and RPC configuration.

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `grpc_timeout` | `int` | `30` | gRPC timeout in seconds. Must be >= 1. |
| `max_retries` | `int` | `3` | Maximum retry attempts. Must be >= 1. |
| `retry_delay` | `float` | `1.0` | Delay between retries in seconds. |

**Validators:**
- `grpc_timeout`, `max_retries`: Must be >= 1.

```yaml
network:
  grpc_timeout: 60
  max_retries: 5
  retry_delay: 2.0
```

---

## 15. TLS Settings

**Section key:** `tls`
**Class:** `TLSSettings`
**Source:** `_network.py`

TLS and mutual TLS (mTLS) configuration for secure connections.

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `enabled` | `bool` | `False` | Enable TLS. |
| `cert_dir` | `str` | `"certs"` | Directory containing certificates. |
| `cert_file` | `str \| None` | `None` | Server certificate file path. |
| `key_file` | `str \| None` | `None` | Server private key file path. |
| `ca_cert_file` | `str \| None` | `None` | CA certificate file path. |
| `client_cert_file` | `str \| None` | `None` | Client certificate for mTLS. Enables mutual TLS when set with `client_key_file`. |
| `client_key_file` | `str \| None` | `None` | Client private key for mTLS. |
| `require_client_cert` | `bool` | `False` | Reject connections without a valid client certificate. Requires `ca_cert_file`. |
| `min_tls_version` | `str` | `"TLSv1.2"` | Minimum TLS version. Allowed: `TLSv1.2`, `TLSv1.3`. |

```yaml
tls:
  enabled: true
  cert_dir: /etc/ssl/distllm
  cert_file: server.crt
  key_file: server.key
  ca_cert_file: ca.crt
  require_client_cert: true
  min_tls_version: TLSv1.3
```

---

## 16. Rate Limit Settings

**Section key:** `rate_limit`
**Class:** `RateLimitSettings`
**Source:** `_network.py`

API rate limiting configuration.

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `enabled` | `bool` | `True` | Enable rate limiting. |
| `default_rpm` | `float` | `60.0` | Default requests per minute. Must be positive. |
| `endpoint_limits` | `dict[str, float]` | See below | Per-endpoint RPM limits. |
| `burst_multiplier` | `float` | `1.5` | Burst multiplier above steady-state limit. Must be positive. |
| `auth_rpm_multiplier` | `float` | `2.0` | RPM multiplier for authenticated clients. Must be positive. |

**Default endpoint limits:**
```yaml
/v1/chat/completions: 30.0
/v1/completions: 30.0
/health: 120.0
/metrics: 120.0
```

```yaml
rate_limit:
  enabled: true
  default_rpm: 120
  burst_multiplier: 2.0
  auth_rpm_multiplier: 3.0
  endpoint_limits:
    /v1/chat/completions: 60.0
    /v1/completions: 60.0
```

---

## 17. Wide Area Settings

**Section key:** `wide_area`
**Class:** `WideAreaSettings`
**Source:** `_network.py`

Wide-area network distributed inference. Enables P2P node forwarding across high-latency links.

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `enabled` | `bool` | `False` | Enable wide-area mode. |
| `p2p_forwarding` | `bool` | `True` | Enable P2P token forwarding. |
| `tokens_before_forward` | `int` | `10` | Tokens to accumulate before forwarding. Must be >= 1. |
| `wan_timeout_seconds` | `int` | `60` | WAN-specific timeout. Must be >= 1. |
| `max_retries` | `int` | `3` | Maximum WAN retries. |
| `backoff_base_seconds` | `float` | `1.0` | Exponential backoff base. |

```yaml
wide_area:
  enabled: true
  p2p_forwarding: true
  tokens_before_forward: 20
  wan_timeout_seconds: 120
```

---

## 18. Chat Router Settings

**Section key:** `chat_router`
**Class:** `ChatRouterSettings`
**Source:** `_network.py`

Multi-model chat router for routing queries to different backend models based on content matching rules.

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `enabled` | `bool` | `False` | Enable chat routing. |
| `name` | `str` | `"hybrid"` | Model name that clients use to invoke this router. |
| `default_model` | `str` | `""` | Default model when no rules match. |
| `routes` | `list[RouteRuleSettings]` | `[]` | Ordered routing rules. |

### Route Rule Settings (`RouteRuleSettings`)

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `name` | `str` | `""` | Rule name for identification. |
| `match_type` | `str` | `"keyword"` | Matching strategy: `keyword`, `regex`, or `workload`. |
| `match` | `str` | `""` | Pattern to match against the user message. |
| `target_model` | `str` | `""` | Model to route to when this rule matches. |
| `priority` | `int` | `0` | Rule priority (higher = evaluated first). Must be >= 0. |

```yaml
chat_router:
  enabled: true
  name: hybrid
  default_model: llama3
  routes:
    - name: code-route
      match_type: keyword
      match: "write a function"
      target_model: codellama
      priority: 10
    - name: creative-route
      match_type: keyword
      match: "write a story"
      target_model: llama3
      priority: 5
```

---

## 19. Prefix Cache Settings

**Section key:** `prefix_cache`
**Class:** `PrefixCacheSettings`
**Source:** `_cache.py`

Prefix cache for KV reuse across requests with common prefixes.

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `enabled` | `bool` | `True` | Enable prefix caching. |
| `max_entries` | `int` | `1024` | Maximum cache entries. Must be >= 1. |
| `min_prefix_len` | `int` | `16` | Minimum prefix length to cache. Must be >= 1. |
| `radix_tree_enabled` | `bool` | `True` | Use RadixTree (trie) instead of hash-based LRU. |

```yaml
prefix_cache:
  enabled: true
  max_entries: 2048
  min_prefix_len: 32
  radix_tree_enabled: true
```

---

## 20. Cache Persistence Settings

**Section key:** `cache_persistence`
**Class:** `CachePersistenceSettings`
**Source:** `_cache.py`

KV cache persistence to disk.

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `enabled` | `bool` | `False` | Enable disk persistence. |
| `storage_path` | `str` | `".distllm_cache"` | Storage path for persisted cache. |
| `max_disk_gb` | `float` | `50.0` | Maximum disk usage in GB. |
| `ttl_hours` | `float` | `24.0` | Cache entry TTL in hours. |

```yaml
cache_persistence:
  enabled: true
  storage_path: /data/distllm_cache
  max_disk_gb: 100
  ttl_hours: 48
```

---

## 21. Gossip Settings

**Section key:** `gossip`
**Class:** `GossipSettings`
**Source:** `_cache.py`

P2P KV cache gossip protocol for sharing cache metadata across nodes.

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `enabled` | `bool` | `False` | Enable gossip protocol. |
| `interval` | `float` | `10.0` | Gossip interval in seconds. |
| `max_peers` | `int` | `16` | Maximum peers to gossip with. |
| `cache_ttl` | `float` | `300.0` | Cache entry TTL in seconds. |

```yaml
gossip:
  enabled: true
  interval: 5.0
  max_peers: 32
  cache_ttl: 600
```

---

## 22. Predictive Cache Settings

**Section key:** `predictive_cache`
**Class:** `PredictiveCacheSettings`
**Source:** `_cache.py`

Predictive KV cache management for pre-warming cache based on usage patterns.

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `enabled` | `bool` | `False` | Enable predictive caching. |
| `gpu_cache_mb` | `int` | `512` | GPU cache budget in MB. |
| `cpu_cache_mb` | `int` | `4096` | CPU cache budget in MB. |
| `pattern_decay_hours` | `float` | `24.0` | Pattern decay time in hours. |
| `min_prefix_len` | `int` | `8` | Minimum prefix length for prediction. |
| `background_compress_interval_s` | `int` | `300` | Background compression interval in seconds. |

```yaml
predictive_cache:
  enabled: true
  gpu_cache_mb: 1024
  cpu_cache_mb: 8192
```

---

## 23. Unified Cache Settings

**Section key:** `cache`
**Class:** `CacheSettings`
**Source:** `_cache.py`

Unified cache configuration consolidating prefix cache, persistence, predictive cache, and gossip into a single section. Sub-configs remain for backward compatibility.

| Name | Type | Default | Description |
|------|------|---------|-------------|
| **Prefix cache** | | | |
| `prefix_enabled` | `bool` | `True` | Enable prefix caching. |
| `prefix_max_entries` | `int` | `1024` | Maximum prefix cache entries. |
| `prefix_min_prefix_len` | `int` | `16` | Minimum prefix length to cache. |
| `radix_tree_enabled` | `bool` | `True` | Use RadixTree trie structure. |
| **Persistence** | | | |
| `persistence_enabled` | `bool` | `False` | Enable cache persistence to disk. |
| `persistence_storage_path` | `str` | `".distllm_cache"` | Disk storage path. |
| `persistence_max_disk_gb` | `float` | `50.0` | Max disk usage in GB. |
| `persistence_ttl_hours` | `float` | `24.0` | Entry TTL in hours. |
| `background_compaction_enabled` | `bool` | `False` | Enable background compaction. |
| `background_compaction_interval_s` | `float` | `300.0` | Compaction interval in seconds. |
| **Predictive** | | | |
| `predictive_enabled` | `bool` | `False` | Enable predictive cache. |
| `predictive_gpu_cache_mb` | `int` | `512` | GPU cache budget in MB. |
| `predictive_cpu_cache_mb` | `int` | `4096` | CPU cache budget in MB. |
| `predictive_pattern_decay_hours` | `float` | `24.0` | Pattern decay in hours. |
| `predictive_min_prefix_len` | `int` | `8` | Min prefix length for prediction. |
| **Gossip** | | | |
| `gossip_enabled` | `bool` | `False` | Enable P2P gossip protocol. |
| `gossip_interval` | `float` | `10.0` | Gossip interval in seconds. |
| `gossip_max_peers` | `int` | `16` | Maximum gossip peers. |
| `gossip_cache_ttl` | `float` | `300.0` | Gossip cache TTL in seconds. |
| **Eviction** | | | |
| `eviction_strategy` | `str` | `"hybrid"` | Eviction strategy. Allowed: `lru`, `lfu`, `hybrid`. |
| `size_aware_admission` | `bool` | `True` | Size-aware cache admission. |
| `memory_adaptive_budget` | `bool` | `True` | Memory-adaptive cache budget. |

```yaml
cache:
  prefix_enabled: true
  prefix_max_entries: 4096
  persistence_enabled: true
  persistence_storage_path: /data/cache
  predictive_enabled: true
  gossip_enabled: true
  eviction_strategy: hybrid
```

---

## 24. Defragmentation Settings

**Section key:** `defragmentation`
**Class:** `DefragmentationSettings`
**Source:** `_cache.py`

GPU memory defragmentation. Compacts fragmented KV cache blocks to prevent OOM errors.

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `enabled` | `bool` | `False` | Enable defragmentation. |
| `policy` | `str` | `"balanced"` | Compaction policy. Allowed: `lazy`, `balanced`, `aggressive`. |
| `interval_seconds` | `float` | `60.0` | Seconds between defrag checks. Must be >= 5.0. |
| `max_blocks_per_pass` | `int` | `0` | Max blocks per pass (0 = unlimited). Must be >= 0. |
| `threshold` | `float` | `0.0` | Override policy threshold (0 = use policy default). Range: 0.0–1.0. |
| `tiered_compaction` | `bool` | `False` | Enable L2 (CPU swap) and L3 (NVMe) compaction. |
| `l2_cpu_swap_threshold` | `float` | `0.60` | L2 CPU swap threshold. Range: 0.0–1.0. |
| `l3_nvme_swap_threshold` | `float` | `0.80` | L3 NVMe swap threshold. Range: 0.0–1.0. |
| `cuda_stream_priority` | `int` | `-1` | CUDA stream priority for copy ops. |
| `enable_predictive` | `bool` | `False` | Predictive (preemptive) defragmentation. |
| `enable_prometheus` | `bool` | `False` | Export Prometheus defrag metrics. |

**Validators:**
- `policy`: Must be one of `lazy`, `balanced`, `aggressive`.

```yaml
defragmentation:
  enabled: true
  policy: aggressive
  interval_seconds: 30
  tiered_compaction: true
  l2_cpu_swap_threshold: 0.5
  l3_nvme_swap_threshold: 0.7
  enable_prometheus: true
```

---

## 25. Node Settings

**Section key:** `nodes` (list)
**Class:** `NodeSettings`
**Source:** `_parallelism.py`

Worker node configuration. `nodes` is a list of node definitions.

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `node_id` | `str` | *(required)* | Unique node identifier. |
| `host` | `str` | `"localhost"` | Node hostname. |
| `port` | `int` | `50051` | Node gRPC port. Must be 1–65535. |
| `start_layer` | `int` | `0` | First layer assigned to this node. |
| `end_layer` | `int` | `3` | Last layer assigned to this node. Must be >= `start_layer`. |
| `device` | `str` | `"cuda"` | Device to run on. |
| `role` | `NodeRole` | `"auto"` | Node role. Allowed: `auto`, `prefill`, `decode`. |

**Validators:**
- `port`: Must be 1–65535.
- `end_layer`: Must be >= `start_layer`.

```yaml
nodes:
  - node_id: node-0
    host: 192.168.1.10
    port: 50051
    start_layer: 0
    end_layer: 15
    device: cuda
    role: prefill
  - node_id: node-1
    host: 192.168.1.11
    port: 50051
    start_layer: 16
    end_layer: 31
    device: cuda
    role: decode
```

---

## 26. Tensor Parallel Settings

**Section key:** `tensor_parallel`
**Class:** `TensorParallelSettings`
**Source:** `_parallelism.py`

Tensor parallelism configuration for splitting model layers across GPUs.

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `enabled` | `bool` | `False` | Enable tensor parallelism. |
| `num_gpus` | `int` | `2` | Number of GPUs for tensor parallelism. Must be >= 1. |

```yaml
tensor_parallel:
  enabled: true
  num_gpus: 4
```

---

## 27. Hybrid Parallel Settings

**Section key:** `hybrid_parallel`
**Class:** `HybridParallelSettings`
**Source:** `_parallelism.py`

Hybrid parallelism combining Tensor Parallel (TP), Pipeline Parallel (PP), and Expert Parallel (EP).

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `enabled` | `bool` | `False` | Enable hybrid parallelism. |
| `auto_detect` | `bool` | `True` | Auto-detect optimal parallelism config. |
| `tp_enabled` | `bool` | `True` | Enable tensor parallelism component. |
| `pp_overlap` | `bool` | `True` | Enable pipeline parallelism overlap. |
| `ep_enabled` | `bool` | `True` | Enable expert parallelism component. |
| `force_tp_world_size` | `int` | `0` | Force TP world size (0 = auto). |
| `force_pp_stages` | `int` | `0` | Force PP stages (0 = auto). |

```yaml
hybrid_parallel:
  enabled: true
  auto_detect: true
  tp_enabled: true
  pp_overlap: true
  ep_enabled: true
```

---

## 28. Zero-Copy Settings

**Section key:** `zero_copy`
**Class:** `ZeroCopySettings`
**Source:** `_parallelism.py`

Zero-copy GPU tensor transfer configuration for high-performance inter-node communication.

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `enabled` | `bool` | `False` | Enable zero-copy transfers. |
| `prefer_rdma` | `bool` | `True` | Prefer RDMA for transfers. |
| `fallback_to_nccl` | `bool` | `True` | Fall back to NCCL if RDMA unavailable. |
| `intranode_ipc` | `bool` | `True` | Use IPC for intra-node transfers. |

```yaml
zero_copy:
  enabled: true
  prefer_rdma: true
  fallback_to_nccl: true
```

---

## 29. Partitioning Settings

**Section key:** `partitioning`
**Class:** `PartitioningSettings`
**Source:** `_parallelism.py`

Layer partitioning strategy configuration (legacy).

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `strategy` | `str` | `"gpu_aware"` | Partitioning strategy. Allowed: `equal`, `gpu_aware`. |
| `safety_margin` | `float` | `0.1` | VRAM safety margin (fraction). |

```yaml
partitioning:
  strategy: gpu_aware
  safety_margin: 0.15
```

---

## 30. Rebalancer Settings

**Section key:** `rebalancer`
**Class:** `RebalancerSettings`
**Source:** `_parallelism.py`

Dynamic pipeline rebalancing for straggler mitigation.

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `enabled` | `bool` | `False` | Enable dynamic rebalancing. |
| `check_interval` | `float` | `30.0` | Seconds between rebalance checks. |
| `straggler_threshold` | `float` | `1.5` | Straggler detection multiplier. |
| `min_improvement_pct` | `float` | `0.1` | Minimum improvement to trigger rebalance. |
| `cooldown_seconds` | `float` | `300.0` | Cooldown between rebalances. |
| `grace_period_steps` | `int` | `3` | Grace period steps after rebalance. |
| `auto_mitigate` | `bool` | `False` | Auto-mitigate detected stragglers. |

```yaml
rebalancer:
  enabled: true
  check_interval: 60
  straggler_threshold: 2.0
  auto_mitigate: true
```

---

## 31. Batching Settings

**Section key:** `batching`
**Class:** `BatchingSettings`
**Source:** `_parallelism.py`

Continuous batching configuration.

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `max_batch_size` | `int` | `32` | Maximum batch size. Must be >= 1. |
| `max_tokens_per_batch` | `int` | `4096` | Maximum tokens per batch. Must be >= 1. |

```yaml
batching:
  max_batch_size: 64
  max_tokens_per_batch: 8192
```

---

## 32. Chunked Prefill Settings

**Section key:** `chunked_prefill`
**Class:** `ChunkedPrefillSettings`
**Source:** `_parallelism.py`

Chunked prefill for processing long prompts in chunks.

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `enabled` | `bool` | `True` | Enable chunked prefill. |
| `chunk_size` | `int` | `512` | Chunk size in tokens. Must be >= 1. |

```yaml
chunked_prefill:
  enabled: true
  chunk_size: 1024
```

---

## 33. Priority Settings

**Section key:** `priority`
**Class:** `PrioritySettings`
**Source:** `_parallelism.py`

Request priority queuing configuration.

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `enabled` | `bool` | `False` | Enable priority queuing. |
| `num_levels` | `int` | `4` | Number of priority levels. |
| `preemption_enabled` | `bool` | `False` | Enable request preemption. |
| `max_preempted` | `int` | `10` | Maximum preempted requests. |

```yaml
priority:
  enabled: true
  num_levels: 4
  preemption_enabled: true
  max_preempted: 20
```

---

## 34. Disaggregation Settings

**Section key:** `disagg`
**Class:** `DisaggSettings`
**Source:** `_parallelism.py`

Disaggregated prefill/decode serving — separate nodes for prefill and decode phases.

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `enabled` | `bool` | `False` | Enable disaggregated serving. |
| `prefill_nodes` | `list[dict]` | `[]` | List of prefill node configurations. |
| `decode_nodes` | `list[dict]` | `[]` | List of decode node configurations. |

```yaml
disagg:
  enabled: true
  prefill_nodes:
    - host: 192.168.1.10
      port: 50051
  decode_nodes:
    - host: 192.168.1.20
      port: 50052
```

---

## 35. CUDA Graph Settings

**Section key:** `cuda_graph`
**Class:** `CudaGraphSettings`
**Source:** `_performance.py`

CUDA graph capture for decode acceleration.

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `enabled` | `bool` | `False` | Enable CUDA graph capture. |
| `batch_sizes` | `list[int]` | `[1, 2, 4, 8, 16, 32]` | Batch sizes to capture graphs for. |

```yaml
cuda_graph:
  enabled: true
  batch_sizes: [1, 2, 4, 8, 16, 32, 64]
```

---

## 36. Compile Settings

**Section key:** `compile`
**Class:** `CompileSettings`
**Source:** `_performance.py`

`torch.compile` integration for kernel fusion.

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `enabled` | `bool` | `False` | Enable torch.compile. |
| `mode` | `str` | `"reduce-overhead"` | Compile mode. |
| `fullgraph` | `bool` | `False` | Require full graph compilation (no graph breaks). |

```yaml
compile:
  enabled: true
  mode: reduce-overhead
  fullgraph: false
```

---

## 37. Adaptive Precision Settings

**Section key:** `adaptive_precision`
**Class:** `AdaptivePrecisionSettings`
**Source:** `_performance.py`

Adaptive precision pipeline — automatically selects precision based on quality constraints.

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `enabled` | `bool` | `False` | Enable adaptive precision. |
| `calibration_samples` | `int` | `64` | Number of calibration samples. |
| `target_precision` | `str` | `"auto"` | Target precision. Allowed: `auto`, `fp16`, `int8`. |
| `max_quality_loss_pct` | `float` | `0.1` | Maximum allowed quality loss percentage. |

```yaml
adaptive_precision:
  enabled: true
  target_precision: auto
  max_quality_loss_pct: 0.5
```

---

## 38. Self-Optimizing Settings

**Section key:** `self_optimizing`
**Class:** `SelfOptimizingSettings`
**Source:** `_performance.py`

Auto-tuning via hill-climbing optimization (legacy). Prefer the `optimization` dict for new setups.

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `enabled` | `bool` | `False` | Enable self-optimizing. |
| `tune_interval_seconds` | `float` | `60.0` | Seconds between tuning iterations. |
| `warmup_seconds` | `float` | `30.0` | Warmup period before tuning. |
| `profile_dir` | `str \| None` | `None` | Directory for profiling output. |

```yaml
self_optimizing:
  enabled: true
  tune_interval_seconds: 120
  warmup_seconds: 60
```

---

## 39. Hardware Settings

**Section key:** `hardware`
**Class:** `HardwareSettings`
**Source:** `_hardware.py`

Multi-architecture hardware configuration for heterogeneous clusters.

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `device_type` | `str` | `"auto"` | Device type. Allowed: `auto`, `cuda`, `rocm`, `mps`, `xpu`, `cpu`. |
| `preferred_backend` | `str` | `"auto"` | Preferred backend. Allowed: `auto`, `vllm`, `pytorch`, `llamacpp`. |
| `force_device_id` | `int` | `-1` | Force specific device ID (-1 = auto-select). |
| `fallback_to_cpu` | `bool` | `True` | Fall back to CPU if GPU unavailable. |
| `rocm_visible_devices` | `str` | `""` | ROCm visible devices override. |
| `mps_optimize_memory` | `bool` | `True` | Optimize memory on Apple MPS. |
| `xpu_oneapi_verbose` | `bool` | `False` | Verbose Intel oneAPI logging. |
| `cpu_threads` | `int` | `0` | CPU thread count (0 = auto-detect via psutil). |
| `cpu_numa_aware` | `bool` | `True` | Enable NUMA-aware CPU scheduling. |

**Validators:**
- `device_type`: Must be one of `auto`, `cuda`, `rocm`, `mps`, `xpu`, `cpu`.
- `preferred_backend`: Must be one of `auto`, `vllm`, `pytorch`, `llamacpp`.

```yaml
hardware:
  device_type: cuda
  preferred_backend: vllm
  fallback_to_cpu: false
  cpu_numa_aware: true
```

---

## 40. vLLM Backend Settings

**Section key:** `vllm`
**Class:** `VLLMSettings`
**Source:** `_backends.py`

vLLM backend configuration for high-performance GPU inference.

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `enabled` | `bool` | `False` | Enable vLLM backend. |
| `tensor_parallel_size` | `int` | `1` | Tensor parallelism degree. Must be >= 1. |
| `gpu_memory_utilization` | `float` | `0.9` | GPU memory utilization fraction. Range: (0, 1]. |
| `max_num_seqs` | `int` | `256` | Maximum concurrent sequences. |
| `max_num_batched_tokens` | `int` | `8192` | Maximum batched tokens. |
| `dtype` | `str` | `"auto"` | Data type. Allowed: `auto`, `float16`, `float32`, `bfloat16`, `half`, `full`. |
| `seed` | `int` | `0` | Random seed. |
| `enforce_eager` | `bool` | `False` | Enforce eager mode (disable CUDA graphs). |
| `max_model_len` | `int \| None` | `None` | Maximum model sequence length. |

**Validators:**
- `gpu_memory_utilization`: Must be in (0, 1].
- `tensor_parallel_size`: Must be >= 1.
- `dtype`: Must be one of `auto`, `float16`, `float32`, `bfloat16`, `half`, `full`.

```yaml
vllm:
  enabled: true
  tensor_parallel_size: 2
  gpu_memory_utilization: 0.9
  max_num_seqs: 512
  dtype: auto
```

---

## 41. llama.cpp Backend Settings

**Section key:** `llamacpp`
**Class:** `LlamacppSettings`
**Source:** `_backends.py`

llama.cpp backend configuration — lightweight alternative for CPU/GPU inference with GGUF models.

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `enabled` | `bool` | `False` | Enable llama.cpp backend. |
| `model_path` | `str` | `""` | Path to GGUF model file. **Required when enabled.** |
| `n_gpu_layers` | `int` | `0` | Number of layers to offload to GPU. Must be >= 0. |
| `n_ctx` | `int` | `2048` | Context window size. Must be >= 128. |
| `n_threads` | `int \| None` | `None` | Number of CPU threads (None = auto). |
| `n_batch` | `int` | `512` | Batch size for prompt processing. |
| `seed` | `int` | `0` | Random seed. |
| `verbose` | `bool` | `False` | Verbose logging. |

**Validators:**
- `model_path`: Required when `enabled` is `True`.
- `n_gpu_layers`: Must be >= 0.
- `n_ctx`: Must be >= 128.

```yaml
llamacpp:
  enabled: true
  model_path: /models/llama-7b-q4_k_m.gguf
  n_gpu_layers: 35
  n_ctx: 4096
  n_threads: 8
```

---

## 42. Generation Settings

**Section key:** `generation`
**Class:** `GenerationSettings`
**Source:** `_generation.py`

Default text generation parameters.

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `max_new_tokens` | `int` | `256` | Maximum new tokens to generate. |
| `temperature` | `float` | `0.7` | Sampling temperature. Range: 0.0–2.0. |
| `top_p` | `float` | `0.9` | Nucleus sampling probability. Range: 0.0–1.0. |
| `top_k` | `int` | `0` | Top-k sampling (0 = disabled). Must be >= 0. |

**Validators:**
- `temperature`: Must be 0.0–2.0.
- `top_p`: Must be 0.0–1.0 (exclusive of 0).
- `top_k`: Must be >= 0.

```yaml
generation:
  max_new_tokens: 512
  temperature: 0.8
  top_p: 0.95
  top_k: 50
```

---

## 43. Monitoring Settings

**Section key:** `monitoring`
**Class:** `MonitoringSettings`
**Source:** `_observability.py`

System monitoring configuration.

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `enabled` | `bool` | `True` | Enable system monitoring. |

```yaml
monitoring:
  enabled: true
```

---

## 44. Alerting Settings

**Section key:** `alerting`
**Class:** `AlertingSettings`
**Source:** `_observability.py`

Prometheus alerting rules configuration.

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `enabled` | `bool` | `False` | Enable alerting. |
| `prometheus_url` | `str` | `"http://localhost:9090"` | Prometheus server URL. Must start with `http://` or `https://`. |
| `rule_file` | `str \| None` | `None` | Path to alerting rules file. |

```yaml
alerting:
  enabled: true
  prometheus_url: http://prometheus:9090
  rule_file: /etc/distllm/alerts.yml
```

---

## 45. Chaos Settings

**Section key:** `chaos`
**Class:** `ChaosSettings`
**Source:** `_observability.py`

Chaos engineering fault injection configuration.

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `enabled` | `bool` | `False` | Enable chaos engineering. |
| `allowed_scenarios` | `list[str]` | `["kill_node", "add_latency", "drop_message", "corrupt_data"]` | Allowed fault injection scenarios. |
| `max_latency_ms` | `int` | `5000` | Maximum injected latency in ms. Must be >= 1. |

```yaml
chaos:
  enabled: true
  allowed_scenarios:
    - kill_node
    - add_latency
  max_latency_ms: 2000
```

---

## 46. Canary Settings

**Section key:** `canary`
**Class:** `CanarySettings`
**Source:** `_deployment.py`

Automated canary deployment configuration.

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `enabled` | `bool` | `False` | Enable canary deployments. |
| `stable_version` | `str` | `"stable"` | Name of the stable version. |
| `canary_version` | `str` | `"canary"` | Name of the canary version. |
| `rollback_threshold` | `float` | `0.05` | Error rate threshold to trigger rollback. Range: 0.0–1.0. |
| `stages` | `list[RolloutStageModel]` | 5 stages: 5%, 25%, 50%, 75%, 100% | Canary rollout stages. |

### Rollout Stage Model (`RolloutStageModel`)

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `weight_pct` | `float` | *(required)* | Traffic weight percentage for this stage. Range: 0–100. |
| `analysis_duration_s` | `int` | `300` | Duration to analyze this stage in seconds. |

**Validators:**
- `rollback_threshold`: Must be 0.0–1.0.
- `stages`: Must not be empty. Each `weight_pct` must be 0–100.

```yaml
canary:
  enabled: true
  stable_version: v1.0
  canary_version: v1.1
  rollback_threshold: 0.02
  stages:
    - weight_pct: 10
      analysis_duration_s: 600
    - weight_pct: 50
      analysis_duration_s: 600
    - weight_pct: 100
      analysis_duration_s: 300
```

---

## 47. Version Settings

**Section key:** `version`
**Class:** `VersionSettings`
**Source:** `_deployment.py`

Model versioning and A/B testing configuration.

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `enabled` | `bool` | `False` | Enable versioning. |
| `max_versions` | `int` | `4` | Maximum model versions to retain. |
| `shadow_enabled` | `bool` | `False` | Enable shadow traffic. |
| `shadow_pct` | `float` | `0.0` | Percentage of traffic to shadow (0–100). |
| `blue_green_enabled` | `bool` | `False` | Enable blue-green deployments. |
| `ab_testing_enabled` | `bool` | `False` | Enable A/B testing. |
| `ab_test_split` | `float` | `50.0` | Percentage for variant B (0–100). |
| `auto_promote_enabled` | `bool` | `False` | Auto-promote winning version. |
| `min_samples` | `int` | `100` | Minimum samples before statistical test. |
| `significance_level` | `float` | `0.05` | p-value threshold for A/B test. |

```yaml
version:
  enabled: true
  max_versions: 6
  ab_testing_enabled: true
  ab_test_split: 50
  auto_promote_enabled: true
  min_samples: 1000
  significance_level: 0.01
```

---

## 48. Cost Settings

**Section key:** `cost`
**Class:** `CostSettings`
**Source:** `_deployment.py`

Cost-aware scheduling configuration.

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `enabled` | `bool` | `False` | Enable cost-aware scheduling. |
| `budget_per_hour` | `float` | `0.0` | Hourly budget in USD. Must be >= 0. |
| `spot_preference` | `float` | `0.8` | Preference for spot instances (0.0–1.0). |

**Validators:**
- `budget_per_hour`: Must be >= 0.
- `spot_preference`: Must be 0.0–1.0.

```yaml
cost:
  enabled: true
  budget_per_hour: 10.0
  spot_preference: 0.9
```

---

## 49. Tenant Settings

**Section key:** `tenant`
**Class:** `TenantSettings`
**Source:** `_deployment.py`

Multi-tenant SaaS configuration.

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `enabled` | `bool` | `False` | Enable multi-tenancy. |
| `default_tier` | `str` | `"free"` | Default tenant tier. |
| `admin_api_key` | `SecretStr \| None` | `None` | Admin API key. **Must be set via env var.** |

```yaml
tenant:
  enabled: true
  default_tier: free
  # admin_api_key: MUST be set via DISTLLM__TENANT__ADMIN_API_KEY env var
```

---

## 50. RAG Settings

**Section key:** `rag`
**Class:** `RAGSettings`
**Source:** `_application.py`

RAG (Retrieval-Augmented Generation) pipeline with FAISS.

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `enabled` | `bool` | `False` | Enable RAG pipeline. |
| `dimension` | `int` | `768` | Embedding dimension. |
| `chunk_size` | `int` | `512` | Document chunk size in tokens. |
| `chunk_overlap` | `int` | `50` | Chunk overlap in tokens. |
| `index_path` | `str \| None` | `None` | Path to FAISS index. |

```yaml
rag:
  enabled: true
  dimension: 1024
  chunk_size: 512
  chunk_overlap: 100
  index_path: /data/faiss.index
```

---

## 51. Agent Settings

**Section key:** `agent`
**Class:** `AgentSettings`
**Source:** `_application.py`

ReAct agent loop configuration.

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `enabled` | `bool` | `False` | Enable ReAct agent loop. |
| `max_iterations` | `int` | `10` | Maximum agent loop iterations. |
| `reflection_enabled` | `bool` | `True` | Enable self-reflection step. |

```yaml
agent:
  enabled: true
  max_iterations: 15
  reflection_enabled: true
```

---

## 52. Plugin Settings

**Section key:** `plugins`
**Class:** `PluginSettings`
**Source:** `_application.py`

Plugin system configuration.

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `enabled` | `bool` | `True` | Enable plugin system. |
| `plugins` | `list[dict[str, Any]]` | `[]` | List of plugin definitions. Each must have a `module` key with a fully-qualified module name. |

**Validators:**
- `plugins`: Each plugin dict with a `module` key must use a fully-qualified module name (must contain a `.`).

```yaml
plugins:
  enabled: true
  plugins:
    - module: my_package.my_plugin
      config:
        key: value
    - module: another_package.plugin
```

---

## Top-Level Flat Fields

These fields exist at the top level of `DistLLMSettings` as raw dicts (not typed settings classes):

| Section Key | Type | Default | Description |
|-------------|------|---------|-------------|
| `auto_partition` | `dict` | `{"enabled": False, "strategy": "auto", "safety_margin": 0.1}` | Auto-partitioning config (legacy). |
| `predictive_migration` | `dict` | `{"enabled": False}` | Predictive migration config. |
| `structured_output` | `dict` | `{"enabled": False}` | Structured output (JSON mode) config. |
| `optimization` | `dict` | `{"enabled": False}` | Bayesian optimization config (replaces self_optimizing). |

---

## Complete YAML Example

```yaml
# ==============================================================================
# DistLLM Complete Configuration Example
# ==============================================================================

# --- Model ---
model:
  name: meta-llama/Llama-3-8B-Instruct
  dtype: bfloat16
  trust_remote_code: false

# --- Quantization ---
quantization:
  method: bnb_4bit
  bnb_4bit_compute_dtype: bfloat16
  bnb_4bit_quant_type: nf4
  bnb_4bit_use_double_quant: true
  kv_cache_quant: true
  kv_cache_bits: 8

# --- Speculative Decoding ---
speculative:
  method: auto
  num_assistant_tokens: 5
  min_acceptance_rate: 0.3

# --- LoRA ---
lora:
  enabled: false

# --- MoE ---
moe:
  enabled: false

# --- Coordinator ---
coordinator:
  host: 0.0.0.0
  port: 50050
  api_port: 8000
  cors_origins: "http://localhost:3000,https://app.example.com"

# --- Network ---
network:
  grpc_timeout: 60
  max_retries: 5
  retry_delay: 2.0

# --- TLS ---
tls:
  enabled: false

# --- Rate Limiting ---
rate_limit:
  enabled: true
  default_rpm: 120
  burst_multiplier: 2.0

# --- Wide Area ---
wide_area:
  enabled: false

# --- Chat Router ---
chat_router:
  enabled: false

# --- Workers ---
nodes:
  - node_id: node-0
    host: 192.168.1.10
    port: 50051
    start_layer: 0
    end_layer: 15
    device: cuda
    role: prefill
  - node_id: node-1
    host: 192.168.1.11
    port: 50051
    start_layer: 16
    end_layer: 31
    device: cuda
    role: decode

# --- Parallelism ---
tensor_parallel:
  enabled: true
  num_gpus: 4

hybrid_parallel:
  enabled: true
  auto_detect: true
  tp_enabled: true
  pp_overlap: true
  ep_enabled: true

zero_copy:
  enabled: true
  prefer_rdma: true

partitioning:
  strategy: gpu_aware
  safety_margin: 0.1

rebalancer:
  enabled: true
  check_interval: 60
  straggler_threshold: 2.0
  auto_mitigate: true

# --- Batching ---
batching:
  max_batch_size: 64
  max_tokens_per_batch: 8192

chunked_prefill:
  enabled: true
  chunk_size: 1024

priority:
  enabled: true
  num_levels: 4
  preemption_enabled: true

# --- Disaggregation ---
disagg:
  enabled: false

# --- Cache ---
prefix_cache:
  enabled: true
  max_entries: 4096
  min_prefix_len: 32
  radix_tree_enabled: true

cache_persistence:
  enabled: true
  storage_path: /data/distllm_cache
  max_disk_gb: 100

gossip:
  enabled: true
  interval: 5.0
  max_peers: 32

predictive_cache:
  enabled: true
  gpu_cache_mb: 1024
  cpu_cache_mb: 8192

cache:
  prefix_enabled: true
  prefix_max_entries: 4096
  persistence_enabled: true
  predictive_enabled: true
  gossip_enabled: true
  eviction_strategy: hybrid
  size_aware_admission: true
  memory_adaptive_budget: true

defragmentation:
  enabled: true
  policy: balanced
  interval_seconds: 60
  tiered_compaction: true
  enable_prometheus: true

# --- Hardware ---
hardware:
  device_type: auto
  preferred_backend: auto
  fallback_to_cpu: true
  cpu_numa_aware: true

# --- Backends ---
vllm:
  enabled: true
  tensor_parallel_size: 2
  gpu_memory_utilization: 0.9
  max_num_seqs: 512
  dtype: auto

llamacpp:
  enabled: false

# --- Performance ---
cuda_graph:
  enabled: true
  batch_sizes: [1, 2, 4, 8, 16, 32]

compile:
  enabled: true
  mode: reduce-overhead

adaptive_precision:
  enabled: false

self_optimizing:
  enabled: false

# --- Generation Defaults ---
generation:
  max_new_tokens: 512
  temperature: 0.7
  top_p: 0.9
  top_k: 0

# --- Model Hub ---
model_hub:
  enabled: true
  max_cache_size_gb: 100
  offline_mode: false
  # hf_token: via DISTLLM__MODEL_HUB__HF_TOKEN env var

# --- Prompt Template ---
prompt_template:
  template: auto

# --- Embedding ---
embedding:
  embedding_model: sentence-transformers/all-MiniLM-L6-v2
  normalize: true
  max_length: 512

# --- Observability ---
monitoring:
  enabled: true

alerting:
  enabled: true
  prometheus_url: http://prometheus:9090

chaos:
  enabled: false

# --- Deployment ---
canary:
  enabled: false

version:
  enabled: false

cost:
  enabled: true
  budget_per_hour: 10.0
  spot_preference: 0.8

tenant:
  enabled: false

# --- Application ---
rag:
  enabled: false

agent:
  enabled: false

plugins:
  enabled: true
  plugins: []

# --- Compression ---
compression:
  enabled: false

adaptive_compression:
  enabled: false

# --- Multi-Model ---
multi_model:
  models: {}
  default_model: ""
  max_models: 4

# --- Slora ---
slora:
  enabled: false
```
