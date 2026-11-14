# DistLLM Configuration Reference

Auto-generated from Pydantic models. All fields can be set via:
- `config.yaml` file
- Environment variables (prefix: `DISTLLM_`, delimiter: `__`)
- CLI arguments

---

## `model`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model.name` | `str` | `''` | Model name or path. Must be explicitly set. Env: `DISTLLM_MODEL__NAME` |
| `model.dtype` | `str` | `'float16'` | Env: `DISTLLM_MODEL__DTYPE` |
| `model.trust_remote_code` | `bool` | `False` | Env: `DISTLLM_MODEL__TRUST_REMOTE_CODE` |
## `coordinator`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `coordinator.host` | `str` | `'localhost'` | Env: `DISTLLM_COORDINATOR__HOST` |
| `coordinator.port` | `int` | `50050` | Env: `DISTLLM_COORDINATOR__PORT` |
| `coordinator.api_port` | `int` | `8000` | Env: `DISTLLM_COORDINATOR__API_PORT` |
| `coordinator.cors_origins` | `str` | `'http://localhost:3000,http://localhost:8080'` | Env: `DISTLLM_COORDINATOR__CORS_ORIGINS` |
| `nodes` | `list` | `PydanticUndefined` | Env: `DISTLLM_NODES` |
## `generation`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `generation.max_new_tokens` | `int` | `256` | Env: `DISTLLM_GENERATION__MAX_NEW_TOKENS` |
| `generation.temperature` | `float` | `0.7` | Env: `DISTLLM_GENERATION__TEMPERATURE` |
| `generation.top_p` | `float` | `0.9` | Env: `DISTLLM_GENERATION__TOP_P` |
| `generation.top_k` | `int` | `0` | Env: `DISTLLM_GENERATION__TOP_K` |
## `network`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `network.grpc_timeout` | `int` | `30` | Env: `DISTLLM_NETWORK__GRPC_TIMEOUT` |
| `network.max_retries` | `int` | `3` | Env: `DISTLLM_NETWORK__MAX_RETRIES` |
| `network.retry_delay` | `float` | `1.0` | Env: `DISTLLM_NETWORK__RETRY_DELAY` |
## `tls`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `tls.enabled` | `bool` | `False` | Env: `DISTLLM_TLS__ENABLED` |
| `tls.cert_dir` | `str` | `'certs'` | Env: `DISTLLM_TLS__CERT_DIR` |
| `tls.cert_file` | `Union` |  | Env: `DISTLLM_TLS__CERT_FILE` |
| `tls.key_file` | `Union` |  | Env: `DISTLLM_TLS__KEY_FILE` |
| `tls.ca_cert_file` | `Union` |  | Env: `DISTLLM_TLS__CA_CERT_FILE` |
| `tls.client_cert_file` | `Union` |  | Client certificate for mTLS. When set together with client_key_file, enables mutual TLS where clients must present a valid certificate signed by the CA. Env: `DISTLLM_TLS__CLIENT_CERT_FILE` |
| `tls.client_key_file` | `Union` |  | Client private key for mTLS. Env: `DISTLLM_TLS__CLIENT_KEY_FILE` |
| `tls.require_client_cert` | `bool` | `False` | If True, reject connections that don't present a valid client certificate. Requires ca_cert_file to be set. Env: `DISTLLM_TLS__REQUIRE_CLIENT_CERT` |
## `batching`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `batching.max_batch_size` | `int` | `32` | Env: `DISTLLM_BATCHING__MAX_BATCH_SIZE` |
| `batching.max_tokens_per_batch` | `int` | `4096` | Env: `DISTLLM_BATCHING__MAX_TOKENS_PER_BATCH` |
## `prefix_cache`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `prefix_cache.enabled` | `bool` | `True` | Env: `DISTLLM_PREFIX_CACHE__ENABLED` |
| `prefix_cache.max_entries` | `int` | `1024` | Env: `DISTLLM_PREFIX_CACHE__MAX_ENTRIES` |
| `prefix_cache.min_prefix_len` | `int` | `16` | Env: `DISTLLM_PREFIX_CACHE__MIN_PREFIX_LEN` |
| `prefix_cache.radix_tree_enabled` | `bool` | `True` | Env: `DISTLLM_PREFIX_CACHE__RADIX_TREE_ENABLED` |
## `chunked_prefill`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `chunked_prefill.enabled` | `bool` | `True` | Env: `DISTLLM_CHUNKED_PREFILL__ENABLED` |
| `chunked_prefill.chunk_size` | `int` | `512` | Env: `DISTLLM_CHUNKED_PREFILL__CHUNK_SIZE` |
## `monitoring`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `monitoring.enabled` | `bool` | `True` | Env: `DISTLLM_MONITORING__ENABLED` |
## `quantization`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `quantization.method` | `str` | `'none'` | Env: `DISTLLM_QUANTIZATION__METHOD` |
| `quantization.bnb_4bit_compute_dtype` | `str` | `'float16'` | Env: `DISTLLM_QUANTIZATION__BNB_4BIT_COMPUTE_DTYPE` |
| `quantization.bnb_4bit_quant_type` | `str` | `'nf4'` | Env: `DISTLLM_QUANTIZATION__BNB_4BIT_QUANT_TYPE` |
| `quantization.bnb_4bit_use_double_quant` | `bool` | `True` | Env: `DISTLLM_QUANTIZATION__BNB_4BIT_USE_DOUBLE_QUANT` |
| `quantization.llm_int8_threshold` | `float` | `6.0` | Env: `DISTLLM_QUANTIZATION__LLM_INT8_THRESHOLD` |
| `quantization.gptq_bits` | `int` | `4` | Env: `DISTLLM_QUANTIZATION__GPTQ_BITS` |
| `quantization.gptq_group_size` | `int` | `128` | Env: `DISTLLM_QUANTIZATION__GPTQ_GROUP_SIZE` |
| `quantization.gptq_desc_act` | `bool` | `False` | Env: `DISTLLM_QUANTIZATION__GPTQ_DESC_ACT` |
| `quantization.gptq_use_marlin` | `bool` | `True` | Env: `DISTLLM_QUANTIZATION__GPTQ_USE_MARLIN` |
| `quantization.awq_bits` | `int` | `4` | Env: `DISTLLM_QUANTIZATION__AWQ_BITS` |
| `quantization.awq_group_size` | `int` | `128` | Env: `DISTLLM_QUANTIZATION__AWQ_GROUP_SIZE` |
| `quantization.fp8_scheme` | `str` | `'e4m3'` | Env: `DISTLLM_QUANTIZATION__FP8_SCHEME` |
| `quantization.fp8_dynamic` | `bool` | `True` | Env: `DISTLLM_QUANTIZATION__FP8_DYNAMIC` |
| `quantization.kv_cache_quant` | `bool` | `False` | Env: `DISTLLM_QUANTIZATION__KV_CACHE_QUANT` |
| `quantization.kv_cache_bits` | `int` | `8` | Env: `DISTLLM_QUANTIZATION__KV_CACHE_BITS` |
## `speculative`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `speculative.draft_model` | `str` | `''` | Env: `DISTLLM_SPECULATIVE__DRAFT_MODEL` |
| `speculative.num_assistant_tokens` | `int` | `5` | Env: `DISTLLM_SPECULATIVE__NUM_ASSISTANT_TOKENS` |
| `speculative.min_acceptance_rate` | `float` | `0.3` | Env: `DISTLLM_SPECULATIVE__MIN_ACCEPTANCE_RATE` |
| `speculative.warmup_steps` | `int` | `10` | Env: `DISTLLM_SPECULATIVE__WARMUP_STEPS` |
| `speculative.method` | `str` | `'draft_model'` | Env: `DISTLLM_SPECULATIVE__METHOD` |
| `speculative.medusa_num_heads` | `int` | `4` | Env: `DISTLLM_SPECULATIVE__MEDUSA_NUM_HEADS` |
| `speculative.medusa_num_tokens_per_head` | `int` | `3` | Env: `DISTLLM_SPECULATIVE__MEDUSA_NUM_TOKENS_PER_HEAD` |
| `speculative.eagle_checkpoint` | `str` | `''` | Env: `DISTLLM_SPECULATIVE__EAGLE_CHECKPOINT` |
| `speculative.eagle_variant` | `str` | `'eagle'` | Env: `DISTLLM_SPECULATIVE__EAGLE_VARIANT` |
| `speculative.eagle_hidden_size` | `int` | `4096` | Env: `DISTLLM_SPECULATIVE__EAGLE_HIDDEN_SIZE` |
| `speculative.eagle_vocab_size` | `int` | `32000` | Env: `DISTLLM_SPECULATIVE__EAGLE_VOCAB_SIZE` |
| `speculative.eagle_num_layers` | `int` | `2` | Env: `DISTLLM_SPECULATIVE__EAGLE_NUM_LAYERS` |
| `speculative.ngram_min_match` | `int` | `4` | Env: `DISTLLM_SPECULATIVE__NGRAM_MIN_MATCH` |
## `partitioning`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `partitioning.strategy` | `str` | `'gpu_aware'` | Env: `DISTLLM_PARTITIONING__STRATEGY` |
| `partitioning.safety_margin` | `float` | `0.1` | Env: `DISTLLM_PARTITIONING__SAFETY_MARGIN` |
| `partitioning.shard_across_nodes` | `bool` | `False` | Env: `DISTLLM_PARTITIONING__SHARD_ACROSS_NODES` |
| `auto_partition` | `dict` | `PydanticUndefined` | Env: `DISTLLM_AUTO_PARTITION` |
## `rebalancer`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `rebalancer.enabled` | `bool` | `False` | Env: `DISTLLM_REBALANCER__ENABLED` |
| `rebalancer.check_interval` | `float` | `30.0` | Env: `DISTLLM_REBALANCER__CHECK_INTERVAL` |
| `rebalancer.straggler_threshold` | `float` | `1.5` | Env: `DISTLLM_REBALANCER__STRAGGLER_THRESHOLD` |
| `rebalancer.min_improvement_pct` | `float` | `0.1` | Env: `DISTLLM_REBALANCER__MIN_IMPROVEMENT_PCT` |
| `rebalancer.cooldown_seconds` | `float` | `300.0` | Env: `DISTLLM_REBALANCER__COOLDOWN_SECONDS` |
| `rebalancer.grace_period_steps` | `int` | `3` | Env: `DISTLLM_REBALANCER__GRACE_PERIOD_STEPS` |
| `rebalancer.auto_mitigate` | `bool` | `False` | Env: `DISTLLM_REBALANCER__AUTO_MITIGATE` |
## `cache_persistence`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `cache_persistence.enabled` | `bool` | `False` | Env: `DISTLLM_CACHE_PERSISTENCE__ENABLED` |
| `cache_persistence.storage_path` | `str` | `'.distllm_cache'` | Env: `DISTLLM_CACHE_PERSISTENCE__STORAGE_PATH` |
| `cache_persistence.max_disk_gb` | `float` | `50.0` | Env: `DISTLLM_CACHE_PERSISTENCE__MAX_DISK_GB` |
| `cache_persistence.ttl_hours` | `float` | `24.0` | Env: `DISTLLM_CACHE_PERSISTENCE__TTL_HOURS` |
## `priority`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `priority.enabled` | `bool` | `False` | Env: `DISTLLM_PRIORITY__ENABLED` |
| `priority.num_levels` | `int` | `4` | Env: `DISTLLM_PRIORITY__NUM_LEVELS` |
| `priority.preemption_enabled` | `bool` | `False` | Env: `DISTLLM_PRIORITY__PREEMPTION_ENABLED` |
| `priority.max_preempted` | `int` | `10` | Env: `DISTLLM_PRIORITY__MAX_PREEMPTED` |
## `multi_model`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `multi_model.models` | `dict` | `PydanticUndefined` | Env: `DISTLLM_MULTI_MODEL__MODELS` |
| `multi_model.default_model` | `str` | `''` | Env: `DISTLLM_MULTI_MODEL__DEFAULT_MODEL` |
| `multi_model.max_models` | `int` | `4` | Maximum number of models to load concurrently. Actual capacity depends on GPU memory. Env: `DISTLLM_MULTI_MODEL__MAX_MODELS` |
## `tensor_parallel`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `tensor_parallel.enabled` | `bool` | `False` | Env: `DISTLLM_TENSOR_PARALLEL__ENABLED` |
| `tensor_parallel.num_gpus` | `int` | `2` | Env: `DISTLLM_TENSOR_PARALLEL__NUM_GPUS` |
## `lora`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `lora.enabled` | `bool` | `False` | Env: `DISTLLM_LORA__ENABLED` |
| `lora.adapters` | `dict` | `PydanticUndefined` | Env: `DISTLLM_LORA__ADAPTERS` |
## `moe`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `moe.enabled` | `bool` | `False` | Env: `DISTLLM_MOE__ENABLED` |
| `moe.num_experts` | `int` | `8` | Env: `DISTLLM_MOE__NUM_EXPERTS` |
| `moe.num_experts_per_tok` | `int` | `2` | Env: `DISTLLM_MOE__NUM_EXPERTS_PER_TOK` |
## `gossip`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `gossip.enabled` | `bool` | `False` | Env: `DISTLLM_GOSSIP__ENABLED` |
| `gossip.interval` | `float` | `10.0` | Env: `DISTLLM_GOSSIP__INTERVAL` |
| `gossip.max_peers` | `int` | `16` | Env: `DISTLLM_GOSSIP__MAX_PEERS` |
| `gossip.cache_ttl` | `float` | `300.0` | Env: `DISTLLM_GOSSIP__CACHE_TTL` |
## `compression`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `compression.enabled` | `bool` | `False` | Env: `DISTLLM_COMPRESSION__ENABLED` |
| `compression.method` | `str` | `'none'` | Env: `DISTLLM_COMPRESSION__METHOD` |
| `compression.target_bits` | `int` | `8` | Env: `DISTLLM_COMPRESSION__TARGET_BITS` |
| `compression.pruning_ratio` | `float` | `0.0` | Env: `DISTLLM_COMPRESSION__PRUNING_RATIO` |
| `compression.distillation_teacher` | `Union` |  | Env: `DISTLLM_COMPRESSION__DISTILLATION_TEACHER` |
| `compression.calibration_samples` | `int` | `128` | Env: `DISTLLM_COMPRESSION__CALIBRATION_SAMPLES` |
| `compression.pruning_targets` | `list` | `['q_proj', 'v_proj']` | Env: `DISTLLM_COMPRESSION__PRUNING_TARGETS` |
## `adaptive_compression`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `adaptive_compression.enabled` | `bool` | `False` | Env: `DISTLLM_ADAPTIVE_COMPRESSION__ENABLED` |
| `adaptive_compression.idle_threshold_pct` | `float` | `30.0` | Env: `DISTLLM_ADAPTIVE_COMPRESSION__IDLE_THRESHOLD_PCT` |
| `adaptive_compression.idle_duration_s` | `int` | `60` | Env: `DISTLLM_ADAPTIVE_COMPRESSION__IDLE_DURATION_S` |
| `adaptive_compression.check_interval_s` | `int` | `15` | Env: `DISTLLM_ADAPTIVE_COMPRESSION__CHECK_INTERVAL_S` |
| `adaptive_compression.compression_method` | `str` | `'int4'` | Env: `DISTLLM_ADAPTIVE_COMPRESSION__COMPRESSION_METHOD` |
| `adaptive_compression.calibration_samples` | `int` | `128` | Env: `DISTLLM_ADAPTIVE_COMPRESSION__CALIBRATION_SAMPLES` |
| `adaptive_compression.output_dir` | `str` | `'/tmp/distllm-compress'` | Env: `DISTLLM_ADAPTIVE_COMPRESSION__OUTPUT_DIR` |
## `alerting`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `alerting.enabled` | `bool` | `False` | Env: `DISTLLM_ALERTING__ENABLED` |
| `alerting.prometheus_url` | `str` | `'http://localhost:9090'` | Env: `DISTLLM_ALERTING__PROMETHEUS_URL` |
| `alerting.rule_file` | `Union` |  | Env: `DISTLLM_ALERTING__RULE_FILE` |
## `chaos`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `chaos.enabled` | `bool` | `False` | Env: `DISTLLM_CHAOS__ENABLED` |
| `chaos.allowed_scenarios` | `list` | `PydanticUndefined` | Env: `DISTLLM_CHAOS__ALLOWED_SCENARIOS` |
| `chaos.max_latency_ms` | `int` | `5000` | Env: `DISTLLM_CHAOS__MAX_LATENCY_MS` |
## `canary`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `canary.enabled` | `bool` | `False` | Env: `DISTLLM_CANARY__ENABLED` |
| `canary.stable_version` | `str` | `'stable'` | Env: `DISTLLM_CANARY__STABLE_VERSION` |
| `canary.canary_version` | `str` | `'canary'` | Env: `DISTLLM_CANARY__CANARY_VERSION` |
| `canary.rollback_threshold` | `float` | `0.05` | Env: `DISTLLM_CANARY__ROLLBACK_THRESHOLD` |
| `canary.stages` | `list` | `PydanticUndefined` | Env: `DISTLLM_CANARY__STAGES` |
## `cost`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `cost.enabled` | `bool` | `False` | Env: `DISTLLM_COST__ENABLED` |
| `cost.budget_per_hour` | `float` | `0.0` | Env: `DISTLLM_COST__BUDGET_PER_HOUR` |
| `cost.spot_preference` | `float` | `0.8` | Env: `DISTLLM_COST__SPOT_PREFERENCE` |
## `tenant`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `tenant.enabled` | `bool` | `False` | Env: `DISTLLM_TENANT__ENABLED` |
| `tenant.default_tier` | `str` | `'free'` | Env: `DISTLLM_TENANT__DEFAULT_TIER` |
| `tenant.admin_api_key` | `Union` |  | Admin API key for tenant management. Set via DISTLLM_TENANT__ADMIN_API_KEY env var. Env: `DISTLLM_TENANT__ADMIN_API_KEY` |
## `rate_limit`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `rate_limit.enabled` | `bool` | `True` | Env: `DISTLLM_RATE_LIMIT__ENABLED` |
| `rate_limit.default_rpm` | `float` | `60.0` | Env: `DISTLLM_RATE_LIMIT__DEFAULT_RPM` |
| `rate_limit.endpoint_limits` | `dict` | `PydanticUndefined` | Env: `DISTLLM_RATE_LIMIT__ENDPOINT_LIMITS` |
| `rate_limit.burst_multiplier` | `float` | `1.5` | Env: `DISTLLM_RATE_LIMIT__BURST_MULTIPLIER` |
| `rate_limit.auth_rpm_multiplier` | `float` | `2.0` | Env: `DISTLLM_RATE_LIMIT__AUTH_RPM_MULTIPLIER` |
## `model_hub`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model_hub.enabled` | `bool` | `True` | Env: `DISTLLM_MODEL_HUB__ENABLED` |
| `model_hub.cache_dir` | `Union` |  | Env: `DISTLLM_MODEL_HUB__CACHE_DIR` |
| `model_hub.max_cache_size_gb` | `float` | `50.0` | Env: `DISTLLM_MODEL_HUB__MAX_CACHE_SIZE_GB` |
| `model_hub.offline_mode` | `bool` | `False` | Env: `DISTLLM_MODEL_HUB__OFFLINE_MODE` |
| `model_hub.hf_token` | `Union` |  | HuggingFace token. Set via DISTLLM_MODEL_HUB__HF_TOKEN env var or .env file, NOT in YAML config. Env: `DISTLLM_MODEL_HUB__HF_TOKEN` |
| `model_hub.download_timeout_s` | `int` | `300` | Env: `DISTLLM_MODEL_HUB__DOWNLOAD_TIMEOUT_S` |
## `prompt_template`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `prompt_template.template` | `str` | `'auto'` | Env: `DISTLLM_PROMPT_TEMPLATE__TEMPLATE` |
| `prompt_template.custom_template_path` | `Union` |  | Env: `DISTLLM_PROMPT_TEMPLATE__CUSTOM_TEMPLATE_PATH` |
## `embedding`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `embedding.embedding_model` | `str` | `''` | Env: `DISTLLM_EMBEDDING__EMBEDDING_MODEL` |
| `embedding.rerank_model` | `str` | `''` | Env: `DISTLLM_EMBEDDING__RERANK_MODEL` |
| `embedding.normalize` | `bool` | `True` | Env: `DISTLLM_EMBEDDING__NORMALIZE` |
| `embedding.max_length` | `int` | `512` | Env: `DISTLLM_EMBEDDING__MAX_LENGTH` |
| `embedding.batch_size` | `int` | `32` | Env: `DISTLLM_EMBEDDING__BATCH_SIZE` |
## `version`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `version.enabled` | `bool` | `False` | Env: `DISTLLM_VERSION__ENABLED` |
| `version.max_versions` | `int` | `4` | Env: `DISTLLM_VERSION__MAX_VERSIONS` |
| `version.shadow_enabled` | `bool` | `False` | Env: `DISTLLM_VERSION__SHADOW_ENABLED` |
| `version.shadow_pct` | `float` | `0.0` | Env: `DISTLLM_VERSION__SHADOW_PCT` |
| `version.blue_green_enabled` | `bool` | `False` | Env: `DISTLLM_VERSION__BLUE_GREEN_ENABLED` |
| `version.ab_testing_enabled` | `bool` | `False` | Env: `DISTLLM_VERSION__AB_TESTING_ENABLED` |
| `version.ab_test_split` | `float` | `50.0` | Env: `DISTLLM_VERSION__AB_TEST_SPLIT` |
| `version.auto_promote_enabled` | `bool` | `False` | Env: `DISTLLM_VERSION__AUTO_PROMOTE_ENABLED` |
| `version.min_samples` | `int` | `100` | Env: `DISTLLM_VERSION__MIN_SAMPLES` |
| `version.significance_level` | `float` | `0.05` | Env: `DISTLLM_VERSION__SIGNIFICANCE_LEVEL` |
## `plugins`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `plugins.enabled` | `bool` | `True` | Env: `DISTLLM_PLUGINS__ENABLED` |
| `plugins.plugins` | `list` | `PydanticUndefined` | Env: `DISTLLM_PLUGINS__PLUGINS` |
## `hybrid_parallel`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `hybrid_parallel.enabled` | `bool` | `False` | Env: `DISTLLM_HYBRID_PARALLEL__ENABLED` |
| `hybrid_parallel.auto_detect` | `bool` | `True` | Env: `DISTLLM_HYBRID_PARALLEL__AUTO_DETECT` |
| `hybrid_parallel.tp_enabled` | `bool` | `True` | Env: `DISTLLM_HYBRID_PARALLEL__TP_ENABLED` |
| `hybrid_parallel.pp_overlap` | `bool` | `True` | Env: `DISTLLM_HYBRID_PARALLEL__PP_OVERLAP` |
| `hybrid_parallel.ep_enabled` | `bool` | `True` | Env: `DISTLLM_HYBRID_PARALLEL__EP_ENABLED` |
| `hybrid_parallel.force_tp_world_size` | `int` | `0` | Env: `DISTLLM_HYBRID_PARALLEL__FORCE_TP_WORLD_SIZE` |
| `hybrid_parallel.force_pp_stages` | `int` | `0` | Env: `DISTLLM_HYBRID_PARALLEL__FORCE_PP_STAGES` |
| `hybrid_parallel.shard_across_nodes` | `bool` | `False` | Env: `DISTLLM_HYBRID_PARALLEL__SHARD_ACROSS_NODES` |
## `zero_copy`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `zero_copy.enabled` | `bool` | `False` | Env: `DISTLLM_ZERO_COPY__ENABLED` |
| `zero_copy.prefer_rdma` | `bool` | `True` | Env: `DISTLLM_ZERO_COPY__PREFER_RDMA` |
| `zero_copy.fallback_to_nccl` | `bool` | `True` | Env: `DISTLLM_ZERO_COPY__FALLBACK_TO_NCCL` |
| `zero_copy.intranode_ipc` | `bool` | `True` | Env: `DISTLLM_ZERO_COPY__INTRANODE_IPC` |
## `adaptive_precision`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `adaptive_precision.enabled` | `bool` | `False` | Env: `DISTLLM_ADAPTIVE_PRECISION__ENABLED` |
| `adaptive_precision.calibration_samples` | `int` | `64` | Env: `DISTLLM_ADAPTIVE_PRECISION__CALIBRATION_SAMPLES` |
| `adaptive_precision.target_precision` | `str` | `'auto'` | Env: `DISTLLM_ADAPTIVE_PRECISION__TARGET_PRECISION` |
| `adaptive_precision.max_quality_loss_pct` | `float` | `0.1` | Env: `DISTLLM_ADAPTIVE_PRECISION__MAX_QUALITY_LOSS_PCT` |
## `predictive_cache`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `predictive_cache.enabled` | `bool` | `False` | Env: `DISTLLM_PREDICTIVE_CACHE__ENABLED` |
| `predictive_cache.gpu_cache_mb` | `int` | `512` | Env: `DISTLLM_PREDICTIVE_CACHE__GPU_CACHE_MB` |
| `predictive_cache.cpu_cache_mb` | `int` | `4096` | Env: `DISTLLM_PREDICTIVE_CACHE__CPU_CACHE_MB` |
| `predictive_cache.pattern_decay_hours` | `float` | `24.0` | Env: `DISTLLM_PREDICTIVE_CACHE__PATTERN_DECAY_HOURS` |
| `predictive_cache.min_prefix_len` | `int` | `8` | Env: `DISTLLM_PREDICTIVE_CACHE__MIN_PREFIX_LEN` |
| `predictive_cache.background_compress_interval_s` | `int` | `300` | Env: `DISTLLM_PREDICTIVE_CACHE__BACKGROUND_COMPRESS_INTERVAL_S` |
## `cache`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `cache.prefix_enabled` | `bool` | `True` | Env: `DISTLLM_CACHE__PREFIX_ENABLED` |
| `cache.prefix_max_entries` | `int` | `1024` | Env: `DISTLLM_CACHE__PREFIX_MAX_ENTRIES` |
| `cache.prefix_min_prefix_len` | `int` | `16` | Env: `DISTLLM_CACHE__PREFIX_MIN_PREFIX_LEN` |
| `cache.radix_tree_enabled` | `bool` | `True` | Env: `DISTLLM_CACHE__RADIX_TREE_ENABLED` |
| `cache.persistence_enabled` | `bool` | `False` | Env: `DISTLLM_CACHE__PERSISTENCE_ENABLED` |
| `cache.persistence_storage_path` | `str` | `'.distllm_cache'` | Env: `DISTLLM_CACHE__PERSISTENCE_STORAGE_PATH` |
| `cache.persistence_max_disk_gb` | `float` | `50.0` | Env: `DISTLLM_CACHE__PERSISTENCE_MAX_DISK_GB` |
| `cache.persistence_ttl_hours` | `float` | `24.0` | Env: `DISTLLM_CACHE__PERSISTENCE_TTL_HOURS` |
| `cache.background_compaction_enabled` | `bool` | `False` | Env: `DISTLLM_CACHE__BACKGROUND_COMPACTION_ENABLED` |
| `cache.background_compaction_interval_s` | `float` | `300.0` | Env: `DISTLLM_CACHE__BACKGROUND_COMPACTION_INTERVAL_S` |
| `cache.predictive_enabled` | `bool` | `False` | Env: `DISTLLM_CACHE__PREDICTIVE_ENABLED` |
| `cache.predictive_gpu_cache_mb` | `int` | `512` | Env: `DISTLLM_CACHE__PREDICTIVE_GPU_CACHE_MB` |
| `cache.predictive_cpu_cache_mb` | `int` | `4096` | Env: `DISTLLM_CACHE__PREDICTIVE_CPU_CACHE_MB` |
| `cache.predictive_pattern_decay_hours` | `float` | `24.0` | Env: `DISTLLM_CACHE__PREDICTIVE_PATTERN_DECAY_HOURS` |
| `cache.predictive_min_prefix_len` | `int` | `8` | Env: `DISTLLM_CACHE__PREDICTIVE_MIN_PREFIX_LEN` |
| `cache.gossip_enabled` | `bool` | `False` | Env: `DISTLLM_CACHE__GOSSIP_ENABLED` |
| `cache.gossip_interval` | `float` | `10.0` | Env: `DISTLLM_CACHE__GOSSIP_INTERVAL` |
| `cache.gossip_max_peers` | `int` | `16` | Env: `DISTLLM_CACHE__GOSSIP_MAX_PEERS` |
| `cache.gossip_cache_ttl` | `float` | `300.0` | Env: `DISTLLM_CACHE__GOSSIP_CACHE_TTL` |
| `cache.eviction_strategy` | `str` | `'hybrid'` | Env: `DISTLLM_CACHE__EVICTION_STRATEGY` |
| `cache.size_aware_admission` | `bool` | `True` | Env: `DISTLLM_CACHE__SIZE_AWARE_ADMISSION` |
| `cache.memory_adaptive_budget` | `bool` | `True` | Env: `DISTLLM_CACHE__MEMORY_ADAPTIVE_BUDGET` |
| `predictive_migration` | `dict` | `PydanticUndefined` | Env: `DISTLLM_PREDICTIVE_MIGRATION` |
| `structured_output` | `dict` | `PydanticUndefined` | Env: `DISTLLM_STRUCTURED_OUTPUT` |
## `self_optimizing`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `self_optimizing.enabled` | `bool` | `False` | Env: `DISTLLM_SELF_OPTIMIZING__ENABLED` |
| `self_optimizing.tune_interval_seconds` | `float` | `60.0` | Env: `DISTLLM_SELF_OPTIMIZING__TUNE_INTERVAL_SECONDS` |
| `self_optimizing.warmup_seconds` | `float` | `30.0` | Env: `DISTLLM_SELF_OPTIMIZING__WARMUP_SECONDS` |
| `self_optimizing.profile_dir` | `Union` |  | Env: `DISTLLM_SELF_OPTIMIZING__PROFILE_DIR` |
| `optimization` | `dict` | `PydanticUndefined` | Env: `DISTLLM_OPTIMIZATION` |
## `cuda_graph`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `cuda_graph.enabled` | `bool` | `False` | Env: `DISTLLM_CUDA_GRAPH__ENABLED` |
| `cuda_graph.batch_sizes` | `list` | `[1, 2, 4, 8, 16, 32]` | Env: `DISTLLM_CUDA_GRAPH__BATCH_SIZES` |
## `compile`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `compile.enabled` | `bool` | `False` | Env: `DISTLLM_COMPILE__ENABLED` |
| `compile.mode` | `str` | `'reduce-overhead'` | Env: `DISTLLM_COMPILE__MODE` |
| `compile.fullgraph` | `bool` | `False` | Env: `DISTLLM_COMPILE__FULLGRAPH` |
## `slora`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `slora.enabled` | `bool` | `False` | Env: `DISTLLM_SLORA__ENABLED` |
| `slora.max_adapters` | `int` | `64` | Env: `DISTLLM_SLORA__MAX_ADAPTERS` |
## `rag`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `rag.enabled` | `bool` | `False` | Env: `DISTLLM_RAG__ENABLED` |
| `rag.dimension` | `int` | `768` | Env: `DISTLLM_RAG__DIMENSION` |
| `rag.chunk_size` | `int` | `512` | Env: `DISTLLM_RAG__CHUNK_SIZE` |
| `rag.chunk_overlap` | `int` | `50` | Env: `DISTLLM_RAG__CHUNK_OVERLAP` |
| `rag.index_path` | `Union` |  | Env: `DISTLLM_RAG__INDEX_PATH` |
## `agent`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `agent.enabled` | `bool` | `False` | Env: `DISTLLM_AGENT__ENABLED` |
| `agent.max_iterations` | `int` | `10` | Env: `DISTLLM_AGENT__MAX_ITERATIONS` |
| `agent.reflection_enabled` | `bool` | `True` | Env: `DISTLLM_AGENT__REFLECTION_ENABLED` |
## `disagg`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `disagg.enabled` | `bool` | `False` | Env: `DISTLLM_DISAGG__ENABLED` |
| `disagg.prefill_nodes` | `list` | `[]` | Env: `DISTLLM_DISAGG__PREFILL_NODES` |
| `disagg.decode_nodes` | `list` | `[]` | Env: `DISTLLM_DISAGG__DECODE_NODES` |
## `chat_router`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `chat_router.enabled` | `bool` | `False` | Env: `DISTLLM_CHAT_ROUTER__ENABLED` |
| `chat_router.name` | `str` | `'hybrid'` | Model name that clients use to invoke this router (e.g., 'hybrid', 'smart-router') Env: `DISTLLM_CHAT_ROUTER__NAME` |
| `chat_router.default_model` | `str` | `''` | Default model when no rules match Env: `DISTLLM_CHAT_ROUTER__DEFAULT_MODEL` |
| `chat_router.routes` | `list` | `PydanticUndefined` | Ordered routing rules Env: `DISTLLM_CHAT_ROUTER__ROUTES` |
## `hardware`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `hardware.device_type` | `str` | `'auto'` | Env: `DISTLLM_HARDWARE__DEVICE_TYPE` |
| `hardware.preferred_backend` | `str` | `'auto'` | Env: `DISTLLM_HARDWARE__PREFERRED_BACKEND` |
| `hardware.force_device_id` | `int` | `-1` | Env: `DISTLLM_HARDWARE__FORCE_DEVICE_ID` |
| `hardware.fallback_to_cpu` | `bool` | `True` | Env: `DISTLLM_HARDWARE__FALLBACK_TO_CPU` |
| `hardware.rocm_visible_devices` | `str` | `''` | Env: `DISTLLM_HARDWARE__ROCM_VISIBLE_DEVICES` |
| `hardware.mps_optimize_memory` | `bool` | `True` | Env: `DISTLLM_HARDWARE__MPS_OPTIMIZE_MEMORY` |
| `hardware.xpu_oneapi_verbose` | `bool` | `False` | Env: `DISTLLM_HARDWARE__XPU_ONEAPI_VERBOSE` |
| `hardware.cpu_threads` | `int` | `0` | Env: `DISTLLM_HARDWARE__CPU_THREADS` |
| `hardware.cpu_numa_aware` | `bool` | `True` | Env: `DISTLLM_HARDWARE__CPU_NUMA_AWARE` |
## `vllm`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `vllm.enabled` | `bool` | `False` | Env: `DISTLLM_VLLM__ENABLED` |
| `vllm.tensor_parallel_size` | `int` | `1` | Env: `DISTLLM_VLLM__TENSOR_PARALLEL_SIZE` |
| `vllm.gpu_memory_utilization` | `float` | `0.9` | Env: `DISTLLM_VLLM__GPU_MEMORY_UTILIZATION` |
| `vllm.max_num_seqs` | `int` | `256` | Env: `DISTLLM_VLLM__MAX_NUM_SEQS` |
| `vllm.max_num_batched_tokens` | `int` | `8192` | Env: `DISTLLM_VLLM__MAX_NUM_BATCHED_TOKENS` |
| `vllm.dtype` | `str` | `'auto'` | Env: `DISTLLM_VLLM__DTYPE` |
| `vllm.seed` | `int` | `0` | Env: `DISTLLM_VLLM__SEED` |
| `vllm.enforce_eager` | `bool` | `False` | Env: `DISTLLM_VLLM__ENFORCE_EAGER` |
| `vllm.max_model_len` | `Union` |  | Env: `DISTLLM_VLLM__MAX_MODEL_LEN` |
## `llamacpp`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `llamacpp.enabled` | `bool` | `False` | Env: `DISTLLM_LLAMACPP__ENABLED` |
| `llamacpp.model_path` | `str` | `''` | Env: `DISTLLM_LLAMACPP__MODEL_PATH` |
| `llamacpp.n_gpu_layers` | `int` | `0` | Env: `DISTLLM_LLAMACPP__N_GPU_LAYERS` |
| `llamacpp.n_ctx` | `int` | `2048` | Env: `DISTLLM_LLAMACPP__N_CTX` |
| `llamacpp.n_threads` | `Union` |  | Env: `DISTLLM_LLAMACPP__N_THREADS` |
| `llamacpp.n_batch` | `int` | `512` | Env: `DISTLLM_LLAMACPP__N_BATCH` |
| `llamacpp.seed` | `int` | `0` | Env: `DISTLLM_LLAMACPP__SEED` |
| `llamacpp.verbose` | `bool` | `False` | Env: `DISTLLM_LLAMACPP__VERBOSE` |
## `wide_area`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `wide_area.enabled` | `bool` | `False` | Env: `DISTLLM_WIDE_AREA__ENABLED` |
| `wide_area.p2p_forwarding` | `bool` | `True` | Env: `DISTLLM_WIDE_AREA__P2P_FORWARDING` |
| `wide_area.tokens_before_forward` | `int` | `10` | Env: `DISTLLM_WIDE_AREA__TOKENS_BEFORE_FORWARD` |
| `wide_area.wan_timeout_seconds` | `int` | `60` | Env: `DISTLLM_WIDE_AREA__WAN_TIMEOUT_SECONDS` |
| `wide_area.max_retries` | `int` | `3` | Env: `DISTLLM_WIDE_AREA__MAX_RETRIES` |
| `wide_area.backoff_base_seconds` | `float` | `1.0` | Env: `DISTLLM_WIDE_AREA__BACKOFF_BASE_SECONDS` |
## `defragmentation`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `defragmentation.enabled` | `bool` | `False` | Env: `DISTLLM_DEFRAGMENTATION__ENABLED` |
| `defragmentation.policy` | `str` | `'balanced'` | Compaction policy: lazy, balanced, or aggressive Env: `DISTLLM_DEFRAGMENTATION__POLICY` |
| `defragmentation.interval_seconds` | `float` | `60.0` | Seconds between background defrag checks Env: `DISTLLM_DEFRAGMENTATION__INTERVAL_SECONDS` |
| `defragmentation.max_blocks_per_pass` | `int` | `0` | Max blocks per pass (0 = unlimited) Env: `DISTLLM_DEFRAGMENTATION__MAX_BLOCKS_PER_PASS` |
| `defragmentation.threshold` | `float` | `0.0` | Override policy threshold (0 = use policy default) Env: `DISTLLM_DEFRAGMENTATION__THRESHOLD` |
| `defragmentation.tiered_compaction` | `bool` | `False` | Enable L2 (CPU swap) and L3 (NVMe) compaction Env: `DISTLLM_DEFRAGMENTATION__TIERED_COMPACTION` |
| `defragmentation.l2_cpu_swap_threshold` | `float` | `0.6` | Env: `DISTLLM_DEFRAGMENTATION__L2_CPU_SWAP_THRESHOLD` |
| `defragmentation.l3_nvme_swap_threshold` | `float` | `0.8` | Env: `DISTLLM_DEFRAGMENTATION__L3_NVME_SWAP_THRESHOLD` |
| `defragmentation.cuda_stream_priority` | `int` | `-1` | CUDA stream priority for copy ops Env: `DISTLLM_DEFRAGMENTATION__CUDA_STREAM_PRIORITY` |
| `defragmentation.enable_predictive` | `bool` | `False` | Predictive (preemptive) defragmentation Env: `DISTLLM_DEFRAGMENTATION__ENABLE_PREDICTIVE` |
| `defragmentation.enable_prometheus` | `bool` | `False` | Export Prometheus defrag metrics Env: `DISTLLM_DEFRAGMENTATION__ENABLE_PROMETHEUS` |

---

## Environment Variable Mapping

Any config field can be overridden via environment variables:

```
config.yaml:  model.name -> DISTLLM_MODEL__NAME
config.yaml:  coordinator.port -> DISTLLM_COORDINATOR__PORT
config.yaml:  tls.enabled -> DISTLLM_TLS__ENABLED
```
