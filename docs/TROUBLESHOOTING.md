# Troubleshooting Runbook

Operational runbook for diagnosing and resolving issues in Distributed LLM (DistLLM) clusters. Each section lists the exact symptom, root cause, step-by-step fix, and prevention measures.

---

## Table of Contents

1. [Installation Issues](#1-installation-issues)
2. [Model Loading Failures](#2-model-loading-failures)
3. [Distributed Pipeline Errors](#3-distributed-pipeline-errors)
4. [API Errors](#4-api-errors)
5. [Performance Issues](#5-performance-issues)
6. [GPU Issues](#6-gpu-issues)
7. [Kubernetes Issues](#7-kubernetes-issues)
8. [Docker Issues](#8-docker-issues)
9. [TLS / Certificate Issues](#9-tls--certificate-issues)
10. [Federation Issues](#10-federation-issues)
11. [Diagnostic Commands](#11-diagnostic-commands)

---

## 1. Installation Issues

### 1.1 `pip install` fails with CUDA errors

**Symptom:**
```
ERROR: Could not find a version that satisfies the requirement torch
```
or
```
ERROR: No matching distribution found for torch
```

**Root cause:** PyTorch requires a specific index URL for CUDA-enabled wheels. The default PyPI index only provides CPU-only builds.

**Fix:**
```bash
# Install PyTorch with CUDA support first
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Then install DistLLM
pip install distllm
```

**Prevention:** Pin the PyTorch index URL in your `requirements.txt`:
```
--extra-index-url https://download.pytorch.org/whl/cu121
torch>=2.1.0
```

---

### 1.2 `grpcio` build fails on Windows

**Symptom:**
```
error: Microsoft Visual C++ 14.0 or greater is required
```

**Root cause:** No pre-built wheel available for your Python version/platform, and the source build requires a C compiler.

**Fix:**
```bash
# Install pre-built wheel (skip source compilation)
pip install grpcio --only-binary=:all:
```

**Prevention:** Use Python 3.10-3.12 on Windows (best wheel coverage). Alternatively, install the [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/).

---

### 1.3 `distllm` command not found after install

**Symptom:**
```
'distllm' is not recognized as an internal or external command
```

**Root cause:** The pip install scripts directory is not on your `PATH`.

**Fix:**
```bash
# Find where pip installed the script
python -m site --user-scripts

# Add that directory to PATH, or run via module
python -m distllm.cli --help
```

**Prevention:** Use a virtual environment (`python -m venv .venv`) so scripts land in a predictable location.

---

### 1.4 Dependency version conflicts

**Symptom:**
```
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed
```
or import errors at runtime like `AttributeError: module 'transformers' has no attribute ...`

**Root cause:** Installed packages have incompatible version ranges.

**Fix:**
```bash
# Create a clean environment
python -m venv .venv-clean
source .venv-clean/bin/activate  # Linux/macOS
# .venv-clean\Scripts\activate   # Windows

# Install with locked requirements
pip install -r requirements.lock
pip install distllm
```

**Prevention:** Always use `pip install -r requirements.lock` in production. Generate the lock file with `pip freeze > requirements.lock`.

---

## 2. Model Loading Failures

### 2.1 CUDA out of memory during model load

**Symptom:**
```
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate X MiB
```
or in DistLLM logs:
```
[ERROR] OOM_ERROR: GPU out of memory on node node_0
```

**Root cause:** The model weights exceed available GPU VRAM. This happens when the model is too large for a single GPU, or residual memory from a previous process is still allocated.

**Fix:**
```bash
# Step 1: Verify GPU memory is free
nvidia-smi
# If another process is using the GPU, kill it or wait

# Step 2: Use quantization to reduce memory footprint
export DISTLLM_QUANTIZATION_BITS=4    # INT4: ~75% savings
# or
export DISTLLM_QUANTIZATION_BITS=8    # INT8: ~50% savings

# Step 3: Enable CPU offloading for KV cache
export DISTLLM_CPU_OFFLOAD=true

# Step 4: Reduce layers assigned to this node
# In config.yaml, narrow the layer range:
#   start_layer: 0
#   end_layer: 7   # was 15

# Step 5: Use more nodes to spread the model
distllm --model meta-llama/Llama-3.1-70B --nodes node0:50051:0:15 node1:50052:16:31
```

**Prevention:**
- Check the [Model Compatibility Matrix](MODEL_COMPATIBILITY.md) for VRAM requirements before deploying.
- Use `distllm model info <model>` to see estimated memory per node.
- Set `DISTLLM_QUANTIZATION_BITS=8` as a safe default for production.

---

### 2.2 Unsupported model architecture

**Symptom:**
```
ModelError: Failed to load model 'org/custom-model': Architecture 'CustomForCausalLM' is not supported
```
or
```
KeyError: 'CustomForCausalLM'
```

**Root cause:** The model's architecture class is not registered in DistLLM's `ModelPartitioner`. This happens with custom or newly released model architectures.

**Fix:**
```bash
# Step 1: Check if the model works with trust_remote_code
export DISTLLM_TRUST_REMOTE_CODE=true
distllm model load org/custom-model

# Step 2: If still failing, check the model's config.json for the architecture name
python -c "from transformers import AutoConfig; c = AutoConfig.from_pretrained('org/custom-model'); print(c.architectures)"

# Step 3: If the architecture is a known variant (e.g., LlamaForCausalLM),
# you may be able to alias it. File an issue with the architecture name.
```

**Prevention:**
- Check [MODEL_COMPATIBILITY.md](MODEL_COMPATIBILITY.md) before selecting a model.
- When using community models, always set `DISTLLM_TRUST_REMOTE_CODE=true`.

---

### 2.3 Missing tokenizer

**Symptom:**
```
OSError: org/model does not appear to have a file named tokenizer.json
```
or
```
Tokenizer not found: Unable to load tokenizer for 'org/model'
```

**Root cause:** The model repository is missing tokenizer files, or the tokenizer class requires `sentencepiece`/`tokenizers` packages that are not installed.

**Fix:**
```bash
# Step 1: Ensure tokenizer dependencies are installed
pip install sentencepiece tokenizers protobuf

# Step 2: For SentencePiece models (Llama, Gemma, etc.)
pip install sentencepiece

# Step 3: If using a GGUF model with llama.cpp backend, the tokenizer
# is embedded in the model file — no separate tokenizer needed

# Step 4: For models with tokenizer in a subdirectory
export DISTLLM_TOKENIZER_SUBFOLDER="custom_tokenizer"
distllm model load org/model
```

**Prevention:** Install the full dependency set: `pip install distllm[all]`.

---

### 2.4 HuggingFace authentication failure for gated models

**Symptom:**
```
huggingface_hub.utils._errors.HfHubHTTPError: 401 Client Error: Unauthorized
```
or
```
403 Forbidden: Access to model meta-llama/Llama-3.1-70B is restricted
```

**Root cause:** The model is gated (requires acceptance of a license agreement on HuggingFace) and no valid token is provided.

**Fix:**
```bash
# Step 1: Accept the model license on HuggingFace
# Visit https://huggingface.co/meta-llama/Llama-3.1-70B and accept the terms

# Step 2: Generate an access token at https://huggingface.co/settings/tokens

# Step 3: Set the token
export HUGGING_FACE_HUB_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxx

# Step 4: Verify
python -c "from huggingface_hub import HfApi; print(HfApi().whoami())"
```

**Prevention:** Use a `.env` file or secret manager. Never commit tokens to version control.

---

### 2.5 Model download interrupted / corrupted cache

**Symptom:**
```
RuntimeError: Error(s) in loading state_dict: size mismatch for model.layers.0.self_attn.q_proj.weight
```
or
```
safetensors_rust.SafetensorError: Error deserializing header
```

**Root cause:** The model download was interrupted, leaving partial or corrupted files in the HuggingFace cache.

**Fix:**
```bash
# Step 1: Clear the corrupted model from cache
huggingface-cli delete-cache --model org/model

# Or manually:
rm -rf ~/.cache/huggingface/hub/models--org--model

# Step 2: Re-download
distllm model download org/model

# Step 3: Verify integrity
python -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('org/model')"
```

**Prevention:** Use `DISTLLM_DOWNLOAD_RESUME=true` (default) to resume interrupted downloads.

---

## 3. Distributed Pipeline Errors

### 3.1 Node unreachable

**Symptom:**
```
NodeUnreachableError: Node node_1 at 192.168.1.11:50051 is unreachable
```
or in gRPC logs:
```
grpc._channel._InactiveRpcError: <InactiveRpcError of RPC that terminated with: StatusCode.UNAVAILABLE>
```

**Root cause:** The worker node process is not running, the host is down, a firewall is blocking the port, or the gRPC server failed to start.

**Fix:**
```bash
# Step 1: Check if the worker process is running
ssh user@192.168.1.11 "ps aux | grep distllm-node"

# Step 2: If not running, start it
distllm-node --node-id node_1 --coordinator coordinator:50051 --port 50051 --model meta-llama/Llama-3.1-8B

# Step 3: Test network connectivity
nc -zv 192.168.1.11 50051
# or
telnet 192.168.1.11 50051

# Step 4: Test gRPC health endpoint
grpcurl -plaintext 192.168.1.11:50051 grpc.health.v1.Health/Check

# Step 5: Check firewall rules
sudo iptables -L -n | grep 50051
# or on Windows:
netsh advfirewall firewall show rule name=all | findstr 50051

# Step 6: Check coordinator logs for the node's last seen time
distllm cluster status
```

**Prevention:**
- Use `systemd` or a process supervisor to auto-restart crashed workers.
- Set up health check monitoring (see [Docker healthchecks](#8-docker-issues) or [Kubernetes probes](#7-kubernetes-issues)).
- Open firewall ports during initial setup.

---

### 3.2 gRPC timeout during forward pass

**Symptom:**
```
GRPCTimeoutError: gRPC call to node node_2 at 192.168.1.12:50051 timed out after 30.0s
```
or
```
grpc.RpcError: StatusCode.DEADLINE_EXCEEDED
```

**Root cause:** The worker node is overloaded (GPU saturated), the network link is slow/lossy, or the request payload (hidden states tensor) is too large for the timeout window.

**Fix:**
```bash
# Step 1: Check the worker's GPU utilization
ssh user@192.168.1.12 "nvidia-smi"
# If GPU utilization is 100% and memory is maxed out, the node is overloaded

# Step 2: Increase the pipeline timeout in config.yaml
# pipeline:
#   timeout_seconds: 60.0   # was 30.0

# Step 3: Check network bandwidth between nodes
iperf3 -c 192.168.1.12 -t 10

# Step 4: Enable gRPC compression to reduce payload size
export DISTLLM_GRPC_COMPRESSION=gzip

# Step 5: Enable activation quantization to shrink tensor transfers
export DISTLLM_ACTIVATION_QUANT_BITS=8

# Step 6: For WAN links, switch to QUIC transport
export DISTLLM_WAN_TRANSPORT=quic
```

**Prevention:**
- Use `iperf3` to validate inter-node bandwidth before deployment.
- Monitor `grpc_server_handling_seconds` via Prometheus.
- Set `pipeline.timeout_seconds` to 2x your expected worst-case latency.

---

### 3.3 KV cache mismatch between nodes

**Symptom:**
```
SerializationError: KV cache shape mismatch: expected [1, 32, 128, 64] got [1, 32, 256, 64]
```
or inference produces garbled/incorrect output after a few tokens.

**Root cause:** Nodes were started with different models, different `total_layers` values, or one node was restarted mid-inference while another still holds stale cache.

**Fix:**
```bash
# Step 1: Verify all nodes are running the same model
distllm cluster status
# Check that model_name, total_layers, and dtype match on every node

# Step 2: If mismatched, restart all nodes with consistent config
distllm cluster restart --all

# Step 3: Clear KV cache on all nodes
curl -X POST http://coordinator:8000/admin/cache/clear

# Step 4: Verify KV cache consistency
curl http://coordinator:8000/v1/metrics | grep kv_cache
```

**Prevention:**
- Always deploy from a single `config.yaml` shared across all nodes.
- Use `distllm config validate` before starting the cluster.
- Do not change `total_layers` or model name without restarting all nodes.

---

### 3.4 Circuit breaker tripped on a node

**Symptom:**
```
CircuitBreakerError: Circuit breaker open for node node_1 after 5 failures, recovery in 30.0s
```

**Root cause:** A node has failed multiple consecutive requests (OOM, timeout, crash). The circuit breaker opens to stop sending traffic to the failing node.

**Fix:**
```bash
# Step 1: Check why the node is failing
distllm system logs --node node_1 --tail 50

# Step 2: Check GPU health on that node
ssh user@node-1 "nvidia-smi"

# Step 3: If the node is healthy now, manually reset the circuit breaker
distllm cluster reset-circuit-breaker node_1

# Step 4: If the node is unhealthy, drain it and redistribute
distllm cluster drain node_1
# The coordinator will redistribute layers to remaining nodes
```

**Prevention:**
- Monitor `distllm_circuit_breaker_state` metric in Prometheus.
- Set appropriate `pipeline.max_retries` (default: 3).
- Configure auto-restart for worker processes.

---

### 3.5 Protobuf serialization error

**Symptom:**
```
ProtoError: Failed to decode ForwardPassRequest: invalid wire type 7 for field hidden_states
```
or
```
google.protobuf.message.DecodeError: Error parsing message
```

**Root cause:** Version mismatch between coordinator and worker protobuf definitions, or the tensor data exceeds the configured size limit.

**Fix:**
```bash
# Step 1: Ensure all nodes run the same DistLLM version
distllm --version  # Run on every node

# Step 2: If versions match, check if the tensor is too large
# Default limit is 4 GB; increase if needed:
export DISTLLM_MAX_TENSOR_BYTES=8589934592  # 8 GB

# Step 3: Rebuild protobuf stubs if you modified .proto files
cd proto && python -m grpc_tools.protoc -I. --python_out=../src/distllm/dist/ --grpc_python_out=../src/distllm/dist/ node.proto
```

**Prevention:** Always deploy the same DistLLM version across all nodes. Use `distllm cluster version-check` before rolling upgrades.

---

## 4. API Errors

### 4.1 HTTP 503 — Service Unavailable

**Symptom:**
```json
{"error": {"message": "Service temporarily unavailable", "type": "service_unavailable", "code": 503}}
```

**Root cause:** The coordinator is starting up, all worker nodes are unhealthy, or the cluster is at capacity and cannot accept new requests.

**Fix:**
```bash
# Step 1: Check coordinator health
curl http://coordinator:8000/health

# Step 2: Check worker node status
curl http://coordinator:8000/v1/nodes

# Step 3: If nodes are down, bring them up
distllm cluster status
distllm cluster start

# Step 4: If at capacity, increase batch size or add nodes
export DISTLLM_MAX_BATCH_SIZE=32
export DISTLLM_MAX_NUM_SEQS=512

# Step 5: Check if the readiness probe is failing (Kubernetes)
kubectl describe pod distllm-coordinator-xxxxx
```

**Prevention:**
- Set appropriate `readinessProbe` values so Kubernetes doesn't route traffic before the coordinator is ready.
- Monitor `distllm_requests_queued` and scale workers when the queue depth is sustained above threshold.

---

### 4.2 HTTP 429 — Too Many Requests

**Symptom:**
```json
{"error": {"message": "Rate limit exceeded", "type": "rate_limit_error", "code": 429}}
```
Headers include: `Retry-After: 12`, `X-RateLimit-Remaining: 0`

**Root cause:** The client has exceeded the configured rate limit (requests per minute).

**Fix:**
```bash
# Step 1: Check current rate limit configuration
curl http://coordinator:8000/v1/metrics | grep rate_limit

# Step 2: If limits are too low, increase them in config.yaml
# api:
#   rate_limit_requests: 1000        # was 100
#   rate_limit_window_seconds: 60

# Step 3: Implement client-side backoff
# Respect the Retry-After header in your client code

# Step 4: If using Redis-backed rate limiting, check Redis health
redis-cli -h redis PING
```

**Client-side backoff example (Python):**
```python
import time
import openai

client = openai.OpenAI(base_url="http://coordinator:8000/v1")

for attempt in range(5):
    try:
        response = client.chat.completions.create(model="llama-3", messages=[...])
        break
    except openai.RateLimitError as e:
        retry_after = int(e.response.headers.get("Retry-After", 5))
        time.sleep(retry_after)
```

**Prevention:**
- Use Redis-backed distributed rate limiting for multi-instance deployments.
- Set per-client API keys with appropriate limits via RBAC.

---

### 4.3 HTTP 504 — Gateway Timeout

**Symptom:**
```json
{"error": {"message": "Request timed out", "type": "timeout", "code": 504}}
```

**Root cause:** The inference request took longer than the configured timeout. Common with very long prompts, large `max_tokens`, or slow WAN-connected nodes.

**Fix:**
```bash
# Step 1: Check if the request is still queued
curl http://coordinator:8000/v1/metrics | grep requests_queued

# Step 2: Increase the API timeout
# config.yaml
# pipeline:
#   timeout_seconds: 120.0   # was 30.0

# Step 3: For long-context requests, enable chunked prefill
export DISTLLM_CHUNKED_PREFILL=true
export DISTLLM_CHUNKED_PREFILL_CHUNK_SIZE=512

# Step 4: Check network latency to all nodes
for node in node_0 node_1 node_2; do
  echo "$node: $(ping -c 3 $node | tail -1)"
done

# Step 5: If using a reverse proxy (nginx), increase proxy timeout
# nginx.conf:
#   proxy_read_timeout 300s;
#   proxy_send_timeout 300s;
```

**Prevention:**
- Set `pipeline.timeout_seconds` based on your P99 latency + buffer.
- Use `DISTLLM_CHUNKED_PREFILL=true` for long prompts.
- Monitor `distllm_request_duration_seconds` histogram.

---

### 4.4 HTTP 401 — Authentication Failure

**Symptom:**
```json
{"error": {"message": "Invalid API key", "type": "authentication_error", "code": 401}}
```

**Root cause:** The `Authorization` header is missing, the key is malformed, or the key hash doesn't match what's stored on the coordinator.

**Fix:**
```bash
# Step 1: Verify the key is set in the environment
echo $DISTLLM_API_KEY

# Step 2: Test with curl
curl -H "Authorization: Bearer $DISTLLM_API_KEY" http://coordinator:8000/v1/models

# Step 3: If the key is wrong, generate a new one
distllm config setup

# Step 4: Check the key store for the hash
distllm config list-keys

# Step 5: If using the cluster key for gRPC auth, ensure it matches on all nodes
echo $DISTLLM_CLUSTER_KEY
# Must be identical on coordinator and all workers
```

**Prevention:**
- Store API keys in a secret manager (Vault, AWS Secrets Manager).
- Use `distllm config validate` to verify the key is properly configured before starting.

---

### 4.5 HTTP 400 — Bad Request (input validation)

**Symptom:**
```json
{"error": {"message": "Invalid input: batch would exceed capacity: 50000 > 32768 tokens", "type": "invalid_request_error", "code": 400}}
```

**Root cause:** The request exceeds configured limits (max tokens, max message length, tensor dimensions).

**Fix:**
```bash
# Step 1: Reduce the input size
# - Shorten the prompt
# - Reduce max_tokens
# - Split into multiple requests

# Step 2: Increase limits if appropriate
# config.yaml
# api:
#   max_request_size_mb: 64
#   max_message_length: 262144
# generation:
#   max_new_tokens: 512
```

---

## 5. Performance Issues

### 5.1 Slow inference (< 1 token/sec)

**Symptom:** Generation takes many seconds per token. Users report extremely slow responses.

**Root cause:** Usually one of: GPU not utilized, network bottleneck, wrong pipeline strategy, or KV cache thrashing.

**Fix:**
```bash
# Step 1: Check GPU utilization
nvidia-smi
# If GPU is idle (< 10%), the model is not loaded on GPU

# Step 2: Check network latency between nodes
ping -c 10 node-1
iperf3 -c node-1

# Step 3: Check pipeline metrics
curl http://coordinator:8000/v1/metrics

# Step 4: Enable overlap pipeline (2-4x throughput)
export DISTLLM_ENABLE_PIPELINE_OVERLAP=true

# Step 5: For WAN links, use QUIC transport
export DISTLLM_WAN_TRANSPORT=quic

# Step 6: Enable Flash Attention
export DISTLLM_FLASH_ATTENTION=true

# Step 7: Enable CUDA Graph capture
export DISTLLM_CUDA_GRAPH=true

# Step 8: Increase batch size
export DISTLLM_MAX_BATCH_SIZE=16
```

**Prevention:**
- Always enable `DISTLLM_ENABLE_PIPELINE_OVERLAP=true` in production.
- Use `distllm benchmark run --model <model>` to establish baseline performance.
- Monitor `distllm_tokens_per_second` in Grafana.

---

### 5.2 High first-token latency (slow time-to-first-token)

**Symptom:** The first token takes 5-30 seconds to appear, but subsequent tokens are fast.

**Root cause:** Long prompt prefill is blocking the decode pipeline. The entire prompt must be processed through all nodes before generation begins.

**Fix:**
```bash
# Step 1: Enable chunked prefill
export DISTLLM_CHUNKED_PREFILL=true
export DISTLLM_CHUNKED_PREFILL_CHUNK_SIZE=512

# Step 2: Enable prefix caching for repeated system prompts
export DISTLLM_PREFIX_CACHE_ENABLED=true
export DISTLLM_PREFIX_CACHE_MAX_ENTRIES=1000

# Step 3: Use Sarathi-Serve pressure adaptation
export DISTLLM_SARATHI_PRESSURE_ADAPTATION=true
```

---

### 5.3 High memory usage / KV cache thrashing

**Symptom:** GPU memory is maxed out, frequent OOM warnings, or `kv_cache_eviction_rate` is high in metrics.

**Root cause:** Too many concurrent requests with long sequences, or KV cache quantization is not enabled.

**Fix:**
```bash
# Step 1: Enable KV cache quantization
export DISTLLM_KV_CACHE_QUANT_BITS=8   # INT8 (50% savings)
# or
export DISTLLM_KV_CACHE_QUANT_BITS=4   # INT4 (75% savings)

# Step 2: Enable PagedAttention
export DISTLLM_PAGED_ATTENTION=true

# Step 3: Enable defragmentation
export DISTLLM_DEFRAG_ENABLED=true
export DISTLLM_DEFRAG_POLICY=balanced

# Step 4: Reduce max sequence length
export DISTLLM_MAX_SEQ_LEN=2048

# Step 5: Enable CPU offloading for KV cache
export DISTLLM_CPU_OFFLOAD=true

# Step 6: Reduce max concurrent sequences
export DISTLLM_MAX_NUM_SEQS=128   # was 256
```

---

### 5.4 Inconsistent latency (jitter)

**Symptom:** P50 latency is acceptable but P99 is 5-10x higher.

**Root cause:** Network jitter on WAN links, GC pauses in the coordinator, or one slow node (straggler) delaying the entire pipeline.

**Fix:**
```bash
# Step 1: Identify the straggler node
curl http://coordinator:8000/v1/metrics | grep node_latency

# Step 2: Use QUIC transport for WAN (no head-of-line blocking)
export DISTLLM_WAN_TRANSPORT=quic

# Step 3: Enable straggler detection and recovery
# The coordinator will automatically bypass slow nodes
export DISTLLM_STRAGGLER_DETECTION=true
export DISTLLM_STRAGGLER_THRESHOLD_MS=5000

# Step 4: Use priority aging to prevent request starvation
export DISTLLM_PRIORITY_AGING_ENABLED=true
export DISTLLM_PRIORITY_AGING_INTERVAL=30
```

---

## 6. GPU Issues

### 6.1 CUDA out of memory during inference

**Symptom:**
```
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 256.00 MiB
(GPU 0; 24.00 GiB total; 22.50 GiB already allocated; 128.00 MiB free)
```

**Root cause:** The model + KV cache + activations exceed GPU memory. Often triggered by a long-sequence request or a batch that's too large.

**Fix:**
```bash
# Step 1: Reduce batch size
export DISTLLM_MAX_BATCH_SIZE=4
export DISTLLM_MAX_NUM_SEQS=32

# Step 2: Enable KV cache quantization
export DISTLLM_KV_CACHE_QUANT_BITS=8

# Step 3: Reduce max sequence length
export DISTLLM_MAX_SEQ_LEN=2048

# Step 4: Enable memory defragmentation
export DISTLLM_DEFRAG_ENABLED=true

# Step 5: Use model quantization
distllm model load llama-3-70b --quantization int4

# Step 6: Split the model across more nodes
# Reduce layers per node in config.yaml

# Step 7: Kill other GPU processes
nvidia-smi  # Check for other processes
kill -9 <PID>
```

**Prevention:**
- Set `DISTLLM_DEFRAG_ENABLED=true` by default.
- Use `nvidia-smi --query-gpu=memory.used,memory.total --format=csv -l 5` to monitor memory continuously.
- Set `DISTLLM_MAX_SEQ_LEN` to a reasonable limit for your use case.

---

### 6.2 CUDA driver version mismatch

**Symptom:**
```
RuntimeError: CUDA error: no kernel image is available for execution on the device
```
or
```
NVIDIA kernel module has version X.XX.XX but this version of PyTorch was compiled with version Y.YY.YY
```

**Root cause:** The NVIDIA driver version is too old for the CUDA toolkit version used to compile PyTorch, or the CUDA toolkit is not installed.

**Fix:**
```bash
# Step 1: Check driver and CUDA versions
nvidia-smi  # Shows driver version and supported CUDA version
nvcc --version  # Shows installed CUDA toolkit version

# Step 2: Verify compatibility
# Driver must support the CUDA version PyTorch was built with
# See: https://docs.nvidia.com/deploy/cuda-compatibility/

# Step 3: Update driver if needed
# Ubuntu:
sudo apt update && sudo apt install -y nvidia-driver-545
sudo reboot

# Step 4: Or install a PyTorch version matching your driver
pip install torch --index-url https://download.pytorch.org/whl/cu118  # For older drivers
```

**Prevention:** Check compatibility matrix before upgrading PyTorch or the driver. Pin PyTorch CUDA version in `requirements.lock`.

---

### 6.3 Multi-GPU not detected

**Symptom:**
```
WARNING: Only 1 GPU detected, but config specifies 2 nodes with cuda:0 and cuda:1
```
or only one GPU shows up in `nvidia-smi`.

**Root cause:** `CUDA_VISIBLE_DEVICES` is set to a single GPU, NVIDIA Container Toolkit is not installed (Docker), or the GPUs are in exclusive mode and one is already claimed.

**Fix:**
```bash
# Step 1: Verify all GPUs are visible
nvidia-smi -L

# Step 2: Check CUDA_VISIBLE_DEVICES
echo $CUDA_VISIBLE_DEVICES
# If set, ensure it includes all GPU indices: "0,1,2,3"

# Step 3: Check GPU compute mode
nvidia-smi -q | grep "Compute Mode"
# If "Exclusive Process", change to "Default":
sudo nvidia-smi -c DEFAULT

# Step 4: For Docker, ensure NVIDIA Container Toolkit is installed
nvidia-ctk --version
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

# Step 5: Verify in DistLLM
distllm system doctor
```

---

### 6.4 GPU temperature throttling

**Symptom:** Performance degrades after sustained load. `nvidia-smi` shows `THRM` in the "Pwr:Cap" column.

**Root cause:** The GPU is thermal throttling because temperature exceeds the threshold (usually 83-90°C).

**Fix:**
```bash
# Step 1: Check GPU temperature
nvidia-smi --query-gpu=temperature.gpu --format=csv -l 1

# Step 2: Set a more aggressive fan curve (if supported)
sudo nvidia-smi -pl 250  # Reduce power limit to 250W

# Step 3: Improve airflow
# - Clean dust from GPU heatsinks
# - Increase case fan speed
# - Ensure proper spacing between GPUs in multi-GPU setups
```

---

## 7. Kubernetes Issues

### 7.1 Pod CrashLoopBackOff

**Symptom:**
```bash
kubectl get pods
# NAME                              READY   STATUS             RESTARTS   AGE
# distllm-coordinator-7f8b9-xk2lp   0/1     CrashLoopBackOff   5          10m
```

**Root cause:** The coordinator or worker process is crashing on startup. Common causes: missing config, OOM killed, or bad environment variables.

**Fix:**
```bash
# Step 1: Check pod events
kubectl describe pod distllm-coordinator-7f8b9-xk2lp

# Step 2: Check container logs
kubectl logs distllm-coordinator-7f8b9-xk2lp --previous

# Step 3: If OOMKilled, increase memory limits
kubectl edit deployment distllm-coordinator
# resources:
#   limits:
#     memory: 32Gi   # was 16Gi

# Step 4: If config error, verify the ConfigMap
kubectl get configmap distllm-config -o yaml

# Step 5: If GPU not available, check node GPU capacity
kubectl get nodes -o yaml | grep -A5 nvidia.com/gpu
```

---

### 7.2 Readiness probe failing

**Symptom:**
```
Readiness probe failed: HTTP probe to /ready returned 503
```
Pod is in `Running` but not receiving traffic (0/1 READY).

**Root cause:** The coordinator started but hasn't loaded the model yet, or worker nodes are not connected.

**Fix:**
```bash
# Step 1: Check if the model is still loading
kubectl logs distllm-coordinator-7f8b9-xk2lp | grep -i "model\|loading"

# Step 2: Increase the readiness probe initialDelaySeconds
# In coordinator-deployment.yaml:
# readinessProbe:
#   httpGet:
#     path: /ready
#     port: 8000
#   initialDelaySeconds: 120   # was 5 — model loading can take minutes
#   periodSeconds: 10

# Step 3: Check the /ready endpoint directly
kubectl exec distllm-coordinator-7f8b9-xk2lp -- curl -s http://localhost:8000/ready
```

**Prevention:** Set `initialDelaySeconds` based on model size. A 70B model can take 60-120 seconds to load.

---

### 7.3 Liveness probe killing healthy pods

**Symptom:** Pods are repeatedly restarted despite working correctly. `kubectl describe pod` shows `Liveness probe failed`.

**Root cause:** The liveness probe timeout is too aggressive. During high load, the `/live` endpoint can't respond within the timeout.

**Fix:**
```yaml
# In coordinator-deployment.yaml:
livenessProbe:
  httpGet:
    path: /live
    port: 8000
  initialDelaySeconds: 30
  periodSeconds: 30          # Check less frequently
  timeoutSeconds: 10         # Allow more time to respond
  failureThreshold: 5        # Require 5 failures before killing
```

**Prevention:** Use a dedicated liveness endpoint that doesn't depend on downstream health. The `/live` endpoint should only check if the process is responsive, not if the cluster is healthy.

---

### 7.4 GPU resource not schedulable

**Symptom:**
```
0/5 nodes are available: 5 Insufficient nvidia.com/gpu.
```

**Root cause:** No nodes in the cluster have GPU resources available, or the NVIDIA device plugin is not installed.

**Fix:**
```bash
# Step 1: Check if the NVIDIA device plugin is running
kubectl get pods -n kube-system | grep nvidia

# Step 2: If not installed, install it
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.3/nvidia-device-plugin.yml

# Step 3: Verify GPU allocatable on nodes
kubectl get nodes -o json | jq '.items[].status.allocatable["nvidia.com/gpu"]'

# Step 4: Add tolerations for GPU taints
# In your deployment spec:
# tolerations:
#   - key: "nvidia.com/gpu"
#     operator: "Exists"
#     effect: "NoSchedule"
```

---

### 7.5 Pod eviction due to memory pressure

**Symptom:**
```
Pod was evicted due to: OOMKilled (memory limit exceeded)
```
or
```
The node was low on resource: memory. Threshold quantity: 100Mi, available: 80Mi
```

**Root cause:** The container exceeded its memory limit, or the node is under memory pressure and Kubernetes is evicting pods.

**Fix:**
```bash
# Step 1: Check actual memory usage
kubectl top pod distllm-coordinator-7f8b9-xk2lp

# Step 2: Increase memory limits
kubectl set resources deployment distllm-coordinator --limits=memory=32Gi

# Step 3: Set memory requests to 50-70% of limits for better scheduling
# resources:
#   requests:
#     memory: 16Gi
#   limits:
#     memory: 32Gi

# Step 4: Enable KV cache quantization to reduce memory
# In ConfigMap:
# DISTLLM_KV_CACHE_QUANT_BITS: "8"
```

---

## 8. Docker Issues

### 8.1 Container can't connect to other containers

**Symptom:**
```
NodeUnreachableError: Node node_1 at node_1:50051 is unreachable
```
or from within a container:
```
ping node_1: Name or service not known
```

**Root cause:** Containers are on different Docker networks, or the service names don't match the container names.

**Fix:**
```bash
# Step 1: Verify all containers are on the same network
docker network ls
docker network inspect distllm-net

# Step 2: If not, connect them
docker network connect distllm-net distllm-node-1

# Step 3: Use service names from docker-compose, not container names
# In docker-compose.yml, the service "node_1" is resolvable as "node_1"
# NOT as "distllm-node-1" (the container_name)

# Step 4: Test DNS resolution between containers
docker exec distllm-coordinator nslookup node_1

# Step 5: Check that ports are exposed
docker port distllm-node-1
```

**Prevention:** Always use the `docker-compose.yml` service names in your DistLLM config. The `container_name` is only for `docker ps` readability.

---

### 8.2 GPU not accessible inside container

**Symptom:**
```
RuntimeError: No CUDA GPUs are available to the container
```
or `nvidia-smi` inside the container returns nothing.

**Root cause:** NVIDIA Container Toolkit is not installed, or the Docker Compose GPU reservation is missing.

**Fix:**
```bash
# Step 1: Verify nvidia-smi works on the host
nvidia-smi

# Step 2: Verify NVIDIA Container Toolkit is installed
nvidia-ctk --version

# Step 3: Test GPU passthrough
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

# Step 4: If that works, check your docker-compose.yml has the GPU reservation:
# deploy:
#   resources:
#     reservations:
#       devices:
#         - driver: nvidia
#           count: 1
#           capabilities: [gpu]

# Step 5: Restart the Docker daemon if needed
sudo systemctl restart docker
```

---

### 8.3 Model cache not shared between containers

**Symptom:** Every container re-downloads the model (slow startup, high bandwidth usage).

**Root cause:** Each container has its own filesystem. Without a shared volume, the HuggingFace cache is per-container.

**Fix:**
```yaml
# In docker-compose.yml, add a shared volume:
services:
  coordinator:
    volumes:
      - model-cache:/root/.cache/huggingface
  worker:
    volumes:
      - model-cache:/root/.cache/huggingface

volumes:
  model-cache:
```

**Prevention:** Always mount `model-cache` as a shared volume in `docker-compose.yml`.

---

### 8.4 Container networking — port conflicts

**Symptom:**
```
Error starting userland proxy: listen tcp4 0.0.0.0:50050: bind: address already in use
```

**Root cause:** Another process (or another DistLLM instance) is already using the port on the host.

**Fix:**
```bash
# Step 1: Find what's using the port
netstat -tlnp | grep 50050
# or on Windows:
netstat -ano | findstr :50050

# Step 2: Kill the conflicting process, or change the host port mapping
# In docker-compose.yml:
# ports:
#   - "15050:50050"  # Map host port 15050 to container port 50050
```

---

## 9. TLS / Certificate Issues

### 9.1 TLS handshake failure

**Symptom:**
```
ssl.SSLError: [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed
```
or
```
grpc._channel._InactiveRpcError: <InactiveRpcError terminated with StatusCode.UNAVAILABLE>
```
(with TLS enabled but certificates misconfigured)

**Root cause:** The certificate is expired, self-signed without the CA being trusted, the hostname doesn't match, or the certificate/key files are in the wrong format.

**Fix:**
```bash
# Step 1: Verify the certificate
openssl x509 -in certs/cert.pem -text -noout
# Check:
#   - "Not After" is in the future
#   - "Subject" or "Subject Alternative Name" matches the hostname

# Step 2: Verify the private key matches the certificate
openssl x509 -noout -modulus -in certs/cert.pem | md5sum
openssl rsa -noout -modulus -in certs/key.pem | md5sum
# Both should produce the same hash

# Step 3: For self-signed certs, add the CA to the trust store
# Linux:
sudo cp certs/ca.pem /usr/local/share/ca-certificates/distllm.crt
sudo update-ca-certificates

# Or tell DistLLM to use the CA file:
# config.yaml
# tls:
#   enabled: true
#   cert_file: /app/certs/cert.pem
#   key_file: /app/certs/key.pem
#   ca_cert_file: /app/certs/ca.pem

# Step 4: Test the TLS connection
openssl s_client -connect coordinator:50050 -servername coordinator
```

**Prevention:** Use Let's Encrypt for production (auto-renewal). Set up a monitoring alert for certificate expiration.

---

### 9.2 Certificate expired

**Symptom:**
```
ssl.SSLCertDateError: certificate has expired
```

**Root cause:** The TLS certificate was not renewed before its expiration date.

**Fix:**
```bash
# Step 1: Renew with certbot (Let's Encrypt)
certbot renew

# Step 2: Reload the coordinator to pick up the new certificate
kill -HUP $(pgrep -f "distllm.*coordinator")
# or restart the container/pod

# Step 3: For self-signed certs, regenerate
distllm security cert create --hostname my-cluster.example.com --days 365
```

**Prevention:** Automate renewal with certbot's systemd timer or cron job. Monitor `distllm_tls_cert_expiry_seconds`.

---

### 9.3 gRPC TLS mutual authentication failure

**Symptom:**
```
grpc.RpcError: StatusCode.UNAVAILABLE: connection error: tls: bad certificate
```

**Root cause:** Mutual TLS (mTLS) is enabled but the client node doesn't present a valid client certificate, or the server doesn't trust the client's CA.

**Fix:**
```bash
# Step 1: Verify mTLS configuration on the coordinator
# config.yaml
# tls:
#   enabled: true
#   cert_file: /app/certs/server-cert.pem
#   key_file: /app/certs/server-key.pem
#   ca_cert_file: /app/certs/ca.pem
#   require_client_cert: true

# Step 2: Ensure each worker has a client certificate signed by the same CA
# certs/
#   ca.pem            # Shared CA
#   server-cert.pem   # Coordinator's server cert
#   server-key.pem    # Coordinator's private key
#   client-cert.pem   # Worker's client cert
#   client-key.pem    # Worker's private key

# Step 3: Verify the client cert is signed by the CA
openssl verify -CAfile certs/ca.pem certs/client-cert.pem
```

---

## 10. Federation Issues

### 10.1 Split-brain between federated clusters

**Symptom:** Two clusters independently accept requests for the same model, leading to inconsistent results. Federation status shows clusters as disconnected despite both being healthy.

**Root cause:** The network link between clusters is down, seed nodes are unreachable, or the gossip protocol has lost quorum.

**Fix:**
```bash
# Step 1: Check federation status
distllm cluster status --federation

# Step 2: Verify seed nodes are reachable from both clusters
curl http://cluster-a-seed:50060/health
curl http://cluster-b-seed:50060/health

# Step 3: Check the gossip protocol state
curl http://coordinator:8000/admin/federation/gossip-state

# Step 4: Force re-join if a cluster is isolated
distllm cluster federation rejoin --seed cluster-a:50060

# Step 5: If split-brain occurred, pick a leader and reconcile
distllm cluster federation resolve --leader cluster-a
```

**Prevention:**
- Use at least 3 seed nodes for fault tolerance.
- Monitor `distllm_federation_peer_count` and alert if it drops.
- Set `federation.heartbeat_interval_seconds` to a value appropriate for your network latency.

---

### 10.2 Peer discovery not working

**Symptom:**
```
FederationWarning: No peers discovered after 60s
```
or `distllm cluster status --federation` shows only the local cluster.

**Root cause:** DNS SRV records are not configured, the federation port is blocked, or seed node addresses are wrong.

**Fix:**
```bash
# Step 1: Verify seed node configuration
# config.yaml
# federation:
#   enabled: true
#   seed_nodes:
#     - "cluster-a:50060"
#     - "cluster-b:50060"
#     - "cluster-c:50060"

# Step 2: Test connectivity to seed nodes
nc -zv cluster-a 50060
telnet cluster-a 50060

# Step 3: If using DNS discovery, verify SRV records
dig _grpc._tcp.federation.example.com SRV

# Step 4: Check firewall rules
# Federation port (default: 50060) must be open between clusters
sudo iptables -L -n | grep 50060

# Step 5: Check federation logs
distllm system logs --component federation --tail 100
```

**Prevention:**
- Use a service mesh (Istio, Linkerd) for automatic mTLS and service discovery.
- Pre-configure seed nodes in a shared ConfigMap.

---

### 10.3 Cross-cluster model routing inconsistency

**Symptom:** Requests routed to a remote cluster fail or return different results than local execution.

**Root cause:** The remote cluster is running a different model version, has different quantization settings, or the federation routing table is stale.

**Fix:**
```bash
# Step 1: Verify model versions match across clusters
distllm cluster status --federation | grep model

# Step 2: Check the routing table
curl http://coordinator:8000/admin/federation/routes

# Step 3: Force a routing table refresh
distllm cluster federation refresh-routes

# Step 4: If models are intentionally different, use model-specific routing
# config.yaml
# federation:
#   routing:
#     meta-llama/Llama-3.1-70B: [cluster-a, cluster-b]
#     meta-llama/Llama-3.1-8B: [cluster-c]
```

---

## 11. Diagnostic Commands

### System Health

```bash
# Full system health check
distllm system doctor

# Cluster status (all nodes)
distllm cluster status

# Federation status
distllm cluster status --federation

# Configuration validation
distllm config validate
```

### Logs

```bash
# Recent logs
distllm system logs --tail 100

# Logs for a specific node
distllm system logs --node node_1 --tail 100

# Logs for a specific component
distllm system logs --component coordinator --tail 100

# Follow logs in real-time
distllm system logs --follow
```

### Performance

```bash
# Run benchmarks
distllm benchmark run --model llama-3-8b

# Compare with baseline
distllm benchmark compare --baseline results/baseline.json

# Profile a specific scenario
distllm benchmark profile --scenario chat --duration 60
```

### GPU

```bash
# GPU status
nvidia-smi

# Continuous monitoring (updates every 1 second)
nvidia-smi -l 1

# Query specific fields
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv -l 5
```

### Network

```bash
# gRPC health check
grpcurl -plaintext coordinator:50050 grpc.health.v1.Health/Check

# REST API health
curl http://coordinator:8000/health

# Metrics
curl http://coordinator:8000/v1/metrics

# Network bandwidth test
iperf3 -c node-1 -t 10
```

### Docker

```bash
# Container status
docker ps -a --filter "label=com.docker.compose.project=distllm"

# Container logs
docker logs distllm-coordinator --tail 100

# Container resource usage
docker stats --no-stream
```

### Kubernetes

```bash
# Pod status
kubectl get pods -l app.kubernetes.io/name=distributed-llm

# Pod logs
kubectl logs -l app.kubernetes.io/component=coordinator --tail=100

# Pod events
kubectl describe pod distllm-coordinator-xxxxx

# Resource usage
kubectl top pods -l app.kubernetes.io/name=distributed-llm

# GPU allocation
kubectl get nodes -o json | jq '.items[].status.allocatable["nvidia.com/gpu"]'
```

---

## Getting Help

1. Run `distllm system doctor` for automated diagnostics.
2. Search [GitHub Issues](https://github.com/distributed-llm/distributed-llm/issues).
3. Ask in [GitHub Discussions](https://github.com/distributed-llm/distributed-llm/discussions).
4. Join [Discord](https://discord.gg/distllm).
5. When filing an issue, include:
   - `distllm --version`
   - `distllm system doctor` output
   - `nvidia-smi` output
   - Relevant logs with `DISTLLM_LOG_LEVEL=DEBUG`
