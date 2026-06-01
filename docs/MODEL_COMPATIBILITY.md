# Model Compatibility Matrix

## Supported Models

### Meta Llama

| Model | Size | Backend | Quantization | Status | Notes |
|-------|------|---------|--------------|--------|-------|
| Llama-3.1-8B | 8B | PyTorch, vLLM | FP16, INT8, INT4 | ✅ Supported | Best for single GPU |
| Llama-3.1-70B | 70B | PyTorch, vLLM | FP16, INT8, INT4 | ✅ Supported | Requires 2+ GPUs |
| Llama-3.1-405B | 405B | PyTorch | FP16, INT8 | ✅ Supported | Requires 8+ GPUs |
| Llama-3-8B-Instruct | 8B | PyTorch, vLLM | FP16, INT8, INT4 | ✅ Supported | Chat model |
| Llama-3-70B-Instruct | 70B | PyTorch, vLLM | FP16, INT8, INT4 | ✅ Supported | Chat model |
| CodeLlama-34B | 34B | PyTorch | FP16, INT8 | ✅ Supported | Code generation |

### Mistral

| Model | Size | Backend | Quantization | Status | Notes |
|-------|------|---------|--------------|--------|-------|
| Mistral-7B-v0.3 | 7B | PyTorch, vLLM | FP16, INT8, INT4 | ✅ Supported | |
| Mixtral-8x7B | 47B | PyTorch | FP16, INT8 | ✅ Supported | MoE architecture |
| Mixtral-8x22B | 141B | PyTorch | FP16, INT8 | ✅ Supported | Large MoE |

### Qwen

| Model | Size | Backend | Quantization | Status | Notes |
|-------|------|---------|--------------|--------|-------|
| Qwen2-7B | 7B | PyTorch, vLLM | FP16, INT8, INT4 | ✅ Supported | |
| Qwen2-72B | 72B | PyTorch | FP16, INT8 | ✅ Supported | |
| Qwen2.5-72B | 72B | PyTorch | FP16, INT8 | ✅ Supported | Latest |

### DeepSeek

| Model | Size | Backend | Quantization | Status | Notes |
|-------|------|---------|--------------|--------|-------|
| DeepSeek-V2 | 236B | PyTorch | FP16, INT8 | ✅ Supported | MoE architecture |
| DeepSeek-Coder-V2 | 236B | PyTorch | FP16, INT8 | ✅ Supported | Code-focused |

### Falcon

| Model | Size | Backend | Quantization | Status | Notes |
|-------|------|---------|--------------|--------|-------|
| Falcon-7B | 7B | PyTorch | FP16, INT8 | ✅ Supported | |
| Falcon-40B | 40B | PyTorch | FP16, INT8 | ✅ Supported | |

### Phi

| Model | Size | Backend | Quantization | Status | Notes |
|-------|------|---------|--------------|--------|-------|
| Phi-3-mini | 3.8B | PyTorch | FP16, INT8, INT4 | ✅ Supported | Small but capable |
| Phi-3-medium | 14B | PyTorch | FP16, INT8 | ✅ Supported | |

### Gemma

| Model | Size | Backend | Quantization | Status | Notes |
|-------|------|---------|--------------|--------|-------|
| Gemma-2-9B | 9B | PyTorch | FP16, INT8 | ✅ Supported | |
| Gemma-2-27B | 27B | PyTorch | FP16, INT8 | ✅ Supported | |

---

## Backend Compatibility

| Backend | Models Supported | Best For | Notes |
|---------|-----------------|----------|-------|
| **PyTorch** | All | Development, research | Full flexibility |
| **vLLM** | Llama, Mistral, Qwen | Production serving | PagedAttention, high throughput |
| **llama.cpp** | GGUF models | CPU inference, edge | Excellent quantization |
| **ExLlamaV2** | GPTQ models | Consumer GPUs | Fast INT4 inference |
| **ONNX** | Exported models | Cross-platform | Limited model support |
| **TensorRT-LLM** | NVIDIA-optimized | NVIDIA GPUs | Best single-GPU perf |

---

## Quantization Support

| Method | Bits | Quality | Speed | Memory Savings | Best For |
|--------|------|---------|-------|----------------|----------|
| **FP16** | 16 | Best | Baseline | 0% | Development |
| **INT8** | 8 | Good | 1.5x | 50% | Production |
| **INT4 (GPTQ)** | 4 | Fair | 2x | 75% | Consumer GPUs |
| **INT4 (AWQ)** | 4 | Good | 2x | 75% | Production |
| **GGUF Q4_K_M** | 4 | Good | 2x | 75% | CPU inference |
| **FP8** | 8 | Good | 1.5x | 50% | H100 GPUs |

---

## Hardware Requirements

### Minimum Requirements

| Model Size | GPU Memory | Recommended GPU | Nodes |
|------------|-----------|-----------------|-------|
| 1-3B | 4 GB | RTX 3060, RTX 4060 | 1 |
| 7-8B | 8 GB | RTX 3070, RTX 4070 | 1 |
| 13B | 16 GB | RTX 4080, A4000 | 1-2 |
| 34B | 24 GB | RTX 4090, A5000 | 1-2 |
| 70B | 48 GB | A6000, 2x RTX 4090 | 2-4 |
| 405B | 160 GB | 4x A100 80GB | 4-8 |

### Network Requirements

| Setup | Bandwidth | Latency | Use Case |
|-------|-----------|---------|----------|
| Same machine | 600+ GB/s | <1ms | NVLink multi-GPU |
| Local LAN | 1-10 Gbps | <1ms | Home/office cluster |
| Datacenter | 10-100 Gbps | <5ms | Enterprise cluster |
| WAN (internet) | 100 Mbps-1 Gbps | 50-200ms | Distributed/friends |

---

## Adding New Models

To add support for a new model architecture:

1. **Check HuggingFace**: Model must be available on HuggingFace Hub
2. **Test loading**: `distllm model load <model-name>`
3. **Verify partitioning**: `distllm model info <model-name>`
4. **Run inference**: `distllm chat --model <model-name>`
5. **Submit PR**: Add to this matrix and update README

### Common Issues

- **Architecture not supported**: May need to add layer mapping in `ModelPartitioner`
- **Trust remote code**: Some models require `--trust-remote-code`
- **Custom tokenizer**: Ensure tokenizer loads correctly
- **Gated models**: Set `HUGGING_FACE_HUB_TOKEN` for gated models
