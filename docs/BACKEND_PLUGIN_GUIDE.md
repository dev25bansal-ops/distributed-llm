# Backend Plugin Guide

How to add custom inference backends to DistLLM.

---

## Architecture

DistLLM uses a plugin-based backend architecture:

```
BackendAdapter (protocol.py)
    ├── PyTorchBackend (pytorch_backend.py)
    ├── VLLMBackend (vllm_backend.py)
    ├── LlamaCppBackend (llamacpp_backend.py)
    ├── ExLlamaBackend (exllama_backend.py) [experimental]
    ├── TensorRTBackend (tensorrt_backend.py) [experimental]
    ├── ONNXBackend (onnx_backend.py)
    └── YourCustomBackend (your_file.py)
```

All backends implement `BackendAdapter` and register via `BackendRegistry`.

---

## Creating a Custom Backend

### 1. Implement BackendAdapter

```python
from distllm.backends.protocol import BackendAdapter

class MyCustomBackend(BackendAdapter):
    """My custom inference engine."""

    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self._model = None

    def load_model(self) -> None:
        """Initialize the engine and load weights."""
        # Your model loading code
        self._model = load_my_model(self.model_name)

    def forward(self, input_ids=None, hidden_states=None, **kwargs):
        """Run a forward pass."""
        # Your inference code
        output = self._model.forward(input_ids)
        return output, []  # (logits, kv_cache)

    def shutdown(self) -> None:
        """Release resources."""
        del self._model

    @classmethod
    def display_name(cls) -> str:
        return "My Custom Backend"

    @classmethod
    def is_available(cls) -> bool:
        try:
            import my_custom_lib
            return True
        except ImportError:
            return False

    @classmethod
    def priority_for(cls, device_type: str) -> int:
        if device_type == "cuda":
            return 8  # High priority on CUDA
        return 0  # Not supported on other devices
```

### 2. Register the Backend

```python
from distllm.backends.registry import BackendRegistry

BackendRegistry.register(MyCustomBackend, name="my-custom")
```

### 3. Configure in YAML

```yaml
# config.yaml
model:
  backend: my-custom
  model_name: my-model
```

---

## Supported Backends

| Backend | Status | Best For | Priority (CUDA) |
|---------|--------|----------|----------------|
| **PyTorch** | Stable | Development, custom models | 5 |
| **vLLM** | Stable | Production throughput | 10 |
| **llama.cpp** | Stable | CPU/edge, GGUF models | 3 (CUDA), 8 (CPU) |
| **SGLang** | Experimental | High-throughput serving | 8 |
| **TensorRT-LLM** | Experimental | Maximum latency | 9 |
| **ExLlamaV2** | Experimental | GPTQ quantized models | 7 |
| **ONNX** | Experimental | Cross-platform | 4 |

---

## Backend Selection

DistLLM auto-selects the best backend based on:

1. **Explicit config**: `model.backend: vllm` in config.yaml
2. **Preferred backend**: `--backend vllm` CLI flag
3. **Auto-detection**: Highest `priority_for(device_type)` among available backends

```python
from distllm.backends.registry import select_backend

# Auto-select best backend for CUDA
Backend = select_backend(device_type="cuda")

# Prefer vLLM but fall back
Backend = select_backend(preferred_backend="vllm")
```

---

## Adding a New Backend to the Registry

### Option 1: Direct Registration

```python
# In your plugin file
from distllm.backends.registry import BackendRegistry
from my_package import MyBackend

BackendRegistry.register(MyBackend, name="my-backend")
```

### Option 2: Entry Point (pip installable)

```toml
# pyproject.toml
[project.entry-points."distllm.backends"]
my-backend = "my_package.backends:MyBackend"
```

### Option 3: Config File

```yaml
# config.yaml
backends:
  plugins:
    - name: my-backend
      module: my_package.backends
      class: MyBackend
```

---

## Testing Your Backend

```python
from distllm.backends.registry import get_backend

Backend = get_backend("my-backend")
assert Backend is not None, "Backend not registered"
assert Backend.is_available(), "Dependencies not installed"

adapter = Backend(model_name="test-model")
adapter.load_model()

# Test forward pass
import torch
input_ids = torch.tensor([[1, 2, 3]])
logits, kv = adapter.forward(input_ids=input_ids)
assert logits.shape[0] == 1

adapter.shutdown()
```
