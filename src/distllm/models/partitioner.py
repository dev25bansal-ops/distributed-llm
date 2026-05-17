"""Model loading and partitioning for distributed LLM inference."""

import torch
import torch.nn as nn
import inspect
from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from loguru import logger

from distllm.core.kv_cache import KVCache
from distllm.config.loader import QuantizationConfig
from distllm.errors import ModelLoadError
from distllm.core.architecture_registry import (
    get_architecture_info,
    get_trusted_models,
)

DTYPE_MAP = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}


def _find_attr(model, candidates: list[str]):
    """Find the first matching attribute on a model object."""
    for attr in candidates:
        if hasattr(model, attr):
            return getattr(model, attr)
    return None


def _get_base_prefix(model) -> str:
    """Get the attribute name used for the base model wrapper."""
    for attr in ['model', 'transformer', 'encoder']:
        if hasattr(model, attr):
            return attr
    return ""


def build_quantization_config(quant_config: QuantizationConfig | None) -> "BitsAndBytesConfig | None":
    """Build a BitsAndBytesConfig from QuantizationConfig dataclass."""
    if quant_config.method == "none":
        return None
    from transformers import BitsAndBytesConfig
    if quant_config.method == "bnb_8bit":
        return BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=quant_config.llm_int8_threshold,
        )
    if quant_config.method == "bnb_4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=DTYPE_MAP.get(quant_config.bnb_4bit_compute_dtype, torch.float16),
            bnb_4bit_quant_type=quant_config.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=quant_config.bnb_4bit_use_double_quant,
        )
    # GPTQ and FP8 require more complex setup; not supported in this simplified version
    if quant_config.method in ("gptq", "fp8"):
        logger.warning(f"Quantization method '{quant_config.method}' is not fully supported, falling back to no quantization")
    return None

# Models known to require trust_remote_code=True for legitimate reasons
# Auto-populated from the ArchitectureRegistry; extend with additional entries as needed.
_TRUSTED_FROM_REGISTRY = get_trusted_models()
TRUSTED_MODELS_ALLOWLIST: set[str] = _TRUSTED_FROM_REGISTRY | {
    "baichuan", "baichuan2",
    "chatglm", "chatglm2", "chatglm3",
    "internlm", "internlm2",
    "stablelm",
    "jina",
}


def _should_trust_remote_code(model_name: str, trust_remote_code: bool | None = None) -> bool:
    """Determine whether to trust remote code for a model.

    Args:
        model_name: HuggingFace model identifier
        trust_remote_code: Explicit override. If None, uses allowlist logic.

    Returns:
        True if remote code should be trusted, False otherwise.
    """
    if trust_remote_code is not None:
        return trust_remote_code

    # Check ArchitectureRegistry first (covers DeepSeek, Phi-3, etc.)
    from distllm.core.architecture_registry import lookup_by_model_name
    arch_info = lookup_by_model_name(model_name)
    if arch_info is not None and arch_info.trust_remote_code:
        return True

    # Extract the model name part (last segment of HF repo path)
    model_lower = model_name.lower().split("/")[-1]

    # Extract model family (prefix before first - or . separator)
    # e.g., "qwen2-7b" -> "qwen2", "my-qwen-exploit" -> "my"
    family = model_lower.split("-")[0].split(".")[0]

    # Match model family against allowlist to prevent false positives
    # (e.g., "my-qwen-exploit" has family "my" which won't match "qwen")
    for trusted in TRUSTED_MODELS_ALLOWLIST:
        if model_lower == trusted or family == trusted:
            return True
    return False


class ModelPartitioner:
    """Splits a model's layers across multiple nodes for pipeline parallelism."""

    def __init__(self, model_name: str, device: str = "auto", dtype: str = "float16", trust_remote_code: bool | None = None, quantization_config: QuantizationConfig | None = None, compression_config=None):
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self.trust_remote_code = trust_remote_code
        self.quantization_config = quantization_config
        self.compression_config = compression_config
        self.config = None
        self.tokenizer = None

        # Model components (only loaded subsets)
        self.embed_tokens = None  # Only on first node
        self.position_embeds = None  # Positional encoding (wpe), only on first node
        self.layers = nn.ModuleList()  # Assigned layers
        self.final_norm = None  # Only on last node
        self.lm_head = None  # Only on last node
        self.rotary_emb = None  # RoPE module for Qwen2, Llama, Mistral

        # Pipeline role
        self.is_first_node = False
        self.is_last_node = False
        self.assigned_layers: list[int] = []

        # Cached layer forward signatures (avoid inspect on every call)
        self._layer_params: list[set] = []

        # Cached causal attention mask (built once, extended incrementally)
        self._causal_mask: torch.Tensor | None = None
        self._causal_mask_device: str = ""

    def load_full_model(self) -> None:
        """Load the complete model (for single-node mode)."""
        logger.info(f"Loading full model: {self.model_name}")
        trust = _should_trust_remote_code(self.model_name, self.trust_remote_code)
        self.config = AutoConfig.from_pretrained(self.model_name, trust_remote_code=trust)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=trust)

        torch_dtype = DTYPE_MAP.get(self.dtype, torch.float16)
        quant_config = build_quantization_config(self.quantization_config) if self.quantization_config else None

        model_kwargs = {
            "config": self.config,
            "torch_dtype": torch_dtype,
            "device_map": "auto" if self.device == "auto" else self.device,
            "trust_remote_code": trust,
            "low_cpu_mem_usage": True,
        }
        if quant_config is not None:
            model_kwargs["quantization_config"] = quant_config

        self.full_model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)

        # Apply compression pipeline
        if self.compression_config and self.compression_config.enabled:
            from distllm.core.compression_pipeline import CompressionPipeline
            from distllm.core.compression_config import CompressionConfig, CompressionMethod
            comp_config = CompressionConfig(
                method=CompressionMethod(self.compression_config.method),
                enabled=self.compression_config.enabled,
                target_bits=self.compression_config.target_bits,
                pruning_ratio=self.compression_config.pruning_ratio,
                distillation_teacher=self.compression_config.distillation_teacher,
                calibration_samples=self.compression_config.calibration_samples,
                pruning_targets=self.compression_config.pruning_targets,
            )
            pipeline = CompressionPipeline(comp_config)
            self.full_model = pipeline.apply(self.full_model, tokenizer=self.tokenizer)

        self.full_model.eval()

        # Extract components for pipeline compatibility
        base_model = _find_attr(self.full_model, ['model', 'transformer', 'encoder']) or self.full_model

        # Embedding layer
        embed_layer = _find_attr(base_model, ['embed_tokens', 'wte', 'word_embeddings'])
        if embed_layer is None:
            decoder = getattr(base_model, 'decoder', None)
            if decoder:
                embed_layer = _find_attr(decoder, ['embed_tokens', 'wte', 'word_embeddings'])
        if embed_layer is not None:
            self.embed_tokens = embed_layer
            self.is_first_node = True

            # Positional encoding (GPT-2/GPT-Neo use 'wpe')
            for attr in ['wpe', 'embed_positions']:
                if hasattr(base_model, attr):
                    self.position_embeds = getattr(base_model, attr)
                    break

        # Transformer layers
        layers_attr = _find_attr(base_model, ['layers', 'block', 'h'])
        if layers_attr is not None:
            self.layers = layers_attr
            self.assigned_layers = list(range(len(layers_attr)))

        # Final norm and LM head
        self.final_norm = _find_attr(base_model, ['norm', 'final_layer_norm', 'ln_f'])
        if hasattr(self.full_model, 'lm_head'):
            self.lm_head = self.full_model.lm_head
            self.is_last_node = True

        # Rotary embedding for RoPE-based models
        for attr in ['rotary_emb', 'rotary']:
            if hasattr(base_model, attr):
                self.rotary_emb = getattr(base_model, attr)
                break

        # Cache layer forward signatures for fast forward() dispatch
        self._layer_params = []
        for layer in self.layers:
            sig = inspect.signature(layer.forward)
            self._layer_params.append(set(sig.parameters.keys()))

        logger.info(f"Full model loaded: {self.model_name}")

    def load_layer_subset(self, start_layer: int, end_layer: int, total_layers: int, device: str | None = None) -> None:
        """Load only a subset of layers (start_layer to end_layer inclusive)."""
        logger.info(f"Loading layers {start_layer}-{end_layer} of {total_layers} for {self.model_name}")
        trust = _should_trust_remote_code(self.model_name, self.trust_remote_code)
        self.config = AutoConfig.from_pretrained(self.model_name, trust_remote_code=trust)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=trust)

        device = device or self.device
        torch_dtype = DTYPE_MAP.get(self.dtype, torch.float16)
        quant_config = build_quantization_config(self.quantization_config) if self.quantization_config else None

        is_first = (start_layer == 0)
        is_last = (end_layer >= total_layers - 1)

        model_kwargs = {
            "config": self.config,
            "torch_dtype": torch_dtype,
            "device_map": "meta",
            "trust_remote_code": trust,
            "low_cpu_mem_usage": True,
        }
        if quant_config is not None:
            model_kwargs["quantization_config"] = quant_config

        temp_model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)

        temp_base = None
        for attr in ['model', 'transformer', 'encoder']:
            if hasattr(temp_model, attr):
                temp_base = getattr(temp_model, attr)
                break
        if temp_base is None:
            temp_base = temp_model

        layers_attr = None
        for attr in ['layers', 'block', 'h']:
            if hasattr(temp_base, attr):
                layers_attr = attr
                break

        if layers_attr is None:
            raise ModelLoadError(self.model_name, "Cannot find transformer layers")

        device_map = {}
        base_prefix = _get_base_prefix(temp_model)

        if is_first:
            embed_attr = _find_attr(temp_base, ['embed_tokens', 'wte', 'word_embeddings'])
            if embed_attr is not None:
                # Find which attribute name matched
                for attr in ['embed_tokens', 'wte', 'word_embeddings']:
                    if hasattr(temp_base, attr):
                        device_map[f"{base_prefix}.{attr}"] = device
                        break
            for attr in ['wpe', 'embed_positions']:
                if hasattr(temp_base, attr):
                    device_map[f"{base_prefix}.{attr}"] = device
                    break

        for i in range(total_layers):
            layer_device = device if start_layer <= i <= end_layer else "meta"
            device_map[f"{base_prefix}.{layers_attr}.{i}"] = layer_device

        if is_last:
            for attr in ['norm', 'final_layer_norm', 'ln_f']:
                if hasattr(temp_base, attr):
                    device_map[f"{base_prefix}.{attr}"] = device
                    break
            device_map[f"lm_head"] = device

        model_kwargs_final = {
            "config": self.config,
            "torch_dtype": torch_dtype,
            "device_map": device_map,
            "trust_remote_code": trust,
            "low_cpu_mem_usage": True,
        }
        if quant_config is not None:
            model_kwargs_final["quantization_config"] = quant_config

        model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs_final)
        model.eval()

        self._extract_subset(model, start_layer, end_layer, total_layers, device)

        del temp_model, model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info(f"Loaded layers {start_layer}-{end_layer} on {device}")

    def _extract_subset(self, full_model, start_layer: int, end_layer: int, total_layers: int, device: str) -> None:
        """Extract specific layers and components from the full model."""
        base_model = None
        for attr in ['model', 'transformer', 'encoder']:
            if hasattr(full_model, attr):
                base_model = getattr(full_model, attr)
                break
        if base_model is None:
            base_model = full_model

        self.is_first_node = (start_layer == 0)
        if self.is_first_node:
            embed_layer = _find_attr(base_model, ['embed_tokens', 'wte', 'word_embeddings'])
            if embed_layer is None:
                decoder = getattr(base_model, 'decoder', None)
                if decoder:
                    embed_layer = _find_attr(decoder, ['embed_tokens', 'wte', 'word_embeddings'])
            if embed_layer is not None:
                self.embed_tokens = embed_layer.to(device)
                logger.info("Loaded embedding layer")

            for attr in ['wpe', 'embed_positions']:
                if hasattr(base_model, attr):
                    self.position_embeds = getattr(base_model, attr).to(device)
                    logger.info(f"Loaded positional encoding ({attr})")
                    break

        layers_attr = _find_attr(base_model, ['layers', 'block', 'h'])

        if layers_attr is None:
            raise ModelLoadError(self.model_name, "Cannot find transformer layers")

        self.layers = nn.ModuleList()
        self.assigned_layers = []
        for i in range(start_layer, min(end_layer + 1, total_layers)):
            self.layers.append(layers_attr[i].to(device))
            self.assigned_layers.append(i)

        logger.info(f"Loaded {len(self.layers)} transformer layers: {self.assigned_layers}")

        self.is_last_node = (end_layer >= total_layers - 1)
        if self.is_last_node:
            self.final_norm = _find_attr(base_model, ['norm', 'final_layer_norm', 'ln_f'])
            if self.final_norm is not None:
                self.final_norm = self.final_norm.to(device)
                logger.info("Loaded final layer norm")

            if hasattr(full_model, 'lm_head'):
                self.lm_head = full_model.lm_head.to(device)
                logger.info("Loaded LM head")

        self._layer_params = []
        for layer in self.layers:
            sig = inspect.signature(layer.forward)
            self._layer_params.append(set(sig.parameters.keys()))

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]] | None]:
        """Run forward pass through assigned layers."""
        new_past_key_values = []
        seq_len = hidden_states.shape[1]

        if position_ids is None:
            past_len = 0
            if past_key_values and len(past_key_values) > 0:
                past_len = past_key_values[0][0].shape[-2]
            position_ids = torch.arange(past_len, past_len + seq_len, device=hidden_states.device).unsqueeze(0)

        if attention_mask is None:
            attention_mask = torch.ones(
                hidden_states.shape[0], hidden_states.shape[1],
                device=hidden_states.device, dtype=torch.long
            )

        batch_size = hidden_states.shape[0]

        past_len = 0
        if past_key_values and len(past_key_values) > 0:
            past_len = past_key_values[0][0].shape[-2]
        total_len = past_len + seq_len

        device = hidden_states.device
        # Use 2D causal mask (not expanded to 4D) to save memory.
        # Each layer will expand it internally if needed, or use FlashAttention.
        if self._causal_mask is None or self._causal_mask.shape[0] < total_len or self._causal_mask_device != str(device):
            self._causal_mask = torch.tril(
                torch.ones(max(total_len, 4096), max(total_len, 4096), device=device, dtype=torch.bool)
            )
            self._causal_mask_device = str(device)
        attention_mask_4d = self._causal_mask[:total_len, :total_len]

        for i, layer in enumerate(self.layers):
            layer_past = None
            if past_key_values and i < len(past_key_values):
                layer_past = past_key_values[i]

            params = self._layer_params[i] if i < len(self._layer_params) else set()

            layer_kwargs = {"hidden_states": hidden_states}

            if "attention_mask" in params:
                layer_kwargs["attention_mask"] = attention_mask_4d
            if "position_ids" in params and position_ids is not None:
                layer_kwargs["position_ids"] = position_ids
            if "past_key_value" in params and layer_past is not None:
                layer_kwargs["past_key_value"] = layer_past
            elif "past_key_values" in params and layer_past is not None:
                layer_kwargs["past_key_values"] = layer_past
            if "use_cache" in params:
                layer_kwargs["use_cache"] = True

            if "position_embeddings" in params and self.rotary_emb is not None:
                past_seq_len = 0
                if past_key_values and len(past_key_values) > 0:
                    past_seq_len = past_key_values[0][0].shape[-2]
                total_seq_len = past_seq_len + seq_len
                full_position_ids = torch.arange(total_seq_len, device=hidden_states.device).unsqueeze(0)
                current_position_ids = full_position_ids[:, past_seq_len:]
                cos, sin = self.rotary_emb(hidden_states, current_position_ids)
                layer_kwargs["position_embeddings"] = (cos, sin)

            outputs = layer(**layer_kwargs)

            if isinstance(outputs, tuple):
                hidden_states = outputs[0]
                for item in outputs[1:]:
                    if isinstance(item, tuple) and len(item) == 2:
                        new_past_key_values.append(item)
                        break
                    if hasattr(item, 'past_key_values') and item.past_key_values is not None:
                        new_past_key_values.extend(item.past_key_values)
                        break
            else:
                hidden_states = outputs

        return hidden_states, new_past_key_values

    def embed_input(self, input_ids: torch.Tensor, position_offset: int = 0) -> torch.Tensor:
        """Convert token IDs to embeddings with positional encoding (first node only)."""
        if self.embed_tokens is None:
            raise RuntimeError("No embedding layer on this node")
        hidden = self.embed_tokens(input_ids)
        if self.position_embeds is not None:
            seq_len = input_ids.shape[-1]
            position_ids = torch.arange(position_offset, position_offset + seq_len, device=input_ids.device).unsqueeze(0)
            hidden = hidden + self.position_embeds(position_ids)
        return hidden

    def get_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute logits from hidden states (last node only)."""
        if not self.is_last_node:
            raise RuntimeError("This node is not the last node in the pipeline")

        if self.final_norm is not None:
            hidden_states = self.final_norm(hidden_states)

        if self.lm_head is not None:
            return self.lm_head(hidden_states)

        raise RuntimeError("No LM head available")

    def get_model_config(self) -> dict:
        """Get model configuration parameters."""
        if self.config is None:
            trust = _should_trust_remote_code(self.model_name, self.trust_remote_code)
            self.config = AutoConfig.from_pretrained(self.model_name, trust_remote_code=trust)
        return {
            "hidden_size": self.config.hidden_size,
            "num_attention_heads": self.config.num_attention_heads,
            "num_hidden_layers": self.config.num_hidden_layers,
            "vocab_size": self.config.vocab_size,
            "num_key_value_heads": getattr(self.config, 'num_key_value_heads', self.config.num_attention_heads),
        }


def partition_model_across_nodes(model_name: str, num_nodes: int, trust_remote_code: bool | None = None) -> list[tuple[int, int]]:
    """Calculate layer assignments for each node using equal split."""
    trust = _should_trust_remote_code(model_name, trust_remote_code)
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=trust)
    total_layers = config.num_hidden_layers

    layers_per_node = total_layers // num_nodes
    remainder = total_layers % num_nodes

    assignments = []
    start = 0
    for i in range(num_nodes):
        extra = 1 if i < remainder else 0
        end = start + layers_per_node + extra - 1
        assignments.append((start, end))
        start = end + 1

    return assignments


def partition_model_gpu_aware(
    node_gpus: dict[str, list],
    model_name: str,
    total_layers: int,
    trust_remote_code: bool | None = None,
    safety_margin: float = 0.1,
) -> dict[str, tuple[int, int]]:
    """Calculate VRAM-aware layer assignments for each node.

    Args:
        node_gpus: dict mapping node_id to list of GPUInfo objects
        model_name: HuggingFace model identifier
        total_layers: total number of transformer layers
        trust_remote_code: whether to trust remote code
        safety_margin: fraction of VRAM to leave free (default 0.1 = 10%)

    Returns:
        dict mapping node_id to (start_layer, end_layer) tuple
    """
    from distllm.core.gpu_profiler import GPUProfiler

    profiler = GPUProfiler()

    # Estimate per-layer VRAM
    per_layer_vram = profiler.estimate_layer_vram(
        model_name, 0, total_layers, trust_remote_code
    )

    if per_layer_vram == 0:
        # Fallback to equal partitioning if estimation fails
        logger.warning("VRAM estimation failed, falling back to equal partitioning")
        assignments = partition_model_across_nodes(model_name, len(node_gpus), trust_remote_code)
        return {node_id: assignments[i] for i, node_id in enumerate(node_gpus)}

    # Calculate available VRAM per node (apply safety margin)
    node_vram = {}
    for node_id, gpus in node_gpus.items():
        total_free = sum(gpu.free_memory for gpu in gpus)
        available = int(total_free * (1 - safety_margin))
        node_vram[node_id] = available

    # Assign layers proportional to available VRAM
    total_available = sum(node_vram.values())
    if total_available == 0:
        logger.warning("No available VRAM, falling back to equal partitioning")
        assignments = partition_model_across_nodes(model_name, len(node_gpus), trust_remote_code)
        return {node_id: assignments[i] for i, node_id in enumerate(node_gpus)}

    # Initial assignment: floor(vram_i / per_layer_vram)
    node_layers = {}
    assigned_total = 0
    node_ids = sorted(node_gpus.keys())

    for node_id in node_ids:
        raw_layers = node_vram[node_id] // per_layer_vram
        node_layers[node_id] = max(1, raw_layers)  # at least 1 layer
        assigned_total += node_layers[node_id]

    # Normalize to match total_layers exactly
    if assigned_total != total_layers:
        # Scale proportionally
        scale = total_layers / assigned_total
        scaled = {}
        for node_id in node_ids:
            scaled[node_id] = max(1, int(node_layers[node_id] * scale))
        node_layers = scaled

    # Distribute remainder
    assigned_total = sum(node_layers.values())
    remainder = total_layers - assigned_total
    if remainder > 0:
        # Give extra layers to nodes with most VRAM headroom
        sorted_by_vram = sorted(node_ids, key=lambda n: node_vram[n], reverse=True)
        for i in range(remainder):
            node_layers[sorted_by_vram[i % len(sorted_by_vram)]] += 1
    elif remainder < 0:
        # Remove layers from nodes with least VRAM
        sorted_by_vram = sorted(node_ids, key=lambda n: node_vram[n])
        for i in range(abs(remainder)):
            node_layers[sorted_by_vram[i % len(sorted_by_vram)]] = max(1, node_layers[sorted_by_vram[i % len(sorted_by_vram)]] - 1)

    # Convert to (start, end) tuples
    result = {}
    start = 0
    for node_id in node_ids:
        count = node_layers[node_id]
        end = start + count - 1
        result[node_id] = (start, end)
        start = end + 1

    logger.info(f"GPU-aware partitioning for {model_name}: {result}")
    return result


def get_model_info(model_name: str, trust_remote_code: bool | None = None) -> dict:
    """Get model configuration info."""
    trust = _should_trust_remote_code(model_name, trust_remote_code)
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=trust)
    return {
        "model_type": config.model_type,
        "num_layers": config.num_hidden_layers,
        "hidden_size": config.hidden_size,
        "num_attention_heads": config.num_attention_heads,
        "num_key_value_heads": getattr(config, 'num_key_value_heads', config.num_attention_heads),
        "vocab_size": config.vocab_size,
        "rope_scaling": getattr(config, 'rope_scaling', None),
        "max_position_embeddings": getattr(config, 'max_position_embeddings', 2048),
        "head_dim": getattr(config, 'hidden_size', 4096) // getattr(config, 'num_attention_heads', 32),
    }


# --- RoPE Scaling for Long Context (128K+) ---

def build_rope_scaling_config(
    model_type: str = "llama",
    original_max_pos: int = 4096,
    target_max_pos: int = 131072,
    scaling_type: str = "yarn",
    rope_theta: float = 10000.0,
    attention_head_dim: int = 128,
) -> dict:
    """Build RoPE scaling configuration for extending context to 128K+.

    Supports NTK-aware, YaRN, and linear scaling methods.

    Args:
        model_type: Model architecture (llama, mistral, gemma, qwen2).
        original_max_pos: Original max position embeddings (e.g., 4096).
        target_max_pos: Desired max position embeddings (e.g., 131072).
        scaling_type: Scaling method: "linear", "ntk", "ntk_aware", "yarn".
        rope_theta: Base theta for RoPE (default 10000.0).
        attention_head_dim: Dimension per attention head (default 128).

    Returns:
        Dict suitable for setting model config's rope_scaling field.

    Raises:
        ValueError: If scaling_type is not recognized.
    """
    scale = target_max_pos / original_max_pos

    if scaling_type == "linear":
        return {
            "type": "linear",
            "factor": scale,
        }

    if scaling_type in ("ntk", "ntk_aware"):
        rope_theta_scaled = rope_theta * (scale ** (attention_head_dim / (attention_head_dim - 2)))
        return {
            "type": "ntk",
            "factor": scale,
            "rope_theta": rope_theta_scaled,
            "original_max_position_embeddings": original_max_pos,
        }

    if scaling_type == "yarn":
        return {
            "type": "yarn",
            "factor": scale,
            "original_max_position_embeddings": original_max_pos,
            "attention_factor": 1.0,
            "beta_fast": 32,
            "beta_slow": 1,
            "mscale": 1.0,
            "mscale_all_dim": 1.0,
        }

    raise ValueError(f"Unknown RoPE scaling type: {scaling_type}. Supported: linear, ntk, ntk_aware, yarn")


def apply_rope_scaling(
    model,
    target_context_len: int = 131072,
    scaling_type: str = "yarn",
) -> bool:
    """Apply RoPE scaling to a loaded model for extended context.

    Modifies the model's config in-place and re-initializes RoPE
    embeddings if possible.

    Args:
        model: Loaded HuggingFace model.
        target_context_len: Desired context window length.
        scaling_type: Scaling method ("linear", "ntk", "ntk_aware", "yarn").

    Returns:
        True if scaling was applied, False if model doesn't support it.
    """
    config = getattr(model, "config", None)
    if config is None:
        logger.warning("Model has no config, cannot apply RoPE scaling")
        return False

    original_max_pos = getattr(config, "max_position_embeddings", 4096)
    head_dim = getattr(config, "hidden_size", 4096) // getattr(config, "num_attention_heads", 32)
    rope_theta = float(getattr(config, "rope_theta", 10000.0))
    model_type = getattr(config, "model_type", "llama")

    rope_config = build_rope_scaling_config(
        model_type=model_type,
        original_max_pos=original_max_pos,
        target_max_pos=target_context_len,
        scaling_type=scaling_type,
        rope_theta=rope_theta,
        attention_head_dim=head_dim,
    )

    config.max_position_embeddings = target_context_len
    config.rope_scaling = rope_config

    if "theta" in rope_config:
        config.rope_theta = rope_config["theta"]

    logger.info(
        f"Applied {scaling_type} RoPE scaling: "
        f"{original_max_pos} -> {target_context_len} "
        f"(factor={target_context_len / original_max_pos:.1f}x)"
    )
    return True


# --- Auto-Partitioning Optimizer ---

@dataclass
class PartitionProfile:
    """Profiling results for a single layer assignment."""
    node_id: str
    start_layer: int
    end_layer: int
    vram_mb: float = 0.0
    compute_ms: float = 0.0
    communication_ms: float = 0.0
    throughput: float = 0.0  # tokens/second


def profile_partition_throughput(
    model_name: str,
    num_nodes: int,
    batch_size: int = 1,
    seq_len: int = 2048,
    trust_remote_code: bool | None = None,
    gpu_info: dict[str, list] | None = None,
) -> list[tuple[int, int, float]]:
    """Profile and find the optimal layer partition for max throughput.

    Estimates each partition's throughput by considering:
    - VRAM capacity per node (or equal split if not provided)
    - Compute cost proportional to layers assigned
    - Communication cost (proportional to activations sent between nodes)

    Args:
        model_name: HuggingFace model identifier.
        num_nodes: Number of pipeline nodes.
        batch_size: Micro-batch size for profiling.
        seq_len: Sequence length for profiling.
        trust_remote_code: Whether to trust remote HF code.
        gpu_info: Optional dict of node_id -> list of GPUInfo objects.

    Returns:
        List of (start_layer, end_layer, estimated_throughput) sorted by
        throughput descending.
    """
    trust = _should_trust_remote_code(model_name, trust_remote_code)
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=trust)
    total_layers = config.num_hidden_layers or 32
    hidden_size = config.hidden_size or 4096
    num_heads = config.num_attention_heads or 32
    head_dim = hidden_size // num_heads

    # Estimate per-layer compute cost (relative)
    per_layer_flops = (
        4 * batch_size * seq_len * hidden_size * hidden_size  # MLP
        + 2 * batch_size * seq_len * hidden_size * (num_heads * head_dim)  # Attention
        + 4 * batch_size * seq_len * hidden_size  # LayerNorm + residual
    )

    # Activation size sent between nodes (bytes per step)
    activation_bytes = batch_size * seq_len * hidden_size * 2  # fp16

    results: list[tuple[int, int, float, float, float, float]] = []

    # Try multiple partition strategies and evaluate
    strategies = [
        ("equal", None),
    ]

    if gpu_info:
        strategies.append(("gpu_aware", gpu_info))

    for strategy_name, gpus in strategies:
        if strategy_name == "equal":
            partitions = partition_model_across_nodes(model_name, num_nodes, trust)
        else:
            result_dict = partition_model_gpu_aware(gpus, model_name, total_layers, trust)
            partitions = [result_dict[nid] for nid in sorted(result_dict.keys())]

        for start, end in partitions:
            num_assigned_layers = end - start + 1

            # Compute cost (proportional to FLOPs)
            compute_cost = num_assigned_layers * per_layer_flops

            # Communication cost (activation transfer)
            comm_cost = activation_bytes  # one send per step

            # Throughput = 1 / (compute + communication)
            total_cost = compute_cost + comm_cost
            throughput = 1.0 / max(total_cost, 1)

            # VRAM estimate
            vram_per_layer_mb = (
                hidden_size * head_dim * 2 * 2  # K/V cache per layer (fp16)
                + hidden_size * hidden_size * 4 * 2 / (1024 ** 2)  # weights (fp16)
            )
            estimated_vram_mb = num_assigned_layers * vram_per_layer_mb

            results.append((
                start, end, throughput, estimated_vram_mb, compute_cost, comm_cost
            ))

    # Sort by throughput descending
    results.sort(key=lambda r: r[2], reverse=True)
    return results


def find_optimal_partition(
    model_name: str,
    num_nodes: int,
    batch_size: int = 1,
    seq_len: int = 2048,
    trust_remote_code: bool | None = None,
    gpu_info: dict[str, list] | None = None,
) -> list[tuple[int, int]]:
    """Find the optimal layer partition maximizing throughput.

    Profiles multiple partition strategies and returns the best one.

    Args:
        Same as profile_partition_throughput.

    Returns:
        List of (start_layer, end_layer) tuples for the optimal partition.
    """
    profiles = profile_partition_throughput(
        model_name, num_nodes, batch_size, seq_len,
        trust_remote_code, gpu_info,
    )
    if not profiles:
        return partition_model_across_nodes(model_name, num_nodes, trust_remote_code)

    best_start, best_end = profiles[0][0], profiles[0][1]
    # Use proportional allocation: faster layer counts get more layers
    total_layers = AutoConfig.from_pretrained(
        model_name,
        trust_remote_code=_should_trust_remote_code(model_name, trust_remote_code),
    ).num_hidden_layers

    throughputs = {i * (total_layers // num_nodes): prof[2] for i, prof in enumerate(profiles[:num_nodes])}
    total_throughput = sum(throughputs.values()) or 1.0
    result = []
    current = 0
    for i in range(num_nodes):
        fraction = throughputs.get(current, 1.0) / total_throughput
        n_layers = max(1, int(total_layers * fraction)) if i < num_nodes - 1 else total_layers - current
        end = min(current + n_layers - 1, total_layers - 1)
        result.append((current, end))
        current = end + 1
    return result
