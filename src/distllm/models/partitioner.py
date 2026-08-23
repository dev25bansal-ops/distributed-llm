"""Model loading and partitioning for distributed LLM inference."""

from __future__ import annotations

import gc
import inspect
import os
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from loguru import logger
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from distllm.config.settings import QuantizationSettings as QuantizationConfig
from distllm.dist.fsdp import FSDPConfig, FSDPShard
from distllm.errors import ModelLoadError
from distllm.security import hf_revision


def _get_trusted_models() -> set[str]:
    return set()


_TRUSTED_FROM_REGISTRY = _get_trusted_models()

DTYPE_MAP = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}


def _find_attr(model: Any, candidates: list[str]) -> Any:
    """Find the first matching attribute on a model object."""
    for attr in candidates:
        if hasattr(model, attr):
            return getattr(model, attr)
    return None


def _get_base_prefix(model: Any) -> str:
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

    def __init__(self, model_name: str, device: str = "auto", dtype: str = "float16", trust_remote_code: bool | None = None, quantization_config: QuantizationConfig | None = None, compression_config: Any = None):
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self.trust_remote_code = trust_remote_code
        self.model_revision = hf_revision()
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
        self._causal_mask_max_len = self._read_causal_mask_max_len()
        self._causal_mask_limit_warned = False

    @staticmethod
    def _read_causal_mask_max_len() -> int:
        raw = os.environ.get("DISTLLM_MAX_CACHED_CAUSAL_MASK", "8192")
        try:
            return max(1, int(raw))
        except ValueError:
            logger.warning(f"Invalid DISTLLM_MAX_CACHED_CAUSAL_MASK={raw!r}; using 8192")
            return 8192

    def load_full_model(self) -> None:
        """Load the complete model (for single-node mode)."""
        logger.info(f"Loading full model: {self.model_name}")
        trust = _should_trust_remote_code(self.model_name, self.trust_remote_code)
        self.config = AutoConfig.from_pretrained(
            self.model_name,
            trust_remote_code=trust,
            revision=self.model_revision,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=trust,
            revision=self.model_revision,
        )

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

        self.full_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            revision=self.model_revision,
            **model_kwargs,
        )

        # Apply compression pipeline (legacy module removed)
        if self.compression_config and self.compression_config.enabled:
            logger.warning("Compression pipeline not available (legacy module removed), skipping")

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
        """Load only a subset of layers (start_layer to end_layer inclusive).

        Tries selective safetensors loading first (download only needed shard
        files), falling back to full-model load + extract for legacy models.
        """
        logger.info(f"Loading layers {start_layer}-{end_layer} of {total_layers} for {self.model_name}")
        trust = _should_trust_remote_code(self.model_name, self.trust_remote_code)
        self.config = AutoConfig.from_pretrained(
            self.model_name,
            trust_remote_code=trust,
            revision=self.model_revision,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=trust,
            revision=self.model_revision,
        )

        device = device or self.device
        torch_dtype = DTYPE_MAP.get(self.dtype, torch.float16)

        # Try selective safetensors loading first
        if self._try_load_selective(start_layer, end_layer, total_layers, device, torch_dtype, trust):
            logger.info(f"Selectively loaded layers {start_layer}-{end_layer} on {device}")
            return

        # Fallback: load the entire model then extract subset
        logger.warning("Falling back to full-model load for layer extraction")
        quant_config = build_quantization_config(self.quantization_config) if self.quantization_config else None

        d = torch.device(device)
        model_kwargs = {
            "config": self.config,
            "torch_dtype": torch_dtype,
            "trust_remote_code": trust,
            "low_cpu_mem_usage": True,
        }
        if quant_config is not None:
            model_kwargs["quantization_config"] = quant_config

        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            revision=self.model_revision,
            **model_kwargs,
        )
        model.eval()
        self._extract_subset(model, start_layer, end_layer, total_layers, device)
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info(f"Loaded layers {start_layer}-{end_layer} on {device}")

    def _try_load_selective(
        self, start_layer: int, end_layer: int, total_layers: int,
        device: str, torch_dtype: torch.dtype, trust: bool,
    ) -> bool:
        """Try to load only the needed shards using safetensors index.

        Uses :class:`SafetensorsIndex` to map layers to shard files and
        :class:`ModelHub.download_layer_subset` to fetch only those
        shards.  Falls back to the full-model loading path on failure.

        Returns:
            ``True`` on success, ``False`` to trigger the full-model fallback.
        """
        try:
            from accelerate import init_empty_weights
            from safetensors import safe_open
        except ImportError:
            return False

        from transformers import AutoModelForCausalLM as ModelBuilder

        from distllm.models.model_hub import ModelHub
        from distllm.models.safetensors_index import SafetensorsIndex

        try:
            target_device = torch.device(device)

            # Step 1: resolve the index (from cache or Hub)
            #         and determine which shards / keys are needed.
            hub = ModelHub()
            index = SafetensorsIndex.from_hub(
                self.model_name, revision=self.model_revision,
            )

            needed_keys = index.get_keys_for_layer_range(start_layer, end_layer)
            needed_shards = index.get_shards_for_layer_range(start_layer, end_layer)

            # Single-file model → use the single-safetensors loader
            if len(needed_shards) <= 1 and not needed_keys:
                return self._try_load_single_safetensors(
                    None, needed_keys, device, target_device,
                    torch_dtype, trust=trust,
                )

            # Step 2: download only the needed shard files.
            #         ``download_layer_subset`` ensures the index + shards
            #         are in the HuggingFace shared cache.
            hub.download_layer_subset(
                self.model_name, start_layer, end_layer,
                revision=self.model_revision,
            )

            # Step 3: resolve downloaded shard paths via hf_hub_download
            #         (returns cached path if already downloaded).
            from huggingface_hub import hf_hub_download

            shard_paths: dict[str, str] = {}
            for shard in sorted(needed_shards):
                if shard == "model.safetensors.index.json":
                    continue
                path = hf_hub_download(
                    self.model_name, shard,
                    revision=self.model_revision,
                )
                shard_paths[shard] = path

            # Step 4: create model skeleton on meta device (no memory allocated)
            with init_empty_weights():
                partial_model = ModelBuilder.from_config(
                    self.config,
                    trust_remote_code=trust,
                    torch_dtype=torch_dtype,
                )
            partial_model.eval()

            # Step 5: load only needed tensors from downloaded shards
            state_dict: dict[str, torch.Tensor] = {}
            for shard, path in shard_paths.items():
                with safe_open(path, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        if key in needed_keys:
                            state_dict[key] = f.get_tensor(key)

            if not state_dict:
                logger.warning("No tensors loaded from safetensors shards")
                return False

            # Step 6: apply weights (only needed keys, rest stay on meta device)
            partial_model.load_state_dict(state_dict, strict=False, assign=True)
            del state_dict

            # Step 7: extract subset and move to target device
            self._extract_subset(
                partial_model, start_layer, end_layer, total_layers, device,
            )
            del partial_model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return True

        except Exception as e:
            logger.warning(f"Selective safetensors loading failed: {e}")
            return False

    def _is_layer_in_range(self, param_key: str, start_layer: int, end_layer: int) -> bool:
        """Check if a parameter key belongs to one of the target layers.

        Matches keys like ``model.layers.5.self_attn.q_proj.weight``
        and includes embedding/LM-head keys for first/last node.
        """
        import re
        match = re.search(r'(?:\.layers|\.block|\.h)\.(\d+)\.', param_key)
        if match:
            layer_num = int(match.group(1))
            return start_layer <= layer_num <= end_layer
        return True  # Non-layer params (embeddings, norm, lm_head) always included

    def _try_load_single_safetensors(
        self, partial_model: Any, needed_keys: set[str] | None, device: str, target_device: torch.device, torch_dtype: torch.dtype,
        trust: bool = False,
    ) -> bool:
        """Fallback: load from a single ``model.safetensors`` file."""
        try:
            from accelerate import init_empty_weights
            from huggingface_hub import hf_hub_download
            from safetensors import safe_open
            from transformers import AutoModelForCausalLM as ModelBuilder

            safetensors_path = hf_hub_download(
                self.model_name, "model.safetensors",
                revision=self.model_revision,
            )

            # Determine keys to load (all if needed_keys is empty/None)
            with safe_open(safetensors_path, framework="pt", device="cpu") as f:
                available_keys = set(f.keys())
                load_keys = needed_keys if needed_keys else available_keys

            # Create model skeleton on meta device
            with init_empty_weights():
                if partial_model is None:
                    partial_model = ModelBuilder.from_config(
                        self.config,
                        trust_remote_code=trust,
                        torch_dtype=torch_dtype,
                    )
                partial_model.eval()

            # Load only the needed tensors
            state_dict = {}
            with safe_open(safetensors_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if key in load_keys:
                        state_dict[key] = f.get_tensor(key)

            if not state_dict:
                return False

            partial_model.load_state_dict(state_dict, strict=False, assign=True)
            return True
        except Exception:
            return False

    def _extract_subset(self, full_model: Any, start_layer: int, end_layer: int, total_layers: int, device: str) -> None:
        """Extract specific layers and components from the full model.

        Supports:
        - Decoder-only (GPT, Llama, Mistral)
        - Encoder-decoder (T5, BART, vision-language models)
        """
        self._is_encoder_decoder = False

        # Detect encoder-decoder architecture
        if hasattr(full_model, 'encoder') and hasattr(full_model, 'decoder'):
            self._is_encoder_decoder = True
            self.encoder_module = full_model.encoder
            self.decoder_module = full_model.decoder
            logger.info("Detected encoder-decoder architecture")

        # Extract encoder layers if this is an encoder-decoder model
        if self._is_encoder_decoder and start_layer == 0:
            encoder_layers = _find_attr(
                self.encoder_module,
                ['layers', 'block', 'layer', 'encoder_layer'],
            ) or _find_attr(full_model, ['encoder_layers', 'encoder_layer'])
            if encoder_layers is None:
                # Fallback: look for 'layer' directly on the encoder
                encoder_layers = getattr(self.encoder_module, 'layer', None) if hasattr(self.encoder_module, 'layer') else None
            if encoder_layers is not None:
                self.encoder_layers = nn.ModuleList()
                for i in range(min(len(encoder_layers), total_layers)):
                    self.encoder_layers.append(encoder_layers[i].to(device) if hasattr(encoder_layers[i], 'to') else encoder_layers[i])
                logger.info(f"Loaded {len(self.encoder_layers)} encoder layers")

        base_model = None
        for attr in ['model', 'transformer', 'decoder' if self._is_encoder_decoder else 'transformer', 'encoder']:
            if hasattr(full_model, attr):
                base_model = getattr(full_model, attr)
                break
        if base_model is None:
            base_model = full_model

        self.is_first_node = (start_layer == 0)
        if self.is_first_node:
            embed_layer = _find_attr(base_model, ['embed_tokens', 'wte', 'word_embeddings'])
            if embed_layer is None and not self._is_encoder_decoder:
                decoder_part = getattr(base_model, 'decoder', None)
                if decoder_part:
                    embed_layer = _find_attr(decoder_part, ['embed_tokens', 'wte', 'word_embeddings'])
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

    # ── FSDP-style weight sharding ────────────────────────────────────────

    def enable_fsdp(
        self,
        world_size: int = 1,
        rank: int = 0,
        cpu_offload: bool = False,
        min_param_size: int = 1024,
    ) -> None:
        """Enable FSDP-style weight sharding across nodes.

        Shards all model parameters across *world_size* ranks.  Each node
        keeps only a 1/*world_size* chunk; the full weights are reconstructed
        via an all-gather before the forward pass and freed afterward.

        Call after ``load_full_model()`` or ``load_layer_subset()``.

        Args:
            world_size: Total number of sharding ranks (nodes).
            rank: Local rank of this node (0-based).
            cpu_offload: If True, offload non-local shards to CPU instead of
                freeing memory after forward.
            min_param_size: Minimum number of elements to shard (smaller
                parameters are replicated on every rank).
        """
        # Collect all components into a single module for FSDP sharding.
        # We build a temporary wrapper so FSDPShard can traverse parameters.
        wrapper = nn.ModuleDict()
        if self.embed_tokens is not None:
            wrapper["embed_tokens"] = self.embed_tokens
        if self.layers:
            wrapper["layers"] = self.layers
        if self.final_norm is not None:
            wrapper["final_norm"] = self.final_norm
        if self.lm_head is not None:
            wrapper["lm_head"] = self.lm_head
        if self.rotary_emb is not None:
            wrapper["rotary_emb"] = self.rotary_emb

        config = FSDPConfig(
            world_size=world_size,
            rank=rank,
            min_param_size=min_param_size,
            cpu_offload=cpu_offload,
        )
        self._fsdp = FSDPShard(wrapper, config=config)
        self._fsdp.shard()

        num_sharded = len(self._fsdp._sharded_params)
        logger.info(
            f"FSDP enabled on rank {rank}/{world_size}: "
            f"{num_sharded} parameter groups sharded, "
            f"min_param_size={min_param_size}"
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]] | None]:
        """Run forward pass through assigned layers.

        If FSDP weight sharding is enabled, this method automatically
        gathers full weights before the layer loop and frees non-local
        shards after.
        """
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
        attention_mask_4d = None
        # Bound the cached dense mask. Long-context models should rely on the
        # backend's causal attention path instead of allocating seq_len^2 memory.
        if total_len <= self._causal_mask_max_len:
            target_len = min(max(total_len, 4096), self._causal_mask_max_len)
            if (
                self._causal_mask is None
                or self._causal_mask.shape[0] < total_len
                or self._causal_mask.shape[0] > self._causal_mask_max_len
                or self._causal_mask_device != str(device)
            ):
                self._causal_mask = torch.tril(
                    torch.ones(target_len, target_len, device=device, dtype=torch.bool)
                )
                self._causal_mask_device = str(device)
            attention_mask_4d = self._causal_mask[:total_len, :total_len]
        else:
            if self._causal_mask is not None and self._causal_mask.shape[0] > self._causal_mask_max_len:
                self._causal_mask = None
                self._causal_mask_device = ""
            if not self._causal_mask_limit_warned:
                logger.warning(
                    f"Skipping dense causal mask cache for sequence length {total_len}; "
                    f"configured limit is {self._causal_mask_max_len}"
                )
                self._causal_mask_limit_warned = True

        # FSDP: gather full weights before the layer forward pass
        fsdp = getattr(self, "_fsdp", None)
        if fsdp is not None:
            fsdp.gather()

        try:
            for i, layer in enumerate(self.layers):
                layer_past = None
                if past_key_values and i < len(past_key_values):
                    layer_past = past_key_values[i]

                params = self._layer_params[i] if i < len(self._layer_params) else set()

                layer_kwargs = {"hidden_states": hidden_states}

                if "attention_mask" in params and attention_mask_4d is not None:
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

        finally:
            # FSDP: free non-local shards after forward
            if fsdp is not None:
                fsdp.free()

        return hidden_states, new_past_key_values

    def embed_input(self, input_ids: torch.Tensor, position_offset: int = 0) -> torch.Tensor:
        """Convert token IDs to embeddings with positional encoding (first node only)."""
        if self.embed_tokens is None:
            raise RuntimeError("No embedding layer on this node")

        fsdp = getattr(self, "_fsdp", None)
        if fsdp is not None:
            fsdp.gather()

        try:
            hidden = self.embed_tokens(input_ids)
            if self.position_embeds is not None:
                seq_len = input_ids.shape[-1]
                position_ids = torch.arange(position_offset, position_offset + seq_len, device=input_ids.device).unsqueeze(0)
                hidden = hidden + self.position_embeds(position_ids)
            return hidden
        finally:
            if fsdp is not None:
                fsdp.free()

    def get_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute logits from hidden states (last node only)."""
        if not self.is_last_node:
            raise RuntimeError("This node is not the last node in the pipeline")

        fsdp = getattr(self, "_fsdp", None)
        if fsdp is not None:
            fsdp.gather()

        try:
            if self.final_norm is not None:
                hidden_states = self.final_norm(hidden_states)

            if self.lm_head is not None:
                return self.lm_head(hidden_states)

            raise RuntimeError("No LM head available")
        finally:
            if fsdp is not None:
                fsdp.free()

    def get_model_config(self) -> dict:
        """Get model configuration parameters."""
        if self.config is None:
            trust = _should_trust_remote_code(self.model_name, self.trust_remote_code)
            self.config = AutoConfig.from_pretrained(
                self.model_name,
                trust_remote_code=trust,
                revision=self.model_revision,
            )
        return {
            "hidden_size": self.config.hidden_size,
            "num_attention_heads": self.config.num_attention_heads,
            "num_hidden_layers": self.config.num_hidden_layers,
            "vocab_size": self.config.vocab_size,
            "num_key_value_heads": getattr(self.config, 'num_key_value_heads', self.config.num_attention_heads),
        }


# Backward-compatible re-exports
from distllm.models.partition_planner import (  # noqa: E402
    PartitionProfile,
    find_optimal_partition,
    get_model_info,
    partition_model_across_nodes,
    partition_model_gpu_aware,
    profile_partition_throughput,
)
from distllm.models.rope_scaling import apply_rope_scaling, build_rope_scaling_config  # noqa: E402
