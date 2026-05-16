"""Model manager for the Coordinator facade.

Handles model loading, draft model loading, and local generation.
Extracted from the Coordinator class.
"""

from typing import Optional

import torch
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer

from distllm.core.token_generator import TokenGenerator
from distllm.core.batch_scheduler import ScheduledBatch


class ModelManager:
    """Manages model loading and local generation.

    Attributes:
        model_name: Name of the base model.
        dtype: Model data type.
        trust_remote_code: Whether to trust remote code.
        quantization_config: Optional quantization configuration.
    """

    def __init__(
        self,
        model_name: str,
        dtype: str = "float16",
        trust_remote_code: Optional[bool] = None,
        quantization_config=None,
    ):
        self.model_name = model_name
        self.dtype = dtype
        self.trust_remote_code = trust_remote_code
        self.quantization_config = quantization_config

    def load_local_model(self, coordinator) -> None:
        """Load the full model locally (for single-node testing).

        Args:
            coordinator: Coordinator instance to update with loaded model.
        """
        logger.info(f"Loading full model locally: {self.model_name}")

        from distllm.models.partitioner import ModelPartitioner

        coordinator.local_partitioner = ModelPartitioner(
            model_name=self.model_name,
            dtype=self.dtype,
            trust_remote_code=self.trust_remote_code,
            quantization_config=self.quantization_config,
        )
        coordinator.local_partitioner.load_full_model()
        coordinator.tokenizer = coordinator.local_partitioner.tokenizer

        # Load adapters if available
        if coordinator.adapter_manager is not None:
            coordinator.adapter_manager.set_base_model(
                coordinator.local_partitioner.full_model,
                coordinator.tokenizer,
            )
            if hasattr(coordinator, "_lora_adapters_config") and coordinator._lora_adapters_config:
                for adapter_id, adapter_path in coordinator._lora_adapters_config.items():
                    coordinator.adapter_manager.load_adapter(adapter_id, adapter_path)

        # Load draft model if configured
        if coordinator._draft_model_name:
            self.load_draft_model(coordinator)

        logger.info("Full model loaded locally")

    def load_draft_model(self, coordinator) -> None:
        """Load a smaller draft model for speculative decoding.

        Args:
            coordinator: Coordinator instance to update with draft model.
        """
        if not coordinator._draft_model_name:
            logger.debug("No draft model configured")
            return

        logger.info(f"Loading draft model: {coordinator._draft_model_name}")
        trust = self.trust_remote_code
        coordinator.draft_model = AutoModelForCausalLM.from_pretrained(
            coordinator._draft_model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=trust,
            low_cpu_mem_usage=True,
        )
        coordinator.draft_model.eval()
        logger.info(f"Draft model loaded: {coordinator._draft_model_name}")

    def load_draft_model_early(self, coordinator) -> None:
        """Load draft model during Coordinator startup (before local model).

        This allows the draft model to be available for distributed pipeline
        speculative decoding even when no local target model is loaded.

        Args:
            coordinator: Coordinator instance.
        """
        if coordinator._draft_model_name and coordinator.draft_model is None:
            self.load_draft_model(coordinator)

    def generate_local_sync(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        local_partitioner,
        tokenizer,
        draft_model=None,
        num_assistant_tokens: int = 5,
    ) -> str:
        """Synchronous local generation helper.

        Args:
            prompt: Input text prompt.
            max_new_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.
            local_partitioner: Model partitioner with full model.
            tokenizer: Tokenizer for encoding/decoding.
            draft_model: Optional draft model for speculative decoding.
            num_assistant_tokens: Number of assistant tokens for speculative decoding.

        Returns:
            Generated text.
        """
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        model_device = next(local_partitioner.full_model.parameters()).device
        input_ids = input_ids.to(model_device)

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": temperature > 0,
            "pad_token_id": tokenizer.eos_token_id,
        }
        if draft_model is not None:
            gen_kwargs["assistant_model"] = draft_model
            gen_kwargs["num_assistant_tokens"] = num_assistant_tokens

        with torch.no_grad():
            output = local_partitioner.full_model.generate(input_ids, **gen_kwargs)
        return tokenizer.decode(output[0], skip_special_tokens=True)

    def generate_local_batch(
        self,
        batch: ScheduledBatch,
        local_partitioner,
        token_gen: TokenGenerator,
        spec_decoder=None,
        draft_model=None,
        tokenizer=None,
    ) -> None:
        """Run a batch through the local model.

        Args:
            batch: Scheduled batch to process.
            local_partitioner: Model partitioner with full model.
            token_gen: Token generator for sampling.
            spec_decoder: Optional speculative decoder.
            draft_model: Optional draft model.
            tokenizer: Tokenizer for constraint masks.
        """
        batch_size = batch.batch_size
        device = next(local_partitioner.full_model.parameters()).device

        max_len = batch.max_seq_len
        input_ids_list = []
        for i, seq in enumerate(batch.sequences):
            if batch.is_prefill[i]:
                start = seq.prefix_match_len
                tokens = seq.prompt_tokens[start:]
            else:
                tokens = [seq.decode_input_token]
            padded = tokens + [0] * (max_len - len(tokens))
            input_ids_list.append(padded)

        input_ids = torch.tensor(input_ids_list, dtype=torch.long, device=device)
        attention_mask = (input_ids != 0).long()

        # Speculative batch decoding
        if spec_decoder and spec_decoder.is_enabled and draft_model is not None:
            draft_tokens_per_seq = []
            for i, seq in enumerate(batch.sequences):
                seq_input = input_ids[i : i + 1]
                drafts, _ = spec_decoder.generate_draft_tokens(draft_model, seq_input)
                draft_tokens_per_seq.append(drafts)

            with torch.no_grad():
                outputs = local_partitioner.full_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                logits = outputs.logits

            # Verify each sequence's draft tokens
            all_next_tokens = []
            for i, seq in enumerate(batch.sequences):
                drafts = draft_tokens_per_seq[i] if i < len(draft_tokens_per_seq) else []
                _, accepted, next_token = spec_decoder.verify_and_accept(
                    drafts, logits[i : i + 1], tokenizer
                )
                all_next_tokens.append(next_token)

            next_tokens = torch.tensor(all_next_tokens, device=device)
        else:
            with torch.no_grad():
                outputs = local_partitioner.full_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                logits = outputs.logits[:, -1, :]

            next_tokens = token_gen.sample_batch(logits, batch.sequences, tokenizer=tokenizer)

        batch.scheduler.step(batch, next_tokens)
