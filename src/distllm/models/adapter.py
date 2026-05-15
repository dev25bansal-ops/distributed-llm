"""LoRA adapter management for multi-tenant serving."""

from typing import Dict, List, Optional
from loguru import logger


class AdapterManager:
    """Manages LoRA adapters on a base model."""

    def __init__(self, base_model: Optional[object] = None, tokenizer: Optional[object] = None):
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.adapters: Dict[str, object] = {}
        self._active_adapter: Optional[str] = None

    def set_base_model(self, base_model: object, tokenizer: Optional[object] = None) -> None:
        """Set the base model after it has been loaded."""
        self.base_model = base_model
        if tokenizer:
            self.tokenizer = tokenizer

    def load_adapter(self, adapter_id: str, adapter_path: str) -> None:
        """Load a LoRA adapter from path or HuggingFace hub."""
        from peft import PeftModel

        if self.base_model is None:
            raise RuntimeError("No base model loaded. Call set_base_model() first.")

        logger.info(f"Loading LoRA adapter '{adapter_id}' from {adapter_path}")
        model = PeftModel.from_pretrained(self.base_model, adapter_path, adapter_name=adapter_id)
        self.adapters[adapter_id] = model
        logger.info(f"Adapter '{adapter_id}' loaded successfully")

    def set_active(self, adapter_id: Optional[str]) -> None:
        """Switch active adapter. None = base model only."""
        if adapter_id is None:
            self._active_adapter = None
            logger.info("Switched to base model (no adapter)")
        elif adapter_id in self.adapters:
            self.adapters[adapter_id].set_adapter(adapter_id)
            self._active_adapter = adapter_id
            logger.info(f"Switched to adapter '{adapter_id}'")
        else:
            raise KeyError(f"Adapter '{adapter_id}' not found. Available: {list(self.adapters.keys())}")

    def list_adapters(self) -> List[str]:
        """Return list of loaded adapter IDs."""
        return list(self.adapters.keys())

    @property
    def active_adapter(self) -> Optional[str]:
        return self._active_adapter
