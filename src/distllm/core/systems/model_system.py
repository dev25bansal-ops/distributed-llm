"""ModelSystem: model loading, multi-model, adapters.

Groups: ModelManager, MultiModelManager, AdapterManager, VersionManager
"""




class ModelSystem:
    """Manages models: loading, multi-model serving, adapters, versions.

    Composes ModelManager, MultiModelManager, AdapterManager,
    and VersionManager into a single interface.
    """

    def __init__(
        self,
        model_name: str = "",
        dtype: str = "float16",
        device: str = "auto",
        trust_remote_code: bool = False,
    ):
        from distllm.core.coordinator_model import ModelManager
        from distllm.core.coordinator_multi_model import MultiModelManager
        from distllm.core.adapter import AdapterManager
        try:
            from distllm.deploy.version_manager import VersionManager
        except ImportError:
            VersionManager = None

        self.model_mgr = ModelManager(
            model_name=model_name,
            dtype=dtype,
            device=device,
            trust_remote_code=trust_remote_code,
        )

        self.multi_model = MultiModelManager()
        self.adapter_mgr = AdapterManager()
        self.version_mgr = VersionManager() if VersionManager else None

        self._model = None
        self._tokenizer = None

    @property
    def model(self):
        return self._model

    @property
    def tokenizer(self):
        return self._tokenizer

    def load_model(self, model_name: str | None = None, **kwargs) -> bool:
        success = self.model_mgr.load_model(model_name or self.model_mgr.model_name, **kwargs)
        if success:
            self._model = self.model_mgr.model
            self._tokenizer = self.model_mgr.tokenizer
        return success

    def load_local_model(self, model_name: str | None = None, **kwargs) -> bool:
        return self.load_model(model_name, **kwargs)

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        return self.model_mgr.generate(prompt, max_new_tokens, temperature, top_p)

    def register_adapter(self, adapter_id: str, adapter_path: str) -> bool:
        return self.adapter_mgr.load_adapter(adapter_id, adapter_path)

    def activate_adapter(self, adapter_id: str) -> bool:
        return self.adapter_mgr.activate(adapter_id)

    def get_model_name(self) -> str:
        return self.model_mgr.model_name

    def list_models(self) -> list[str]:
        return self.multi_model.list_models()

    def stats(self) -> dict:
        return {
            "model": self.model_mgr.model_name,
            "adapters": self.adapter_mgr.list_adapters(),
        }
