from unittest.mock import MagicMock, patch
import pytest


class TestBackendProtocol:
    def test_backend_interface_methods(self):
        from distllm.backends.protocol import BackendAdapter
        import inspect
        methods = ["load_model", "forward", "shutdown", "generate"]
        for m in methods:
            assert hasattr(BackendAdapter, m), f"Missing method: {m}"

    def test_backend_classmethods(self):
        from distllm.backends.protocol import BackendAdapter
        assert hasattr(BackendAdapter, "display_name")
        assert hasattr(BackendAdapter, "is_available")
        assert hasattr(BackendAdapter, "priority_for")

    def test_generate_default_raises_not_implemented(self):
        from distllm.backends.protocol import BackendAdapter

        class MinimalAdapter(BackendAdapter):
            def load_model(self):
                pass
            def forward(self, hidden_states=None, attention_mask=None,
                       position_ids=None, past_key_values=None, input_ids=None):
                return None, []
            def shutdown(self):
                pass
            @classmethod
            def display_name(cls):
                return "test"
            @classmethod
            def is_available(cls):
                return True
            @classmethod
            def priority_for(cls, device_type):
                return 0

        adapter = MinimalAdapter()
        with pytest.raises(NotImplementedError):
            adapter.generate("hello")

    def test_get_tokenizer_default_none(self):
        from distllm.backends.protocol import BackendAdapter

        class MinimalAdapter(BackendAdapter):
            def load_model(self):
                pass
            def forward(self, hidden_states=None, attention_mask=None,
                       position_ids=None, past_key_values=None, input_ids=None):
                return None, []
            def shutdown(self):
                pass
            @classmethod
            def display_name(cls):
                return "test"
            @classmethod
            def is_available(cls):
                return True
            @classmethod
            def priority_for(cls, device_type):
                return 0

        adapter = MinimalAdapter()
        assert adapter.get_tokenizer() is None

    def test_description_default(self):
        from distllm.backends.protocol import BackendAdapter

        class MinimalAdapter(BackendAdapter):
            def load_model(self):
                pass
            def forward(self, hidden_states=None, attention_mask=None,
                       position_ids=None, past_key_values=None, input_ids=None):
                return None, []
            def shutdown(self):
                pass
            @classmethod
            def display_name(cls):
                return "test"
            @classmethod
            def is_available(cls):
                return True
            @classmethod
            def priority_for(cls, device_type):
                return 0

        assert MinimalAdapter.description() == ""

    def test_version_default(self):
        from distllm.backends.protocol import BackendAdapter

        class MinimalAdapter(BackendAdapter):
            def load_model(self):
                pass
            def forward(self, hidden_states=None, attention_mask=None,
                       position_ids=None, past_key_values=None, input_ids=None):
                return None, []
            def shutdown(self):
                pass
            @classmethod
            def display_name(cls):
                return "test"
            @classmethod
            def is_available(cls):
                return True
            @classmethod
            def priority_for(cls, device_type):
                return 0

        assert MinimalAdapter.version() == "1.0.0"


class TestBackendRegistry:
    @patch("distllm.backends.registry._registry", {})
    def test_register_and_list(self):
        from distllm.backends.registry import BackendRegistry
        from distllm.backends.protocol import BackendAdapter

        class TestBackend(BackendAdapter):
            def load_model(self):
                pass
            def forward(self, hidden_states=None, attention_mask=None,
                       position_ids=None, past_key_values=None, input_ids=None):
                return None, []
            def shutdown(self):
                pass
            @classmethod
            def display_name(cls):
                return "Test"
            @classmethod
            def is_available(cls):
                return True
            @classmethod
            def priority_for(cls, device_type):
                return 5 if device_type == "cuda" else 0

        BackendRegistry.register(TestBackend, name="test_backend")
        backends = BackendRegistry.list_backends()
        assert len(backends) == 1
        assert backends[0].name == "test_backend"

    @patch("distllm.backends.registry._registry", {})
    def test_get_backend(self):
        from distllm.backends.registry import BackendRegistry
        from distllm.backends.protocol import BackendAdapter

        class TestBackend(BackendAdapter):
            def load_model(self):
                pass
            def forward(self, hidden_states=None, attention_mask=None,
                       position_ids=None, past_key_values=None, input_ids=None):
                return None, []
            def shutdown(self):
                pass
            @classmethod
            def display_name(cls):
                return "Test"
            @classmethod
            def is_available(cls):
                return True
            @classmethod
            def priority_for(cls, device_type):
                return 5

        BackendRegistry.register(TestBackend, name="test_get")
        cls = BackendRegistry.get("test_get")
        assert cls is TestBackend

    @patch("distllm.backends.registry._registry", {})
    def test_get_nonexistent_backend(self):
        from distllm.backends.registry import BackendRegistry
        cls = BackendRegistry.get("nonexistent")
        assert cls is None

    @patch("distllm.backends.registry._registry", {})
    def test_register_duplicate_raises_keyerror(self):
        from distllm.backends.registry import BackendRegistry
        from distllm.backends.protocol import BackendAdapter

        class TestBackend(BackendAdapter):
            def load_model(self):
                pass
            def forward(self, hidden_states=None, attention_mask=None,
                       position_ids=None, past_key_values=None, input_ids=None):
                return None, []
            def shutdown(self):
                pass
            @classmethod
            def display_name(cls):
                return "Test"
            @classmethod
            def is_available(cls):
                return True
            @classmethod
            def priority_for(cls, device_type):
                return 5

        BackendRegistry.register(TestBackend, name="dup")
        with pytest.raises(KeyError):
            BackendRegistry.register(TestBackend, name="dup")

    @patch("distllm.backends.registry._registry", {})
    def test_register_duplicate_with_force(self):
        from distllm.backends.registry import BackendRegistry
        from distllm.backends.protocol import BackendAdapter

        class TestBackendA(BackendAdapter):
            def load_model(self):
                pass
            def forward(self, hidden_states=None, attention_mask=None,
                       position_ids=None, past_key_values=None, input_ids=None):
                return None, []
            def shutdown(self):
                pass
            @classmethod
            def display_name(cls):
                return "A"
            @classmethod
            def is_available(cls):
                return True
            @classmethod
            def priority_for(cls, device_type):
                return 5

        class TestBackendB(BackendAdapter):
            def load_model(self):
                pass
            def forward(self, hidden_states=None, attention_mask=None,
                       position_ids=None, past_key_values=None, input_ids=None):
                return None, []
            def shutdown(self):
                pass
            @classmethod
            def display_name(cls):
                return "B"
            @classmethod
            def is_available(cls):
                return True
            @classmethod
            def priority_for(cls, device_type):
                return 5

        BackendRegistry.register(TestBackendA, name="force_test")
        BackendRegistry.register(TestBackendB, name="force_test", force=True)
        cls = BackendRegistry.get("force_test")
        assert cls is TestBackendB

    @patch("distllm.backends.registry._registry", {})
    def test_register_non_adapter_raises_valueerror(self):
        from distllm.backends.registry import BackendRegistry

        class NotAnAdapter:
            pass

        with pytest.raises(ValueError):
            BackendRegistry.register(NotAnAdapter)

    @patch("distllm.backends.registry._registry", {})
    def test_unregister(self):
        from distllm.backends.registry import BackendRegistry
        from distllm.backends.protocol import BackendAdapter

        class TestBackend(BackendAdapter):
            def load_model(self):
                pass
            def forward(self, hidden_states=None, attention_mask=None,
                       position_ids=None, past_key_values=None, input_ids=None):
                return None, []
            def shutdown(self):
                pass
            @classmethod
            def display_name(cls):
                return "Test"
            @classmethod
            def is_available(cls):
                return True
            @classmethod
            def priority_for(cls, device_type):
                return 5

        BackendRegistry.register(TestBackend, name="to_unregister")
        assert BackendRegistry.get("to_unregister") is not None
        BackendRegistry.unregister("to_unregister")
        assert BackendRegistry.get("to_unregister") is None

    @patch("distllm.backends.registry._registry", {})
    def test_get_plugin(self):
        from distllm.backends.registry import BackendRegistry, BackendPlugin
        from distllm.backends.protocol import BackendAdapter

        class TestBackend(BackendAdapter):
            def load_model(self):
                pass
            def forward(self, hidden_states=None, attention_mask=None,
                       position_ids=None, past_key_values=None, input_ids=None):
                return None, []
            def shutdown(self):
                pass
            @classmethod
            def display_name(cls):
                return "Test"
            @classmethod
            def is_available(cls):
                return True
            @classmethod
            def priority_for(cls, device_type):
                return 5

        BackendRegistry.register(TestBackend, name="plugin_test")
        plugin = BackendRegistry.get_plugin("plugin_test")
        assert isinstance(plugin, BackendPlugin)
        assert plugin.name == "plugin_test"
        assert plugin.adapter_class is TestBackend

    @patch("distllm.backends.registry._registry", {})
    def test_list_available(self):
        from distllm.backends.registry import BackendRegistry
        from distllm.backends.protocol import BackendAdapter

        class AvailableBackend(BackendAdapter):
            def load_model(self):
                pass
            def forward(self, hidden_states=None, attention_mask=None,
                       position_ids=None, past_key_values=None, input_ids=None):
                return None, []
            def shutdown(self):
                pass
            @classmethod
            def display_name(cls):
                return "Available"
            @classmethod
            def is_available(cls):
                return True
            @classmethod
            def priority_for(cls, device_type):
                return 5

        class UnavailableBackend(BackendAdapter):
            def load_model(self):
                pass
            def forward(self, hidden_states=None, attention_mask=None,
                       position_ids=None, past_key_values=None, input_ids=None):
                return None, []
            def shutdown(self):
                pass
            @classmethod
            def display_name(cls):
                return "Unavailable"
            @classmethod
            def is_available(cls):
                return False
            @classmethod
            def priority_for(cls, device_type):
                return 0

        BackendRegistry.register(AvailableBackend, name="avail")
        BackendRegistry.register(UnavailableBackend, name="unavail")
        available = BackendRegistry.list_available()
        names = [p.name for p in available]
        assert "avail" in names
        assert "unavail" not in names

    @patch("distllm.backends.registry._registry", {})
    def test_select_preferred_backend(self):
        from distllm.backends.registry import BackendRegistry
        from distllm.backends.protocol import BackendAdapter

        class TestBackend(BackendAdapter):
            def load_model(self):
                pass
            def forward(self, hidden_states=None, attention_mask=None,
                       position_ids=None, past_key_values=None, input_ids=None):
                return None, []
            def shutdown(self):
                pass
            @classmethod
            def display_name(cls):
                return "Test"
            @classmethod
            def is_available(cls):
                return True
            @classmethod
            def priority_for(cls, device_type):
                return 5

        BackendRegistry.register(TestBackend, name="test_select")
        selected = BackendRegistry.select(preferred_backend="test_select")
        assert selected is TestBackend

    @patch("distllm.backends.registry._registry", {})
    def test_select_no_available(self):
        from distllm.backends.registry import BackendRegistry
        selected = BackendRegistry.select(device_type="cpu")
        assert selected is None

    @patch("distllm.backends.registry._registry", {})
    def test_register_with_default_name(self):
        from distllm.backends.registry import BackendRegistry, _default_name
        from distllm.backends.protocol import BackendAdapter

        class MyCustomBackend(BackendAdapter):
            def load_model(self):
                pass
            def forward(self, hidden_states=None, attention_mask=None,
                       position_ids=None, past_key_values=None, input_ids=None):
                return None, []
            def shutdown(self):
                pass
            @classmethod
            def display_name(cls):
                return "MyCustom"
            @classmethod
            def is_available(cls):
                return True
            @classmethod
            def priority_for(cls, device_type):
                return 5

        BackendRegistry.register(MyCustomBackend)
        name = _default_name(MyCustomBackend)
        assert BackendRegistry.get(name) is MyCustomBackend

    @patch("distllm.backends.registry._registry", {})
    def test_select_plugin(self):
        from distllm.backends.registry import BackendRegistry
        from distllm.backends.protocol import BackendAdapter

        class TestBackend(BackendAdapter):
            def load_model(self):
                pass
            def forward(self, hidden_states=None, attention_mask=None,
                       position_ids=None, past_key_values=None, input_ids=None):
                return None, []
            def shutdown(self):
                pass
            @classmethod
            def display_name(cls):
                return "Test"
            @classmethod
            def is_available(cls):
                return True
            @classmethod
            def priority_for(cls, device_type):
                return 5

        BackendRegistry.register(TestBackend, name="sp")
        plugin = BackendRegistry.select_plugin(preferred_backend="sp")
        assert plugin is not None
        assert plugin.name == "sp"


class TestConvenienceFunctions:
    @patch("distllm.backends.registry._registry", {})
    def test_get_backend(self):
        from distllm.backends.registry import get_backend, BackendRegistry
        from distllm.backends.protocol import BackendAdapter

        class TestBackend(BackendAdapter):
            def load_model(self):
                pass
            def forward(self, hidden_states=None, attention_mask=None,
                       position_ids=None, past_key_values=None, input_ids=None):
                return None, []
            def shutdown(self):
                pass
            @classmethod
            def display_name(cls):
                return "Test"
            @classmethod
            def is_available(cls):
                return True
            @classmethod
            def priority_for(cls, device_type):
                return 5

        BackendRegistry.register(TestBackend, name="conv_test")
        cls = get_backend("conv_test")
        assert cls is TestBackend

    @patch("distllm.backends.registry._registry", {})
    def test_list_backends(self):
        from distllm.backends.registry import list_backends, BackendRegistry
        from distllm.backends.protocol import BackendAdapter

        class TestBackend(BackendAdapter):
            def load_model(self):
                pass
            def forward(self, hidden_states=None, attention_mask=None,
                       position_ids=None, past_key_values=None, input_ids=None):
                return None, []
            def shutdown(self):
                pass
            @classmethod
            def display_name(cls):
                return "Test"
            @classmethod
            def is_available(cls):
                return True
            @classmethod
            def priority_for(cls, device_type):
                return 5

        BackendRegistry.register(TestBackend, name="lb")
        result = list_backends()
        assert len(result) == 1


class TestDefaultName:
    def test_default_name_strips_suffixes(self):
        from distllm.backends.registry import _default_name

        class VLLMNodeAdapter:
            pass

        class PyTorchNodeAdapter:
            pass

        class CustomAdapter:
            pass

        class MyBackend:
            pass

        assert _default_name(VLLMNodeAdapter) == "vllm"
        assert _default_name(PyTorchNodeAdapter) == "pytorch"


class TestDetectDevice:
    @patch("distllm.backends.registry.torch")
    def test_detect_cuda(self, mock_torch):
        from distllm.backends.registry import _detect_device
        mock_torch.cuda.is_available.return_value = True
        mock_torch.__version__ = "2.0.0"
        device = _detect_device()
        assert device == "cuda"

    @patch("distllm.backends.registry.torch")
    def test_detect_cpu_fallback(self, mock_torch):
        from distllm.backends.registry import _detect_device
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends = MagicMock()
        mock_torch.mps.is_available.return_value = False
        mock_torch.xpu.is_available.return_value = False
        device = _detect_device()
        assert device == "cpu"
