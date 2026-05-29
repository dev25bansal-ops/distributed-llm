"""Config tests — RemoteDraftConfig creation, validation, defaults."""


from distllm.core.distributed_speculative import (
    RemoteDraftConfig,
    RemoteDraftModel,
    DistributedSpeculativeDecoder,
)


class TestRemoteDraftConfig:
    def test_defaults(self):
        cfg = RemoteDraftConfig(endpoint_url="http://draft:8000/v1/completions")
        assert cfg.model_name == ""
        assert cfg.api_key == ""
        assert cfg.timeout_seconds == 30.0
        assert cfg.max_retries == 2
        assert cfg.transport == "http"
        assert cfg.prompt_format == "auto"
        assert cfg.verify_ssl is True

    def test_custom_values(self):
        cfg = RemoteDraftConfig(
            endpoint_url="https://api.openai.com/v1/chat/completions",
            model_name="gpt-4o-mini",
            api_key="sk-test",
            timeout_seconds=10.0,
            max_retries=3,
            transport="http",
            prompt_format="text",
            verify_ssl=False,
        )
        assert cfg.model_name == "gpt-4o-mini"
        assert cfg.api_key == "sk-test"
        assert cfg.timeout_seconds == 10.0
        assert cfg.max_retries == 3
        assert cfg.prompt_format == "text"
        assert cfg.verify_ssl is False

    def test_grpc_transport(self):
        cfg = RemoteDraftConfig(
            endpoint_url="grpc://draft-node:50051",
            transport="grpc",
        )
        assert cfg.transport == "grpc"


class TestStringURLAutoConfig:
    def test_string_creates_config(self):
        model = RemoteDraftModel("http://draft:8000/v1/completions")
        assert model._config.endpoint_url == "http://draft:8000/v1/completions"
        assert model._config.verify_ssl is True
        model.close()

    def test_decoder_string_url(self):
        sd = DistributedSpeculativeDecoder(
            target_forward=lambda x, **kw: x,
            draft_model="http://draft:8000/v1/completions",
        )
        assert isinstance(sd._draft, RemoteDraftModel)
        assert sd._draft._config.endpoint_url == "http://draft:8000/v1/completions"
        sd.close()

    def test_decoder_config_object(self):
        cfg = RemoteDraftConfig(
            endpoint_url="https://secure-draft:9000/v1/completions",
            api_key="sk-secret",
            verify_ssl=False,
        )
        sd = DistributedSpeculativeDecoder(
            target_forward=lambda x, **kw: x,
            draft_model=cfg,
        )
        assert sd._draft._config.api_key == "sk-secret"
        assert sd._draft._config.verify_ssl is False
        sd.close()


class TestSSLConfig:
    def test_verify_ssl_true_by_default(self):
        model = RemoteDraftModel(RemoteDraftConfig(
            endpoint_url="https://draft:8000/v1/completions",
        ))
        assert model._config.verify_ssl is True
        model.close()

    def test_verify_ssl_false(self):
        model = RemoteDraftModel(RemoteDraftConfig(
            endpoint_url="https://self-signed:8000/v1/completions",
            verify_ssl=False,
        ))
        assert model._config.verify_ssl is False
        model.close()


class TestPromptFormat:
    def test_auto_detect_openai(self):
        model = RemoteDraftModel(RemoteDraftConfig(
            endpoint_url="https://api.openai.com/v1/completions",
            prompt_format="auto",
        ))
        assert model._should_use_chat_completions("") is True
        model.close()

    def test_auto_detect_plain(self):
        model = RemoteDraftModel(RemoteDraftConfig(
            endpoint_url="http://draft:8000/v1/completions",
            prompt_format="auto",
        ))
        assert model._should_use_chat_completions("") is False
        model.close()

    def test_force_text(self):
        model = RemoteDraftModel(RemoteDraftConfig(
            endpoint_url="http://draft:8000/v1/completions",
            prompt_format="text",
        ))
        assert model._should_use_chat_completions("") is True
        model.close()

    def test_force_tokens(self):
        model = RemoteDraftModel(RemoteDraftConfig(
            endpoint_url="https://api.openai.com/v1/completions",
            prompt_format="tokens",
        ))
        assert model._should_use_chat_completions("") is False
        model.close()
