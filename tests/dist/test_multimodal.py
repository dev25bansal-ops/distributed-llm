"""Real tests for multi-modal pipeline parallelism — modality detection, vision
encoder, routing, projection layers, and token interleaving.

Zero mocks — all tests use real instances and deterministic logic.
"""

from __future__ import annotations

import pytest
import torch

from distllm.dist.multimodal import (
    ModalityDetection,
    ModalityRouter,
    ModalityType,
    MultiModalPipelineConfig,
    VisionEncoder,
    VisionPipelineConfig,
    build_projection_layer,
    detect_modality,
    get_modality_router,
    interleave_multimodal_tokens,
)


class TestModalityType:
    def test_values(self):
        assert ModalityType.TEXT.value == "text"
        assert ModalityType.IMAGE.value == "image"
        assert ModalityType.MULTI_MODAL.value == "multi_modal"
        assert ModalityType.AUDIO.value == "audio"

    def test_enum_members(self):
        assert len(ModalityType) == 4


class TestModalityDetection:
    def test_defaults(self):
        d = ModalityDetection()
        assert d.primary_modality == ModalityType.TEXT
        assert d.has_images is False
        assert d.has_text is True
        assert d.image_count == 0
        assert d.text_length == 0
        assert d.image_urls == []
        assert d.image_sizes == []

    def test_custom_values(self):
        d = ModalityDetection(
            primary_modality=ModalityType.MULTI_MODAL,
            has_images=True,
            has_text=True,
            image_count=3,
            text_length=100,
            image_urls=["http://example.com/img1.png"],
            image_sizes=[(224, 224)],
        )
        assert d.primary_modality == ModalityType.MULTI_MODAL
        assert d.has_images is True
        assert d.image_count == 3
        assert d.image_urls == ["http://example.com/img1.png"]
        assert d.image_sizes == [(224, 224)]


class TestVisionPipelineConfig:
    def test_defaults(self):
        cfg = VisionPipelineConfig()
        assert cfg.vision_model == "openai/clip-vit-large-patch14-336"
        assert cfg.image_size == (336, 336)
        assert cfg.patch_size == 14
        assert cfg.vision_dim == 1024
        assert cfg.num_image_tokens == 576
        assert cfg.dtype == "float16"
        assert cfg.device == "auto"

    def test_custom_values(self):
        cfg = VisionPipelineConfig(
            vision_model="openai/clip-vit-base-patch32",
            image_size=(224, 224),
            patch_size=32,
            vision_dim=768,
            num_image_tokens=49,
            dtype="float32",
            device="cpu",
        )
        assert cfg.vision_model == "openai/clip-vit-base-patch32"
        assert cfg.num_image_tokens == 49
        assert cfg.dtype == "float32"
        assert cfg.device == "cpu"


class TestMultiModalPipelineConfig:
    def test_defaults(self):
        cfg = MultiModalPipelineConfig()
        assert cfg.enabled is False
        assert isinstance(cfg.vision, VisionPipelineConfig)
        assert cfg.projection_type == "linear"
        assert cfg.max_images == 5
        assert cfg.separate_vision_device is True

    def test_enabled_and_custom_vision(self):
        vision = VisionPipelineConfig(device="cpu", dtype="float32")
        cfg = MultiModalPipelineConfig(
            enabled=True,
            vision=vision,
            projection_type="mlp",
            max_images=10,
            separate_vision_device=False,
        )
        assert cfg.enabled is True
        assert cfg.vision is vision
        assert cfg.projection_type == "mlp"
        assert cfg.max_images == 10
        assert cfg.separate_vision_device is False


class TestDetectModality:
    def test_empty_messages(self):
        detection = detect_modality([])
        assert detection.primary_modality == ModalityType.TEXT
        assert detection.has_images is False
        assert detection.has_text is True
        assert detection.text_length == 0

    def test_text_only_string_content(self):
        messages = [{"role": "user", "content": "Hello, world!"}]
        detection = detect_modality(messages)
        assert detection.primary_modality == ModalityType.TEXT
        assert detection.has_images is False
        assert detection.has_text is True
        assert detection.text_length == 13

    def test_text_only_multiple_messages(self):
        messages = [
            {"role": "user", "content": "First message"},
            {"role": "assistant", "content": "Second message"},
        ]
        detection = detect_modality(messages)
        assert detection.primary_modality == ModalityType.TEXT
        assert detection.text_length == 27  # 13 + 14

    def test_empty_string_content(self):
        messages = [{"role": "user", "content": ""}]
        detection = detect_modality(messages)
        assert detection.primary_modality == ModalityType.TEXT
        assert detection.text_length == 0

    def test_whitespace_only_content(self):
        messages = [{"role": "user", "content": "   \n  \t  "}]
        detection = detect_modality(messages)
        assert detection.primary_modality == ModalityType.TEXT
        assert detection.text_length == 0
        assert detection.has_text is True  # default

    def test_image_only_content_block(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "http://example.com/img.png"}},
                ],
            }
        ]
        detection = detect_modality(messages)
        # has_text defaults to True, so combined images+text → MULTI_MODAL
        assert detection.primary_modality == ModalityType.MULTI_MODAL
        assert detection.has_images is True
        assert detection.has_text is True
        assert detection.image_count == 1
        assert detection.image_urls == ["http://example.com/img.png"]

    def test_mixed_text_and_image_content_block(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is in this image?"},
                    {"type": "image_url", "image_url": {"url": "http://example.com/img.png"}},
                ],
            }
        ]
        detection = detect_modality(messages)
        assert detection.primary_modality == ModalityType.MULTI_MODAL
        assert detection.has_images is True
        assert detection.has_text is True
        assert detection.image_count == 1
        assert detection.text_length == 22

    def test_multiple_images(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "http://example.com/a.png"}},
                    {"type": "image_url", "image_url": {"url": "http://example.com/b.png"}},
                    {"type": "image_url", "image_url": {"url": "http://example.com/c.png"}},
                ],
            }
        ]
        detection = detect_modality(messages)
        # has_text defaults to True → MULTI_MODAL (since Detected has_text is never toggled below)
        assert detection.primary_modality == ModalityType.MULTI_MODAL
        assert detection.image_count == 3
        assert len(detection.image_urls) == 3

    def test_text_with_blank_text_block(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "  "},
                    {"type": "image_url", "image_url": {"url": "http://example.com/img.png"}},
                ],
            }
        ]
        detection = detect_modality(messages)
        # Whitespace-only text block does not set has_text from default,
        # but default is True → combined → MULTI_MODAL
        assert detection.primary_modality == ModalityType.MULTI_MODAL
        assert detection.has_text is True
        assert detection.text_length == 0

    def test_missing_content_key(self):
        messages = [{"role": "user"}]
        detection = detect_modality(messages)
        assert detection.primary_modality == ModalityType.TEXT

    def test_content_is_non_string_non_list(self):
        messages = [{"role": "user", "content": 42}]
        detection = detect_modality(messages)
        assert detection.primary_modality == ModalityType.TEXT

    def test_image_url_as_object_with_url_attribute(self):
        """Test the hasattr(url_obj, 'url') branch."""
        class URLObject:
            url = "http://example.com/obj.png"

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": URLObject()},
                ],
            }
        ]
        detection = detect_modality(messages)
        assert detection.image_count == 1
        assert detection.image_urls == ["http://example.com/obj.png"]

    def test_image_url_none_value(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": None},
                ],
            }
        ]
        detection = detect_modality(messages)
        assert detection.image_count == 1
        # When image_url is None, the isinstance check fails and hasattr also fails
        # so the url is never appended → image_urls stays []
        assert detection.image_urls == []


class TestVisionEncoder:
    def test_default_construction(self):
        encoder = VisionEncoder()
        assert encoder.is_loaded is False
        assert encoder._device == "cpu"

    def test_custom_config_construction(self):
        cfg = VisionPipelineConfig(device="cpu", dtype="float32")
        encoder = VisionEncoder(config=cfg)
        assert encoder.is_loaded is False
        assert encoder._config.device == "cpu"

    def test_encode_images_returns_none_when_not_loaded(self):
        encoder = VisionEncoder()
        result = encoder.encode_images([])
        assert result is None

    def test_encode_single_image_returns_none_when_not_loaded(self):
        encoder = VisionEncoder()
        result = encoder.encode_single_image(None)
        assert result is None


class TestModalityRouter:
    def test_default_construction(self):
        router = ModalityRouter()
        assert router.has_vision is False

    def test_custom_config_construction(self):
        cfg = VisionPipelineConfig(device="cpu")
        router = ModalityRouter(vision_config=cfg, separate_device=False)
        assert router.has_vision is False
        assert router._vision_config.device == "cpu"
        assert router._separate_device is False

    def test_route_request_text_only(self):
        router = ModalityRouter()
        messages = [{"role": "user", "content": "Hello"}]
        detection, features = router.route_request(messages)
        assert detection.primary_modality == ModalityType.TEXT
        assert detection.has_images is False
        assert features is None

    def test_route_request_with_images_no_vision(self):
        router = ModalityRouter()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "http://example.com/img.png"}},
                ],
            }
        ]
        detection, features = router.route_request(messages)
        # Without vision loaded, features should be None
        # has_text defaults to True, so combined → MULTI_MODAL
        assert detection.primary_modality == ModalityType.MULTI_MODAL
        assert detection.image_count == 1
        assert features is None

    def test_route_request_empty_messages(self):
        router = ModalityRouter()
        detection, features = router.route_request([])
        assert detection.primary_modality == ModalityType.TEXT
        assert features is None

    def test_route_request_mixed_content_no_vision(self):
        router = ModalityRouter()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image"},
                    {"type": "image_url", "image_url": {"url": "http://example.com/img.png"}},
                ],
            }
        ]
        detection, features = router.route_request(messages)
        assert detection.primary_modality == ModalityType.MULTI_MODAL
        assert features is None


class TestBuildProjectionLayer:
    def test_linear_projection(self):
        layer = build_projection_layer(vision_dim=1024, language_dim=4096, projection_type="linear")
        assert isinstance(layer, torch.nn.Linear)
        assert layer.in_features == 1024
        assert layer.out_features == 4096

    def test_mlp_projection(self):
        layer = build_projection_layer(vision_dim=768, language_dim=2048, projection_type="mlp")
        assert isinstance(layer, torch.nn.Sequential)
        assert len(layer) == 3
        assert isinstance(layer[0], torch.nn.Linear)
        assert layer[0].in_features == 768
        assert layer[0].out_features == 2048
        assert isinstance(layer[1], torch.nn.GELU)
        assert isinstance(layer[2], torch.nn.Linear)
        assert layer[2].in_features == 2048
        assert layer[2].out_features == 2048

    def test_unknown_type_falls_back_to_linear(self):
        layer = build_projection_layer(vision_dim=512, language_dim=1024, projection_type="cross_attention")
        assert isinstance(layer, torch.nn.Linear)
        assert layer.in_features == 512
        assert layer.out_features == 1024

    def test_projection_forward(self):
        layer = build_projection_layer(vision_dim=1024, language_dim=4096)
        x = torch.randn(2, 576, 1024)
        out = layer(x)
        assert out.shape == (2, 576, 4096)


class TestInterleaveMultimodalTokens:
    def test_no_images(self):
        text_emb = torch.randn(1, 10, 512)
        img_feats = torch.randn(1, 0, 16, 512)
        result = interleave_multimodal_tokens(text_emb, img_feats, [], num_image_tokens=16)
        assert result.shape == (1, 10, 512)
        assert torch.equal(result, text_emb)

    def test_single_image_at_position_zero(self):
        text_emb = torch.randn(1, 10, 512)
        img_feats = torch.randn(1, 1, 16, 512)
        result = interleave_multimodal_tokens(text_emb, img_feats, [0], num_image_tokens=16)
        assert result.shape == (1, 26, 512)

    def test_single_image_at_end(self):
        text_emb = torch.randn(1, 10, 512)
        img_feats = torch.randn(1, 1, 16, 512)
        result = interleave_multimodal_tokens(text_emb, img_feats, [10], num_image_tokens=16)
        assert result.shape == (1, 26, 512)

    def test_multiple_images(self):
        text_emb = torch.randn(1, 20, 512)
        img_feats = torch.randn(1, 3, 16, 512)
        result = interleave_multimodal_tokens(text_emb, img_feats, [5, 10, 15], num_image_tokens=16)
        assert result.shape == (1, 68, 512)

    def test_batch_size_preserved(self):
        text_emb = torch.randn(4, 10, 256)
        img_feats = torch.randn(4, 1, 8, 256)
        result = interleave_multimodal_tokens(text_emb, img_feats, [5], num_image_tokens=8)
        assert result.shape == (4, 18, 256)

    def test_fewer_patches_than_num_tokens_raises(self):
        """RuntimeError when num_image_tokens exceeds available patches (module limitation)."""
        text_emb = torch.randn(1, 10, 64)
        img_feats = torch.randn(1, 1, 5, 64)
        with pytest.raises(RuntimeError):
            interleave_multimodal_tokens(text_emb, img_feats, [3], num_image_tokens=16)

    def test_num_tokens_matches_patches(self):
        """When num_image_tokens == available patches, it works."""
        text_emb = torch.randn(1, 10, 64)
        img_feats = torch.randn(1, 1, 5, 64)
        result = interleave_multimodal_tokens(text_emb, img_feats, [3], num_image_tokens=5)
        assert result.shape == (1, 15, 64)

    def test_content_preserved_after_interleave(self):
        """Verify text content before and after image insertion is present."""
        text_emb = torch.zeros(1, 10, 4)
        text_emb[:, :5] = 1.0
        text_emb[:, 5:] = 2.0
        img_feats = torch.full((1, 1, 4, 4), 9.0)
        result = interleave_multimodal_tokens(text_emb, img_feats, [5], num_image_tokens=4)
        assert result.shape == (1, 14, 4)
        assert torch.allclose(result[:, :5], torch.full((1, 5, 4), 1.0))
        assert torch.allclose(result[:, 5:9], torch.full((1, 4, 4), 9.0))
        assert torch.allclose(result[:, 9:], torch.full((1, 5, 4), 2.0))

    def test_more_images_than_positions(self):
        """Extra images beyond image_positions should be ignored."""
        text_emb = torch.randn(1, 10, 32)
        img_feats = torch.randn(1, 5, 4, 32)
        result = interleave_multimodal_tokens(text_emb, img_feats, [3], num_image_tokens=4)
        expected_len = 10 + 4
        assert result.shape == (1, expected_len, 32)

    def test_unsorted_positions(self):
        """Positions do not need to be pre-sorted."""
        text_emb = torch.randn(1, 20, 16)
        img_feats = torch.randn(1, 2, 4, 16)
        result = interleave_multimodal_tokens(text_emb, img_feats, [15, 3], num_image_tokens=4)
        expected_len = 20 + 8
        assert result.shape == (1, expected_len, 16)

    def test_different_dtype_and_device(self):
        text_emb = torch.randn(1, 5, 8, dtype=torch.float64)
        img_feats = torch.randn(1, 1, 3, 8, dtype=torch.float64)
        result = interleave_multimodal_tokens(text_emb, img_feats, [2], num_image_tokens=3)
        assert result.dtype == torch.float64
        assert result.device == text_emb.device


class TestGetModalityRouter:
    def test_returns_singleton(self):
        router_a = get_modality_router()
        router_b = get_modality_router()
        assert router_a is router_b

    def test_singleton_ignores_subsequent_config(self):
        router_a = get_modality_router()
        cfg = MultiModalPipelineConfig(enabled=True)
        router_b = get_modality_router(config=cfg)
        assert router_a is router_b

    # Reset the global singleton for other tests that depend on clean state
    def test_reset_between_tests(self):
        """Force a reset by importing and reassigning."""
        import distllm.dist.multimodal as mm

        mm._router = None
        router = get_modality_router()
        assert router is not None
        assert router.has_vision is False
        mm._router = None

    def test_router_with_config(self):
        import distllm.dist.multimodal as mm

        mm._router = None
        cfg = MultiModalPipelineConfig(
            enabled=True,
            vision=VisionPipelineConfig(device="cpu"),
        )
        router = get_modality_router(config=cfg)
        assert router._vision_config.device == "cpu"
        assert router._separate_device is True
        mm._router = None
