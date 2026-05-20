"""Tests for structured output / response_format wiring across all layers.

Tests:
- SchemaConstrainedDecoder.from_response_format() uses JSONSchemaFSM
- request_pipeline generate_async() creates constraint from response_format
- completions route accepts response_format field
- Backward compatibility: json_object, json_schema, grammar, regex types
"""

from contextlib import contextmanager
from unittest.mock import MagicMock, patch
import pytest
import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@contextmanager
def _patch_coord(module, mock_coord):
    """Safely patch g.coordinator (property with no deleter)."""
    original = module.g.coordinator
    try:
        module.g.coordinator = mock_coord
        yield
    finally:
        module.g.coordinator = original


def _make_real_tokenizer():
    """Create a tokenizer that works with TokenIndex.build()."""
    tok = MagicMock()
    tok.vocab_size = 256
    tok.eos_token_id = 1
    tok.pad_token_id = 0
    # TokenIndex.build calls get_vocab() → dict of {token_str: id}
    tok.get_vocab.return_value = {chr(i): i for i in range(32, 128)}
    # decode: if list, return char; if int, return char
    def decode_side_effect(ids, **kw):
        if isinstance(ids, list):
            return chr(ids[0]) if ids else ""
        if isinstance(ids, int):
            return chr(ids)
        if isinstance(ids, torch.Tensor):
            return chr(int(ids.item()))
        return ""
    tok.decode.side_effect = decode_side_effect
    def encode_side_effect(text, **kw):
        tokens = [ord(c) for c in text[:10]]
        if kw.get("return_tensors") == "pt":
            return torch.tensor([tokens])
        return tokens
    tok.encode.side_effect = encode_side_effect
    return tok


def _make_minimal_coord():
    """Create a coordinator mock safe for request_pipeline.generate_async()."""
    coord = MagicMock()
    coord.model_info = None
    coord.tokenizer = _make_real_tokenizer()
    coord.scheduler = MagicMock()
    coord.scheduler.add = MagicMock()
    coord._param_update_channel = MagicMock()
    coord._param_update_channel.register = MagicMock()
    coord._rate_limiter = None
    coord._cache_mgr = MagicMock()
    coord._cache_mgr.lookup_prefix.return_value = (0, None)
    coord._cache_mgr.maybe_chunk.return_value = None
    coord.prefix_cache = None
    coord._request_tracker = MagicMock()
    return coord


# ---------------------------------------------------------------------------
# SchemaConstrainedDecoder → JSONSchemaFSM integration
# ---------------------------------------------------------------------------

class TestSchemaConstrainedDecoder:
    """Verify constraint creation from response_format uses JSONSchemaFSM."""

    @pytest.fixture
    def tokenizer(self):
        return _make_real_tokenizer()

    def test_json_object_creates_constraint(self, tokenizer):
        from distllm.core.constrained_decoder import SchemaConstrainedDecoder
        constraint = SchemaConstrainedDecoder.from_response_format(
            {"type": "json_object"}, tokenizer=tokenizer
        )
        assert constraint is not None
        mask = constraint.get_logits_mask(256)
        assert mask.shape == (256,)
        assert mask.dtype == torch.bool

    def test_json_schema_creates_constraint(self, tokenizer):
        from distllm.core.constrained_decoder import SchemaConstrainedDecoder
        constraint = SchemaConstrainedDecoder.from_response_format(
            {"type": "json_schema", "schema": {"type": "object", "properties": {"name": {"type": "string"}}}},
            tokenizer=tokenizer,
        )
        assert constraint is not None
        mask = constraint.get_logits_mask(256)
        assert mask.shape == (256,)

    def test_grammar_creates_constraint(self, tokenizer):
        from distllm.core.constrained_decoder import SchemaConstrainedDecoder
        constraint = SchemaConstrainedDecoder.from_response_format(
            {"type": "grammar", "grammar": 'root ::= "hello"'},
            tokenizer=tokenizer,
        )
        assert constraint is not None
        mask = constraint.get_logits_mask(256)
        assert mask.shape == (256,)

    def test_regex_creates_constraint(self, tokenizer):
        from distllm.core.constrained_decoder import SchemaConstrainedDecoder
        constraint = SchemaConstrainedDecoder.from_response_format(
            {"type": "regex", "pattern": "[a-z]+"},
            tokenizer=tokenizer,
        )
        assert constraint is not None
        mask = constraint.get_logits_mask(256)
        assert mask.shape == (256,)

    def test_unknown_type_returns_none(self, tokenizer):
        from distllm.core.constrained_decoder import SchemaConstrainedDecoder
        constraint = SchemaConstrainedDecoder.from_response_format(
            {"type": "unknown"}, tokenizer=tokenizer
        )
        assert constraint is None

    def test_none_response_format_returns_none(self, tokenizer):
        from distllm.core.constrained_decoder import SchemaConstrainedDecoder
        constraint = SchemaConstrainedDecoder.from_response_format(
            {}, tokenizer=tokenizer
        )
        assert constraint is None

    def test_mask_evolves_after_brace(self, tokenizer):
        """Verify constraint mask changes after emitting '{'."""
        from distllm.core.constrained_decoder import SchemaConstrainedDecoder
        constraint = SchemaConstrainedDecoder.from_response_format(
            {"type": "json_object"}, tokenizer=tokenizer
        )
        mask_before = constraint.get_logits_mask(256)
        constraint.update("{")
        mask_after = constraint.get_logits_mask(256)
        # After '{', the mask should allow different tokens
        assert mask_after.sum() > 0, "After '{', some tokens should be allowed"

    def test_eos_not_allowed_in_non_accepting_state(self, tokenizer):
        """EOS token should not be allowed when JSON is incomplete."""
        from distllm.core.constrained_decoder import SchemaConstrainedDecoder
        constraint = SchemaConstrainedDecoder.from_response_format(
            {"type": "json_object"}, tokenizer=tokenizer
        )
        mask = constraint.get_logits_mask(256)
        assert not mask[1].item(), "EOS should not be allowed in non-accepting state"


# ---------------------------------------------------------------------------
# request_pipeline generate_async constraint creation
# ---------------------------------------------------------------------------

class TestRequestPipelineConstraint:
    """Verify request_pipeline.generate_async() creates constraints from response_format."""

    def test_response_format_uses_schema_constrained_decoder(self):
        """SchemaConstrainedDecoder should be tried first for response_format."""
        from distllm.core.request_pipeline import RequestPipeline
        from distllm.core.constrained_decoder import ConstrainedConstraint

        coord = _make_minimal_coord()
        pipeline = RequestPipeline(coord)
        pipeline.generate_async(
            "test prompt",
            response_format={"type": "json_object"},
        )
        args, kwargs = coord.scheduler.add.call_args
        seq = args[0]
        assert seq.constraint is not None
        assert isinstance(seq.constraint, ConstrainedConstraint)

    def test_schema_fallback_to_json_constraint(self):
        """If SchemaConstrainedDecoder fails, fall back to JSONSchemaConstraint."""
        from distllm.core.request_pipeline import RequestPipeline
        from distllm.core.structured_output import JSONSchemaConstraint

        coord = _make_minimal_coord()
        pipeline = RequestPipeline(coord)

        with patch('distllm.core.request_pipeline.SchemaConstrainedDecoder.from_response_format', return_value=None):
            pipeline.generate_async(
                "test prompt",
                response_format={"type": "json_object"},
            )
            args, kwargs = coord.scheduler.add.call_args
            seq = args[0]
            assert seq.constraint is not None
            assert isinstance(seq.constraint, JSONSchemaConstraint)

    def test_schema_passthrough_uses_json_constraint(self):
        """Schema (not response_format) should use JSONSchemaConstraint directly."""
        from distllm.core.request_pipeline import RequestPipeline
        from distllm.core.structured_output import JSONSchemaConstraint

        coord = _make_minimal_coord()
        pipeline = RequestPipeline(coord)
        pipeline.generate_async(
            "test prompt",
            schema={"type": "object"},
        )
        args, kwargs = coord.scheduler.add.call_args
        seq = args[0]
        assert seq.constraint is not None
        assert isinstance(seq.constraint, JSONSchemaConstraint)


# ---------------------------------------------------------------------------
# Completions route: response_format field
# ---------------------------------------------------------------------------

class TestCompletionRouteResponseFormat:
    """Verify POST /v1/completions accepts and processes response_format."""

    def test_completion_request_has_response_format_field(self):
        from distllm.api.routes.completion import CompletionRequest
        req = CompletionRequest(
            prompt="Hello",
            response_format={"type": "json_object"},
        )
        assert req.response_format == {"type": "json_object"}

    def test_completion_request_defaults_to_none(self):
        from distllm.api.routes.completion import CompletionRequest
        req = CompletionRequest(prompt="Hello")
        assert req.response_format is None

    def test_completion_request_json_schema_field(self):
        from distllm.api.routes.completion import CompletionRequest
        schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
        req = CompletionRequest(
            prompt="JSON please",
            response_format={"type": "json_schema", "schema": schema},
        )
        assert req.response_format["type"] == "json_schema"
        assert req.response_format["schema"] == schema


# ---------------------------------------------------------------------------
# Streaming: constraint integration (unit-level)
# ---------------------------------------------------------------------------

class TestStreamingConstraint:
    """Verify _generate_tokens handles constraint creation without error."""

    def test_constraint_created_from_response_format(self):
        """Verify constraint is created from response_format."""
        from distllm.core.constrained_decoder import SchemaConstrainedDecoder
        tok = _make_real_tokenizer()
        constraint = SchemaConstrainedDecoder.from_response_format(
            {"type": "json_object"}, tokenizer=tok
        )
        assert constraint is not None
        mask = constraint.get_logits_mask(256)
        assert mask.shape == (256,)
        assert mask.dtype == torch.bool

    def test_constraint_update_and_mask(self):
        """Verify constraint.update() advances the FSM."""
        from distllm.core.constrained_decoder import SchemaConstrainedDecoder
        tok = _make_real_tokenizer()
        constraint = SchemaConstrainedDecoder.from_response_format(
            {"type": "json_object"}, tokenizer=tok
        )
        mask_before = constraint.get_logits_mask(256)
        constraint.update("{")
        mask_after = constraint.get_logits_mask(256)
        assert mask_after.sum() > 0, "After '{', some tokens should be allowed"

    @pytest.mark.asyncio
    async def test_generate_tokens_passes_response_format(self):
        """Verify _generate_tokens passes response_format to the constraint path (local path)."""
        from distllm.api.streaming import _generate_tokens

        mock_model = MagicMock()
        logits = torch.randn(1, 1, 256)
        mock_output = MagicMock()
        mock_output.logits = logits
        mock_output.past_key_values = None
        mock_model.side_effect = lambda *a, **kw: mock_output
        # Provide a device via model parameters
        mock_param = MagicMock()
        mock_param.device = torch.device("cpu")
        mock_model.parameters.return_value = iter([mock_param])

        mock_coord = MagicMock()
        mock_coord.model_name = "test"
        mock_coord.nodes = {}
        mock_coord.node_order = []
        mock_coord.local_partitioner = MagicMock()
        mock_coord.local_partitioner.full_model = mock_model
        mock_coord.tokenizer = _make_real_tokenizer()

        import distllm.api.streaming as streaming_mod
        with _patch_coord(streaming_mod, mock_coord):
            with patch.object(streaming_mod, '_get_token_gen') as mock_tg:
                mock_tg_instance = MagicMock()
                mock_tg.return_value = mock_tg_instance
                mock_tg_instance.sample.return_value = (torch.tensor([42]), None)

                gen = _generate_tokens(
                    "test prompt", "req-1", max_tokens=2,
                    temperature=0.7, top_p=0.9, top_k=0,
                    response_format={"type": "json_object"},
                )
                results = [r async for r in gen]
                assert len(results) == 2

    @pytest.mark.asyncio
    async def test_generate_tokens_no_constraint_no_error(self):
        """Verify _generate_tokens works without response_format (local path)."""
        from distllm.api.streaming import _generate_tokens

        mock_model = MagicMock()
        logits = torch.randn(1, 1, 256)
        mock_output = MagicMock()
        mock_output.logits = logits
        mock_output.past_key_values = None
        mock_model.side_effect = lambda *a, **kw: mock_output
        mock_param = MagicMock()
        mock_param.device = torch.device("cpu")
        mock_model.parameters.return_value = iter([mock_param])

        mock_coord = MagicMock()
        mock_coord.model_name = "test"
        mock_coord.nodes = {}
        mock_coord.node_order = []
        mock_coord.local_partitioner = MagicMock()
        mock_coord.local_partitioner.full_model = mock_model
        mock_coord.tokenizer = _make_real_tokenizer()

        import distllm.api.streaming as streaming_mod
        with _patch_coord(streaming_mod, mock_coord):
            with patch.object(streaming_mod, '_get_token_gen') as mock_tg:
                mock_tg_instance = MagicMock()
                mock_tg.return_value = mock_tg_instance
                mock_tg_instance.sample.return_value = (torch.tensor([42]), None)

                gen = _generate_tokens(
                    "test", "req-1", max_tokens=3,
                    temperature=0.7, top_p=0.9, top_k=0,
                )
                results = [r async for r in gen]
                assert len(results) == 3


# ---------------------------------------------------------------------------
# Chat route: response_format passthrough to _stream_response
# ---------------------------------------------------------------------------

class TestChatRouteStreamingPassthrough:
    """Verify chat streaming passes response_format to _stream_response."""

    def test_chat_route_passes_response_format(self):
        from distllm.api.routes.chat import ChatCompletionRequest
        import distllm.api.routes.chat as chat_mod

        mock_coord = MagicMock()
        mock_coord.model_name = "test"
        mock_coord.nodes = {}
        mock_coord.node_order = []
        mock_coord.scheduler = None
        mock_coord.tokenizer = _make_real_tokenizer()
        mock_coord.list_models = MagicMock(return_value=["test"])
        # Avoid the multi-modal pipeline path
        mock_coord._vlm_pipeline = None

        with _patch_coord(chat_mod, mock_coord):
            with patch.object(chat_mod, '_stream_response') as mock_stream:
                mock_request = MagicMock()
                mock_request.state.model = "test"
                mock_request.state.tenant = "default"

                body = ChatCompletionRequest(
                    messages=[{"role": "user", "content": "Hi"}],
                    stream=True,
                    response_format={"type": "json_object"},
                )
                import asyncio
                asyncio.run(chat_mod.chat_completions(mock_request, body))
                mock_stream.assert_called_once()
                call_kwargs = mock_stream.call_args[1]
                assert call_kwargs.get("response_format") == {"type": "json_object"}


# ---------------------------------------------------------------------------
# Completion route: response_format passthrough to generate_async
# ---------------------------------------------------------------------------

class TestCompletionRoutePassthrough:
    """Verify completions handler passes response_format to generate_async."""

    @pytest.mark.asyncio
    async def test_completion_passes_response_format(self):
        from distllm.api.routes.completion import CompletionRequest
        import distllm.api.routes.completion as compl_mod

        mock_coord = MagicMock()
        mock_coord.model_name = "test"
        mock_coord.nodes = {}
        mock_coord.node_order = []
        mock_coord.scheduler = MagicMock()
        mock_coord.generate_async.return_value = "req-123"
        mock_coord.wait_for_result.return_value = '{"answer": "42"}'
        mock_coord.tokenizer = _make_real_tokenizer()

        with _patch_coord(compl_mod, mock_coord):
            mock_request = MagicMock()
            mock_request.state.model = "test"
            mock_request.state.tenant = "default"

            body = CompletionRequest(
                prompt="Return JSON",
                response_format={"type": "json_schema", "schema": {"type": "object"}},
                max_tokens=50,
            )
            result = await compl_mod.completions(mock_request, body)
            mock_coord.generate_async.assert_called_once()
            call_kwargs = mock_coord.generate_async.call_args[1]
            assert call_kwargs["response_format"] == {"type": "json_schema", "schema": {"type": "object"}}
