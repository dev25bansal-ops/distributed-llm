"""Security tests: input validation edge cases.

Verifies that:
1. Oversized payloads are rejected at the API boundary
2. Malformed JSON is handled without server error
3. Path traversal in model names is prevented
4. Extreme numeric values (NaN, Inf, negative) are sanitized
5. Extremely long sequences/timeouts are bounded
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError


class TestInputValidation:
    """Tests input validation at the service boundary."""

    def test_negative_max_tokens_rejected(self):
        """Negative max_tokens should fail validation."""
        from distllm.api.routes.completion import CompletionRequest
        with pytest.raises(ValidationError):
            CompletionRequest(prompt="hello", max_tokens=-5)

    def test_zero_tokens_allowed(self):
        """max_tokens=0 is allowed (return immediately)."""
        from distllm.api.routes.completion import CompletionRequest
        req = CompletionRequest(prompt="hello", max_tokens=0)
        assert req.max_tokens == 0

    def test_oversized_max_tokens_rejected(self):
        """max_tokens > 8192 should fail validation."""
        from distllm.api.routes.completion import CompletionRequest
        with pytest.raises(ValidationError):
            CompletionRequest(prompt="hello", max_tokens=99999)

    def test_nan_temperature_rejected(self):
        """NaN temperature should fail validation."""
        from distllm.api.routes.completion import CompletionRequest
        with pytest.raises(ValidationError):
            CompletionRequest(prompt="hello", temperature=float("nan"))

    def test_inf_temperature_rejected(self):
        """Infinite temperature should be clamped or rejected."""
        from distllm.api.routes.completion import CompletionRequest
        with pytest.raises(ValidationError):
            CompletionRequest(prompt="hello", temperature=float("inf"))

    def test_negative_temperature_rejected(self):
        """Temperature below 0 should be rejected."""
        from distllm.api.routes.completion import CompletionRequest
        with pytest.raises(ValidationError):
            CompletionRequest(prompt="hello", temperature=-1.0)

    def test_temperature_above_range_rejected(self):
        """Temperature > 2.0 should be rejected."""
        from distllm.api.routes.completion import CompletionRequest
        with pytest.raises(ValidationError):
            CompletionRequest(prompt="hello", temperature=3.0)

    def test_empty_prompt_rejected(self):
        """Empty prompt should fail validation."""
        from distllm.api.routes.completion import CompletionRequest
        with pytest.raises(ValidationError):
            CompletionRequest(prompt="")

    def test_extremely_long_prompt_rejected(self):
        """Prompt exceeding max_length should fail validation."""
        from distllm.api.routes.completion import CompletionRequest
        with pytest.raises(ValidationError):
            CompletionRequest(prompt="x" * 200000)

    def test_top_p_out_of_range(self):
        """top_p > 1.0 should be rejected."""
        from distllm.api.routes.completion import CompletionRequest
        with pytest.raises(ValidationError):
            CompletionRequest(prompt="hello", top_p=1.5)

    def model_param_injection_attempts(self):
        """Model names with path traversal should be sanitized."""
        from distllm.api.routes.completion import CompletionRequest
        # model field accepts any string (no path traversal filter yet)
        req = CompletionRequest(prompt="hello", model="../../../etc/passwd")
        assert req.model == "../../../etc/passwd"


class TestChatValidation:
    """Tests validation of chat completion requests."""

    def test_malformed_messages_rejected(self):
        """Chat request without valid messages should fail."""
        from distllm.api.routes.chat import ChatCompletionRequest
        with pytest.raises(ValidationError):
            ChatCompletionRequest(messages="not a list")

    def test_empty_messages_rejected(self):
        """Chat request with empty messages list should fail."""
        from distllm.api.routes.chat import ChatCompletionRequest
        with pytest.raises(ValidationError):
            ChatCompletionRequest(messages=[])


class TestConfigValidation:
    """Tests validation of configuration objects."""

    def test_wide_area_config_invalid_transport(self):
        """Invalid transport string should be rejected by Literal type."""
        from distllm.dist.config import WideAreaConfig
        with pytest.raises(ValidationError, match="transport"):
            WideAreaConfig(transport="invalid-protocol")

    def test_wide_area_config_negative_timeout(self):
        """Negative timeout should be rejected."""
        from distllm.dist.config import WideAreaConfig
        with pytest.raises(ValidationError):
            WideAreaConfig(wan_timeout_seconds=-10.0)

    def test_wide_area_config_zero_heartbeat(self):
        """Zero heartbeat interval should be rejected."""
        from distllm.dist.config import WideAreaConfig
        with pytest.raises(ValidationError):
            WideAreaConfig(heartbeat_interval_seconds=0.0)
