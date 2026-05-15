"""Tests for DistLLM error types and hierarchy."""

import pytest

from distllm.errors.types import (
    DistLLMError,
    NodeError, NodeUnreachableError, CircuitBreakerError,
    CommunicationError, SerializationError, GRPCTimeoutError,
    ModelError, ModelNotFoundError, ModelLoadError,
    ConfigError, ConfigValidationError,
    BatchError, BatchCapacityError,
    ConstraintError, ConstraintViolationError,
    OOMError,
    InputValidationError,
    ProtoError,
)


class TestDistLLMError:
    def test_base_error_message(self):
        err = DistLLMError("something went wrong")
        assert str(err) == "something went wrong"
        assert err.message == "something went wrong"

    def test_base_error_context(self):
        err = DistLLMError("error", context={"key": "value"})
        assert err.context == {"key": "value"}

    def test_base_error_default_empty_context(self):
        err = DistLLMError("error")
        assert err.context == {}


class TestNodeErrors:
    def test_node_error_message(self):
        err = NodeError("node failed")
        assert isinstance(err, DistLLMError)
        assert err.message == "node failed"

    def test_node_unreachable_error_attributes(self):
        err = NodeUnreachableError("node-1", "10.0.0.1", 50051)
        assert err.node_id == "node-1"
        assert err.host == "10.0.0.1"
        assert err.port == 50051
        assert "node-1" in str(err)
        assert "10.0.0.1:50051" in str(err)

    def test_node_unreachable_error_context(self):
        err = NodeUnreachableError("node-1", "10.0.0.1", 50051)
        assert err.context["node_id"] == "node-1"
        assert err.context["host"] == "10.0.0.1"
        assert err.context["port"] == 50051

    def test_node_unreachable_with_original_error(self):
        original = ConnectionError("refused")
        err = NodeUnreachableError("node-1", "10.0.0.1", 50051, original_error=original)
        assert err.original_error is original

    def test_circuit_breaker_error_message(self):
        err = CircuitBreakerError("node-1", failures=5, recovery_in=30.0)
        assert "node-1" in str(err)
        assert "5 failures" in str(err)
        assert "30.0s" in str(err)

    def test_circuit_breaker_error_recovery_time(self):
        err = CircuitBreakerError("node-1", failures=3, recovery_in=60.0)
        assert err.node_id == "node-1"
        assert err.failures == 3
        assert err.recovery_in == 60.0
        assert err.context["node_id"] == "node-1"
        assert err.context["failures"] == 3


class TestCommunicationErrors:
    def test_communication_error_base(self):
        err = CommunicationError("comm failed")
        assert isinstance(err, DistLLMError)

    def test_serialization_error_with_field(self):
        err = SerializationError("invalid data", field="tensor")
        assert err.context["field"] == "tensor"
        assert "invalid data" in str(err)

    def test_serialization_error_without_field(self):
        err = SerializationError("invalid data")
        assert err.context == {}

    def test_grpc_timeout_error_attributes(self):
        err = GRPCTimeoutError("node-1", timeout=5.0)
        assert err.context["node_id"] == "node-1"
        assert err.context["timeout"] == 5.0
        assert "node-1" in str(err)
        assert "5.0s" in str(err)


class TestModelErrors:
    def test_model_not_found_error(self):
        err = ModelNotFoundError("gpt2")
        assert err.model_name == "gpt2"
        assert err.context["model_name"] == "gpt2"
        assert "gpt2" in str(err)

    def test_model_load_error(self):
        err = ModelLoadError("gpt2", reason="disk full")
        assert err.model_name == "gpt2"
        assert err.context["model_name"] == "gpt2"
        assert "gpt2" in str(err)
        assert "disk full" in str(err)


class TestConfigErrors:
    def test_config_validation_error_field(self):
        err = ConfigValidationError("port", "must be positive")
        assert err.field == "port"
        assert err.context["field"] == "port"

    def test_config_validation_error_message(self):
        err = ConfigValidationError("temperature", "out of range")
        assert "temperature" in str(err)
        assert "out of range" in str(err)


class TestBatchErrors:
    def test_batch_error_base(self):
        err = BatchError("batch failed")
        assert isinstance(err, DistLLMError)

    def test_batch_capacity_error_attributes(self):
        err = BatchCapacityError(current_tokens=5000, max_tokens=4096)
        assert err.context["current_tokens"] == 5000
        assert err.context["max_tokens"] == 4096
        assert "5000" in str(err)
        assert "4096" in str(err)


class TestConstraintErrors:
    def test_constraint_violation_error(self):
        err = ConstraintViolationError("json_schema", "missing required field")
        assert err.constraint_type == "json_schema"
        assert err.context["constraint_type"] == "json_schema"
        assert "json_schema" in str(err)
        assert "missing required field" in str(err)


class TestOOMError:
    def test_oom_error_with_detail(self):
        err = OOMError("node-1", detail="tried to allocate 10GB")
        assert err.node_id == "node-1"
        assert err.context["node_id"] == "node-1"
        assert "node-1" in str(err)
        assert "tried to allocate 10GB" in str(err)

    def test_oom_error_without_detail(self):
        err = OOMError("node-2")
        assert err.node_id == "node-2"
        assert ": " not in str(err)


class TestInputValidationError:
    def test_input_validation_error_with_field(self):
        err = InputValidationError("too long", field="prompt")
        assert err.context["field"] == "prompt"
        assert "too long" in str(err)

    def test_input_validation_error_without_field(self):
        err = InputValidationError("invalid value")
        assert err.context == {}


class TestProtoError:
    def test_proto_error_with_field(self):
        err = ProtoError("decode failed", field="hidden_states")
        assert err.context["field"] == "hidden_states"

    def test_proto_error_without_field(self):
        err = ProtoError("encode failed")
        assert err.context == {}


class TestErrorHierarchy:
    def test_node_error_is_distllm_error(self):
        assert issubclass(NodeError, DistLLMError)

    def test_communication_error_is_distllm_error(self):
        assert issubclass(CommunicationError, DistLLMError)

    def test_model_error_is_distllm_error(self):
        assert issubclass(ModelError, DistLLMError)

    def test_config_error_is_distllm_error(self):
        assert issubclass(ConfigError, DistLLMError)

    def test_batch_error_is_distllm_error(self):
        assert issubclass(BatchError, DistLLMError)

    def test_constraint_error_is_distllm_error(self):
        assert issubclass(ConstraintError, DistLLMError)

    def test_node_unreachable_is_node_error(self):
        assert issubclass(NodeUnreachableError, NodeError)

    def test_circuit_breaker_is_node_error(self):
        assert issubclass(CircuitBreakerError, NodeError)

    def test_serialization_is_communication_error(self):
        assert issubclass(SerializationError, CommunicationError)

    def test_grpc_timeout_is_communication_error(self):
        assert issubclass(GRPCTimeoutError, CommunicationError)

    def test_proto_is_communication_error(self):
        assert issubclass(ProtoError, CommunicationError)

    def test_model_not_found_is_model_error(self):
        assert issubclass(ModelNotFoundError, ModelError)

    def test_model_load_is_model_error(self):
        assert issubclass(ModelLoadError, ModelError)

    def test_oom_is_model_error(self):
        assert issubclass(OOMError, ModelError)

    def test_batch_capacity_is_batch_error(self):
        assert issubclass(BatchCapacityError, BatchError)

    def test_constraint_violation_is_constraint_error(self):
        assert issubclass(ConstraintViolationError, ConstraintError)
