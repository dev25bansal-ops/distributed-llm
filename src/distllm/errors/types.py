"""Standardized error hierarchy for distributed LLM inference.

Every error carries a unique ``code`` string for programmatic handling,
an optional ``context`` dict for structured logging, and a
``troubleshooting_url`` linking to the relevant docs section.

Error Code Reference: https://distllm.dev/docs/errors
Troubleshooting: https://distllm.dev/docs/troubleshooting
"""

from typing import Dict, Any, Optional

# Base URL for error documentation
_DOCS_BASE = "https://distllm.dev/docs/troubleshooting"


class DistLLMError(Exception):
    """Base exception for all distributed LLM errors."""

    code: str = "DISTLLM_ERROR"
    troubleshooting_section: str = ""

    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.context = context or {}

    @property
    def troubleshooting_url(self) -> str:
        """Link to the troubleshooting docs for this error."""
        if self.troubleshooting_section:
            return f"{_DOCS_BASE}#{self.troubleshooting_section}"
        return _DOCS_BASE

    def to_dict(self) -> dict:
        """Serialize error for API responses."""
        return {
            "error": {
                "code": self.code,
                "message": self.message,
                "troubleshooting_url": self.troubleshooting_url,
                "context": self.context,
            }
        }


# Node errors

class NodeError(DistLLMError):
    """Error related to a worker node."""
    code = "NODE_ERROR"
    troubleshooting_section = "3-distributed-pipeline-errors"


class NodeUnreachableError(NodeError):
    """A worker node cannot be reached via gRPC."""
    code = "NODE_UNREACHABLE"
    troubleshooting_section = "3-distributed-pipeline-errors"

    def __init__(self, node_id: str, host: str, port: int, original_error: Optional[Exception] = None):
        message = f"Node {node_id} at {host}:{port} is unreachable"
        super().__init__(message, context={"node_id": node_id, "host": host, "port": port})
        self.node_id = node_id
        self.host = host
        self.port = port
        self.original_error = original_error


class CircuitBreakerError(NodeError):
    """Circuit breaker is open for a node after repeated failures."""
    code = "CIRCUIT_BREAKER_OPEN"
    troubleshooting_section = "3-distributed-pipeline-errors"

    def __init__(self, node_id: str, failures: int, recovery_in: float):
        message = (
            f"Circuit breaker open for node {node_id} "
            f"after {failures} failures, recovery in {recovery_in:.1f}s"
        )
        super().__init__(message, context={"node_id": node_id, "failures": failures})
        self.node_id = node_id
        self.failures = failures
        self.recovery_in = recovery_in


# Gateway errors

class GatewayError(DistLLMError):
    """Error from the multi-provider gateway (routing, fallback, upstream)."""
    code = "GATEWAY_ERROR"
    troubleshooting_section = "4-api-errors"

    def __init__(self, provider: str, model: str, status_code: int = 0, detail: str = ""):
        message = f"Gateway error from {provider} for {model}"
        if detail:
            message += f": {detail}"
        super().__init__(message, context={"provider": provider, "model": model, "status_code": status_code})
        self.provider = provider
        self.model = model
        self.status_code = status_code


class ProviderTimeoutError(GatewayError):
    """Upstream provider request timed out."""
    code = "PROVIDER_TIMEOUT"
    troubleshooting_section = "4-api-errors"


# Communication errors

class CommunicationError(DistLLMError):
    """Error related to inter-node communication."""
    code = "COMMUNICATION_ERROR"
    troubleshooting_section = "3-distributed-pipeline-errors"


class SerializationError(CommunicationError):
    """Failed to serialize or deserialize protobuf messages."""
    code = "SERIALIZATION_ERROR"

    def __init__(self, message: str, field: Optional[str] = None):
        ctx = {"field": field} if field else {}
        super().__init__(message, context=ctx)


class GRPCTimeoutError(CommunicationError):
    """gRPC call timed out."""
    code = "GRPC_TIMEOUT"

    def __init__(self, node_id: str, timeout: float, host: str | None = None, port: int | None = None):
        target = f"{node_id} at {host}:{port}" if host is not None and port is not None else node_id
        message = f"gRPC call to node {target} timed out after {timeout}s"
        context = {"node_id": node_id, "timeout": timeout}
        if host is not None:
            context["host"] = host
        if port is not None:
            context["port"] = port
        super().__init__(message, context=context)
        self.node_id = node_id
        self.timeout = timeout
        self.host = host
        self.port = port


# Model errors

class ModelError(DistLLMError):
    """Error related to model loading or inference."""
    code = "MODEL_ERROR"
    troubleshooting_section = "2-model-loading-failures"


class ModelNotFoundError(ModelError):
    """Model not found on HuggingFace or local filesystem."""
    code = "MODEL_NOT_FOUND"
    troubleshooting_section = "2-model-loading-failures"

    def __init__(self, model_name: str):
        message = f"Model '{model_name}' not found"
        super().__init__(message, context={"model_name": model_name})
        self.model_name = model_name


class ModelLoadError(ModelError):
    """Failed to load model weights or architecture."""
    code = "MODEL_LOAD_ERROR"
    troubleshooting_section = "2-model-loading-failures"

    def __init__(self, model_name: str, reason: str):
        message = f"Failed to load model '{model_name}': {reason}"
        super().__init__(message, context={"model_name": model_name})
        self.model_name = model_name


# Config errors

class ConfigError(DistLLMError):
    """Error related to configuration."""
    code = "CONFIG_ERROR"
    troubleshooting_section = "1-installation-issues"


class ConfigValidationError(ConfigError):
    """Configuration validation failed."""
    code = "CONFIG_VALIDATION_ERROR"
    troubleshooting_section = "1-installation-issues"

    def __init__(self, field: str, message: str):
        super().__init__(f"Config validation error for '{field}': {message}", context={"field": field})
        self.field = field


# Batch errors

class BatchError(DistLLMError):
    """Error related to batch processing."""
    code = "BATCH_ERROR"


class BatchCapacityError(BatchError):
    """Batch capacity exceeded."""
    code = "BATCH_CAPACITY_ERROR"

    def __init__(self, current_tokens: int, max_tokens: int):
        message = f"Batch would exceed capacity: {current_tokens} > {max_tokens} tokens"
        super().__init__(message, context={"current_tokens": current_tokens, "max_tokens": max_tokens})


# Constraint errors

class ConstraintError(DistLLMError):
    """Error related to output constraints (e.g., JSON schema)."""
    code = "CONSTRAINT_ERROR"


class ConstraintViolationError(ConstraintError):
    """Generated output violates the specified constraint."""
    code = "CONSTRAINT_VIOLATION"

    def __init__(self, constraint_type: str, detail: str):
        message = f"Constraint '{constraint_type}' violated: {detail}"
        super().__init__(message, context={"constraint_type": constraint_type})
        self.constraint_type = constraint_type


# GPU/Memory errors

class OOMError(ModelError):
    """GPU out of memory during inference."""
    code = "OOM_ERROR"
    troubleshooting_section = "6-gpu-issues"

    def __init__(self, node_id: str, detail: str = ""):
        message = f"GPU out of memory on node {node_id}"
        if detail:
            message += f": {detail}"
        super().__init__(message, context={"node_id": node_id})
        self.node_id = node_id


# Input validation errors

class InputValidationError(DistLLMError):
    """Input data failed validation."""
    code = "INPUT_VALIDATION_ERROR"

    def __init__(self, detail: str, field: str = ""):
        message = f"Invalid input: {detail}"
        ctx = {"field": field} if field else {}
        super().__init__(message, context=ctx)


# Protocol errors

class ProtoError(CommunicationError):
    """Protocol buffer encoding/decoding error."""
    code = "PROTO_ERROR"

    def __init__(self, message: str, field: str = ""):
        ctx = {"field": field} if field else {}
        super().__init__(message, context=ctx)
