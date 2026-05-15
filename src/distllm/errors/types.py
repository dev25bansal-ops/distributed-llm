"""Standardized error hierarchy for distributed LLM inference."""

from typing import Dict, Any, Optional


class DistLLMError(Exception):
    """Base exception for all distributed LLM errors."""

    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.context = context or {}


# Node errors

class NodeError(DistLLMError):
    """Error related to a worker node."""


class NodeUnreachableError(NodeError):
    """A worker node cannot be reached via gRPC."""

    def __init__(self, node_id: str, host: str, port: int, original_error: Optional[Exception] = None):
        message = f"Node {node_id} at {host}:{port} is unreachable"
        super().__init__(message, context={"node_id": node_id, "host": host, "port": port})
        self.node_id = node_id
        self.host = host
        self.port = port
        self.original_error = original_error


class CircuitBreakerError(NodeError):
    """Circuit breaker is open for a node after repeated failures."""

    def __init__(self, node_id: str, failures: int, recovery_in: float):
        message = (
            f"Circuit breaker open for node {node_id} "
            f"after {failures} failures, recovery in {recovery_in:.1f}s"
        )
        super().__init__(message, context={"node_id": node_id, "failures": failures})
        self.node_id = node_id
        self.failures = failures
        self.recovery_in = recovery_in


# Communication errors

class CommunicationError(DistLLMError):
    """Error related to gRPC communication."""


class SerializationError(CommunicationError):
    """Failed to serialize or deserialize protobuf messages."""

    def __init__(self, message: str, field: Optional[str] = None):
        ctx = {"field": field} if field else {}
        super().__init__(message, context=ctx)


class GRPCTimeoutError(CommunicationError):
    """gRPC call timed out."""

    def __init__(self, node_id: str, timeout: float):
        message = f"gRPC call to node {node_id} timed out after {timeout}s"
        super().__init__(message, context={"node_id": node_id, "timeout": timeout})


# Model errors

class ModelError(DistLLMError):
    """Error related to model loading or inference."""


class ModelNotFoundError(ModelError):
    """Model not found on HuggingFace or local filesystem."""

    def __init__(self, model_name: str):
        message = f"Model '{model_name}' not found"
        super().__init__(message, context={"model_name": model_name})
        self.model_name = model_name


class ModelLoadError(ModelError):
    """Failed to load model weights or architecture."""

    def __init__(self, model_name: str, reason: str):
        message = f"Failed to load model '{model_name}': {reason}"
        super().__init__(message, context={"model_name": model_name})
        self.model_name = model_name


# Config errors

class ConfigError(DistLLMError):
    """Error related to configuration."""


class ConfigValidationError(ConfigError):
    """Configuration validation failed."""

    def __init__(self, field: str, message: str):
        super().__init__(f"Config validation error for '{field}': {message}", context={"field": field})
        self.field = field


# Batch errors

class BatchError(DistLLMError):
    """Error related to batch processing."""


class BatchCapacityError(BatchError):
    """Batch capacity exceeded."""

    def __init__(self, current_tokens: int, max_tokens: int):
        message = f"Batch would exceed capacity: {current_tokens} > {max_tokens} tokens"
        super().__init__(message, context={"current_tokens": current_tokens, "max_tokens": max_tokens})


# Constraint errors

class ConstraintError(DistLLMError):
    """Error related to output constraints (e.g., JSON schema)."""


class ConstraintViolationError(ConstraintError):
    """Generated output violates the specified constraint."""

    def __init__(self, constraint_type: str, detail: str):
        message = f"Constraint '{constraint_type}' violated: {detail}"
        super().__init__(message, context={"constraint_type": constraint_type})
        self.constraint_type = constraint_type


# GPU/Memory errors

class OOMError(ModelError):
    """GPU out of memory during inference."""

    def __init__(self, node_id: str, detail: str = ""):
        message = f"GPU out of memory on node {node_id}"
        if detail:
            message += f": {detail}"
        super().__init__(message, context={"node_id": node_id})
        self.node_id = node_id


# Input validation errors

class InputValidationError(DistLLMError):
    """Input data failed validation."""

    def __init__(self, detail: str, field: str = ""):
        message = f"Invalid input: {detail}"
        ctx = {"field": field} if field else {}
        super().__init__(message, context=ctx)


# Protocol errors

class ProtoError(CommunicationError):
    """Protocol buffer encoding/decoding error."""

    def __init__(self, message: str, field: str = ""):
        ctx = {"field": field} if field else {}
        super().__init__(message, context=ctx)
