"""Standardized error hierarchy for distributed LLM inference.

Every error carries a unique ``code`` string for programmatic handling,
an optional ``context`` dict for structured logging, a user-friendly
``user_message`` and ``remediation_hint``, and a ``docs_url`` linking to
the relevant documentation section.

Error Code Reference: https://distllm.dev/docs/errors
Troubleshooting: https://distllm.dev/docs/troubleshooting
"""

from typing import Any

# Base URL for error documentation
_DOCS_BASE = "https://distllm.dev/docs/troubleshooting"


class DistLLMError(Exception):
    """Base exception for all distributed LLM errors.

    Attributes:
        message: Technical error message for logs.
        user_message: User-friendly message for CLI/API display.
        remediation_hint: What the user can do to fix the issue.
        docs_url: Link to relevant documentation.
        context: Structured dict for logging/tracing.
        code: Programmatic error code string.
    """

    code: str = "DISTLLM_ERROR"
    troubleshooting_section: str = ""

    def __init__(
        self,
        message: str,
        context: dict[str, Any] | None = None,
        *,
        user_message: str | None = None,
        remediation_hint: str | None = None,
        docs_url: str | None = None,
    ):
        super().__init__(message)
        self.message = message
        self.user_message = user_message or message
        self.remediation_hint = remediation_hint or ""
        self.context = context or {}
        self.docs_url = docs_url or self.troubleshooting_url

    @property
    def troubleshooting_url(self) -> str:
        """Link to the troubleshooting docs for this error."""
        if self.troubleshooting_section:
            return f"{_DOCS_BASE}#{self.troubleshooting_section}"
        return _DOCS_BASE

    def to_dict(self) -> dict[str, Any]:
        """Serialize error for API responses."""
        return {
            "error": {
                "code": self.code,
                "message": self.message,
                "user_message": self.user_message,
                "remediation_hint": self.remediation_hint,
                "troubleshooting_url": self.docs_url,
                "docs_url": self.docs_url,
                "context": self.context,
            }
        }

    def __str__(self) -> str:
        return f"[{self.code}] {self.message}"

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"code={self.code!r}, "
            f"message={self.message!r}, "
            f"user_message={self.user_message!r})"
        )


# ═══════════════════════════════════════════════════════════════════════
# Node errors
# ═══════════════════════════════════════════════════════════════════════

class NodeError(DistLLMError):
    """Error related to a worker node."""
    code = "NODE_ERROR"
    troubleshooting_section = "3-distributed-pipeline-errors"


class NodeUnreachableError(NodeError):
    """A worker node cannot be reached via gRPC."""
    code = "NODE_UNREACHABLE"
    troubleshooting_section = "3-distributed-pipeline-errors"

    def __init__(
        self,
        node_id: str,
        host: str,
        port: int,
        original_error: Exception | None = None,
    ):
        message = f"Node {node_id} at {host}:{port} is unreachable"
        super().__init__(
            message,
            context={"node_id": node_id, "host": host, "port": port},
            user_message=f"Cannot reach worker node '{node_id}' at {host}:{port}.",
            remediation_hint=(
                "Check that the worker node is running and reachable on the network. "
                "Verify firewall rules and that the port is open."
            ),
        )
        self.node_id = node_id
        self.host = host
        self.port = port
        self.original_error = original_error


class NodeTimeoutError(NodeError):
    """A worker node did not respond within the timeout period."""
    code = "NODE_TIMEOUT"
    troubleshooting_section = "3-distributed-pipeline-errors"

    def __init__(self, node_id: str, timeout: float, operation: str = "inference"):
        message = f"Node {node_id} timed out during {operation} after {timeout}s"
        super().__init__(
            message,
            context={"node_id": node_id, "timeout": timeout, "operation": operation},
            user_message=f"Worker node '{node_id}' did not respond within {timeout}s.",
            remediation_hint=(
                "The node may be overloaded or the network may be slow. "
                "Try increasing the timeout or reducing the request size."
            ),
        )
        self.node_id = node_id
        self.timeout = timeout
        self.operation = operation


class NodeOOMError(NodeError):
    """GPU out of memory on a worker node."""
    code = "NODE_OOM"
    troubleshooting_section = "6-gpu-issues"

    def __init__(self, node_id: str, detail: str = ""):
        message = f"GPU out of memory on node {node_id}"
        if detail:
            message += f": {detail}"
        super().__init__(
            message,
            context={"node_id": node_id},
            user_message=f"Worker node '{node_id}' ran out of GPU memory.",
            remediation_hint=(
                "Reduce the model size, batch size, or sequence length. "
                "Try enabling memory optimizations like gradient checkpointing "
                "or offloading layers to CPU. Consider using a GPU with more VRAM."
            ),
        )
        self.node_id = node_id


class CircuitBreakerError(NodeError):
    """Circuit breaker is open for a node after repeated failures."""
    code = "CIRCUIT_BREAKER_OPEN"
    troubleshooting_section = "3-distributed-pipeline-errors"

    def __init__(self, node_id: str, failures: int, recovery_in: float):
        message = (
            f"Circuit breaker open for node {node_id} "
            f"after {failures} failures, recovery in {recovery_in:.1f}s"
        )
        super().__init__(
            message,
            context={"node_id": node_id, "failures": failures},
            user_message=f"Worker node '{node_id}' is temporarily unavailable after {failures} failures.",
            remediation_hint=(
                "Wait for the circuit breaker to close (recovery in progress). "
                "Check the node's health and logs to understand why it keeps failing."
            ),
        )
        self.node_id = node_id
        self.failures = failures
        self.recovery_in = recovery_in


# ═══════════════════════════════════════════════════════════════════════
# Model errors
# ═══════════════════════════════════════════════════════════════════════

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
        super().__init__(
            message,
            context={"model_name": model_name},
            user_message=f"Model '{model_name}' could not be found.",
            remediation_hint=(
                "Check the model name for typos. For HuggingFace models, "
                "use the full repository ID (e.g., 'meta-llama/Llama-3.1-8B'). "
                "For local models, verify the path exists and is accessible."
            ),
        )
        self.model_name = model_name


class ModelLoadError(ModelError):
    """Failed to load model weights or architecture."""
    code = "MODEL_LOAD_ERROR"
    troubleshooting_section = "2-model-loading-failures"

    def __init__(self, model_name: str, reason: str):
        message = f"Failed to load model '{model_name}': {reason}"
        super().__init__(
            message,
            context={"model_name": model_name},
            user_message=f"Could not load model '{model_name}'.",
            remediation_hint=(
                f"The model failed to load because: {reason}. "
                "Check that the model is compatible with your hardware and software versions. "
                "Ensure sufficient disk space and memory are available."
            ),
        )
        self.model_name = model_name


class ModelOOMError(ModelError):
    """GPU out of memory during model inference."""
    code = "MODEL_OOM"
    troubleshooting_section = "6-gpu-issues"

    def __init__(self, model_name: str, detail: str = ""):
        message = f"Out of memory while running model {model_name}"
        if detail:
            message += f": {detail}"
        super().__init__(
            message,
            context={"model_name": model_name},
            user_message=f"The model '{model_name}' ran out of GPU memory.",
            remediation_hint=(
                "Reduce the model size, batch size, or sequence length. "
                "Try enabling quantization (INT8/INT4) or memory optimizations. "
                "Consider using a GPU with more VRAM."
            ),
        )
        self.model_name = model_name


# Legacy OOM error: subclasses ModelError (historical hierarchy) but carries
# a node_id — callers construct it as OOMError(node_id, detail=...).
class OOMError(ModelError):
    """GPU out of memory (legacy class: model-hierarchy, node-scoped)."""
    code = "MODEL_OOM"
    troubleshooting_section = "6-gpu-issues"

    def __init__(self, node_id: str, detail: str = ""):
        message = f"Out of memory on node {node_id}"
        if detail:
            message += f": {detail}"
        super().__init__(
            message,
            context={"node_id": node_id},
            user_message=f"Node '{node_id}' ran out of GPU memory.",
            remediation_hint=(
                "Reduce the batch size or sequence length, free GPU memory, "
                "or add another node to the pool."
            ),
        )
        self.node_id = node_id


# ═══════════════════════════════════════════════════════════════════════
# Config errors
# ═══════════════════════════════════════════════════════════════════════

class ConfigError(DistLLMError):
    """Error related to configuration."""
    code = "CONFIG_ERROR"
    troubleshooting_section = "1-installation-issues"


class ConfigValidationError(ConfigError):
    """Configuration validation failed."""
    code = "CONFIG_VALIDATION_ERROR"
    troubleshooting_section = "1-installation-issues"

    def __init__(self, field: str, message: str):
        super().__init__(
            f"Config validation error for '{field}': {message}",
            context={"field": field},
            user_message=f"Configuration error in '{field}'.",
            remediation_hint=(
                f"Check the configuration value for '{field}'. {message} "
                "Refer to the configuration reference for valid values."
            ),
        )
        self.field = field


class ConfigFileNotFoundError(ConfigError):
    """Configuration file not found at the specified path."""
    code = "CONFIG_FILE_NOT_FOUND"
    troubleshooting_section = "1-installation-issues"

    def __init__(self, path: str):
        message = f"Configuration file not found: {path}"
        super().__init__(
            message,
            context={"path": path},
            user_message=f"Configuration file '{path}' was not found.",
            remediation_hint=(
                "Check that the config file path is correct and the file exists. "
                "Use 'distllm config setup' to create a new configuration."
            ),
        )
        self.path = path


# ═══════════════════════════════════════════════════════════════════════
# Network errors
# ═══════════════════════════════════════════════════════════════════════

class NetworkError(DistLLMError):
    """Error related to network communication between services."""
    code = "NETWORK_ERROR"
    troubleshooting_section = "3-distributed-pipeline-errors"


class NetworkTimeoutError(NetworkError):
    """Network request timed out."""
    code = "NETWORK_TIMEOUT"
    troubleshooting_section = "3-distributed-pipeline-errors"

    def __init__(self, host: str, port: int, timeout: float):
        message = f"Network timeout connecting to {host}:{port} after {timeout}s"
        super().__init__(
            message,
            context={"host": host, "port": port, "timeout": timeout},
            user_message=f"Connection to {host}:{port} timed out.",
            remediation_hint=(
                "The target service may be down, overloaded, or unreachable. "
                "Verify the service is running and the network is stable."
            ),
        )
        self.host = host
        self.port = port
        self.timeout = timeout


class ConnectionLostError(NetworkError):
    """Network connection was lost during an operation."""
    code = "CONNECTION_LOST"
    troubleshooting_section = "3-distributed-pipeline-errors"

    def __init__(self, host: str, port: int, operation: str = ""):
        message = f"Connection lost to {host}:{port}"
        if operation:
            message += f" during {operation}"
        super().__init__(
            message,
            context={"host": host, "port": port, "operation": operation},
            user_message=f"Connection to {host}:{port} was lost.",
            remediation_hint=(
                "The network connection may have been interrupted. "
                "Check network stability and that the remote service is still running."
            ),
        )
        self.host = host
        self.port = port
        self.operation = operation


# ═══════════════════════════════════════════════════════════════════════
# Auth errors
# ═══════════════════════════════════════════════════════════════════════

class AuthError(DistLLMError):
    """Error related to authentication or authorization."""
    code = "AUTH_ERROR"
    troubleshooting_section = "4-api-errors"


class AuthenticationError(AuthError):
    """Authentication failed (invalid or missing credentials)."""
    code = "AUTHENTICATION_ERROR"
    troubleshooting_section = "4-api-errors"

    def __init__(self, detail: str = ""):
        message = "Authentication failed"
        if detail:
            message += f": {detail}"
        super().__init__(
            message,
            context={},
            user_message="Authentication failed. Please provide valid credentials.",
            remediation_hint=(
                "Check that your API key or authentication token is correct and not expired. "
                "Set the API_KEY environment variable or pass --api-key."
            ),
        )
        self.detail = detail


class AuthorizationError(AuthError):
    """Authorization failed (insufficient permissions)."""
    code = "AUTHORIZATION_ERROR"
    troubleshooting_section = "4-api-errors"

    def __init__(self, resource: str = "", required_role: str = ""):
        message = "Authorization failed"
        parts: list[str] = []
        if resource:
            parts.append(f"for {resource}")
        if required_role:
            parts.append(f"requires {required_role}")
        if parts:
            message += f" ({', '.join(parts)})"
        super().__init__(
            message,
            context={"resource": resource, "required_role": required_role},
            user_message="You do not have permission to perform this action.",
            remediation_hint=(
                "Contact your administrator to request the required access level. "
                "Verify you are using the correct API key with sufficient permissions."
            ),
        )
        self.resource = resource
        self.required_role = required_role


# ═══════════════════════════════════════════════════════════════════════
# API errors
# ═══════════════════════════════════════════════════════════════════════

class APIError(DistLLMError):
    """Error related to API usage and limits."""
    code = "API_ERROR"
    troubleshooting_section = "4-api-errors"


class RateLimitError(APIError):
    """Request rate limit exceeded."""
    code = "RATE_LIMIT_EXCEEDED"
    troubleshooting_section = "4-api-errors"

    def __init__(self, retry_after: float = 0.0, limit: str = ""):
        message = "Rate limit exceeded"
        if limit:
            message += f" ({limit})"
        super().__init__(
            message,
            context={"retry_after": retry_after, "limit": limit},
            user_message="Too many requests. Please slow down.",
            remediation_hint=(
                f"Wait {retry_after:.0f} seconds before sending another request. "
                "Consider reducing the request rate or increasing your rate limit."
            ),
        )
        self.retry_after = retry_after
        self.limit = limit


class QuotaExceededError(APIError):
    """Usage quota exceeded for the current billing period."""
    code = "QUOTA_EXCEEDED"
    troubleshooting_section = "4-api-errors"

    def __init__(self, limit_name: str = "", limit_value: str = ""):
        message = "Quota exceeded"
        if limit_name:
            message += f": {limit_name}"
        super().__init__(
            message,
            context={"limit_name": limit_name, "limit_value": limit_value},
            user_message="Your usage quota has been exceeded.",
            remediation_hint=(
                "Upgrade your plan to increase your quota, or wait for the quota "
                "to reset at the end of the current billing period."
            ),
        )
        self.limit_name = limit_name
        self.limit_value = limit_value


# ═══════════════════════════════════════════════════════════════════════
# Gateway errors
# ═══════════════════════════════════════════════════════════════════════

class GatewayError(DistLLMError):
    """Error from the multi-provider gateway (routing, fallback, upstream)."""
    code = "GATEWAY_ERROR"
    troubleshooting_section = "4-api-errors"

    def __init__(
        self,
        provider: str,
        model: str,
        status_code: int = 0,
        detail: str = "",
    ):
        message = f"Gateway error from {provider} for {model}"
        if detail:
            message += f": {detail}"
        super().__init__(
            message,
            context={"provider": provider, "model": model, "status_code": status_code},
            user_message=f"Upstream provider '{provider}' returned an error for model '{model}'.",
            remediation_hint=(
                "The upstream provider may be experiencing issues. "
                "Try again later or switch to a different provider."
            ),
        )
        self.provider = provider
        self.model = model
        self.status_code = status_code


class ProviderTimeoutError(GatewayError):
    """Upstream provider request timed out."""
    code = "PROVIDER_TIMEOUT"
    troubleshooting_section = "4-api-errors"


# ═══════════════════════════════════════════════════════════════════════
# Communication errors
# ═══════════════════════════════════════════════════════════════════════

class CommunicationError(DistLLMError):
    """Error related to inter-node communication."""
    code = "COMMUNICATION_ERROR"
    troubleshooting_section = "3-distributed-pipeline-errors"


class SerializationError(CommunicationError):
    """Failed to serialize or deserialize protobuf messages."""
    code = "SERIALIZATION_ERROR"

    def __init__(self, message: str, field: str | None = None):
        ctx: dict[str, Any] = {}
        if field:
            ctx["field"] = field
        super().__init__(
            message,
            context=ctx,
            user_message="A data serialization error occurred.",
            remediation_hint=(
                "Check that the data format is correct and compatible between nodes. "
                "Verify protobuf schemas are in sync across the cluster."
            ),
        )
        self.field = field


class GRPCTimeoutError(CommunicationError):
    """gRPC call timed out."""
    code = "GRPC_TIMEOUT"

    def __init__(
        self,
        node_id: str,
        timeout: float,
        host: str | None = None,
        port: int | None = None,
    ):
        target = f"{node_id} at {host}:{port}" if host is not None and port is not None else node_id
        message = f"gRPC call to node {target} timed out after {timeout}s"
        context: dict[str, Any] = {"node_id": node_id, "timeout": timeout}
        if host is not None:
            context["host"] = host
        if port is not None:
            context["port"] = port
        super().__init__(
            message,
            context=context,
            user_message=f"gRPC call to node '{node_id}' timed out after {timeout}s.",
            remediation_hint=(
                "The node may be overloaded or the network may be slow. "
                "Try increasing the gRPC timeout or reducing the workload."
            ),
        )
        self.node_id = node_id
        self.timeout = timeout
        self.host = host
        self.port = port


class ProtoError(CommunicationError):
    """Protocol buffer encoding/decoding error."""
    code = "PROTO_ERROR"

    def __init__(self, message: str, field: str = ""):
        ctx: dict[str, Any] = {}
        if field:
            ctx["field"] = field
        super().__init__(
            message,
            context=ctx,
            user_message="A protocol buffer error occurred.",
            remediation_hint=(
                "Check that protobuf definitions are compatible across versions. "
                "Ensure all nodes are running the same software version."
            ),
        )
        self.field = field


# ═══════════════════════════════════════════════════════════════════════
# Batch errors
# ═══════════════════════════════════════════════════════════════════════

class BatchError(DistLLMError):
    """Error related to batch processing."""
    code = "BATCH_ERROR"


class BatchCapacityError(BatchError):
    """Batch capacity exceeded."""
    code = "BATCH_CAPACITY_ERROR"

    def __init__(self, current_tokens: int, max_tokens: int):
        message = f"Batch would exceed capacity: {current_tokens} > {max_tokens} tokens"
        super().__init__(
            message,
            context={"current_tokens": current_tokens, "max_tokens": max_tokens},
            user_message="Request exceeds the maximum batch capacity.",
            remediation_hint=(
                "Reduce the batch size or sequence length. "
                "Increase the max_tokens_per_batch setting if your hardware supports it."
            ),
        )
        self.current_tokens = current_tokens
        self.max_tokens = max_tokens


# ═══════════════════════════════════════════════════════════════════════
# HA / leadership errors
# ═══════════════════════════════════════════════════════════════════════

class NotLeaderError(DistLLMError):
    """This coordinator is a HA standby and cannot serve requests.

    Raised when a request reaches a coordinator that is not the elected
    leader in an HA deployment.  Clients should retry against the elected
    leader instead of the standby.
    """
    code = "NOT_LEADER"
    troubleshooting_section = "4-ha-election-errors"

    def __init__(self, leader_id: str | None = None):
        message = (
            "This coordinator is a HA standby and cannot serve requests"
            + (f"; leader is {leader_id}" if leader_id else "")
            + " — retry on the elected leader."
        )
        super().__init__(
            message,
            context={"leader_id": leader_id},
            user_message="This server is a standby; please retry on the leader.",
            remediation_hint="Retry the request against the elected leader coordinator.",
        )
        self.leader_id = leader_id


# ═══════════════════════════════════════════════════════════════════════
# Constraint errors
# ═══════════════════════════════════════════════════════════════════════

class ConstraintError(DistLLMError):
    """Error related to output constraints (e.g., JSON schema)."""
    code = "CONSTRAINT_ERROR"


class ConstraintViolationError(ConstraintError):
    """Generated output violates the specified constraint."""
    code = "CONSTRAINT_VIOLATION"

    def __init__(self, constraint_type: str, detail: str):
        message = f"Constraint '{constraint_type}' violated: {detail}"
        super().__init__(
            message,
            context={"constraint_type": constraint_type},
            user_message=f"Output violates the '{constraint_type}' constraint.",
            remediation_hint=(
                "Relax the constraint or adjust the generation parameters. "
                "Check that the constraint definition is correct."
            ),
        )
        self.constraint_type = constraint_type


# ═══════════════════════════════════════════════════════════════════════
# Input validation errors
# ═══════════════════════════════════════════════════════════════════════

class InputValidationError(DistLLMError):
    """Input data failed validation."""
    code = "INPUT_VALIDATION_ERROR"

    def __init__(self, detail: str, field: str = ""):
        message = f"Invalid input: {detail}"
        ctx: dict[str, Any] = {}
        if field:
            ctx["field"] = field
        super().__init__(
            message,
            context=ctx,
            user_message=(
                f"Invalid input{f' for {field}' if field else ''}: {detail}"
            ),
            remediation_hint="Check your input values and try again.",
        )
