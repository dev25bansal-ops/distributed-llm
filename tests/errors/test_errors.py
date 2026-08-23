"""Tests for distllm.errors -- types, retry, and policies.

No MagicMock -- real callable functions and closures for retry tests.
"""

from __future__ import annotations

import asyncio

import pytest

from distllm.errors.types import (
    DistLLMError,
    NodeError,
    NodeUnreachableError,
    CircuitBreakerError,
    GatewayError,
    ProviderTimeoutError,
    CommunicationError,
    SerializationError,
    GRPCTimeoutError,
    ProtoError,
    ModelError,
    ModelNotFoundError,
    ModelLoadError,
    OOMError,
    ConfigError,
    ConfigValidationError,
    BatchError,
    BatchCapacityError,
    ConstraintError,
    ConstraintViolationError,
    InputValidationError,
)

from distllm.errors.retry import RetryPolicy, with_retry, with_retry_async, retry_grpc_call

from distllm.errors.policies import (
    ERROR_RETRY_POLICIES,
    get_retry_policy,
    should_retry,
    get_retry_delay,
)


# ============================================================================
# ERROR TYPES
# ============================================================================


class TestDistLLMErrorBase:
    """DistLLMError base class behaviour."""

    def test_message(self):
        err = DistLLMError("something went wrong")
        assert str(err) == "something went wrong"
        assert err.message == "something went wrong"

    def test_default_empty_context(self):
        err = DistLLMError("error")
        assert err.context == {}

    def test_context(self):
        err = DistLLMError("error", context={"key": "value"})
        assert err.context == {"key": "value"}

    def test_code_attribute(self):
        err = DistLLMError("msg")
        assert err.code == "DISTLLM_ERROR"

    def test_troubleshooting_url_no_section(self):
        err = DistLLMError("msg")
        assert err.troubleshooting_url == "https://distllm.dev/docs/troubleshooting"

    def test_troubleshooting_url_with_section(self):
        err = ConfigError("msg")
        assert err.troubleshooting_section == "1-installation-issues"
        assert err.troubleshooting_url.endswith("#1-installation-issues")

    def test_to_dict(self):
        err = DistLLMError("test msg", context={"node": "n1"})
        d = err.to_dict()
        assert d["error"]["code"] == "DISTLLM_ERROR"
        assert d["error"]["message"] == "test msg"
        assert d["error"]["context"] == {"node": "n1"}
        assert "troubleshooting_url" in d["error"]

    def test_is_exception(self):
        assert issubclass(DistLLMError, Exception)


class TestNodeErrors:
    def test_node_error(self):
        err = NodeError("node failed")
        assert isinstance(err, DistLLMError)
        assert err.code == "NODE_ERROR"
        assert err.troubleshooting_section == "3-distributed-pipeline-errors"

    def test_node_unreachable_defaults(self):
        err = NodeUnreachableError("node-1", "10.0.0.1", 50051)
        assert err.node_id == "node-1"
        assert err.host == "10.0.0.1"
        assert err.port == 50051
        assert err.original_error is None
        assert err.code == "NODE_UNREACHABLE"
        assert "node-1" in str(err)
        assert "10.0.0.1:50051" in str(err)

    def test_node_unreachable_with_original(self):
        original = ConnectionError("refused")
        err = NodeUnreachableError("node-1", "10.0.0.1", 50051, original_error=original)
        assert err.original_error is original

    def test_node_unreachable_context(self):
        err = NodeUnreachableError("n1", "1.2.3.4", 8080)
        assert err.context["node_id"] == "n1"
        assert err.context["host"] == "1.2.3.4"
        assert err.context["port"] == 8080

    def test_circuit_breaker(self):
        err = CircuitBreakerError("node-1", failures=5, recovery_in=30.0)
        assert err.node_id == "node-1"
        assert err.failures == 5
        assert err.recovery_in == 30.0
        assert err.code == "CIRCUIT_BREAKER_OPEN"
        assert "5 failures" in str(err)
        assert "30.0s" in str(err)
        assert err.context["failures"] == 5


class TestGatewayErrors:
    def test_gateway_error_defaults(self):
        err = GatewayError("openai", "gpt-4")
        assert err.provider == "openai"
        assert err.model == "gpt-4"
        assert err.status_code == 0
        assert err.code == "GATEWAY_ERROR"

    def test_gateway_error_with_detail(self):
        err = GatewayError("anthropic", "claude-3", status_code=429, detail="rate limited")
        assert err.status_code == 429
        assert "rate limited" in str(err)

    def test_provider_timeout(self):
        err = ProviderTimeoutError("openai", "gpt-4")
        assert isinstance(err, GatewayError)
        assert err.code == "PROVIDER_TIMEOUT"
        assert err.troubleshooting_section == "4-api-errors"


class TestCommunicationErrors:
    def test_communication_error(self):
        err = CommunicationError("comm broke")
        assert isinstance(err, DistLLMError)
        assert err.code == "COMMUNICATION_ERROR"

    def test_serialization_error_with_field(self):
        err = SerializationError("bad data", field="tensor")
        assert err.context["field"] == "tensor"
        assert "bad data" in str(err)

    def test_serialization_error_without_field(self):
        err = SerializationError("bad data")
        assert err.context == {}

    def test_grpc_timeout_node_only(self):
        err = GRPCTimeoutError("node-1", timeout=5.0)
        assert err.node_id == "node-1"
        assert err.timeout == 5.0
        assert err.host is None
        assert err.port is None
        assert err.code == "GRPC_TIMEOUT"
        assert "node-1" in str(err)
        assert "5.0s" in str(err)

    def test_grpc_timeout_with_host_port(self):
        err = GRPCTimeoutError("node-1", timeout=2.5, host="10.0.0.1", port=50051)
        assert err.host == "10.0.0.1"
        assert err.port == 50051
        assert "10.0.0.1:50051" in str(err)

    def test_proto_error_with_field(self):
        err = ProtoError("decode fail", field="logits")
        assert err.context["field"] == "logits"
        assert err.code == "PROTO_ERROR"

    def test_proto_error_without_field(self):
        err = ProtoError("encode fail")
        assert err.context == {}


class TestModelErrors:
    def test_model_error(self):
        err = ModelError("model fail")
        assert isinstance(err, DistLLMError)
        assert err.code == "MODEL_ERROR"

    def test_model_not_found(self):
        err = ModelNotFoundError("gpt2")
        assert err.model_name == "gpt2"
        assert err.code == "MODEL_NOT_FOUND"
        assert "gpt2" in str(err)

    def test_model_load_error(self):
        err = ModelLoadError("gpt2", reason="disk full")
        assert err.model_name == "gpt2"
        assert err.code == "MODEL_LOAD_ERROR"
        assert "disk full" in str(err)

    def test_oom_with_detail(self):
        err = OOMError("node-1", detail="tried to allocate 10GB")
        assert err.node_id == "node-1"
        assert err.code == "OOM_ERROR"
        assert "10GB" in str(err)

    def test_oom_without_detail(self):
        err = OOMError("node-2")
        assert err.node_id == "node-2"
        assert ": " not in str(err)


class TestConfigErrors:
    def test_config_error(self):
        err = ConfigError("bad config")
        assert isinstance(err, DistLLMError)
        assert err.code == "CONFIG_ERROR"

    def test_config_validation_error(self):
        err = ConfigValidationError("port", "must be positive")
        assert err.field == "port"
        assert err.code == "CONFIG_VALIDATION_ERROR"
        assert "port" in str(err)
        assert "must be positive" in str(err)


class TestBatchErrors:
    def test_batch_error(self):
        err = BatchError("batch broke")
        assert isinstance(err, DistLLMError)
        assert err.code == "BATCH_ERROR"

    def test_batch_capacity_error(self):
        err = BatchCapacityError(current_tokens=5000, max_tokens=4096)
        assert err.context["current_tokens"] == 5000
        assert err.context["max_tokens"] == 4096
        assert "5000" in str(err)


class TestConstraintErrors:
    def test_constraint_error(self):
        err = ConstraintError("constraint fail")
        assert isinstance(err, DistLLMError)
        assert err.code == "CONSTRAINT_ERROR"

    def test_constraint_violation_error(self):
        err = ConstraintViolationError("json_schema", "missing required field")
        assert err.constraint_type == "json_schema"
        assert err.code == "CONSTRAINT_VIOLATION"
        assert "missing required field" in str(err)


class TestInputValidationError:
    def test_with_field(self):
        err = InputValidationError("too long", field="prompt")
        assert err.context["field"] == "prompt"
        assert err.code == "INPUT_VALIDATION_ERROR"
        assert "too long" in str(err)

    def test_without_field(self):
        err = InputValidationError("invalid value")
        assert err.context == {}


class TestErrorHierarchy:
    """Verify every error is a DistLLMError and has correct parent."""

    def test_base(self):
        assert issubclass(DistLLMError, Exception)

    @pytest.mark.parametrize(
        "cls,parent",
        [
            (NodeError, DistLLMError),
            (NodeUnreachableError, NodeError),
            (CircuitBreakerError, NodeError),
            (GatewayError, DistLLMError),
            (ProviderTimeoutError, GatewayError),
            (CommunicationError, DistLLMError),
            (SerializationError, CommunicationError),
            (GRPCTimeoutError, CommunicationError),
            (ProtoError, CommunicationError),
            (ModelError, DistLLMError),
            (ModelNotFoundError, ModelError),
            (ModelLoadError, ModelError),
            (OOMError, ModelError),
            (ConfigError, DistLLMError),
            (ConfigValidationError, ConfigError),
            (BatchError, DistLLMError),
            (BatchCapacityError, BatchError),
            (ConstraintError, DistLLMError),
            (ConstraintViolationError, ConstraintError),
            (InputValidationError, DistLLMError),
        ],
    )
    def test_subclass_of(self, cls, parent):
        assert issubclass(cls, parent), f"{cls.__name__} should inherit from {parent.__name__}"

    @pytest.mark.parametrize(
        "cls,expected_code",
        [
            (DistLLMError, "DISTLLM_ERROR"),
            (NodeError, "NODE_ERROR"),
            (NodeUnreachableError, "NODE_UNREACHABLE"),
            (CircuitBreakerError, "CIRCUIT_BREAKER_OPEN"),
            (GatewayError, "GATEWAY_ERROR"),
            (ProviderTimeoutError, "PROVIDER_TIMEOUT"),
            (CommunicationError, "COMMUNICATION_ERROR"),
            (SerializationError, "SERIALIZATION_ERROR"),
            (GRPCTimeoutError, "GRPC_TIMEOUT"),
            (ProtoError, "PROTO_ERROR"),
            (ModelError, "MODEL_ERROR"),
            (ModelNotFoundError, "MODEL_NOT_FOUND"),
            (ModelLoadError, "MODEL_LOAD_ERROR"),
            (OOMError, "OOM_ERROR"),
            (ConfigError, "CONFIG_ERROR"),
            (ConfigValidationError, "CONFIG_VALIDATION_ERROR"),
            (BatchError, "BATCH_ERROR"),
            (BatchCapacityError, "BATCH_CAPACITY_ERROR"),
            (ConstraintError, "CONSTRAINT_ERROR"),
            (ConstraintViolationError, "CONSTRAINT_VIOLATION"),
            (InputValidationError, "INPUT_VALIDATION_ERROR"),
        ],
    )
    def test_error_code(self, cls, expected_code):
        kwargs = _minimal_args(cls)
        err = cls(**kwargs)
        assert err.code == expected_code, f"{cls.__name__}.code should be {expected_code!r}"


def _minimal_args(cls: type) -> dict:
    """Return a dict of minimal constructor arguments for *cls*."""
    table: dict[type, dict] = {
        DistLLMError: {"message": "err"},
        NodeError: {"message": "err"},
        NodeUnreachableError: {"node_id": "n1", "host": "h", "port": 1},
        CircuitBreakerError: {"node_id": "n1", "failures": 1, "recovery_in": 1.0},
        GatewayError: {"provider": "p", "model": "m"},
        ProviderTimeoutError: {"provider": "p", "model": "m"},
        CommunicationError: {"message": "err"},
        SerializationError: {"message": "err"},
        GRPCTimeoutError: {"node_id": "n1", "timeout": 1.0},
        ProtoError: {"message": "err"},
        ModelError: {"message": "err"},
        ModelNotFoundError: {"model_name": "m"},
        ModelLoadError: {"model_name": "m", "reason": "r"},
        OOMError: {"node_id": "n1"},
        ConfigError: {"message": "err"},
        ConfigValidationError: {"field": "f", "message": "m"},
        BatchError: {"message": "err"},
        BatchCapacityError: {"current_tokens": 1, "max_tokens": 1},
        ConstraintError: {"message": "err"},
        ConstraintViolationError: {"constraint_type": "c", "detail": "d"},
        InputValidationError: {"detail": "d"},
    }
    return table.get(cls, {"message": "err"})


# ============================================================================
# RETRY POLICY
# ============================================================================


class TestRetryPolicy:
    def test_defaults(self):
        p = RetryPolicy()
        assert p.max_retries == 3
        assert p.base_delay == 1.0
        assert p.max_delay == 60.0
        assert p.retryable == (IOError, TimeoutError, ConnectionError, OSError)
        assert p.backoff_multiplier == 2.0

    def test_custom_values(self):
        p = RetryPolicy(max_retries=5, base_delay=0.5, max_delay=10.0, retryable=(ValueError,), backoff_multiplier=3.0)
        assert p.max_retries == 5
        assert p.base_delay == 0.5
        assert p.max_delay == 10.0
        assert p.retryable == (ValueError,)
        assert p.backoff_multiplier == 3.0


# ============================================================================
# WITH_RETRY (synchronous)
# ============================================================================


class TestWithRetry:
    def test_success_first_attempt(self):
        """Succeeds on the first call without any retry."""
        call_count = 0

        def fn():
            nonlocal call_count
            call_count += 1
            return 42

        decorated = with_retry(RetryPolicy(max_retries=3))(fn)
        assert decorated() == 42
        assert call_count == 1

    def test_succeeds_after_retries(self):
        """Fails twice, succeeds on third attempt."""
        call_count = 0

        def flaky(x: int) -> int:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("try again")
            return x + 1

        decorated = with_retry(RetryPolicy(max_retries=3, base_delay=0.001, retryable=(ConnectionError,)))(flaky)
        result = decorated(41)
        assert result == 42
        assert call_count == 3

    def test_exhaust_retries_raises(self):
        """Raises the last exception once max_retries is exhausted."""
        call_count = 0

        def fn():
            nonlocal call_count
            call_count += 1
            raise ConnectionError("always fails")

        decorated = with_retry(RetryPolicy(max_retries=2, base_delay=0.001, retryable=(ConnectionError,)))(fn)
        with pytest.raises(ConnectionError, match="always fails"):
            decorated()
        assert call_count == 3

    def test_max_retries_zero(self):
        """With max_retries=0, the function is called once and raises immediately."""
        call_count = 0

        def fn():
            nonlocal call_count
            call_count += 1
            raise ValueError("boom")

        decorated = with_retry(RetryPolicy(max_retries=0, retryable=(ValueError,)))(fn)
        with pytest.raises(ValueError, match="boom"):
            decorated()
        assert call_count == 1

    def test_non_retryable_exception_propagates(self):
        """If the exception is not in retryable, it propagates immediately."""
        call_count = 0

        def fn():
            nonlocal call_count
            call_count += 1
            raise TypeError("bad type")

        decorated = with_retry(RetryPolicy(max_retries=3, retryable=(ConnectionError,)))(fn)
        with pytest.raises(TypeError, match="bad type"):
            decorated()
        assert call_count == 1

    def test_async_function_raises_typeerror(self):
        """Decorating an async function with @with_retry raises TypeError."""
        async def async_fn():
            return 42

        with pytest.raises(TypeError, match="async"):
            with_retry(RetryPolicy())(async_fn)

    def test_sync_function_returning_coroutine_raises_typeerror(self):
        """If the sync wrapper returns a coroutine, it raises TypeError."""
        async def inner():
            return 42

        def bad_fn():
            return inner()

        decorated = with_retry(RetryPolicy(max_retries=1, retryable=(ConnectionError,)))(bad_fn)
        with pytest.raises(TypeError, match="coroutine"):
            decorated()

    def test_backoff_delay_increases(self):
        """Each successive retry increases the delay exponentially (tiny base for speed)."""
        call_count = 0

        def fn():
            nonlocal call_count
            call_count += 1
            raise ConnectionError("fail")

        policy = RetryPolicy(max_retries=3, base_delay=0.001, backoff_multiplier=2.0, max_delay=60.0, retryable=(ConnectionError,))
        decorated = with_retry(policy)(fn)

        with pytest.raises(ConnectionError):
            decorated()

        assert call_count == 4

    def test_backoff_capped_at_max_delay(self):
        """Delay is capped at max_delay (tiny base for speed)."""
        call_count = 0

        def fn():
            nonlocal call_count
            call_count += 1
            raise ConnectionError("fail")

        policy = RetryPolicy(max_retries=4, base_delay=0.001, backoff_multiplier=10.0, max_delay=5.0, retryable=(ConnectionError,))
        decorated = with_retry(policy)(fn)

        with pytest.raises(ConnectionError):
            decorated()

        assert call_count == 5


# ============================================================================
# WITH_RETRY_ASYNC
# ============================================================================


class TestWithRetryAsync:
    @pytest.mark.asyncio
    async def test_success_first_attempt(self):
        call_count = 0

        @with_retry_async(RetryPolicy(max_retries=3))
        async def do_it():
            nonlocal call_count
            call_count += 1
            return 42

        result = await do_it()
        assert result == 42
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_succeeds_after_retries(self):
        call_count = 0

        @with_retry_async(RetryPolicy(max_retries=3, base_delay=0.01, retryable=(ConnectionError,)))
        async def flaky(x: int) -> int:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("try again")
            return x + 1

        result = await flaky(41)
        assert result == 42
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_exhaust_retries_raises(self):
        call_count = 0

        @with_retry_async(RetryPolicy(max_retries=2, base_delay=0.01, retryable=(ConnectionError,)))
        async def do_it():
            nonlocal call_count
            call_count += 1
            raise ConnectionError("always fails")

        with pytest.raises(ConnectionError, match="always fails"):
            await do_it()
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_max_retries_zero(self):
        @with_retry_async(RetryPolicy(max_retries=0, retryable=(ValueError,)))
        async def do_it():
            raise ValueError("boom")

        with pytest.raises(ValueError, match="boom"):
            await do_it()

    @pytest.mark.asyncio
    async def test_non_retryable_exception_propagates(self):
        @with_retry_async(RetryPolicy(max_retries=3, retryable=(ConnectionError,)))
        async def do_it():
            raise TypeError("bad type")

        with pytest.raises(TypeError, match="bad type"):
            await do_it()

    @pytest.mark.asyncio
    async def test_backoff_delay(self):
        """Retries execute with real sleeps (tiny base_delay so the test is fast)."""
        call_count = 0

        @with_retry_async(RetryPolicy(max_retries=3, base_delay=0.001, backoff_multiplier=2.0, max_delay=60.0, retryable=(ConnectionError,)))
        async def do_it():
            nonlocal call_count
            call_count += 1
            raise ConnectionError("fail")

        with pytest.raises(ConnectionError):
            await do_it()

        assert call_count == 4


# ============================================================================
# RETRY_GRPC_CALL
# ============================================================================


class TestRetryGrpcCall:
    def test_success(self):
        call_count = 0

        def fn():
            nonlocal call_count
            call_count += 1
            return "ok"

        result = retry_grpc_call(fn, max_retries=3, base_delay=0.001, retryable_exceptions=(ConnectionError,))
        assert result == "ok"
        assert call_count == 1

    def test_exhaust_retries(self):
        call_count = 0

        def fn():
            nonlocal call_count
            call_count += 1
            raise ConnectionError("grpc fail")

        with pytest.raises(ConnectionError, match="grpc fail"):
            retry_grpc_call(fn, max_retries=2, base_delay=0.001, retryable_exceptions=(ConnectionError,))
        assert call_count == 3

    def test_non_retryable_propagates(self):
        call_count = 0

        def fn():
            nonlocal call_count
            call_count += 1
            raise TypeError("bad")

        with pytest.raises(TypeError, match="bad"):
            retry_grpc_call(fn, max_retries=2, base_delay=0.001, retryable_exceptions=(ConnectionError,))
        assert call_count == 1

    def test_default_exceptions_requires_grpc(self):
        """When no retryable_exceptions is provided, the function uses grpc.RpcError.
        If grpc is not installed this test is skipped."""
        grpc = pytest.importorskip("grpc")
        call_count = 0

        def fn():
            nonlocal call_count
            call_count += 1
            raise grpc.RpcError()

        with pytest.raises(grpc.RpcError):
            retry_grpc_call(fn, max_retries=1, base_delay=0.001)
        assert call_count == 2


# ============================================================================
# POLICIES
# ============================================================================


class TestErrorRetryPolicies:
    def test_has_expected_keys(self):
        """The registry contains all expected error types."""
        expected = {
            NodeUnreachableError,
            CircuitBreakerError,
            GRPCTimeoutError,
            SerializationError,
            ModelNotFoundError,
            ModelLoadError,
            ConfigValidationError,
            BatchCapacityError,
            ConstraintViolationError,
        }
        assert expected.issubset(ERROR_RETRY_POLICIES.keys())

    def test_circuit_breaker_not_retryable(self):
        policy = ERROR_RETRY_POLICIES[CircuitBreakerError]
        assert policy.max_retries == 0
        assert policy.retryable == ()

    def test_grpc_timeout_retryable(self):
        policy = ERROR_RETRY_POLICIES[GRPCTimeoutError]
        assert policy.max_retries == 5
        assert policy.base_delay == 0.5
        assert policy.max_delay == 30.0
        assert GRPCTimeoutError in policy.retryable

    def test_model_not_found_not_retryable(self):
        policy = ERROR_RETRY_POLICIES[ModelNotFoundError]
        assert policy.max_retries == 0

    def test_serialization_retryable(self):
        policy = ERROR_RETRY_POLICIES[SerializationError]
        assert policy.max_retries == 2
        assert SerializationError in policy.retryable


class TestGetRetryPolicy:
    def test_exact_match(self):
        err = GRPCTimeoutError("n1", timeout=1.0)
        policy = get_retry_policy(err)
        assert policy.max_retries == 5

    def test_mro_fallback(self):
        """A subclass without its own policy falls back to its parent's policy."""
        err = ProviderTimeoutError("openai", "gpt-4")
        policy = get_retry_policy(err)
        assert policy.max_retries == 2
        assert policy.retryable == (ProviderTimeoutError,)

    def test_default_fallback(self):
        """An error type not in the registry at all gets the default policy."""
        err = InputValidationError("bad")
        policy = get_retry_policy(err)
        assert policy.max_retries == 2
        assert policy.base_delay == 1.0
        assert policy.max_delay == 10.0

    def test_mro_walks_to_node_error(self):
        """NodeUnreachableError matches its direct policy, not NodeError's non-existent entry."""
        err = NodeUnreachableError("n1", "h", 1)
        policy = get_retry_policy(err)
        assert policy.max_retries == 3


class TestShouldRetry:
    def test_retryable_below_max(self):
        """Returns True when attempt < max_retries and type is retryable."""
        err = GRPCTimeoutError("n1", timeout=1.0)
        assert should_retry(err, attempt=0) is True
        assert should_retry(err, attempt=4) is True

    def test_retryable_at_max(self):
        """Returns False when attempt == max_retries (exhausted)."""
        err = GRPCTimeoutError("n1", timeout=1.0)
        assert should_retry(err, attempt=5) is False
        assert should_retry(err, attempt=6) is False

    def test_non_retryable_error_type(self):
        """Returns False when the error type is not in retryable."""
        err = CircuitBreakerError("n1", failures=3, recovery_in=10.0)
        assert should_retry(err, attempt=0) is False

    def test_unregistered_error(self):
        """Unregistered errors are retryable (type in default retryable) up to max_retries."""
        err = InputValidationError("bad")
        assert should_retry(err, attempt=0) is True
        assert should_retry(err, attempt=2) is False


class TestGetRetryDelay:
    def test_exponential_backoff(self):
        err = GRPCTimeoutError("n1", timeout=1.0)
        assert get_retry_delay(err, attempt=0) == pytest.approx(0.5)
        assert get_retry_delay(err, attempt=1) == pytest.approx(1.0)
        assert get_retry_delay(err, attempt=2) == pytest.approx(2.0)

    def test_capped_at_max_delay(self):
        err = GRPCTimeoutError("n1", timeout=1.0)
        assert get_retry_delay(err, attempt=6) == pytest.approx(30.0)

    def test_default_policy_backoff(self):
        err = InputValidationError("bad")
        assert get_retry_delay(err, attempt=0) == pytest.approx(1.0)
        assert get_retry_delay(err, attempt=1) == pytest.approx(2.0)
        assert get_retry_delay(err, attempt=2) == pytest.approx(4.0)
        assert get_retry_delay(err, attempt=10) == pytest.approx(10.0)
