"""Tests for retry utilities — RetryPolicy, with_retry, with_retry_async, retry_grpc_call."""

import asyncio
import time
from unittest.mock import patch, MagicMock

import pytest

from distllm.errors.retry import RetryPolicy, with_retry, with_retry_async, retry_grpc_call


# ---------------------------------------------------------------------------
# RetryPolicy defaults
# ---------------------------------------------------------------------------


class TestRetryPolicyDefaults:
    """Verify the dataclass defaults match documentation."""

    def test_default_max_retries(self):
        policy = RetryPolicy()
        assert policy.max_retries == 3

    def test_default_base_delay(self):
        policy = RetryPolicy()
        assert policy.base_delay == 1.0

    def test_default_max_delay(self):
        policy = RetryPolicy()
        assert policy.max_delay == 60.0

    def test_default_backoff_multiplier(self):
        policy = RetryPolicy()
        assert policy.backoff_multiplier == 2.0

    def test_default_retryable_includes_ioerror(self):
        policy = RetryPolicy()
        assert IOError in policy.retryable

    def test_default_retryable_includes_timeout(self):
        policy = RetryPolicy()
        assert TimeoutError in policy.retryable

    def test_default_retryable_includes_connection(self):
        policy = RetryPolicy()
        assert ConnectionError in policy.retryable

    def test_default_retryable_includes_oserror(self):
        policy = RetryPolicy()
        assert OSError in policy.retryable

    def test_custom_policy_fields(self):
        policy = RetryPolicy(
            max_retries=5,
            base_delay=0.5,
            max_delay=30.0,
            retryable=(ValueError,),
            backoff_multiplier=3.0,
        )
        assert policy.max_retries == 5
        assert policy.base_delay == 0.5
        assert policy.max_delay == 30.0
        assert policy.retryable == (ValueError,)
        assert policy.backoff_multiplier == 3.0


# ---------------------------------------------------------------------------
# with_retry — synchronous decorator
# ---------------------------------------------------------------------------


class TestWithRetrySync:
    """Test the @with_retry decorator for synchronous functions."""

    @patch("distllm.errors.retry.time.sleep", return_value=None)
    def test_success_on_first_try(self, mock_sleep):
        """Should return the result without sleeping on first success."""
        policy = RetryPolicy(max_retries=3, retryable=(IOError,))

        @with_retry(policy)
        def succeed():
            return "ok"

        result = succeed()
        assert result == "ok"
        mock_sleep.assert_not_called()

    @patch("distllm.errors.retry.time.sleep", return_value=None)
    def test_retries_on_retryable_exception(self, mock_sleep):
        """Should retry when a retryable exception is raised."""
        call_count = 0
        policy = RetryPolicy(max_retries=2, base_delay=1.0, retryable=(IOError,))

        @with_retry(policy)
        def flaky():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise IOError("transient")
            return "recovered"

        result = flaky()
        assert result == "recovered"
        assert call_count == 3

    @patch("distllm.errors.retry.time.sleep", return_value=None)
    def test_raises_after_max_retries(self, mock_sleep):
        """Should re-raise the exception after exhausting retries."""
        policy = RetryPolicy(max_retries=2, base_delay=0.1, retryable=(IOError,))

        @with_retry(policy)
        def always_fail():
            raise IOError("permanent")

        with pytest.raises(IOError, match="permanent"):
            always_fail()
        assert mock_sleep.call_count == 2

    @patch("distllm.errors.retry.time.sleep", return_value=None)
    def test_does_not_catch_non_retryable(self, mock_sleep):
        """Should let non-retryable exceptions propagate immediately."""
        policy = RetryPolicy(max_retries=3, retryable=(IOError,))

        @with_retry(policy)
        def raise_value_error():
            raise ValueError("not retryable")

        with pytest.raises(ValueError, match="not retryable"):
            raise_value_error()
        mock_sleep.assert_not_called()

    @patch("distllm.errors.retry.time.sleep", return_value=None)
    def test_exponential_backoff_delays(self, mock_sleep):
        """Should compute delays: base * multiplier^attempt, capped at max_delay."""
        policy = RetryPolicy(
            max_retries=3,
            base_delay=1.0,
            max_delay=10.0,
            backoff_multiplier=2.0,
            retryable=(IOError,),
        )

        @with_retry(policy)
        def fail_three_times():
            raise IOError("boom")

        with pytest.raises(IOError):
            fail_three_times()

        expected_delays = [1.0, 2.0, 4.0]
        actual_delays = [call.args[0] for call in mock_sleep.call_args_list]
        assert actual_delays == expected_delays

    @patch("distllm.errors.retry.time.sleep", return_value=None)
    def test_delay_capped_at_max(self, mock_sleep):
        """Should cap the delay at max_delay."""
        policy = RetryPolicy(
            max_retries=5,
            base_delay=1.0,
            max_delay=5.0,
            backoff_multiplier=4.0,
            retryable=(IOError,),
        )

        @with_retry(policy)
        def always_fail():
            raise IOError("boom")

        with pytest.raises(IOError):
            always_fail()

        actual_delays = [call.args[0] for call in mock_sleep.call_args_list]
        for delay in actual_delays:
            assert delay <= 5.0

    def test_rejects_async_function(self):
        """Should raise TypeError when decorating an async function."""
        policy = RetryPolicy(max_retries=1, retryable=(IOError,))

        with pytest.raises(TypeError, match="async"):
            @with_retry(policy)
            async def async_fn():
                pass

    @patch("distllm.errors.retry.time.sleep", return_value=None)
    def test_rejects_coroutine_return(self, mock_sleep):
        """Should raise TypeError if a sync-decorated function returns a coroutine."""
        policy = RetryPolicy(max_retries=1, retryable=(IOError,))

        async def coro():
            return 42

        def sneaky():
            return coro()

        decorated = with_retry(policy)(sneaky)

        with pytest.raises(TypeError, match="coroutine"):
            decorated()

    @patch("distllm.errors.retry.time.sleep", return_value=None)
    def test_passes_args_and_kwargs(self, mock_sleep):
        """Should forward positional and keyword arguments to the wrapped function."""
        policy = RetryPolicy(max_retries=0, retryable=(IOError,))

        @with_retry(policy)
        def add(a, b, scale=1):
            return (a + b) * scale

        assert add(2, 3, scale=10) == 50

    @patch("distllm.errors.retry.time.sleep", return_value=None)
    def test_zero_max_retries_raises_immediately(self, mock_sleep):
        """With max_retries=0 the function should be called exactly once and raise."""
        policy = RetryPolicy(max_retries=0, retryable=(IOError,))

        @with_retry(policy)
        def fail():
            raise IOError("once")

        with pytest.raises(IOError, match="once"):
            fail()
        mock_sleep.assert_not_called()


# ---------------------------------------------------------------------------
# with_retry_async — asynchronous decorator
# ---------------------------------------------------------------------------


class TestWithRetryAsync:
    """Test the @with_retry_async decorator for async functions."""

    @pytest.mark.asyncio
    @patch("distllm.errors.retry.asyncio.sleep", return_value=None)
    async def test_success_on_first_try(self, mock_sleep):
        policy = RetryPolicy(max_retries=3, retryable=(IOError,))

        @with_retry_async(policy)
        async def succeed():
            return "ok"

        result = await succeed()
        assert result == "ok"
        mock_sleep.assert_not_called()

    @pytest.mark.asyncio
    @patch("distllm.errors.retry.asyncio.sleep", return_value=None)
    async def test_retries_on_retryable_exception(self, mock_sleep):
        call_count = 0
        policy = RetryPolicy(max_retries=2, base_delay=0.5, retryable=(IOError,))

        @with_retry_async(policy)
        async def flaky():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise IOError("transient")
            return "recovered"

        result = await flaky()
        assert result == "recovered"
        assert call_count == 3

    @pytest.mark.asyncio
    @patch("distllm.errors.retry.asyncio.sleep", return_value=None)
    async def test_raises_after_max_retries(self, mock_sleep):
        policy = RetryPolicy(max_retries=1, base_delay=0.1, retryable=(IOError,))

        @with_retry_async(policy)
        async def always_fail():
            raise IOError("permanent")

        with pytest.raises(IOError, match="permanent"):
            await always_fail()

    @pytest.mark.asyncio
    @patch("distllm.errors.retry.asyncio.sleep", return_value=None)
    async def test_exponential_backoff_delays(self, mock_sleep):
        policy = RetryPolicy(
            max_retries=3,
            base_delay=1.0,
            max_delay=10.0,
            backoff_multiplier=2.0,
            retryable=(TimeoutError,),
        )

        @with_retry_async(policy)
        async def always_fail():
            raise TimeoutError("timeout")

        with pytest.raises(TimeoutError):
            await always_fail()

        expected_delays = [1.0, 2.0, 4.0]
        actual_delays = [call.args[0] for call in mock_sleep.call_args_list]
        assert actual_delays == expected_delays

    @pytest.mark.asyncio
    @patch("distllm.errors.retry.asyncio.sleep", return_value=None)
    async def test_does_not_catch_non_retryable(self, mock_sleep):
        policy = RetryPolicy(max_retries=3, retryable=(IOError,))

        @with_retry_async(policy)
        async def raise_value_error():
            raise ValueError("not retryable")

        with pytest.raises(ValueError, match="not retryable"):
            await raise_value_error()
        mock_sleep.assert_not_called()

    @pytest.mark.asyncio
    @patch("distllm.errors.retry.asyncio.sleep", return_value=None)
    async def test_passes_args_and_kwargs(self, mock_sleep):
        policy = RetryPolicy(max_retries=0, retryable=(IOError,))

        @with_retry_async(policy)
        async def multiply(a, b):
            return a * b

        assert await multiply(3, 7) == 21


# ---------------------------------------------------------------------------
# retry_grpc_call — imperative retry helper
# ---------------------------------------------------------------------------


class TestRetryGrpcCall:
    """Test the retry_grpc_call helper function."""

    @patch("distllm.errors.retry.time.sleep", return_value=None)
    def test_success_on_first_call(self, mock_sleep):
        fn = MagicMock(return_value="result")
        result = retry_grpc_call(fn, max_retries=3)
        assert result == "result"
        fn.assert_called_once()
        mock_sleep.assert_not_called()

    @patch("distllm.errors.retry.time.sleep", return_value=None)
    def test_retries_on_custom_exception(self, mock_sleep):
        class TransientError(Exception):
            pass

        fn = MagicMock(side_effect=[TransientError("fail"), "ok"])
        result = retry_grpc_call(
            fn,
            max_retries=2,
            base_delay=0.1,
            retryable_exceptions=(TransientError,),
        )
        assert result == "ok"
        assert fn.call_count == 2

    @patch("distllm.errors.retry.time.sleep", return_value=None)
    def test_raises_after_exhausting_retries(self, mock_sleep):
        class TransientError(Exception):
            pass

        fn = MagicMock(side_effect=TransientError("always fail"))
        with pytest.raises(TransientError):
            retry_grpc_call(
                fn,
                max_retries=2,
                base_delay=0.01,
                retryable_exceptions=(TransientError,),
            )
        assert fn.call_count == 3  # 1 initial + 2 retries

    @patch("distllm.errors.retry.time.sleep", return_value=None)
    def test_does_not_retry_non_retryable(self, mock_sleep):
        fn = MagicMock(side_effect=ValueError("bad"))
        with pytest.raises(ValueError, match="bad"):
            retry_grpc_call(
                fn,
                max_retries=3,
                retryable_exceptions=(IOError,),
            )
        fn.assert_called_once()

    @patch("distllm.errors.retry.time.sleep", return_value=None)
    def test_exponential_backoff_delays(self, mock_sleep):
        class TransientError(Exception):
            pass

        fn = MagicMock(side_effect=TransientError("fail"))
        with pytest.raises(TransientError):
            retry_grpc_call(
                fn,
                max_retries=3,
                base_delay=1.0,
                max_delay=10.0,
                retryable_exceptions=(TransientError,),
            )

        expected_delays = [1.0, 2.0, 4.0]
        actual_delays = [call.args[0] for call in mock_sleep.call_args_list]
        assert actual_delays == expected_delays

    @patch("distllm.errors.retry.time.sleep", return_value=None)
    def test_uses_default_retryable_when_none(self, mock_sleep):
        """When retryable_exceptions is None it defaults to grpc.RpcError."""
        mock_grpc = MagicMock()
        mock_rpc_error = type("RpcError", (Exception,), {})
        mock_grpc.RpcError = mock_rpc_error

        with patch.dict("sys.modules", {"grpc": mock_grpc}):
            import importlib
            import distllm.errors.retry as retry_mod
            importlib.reload(retry_mod)

            fn = MagicMock(side_effect=[mock_rpc_error("fail"), "ok"])
            result = retry_mod.retry_grpc_call(fn, max_retries=1, base_delay=0.01)
            assert result == "ok"
