from distllm_sdk.types import ApiError


class AuthenticationError(ApiError):
    def __init__(self, message: str = "Authentication failed", request_id: str | None = None):
        super().__init__(message, status_code=401, error_type="authentication_error", request_id=request_id)


class RateLimitError(ApiError):
    def __init__(self, message: str = "Rate limit exceeded", retry_after: float | None = None, request_id: str | None = None):
        super().__init__(message, status_code=429, error_type="rate_limit_error", request_id=request_id)
        self.retry_after = retry_after


class TimeoutError(ApiError):
    def __init__(self, message: str = "Request timed out", request_id: str | None = None):
        super().__init__(message, status_code=504, error_type="timeout_error", request_id=request_id)


class ModelNotFoundError(ApiError):
    def __init__(self, model: str, request_id: str | None = None):
        super().__init__(
            f"Model '{model}' not found",
            status_code=404,
            error_type="model_not_found",
            request_id=request_id,
        )
        self.model = model


class ServiceUnavailableError(ApiError):
    def __init__(self, message: str = "Service unavailable", retry_after: float | None = None, request_id: str | None = None):
        super().__init__(message, status_code=503, error_type="service_unavailable", request_id=request_id)
        self.retry_after = retry_after


class InvalidRequestError(ApiError):
    def __init__(self, message: str = "Invalid request", param: str | None = None, request_id: str | None = None):
        super().__init__(message, status_code=400, error_type="invalid_request_error", request_id=request_id)
        self.param = param
