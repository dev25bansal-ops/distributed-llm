DEFAULT_HTTP_TIMEOUT: float = 120.0
MAX_RETRIES: int = 3
RETRY_DELAY: float = 1.0
# Floor (seconds) applied when honoring a server-provided Retry-After header,
# so a tiny/zero value cannot cause a tight retry loop.
RETRY_AFTER_FLOOR: float = 0.5
