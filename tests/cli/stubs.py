"""Stub classes for CLI tests — replace MagicMock with explicit stubs.

Each stub explicitly implements the interface needed for the tests.
Call tracking is built into each stub method.
"""

from __future__ import annotations

from typing import Any


# ── Response stub ──────────────────────────────────────────────────────────


class StubResponse:
    """Stub for HTTP response objects (httpx.Response)."""

    def __init__(
        self,
        json_data: Any = None,
        text: str = "",
        status_code: int = 200,
    ):
        self._json_data = json_data
        self.text = text
        self.status_code = status_code

    def json(self) -> Any:
        return self._json_data

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            import httpx
            raise httpx.HTTPStatusError(
                f"HTTP {self.status_code} error",
                request=_StubHttpxRequest(),
                response=self,
            )


class _StubHttpxRequest:
    """Minimal httpx.Request stub for HTTPStatusError in raise_for_status."""
    url = "http://localhost/"
    method = "GET"


# ── Generic callable stub ──────────────────────────────────────────────────


class StubFn:
    """Callable stub that records calls.

    Usage::

        fn = StubFn(return_value=42)
        result = fn("arg", key="val")
        assert result == 42
        assert fn.calls == [(("arg",), {"key": "val"})]
    """

    def __init__(self, return_value: Any = None, side_effect: Any = None):
        self.calls: list[tuple[tuple, dict]] = []
        self.return_value = return_value
        self._side_effect = side_effect

    @property
    def side_effect(self) -> Any:
        return self._side_effect

    @side_effect.setter
    def side_effect(self, value: Any) -> None:
        self._side_effect = value

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.calls.append((args, kwargs))
        if self._side_effect is not None:
            if callable(self._side_effect):
                return self._side_effect(*args, **kwargs)
            raise self._side_effect
        return self.return_value

    @property
    def call_count(self) -> int:
        """Number of times this stub was called (compat with MagicMock.call_count)."""
        return len(self.calls)

    @property
    def call_args_list(self) -> list:
        """Return recorded calls (compat with MagicMock.call_args_list)."""
        return [
            type("_Call", (), {"args": a, "kwargs": kw})()
            for a, kw in self.calls
        ]

    @property
    def called(self) -> bool:
        """Whether this stub was called at least once (compat with MagicMock.called)."""
        return len(self.calls) > 0

    @property
    def call_count(self) -> int:
        """Alias for len(self.calls) (compat with MagicMock)."""
        return len(self.calls)

    def assert_any_call(self, *args: Any, **kwargs: Any) -> None:
        """Assert the callable was called at least once with the given args."""
        for a, kw in self.calls:
            if a == args and kw == kwargs:
                return
        raise AssertionError(
            f"Expected call({args!r}, {kwargs!r}) not found in {self.calls!r}"
        )

    def assert_called_once(self) -> None:
        assert len(self.calls) == 1, f"Expected 1 call, got {len(self.calls)}"

    def assert_called_once_with(self, *args: Any, **kwargs: Any) -> None:
        self.assert_called_once()
        recorded = self.calls[0]
        assert recorded[0] == args, (
            f"Expected args {args!r}, got {recorded[0]!r}"
        )
        assert recorded[1] == kwargs, (
            f"Expected kwargs {kwargs!r}, got {recorded[1]!r}"
        )

    def assert_not_called(self) -> None:
        assert len(self.calls) == 0, f"Expected no calls, got {len(self.calls)}"


# ── Client stub ────────────────────────────────────────────────────────────


class StubClientConfig:
    """Minimal config for StubClient."""
    base_url = "http://localhost:8000"


class StubClient:
    """Stub for DistLLMClient used by CLI modules.

    Tracks get/post calls and returns pre-configured results.
    Supports URL-specific responses via side_effect function.
    """

    def __init__(self):
        self.get_calls: list[tuple[str, dict]] = []
        self.post_calls: list[tuple[str, dict]] = []
        self._get_result: Any = None
        self._get_side_effect: Any = None
        self._post_result: Any = None
        self._post_side_effect: Any = None

    def set_get(self, result: Any = None, side_effect: Any = None) -> None:
        """Set default return value or side effect for get()."""
        self._get_result = result
        self._get_side_effect = side_effect

    def set_post(self, result: Any = None, side_effect: Any = None) -> None:
        """Set default return value or side effect for post()."""
        self._post_result = result
        self._post_side_effect = side_effect

    def get(self, path: str, **kwargs: Any) -> Any:
        self.get_calls.append((path, kwargs))
        if self._get_side_effect is not None:
            if callable(self._get_side_effect):
                return self._get_side_effect(path, **kwargs)
            raise self._get_side_effect
        return self._get_result

    def post(self, path: str, **kwargs: Any) -> Any:
        self.post_calls.append((path, kwargs))
        if self._post_side_effect is not None:
            if callable(self._post_side_effect):
                return self._post_side_effect(path, **kwargs)
            raise self._post_side_effect
        return self._post_result

    def _get_session(self) -> StubClient:
        """Stub for internal session access (used by status.py)."""
        return self

    @property
    def _config(self) -> StubClientConfig:
        return StubClientConfig()

    def assert_get_called_once_with(self, path: str) -> None:
        assert len(self.get_calls) == 1, (
            f"Expected 1 GET call, got {len(self.get_calls)}"
        )
        assert self.get_calls[0][0] == path, (
            f"Expected GET {path!r}, got {self.get_calls[0][0]!r}"
        )

    def assert_post_called_once(self) -> None:
        assert len(self.post_calls) == 1, (
            f"Expected 1 POST call, got {len(self.post_calls)}"
        )


# ── Httpx client stub (context manager) ────────────────────────────────────


class StubHttpxClient:
    """Stub for httpx.Client used as a context manager in cluster.py / benchmark.py."""

    def __init__(self):
        self.get_calls: list[tuple[str, dict]] = []
        self.post_calls: list[tuple[str, dict]] = []
        self._get_result: StubResponse | None = None
        self._get_side_effect: Any = None
        self._post_result: StubResponse | None = None
        self._post_side_effect: Any = None

    def set_get(self, result: StubResponse | None = None,
                side_effect: Any = None) -> None:
        self._get_result = result
        self._get_side_effect = side_effect

    def set_post(self, result: StubResponse | None = None,
                 side_effect: Any = None) -> None:
        self._post_result = result
        self._post_side_effect = side_effect

    def __enter__(self) -> StubHttpxClient:
        return self

    def __exit__(self, *args: Any) -> None:
        pass

    def get(self, url: str, **kwargs: Any) -> StubResponse | None:
        self.get_calls.append((url, kwargs))
        if self._get_side_effect is not None:
            if callable(self._get_side_effect):
                return self._get_side_effect(url, **kwargs)
            raise self._get_side_effect
        return self._get_result

    def post(self, url: str, **kwargs: Any) -> StubResponse | None:
        self.post_calls.append((url, kwargs))
        if self._post_side_effect is not None:
            if callable(self._post_side_effect):
                return self._post_side_effect(url, **kwargs)
            raise self._post_side_effect
        return self._post_result


# ── Console stub ───────────────────────────────────────────────────────────


class StubConsole:
    """Stub for rich.console.Console used in CLI tests.

    ``console.print`` is a StubFn so MagicMock-compatible attributes
    (``call_count``, ``call_args_list``, ``assert_any_call()``) work
    transparently. ``console.input`` is a real method that can be
    configured via ``set_input_values()`` or ``console.input.return_value``.
    """

    def __init__(self):
        self.print = StubFn()
        self.input_calls: list[str] = []
        self._input_fn = lambda prompt="": ""

    def input(self, prompt: str = "") -> str:
        result = self._input_fn(prompt)
        self.input_calls.append(result)
        return result

    def set_input_values(self, values: list[str]) -> None:
        """Set a sequence of return values for successive input() calls."""
        it = iter(values)
        self._input_fn = lambda prompt="": next(it)

    def assert_called_once(self) -> None:
        """Assert print was called exactly once."""
        self.print.assert_called_once()

    def assert_not_called(self) -> None:
        """Assert print was never called."""
        self.print.assert_not_called()


# ── Backup stubs ───────────────────────────────────────────────────────────


class StubBackupManifest:
    """Stub for backup manifest return value."""

    def __init__(
        self,
        backup_id: str = "bkp-001",
        size_bytes: int = 1024,
        entries: int = 5,
        backup_type: str = "full",
        created_at: int = 1700000000,
        cluster_name: str = "default",
    ):
        self.backup_id = backup_id
        self.size_bytes = size_bytes
        self.entries = entries
        self.backup_type = backup_type
        self.created_at = created_at
        self.cluster_name = cluster_name


class StubBackupManager:
    """Stub for BackupManager."""

    def __init__(self):
        self.create_full_calls: list = []
        self.restore_calls: list = []
        self.delete_backup_calls: list = []
        self.list_backups_calls: list = []
        self._create_full_result = StubBackupManifest()
        self._restore_result: Any = {"model": "test", "config": {}}
        self._delete_result: bool = True
        self._list_backups_result: list = []

    def create_full(self, *args: Any, **kwargs: Any) -> StubBackupManifest:
        self.create_full_calls.append((args, kwargs))
        return self._create_full_result

    def list_backups(self, *args: Any, **kwargs: Any) -> list:
        self.list_backups_calls.append((args, kwargs))
        return self._list_backups_result

    def restore(self, backup_id: str, *args: Any, **kwargs: Any) -> Any:
        self.restore_calls.append((backup_id, args, kwargs))
        return self._restore_result

    def delete_backup(self, backup_id: str, *args: Any, **kwargs: Any) -> bool:
        self.delete_backup_calls.append((backup_id, args, kwargs))
        return self._delete_result

    def assert_create_full_called(self) -> None:
        assert len(self.create_full_calls) > 0, "create_full was not called"


# ── Certificate stubs ─────────────────────────────────────────────────────


class StubCertificateInfo:
    """Stub for certificate info."""

    def __init__(
        self,
        common_name: str = "test.local",
        subject_alt_names: list[str] | None = None,
        issuer: str = "self",
        not_before: int = 1700000000,
        not_after: int = 1800000000,
        fingerprint_sha256: str = "abcd1234",
        is_self_signed: bool = True,
        cert_path: str = "/tmp/certs/test.pem",
        key_path: str = "/tmp/certs/test-key.pem",
    ):
        self.common_name = common_name
        self.subject_alt_names = subject_alt_names or [common_name]
        self.issuer = issuer
        self.not_before = not_before
        self.not_after = not_after
        self.fingerprint_sha256 = fingerprint_sha256
        self.is_self_signed = is_self_signed
        self.cert_path = cert_path
        self.key_path = key_path


class StubCertificateManager:
    """Stub for CertificateManager."""

    def __init__(self):
        self.ensure_certificate_calls: list = []
        self.get_certificate_info_calls: list = []
        self.renew_all_calls: list = []
        self._cert_info = StubCertificateInfo()

    def ensure_certificate(
        self, *args: Any, **kwargs: Any
    ) -> StubCertificateInfo:
        self.ensure_certificate_calls.append((args, kwargs))
        return self._cert_info

    def get_certificate_info(
        self, *args: Any, **kwargs: Any
    ) -> StubCertificateInfo | None:
        self.get_certificate_info_calls.append((args, kwargs))
        return self._cert_info

    def renew_all(self, *args: Any, **kwargs: Any) -> list:
        self.renew_all_calls.append((args, kwargs))
        return [self._cert_info]

    def revoke(self, *args: Any, **kwargs: Any) -> None:
        pass


# ── Notification stubs ────────────────────────────────────────────────────


class _SeverityValue:
    value = "info"


class _ChannelValue:
    value = "console"


class StubNotification:
    """Stub for notification record."""

    def __init__(
        self,
        timestamp: int = 1700000000,
        title: str = "Test notification",
        message: str = "This is a test",
    ):
        self.timestamp = timestamp
        self.severity = _SeverityValue()
        self.channel = _ChannelValue()
        self.title = title
        self.message = message


class StubNotificationManager:
    """Stub for NotificationManager."""

    def __init__(self):
        self.send_calls: list = []
        self.recent_calls: list = []
        self.configure_slack_calls: list = []
        self._send_result: bool = True
        self._recent_result: list = []

    def send(self, *args: Any, **kwargs: Any) -> bool:
        self.send_calls.append((args, kwargs))
        return self._send_result

    def recent(self, *args: Any, **kwargs: Any) -> list:
        self.recent_calls.append((args, kwargs))
        return self._recent_result

    def configure_slack(self, *args: Any, **kwargs: Any) -> None:
        self.configure_slack_calls.append((args, kwargs))

    def assert_send_called(self) -> None:
        assert len(self.send_calls) > 0, "send was not called"

    def assert_configure_slack_called_once(self) -> None:
        assert len(self.configure_slack_calls) == 1, (
            f"Expected 1 configure_slack call, got {len(self.configure_slack_calls)}"
        )


# ── Quota/Usage stubs ─────────────────────────────────────────────────────


class StubQuota:
    """Stub for quota data."""
    max_tokens_per_day = 100000
    max_requests_per_minute = 100
    max_tokens_per_request = 4096
    max_concurrent_requests = 5
    cost_budget_per_month = 100.0
    overage_allowed = False


class StubUsage:
    """Stub for usage data."""
    total_requests = 500
    total_input_tokens = 50000
    total_output_tokens = 100000
    total_cost = 0.5
    daily_tokens: dict = {}


class StubUsageMeter:
    """Stub for UsageMeter."""

    def __init__(self):
        self.set_quota_calls: list = []
        self.get_quota_calls: list = []
        self.tenant_usage_calls: list = []
        self.all_tenants_calls: list = []
        self.generate_invoice_calls: list = []
        self.export_csv_calls: list = []
        self._quota: Any = StubQuota()
        self._usage: Any = StubUsage()
        self._invoice: dict = {
            "period_start": 1700000000,
            "period_end": 1700086400,
            "total_requests": 100,
            "total_input_tokens": 5000,
            "total_output_tokens": 10000,
            "total_cost": 0.05,
            "overage_cost": 0.0,
            "grand_total": 0.05,
        }
        self._export_result: str = "/tmp/export.csv"

    def set_quota(self, *args: Any, **kwargs: Any) -> None:
        self.set_quota_calls.append((args, kwargs))

    def get_quota(self, *args: Any, **kwargs: Any) -> Any:
        self.get_quota_calls.append((args, kwargs))
        return self._quota

    def tenant_usage(self, *args: Any, **kwargs: Any) -> Any:
        self.tenant_usage_calls.append((args, kwargs))
        return self._usage

    def all_tenants(self, *args: Any, **kwargs: Any) -> list:
        self.all_tenants_calls.append((args, kwargs))
        return []

    def generate_invoice(self, *args: Any, **kwargs: Any) -> dict:
        self.generate_invoice_calls.append((args, kwargs))
        return self._invoice

    def export_csv(self, *args: Any, **kwargs: Any) -> str:
        self.export_csv_calls.append((args, kwargs))
        return self._export_result

    def assert_set_quota_called(self) -> None:
        assert len(self.set_quota_calls) > 0, "set_quota was not called"


# ── Webhook stubs ─────────────────────────────────────────────────────────


class StubWebhookTarget:
    """Stub for webhook target data."""

    def __init__(
        self,
        url: str = "https://hooks.example.com/events",
        active: bool = True,
        events: list[str] | None = None,
        label: str = "test",
        success_rate: float = 1.0,
    ):
        self.url = url
        self.active = active
        self.events = events or ["model.loaded", "node.joined"]
        self.label = label
        self.success_rate = success_rate


class StubWebhookManager:
    """Stub for WebhookManager."""

    def __init__(self):
        self.register_calls: list = []
        self.list_targets_calls: list = []
        self.unregister_calls: list = []
        self.dispatch_calls: list = []
        self.success_rate_calls: list = []
        self._register_result: bool = True
        self._unregister_result: bool = True
        self._list_result: list = []

    def register(self, *args: Any, **kwargs: Any) -> bool:
        self.register_calls.append((args, kwargs))
        return self._register_result

    def list_targets(self, *args: Any, **kwargs: Any) -> list:
        self.list_targets_calls.append((args, kwargs))
        return self._list_result

    def unregister(self, *args: Any, **kwargs: Any) -> bool:
        self.unregister_calls.append((args, kwargs))
        return self._unregister_result

    def dispatch(self, *args: Any, **kwargs: Any) -> None:
        self.dispatch_calls.append((args, kwargs))

    def success_rate(self, *args: Any, **kwargs: Any) -> float:
        self.success_rate_calls.append((args, kwargs))
        return 1.0

    def assert_register_called(self) -> None:
        assert len(self.register_calls) > 0, "register was not called"

    def assert_dispatch_called(self) -> None:
        assert len(self.dispatch_calls) > 0, "dispatch was not called"


# ── Model/tokenizer stubs (for compress.py tests) ─────────────────────────


class StubAutoModel:
    """Stub for AutoModelForCausalLM."""

    def __init__(self):
        self.from_pretrained_calls: list = []
        self.save_pretrained_calls: list = []
        self.to_calls: list = []

    def from_pretrained(self, *args: Any, **kwargs: Any) -> StubAutoModel:
        self.from_pretrained_calls.append((args, kwargs))
        return self

    def save_pretrained(self, *args: Any, **kwargs: Any) -> None:
        self.save_pretrained_calls.append((args, kwargs))

    def to(self, *args: Any, **kwargs: Any) -> StubAutoModel:
        self.to_calls.append((args, kwargs))
        return self

    def parameters(self) -> list:
        return []


class StubAutoTokenizer:
    """Stub for AutoTokenizer."""

    def __init__(self):
        self.from_pretrained_calls: list = []
        self.save_pretrained_calls: list = []
        self.pad_token: Any = None
        self.eos_token: str = "<eos>"

    def from_pretrained(self, *args: Any, **kwargs: Any) -> StubAutoTokenizer:
        self.from_pretrained_calls.append((args, kwargs))
        return self

    def save_pretrained(self, *args: Any, **kwargs: Any) -> None:
        self.save_pretrained_calls.append((args, kwargs))


# ── Prompt stubs ──────────────────────────────────────────────────────────


class StubPrompt:
    """Stub for a prompt record in prompts.py tests."""

    def __init__(
        self,
        prompt_id: str = "test-prompt",
        category: str = "general",
        name: str = "Test Prompt",
        description: str = "A test prompt",
        tags: list[str] | None = None,
        version: str = "1.0",
        prompt_text: str = "You are a test assistant.",
    ):
        self.id = prompt_id
        self.category = category
        self.name = name
        self.description = description
        self.tags = tags or ["test"]
        self.version = version
        self.prompt = prompt_text


# ── Backend stub (for verify.py tests) ────────────────────────────────────


class StubAdapterClass:
    """Stub for a backend adapter class (display_name, version, description)."""

    def __init__(self):
        self.display_name = StubFn(return_value="Test Backend")
        self.version = StubFn(return_value="1.0")
        self.description = StubFn(return_value="A test backend")


class StubBackend:
    """Stub for a backend record in verify.py tests."""

    def __init__(self, name: str = "test-backend"):
        self.name = name
        self.adapter_class = StubAdapterClass()


# ── Uvicorn stub ──────────────────────────────────────────────────────────


class StubUvicorn:
    """Stub for uvicorn module in run.py tests."""

    def __init__(self):
        self.run_calls: list = []
        self._run_side_effect: Any = None

    def run(self, *args: Any, **kwargs: Any) -> None:
        self.run_calls.append((args, kwargs))
        if self._run_side_effect is not None:
            raise self._run_side_effect


# ── Verification report stub ──────────────────────────────────────────────


class StubVerificationReport:
    """Stub for verification report in verify.py tests."""

    def __init__(
        self,
        model_name: str = "test-model",
        num_nodes: int = 2,
        dtype: str = "float16",
        temperature: float = 0.0,
        per_prompt: list | None = None,
        summary: dict | None = None,
    ):
        self.model_name = model_name
        self.num_nodes = num_nodes
        self.dtype = dtype
        self.temperature = temperature
        self.per_prompt = per_prompt or []
        self._summary = summary or {
            "total": 3, "passed": 3, "failed": 0, "pass_rate": 1.0,
        }

    def summary(self) -> dict:
        return self._summary


# ── Torch stub ────────────────────────────────────────────────────────────


class StubTorchCuda:
    """Stub for torch.cuda in tutorial tests."""

    def __init__(self):
        self._is_available = True
        self._device_count = 2
        self._device_name_fn = lambda i: f"GPU {i}"

    def is_available(self) -> bool:
        return self._is_available

    def device_count(self) -> int:
        return self._device_count

    def get_device_name(self, i: int) -> str:
        return self._device_name_fn(i)


class StubTorch:
    """Stub for torch module in tutorial tests."""
    __version__ = "2.1.0"

    def __init__(self):
        self.cuda = StubTorchCuda()


# ── RemoteDraftModel stub (for correctness tests) ─────────────────────────

class StubRemoteDraftModel:
    """Stub for RemoteDraftModel in speculative decoding correctness tests.

    Provides realistic default values for token generation and stats
    that the DistributedSpeculativeDecoder reads during verification.
    """

    def __init__(self):
        self.stats: dict[str, float] = {
            "total_calls": 0,
            "total_tokens": 0,
            "avg_latency_ms": 0,
            "tokens_per_second": 0,
            "errors": 0,
        }
        self.generate_tokens = StubFn(return_value=None)  # To be configured


# ── shutil.disk_usage stub ────────────────────────────────────────────────


class StubDiskUsage:
    """Stub for shutil.disk_usage namedtuple."""

    def __init__(self, total: float = 500e9, used: float = 100e9, free: float = 400e9):
        self.total = total
        self.used = used
        self.free = free
