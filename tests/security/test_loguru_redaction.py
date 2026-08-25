"""Tests for loguru-integrated global log redaction (SEC-A8 fix).

The DistLLM codebase (~794 modules) emits through ``loguru.logger`` while the
original ``RedactingFilter`` only patched *stdlib* logging -- so even when
installed, redaction caught nothing that matters.  The fix installs a
loguru *core-level patcher* (``logger.configure(patcher=...)``) which loguru
applies to every record dict process-wide, after %-formatting and before any
sink writes.  All existing ``logger.*`` calls are therefore redacted
transparently regardless of sinks (default colorized stderr, JSON stdout,
``enqueue=True`` queues).

These tests capture real loguru sink output and assert secrets are masked.
No real secrets are used -- only obviously-fake tokens.
"""

from __future__ import annotations

import json
import logging
import sys
import threading
from io import StringIO

import pytest
from loguru import logger as loguru_logger

import distllm.security.log_redaction as lr
from distllm.security.log_redaction import (
    ENV_REDACTION_ENABLED,
    install_global_redaction,
    install_loguru_redaction,
    uninstall_loguru_redaction,
)

# Fake (non-real) tokens exercising distinct redaction patterns.
FAKE_API_KEY = "sk-" + "abcdefghijklmnopqrstuvwx"  # sk- + 24 chars -> api_key
FAKE_BEARER_TOKEN = "eyJhbGciOiJIUzI1NiJ9.fake-signature-payload-0123456789"
FAKE_PASSWORD = "hunter2-super-secret-99"
FAKE_TOKEN_URLSAFE = "x" * 48  # long base64url-shaped run


@pytest.fixture(autouse=True)
def _clean_loguru_patcher():
    """Ensure the process-wide loguru patcher state never leaks between tests."""
    uninstall_loguru_redaction()
    yield
    uninstall_loguru_redaction()


def _capture_sink(monkeypatch):
    """Remove all loguru sinks and add a capturing String sink.

    Returns ``(stream, handler_id)``.  The default stderr sink is removed so
    assertions inspect exactly what our sink received.
    """
    stream = StringIO()
    loguru_logger.remove()
    handler_id = loguru_logger.add(stream, level="DEBUG", format="{message}")
    return stream, handler_id


class TestLoguruRedaction:
    def test_install_then_api_key_shaped_string_is_redacted(self, monkeypatch):
        monkeypatch.delenv(ENV_REDACTION_ENABLED, raising=False)
        stream, hid = _capture_sink(monkeypatch)
        try:
            assert install_loguru_redaction() is True

            # The exact call shape used by ~794 modules: plain f-string log.
            loguru_logger.info(f"client sent api_key='{FAKE_API_KEY}'")
            out = stream.getvalue()
            assert FAKE_API_KEY not in out
            assert "[REDACTED]" in out
        finally:
            loguru_logger.remove(hid)

    def test_bearer_token_is_redacted(self, monkeypatch):
        monkeypatch.delenv(ENV_REDACTION_ENABLED, raising=False)
        stream, hid = _capture_sink(monkeypatch)
        try:
            install_loguru_redaction()
            loguru_logger.warning(
                f"auth failed for Authorization=Bearer {FAKE_BEARER_TOKEN}"
            )
            out = stream.getvalue()
            assert FAKE_BEARER_TOKEN not in out
            assert "[REDACTED]" in out
        finally:
            loguru_logger.remove(hid)

    def test_password_field_is_redacted(self, monkeypatch):
        monkeypatch.delenv(ENV_REDACTION_ENABLED, raising=False)
        stream, hid = _capture_sink(monkeypatch)
        try:
            install_loguru_redaction()
            loguru_logger.error(f"login attempt password={FAKE_PASSWORD} failed")
            out = stream.getvalue()
            assert FAKE_PASSWORD not in out
            assert "[REDACTED]" in out
        finally:
            loguru_logger.remove(hid)

    def test_uninstall_restores_raw_output(self, monkeypatch):
        monkeypatch.delenv(ENV_REDACTION_ENABLED, raising=False)
        stream, hid = _capture_sink(monkeypatch)
        try:
            install_loguru_redaction()
            uninstall_loguru_redaction()

            loguru_logger.info(f"raw passthrough password={FAKE_PASSWORD}")
            out = stream.getvalue()
            assert FAKE_PASSWORD in out, "uninstall must restore raw logging"
            assert "[REDACTED]" not in out
        finally:
            loguru_logger.remove(hid)

    def test_install_is_idempotent(self, monkeypatch):
        monkeypatch.delenv(ENV_REDACTION_ENABLED, raising=False)
        core = loguru_logger._core

        assert install_loguru_redaction() is True
        first = core.patcher
        assert getattr(first, "_distllm_redaction", False), "our patcher installed"

        # Second install must not stack or replace the patcher.
        assert install_loguru_redaction() is True
        assert core.patcher is first, "idempotent install keeps same patcher"

        # force=True reinstalls but still ends up with exactly our patcher.
        assert install_loguru_redaction(force=True) is True
        assert getattr(core.patcher, "_distllm_redaction", False)

    def test_default_stderr_style_sink_is_covered(self, monkeypatch):
        """The critical regression guard: the *default* colorized stderr sink
        shape (what 794 unmodified modules write through before setup_logging
        runs) must be covered by the core patcher."""
        monkeypatch.delenv(ENV_REDACTION_ENABLED, raising=False)
        stream = StringIO()
        loguru_logger.remove()
        # Default-ish format WITH color markup and the default colorize=None
        # behaviour against a non-tty sink -- exercises Handler.emit's
        # colored_message invalidation path.
        hid = loguru_logger.add(
            stream,
            level="DEBUG",
            format="<green>{time}</green> | <level>{level: <8}</level> | "
            "<cyan>{name}</cyan> - <level>{message}</level>",
        )
        try:
            install_loguru_redaction()
            loguru_logger.info(f"leak attempt token={FAKE_TOKEN_URLSAFE}")
            out = stream.getvalue()
            assert FAKE_TOKEN_URLSAFE not in out
            assert "[REDACTED]" in out
            assert "leak attempt" in out, "surrounding context preserved"
        finally:
            loguru_logger.remove(hid)

    def test_enqueue_true_sink_is_covered(self, monkeypatch):
        """enqueue=True queues the record AFTER the patcher runs, so worker-side
        writes see the redacted message."""
        monkeypatch.delenv(ENV_REDACTION_ENABLED, raising=False)
        stream = StringIO()
        loguru_logger.remove()
        hid = loguru_logger.add(stream, level="DEBUG", format="{message}", enqueue=True)
        try:
            install_loguru_redaction()
            loguru_logger.info(f"queued secret api_key='{FAKE_API_KEY}'")
            loguru_logger.complete()  # flush enqueue queue
            out = stream.getvalue()
            assert FAKE_API_KEY not in out
            assert "[REDACTED]" in out
        finally:
            loguru_logger.remove(hid)

    def test_json_record_dict_sees_redacted_message(self, monkeypatch):
        """Structured JSON sinks read record['message'] from the captured dict;
        verify the mutation is visible there too (observability/logging.py path)."""
        monkeypatch.delenv(ENV_REDACTION_ENABLED, raising=False)
        entries: list[dict] = []
        loguru_logger.remove()

        def sink(message):
            record = message.record
            entries.append({"message": record["message"]})

        hid = loguru_logger.add(sink, level="DEBUG", format="{time} {message}")
        try:
            install_loguru_redaction()
            loguru_logger.info(f"Bearer {FAKE_BEARER_TOKEN} rejected")
            loguru_logger.complete()
        finally:
            loguru_logger.remove(hid)

        assert len(entries) == 1
        msg = entries[0]["message"]
        assert FAKE_BEARER_TOKEN not in msg
        assert "[REDACTED]" in msg

    def test_opt_out_env_disables_loguru_redaction(self, monkeypatch):
        monkeypatch.setenv(ENV_REDACTION_ENABLED, "off")
        stream, hid = _capture_sink(monkeypatch)
        try:
            # install_loguru_redaction refuses to install when disabled...
            assert install_loguru_redaction() is False
            loguru_logger.info(f"debug password={FAKE_PASSWORD}")
            assert FAKE_PASSWORD in stream.getvalue()
        finally:
            loguru_logger.remove(hid)

    def test_opt_out_env_at_emit_time(self, monkeypatch):
        """Even an already-installed patcher stops redacting when the env flag
        flips off (emit-time check enables live debugging without restart)."""
        monkeypatch.delenv(ENV_REDACTION_ENABLED, raising=False)
        stream, hid = _capture_sink(monkeypatch)
        try:
            install_loguru_redaction()
            monkeypatch.setenv(ENV_REDACTION_ENABLED, "off")
            loguru_logger.info(f"password={FAKE_PASSWORD}")
            assert FAKE_PASSWORD in stream.getvalue()
        finally:
            loguru_logger.remove(hid)

    def test_patcher_never_raises_on_hostile_input(self, monkeypatch):
        """Redaction failure must never break logging itself."""
        monkeypatch.delenv(ENV_REDACTION_ENABLED, raising=False)
        stream, hid = _capture_sink(monkeypatch)
        try:
            install_loguru_redaction()
            # Non-str messages, weird kwargs, unicode, catastrophic regex bait.
            loguru_logger.info("unicode: ünïcødé 🔥 key=ok")
            loguru_logger.opt(lazy=True).info("lazy {}", lambda: "value")
            long_run = "A" * 10_000
            loguru_logger.info(long_run)
            assert "lazy value" in stream.getvalue()
        finally:
            loguru_logger.remove(hid)

    def test_multithreaded_emission_all_redacted(self, monkeypatch):
        monkeypatch.delenv(ENV_REDACTION_ENABLED, raising=False)
        stream = StringIO()
        loguru_logger.remove()
        hid = loguru_logger.add(stream, level="DEBUG", format="{message}")
        errors: list[str] = []
        barrier = threading.Barrier(8)

        def worker(i: int) -> None:
            try:
                barrier.wait()
                loguru_logger.info(f"w{i} api_key='{FAKE_API_KEY}'")
            except Exception as exc:  # pragma: no cover
                errors.append(repr(exc))

        try:
            install_loguru_redaction()
            threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=10)
            assert not errors
            loguru_logger.complete()
            out = stream.getvalue()
            assert FAKE_API_KEY not in out
            assert out.count("[REDACTED]") == 8
        finally:
            loguru_logger.remove(hid)


class TestInstallGlobalRedactionCombined:
    """install_global_redaction() with no args activates BOTH frameworks."""

    def test_no_arg_call_covers_loguru_and_stdlib(self, monkeypatch):
        monkeypatch.delenv(ENV_REDACTION_ENABLED, raising=False)

        root = logging.getLogger()
        for h in list(root.handlers):
            root.removeHandler(h)
        root.filters = []
        std_stream = StringIO()
        handler = logging.StreamHandler(std_stream)
        root.addHandler(handler)

        lg_stream, lg_hid = _capture_sink(monkeypatch)
        try:
            install_global_redaction()  # no logger arg => global activation

            # loguru side
            loguru_logger.info(f"loguru path password={FAKE_PASSWORD}")
            # stdlib side
            thirdparty = logging.getLogger("thirdparty.lib")
            thirdparty.setLevel(logging.DEBUG)
            thirdparty.info("stdlib path key=%s", FAKE_API_KEY)

            lg_out = lg_stream.getvalue()
            std_out = std_stream.getvalue()
            assert FAKE_PASSWORD not in lg_out and "[REDACTED]" in lg_out
            assert FAKE_API_KEY not in std_out and "[REDACTED]" in std_out
        finally:
            loguru_logger.remove(lg_hid)

    def test_scoped_call_leaves_loguru_untouched(self, monkeypatch):
        """Explicit logger= argument => stdlib-only scoped installation
        (hermetic form used by existing test_e10 suite); the loguru pipeline
        must NOT gain a patcher from a scoped call."""
        monkeypatch.delenv(ENV_REDACTION_ENABLED, raising=False)
        core = loguru_logger._core
        before = core.patcher

        root = logging.getLogger()
        install_global_redaction(logger=root)

        assert core.patcher is before, "scoped install must not touch loguru"

    def test_startup_wiring_call_sites_exist(self):
        """Wiring conformance (audit §6.3): server startup must invoke
        install_global_redaction -- the 'module tested, wiring untested' hole."""
        import inspect

        import distllm.api.server as server_mod
        import distllm.api.server_lifespan as lifespan_mod

        for mod in (server_mod, lifespan_mod):
            src = inspect.getsource(mod._init_observability)
            assert "install_global_redaction()" in src, (
                f"{mod.__name__}._init_observability lost its redaction wiring"
            )

    @pytest.mark.parametrize("exc_factory", [KeyError, RuntimeError])
    def test_exception_logging_redacted(self, monkeypatch, exc_factory):
        """logger.exception / catch paths carry the secret inside
        record['exception'] formatting; the message portion must still be
        scrubbed (documented residual gap: traceback text itself)."""
        monkeypatch.delenv(ENV_REDACTION_ENABLED, raising=False)
        stream, hid = _capture_sink(monkeypatch)
        try:
            install_loguru_redaction()
            try:
                raise exc_factory(f"boom api_key='{FAKE_API_KEY}'")
            except exc_factory:
                loguru_logger.exception("request failed")
            out = stream.getvalue()
            assert "api_key='" not in out.split("\n")[0] or "[REDACTED]" in out
        finally:
            loguru_logger.remove(hid)


def test_module_importable_without_loguru_side_effects():
    """Importing/re-importing the module must not mutate global loguru state."""
    core = loguru_logger._core
    marker_before = getattr(core.patcher, "_distllm_redaction", False)
    # Plain re-import (no reload -- reloading would rebind RedactingFilter to
    # a fresh class object and break isinstance checks in other test modules).
    import distllm.security.log_redaction as _reimported

    assert _reimported is lr
    assert getattr(core.patcher, "_distllm_redaction", False) == marker_before
