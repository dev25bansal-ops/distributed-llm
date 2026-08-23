"""Regression: Dify provider must not double-prefix /v1 and must honor model.

F-003: ``DistLLMProvider`` defaulted ``base_url`` to ``http://localhost:8000/v1``
and then posted to ``/v1/chat/completions`` — httpx concatenated to
``/v1/v1/chat/completions`` (404).  The base URL must be the bare origin.
"""

from __future__ import annotations

import json
from unittest import mock

from integrations.dify.distllm_provider import DistLLMProvider


class _FakeResponse:
    def __init__(self, status_code=200, data=None):
        self.status_code = status_code
        self._data = data or {"choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}], "usage": {}}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._data


def _recorded_calls():
    """Return a (paths, bodies) recorder for httpx client.post/get."""
    calls = []

    class _Ctx:
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False

    client = _Ctx()
    client.post = lambda path, json=None: (calls.append(("post", path, json)) or _FakeResponse())
    client.get = lambda path: (calls.append(("get", path, None)) or _FakeResponse())
    client.stream = lambda method, path, json=None: _StreamCtx(calls, path, json)
    return client, calls


class _StreamCtx:
    def __init__(self, calls, path, json):
        self._calls = calls
        self._path = path
        self._json = json

    def __enter__(self):
        self._calls.append(("stream", self._path, self._json))
        return self

    def __exit__(self, *a):
        return False

    def iter_text(self):
        payload = json.dumps({"choices": [{"delta": {"content": "x"}}]})
        yield f"data: {payload}\n"
        yield "data: [DONE]\n"


def _provider():
    client, calls = _recorded_calls()
    p = DistLLMProvider()
    p._get_client = mock.Mock(return_value=client)
    return p, calls


class TestDifyProviderURLs:
    def test_base_url_has_no_v1_prefix(self):
        p = DistLLMProvider()
        assert not p._api_base.endswith("/v1"), f"base_url must be bare origin, got {p._api_base!r}"

    def test_invoke_posts_to_single_v1_chat(self):
        p, calls = _provider()
        p.invoke("m1", {}, {"temperature": 0.5}, [{"role": "user", "content": "hi"}])
        paths = [c[1] for c in calls]
        assert any(pth == "/v1/chat/completions" for pth in paths), paths
        # No double-prefix.
        assert all("/v1/v1" not in pth for pth in paths)

    def test_invoke_honors_per_call_model(self):
        p, calls = _provider()
        p.invoke("my-model", {}, {}, [{"role": "user", "content": "hi"}])
        posted = [c[2] for c in calls if c[0] == "post"]
        assert posted and posted[0]["model"] == "my-model"

    def test_validate_credentials_hits_health_once(self):
        p, calls = _provider()
        p.validate_credentials("m", {})
        paths = [c[1] for c in calls if c[0] == "get"]
        assert paths and paths[0].endswith("/health"), paths

    def test_get_models_lists_models(self):
        p, calls = _provider()
        p._get_client = mock.Mock(return_value=_recorded_calls()[0])
        models = p.get_models()
        assert isinstance(models, list)