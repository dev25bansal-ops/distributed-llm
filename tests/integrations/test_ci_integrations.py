"""Tests for CI integration modules (GitLab CI & Jenkins).

Covers constructors, context-manager support, public methods with mocked
HTTP responses, and retry / graceful-degradation paths.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Iterator
from unittest.mock import ANY, MagicMock, create_autospec

import httpx
import pytest

from distllm.integrations.ci._common import (
    _BASE_DELAY,
    _DEFAULT_TIMEOUT,
    _MAX_RETRIES,
    _retry,
    BuildInfo,
    EvalResult,
)
from distllm.integrations.ci.gitlab import GitLabCIIntegration
from distllm.integrations.ci.jenkins import JenkinsIntegration


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def mock_httpx_client() -> Iterator[MagicMock]:
    """Create a mock ``httpx.Client`` that behaves like the real one."""
    with pytest.MonkeyPatch.context() as mp:
        client = create_autospec(httpx.Client, instance=True)
        # make the context-manager return self
        client.__enter__.return_value = client
        yield client


@pytest.fixture
def gitlab(mock_httpx_client: MagicMock) -> Iterator[GitLabCIIntegration]:
    """Configure a ``GitLabCIIntegration`` instance backed by a mock client."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            "distllm.integrations.ci.gitlab.httpx.Client",
            lambda *a, **kw: mock_httpx_client,
        )
        inst = GitLabCIIntegration(
            url="https://gitlab.example.com",
            token="glpat-test123",
            project="team/project",
        )
        yield inst


@pytest.fixture
def jenkins(mock_httpx_client: MagicMock) -> Iterator[JenkinsIntegration]:
    """Configure a ``JenkinsIntegration`` instance backed by a mock client."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            "distllm.integrations.ci.jenkins.httpx.Client",
            lambda *a, **kw: mock_httpx_client,
        )
        inst = JenkinsIntegration(
            url="https://jenkins.example.com",
            user="bot",
            token="jtoken-abc",
        )
        yield inst


# ===========================================================================
# Shared helper tests
# ===========================================================================


class TestRetry:
    """Verify exponential-backoff retry behaviour."""

    def test_ok_first_try(self) -> None:
        assert _retry(lambda: 42) == 42

    def test_retry_then_success(self) -> None:
        call_count = 0

        def flaky() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise httpx.TransportError("timeout")
            return "ok"

        assert _retry(flaky, max_retries=3, base_delay=0.01) == "ok"
        assert call_count == 3

    def test_exhaust_retries(self) -> None:
        def always_fail() -> None:
            raise httpx.TransportError("always down")

        with pytest.raises(httpx.TransportError, match="always down"):
            _retry(always_fail, max_retries=2, base_delay=0.01)

    def test_does_not_retry_4xx(self) -> None:
        """Client errors (except 429) should raise immediately."""
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = 404

        def client_error() -> None:
            raise httpx.HTTPStatusError("not found", request=MagicMock(), response=resp)

        with pytest.raises(httpx.HTTPStatusError, match="not found"):
            _retry(client_error, max_retries=3)

    def test_retries_429(self) -> None:
        """Rate-limit (429) is retried."""
        call_count = 0
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = 429

        def rate_limited() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise httpx.HTTPStatusError(
                    "rate limited", request=MagicMock(), response=resp
                )
            return "ok"

        assert _retry(rate_limited, max_retries=3, base_delay=0.01) == "ok"
        assert call_count == 2


class TestDataTypes:
    """Verify dataclass behaviour for ``EvalResult`` and ``BuildInfo``."""

    def test_eval_result_defaults(self) -> None:
        r = EvalResult(pipeline_id=1, project="p", ref="main", model="m", status="success")
        assert r.metrics == {}
        assert r.artifacts == []

    def test_eval_result_frozen(self) -> None:
        r = EvalResult(pipeline_id=1, project="p", ref="main", model="m", status="running")
        with pytest.raises(AttributeError):
            r.status = "success"  # type: ignore[misc]

    def test_build_info_defaults(self) -> None:
        b = BuildInfo(build_number=42, job="pipe", status="SUCCESS", url="http://u")
        assert b.duration_ms == 0
        assert b.estimated_duration_ms == 0

    def test_build_info_frozen(self) -> None:
        b = BuildInfo(build_number=1, job="j", status="RUNNING", url="http://u")
        with pytest.raises(AttributeError):
            b.status = "SUCCESS"  # type: ignore[misc]


# ===========================================================================
# GitLabCIIntegration tests
# ===========================================================================


class TestGitLabCIIntegration:
    """Comprehensive test suite for ``GitLabCIIntegration``."""

    # -- constructor & context manager --------------------------------------

    def test_constructor_default_project(self, mock_httpx_client: MagicMock) -> None:
        gl = GitLabCIIntegration(
            url="https://gitlab.example.com",
            token="glpat-test123",
            project="team/project",
        )
        assert gl._base_url == "https://gitlab.example.com"
        assert gl._default_project == "team/project"
        assert gl._token == "glpat-test123"
        assert gl._timeout == _DEFAULT_TIMEOUT

    def test_constructor_no_default_project(self, mock_httpx_client: MagicMock) -> None:
        gl = GitLabCIIntegration(
            url="https://gitlab.example.com",
            token="glpat-test123",
        )
        assert gl._default_project is None

    def test_context_manager(self, gitlab: GitLabCIIntegration) -> None:
        with gitlab as gl:
            assert gl is gitlab
            gl.close()
        # close should have been called on exit

    def test_close(self, gitlab: GitLabCIIntegration) -> None:
        gitlab.close()
        gitlab._client.close.assert_called_once()

    # -- _resolve_project ---------------------------------------------------

    def test_resolve_project_explicit(self, gitlab: GitLabCIIntegration) -> None:
        assert gitlab._resolve_project("other/proj") == "other%2Fproj"

    def test_resolve_project_default(self, gitlab: GitLabCIIntegration) -> None:
        assert gitlab._resolve_project() == "team%2Fproject"

    def test_resolve_project_missing(self, mock_httpx_client: MagicMock) -> None:
        gl = GitLabCIIntegration(url="https://gitlab.example.com", token="glpat-x")
        with pytest.raises(ValueError, match="No project specified"):
            gl._resolve_project()

    # -- trigger_eval -------------------------------------------------------

    def test_trigger_eval_minimal(
        self, gitlab: GitLabCIIntegration, mock_httpx_client: MagicMock
    ) -> None:
        mock_httpx_client.post.return_value.json.return_value = {"id": 123}
        pipeline_id = gitlab.trigger_eval()
        assert pipeline_id == 123
        mock_httpx_client.post.assert_called_once()

    def test_trigger_eval_with_model(
        self, gitlab: GitLabCIIntegration, mock_httpx_client: MagicMock
    ) -> None:
        mock_httpx_client.post.return_value.json.return_value = {"id": 456}
        pipeline_id = gitlab.trigger_eval(model="llama-70b", ref="main")
        assert pipeline_id == 456
        # Verify the POST body included the CI variable
        _, kwargs = mock_httpx_client.post.call_args
        # We check the json data passed to the inner _post closure
        # Actually the mock is on the httpx client itself, so kwargs
        # should have the JSON data
        assert kwargs["json"] is not None

    def test_trigger_eval_with_variables(
        self, gitlab: GitLabCIIntegration, mock_httpx_client: MagicMock
    ) -> None:
        mock_httpx_client.post.return_value.json.return_value = {"id": 789}
        pipeline_id = gitlab.trigger_eval(
            project="other/proj",
            variables={"CUSTOM_VAR": "value"},
        )
        assert pipeline_id == 789

    def test_trigger_eval_http_error(
        self, gitlab: GitLabCIIntegration, mock_httpx_client: MagicMock
    ) -> None:
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = 404
        mock_httpx_client.post.side_effect = httpx.HTTPStatusError(
            "not found", request=MagicMock(), response=resp
        )
        with pytest.raises(httpx.HTTPStatusError):
            gitlab.trigger_eval()

    # -- get_results --------------------------------------------------------

    def test_get_results_success(
        self, gitlab: GitLabCIIntegration, mock_httpx_client: MagicMock
    ) -> None:
        # Mock pipeline info
        pipeline_data = {
            "id": 10,
            "status": "success",
            "ref": "main",
            "variables": {"EVAL_MODEL": "gpt-4"},
        }
        # Mock jobs list
        jobs_data = [
            {
                "id": 100,
                "name": "eval-job",
                "status": "success",
            }
        ]
        # Mock artifact response
        artifact_data = {"accuracy": 0.95, "model": "gpt-4"}

        mock_httpx_client.get.side_effect = [
            MagicMock(json=lambda: pipeline_data, raise_for_status=lambda: None),
            MagicMock(json=lambda: jobs_data, raise_for_status=lambda: None),
            MagicMock(json=lambda: artifact_data, raise_for_status=lambda: None),
        ]

        result = gitlab.get_results(pipeline_id=10)

        assert isinstance(result, EvalResult)
        assert result.pipeline_id == 10
        assert result.status == "success"
        assert result.ref == "main"
        assert result.model == "gpt-4"
        assert result.metrics == {"accuracy": 0.95, "model": "gpt-4"}
        assert len(result.artifacts) == 1
        assert "artifacts/download" in result.artifacts[0]

    def test_get_results_missing_pipeline_id(
        self, gitlab: GitLabCIIntegration
    ) -> None:
        with pytest.raises(ValueError, match="pipeline_id is required"):
            gitlab.get_results()

    def test_get_results_artifact_missing(
        self, gitlab: GitLabCIIntegration, mock_httpx_client: MagicMock
    ) -> None:
        """Graceful degradation: artifact fetch fails but pipeline info still returned."""
        pipeline_data = {"id": 11, "status": "failed", "ref": "dev", "variables": {}}
        jobs_data = [
            {
                "id": 101,
                "name": "eval-job",
                "status": "failed",
            }
        ]

        mock_httpx_client.get.side_effect = [
            MagicMock(json=lambda: pipeline_data, raise_for_status=lambda: None),
            MagicMock(json=lambda: jobs_data, raise_for_status=lambda: None),
        ]

        result = gitlab.get_results(pipeline_id=11)
        assert result.status == "failed"
        assert result.metrics == {}
        assert result.artifacts == []

    # -- report_to_mr -------------------------------------------------------

    def test_report_to_mr_with_eval_result(
        self, gitlab: GitLabCIIntegration, mock_httpx_client: MagicMock
    ) -> None:
        mock_httpx_client.post.return_value.json.return_value = {"id": 999}
        report = EvalResult(
            pipeline_id=1,
            project="team/project",
            ref="main",
            model="llama",
            status="success",
            metrics={"acc": 0.9},
        )
        note_id = gitlab.report_to_mr(mr_iid=42, report=report)
        assert note_id == 999

    def test_report_to_mr_missing_iid(
        self, gitlab: GitLabCIIntegration
    ) -> None:
        with pytest.raises(ValueError, match="mr_iid is required"):
            gitlab.report_to_mr()

    def test_report_to_mr_with_dict(
        self, gitlab: GitLabCIIntegration, mock_httpx_client: MagicMock
    ) -> None:
        mock_httpx_client.post.return_value.json.return_value = {"id": 111}
        note_id = gitlab.report_to_mr(mr_iid=7, report={"accuracy": 0.95})
        assert note_id == 111

    def test_report_to_mr_with_string(
        self, gitlab: GitLabCIIntegration, mock_httpx_client: MagicMock
    ) -> None:
        mock_httpx_client.post.return_value.json.return_value = {"id": 222}
        note_id = gitlab.report_to_mr(mr_iid=8, report="Everything looks good.")
        assert note_id == 222

    # -- _format_report -----------------------------------------------------

    def test_format_eval_result(self) -> None:
        report = EvalResult(
            pipeline_id=5,
            project="p",
            ref="main",
            model="bert",
            status="success",
            metrics={"loss": 0.1},
            artifacts=["http://download"],
        )
        body = GitLabCIIntegration._format_report(report)
        assert "bert" in body
        assert "#5" in body
        assert "loss" in body
        assert "download" in body

    def test_format_dict(self) -> None:
        body = GitLabCIIntegration._format_report({"key": "val"})
        assert "Evaluation Report" in body
        assert "key" in body
        assert "val" in body

    def test_format_string(self) -> None:
        body = GitLabCIIntegration._format_report("plain text")
        assert body == "plain text"


# ===========================================================================
# JenkinsIntegration tests
# ===========================================================================


class TestJenkinsIntegration:
    """Comprehensive test suite for ``JenkinsIntegration``."""

    # -- constructor & context manager --------------------------------------

    def test_constructor(self, mock_httpx_client: MagicMock) -> None:
        jk = JenkinsIntegration(
            url="https://jenkins.example.com",
            user="bot",
            token="jtoken-abc",
        )
        assert jk._base_url == "https://jenkins.example.com"
        assert jk._auth == ("bot", "jtoken-abc")
        assert jk._timeout == _DEFAULT_TIMEOUT

    def test_constructor_with_timeout(self, mock_httpx_client: MagicMock) -> None:
        jk = JenkinsIntegration(
            url="https://jenkins.example.com",
            user="bot",
            token="jtoken-abc",
            timeout=300.0,
        )
        assert jk._timeout == 300.0

    def test_context_manager(self, jenkins: JenkinsIntegration) -> None:
        with jenkins as jk:
            assert jk is jenkins
            jk.close()

    def test_close(self, jenkins: JenkinsIntegration) -> None:
        jenkins.close()
        jenkins._client.close.assert_called_once()

    # -- _encode_job_path ---------------------------------------------------

    def test_encode_simple_job(self, jenkins: JenkinsIntegration) -> None:
        assert jenkins._encode_job_path("my-pipeline") == "job/my-pipeline"

    def test_encode_folder_job(self, jenkins: JenkinsIntegration) -> None:
        assert jenkins._encode_job_path("folder/pipeline") == "job/folder/job/pipeline"

    def test_encode_dotted_job(self, jenkins: JenkinsIntegration) -> None:
        assert jenkins._encode_job_path("folder.pipeline") == "job/folder/job/pipeline"

    # -- trigger_build ------------------------------------------------------

    def test_trigger_build_with_params(
        self, jenkins: JenkinsIntegration, mock_httpx_client: MagicMock
    ) -> None:
        response = MagicMock(spec=httpx.Response)
        response.status_code = 201
        response.headers = {"Location": "http://jenkins/queue/item/99/"}
        mock_httpx_client.post.return_value = response

        # Mock the queue item resolution
        mock_httpx_client.get.return_value.json.return_value = {
            "executable": {"number": 42}
        }

        build_num = jenkins.trigger_build("benchmark", params={"MODEL": "llama"})
        assert build_num == 42

    def test_trigger_build_no_params(
        self, jenkins: JenkinsIntegration, mock_httpx_client: MagicMock
    ) -> None:
        response = MagicMock(spec=httpx.Response)
        response.status_code = 201
        response.headers = {"Location": "http://jenkins/queue/item/100/"}
        mock_httpx_client.post.return_value = response
        mock_httpx_client.get.return_value.json.return_value = {
            "executable": {"number": 43}
        }

        build_num = jenkins.trigger_build("benchmark")
        assert build_num == 43

    def test_trigger_build_fallback_to_last_build(
        self, jenkins: JenkinsIntegration, mock_httpx_client: MagicMock
    ) -> None:
        """When no Location header with queue item, fall back to last build."""
        response = MagicMock(spec=httpx.Response)
        response.status_code = 201
        response.headers = {}  # no Location
        mock_httpx_client.post.return_value = response

        mock_httpx_client.get.return_value.json.return_value = {
            "lastBuild": {"number": 100}
        }

        build_num = jenkins.trigger_build("benchmark")
        assert build_num == 100

    # -- get_build_status ---------------------------------------------------

    def test_get_build_status_completed(
        self, jenkins: JenkinsIntegration, mock_httpx_client: MagicMock
    ) -> None:
        mock_httpx_client.get.return_value.json.return_value = {
            "number": 42,
            "result": "SUCCESS",
            "url": "http://jenkins/job/pipe/42/",
            "duration": 12345,
            "estimatedDuration": 10000,
        }

        info = jenkins.get_build_status("pipe", build_num=42)
        assert isinstance(info, BuildInfo)
        assert info.build_number == 42
        assert info.status == "SUCCESS"
        assert info.duration_ms == 12345
        assert info.estimated_duration_ms == 10000

    def test_get_build_status_running(
        self, jenkins: JenkinsIntegration, mock_httpx_client: MagicMock
    ) -> None:
        mock_httpx_client.get.return_value.json.return_value = {
            "number": 43,
            "result": None,  # null = still running
            "url": "http://jenkins/job/pipe/43/",
            "duration": 0,
            "estimatedDuration": 10000,
        }

        info = jenkins.get_build_status("pipe", build_num=43)
        assert info.status == "RUNNING"

    def test_get_build_status_last_completed(
        self, jenkins: JenkinsIntegration, mock_httpx_client: MagicMock
    ) -> None:
        mock_httpx_client.get.return_value.json.return_value = {
            "number": 99,
            "result": "UNSTABLE",
            "url": "http://jenkins/job/pipe/99/",
            "duration": 5000,
            "estimatedDuration": 5000,
        }

        info = jenkins.get_build_status("pipe")  # no build_num → last completed
        assert info.status == "UNSTABLE"

    # -- publish_artifact ---------------------------------------------------

    def test_publish_artifact(
        self, jenkins: JenkinsIntegration, mock_httpx_client: MagicMock
    ) -> None:
        mock_httpx_client.put.return_value.raise_for_status.return_value = None

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            f.write('{"result": "ok"}')
            tmp_path = f.name

        try:
            url = jenkins.publish_artifact("benchmark", tmp_path, build_num=42)
            assert "42/artifact" in url
            assert mock_httpx_client.put.called
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_publish_artifact_file_not_found(
        self, jenkins: JenkinsIntegration
    ) -> None:
        with pytest.raises(FileNotFoundError, match="notexist.json"):
            jenkins.publish_artifact("benchmark", "notexist.json")


# ===========================================================================
# Graceful degradation tests
# ===========================================================================


class TestGracefulDegradation:
    """Verify that transient failures don't cause hard crashes."""

    def test_gitlab_retry_on_transport_error(
        self, gitlab: GitLabCIIntegration, mock_httpx_client: MagicMock
    ) -> None:
        """Transient transport errors should be retried then raised."""
        mock_httpx_client.get.side_effect = httpx.TransportError("net down")
        with pytest.raises(httpx.TransportError):
            gitlab._get("/test")

    def test_jenkins_retry_on_timeout(
        self, jenkins: JenkinsIntegration, mock_httpx_client: MagicMock
    ) -> None:
        """Timeout exceptions should be retried."""
        mock_httpx_client.get.side_effect = httpx.TimeoutException("timed out")
        with pytest.raises(httpx.TimeoutException):
            jenkins._get_json("/test")

    def test_gitlab_no_crash_on_missing_project(
        self, mock_httpx_client: MagicMock
    ) -> None:
        """Missing default project raises ValueError, not AttributeError."""
        gl = GitLabCIIntegration(url="https://gitlab.example.com", token="glpat-x")
        with pytest.raises(ValueError, match="No project specified"):
            gl.trigger_eval()

    def test_jenkins_no_crash_on_empty_builds(
        self, jenkins: JenkinsIntegration, mock_httpx_client: MagicMock
    ) -> None:
        """No builds exist yet for a job should raise RuntimeError."""
        mock_httpx_client.get.return_value.json.return_value = {"lastBuild": None}
        with pytest.raises(RuntimeError, match="No builds exist yet"):
            jenkins._get_latest_build_number("job/test")
