"""GitLab CI / Jenkins integrations for model evaluation and benchmarking.

Provides two integration classes:

- :class:`GitLabCIIntegration` — trigger evaluation pipelines on GitLab CI,
  fetch pipeline results, and post reports as merge-request comments.
- :class:`JenkinsIntegration` — trigger Jenkins builds with arbitrary
  parameters, query build status, and publish build artifacts.

Usage::

    from integrations.gitlab_ci import GitLabCIIntegration, JenkinsIntegration

    # GitLab: trigger model eval on a merge-request ref
    gl = GitLabCIIntegration(url="https://gitlab.example.com", token="glpat-xxx")
    pipeline_id = gl.trigger_eval("my-team/distllm-bench", "refs/heads/main", "llama-70b")
    results = gl.get_results("my-team/distllm-bench", pipeline_id)
    gl.report_to_mr("my-team/distllm-bench", 42, results)

    # Jenkins: trigger a benchmark build
    jk = JenkinsIntegration(url="https://jenkins.example.com", user="bot", token="xxxx")
    build_num = jk.trigger_build("benchmark-pipeline", {"MODEL": "llama-70b"})
    status = jk.get_build_status("benchmark-pipeline", build_num)
    jk.publish_artifact("benchmark-pipeline", "results.json")
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import httpx

logger = logging.getLogger("distllm")

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_DEFAULT_TIMEOUT = 120.0  # seconds
_MAX_RETRIES = 3
_BASE_DELAY = 1.0


def _retry(
    fn: Any,
    *args: Any,
    max_retries: int = _MAX_RETRIES,
    base_delay: float = _BASE_DELAY,
    **kwargs: Any,
) -> Any:
    """Call *fn* with exponential backoff on transient failures."""
    last_exc: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            return fn(*args, **kwargs)
        except httpx.HTTPStatusError as exc:
            # Do not retry 4xx client errors (except 429 rate-limit).
            status = exc.response.status_code
            if 400 <= status < 500 and status != 429:
                raise
            last_exc = exc
        except (httpx.TransportError, httpx.TimeoutException) as exc:
            last_exc = exc
        if attempt < max_retries - 1:
            delay = min(base_delay * (2**attempt), 30.0)
            logger.warning(
                "Attempt %d/%d failed (%s), retrying in %.1fs …",
                attempt + 1,
                max_retries,
                last_exc,
                delay,
            )
            time.sleep(delay)
    raise last_exc  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvalResult:
    """Outcome of a single evaluation run."""

    pipeline_id: int
    project: str
    ref: str
    model: str
    status: str  # e.g. "success", "failed", "running"
    metrics: dict[str, Any] = field(default_factory=dict)
    artifacts: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class BuildInfo:
    """Jenkins build status and metadata."""

    build_number: int
    job: str
    status: str  # e.g. "SUCCESS", "FAILURE", "UNSTABLE", "RUNNING"
    url: str
    duration_ms: int = 0
    estimated_duration_ms: int = 0


# ---------------------------------------------------------------------------
# GitLab CI integration
# ---------------------------------------------------------------------------


class GitLabCIIntegration:
    """Trigger and monitor model evaluation pipelines on GitLab CI.

    Parameters
    ----------
    url:
        GitLab instance base URL (e.g. ``https://gitlab.com``).
    token:
        Personal access token or project CI/CD token with ``api`` scope.
    project:
        Default project path (e.g. ``"my-org/my-project"``).  When
        *project* is omitted from method calls this default is used.
    timeout:
        HTTP request timeout in seconds.
    verify_ssl:
        Whether to verify TLS certificates.
    """

    def __init__(
        self,
        url: str,
        token: str,
        project: Optional[str] = None,
        timeout: float = _DEFAULT_TIMEOUT,
        verify_ssl: bool = True,
    ) -> None:
        self._base_url = url.rstrip("/")
        self._default_project = project
        self._token = token
        self._timeout = timeout
        self._client = httpx.Client(
            base_url=self._base_url,
            headers={
                "PRIVATE-TOKEN": token,
                "Accept": "application/json",
            },
            timeout=httpx.Timeout(timeout),
            verify=verify_ssl,
        )

    # -- convenience helpers ------------------------------------------------

    def _api_url(self, path: str) -> str:
        return f"{self._base_url}/api/v4{path}"

    def _resolve_project(self, project: Optional[str] = None) -> str:
        resolved = project or self._default_project
        if not resolved:
            raise ValueError(
                "No project specified. Pass it explicitly or set a default "
                "in the constructor."
            )
        return resolved.replace("/", "%2F")

    def _get(self, path: str, **params: Any) -> Any:
        url = self._api_url(path)

        def request() -> Any:
            resp = self._client.get(url, params=params)
            resp.raise_for_status()
            return resp.json()

        return _retry(request)

    def _post(self, path: str, json_data: dict[str, Any]) -> Any:
        url = self._api_url(path)

        def request() -> Any:
            resp = self._client.post(url, json=json_data)
            resp.raise_for_status()
            return resp.json()

        return _retry(request)

    # -- public API ---------------------------------------------------------

    def trigger_eval(
        self,
        project: Optional[str] = None,
        ref: Optional[str] = None,
        model: Optional[str] = None,
        variables: Optional[dict[str, str]] = None,
    ) -> int:
        """Trigger an evaluation pipeline on GitLab CI.

        Parameters
        ----------
        project:
            Project path (e.g. ``"my-org/my-project"``).  Falls back to
            the default passed at construction.
        ref:
            Branch or tag to run the pipeline on.  Defaults to the
            repository's default branch when ``None``.
        model:
            Short model identifier injected as a CI variable
            ``EVAL_MODEL`` so downstream jobs know which model to
            benchmark.  When set, it is merged into *variables*.
        variables:
            Additional CI variables to pass to the pipeline.

        Returns
        -------
        int
            The created pipeline ID.

        Raises
        ------
        httpx.HTTPStatusError
            On 4xx / 5xx responses (client errors are not retried).
        httpx.TransportError
            On network failures after all retries are exhausted.
        """
        proj = self._resolve_project(project)
        merged_vars: dict[str, str] = dict(variables or {})
        if model:
            merged_vars.setdefault("EVAL_MODEL", model)

        body: dict[str, Any] = {}
        if ref:
            body["ref"] = ref
        if merged_vars:
            body["variables"] = merged_vars

        data = self._post(f"/projects/{proj}/pipeline", body)
        pipeline_id: int = data["id"]
        logger.info(
            "Triggered pipeline %d on %s (ref=%s, model=%s)",
            pipeline_id,
            project or self._default_project,
            ref or "default",
            model or "unspecified",
        )
        return pipeline_id

    def get_results(
        self,
        project: Optional[str] = None,
        pipeline_id: Optional[int] = None,
    ) -> EvalResult:
        """Fetch evaluation results for a completed pipeline.

        Gathers the pipeline status and any evaluation metrics stored in
        job artifacts (``eval-metrics.json``), then wraps everything in an
        :class:`EvalResult`.

        Parameters
        ----------
        project:
            Project path.  Falls back to default.
        pipeline_id:
            Pipeline ID returned by :meth:`trigger_eval`.

        Returns
        -------
        EvalResult
            Aggregated evaluation result with metrics and artifact URLs.
        """
        if pipeline_id is None:
            raise ValueError("pipeline_id is required.")

        proj = self._resolve_project(project)

        # 1. Pipeline-level status.
        pipeline = self._get(f"/projects/{proj}/pipelines/{pipeline_id}")
        status: str = pipeline.get("status", "unknown")
        ref: str = pipeline.get("ref", "unknown")

        # 2. List jobs to find evaluation jobs and their artifacts.
        jobs = self._get(
            f"/projects/{proj}/pipelines/{pipeline_id}/jobs",
            per_page=100,
        )

        metrics: dict[str, Any] = {}
        artifact_urls: list[str] = []

        for job in jobs:
            job_id = job["id"]
            job_name = job.get("name", "")
            if job.get("status") == "success":
                # Attempt to fetch an eval-metrics.json artifact from each
                # successful job so downstream evaluation harvesters can store
                # structured results.
                try:
                    artifact = self._get(
                        f"/projects/{proj}/jobs/{job_id}/artifacts/eval-metrics.json",  # noqa: E501
                    )
                    if isinstance(artifact, dict):
                        metrics.update(artifact)
                except (httpx.HTTPStatusError, httpx.TransportError):
                    pass

                artifact_urls.append(
                    f"{self._base_url}/{project}/-/jobs/{job_id}/artifacts/download"  # noqa: E501
                )

        # 3. Extract eval-related model name from metrics or fall back.
        model = metrics.get("model") or pipeline.get("variables", {}).get(
            "EVAL_MODEL", ""
        )

        logger.info(
            "Pipeline %d on %s is %s (%d metrics)",
            pipeline_id,
            project or self._default_project,
            status,
            len(metrics),
        )

        return EvalResult(
            pipeline_id=pipeline_id,
            project=project or self._default_project or "",
            ref=ref,
            model=str(model) if model else "",
            status=status,
            metrics=metrics,
            artifacts=artifact_urls,
        )

    def report_to_mr(
        self,
        project: Optional[str] = None,
        mr_iid: Optional[int] = None,
        report: Optional[Any] = None,
    ) -> int:
        """Post evaluation results as a merge-request note.

        Parameters
        ----------
        project:
            Project path.  Falls back to default.
        mr_iid:
            Merge-request IID (the number shown in the MR URL, not the
            global ID).
        report:
            Structured report to post.  Can be an :class:`EvalResult`, a
            ``dict``, a ``str``, or any object with a ``__str__`` method.
            Dicts are formatted as a Markdown table; everything else is
            stringified.

        Returns
        -------
        int
            The created note ID.

        Raises
        ------
        ValueError
            If *mr_iid* is not provided.
        """
        if mr_iid is None:
            raise ValueError("mr_iid is required to post a report.")

        proj = self._resolve_project(project)

        body = self._format_report(report)
        data = self._post(
            f"/projects/{proj}/merge_requests/{mr_iid}/notes",
            {"body": body},
        )

        note_id: int = data["id"]
        logger.info(
            "Posted evaluation report as note %d on MR !%d (%s)",
            note_id,
            mr_iid,
            project or self._default_project,
        )
        return note_id

    @staticmethod
    def _format_report(report: Any) -> str:
        """Convert a structured report into a Markdown comment body."""
        if isinstance(report, EvalResult):
            lines = [
                f"## Evaluation Report — `{report.model}`",
                "",
                f"- **Pipeline**: #{report.pipeline_id}",
                f"- **Ref**: `{report.ref}`",
                f"- **Status**: {report.status}",
                "",
            ]
            if report.metrics:
                lines.append("| Metric | Value |")
                lines.append("|--------|-------|")
                for k, v in report.metrics.items():
                    lines.append(f"| {k} | {v} |")
                lines.append("")
            if report.artifacts:
                lines.append("**Artifacts:**")
                for url in report.artifacts:
                    lines.append(f"- [Download]({url})")
            return "\n".join(lines)

        if isinstance(report, dict):
            lines = ["## Evaluation Report", "", "| Metric | Value |", "|--------|-------|"]
            for k, v in report.items():
                lines.append(f"| {k} | {v} |")
            return "\n".join(lines)

        return str(report)

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._client.close()


# ---------------------------------------------------------------------------
# Jenkins integration
# ---------------------------------------------------------------------------


class JenkinsIntegration:
    """Trigger and monitor benchmark builds on Jenkins.

    Parameters
    ----------
    url:
        Jenkins server base URL (e.g. ``https://jenkins.example.com``).
    user:
        Jenkins username (for HTTP basic auth).
    token:
        Jenkins API token or password for *user*.
    timeout:
        HTTP request timeout in seconds.
    verify_ssl:
        Whether to verify TLS certificates.
    """

    def __init__(
        self,
        url: str,
        user: str,
        token: str,
        timeout: float = _DEFAULT_TIMEOUT,
        verify_ssl: bool = True,
    ) -> None:
        self._base_url = url.rstrip("/")
        self._auth = (user, token)
        self._timeout = timeout
        self._client = httpx.Client(
            base_url=self._base_url,
            auth=self._auth,
            headers={"Accept": "application/json"},
            timeout=httpx.Timeout(timeout),
            verify=verify_ssl,
        )

    # -- helpers ------------------------------------------------------------

    def _encode_job_path(self, job: str) -> str:
        """Encode a dotted or slash-separated Jenkins job path.

        ``"folder / pipeline"`` → ``"folder/job/pipeline"``.
        """
        parts = job.replace(".", "/").split("/")
        return "/".join(f"job/{p}" for p in parts if p)

    def _get_json(self, path: str, **params: Any) -> Any:
        url = f"{self._base_url}{path}"

        def request() -> Any:
            resp = self._client.get(url, params=params)
            resp.raise_for_status()
            return resp.json()

        return _retry(request)

    def _post_no_body(self, path: str, **params: Any) -> httpx.Response:
        url = f"{self._base_url}{path}"

        def request() -> httpx.Response:
            resp = self._client.post(url, params=params)
            resp.raise_for_status()
            return resp

        return _retry(request)

    # -- public API ---------------------------------------------------------

    def trigger_build(
        self,
        job: str,
        params: Optional[dict[str, str]] = None,
    ) -> int:
        """Trigger a parameterised Jenkins build.

        Parameters
        ----------
        job:
            Job name.  Supports folder notation (``"folder/pipeline"`` or
            ``"folder.pipeline"``).
        params:
            Build parameters as key-value pairs.

        Returns
        -------
        int
            The build number assigned to the triggered build.

        Raises
        ------
        httpx.HTTPStatusError
            On 4xx / 5xx (e.g. job not found, missing parameters).
        httpx.TransportError
            After retries are exhausted.
        """
        encoded = self._encode_job_path(job)

        if params:
            # Parameterised build via buildWithParameters.
            path = f"/{encoded}/buildWithParameters"
            response = self._post_no_body(path, **params)
        else:
            # Simple build without parameters.
            path = f"/{encoded}/build"
            response = self._post_no_body(path)

        # Jenkins responds 201 with a ``Location`` header that includes
        # the queue item URI.  We parse the queue item to find the
        # eventual build number — or poll the queue briefly.
        location = response.headers.get("Location", "")
        if "queue/item" in location:
            build_num = self._resolve_queue_item(location)
        else:
            # Fall back to fetching the last build number from the job API.
            build_num = self._get_latest_build_number(encoded)

        logger.info("Triggered Jenkins build #%d on %s", build_num, job)
        return build_num

    def _resolve_queue_item(self, location: str) -> int:
        """Poll a Jenkins queue item until it is assigned a build number."""
        queue_api = f"{location}api/json/"
        for _ in range(30):
            data = self._get_json(queue_api)
            if data.get("executable"):
                return data["executable"]["number"]
            time.sleep(1.0)
        raise RuntimeError(
            f"Queue item at {location} was not picked up within 30 s."
        )

    def _get_latest_build_number(self, encoded_job: str) -> int:
        """Fetch the last build number from the job API."""
        data = self._get_json(f"/{encoded_job}/api/json/")
        last_build = data.get("lastBuild")
        if last_build is None:
            raise RuntimeError(f"No builds exist yet for job '{encoded_job}'.")
        return last_build["number"]

    def get_build_status(
        self,
        job: str,
        build_num: Optional[int] = None,
    ) -> BuildInfo:
        """Query the status of a Jenkins build.

        Parameters
        ----------
        job:
            Job name (folder notation supported).
        build_num:
            Build number.  When ``None``, returns the status of the last
            completed build.

        Returns
        -------
        BuildInfo
            Current build status and metadata.
        """
        encoded = self._encode_job_path(job)
        if build_num is None:
            path = f"/{encoded}/lastCompletedBuild/api/json/"
        else:
            path = f"/{encoded}/{build_num}/api/json/"

        data = self._get_json(path)

        raw_result = data.get("result")
        if raw_result is None:
            # ``result`` is null while the build is still running.
            status = "RUNNING"
        else:
            status = str(raw_result)

        return BuildInfo(
            build_number=data["number"],
            job=job,
            status=status,
            url=data.get("url", ""),
            duration_ms=data.get("duration", 0),
            estimated_duration_ms=data.get("estimatedDuration", 0),
        )

    def publish_artifact(
        self,
        job: str,
        path: str,
        build_num: Optional[int] = None,
    ) -> str:
        """Upload a local file as a build artifact *via the Jenkins CLI API*.

        This uses the ``/artifact`` upload mechanism on a parameterised
        pipeline or freestyle job that exposes a file parameter.  For
        Pipeline jobs, prefer storing the file in a shared volume and
        referencing it from the ``Jenkinsfile``.

        Parameters
        ----------
        job:
            Job name (folder notation supported).
        path:
            Absolute or relative path to the local file to publish.
        build_num:
            Build number to attach the artifact to.  When ``None``, uses
            the last completed build.

        Returns
        -------
        str
            The public artifact URL on the Jenkins server.
        """
        encoded = self._encode_job_path(job)
        if build_num is None:
            build_info = self.get_build_status(job)
            build_num = build_info.build_number

        filepath = Path(path)
        if not filepath.is_file():
            raise FileNotFoundError(f"Artifact file not found: {path}")

        # Upload through the Jenkins file parameter endpoint.
        # This is most reliable with a parameterised freestyle that has a
        # File Parameter defined.  For Pipeline jobs the recommended
        # approach is to have the Jenkinsfile pull from a known location.
        url = f"{self._base_url}/{encoded}/{build_num}/artifact/{filepath.name}"

        with filepath.open("rb") as fh:
            resp = self._client.put(url, content=fh)
        resp.raise_for_status()

        logger.info("Published artifact %s to %s", filepath.name, url)
        return url

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._client.close()
