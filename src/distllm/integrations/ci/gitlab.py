"""GitLab CI integration for model evaluation and benchmarking.

Provides :class:`GitLabCIIntegration` which can trigger evaluation pipelines,
fetch pipeline results, and post reports as merge-request comments.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import httpx

from distllm.integrations.ci._common import (
    _DEFAULT_TIMEOUT,
    _retry,
    EvalResult,
)

logger = logging.getLogger("distllm")


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

    # -- context manager support --------------------------------------------

    def __enter__(self) -> GitLabCIIntegration:
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],
    ) -> None:
        self.close()

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
                f"## Evaluation Report -- `{report.model}`",
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
            lines = [
                "## Evaluation Report",
                "",
                "| Metric | Value |",
                "|--------|-------|",
            ]
            for k, v in report.items():
                lines.append(f"| {k} | {v} |")
            return "\n".join(lines)

        return str(report)

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._client.close()
