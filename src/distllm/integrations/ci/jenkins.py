"""Jenkins CI integration for model evaluation and benchmarking.

Provides :class:`JenkinsIntegration` which can trigger parameterised builds,
query build status, and publish build artifacts.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Optional

import httpx

from distllm.integrations.ci._common import (
    _DEFAULT_TIMEOUT,
    _retry,
    BuildInfo,
)

logger = logging.getLogger("distllm")


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

    # -- context manager support --------------------------------------------

    def __enter__(self) -> JenkinsIntegration:
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],
    ) -> None:
        self.close()

    # -- helpers ------------------------------------------------------------

    def _encode_job_path(self, job: str) -> str:
        """Encode a dotted or slash-separated Jenkins job path.

        ``"folder / pipeline"`` becomes ``"folder/job/pipeline"``.
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
        # eventual build number -- or poll the queue briefly.
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
