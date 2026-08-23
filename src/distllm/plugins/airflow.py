"""Airflow plugin — submit batch inference jobs from Airflow DAGs.

Usage in a DAG::

    from distllm.plugins.airflow import DistLLMBatchOperator

    infer = DistLLMBatchOperator(
        task_id="run_inference",
        model="meta-llama/Llama-3.1-70B",
        input_data="s3://bucket/inputs/*.json",
        output_path="s3://bucket/outputs/",
        max_tokens=4096,
        tenant_id="my-team",
        distllm_endpoint="http://coordinator:8000",
    )
"""

from __future__ import annotations

from typing import Any

from loguru import logger


class DistLLMBatchOperator:
    """Submit a batch inference job to DistLLM from an Airflow DAG.

    This is a minimal operator that can be used directly or wrapped
    in an Airflow ``PythonOperator`` for full integration.

    Args:
        task_id: Airflow task ID.
        model: Model name/path to run.
        input_data: Path/URI/gloss for input data.
        output_path: Destination for output data.
        max_tokens: Max tokens to generate per item.
        tenant_id: Tenant identifier for SLO enforcement.
        distllm_endpoint: DistLLM coordinator HTTP endpoint.
        **kwargs: Additional job parameters.
    """

    def __init__(
        self,
        task_id: str,
        model: str,
        input_data: str,
        output_path: str,
        max_tokens: int = 2048,
        tenant_id: str = "default",
        distllm_endpoint: str = "http://localhost:8000",
        **kwargs: Any,
    ):
        self.task_id = task_id
        self.model = model
        self.input_data = input_data
        self.output_path = output_path
        self.max_tokens = max_tokens
        self.tenant_id = tenant_id
        self.endpoint = distllm_endpoint
        self._extra = kwargs

    def execute(self, context: dict[str, Any] | None = None) -> dict[str, Any]:
        """Execute the batch inference job.

        Sends a job submission request to the DistLLM coordinator and
        polls until completion.

        Returns:
            Dict with ``job_id``, ``status``, ``output_location``, ``metrics``.
        """
        import requests
        import time

        payload = {
            "model": self.model,
            "input_data": self.input_data,
            "output_path": self.output_path,
            "max_tokens": self.max_tokens,
            "tenant_id": self.tenant_id,
            **{k: v for k, v in self._extra.items()
               if k in ("batch_size", "priority", "tags")},
        }

        # Submit job
        resp = requests.post(f"{self.endpoint}/api/v1/jobs", json=payload, timeout=30)
        resp.raise_for_status()
        job_id = resp.json().get("job_id", "unknown")
        logger.info(f"Airflow: submitted job {job_id} ({self.model}, {self.input_data})")

        # Poll for completion
        status = "running"
        while status in ("running", "pending", "queued"):
            time.sleep(5)
            status_resp = requests.get(
                f"{self.endpoint}/api/v1/jobs/{job_id}", timeout=30
            )
            status_resp.raise_for_status()
            status = status_resp.json().get("status", "unknown")
            logger.debug(f"Airflow: job {job_id} status={status}")

        return {
            "job_id": job_id,
            "status": status,
            "output_location": self.output_path,
            "metrics": {"model": self.model, "max_tokens": self.max_tokens},
        }

    def __call__(self, **context: Any) -> dict[str, Any]:
        """Make callable for Airflow ``PythonOperator``."""
        return self.execute(dict(**context))
