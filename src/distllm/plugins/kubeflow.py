"""Kubeflow Pipelines component — submit batch inference from KFP.

Usage in a pipeline::

    from distllm.plugins.kubeflow import kubeflow_batch_inference_op

    @dsl.pipeline
    def my_pipeline():
        infer_task = kubeflow_batch_inference_op(
            model="meta-llama/Llama-3.1-70B",
            input_data="gs://bucket/inputs/",
            output_path="gs://bucket/outputs/",
        )
"""

from __future__ import annotations

from typing import Any

from loguru import logger


def kubeflow_batch_inference_op(
    model: str,
    input_data: str,
    output_path: str,
    max_tokens: int = 2048,
    tenant_id: str = "default",
    distllm_endpoint: str = "http://distllm-coordinator:8000",
    **kwargs: Any,
) -> dict[str, Any]:
    """Kubeflow Pipelines component for batch inference.

    This is a light wrapper that generates a container op spec.
    In real usage this would be decorated with ``@kfp.dsl.component``.

    Args:
        model: Model name/path.
        input_data: GCS/S3 path to input data.
        output_path: GCS/S3 path for outputs.
        max_tokens: Max tokens per item.
        tenant_id: Tenant identifier.
        distllm_endpoint: DistLLM coordinator URL.
        **kwargs: Additional params.

    Returns:
        Dict with job metadata.
    """
    import requests
    import time

    payload = {
        "model": model,
        "input_data": input_data,
        "output_path": output_path,
        "max_tokens": max_tokens,
        "tenant_id": tenant_id,
        "source": "kubeflow",
        **{k: v for k, v in kwargs.items()
           if k in ("batch_size", "priority", "tags")},
    }

    # Submit
    resp = requests.post(f"{distllm_endpoint}/api/v1/jobs", json=payload, timeout=30)
    resp.raise_for_status()
    job_id = resp.json().get("job_id", "unknown")
    logger.info(f"Kubeflow: submitted job {job_id} ({model})")

    # Poll
    status = "running"
    while status in ("running", "pending", "queued"):
        time.sleep(5)
        s_resp = requests.get(
            f"{distllm_endpoint}/api/v1/jobs/{job_id}", timeout=30
        )
        s_resp.raise_for_status()
        status = s_resp.json().get("status", "unknown")

    return {
        "job_id": job_id,
        "status": status,
        "output_location": output_path,
        "model": model,
    }
