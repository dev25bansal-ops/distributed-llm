"""FaaS-7B: Fully Serverless Inference Workers on AWS Lambda / Cloudflare Workers."""

from __future__ import annotations

import base64
import json
import os
import time
from typing import Any

from loguru import logger

try:
    import boto3
    HAS_BOTO3 = True
except ImportError:
    HAS_BOTO3 = False


class LambdaInferenceWorker:
    """AWS Lambda-based serverless inference worker."""

    def __init__(self, function_name: str = "distllm-inference", region: str = ""):
        self._function_name = function_name
        self._region = region or os.environ.get("AWS_REGION", "us-east-1")
        self._client = None
        self._cold_start = True
        self._stats = {"invocations": 0, "cold_starts": 0, "total_time_ms": 0}

    def _get_client(self):
        if self._client is None and HAS_BOTO3:
            self._client = boto3.client("lambda", region_name=self._region)
        return self._client

    def deploy(self, model_name: str, memory_mb: int = 10240, timeout_s: int = 300) -> bool:
        """Deploy model as Lambda function using container image."""
        logger.info(f"Deploying {model_name} as Lambda ({memory_mb}MB, {timeout_s}s timeout)")
        if not HAS_BOTO3:
            logger.warning("boto3 not available — skipping deploy")
            return False
        try:
            client = self._get_client()
            client.create_function(
                FunctionName=self._function_name,
                PackageType="Image",
                Code={"ImageUri": f"distllm/inference:{model_name.lower().replace('/', '-')}"},
                Role=os.environ.get("AWS_LAMBDA_ROLE_ARN", ""),
                MemorySize=memory_mb,
                Timeout=timeout_s,
            )
            return True
        except client.exceptions.ResourceConflictException:
            logger.info(f"Function {self._function_name} already exists")
            return True
        except Exception as e:
            logger.error(f"Lambda deploy failed: {e}")
            return False

    def invoke(self, payload: dict) -> dict:
        """Invoke inference Lambda synchronously."""
        client = self._get_client()
        if client is None:
            return {"error": "AWS Lambda not available"}

        start = time.time()
        try:
            resp = client.invoke(
                FunctionName=self._function_name,
                InvocationType="RequestResponse",
                Payload=json.dumps(payload),
            )
            result = json.loads(resp["Payload"].read())
            elapsed = (time.time() - start) * 1000
            self._stats["invocations"] += 1
            if self._cold_start:
                self._stats["cold_starts"] += 1
                self._cold_start = False
            self._stats["total_time_ms"] += elapsed
            return result
        except Exception as e:
            return {"error": str(e)}

    def invoke_async(self, payload: dict) -> str:
        """Invoke inference Lambda asynchronously."""
        client = self._get_client()
        if client is None:
            return ""
        try:
            resp = client.invoke(
                FunctionName=self._function_name,
                InvocationType="Event",
                Payload=json.dumps(payload),
            )
            return resp.get("ResponseMetadata", {}).get("RequestId", "")
        except Exception as e:
            logger.error(f"Async invoke failed: {e}")
            return ""

    def get_concurrency(self) -> int:
        """Get reserved concurrency."""
        client = self._get_client()
        if client is None:
            return 0
        try:
            resp = client.get_function_concurrency(FunctionName=self._function_name)
            return resp.get("ReservedConcurrentExecutions", 0)
        except Exception:
            return 0

    def set_concurrency(self, concurrency: int) -> bool:
        """Set reserved concurrency for the function."""
        client = self._get_client()
        if client is None:
            return False
        try:
            client.put_function_concurrency(
                FunctionName=self._function_name,
                ReservedConcurrentExecutions=concurrency,
            )
            return True
        except Exception:
            return False

    @property
    def stats(self) -> dict:
        s = dict(self._stats)
        if s["invocations"]:
            s["avg_latency_ms"] = round(s["total_time_ms"] / s["invocations"], 1)
        return s


class CloudflareWorker:
    """Cloudflare Workers-based serverless inference."""

    def __init__(self, worker_url: str = "", api_token: str = ""):
        self._worker_url = worker_url or os.environ.get("CF_WORKER_URL", "")
        self._api_token = api_token or os.environ.get("CF_API_TOKEN", "")

    def deploy_worker(self, script: str, name: str = "distllm-inference") -> bool:
        """Deploy a Cloudflare Worker script."""
        try:
            import httpx
            resp = httpx.post(
                f"https://api.cloudflare.com/client/v4/accounts/me/workers/scripts/{name}",
                headers={"Authorization": f"Bearer {self._api_token}"},
                data={"metadata": json.dumps({"main_module": "index.mjs"})},
                files={"index.mjs": ("index.mjs", script, "application/javascript+module")},
            )
            return resp.is_success
        except Exception as e:
            logger.error(f"Cloudflare deploy failed: {e}")
            return False

    def invoke(self, prompt: str, max_tokens: int = 256) -> dict:
        """Send inference request to Cloudflare Worker."""
        if not self._worker_url:
            return {"error": "CF_WORKER_URL not configured"}
        try:
            import httpx
            resp = httpx.post(
                self._worker_url,
                json={"prompt": prompt, "max_tokens": max_tokens},
                timeout=30.0,
            )
            return resp.json() if resp.is_success else {"error": resp.text}
        except Exception as e:
            return {"error": str(e)}


class FaaS7B:
    """Orchestrates serverless inference across Lambda and Cloudflare Workers."""

    def __init__(self):
        self._lambda_worker = LambdaInferenceWorker()
        self._cf_worker = CloudflareWorker()
        self._stats = {"lambda_calls": 0, "cf_calls": 0, "errors": 0}

    def infer(self, prompt: str, provider: str = "auto", max_tokens: int = 256) -> dict:
        """Run inference via serverless provider."""
        if provider == "lambda" or (provider == "auto" and HAS_BOTO3):
            result = self._lambda_worker.invoke({"prompt": prompt, "max_tokens": max_tokens})
            self._stats["lambda_calls"] += 1
            if "error" in result:
                self._stats["errors"] += 1
            return result
        else:
            result = self._cf_worker.invoke(prompt, max_tokens)
            self._stats["cf_calls"] += 1
            if "error" in result:
                self._stats["errors"] += 1
            return result

    def deploy(self, model_name: str, provider: str = "lambda") -> bool:
        """Deploy model to serverless provider."""
        if provider == "lambda":
            return self._lambda_worker.deploy(model_name)
        return False

    @property
    def stats(self) -> dict:
        return dict(self._stats)
