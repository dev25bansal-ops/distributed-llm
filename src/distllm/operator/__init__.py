"""Kubernetes operator package (kopf-based)."""

from distllm.operator.hpa import CustomHPA, _validate_prometheus_url

__all__ = ["CustomHPA", "_validate_prometheus_url"]
