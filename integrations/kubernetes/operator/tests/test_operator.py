"""Tests for the DistLLM Kubernetes operator lifecycle.

These tests mock the kubernetes client module so no cluster is required.
"""

import asyncio
import logging
import unittest.mock as mock

import pytest

from distllm_operator import operator

LOGGER = logging.getLogger("test_operator")


class FakeApiException(Exception):
    """Stand-in for kubernetes.client.exceptions.ApiException."""


@pytest.fixture
def k8s():
    """Patch distllm_operator.operator.kubernetes with an in-memory mock."""
    mock_k8s = mock.MagicMock()
    mock_k8s.client.exceptions.ApiException = FakeApiException
    mock_k8s.config.load_incluster_config.return_value = None

    mock_apps = mock.MagicMock()
    mock_core = mock.MagicMock()
    mock_custom = mock.MagicMock()
    mock_k8s.client.AppsV1Api.return_value = mock_apps
    mock_k8s.client.CoreV1Api.return_value = mock_core
    mock_k8s.client.CustomObjectsApi.return_value = mock_custom
    mock_k8s.client.ApiClient.return_value.sanitize_for_serialization.return_value = (
        "2026-01-01T00:00:00"
    )

    with mock.patch.object(operator, "kubernetes", mock_k8s):
        ctx = mock.MagicMock()
        ctx.k8s = mock_k8s
        ctx.apps = mock_apps
        ctx.core = mock_core
        ctx.custom = mock_custom
        yield ctx


def _last_phase(custom_mock):
    calls = custom_mock.patch_namespaced_custom_object_status.call_args_list
    if not calls:
        return None
    return calls[-1].kwargs["body"]["status"]["phase"]


# ----------------------------------------------------------------------
# Create lifecycle
# ----------------------------------------------------------------------


def test_create_invokes_resources_and_status_deploying(k8s):
    operator.create_fn(
        spec={"replicas": 2, "model": "llama"},
        meta={"name": "c1", "namespace": "ns"},
        logger=LOGGER,
    )

    # coordinator + worker deployments, plus the service
    assert k8s.apps.create_namespaced_deployment.call_count == 2
    k8s.core.create_namespaced_service.assert_called_once()
    assert _last_phase(k8s.custom) == "Deploying"


def test_create_uses_config_defaults(k8s):
    # no replicas/model supplied -> fall back to OperatorConfig defaults
    operator.create_fn(spec={}, meta={"name": "c1"}, logger=LOGGER)

    calls = k8s.apps.create_namespaced_deployment.call_args_list
    worker_body = [
        c.args[1]
        for c in calls
        if c.args[1]["metadata"]["name"].startswith("distllm-worker-")
    ][0]

    assert worker_body["spec"]["replicas"] == 1
    env_value = worker_body["spec"]["template"]["spec"]["containers"][0]["env"][0][
        "value"
    ]
    assert env_value == "distributed-llm"


# ----------------------------------------------------------------------
# Update lifecycle
# ----------------------------------------------------------------------


def test_update_scales_workers(k8s):
    operator.update_fn(
        spec={"replicas": 3},
        meta={"name": "c1", "namespace": "ns"},
        diff=None,
        logger=LOGGER,
    )

    scale_call = k8s.apps.patch_namespaced_deployment_scale.call_args
    assert scale_call.kwargs["name"] == "distllm-worker-c1"
    assert scale_call.kwargs["body"] == {"spec": {"replicas": 3}}
    assert _last_phase(k8s.custom) == "Scaling"


def test_update_scale_failure_sets_error_status(k8s):
    k8s.apps.patch_namespaced_deployment_scale.side_effect = FakeApiException(
        "boom"
    )

    operator.update_fn(
        spec={"replicas": 3},
        meta={"name": "c1", "namespace": "ns"},
        diff=None,
        logger=LOGGER,
    )

    assert _last_phase(k8s.custom) == "Error"


# ----------------------------------------------------------------------
# Delete lifecycle
# ----------------------------------------------------------------------


def test_delete_removes_resources(k8s):
    operator.delete_fn(
        spec={}, meta={"name": "c1", "namespace": "ns"}, logger=LOGGER
    )

    deleted = [
        c.args[0] for c in k8s.apps.delete_namespaced_deployment.call_args_list
    ]
    assert "distllm-coordinator-c1" in deleted
    assert "distllm-worker-c1" in deleted
    k8s.core.delete_namespaced_service.assert_called_once_with(
        "distllm-c1", "ns"
    )


def test_delete_handles_api_exception(k8s):
    k8s.apps.delete_namespaced_deployment.side_effect = FakeApiException("gone")
    k8s.core.delete_namespaced_service.side_effect = FakeApiException("gone")

    # should swallow exceptions, not raise
    operator.delete_fn(
        spec={}, meta={"name": "c1", "namespace": "ns"}, logger=LOGGER
    )

    assert k8s.apps.delete_namespaced_deployment.call_count == 2


# ----------------------------------------------------------------------
# Resume lifecycle
# ----------------------------------------------------------------------


def test_resume_triggers_create(k8s):
    operator.resume_fn(
        spec={"replicas": 1, "model": "llama"},
        meta={"name": "c1", "namespace": "ns"},
        logger=LOGGER,
    )

    assert k8s.apps.create_namespaced_deployment.call_count == 2
    k8s.core.create_namespaced_service.assert_called_once()


# ----------------------------------------------------------------------
# Health checking / pod status
# ----------------------------------------------------------------------


def _fake_deployment(ready):
    dep = mock.MagicMock()
    dep.status.ready_replicas = ready
    return dep


def test_health_running(k8s):
    k8s.apps.read_namespaced_deployment.side_effect = [
        _fake_deployment(1),  # coordinator
        _fake_deployment(2),  # workers
    ]

    asyncio.run(
        operator._reconcile_health(
            "c1", "ns", {"replicas": 2}, LOGGER
        )
    )

    assert _last_phase(k8s.custom) == "Running"


def test_health_pending(k8s):
    k8s.apps.read_namespaced_deployment.side_effect = [
        _fake_deployment(0),  # coordinator not ready
        _fake_deployment(0),
    ]

    asyncio.run(
        operator._reconcile_health(
            "c1", "ns", {"replicas": 2}, LOGGER
        )
    )

    assert _last_phase(k8s.custom) == "Pending"


def test_watch_triggers_reconcile(k8s):
    k8s.apps.read_namespaced_deployment.side_effect = [
        _fake_deployment(1),
        _fake_deployment(2),
    ]

    captured = []

    def fake_create_task(coro):
        captured.append(coro)
        return coro

    with mock.patch.object(asyncio, "create_task", side_effect=fake_create_task):
        asyncio.run(
            operator.watch_fn(
                spec={"replicas": 2},
                meta={"name": "c1", "namespace": "ns"},
                event={"type": "ADDED"},
                logger=LOGGER,
            )
        )

    assert captured, "expected a reconcile task to be scheduled"
    asyncio.run(captured[0])
    assert _last_phase(k8s.custom) == "Running"


def test_watch_ignores_non_administrative_events(k8s):
    with mock.patch.object(asyncio, "create_task") as spy:
        asyncio.run(
            operator.watch_fn(
                spec={"replicas": 2},
                meta={"name": "c1", "namespace": "ns"},
                event={"type": "DELETED"},
                logger=LOGGER,
            )
        )
    spy.assert_not_called()
