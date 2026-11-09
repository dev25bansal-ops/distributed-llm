"""Tests: ``distllm pool`` — renting GPUs from spot marketplaces.

These commands were a stub that printed "week-5 feature". The backend they
need (``distllm.cloud.spot_orchestrator``) already existed and speaks to
RunPod, Vast and Salad over their real HTTP APIs, so the commands are wired to
it now. These tests pin the behaviour that matters: no credentials means a
clear message and a non-zero exit, never invented data.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from distllm.cli.main import app
from distllm.cli.pool import SUPPORTED_PROVIDERS, _ENV_KEYS, _missing_keys

runner = CliRunner()


@pytest.fixture()
def no_creds(monkeypatch):
    for env in _ENV_KEYS.values():
        monkeypatch.delenv(env, raising=False)
    return monkeypatch


class TestCredentialHandling:
    def test_supported_providers_are_real_marketplaces(self):
        assert set(SUPPORTED_PROVIDERS) == {"runpod", "vast", "salad"}

    def test_every_provider_has_an_env_key(self):
        for provider in SUPPORTED_PROVIDERS:
            assert provider in _ENV_KEYS

    def test_missing_keys_reports_all_when_unset(self, no_creds):
        assert _missing_keys(list(SUPPORTED_PROVIDERS)) == list(SUPPORTED_PROVIDERS)

    def test_missing_keys_excludes_configured_provider(self, no_creds):
        no_creds.setenv("RUNPOD_API_KEY", "test-key")
        assert "runpod" not in _missing_keys(list(SUPPORTED_PROVIDERS))
        assert "vast" in _missing_keys(list(SUPPORTED_PROVIDERS))


class TestCommandsWithoutCredentials:
    def test_list_explains_what_is_missing_and_exits_nonzero(self, no_creds):
        result = runner.invoke(app, ["pool", "list"])
        assert result.exit_code != 0
        assert "RUNPOD_API_KEY" in result.output

    def test_add_refuses_without_credentials(self, no_creds):
        result = runner.invoke(app, ["pool", "add", "--gpu", "4090"])
        assert result.exit_code != 0
        assert "API_KEY" in result.output

    def test_remove_requires_provider(self, no_creds):
        """Instance ids are only unique within a marketplace."""
        result = runner.invoke(app, ["pool", "remove", "abc123", "--yes"])
        assert result.exit_code != 0
        assert "provider" in result.output.lower()


class TestCommandsWithCredentials:
    def test_list_renders_offers(self, no_creds):
        no_creds.setenv("RUNPOD_API_KEY", "test-key")
        offer = MagicMock()
        offer.gpu_type = "RTX 4090"
        offer.gpu_count = 1
        offer.price_per_hour = 0.44
        offer.region = "us-east"
        offer.provider = "runpod"
        offer.instance_id = "pod-123"

        with patch(
            "distllm.cloud.spot_orchestrator.SpotOrchestrator.find_cheapest",
            return_value=[offer],
        ):
            result = runner.invoke(app, ["pool", "list", "--gpu", "4090"])
        assert result.exit_code == 0
        assert "RTX 4090" in result.output
        assert "0.440" in result.output

    def test_list_reports_no_matches_cleanly(self, no_creds):
        no_creds.setenv("RUNPOD_API_KEY", "test-key")
        with patch(
            "distllm.cloud.spot_orchestrator.SpotOrchestrator.find_cheapest",
            return_value=[],
        ):
            result = runner.invoke(app, ["pool", "list", "--gpu", "nonexistent"])
        assert result.exit_code == 0
        assert "No offers" in result.output

    def test_add_dry_run_does_not_rent(self, no_creds):
        no_creds.setenv("RUNPOD_API_KEY", "test-key")
        offer = MagicMock()
        offer.gpu_type = "RTX 4090"
        offer.gpu_count = 1
        offer.price_per_hour = 0.44
        offer.provider = "runpod"

        with patch(
            "distllm.cloud.spot_orchestrator.SpotOrchestrator.find_cheapest",
            return_value=[offer],
        ), patch(
            "distllm.cloud.spot_orchestrator.SpotOrchestrator.launch_cluster"
        ) as launch:
            result = runner.invoke(
                app, ["pool", "add", "--gpu", "4090", "--max-price", "1.0", "--dry-run"]
            )
        assert result.exit_code == 0
        assert "dry-run" in result.output.lower()
        launch.assert_not_called()