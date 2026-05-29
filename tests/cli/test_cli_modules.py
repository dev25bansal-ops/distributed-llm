"""Tests for all CLI modules — adapters, autopsy, backup, cert, compress,
cost_avoid, deploy, logs, models, notify, profile, prompts, quota, run,
setup, status, tutorial, verify, webhook.
"""
from unittest.mock import MagicMock, patch, PropertyMock
import pytest
from typer.testing import CliRunner

runner = CliRunner()


# ===========================================================================
# adapters.py
# ===========================================================================


class TestCliAdapters:
    """Tests for distllm.cli.adapters — _list_adapters, _load_adapter,
    _set_adapter, _unload_adapter.
    """

    def test_list_adapters_happy_path(self):
        from distllm.cli.adapters import _list_adapters

        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "adapters": [
                {"id": "lora-1", "source": "hf://user/lora",
                 "status": "loaded", "active": True},
                {"id": "lora-2", "source": "local://path",
                 "status": "loaded", "active": False},
            ]
        }
        with patch("distllm.cli.adapters.httpx.Client") as mc:
            inst = MagicMock()
            inst.get.return_value = mock_resp
            mc.return_value.__enter__ = MagicMock(return_value=inst)
            mc.return_value.__exit__ = MagicMock(return_value=False)
            _list_adapters("localhost", 8000)

        inst.get.assert_called_once_with("/v1/adapters")

    def test_list_adapters_empty(self):
        from distllm.cli.adapters import _list_adapters

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"adapters": []}
        with patch("distllm.cli.adapters.httpx.Client") as mc:
            inst = MagicMock()
            inst.get.return_value = mock_resp
            mc.return_value.__enter__ = MagicMock(return_value=inst)
            mc.return_value.__exit__ = MagicMock(return_value=False)
            _list_adapters("localhost", 8000)

    def test_list_adapters_connect_error(self):
        from distllm.cli.adapters import _list_adapters

        with patch("distllm.cli.adapters.httpx.Client") as mc:
            inst = MagicMock()
            from httpx import ConnectError
            inst.get.side_effect = ConnectError("refused")
            mc.return_value.__enter__ = MagicMock(return_value=inst)
            mc.return_value.__exit__ = MagicMock(return_value=False)
            _list_adapters("badhost", 9999)

    def test_list_adapters_http_error(self):
        from distllm.cli.adapters import _list_adapters

        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.text = "Internal error"
        from httpx import HTTPStatusError
        mock_resp.raise_for_status.side_effect = HTTPStatusError(
            "error", request=MagicMock(), response=mock_resp
        )
        with patch("distllm.cli.adapters.httpx.Client") as mc:
            inst = MagicMock()
            inst.get.return_value = mock_resp
            mc.return_value.__enter__ = MagicMock(return_value=inst)
            mc.return_value.__exit__ = MagicMock(return_value=False)
            _list_adapters("localhost", 8000)

    def test_load_adapter_happy_path(self):
        from distllm.cli.adapters import _load_adapter

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"status": "ok"}
        with patch("distllm.cli.adapters.httpx.Client") as mc:
            inst = MagicMock()
            inst.post.return_value = mock_resp
            mc.return_value.__enter__ = MagicMock(return_value=inst)
            mc.return_value.__exit__ = MagicMock(return_value=False)
            _load_adapter("localhost", 8000, "my-lora", "hf://user/lora")

        inst.post.assert_called_once()
        assert "/v1/adapters/load" in str(inst.post.call_args[0])

    def test_set_adapter_happy_path(self):
        from distllm.cli.adapters import _set_adapter

        mock_resp = MagicMock()
        with patch("distllm.cli.adapters.httpx.Client") as mc:
            inst = MagicMock()
            inst.post.return_value = mock_resp
            mc.return_value.__enter__ = MagicMock(return_value=inst)
            mc.return_value.__exit__ = MagicMock(return_value=False)
            _set_adapter("localhost", 8000, "my-lora")

    def test_unload_adapter_happy_path(self):
        from distllm.cli.adapters import _unload_adapter

        mock_resp = MagicMock()
        with patch("distllm.cli.adapters.httpx.Client") as mc:
            inst = MagicMock()
            inst.post.return_value = mock_resp
            mc.return_value.__enter__ = MagicMock(return_value=inst)
            mc.return_value.__exit__ = MagicMock(return_value=False)
            _unload_adapter("localhost", 8000, "my-lora")

    def test_load_adapter_http_error(self):
        from distllm.cli.adapters import _load_adapter

        mock_resp = MagicMock()
        mock_resp.status_code = 400
        mock_resp.text = "Bad adapter"
        from httpx import HTTPStatusError
        mock_resp.raise_for_status.side_effect = HTTPStatusError(
            "error", request=MagicMock(), response=mock_resp
        )
        with patch("distllm.cli.adapters.httpx.Client") as mc:
            inst = MagicMock()
            inst.post.return_value = mock_resp
            mc.return_value.__enter__ = MagicMock(return_value=inst)
            mc.return_value.__exit__ = MagicMock(return_value=False)
            _load_adapter("localhost", 8000, "bad-lora", "hf://bad")


# ===========================================================================
# autopsy.py
# ===========================================================================


class TestCliAutopsy:
    """Tests for distllm.cli.autopsy — collect_autopsy."""

    @patch("distllm.cli.autopsy.httpx.get")
    @patch("distllm.cli.autopsy.os.path.exists", return_value=False)
    @patch("distllm.cli.autopsy.zipfile.ZipFile")
    def test_collect_autopsy_happy_path(self, mock_zip, mock_exists, mock_httpx_get):
        from distllm.cli.autopsy import collect_autopsy

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"status": "healthy", "nodes": 2}
        mock_resp.text = "test_metric 42\n"
        mock_httpx_get.return_value = mock_resp

        # Clear config file lookup so no configs found
        result = collect_autopsy("localhost", 8000, 60, "/tmp/test-report.zip")
        assert result == "/tmp/test-report.zip"
        assert mock_httpx_get.call_count >= 4

    @patch("distllm.cli.autopsy.httpx.get")
    @patch("distllm.cli.autopsy.os.path.exists", return_value=False)
    @patch("distllm.cli.autopsy.zipfile.ZipFile")
    def test_collect_autopsy_connection_errors(self, mock_zip, mock_exists, mock_httpx_get):
        from distllm.cli.autopsy import collect_autopsy

        mock_httpx_get.side_effect = ConnectionError("refused")
        result = collect_autopsy("badhost", 8000, 60, "/tmp/test-report-err.zip")
        assert result.endswith(".zip")

    @patch("distllm.cli.autopsy.httpx.get")
    @patch("distllm.cli.autopsy.os.path.exists")
    @patch("distllm.cli.autopsy.zipfile.ZipFile")
    def test_collect_autopsy_with_configs(self, mock_zip, mock_exists, mock_httpx_get):
        from distllm.cli.autopsy import collect_autopsy

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"status": "healthy"}
        mock_resp.text = "cpu 10\n"
        mock_httpx_get.return_value = mock_resp
        mock_exists.return_value = True

        mock_file = MagicMock()
        mock_file.read.return_value = "model:\n  name: test\n"
        mock_open = MagicMock(return_value=mock_file)

        with patch("builtins.open", mock_open):
            result = collect_autopsy("localhost", 8000, output_path="/tmp/test-cfg.zip")
            assert result == "/tmp/test-cfg.zip"

    @patch("distllm.cli.autopsy.httpx.get")
    @patch("distllm.cli.autopsy.os.path.exists", return_value=False)
    @patch("distllm.cli.autopsy.zipfile.ZipFile")
    def test_collect_autopsy_auto_path(self, mock_zip, mock_exists, mock_httpx_get):
        from distllm.cli.autopsy import collect_autopsy

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"status": "healthy"}
        mock_resp.text = ""
        mock_httpx_get.return_value = mock_resp

        result = collect_autopsy("localhost", 8000)
        assert result.startswith("distllm-autopsy-")
        assert result.endswith(".zip")


# ===========================================================================
# backup.py
# ===========================================================================


class TestCliBackup:
    """Tests for distllm.cli.backup — backup_app Typer commands."""

    @patch("distllm.cli.backup.BackupManager")
    def test_backup_create(self, mock_mgr_cls):
        from distllm.cli.backup import backup_app

        mock_mgr = MagicMock()
        mock_manifest = MagicMock()
        mock_manifest.backup_id = "bkp-001"
        mock_manifest.size_bytes = 1024
        mock_manifest.entries = 5
        mock_manifest.backup_type = "full"
        mock_mgr.create_full.return_value = mock_manifest
        mock_mgr_cls.return_value = mock_mgr

        with patch("builtins.open", MagicMock()):
            result = runner.invoke(
                backup_app, ["create", "--dir", "/tmp/backups"],
                catch_exceptions=False,
            )
        assert result.exit_code == 0

    @patch("distllm.cli.backup.BackupManager")
    def test_backup_list_empty(self, mock_mgr_cls):
        from distllm.cli.backup import backup_app

        mock_mgr = MagicMock()
        mock_mgr.list_backups.return_value = []
        mock_mgr_cls.return_value = mock_mgr

        result = runner.invoke(backup_app, ["list", "--dir", "/tmp/backups"],
                               catch_exceptions=False)
        assert result.exit_code == 0

    @patch("distllm.cli.backup.BackupManager")
    def test_backup_list_with_data(self, mock_mgr_cls):
        from distllm.cli.backup import backup_app

        mock_backup = MagicMock()
        mock_backup.backup_id = "bkp-001"
        mock_backup.backup_type = "full"
        mock_backup.created_at = 1700000000
        mock_backup.size_bytes = 2048
        mock_backup.entries = 10
        mock_backup.cluster_name = "default"

        mock_mgr = MagicMock()
        mock_mgr.list_backups.return_value = [mock_backup]
        mock_mgr_cls.return_value = mock_mgr

        result = runner.invoke(backup_app, ["list"], catch_exceptions=False)
        assert result.exit_code == 0

    @patch("distllm.cli.backup.BackupManager")
    def test_backup_restore_found(self, mock_mgr_cls):
        from distllm.cli.backup import backup_app

        mock_mgr = MagicMock()
        mock_mgr.restore.return_value = {"model": "test", "config": {}}
        mock_mgr_cls.return_value = mock_mgr

        result = runner.invoke(backup_app, ["restore", "bkp-001"],
                               catch_exceptions=False)
        assert result.exit_code == 0

    @patch("distllm.cli.backup.BackupManager")
    def test_backup_restore_not_found(self, mock_mgr_cls):
        from distllm.cli.backup import backup_app

        mock_mgr = MagicMock()
        mock_mgr.restore.return_value = None
        mock_mgr_cls.return_value = mock_mgr

        result = runner.invoke(backup_app, ["restore", "bkp-missing"],
                               catch_exceptions=False)
        assert result.exit_code == 1

    @patch("distllm.cli.backup.BackupManager")
    def test_backup_delete_found(self, mock_mgr_cls):
        from distllm.cli.backup import backup_app

        mock_mgr = MagicMock()
        mock_mgr.delete_backup.return_value = True
        mock_mgr_cls.return_value = mock_mgr

        result = runner.invoke(backup_app, ["delete", "bkp-001"],
                               catch_exceptions=False)
        assert result.exit_code == 0

    @patch("distllm.cli.backup.BackupManager")
    def test_backup_delete_not_found(self, mock_mgr_cls):
        from distllm.cli.backup import backup_app

        mock_mgr = MagicMock()
        mock_mgr.delete_backup.return_value = False
        mock_mgr_cls.return_value = mock_mgr

        result = runner.invoke(backup_app, ["delete", "bkp-missing"],
                               catch_exceptions=False)
        assert result.exit_code == 0


# ===========================================================================
# cert.py
# ===========================================================================


class TestCliCert:
    """Tests for distllm.cli.cert — cert_app Typer commands."""

    @patch("distllm.cli.cert.CertificateManager")
    def test_cert_create(self, mock_mgr_cls):
        from distllm.cli.cert import cert_app

        mock_info = MagicMock()
        mock_info.cert_path = "/tmp/certs/test.pem"
        mock_info.key_path = "/tmp/certs/test-key.pem"
        mock_info.not_after = 1800000000
        mock_info.subject_alt_names = ["test.local"]
        mock_mgr = MagicMock()
        mock_mgr.ensure_certificate.return_value = mock_info
        mock_mgr_cls.return_value = mock_mgr

        result = runner.invoke(
            cert_app, ["create", "test.local"],
            catch_exceptions=False,
        )
        assert result.exit_code == 0

    @patch("distllm.cli.cert.CertificateManager")
    def test_cert_info_found(self, mock_mgr_cls):
        from distllm.cli.cert import cert_app

        mock_info = MagicMock()
        mock_info.common_name = "test.local"
        mock_info.subject_alt_names = ["test.local"]
        mock_info.issuer = "self"
        mock_info.not_before = 1700000000
        mock_info.not_after = 1800000000
        mock_info.fingerprint_sha256 = "abcd1234"
        mock_info.is_self_signed = True
        mock_info.cert_path = "/tmp/certs/test.pem"
        mock_info.key_path = "/tmp/certs/test-key.pem"
        mock_mgr = MagicMock()
        mock_mgr.get_certificate_info.return_value = mock_info
        mock_mgr_cls.return_value = mock_mgr

        result = runner.invoke(cert_app, ["info", "test.local"],
                               catch_exceptions=False)
        assert result.exit_code == 0

    @patch("distllm.cli.cert.CertificateManager")
    def test_cert_info_not_found(self, mock_mgr_cls):
        from distllm.cli.cert import cert_app

        mock_mgr = MagicMock()
        mock_mgr.get_certificate_info.return_value = None
        mock_mgr_cls.return_value = mock_mgr

        result = runner.invoke(cert_app, ["info", "missing.local"],
                               catch_exceptions=False)
        assert result.exit_code == 1

    @patch("distllm.cli.cert.CertificateManager")
    def test_cert_renew_with_renewals(self, mock_mgr_cls):
        from distllm.cli.cert import cert_app

        mock_info = MagicMock()
        mock_info.common_name = "test.local"
        mock_mgr = MagicMock()
        mock_mgr.renew_all.return_value = [mock_info]
        mock_mgr_cls.return_value = mock_mgr

        result = runner.invoke(cert_app, ["renew"], catch_exceptions=False)
        assert result.exit_code == 0

    @patch("distllm.cli.cert.CertificateManager")
    def test_cert_renew_no_renewals(self, mock_mgr_cls):
        from distllm.cli.cert import cert_app

        mock_mgr = MagicMock()
        mock_mgr.renew_all.return_value = []
        mock_mgr_cls.return_value = mock_mgr

        result = runner.invoke(cert_app, ["renew"], catch_exceptions=False)
        assert result.exit_code == 0

    @patch("distllm.cli.cert.CertificateManager")
    def test_cert_revoke(self, mock_mgr_cls):
        from distllm.cli.cert import cert_app

        mock_mgr = MagicMock()
        mock_mgr_cls.return_value = mock_mgr

        result = runner.invoke(cert_app, ["revoke", "test.local"],
                               catch_exceptions=False)
        assert result.exit_code == 0


# ===========================================================================
# compress.py
# ===========================================================================


class TestCliCompress:
    """Tests for distllm.cli.compress — run_compress."""

    @patch("distllm.cli.compress.AutoModelForCausalLM")
    @patch("distllm.cli.compress.AutoTokenizer")
    def test_run_compress_happy_path(self, mock_tok_cls, mock_model_cls):
        from distllm.cli.compress import run_compress

        mock_model = MagicMock()
        mock_model_cls.from_pretrained.return_value = mock_model
        mock_tok = MagicMock()
        mock_tok.pad_token = None
        mock_tok.eos_token = "<eos>"
        mock_tok_cls.from_pretrained.return_value = mock_tok

        console = MagicMock()
        with patch("pathlib.Path.mkdir"):
            run_compress("test-model", "int8", "/tmp/out", None,
                         0.0, 10, "awq", False, console)

        mock_model_cls.from_pretrained.assert_called_once()
        mock_tok_cls.from_pretrained.assert_called_once()
        mock_model.save_pretrained.assert_called_once()
        mock_tok.save_pretrained.assert_called_once()

    @patch("distllm.cli.compress.AutoModelForCausalLM")
    @patch("distllm.cli.compress.AutoTokenizer")
    def test_run_compress_fp32(self, mock_tok_cls, mock_model_cls):
        from distllm.cli.compress import run_compress

        mock_model = MagicMock()
        mock_model_cls.from_pretrained.return_value = mock_model
        mock_tok = MagicMock()
        mock_tok.pad_token = None
        mock_tok.eos_token = "<eos>"
        mock_tok_cls.from_pretrained.return_value = mock_tok

        console = MagicMock()
        with patch("pathlib.Path.mkdir"):
            run_compress("test-model", "fp32", "/tmp/out", "tok-name",
                         0.0, 5, "none", True, console)

    def test_run_compress_invalid_target(self):
        from distllm.cli.compress import run_compress

        console = MagicMock()
        with pytest.raises(SystemExit):
            run_compress("test-model", "invalid", "/tmp/out", None,
                         0.0, 5, "none", False, console)


# ===========================================================================
# cost_avoid.py
# ===========================================================================


class TestCliCostAvoid:
    """Tests for distllm.cli.cost_avoid — calculate_cost_avoidance and helpers."""

    def test_estimate_model_size_various(self):
        from distllm.cli.cost_avoid import _estimate_model_size

        assert _estimate_model_size("meta-llama/Llama-3.1-70B") == 70
        assert _estimate_model_size("meta-llama/Llama-3.2-8B") == 8
        assert _estimate_model_size("mistralai/Mistral-7B-v0.1") == 7
        assert _estimate_model_size("microsoft/Phi-3-mini-4k-instruct") == 7  # default
        assert _estimate_model_size("TinyLlama/TinyLlama-1B") == 1

    def test_estimate_vram_gb(self):
        from distllm.cli.cost_avoid import _estimate_vram_gb

        assert _estimate_vram_gb(7, "fp16") == 14
        assert _estimate_vram_gb(7, "int8") == 7
        assert _estimate_vram_gb(7, "int4") == 3  # 14 // 4 = 3
        assert _estimate_vram_gb(70, "fp16") == 140

    def test_calculate_cost_avoidance_basic(self):
        from distllm.cli.cost_avoid import calculate_cost_avoidance

        result = calculate_cost_avoidance(
            model_name="meta-llama/Llama-3.1-70B",
            requests_per_day=50000,
            gpu_type="RTX 4090",
            cloud_api="llama-3.1-70b-deepinfra",
        )
        assert result["model_size_b"] == 70
        assert result["gpus_needed"] >= 1
        assert result["monthly_savings"] >= 0
        assert "monthly_api_cost" in result
        assert "monthly_self_hosted_cost" in result
        assert "monthly_savings" in result

    def test_calculate_cost_avoidance_small_model(self):
        from distllm.cli.cost_avoid import calculate_cost_avoidance

        result = calculate_cost_avoidance(
            model_name="microsoft/Phi-3-mini-4k-instruct",
            requests_per_day=1000,
            gpu_type="RTX 4060",
            cloud_api="gpt-4o-mini",
        )
        assert result["monthly_self_hosted_cost"] > 0

    def test_calculate_cost_avoidance_no_savings(self):
        from distllm.cli.cost_avoid import calculate_cost_avoidance

        result = calculate_cost_avoidance(
            model_name="meta-llama/Llama-3.1-70B",
            requests_per_day=1,
            gpu_type="H100",
            cloud_api="llama-3.1-70b-deepinfra",
        )
        assert result["monthly_savings"] >= 0

    def test_main_with_args(self):
        from distllm.cli.cost_avoid import main

        with patch("sys.argv", ["cost_avoid", "--model", "test/model"]):
            with patch("distllm.cli.cost_avoid.calculate_cost_avoidance") as mock_calc:
                mock_calc.return_value = {
                    "model": "test/model", "model_size_b": 7,
                    "estimated_vram_gb": 14, "gpus_needed": 1,
                    "requests_per_day": 10000,
                    "monthly_api_cost": 100.0, "monthly_self_hosted_cost": 50.0,
                    "monthly_savings": 50.0, "savings_percent": 50.0,
                    "payback_period_days": 100.0,
                }
                main()

    def test_main_with_no_savings(self):
        from distllm.cli.cost_avoid import main

        with patch("sys.argv", ["cost_avoid", "--requests-per-day", "1"]):
            with patch("distllm.cli.cost_avoid.calculate_cost_avoidance") as mock_calc:
                mock_calc.return_value = {
                    "model": "test", "model_size_b": 7,
                    "estimated_vram_gb": 14, "gpus_needed": 1,
                    "requests_per_day": 1,
                    "monthly_api_cost": 1.0, "monthly_self_hosted_cost": 50.0,
                    "monthly_savings": 0.0, "savings_percent": 0.0,
                    "payback_period_days": float("inf"),
                }
                main()


# ===========================================================================
# deploy.py
# ===========================================================================


class TestCliDeploy:
    """Tests for distllm.cli.deploy — run_deploy."""

    def test_deploy_dry_run(self):
        from distllm.cli.deploy import run_deploy

        console = MagicMock()
        run_deploy("test-model", 4, "float16", "none", None, True,
                   False, "localhost", 8000, console)
        console.print.assert_any_call("[yellow]Dry run — no changes made.[/yellow]")

    def test_deploy_with_wait_success(self):
        from distllm.cli.deploy import run_deploy
        import httpx

        console = MagicMock()
        mock_post_resp = MagicMock()
        mock_get_resp = MagicMock()
        mock_get_resp.json.return_value = {"nodes": 2}

        with patch("distllm.cli.deploy.httpx.Client") as mc:
            inst = MagicMock()
            inst.post.return_value = mock_post_resp
            inst.get.return_value = mock_get_resp
            mc.return_value.__enter__ = MagicMock(return_value=inst)
            mc.return_value.__exit__ = MagicMock(return_value=False)

            run_deploy("test-model", 2, "float16", "none", None, False,
                       True, "localhost", 8000, console)

    def test_deploy_connect_error(self):
        from distllm.cli.deploy import run_deploy
        import httpx

        console = MagicMock()
        with patch("distllm.cli.deploy.httpx.Client") as mc:
            inst = MagicMock()
            inst.post.side_effect = httpx.ConnectError("refused")
            mc.return_value.__enter__ = MagicMock(return_value=inst)
            mc.return_value.__exit__ = MagicMock(return_value=False)

            run_deploy("test-model", 2, "float16", "none", None, False,
                       False, "badhost", 9999, console)

    def test_deploy_http_error(self):
        from distllm.cli.deploy import run_deploy
        import httpx

        console = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.text = "Internal error"
        mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "error", request=MagicMock(), response=mock_resp
        )

        with patch("distllm.cli.deploy.httpx.Client") as mc:
            inst = MagicMock()
            inst.post.return_value = mock_resp
            mc.return_value.__enter__ = MagicMock(return_value=inst)
            mc.return_value.__exit__ = MagicMock(return_value=False)

            run_deploy("test-model", 2, "float16", "none", None, False,
                       False, "localhost", 8000, console)


# ===========================================================================
# logs.py
# ===========================================================================


class TestCliLogs:
    """Tests for distllm.cli.logs — _stream_logs."""

    def test_logs_non_follow_empty(self):
        from distllm.cli.logs import _stream_logs

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"logs": []}
        with patch("distllm.cli.logs.httpx.Client") as mc:
            inst = MagicMock()
            inst.get.return_value = mock_resp
            mc.return_value.__enter__ = MagicMock(return_value=inst)
            mc.return_value.__exit__ = MagicMock(return_value=False)
            _stream_logs("localhost", 8000, follow=False, lines=10)

    def test_logs_non_follow_with_entries(self):
        from distllm.cli.logs import _stream_logs

        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "logs": [
                {"level": "INFO", "timestamp": "12:00:00",
                 "component": "coordinator", "message": "started"},
                {"level": "ERROR", "timestamp": "12:00:01",
                 "component": "worker", "message": "failed"},
            ]
        }
        with patch("distllm.cli.logs.httpx.Client") as mc:
            inst = MagicMock()
            inst.get.return_value = mock_resp
            mc.return_value.__enter__ = MagicMock(return_value=inst)
            mc.return_value.__exit__ = MagicMock(return_value=False)
            _stream_logs("localhost", 8000, follow=False, lines=50)

    def test_logs_connect_error(self):
        from distllm.cli.logs import _stream_logs
        import httpx

        with patch("distllm.cli.logs.httpx.Client") as mc:
            inst = MagicMock()
            inst.get.side_effect = httpx.ConnectError("refused")
            mc.return_value.__enter__ = MagicMock(return_value=inst)
            mc.return_value.__exit__ = MagicMock(return_value=False)
            _stream_logs("badhost", 9999)

    def test_logs_http_error(self):
        from distllm.cli.logs import _stream_logs
        import httpx

        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.text = "error"
        mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "err", request=MagicMock(), response=mock_resp
        )
        with patch("distllm.cli.logs.httpx.Client") as mc:
            inst = MagicMock()
            inst.get.return_value = mock_resp
            mc.return_value.__enter__ = MagicMock(return_value=inst)
            mc.return_value.__exit__ = MagicMock(return_value=False)
            _stream_logs("localhost", 8000)

    def test_logs_with_filters(self):
        from distllm.cli.logs import _stream_logs

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"logs": []}
        with patch("distllm.cli.logs.httpx.Client") as mc:
            inst = MagicMock()
            inst.get.return_value = mock_resp
            mc.return_value.__enter__ = MagicMock(return_value=inst)
            mc.return_value.__exit__ = MagicMock(return_value=False)
            _stream_logs("localhost", 8000, follow=False, level="ERROR",
                         component="worker", search="oom")

        # Verify params include level, component, search
        call_kwargs = inst.get.call_args.kwargs
        assert call_kwargs["params"]["level"] == "ERROR"
        assert call_kwargs["params"]["component"] == "worker"
        assert call_kwargs["params"]["search"] == "oom"


# ===========================================================================
# models.py
# ===========================================================================


class TestCliModels:
    """Tests for distllm.cli.models — _list_models, _model_info,
    _load_model, _unload_model.
    """

    def test_list_models_happy_path(self):
        from distllm.cli.models import _list_models

        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "data": [
                {"id": "model-1", "object": "model", "owned_by": "org"},
                {"id": "model-2", "object": "model", "owned_by": "org"},
            ]
        }
        with patch("distllm.cli.models.httpx.Client") as mc:
            inst = MagicMock()
            inst.get.return_value = mock_resp
            mc.return_value.__enter__ = MagicMock(return_value=inst)
            mc.return_value.__exit__ = MagicMock(return_value=False)
            _list_models("localhost", 8000)

    def test_list_models_empty(self):
        from distllm.cli.models import _list_models

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"data": []}
        with patch("distllm.cli.models.httpx.Client") as mc:
            inst = MagicMock()
            inst.get.return_value = mock_resp
            mc.return_value.__enter__ = MagicMock(return_value=inst)
            mc.return_value.__exit__ = MagicMock(return_value=False)
            _list_models("localhost", 8000)

    def test_list_models_connect_error(self):
        from distllm.cli.models import _list_models
        import httpx

        with patch("distllm.cli.models.httpx.Client") as mc:
            inst = MagicMock()
            inst.get.side_effect = httpx.ConnectError("refused")
            mc.return_value.__enter__ = MagicMock(return_value=inst)
            mc.return_value.__exit__ = MagicMock(return_value=False)
            _list_models("badhost", 9999)

    def test_model_info_found(self):
        from distllm.cli.models import _model_info

        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "data": [
                {"id": "my-model", "object": "model", "owned_by": "me"},
            ]
        }
        with patch("distllm.cli.models.httpx.Client") as mc:
            inst = MagicMock()
            inst.get.return_value = mock_resp
            mc.return_value.__enter__ = MagicMock(return_value=inst)
            mc.return_value.__exit__ = MagicMock(return_value=False)
            _model_info("localhost", 8000, "my-model")

    def test_model_info_not_found(self):
        from distllm.cli.models import _model_info

        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "data": [
                {"id": "other-model", "object": "model", "owned_by": "org"},
            ]
        }
        with patch("distllm.cli.models.httpx.Client") as mc:
            inst = MagicMock()
            inst.get.return_value = mock_resp
            mc.return_value.__enter__ = MagicMock(return_value=inst)
            mc.return_value.__exit__ = MagicMock(return_value=False)
            _model_info("localhost", 8000, "missing-model")

    def test_load_model_happy_path(self):
        from distllm.cli.models import _load_model

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"model": "test-model"}
        with patch("distllm.cli.models.httpx.Client") as mc:
            inst = MagicMock()
            inst.post.return_value = mock_resp
            mc.return_value.__enter__ = MagicMock(return_value=inst)
            mc.return_value.__exit__ = MagicMock(return_value=False)
            _load_model("localhost", 8000, "test-model")
        inst.post.assert_called_once()

    def test_unload_model_happy_path(self):
        from distllm.cli.models import _unload_model

        mock_resp = MagicMock()
        with patch("distllm.cli.models.httpx.Client") as mc:
            inst = MagicMock()
            inst.post.return_value = mock_resp
            mc.return_value.__enter__ = MagicMock(return_value=inst)
            mc.return_value.__exit__ = MagicMock(return_value=False)
            _unload_model("localhost", 8000, "test-model")
        inst.post.assert_called_once()

    def test_load_model_http_error(self):
        from distllm.cli.models import _load_model
        import httpx

        mock_resp = MagicMock()
        mock_resp.status_code = 400
        mock_resp.text = "bad model"
        mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "err", request=MagicMock(), response=mock_resp
        )
        with patch("distllm.cli.models.httpx.Client") as mc:
            inst = MagicMock()
            inst.post.return_value = mock_resp
            mc.return_value.__enter__ = MagicMock(return_value=inst)
            mc.return_value.__exit__ = MagicMock(return_value=False)
            _load_model("localhost", 8000, "bad-model")


# ===========================================================================
# notify.py
# ===========================================================================


class TestCliNotify:
    """Tests for distllm.cli.notify — notify_app Typer commands."""

    @patch("distllm.cli.notify.NotificationManager")
    def test_notify_send_console(self, mock_nm_cls):
        from distllm.cli.notify import notify_app

        mock_nm = MagicMock()
        mock_nm.send.return_value = True
        mock_nm_cls.return_value = mock_nm

        result = runner.invoke(notify_app, [
            "send", "--title", "Test", "--message", "Hello",
        ], catch_exceptions=False)
        assert result.exit_code == 0

    @patch("distllm.cli.notify.NotificationManager")
    def test_notify_send_failure(self, mock_nm_cls):
        from distllm.cli.notify import notify_app

        mock_nm = MagicMock()
        mock_nm.send.return_value = False
        mock_nm_cls.return_value = mock_nm

        result = runner.invoke(notify_app, [
            "send", "--title", "Fail", "--message", "Oops",
        ], catch_exceptions=False)
        assert result.exit_code == 1

    @patch("distllm.cli.notify.NotificationManager")
    def test_notify_send_slack_webhook(self, mock_nm_cls):
        from distllm.cli.notify import notify_app

        mock_nm = MagicMock()
        mock_nm.send.return_value = True
        mock_nm_cls.return_value = mock_nm

        result = runner.invoke(notify_app, [
            "send", "--title", "Alert", "--message", "Slack test",
            "--channel", "slack", "--webhook-url", "https://hooks.slack.com/test",
        ], catch_exceptions=False)
        assert result.exit_code == 0
        mock_nm.configure_slack.assert_called_once()

    @patch("distllm.cli.notify.NotificationManager")
    def test_notify_history_empty(self, mock_nm_cls):
        from distllm.cli.notify import notify_app

        mock_nm = MagicMock()
        mock_nm.recent.return_value = []
        mock_nm_cls.return_value = mock_nm

        result = runner.invoke(notify_app, ["history"], catch_exceptions=False)
        assert result.exit_code == 0

    @patch("distllm.cli.notify.NotificationManager")
    def test_notify_history_with_data(self, mock_nm_cls):
        from distllm.cli.notify import notify_app

        mock_note = MagicMock()
        mock_note.timestamp = 1700000000
        mock_note.severity = MagicMock()
        mock_note.severity.value = "info"
        mock_note.channel = MagicMock()
        mock_note.channel.value = "console"
        mock_note.title = "Test notification"
        mock_note.message = "This is a test"

        mock_nm = MagicMock()
        mock_nm.recent.return_value = [mock_note]
        mock_nm_cls.return_value = mock_nm

        result = runner.invoke(notify_app, ["history", "--limit", "5"],
                               catch_exceptions=False)
        assert result.exit_code == 0


# ===========================================================================
# profile.py
# ===========================================================================


class TestCliProfile:
    """Tests for distllm.cli.profile — run_profile."""

    def test_profile_all_successful(self):
        from distllm.cli.profile import run_profile

        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "usage": {"completion_tokens": 10, "tokens_per_second": 25.0},
            "generation_time": 0.3,
        }
        console = MagicMock()
        with patch("distllm.cli.profile.httpx.Client") as mc:
            inst = MagicMock()
            inst.post.return_value = mock_resp
            mc.return_value.__enter__ = MagicMock(return_value=inst)
            mc.return_value.__exit__ = MagicMock(return_value=False)
            run_profile("test", "localhost", 8000, 10, 20, 3, None, console)

        assert console.print.call_count > 0

    def test_profile_all_fail(self):
        from distllm.cli.profile import run_profile
        import httpx

        console = MagicMock()
        with patch("distllm.cli.profile.httpx.Client") as mc:
            inst = MagicMock()
            inst.post.side_effect = httpx.ConnectError("refused")
            mc.return_value.__enter__ = MagicMock(return_value=inst)
            mc.return_value.__exit__ = MagicMock(return_value=False)
            run_profile("test", "badhost", 9999, 10, 20, 2, None, console)

        # Should print no-successful message
        assert any(
            "No successful iterations" in str(call)
            for call in console.print.call_args_list
        )

    def test_profile_partial_failures(self):
        from distllm.cli.profile import run_profile
        import httpx

        console = MagicMock()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "usage": {"completion_tokens": 5, "tokens_per_second": 10.0},
            "generation_time": 0.1,
        }

        call_count = [0]

        def side_effect(*a, **kw):
            call_count[0] += 1
            if call_count[0] == 1:
                return mock_resp
            raise httpx.ConnectError("refused on second attempt")

        with patch("distllm.cli.profile.httpx.Client") as mc:
            inst = MagicMock()
            inst.post.side_effect = side_effect
            mc.return_value.__enter__ = MagicMock(return_value=inst)
            mc.return_value.__exit__ = MagicMock(return_value=False)
            run_profile("test", "localhost", 8000, 10, 10, 2, None, console)

    def test_profile_json_output(self):
        from distllm.cli.profile import run_profile
        import tempfile, os, json

        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "usage": {"completion_tokens": 10, "tokens_per_second": 20.0},
            "generation_time": 0.2,
        }
        console = MagicMock()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            out_path = f.name

        try:
            with patch("distllm.cli.profile.httpx.Client") as mc:
                inst = MagicMock()
                inst.post.return_value = mock_resp
                mc.return_value.__enter__ = MagicMock(return_value=inst)
                mc.return_value.__exit__ = MagicMock(return_value=False)
                run_profile("test", "localhost", 8000, 10, 20, 3, out_path, console)

            with open(out_path) as f:
                data = json.load(f)
                assert data["model"] == "test"
                assert data["successful"] == 3
        finally:
            os.unlink(out_path)


# ===========================================================================
# prompts.py
# ===========================================================================


class TestCliPrompts:
    """Tests for distllm.cli.prompts — prompt_app Typer commands."""

    @patch("distllm.cli.prompts.list_by_category")
    @patch("distllm.cli.prompts.SYSTEM_PROMPTS")
    def test_prompt_list_all(self, mock_prompts, mock_list_cat):
        from distllm.cli.prompts import prompt_app

        mock_p = MagicMock()
        mock_p.id = "test-prompt"
        mock_p.category = "general"
        mock_p.name = "Test Prompt"
        mock_p.description = "A test prompt"
        mock_p.tags = ["test"]
        mock_prompts.values.return_value = [mock_p]

        result = runner.invoke(prompt_app, ["list"], catch_exceptions=False)
        assert result.exit_code == 0

    @patch("distllm.cli.prompts.search_prompts")
    def test_prompt_list_search(self, mock_search):
        from distllm.cli.prompts import prompt_app

        mock_p = MagicMock()
        mock_p.id = "code-review"
        mock_p.category = "code"
        mock_p.name = "Code Review"
        mock_p.description = "Review code"
        mock_p.tags = ["code", "review"]
        mock_search.return_value = [mock_p]

        result = runner.invoke(prompt_app, ["list", "--search", "code"],
                               catch_exceptions=False)
        assert result.exit_code == 0

    @patch("distllm.cli.prompts.list_by_category", return_value=[])
    def test_prompt_list_empty(self, mock_list_cat):
        from distllm.cli.prompts import prompt_app

        result = runner.invoke(prompt_app, ["list", "--category", "missing"],
                               catch_exceptions=False)
        assert result.exit_code == 0

    @patch("distllm.cli.prompts.get_prompt")
    def test_prompt_show_found(self, mock_get):
        from distllm.cli.prompts import prompt_app

        mock_p = MagicMock()
        mock_p.id = "test-prompt"
        mock_p.category = "general"
        mock_p.name = "Test"
        mock_p.description = "A test"
        mock_p.tags = ["test"]
        mock_p.version = "1.0"
        mock_p.prompt = "You are a test assistant."
        mock_get.return_value = mock_p

        result = runner.invoke(prompt_app, ["show", "test-prompt"],
                               catch_exceptions=False)
        assert result.exit_code == 0

    @patch("distllm.cli.prompts.get_prompt", return_value=None)
    def test_prompt_show_not_found(self, mock_get):
        from distllm.cli.prompts import prompt_app

        result = runner.invoke(prompt_app, ["show", "missing"],
                               catch_exceptions=False)
        assert result.exit_code == 1

    @patch("distllm.cli.prompts.list_categories")
    @patch("distllm.cli.prompts.list_by_category")
    def test_prompt_categories(self, mock_list_cat, mock_cats):
        from distllm.cli.prompts import prompt_app

        mock_cats.return_value = ["general", "code"]
        mock_list_cat.return_value = [MagicMock()]

        result = runner.invoke(prompt_app, ["categories"], catch_exceptions=False)
        assert result.exit_code == 0

    @patch("distllm.cli.prompts.get_prompt")
    def test_prompt_use_found(self, mock_get):
        from distllm.cli.prompts import prompt_app

        mock_p = MagicMock()
        mock_p.id = "test-prompt"
        mock_p.prompt = "You are a test assistant."
        mock_get.return_value = mock_p

        result = runner.invoke(prompt_app, ["use", "test-prompt"],
                               catch_exceptions=False)
        assert result.exit_code == 0

    @patch("distllm.cli.prompts.get_prompt", return_value=None)
    def test_prompt_use_not_found(self, mock_get):
        from distllm.cli.prompts import prompt_app

        result = runner.invoke(prompt_app, ["use", "missing"],
                               catch_exceptions=False)
        assert result.exit_code == 1


# ===========================================================================
# quota.py
# ===========================================================================


class TestCliQuota:
    """Tests for distllm.cli.quota — quota_app Typer commands."""

    @patch("distllm.cli.quota.UsageMeter")
    def test_quota_set(self, mock_meter_cls):
        from distllm.cli.quota import quota_app

        mock_meter = MagicMock()
        mock_meter_cls.return_value = mock_meter

        result = runner.invoke(quota_app, [
            "set", "tenant-1", "--tokens-per-day", "100000",
        ], catch_exceptions=False)
        assert result.exit_code == 0
        mock_meter.set_quota.assert_called_once()

    @patch("distllm.cli.quota.UsageMeter")
    def test_quota_show_with_data(self, mock_meter_cls):
        from distllm.cli.quota import quota_app

        mock_quota = MagicMock()
        mock_quota.max_tokens_per_day = 100000
        mock_quota.max_requests_per_minute = 100
        mock_quota.max_tokens_per_request = 4096
        mock_quota.max_concurrent_requests = 5
        mock_quota.cost_budget_per_month = 100.0
        mock_quota.overage_allowed = False

        mock_usage = MagicMock()
        mock_usage.total_requests = 500
        mock_usage.total_input_tokens = 50000
        mock_usage.total_output_tokens = 100000
        mock_usage.total_cost = 0.5
        mock_usage.daily_tokens = {}

        mock_meter = MagicMock()
        mock_meter.get_quota.return_value = mock_quota
        mock_meter.tenant_usage.return_value = mock_usage
        mock_meter_cls.return_value = mock_meter

        result = runner.invoke(quota_app, ["show", "tenant-1"],
                               catch_exceptions=False)
        assert result.exit_code == 0

    @patch("distllm.cli.quota.UsageMeter")
    def test_quota_show_no_data(self, mock_meter_cls):
        from distllm.cli.quota import quota_app

        mock_meter = MagicMock()
        mock_meter.get_quota.return_value = None
        mock_meter.tenant_usage.return_value = None
        mock_meter_cls.return_value = mock_meter

        result = runner.invoke(quota_app, ["show", "missing-tenant"],
                               catch_exceptions=False)
        assert result.exit_code == 0

    @patch("distllm.cli.quota.UsageMeter")
    def test_quota_list_empty(self, mock_meter_cls):
        from distllm.cli.quota import quota_app

        mock_meter = MagicMock()
        mock_meter.all_tenants.return_value = []
        mock_meter_cls.return_value = mock_meter

        result = runner.invoke(quota_app, ["list"], catch_exceptions=False)
        assert result.exit_code == 0

    @patch("distllm.cli.quota.UsageMeter")
    def test_quota_invoice(self, mock_meter_cls):
        from distllm.cli.quota import quota_app

        mock_meter = MagicMock()
        mock_meter.generate_invoice.return_value = {
            "period_start": 1700000000, "period_end": 1700086400,
            "total_requests": 100, "total_input_tokens": 5000,
            "total_output_tokens": 10000, "total_cost": 0.05,
            "overage_cost": 0.0, "grand_total": 0.05,
        }
        mock_meter_cls.return_value = mock_meter

        result = runner.invoke(quota_app, ["invoice", "tenant-1"],
                               catch_exceptions=False)
        assert result.exit_code == 0

    @patch("distllm.cli.quota.UsageMeter")
    def test_quota_export(self, mock_meter_cls):
        from distllm.cli.quota import quota_app

        mock_meter = MagicMock()
        mock_meter.export_csv.return_value = "/tmp/export.csv"
        mock_meter_cls.return_value = mock_meter

        result = runner.invoke(quota_app, ["export", "/tmp/export.csv"],
                               catch_exceptions=False)
        assert result.exit_code == 0


# ===========================================================================
# run.py
# ===========================================================================


class TestCliRun:
    """Tests for distllm.cli.run — run_inference."""

    def test_run_inference_debug_mode(self):
        from distllm.cli.run import run_inference

        console = MagicMock()
        with patch("distllm.cli.run.set_debug_mode") as mock_debug:
            try:
                with patch("distllm.cli.run.uvicorn") as mock_uvicorn:
                    mock_uvicorn.run.side_effect = KeyboardInterrupt()
                    run_inference("test-model", True, "", 8000, "float16",
                                  256, 0.7, console, debug=True)
            except SystemExit:
                pass
            mock_debug.assert_called_once_with(True)

    def test_run_inference_keyboard_interrupt(self):
        from distllm.cli.run import run_inference

        console = MagicMock()
        with patch("distllm.cli.run.uvicorn") as mock_uvicorn:
            mock_uvicorn.run.side_effect = KeyboardInterrupt()
            run_inference("test-model", True, "", 8000, "float16",
                          256, 0.7, console)

    def test_run_inference_error_exits(self):
        from distllm.cli.run import run_inference

        console = MagicMock()
        with pytest.raises(SystemExit):
            with patch("distllm.cli.run.uvicorn") as mock_uvicorn:
                mock_uvicorn.run.side_effect = RuntimeError("startup failed")
                run_inference("test-model", True, "", 8000, "float16",
                              256, 0.7, console)

    def test_run_inference_with_config(self):
        from distllm.cli.run import run_inference
        import tempfile, os, yaml

        config = {"model": {"name": "cfg-model", "dtype": "bfloat16"}}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config, f)
            cfg_path = f.name

        try:
            console = MagicMock()
            with patch("distllm.cli.run.uvicorn") as mock_uvicorn:
                with patch("distllm.cli.run.create_coordinator") as mock_create:
                    mock_uvicorn.run.side_effect = KeyboardInterrupt()
                    run_inference("cli-model", True, cfg_path, 8000,
                                  "float16", 256, 0.7, console)
                    # Config should override model name
                    mock_create.assert_called_once()
                    args, kwargs = mock_create.call_args
                    assert kwargs.get("model_name") == "cfg-model"
                    assert kwargs.get("dtype") == "bfloat16"
        finally:
            os.unlink(cfg_path)


# ===========================================================================
# setup.py
# ===========================================================================


class TestCliSetup:
    """Tests for distllm.cli.setup — run_setup."""

    def test_setup_local_mode(self, tmp_path):
        from distllm.cli.setup import run_setup

        config_path = str(tmp_path / "config.yaml")
        console = MagicMock()

        inputs = iter([
            "",       # default model
            "local",  # mode
            "",       # default dtype
            "",       # default max_tokens
            "",       # default temperature
            "n",      # no TLS
        ])
        console.input.side_effect = lambda prompt="": next(inputs)

        run_setup(config_path, console)
        assert (tmp_path / "config.yaml").exists()

    def test_setup_distributed_mode(self, tmp_path):
        from distllm.cli.setup import run_setup

        config_path = str(tmp_path / "dist-config.yaml")
        console = MagicMock()

        inputs = iter([
            "test-model",
            "distributed",
            "float32",
            "2",    # num nodes
            "",     # node 0 host
            "",     # node 0 port
            "0",    # node 0 start layer
            "15",   # node 0 end layer
            "",     # node 1 host
            "",     # node 1 port
            "16",   # node 1 start layer
            "31",   # node 1 end layer
            "512",  # max tokens
            "0.8",  # temperature
            "n",    # no TLS
        ])
        console.input.side_effect = lambda prompt="": next(inputs)

        run_setup(config_path, console)
        assert (tmp_path / "dist-config.yaml").exists()

    def test_setup_overwrite_existing(self, tmp_path):
        from distllm.cli.setup import run_setup

        config_path = str(tmp_path / "existing.yaml")
        with open(config_path, "w") as f:
            f.write("old: config\n")

        console = MagicMock()
        inputs = iter([
            "",
            "local",
            "",
            "",
            "",
            "y",     # confirm overwrite
            "n",     # no TLS
        ])
        console.input.side_effect = lambda prompt="": next(inputs)

        run_setup(config_path, console)
        assert (tmp_path / "existing.yaml").exists()

    def test_setup_decline_overwrite(self, tmp_path):
        from distllm.cli.setup import run_setup

        config_path = str(tmp_path / "existing2.yaml")
        with open(config_path, "w") as f:
            f.write("old: config\n")

        console = MagicMock()
        inputs = iter([
            "",
            "local",
            "",
            "",
            "",
            "n",     # decline overwrite
        ])
        console.input.side_effect = lambda prompt="": next(inputs)

        run_setup(config_path, console)
        # Config should not be overwritten (still has old content)
        with open(config_path) as f:
            assert f.read().strip() == "old: config"


# ===========================================================================
# status.py
# ===========================================================================


class TestCliStatus:
    """Tests for distllm.cli.status — show_status."""

    def test_status_healthy(self):
        from distllm.cli.status import show_status

        health_resp = MagicMock()
        health_resp.json.return_value = {
            "status": "healthy", "model": "test", "nodes": 2,
            "node_health": {
                "node-0": {"healthy": True, "memory_used": 1024, "memory_total": 8192},
                "node-1": {"healthy": False, "memory_used": 0, "memory_total": 0},
            },
        }
        models_resp = MagicMock()
        models_resp.json.return_value = {
            "data": [{"id": "model-1"}, {"id": "model-2"}],
        }
        metrics_resp = MagicMock()
        metrics_resp.text = "# HELP cpu_usage\ncpu_usage 42\n"

        with patch("distllm.cli.status.httpx.Client") as mc:
            inst = MagicMock()

            def get_side_effect(url, **kw):
                if "/health" in url:
                    return health_resp
                if "/v1/models" in url:
                    return models_resp
                if "/metrics" in url:
                    return metrics_resp
                return MagicMock()

            inst.get.side_effect = get_side_effect
            mc.return_value.__enter__ = MagicMock(return_value=inst)
            mc.return_value.__exit__ = MagicMock(return_value=False)

            show_status("localhost", 8000, MagicMock())

    def test_status_connect_error(self):
        from distllm.cli.status import show_status
        import httpx

        with patch("distllm.cli.status.httpx.Client") as mc:
            inst = MagicMock()
            inst.get.side_effect = httpx.ConnectError("refused")
            mc.return_value.__enter__ = MagicMock(return_value=inst)
            mc.return_value.__exit__ = MagicMock(return_value=False)

            console = MagicMock()
            show_status("badhost", 9999, console)

    def test_status_generic_exception(self):
        from distllm.cli.status import show_status

        with patch("distllm.cli.status.httpx.Client") as mc:
            inst = MagicMock()
            inst.get.side_effect = ValueError("unexpected")
            mc.return_value.__enter__ = MagicMock(return_value=inst)
            mc.return_value.__exit__ = MagicMock(return_value=False)

            console = MagicMock()
            show_status("localhost", 8000, console)


# ===========================================================================
# tutorial.py
# ===========================================================================


class TestCliTutorial:
    """Tests for distllm.cli.tutorial — _run_tutorial, main."""

    @patch("distllm.cli.tutorial.torch")
    def test_run_tutorial_cuda_available(self, mock_torch):
        from distllm.cli.tutorial import _run_tutorial

        mock_torch.__version__ = "2.1.0"
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 2
        mock_torch.cuda.get_device_name.side_effect = lambda i: f"GPU {i}"

        with patch("builtins.input", side_effect=["", "", "", "", ""]):
            _run_tutorial()

    @patch("distllm.cli.tutorial.torch")
    def test_run_tutorial_no_cuda(self, mock_torch):
        from distllm.cli.tutorial import _run_tutorial

        mock_torch.__version__ = "2.1.0"
        mock_torch.cuda.is_available.return_value = False

        with patch("builtins.input", side_effect=["", "", "", "", ""]):
            _run_tutorial()

    @patch("distllm.cli.tutorial.torch", side_effect=ImportError("no torch"))
    def test_run_tutorial_no_torch(self, mock_torch):
        from distllm.cli.tutorial import _run_tutorial

        with patch("builtins.input", side_effect=["", "", "", "", ""]):
            _run_tutorial()

    def test_main_normal(self):
        from distllm.cli.tutorial import main

        with patch("distllm.cli.tutorial._run_tutorial"):
            main()

    def test_main_keyboard_interrupt(self):
        from distllm.cli.tutorial import main

        with patch("distllm.cli.tutorial._run_tutorial",
                   side_effect=KeyboardInterrupt):
            with pytest.raises(SystemExit):
                main()


# ===========================================================================
# verify.py
# ===========================================================================


class TestCliVerify:
    """Tests for distllm.cli.verify — verify_app Typer commands."""

    @patch("distllm.cli.verify.AccuracyVerifier")
    def test_verify_run_success(self, mock_verifier_cls):
        from distllm.cli.verify import verify_app

        mock_report = MagicMock()
        mock_report.summary.return_value = {"total": 3, "passed": 3, "failed": 0}
        mock_verifier = MagicMock()
        mock_verifier.verify.return_value = mock_report
        mock_verifier_cls.return_value = mock_verifier

        result = runner.invoke(verify_app, [
            "run", "--model", "test-model",
        ], catch_exceptions=False)
        assert result.exit_code == 0

    @patch("distllm.cli.verify.AccuracyVerifier")
    def test_verify_run_with_failures(self, mock_verifier_cls):
        from distllm.cli.verify import verify_app

        mock_report = MagicMock()
        mock_report.summary.return_value = {"total": 3, "passed": 2, "failed": 1}
        mock_verifier = MagicMock()
        mock_verifier.verify.return_value = mock_report
        mock_verifier_cls.return_value = mock_verifier

        result = runner.invoke(verify_app, [
            "run", "--model", "test-model", "--nodes", "4",
            "--prompt", "Hello", "--prompt", "World",
        ], catch_exceptions=False)
        assert result.exit_code == 1

    @patch("distllm.cli.verify.list_available_backends")
    def test_verify_list_backends_with_data(self, mock_list):
        from distllm.cli.verify import verify_app

        mock_backend = MagicMock()
        mock_backend.name = "test-backend"
        mock_backend.adapter_class = MagicMock()
        mock_backend.adapter_class.display_name.return_value = "Test Backend"
        mock_backend.adapter_class.version.return_value = "1.0"
        mock_backend.adapter_class.description.return_value = "A test backend"
        mock_list.return_value = [mock_backend]

        result = runner.invoke(verify_app, ["list-backends"],
                               catch_exceptions=False)
        assert result.exit_code == 0

    @patch("distllm.cli.verify.list_available_backends", return_value=[])
    def test_verify_list_backends_empty(self, mock_list):
        from distllm.cli.verify import verify_app

        result = runner.invoke(verify_app, ["list-backends"],
                               catch_exceptions=False)
        assert result.exit_code == 0


# ===========================================================================
# webhook.py
# ===========================================================================


class TestCliWebhook:
    """Tests for distllm.cli.webhook — webhook_app Typer commands."""

    @patch("distllm.cli.webhook.WebhookManager")
    def test_webhook_register_success(self, mock_mgr_cls):
        from distllm.cli.webhook import webhook_app

        mock_mgr = MagicMock()
        mock_mgr.register.return_value = True
        mock_mgr_cls.return_value = mock_mgr

        result = runner.invoke(webhook_app, [
            "register", "https://hooks.example.com/events",
            "--event", "model.loaded", "--event", "node.joined",
            "--label", "test-webhook",
        ], catch_exceptions=False)
        assert result.exit_code == 0

    @patch("distllm.cli.webhook.WebhookManager")
    def test_webhook_register_failure(self, mock_mgr_cls):
        from distllm.cli.webhook import webhook_app

        mock_mgr = MagicMock()
        mock_mgr.register.return_value = False
        mock_mgr_cls.return_value = mock_mgr

        result = runner.invoke(webhook_app, [
            "register", "https://hooks.example.com/bad",
        ], catch_exceptions=False)
        assert result.exit_code == 1

    @patch("distllm.cli.webhook.WebhookManager")
    def test_webhook_list_empty(self, mock_mgr_cls):
        from distllm.cli.webhook import webhook_app

        mock_mgr = MagicMock()
        mock_mgr.list_targets.return_value = []
        mock_mgr_cls.return_value = mock_mgr

        result = runner.invoke(webhook_app, ["list"], catch_exceptions=False)
        assert result.exit_code == 0

    @patch("distllm.cli.webhook.WebhookManager")
    def test_webhook_list_with_targets(self, mock_mgr_cls):
        from distllm.cli.webhook import webhook_app

        mock_target = MagicMock()
        mock_target.url = "https://hooks.example.com/events"
        mock_target.active = True
        mock_target.events = ["model.loaded", "node.joined"]
        mock_target.label = "test"
        mock_target.success_rate = 1.0

        mock_mgr = MagicMock()
        mock_mgr.list_targets.return_value = [mock_target]
        mock_mgr.success_rate.return_value = 1.0
        mock_mgr_cls.return_value = mock_mgr

        result = runner.invoke(webhook_app, ["list"], catch_exceptions=False)
        assert result.exit_code == 0

    @patch("distllm.cli.webhook.WebhookManager")
    def test_webhook_unregister_found(self, mock_mgr_cls):
        from distllm.cli.webhook import webhook_app

        mock_mgr = MagicMock()
        mock_mgr.unregister.return_value = True
        mock_mgr_cls.return_value = mock_mgr

        result = runner.invoke(webhook_app, [
            "unregister", "https://hooks.example.com/events",
        ], catch_exceptions=False)
        assert result.exit_code == 0

    @patch("distllm.cli.webhook.WebhookManager")
    def test_webhook_unregister_not_found(self, mock_mgr_cls):
        from distllm.cli.webhook import webhook_app

        mock_mgr = MagicMock()
        mock_mgr.unregister.return_value = False
        mock_mgr_cls.return_value = mock_mgr

        result = runner.invoke(webhook_app, [
            "unregister", "https://hooks.example.com/missing",
        ], catch_exceptions=False)
        assert result.exit_code == 0

    @patch("distllm.cli.webhook.WebhookManager")
    def test_webhook_test(self, mock_mgr_cls):
        from distllm.cli.webhook import webhook_app

        mock_mgr = MagicMock()
        mock_mgr_cls.return_value = mock_mgr

        result = runner.invoke(webhook_app, [
            "test", "https://hooks.example.com/events",
        ], catch_exceptions=False)
        assert result.exit_code == 0
        mock_mgr.register.assert_called_once()
        mock_mgr.dispatch.assert_called_once()
