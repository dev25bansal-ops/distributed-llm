"""Security regression tests for B-01 (sec): model-name supply chain.

A traversal path supplied as ``model_name`` / ``model_path`` must be
rejected by every backend adapter before any load attempt.
"""

from __future__ import annotations

import pytest

from distllm.backends.model_validation import (
    resolve_trust_remote_code,
    validate_model_name,
    validate_model_path,
)


class TestValidateModelName:
    def test_valid_org_name(self) -> None:
        assert validate_model_name("meta-llama/Llama-2-7b") == "meta-llama/Llama-2-7b"

    def test_accepts_env_allowlist_match(self, monkeypatch) -> None:
        monkeypatch.setenv("DISTLLM_ALLOWED_MODELS", "openai/gpt-oss-120b,meta-llama/*")
        assert validate_model_name("meta-llama/Llama-2-7b") == "meta-llama/Llama-2-7b"

    def test_rejects_not_in_env_allowlist(self, monkeypatch) -> None:
        monkeypatch.setenv("DISTLLM_ALLOWED_MODELS", "openai/gpt-oss-120b")
        with pytest.raises(ValueError, match="DISTLLM_ALLOWED_MODELS"):
            validate_model_name("evil-org/evil-model")

    def test_rejects_path_traversal(self) -> None:
        with pytest.raises(ValueError, match="path traversal"):
            validate_model_name("../../etc/passwd")

    def test_rejects_absolute_path(self) -> None:
        with pytest.raises(ValueError, match="absolute path"):
            validate_model_name("/etc/passwd.onnx")

    def test_rejects_bare_name_without_org(self) -> None:
        # Default mode accepts bare names as local identifiers; allowlist
        # mode (DISTLLM_ALLOWED_MODELS set) enforces org/name form.
        import os
        os.environ["DISTLLM_ALLOWED_MODELS"] = "meta-llama/*"
        try:
            with pytest.raises(ValueError, match="org/name"):
                validate_model_name("Llama-2-7b")
        finally:
            del os.environ["DISTLLM_ALLOWED_MODELS"]

    def test_rejects_windows_path(self) -> None:
        with pytest.raises(ValueError, match="Windows path"):
            validate_model_name("C:\\models\\llama.bin")

    def test_rejects_home_dir(self) -> None:
        with pytest.raises(ValueError, match="home-directory"):
            validate_model_name("~/models/llama")

    def test_rejects_empty_string(self) -> None:
        with pytest.raises(ValueError):
            validate_model_name("")


class TestValidateModelPath:
    def test_traversal_path_rejected(self) -> None:
        with pytest.raises(ValueError, match="path traversal"):
            validate_model_path("../../etc/passwd.onnx", allow_local_paths=True)

    def test_home_dir_rejected(self) -> None:
        with pytest.raises(ValueError, match="home-directory"):
            validate_model_path("~/models/llama.gguf", allow_local_paths=True)

    def test_local_path_without_opt_in_rejected(self) -> None:
        with pytest.raises(ValueError, match="allow_local_paths"):
            validate_model_path("/models/model.gguf")

    def test_local_path_with_opt_in_accepted(self) -> None:
        assert (
            validate_model_path("/models/model.gguf", allow_local_paths=True)
            == "/models/model.gguf"
        )


class TestAdapterRejections:
    """Traversal paths are rejected at adapter construction."""

    def test_pytorch_rejects_traversal(self) -> None:
        from distllm.backends.pytorch_backend import PyTorchNodeAdapter

        with pytest.raises(ValueError, match="path traversal"):
            PyTorchNodeAdapter(model_name="../../etc/passwd")

    def test_onnx_rejects_traversal(self) -> None:
        from distllm.backends.onnx_backend import ONNXNodeAdapter

        with pytest.raises(ValueError, match="rejected"):
            ONNXNodeAdapter(model_name="/etc/passwd.onnx")

    def test_llamacpp_rejects_traversal(self) -> None:
        from distllm.backends.llamacpp_backend import LlamacppNodeAdapter

        with pytest.raises(ValueError, match="path traversal"):
            LlamacppNodeAdapter(model_path="../../etc/passwd.gguf")

    def test_exllama_rejects_traversal(self) -> None:
        from distllm.backends.exllama_backend import ExLlamaV2NodeAdapter

        with pytest.raises(ValueError, match="path traversal"):
            ExLlamaV2NodeAdapter(model_name="../../etc/passwd")

    def test_tensorrt_rejects_traversal(self) -> None:
        from distllm.backends.tensorrt_backend import TensorRTLLMAdapter

        with pytest.raises(ValueError, match="rejected"):
            TensorRTLLMAdapter(model_name="/etc/passwd")

    def test_mlx_rejects_traversal(self) -> None:
        from distllm.backends.mlx_backend import MLXNodeAdapter

        with pytest.raises(ValueError, match="path traversal"):
            MLXNodeAdapter(model_name="../../etc/passwd")

    def test_nim_rejects_traversal(self) -> None:
        from distllm.backends.nim_backend import NimNodeAdapter

        with pytest.raises(ValueError, match="rejected"):
            NimNodeAdapter(model_name="/etc/passwd")

    def test_vllm_rejects_traversal(self) -> None:
        from distllm.backends.vllm_backend import VLLMNodeAdapter

        with pytest.raises(ValueError, match="path traversal"):
            VLLMNodeAdapter(model_name="../../etc/passwd")


class TestTrustRemoteCodeDefault:
    def test_default_is_false(self) -> None:
        assert resolve_trust_remote_code(None) is False
        assert resolve_trust_remote_code(False) is False

    def test_explicit_true_preserved(self) -> None:
        assert resolve_trust_remote_code(True) is True
