"""Comprehensive cross-field validation tests for settings.py coverage."""
import os
import tempfile
import pytest
from distllm.config.settings import DistLLMSettings
from distllm.config._model import ModelSettings, SpeculativeSettings
from distllm.config._network import TLSSettings, NetworkSettings
from distllm.config._parallelism import TensorParallelSettings, BatchingSettings, ChunkedPrefillSettings
from distllm.config._backends import VLLMSettings


class TestCrossFieldAllBranches:
    def test_vllm_dtype_match(self):
        s = DistLLMSettings(model=ModelSettings(name="test", dtype="bfloat16"), vllm=VLLMSettings(enabled=True, dtype="bfloat16"))
        s._validate_cross_field()

    def test_vllm_dtype_auto(self):
        s = DistLLMSettings(model=ModelSettings(name="test"), vllm=VLLMSettings(enabled=True, dtype="auto"))
        s._validate_cross_field()

    def test_chunked_prefill_zero_budget(self):
        with pytest.raises(ValueError):
            DistLLMSettings(model=ModelSettings(name="test"), chunked_prefill=ChunkedPrefillSettings(enabled=True), batching=BatchingSettings(max_tokens_per_batch=0))

    def test_chunked_prefill_ok(self):
        s = DistLLMSettings(model=ModelSettings(name="test"), chunked_prefill=ChunkedPrefillSettings(enabled=True), batching=BatchingSettings(max_tokens_per_batch=4096))
        s._validate_cross_field()

    def test_spec_empty_non_default(self):
        with pytest.raises(ValueError):
            DistLLMSettings(model=ModelSettings(name="test"), speculative=SpeculativeSettings(method="draft_model", num_assistant_tokens=10))

    def test_spec_empty_default_ok(self):
        s = DistLLMSettings(model=ModelSettings(name="test"), speculative=SpeculativeSettings(method="draft_model"))
        s._validate_cross_field()

    def test_spec_eagle_empty(self):
        with pytest.raises(ValueError):
            DistLLMSettings(model=ModelSettings(name="test"), speculative=SpeculativeSettings(method="eagle"))

    def test_spec_eagle_with_path(self):
        s = DistLLMSettings(model=ModelSettings(name="test"), speculative=SpeculativeSettings(method="eagle", eagle_checkpoint="/ckpt"))
        s._validate_cross_field()

    def test_tls_no_cert(self):
        with pytest.raises(ValueError):
            DistLLMSettings(model=ModelSettings(name="test"), tls=TLSSettings(enabled=True))

    def test_tls_with_files(self):
        with tempfile.NamedTemporaryFile(suffix=".pem", delete=False) as cert:
            cert.write(b"cert"); cert_path = cert.name
        with tempfile.NamedTemporaryFile(suffix=".pem", delete=False) as key:
            key.write(b"key"); key_path = key.name
        try:
            s = DistLLMSettings(model=ModelSettings(name="test"), tls=TLSSettings(enabled=True, cert_file=cert_path, key_file=key_path))
            s._validate_cross_field()
        finally:
            os.unlink(cert_path); os.unlink(key_path)

    def test_tp_bad_network(self):
        with pytest.raises(ValueError):
            DistLLMSettings(model=ModelSettings(name="test"), tensor_parallel=TensorParallelSettings(enabled=True, num_gpus=4), network=NetworkSettings(grpc_timeout=0, max_retries=0))

    def test_tp_single_gpu(self):
        s = DistLLMSettings(model=ModelSettings(name="test"), tensor_parallel=TensorParallelSettings(enabled=True, num_gpus=1))
        s._validate_cross_field()

    def test_tp_good_network(self):
        s = DistLLMSettings(model=ModelSettings(name="test"), tensor_parallel=TensorParallelSettings(enabled=True, num_gpus=1), network=NetworkSettings(grpc_timeout=10, max_retries=3))
        s._validate_cross_field()

    def test_full_valid(self):
        with tempfile.NamedTemporaryFile(suffix=".pem", delete=False) as cert:
            cert.write(b"c"); cert_path = cert.name
        with tempfile.NamedTemporaryFile(suffix=".pem", delete=False) as key:
            key.write(b"k"); key_path = key.name
        try:
            s = DistLLMSettings(model=ModelSettings(name="test", dtype="bfloat16"), vllm=VLLMSettings(enabled=True, dtype="bfloat16"), chunked_prefill=ChunkedPrefillSettings(enabled=True), batching=BatchingSettings(max_batch_size=16, max_tokens_per_batch=4096), tensor_parallel=TensorParallelSettings(enabled=True, num_gpus=1), network=NetworkSettings(grpc_timeout=30, max_retries=3), tls=TLSSettings(enabled=True, cert_file=cert_path, key_file=key_path), speculative=SpeculativeSettings(method="draft_model", draft_model="test-draft"))
            s._validate_cross_field()
        finally:
            os.unlink(cert_path); os.unlink(key_path)

    def test_from_profile_dev(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("model:\n  name: pt\n")
            path = f.name
        try:
            assert DistLLMSettings.from_profile(config_path=path, profile="dev").model.name == "pt"
        finally:
            os.unlink(path)
