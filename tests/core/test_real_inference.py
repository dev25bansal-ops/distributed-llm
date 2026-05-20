"""Real model inference test.

Downloads a tiny model and runs a real forward pass through
the pipeline to verify inference actually works end-to-end.

Run: pytest tests/core/test_real_inference.py -v --timeout=120
"""

import pytest
import torch


def _get_small_model_name() -> str:
    """Return the smallest available model for testing.

    Uses SmolLM-135M which is ~270MB and fast to download.
    """
    return "HuggingFaceTB/SmolLM-135M"


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires GPU for real inference")
class TestRealInference:
    """Real model inference tests using a tiny downloaded model."""

    @pytest.fixture(scope="class")
    def model_and_tokenizer(self):
        """Download model once per class."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model_name = _get_small_model_name()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        model.eval()
        return model, tokenizer, device

    def test_model_loaded(self, model_and_tokenizer):
        """Verify model loads without errors."""
        model, tokenizer, device = model_and_tokenizer
        assert model is not None
        assert tokenizer is not None
        assert next(model.parameters()).device.type == device

    def test_single_token_forward(self, model_and_tokenizer):
        """Run a single forward pass and verify output shape."""
        model, tokenizer, _ = model_and_tokenizer
        inputs = tokenizer("Hello, my name is", return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(**inputs)
        assert outputs.logits is not None
        assert outputs.logits.shape[0] == 1
        assert outputs.logits.shape[1] == inputs.input_ids.shape[1]
        last_logits = outputs.logits[:, -1, :]
        next_token_id = last_logits.argmax(dim=-1)
        assert next_token_id.shape == (1,)
        next_token = tokenizer.decode(next_token_id[0])
        assert isinstance(next_token, str)
        assert len(next_token) > 0

    def test_generate_text(self, model_and_tokenizer):
        """Generate a short completion and verify it produces text."""
        model, tokenizer, _ = model_and_tokenizer
        prompt = "The capital of France is"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        assert len(generated) > len(prompt)
        assert isinstance(generated, str)

    def test_different_precisions_match(self, model_and_tokenizer):
        """FP16 and FP32 outputs should be close within tolerance."""
        model, tokenizer, device = model_and_tokenizer
        prompt = "Machine learning is"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output_fp16 = model(**inputs).logits[:, -1, :]

        model_fp32 = model.float()
        with torch.no_grad():
            output_fp32 = model_fp32(**inputs).logits[:, -1, :]

        cos = torch.nn.CosineSimilarity(dim=-1)
        similarity = cos(output_fp16.float(), output_fp32).item()
        assert similarity > 0.95, f"FP16/FP32 similarity {similarity} < 0.95"
