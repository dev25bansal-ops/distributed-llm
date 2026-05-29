from unittest.mock import MagicMock
import pytest


@pytest.fixture
def mock_model():
    model = MagicMock()
    model.config = MagicMock()
    model.config.num_hidden_layers = 32
    model.config.hidden_size = 4096
    model.config.num_attention_heads = 32
    model.config.vocab_size = 32000
    model.device = "cpu"
    return model


@pytest.fixture
def mock_tokenizer_backend():
    tokenizer = MagicMock()
    tokenizer.vocab_size = 32000
    tokenizer.eos_token_id = 2
    tokenizer.pad_token_id = 0
    tokenizer.encode.return_value = [1, 2, 3]
    tokenizer.decode.return_value = "hello world"
    return tokenizer
