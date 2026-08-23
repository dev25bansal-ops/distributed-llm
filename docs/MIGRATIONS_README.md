# DistLLM Migration Guides

Practical, drop-in migration guides for teams moving to DistLLM's OpenAI-compatible API.

| Guide | Audience | Core change |
|-------|----------|-------------|
| [MIGRATION_OPENAI.md](MIGRATION_OPENAI.md) | OpenAI API users | Point `base_url` at `http://localhost:8000/v1`; `api_key` optional |
| [MIGRATION_VLLM.md](MIGRATION_VLLM.md) | vLLM OpenAI-server users | Same `/v1` base URL swap; operational differences (distributed HW, model loading) |
| [MIGRATION_OLLAMA.md](MIGRATION_OLLAMA.md) | Ollama users | Swap `/v1` host/port or run the Ollama proxy; map tags → DistLLM model ids |

## Common facts across all three

- **DistLLM base URL:** `http://localhost:8000/v1`
- **API key:** optional. Use `--no-auth` for local dev (pass any non-empty string such as
  `"not-needed"`), or set `API_KEY` and send `Authorization: Bearer <key>` in production.
- **Supported OpenAI endpoints:** `POST /v1/chat/completions`, `POST /v1/completions`,
  `POST /v1/embeddings`, `GET /v1/models`.
- **Models must be loaded** into the cluster before use (`distllm model load <hf-id>` or
  `multi_model.models` in `config.yaml`). DistLLM model ids are HuggingFace ids or local paths, not
  OpenAI/Ollama names.
- **Verify any migration** with:

  ```bash
  curl http://localhost:8000/v1/models
  ```

## References

- [docs/adr/0004-openai-compatible-api.md](adr/0004-openai-compatible-api.md) — why the API is OpenAI-compatible
- [docs/QUICKSTART.md](QUICKSTART.md) — starting `distllm system api`
- [docs/api.md](api.md) — full endpoint reference (incl. DistLLM extensions)
- [docs/CONFIG_REFERENCE.md](CONFIG_REFERENCE.md) — complete config schema
- [docs/MODEL_COMPATIBILITY.md](MODEL_COMPATIBILITY.md) — supported models and backends
- `examples/` — SDK and framework integration samples (`sdk_example.py`, `langchain_example.py`, …)
