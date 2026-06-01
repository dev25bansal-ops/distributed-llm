# ADR-0004: OpenAI-Compatible API Design

**Date:** 2024-04-05
**Status:** Accepted
**Deciders:** Core team

## Context

Users want to switch between LLM providers without changing their application code. The OpenAI API has become a de facto standard.

## Decision

We implemented a **fully OpenAI-compatible API**:

1. **Endpoint Compatibility**: Same paths as OpenAI
   - `POST /v1/chat/completions`
   - `POST /v1/completions`
   - `POST /v1/embeddings`
   - `GET /v1/models`

2. **Request/Response Format**: Identical to OpenAI
   - Same JSON schema for requests
   - Same response structure with choices, usage, etc.
   - Streaming via SSE with same format

3. **Drop-in SDK Replacement**: Users can switch by changing base URL
   ```python
   # Before
   client = openai.OpenAI(api_key="sk-...")
   
   # After
   client = openai.OpenAI(base_url="http://localhost:8000/v1", api_key="distllm-...")
   ```

4. **Versioned API**: v1 (stable), v2 (latest) with proper deprecation

## Consequences

### Positive
- Zero-friction adoption for existing OpenAI users
- Compatible with LangChain, LlamaIndex, CrewAI, etc.
- Large ecosystem of OpenAI-compatible tools

### Negative
- Some OpenAI features don't apply to distributed inference
- Must maintain backward compatibility
- Cannot add custom fields without breaking compatibility

### Mitigations
- Custom fields in separate headers (X-DistLLM-*)
- Extensions API for non-standard features
- Clear documentation of supported/unsupported features

## Related ADRs
- ADR-0001: Pipeline Parallelism
