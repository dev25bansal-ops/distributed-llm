# Contributing to Distributed LLM

Thank you for your interest in contributing! This project aims to make distributed LLM inference accessible, fast, and easy to deploy.

## Ways to Contribute

### Bug Reports
- Search existing issues first
- Include: OS, Python version, GPU model, model name, error logs
- Provide a minimal reproduction case if possible

### Feature Requests
- Explain the use case and why it matters
- Consider whether it fits the project's scope (distributed inference)
- Discuss implementation approach if you have ideas

### Code Contributions

#### Setup Development Environment
```bash
git clone https://github.com/distributed-llm/distributed-llm.git
cd distributed-llm
pip install -e ".[dev]"
```

#### Code Style
- **Black** for formatting (line length: 100)
- **Ruff** for linting
- **Type hints** for all new code
- Follow existing patterns in the codebase

#### Running Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=term-missing

# Run specific test file
pytest tests/test_coordinator.py
```

#### Before Submitting a PR
1. Run the full test suite: `pytest`
2. Run the linter: `ruff check src/`
3. Format code: `black src/`
4. Test locally with at least one model (TinyStories-1M is fastest)
5. Update CHANGELOG.md with your changes

### Adding Model Architecture Support

To add support for a new model architecture:

1. Update `ModelPartitioner._extract_subset()` to handle the model's layer structure
2. Add rotary embedding support if the model uses RoPE
3. Test with `test.py` in both local and distributed modes
4. Add the model to the compatibility matrix in README.md

### Performance Improvements

Performance work is highly valued. When submitting optimizations:

1. Include benchmark results (before/after)
2. Use the `benchmarks/` directory for your benchmark script
3. Specify hardware used for testing (GPU model, network type)

## Architecture Overview

```
Coordinator (orchestrates)
    |
    | gRPC (protobuf)
    |
    v
Node 0 -> Node 1 -> ... -> Node N
(layers 0-M)  (layers M+1-N)
```

Key components:
- `src/models.py` - Model loading, layer partitioning, forward pass
- `src/coordinator.py` - Inference orchestration, token sampling
- `src/node.py` - Worker node gRPC server
- `src/communication.py` - gRPC layer, tensor serialization
- `src/kv_cache.py` - KV cache management
- `src/api.py` - OpenAI-compatible REST API
- `proto/node.proto` - Protobuf service definitions

## First-Time Contributor Guide

Good starting issues:
- Issues labeled `good first issue`
- Adding support for new model architectures
- Writing tests
- Performance benchmarks
- Documentation improvements

## Development Workflow

1. Fork the repository
2. Create a feature branch: `git checkout -b feat/your-feature`
3. Make your changes
4. Run tests and linting
5. Commit with descriptive messages: `feat: add Llama-3 support`
6. Push and open a PR

### Commit Convention
- `feat:` new features
- `fix:` bug fixes
- `perf:` performance improvements
- `docs:` documentation changes
- `test:` test additions/changes
- `refactor:` code refactoring
- `ci:` CI/CD changes

## Questions?

Open a GitHub Discussion or ask in an issue. We're happy to help!
