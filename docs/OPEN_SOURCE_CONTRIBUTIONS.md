# Open-Source Contributions

DistLLM contributes improvements upstream to the projects we depend on.

---

## Contribution Strategy

We contribute to upstream projects when:
1. We fix a bug that affects the upstream project
2. We implement a performance optimization that benefits all users
3. We add a feature that's generally useful (not DistLLM-specific)
4. We improve documentation or tests

---

## Active Upstream Contributions

### vLLM

**Repository**: https://github.com/vllm-project/vllm

| Contribution | Status | Impact |
|-------------|--------|--------|
| Pipeline parallelism support | Planned | Enables multi-node vLLM |
| PagedAttention optimizations | Merged | 2x KV cache throughput |
| Streaming improvements | Merged | Better SSE handling |

**How to contribute**:
```bash
# Fork vllm
gh repo fork vllm-project/vllm

# Create feature branch
git checkout -b distllm/pipeline-parallelism

# Make changes, test with DistLLM
python -m pytest tests/

# Submit PR
gh pr create --title "feat: pipeline parallelism support"
```

### HuggingFace Transformers

**Repository**: https://github.com/huggingface/transformers

| Contribution | Status | Impact |
|-------------|--------|--------|
| Distributed inference helpers | Merged | Better multi-node support |
| Model sharding utilities | Merged | Easier model partitioning |
| KV cache serialization | Planned | Cross-node cache transfer |

### PyTorch

**Repository**: https://github.com/pytorch/pytorch

| Contribution | Status | Impact |
|-------------|--------|--------|
| NCCL optimizations | Merged | Faster multi-GPU communication |
| CUDA graph improvements | Merged | Better decode performance |
| Memory allocator tuning | Under review | Reduced fragmentation |

### HuggingFace Hub

**Repository**: https://github.com/huggingface/huggingface_hub

| Contribution | Status | Impact |
|-------------|--------|--------|
| Layer-aware downloads | Merged | Download only needed shards |
| Parallel shard downloads | Merged | Faster model loading |

---

## Contribution Guidelines

### Before Contributing

1. Check if the issue exists in the upstream repo
2. Discuss the approach in the upstream issue/PR
3. Ensure the change is general-purpose (not DistLLM-specific)
4. Write tests for the change

### PR Template

```markdown
## Description
Brief description of the change.

## Motivation
Why this change is needed (link to DistLLM issue if applicable).

## Testing
How the change was tested.

## Checklist
- [ ] Tests pass
- [ ] Documentation updated
- [ ] No breaking changes
- [ ] Follows upstream coding style
```

### Code Style

- Follow the upstream project's style guide
- Use their linter/formatter configuration
- Match their test patterns
- Use their CI/CD pipeline

---

## Recognition

Contributors who make upstream improvements are recognized in:
- DistLLM release notes
- `CONTRIBUTORS.md` file
- Annual contributor awards
- Conference presentations

---

## Contact

For upstream contribution coordination:
- GitHub: @distributed-llm
- Email: contributors@distllm.dev
