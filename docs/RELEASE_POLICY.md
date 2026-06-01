# Release Policy

## Versioning

DistLLM follows [Semantic Versioning](https://semver.org/):

- **MAJOR** (X.0.0): Breaking API changes, incompatible config changes
- **MINOR** (0.X.0): New features, backward-compatible changes
- **PATCH** (0.0.X): Bug fixes, security patches, documentation

## Release Cadence

| Release Type | Cadence | Typical Content |
|-------------|---------|-----------------|
| **Patch** | Weekly (Wednesdays) | Bug fixes, security patches, dependency updates |
| **Minor** | Monthly (first Monday) | New features, performance improvements, new backends |
| **Major** | As needed (quarterly max) | Breaking changes, architecture rewrites |

## Release Process

### Patch Release

1. Cherry-pick fixes from `main` to `release/vX.Y` branch
2. Run full test suite: `make test-all`
3. Tag: `git tag vX.Y.Z`
4. Push tag: `git push origin vX.Y.Z`
5. GitHub Actions builds Docker images, publishes to PyPI, creates GitHub Release

### Minor Release

1. Create `release/vX.Y` branch from `main`
2. Run full test suite + security scan + benchmark regression
3. Update CHANGELOG.md with all changes since last minor
4. Tag: `git tag vX.Y.0`
5. Create GitHub Release with release notes
6. Announce on Discussions, Twitter, Discord

### Major Release

1. Create RFC for breaking changes (minimum 2-week review period)
2. Create migration guide in `docs/MIGRATION_vX_to_vY.md`
3. Deprecation period: warn for at least 1 minor version before removing
4. Full security audit
5. Tag and release with comprehensive release notes

## Branch Strategy

```
main ───────────────────────────────────────────── (development)
  │
  ├── release/v0.4 ─────────────────────────────── (patch releases)
  │
  ├── release/v0.5 ─────────────────────────────── (next minor)
  │
  └── feat/feature-name ────────────────────────── (feature branches)
```

- `main`: Always deployable, all tests pass
- `release/vX.Y`: Receives cherry-picked fixes only
- Feature branches: Merge to `main` via PR with 1+ approvals

## What Goes in Each Release

### Patch (vX.Y.Z)
- Critical bug fixes
- Security vulnerability patches
- Dependency updates (minor/patch)
- Documentation corrections
- Performance regressions fixed

### Minor (vX.Y.0)
- New features
- New backend support
- New API endpoints
- Performance improvements
- New CLI commands
- Plugin API enhancements

### Major (X.0.0)
- Breaking API changes
- Config format changes
- Removal of deprecated features
- Architecture changes

## Deprecation Policy

1. **Announce**: Mark deprecated in CHANGELOG + runtime warning
2. **Warn**: At least 1 minor version with deprecation warning
3. **Remove**: Only in next major version

Example timeline:
- v0.4.0: Feature marked deprecated, warning emitted
- v0.5.0: Still available, warning louder
- v1.0.0: Feature removed

## Release Checklist

- [ ] All tests pass (`make test-all`)
- [ ] Security scan clean (`make security`)
- [ ] Benchmark regression check (`make bench-regression`)
- [ ] CHANGELOG.md updated
- [ ] Version bumped in `pyproject.toml`
- [ ] Migration guide written (if breaking changes)
- [ ] Docker images build successfully
- [ ] Helm chart version updated
- [ ] SDK version updated (if API changes)
- [ ] Release notes drafted
