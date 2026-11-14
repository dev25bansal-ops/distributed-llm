# Fix brief: docs

Audit findings for the files in your exclusive scope. For EACH finding:
reproduce-or-refute first (run the check command or read the cited code
carefully), then either FIX it or record an explicit refutation. Never
leave a finding unaddressed. After fixing, extend or add a test in the
named test file and run it. Do NOT commit; leave changes in the tree.

## [high] docs/BENCHMARKS.md:127
**Summary**: The flagship 92.0/153.1 tok/s figures are labeled 'measured 2026-06-26' but no in-repo artifact or code path produces them, and the documented reproduction command runs an analytical estimator that prints 99.1 tok/s instead of 153.1.
**Failure scenario**: BUILDATHON-APPLICATION.md (lines 114-117, 316-317) tells the jury these figures are 'recorded in the repository with their measurement date and hardware' and 'ones we can defend and re-run'. Running BENCHMARKS.md's own Reproduction command (`python -m benchmarks.cluster_benchmark --models llama8b --clusters 2 --prompt-len 512 --max-tokens 128`, line 77) prints 'Running analytical estimates (use --live for real hardware)' and reports 99.1 tok/s for 2 nodes (1 GbE) — not 153.1. The only appearance of the 92.0 number in any script is `single_gpu_tok_s=92.0` in benchmarks/cluster_benchmark.py:76, hardcoded as an input to that roofline estimator and commented 'empirical tok/s on 1x H100' (line 51) — contradicting the RTX 4090 attribution in BENCHMARKS.md. No raw measurement JSON, no CI run, and no committed artifact exists: benchmarks/results/ is gitignored, GitHub Actions has zero June-2026 artifacts, and only CI Dashboard/Chaos ran on 2026-06-26. The 153.1 row's 'Run ID' is literally '—' (line 218-219). If a jury engineer runs the documented command, they get a different number in estimation mode — exactly the check the application invites.
**Check command**: `cd "D:\distributed-llm\.claude\worktrees\bubbly-pondering-whisper" && python -m benchmarks.cluster_benchmark --models llama8b --clusters 2 --prompt-len 512 --max-tokens 128  (prints 99.1 tok/s, labeled analytical estimates)`
**Reproduced by auditor**: True

## [med] docs/OPS_RUNBOOK.md:41
**Summary**: OPS_RUNBOOK documents `distllm backup list/restore/create` but no top-level `backup` command exists (it is `config backup`), and `config backup create` has no `--type` flag.
**Failure scenario**: An operator following the recovery runbook ('Scenario: config change causes performance degradation' at line 35, and 'Upgrading DistLLM Version' at line 165) types `distllm backup list` and gets exit code 2 with 'No such command backup.'; `distllm backup create --type full` similarly fails, and the real command `distllm config backup create` rejects `--type` ('No such option: --type'). Unlike SCALING.md/PLUGIN_DEVELOPMENT.md, OPS_RUNBOOK.md carries zero hygiene 'UNVERIFIED' annotations, so these read as working instructions. Backup steps that silently fail during an incident are exactly when they matter.
**Check command**: `cd "D:\distributed-llm\.claude\worktrees\bubbly-pondering-whisper" && PYTHONPATH=src python -m distllm.cli.main backup list  (exit 2, 'No such command backup'); PYTHONPATH=src python -m distllm.cli.main config backup create --type full  (exit 2, 'No such option: --type')`
**Reproduced by auditor**: True

## [med] docs/QUICKSTART.md:173
**Summary**: QUICKSTART documents `distllm system run --coordinator-host your-server.com` but `system run` has no --coordinator-host option.
**Failure scenario**: A user setting up WAN mode per QUICKSTART lines 169-173 runs the documented worker command and gets exit code 2 with 'No such option: --coordinator-host'. The flag exists only on the separate `distllm-node` console script (distllm.dist.worker:main). SCALING.md repeats the same wrong command but has a hygiene banner correcting it (SCALING.md:27); QUICKSTART.md has no such annotation, so the first-run document presents a command that cannot run.
**Check command**: `cd "D:\distributed-llm\.claude\worktrees\bubbly-pondering-whisper" && PYTHONPATH=src python -m distllm.cli.main system run --coordinator-host x  (exit 2, 'No such option: --coordinator-host')`
**Reproduced by auditor**: True

## [med] docs/TROUBLESHOOTING.md:516
**Summary**: TROUBLESHOOTING documents `distllm system logs --node node_1 --tail 50` and `system logs --tail 100`, but the `system logs` command accepts no --node or --tail options.
**Failure scenario**: During the circuit-breaker incident procedure (line 516) and general log inspection (line 1647), the documented command fails with exit code 2 and 'No such option: --node' (and --tail does not exist either; `system logs stream` offers --lines/-n, --level, --component, --search, --follow). The user following the troubleshooting steps to diagnose a failing node hits a CLI error instead of the intended logs.
**Check command**: `cd "D:\distributed-llm\.claude\worktrees\bubbly-pondering-whisper" && PYTHONPATH=src python -m distllm.cli.main system logs --node node_1 --tail 50  (exit 2, 'No such option: --node')`
**Reproduced by auditor**: True

## [med] docs/benchmark-results.md:281
**Summary**: The reproduction section (and four other places in the file) documents `python benchmarks/kv_reuse_bench.py`, but that script has never existed in any commit or checkout of the repo.
**Failure scenario**: benchmark-results.md presents measured kv-reuse numbers ('engine_stream_default 97.2 -> 174.5 tok/s', 'Artifacts: benchmarks/results/kv-reuse-aligned-*.json') and tells the reader to reproduce them with `python benchmarks/kv_reuse_bench.py --device cpu|--device cuda` (lines 152, 231, 281-282). The file does not exist: `ls benchmarks/kv_reuse_bench.py` fails and `git log --all -- benchmarks/kv_reuse_bench.py` is empty — it was never committed, so the documented measurements cannot be re-run even though the result JSONs sit in the (gitignored) benchmarks/results/ of the main checkout. This weakens the 'numbers that remain are ones we can defend and re-run' claim the Buildathon application leans on.
**Check command**: `cd "D:\distributed-llm\.claude\worktrees\bubbly-pondering-whisper" && ls benchmarks/kv_reuse_bench.py; git log --all --oneline -- benchmarks/kv_reuse_bench.py  (both empty/missing)`
**Reproduced by auditor**: True

## [low] README.md:174
**Summary**: README roadmap states 'v0.4.x (current)' while the repository's actual current version is 0.7.0.
**Failure scenario**: README.md:174 labels the roadmap bullet 'v0.4.x (current)', but pyproject.toml declares version = '0.7.0', CHANGELOG.md's newest section is [0.7.0] - 2026-11-12, and dist/ contains distributed_llm-0.7.0 wheels; PyPI serves 0.6.2.2 as the published version. Anyone checking the front page against the shipped artifacts sees a version claim three releases stale.
**Check command**: `cd "D:\distributed-llm\.claude\worktrees\bubbly-pondering-whisper" && grep -n 'version =' pyproject.toml; grep -n 'v0.4.x' README.md`
**Reproduced by auditor**: True
