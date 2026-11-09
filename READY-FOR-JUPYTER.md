# Ready for the Jupyter A10 test

Date: 2026-11-09
Installed CLI: **distributed-llm 0.6.2.2** (user site-packages)
Wheel built and validated: `dist/distributed_llm-0.6.2.2-py3-none-any.whl`

---

## What is done

**All 23 defects from the live test are fixed and verified.**

| Fix | Verified how |
|---|---|
| `/api/cost/history` 500 | live: 200 |
| `/api/pipeline/health` 500 | live: 200 |
| `/v1/scheduler/config` 500 (and PATCH writing a dead attribute) | live: 200 |
| `/ready` 500 | live: 200 |
| Leaderboard `/summary`, `/top`, `/community`, `/verified` shadowed | live: 200 × 4 |
| `warmup` / registry-detail unreachable for `org/name` model ids | live: 200 with the real HF id |
| `model load` / `model unload` path drift | live: "Model loaded" |
| `system logs stream` path drift | live: reached `/admin/v1/logs` |
| `benchmark eval` path drift | path corrected, unit-tested |
| `cluster drain` → real admin route | path corrected, unit-tested |
| `cluster deploy` → `/api/models/load` | path corrected, unit-tested |
| `cluster scale` / `rebalance` / `model adapters` | now exit 2 with a clear message |
| 10 commands that never authenticated | `status` now reports `models=1` (was 0) |
| `config webhook test` claimed delivery, delivered nothing | **real receiver got the payload** |
| `benchmark live` fabricated metrics | every frame now carries `source="synthetic"` |

Tests: **1433 API+SDK passed**, **382 CLI passed**, 0 failures.

---

## The one thing left: remove the impostor package

`distllm` on PATH is **not this project**. A PyPI package named `distllm` v1.0.2
installs into `C:\Python314\Lib\site-packages`, and its `distllm.exe` lives in
`C:\Python314\Scripts`, which comes **before** the real CLI's directory on PATH:

```
C:\Python314\Scripts                              <- impostor, wins
C:\Users\dev25\AppData\Roaming\Python\Python314\Scripts   <- real 0.6.2.2
```

Result: `distllm --version` → `TypeError: 'module' object is not callable`.

Removing it needs Administrator. Run `scripts/remove-impostor-distllm.bat` as
Administrator (right-click → Run as administrator). It removes exactly:
- `C:\Python314\Lib\site-packages\distllm\`
- `C:\Python314\Lib\site-packages\distllm-1.0.2.dist-info\`
- the four `distllm*.exe` launchers in `C:\Python314\Scripts\`

It does not touch the real install.

**Until that is run**, use the full path or the module form:

```bash
"C:/Users/dev25/AppData/Roaming/Python/Python314/Scripts/distllm.exe" --version
```

```bash
python -m distllm.cli.main --version
```

---

## Publishing 0.6.2.2 to PyPI

The wheel is built and `twine check` passes, but publishing needs your token —
none is stored on this machine. Once you have it:

```bash
cd "D:/distributed-llm/.claude/worktrees/bubbly-pondering-whisper"
python -m twine upload dist/distributed_llm-0.6.2.2-py3-none-any.whl
```

You do **not** need this for the Jupyter test — 0.6.2.2 is already installed
locally, and the wheel can be copied to the Jupyter machine directly:

```bash
pip install dist/distributed_llm-0.6.2.2-py3-none-any.whl
```

---

## For the Jupyter A10 test

Your laptop: RTX 5060 Laptop, 8 GB VRAM.
Jupyter DSW: A10, 24 GB VRAM.

```bash
# On the Jupyter machine
pip install /path/to/distributed_llm-0.6.2.2-py3-none-any.whl

# Find a model that fits 24 GB
distllm model search "qwen3 7b"

# Start the API
distllm system run --model Qwen/Qwen3-7B-Instruct --port 8000 --host 0.0.0.0

# On the laptop, join as a worker or just point at the Jupyter API
distllm cluster join --coordinator <jupyter-ip>:50050
```

The OpenAI-compatible surface is verified working end to end (chat blocking and
streaming, completions, embeddings, batch, files, fine-tuning), so any OpenAI
client works against `http://<jupyter-ip>:8000/v1` with only a `base_url` change.

---

## Not done

- **Impostor package removal** — needs Administrator (script is ready).
- **PyPI publish** — needs your token.
- `benchmark live` is still synthetic; it is now labelled instead of silently faked.
- The PR's 14 pre-existing CI failures (terraform, k8s, vscode, some integrations)
  are unrelated to these changes.
