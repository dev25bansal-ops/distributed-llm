# DistLLM Laptop + Colab Demo (v3 — true 2-node pipeline)

This demo uses **`bore.pub` to tunnel the gRPC port** so a free Google Colab notebook can join your laptop's DistLLM cluster as a real worker. The model is **sharded across both boxes** (laptop + Colab = 24 GB combined VRAM) and inferences cross the public internet.

## The architecture

```
+-------------------+         gRPC (bore)       +--------------------+
|  Google Colab     |  <------------------->  |  Your laptop       |
|  (free T4, 16 GB) |   port 50050             |  (RTX, 8 GB)       |
|                   |                          |                    |
|  * distllm.dist   |  cluster_key auth        |  * coordinator     |
|    .worker        |                          |  * REST API        |
|  * layers 16-31   |  <------------------->    |  * layers 0-15     |
|    of Llama-8B    |   /v1/chat/completions    |    of Llama-8B     |
|                   |     (optional, laptop)   |  * 5 GB VRAM used  |
+-------------------+                          +--------------------+
       ^                                               ^
       |                                               |
   real worker,                              coordinator + API server
   sharded model
```

## What this proves

1. **Real pipeline parallelism across the public internet** — Llama-3.1-8B's 32 layers split as `0-15` on the laptop, `16-31` on Colab.
2. **24 GB combined VRAM** holding one model across two physical boxes.
3. **bore.pub** tunnels the gRPC port (TCP) and prints the assigned port to stdout — the Colab worker can discover the laptop from a notebook.
4. **DISTLLM_CLUSTER_KEY** authenticates the worker (HMAC over the cluster gRPC).
5. **OpenAI-compatible API** still works at `http://localhost:8000/v1/chat/completions` on the laptop; just the inference crosses the tunnel.

## Prerequisites

On your laptop:

- Python 3.10+ with `distllm` installed.
- `bitsandbytes` for 4-bit quantization (`pip install bitsandbytes`).
- `bore` (TCP tunnel) on `PATH`:
  - **Windows**: download `bore.exe` from <https://github.com/ekzhang/bore/releases/latest> (pick `bore-v0.6.0-x86_64-pc-windows-msvc.zip`) and put it somewhere on PATH, e.g. `C:\Users\<you>\bin\`.
  - **macOS**: `brew install bore`
  - **Linux**: `cargo install bore-cli` or download the binary.
- A free HuggingFace account + token if you want gated models (Llama-3.1-8B).

In Google Colab:

- Just a free account. No installs in advance; the notebook installs DistLLM.

## Run order

### 1. Start the laptop coordinator + bore tunnel

```powershell
# PowerShell
$env:DISTLLM_DEMO_MODEL    = "meta-llama/Meta-Llama-3.1-8B-Instruct"
$env:DISTLLM_DEMO_API_KEY  = "demo-laptop-2026"
$env:HUGGING_FACE_HUB_TOKEN = "hf_your_token_here"   # gated model

python D:\distributed-llm\examples\distllm_laptop_coordinator.py
```

You will see (after ~10-30 s):

```
bore tunnel up: bore.pub:17128
----------------------------------------------------------------------
 Paste these into the Colab notebook (cell #2):
----------------------------------------------------------------------
  DISTLLM_PUBLIC_HOST = 'bore.pub'
  DISTLLM_PUBLIC_PORT = '17128'
  CLUSTER_KEY         = '<random>'
----------------------------------------------------------------------
coordinator PID ..., api PID ...
REST API ready: http://localhost:8000/v1/chat/completions
```

Leave the terminal running.

### 2. Open the Colab notebook

Upload `examples/notebooks/tinylm_colab_worker.ipynb` to Colab.

Pick **Runtime → Change runtime type → T4 GPU** (free), then save.

Run the cells in order:

- **Cell #1** — installs DistLLM + `transformers` + `torch`. Reports GPU + VRAM.
- **Cell #2** — paste the three values from the laptop terminal into the form.
- **Cell #3** — starts `distllm.dist.worker` on Colab, streaming logs. When you see `worker is ready; the cluster is now live`, the model is sharded and the cluster is fully operational.
- **Cell #4** — instructions for the next step (chat from the laptop terminal, since the REST API is local).

### 3. Verify the cluster

Back on the laptop terminal (while the Colab worker is running):

```powershell
curl -s http://localhost:8000/v1/chat/completions `
  -H "Content-Type: application/json" `
  -H "Authorization: Bearer demo-laptop-2026" `
  -d '{\"model\":\"meta-llama/Meta-Llama-3.1-8B-Instruct\",\"messages\":[{\"role\":\"user\",\"content\":\"Why is distributed inference useful?\"}],\"max_tokens\":200}'
```

You should get back a coherent, multi-sentence response from the **full** Llama-3.1-8B (with the second half physically running on the Colab T4).

To **prove** the model is really split, run the same prompt on the laptop and watch `/admin/v1/nodes` on the laptop show both nodes healthy and serving their layer halves:

```powershell
python -m distllm.cli.main cluster list-nodes --coordinator localhost --port 50050
```

You should see two rows: the laptop (laptop-side, layers 0-15) and `colab-t4` (Colab-side, layers 16-31).

## What model to use

| Model | Layers | 4-bit size | Laptop 8 GB | Colab 16 GB | Notes |
|---|---|---|---|---|---|
| **Llama-3.1-8B-Instruct** (default) | 32 | ~5 GB | ✅ (16 layers) | ✅ (16 layers) | Gated, best quality |
| **Mistral-7B-Instruct-v0.3** | 32 | ~5 GB | ✅ (16 layers) | ✅ (16 layers) | Un-gated Llama alternative |
| **Qwen2.5-7B-Instruct** | 32 | ~5 GB | ✅ (16 layers) | ✅ (16 layers) | Un-gated, top code/multilingual |
| **Phi-3-mini-4k-instruct (3.8B)** | 32 | ~2.5 GB | ✅ (16 layers) | ✅ (16 layers) | Tiny, fast |
| **Llama-3.2-3B-Instruct** | 28 | ~2.5 GB | ✅ (14 layers) | ✅ (14 layers) | Un-gated, very clean |
| **Qwen2.5-32B-Instruct (AWQ 4-bit)** | 64 | ~20 GB | ✅ (32 layers ≈ 10 GB) | ✅ (32 layers ≈ 10 GB) | Fits the 24 GB combined budget |
| **Llama-3-70B-Instruct (AWQ 4-bit)** | 80 | ~40 GB | ❌ | ❌ | Total exceeds combined 24 GB |

To use a different model, just override `DISTLLM_DEMO_MODEL`:

```powershell
$env:DISTLLM_DEMO_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
```

For 70B-class, you would need ~40 GB combined — add a second real GPU (e.g. desktop + laptop + Colab) and update the layer split in the notebook cell #2.

## Cleanup

- Colab: `Runtime → Interrupt execution` (the worker process is killed).
- Laptop: `Ctrl+C` in the coordinator terminal. The script terminates bore, the coordinator, and the REST API.

## What if bore.pub is down?

Set `BORE_SERVER=my-bore.example.com` in the env to point at a self-hosted bore server. The protocol is documented at <https://github.com/ekzhang/bore>.
