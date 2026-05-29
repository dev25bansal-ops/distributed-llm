# DistLLM for Visual Studio Code

[DistLLM](https://github.com/dev25bansal-ops/distributed-llm) is a distributed LLM inference system. This extension gives you real-time cluster health monitoring and inline LLM inference directly from your editor.

## Features

### 🟢 Cluster Health in Status Bar
Three status bar items show real-time cluster state:
- **Model name** — click to open the DistLLM dashboard
- **Node health** — healthy/total node count
- **Throughput** — tokens/second with color coding (green >50, yellow >20, red <20)

### 📝 Send Selection to DistLLM
Right-click any selected text in the editor → "Send to DistLLM" to run inference on your cluster. Results appear in the output panel.

### ⚙️ Configurable
| Setting | Default | Description |
|---------|---------|-------------|
| `distllm.apiUrl` | `http://localhost:8000` | DistLLM API server URL |
| `distllm.model` | Auto-detect | Model name override |
| `distllm.refreshInterval` | 10s | Status bar refresh interval |
| `distllm.maxTokens` | 256 | Default max generation tokens |
| `distllm.temperature` | 0.7 | Default sampling temperature |

## Requirements

- A running DistLLM API server (local or remote)
- VS Code 1.85+

## Quick Start

1. Install the extension
2. Start your DistLLM cluster: `distllm cluster start`
3. The status bar updates automatically with cluster health
4. Select code in the editor, right-click, choose "Send to DistLLM"

## Commands

| Command | Title |
|---------|-------|
| `distllm.sendSelection` | Send to DistLLM |
| `distllm.openDashboard` | Open DistLLM Dashboard |
| `distllm.refreshStatus` | Refresh DistLLM Status |

## Extension Settings

This extension contributes the following settings:
- `distllm.apiUrl`: DistLLM API endpoint URL
- `distllm.model`: Model name (empty = auto-detect)
- `distllm.refreshInterval`: Status bar refresh interval in seconds
- `distllm.maxTokens`: Default max tokens for "Send to DistLLM"
- `distllm.temperature`: Default temperature for generation

## Known Issues

- API server must be running before extension can connect
- No embedded webview dashboard (opens in external browser)

## Release Notes

See [CHANGELOG.md](CHANGELOG.md) for version history.

---

**Apache-2.0 License**
