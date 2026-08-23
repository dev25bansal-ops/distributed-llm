# DistLLM Desktop App (Tauri + Svelte)

A cross-platform desktop application for monitoring and interacting with
your DistLLM cluster, built with [Tauri 2](https://v2.tauri.app/)
and [Svelte 5](https://svelte.dev/).

## Prerequisites

- **Node.js** 18+ (with npm)
- **Rust** toolchain (stable, via rustup)
- **Tauri CLI**: `cargo install tauri-cli --version "^2.0"`
- **System dependencies** (Windows):
  - Microsoft Visual Studio C++ Build Tools
  - WebView2 (included with Windows 10+)

## Quick Start

```bash
# 1. Install JS dependencies
cd tauri
npm install

# 2. Run in development mode (hot-reload)
npm run tauri dev

# 3. Build for production
npm run tauri build
# Output: tauri/src-tauri/target/release/distllm-app.exe
```

## Configuration

The app connects to a DistLLM API server. Configure the connection
before launching:

| Setting | Default | Env Var | Description |
|---------|---------|---------|-------------|
| API URL | `http://localhost:8000` | `DISTLLM_API_URL` | Coordinator API endpoint |
| API Key | — | `DISTLLM_API_KEY` | Authentication key |
| Refresh interval | 5s | — | Dashboard refresh rate |

### Via `tauri/src-tauri/tauri.conf.json`:

```json
{
  "plugins": {
    "shell": {
      "open": true
    }
  }
}
```

## Features

- **Dashboard**: Real-time cluster health (nodes, throughput, latency)
- **Chat**: Interactive chat with any loaded model
- **Model Management**: Load/unload models, view registry
- **Cluster Map**: Visual node topology with GPU utilization
- **Logs**: Stream coordinator and worker logs in real-time

## Building Installers

```bash
# Windows MSI installer
npm run tauri build -- --bundles msi

# Portable executable
npm run tauri build -- --bundles portable
```

Output path: `tauri/src-tauri/target/release/bundle/`

## Development

```bash
# Frontend only (browser-based, no Tauri)
npm run dev
# -> http://localhost:5173

# Full Tauri app with hot-reload
npm run tauri dev
```

## Project Structure

```
tauri/
├── src/                  # Svelte frontend
│   ├── App.svelte        # Root component
│   ├── lib/              # Shared components
│   │   ├── Dashboard.svelte
│   │   ├── Chat.svelte
│   │   └── ...
│   └── main.ts           # Entry point
├── src-tauri/            # Rust backend
│   ├── src/main.rs       # Tauri app entry
│   ├── Cargo.toml        # Rust dependencies
│   └── tauri.conf.json   # Tauri configuration
├── package.json
├── vite.config.ts
└── svelte.config.js
```

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `WebView2` not found | Install from Microsoft: `aka.ms/webview2installer` |
| Rust build fails | Run `rustup update stable` |
| Tauri CLI not found | Run `cargo install tauri-cli --version "^2.0"` |
| API connection refused | Start a coordinator: `distllm cluster start` |
