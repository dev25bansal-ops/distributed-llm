#!/usr/bin/env bash
# DistLLM — One-line install script
# Usage: curl -sfL https://distllm.ai/install.sh | sh
# Or: curl -sfL https://raw.githubusercontent.com/distributed-llm/distributed-llm/main/scripts/install.sh | sh

set -e

BOLD="\033[1m"
GREEN="\033[0;32m"
YELLOW="\033[0;33m"
CYAN="\033[0;36m"
NC="\033[0m"

echo -e "${BOLD}${CYAN}DistLLM — Distributed LLM Inference${NC}"
echo -e "${CYAN}Pool GPUs across all your devices to run models no single machine can handle${NC}"
echo ""

# ── Detect OS ──────────────────────────────────────────────────────────
OS="$(uname -s)"
ARCH="$(uname -m)"
case "$OS" in
    Linux)   PLATFORM="linux" ;;
    Darwin)  PLATFORM="macos" ;;
    MINGW*|MSYS*) PLATFORM="windows" ;;
    *)       echo -e "${YELLOW}Unknown OS: $OS. Attempting pip install anyway...${NC}" ;;
esac
echo -e "  ${BOLD}OS:${NC} $OS  ${BOLD}Arch:${NC} $ARCH  ${BOLD}Platform:${NC} $PLATFORM"

# ── Check Python ───────────────────────────────────────────────────────
PYTHON=""
for cmd in python3 python; do
    if command -v "$cmd" >/dev/null 2>&1; then
        PYTHON="$cmd"
        break
    fi
done

if [ -z "$PYTHON" ]; then
    echo -e "${YELLOW}Python not found. Please install Python 3.10+ from https://python.org${NC}"
    exit 1
fi

PY_VER="$($PYTHON --version 2>&1 | grep -oP '\d+\.\d+')"
PY_MAJOR="${PY_VER%%.*}"
PY_MINOR="${PY_VER#*.}"

if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 10 ]; }; then
    echo -e "${YELLOW}Python 3.10+ required. Found: $PYTHON ($PY_VER)${NC}"
    exit 1
fi
echo -e "  ${BOLD}Python:${NC} $PYTHON ($PY_VER)"

# ── Check CUDA (optional, for NVIDIA GPUs) ────────────────────────────
CUDA_AVAILABLE=0
if command -v nvidia-smi >/dev/null 2>&1; then
    CUDA_AVAILABLE=1
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    echo -e "  ${BOLD}GPU:${NC} $GPU_NAME"
fi

# ── Install ─────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}Installing DistLLM...${NC}"

INSTALL_DIR="${DISTLLM_DIR:-$HOME/.distllm}"
mkdir -p "$INSTALL_DIR"

if [ -d "$INSTALL_DIR/distributed-llm" ]; then
    echo -e "  Updating existing installation..."
    cd "$INSTALL_DIR/distributed-llm"
    git pull --ff-only 2>/dev/null || true
else
    echo -e "  Cloning repository..."
    git clone --depth 1 https://github.com/distributed-llm/distributed-llm.git "$INSTALL_DIR/distributed-llm"
    cd "$INSTALL_DIR/distributed-llm"
fi

# Install core package
echo -e "  Installing Python package (core)..."
$PYTHON -m pip install --quiet -e . 2>/dev/null

if [ "$CUDA_AVAILABLE" = "1" ]; then
    echo -e "  Installing with CUDA support..."
    $PYTHON -m pip install --quiet torch --index-url https://download.pytorch.org/whl/cu128 2>/dev/null || true
    $PYTHON -m pip install --quiet -e ".[vllm]" 2>/dev/null || true
fi

# ── Path setup ──────────────────────────────────────────────────────────
SHELL_NAME="$(basename "$SHELL" 2>/dev/null || echo 'bash')"
PROFILE_FILE="$HOME/.${SHELL_NAME}rc"
if [ "$SHELL_NAME" = "zsh" ]; then
    PROFILE_FILE="$HOME/.zshrc"
fi

INSTALL_BIN="$INSTALL_DIR/bin"
mkdir -p "$INSTALL_BIN"

# Create wrapper scripts
cat > "$INSTALL_BIN/distllm" << 'SCRIPT'
#!/usr/bin/env bash
exec python -m distllm.cli.main "$@"
SCRIPT
chmod +x "$INSTALL_BIN/distllm"

for CMD in distllm-coordinator distllm-node distllm-api; do
    cat > "$INSTALL_BIN/$CMD" << SCRIPT
#!/usr/bin/env bash
exec python -m distllm.core.coordinator "\$@"
SCRIPT
    chmod +x "$INSTALL_BIN/$CMD"
done

# Fix the node wrapper
cat > "$INSTALL_BIN/distllm-node" << 'SCRIPT'
#!/usr/bin/env bash
exec python -m distllm.dist.worker "$@"
SCRIPT
chmod +x "$INSTALL_BIN/distllm-node"

cat > "$INSTALL_BIN/distllm-api" << 'SCRIPT'
#!/usr/bin/env bash
exec python -m distllm.api.server "$@"
SCRIPT
chmod +x "$INSTALL_BIN/distllm-api"

# Add to PATH if not already there
case ":$PATH:" in
    *:$INSTALL_BIN:*) ;;
    *)
        echo ""
        echo -e "  ${YELLOW}Add to your PATH:${NC}"
        echo -e "  export PATH=\"\$PATH:$INSTALL_BIN\""
        echo ""
        echo -e "  Or run: echo 'export PATH=\"\$PATH:$INSTALL_BIN\"' >> $PROFILE_FILE"
        ;;
esac

# ── Done ────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}${BOLD}✓ DistLLM installed!${NC}"
echo ""
echo -e "  ${BOLD}Quick start (local mode):${NC}"
echo -e "  $ distllm-coordinator --model HuggingFaceTB/SmolLM-135M --local --chat"
echo ""
echo -e "  ${BOLD}Cluster mode (multi-device):${NC}"
echo -e "  Machine 1: $ ${CYAN}distllm-coordinator --model meta-llama/Llama-3.2-1B --port 50050${NC}"
echo -e "  Machine 2: $ ${CYAN}distllm-node --node-id worker0 --model meta-llama/Llama-3.2-1B \\"
echo -e "                        --start-layer 0 --end-layer 5 --total-layers 12 \\"
echo -e "                        --coordinator-host <IP_OF_MACHINE_1> --port 50051${NC}"
echo ""
echo -e "  ${BOLD}API server:${NC}"
echo -e "  $ distllm-api --model HuggingFaceTB/SmolLM-135M --local --port 8000"
echo ""
echo -e "  ${BOLD}Documentation:${NC} https://github.com/distributed-llm/distributed-llm"
echo -e "  ${BOLD}Need help?${NC} https://github.com/distributed-llm/distributed-llm/issues"
