#!/bin/sh
# ============================================================================
# DistLLM — One-Command Install Script
#   curl -sfL https://distllm.ai/install.sh | sh
#
# Detects CUDA version, installs compatible PyTorch, downloads model,
# and starts coordinator + 2 worker nodes via Docker Compose.
# ============================================================================
set -e

# ---- helpers ---------------------------------------------------------------

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

info()  { printf "${BLUE} INFO${NC}  %s\n" "$*"; }
ok()    { printf "${GREEN}  OK${NC}  %s\n" "$*"; }
warn()  { printf "${YELLOW} WARN${NC}  %s\n" "$*"; }
fail()  { printf "${RED}FAIL${NC}  %s\n" "$*"; exit 1; }

REPO="https://github.com/dev25bansal-ops/distributed-llm.git"
DISTLLM_DIR="${DISTLLM_DIR:-$HOME/.distllm}"
MODEL="${DISTLLM_MODEL:-roneneldan/TinyStories-1M}"
COMPOSE_PROJECT="distllm"

# ---- prerequisites ---------------------------------------------------------

info "Checking prerequisites..."

for cmd in curl git docker; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    fail "$cmd is required but not installed. Install it and re-run."
  fi
done
ok "curl, git, docker found"

if ! docker info >/dev/null 2>&1; then
  fail "Docker daemon is not running. Start Docker and re-run."
fi
ok "Docker daemon running"

# ---- CUDA detection --------------------------------------------------------

CUDA_MAJOR=""
CUDA_MINOR=""

if command -v nvidia-smi >/dev/null 2>&1; then
  CUDA_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
  CUDA_MAJOR=$(echo "$CUDA_VERSION" | cut -d. -f1)
  CUDA_MINOR=$(echo "$CUDA_VERSION" | cut -d. -f2)
  info "NVIDIA driver detected: $CUDA_VERSION"
else
  # Fallback: check for nvcc
  if command -v nvcc >/dev/null 2>&1; then
    CUDA_MAJOR=$(nvcc --version | grep "release" | sed 's/.*release //' | sed 's/,.*//' | cut -d. -f1)
    CUDA_MINOR=$(nvcc --version | grep "release" | sed 's/.*release //' | sed 's/,.*//' | cut -d. -f2)
  fi
fi

if [ -z "$CUDA_MAJOR" ]; then
  fail "No NVIDIA GPU / CUDA detected. Distributed LLM requires CUDA-capable GPUs."
fi

# Map driver version to Docker CUDA image tag
if [ "$CUDA_MAJOR" -ge 12 ] && [ "$CUDA_MINOR" -ge 8 ]; then
  CUDA_TAG="12.8.0"
  DOCKERFILE="Dockerfile"
elif [ "$CUDA_MAJOR" -ge 12 ] && [ "$CUDA_MINOR" -ge 6 ]; then
  CUDA_TAG="12.6.0"
  DOCKERFILE="Dockerfile.cuda12.6"
elif [ "$CUDA_MAJOR" -ge 12 ] && [ "$CUDA_MINOR" -ge 1 ]; then
  CUDA_TAG="12.1.0"
  DOCKERFILE="Dockerfile.cuda12.1"
else
  warn "CUDA $CUDA_MAJOR.$CUDA_MINOR is not officially supported. Trying CUDA 12.1."
  CUDA_TAG="12.1.0"
  DOCKERFILE="Dockerfile.cuda12.1"
fi

ok "CUDA $CUDA_MAJOR.$CUDA_MINOR detected, using $DOCKERFILE"

# Determine docker compose command
if docker compose version >/dev/null 2>&1; then
  DC="docker compose"
elif docker-compose version >/dev/null 2>&1; then
  DC="docker-compose"
else
  fail "Neither 'docker compose' nor 'docker-compose' found. Install Docker Compose."
fi

# ---- nvidia-container-toolkit check ----------------------------------------

if ! docker info 2>/dev/null | grep -qi "nvidia"; then
  warn "nvidia-container-toolkit not detected. Attempting to install..."
  if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS_ID="$ID"
  else
    OS_ID="unknown"
  fi
  case "$OS_ID" in
    ubuntu|debian)
      curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
        gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg 2>/dev/null || true
      curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        tee /etc/apt/sources.list.d/nvidia-container-toolkit.list 2>/dev/null || true
      apt-get update -qq && apt-get install -y -qq nvidia-container-toolkit 2>/dev/null || \
        warn "Could not install nvidia-container-toolkit. Install manually: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
      ;;
    rhel|centos|fedora|rocky|almalinux)
      curl -fsSL https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo | \
        tee /etc/yum.repos.d/nvidia-container-toolkit.repo 2>/dev/null || true
      yum install -y nvidia-container-toolkit 2>/dev/null || \
        warn "Could not install nvidia-container-toolkit. Install manually."
      ;;
    *)
      warn "Unsupported OS ($OS_ID). Install nvidia-container-toolkit manually."
      ;;
  esac
  if command -v nvidia-ctk >/dev/null 2>&1; then
    nvidia-ctk runtime configure --runtime=docker 2>/dev/null || true
    if pidof dockerd >/dev/null 2>&1; then
      kill -s SIGHUP "$(pidof dockerd)" 2>/dev/null || true
    fi
    ok "nvidia-container-toolkit installed and configured"
  fi
fi

# ---- clone / update repo ---------------------------------------------------

if [ -d "$DISTLLM_DIR" ]; then
  info "Updating existing installation at $DISTLLM_DIR"
  cd "$DISTLLM_DIR"
  git pull --ff-only 2>/dev/null || true
else
  info "Cloning repository to $DISTLLM_DIR"
  git clone --depth 1 "$REPO" "$DISTLLM_DIR"
  cd "$DISTLLM_DIR"
fi

# ---- docker compose build --------------------------------------------------

info "Building DistLLM Docker image (CUDA ${CUDA_TAG})..."

if [ ! -f "$DOCKERFILE" ]; then
  warn "$DOCKERFILE not found, falling back to Dockerfile"
  DOCKERFILE="Dockerfile"
fi

export DOCKER_BUILDKIT=1

$DC build \
  --build-arg "CUDA_VERSION=$CUDA_TAG" \
  --build-arg "TORCH_CUDA_ARCH_LIST=8.0;8.6;8.9;9.0" \
  2>&1 | tail -5

ok "Docker image built"

# ---- start services --------------------------------------------------------

info "Starting coordinator + 2 worker nodes via Docker Compose..."
$DC up -d 2>&1

# ---- wait for health -------------------------------------------------------

info "Waiting for API server at http://localhost:8000/health ..."
ATTEMPTS=0
MAX_ATTEMPTS=60
until curl -sf http://localhost:8000/health >/dev/null 2>&1; do
  ATTEMPTS=$((ATTEMPTS + 1))
  if [ "$ATTEMPTS" -ge "$MAX_ATTEMPTS" ]; then
    warn "Timed out waiting for API server. Check logs: $DC logs"
    break
  fi
  sleep 2
done

# ---- print summary ---------------------------------------------------------

printf "\n"
printf "╔══════════════════════════════════════════════════════════════╗\n"
printf "║  ${GREEN}DistLLM is running!${NC}                                   ║\n"
printf "╠══════════════════════════════════════════════════════════════╣\n"
printf "║  ${BLUE}Your API is at${NC}                                           ║\n"
printf "║  ${GREEN}http://localhost:8000${NC}                                    ║\n"
printf "╠══════════════════════════════════════════════════════════════╣\n"
printf "║  Model:     ${MODEL}          ║\n"
printf "║  CUDA:      ${CUDA_MAJOR}.${CUDA_MINOR}                                       ║\n"
printf "║  Services:  coordinator, node_0, node_1                    ║\n"
printf "╠══════════════════════════════════════════════════════════════╣\n"
printf "║  ${YELLOW}Useful commands:${NC}                                          ║\n"
printf "║  %s logs -f             # tail logs        ║\n" "$DC"
printf "║  %s down                # stop all          ║\n" "$DC"
printf "║  curl http://localhost:8000/v1/chat/completions  # test API║\n"
printf "╚══════════════════════════════════════════════════════════════╝\n"
printf "\n"
