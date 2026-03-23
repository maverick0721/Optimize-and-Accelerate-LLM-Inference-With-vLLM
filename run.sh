#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="local"
PYTHON_BIN="python3"
VENV_DIR="$PROJECT_DIR/.venv"
IMAGE="${IMAGE:-nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"
DOCKER_GPU_ARG="${DOCKER_GPU_ARG:-auto}"

SKIP_INSTALL=0
NO_VENV=0
CREATED_VENV=0

DEFAULT_BENCH_ARGS=(--seed 42 --max-new-tokens 100 --output vllm_vs_hf_results.csv --quiet)
BENCH_ARGS=()

print_help() {
  cat <<'EOF'
Usage: ./run.sh [options] [-- run_benchmark.py args]

Single script launcher for this project.
It supports local execution and Docker no-build fallback execution.

Options:
  --mode <local|docker-nobuild>   Execution mode (default: local)
  --skip-install                  Skip dependency installation
  --no-venv                       Use system Python for local mode
  --image <docker-image>          Base image for docker-nobuild mode
  --torch-index-url <url>         PyTorch wheel index URL
  --docker-gpu-arg <value>        Docker GPU arg; use auto, '--gpus all', or '--device nvidia.com/gpu=all'
  -h, --help                      Show this help

Benchmark args:
  Pass benchmark args after -- to forward them to run_benchmark.py.

Examples:
  ./run.sh
  ./run.sh -- --max-new-tokens 64 --output demo_results.csv
  ./run.sh --mode local --skip-install -- --batch-sizes 1 4 8
  ./run.sh --mode docker-nobuild
  ./run.sh --mode docker-nobuild --docker-gpu-arg "--device nvidia.com/gpu=all" -- --batch-sizes 1
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="${2:-}"
      shift 2
      ;;
    --skip-install)
      SKIP_INSTALL=1
      shift
      ;;
    --no-venv)
      NO_VENV=1
      shift
      ;;
    --image)
      IMAGE="${2:-}"
      shift 2
      ;;
    --torch-index-url)
      TORCH_INDEX_URL="${2:-}"
      shift 2
      ;;
    --docker-gpu-arg)
      DOCKER_GPU_ARG="${2:-}"
      shift 2
      ;;
    -h|--help)
      print_help
      exit 0
      ;;
    --)
      shift
      BENCH_ARGS=("$@")
      break
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help for usage."
      exit 1
      ;;
  esac
done

if [[ ${#BENCH_ARGS[@]} -eq 0 ]]; then
  BENCH_ARGS=("${DEFAULT_BENCH_ARGS[@]}")
fi

run_local() {
  echo "[1/5] Entering project directory"
  cd "$PROJECT_DIR"

  if [[ "$NO_VENV" -eq 0 ]]; then
    echo "[2/5] Preparing virtual environment at .venv"
    if [[ ! -d "$VENV_DIR" ]]; then
      "$PYTHON_BIN" -m venv "$VENV_DIR"
      CREATED_VENV=1
    fi
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
  else
    echo "[2/5] Skipping virtual environment (using system Python)"
  fi

  if [[ "$SKIP_INSTALL" -eq 1 ]] && [[ "$NO_VENV" -eq 0 ]] && [[ "$CREATED_VENV" -eq 1 ]]; then
    echo "Error: --skip-install was set, but a new .venv was just created and has no dependencies yet."
    echo "Run again without --skip-install, or use --no-venv to use your current system environment."
    exit 1
  fi

  if [[ "$SKIP_INSTALL" -eq 0 ]]; then
    echo "[3/5] Installing dependencies"
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
  else
    echo "[3/5] Skipping dependency installation"
  fi

  echo "[4/5] Validating environment"
  python validate_setup.py

  echo "[5/5] Running benchmark"
  python run_benchmark.py "${BENCH_ARGS[@]}"
}

resolve_docker_gpu_args() {
  local value="$1"
  if [[ "$value" == "auto" ]]; then
    if docker run --rm --gpus all hello-world >/dev/null 2>&1; then
      echo "--gpus all"
      return
    fi
    if docker run --rm --device nvidia.com/gpu=all hello-world >/dev/null 2>&1; then
      echo "--device nvidia.com/gpu=all"
      return
    fi
    echo "Error: could not determine a supported Docker GPU flag." >&2
    echo "Set --docker-gpu-arg or DOCKER_GPU_ARG manually." >&2
    exit 1
  fi
  echo "$value"
}

run_docker_nobuild() {
  cd "$PROJECT_DIR"
  mkdir -p "$PROJECT_DIR/.cache/huggingface"

  local gpu_arg
  gpu_arg="$(resolve_docker_gpu_args "$DOCKER_GPU_ARG")"

  local bench_cmd=""
  local arg
  for arg in "${BENCH_ARGS[@]}"; do
    bench_cmd+=" $(printf '%q' "$arg")"
  done

  # shellcheck disable=SC2206
  local gpu_args=($gpu_arg)

  docker run --rm "${gpu_args[@]}" \
    -v "$PROJECT_DIR":/workspace \
    -w /workspace \
    -e HF_HOME=/workspace/.cache/huggingface \
    -e TRANSFORMERS_CACHE=/workspace/.cache/huggingface \
    "$IMAGE" \
    /bin/bash -lc "set -euo pipefail; \
      apt-get update; \
      apt-get install -y --no-install-recommends python3 python3-pip python3-venv git; \
      ln -sf /usr/bin/python3 /usr/bin/python; \
      python -m pip install --upgrade pip; \
      python -m pip install --index-url $TORCH_INDEX_URL torch torchvision torchaudio; \
      python -m pip install -r requirements.txt; \
      python validate_setup.py; \
      python run_benchmark.py$bench_cmd"
}

case "$MODE" in
  local)
    run_local
    ;;
  docker-nobuild)
    run_docker_nobuild
    ;;
  *)
    echo "Invalid --mode value: $MODE"
    echo "Allowed values: local, docker-nobuild"
    exit 1
    ;;
esac
