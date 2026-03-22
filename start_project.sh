#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"
PYTHON_BIN="python3"

SKIP_INSTALL=0
NO_VENV=0
CREATED_VENV=0

BENCH_ARGS=()
DEFAULT_ARGS=(--seed 42 --max-new-tokens 100 --output vllm_vs_hf_results.csv --quiet)
OUTPUT_FILE="vllm_vs_hf_results.csv"

print_help() {
  cat <<'EOF'
Usage: ./start_project.sh [options] [run_benchmark.py args]

One-command start-to-end launcher for this project.
It can create a virtual environment, install dependencies, validate setup,
and run the benchmark with deterministic defaults.

Options:
  --skip-install   Skip pip install -r requirements.txt
  --no-venv        Use system Python instead of .venv
  -h, --help       Show this help

Any additional arguments are forwarded to run_benchmark.py.
Examples:
  ./start_project.sh
  ./start_project.sh --max-new-tokens 64 --output demo_results.csv
  ./start_project.sh --skip-install --batch-sizes 1 4 8 --quiet
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-install)
      SKIP_INSTALL=1
      shift
      ;;
    --no-venv)
      NO_VENV=1
      shift
      ;;
    -h|--help)
      print_help
      exit 0
      ;;
    *)
      BENCH_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ ${#BENCH_ARGS[@]} -eq 0 ]]; then
  BENCH_ARGS=("${DEFAULT_ARGS[@]}")
fi

for ((i = 0; i < ${#BENCH_ARGS[@]}; i++)); do
  if [[ "${BENCH_ARGS[$i]}" == "--output" ]] && [[ $((i + 1)) -lt ${#BENCH_ARGS[@]} ]]; then
    OUTPUT_FILE="${BENCH_ARGS[$((i + 1))]}"
  fi
done

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

cat <<EOF

Done. Project run completed.

Project summary:
- Compares HuggingFace vs vLLM inference on TinyLlama.
- Measures latency, tokens/sec, throughput scaling, and peak GPU memory.
- Uses deterministic decode settings for reproducible comparisons.
- Results saved to: $OUTPUT_FILE
EOF
