#!/usr/bin/env bash
set -euo pipefail

python validate_setup.py
python run_benchmark.py "$@"
