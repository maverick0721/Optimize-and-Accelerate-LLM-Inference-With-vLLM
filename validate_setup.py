#!/usr/bin/env python3
"""Quick environment validation for this benchmark project."""

import importlib
import sys

REQUIRED = [
    "torch",
    "transformers",
    "vllm",
    "matplotlib",
    "pandas",
    "numpy",
]


def main() -> int:
    print(f"Python: {sys.version.split()[0]}")
    failed = []

    for module in REQUIRED:
        try:
            mod = importlib.import_module(module)
            version = getattr(mod, "__version__", "unknown")
            print(f"[OK] {module} ({version})")
        except Exception as exc:
            print(f"[FAIL] {module}: {type(exc).__name__}: {exc}")
            failed.append(module)

    try:
        import torch

        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"CUDA runtime: {torch.version.cuda}")
    except Exception as exc:
        print(f"[WARN] torch CUDA check failed: {exc}")

    if failed:
        print("\nMissing modules detected. Install dependencies with:")
        print("pip install -r requirements.txt")
        return 1

    print("\nEnvironment looks ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
