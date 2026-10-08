#!/usr/bin/env bash
# Idempotent software setup. Safe to run twice.
set -euo pipefail
cd "$(dirname "$0")/.."

if ! command -v uv >/dev/null 2>&1; then
  echo "missing: uv (https://docs.astral.sh/uv/)" >&2
  exit 1
fi

# Converge: reuse .venv if present, create if absent.
if [ ! -d .venv ]; then
  uv venv .venv
fi

uv pip install --python .venv/Scripts/python.exe -e ".[dev]" 2>/dev/null \
  || uv pip install --python .venv/Scripts/python.exe numpy onnx pytest

echo "--- toolchain ---"
.venv/Scripts/python.exe --version
git --version
gcc --version | head -n 1
.venv/Scripts/python.exe -c "import numpy, onnx, pytest; print('py deps ok:', numpy.__version__)"
echo "setup ok. next: bash scripts/check.sh"
