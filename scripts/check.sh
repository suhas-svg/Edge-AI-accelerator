#!/usr/bin/env bash
# One green check for the software slice. Fails loud, no partial pass.
set -euo pipefail
cd "$(dirname "$0")/.."
.venv/Scripts/python.exe -m pytest tests/ -q
mkdir -p build
gcc -c runtime/src/edge_npu.c -o build/edge_npu_check.o -I runtime/include -Wall
echo "check ok"
