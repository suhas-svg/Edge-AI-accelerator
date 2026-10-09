#!/usr/bin/env bash
# One green check for the software slice. Fails loud, no partial pass.
set -euo pipefail
cd "$(dirname "$0")/.."
.venv/Scripts/python.exe tools/gen_vectors.py
.venv/Scripts/python.exe -m pytest tests/ -q
mkdir -p build
# Prove the harness still fails on a corrupted vector. selfcheck writes a
# deliberately wrong rtl_out, so verify_rtl must exit 1 here.
.venv/Scripts/python.exe tools/selfcheck.py build/selfcheck
if .venv/Scripts/python.exe tools/verify_rtl.py build/selfcheck; then
  echo "FAIL: harness did not catch a corrupted vector" >&2; exit 1
fi
gcc -c runtime/src/edge_npu.c -o build/edge_npu_check.o -I runtime/include -Wall
echo "check ok"
