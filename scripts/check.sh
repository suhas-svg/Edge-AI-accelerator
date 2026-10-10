#!/usr/bin/env bash
# One green check for the software slice. Fails loud, no partial pass.
set -euo pipefail
cd "$(dirname "$0")/.."

# --- environment repair (verified, see scripts/doctor.sh for the probe) ---
# PATH: this host has an unrelated /mingw64/bin ahead of /c/MinGW/bin carrying a
# DIFFERENT build of libgmp-10.dll. The Windows loader rejects the ABI mismatch
# and kills cc1.exe with exit 127 and ZERO diagnostics, so gcc looks like it
# silently does nothing. Putting the real MinGW bin first fixes it.
export PATH="/c/MinGW/bin:$PATH"
# PYTHONPATH: the Hermes agent's own venv is exported here and shadows the
# project's healthy numpy, so imports die with "DLL load failed: Access denied".
unset PYTHONPATH

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
gcc runtime/tests/test_runtime.c runtime/src/edge_npu.c -o build/runtime_check -I runtime/include -Wall
./build/runtime_check
g++ -c runtime/src/edge_npu_cpp.cpp -o build/edge_npu_cpp_check.o -I runtime/include -Wall -std=c++14
g++ runtime/tests/test_runtime_cpp.cpp runtime/src/edge_npu_cpp.cpp runtime/src/edge_npu.c -o build/runtime_cpp_check -I runtime/include -Wall -std=c++14
./build/runtime_cpp_check
.venv/Scripts/python.exe demo/demo_inference.py > /dev/null
echo "check ok"
