#!/usr/bin/env bash
# One green check for the software slice. Fails loud, no partial pass.
set -euo pipefail
cd "$(dirname "$0")/.."
.venv/Scripts/python.exe tools/gen_vectors.py
.venv/Scripts/python.exe -m pytest tests/ -q
mkdir -p build
mkdir -p build/selfcheck
.venv/Scripts/python.exe - <<'EOF'
import numpy as np, os
out = "build/selfcheck"
os.makedirs(out, exist_ok=True)
d = dict(np.load("tests/vectors/matmul_8x8x8.npz"))
d["rtl_out"] = d["expected"].copy()
d["rtl_out"][0, 0] += 1
np.savez(os.path.join(out, "matmul_8x8x8.npz"), **d)
EOF
if .venv/Scripts/python.exe tools/verify_rtl.py build/selfcheck; then
  echo "FAIL: harness did not catch a corrupted vector" >&2; exit 1
fi
gcc -c runtime/src/edge_npu.c -o build/edge_npu_check.o -I runtime/include -Wall
echo "check ok"
