#!/usr/bin/env bash
# Environment doctor for the EdgeNPU software slice.
#
# Answers "why does check.sh fail on a clean machine" in one command, printing
# the evidence instead of describing it. Both faults below were found by
# bisection; scripts/check.sh applies the same two repairs.
set -uo pipefail
cd "$(dirname "$0")/.."

pass=0; fail=0; warn=0
ok()   { echo "  [ OK ] $1"; pass=$((pass+1)); }
bad()  { echo "  [FAIL] $1"; fail=$((fail+1)); }
warn() { echo "  [WARN] $1"; warn=$((warn+1)); }

mkdir -p build

# gcc on this host can exit nonzero with ZERO stderr, so only a produced object
# counts as success.
gcc_ok() { gcc -c runtime/src/edge_npu.c -o build/.doctor_probe.o -I runtime/include -Wall >/dev/null 2>&1 && [ -f build/.doctor_probe.o ]; }

echo "=== 1. interpreter and imports ==="
PY=.venv/Scripts/python.exe
if [ ! -x "$PY" ]; then
  bad ".venv missing - run: bash scripts/setup.sh"
elif "$PY" -c "import numpy" 2>/dev/null; then
  ok "numpy imports in the project .venv"
elif env -u PYTHONPATH "$PY" -c "import numpy" 2>/dev/null; then
  warn "numpy imports ONLY with PYTHONPATH cleared - the ambient env shadows it."
  echo "         check.sh already does 'unset PYTHONPATH', so this is diagnosed,"
  echo "         not blocking. In your own shell run: export PYTHONPATH=/dev/null"
  echo "         ambient PYTHONPATH=${PYTHONPATH:-<unset>}"
else
  bad ".venv cannot import numpy at all - rebuild with: bash scripts/setup.sh"
fi

echo "=== 2. gcc can reach its own sub-tools ==="
# These two exports are exactly what check.sh does FIRST, so this section
# measures the post-repair state rather than the bare ambient one.
export PATH="/c/MinGW/bin:$PATH"
unset PYTHONPATH
if gcc_ok; then
  ok "gcc compiles the runtime"
else
  bad "gcc cannot compile even with /c/MinGW/bin first - inspect the install"
fi

echo "=== 3. the suite itself ==="
if "$PY" -m pytest tests/ -q 2>/dev/null; then
  ok "pytest green"
else
  bad "pytest failed - run: .venv/Scripts/python.exe -m pytest tests/ -q"
fi

echo
echo "doctor: $pass ok, $warn warned, $fail blocked"
[ "$fail" -eq 0 ]
