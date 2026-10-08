# Phase 2 software verification harness implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the golden vectors to non-square tile-boundary and overflow cases, and build the harness that diffs RTL simulation output against the Python reference.

**Architecture:** `tools/gen_vectors.py` owns vector generation and stays deterministic so a rerun is byte-identical. `tools/verify_rtl.py` owns the diff: it reads per-vector `rtl_out` arrays produced by whatever RTL simulator the hardware partner wires in, compares them element-wise against `expected`, and exits nonzero on any mismatch. A self-test proves the harness fails on a deliberately corrupted result, so the harness has teeth before any RTL exists.

**Tech Stack:** Python 3.10+, numpy, pytest.

**Spec:** The EdgeNPU spec at the top of this project conversation. Sections 22 (simulator), 23 (hardware verification against golden reference), 26 (benchmark models and sizes), 5 (8x8 INT8 MAC array, INT32 accumulator).

## Global Constraints

- MAC array is 8x8. INT8 inputs and weights, INT32 accumulators.
- Vectors live in `tests/vectors/`, named `matmul_<M>x<K>x<N>.npz`, with keys `a_q`, `b_q`, `expected` as they are today.
- `expected` is always the exact INT32 result from `python.edge_npu.reference.matmul_int8`.
- Repo python is `.venv/Scripts/python.exe`. Shell is Windows Git Bash.
- Every task ends green on `bash scripts/check.sh`.

## Review Focus

- A non-square vector reaches the harness and a reasonable person expects it compared by shape, not by a square-only code path. Pinned by the non-square case in Task 2.
- A vector whose `expected` value exceeds INT32 range reaches generation and a reasonable person expects an error naming the vector, not a silently wrapped result. Pinned by the overflow-range test in Task 1.
- A vector directory is missing an `rtl_out` key and a reasonable person expects the harness to name the file and fail, not skip it. Pinned by the missing-key test in Task 2.
- A harness that never sees a mismatch reports success and a reasonable person expects proof it can fail. Pinned by the corrupted-input test in Task 2.
- Generation runs twice and a reasonable person expects the same bytes. Pinned by the determinism test in Task 1.

---

### Task 1: Extend vectors to tile boundaries and overflow

**Files:**
- Modify: `tools/gen_vectors.py`
- Test: `tests/test_gen_vectors.py`

**Interfaces:**
- Consumes: `quantize_int8`, `matmul_int8` from `python/edge_npu/reference.py`.
- Produces: `SIZES: list[tuple[int, int, int]]` as `(m, k, n)` triples, and `main()` writing `tests/vectors/matmul_<m>x<k>x<n>.npz` with keys `a_q`, `b_q`, `expected`. Also writes `matmul_overflow.npz`, an `(m=64, k=64, n=1)` vector where every element of `a_q` is -128 and every element of `b_q` is 127, so `expected` is exactly -1040384. `main()` raises `ValueError` naming the vector if any `expected` value falls outside INT32 range.

- [ ] **Step 1: Write the failing tests**

```python
def test_sizes_are_triples():
    from tools import gen_vectors
    assert all(len(s) == 3 for s in gen_vectors.SIZES)


def test_generation_is_byte_identical(tmp_path, monkeypatch):
    import hashlib, subprocess, sys, os
    repo = os.path.join(os.path.dirname(__file__), "..")

    def digest() -> dict:
        out = {}
        d = os.path.join(repo, "tests", "vectors")
        for f in sorted(os.listdir(d)):
            if f.endswith(".npz"):
                with open(os.path.join(d, f), "rb") as fh:
                    out[f] = hashlib.sha256(fh.read()).hexdigest()
        return out

    first = digest()
    subprocess.run([sys.executable, os.path.join(repo, "tools", "gen_vectors.py")],
                   check=True, capture_output=True)
    assert digest() == first


def test_overflow_vector_expected_value():
    import numpy as np
    d = np.load("tests/vectors/matmul_overflow.npz")
    assert d["a_q"].shape == (64, 64)
    assert d["a_q"].min() == -128 and d["a_q"].max() == -128
    assert d["b_q"].min() == 127 and d["b_q"].max() == 127
    assert d["expected"].tolist() == [[-1040384] * 64]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_gen_vectors.py -v`
Expected: FAIL. `test_sizes_are_triples` fails because `SIZES` is currently `list[int]`. `test_overflow_vector_expected_value` fails with `FileNotFoundError`.

- [ ] **Step 3: Extend `tools/gen_vectors.py`**

Set `SIZES` to `[(8, 8, 8), (8, 8, 16), (16, 8, 8), (64, 64, 64), (128, 128, 128), (256, 256, 256)]`. Loop unpacks `m, k, n` and builds `a` as `(m, k)` and `b` as `(k, n)`. After computing `expected`, check `expected.min() >= -2**31 and expected.max() <= 2**31 - 1`, else raise `ValueError` naming the size. Add the overflow vector case with a fixed array, not a random one.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_gen_vectors.py -v`
Expected: PASS, 3/3.

- [ ] **Step 5: Commit**

```bash
git add tools/gen_vectors.py tests/test_gen_vectors.py tests/vectors/
git commit -m "test: extend golden vectors to tile boundaries and overflow"
```

### Task 2: RTL verification harness

**Files:**
- Create: `tools/verify_rtl.py`
- Test: `tests/test_verify_rtl.py`
- Modify: `scripts/check.sh` to run the harness self-check

**Interfaces:**
- Consumes: vectors in `tests/vectors/`, each optionally carrying an `rtl_out` key of the same shape as `expected`.
- Produces: `compare_vector(path: str) -> list[str]` returning a list of mismatch descriptions, empty on success, in `tools/verify_rtl.py`. Also `main(vec_dir: str) -> int` that prints one line per vector and exits `0` when every vector matches and `1` otherwise. A vector with no `rtl_out` key counts as a failure with a message naming the file and the word `MISSING`. A shape mismatch reports both shapes. A value mismatch reports the first index and both values.

- [ ] **Step 1: Write the failing tests**

```python
def test_matching_vector_passes(tmp_path):
    import numpy as np
    from tools.verify_rtl import compare_vector
    path = str(tmp_path / "m.npz")
    np.savez(path, a_q=np.ones((8, 8), dtype=np.int8),
             b_q=np.ones((8, 8), dtype=np.int8),
             expected=np.full((8, 8), 8, dtype=np.int32),
             rtl_out=np.full((8, 8), 8, dtype=np.int32))
    assert compare_vector(path) == []


def test_missing_rtl_out_is_a_failure(tmp_path):
    import numpy as np
    from tools.verify_rtl import compare_vector
    path = str(tmp_path / "m.npz")
    np.savez(path, a_q=np.ones((8, 8), dtype=np.int8),
             b_q=np.ones((8, 8), dtype=np.int8),
             expected=np.full((8, 8), 8, dtype=np.int32))
    msgs = compare_vector(path)
    assert len(msgs) == 1
    assert "MISSING" in msgs[0] and "m.npz" in msgs[0]


def test_wrong_value_reports_first_index(tmp_path):
    import numpy as np
    from tools.verify_rtl import compare_vector
    exp = np.full((2, 2), 8, dtype=np.int32)
    rtl = exp.copy()
    rtl[1, 1] = 9
    path = str(tmp_path / "m.npz")
    np.savez(path, a_q=np.ones((2, 2), dtype=np.int8),
             b_q=np.ones((2, 2), dtype=np.int8),
             expected=exp, rtl_out=rtl)
    msgs = compare_vector(path)
    assert len(msgs) == 1
    assert "(1, 1)" in msgs[0] and "8" in msgs[0] and "9" in msgs[0]


def test_shape_mismatch_reported(tmp_path):
    import numpy as np
    from tools.verify_rtl import compare_vector
    path = str(tmp_path / "m.npz")
    np.savez(path, a_q=np.ones((8, 8), dtype=np.int8),
             b_q=np.ones((8, 8), dtype=np.int8),
             expected=np.full((8, 8), 8, dtype=np.int32),
             rtl_out=np.full((4, 16), 8, dtype=np.int32))
    msgs = compare_vector(path)
    assert len(msgs) == 1
    assert "(8, 8)" in msgs[0] and "(4, 16)" in msgs[0]


def test_main_exits_nonzero_on_mismatch(tmp_path):
    import numpy as np
    from tools.verify_rtl import main
    exp = np.full((2, 2), 8, dtype=np.int32)
    rtl = exp.copy()
    rtl[0, 0] = 7
    np.savez(str(tmp_path / "m.npz"), a_q=np.ones((2, 2), dtype=np.int8),
             b_q=np.ones((2, 2), dtype=np.int8), expected=exp, rtl_out=rtl)
    assert main(str(tmp_path)) == 1


def test_self_check_catches_deliberate_corruption(tmp_path):
    """The harness must fail on a corrupted copy of a real golden vector."""
    import numpy as np, os, shutil, subprocess, sys
    repo = os.path.join(os.path.dirname(__file__), "..")
    vdir = os.path.join(repo, "tests", "vectors")
    tmp = str(tmp_path)
    for f in os.listdir(vdir):
        if f.endswith(".npz"):
            d = dict(np.load(os.path.join(vdir, f)))
            d["rtl_out"] = d["expected"].copy()
            np.savez(os.path.join(tmp, f), **d)
    # corrupt exactly one output value
    target = os.path.join(tmp, "matmul_8x8x8.npz")
    d = dict(np.load(target))
    d["rtl_out"] = d["rtl_out"].copy()
    d["rtl_out"][0, 0] = d["rtl_out"][0, 0] + 1
    np.savez(target, **d)
    r = subprocess.run([sys.executable, os.path.join(repo, "tools", "verify_rtl.py"), tmp],
                       capture_output=True, text=True)
    assert r.returncode == 1
    assert "matmul_8x8x8.npz" in r.stdout
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_verify_rtl.py -v`
Expected: FAIL with "cannot import name 'compare_vector'".

- [ ] **Step 3: Implement `tools/verify_rtl.py`**

`compare_vector` loads the npz, returns one message per defect. `main(vec_dir)` iterates `matmul_*.npz` sorted, prints `PASS` or each message per file, and returns 1 if any file has messages. Guard the entry point with `if __name__ == "__main__": sys.exit(mame(sys.argv[1] if len(sys.argv) > 1 else default))`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_verify_rtl.py -v`
Expected: PASS, 6/6.

- [ ] **Step 5: Wire the self-check into `scripts/check.sh` and commit**

Add one line before pytest that builds a corrupted copy in `build/` and asserts the harness exits 1, so `check.sh` fails if the harness ever stops catching a defect.

```bash
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
```

```bash
git add tools/verify_rtl.py tests/test_verify_rtl.py scripts/check.sh
git commit -m "test: add RTL golden-vector verification harness with self-check"
```
