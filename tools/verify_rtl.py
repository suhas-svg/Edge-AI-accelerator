"""Golden-vector verification harness.

Reads every tests/vectors/matmul_*.npz. When a file carries an ``rtl_out``
key it is compared element-wise against ``expected``. A file without the
key counts as a failure, so a half-finished run of the hardware partner's
testbench cannot silently pass.

Exit code 0 when every vector matches, 1 otherwise.
"""
from __future__ import annotations
import os
import sys
import glob
import numpy as np

DEFAULT_VEC_DIR = os.path.join(os.path.dirname(__file__), "..", "tests", "vectors")


def compare_vector(path: str) -> list[str]:
    """Return one message per defect in a single vector file. Empty means match.

    Reports every defect found, not just the first, so a vector with both a
    shape error and value errors names both.
    """
    name = os.path.basename(path)
    d = np.load(path)
    if "rtl_out" not in d.files:
        return [f"{name}: MISSING rtl_out key; hardware output not present"]
    expected, rtl = d["expected"], d["rtl_out"]
    msgs: list[str] = []
    if expected.shape != rtl.shape:
        return [f"{name}: shape mismatch expected {expected.shape} vs rtl_out {rtl.shape}"]
    if expected.dtype != rtl.dtype:
        msgs.append(f"{name}: dtype mismatch expected {expected.dtype} vs rtl_out {rtl.dtype}")
    diff = np.argwhere(expected != rtl)
    if diff.size == 0:
        return msgs
    idx = tuple(int(i) for i in diff[0])
    msgs.append(f"{name}: mismatch at {idx} expected {expected[idx]} vs rtl_out {rtl[idx]} "
                f"({diff.shape[0]} of {expected.size} values differ)")
    return msgs


def main(vec_dir: str = DEFAULT_VEC_DIR) -> int:
    paths = sorted(glob.glob(os.path.join(vec_dir, "matmul_*.npz")))
    if not paths:
        print(f"no vectors found in {vec_dir}")
        return 1
    failures = 0
    for p in paths:
        msgs = compare_vector(p)
        if msgs:
            failures += 1
            for m in msgs:
                print(f"FAIL {m}")
        else:
            print(f"PASS {os.path.basename(p)}")
    total = len(paths)
    print(f"{total - failures}/{total} vectors match")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_VEC_DIR))
