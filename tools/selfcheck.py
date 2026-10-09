"""Build a deliberately corrupted copy of the golden vectors.

Single source of truth for "what a failing RTL run looks like". Used by
scripts/check.sh and by tests/test_verify_rtl.py so the two cannot drift.
"""
from __future__ import annotations
import os
import sys
import numpy as np

REPO = os.path.join(os.path.dirname(__file__), "..")
VEC_DIR = os.path.join(REPO, "tests", "vectors")
VECTOR = "matmul_8x8x8.npz"


def build_corrupted(out_dir: str) -> str:
    """Copy every golden vector into out_dir, then corrupt one value."""
    os.makedirs(out_dir, exist_ok=True)
    for f in os.listdir(VEC_DIR):
        if f.endswith(".npz"):
            d = dict(np.load(os.path.join(VEC_DIR, f)))
            d["rtl_out"] = d["expected"].copy()
            np.savez(os.path.join(out_dir, f), **d)
    target = os.path.join(out_dir, VECTOR)
    d = dict(np.load(target))
    d["rtl_out"] = d["rtl_out"].copy()
    d["rtl_out"][0, 0] += 1
    np.savez(target, **d)
    return target


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "build/selfcheck"
    print(f"wrote corrupted copy to {build_corrupted(out)}")
