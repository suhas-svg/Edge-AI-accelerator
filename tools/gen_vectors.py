"""Golden INT8 matmul vectors. Deterministic: fixed seed, fixed scale. Rerun converges."""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from python.edge_npu.reference import quantize_int8, matmul_int8

SIZES = [8, 64, 128, 256]
SCALE = 0.05
OUT = os.path.join(os.path.dirname(__file__), "..", "tests", "vectors")


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(0)
    for s in SIZES:
        a = rng.standard_normal((s, s)).astype(np.float32)
        b = rng.standard_normal((s, s)).astype(np.float32)
        aq, bq = quantize_int8(a, SCALE), quantize_int8(b, SCALE)
        expected = matmul_int8(aq, bq)
        path = os.path.join(OUT, f"matmul_{s}x{s}.npz")
        np.savez(path, a_q=aq, b_q=bq, expected=expected)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
