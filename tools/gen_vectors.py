"""Golden INT8 matmul vectors. Deterministic: fixed seed, fixed scale. Rerun converges."""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from python.edge_npu.reference import quantize_int8, matmul_int8

SIZES = [(8, 8, 8), (8, 8, 16), (16, 8, 8), (64, 64, 64), (128, 128, 128), (256, 256, 256)]
SCALE = 0.05
OUT = os.path.join(os.path.dirname(__file__), "..", "tests", "vectors")
INT32_MIN, INT32_MAX = -(2 ** 31), 2 ** 31 - 1


def _write(path: str, a_q: np.ndarray, b_q: np.ndarray, expected: np.ndarray) -> None:
    if expected.min() < INT32_MIN or expected.max() > INT32_MAX:
        raise ValueError(f"{path}: expected exceeds INT32 range")
    np.savez(path, a_q=a_q, b_q=b_q, expected=expected)
    print(f"wrote {path}")


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(0)
    for m, k, n in SIZES:
        a = rng.standard_normal((m, k)).astype(np.float32)
        b = rng.standard_normal((k, n)).astype(np.float32)
        aq, bq = quantize_int8(a, SCALE), quantize_int8(b, SCALE)
        expected = matmul_int8(aq, bq)
        _write(os.path.join(OUT, f"matmul_{m}x{k}x{n}.npz"), aq, bq, expected)
    ov_a = np.full((64, 64), -128, dtype=np.int8)
    ov_b = np.full((64, 64), 127, dtype=np.int8)
    _write(os.path.join(OUT, "matmul_overflow.npz"), ov_a, ov_b, matmul_int8(ov_a, ov_b))


if __name__ == "__main__":
    main()
