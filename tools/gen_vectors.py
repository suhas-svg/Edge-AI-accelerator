"""Golden INT8 matmul vectors. Deterministic: fixed seed, fixed scale. Rerun converges."""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from python.edge_npu.reference import quantize_int8, matmul_int8, conv2d_int8, max_pool

SIZES = [(8, 8, 8), (8, 8, 16), (16, 8, 8), (64, 64, 64), (128, 128, 128), (256, 256, 256)]
# Odd dims never reach hardware unpadded; these are software-side only and
# live in a separate directory so the RTL harness glob never sees them.
ODD_SIZES = [(20, 12, 28), (1, 8, 8), (60, 64, 64)]
CONV_ODD_SIZES = [(4, 9, 9, 8, 3), (8, 15, 15, 8, 3), (3, 10, 12, 8, 2)]
POOL_ODD_SIZES = [(8, 7, 7, 2, 2), (4, 9, 9, 3, 1), (5, 9, 9, 2, 2)]
SCALE = 0.05
OUT = os.path.join(os.path.dirname(__file__), "..", "tests", "vectors")
ODD_OUT = os.path.join(os.path.dirname(__file__), "..", "tests", "vectors_odd")


def _write(path: str, a_q: np.ndarray, b_q: np.ndarray, expected: np.ndarray) -> None:
    # matmul_int8 raises on an out-of-range result, so no range check here.
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
    os.makedirs(ODD_OUT, exist_ok=True)
    orng = np.random.default_rng(11)
    for m, k, n in ODD_SIZES:
        a = orng.standard_normal((m, k)).astype(np.float32)
        b = orng.standard_normal((k, n)).astype(np.float32)
        aq, bq = quantize_int8(a, SCALE), quantize_int8(b, SCALE)
        expected = matmul_int8(aq, bq)
        _write(os.path.join(ODD_OUT, f"matmul_{m}x{k}x{n}.npz"), aq, bq, expected)
    for c, h, wd, k, kh in CONV_ODD_SIZES:
        x = orng.standard_normal((c, h, wd)).astype(np.float32)
        w = orng.standard_normal((k, c, kh, kh)).astype(np.float32)
        xq, wq = quantize_int8(x, SCALE), quantize_int8(w, SCALE)
        expected = conv2d_int8(xq, wq)
        path = os.path.join(ODD_OUT, f"conv_{c}x{h}x{wd}_{k}x{c}x{kh}x{kh}.npz")
        np.savez(path, x_q=xq, w_q=wq, expected=expected)
        print(f"wrote {path}")
    for c, h, wd, win, st in POOL_ODD_SIZES:
        x = orng.integers(-10000, 10000, size=(c, h, wd)).astype(np.int32)
        expected = max_pool(x, win, st)
        path = os.path.join(ODD_OUT, f"pool_{c}x{h}x{wd}_k{win}_s{st}.npz")
        np.savez(path, x=x, expected=expected)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
