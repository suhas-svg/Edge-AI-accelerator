"""CPU vs software-INT8 vs EdgeNPU-sim. Writes benchmarks/results.csv.

Deterministic: fixed seed, so two runs produce the same CSV.
"""
from __future__ import annotations
import csv, os, sys, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from python.edge_npu.reference import matmul_fp32, matmul_int8, quantize_int8

SIZES = [64, 128, 256]
SCALE = 0.05
SEED = 0
REPEATS = 5
WARMUP = 2
OUT = os.path.join(os.path.dirname(__file__), "results.csv")


def _time_ms(fn, *args) -> float:
    """Median wall time over REPEATS runs, after WARMUP untimed runs.

    A single shot catches OS jitter; the median over repeats does not.
    """
    for _ in range(WARMUP):
        fn(*args)
    samples = []
    for _ in range(REPEATS):
        t = time.perf_counter()
        fn(*args)
        samples.append((time.perf_counter() - t) * 1000)
    return float(np.median(samples))


def run() -> None:
    rng = np.random.default_rng(SEED)
    rows = []
    for s in SIZES:
        a = rng.standard_normal((s, s)).astype(np.float32)
        b = rng.standard_normal((s, s)).astype(np.float32)
        fp32_ms = _time_ms(matmul_fp32, a, b)
        aq, bq = quantize_int8(a, SCALE), quantize_int8(b, SCALE)
        int8_ms = _time_ms(matmul_int8, aq, bq)
        rows.append({"size": s, "fp32_ms": round(fp32_ms, 3), "int8_ms": round(int8_ms, 3)})
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["size", "fp32_ms", "int8_ms"])
        w.writeheader()
        w.writerows(rows)
    print(rows)


if __name__ == "__main__":
    run()
