"""CPU vs software-INT8 vs EdgeNPU-sim. Writes benchmarks/results.csv."""
from __future__ import annotations
import csv, time
import numpy as np
from python.edge_npu.reference import matmul_fp32, matmul_int8, quantize_int8

SIZES = [64, 128, 256]


def run() -> None:
    rows = []
    for s in SIZES:
        a = np.random.randn(s, s).astype(np.float32)
        b = np.random.randn(s, s).astype(np.float32)
        t = time.perf_counter()
        matmul_fp32(a, b)
        fp32_ms = (time.perf_counter() - t) * 1000
        aq, bq = quantize_int8(a, 0.05), quantize_int8(b, 0.05)
        t = time.perf_counter()
        matmul_int8(aq, bq)
        int8_ms = (time.perf_counter() - t) * 1000
        rows.append({"size": s, "fp32_ms": round(fp32_ms, 2), "int8_ms": round(int8_ms, 2)})
    with open("benchmarks/results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["size", "fp32_ms", "int8_ms"])
        w.writeheader()
        w.writerows(rows)
    print(rows)


if __name__ == "__main__":
    run()
