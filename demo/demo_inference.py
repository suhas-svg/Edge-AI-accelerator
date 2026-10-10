"""End-to-end EdgeNPU demo: synthesize → quantize → compile → pack → load → predict.

Runs the spec section 40 pipeline in miniature on a single 64x64 INT8
matmul (no torch needed; weights are seeded synthetic). Asserts bit-exact
match against the golden reference, then prints the dashboard.
"""
from __future__ import annotations

import os
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from compiler.binary import write_model
from compiler.codegen import lower_matmul
from compiler.graph import Graph, Node
from python.edge_npu.commands import Command  # noqa: F401  (re-export check)
from python.edge_npu.reference import (
    dequantize_int8,
    matmul_int8,
    quantization_error,
    quantize_int8,
)
from python.edge_npu.sdk import Device
from python.edge_npu.tensor import TensorSpec

SIZE = 64
SCALE = 0.05
SEED = 0
REPEATS = 5


def _median_ms(fn, *args) -> float:
    for _ in range(2):
        fn(*args)
    samples = []
    for _ in range(REPEATS):
        t = time.perf_counter()
        fn(*args)
        samples.append((time.perf_counter() - t) * 1000)
    return float(np.median(samples))


def main() -> None:
    rng = np.random.default_rng(SEED)
    a = rng.standard_normal((SIZE, SIZE)).astype(np.float32)
    b = rng.standard_normal((SIZE, SIZE)).astype(np.float32)

    aq, bq = quantize_int8(a, SCALE), quantize_int8(b, SCALE)
    ref = matmul_int8(aq, bq)
    qerr = quantization_error(
        a.astype(np.float32) @ b.astype(np.float32),
        dequantize_int8(aq, SCALE) @ dequantize_int8(bq, SCALE),
    )

    node = Node(op="matmul", inputs=("a", "w"), output="c")
    specs = {"a": TensorSpec(name="a", dtype="int8", shape=(SIZE, SIZE)),
             "w": TensorSpec(name="w", dtype="int8", shape=(SIZE, SIZE)),
             "c": TensorSpec(name="c", dtype="int32", shape=(SIZE, SIZE))}
    cmds = lower_matmul(node, specs, base=0x1000)

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "demo.bin")
        write_model(path, Graph(nodes=(node,)), {"w": bq}, cmds)
        model = Device().load_model(path)
        out = model.predict(aq)
        cpu_ms = _median_ms(matmul_int8, aq, bq)
        sim = Device().benchmark(model, aq, repeats=REPEATS)
        stats = model.get_stats()

    diff = int(np.max(np.abs(out.astype(np.int64) - ref.astype(np.int64))))
    assert diff == 0, f"sim diverged from reference by {diff}"

    print("╔══════════════════════════════════════╗")
    print("║           EdgeNPU Dashboard          ║")
    print("╠══════════════════════════════════════╣")
    print("║ Model:        DemoMatmul 64x64       ║")
    print("║ Precision:    INT8                   ║")
    print("║                                      ║")
    print(f"║ CPU INT8:     {cpu_ms:>8.3f} ms           ║")
    print(f"║ EdgeNPU-sim:  {sim['median_ms']:>8.3f} ms           ║")
    print("║                                      ║")
    print(f"║ Sim cycles:   {stats['cycles']:<10d}         ║")
    print(f"║ MAC util:     {stats['mac_utilization'] * 100:>7.1f} %           ║")
    print(f"║ Quant err:    {qerr:>8.4f}               ║")
    print(f"║ Correctness:  max diff {diff} (exact)      ║")
    print("╚══════════════════════════════════════╝")


if __name__ == "__main__":
    main()
