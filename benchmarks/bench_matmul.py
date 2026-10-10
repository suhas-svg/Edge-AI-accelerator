"""CPU vs software-INT8 vs EdgeNPU-sim. Writes benchmarks/results.csv.

Deterministic: fixed seed, so two runs produce the same CSV.
"""
from __future__ import annotations
import csv, os, sys, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from python.edge_npu.reference import matmul_fp32, matmul_int8, quantize_int8
from python.edge_npu.sdk import Device
from python.edge_npu.tensor import TensorSpec
from compiler.binary import write_model
from compiler.graph import Graph, Node
from compiler.middleend import lower_graph

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


def _edgenpu_sim(aq: np.ndarray, bq: np.ndarray, tmp: str) -> tuple[float, int]:
    """Pack aq/bq's shapes through compiler → model.bin → SDK predict.

    Returns (median wall ms, sim cycles). Numerics equal matmul_int8;
    the point is the full compile→pack→load→infer path plus cycle counts.
    """
    s = aq.shape[0]
    node = Node(op="matmul", inputs=("a", "w"), output="c")
    specs = {"a": TensorSpec(name="a", dtype="int8", shape=(s, s)),
             "w": TensorSpec(name="w", dtype="int8", shape=(s, s)),
             "c": TensorSpec(name="c", dtype="int32", shape=(s, s))}
    lg = lower_graph(Graph(nodes=(node,)), specs, {"w": bq}, {})
    path = os.path.join(tmp, f"bench_{s}.bin")
    write_model(path, lg.graph, lg.weights, lg.cmds)
    model = Device().load_model(path)
    bench = Device().benchmark(model, aq, repeats=REPEATS)
    return round(bench["median_ms"], 3), model.get_stats()["cycles"]


def run() -> None:
    import tempfile

    rng = np.random.default_rng(SEED)
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for s in SIZES:
            a = rng.standard_normal((s, s)).astype(np.float32)
            b = rng.standard_normal((s, s)).astype(np.float32)
            fp32_ms = _time_ms(matmul_fp32, a, b)
            aq, bq = quantize_int8(a, SCALE), quantize_int8(b, SCALE)
            int8_ms = _time_ms(matmul_int8, aq, bq)
            sim_ms, cycles = _edgenpu_sim(aq, bq, tmp)
            rows.append({"size": s, "fp32_ms": round(fp32_ms, 3),
                         "int8_ms": round(int8_ms, 3),
                         "edgenpu_sim_ms": sim_ms, "edgenpu_cycles": cycles})
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["size", "fp32_ms", "int8_ms",
                                          "edgenpu_sim_ms", "edgenpu_cycles"])
        w.writeheader()
        w.writerows(rows)
    print(rows)


if __name__ == "__main__":
    run()
