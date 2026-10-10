"""CPU vs software-INT8 vs EdgeNPU-sim. Writes benchmarks/results.csv.

Deterministic inputs (fixed seed); wall-clock medians are reference
snapshots for humans, not gates. fp32_ms is blank where no FP32 reference
exists for that op (conv and multi-op chains).
"""
from __future__ import annotations
import csv, os, sys, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from python.edge_npu.reference import (
    bias_add, conv2d_int8, matmul_fp32, matmul_int8, max_pool, quantize_int8,
    relu, requantize,
)
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
FIELDS = ["case", "fp32_ms", "int8_ms", "edgenpu_sim_ms", "edgenpu_cycles",
          "busy_cycles", "dma_bytes", "command_counts"]


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


def _row(case: str, fp32_ms, int8_ms, sim_ms: float, stats: dict) -> dict:
    return {"case": case, "fp32_ms": fp32_ms, "int8_ms": int8_ms,
            "edgenpu_sim_ms": sim_ms, "edgenpu_cycles": stats["cycles"],
            "busy_cycles": stats["busy_cycles"], "dma_bytes": stats["dma_bytes"],
            "command_counts": _counts_str(stats["command_counts"])}


def _counts_str(counts: dict) -> str:
    return ";".join(f"{op}:{n}" for op, n in counts.items())


def _sim_run(path: str, graph: Graph, weights: dict, cmds: list,
             x: np.ndarray) -> tuple[float, dict]:
    """Pack → load → benchmark → stats through the full SDK path."""
    write_model(path, graph, weights, cmds)
    model = Device().load_model(path)
    bench = Device().benchmark(model, x, repeats=REPEATS)
    return round(bench["median_ms"], 3), model.get_stats()


def matmul_row(label: str, m: int, k: int, n: int, tmp: str) -> dict:
    rng = np.random.default_rng(SEED)
    a = rng.standard_normal((m, k)).astype(np.float32)
    b = rng.standard_normal((k, n)).astype(np.float32)
    fp32_ms = round(_time_ms(matmul_fp32, a, b), 3)
    aq, bq = quantize_int8(a, SCALE), quantize_int8(b, SCALE)
    int8_ms = round(_time_ms(matmul_int8, aq, bq), 3)
    node = Node(op="matmul", inputs=("a", "w"), output="c")
    specs = {"a": TensorSpec(name="a", dtype="int8", shape=(m, k)),
             "w": TensorSpec(name="w", dtype="int8", shape=(k, n)),
             "c": TensorSpec(name="c", dtype="int32", shape=(m, n))}
    lg = lower_graph(Graph(nodes=(node,)), specs, {"w": bq}, {})
    sim_ms, stats = _sim_run(os.path.join(tmp, f"bench_{label}.bin"),
                             lg.graph, lg.weights, lg.cmds, aq)
    return _row(label, fp32_ms, int8_ms, sim_ms, stats)


def conv_chain_row(tmp: str) -> dict:
    rng = np.random.default_rng(SEED + 1)
    x = rng.integers(-128, 127, size=(8, 10, 10), dtype=np.int8)
    w = rng.integers(-128, 127, size=(8, 8, 3, 3), dtype=np.int8)
    int8_ms = round(_time_ms(lambda: max_pool(relu(conv2d_int8(x, w)), 2, 2)), 3)
    nodes = (Node(op="conv2d", inputs=("x", "w"), output="c"),
             Node(op="relu", inputs=("c",), output="r"),
             Node(op="max_pool", inputs=("r",), output="p"))
    specs = {"x": TensorSpec(name="x", dtype="int8", shape=(8, 10, 10)),
             "w": TensorSpec(name="w", dtype="int8", shape=(8, 8, 3, 3)),
             "c": TensorSpec(name="c", dtype="int32", shape=(8, 8, 8)),
             "r": TensorSpec(name="r", dtype="int32", shape=(8, 8, 8)),
             "p": TensorSpec(name="p", dtype="int32", shape=(8, 4, 4))}
    lg = lower_graph(Graph(nodes=nodes), specs, {"w": w},
                     {"p": {"size": 2, "stride": 2}})
    sim_ms, stats = _sim_run(os.path.join(tmp, "bench_conv_chain.bin"),
                             lg.graph, lg.weights, lg.cmds, x)
    return _row("conv-chain-8ch", "", int8_ms, sim_ms, stats)


def requantize_chain_row(tmp: str) -> dict:
    s, scale = 64, 0.02
    rng = np.random.default_rng(SEED + 2)
    aq = quantize_int8(rng.standard_normal((s, s)).astype(np.float32), SCALE)
    wq = quantize_int8(rng.standard_normal((s, s)).astype(np.float32), SCALE)
    b = np.random.default_rng(SEED + 3).integers(-100, 100, size=(s,),
                                                 dtype=np.int32)
    int8_ms = round(_time_ms(
        lambda: requantize(bias_add(relu(matmul_int8(aq, wq)), b), scale)), 3)
    nodes = (Node(op="matmul", inputs=("a", "w"), output="c"),
             Node(op="relu", inputs=("c",), output="r"),
             Node(op="bias_add", inputs=("r", "b"), output="d"),
             Node(op="requantize", inputs=("d",), output="q"))
    specs = {"a": TensorSpec(name="a", dtype="int8", shape=(s, s)),
             "w": TensorSpec(name="w", dtype="int8", shape=(s, s)),
             "c": TensorSpec(name="c", dtype="int32", shape=(s, s)),
             "r": TensorSpec(name="r", dtype="int32", shape=(s, s)),
             "b": TensorSpec(name="b", dtype="int32", shape=(s,)),
             "d": TensorSpec(name="d", dtype="int32", shape=(s, s)),
             "q": TensorSpec(name="q", dtype="int8", shape=(s, s), scale=scale)}
    lg = lower_graph(Graph(nodes=nodes), specs, {"w": wq, "b": b}, {})
    sim_ms, stats = _sim_run(os.path.join(tmp, "bench_requant_chain.bin"),
                             lg.graph, lg.weights, lg.cmds, aq)
    return _row("requantize-chain-64", "", int8_ms, sim_ms, stats)


def run() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        rows = ([matmul_row(f"matmul-{s}", s, s, s, tmp) for s in SIZES]
                + [matmul_row("matmul-odd-20x12x28", 20, 12, 28, tmp),
                   matmul_row("matmul-tiny-1x8", 1, 8, 8, tmp),
                   conv_chain_row(tmp), requantize_chain_row(tmp)])
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(rows)


if __name__ == "__main__":
    run()
