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
from compiler.middleend import lower_graph
from compiler.graph import Graph, Node
from python.edge_npu.commands import Command  # noqa: F401  (re-export check)
from python.edge_npu.reference import (
    bias_add,
    dequantize_int8,
    matmul_int8,
    quantization_error,
    quantize_int8,
    relu,
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
    lg = lower_graph(Graph(nodes=(node,)), specs, {"w": bq}, {})

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "demo.bin")
        write_model(path, lg.graph, lg.weights, lg.cmds)
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

    _demo_chain(rng)
    _demo_tinycnn()
    _demo_onnx_pipeline()


def _demo_chain(rng: np.random.Generator) -> None:
    """v0.2 chain + v0.4 tail: matmul → relu → bias_add → requantize."""
    brng = np.random.default_rng(SEED + 1)
    b = brng.integers(-100, 100, size=(SIZE,), dtype=np.int32)
    aq = quantize_int8(rng.standard_normal((SIZE, SIZE)).astype(np.float32), SCALE)
    w = quantize_int8(rng.standard_normal((SIZE, SIZE)).astype(np.float32), SCALE)
    scale = 0.02
    nodes = (Node(op="matmul", inputs=("a", "w"), output="c"),
             Node(op="relu", inputs=("c",), output="r"),
             Node(op="bias_add", inputs=("r", "b"), output="d"),
             Node(op="requantize", inputs=("d",), output="q"))
    specs = {"a": TensorSpec(name="a", dtype="int8", shape=(SIZE, SIZE)),
             "w": TensorSpec(name="w", dtype="int8", shape=(SIZE, SIZE)),
             "c": TensorSpec(name="c", dtype="int32", shape=(SIZE, SIZE)),
             "r": TensorSpec(name="r", dtype="int32", shape=(SIZE, SIZE)),
             "b": TensorSpec(name="b", dtype="int32", shape=(SIZE,)),
             "d": TensorSpec(name="d", dtype="int32", shape=(SIZE, SIZE)),
             "q": TensorSpec(name="q", dtype="int8", shape=(SIZE, SIZE), scale=scale)}
    lg = lower_graph(Graph(nodes=nodes), specs, {"w": w, "b": b}, {})
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "chain.bin")
        write_model(path, lg.graph, lg.weights, lg.cmds)
        model = Device().load_model(path)
        out = model.predict(aq)
        stats = model.get_stats()
    from python.edge_npu.reference import requantize as _requantize
    expected = _requantize(bias_add(relu(matmul_int8(aq, w)), b), scale)
    diff = int(np.max(np.abs(out.astype(np.int64) - expected.astype(np.int64))))
    assert diff == 0, f"chain diverged from reference by {diff}"

    print()
    print("╔══════════════════════════════════════╗")
    print("║      EdgeNPU Chain (v0.2+v0.4)       ║")
    print("╠══════════════════════════════════════╣")
    print("║ matmul→relu→bias→requant, 64x64      ║")
    print(f"║ Sim cycles:   {stats['cycles']:<10d}         ║")
    print(f"║ Out dtype:    {str(out.dtype):<10}           ║")
    print(f"║ Correctness:  max diff {diff} (exact)      ║")
    print("╚══════════════════════════════════════╝")


def _demo_tinycnn() -> None:
    """v0.3 chain: conv → relu → max_pool, bit-exact vs reference."""
    from python.edge_npu.reference import conv2d_int8, max_pool as _max_pool

    crng = np.random.default_rng(SEED + 2)
    x = crng.integers(-128, 127, size=(8, 10, 10), dtype=np.int8)
    w = crng.integers(-128, 127, size=(8, 8, 3, 3), dtype=np.int8)
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
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "tinycnn.bin")
        write_model(path, lg.graph, lg.weights, lg.cmds)
        model = Device().load_model(path)
        out = model.predict(x)
        stats = model.get_stats()
    expected = _max_pool(relu(conv2d_int8(x, w)), 2, 2)
    diff = int(np.max(np.abs(out.astype(np.int64) - expected.astype(np.int64))))
    assert diff == 0, f"tinycnn diverged from reference by {diff}"

    print()
    print("╔══════════════════════════════════════╗")
    print("║        EdgeNPU TinyCNN (v0.3)        ║")
    print("╠══════════════════════════════════════╣")
    print("║ conv → relu → max_pool, 8ch 10x10    ║")
    print(f"║ Sim cycles:   {stats['cycles']:<10d}         ║")
    print(f"║ Correctness:  max diff {diff} (exact)      ║")
    print("╚══════════════════════════════════════╝")


def _demo_onnx_pipeline() -> None:
    """Full pipeline: .onnx file → parse → quantize → lower → pack → predict."""
    from onnx import TensorProto, helper, save_model
    from compiler.frontend import load_onnx
    from compiler.quantize import quantize_activations, quantize_specs, quantize_weights

    orng = np.random.default_rng(SEED + 3)
    af = orng.standard_normal((SIZE, SIZE)).astype(np.float32)
    wf = orng.standard_normal((SIZE, SIZE)).astype(np.float32)
    with tempfile.TemporaryDirectory() as tmp:
        onnx_path = os.path.join(tmp, "tiny.onnx")
        graph_proto = helper.make_graph(
            [helper.make_node("MatMul", ["a", "w"], ["c"])], "tiny-mm",
            [helper.make_tensor_value_info("a", TensorProto.FLOAT, [SIZE, SIZE])],
            [helper.make_tensor_value_info("c", TensorProto.FLOAT, [SIZE, SIZE])],
            [helper.make_tensor("w", TensorProto.FLOAT, [SIZE, SIZE],
                                wf.flatten().tolist())])
        save_model(helper.make_model(
            graph_proto, opset_imports=[helper.make_opsetid("", 17)]), onnx_path)
        graph, weights, specs, attrs = load_onnx(onnx_path)
        qw, wscales = quantize_weights(weights)
        ascales = quantize_activations({"a": [af]})
        qspecs = quantize_specs(graph, specs, {**wscales, **ascales})
        lg = lower_graph(graph, qspecs, qw, attrs)
        bin_path = os.path.join(tmp, "onnx.bin")
        write_model(bin_path, lg.graph, lg.weights, lg.cmds)
        model = Device().load_model(bin_path)
        aq = quantize_int8(af, ascales["a"])
        out = model.predict(aq)
        stats = model.get_stats()
    expected = matmul_int8(aq, qw["w"])
    diff = int(np.max(np.abs(out.astype(np.int64) - expected.astype(np.int64))))
    assert diff == 0, f"onnx pipeline diverged by {diff}"
    qerr = quantization_error(
        af.astype(np.float32) @ wf.astype(np.float32),
        dequantize_int8(aq, ascales["a"]) @ dequantize_int8(qw["w"], wscales["w"]),
    )

    print()
    print("╔══════════════════════════════════════╗")
    print("║       EdgeNPU ONNX Pipeline        ║")
    print("╠══════════════════════════════════════╣")
    print("║ onnx→parse→quant→lower→pack→run      ║")
    print(f"║ w scale:      {wscales['w']:>8.5f}             ║")
    print(f"║ Sim cycles:   {stats['cycles']:<10d}         ║")
    print(f"║ Quant err:    {qerr:>8.4f}               ║")
    print(f"║ Correctness:  max diff {diff} (exact)      ║")
    print("╚══════════════════════════════════════╝")


if __name__ == "__main__":
    main()
