import struct

import numpy as np
import pytest

from compiler.binary import write_model
from compiler.graph import Graph, Node
from compiler.middleend import lower_graph
from python.edge_npu.commands import Command
from python.edge_npu.reference import matmul_int8, relu, requantize
from python.edge_npu.sdk import Device
from python.edge_npu.tensor import TensorSpec


def _mlp():
    nodes = (Node(op="matmul", inputs=("a", "w1"), output="c"),
             Node(op="relu", inputs=("c",), output="r"),
             Node(op="matmul", inputs=("r", "w2"), output="d"),
             Node(op="requantize", inputs=("d",), output="q"))
    specs = {"a": TensorSpec(name="a", dtype="int8", shape=(8, 8)),
             "w1": TensorSpec(name="w1", dtype="int8", shape=(8, 8)),
             "c": TensorSpec(name="c", dtype="int32", shape=(8, 8)),
             "r": TensorSpec(name="r", dtype="int32", shape=(8, 8)),
             "w2": TensorSpec(name="w2", dtype="int8", shape=(8, 8)),
             "d": TensorSpec(name="d", dtype="int32", shape=(8, 8)),
             "q": TensorSpec(name="q", dtype="int8", shape=(8, 8), scale=0.02)}
    return nodes, specs


def test_second_matmul_loads_only_sources():
    nodes, specs = _mlp()
    rng = np.random.default_rng(41)
    w1 = rng.integers(-128, 127, size=(8, 8), dtype=np.int8)
    w2 = rng.integers(-128, 127, size=(8, 8), dtype=np.int8)
    lg = lower_graph(Graph(nodes=nodes), specs, {"w1": w1, "w2": w2}, {})
    assert [c.op for c in lg.cmds] == ["LOAD", "LOAD", "LOAD", "MATMUL", "STORE",
                                       "MATMUL", "STORE",
                                       "REQUANTIZE", "STORE"]
    assert [c.size for c in lg.cmds if c.op == "LOAD"] == [64, 64, 64]
    mms = [(c.m, c.n, c.k) for c in lg.cmds if c.op == "MATMUL"]
    assert mms == [(8, 8, 8), (8, 8, 8)]


def test_mlp_end_to_end_matches_reference(tmp_path):
    nodes, specs = _mlp()
    rng = np.random.default_rng(42)
    w1 = rng.integers(-128, 127, size=(8, 8), dtype=np.int8)
    w2 = rng.integers(-128, 127, size=(8, 8), dtype=np.int8)
    lg = lower_graph(Graph(nodes=nodes), specs, {"w1": w1, "w2": w2}, {})
    path = str(tmp_path / "mlp.bin")
    write_model(path, lg.graph, lg.weights, lg.cmds)
    x = rng.integers(-128, 127, size=(8, 8), dtype=np.int8)
    model = Device().load_model(path)
    out = model.predict(x)
    expected = requantize(matmul_int8(relu(matmul_int8(x, w1)), w2), 0.02)
    assert out.shape == (8, 8)
    np.testing.assert_array_equal(out, expected)
    assert model.get_stats()["cycles"] == 2 * (8 * 8 * 8 // 64)
    assert model.get_stats()["dma_bytes"] == 64 + 64 + 256 + 64 + 256 + 64
    assert model.get_stats()["command_counts"] == {
        "LOAD": 3, "MATMUL": 2, "STORE": 3, "REQUANTIZE": 1}


def test_mlp_odd_second_layer_matches_reference(tmp_path):
    nodes = (Node(op="matmul", inputs=("a", "w1"), output="c"),
             Node(op="matmul", inputs=("c", "w2"), output="d"))
    specs = {"a": TensorSpec(name="a", dtype="int8", shape=(20, 12)),
             "w1": TensorSpec(name="w1", dtype="int8", shape=(12, 28)),
             "c": TensorSpec(name="c", dtype="int32", shape=(20, 28)),
             "w2": TensorSpec(name="w2", dtype="int8", shape=(28, 16)),
             "d": TensorSpec(name="d", dtype="int32", shape=(20, 16))}
    rng = np.random.default_rng(43)
    w1 = rng.integers(-128, 127, size=(12, 28), dtype=np.int8)
    w2 = rng.integers(-128, 127, size=(28, 16), dtype=np.int8)
    lg = lower_graph(Graph(nodes=nodes), specs, {"w1": w1, "w2": w2}, {})
    path = str(tmp_path / "mlp_odd.bin")
    write_model(path, lg.graph, lg.weights, lg.cmds)
    x = rng.integers(-128, 127, size=(20, 12), dtype=np.int8)
    out = Device().load_model(path).predict(x)
    np.testing.assert_array_equal(out, matmul_int8(matmul_int8(x, w1), w2))
