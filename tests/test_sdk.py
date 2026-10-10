import numpy as np
import pytest

from compiler.binary import write_model
from compiler.graph import Graph, Node
from python.edge_npu.commands import Command
from python.edge_npu.reference import matmul_int8
from python.edge_npu.sdk import Device


def _tiny_model(path, shape=(8, 8)):
    m, k = shape
    rng = np.random.default_rng(0)
    w = rng.integers(-128, 127, size=(k, m), dtype=np.int8)
    graph = Graph(nodes=(Node(op="matmul", inputs=("a", "w"), output="c"),))
    cmds = [Command(op="LOAD", address=0x1000, size=m * k),
            Command(op="LOAD", address=0x1000 + m * k, size=k * m),
            Command(op="MATMUL", m=m, n=m, k=k),
            Command(op="STORE", address=0x1000 + 2 * m * k, size=m * m * 4)]
    write_model(path, graph, {"w": w}, cmds)
    return w


def test_predict_matches_reference(tmp_path):
    path = str(tmp_path / "tiny.bin")
    w = _tiny_model(path)
    rng = np.random.default_rng(1)
    x = rng.integers(-128, 127, size=(8, 8), dtype=np.int8)
    model = Device().load_model(path)
    np.testing.assert_array_equal(model.predict(x), matmul_int8(x, w))


def test_stats_report_sim_cycles(tmp_path):
    path = str(tmp_path / "tiny.bin")
    _tiny_model(path)
    rng = np.random.default_rng(1)
    model = Device().load_model(path)
    model.predict(rng.integers(-128, 127, size=(8, 8), dtype=np.int8))
    stats = model.get_stats()
    assert stats["cycles"] == 8 * 8 * 8 // 64
    assert stats["mac_utilization"] == pytest.approx(1.0)


def test_load_corrupt_file_raises(tmp_path):
    path = str(tmp_path / "bad.bin")
    with open(path, "wb") as f:
        f.write(b"not a model")
    with pytest.raises(ValueError):
        Device().load_model(path)


def test_multi_matmul_rejected(tmp_path):
    graph = Graph(nodes=(Node(op="matmul", inputs=("a", "w"), output="c"),
                         Node(op="matmul", inputs=("c", "w2"), output="y"),))
    w = np.ones((8, 8), dtype=np.int8)
    cmds = [Command(op="MATMUL", m=8, n=8, k=8)]
    path = str(tmp_path / "two.bin")
    write_model(path, graph, {"w": w, "w2": w}, cmds)
    with pytest.raises(ValueError):
        Device().load_model(path)


def _chain_model(path):
    from compiler.codegen import lower_bias_add, lower_matmul, lower_relu
    from python.edge_npu.tensor import TensorSpec
    rng = np.random.default_rng(2)
    w = rng.integers(-128, 127, size=(8, 8), dtype=np.int8)
    b = rng.integers(-100, 100, size=(8,), dtype=np.int32)
    nodes = (Node(op="matmul", inputs=("a", "w"), output="c"),
             Node(op="relu", inputs=("c",), output="r"),
             Node(op="bias_add", inputs=("r", "b"), output="y"))
    specs = {"a": TensorSpec(name="a", dtype="int8", shape=(8, 8)),
             "w": TensorSpec(name="w", dtype="int8", shape=(8, 8)),
             "c": TensorSpec(name="c", dtype="int32", shape=(8, 8)),
             "r": TensorSpec(name="r", dtype="int32", shape=(8, 8)),
             "b": TensorSpec(name="b", dtype="int32", shape=(8,)),
             "y": TensorSpec(name="y", dtype="int32", shape=(8, 8))}
    cmds = (lower_matmul(nodes[0], specs, base=0x1000)
            + lower_relu(nodes[1], specs, base=0x3000)
            + lower_bias_add(nodes[2], specs, base=0x3000))
    write_model(path, Graph(nodes=nodes), {"w": w, "b": b}, cmds)
    return w, b


def test_chain_predict_matches_reference(tmp_path):
    from python.edge_npu.reference import bias_add, matmul_int8, relu
    path = str(tmp_path / "chain.bin")
    w, b = _chain_model(path)
    rng = np.random.default_rng(3)
    x = rng.integers(-128, 127, size=(8, 8), dtype=np.int8)
    model = Device().load_model(path)
    expected = bias_add(relu(matmul_int8(x, w)), b)
    np.testing.assert_array_equal(model.predict(x), expected)


def test_forked_chain_rejected(tmp_path):
    graph = Graph(nodes=(Node(op="matmul", inputs=("a", "w"), output="c"),
                         Node(op="relu", inputs=("a",), output="y"),))
    w = np.ones((8, 8), dtype=np.int8)
    cmds = [Command(op="MATMUL", m=8, n=8, k=8)]
    path = str(tmp_path / "fork.bin")
    write_model(path, graph, {"w": w}, cmds)
    with pytest.raises(ValueError):
        Device().load_model(path)
