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


def _conv_chain_model(path):
    from compiler.codegen import lower_conv2d, lower_max_pool, lower_relu
    from python.edge_npu.tensor import TensorSpec
    rng = np.random.default_rng(4)
    w = rng.integers(-128, 127, size=(8, 8, 3, 3), dtype=np.int8)
    nodes = (Node(op="conv2d", inputs=("x", "w"), output="c"),
             Node(op="relu", inputs=("c",), output="r"),
             Node(op="max_pool", inputs=("r",), output="p"))
    specs = {"x": TensorSpec(name="x", dtype="int8", shape=(8, 10, 10)),
             "w": TensorSpec(name="w", dtype="int8", shape=(8, 8, 3, 3)),
             "c": TensorSpec(name="c", dtype="int32", shape=(8, 8, 8)),
             "r": TensorSpec(name="r", dtype="int32", shape=(8, 8, 8)),
             "p": TensorSpec(name="p", dtype="int32", shape=(8, 4, 4))}
    nc, nh, nw = 8, 10, 10
    cmds = ([Command(op="LOAD", address=0x1000, size=nc * nh * nw),
             Command(op="LOAD", address=0x1000 + nc * nh * nw, size=8 * 8 * 3 * 3)]
            + lower_conv2d(nodes[0], specs, base=0x3000)
            + lower_relu(nodes[1], specs, base=0x3000)
            + lower_max_pool(nodes[2], specs, base=0x4000, size=2, stride=2)
            + [Command(op="STORE", address=0x5000, size=8 * 4 * 4 * 4)])
    write_model(path, Graph(nodes=nodes), {"w": w}, cmds)
    return w


def test_conv_chain_predict_matches_reference(tmp_path):
    from python.edge_npu.reference import conv2d_int8, max_pool, relu
    path = str(tmp_path / "conv_chain.bin")
    w = _conv_chain_model(path)
    rng = np.random.default_rng(5)
    x = rng.integers(-128, 127, size=(8, 10, 10), dtype=np.int8)
    model = Device().load_model(path)
    expected = max_pool(relu(conv2d_int8(x, w)), 2, 2)
    np.testing.assert_array_equal(model.predict(x), expected)
    assert model.get_stats()["cycles"] == (8 * 8 * 8 * 8 * 3 * 3) // 64


def test_pool_first_rejected(tmp_path):
    graph = Graph(nodes=(Node(op="max_pool", inputs=("x",), output="p"),))
    w = np.ones((8, 8), dtype=np.int8)
    cmds = [Command(op="MAX_POOL", address=0x4000, size=256, m=2, n=2)]
    path = str(tmp_path / "poolfirst.bin")
    write_model(path, graph, {"w": w}, cmds)
    with pytest.raises(ValueError):
        Device().load_model(path)


def test_requantize_chain_predict_matches_reference(tmp_path):
    from compiler.codegen import lower_bias_add, lower_matmul, lower_requantize
    from python.edge_npu.reference import bias_add, matmul_int8, requantize
    from python.edge_npu.tensor import TensorSpec
    rng = np.random.default_rng(6)
    w = rng.integers(-128, 127, size=(8, 8), dtype=np.int8)
    b = rng.integers(-100, 100, size=(8,), dtype=np.int32)
    scale = 0.05
    nodes = (Node(op="matmul", inputs=("a", "w"), output="c"),
             Node(op="bias_add", inputs=("c", "b"), output="d"),
             Node(op="requantize", inputs=("d",), output="q"))
    specs = {"a": TensorSpec(name="a", dtype="int8", shape=(8, 8)),
             "w": TensorSpec(name="w", dtype="int8", shape=(8, 8)),
             "c": TensorSpec(name="c", dtype="int32", shape=(8, 8)),
             "b": TensorSpec(name="b", dtype="int32", shape=(8,)),
             "d": TensorSpec(name="d", dtype="int32", shape=(8, 8)),
             "q": TensorSpec(name="q", dtype="int8", shape=(8, 8))}
    cmds = (lower_matmul(nodes[0], specs, base=0x1000)
            + lower_bias_add(nodes[1], specs, base=0x3000)
            + lower_requantize(nodes[2], specs, base=0x5000, scale=scale)
            + [Command(op="STORE", address=0x6000, size=64)])
    path = str(tmp_path / "requant.bin")
    write_model(path, Graph(nodes=nodes), {"w": w, "b": b}, cmds)
    x = rng.integers(-128, 127, size=(8, 8), dtype=np.int8)
    model = Device().load_model(path)
    expected = requantize(bias_add(matmul_int8(x, w), b), scale)
    np.testing.assert_array_equal(model.predict(x), expected)
    assert model.predict(x).dtype == np.int8
