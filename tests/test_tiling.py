import numpy as np
import pytest

from compiler.binary import write_model
from compiler.graph import Graph, Node
from compiler.middleend import legalize, lower_graph, schedule
from python.edge_npu.commands import Command
from python.edge_npu.reference import bias_add, matmul_int8, relu
from python.edge_npu.sdk import Device
from python.edge_npu.tensor import TensorSpec


def _mm_specs(a, w, c):
    return {"a": TensorSpec(name="a", dtype="int8", shape=a),
            "w": TensorSpec(name="w", dtype="int8", shape=w),
            "c": TensorSpec(name="c", dtype="int32", shape=c)}


def _order(nodes, specs):
    return schedule(Graph(nodes=nodes), specs)


def test_legalize_aligned_is_identity():
    nodes = (Node(op="matmul", inputs=("a", "w"), output="c"),)
    specs = _mm_specs((64, 64), (64, 64), (64, 64))
    assert legalize(_order(nodes, specs), specs, {}) == specs


def test_legalize_odd_matmul_pads_to_tiles():
    nodes = (Node(op="matmul", inputs=("a", "w"), output="c"),)
    specs = _mm_specs((60, 64), (64, 48), (60, 48))
    got = legalize(_order(nodes, specs), specs, {})
    assert got["a"].shape == (64, 64)
    assert got["w"].shape == (64, 48)
    assert got["c"].shape == (64, 48)
    assert (got["a"].dtype, got["c"].dtype) == ("int8", "int32")


def test_legalize_inner_mismatch_still_rejected():
    nodes = (Node(op="matmul", inputs=("a", "w"), output="c"),)
    specs = _mm_specs((8, 7), (9, 8), (8, 8))
    with pytest.raises(ValueError, match="inner dim mismatch 7 vs 9"):
        legalize(_order(nodes, specs), specs, {})


def test_legalize_conv_pads_output_spatial():
    nodes = (Node(op="conv2d", inputs=("x", "w"), output="y"),)
    specs = {"x": TensorSpec(name="x", dtype="int8", shape=(8, 9, 9)),
             "w": TensorSpec(name="w", dtype="int8", shape=(8, 8, 3, 3)),
             "y": TensorSpec(name="y", dtype="int32", shape=(8, 7, 7))}
    got = legalize(_order(nodes, specs), specs, {})
    assert got["x"].shape == (8, 10, 10)
    assert got["w"].shape == (8, 8, 3, 3)
    assert got["y"].shape == (8, 8, 8)


def test_legalize_elementwise_inherits_padded_shape():
    nodes = (Node(op="matmul", inputs=("a", "w"), output="c"),
             Node(op="relu", inputs=("c",), output="r"))
    specs = dict(_mm_specs((60, 64), (64, 64), (60, 64)),
                 r=TensorSpec(name="r", dtype="int32", shape=(60, 64)))
    got = legalize(_order(nodes, specs), specs, {})
    assert got["c"].shape == got["r"].shape == (64, 64)


def test_legalize_pads_bias_to_padded_last_dim():
    nodes = (Node(op="matmul", inputs=("a", "w"), output="c"),
             Node(op="bias_add", inputs=("c", "b"), output="d"))
    specs = dict(_mm_specs((60, 60), (60, 60), (60, 60)),
                 b=TensorSpec(name="b", dtype="int32", shape=(60,)),
                 d=TensorSpec(name="d", dtype="int32", shape=(60, 60)))
    got = legalize(_order(nodes, specs), specs, {})
    assert got["d"].shape == (64, 64)
    assert got["b"].shape == (64,)


def test_legalize_pool_derives_both_shapes():
    nodes = (Node(op="conv2d", inputs=("x", "w"), output="c"),
             Node(op="max_pool", inputs=("c",), output="p"))
    specs = {"x": TensorSpec(name="x", dtype="int8", shape=(8, 9, 9)),
             "w": TensorSpec(name="w", dtype="int8", shape=(8, 8, 3, 3)),
             "c": TensorSpec(name="c", dtype="int32", shape=(8, 7, 7)),
             "p": TensorSpec(name="p", dtype="int32", shape=(8, 3, 3))}
    got = legalize(_order(nodes, specs), specs, {"p": {"size": 2, "stride": 2}})
    assert got["c"].shape == (8, 8, 8)
    assert got["p"].shape == (8, 4, 4)


def test_legalize_pool_needs_attrs():
    nodes = (Node(op="max_pool", inputs=("x",), output="p"),)
    specs = {"x": TensorSpec(name="x", dtype="int8", shape=(8, 8, 8)),
             "p": TensorSpec(name="p", dtype="int8", shape=(8, 4, 4))}
    with pytest.raises(ValueError, match="'p'"):
        legalize(_order(nodes, specs), specs, {})


def test_lower_graph_odd_matmul_stream_is_tiled():
    node = Node(op="matmul", inputs=("a", "w"), output="c")
    specs = _mm_specs((20, 12), (12, 28), (20, 28))
    lg = lower_graph(Graph(nodes=(node,)), specs,
                     {"w": np.ones((12, 28), dtype=np.int8)}, {})
    assert lg.cmds == [
        Command(op="LOAD", address=0x1000, size=384),
        Command(op="LOAD", address=0x1180, size=512),
        Command(op="MATMUL", m=24, n=32, k=16),
        Command(op="STORE", address=0x1380, size=3072),
    ]
    assert lg.weights["w"].shape == (12, 28)  # stored logical


def test_lower_graph_padded_budget_boundary():
    from compiler.memory import MemoryConfig
    node = Node(op="matmul", inputs=("a", "w"), output="c")
    specs = _mm_specs((20, 12), (12, 28), (20, 28))
    weights = {"w": np.ones((12, 28), dtype=np.int8)}
    lower_graph(Graph(nodes=(node,)), specs, weights, {},
                memory=MemoryConfig(size=3968))  # padded peak exactly
    with pytest.raises(ValueError, match="3968"):
        lower_graph(Graph(nodes=(node,)), specs, weights, {},
                    memory=MemoryConfig(size=3967))


def test_lower_graph_odd_dims_still_rejected_below_legalize():
    from compiler.codegen import lower_matmul
    node = Node(op="matmul", inputs=("a", "w"), output="c")
    with pytest.raises(ValueError, match="not divisible"):
        lower_matmul(node, _mm_specs((60, 64), (64, 64), (60, 64)), base=0x1000)


def test_odd_matmul_end_to_end_matches_reference(tmp_path):
    node = Node(op="matmul", inputs=("a", "w"), output="c")
    specs = _mm_specs((20, 12), (12, 28), (20, 28))
    rng = np.random.default_rng(21)
    w = rng.integers(-128, 127, size=(12, 28), dtype=np.int8)
    lg = lower_graph(Graph(nodes=(node,)), specs, {"w": w}, {})
    path = str(tmp_path / "odd.bin")
    write_model(path, lg.graph, lg.weights, lg.cmds)
    x = rng.integers(-128, 127, size=(20, 12), dtype=np.int8)
    out = Device().load_model(path).predict(x)
    assert out.shape == (20, 28)
    np.testing.assert_array_equal(out, matmul_int8(x, w))


def test_tiny_row_tiles_and_trims(tmp_path):
    node = Node(op="matmul", inputs=("a", "w"), output="c")
    specs = _mm_specs((1, 8), (8, 8), (1, 8))
    rng = np.random.default_rng(22)
    w = rng.integers(-128, 127, size=(8, 8), dtype=np.int8)
    lg = lower_graph(Graph(nodes=(node,)), specs, {"w": w}, {})
    path = str(tmp_path / "tiny.bin")
    write_model(path, lg.graph, lg.weights, lg.cmds)
    x = rng.integers(-128, 127, size=(1, 8), dtype=np.int8)
    out = Device().load_model(path).predict(x)
    assert out.shape == (1, 8)
    np.testing.assert_array_equal(out, matmul_int8(x, w))


def test_odd_conv_pool_matches_reference(tmp_path):
    from python.edge_npu.reference import conv2d_int8, max_pool
    nodes = (Node(op="conv2d", inputs=("x", "w"), output="c"),
             Node(op="relu", inputs=("c",), output="r"),
             Node(op="max_pool", inputs=("r",), output="p"))
    specs = {"x": TensorSpec(name="x", dtype="int8", shape=(4, 9, 9)),
             "w": TensorSpec(name="w", dtype="int8", shape=(8, 4, 3, 3)),
             "c": TensorSpec(name="c", dtype="int32", shape=(8, 7, 7)),
             "r": TensorSpec(name="r", dtype="int32", shape=(8, 7, 7)),
             "p": TensorSpec(name="p", dtype="int32", shape=(8, 3, 3))}
    rng = np.random.default_rng(23)
    w = rng.integers(-128, 127, size=(8, 4, 3, 3), dtype=np.int8)
    lg = lower_graph(Graph(nodes=nodes), specs, {"w": w},
                     {"p": {"size": 2, "stride": 2}})
    path = str(tmp_path / "oddcnn.bin")
    write_model(path, lg.graph, lg.weights, lg.cmds)
    x = rng.integers(-128, 127, size=(4, 9, 9), dtype=np.int8)
    out = Device().load_model(path).predict(x)
    assert out.shape == (8, 3, 3)
    np.testing.assert_array_equal(out, max_pool(relu(conv2d_int8(x, w)), 2, 2))
